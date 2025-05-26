import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.parse import urljoin, urlparse

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts"
                      
                     ]

def get_relevant_documents(document_links, query, ai_response, max_docs=3):
    """
    Simple approach: just return the first few document links from retrieved context
    """
    if not document_links:
        return []
    
    # Basic filtering - only keep documents with meaningful titles
    filtered_docs = []
    for doc_link in document_links:
        title = doc_link.get('title', '').strip()
        if len(title) > 10 and title.lower() not in ['click here', 'read more', 'download']:
            filtered_docs.append(doc_link)
    
    # Just return the first few - they're already from relevant retrieved documents
    return filtered_docs[:max_docs]

def enhanced_web_scrape(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text
        
    except Exception as e:
        st.error(f"Enhanced scraping failed for {url}: {e}")
        return None

def extract_document_links(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
    document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
    
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href')
        link_text = link.get_text(strip=True)
        
        if not href or len(link_text) < 3:
            continue
            
        if href.startswith('/'):
            href = urljoin(url, href)
        elif not href.startswith(('http://', 'https://')):
            href = urljoin(url, href)
        
        is_document_link = any(ext in href.lower() for ext in document_extensions)
        
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive']
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if is_document_link or has_doc_keywords:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'document' if is_document_link else 'content'
            })
    
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                for cell in cells:
                    links_in_cell = cell.find_all('a', href=True)
                    for link in links_in_cell:
                        href = link.get('href')
                        link_text = link.get_text(strip=True)
                        
                        if not href or len(link_text) < 5:
                            continue
                            
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        elif not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        
                        document_patterns = [
                            r'act.*\d{4}',
                            r'circular.*\d+',
                            r'amendment.*act',
                            r'insurance.*act',
                            r'guideline',
                            r'master.*direction',
                            r'regulation.*\d+'
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        is_document_extension = any(ext in href.lower() for ext in document_extensions)
                        
                        if is_likely_document or is_document_extension:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'document' if is_document_extension else 'reference'
                            })
    
    content_sections = soup.find_all(['div', 'section', 'article'])
    for section in content_sections:
        section_text = section.get_text().lower()
        if any(keyword in section_text for keyword in ['act', 'circular', 'regulation', 'guideline']):
            links_in_section = section.find_all('a', href=True)
            for link in links_in_section:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if not href or len(link_text) < 5:
                    continue
                
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                if len(link_text) > 10 and len(link_text) < 200:
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy']
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        document_links.append({
                            'title': link_text,
                            'link': href,
                            'type': 'reference'
                        })
    
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    unique_document_links.sort(key=lambda x: (x['type'] != 'document', x['title']))
    
    return unique_document_links

def extract_structured_content(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content_sections = {}
    
    news_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ['news', 'update', 'recent', 'latest', 'whats-new']
    ))
    
    if news_sections:
        content_sections['news'] = []
        for section in news_sections:
            text = section.get_text(strip=True)
            if len(text) > 50:
                content_sections['news'].append(text)
    
    content_sections['document_links'] = extract_document_links(html_content, url)
    
    main_text = soup.get_text()
    
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return clean_text, content_sections

def load_hardcoded_websites():
    loaded_docs = []
    
    for url in HARDCODED_WEBSITES:
        try:
            st.write(f"Loading URL: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content(html_content, url)
                
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url, "sections": sections}
                )
                loaded_docs.append(doc)
                
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                if sections.get('document_links'):
                    with st.expander(f"Document Links found from {url}"):
                        st.write(f"**Total documents found:** {len(sections['document_links'])}")
                        
                        pdf_docs = [link for link in sections['document_links'] if link['type'] == 'document']
                        ref_docs = [link for link in sections['document_links'] if link['type'] in ['reference', 'content']]
                        
                        if pdf_docs:
                            st.write("**Direct Document Downloads:**")
                            for i, link_info in enumerate(pdf_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if ref_docs:
                            st.write("**Document References:**")
                            for i, link_info in enumerate(ref_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.write(f"No document links found from {url}")
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    return loaded_docs

if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False

st.title("Web GEN-ie")

api_key = "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"

if not st.session_state['docs_loaded']:
    if st.button("Load Websites"):
        st.session_state['loaded_docs'] = load_hardcoded_websites()
        st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
        
        if api_key and st.session_state['loaded_docs']:
            with st.spinner("Processing documents..."):
                llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
                hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.
                    
                    IMPORTANT INSTRUCTIONS:
                    - ONLY answer questions that can be addressed using the provided context from the provided websites
                    - If a question is completely outside the insurance/regulatory domain or if the information is not available in the provided context, respond with: "I can only provide information based on the regulatory documents and guidelines available in my dataset. The specific details you've asked about are not available in the current context. Please ask questions related to insurance regulations, acts, circulars, guidelines, or policy updates."
                    - Pay special attention to dates, recent updates, and chronological information
                    - When asked about "what's new" or recent developments, focus on the most recent information available
                    - Look for press releases, circulars, guidelines, and policy updates
                    - Provide specific details about new regulations, policy changes, or announcements
                    - If you find dated information, mention the specific dates
                    - When mentioning any acts, circulars, or regulations, try to reference the available document links
                    
                    Based on the context provided from the insurance regulatory website(s), answer the user's question accurately and comprehensively.
                    
                    <context>
                    {context}
                    </context>
                    
                    Question: {input}
                    
                    Answer with specific details, dates, and references where available. If relevant documents are mentioned, note that direct links may be available in the sources section.
                    """
                )
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
                st.write(f"Number of chunks created: {len(document_chunks)}")
                
                st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 6})
                st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
                
                st.session_state['docs_loaded'] = True
                st.success("Documents processed and ready for querying!")

st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            response = st.session_state['retrieval_chain'].invoke({"input": query})
            
            st.subheader("Response:")
            st.write(response['answer'])
            
            # Show additional information for all responses
            retrieved_docs = response.get('context', [])
            
            # Extract and show relevant documents
            all_document_links = []
            for doc in retrieved_docs:
                if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                    for link_info in doc.metadata['sections']['document_links']:
                        if link_info not in all_document_links:
                            all_document_links.append(link_info)
            
            relevant_docs = get_relevant_documents(all_document_links, query, response['answer']) if all_document_links else []
            
            if relevant_docs:
                st.write("\n**📄 Related Documents:**")
                for i, link_info in enumerate(relevant_docs[:3]):
                    st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
            
            # Show sources
            st.write("\n**📍 Information Sources:**")
            sources = set()
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown')
                sources.add(source)
            
            for i, source in enumerate(sources, 1):
                st.write(f"{i}. [{source}]({source})")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
