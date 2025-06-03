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

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/rules"
                     ]

def filter_relevant_documents_with_llm(document_links, query, ai_response, llm, max_docs=3):
    """
    Use LLM to intelligently filter document links based on query and AI response relevance
    """
    if not document_links or not llm:
        return []
    
    # First, basic filtering to remove obviously irrelevant links
    filtered_docs = []
    for doc_link in document_links:
        title = doc_link.get('title', '').strip()
        if (len(title) > 10 and 
            title.lower() not in ['click here', 'read more', 'download', 'view more', 'see all'] and
            not any(skip_word in title.lower() for skip_word in ['home', 'contact', 'about us', 'sitemap'])):
            filtered_docs.append(doc_link)
    
    if not filtered_docs:
        return []
    
    # Prepare document titles and links for LLM evaluation
    doc_list = []
    for i, doc in enumerate(filtered_docs):
        doc_list.append(f"{i+1}. {doc['title']} - {doc['link']}")
    
    doc_text = "\n".join(doc_list)
    
    # Create LLM prompt for document relevance filtering
    relevance_prompt = f"""
    Given the following user query and AI response, identify the most relevant document links.
    
    User Query: {query}
    
    AI Response Summary: {ai_response[:500]}...
    
    Available Documents:
    {doc_text}
    
    Instructions:
    1. Select only documents that are directly relevant to the user's query and mentioned concepts in the AI response
    2. Focus on documents that contain regulations, acts, circulars, guidelines, or policies related to the query topic
    3. Exclude generic, administrative, or unrelated documents
    4. Return maximum {max_docs} most relevant documents
    5. Return only the document numbers (e.g., "1, 3, 5") of the most relevant ones
    
    Relevant document numbers (comma-separated):
    """
    
    try:
        # Get LLM's assessment of document relevance
        llm_response = llm.invoke(relevance_prompt)
        relevant_numbers = []
        
        # Parse the LLM response to extract document numbers
        if hasattr(llm_response, 'content'):
            response_text = llm_response.content
        else:
            response_text = str(llm_response)
        
        # Extract numbers from the response
        numbers = re.findall(r'\b\d+\b', response_text)
        relevant_numbers = [int(num) - 1 for num in numbers if int(num) <= len(filtered_docs)]
        
        # Return the relevant documents based on LLM selection
        relevant_docs = [filtered_docs[i] for i in relevant_numbers[:max_docs] if 0 <= i < len(filtered_docs)]
        
        return relevant_docs
        
    except Exception as e:
        st.warning(f"LLM filtering failed, using fallback method: {e}")
        # Fallback to keyword-based filtering
        return keyword_based_filter(filtered_docs, query, ai_response, max_docs)

def keyword_based_filter(document_links, query, ai_response, max_docs=3):
    """
    Fallback method for document filtering using keyword matching
    """
    # Extract key terms from query and response
    query_terms = set(query.lower().split())
    response_terms = set(ai_response.lower().split())
    
    # Common regulatory keywords
    regulatory_keywords = {'act', 'regulation', 'circular', 'guideline', 'amendment', 
                          'notification', 'policy', 'rule', 'framework', 'directive'}
    
    all_terms = query_terms.union(response_terms).union(regulatory_keywords)
    
    # Score documents based on keyword relevance
    scored_docs = []
    for doc in document_links:
        title_lower = doc['title'].lower()
        score = 0
        
        # Score based on keyword matches
        for term in all_terms:
            if len(term) > 2 and term in title_lower:
                score += 1
        
        # Bonus for regulatory documents
        if any(keyword in title_lower for keyword in regulatory_keywords):
            score += 2
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and return top documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_docs]]

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
    
    # Focus on clickable content links, not just file downloads
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
        
        # Focus on content pages rather than file downloads
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive',
                           'insurance', 'aadhaar', 'compliance', 'licensing']
        
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        # Include content links, not just PDF downloads
        if has_doc_keywords and len(link_text) > 5:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'content'
            })
    
    # Extract from tables (common structure for regulatory websites)
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
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
                            r'regulation.*\d+',
                            r'aadhaar.*act'
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        
                        if is_likely_document:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'reference'
                            })
    
    # Extract from content sections
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
                
                if 10 < len(link_text) < 200:
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy', 'aadhaar']
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        document_links.append({
                            'title': link_text,
                            'link': href,
                            'type': 'reference'
                        })
    
    # Remove duplicates
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    # Sort by relevance (content type first, then alphabetically)
    unique_document_links.sort(key=lambda x: (x['type'] != 'content', x['title']))
    
    return unique_document_links

def extract_structured_content(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content_sections = {}
    
    # Extract news/updates sections
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
    
    # Extract main text content
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
                        st.write(f"**Total document links found:** {len(sections['document_links'])}")
                        
                        content_docs = [link for link in sections['document_links'] if link['type'] == 'content']
                        ref_docs = [link for link in sections['document_links'] if link['type'] == 'reference']
                        
                        if content_docs:
                            st.write("**Content Pages:**")
                            for i, link_info in enumerate(content_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if ref_docs:
                            st.write("**Reference Documents:**")
                            for i, link_info in enumerate(ref_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.write(f"No relevant document links found from {url}")
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    return loaded_docs

def is_fallback_response(response_text):
    fallback_phrases = [
        "fall outside the scope of the data I've been trained on",
        "details you've asked for fall outside the scope",
        "outside the scope of the data",
        "However, I've gathered information that closely aligns"
    ]
    
    return any(phrase in response_text for phrase in fallback_phrases)

# Initialize session state
if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
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
                st.session_state['llm'] = llm
                
                hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.
                    
                    IMPORTANT INSTRUCTIONS:
                    - ONLY answer questions that can be addressed using the provided context ONLY from the provided websites
                    - If a question is completely outside the insurance/regulatory domain or if the information is not available in the provided context, respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."
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
            
            if not is_fallback_response(response['answer']):
                retrieved_docs = response.get('context', [])
                
                # Collect all document links from retrieved documents
                all_document_links = []
                for doc in retrieved_docs:
                    if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                        for link_info in doc.metadata['sections']['document_links']:
                            if link_info not in all_document_links:
                                all_document_links.append(link_info)
                
                # Use LLM-based intelligent filtering for document relevance
                if all_document_links and st.session_state['llm']:
                    relevant_docs = filter_relevant_documents_with_llm(
                        all_document_links, 
                        query, 
                        response['answer'], 
                        st.session_state['llm'],
                        max_docs=3
                    )
                    
                    if relevant_docs:
                        st.write("\n**📄 Most Relevant Documents:**")
                        for i, link_info in enumerate(relevant_docs):
                            st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                    else:
                        st.info("No highly relevant document links found for this specific query.")
                
                st.write("\n**📍 Information Sources:**")
                sources = set()
                for doc in retrieved_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    sources.add(source)
                
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
            else:
                st.info("ℹ️ No specific documents or sources are available for this query as it falls outside the current data scope.")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
