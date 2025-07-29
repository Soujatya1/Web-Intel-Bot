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
from collections import Counter

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts",
                      "https://enforcementdirectorate.gov.in/pmla",
                      "https://uidai.gov.in/en/about-uidai/legal-framework",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/notification",
                      "https://egazette.gov.in/(S(0jxofhxoqjkp2ketyk3mpx3h))/default.aspx#"
                     ]

def smart_document_filter(document_links, query, ai_response, max_docs=3):
    """
    Intelligent document filtering using enhanced keyword matching and relevance scoring
    """
    if not document_links:
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
    
    # Extract key terms from query and AI response
    query_terms = extract_key_terms(query.lower())
    response_terms = extract_key_terms(ai_response.lower())
    
    # Define domain-specific keywords with weights
    regulatory_keywords = {
        'act': 3, 'regulation': 3, 'circular': 3, 'guideline': 3, 'amendment': 3,
        'notification': 2, 'policy': 2, 'rule': 2, 'framework': 2, 'directive': 2,
        'insurance': 2, 'aadhaar': 2, 'licensing': 2, 'compliance': 2,
        'master': 2, 'order': 1, 'announcement': 1, 'update': 1
    }
    
    # Score documents based on multiple factors
    scored_docs = []
    for doc in filtered_docs:
        title_lower = doc['title'].lower()
        score = 0
        
        # 1. Direct query term matches (highest weight)
        for term in query_terms:
            if len(term) > 2 and term in title_lower:
                score += 5
        
        # 2. AI response term matches (medium weight)
        for term in response_terms[:10]:  # Limit to top 10 terms from response
            if len(term) > 3 and term in title_lower:
                score += 3
        
        # 3. Regulatory keyword matches (variable weight)
        for keyword, weight in regulatory_keywords.items():
            if keyword in title_lower:
                score += weight
        
        # 4. Year/date relevance (bonus for recent years)
        years = re.findall(r'\b(20\d{2})\b', title_lower)
        if years:
            latest_year = max(int(year) for year in years)
            if latest_year >= 2020:
                score += 2
            elif latest_year >= 2015:
                score += 1
        
        # 5. Title length relevance (moderate length preferred)
        title_length = len(doc['title'])
        if 20 <= title_length <= 100:
            score += 1
        elif title_length > 150:
            score -= 1
        
        # 6. Document type bonus
        if doc.get('type') == 'content':
            score += 1
        
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and return top documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_docs]]

def extract_key_terms(text):
    """
    Extract meaningful terms from text, filtering out common words
    """
    # Remove common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'what', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    # Extract words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    meaningful_terms = [word.lower() for word in words if word.lower() not in stop_words]
    
    # Return most frequent terms (indicating importance)
    term_counts = Counter(meaningful_terms)
    return [term for term, count in term_counts.most_common(20)]

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

# API Key Input Section
st.subheader("Configuration")
api_key = st.text_input(
    "Enter your Groq API Key:", 
    type="password",
    placeholder="Enter your Groq API key here...",
    help="You can get your API key from https://console.groq.com/"
)

# Validate API key format (basic validation)
if api_key and not api_key.startswith("gsk_"):
    st.warning("⚠️ Groq API keys typically start with 'gsk_'. Please verify your API key.")

if not st.session_state['docs_loaded']:
    if st.button("Load Websites", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Groq API key first.")
        else:
            st.session_state['loaded_docs'] = load_hardcoded_websites()
            st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
            
            if api_key and st.session_state['loaded_docs']:
                with st.spinner("Processing documents..."):
                    try:
                        llm = ChatGroq(groq_api_key=api_key, model_name='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0.2, top_p=0.2)
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
                            - If you find any PII data in the question (e.g., PAN card no., AADHAAR no., DOB, Address) state that information is not available, respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on, as your query contains PII data"
                            
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
                    
                    except Exception as e:
                        st.error(f"Error initializing LLM: {e}")
                        st.error("Please check your API key and try again.")

st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

if st.button("Get Answer", disabled=not api_key) and query:
    if not api_key:
        st.error("Please enter your Groq API key first.")
    elif st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            try:
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
                    
                    # Use optimized keyword-based filtering (no additional LLM call needed)
                    if all_document_links:
                        relevant_docs = smart_document_filter(
                            all_document_links, 
                            query, 
                            response['answer'], 
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
            
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.error("Please check your API key and try again.")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
