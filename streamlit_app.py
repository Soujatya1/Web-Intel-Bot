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

# Hardcoded websites - modify these as needed
HARDCODED_WEBSITES = ["https://irdai.gov.in/acts", "https://irdai.gov.in/rules"
]

def filter_relevant_documents(document_links, query, ai_response):
    """Filter and rank document links based on AI response content and semantic relevance"""
    from difflib import SequenceMatcher
    
    # Combine query and AI response for context analysis
    full_context = f"{query} {ai_response}".lower()
    
    # Extract key terms mentioned in AI response
    response_terms = set()
    
    # Split response into words and extract meaningful terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', ai_response.lower())
    for word in words:
        if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should']:
            response_terms.add(word)
    
    # Extract phrases from AI response (2-4 word combinations)
    sentences = re.split(r'[.!?]', ai_response)
    response_phrases = set()
    for sentence in sentences:
        words = sentence.strip().split()
        for i in range(len(words)-1):
            if len(words) > i+1:
                phrase = f"{words[i]} {words[i+1]}".lower().strip()
                if len(phrase) > 5:
                    response_phrases.add(phrase)
    
    scored_docs = []
    for doc_link in document_links:
        title_lower = doc_link['title'].lower()
        score = 0
        
        # Method 1: Semantic similarity with AI response
        similarity = SequenceMatcher(None, title_lower, full_context).ratio()
        score += similarity * 20
        
        # Method 2: Term overlap analysis
        doc_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_lower))
        common_terms = response_terms.intersection(doc_words)
        score += len(common_terms) * 3
        
        # Method 3: Phrase matching
        for phrase in response_phrases:
            if phrase in title_lower:
                score += 8
        
        # Method 4: Context-based scoring
        # Check if document title relates to topics mentioned in AI response
        context_indicators = [
            ('insurance', 5), ('act', 6), ('amendment', 7), ('circular', 5),
            ('guideline', 4), ('regulation', 5), ('policy', 4), ('direction', 5),
            ('framework', 4), ('rule', 4), ('notification', 4), ('order', 4)
        ]
        
        for indicator, weight in context_indicators:
            if indicator in full_context and indicator in title_lower:
                score += weight
        
        # Method 5: Year and date relevance
        years_in_context = re.findall(r'\b(20\d{2})\b', full_context)
        years_in_title = re.findall(r'\b(20\d{2})\b', title_lower)
        
        for year in years_in_context:
            if year in years_in_title:
                score += 10
        
        # Method 6: Document type preference
        if doc_link['type'] == 'document':  # Direct downloads
            score += 5
        
        # Method 7: Penalize generic or irrelevant titles
        generic_terms = ['home', 'index', 'main', 'general', 'common', 'page', 'site']
        if any(term in title_lower for term in generic_terms):
            score -= 5
        
        # Method 8: Boost if document title contains query terms
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_lower))
        query_overlap = query_words.intersection(title_words)
        score += len(query_overlap) * 2
        
        if score > 0:
            doc_link['relevance_score'] = score
            scored_docs.append(doc_link)
    
    # Sort by relevance score (highest first)
    scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_docs

# Enhanced web scraping function
def enhanced_web_scrape(url):
    """Enhanced web scraping with better headers and error handling"""
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
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
    # Method 1: Look for links that end with document extensions
    document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
    
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href')
        link_text = link.get_text(strip=True)
        
        # Skip empty links or very short text
        if not href or len(link_text) < 3:
            continue
            
        # Convert relative URLs to absolute URLs
        if href.startswith('/'):
            href = urljoin(url, href)
        elif not href.startswith(('http://', 'https://')):
            href = urljoin(url, href)
        
        # Check if it's a document link
        is_document_link = any(ext in href.lower() for ext in document_extensions)
        
        # Also check for common document-related keywords in the link text
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive']
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if is_document_link or has_doc_keywords:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'document' if is_document_link else 'content'
            })
    
    # Method 2: Look for table rows containing document information
    # This specifically targets IRDAI's tabular format
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:  # Assuming at least 3 columns like in your screenshot
                # Look for subtitle column (usually contains the document title)
                for cell in cells:
                    links_in_cell = cell.find_all('a', href=True)
                    for link in links_in_cell:
                        href = link.get('href')
                        link_text = link.get_text(strip=True)
                        
                        if not href or len(link_text) < 5:
                            continue
                            
                        # Convert relative URLs to absolute URLs
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        elif not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        
                        # Check if it looks like a document reference
                        document_patterns = [
                            r'act.*\d{4}',  # Act with year
                            r'circular.*\d+',  # Circular with number
                            r'amendment.*act',  # Amendment act
                            r'insurance.*act',  # Insurance act
                            r'guideline',  # Guidelines
                            r'master.*direction',  # Master directions
                            r'regulation.*\d+'  # Regulations with numbers
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        is_document_extension = any(ext in href.lower() for ext in document_extensions)
                        
                        if is_likely_document or is_document_extension:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'document' if is_document_extension else 'reference'
                            })
    
    # Method 3: Look for specific IRDAI document patterns in div/section structures
    content_sections = soup.find_all(['div', 'section', 'article'])
    for section in content_sections:
        # Look for sections that might contain document listings
        section_text = section.get_text().lower()
        if any(keyword in section_text for keyword in ['act', 'circular', 'regulation', 'guideline']):
            links_in_section = section.find_all('a', href=True)
            for link in links_in_section:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if not href or len(link_text) < 5:
                    continue
                
                # Convert relative URLs to absolute URLs
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                # Filter for substantial document-like content
                if len(link_text) > 10 and len(link_text) < 200:
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy']
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        document_links.append({
                            'title': link_text,
                            'link': href,
                            'type': 'reference'
                        })
    
    # Remove duplicates while preserving order
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    # Sort by type: documents first, then references
    unique_document_links.sort(key=lambda x: (x['type'] != 'document', x['title']))
    
    return unique_document_links

def extract_structured_content(html_content, url):
    """Extract structured content with better parsing and document links"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract different sections
    content_sections = {}
    
    # Look for news/updates sections
    news_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ['news', 'update', 'recent', 'latest', 'whats-new']
    ))
    
    if news_sections:
        content_sections['news'] = []
        for section in news_sections:
            text = section.get_text(strip=True)
            if len(text) > 50:  # Only include substantial content
                content_sections['news'].append(text)
    
    # Extract document links using our enhanced function
    content_sections['document_links'] = extract_document_links(html_content, url)
    
    # Extract all text content
    main_text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return clean_text, content_sections

# Function to load hardcoded websites
def load_hardcoded_websites():
    """Load and process hardcoded websites"""
    loaded_docs = []
    
    for url in HARDCODED_WEBSITES:
        try:
            st.write(f"Loading URL: {url}")
            
            # Use enhanced scraping
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content(html_content, url)
                
                # Create document object
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url, "sections": sections}
                )
                loaded_docs.append(doc)
                
                # Show extracted sections
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                # Show extracted document links
                if sections.get('document_links'):
                    with st.expander(f"Document Links found from {url}"):
                        st.write(f"**Total documents found:** {len(sections['document_links'])}")
                        
                        # Group by type
                        pdf_docs = [link for link in sections['document_links'] if link['type'] == 'document']
                        ref_docs = [link for link in sections['document_links'] if link['type'] in ['reference', 'content']]
                        
                        if pdf_docs:
                            st.write("**📄 Direct Document Downloads (PDFs/Files):**")
                            for i, link_info in enumerate(pdf_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if ref_docs:
                            st.write("**🔗 Document References/Content Pages:**")
                            for i, link_info in enumerate(ref_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.write(f"No document links found from {url}")
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    return loaded_docs

# Initialize session state variables
if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False

# Streamlit UI
st.title("Web GEN-ie")

api_key = "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"

# Auto-load hardcoded websites if not already loaded
if not st.session_state['docs_loaded']:
    if st.button("Load Websites"):
        st.session_state['loaded_docs'] = load_hardcoded_websites()
        st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
        
        # Process documents if any were loaded
        if api_key and st.session_state['loaded_docs']:
            with st.spinner("Processing documents..."):
                llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
                hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # Enhanced prompt for IRDAI-specific queries with document links
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are Website expert assistant.
                    
                    IMPORTANT INSTRUCTIONS:
                    - Pay special attention to dates, recent updates, and chronological information
                    - When asked about "what's new" or recent developments, focus on the most recent information available
                    - Look for press releases, circulars, guidelines, and policy updates
                    - Provide specific details about new regulations, policy changes, or announcements
                    - If you find dated information, mention the specific dates
                    - When mentioning any acts, circulars, or regulations, try to reference the available document links
                    
                    Based on the context provided from the website(s), answer the user's question accurately and comprehensively.

                    If no context for a question is not found, or no answer is generated, the response should show: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."
                    
                    <context>
                    {context}
                    </context>
                    
                    Question: {input}
                    
                    Answer with specific details, dates, and references where available. If relevant documents are mentioned, note that direct links may be available in the sources section.
                    """
                )
                
                # Text Splitting with smaller chunks for better retrieval
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
                st.write(f"Number of chunks created: {len(document_chunks)}")
                
                # Vector database storage
                st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                
                # Create chains
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 6})  # Retrieve more chunks
                st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
                
                st.session_state['docs_loaded'] = True
                st.success("Documents processed and ready for querying!")

# Query Section
st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            response = st.session_state['retrieval_chain'].invoke({"input": query})
            
            # Get relevant documents first
            retrieved_docs = response.get('context', [])
            all_document_links = []
            
            for doc in retrieved_docs:
                # Extract document links from metadata
                if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                    for link_info in doc.metadata['sections']['document_links']:
                        if link_info not in all_document_links:
                            all_document_links.append(link_info)
            
            # Filter and rank document links based on query relevance
            relevant_docs = filter_relevant_documents(all_document_links, query, response['answer']) if all_document_links else []
            
            # Display response with relevant documents integrated
            st.subheader("Response:")
            st.write(response['answer'])
            
            # Add relevant documents as part of the normal response
            if relevant_docs:
                st.write("\n**📄 Most Relevant Documents:**")
                for i, link_info in enumerate(relevant_docs[:3]):  # Show top 3 most relevant
                    st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
