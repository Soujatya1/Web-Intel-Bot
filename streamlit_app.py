import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
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
from sklearn.metrics.pairwise import cosine_similarity
import json

# Configuration
HARDCODED_WEBSITES = [
    "https://irdai.gov.in/acts",
    "https://irdai.gov.in/home",
    "https://irdai.gov.in/rules",
    "https://irdai.gov.in/consolidated-gazette-notified-regulations",
    "https://irdai.gov.in/updated-regulations",
    "https://irdai.gov.in/notifications",
    "https://irdai.gov.in/circulars",
    "https://irdai.gov.in/guidelines",
    "https://irdai.gov.in/orders1",
    "https://enforcementdirectorate.gov.in/fema",
    "https://enforcementdirectorate.gov.in/fema?page=1",
    "https://enforcementdirectorate.gov.in/fema?page=2"
]

# Enhanced System Prompt with Explicit Citation Requirements
SYSTEM_PROMPT_TEMPLATE = """
You are a specialized assistant for IRDAI, UIDAI, PMLA and egazette websites.

**CRITICAL CITATION REQUIREMENTS:**
1. ALWAYS extract and include the "Source URL:" from each context chunk
2. ALWAYS include document links when they appear in the context
3. Format sources as: "**Source:** [URL]"
4. Format document links as: "[Link Title](URL)"
5. End your response with a "**Sources Used:**" section listing all source URLs

**RESPONSE STRUCTURE:**
- Answer the question using the provided context
- Include inline citations with source URLs
- When mentioning acts, regulations, or documents, include the relevant links
- Focus on recent information when asked about "latest" or "new" developments
- Always end with "**Sources Used:**" section

**IMPORTANT:**
- If PII data detected, respond: "This query contains PII data which is outside my scope."
- Each context chunk starts with "Source URL:" - YOU MUST use this information

**Context:** {context}
**Question:** {input}

**Your Response Must Include Sources and Links:**
"""

RELEVANCE_SCORE_THRESHOLD = 0.1

# Utility Functions
def enhanced_web_scrape(url):
    """Enhanced web scraping with proper headers"""
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
        st.error(f"Scraping failed for {url}: {e}")
        return None

def extract_document_links(html_content, url):
    """Extract document links with improved filtering"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                        'amendment', 'notification', 'insurance', 'policy', 'aadhaar',
                        'master direction', 'gazette', 'pmla', 'fema']
    
    # Extract all links
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href')
        link_text = link.get_text(strip=True)
        
        if not href or len(link_text) < 3:
            continue
            
        # Convert relative URLs to absolute
        if href.startswith('/'):
            href = urljoin(url, href)
        elif not href.startswith(('http://', 'https://')):
            href = urljoin(url, href)
        
        # Check if link is relevant
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if has_doc_keywords and 5 < len(link_text) < 200:
            document_links.append({
                'title': link_text.strip(),
                'link': href,
                'type': 'document'
            })
    
    # Extract from tables (common in government sites)
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
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
                    
                    has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
                    
                    if has_doc_keywords and 5 < len(link_text) < 200:
                        document_links.append({
                            'title': link_text.strip(),
                            'link': href,
                            'type': 'table_document'
                        })
    
    # Remove duplicates
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links and len(link_info['title']) > 5:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    return unique_document_links[:15]  # Limit to prevent overwhelming

def create_enhanced_content(html_content, url):
    """Create enhanced content with embedded links"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract document links
    document_links = extract_document_links(html_content, url)
    
    # Get main text content
    main_text = soup.get_text()
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 10)
    
    # Create source header with embedded links
    source_header = f"Source URL: {url}"
    
    if document_links:
        source_header += "\nDocument Links Available:\n"
        for i, link in enumerate(document_links[:10], 1):  # Limit display
            source_header += f"{i}. {link['title']} - {link['link']}\n"
    
    # Combine content
    enhanced_content = f"{source_header}\n\n--- MAIN CONTENT ---\n{clean_text}"
    
    return enhanced_content, document_links

def load_websites():
    """Load all hardcoded websites with enhanced content processing"""
    loaded_docs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(HARDCODED_WEBSITES):
        try:
            status_text.text(f"Loading: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                enhanced_content, document_links = create_enhanced_content(html_content, url)
                
                doc = Document(
                    page_content=enhanced_content,
                    metadata={
                        "source": url,
                        "document_links": document_links,
                        "total_links": len(document_links),
                        "has_source_url": True
                    }
                )
                loaded_docs.append(doc)
                
                # Show progress
                with st.expander(f"✅ Loaded: {url} ({len(document_links)} links found)"):
                    if document_links:
                        st.write("**Sample Document Links:**")
                        for link in document_links[:5]:
                            st.write(f"- [{link['title']}]({link['link']})")
                    else:
                        st.write("No document links found")
            else:
                st.error(f"Failed to load: {url}")
                
            progress_bar.progress((i + 1) / len(HARDCODED_WEBSITES))
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    status_text.text(f"✅ Loaded {len(loaded_docs)} documents successfully!")
    return loaded_docs

def relevance_score(query, document, embeddings):
    """Calculate relevance score for document ranking"""
    try:
        query_embedding = embeddings.embed_query(query)
        document_embedding = embeddings.embed_documents([document.page_content])[0]
        similarity = cosine_similarity([query_embedding], [document_embedding])[0][0]
        
        # Keyword matching bonus
        keywords = query.lower().split()
        keyword_matches = sum(1 for keyword in keywords if keyword in document.page_content.lower())
        keyword_bonus = keyword_matches * 0.1
        
        return similarity + keyword_bonus
    except Exception:
        # Fallback to keyword matching
        keywords = query.lower().split()
        keyword_matches = sum(1 for keyword in keywords if keyword in document.page_content.lower())
        return keyword_matches * 0.2

def rerank_documents(query, documents, embeddings):
    """Rerank documents based on relevance"""
    if not documents:
        return []
    
    try:
        scored_docs = [(doc, relevance_score(query, doc, embeddings)) for doc in documents]
        scored_docs = [(doc, score) for doc, score in scored_docs if score >= RELEVANCE_SCORE_THRESHOLD]
        
        if not scored_docs:
            return documents[:6]
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:6]]
        
    except Exception as e:
        st.warning(f"Reranking failed: {e}. Using original order.")
        return documents[:6]

def custom_retrieval_with_enhanced_context(query_dict):
    """Custom retrieval function that ensures source URLs and links are preserved"""
    query = query_dict["input"]
    
    # Get initial documents
    raw_retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 20})
    raw_docs = raw_retriever.get_relevant_documents(query)
    
    # Rerank documents
    reranked_docs = rerank_documents(query, raw_docs, st.session_state.get('hf_embedding'))
    
    # Ensure all documents have proper source formatting
    enhanced_docs = []
    for doc in reranked_docs:
        if not doc.page_content.startswith("Source URL:"):
            # Add source URL if missing
            source_url = doc.metadata.get('source', 'Unknown Source')
            enhanced_content = f"Source URL: {source_url}\n\n{doc.page_content}"
            doc.page_content = enhanced_content
        
        enhanced_docs.append(doc)
    
    # Create document chain and generate response
    document_chain = create_stuff_documents_chain(st.session_state['llm'], st.session_state['prompt'])
    result = document_chain.invoke({"input": query, "context": enhanced_docs})
    
    return {"answer": result, "context": enhanced_docs}

def extract_sources_and_links(response_text, context_docs):
    """Extract and display sources and links from response and context"""
    sources = set()
    document_links = []
    
    # Extract from context documents
    for doc in context_docs:
        content = doc.page_content
        
        # Extract source URL
        if "Source URL:" in content:
            lines = content.split('\n')
            for line in lines:
                if line.startswith("Source URL:"):
                    source_url = line.replace("Source URL:", "").strip()
                    sources.add(source_url)
                    break
        
        # Extract document links
        if "Document Links Available:" in content:
            lines = content.split('\n')
            in_links_section = False
            for line in lines:
                if "Document Links Available:" in line:
                    in_links_section = True
                    continue
                if in_links_section and line.strip():
                    if line.startswith("--- MAIN CONTENT ---"):
                        break
                    # Parse link format: "1. Title - URL"
                    match = re.search(r'\d+\.\s*(.+?)\s*-\s*(https?://[^\s]+)', line)
                    if match:
                        title, url = match.groups()
                        document_links.append({"title": title.strip(), "url": url.strip()})
        
        # Also check metadata
        meta_source = doc.metadata.get('source')
        if meta_source:
            sources.add(meta_source)
        
        meta_links = doc.metadata.get('document_links', [])
        for link in meta_links:
            document_links.append({"title": link.get('title', ''), "url": link.get('link', '')})
    
    return list(sources), document_links

def display_enhanced_response(response, show_debug=False):
    """Display response with proper source citations and links"""
    response_text = response['answer']
    context_docs = response.get('context', [])
    
    st.subheader("📋 Response:")
    st.write(response_text)
    
    # Extract sources and links
    sources, doc_links = extract_sources_and_links(response_text, context_docs)
    
    # Display sources
    if sources:
        st.write("\n**📍 Information Sources:**")
        for i, source in enumerate(sorted(sources), 1):
            st.markdown(f"{i}. [{source}]({source})")
    else:
        st.warning("⚠️ No sources were found in the context")
    
    # Display document links
    if doc_links:
        st.write("\n**📄 Related Document Links:**")
        unique_links = {}
        for link in doc_links:
            if link['url'] and link['title']:
                unique_links[link['url']] = link['title']
        
        for i, (url, title) in enumerate(unique_links.items(), 1):
            st.markdown(f"{i}. [{title}]({url})")
    else:
        st.info("ℹ️ No document links found in the context")
    
    # Debug information
    if show_debug:
        st.subheader("🔍 Debug Information")
        st.write(f"**Context documents:** {len(context_docs)}")
        docs_with_sources = sum(1 for doc in context_docs if "Source URL:" in doc.page_content)
        docs_with_links = sum(1 for doc in context_docs if "Document Links Available:" in doc.page_content)
        st.write(f"**Docs with source URLs:** {docs_with_sources}")
        st.write(f"**Docs with document links:** {docs_with_links}")
        
        for i, doc in enumerate(context_docs):
            with st.expander(f"Context Doc {i+1} Preview"):
                preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                st.code(preview)

# Streamlit App
def main():
    # Initialize session state
    if 'docs_loaded' not in st.session_state:
        st.session_state['docs_loaded'] = False
    if 'loaded_docs' not in st.session_state:
        st.session_state['loaded_docs'] = []
    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = None
    if 'retrieval_chain' not in st.session_state:
        st.session_state['retrieval_chain'] = None
    if 'llm' not in st.session_state:
        st.session_state['llm'] = None
    if 'hf_embedding' not in st.session_state:
        st.session_state['hf_embedding'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = None

    st.title("🌐 Web GEN-ie")
    st.markdown("*Enhanced RAG System for Government Websites with Source Citations*")

    # Configuration Section
    st.subheader("🔧 Azure OpenAI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint:", 
            placeholder="https://your-resource.openai.azure.com/",
            help="Your Azure OpenAI resource endpoint URL"
        )
        
        api_key = st.text_input(
            "Azure OpenAI API Key:", 
            type="password",
            placeholder="Enter your Azure OpenAI API key here...",
            help="Your Azure OpenAI API key"
        )

    with col2:
        deployment_name = st.text_input(
            "Deployment Name:", 
            placeholder="gpt-4o",
            help="The name of your deployed model"
        )
        
        api_version = st.selectbox(
            "API Version:",
            ["2025-01-01-preview", "2024-10-21-preview", "2024-08-01-preview"],
            index=0,
            help="Azure OpenAI API version"
        )

    config_complete = all([azure_endpoint, api_key, deployment_name, api_version])

    if not config_complete:
        st.warning("⚠️ Please fill in all Azure OpenAI configuration fields to proceed.")
        return

    # Document Loading Section
    st.subheader("📚 Document Loading")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state['docs_loaded']:
            st.success(f"✅ {len(st.session_state['loaded_docs'])} documents loaded and ready!")
        else:
            st.info("Click 'Load Websites' to begin document processing")
    
    with col2:
        load_button = st.button("🔄 Load Websites", disabled=st.session_state['docs_loaded'])
    
    with col3:
        if st.session_state['docs_loaded']:
            reset_button = st.button("🗑️ Reset")
            if reset_button:
                # Reset all session state
                for key in ['docs_loaded', 'loaded_docs', 'vector_db', 'retrieval_chain', 'llm', 'hf_embedding', 'prompt']:
                    st.session_state[key] = None if key != 'docs_loaded' else False
                    if key == 'loaded_docs':
                        st.session_state[key] = []
                st.experimental_rerun()

    # Load websites if button clicked
    if load_button and not st.session_state['docs_loaded']:
        with st.spinner("🔍 Loading and processing websites..."):
            try:
                # Load documents
                st.session_state['loaded_docs'] = load_websites()
                
                if not st.session_state['loaded_docs']:
                    st.error("❌ No documents were loaded successfully.")
                    return
                
                # Initialize components
                st.subheader("🤖 Initializing AI Components")
                
                # Initialize LLM
                with st.spinner("Initializing Azure OpenAI..."):
                    llm = AzureChatOpenAI(
                        azure_endpoint=azure_endpoint,
                        api_key=api_key,
                        azure_deployment=deployment_name,
                        api_version=api_version,
                        temperature=0.0,
                        top_p=0.1
                    )
                    st.session_state['llm'] = llm
                    st.success("✅ Azure OpenAI initialized")
                
                # Initialize embeddings
                with st.spinner("Loading embeddings model..."):
                    hf_embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
                    st.session_state['hf_embedding'] = hf_embedding
                    st.success("✅ Embeddings model loaded")
                
                # Create prompt template
                prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                st.session_state['prompt'] = prompt
                
                # Process documents
                with st.spinner("Creating document chunks and vector database..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1200,
                        chunk_overlap=200,
                        length_function=len
                    )
                    
                    document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
                    st.write(f"📄 Created {len(document_chunks)} document chunks")
                    
                    # Verify chunks have source URLs
                    chunks_with_sources = sum(1 for chunk in document_chunks if "Source URL:" in chunk.page_content)
                    st.success(f"✅ {chunks_with_sources}/{len(document_chunks)} chunks contain source URLs")
                    
                    # Create vector database
                    st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                    st.success("✅ Vector database created")
                
                # Set up retrieval chain
                st.session_state['retrieval_chain'] = custom_retrieval_with_enhanced_context
                st.session_state['docs_loaded'] = True
                
                st.success("🎉 All components initialized successfully! Ready for queries.")
                
            except Exception as e:
                st.error(f"❌ Error during initialization: {e}")
                st.error("Please check your configuration and try again.")
                return

    # Query Section
    if st.session_state['docs_loaded']:
        st.subheader("❓ Ask Questions")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter your question:",
                value="What are the recent Insurance Acts and amendments?",
                placeholder="Ask about regulations, acts, guidelines, or recent updates..."
            )
        
        with col2:
            show_debug = st.checkbox("🔍 Show Debug Info", value=False)
        
        if st.button("🔍 Get Answer") and query:
            with st.spinner("🤖 Processing your question..."):
                try:
                    response = st.session_state['retrieval_chain']({"input": query})
                    display_enhanced_response(response, show_debug=show_debug)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    st.error("Please try rephrasing your question or check the configuration.")

if __name__ == "__main__":
    main()
