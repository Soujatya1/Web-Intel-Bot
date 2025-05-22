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

def extract_structured_content(html_content, url):
    """Extract structured content with better parsing"""
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
    
    # Extract all text content
    main_text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return clean_text, content_sections

# Initialize session state variables
if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None

# Streamlit UI
st.title("Enhanced Website Intelligence for IRDAI")

api_key = "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"

websites_input = st.text_area("Enter website URLs (one per line):")

if st.button("Load and Process"):
    website_urls = websites_input.strip().splitlines()
    st.session_state['loaded_docs'] = []
    
    for url in website_urls:
        if not url.strip():
            continue
            
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
                st.session_state['loaded_docs'].append(doc)
                
                # Show extracted sections
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
            
            st.success(f"Successfully loaded content from {url}")
            
            # Display extracted content preview
            if st.session_state['loaded_docs']:
                latest_doc = st.session_state['loaded_docs'][-1]
                with st.expander(f"Content Preview from {url}"):
                    st.write("**Metadata:**")
                    st.json(latest_doc.metadata)
                    st.write("**Content Preview (first 1000 characters):**")
                    st.text(latest_doc.page_content[:1000] + "..." if len(latest_doc.page_content) > 1000 else latest_doc.page_content)
                    st.write(f"**Total Content Length:** {len(latest_doc.page_content)} characters")
                    
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
    
    # Process documents if any were loaded
    if api_key and st.session_state['loaded_docs']:
        with st.spinner("Processing documents..."):
            llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
            hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Enhanced prompt for IRDAI-specific queries
            prompt = ChatPromptTemplate.from_template(
                """
                You are an IRDAI (Insurance Regulatory and Development Authority of India) expert assistant.
                
                IMPORTANT INSTRUCTIONS:
                - Pay special attention to dates, recent updates, and chronological information
                - When asked about "what's new" or recent developments, focus on the most recent information available
                - Look for press releases, circulars, guidelines, and policy updates
                - Provide specific details about new regulations, policy changes, or announcements
                - If you find dated information, mention the specific dates
                
                Based on the context provided from IRDAI website(s), answer the user's question accurately and comprehensively.
                
                <context>
                {context}
                </context>
                
                Question: {input}
                
                Answer with specific details, dates, and references where available.
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
            
            st.success("Documents processed and ready for querying!")

# Query Section
st.subheader("Ask Questions about IRDAI")
query = st.text_input("Enter your query:", value="What's new in IRDAI?")

show_retrieved = st.checkbox("Show retrieved documents with answer", value=True)

if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            response = st.session_state['retrieval_chain'].invoke({"input": query})
            
            st.subheader("Response:")
            st.write(response['answer'])
            
            # Show retrieved documents
            if show_retrieved and 'context' in response:
                st.subheader("Retrieved Context:")
                retrieved_docs = response.get('context', [])
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.write("**Retrieved Content:**")
                        st.write(doc.page_content)
                        if 'sections' in doc.metadata:
                            st.write("**Structured Sections:**")
                            st.json(doc.metadata['sections'])
    else:
        st.warning("Please load and process documents first.")

# Debug section
if st.checkbox("Show Debug Information"):
    if st.session_state['loaded_docs']:
        st.subheader("Debug: All Loaded Content")
        for i, doc in enumerate(st.session_state['loaded_docs']):
            with st.expander(f"Document {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                st.write("**Full Content:**")
                st.text(doc.page_content)
