import streamlit as st
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time

st.title("Document GeN-ie")
st.subheader("Chat with your websites")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.

The document may contain tables. Tables are formatted as CSV data and preceded by [TABLE] markers.

Question: {question} 
Context: {context} 
Answer:
"""

# Initialize embeddings and model
@st.cache_resource
def initialize_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model = ChatGroq(
        groq_api_key="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", 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.3
    )
    return embeddings, model

embeddings, model = initialize_components()

# Global variable for the vector store
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def debug_print(message, force=False):
    """Print debug messages if debug mode is enabled"""
    if debug_mode or force:
        st.write(f"🔍 DEBUG: {message}")

def load_website_simple(url):
    """Simplified website loading with extensive debugging using RecursiveUrlLoader"""
    debug_print(f"Starting to load website: {url}")
    
    try:
        # Method 1: Try RecursiveUrlLoader first
        debug_print("Trying RecursiveUrlLoader...")
        
        # Configure RecursiveUrlLoader with appropriate settings
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=1,  # Only load the main page, not recursive links
            extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
            prevent_outside_links=True,
            use_async=False,
            timeout=30,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        documents = loader.load()
        debug_print(f"RecursiveUrlLoader returned {len(documents)} documents")
        
        if documents and len(documents) > 0:
            for i, doc in enumerate(documents):
                debug_print(f"Document {i}: {len(doc.page_content)} characters")
                debug_print(f"First 200 chars: {doc.page_content[:200]}")
                doc.metadata["source"] = url
                doc.metadata["type"] = "web_content"
                
                # Clean up the content
                content = doc.page_content
                # Remove excessive whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_content = ' '.join(chunk for chunk in chunks if chunk)
                doc.page_content = clean_content
                
                debug_print(f"Cleaned document {i}: {len(doc.page_content)} characters")
        
        if not documents or all(not doc.page_content.strip() for doc in documents):
            debug_print("RecursiveUrlLoader failed or returned empty content, trying direct requests...")
            
            # Method 2: Try direct requests as fallback
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)
            
            debug_print(f"Direct extraction got {len(clean_text)} characters")
            debug_print(f"Sample text: {clean_text[:200]}")
            
            if clean_text.strip():
                documents = [Document(
                    page_content=clean_text,
                    metadata={"source": url, "type": "web_content"}
                )]
            else:
                debug_print("No content extracted from direct method either")
                return []
        
        return documents
        
    except Exception as e:
        debug_print(f"Error loading website: {str(e)}", force=True)
        st.error(f"Failed to load {url}: {str(e)}")
        return []

def split_text_simple(documents):
    """Simplified text splitting with debugging"""
    debug_print(f"Starting to split {len(documents)} documents")
    
    if not documents:
        debug_print("No documents to split")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_splits = []
    for i, doc in enumerate(documents):
        debug_print(f"Splitting document {i} with {len(doc.page_content)} characters")
        
        if len(doc.page_content.strip()) < 50:
            debug_print(f"Document {i} too short, skipping")
            continue
            
        try:
            splits = text_splitter.split_documents([doc])
            debug_print(f"Document {i} split into {len(splits)} chunks")
            
            for j, split in enumerate(splits):
                split.metadata = doc.metadata.copy()
                split.metadata["chunk_id"] = f"{i}_{j}"
                debug_print(f"Chunk {i}_{j}: {len(split.page_content)} chars")
                
            all_splits.extend(splits)
        except Exception as e:
            debug_print(f"Error splitting document {i}: {str(e)}")
            # Add original doc if splitting fails
            all_splits.append(doc)
    
    debug_print(f"Total chunks created: {len(all_splits)}")
    return all_splits

def create_vector_store(documents):
    """Create vector store with debugging"""
    debug_print(f"Creating vector store with {len(documents)} documents")
    
    if not documents:
        debug_print("No documents to index")
        return False
    
    # Filter out empty documents
    valid_docs = []
    for i, doc in enumerate(documents):
        if doc.page_content.strip():
            valid_docs.append(doc)
            debug_print(f"Valid doc {i}: {len(doc.page_content)} chars")
        else:
            debug_print(f"Skipping empty doc {i}")
    
    if not valid_docs:
        debug_print("No valid documents after filtering")
        return False
    
    try:
        debug_print("Creating FAISS index...")
        st.session_state.vector_store = FAISS.from_documents(valid_docs, embeddings)
        debug_print(f"Successfully created vector store with {len(valid_docs)} documents")
        return True
    except Exception as e:
        debug_print(f"Error creating vector store: {str(e)}", force=True)
        st.error(f"Failed to create vector store: {str(e)}")
        return False

def search_documents(query, k=4):
    """Search documents with debugging"""
    debug_print(f"Searching for: '{query}'")
    
    if st.session_state.vector_store is None:
        debug_print("No vector store available")
        return []
    
    try:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        debug_print(f"Found {len(docs)} relevant documents")
        
        for i, doc in enumerate(docs):
            debug_print(f"Result {i}: {len(doc.page_content)} chars from {doc.metadata.get('source', 'unknown')}")
            debug_print(f"Preview: {doc.page_content[:100]}...")
        
        return docs
    except Exception as e:
        debug_print(f"Error searching documents: {str(e)}", force=True)
        return []

def generate_answer(question, context_docs):
    """Generate answer with debugging"""
    debug_print(f"Generating answer for: '{question}'")
    debug_print(f"Using {len(context_docs)} context documents")
    
    if not context_docs:
        return "I don't have any relevant information to answer your question. Please make sure you've processed some websites first."
    
    # Combine contexts
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    debug_print(f"Combined context length: {len(context)} characters")
    debug_print(f"Context preview: {context[:300]}...")
    
    if not context.strip():
        return "The retrieved context appears to be empty. Please try processing the websites again."
    
    try:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        debug_print("Calling LLM...")
        response = chain.invoke({"question": question, "context": context})
        
        debug_print(f"LLM response length: {len(response.content) if response.content else 0}")
        debug_print(f"Response preview: {response.content[:200] if response.content else 'None'}...")
        
        if not response.content or not response.content.strip():
            return "I received an empty response from the AI model. Please try rephrasing your question."
        
        return response.content
        
    except Exception as e:
        debug_print(f"Error generating answer: {str(e)}", force=True)
        st.error(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Website processing interface
st.header("Process Website")
website_url = st.text_input("Enter website URL:", placeholder="https://example.com")

if st.button("🔄 Process Website", type="primary"):
    if not website_url:
        st.warning("Please enter a website URL")
    else:
        # Add protocol if missing
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        with st.spinner(f"Processing {website_url}..."):
            # Step 1: Load website
            st.write("📥 Loading website content...")
            documents = load_website_simple(website_url)
            
            if not documents:
                st.error("❌ Failed to load website content")
                st.stop()
            
            st.success(f"✅ Loaded {len(documents)} documents from website")
            
            # Step 2: Split into chunks
            st.write("✂️ Splitting into chunks...")
            chunks = split_text_simple(documents)
            
            if not chunks:
                st.error("❌ Failed to create text chunks")
                st.stop()
            
            st.success(f"✅ Created {len(chunks)} text chunks")
            
            # Step 3: Create vector store
            st.write("🔍 Creating searchable index...")
            if create_vector_store(chunks):
                st.success("✅ Successfully created searchable index!")
                st.session_state.documents_processed = True
                
                # Show preview
                with st.expander("📋 Preview processed content"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.write(f"**Chunk {i+1}:**")
                        st.text(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
                        st.divider()
            else:
                st.error("❌ Failed to create searchable index")

# Test search functionality
if st.session_state.documents_processed:
    st.header("🔍 Test Search")
    test_query = st.text_input("Enter a test search query:")
    if st.button("Search") and test_query:
        results = search_documents(test_query)
        st.write(f"Found {len(results)} results:")
        for i, doc in enumerate(results):
            with st.expander(f"Result {i+1}"):
                st.text(doc.page_content[:500])

# Chat interface
st.header("💬 Chat")
if st.session_state.documents_processed:
    question = st.chat_input("Ask me anything about the processed website...")
    
    if question:
        # Add user message
        st.session_state.conversation_history.append({"role": "user", "content": question})
        
        # Search for relevant documents
        with st.spinner("Searching for relevant information..."):
            relevant_docs = search_documents(question)
        
        # Generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(question, relevant_docs)
        
        # Add assistant message
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
else:
    st.info("👆 Please process a website first before asking questions")

# Display conversation
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear conversation button
if st.session_state.conversation_history:
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.experimental_rerun()
