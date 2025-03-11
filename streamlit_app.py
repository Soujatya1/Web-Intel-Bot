import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import threading
import time

# Predefined URLs to process
PREDEFINED_URLS = [
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html", "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html"
]

# Set page title
st.title("Web-based Q&A System")

# API configuration sidebar
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    model_name = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    embedding_model = st.selectbox(
        "Select Embedding Model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    )
    
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
    
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = {}
    
if "source_documents" not in st.session_state:
    st.session_state.source_documents = {}
    
if "initial_load_done" not in st.session_state:
    st.session_state.initial_load_done = False

# Function to load and process the website in background
def process_website_background(url):
    try:
        # Load the website content
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Store source URL with each document
        for doc in documents:
            doc.metadata["source_url"] = url
        
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name=st.session_state.embedding_model)
        
        # If vectorstore exists, add to it; otherwise create new one
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore.add_documents(chunks)
        else:
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Store document chunks for reference
        for chunk in chunks:
            chunk_id = str(hash(chunk.page_content))
            st.session_state.source_documents[chunk_id] = {
                "content": chunk.page_content,
                "url": url
            }
        
        # Update processing status
        st.session_state.processed_urls[url] = "Completed"
        st.session_state.processing_status = f"Successfully processed {url}"
        
    except Exception as e:
        st.session_state.processed_urls[url] = f"Failed: {str(e)}"
        st.session_state.processing_status = f"Error processing {url}: {str(e)}"

# Automatically start processing all predefined URLs when the app loads
if not st.session_state.initial_load_done:
    st.session_state.embedding_model = embedding_model  # Store selected embedding model
    
    # Start a background thread for each URL
    for url in PREDEFINED_URLS:
        if url not in st.session_state.processed_urls:
            st.session_state.processed_urls[url] = "Processing..."
            thread = threading.Thread(target=process_website_background, args=(url,))
            thread.start()
    
    st.session_state.initial_load_done = True
    st.info("Processing URLs in the background. You can start chatting once processing is complete.")

# Display processing status
st.header("Website Processing Status")
if st.session_state.processed_urls:
    for url, status in st.session_state.processed_urls.items():
        status_icon = "✅" if status == "Completed" else "⏳" if status == "Processing..." else "❌"
        st.text(f"{status_icon} {url}: {status}")

# Q&A section
st.header("Ask Questions")

# Create the LLM-based QA chain when API key is provided
if groq_api_key and st.session_state.vectorstore is not None:
    # Set the API key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Create the ChatGroq instance
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.5,
    )
    
    # Create a conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the conversation chain with source documents
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        ),
        memory=memory,
        return_source_documents=True  # This returns the source documents used for the answer
    )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display source links if available
            if "sources" in message:
                with st.expander("Source Links"):
                    for source in message["sources"]:
                        st.markdown(f"- [{source}]({source})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the website content:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get the response from the conversation chain
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            # Extract source URLs from the response
            source_urls = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    if "source_url" in doc.metadata and doc.metadata["source_url"] not in source_urls:
                        source_urls.append(doc.metadata["source_url"])
            
            # Update conversation history
            st.session_state.chat_history.append((prompt, response['answer']))
        
        # Display assistant response in chat with source links
        with st.chat_message("assistant"):
            st.write(response['answer'])
            
            if source_urls:
                with st.expander("Source Links"):
                    for url in source_urls:
                        st.markdown(f"- [{url}]({url})")
        
        # Add assistant response to chat history with sources
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response['answer'],
            "sources": source_urls
        })

elif st.session_state.vectorstore is None and st.session_state.processed_urls:
    st.info("Website processing in progress. Please wait until processing is complete.")
elif not groq_api_key and st.session_state.vectorstore is not None:
    st.warning("Please enter your Groq API key in the sidebar.")
else:
    st.info("Website processing in progress. Chat will be available once processing is complete.")

# Add information about the application
with st.expander("About this App"):
    st.markdown("""
    This Q&A application allows you to:
    1. Automatically processes predefined websites in the background
    2. Indexes the content using FAISS vector database
    3. Ask questions about the website content
    4. Get AI-powered answers using Groq's language models with source links
    
    The application uses:
    - LangChain for the document processing pipeline
    - HuggingFace embeddings for vectorization
    - Groq for generating responses
    - Streamlit for the user interface
    
    To use:
    1. Enter your Groq API key in the sidebar
    2. Wait until website processing is complete
    3. Ask questions in the chat input
    """)
