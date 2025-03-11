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
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Set page title
st.title("Website Q&A System")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "websites_processed" not in st.session_state:
    st.session_state.websites_processed = False

if "websites" not in st.session_state:
    st.session_state.websites = []

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

# User input for websites
st.header("Enter Websites to Process")
website_input = st.text_area("Enter website URLs (one per line):")
add_website_button = st.button("Add Websites")

if add_website_button and website_input:
    new_websites = [url.strip() for url in website_input.split('\n') if url.strip()]
    st.session_state.websites.extend(new_websites)
    st.success(f"Added {len(new_websites)} websites")

# Display added websites
if st.session_state.websites:
    st.header("Websites to Process")
    for i, website in enumerate(st.session_state.websites, 1):
        st.write(f"{i}. {website}")
        
    # Option to clear the list
    if st.button("Clear Websites List"):
        st.session_state.websites = []
        st.session_state.websites_processed = False
        st.session_state.vectorstore = None
        st.experimental_rerun()

# Function to load and process websites
def process_websites(urls_list):
    with st.spinner("Loading and processing websites... This may take a few minutes."):
        try:
            all_chunks = []
            
            for url in urls_list:
                st.write(f"Processing: {url}")
                
                # Create a simple name for the website based on domain and path
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if path_parts:
                    webpage_id = f"{domain}_{path_parts[-1]}"
                else:
                    webpage_id = domain
                
                # Load the website content
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                # Make sure each document has metadata with source URL
                for doc in documents:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = url
                
                # Split the content into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.session_state.websites_processed = True
            st.success(f"Successfully processed all websites. You can now ask questions!")
            return True
        
        except Exception as e:
            st.error(f"Error processing websites: {str(e)}")
            return False

# Button to process websites
if st.session_state.websites and not st.session_state.websites_processed:
    if st.button("Process Websites"):
        process_websites(st.session_state.websites)

# Function to get relevant sources for a query
def get_relevant_sources(query, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    sources = []
    for doc in relevant_docs:
        if 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    return sources

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
    
    # Create a conversation memory with explicit output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly set which output to store
    )
    
    # Create the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True  # This ensures source docs are returned
    )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the websites:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get relevant sources before the chain call
        relevant_sources = get_relevant_sources(prompt, st.session_state.vectorstore, k=3)
        
        # Get the response from the conversation chain
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            # Get source documents from the response
            source_docs = response.get('source_documents', [])
            
            # Extract source URLs from returned documents
            sources = []
            for doc in source_docs:
                if 'source' in doc.metadata:
                    source_url = doc.metadata['source']
                    if source_url not in sources:
                        sources.append(source_url)
            
            # If no sources from response, use the pre-fetched relevant sources
            if not sources:
                sources = relevant_sources
            
            # Get the answer
            answer = response['answer']
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.write(answer)
            
            # Display source links
            if sources:
                st.write("---")
                st.write("**Sources:**")
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
        
        # Add assistant response to chat history
        response_with_sources = answer
        if sources:
            source_text = "\n\nSources:\n" + "\n".join(sources)
            response_with_sources += source_text
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_sources
        })

elif not st.session_state.websites:
    st.info("Please add websites to process.")
elif not st.session_state.websites_processed:
    st.info("Please process the websites to enable Q&A.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to enable Q&A functionality.")
