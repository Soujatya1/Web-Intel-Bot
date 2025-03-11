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

# Set page title
st.title("Web-based Q&A System")

# Hardcoded website options
websites = {
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html"
}

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
    
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Web content selection section
st.header("Select Web Content")
selected_website = st.selectbox("Choose a website to analyze:", list(websites.keys()))
web_url = websites[selected_website]
st.write(f"Selected URL: {web_url}")
process_button = st.button("Process Website")

# Function to load and process the website
def process_website(url):
    with st.spinner("Loading and processing the website..."):
        try:
            # Load the website content
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Split the content into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.success(f"Successfully processed {url}. You can now ask questions!")
            return True
        
        except Exception as e:
            st.error(f"Error processing the website: {str(e)}")
            return False

# Process website if button is clicked
if process_button:
    process_website(web_url)

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
    
    # Create the conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory
    )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
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
            
            # Update conversation history
            st.session_state.chat_history.append((prompt, response['answer']))
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.write(response['answer'])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

elif st.session_state.vectorstore is None:
    st.info("Process a website first before asking questions.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
