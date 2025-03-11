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

st.title("Website Q&A System")

# Hardcoded list of websites to process
WEBSITES = [
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html", "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html"
]

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "websites_processed" not in st.session_state:
    st.session_state.websites_processed = False

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

# Display the hardcoded websites
st.header("Websites to Process")
for i, website in enumerate(WEBSITES, 1):
    st.write(f"{i}. {website}")

def process_websites(urls_list):
    with st.spinner("Loading and processing websites... This may take a few minutes."):
        try:
            all_chunks = []
            
            for url in urls_list:
                st.write(f"Processing: {url}")
                
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
                
                for doc in documents:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = url
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
            
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.session_state.websites_processed = True
            st.success(f"Successfully processed all websites. You can now ask questions!")
            return True
        
        except Exception as e:
            st.error(f"Error processing websites: {str(e)}")
            return False

if not st.session_state.websites_processed:
    if st.button("Process Websites"):
        process_websites(WEBSITES)

def get_relevant_sources(query, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    sources = []
    for doc in relevant_docs:
        if 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    return sources

st.header("Ask Questions")

if groq_api_key and st.session_state.vectorstore is not None:
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.5,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about the websites:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        relevant_sources = get_relevant_sources(prompt, st.session_state.vectorstore, k=3)
        
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            source_docs = response.get('source_documents', [])
            
            sources = []
            for doc in source_docs:
                if 'source' in doc.metadata:
                    source_url = doc.metadata['source']
                    if source_url not in sources:
                        sources.append(source_url)
            
            if not sources:
                sources = relevant_sources
            
            answer = response['answer']
        
        with st.chat_message("assistant"):
            st.write(answer)
            
            if sources:
                st.write("---")
                st.write("**Sources:**")
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
        
        response_with_sources = answer
        if sources:
            source_text = "\n\nSources:\n" + "\n".join(sources)
            response_with_sources += source_text
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_sources
        })

elif not st.session_state.websites_processed:
    st.info("Please process the websites to enable Q&A.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to enable Q&A functionality.")
