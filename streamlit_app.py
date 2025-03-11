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
import pandas as pd
from urllib.parse import urljoin, urlparse
import re

# Set page title
st.title("Website PDF Extractor & Q&A System")

# Hardcoded websites to process - you can modify this list as needed
WEBSITES = [
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html"
]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "websites_processed" not in st.session_state:
    st.session_state.websites_processed = False

if "pdf_links" not in st.session_state:
    st.session_state.pdf_links = {}

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
    
    # PDF extraction settings
    st.subheader("PDF Extraction Settings")
    extraction_depth = st.slider("Link Search Depth", 0, 2, 0, 
                               help="0 = Only search main page, 1 = Follow one level of links, 2 = Follow two levels")
    filter_keywords = st.text_input("Filter PDFs by Keywords (comma-separated)", 
                                   help="Only extract PDFs with these keywords in title or URL")

# Display the hardcoded websites
st.header("Websites to Process")
for i, website in enumerate(WEBSITES, 1):
    st.write(f"{i}. {website}")

# Function to extract PDF links from any website
def extract_pdf_links(url, depth=0, max_depth=0, visited=None, keywords=None):
    if visited is None:
        visited = set()
    
    if url in visited:
        return {}
    
    visited.add(url)
    
    if keywords:
        keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
    else:
        keyword_list = []
    
    try:
        st.write(f"Scanning: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pdf_links = {}
        
        # Find all links on the page
        links = soup.find_all('a')
        for link in links:
            href = link.get('href')
            if not href:
                continue
                
            # Make the URL absolute
            full_url = urljoin(url, href)
            
            # Extract PDF links
            if href.lower().endswith('.pdf'):
                title = link.get_text().strip()
                if not title:
                    # Use the filename as title if no text
                    title = os.path.basename(href)
                
                # Apply keyword filter if specified
                if keyword_list:
                    if not any(keyword in title.lower() or keyword in full_url.lower() for keyword in keyword_list):
                        continue
                
                pdf_links[title] = full_url
            
            # Recursively follow links if depth allows
            elif depth < max_depth and full_url not in visited:
                # Only follow links within the same domain
                if urlparse(url).netloc == urlparse(full_url).netloc:
                    sub_links = extract_pdf_links(full_url, depth + 1, max_depth, visited, keywords)
                    pdf_links.update(sub_links)
        
        return pdf_links
    
    except Exception as e:
        st.warning(f"Error scanning {url}: {str(e)}")
        return {}

# Function to load and process websites
def process_websites(urls_list, max_depth=0, keywords=None):
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
                
                # Extract PDF links
                pdf_links = extract_pdf_links(url, max_depth=max_depth, keywords=keywords)
                st.session_state.pdf_links[url] = pdf_links
                st.write(f"Found {len(pdf_links)} PDF links.")
                
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
if not st.session_state.websites_processed:
    if st.button("Process Websites and Extract PDFs"):
        keywords = filter_keywords if filter_keywords else None
        process_websites(WEBSITES, max_depth=extraction_depth, keywords=keywords)

# Display extracted PDF links
if st.session_state.websites_processed and st.session_state.pdf_links:
    st.header("Extracted PDF Links")
    
    total_pdfs = sum(len(pdfs) for pdfs in st.session_state.pdf_links.values())
    st.write(f"Total PDFs found: {total_pdfs}")
    
    # Combine all PDFs into a single dataframe
    all_pdf_data = []
    for url, links in st.session_state.pdf_links.items():
        for title, pdf_url in links.items():
            all_pdf_data.append({
                "Title": title,
                "PDF Link": pdf_url,
                "Source Website": url
            })
    
    if all_pdf_data:
        all_pdf_df = pd.DataFrame(all_pdf_data)
        st.dataframe(all_pdf_df)
        
        # Add a download button for all PDF links
        csv = all_pdf_df.to_csv(index=False)
        st.download_button(
            label="Download All PDF Links as CSV",
            data=csv,
            file_name="all_pdf_links.csv",
            mime="text/csv"
        )
    else:
        st.write("No PDF links found.")

# Q&A section
st.header("Ask Questions")

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
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.write(response['answer'])
            
            # Display source links
            if sources:
                st.write("---")
                st.write("**Relevant Sources:**")
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
        
        # Add assistant response to chat history
        response_with_sources = response['answer']
        if sources:
            source_text = "\n\nRelevant Sources:\n" + "\n".join(sources)
            response_with_sources += source_text
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_sources
        })

elif not st.session_state.websites_processed:
    st.info("Please process the websites to extract PDF links and enable Q&A.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to enable Q&A functionality.")
