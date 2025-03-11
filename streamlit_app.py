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

if "pdf_metadata" not in st.session_state:
    st.session_state.pdf_metadata = {}

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
        st.session_state.pdf_links = {}
        st.session_state.pdf_metadata = {}
        st.session_state.vectorstore = None
        st.experimental_rerun()

# Function to extract PDF links and metadata from any website
def extract_pdf_links(url, visited=None):
    if visited is None:
        visited = set()
    
    if url in visited:
        return {}
    
    visited.add(url)
    
    try:
        st.write(f"Scanning: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pdf_links = {}
        pdf_metadata = {}
        
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
                
                pdf_links[title] = full_url
                
                # Extract surrounding text for context (up to 300 characters)
                parent = link.parent
                context = parent.get_text().strip()
                # Limit context length
                context = context[:300] + ('...' if len(context) > 300 else '')
                
                # Store metadata
                pdf_metadata[full_url] = {
                    'title': title,
                    'context': context,
                    'source_url': url,
                    'keywords': extract_keywords(title + " " + context)
                }
                
                # Try to identify if it's a circular/regulation
                is_circular = any(term in title.lower() or term in context.lower() 
                                for term in ['circular', 'regulation', 'policy', 
                                             'guideline', 'notice', 'notification',
                                             'directive', 'rule', 'act', 'law'])
                pdf_metadata[full_url]['is_circular'] = is_circular
        
        return pdf_links, pdf_metadata
    
    except Exception as e:
        st.warning(f"Error scanning {url}: {str(e)}")
        return {}, {}

# Function to extract keywords from text
def extract_keywords(text):
    # Simple keyword extraction - split by spaces and remove common words
    common_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'by'}
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in common_words and len(word) > 2]
    return keywords

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
                
                # Extract PDF links and metadata
                pdf_links, pdf_metadata = extract_pdf_links(url)
                st.session_state.pdf_links[url] = pdf_links
                st.session_state.pdf_metadata.update(pdf_metadata)
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
                
                # Create chunks for PDF information to include in vector search
                for pdf_url, metadata in pdf_metadata.items():
                    if metadata['title'] and metadata['context']:
                        pdf_content = f"PDF Document: {metadata['title']}\nContext: {metadata['context']}"
                        from langchain.schema import Document
                        pdf_doc = Document(
                            page_content=pdf_content,
                            metadata={
                                'source': url,
                                'pdf_url': pdf_url,
                                'is_pdf': True,
                                'title': metadata['title'],
                                'is_circular': metadata.get('is_circular', False)
                            }
                        )
                        chunks.append(pdf_doc)
            
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
    if st.button("Process Websites and Extract PDFs"):
        process_websites(st.session_state.websites)
        # Add a simple confirmation if processing was successful
        if st.session_state.websites_processed:
            pdf_count = sum(len(links) for links in st.session_state.pdf_links.values())
            circular_count = sum(1 for metadata in st.session_state.pdf_metadata.values() 
                               if metadata.get('is_circular', False))
            st.success(f"Found {pdf_count} PDF files across all websites, including {circular_count} potential circulars/regulations.")

# Function to get relevant sources for a query
def get_relevant_sources(query, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    sources = []
    pdf_sources = []
    
    for doc in relevant_docs:
        if 'source' in doc.metadata:
            source_url = doc.metadata['source']
            
            # Check if this is a PDF reference
            if doc.metadata.get('is_pdf', False) and 'pdf_url' in doc.metadata:
                pdf_url = doc.metadata['pdf_url']
                title = doc.metadata.get('title', 'PDF Document')
                is_circular = doc.metadata.get('is_circular', False)
                
                pdf_sources.append({
                    'url': pdf_url,
                    'title': title,
                    'source_website': source_url,
                    'is_circular': is_circular
                })
            elif source_url not in sources:
                sources.append(source_url)
    
    return sources, pdf_sources

# Improved function to find relevant PDF links for a query
def get_relevant_pdfs(query, pdf_metadata, max_results=5):
    query_terms = set(query.lower().split())
    # Add terms related to circulars/regulations if the query seems to be about regulations
    regulation_terms = {'circular', 'regulation', 'policy', 'guideline', 'notice', 
                       'notification', 'directive', 'rule', 'act', 'law'}
    
    is_regulation_query = any(term in query_terms for term in regulation_terms)
    
    scored_pdfs = []
    
    # Score each PDF by relevance to the query
    for pdf_url, metadata in pdf_metadata.items():
        score = 0
        pdf_text = metadata['title'] + " " + metadata['context']
        pdf_terms = set(pdf_text.lower().split())
        
        # Score based on keyword matches
        matching_terms = query_terms.intersection(pdf_terms)
        score += len(matching_terms) * 2  # Higher weight for direct matches
        
        # Boost score for PDFs that are circulars/regulations if query is about regulations
        if is_regulation_query and metadata.get('is_circular', False):
            score += 5
        
        # Check for specific keywords in PDF metadata
        for keyword in metadata.get('keywords', []):
            if any(term in keyword for term in query_terms):
                score += 1
        
        if score > 0:
            scored_pdfs.append((score, pdf_url, metadata))
    
    # Sort by score (descending) and take top results
    scored_pdfs.sort(reverse=True)
    
    relevant_pdfs = []
    for _, pdf_url, metadata in scored_pdfs[:max_results]:
        relevant_pdfs.append({
            'title': metadata['title'],
            'url': pdf_url,
            'website': metadata['source_url'],
            'is_circular': metadata.get('is_circular', False)
        })
    
    return relevant_pdfs

# Function to prepare the prompt for the LLM with PDF awareness
def prepare_prompt_with_pdf_context(query, relevant_pdfs):
    if not relevant_pdfs:
        return query
    
    # Enhance the prompt with PDF context
    pdf_context = "I found these potentially relevant documents:\n"
    for i, pdf in enumerate(relevant_pdfs[:3], 1):  # Limit to top 3
        pdf_type = "circular/regulation" if pdf.get('is_circular') else "document"
        pdf_context += f"{i}. {pdf['title']} ({pdf_type})\n"
    
    enhanced_prompt = f"{pdf_context}\n\nBased on the information above and your knowledge, please answer: {query}"
    return enhanced_prompt

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
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the websites:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get relevant sources before the chain call
        sources, pdf_sources = get_relevant_sources(prompt, st.session_state.vectorstore, k=5)
        
        # Find relevant PDFs for the query
        keyword_relevant_pdfs = get_relevant_pdfs(prompt, st.session_state.pdf_metadata)
        
        # Combine PDFs from vector search and keyword search (avoiding duplicates)
        all_pdf_urls = set()
        relevant_pdfs = []
        
        # First add PDFs found through vector search
        for pdf in pdf_sources:
            if pdf['url'] not in all_pdf_urls:
                relevant_pdfs.append(pdf)
                all_pdf_urls.add(pdf['url'])
        
        # Then add PDFs found through keyword search
        for pdf in keyword_relevant_pdfs:
            if pdf['url'] not in all_pdf_urls:
                relevant_pdfs.append(pdf)
                all_pdf_urls.add(pdf['url'])
        
        # Prepare enhanced prompt with PDF context
        enhanced_prompt = prepare_prompt_with_pdf_context(prompt, relevant_pdfs)
        
        # Get the response from the conversation chain
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({
                "question": enhanced_prompt,
                "chat_history": st.session_state.chat_history
            })
            
            # Get source documents from the response
            source_docs = response.get('source_documents', [])
            
            # Extract source URLs from returned documents
            sources_from_response = []
            for doc in source_docs:
                if 'source' in doc.metadata:
                    source_url = doc.metadata['source']
                    if source_url not in sources_from_response:
                        sources_from_response.append(source_url)
            
            # If no sources from response, use the pre-fetched relevant sources
            if not sources_from_response:
                sources_from_response = sources
            
            # Prepare the answer
            answer = response['answer']
            
            # Identify if the query is about circulars/regulations
            regulation_terms = {'circular', 'regulation', 'policy', 'guideline', 'notice', 
                               'notification', 'directive', 'rule', 'act', 'law'}
            query_terms = set(prompt.lower().split())
            is_regulation_query = any(term in query_terms for term in regulation_terms)
            
            # Filter PDFs to prioritize circulars/regulations for regulation queries
            if is_regulation_query:
                circular_pdfs = [pdf for pdf in relevant_pdfs if pdf.get('is_circular', False)]
                if circular_pdfs:
                    relevant_pdfs = circular_pdfs
            
            # Add PDF references to the answer
            if relevant_pdfs:
                answer += "\n\n## Relevant Documents\n"
                for i, pdf in enumerate(relevant_pdfs[:5], 1):  # Limit to top 5
                    doc_type = "📜 Circular/Regulation" if pdf.get('is_circular') else "📄 Document"
                    answer += f"{i}. [{pdf['title']}]({pdf['url']}) - {doc_type}\n"
        
        # Display assistant response in chat
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Display source links
            if sources_from_response:
                st.markdown("---")
                st.markdown("**Website Sources:**")
                for i, source in enumerate(sources_from_response, 1):
                    st.markdown(f"{i}. [{source}]({source})")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer
        })

elif not st.session_state.websites:
    st.info("Please add websites to process.")
elif not st.session_state.websites_processed:
    st.info("Please process the websites to extract PDF links and enable Q&A.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to enable Q&A functionality.")
