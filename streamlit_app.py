import streamlit as st
import requests
import os
import re
import faiss
import numpy as np
import tempfile
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set page configuration
st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

# Define the list of websites to query
WEBSITES = [
    "https://python.org",
    "https://docs.python.org",
    "https://streamlit.io",
    "https://langchain.com",
    "https://docs.groq.com",
    "https://github.com",
    "https://huggingface.co",
    "https://pytorch.org",
    "https://tensorflow.org",
    "https://keras.io",
    "https://scikit-learn.org",
    "https://pandas.pydata.org",
    "https://numpy.org",
    "https://matplotlib.org",
    "https://scipy.org",
    "https://fastapi.tiangolo.com",
    "https://flask.palletsprojects.com",
    "https://django-project.com",
    "https://kaggle.com",
    "https://paperswithcode.com"
]

# Define the cache directory
CACHE_DIR = ".web_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Create a class for sentence transformer embeddings compatible with LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

# Functions for web scraping and content extraction
def fetch_website_content(url: str) -> Tuple[str, List[Dict]]:
    """Fetch content from a website and extract text and PDF links."""
    
    # Use cached version if available
    cache_file = os.path.join(CACHE_DIR, urllib.parse.quote_plus(url))
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            content = response.text
            
            # Cache the content
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return f"Error fetching {url}: {str(e)}", []
    
    # Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')
    
    # Extract all text
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Extract PDF links
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            # Make relative URLs absolute
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(url, href)
            
            # Get text around the link
            surrounding_text = link.get_text() or "PDF Document"
            parent_text = ""
            parent = link.parent
            if parent:
                parent_text = parent.get_text() or ""
            
            pdf_links.append({
                'url': href,
                'text': surrounding_text,
                'context': parent_text[:100]
            })
    
    return text, pdf_links

def initialize_rag_system():
    """Initialize the RAG system by scraping websites and creating vector store."""
    st.session_state.status = "Initializing RAG system..."
    
    # Scrape website contents
    all_docs = []
    all_pdf_links = []
    
    progress_bar = st.progress(0)
    for i, website in enumerate(WEBSITES):
        st.session_state.status = f"Processing {website}..."
        content, pdf_links = fetch_website_content(website)
        
        # Create text chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        if content and not content.startswith("Error"):
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"source": website}
                ))
            all_pdf_links.extend(pdf_links)
        
        progress_bar.progress((i + 1) / len(WEBSITES))
    
    # Initialize embeddings
    st.session_state.status = "Creating embeddings..."
    embeddings = SentenceTransformerEmbeddings()
    
    # Create FAISS vector store
    st.session_state.status = "Building vector store..."
    vector_store = LangchainFAISS.from_documents(all_docs, embeddings)
    
    # Store in session state
    st.session_state.vector_store = vector_store
    st.session_state.pdf_links = all_pdf_links
    st.session_state.status = "RAG system initialized!"
    st.session_state.initialized = True

# Initialize Groq LLM with API key
def initialize_llm():
    groq_api_key = st.session_state.groq_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192"
    )
    
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    template = """
    Answer the question based on the provided context. If the information is not available in the context, say so clearly.
    Include references to the sources of your information. If there are PDF links that might be relevant, mention them.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    st.session_state.qa_chain = qa_chain

# Function to find relevant PDF links based on query
def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 3):
    if not pdf_links:
        return []
    
    # Get embeddings for the query and pdf contexts
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query)
    
    pdf_texts = [f"{pdf['text']} {pdf['context']}" for pdf in pdf_links]
    pdf_embeddings = model.encode(pdf_texts)
    
    # Set up FAISS for quick similarity search
    dimension = pdf_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(pdf_embeddings)
    
    # Search
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Return the top matching PDFs
    results = []
    for idx in indices[0]:
        if idx < len(pdf_links):
            results.append(pdf_links[idx])
    
    return results

# Main Streamlit UI
st.title("Web Intelligence BOT")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
        
        if st.button("Initialize System") or ('initialized' not in st.session_state):
            st.session_state.initialized = False
            initialize_rag_system()
            if st.session_state.initialized:
                initialize_llm()

# Display initialization status
if 'status' in st.session_state:
    st.info(st.session_state.status)

# Main content area
if 'initialized' in st.session_state and st.session_state.initialized:
    st.subheader("Ask a question")
    query = st.text_input("What would you like to know?")
    
    if query and st.button("Search"):
        with st.spinner("Searching for information..."):
            # Run the query through the QA chain
            result = st.session_state.qa_chain({"query": query})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Find relevant PDF links
            relevant_pdfs = find_relevant_pdfs(query, st.session_state.pdf_links)
            
            # Display the answer
            st.subheader("Answer")
            st.write(answer)
            
            # Display sources
            with st.expander("Sources"):
                sources = set()
                for doc in source_docs:
                    sources.add(doc.metadata["source"])
                
                for source in sources:
                    st.write(f"- [{source}]({source})")
            
            # Display relevant PDF links
            if relevant_pdfs:
                st.subheader("Relevant PDF Documents")
                for pdf in relevant_pdfs:
                    st.markdown(f"[{pdf['text']}]({pdf['url']})")
                    st.caption(f"Context: {pdf['context']}")
            else:
                st.info("No relevant PDF documents found.")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

# Display info about indexed websites
with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
