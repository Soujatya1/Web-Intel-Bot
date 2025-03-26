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
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation"
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
    """Fetch content from a website and extract text, tables, and PDF links."""
    
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
    
    # Extract text with better structure preservation
    for script in soup(["script", "style"]):
        script.extract()
    
    # Extract table data specifically (for structured data like acts)
    table_data = extract_table_data(soup, url)
    
    # Extract general text
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Combine general text with structured table data
    combined_text = text + "\n\n" + table_data
    
    # Extract PDF links with better metadata
    pdf_links = extract_pdf_links(soup, url)
    
    return combined_text, pdf_links

def extract_table_data(soup, base_url):
    """Extract structured data from tables with special handling for IRDAI acts table."""
    table_data = ""
    
    # Look for tables in the document
    tables = soup.find_all('table')
    
    for table in tables:
        # Check if this looks like the Acts table (has headers like "Archive/Non Archive", "Short Description", etc.)
        headers = [th.get_text().strip() for th in table.find_all('th')]
        
        # If this appears to be the Acts table
        if any(header in " ".join(headers) for header in ["Archive", "Description", "Last Updated", "Documents"]):
            table_data += "IRDAI Acts Information:\n"
            
            # Process each row
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:  # Ensure we have enough cells
                    # Extract data with position awareness
                    archive_status = cells[0].get_text().strip()
                    description = cells[1].get_text().strip()
                    last_updated = cells[2].get_text().strip()
                    
                    # Get document links
                    doc_cell = cells[-1]  # Usually the last cell
                    pdf_links = []
                    for link in doc_cell.find_all('a'):
                        if link.has_attr('href') and link['href'].lower().endswith('.pdf'):
                            pdf_url = link['href']
                            if not pdf_url.startswith(('http://', 'https://')):
                                pdf_url = urllib.parse.urljoin(base_url, pdf_url)
                            
                            # Try to extract file name and size
                            file_info = link.get_text().strip()
                            pdf_links.append(f"{file_info} ({pdf_url})")
                    
                    # Format as structured text
                    row_data = f"Act: {description}\n"
                    row_data += f"Status: {archive_status}\n"
                    row_data += f"Last Updated: {last_updated}\n"
                    
                    if pdf_links:
                        row_data += "Documents: " + ", ".join(pdf_links) + "\n"
                    
                    table_data += row_data + "\n"
            
            # Add special sections to help with retrieval
            table_data += "\nLatest Acts Information:\n"
            
            # Find the most recent date
            latest_dates = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    date_text = cells[2].get_text().strip()
                    if re.search(r'\d{2}-\d{2}-\d{4}', date_text):
                        latest_dates.append((date_text, cells[1].get_text().strip()))
            
            # Sort by date (most recent first) and add the information
            if latest_dates:
                latest_dates.sort(reverse=True)
                latest_date, latest_act = latest_dates[0]
                table_data += f"The latest updated Act under IRDAI is {latest_act} with the last updated date as {latest_date}\n"
    
    return table_data

def extract_pdf_links(soup, base_url):
    """Extract PDF links with improved metadata extraction."""
    pdf_links = []
    
    # First pass: look for PDF links in tables with better metadata
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            # Skip if not enough cells
            if len(cells) < 3:
                continue
                
            # Try to extract metadata from the row
            try:
                description = cells[1].get_text().strip() if len(cells) > 1 else ""
                last_updated = cells[2].get_text().strip() if len(cells) > 2 else ""
                
                # Look for PDF links in the last cell (documents column)
                doc_cell = cells[-1]
                for link in doc_cell.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        # Make relative URLs absolute
                        if not href.startswith(('http://', 'https://')):
                            href = urllib.parse.urljoin(base_url, href)
                        
                        # Get link text and file size if available
                        link_text = link.get_text().strip()
                        
                        # Create a rich context including the description and update date
                        context = f"Act: {description}, Last Updated: {last_updated}"
                        
                        pdf_links.append({
                            'url': href,
                            'text': link_text or description,
                            'context': context,
                            'metadata': {
                                'description': description,
                                'last_updated': last_updated
                            }
                        })
            except Exception as e:
                # If error parsing the row, continue to the next
                continue
    
    # Second pass: general PDF links (in case we missed any)
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            # Skip if we already found this URL in the table
            if any(pdf['url'] == href for pdf in pdf_links):
                continue
                
            # Make relative URLs absolute
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(base_url, href)
            
            # Get text around the link
            surrounding_text = link.get_text() or "PDF Document"
            parent_text = ""
            parent = link.parent
            if parent:
                parent_text = parent.get_text() or ""
            
            pdf_links.append({
                'url': href,
                'text': surrounding_text,
                'context': parent_text[:100],
                'metadata': {}
            })
    
    return pdf_links

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
    st.session_state.status = "System initialized!"
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
    You are an expert assistant for insurance regulatory information. Answer the question based on the provided context.
    If the information is not available in the context, say so clearly.
    
    When asked about the "latest" acts or documents, focus on the most recently updated ones based on dates in the context.
    Pay special attention to the "Last Updated" dates and present them in your answer.
    
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
    
    # Include metadata in the context for better matching
    pdf_texts = []
    for pdf in pdf_links:
        context_text = f"{pdf['text']} {pdf['context']}"
        # Add metadata if available
        if 'metadata' in pdf and pdf['metadata']:
            for key, value in pdf['metadata'].items():
                context_text += f" {key}: {value}"
        pdf_texts.append(context_text)
    
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
                    # Display with metadata when available
                    metadata_text = ""
                    if 'metadata' in pdf and pdf['metadata']:
                        for key, value in pdf['metadata'].items():
                            if value:  # Only show non-empty values
                                metadata_text += f"{key}: {value}, "
                        metadata_text = metadata_text.rstrip(", ")
                    
                    st.markdown(f"[{pdf['text']}]({pdf['url']})")
                    if metadata_text:
                        st.caption(f"{metadata_text}")
                    else:
                        st.caption(f"Context: {pdf['context']}")
            else:
                st.info(" ")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

# Display info about indexed websites
with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
