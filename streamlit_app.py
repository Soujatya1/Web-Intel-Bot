import streamlit as st
import requests
import os
import re
import faiss
import numpy as np
import tempfile
import urllib.parse
import logging
import io
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from pathlib import Path
import shutil

st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEBSITES = [
    "https://irdai.gov.in/rules",
    "https://irdai.gov.in/consolidated-gazette-notified-regulations",
    "https://irdai.gov.in/updated-regulations",
    "https://irdai.gov.in/notifications",
    "https://irdai.gov.in/circulars",
    "https://irdai.gov.in/orders1",
    "https://irdai.gov.in/exposure-drafts",
    "https://irdai.gov.in/programmes-to-advance-understanding-of-rti",
    "https://irdai.gov.in/cic-orders",
    "https://irdai.gov.in/antimoney-laundering",
    "https://irdai.gov.in/other-communication",
    "https://irdai.gov.in/directory-of-employees",
    "https://irdai.gov.in/warnings-and-penalties",
    "https://uidai.gov.in/en/",
    "https://uidai.gov.in/en/about-uidai/legal-framework.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/rules.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/notifications.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/judgements.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules",
    "https://enforcementdirectorate.gov.in/pmla",
    "https://enforcementdirectorate.gov.in/fema",
    "https://enforcementdirectorate.gov.in/bns",
    "https://enforcementdirectorate.gov.in/bnss",
    "https://enforcementdirectorate.gov.in/bsa"
]

CACHE_DIR = ".web_cache"
MODEL_CACHE_DIR = ".model_cache"
PDF_CACHE_DIR = ".pdf_cache"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# Define abbreviations for query expansion
ABBREVIATIONS = {
    "IRDAI": "Insurance Regulatory and Development Authority of India",
    "UIDAI": "Unique Identification Authority of India",
    "ED": "Enforcement Directorate",
    "PMLA": "Prevention of Money Laundering Act",
    "FEMA": "Foreign Exchange Management Act",
    "KYC": "Know Your Customer",
    "AML": "Anti-Money Laundering",
    "CFT": "Combating Financing of Terrorism"
}

# Custom embeddings class with robust offline support
class CustomEmbeddings(Embeddings):
    def __init__(self, cache_folder=MODEL_CACHE_DIR):
        self.cache_folder = cache_folder
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        try:
            # Try to import and use SentenceTransformer
            from sentence_transformers import SentenceTransformer
            
            # Try multiple model options with fallback
            model_options = [
                "all-MiniLM-L6-v2",
                "paraphrase-MiniLM-L6-v2",
                "distiluse-base-multilingual-cased-v1"
            ]
            
            for model_name in model_options:
                try:
                    self.model = SentenceTransformer(model_name, cache_folder=self.cache_folder)
                    logger.info(f"Successfully loaded model {model_name}")
                    return
                except Exception as e:
                    logger.warning(f"Could not load model {model_name}: {str(e)}")
                    
            # If all models fail, try one more time with download
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to download model (attempt {attempt+1}/{max_retries})")
                    self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=self.cache_folder)
                    logger.info("Successfully downloaded model")
                    break
                except Exception as e:
                    logger.error(f"Download attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retrying")
                        time.sleep(wait_time)
                    else:
                        raise
        except ImportError:
            logger.error("sentence_transformers module not available")
            st.error("Required modules not available. Please install sentence_transformers.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embeddings model: {str(e)}")
            st.error(f"Failed to initialize embeddings model: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            raise ValueError("Model not initialized")
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding documents: {str(e)}")
            # Fall back to simple embeddings if model fails
            return self._fallback_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if self.model is None:
            raise ValueError("Model not initialized")
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error encoding query: {str(e)}")
            # Fall back to simple embeddings if model fails
            return self._fallback_embeddings([text])[0]
    
    def _fallback_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Extremely simple fallback embedding method using hashing"""
        logger.warning("Using fallback embedding method - results will be suboptimal")
        dimension = 384  # Match MiniLM dimension
        results = []
        
        for text in texts:
            # Create a simple hash-based embedding
            import hashlib
            embedding = np.zeros(dimension)
            
            # Break text into words and hash each word
            words = text.split()
            for i, word in enumerate(words):
                word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
                # Use the hash to set values in the embedding
                for j in range(min(20, len(word))):  # Use first 20 chars max
                    pos = (word_hash + ord(word[j]) + j) % dimension
                    embedding[pos] = 0.1 * ((i % 10) + 1) + 0.01 * ord(word[j]) / 255.0
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            results.append(embedding.tolist())
            
        return results

def fetch_website_content(url: str) -> Tuple[str, List[Dict]]:
    """Fetch and parse website content with more robust error handling"""
    
    cache_file = os.path.join(CACHE_DIR, urllib.parse.quote_plus(url))
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            # Check response status
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return f"Error fetching {url}: HTTP {response.status_code}", []
                
            content = response.text
            
            # Cache the content
            with open(cache_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return f"Error fetching {url}: {str(e)}", []
    
    try:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts, styles, and hidden elements
        for element in soup(['script', 'style', '[style*="display:none"]', '[style*="display: none"]']):
            element.decompose()
        
        # Extract table data more comprehensively
        table_data = extract_table_data(soup, url)
        
        # Extract text content
        text_parts = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text(strip=True)
            if text:
                text_parts.append(text)
        
        # Get the title of the page
        title = soup.title.string if soup.title else ""
        text_parts.insert(0, f"Title: {title}")
        
        # Combine all text content
        main_text = "\n\n".join(text_parts)
        
        # Add context about the URL
        url_context = f"Source URL: {url}\n\n"
        
        # Combine everything
        combined_text = url_context + main_text + "\n\n" + table_data
        
        # Clean up the text
        combined_text = re.sub(r'\n{3,}', '\n\n', combined_text)
        combined_text = re.sub(r'\s{2,}', ' ', combined_text)
        
        # Extract PDF links
        pdf_links = extract_pdf_links(soup, url)
        
        return combined_text, pdf_links
    except Exception as e:
        logger.error(f"Error parsing {url}: {str(e)}")
        return f"Error parsing {url}: {str(e)}", []

def extract_table_data(soup, base_url):
    """Extract structured information from tables"""
    table_data = ""
    
    tables = soup.find_all('table')
    
    for table_idx, table in enumerate(tables):
        # Add table identifier
        table_data += f"Table {table_idx + 1} content:\n"
        
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            if headers:
                table_data += "Headers: " + " | ".join(headers) + "\n"
        
        # Process rows
        rows = table.find_all('tr')[1:] if headers else table.find_all('tr')
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            if cells:
                row_data = []
                for cell_idx, cell in enumerate(cells):
                    cell_text = cell.get_text().strip()
                    
                    # Label the content if we have headers
                    if cell_idx < len(headers) and headers:
                        header = headers[cell_idx]
                        if header:
                            row_data.append(f"{header}: {cell_text}")
                    else:
                        row_data.append(cell_text)
                
                # Check for links in this row
                for cell in cells:
                    links = cell.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if not href.startswith(('http://', 'https://')):
                            href = urllib.parse.urljoin(base_url, href)
                        link_text = link.get_text().strip()
                        if link_text and href:
                            row_data.append(f"Link: {link_text} ({href})")
                
                table_data += " | ".join(row_data) + "\n"
        
        table_data += "\n"
    
    return table_data

def extract_pdf_links(soup, base_url):
    """Extract PDF links with improved context capture"""
    pdf_links = []
    
    # Process tables for PDF links with context
    tables = soup.find_all('table')
    for table in tables:
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
        
        rows = table.find_all('tr')[1:] if headers else table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
            
            # Build row context
            row_context = {}
            for idx, cell in enumerate(cells):
                if idx < len(headers) and headers[idx]:
                    row_context[headers[idx]] = cell.get_text().strip()
            
            # Find PDF links in this row
            for cell in cells:
                for link in cell.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        if not href.startswith(('http://', 'https://')):
                            href = urllib.parse.urljoin(base_url, href)
                        
                        # Get link text or cell text
                        link_text = link.get_text().strip()
                        if not link_text:
                            link_text = cell.get_text().strip()
                        
                        pdf_links.append({
                            'url': href,
                            'text': link_text,
                            'context': row_context,
                            'source_url': base_url
                        })
    
    # Find PDF links outside tables
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            if any(pdf['url'] == href for pdf in pdf_links):
                continue
                
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(base_url, href)
            
            # Get surrounding content for context
            context = {}
            parent = link.parent
            if parent:
                # Look for nearby headings
                prev_heading = parent.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev_heading:
                    context['heading'] = prev_heading.get_text().strip()
                
                # Look for nearby paragraphs
                prev_paragraph = parent.find_previous('p')
                if prev_paragraph:
                    context['paragraph'] = prev_paragraph.get_text().strip()
            
            pdf_links.append({
                'url': href,
                'text': link.get_text().strip() or "PDF Document",
                'context': context,
                'source_url': base_url
            })
    
    return pdf_links

def process_pdf(pdf_url, cache_dir=PDF_CACHE_DIR):
    """Download and extract text from PDF with caching"""
    try:
        import PyPDF2
    except ImportError:
        logger.warning("PyPDF2 not installed. PDF processing disabled.")
        return None
    
    cache_file = os.path.join(cache_dir, urllib.parse.quote_plus(pdf_url) + ".txt")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            pdf_file = io.BytesIO(response.content)
            
            try:
                reader = PyPDF2.PdfReader(pdf_file)
                
                text = ""
                # Try to extract PDF metadata
                if hasattr(reader, 'metadata') and reader.metadata:
                    meta = reader.metadata
                    text += f"PDF Title: {meta.get('/Title', 'N/A')}\n"
                    text += f"PDF Author: {meta.get('/Author', 'N/A')}\n"
                    text += f"PDF Subject: {meta.get('/Subject', 'N/A')}\n"
                    text += f"PDF Keywords: {meta.get('/Keywords', 'N/A')}\n"
                    text += f"PDF Creation Date: {meta.get('/CreationDate', 'N/A')}\n"
                    text += f"PDF Modification Date: {meta.get('/ModDate', 'N/A')}\n\n"
                
                text += f"PDF Source: {pdf_url}\n\n"
                text += f"Total Pages: {len(reader.pages)}\n\n"
                
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text() or ""
                    if page_text:
                        text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                
                # Cache the extracted text
                with open(cache_file, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(text)
                    
                return text
            except Exception as e:
                logger.error(f"Error parsing PDF {pdf_url}: {str(e)}")
                return None
        else:
            logger.warning(f"Failed to download PDF: {pdf_url}, status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_url}: {str(e)}")
        return None

def preprocess_query(query):
    """Enhance query with expanded abbreviations and related terms"""
    # Expand common abbreviations
    expanded_query = query
    
    # Add keywords to enhance retrieval
    for abbr, expansion in ABBREVIATIONS.items():
        if abbr in query and expansion not in query:
            expanded_query += f" ({expansion})"
    
    # Add relevant keywords based on the query content
    if "insurance" in query.lower() or "irdai" in query.lower():
        expanded_query += " regulation policy circular notification"
    elif "aadhaar" in query.lower() or "uidai" in query.lower():
        expanded_query += " identity verification biometric authentication"
    elif "money laundering" in query.lower() or "pmla" in query.lower() or "fema" in query.lower():
        expanded_query += " enforcement directorate financial crime"
    
    logger.info(f"Original query: {query}")
    logger.info(f"Expanded query: {expanded_query}")
    
    return expanded_query

def rerank_documents(docs, query, top_k=5):
    """Rerank retrieved documents for better relevance"""
    try:
        from sentence_transformers import CrossEncoder
        
        # Initialize cross-encoder for reranking
        model_path = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        try:
            cross_encoder = CrossEncoder(model_path, cache_folder=MODEL_CACHE_DIR)
        except Exception as e:
            logger.warning(f"Error loading cross-encoder model: {str(e)}")
            return docs[:top_k]
        
        # Prepare document-query pairs for scoring
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Get relevance scores
        scores = cross_encoder.predict(pairs)
        
        # Create document-score pairs
        doc_score_pairs = list(zip(docs, scores))
        
        # Sort by score in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k documents
        return [doc for doc, _ in doc_score_pairs[:top_k]]
    except Exception as e:
        logger.warning(f"Reranking failed: {str(e)}, returning original docs")
        return docs[:top_k]

def fetch_all_websites():
    """Process all websites in parallel with improved error handling"""
    all_docs = []
    all_pdf_links = []
    
    def process_website(website):
        try:
            logger.info(f"Processing website: {website}")
            content, pdf_links = fetch_website_content(website)
            
            docs = []
            if content and not content.startswith("Error"):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=150,
                    separators=["\n\n", "\n", " ", ""],
                    length_function=len
                )
                
                chunks = text_splitter.split_text(content)
                docs = [
                    Document(page_content=chunk, metadata={"source": website})
                    for chunk in chunks
                ]
                
                # Process top PDFs from this website
                pdf_processed = 0
                for pdf_link in pdf_links[:3]:  # Limit to top 3 PDFs per website
                    pdf_text = process_pdf(pdf_link['url'])
                    if pdf_text:
                        pdf_chunks = text_splitter.split_text(pdf_text)
                        pdf_docs = [
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": pdf_link['url'],
                                    "source_website": website,
                                    "type": "pdf",
                                    "text": pdf_link['text']
                                }
                            )
                            for chunk in pdf_chunks
                        ]
                        docs.extend(pdf_docs)
                        pdf_processed += 1
                
                logger.info(f"Processed {website}: {len(chunks)} chunks, {pdf_processed} PDFs")
            
            return docs, pdf_links
        except Exception as e:
            logger.error(f"Error processing {website}: {str(e)}")
            return [], []
    
    # Process websites in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_website, WEBSITES))
    
    # Combine results
    for docs, pdf_links in results:
        all_docs.extend(docs)
        all_pdf_links.extend(pdf_links)
    
    return all_docs, all_pdf_links

def create_hybrid_retriever(vector_store):
    """Create a hybrid retriever that combines keyword and semantic search"""
    try:
        # Vector store retriever (semantic)
        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Retrieve more and then filter
        )
        
        # BM25 retriever (keyword-based)
        all_docs = list(vector_store.docstore.values())
        try:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 8
            
            # Ensemble retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]  # Favor semantic search but include keyword results
            )
            
            logger.info("Successfully created hybrid retriever")
            return hybrid_retriever
        except Exception as e:
            logger.warning(f"Failed to create BM25 retriever: {str(e)}. Falling back to vector retriever.")
            return vector_retriever
    except Exception as e:
        logger.error(f"Error creating hybrid retriever: {str(e)}")
        # Fall back to basic vector retrieval
        return vector_store.as_retriever(search_kwargs={"k": 5})

def create_qa_chain(llm, retriever, prompt):
    """Create QA chain with fact checking"""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Add answer verification step
    def run_with_verification(query_dict):
        # Get original results
        result = qa(query_dict)
        answer = result["result"]
        sources = result["source_documents"]
        
        # Check if answer is supported by sources
        verification_prompt = f"""
        Answer: {answer}
        
        Source documents:
        {[doc.page_content for doc in sources[:3]]}
        
        Is the above answer completely supported by the source documents? 
        If not, provide a revised answer that only contains facts from the source documents.
        """
        
        try:
            verification = llm.predict(verification_prompt)
            
            if "not supported" in verification.lower() or "revised answer" in verification.lower():
                # Extract the revised answer
                if "revised answer:" in verification.lower():
                    revised_parts = verification.lower().split("revised answer:")
                    if len(revised_parts) > 1:
                        result["result"] = revised_parts[1].strip()
                else:
                    result["result"] += "\n\nNOTE: The information provided is limited to what's available in the source documents."
        except Exception as e:
            logger.error(f"Verification step failed: {str(e)}")
        
        return result
    
    qa.run = run_with_verification
    return qa

def initialize_conversation():
    """Initialize conversation memory"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    
    st.session_state.memory = memory

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 5):
    """Find the most relevant PDF links based on query"""
    if not pdf_links:
        return []
    
    try:
        # Use our custom embeddings for consistency
        embedder = CustomEmbeddings()
        query_embedding = embedder.embed_query(query)
        
        pdf_texts = []
        for pdf in pdf_links:
            # Combine all text elements for matching
            context_text = f"{pdf['text']} "
            
            if 'context' in pdf and pdf['context']:
                if isinstance(pdf['context'], dict):
                    for key, value in pdf['context'].items():
                        context_text += f"{key}: {value} "
                else:
                    context_text += str(pdf['context'])
            
            pdf_texts.append(context_text)
        
        # Generate embeddings
        pdf_embeddings = np.array(embedder.embed_documents(pdf_texts))
        
        # Search for similar PDFs
        dimension = len(query_embedding)
        index = faiss.IndexFlatL2(dimension)
        index.add(pdf_embeddings)
        
        distances, indices = index.search(np.array([query_embedding]), top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(pdf_links):
                results.append(pdf_links[idx])
        
        return results
    except Exception as e:
        logger.error(f"Error finding relevant PDFs: {str(e)}")
        st.warning(f"Could not find relevant PDFs: {str(e)}")
        return []

def initialize_rag_system():
    """Initialize the RAG system with all components"""
    st.session_state.status = "Initializing RAG system"
    
    # Fetch all website content in parallel
    progress_bar = st.progress(0)
    st.session_state.status = "Fetching website content..."
    
    all_docs, all_pdf_links = fetch_all_websites()
    
    # Update progress
    progress_bar.progress(0.5)
    st.session_state.status = "Creating embeddings..."
    
    try:
        # Use our custom embeddings class with offline support
        embeddings = CustomEmbeddings()
        
        # Filter out empty documents
        valid_docs = [doc for doc in all_docs if doc.page_content.strip()]
        
        # Check for empty docs
        if not valid_docs:
            st.error("No valid documents found. Please check the websites and try again.")
            st.session_state.status = "Initialization failed. No valid documents found."
            return
        
        st.session_state.status = "Building vector store..."
        vector_store = LangchainFAISS.from_documents(valid_docs, embeddings)
        
        # Update progress
        progress_bar.progress(0.8)
        st.session_state.status = "Finalizing system..."
        
        # Create hybrid retriever
        retriever = create_hybrid_retriever(vector_store)
        
        # Initialize conversation memory
        initialize_conversation()
        
        # Store components in session state
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.pdf_links = all_pdf_links
        st.session_state.status = f"System initialized with {len(valid_docs)} documents and {len(all_pdf_links)} PDF links!"
        st.session_state.initialized = True
        
        # Complete progress
        progress_bar.progress(1.0)
    except Exception as e:
        st.session_state.status = f"Error during initialization: {str(e)}"
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        st.error(f"Failed to initialize RAG system: {str(e)}")
        progress_bar.progress(1.0)
