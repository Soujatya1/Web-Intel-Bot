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
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Add the following imports for handling offline mode
import logging
from pathlib import Path
import shutil

st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEBSITES = [
    "https://irdai.gov.in/rules"
]

CACHE_DIR = ".web_cache"
MODEL_CACHE_DIR = ".model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Custom embeddings class with offline support
class CustomEmbeddings(Embeddings):
    def __init__(self, cache_folder=MODEL_CACHE_DIR):
        self.cache_folder = cache_folder
        self.model = None
        self.initialize_model()
        
    def initialize_model(self):
        try:
            # Try to import and use SentenceTransformer
            from sentence_transformers import SentenceTransformer
            
            try:
                # First try using the local cache
                self.model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=self.cache_folder)
                logger.info("Successfully loaded model from cache")
            except Exception as e:
                logger.warning(f"Could not load model from cache: {str(e)}")
                # If that fails, try downloading with retries
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
            logger.error(f"Error fetching {url}: {str(e)}")
            return f"Error fetching {url}: {str(e)}", []
    
    soup = BeautifulSoup(content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.extract()
    
    table_data = extract_table_data(soup, url)
    
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    combined_text = text + "\n\n" + table_data
    
    pdf_links = extract_pdf_links(soup, url)
    
    return combined_text, pdf_links

def extract_table_data(soup, base_url):
    table_data = ""
    
    tables = soup.find_all('table')
    
    for table in tables:
        headers = [th.get_text().strip() for th in table.find_all('th')]
        
        if any(header in " ".join(headers) for header in ["Archive", "Description", "Last Updated", "Documents"]):
            table_data += "IRDAI Acts Information:\n"
            
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    archive_status = cells[0].get_text().strip()
                    description = cells[1].get_text().strip()
                    last_updated = cells[2].get_text().strip()
                    
                    doc_cell = cells[-1]
                    pdf_links = []
                    for link in doc_cell.find_all('a'):
                        if link.has_attr('href') and link['href'].lower().endswith('.pdf'):
                            pdf_url = link['href']
                            if not pdf_url.startswith(('http://', 'https://')):
                                pdf_url = urllib.parse.urljoin(base_url, pdf_url)
                            
                            file_info = link.get_text().strip()
                            pdf_links.append(f"{file_info} ({pdf_url})")
                    
                    row_data = f"Act: {description}\n"
                    row_data += f"Status: {archive_status}\n"
                    row_data += f"Last Updated: {last_updated}\n"
                    
                    if pdf_links:
                        row_data += "Documents: " + ", ".join(pdf_links) + "\n"
                    
                    table_data += row_data + "\n"
            
            table_data += "\nLatest Acts Information:\n"
            
            latest_dates = []
            for row in table.find_all('tr')[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    date_text = cells[2].get_text().strip()
                    if re.search(r'\d{2}-\d{2}-\d{4}', date_text):
                        latest_dates.append((date_text, cells[1].get_text().strip()))
            
            if latest_dates:
                latest_dates.sort(reverse=True)
                latest_date, latest_act = latest_dates[0]
                table_data += f"The latest updated Act under IRDAI is {latest_act} with the last updated date as {latest_date}\n"
    
    return table_data

def extract_pdf_links(soup, base_url):
    pdf_links = []
    
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            if len(cells) < 3:
                continue
                
            try:
                description = cells[1].get_text().strip() if len(cells) > 1 else ""
                last_updated = cells[2].get_text().strip() if len(cells) > 2 else ""
                
                doc_cell = cells[-1]
                for link in doc_cell.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        if not href.startswith(('http://', 'https://')):
                            href = urllib.parse.urljoin(base_url, href)
                        
                        link_text = link.get_text().strip()
                        
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
                continue
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            if any(pdf['url'] == href for pdf in pdf_links):
                continue
                
            if not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(base_url, href)
            
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
    st.session_state.status = "Initializing RAG system"
    
    all_docs = []
    all_pdf_links = []
    
    progress_bar = st.progress(0)
    for i, website in enumerate(WEBSITES):
        st.session_state.status = f"Processing {website}..."
        content, pdf_links = fetch_website_content(website)
        
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
    
    st.session_state.status = "Creating embeddings"
    try:
        # Use our custom embeddings class with offline support
        embeddings = CustomEmbeddings()
        
        st.session_state.status = "Building vector store"
        vector_store = LangchainFAISS.from_documents(all_docs, embeddings)
        
        st.session_state.vector_store = vector_store
        st.session_state.pdf_links = all_pdf_links
        st.session_state.status = "System initialized!"
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.status = f"Error during initialization: {str(e)}"
        logger.error(f"Initialization error: {str(e)}")
        st.error(f"Error: {str(e)}")

def initialize_llm():
    groq_api_key = st.session_state.groq_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile"
        )
        
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        template = """
You are an expert assistant for insurance regulatory information in India with knowledge about IRDAI, UIDAI, and Enforcement Directorate regulations. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the information provided in the context.
2. If the information is not available in the context, say "Based on the provided information, I cannot answer this question."
3. When mentioning dates, always specify them explicitly (e.g., "as of 15 January 2023").
4. Provide specific section numbers, regulation names, and exact quotes where possible.
5. If relevant PDF documents are mentioned in the context, include their details.
6. Always cite the source websites or documents that your answer is based on.

ANSWER:
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
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        logger.error(f"LLM initialization error: {str(e)}")

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 3):
    if not pdf_links:
        return []
    
    try:
        # Use our custom embeddings for consistency
        embedder = CustomEmbeddings()
        query_embedding = embedder.embed_query(query)
        
        pdf_texts = []
        for pdf in pdf_links:
            context_text = f"{pdf['text']} {pdf['context']}"
            if 'metadata' in pdf and pdf['metadata']:
                for key, value in pdf['metadata'].items():
                    context_text += f" {key}: {value}"
            pdf_texts.append(context_text)
        
        pdf_embeddings = np.array(embedder.embed_documents(pdf_texts))
        
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

# Add instructions for offline mode setup
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
        
        st.markdown("### Offline Mode Setup")
        st.markdown("""
        If you're experiencing connection issues to Hugging Face, you can use offline mode by:
        1. Downloading the model manually from [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
        2. Place the files in the `.model_cache` directory
        """)
        
        if st.button("Initialize System") or ('initialized' not in st.session_state):
            st.session_state.initialized = False
            initialize_rag_system()
            if st.session_state.initialized:
                initialize_llm()

# Main UI
if 'status' in st.session_state:
    st.info(st.session_state.status)

if 'initialized' in st.session_state and st.session_state.initialized:
    st.subheader("Ask a question")
    query = st.text_input("What would you like to know?")
    
    if query and st.button("Search"):
        with st.spinner("Searching for information..."):
            try:
                result = st.session_state.qa_chain({"query": query})
                answer = result["result"]
                source_docs = result["source_documents"]
                
                relevant_pdfs = find_relevant_pdfs(query, st.session_state.pdf_links)
                
                st.subheader("Answer")
                st.write(answer)
                
                with st.expander("Sources"):
                    sources = set()
                    for doc in source_docs:
                        sources.add(doc.metadata["source"])
                    
                    for source in sources:
                        st.write(f"- [{source}]({source})")
                
                if relevant_pdfs:
                    st.subheader("Relevant PDF Documents")
                    for pdf in relevant_pdfs:
                        metadata_text = ""
                        if 'metadata' in pdf and pdf['metadata']:
                            for key, value in pdf['metadata'].items():
                                if value:
                                    metadata_text += f"{key}: {value}, "
                            metadata_text = metadata_text.rstrip(", ")
                        
                        st.markdown(f"[{pdf['text']}]({pdf['url']})")
                        if metadata_text:
                            st.caption(f"{metadata_text}")
                        else:
                            st.caption(f"Context: {pdf['context']}")
                else:
                    st.info("No relevant PDF documents found.")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                logger.error(f"Query processing error: {str(e)}")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
