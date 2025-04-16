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
    "https://irdai.gov.in/rules",
    "https://irdai.gov.in/consolidated-gazette-notified-regulations",
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
    """
    Extract and structure data from tables on the webpage with improved context gathering
    """
    table_data = ""
    
    # Get page title for additional context
    page_title = soup.title.get_text().strip() if soup.title else ""
    if page_title:
        table_data += f"Page Title: {page_title}\n\n"
    
    # Extract header information that might provide context
    headers = soup.find_all(['h1', 'h2', 'h3'])
    header_text = "\n".join([h.get_text().strip() for h in headers if h.get_text().strip()])
    if header_text:
        table_data += f"Page Headers: {header_text}\n\n"
    
    # Extract tables with better structure and error handling
    tables = soup.find_all('table')
    
    for table_index, table in enumerate(tables):
        try:
            # Try to find a caption or heading near the table
            caption = table.find('caption')
            caption_text = caption.get_text().strip() if caption else ""
            
            # Look for preceding headers
            preceding_header = None
            current = table
            while current and current.name != 'body':
                prev = current.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if prev:
                    preceding_header = prev.get_text().strip()
                    break
                current = current.parent
            
            # Add table identifier with any available context
            table_data += f"\n--- TABLE {table_index + 1}"
            if caption_text:
                table_data += f": {caption_text}"
            elif preceding_header:
                table_data += f" (Section: {preceding_header})"
            table_data += " ---\n"
            
            # Get headers from the table
            headers = []
            thead = table.find('thead')
            if thead:
                headers = [th.get_text().strip() for th in thead.find_all('th')]
            else:
                # Try getting headers from first row
                first_row = table.find('tr')
                if first_row:
                    headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
            
            # Check if this looks like an IRDAI Acts table
            is_irdai_table = any(keyword in " ".join(headers).lower() for keyword in 
                               ["archive", "description", "last updated", "documents", 
                                "act", "regulation", "circular", "notification"])
            
            if is_irdai_table:
                table_data += "IRDAI Regulatory Information:\n"
                
                # Process rows with better structure
                rows = table.find_all('tr')
                start_idx = 1 if (thead or len(headers) > 0) else 0
                
                for row in rows[start_idx:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2:  # Skip rows with insufficient data
                        continue
                    
                    # Extract cell data with fallbacks for missing columns
                    archive_status = cells[0].get_text().strip() if len(cells) > 0 else "N/A"
                    description = cells[1].get_text().strip() if len(cells) > 1 else "N/A"
                    last_updated = cells[2].get_text().strip() if len(cells) > 2 else "N/A"
                    
                    # Extract document links with full context
                    doc_cell = cells[-1]
                    pdf_links = []
                    for link in doc_cell.find_all('a'):
                        if link.has_attr('href'):
                            href = link['href']
                            if href.lower().endswith('.pdf'):
                                pdf_url = href
                                if not pdf_url.startswith(('http://', 'https://')):
                                    pdf_url = urllib.parse.urljoin(base_url, pdf_url)
                                
                                file_info = link.get_text().strip() or "PDF Document"
                                pdf_links.append(f"{file_info} ({pdf_url})")
                    
                    # Format row data with clear labeling
                    row_data = f"Document: {description}\n"
                    row_data += f"Status: {archive_status}\n"
                    row_data += f"Last Updated: {last_updated}\n"
                    
                    if pdf_links:
                        row_data += "Related Documents: " + ", ".join(pdf_links) + "\n"
                    
                    table_data += row_data + "\n"
                
                # Extract and highlight the latest acts/regulations
                table_data += "\nLatest Regulatory Updates:\n"
                
                latest_dates = []
                date_pattern = re.compile(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}[\/\-\.]\w+[\/\-\.]\d{2,4}')
                
                for row in rows[start_idx:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:
                        continue
                    
                    date_cell = cells[2].get_text().strip()
                    if date_pattern.search(date_cell):
                        latest_dates.append((date_cell, cells[1].get_text().strip()))
                
                if latest_dates:
                    # Sort by date if possible (assuming DD-MM-YYYY format)
                    try:
                        # Try various date formats
                        for i, (date_str, act) in enumerate(latest_dates):
                            try:
                                # Try to parse the date using different formats
                                for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %b %Y"]:
                                    try:
                                        parsed_date = time.strptime(date_str, fmt)
                                        latest_dates[i] = (parsed_date, act)
                                        break
                                    except ValueError:
                                        continue
                            except:
                                # Keep as string if parsing fails
                                pass
                        
                        # Sort based on parsed dates or original strings
                        latest_dates.sort(reverse=True)
                        
                        # Convert back to string representation if needed
                        formatted_dates = []
                        for date_info, act in latest_dates:
                            if isinstance(date_info, time.struct_time):
                                date_str = time.strftime("%d-%m-%Y", date_info)
                            else:
                                date_str = date_info
                            formatted_dates.append((date_str, act))
                        
                        latest_dates = formatted_dates
                    except:
                        # If sorting fails, use original data
                        pass
                    
                    # Add latest updates to the table data
                    for i, (date_str, act) in enumerate(latest_dates[:3]):  # Show top 3 latest
                        table_data += f"- {act} (Last updated: {date_str})\n"
            else:
                # For non-IRDAI specific tables, extract general information
                table_data += "Table contents:\n"
                
                # Get headers
                if headers:
                    table_data += " | ".join(headers) + "\n"
                    table_data += "-" * (sum(len(h) for h in headers) + (3 * (len(headers) - 1))) + "\n"
                
                # Get rows
                rows = table.find_all('tr')
                start_idx = 1 if (thead or len(headers) > 0) else 0
                
                for row in rows[start_idx:]:
                    cells = row.find_all(['td', 'th'])
                    if not cells:
                        continue
                    
                    row_data = " | ".join(cell.get_text().strip() for cell in cells)
                    table_data += row_data + "\n"
                
                table_data += "\n"
        except Exception as e:
            table_data += f"\nError processing table {table_index + 1}: {str(e)}\n"
            continue
    
    return table_data

def extract_pdf_links(soup, base_url):
    """
    Extract PDF links with comprehensive context and metadata for better relevance matching
    """
    pdf_links = []
    processed_urls = set()  # Track processed URLs to avoid duplicates
    
    # Helper function to get the most relevant heading context for an element
    def get_nearest_header(element):
        header = None
        current = element
        max_depth = 10  # Limit search depth to avoid infinite loops
        depth = 0
        
        while current and current.name != 'body' and depth < max_depth:
            # Search for the nearest header
            prev = current.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if prev:
                header = prev.get_text().strip()
                break
            current = current.parent
            depth += 1
        
        return header or ""
    
    # Helper function to get surrounding paragraph text
    def get_surrounding_text(element, max_chars=200):
        # Try to find parent paragraph
        parent_p = element.find_parent('p')
        if parent_p:
            return parent_p.get_text().strip()[:max_chars]
        
        # Try to find the containing div
        parent_div = element.find_parent('div')
        if parent_div:
            # Get text but exclude nested elements that would create noise
            text_parts = []
            for content in parent_div.contents:
                if isinstance(content, str):
                    text_parts.append(content.strip())
                elif getattr(content, 'name', None) not in ['div', 'table', 'ul', 'ol']:
                    text_parts.append(content.get_text().strip())
            
            return " ".join(text_parts)[:max_chars]
        
        # If no good container found, get siblings text
        siblings_text = []
        for sibling in element.next_siblings:
            if isinstance(sibling, str):
                siblings_text.append(sibling.strip())
            elif getattr(sibling, 'name', None) == 'br':
                continue
            elif getattr(sibling, 'name', None) not in ['div', 'table', 'ul', 'ol']:
                siblings_text.append(sibling.get_text().strip())
            else:
                break
            
            if len(" ".join(siblings_text)) > max_chars:
                break
                
        return " ".join(siblings_text)[:max_chars]
    
    # First extract PDFs from tables which often contain the most structured data
    tables = soup.find_all('table')
    for table_index, table in enumerate(tables):
        try:
            # Get table caption or nearest header for context
            table_caption = ""
            caption = table.find('caption')
            if caption:
                table_caption = caption.get_text().strip()
            else:
                nearest_header = get_nearest_header(table)
                if nearest_header:
                    table_caption = nearest_header
            
            # Process table rows
            rows = table.find_all('tr')
            for row_index, row in enumerate(rows):
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 3:
                    continue
                    
                try:
                    # Extract row data for context
                    row_data = {}
                    
                    # Use header cells from first row if available
                    if row_index == 0 and any(cell.name == 'th' for cell in cells):
                        continue
                    
                    # Create a mapping between possible column positions and their meanings
                    headers = []
                    if row_index > 0 and rows[0]:
                        header_cells = rows[0].find_all(['th', 'td'])
                        headers = [h.get_text().strip().lower() for h in header_cells]
                    
                    # Map columns based on typical IRDAI table structure or headers
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text().strip()
                        
                        # Try to identify column by header text if available
                        if i < len(headers):
                            header = headers[i]
                            if any(key in header for key in ["descrip", "title", "name"]):
                                row_data["description"] = cell_text
                            elif any(key in header for key in ["date", "updated"]):
                                row_data["last_updated"] = cell_text
                            elif any(key in header for key in ["status", "archive"]):
                                row_data["status"] = cell_text
                        
                        # Fallback to position-based mapping
                        if "description" not in row_data and i == 1:
                            row_data["description"] = cell_text
                        if "last_updated" not in row_data and i == 2:
                            row_data["last_updated"] = cell_text
                        if "status" not in row_data and i == 0:
                            row_data["status"] = cell_text
                    
                    # Process links in the cells, typically in the last cell
                    doc_cell = cells[-1]
                    for link in doc_cell.find_all('a', href=True):
                        href = link['href']
                        if href.lower().endswith('.pdf'):
                            full_url = href
                            if not full_url.startswith(('http://', 'https://')):
                                full_url = urllib.parse.urljoin(base_url, href)
                            
                            # Skip if we've already processed this URL
                            if full_url in processed_urls:
                                continue
                            
                            processed_urls.add(full_url)
                            
                            # Get link text or fallback to description
                            link_text = link.get_text().strip()
                            if not link_text and "description" in row_data:
                                link_text = row_data["description"]
                            if not link_text:
                                link_text = "PDF Document"
                            
                            # Build context from row data and table caption
                            context_parts = []
                            if table_caption:
                                context_parts.append(f"Section: {table_caption}")
                            
                            for key, value in row_data.items():
                                if value:
                                    context_parts.append(f"{key.capitalize()}: {value}")
                            
                            context = " | ".join(context_parts)
                            
                            # Add the PDF link with rich metadata
                            pdf_links.append({
                                'url': full_url,
                                'text': link_text,
                                'context': context,
                                'metadata': {
                                    'table': f"Table {table_index + 1}",
                                    'section': table_caption,
                                    **row_data  # Include all row data as metadata
                                }
                            })
                except Exception as e:
                    logger.warning(f"Error processing row in table {table_index}: {str(e)}")
                    continue
        except Exception as e:
            logger.warning(f"Error processing table {table_index}: {str(e)}")
            continue
    
    # Then get PDFs from general links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            full_url = href
            if not full_url.startswith(('http://', 'https://')):
                full_url = urllib.parse.urljoin(base_url, href)
            
            # Skip if we've already processed this URL
            if full_url in processed_urls:
                continue
            
            processed_urls.add(full_url)
            
            # Get link text
            link_text = link.get_text().strip() or "PDF Document"
            
            # Get section context
            section_header = get_nearest_header(link)
            
            # Get surrounding text
            surrounding_text = get_surrounding_text(link)
            
            # Try to find a date near the link
            date_pattern = re.compile(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{1,2}[\/\-\.]\w+[\/\-\.]\d{2,4}')
            date_match = date_pattern.search(surrounding_text)
            date_str = date_match.group(0) if date_match else ""
            
            # Try to find if this is an act, regulation, circular, etc.
            doc_type = ""
            for doc_keyword in ["Act", "Regulation", "Circular", "Notification", "Guidelines", "Master Circular"]:
                if doc_keyword.lower() in link_text.lower() or doc_keyword.lower() in surrounding_text.lower():
                    doc_type = doc_keyword
                    break
            
            # Build a context string that combines all the information
            context = ""
            if section_header:
                context += f"Section: {section_header}"
            if doc_type:
                context += f" | Type: {doc_type}"
            if date_str:
                context += f" | Date: {date_str}"
            if surrounding_text:
                context += f" | Context: {surrounding_text[:100]}..."
            
            # Add to PDF links with rich metadata
            pdf_links.append({
                'url': full_url,
                'text': link_text,
                'context': context,
                'metadata': {
                    'section': section_header,
                    'document_type': doc_type,
                    'date': date_str,
                    'surrounding_text': surrounding_text
                }
            })
    
    # Sort PDFs by relevance: prioritize those with more metadata
    pdf_links.sort(key=lambda x: sum(1 for v in x['metadata'].values() if v), reverse=True)
    
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
            chunk_size=800,
            chunk_overlap=150
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
            search_kwargs={"k": 5,
                          "fetch_k": 20,
                          "lanbd_mult": 0.7}
        )
        
        template = """
You are an expert assistant for insurance regulatory information in India with knowledge about IRDAI, UIDAI, and Enforcement Directorate regulations. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the information provided in the context above.
2. If the information is not available or unclear in the context, explicitly say "Based on the provided information, I cannot answer this question" or "The context does not provide sufficient information about this."
3. Always specify dates explicitly (e.g., "as of 15 January 2023") when mentioning time-sensitive information.
4. When relevant, cite specific section numbers, regulation names, and direct quotes from the context.
5. If PDF documents are mentioned in the context, highlight their relevance to the answer.
6. Begin your answer by mentioning which sources from the context contain the information you're using.
7. Do not speculate beyond what's in the context, even if you think you know the answer.
8. Focus on the most recent information when dates are provided in the context.
9. Use bullet points for clarity when listing multiple items or requirements.

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

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 5):
    if not pdf_links:
        return []
    
    try:
        # Semantic search with embeddings
        embedder = CustomEmbeddings()
        query_embedding = embedder.embed_query(query)
        
        pdf_texts = []
        for pdf in pdf_links:
            context_text = f"{pdf['text']} {pdf.get('context', '')}"
            if 'metadata' in pdf and pdf['metadata']:
                for key, value in pdf['metadata'].items():
                    if value:
                        context_text += f" {key}: {value}"
            pdf_texts.append(context_text)
        
        pdf_embeddings = np.array(embedder.embed_documents(pdf_texts))
        
        dimension = len(query_embedding)
        index = faiss.IndexFlatL2(dimension)
        index.add(pdf_embeddings)
        
        distances, indices = index.search(np.array([query_embedding]), top_k)
        
        # Also perform keyword matching
        query_terms = set(query.lower().split())
        keyword_scores = []
        
        for i, pdf in enumerate(pdf_links):
            pdf_text = f"{pdf['text']} {pdf.get('context', '')}"
            if 'metadata' in pdf and pdf['metadata']:
                for k, v in pdf['metadata'].items():
                    if v:
                        pdf_text += f" {k}: {v}"
            
            pdf_text = pdf_text.lower()
            matches = sum(1 for term in query_terms if term in pdf_text)
            keyword_scores.append(matches / len(query_terms) if query_terms else 0)
        
        # Combine semantic and keyword results (hybrid approach)
        combined_results = {}
        
        # Add semantic search results
        for rank, idx in enumerate(indices[0]):
            if idx < len(pdf_links):
                combined_results[idx] = 0.7 * (1 / (rank + 1))  # Higher rank = better score
        
        # Add keyword results
        for idx, score in enumerate(keyword_scores):
            if idx in combined_results:
                combined_results[idx] += 0.3 * score
            elif score > 0:
                combined_results[idx] = 0.3 * score
        
        # Sort by combined score
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, _ in sorted_results[:top_k]:
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
