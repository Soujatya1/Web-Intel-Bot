import streamlit as st
import requests
import os
import re
import faiss
import numpy as np
import tempfile
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

# Same website list as before
WEBSITES = ["https://irdai.gov.in/rules",
            "https://irdai.gov.in/consolidated-gazette-notified-regulations",
            # ... rest of your websites
]

CACHE_DIR = ".web_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-mpnet-base-v2"):  # Using a stronger embedding model
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

def fetch_website_content(url: str) -> Tuple[str, List[Dict]]:
    """Fetch content from a website with improved extraction."""
    
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
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove non-content elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
        element.extract()
    
    # Extract main content areas more intelligently
    main_content = ""
    main_elements = soup.select('main, article, .content, #content, .main, #main')
    if main_elements:
        for element in main_elements:
            main_content += element.get_text(separator="\n") + "\n\n"
    else:
        # Fallback to body content
        main_content = soup.body.get_text(separator="\n") if soup.body else ""
    
    table_data = extract_table_data(soup, url)
    
    # Preserve paragraph structure better
    paragraphs = re.split(r'\n\s*\n', main_content)
    cleaned_paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in paragraphs if p.strip()]
    text = "\n\n".join(cleaned_paragraphs)
    
    # Add source URL as context
    combined_text = f"Source: {url}\n\n{text}\n\n{table_data}"
    
    pdf_links = extract_pdf_links(soup, url)
    
    return combined_text, pdf_links

def extract_table_data(soup, base_url):
    """Extract tabular data with improved structure preservation."""
    table_data = ""
    
    tables = soup.find_all('table')
    
    for i, table in enumerate(tables):
        # Add table identifier to help maintain context
        table_data += f"\nTable {i+1} from {base_url}:\n"
        
        # Extract headers
        headers = [th.get_text().strip() for th in table.find_all('th')]
        if headers:
            table_data += " | ".join(headers) + "\n"
            table_data += "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)) + "\n"
        
        # Extract rows with better formatting
        for row in table.find_all('tr')[1:] if headers else table.find_all('tr'):
            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
            if cells:
                table_data += " | ".join(cells) + "\n"
        
        table_data += "\n"
        
        # Handle IRDAI specific tables (keeping your logic here)
        if any(header in " ".join(headers) for header in ["Archive", "Description", "Last Updated", "Documents"]):
            table_data += "IRDAI Acts Information:\n"
            
            for row in table.find_all('tr')[1:]:
                # Your existing IRDAI table processing...
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
            
            # Find latest acts (keeping your logic)
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

# extract_pdf_links function remains mostly the same
def extract_pdf_links(soup, base_url):
    """Extract PDF links with improved metadata extraction."""
    # Same implementation as your original function
    pdf_links = []
    
    # Your existing PDF extraction code...
    
    return pdf_links

def initialize_rag_system():
    """Initialize the RAG system with improved document processing."""
    st.session_state.status = "Initializing RAG system..."
    
    all_docs = []
    all_pdf_links = []
    
    progress_bar = st.progress(0)
    for i, website in enumerate(WEBSITES):
        st.session_state.status = f"Processing {website}..."
        content, pdf_links = fetch_website_content(website)
        
        # Using a better chunking strategy with larger chunks and more overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better context
            chunk_overlap=200,  # More overlap to preserve context across chunks
            separators=["\n\n", "\n", ". ", " ", ""],  # Better splitting at natural boundaries
            keep_separator=True
        )
        
        if content and not content.startswith("Error"):
            # Add the source URL and publish date as metadata for better context
            source_metadata = {"source": website, "domain": urllib.parse.urlparse(website).netloc}
            
            # Extract and add date information if available
            date_match = re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', content[:1000])
            if date_match:
                source_metadata["date"] = date_match.group(1)
            
            # Create documents with rich metadata
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **source_metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                ))
            all_pdf_links.extend(pdf_links)
        
        progress_bar.progress((i + 1) / len(WEBSITES))
    
    st.session_state.status = "Creating embeddings..."
    embeddings = SentenceTransformerEmbeddings()
    
    st.session_state.status = "Building vector store..."
    vector_store = LangchainFAISS.from_documents(all_docs, embeddings)
    
    st.session_state.vector_store = vector_store
    st.session_state.pdf_links = all_pdf_links
    st.session_state.status = "System initialized!"
    st.session_state.initialized = True

def initialize_llm():
    """Initialize the language model with improved retrieval."""
    groq_api_key = st.session_state.groq_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.1  # Lower temperature for more factual responses
    )
    
    # Create a base retriever with similarity search and MMR combination
    base_retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",  # Using MMR for better diversity
        search_kwargs={
            "k": 10,  # Retrieve more documents initially
            "fetch_k": 20,  # Consider more candidates
            "lambda_mult": 0.7  # Balance relevance and diversity
        }
    )
    
    # Add contextual compression to filter out irrelevant passages
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Improved prompt template with specific guidance
    template = """
    You are an expert assistant specializing in insurance and regulatory information for IRDAI (Insurance Regulatory and Development Authority of India) and UIDAI (Unique Identification Authority of India). 
    
    Answer the question based ONLY on the provided context. Be direct, accurate, and provide specific details from the context.
    
    Guidelines:
    1. When mentioning regulations or acts, always include their full names, dates, and reference numbers if available.
    2. If asked about the "latest" regulations, focus on the most recently updated ones based on dates in the context.
    3. Always cite the source of your information by mentioning the specific website or document.
    4. If the information is not available in the context, clearly state that you cannot provide an answer based on the available information.
    5. Do not make up information or guess if it's not in the context.
    6. If there are PDF documents that might contain relevant information, suggest them as additional resources.
    
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

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 5):
    """Find relevant PDFs with improved semantic matching."""
    if not pdf_links:
        return []
    
    # Use same model as main embeddings for consistency
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Enhanced query expansion for better matching
    expanded_query = f"{query} insurance regulation policy document IRDAI UIDAI"
    query_embedding = model.encode(expanded_query)
    
    # Build richer context for each PDF
    pdf_texts = []
    for pdf in pdf_links:
        # Create a more comprehensive representation
        context_text = f"Title: {pdf['text']} "
        context_text += f"Context: {pdf['context']} "
        
        if 'metadata' in pdf and pdf['metadata']:
            for key, value in pdf['metadata'].items():
                context_text += f"{key}: {value} "
        
        # Add the URL as a feature (domain name can be informative)
        domain = urllib.parse.urlparse(pdf['url']).netloc
        context_text += f"Source: {domain}"
        
        pdf_texts.append(context_text)
    
    pdf_embeddings = model.encode(pdf_texts)
    
    # Create FAISS index with cosine similarity (L2 normalized vectors)
    dimension = pdf_embeddings.shape[1]
    pdf_embeddings_normalized = pdf_embeddings / np.linalg.norm(pdf_embeddings, axis=1, keepdims=True)
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(pdf_embeddings_normalized)
    
    distances, indices = index.search(np.array([query_embedding_normalized]), top_k)
    
    # Filter by relevance threshold
    relevance_threshold = 0.5  # Adjust as needed
    results = []
    
    for i, idx in enumerate(indices[0]):
        similarity = distances[0][i]
        if similarity > relevance_threshold and idx < len(pdf_links):
            # Add similarity score to the result
            result = pdf_links[idx].copy()
            result['similarity'] = float(similarity)
            results.append(result)
    
    return results

# UI improvements
st.title("Insurance Regulatory Intelligence Bot")

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
    
    # Add advanced settings
    with st.expander("Advanced Settings"):
        if 'initialized' in st.session_state and st.session_state.initialized:
            st.slider("Number of documents to retrieve", 3, 15, 10, key="k_documents")
            st.slider("Relevance threshold (0-100)", 0, 100, 50, key="relevance_threshold")

if 'status' in st.session_state:
    st.info(st.session_state.status)

if 'initialized' in st.session_state and st.session_state.initialized:
    st.subheader("Ask a question about IRDAI or UIDAI regulations")
    
    # Suggested questions for better user experience
    st.caption("Example questions:")
    example_questions = [
        "What are the latest IRDAI regulations?",
        "What are the KYC requirements for insurance?",
        "What is the process for filing a complaint against an insurance company?",
        "What are the penalties for non-compliance with IRDAI regulations?",
        "What are the latest updates to the Aadhaar Act?"
    ]
    for q in example_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state.query = q
    
    # Query input with history
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    query = st.text_input("What would you like to know?", value=st.session_state.query)
    
    if query and st.button("Search", use_container_width=True):
        st.session_state.query = query  # Save query for context
        
        with st.spinner("Searching for information..."):
            # Update retrieval parameters from UI if available
            if 'k_documents' in st.session_state:
                st.session_state.qa_chain.retriever.search_kwargs["k"] = st.session_state.k_documents
            
            if 'relevance_threshold' in st.session_state:
                threshold = st.session_state.relevance_threshold / 100
                # Apply threshold in the find_relevant_pdfs function
            
            result = st.session_state.qa_chain({"query": query})
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Find relevant PDFs with updated threshold
            relevant_pdfs = find_relevant_pdfs(
                query, 
                st.session_state.pdf_links, 
                top_k=5
            )
            
            st.subheader("Answer")
            st.markdown(answer)
            
            # Source citation with better formatting
            with st.expander("Sources"):
                sources = {}
                for doc in source_docs:
                    source_url = doc.metadata["source"]
                    domain = urllib.parse.urlparse(source_url).netloc
                    
                    if source_url not in sources:
                        sources[source_url] = {
                            "domain": domain,
                            "chunks": []
                        }
                    
                    # Add a snippet from the document with highlighting
                    snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources[source_url]["chunks"].append(snippet)
                
                # Display sources with snippets
                for source_url, info in sources.items():
                    st.markdown(f"#### [{info['domain']}]({source_url})")
                    for i, chunk in enumerate(info["chunks"][:3]):  # Limit to 3 snippets per source
                        st.markdown(f"**Excerpt {i+1}:**")
                        st.text(chunk)
                    
                    if len(info["chunks"]) > 3:
                        st.caption(f"...and {len(info['chunks']) - 3} more relevant sections")
            
            # PDF recommendations with clear relevance indicators
            if relevant_pdfs:
                st.subheader("Relevant PDF Documents")
                
                for pdf in sorted(relevant_pdfs, key=lambda x: x.get('similarity', 0), reverse=True):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"[{pdf['text']}]({pdf['url']})")
                        
                        metadata_text = ""
                        if 'metadata' in pdf and pdf['metadata']:
                            for key, value in pdf['metadata'].items():
                                if value:
                                    metadata_text += f"{key}: {value}, "
                            metadata_text = metadata_text.rstrip(", ")
                        
                        if metadata_text:
                            st.caption(f"{metadata_text}")
                        else:
                            st.caption(f"Context: {pdf['context'][:100]}...")
                    
                    with col2:
                        # Show relevance score as a progress bar
                        if 'similarity' in pdf:
                            relevance = int(pdf['similarity'] * 100)
                            st.caption("Relevance:")
                            st.progress(pdf['similarity'])
                            st.caption(f"{relevance}%")
            else:
                st.info("No relevant PDF documents found")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
