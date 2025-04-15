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
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationRetrievalChain
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

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
os.makedirs(CACHE_DIR, exist_ok=True)

# Replace TF-IDF with Neural Embeddings
class NeuralEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# Use neural embeddings instead of TF-IDF
embeddings = NeuralEmbeddings()

def rerank_documents(query, documents, top_k=5):
    try:
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        
        # Create (score, doc) pairs and sort by score
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(reverse=True)  # Higher score = better
        
        # Return top_k documents
        return [doc for _, doc in scored_docs[:top_k]]
    except Exception as e:
        st.warning(f"Error during reranking: {str(e)}")
        return documents[:top_k]  # Fallback to original ordering

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

# New function to extract text from PDFs
def extract_pdf_content(pdf_url):
    try:
        import fitz  # PyMuPDF
        
        response = requests.get(pdf_url, stream=True, timeout=10)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                
                doc = fitz.open(temp_file.name)
                text = ""
                for page in doc:
                    text += page.get_text()
                
                return text
        return None
    except Exception as e:
        print(f"Error extracting PDF content: {str(e)}")
        return None

def initialize_rag_system():
    st.session_state.status = "Initializing RAG system"
    
    all_docs = []
    all_pdf_links = []
    
    progress_bar = st.progress(0)
    for i, website in enumerate(WEBSITES):
        st.session_state.status = f"Processing {website}..."
        content, pdf_links = fetch_website_content(website)
        
        # Improved text splitting with larger chunks and more overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500
            chunk_overlap=200,  # Increased from 50
            separators=["\n\n", "\n", " ", ""]  # Better natural breaks
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
    
    # Process a limited number of PDFs to extract their content
    st.session_state.status = "Processing PDF documents..."
    pdf_progress = st.progress(0)
    max_pdfs = min(10, len(all_pdf_links))  # Limit to 10 PDFs for performance
    
    for i, pdf_link in enumerate(all_pdf_links[:max_pdfs]):
        pdf_content = extract_pdf_content(pdf_link['url'])
        if pdf_content:
            chunks = text_splitter.split_text(pdf_content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_link['url'], 
                        "type": "pdf", 
                        "title": pdf_link['text'],
                        "description": pdf_link['metadata'].get('description', ''),
                        "last_updated": pdf_link['metadata'].get('last_updated', '')
                    }
                ))
        pdf_progress.progress((i + 1) / max_pdfs)
    
    st.session_state.status = "Creating embeddings and vector store..."
    
    # Create the vector store
    vector_store = LangchainFAISS.from_documents(all_docs, embeddings)
    
    # Create BM25 retriever for keyword search
    texts = [doc.page_content for doc in all_docs]
    bm25_retriever = BM25Retriever.from_texts(texts)
    bm25_retriever.k = 5
    
    # Create vector store retriever for semantic search
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create ensemble retriever combining both approaches
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]  # Weight semantic search higher
    )
    
    st.session_state.vector_store = vector_store
    st.session_state.retriever = ensemble_retriever
    st.session_state.pdf_links = all_pdf_links
    st.session_state.status = "System initialized!"
    st.session_state.initialized = True

def initialize_llm():
    groq_api_key = st.session_state.groq_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    
    # Enhanced prompt with better instructions
    template = """
    You are an expert assistant specializing in Indian insurance and identity regulatory information from IRDAI and UIDAI websites. Answer the question based on the provided context.
    
    Follow these guidelines:
    1. Focus on extracting factual information directly from the provided context
    2. When mentioning dates, always include the full date format as it appears in the source
    3. If referring to regulatory documents, include their reference numbers and publication dates
    4. When citing information, clearly indicate which specific website the information comes from
    5. If the question asks about "latest" information, identify the most recent document based on date and explain why it's the most current
    6. If the context doesn't contain enough information to answer confidently, state this clearly
    
    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    
    Step by step thinking:
    1. First, identify the key entities and concepts in the question
    2. Scan the context for relevant information related to these entities
    3. Analyze any dates to determine chronology and recency
    4. Verify if the information directly answers the question
    5. Formulate a precise, factual answer based strictly on the context provided
    
    Your answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    
    # Add conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"  # Make sure this matches what your chain returns
    )
    
    # Use RetrievalQA chain with memory integration
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "memory": memory
        }
    )
    
    st.session_state.qa_chain = qa_chain
    st.session_state.memory = memory

def get_relevant_context(query):
    # First get more documents than needed
    initial_docs = st.session_state.retriever.get_relevant_documents(query)
    
    # Then rerank to get the most relevant ones
    reranked_docs = rerank_documents(query, initial_docs, top_k=5)
    
    return reranked_docs

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 3):
    if not pdf_links:
        return []
    
    # Use the same neural embeddings for PDF retrieval
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    pdf_texts = []
    for pdf in pdf_links:
        context_text = f"{pdf['text']} {pdf['context']}"
        if 'metadata' in pdf and pdf['metadata']:
            for key, value in pdf['metadata'].items():
                context_text += f" {key}: {value}"
        pdf_texts.append(context_text)
    
    # Get embeddings for pdf_texts
    pdf_embeddings = np.array(model.encode(pdf_texts))
    
    # Get query embedding
    query_embedding = np.array(model.encode([query])[0])
    query_embedding_np = query_embedding.reshape(1, -1)
    
    dimension = pdf_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(pdf_embeddings)
    
    distances, indices = index.search(query_embedding_np, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(pdf_links):
            results.append(pdf_links[idx])
    
    return results

st.title("Web Intelligence BOT")

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

if 'status' in st.session_state:
    st.info(st.session_state.status)

if 'initialized' in st.session_state and st.session_state.initialized:
    st.subheader("Ask a question")
    query = st.text_input("What would you like to know?")
    
    if query and st.button("Search"):
        with st.spinner("Searching for information..."):
            # Get relevant context documents via reranking
            relevant_docs = get_relevant_context(query)
            
            # Process query with conversation chain
            result = st.session_state.qa_chain({"question": query})
            
            # Extract answer and sources
            if "answer" in result:  # ConversationRetrievalChain format
                answer = result["answer"]
            else:  # Fallback for other chain formats
                answer = result["result"]
                
            source_docs = result["source_documents"] if "source_documents" in result else []
            
            # Find relevant PDFs
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
                st.info("No specific PDF documents found for this query.")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
