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
from langchain_community.embeddings import FastEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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

embeddings = FastEmbeddings()

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
    
    st.session_state.status = "Building vector store"
    vector_store = LangchainFAISS.from_documents(all_docs, embeddings)
    
    st.session_state.vector_store = vector_store
    st.session_state.pdf_links = all_pdf_links
    st.session_state.status = "System initialized!"
    st.session_state.initialized = True

def initialize_llm():
    groq_api_key = st.session_state.groq_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-4-scout-17b-16e-instruct"
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

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 3):
    if not pdf_links:
        return []
    
    # Replace SentenceTransformer with FastEmbeddings
    embedder = FastEmbeddings()
    query_embedding = embedder.embed_query(query)
    
    pdf_texts = []
    for pdf in pdf_links:
        context_text = f"{pdf['text']} {pdf['context']}"
        if 'metadata' in pdf and pdf['metadata']:
            for key, value in pdf['metadata'].items():
                context_text += f" {key}: {value}"
        pdf_texts.append(context_text)
    
    # Use the embedder to encode all pdf texts
    pdf_embeddings = np.array(embedder.embed_documents(pdf_texts))
    
    dimension = pdf_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(pdf_embeddings)
    
    # Convert query_embedding to numpy array with correct shape
    query_embedding_np = np.array([query_embedding])
    
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
                st.info(" ")
else:
    if 'initialized' not in st.session_state:
        st.info("Please enter your Groq API key and initialize the system.")
    elif not st.session_state.initialized:
        st.info("System initialization in progress...")

with st.expander("Indexed Websites"):
    for website in WEBSITES:
        st.write(f"- {website}")
