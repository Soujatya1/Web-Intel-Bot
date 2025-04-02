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

st.set_page_config(page_title="Web Intelligence BOT", layout="wide")

# Group websites by category to improve context and retrieval
WEBSITES = {
    "Rules": [
        "https://irdai.gov.in/rules",
        "https://irdai.gov.in/rules2",
        "https://uidai.gov.in/en/about-uidai/legal-framework/rules.html",
        "https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules"
    ],
    "Regulations": [
        "https://irdai.gov.in/consolidated-gazette-notified-regulations",
        "https://irdai.gov.in/updated-regulations",
        "https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html",
        "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation"
    ],
    "Circulars": [
        "https://irdai.gov.in/circulars",
        "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html"
    ],
    "Notifications": [
        "https://irdai.gov.in/notifications",
        "https://uidai.gov.in/en/about-uidai/legal-framework/notifications.html"
    ],
    "Legal Framework": [
        "https://uidai.gov.in/en/about-uidai/legal-framework.html",
        "https://uidai.gov.in/en/about-uidai/legal-framework/judgements.html"
    ],
    "Enforcement": [
        "https://enforcementdirectorate.gov.in/pmla",
        "https://enforcementdirectorate.gov.in/fema",
        "https://enforcementdirectorate.gov.in/bns",
        "https://enforcementdirectorate.gov.in/bnss",
        "https://enforcementdirectorate.gov.in/bsa"
    ],
    "Other": [
        "https://irdai.gov.in/orders1",
        "https://irdai.gov.in/exposure-drafts",
        "https://irdai.gov.in/programmes-to-advance-understanding-of-rti",
        "https://irdai.gov.in/cic-orders",
        "https://irdai.gov.in/antimoney-laundering",
        "https://irdai.gov.in/other-communication",
        "https://irdai.gov.in/directory-of-employees",
        "https://irdai.gov.in/warnings-and-penalties",
        "https://uidai.gov.in/en/"
    ]
}

# Flatten the website list for processing
ALL_WEBSITES = [url for category, urls in WEBSITES.items() for url in urls]

CACHE_DIR = ".web_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])[0]
        return embedding.tolist()

def categorize_url(url: str) -> str:
    """Determine the category of a URL."""
    for category, urls in WEBSITES.items():
        if url in urls:
            return category
    return "Uncategorized"

def fetch_website_content(url: str) -> Tuple[str, List[Dict]]:
    """Fetch content from a website and extract text, tables, and PDF links."""
    
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
    
    # Get the category for context enrichment
    category = categorize_url(url)
    
    # Enhanced table extraction with category metadata
    table_data = extract_table_data(soup, url, category)
    
    # Add more structure to the extracted text
    text = extract_structured_text(soup, url, category)
    
    # Add category context to beginning of document
    category_context = f"DOCUMENT CATEGORY: {category}\nSOURCE URL: {url}\n\n"
    combined_text = category_context + text + "\n\n" + table_data
    
    # Extract PDF links with improved metadata
    pdf_links = extract_pdf_links(soup, url, category)
    
    return combined_text, pdf_links

def extract_structured_text(soup, url, category):
    """Extract text with improved structure and context preservation."""
    text_parts = []
    
    # Add the page title for context
    title = soup.find('title')
    if title:
        text_parts.append(f"PAGE TITLE: {title.get_text().strip()}")
    
    # Extract headings and their content for better structure
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        heading_text = heading.get_text().strip()
        if heading_text:
            # Get the next siblings until the next heading
            content_parts = []
            sibling = heading.next_sibling
            while sibling and not sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if sibling.name and sibling.get_text().strip():
                    content_parts.append(sibling.get_text().strip())
                sibling = sibling.next_sibling
            
            section_text = f"{heading.name.upper()}: {heading_text}\n"
            if content_parts:
                section_text += "\n".join(content_parts)
            text_parts.append(section_text)
    
    # For circulars and rules, make additional effort to extract structured data
    if category in ["Circulars", "Rules"]:
        # Look for divs with class containing terms like "circular", "rule", "document", etc.
        for div in soup.find_all('div', class_=re.compile(r'(circular|rule|document|content|item)', re.IGNORECASE)):
            div_text = div.get_text().strip()
            if div_text:
                # Look for dates in the text as they are important for circulars/rules
                date_matches = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}', div_text)
                if date_matches:
                    text_parts.append(f"DATED CONTENT ({', '.join(date_matches)}):\n{div_text}")
                else:
                    text_parts.append(f"RELEVANT CONTENT:\n{div_text}")
        
        # Look for list items which often contain circulars or rules
        list_texts = []
        for li in soup.find_all('li'):
            li_text = li.get_text().strip()
            if re.search(r'(circular|rule|notification|dated|issued|reference)', li_text, re.IGNORECASE):
                list_texts.append(f"- {li_text}")
        
        if list_texts:
            text_parts.append("LIST ITEMS:\n" + "\n".join(list_texts))
    
    # Get main text if the above methods missed something
    main_text = soup.get_text(separator="\n")
    main_text = re.sub(r'\n+', '\n', main_text)
    main_text = re.sub(r'\s+', ' ', main_text)
    
    # Combine all parts
    combined = "\n\n".join(text_parts)
    
    # If we didn't extract much structured content, use the main text
    if len(combined) < 200:
        return f"CATEGORY: {category}\n{main_text}"
    
    return combined

def extract_table_data(soup, base_url, category):
    table_data = ""
    
    tables = soup.find_all('table')
    
    for table_idx, table in enumerate(tables):
        headers = [th.get_text().strip() for th in table.find_all('th')]
        if not headers and table.find('tr'):
            # Try to extract headers from the first row if no <th> tags
            headers = [td.get_text().strip() for td in table.find('tr').find_all('td')]
        
        table_data += f"\nTABLE {table_idx+1} IN {category} SECTION:\n"
        table_data += f"Headers: {', '.join(headers) if headers else 'No headers'}\n"
        
        # General table extraction regardless of header names
        rows_data = []
        
        for row_idx, row in enumerate(table.find_all('tr')[1:] if headers else table.find_all('tr')):
            cells = row.find_all(['td', 'th'])
            if cells:
                row_text = [cell.get_text().strip() for cell in cells]
                
                # Check for dates or document IDs in this row
                date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}')
                id_pattern = re.compile(r'[A-Z]+-\d+|No\.\s*\d+|Circular No|Rule \d+', re.IGNORECASE)
                
                dates_found = []
                ids_found = []
                
                for cell_text in row_text:
                    date_matches = date_pattern.findall(cell_text)
                    id_matches = id_pattern.findall(cell_text)
                    dates_found.extend(date_matches)
                    ids_found.extend(id_matches)
                
                # Format the row with header names if available
                if headers and len(cells) <= len(headers):
                    row_data = "Row " + str(row_idx+1) + ": "
                    for i, cell in enumerate(cells):
                        header_name = headers[i] if i < len(headers) else f"Column {i+1}"
                        cell_text = cell.get_text().strip()
                        if cell_text:
                            row_data += f"{header_name}: {cell_text}; "
                    
                    if dates_found:
                        row_data += f"Dates: {', '.join(dates_found)}; "
                    if ids_found:
                        row_data += f"IDs: {', '.join(ids_found)}; "
                    
                    rows_data.append(row_data.rstrip("; "))
                else:
                    # Simple row extraction without headers
                    rows_data.append(f"Row {row_idx+1}: " + "; ".join([cell for cell in row_text if cell]))
                
                # Extract PDF links from this row
                for cell in cells:
                    for link in cell.find_all('a', href=True):
                        href = link['href']
                        if href.lower().endswith('.pdf'):
                            if not href.startswith(('http://', 'https://')):
                                href = urllib.parse.urljoin(base_url, href)
                            link_text = link.get_text().strip()
                            rows_data.append(f"PDF Link: [{link_text}]({href})")
        
        table_data += "\n".join(rows_data) + "\n"
    
    # For circular and rules pages, make an extra effort to identify the latest information
    if category in ["Circulars", "Rules"]:
        # Try to find the latest circular/rule by date
        date_entries = []
        for table in tables:
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                row_text = " ".join([cell.get_text().strip() for cell in cells])
                
                date_matches = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}', row_text)
                if date_matches:
                    # Look for circular/rule identifiers
                    id_matches = re.findall(r'[A-Z]+-\d+|No\.\s*\d+|Circular No|Rule \d+', row_text, re.IGNORECASE)
                    id_text = ", ".join(id_matches) if id_matches else "Unknown ID"
                    
                    # Get any links in this row
                    links = []
                    for cell in cells:
                        for link in cell.find_all('a', href=True):
                            href = link['href']
                            if not href.startswith(('http://', 'https://')):
                                href = urllib.parse.urljoin(base_url, href)
                            link_text = link.get_text().strip()
                            links.append(f"[{link_text}]({href})")
                    
                    link_text = ", ".join(links) if links else "No direct link"
                    
                    date_entries.append({
                        "date": date_matches[0],
                        "id": id_text,
                        "text": row_text[:200],  # First 200 chars of row text
                        "links": link_text
                    })
        
        if date_entries:
            # Sort by date (this is simplistic - would need better date parsing for production)
            table_data += f"\nLATEST {category.upper()} FOUND:\n"
            for entry in date_entries[:5]:  # Show top 5 entries
                table_data += f"Date: {entry['date']}, ID: {entry['id']}\n"
                table_data += f"Description: {entry['text']}\n"
                table_data += f"Links: {entry['links']}\n\n"
    
    return table_data

def extract_pdf_links(soup, base_url, category):
    """Extract PDF links with improved metadata extraction."""
    pdf_links = []
    
    # Function to standardize PDF URL
    def standardize_url(href):
        if not href.startswith(('http://', 'https://')):
            return urllib.parse.urljoin(base_url, href)
        return href
    
    # Find PDFs in tables first (more structured)
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            # Skip if not enough cells
            if len(cells) < 2:
                continue
                
            try:
                # Get row context
                row_text = " ".join([cell.get_text().strip() for cell in cells])
                
                # Extract dates
                date_matches = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}', row_text)
                date_info = date_matches[0] if date_matches else ""
                
                # Extract IDs/references
                id_matches = re.findall(r'[A-Z]+-\d+|No\.\s*\d+|Circular No|Rule \d+', row_text, re.IGNORECASE)
                id_info = id_matches[0] if id_matches else ""
                
                # Look for description - often in 2nd column
                description = cells[1].get_text().strip() if len(cells) > 1 else ""
                
                # Look for PDF links in any cell
                for cell in cells:
                    for link in cell.find_all('a', href=True):
                        href = link['href']
                        if href.lower().endswith('.pdf'):
                            pdf_url = standardize_url(href)
                            link_text = link.get_text().strip()
                            
                            # Create rich metadata
                            pdf_links.append({
                                'url': pdf_url,
                                'text': link_text or description,
                                'context': f"{category} document: {description}" + (f", dated {date_info}" if date_info else ""),
                                'metadata': {
                                    'category': category,
                                    'description': description,
                                    'date': date_info,
                                    'id': id_info,
                                    'source_url': base_url
                                }
                            })
            except Exception as e:
                continue
    
    # Find PDFs in lists (common for circulars and rules)
    list_items = soup.find_all('li')
    for item in list_items:
        item_text = item.get_text().strip()
        
        # Extract dates
        date_matches = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}', item_text)
        date_info = date_matches[0] if date_matches else ""
        
        # Extract IDs/references
        id_matches = re.findall(r'[A-Z]+-\d+|No\.\s*\d+|Circular No|Rule \d+', item_text, re.IGNORECASE)
        id_info = id_matches[0] if id_matches else ""
        
        for link in item.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                pdf_url = standardize_url(href)
                link_text = link.get_text().strip()
                
                pdf_links.append({
                    'url': pdf_url,
                    'text': link_text,
                    'context': f"{category} document in list: {item_text[:100]}",
                    'metadata': {
                        'category': category,
                        'list_text': item_text[:200],
                        'date': date_info,
                        'id': id_info,
                        'source_url': base_url
                    }
                })
    
    # Find any remaining PDF links
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            # Skip if we already found this PDF
            pdf_url = standardize_url(href)
            if any(pdf['url'] == pdf_url for pdf in pdf_links):
                continue
            
            # Get surrounding context
            surrounding_text = link.get_text().strip() or "PDF Document"
            parent = link.parent
            parent_text = parent.get_text().strip() if parent else ""
            
            # Extract dates from context
            date_matches = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2} [A-Za-z]+ \d{4}', parent_text)
            date_info = date_matches[0] if date_matches else ""
            
            pdf_links.append({
                'url': pdf_url,
                'text': surrounding_text,
                'context': f"{category} document: {parent_text[:100]}",
                'metadata': {
                    'category': category,
                    'surrounding_text': parent_text[:200],
                    'date': date_info,
                    'source_url': base_url
                }
            })
    
    return pdf_links

def initialize_rag_system():
    """Initialize the RAG system by scraping websites and creating vector store."""
    st.session_state.status = "Initializing RAG system..."
    
    all_docs = []
    all_pdf_links = []
    
    progress_bar = st.progress(0)
    for i, website in enumerate(ALL_WEBSITES):
        st.session_state.status = f"Processing {website}..."
        content, pdf_links = fetch_website_content(website)
        
        # Use larger chunk size to keep more context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500
            chunk_overlap=200  # Increased from 50
        )
        
        if content and not content.startswith("Error"):
            chunks = text_splitter.split_text(content)
            
            # Add category metadata to help with retrieval
            category = categorize_url(website)
            
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": website,
                        "category": category
                    }
                ))
            all_pdf_links.extend(pdf_links)
        
        progress_bar.progress((i + 1) / len(ALL_WEBSITES))
    
    st.session_state.status = "Creating embeddings..."
    embeddings = SentenceTransformerEmbeddings()
    
    st.session_state.status = "Building vector store..."
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
        model_name="llama3-70b-8192"
    )
    
    # Use a hybrid retriever that can fetch by both source and similarity
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}  # Increased from 5
    )
    
    # Improved prompt template that emphasizes source categories
    template = """
    You are an expert assistant for insurance and regulatory information, specializing in IRDAI (Insurance Regulatory and Development Authority of India) and UIDAI (Unique Identification Authority of India) regulations, circulars, and rules.
    
    Answer the question based on the provided context. Be specific and detailed in your responses.
    If the information is not available in the context, say so clearly.
    
    When asked about "Circulars" or "Rules" specifically:
    1. Pay special attention to documents from the Circulars or Rules categories
    2. Include information on the latest circulars/rules if available (look for dates)
    3. Include circular/rule numbers and references when available
    
    When asked about the "latest" acts or documents, focus on the most recently updated ones based on dates in the context.
    Present dates in a clear format and include them in your answer.
    
    Include references to the sources of your information. If PDF links are available, mention them with their titles.
    
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

def preprocess_query(query: str) -> str:
    """Enhance query with category context if applicable."""
    query_lower = query.lower()
    
    if "circular" in query_lower or "circulars" in query_lower:
        return f"Category: Circulars. {query}"
    elif "rule" in query_lower or "rules" in query_lower:
        return f"Category: Rules. {query}"
    elif "regulation" in query_lower or "regulations" in query_lower:
        return f"Category: Regulations. {query}"
    
    return query

def find_relevant_pdfs(query: str, pdf_links: List[Dict], top_k: int = 5):  # Increased from 3
    if not pdf_links:
        return []
    
    # Check if query is about specific categories
    query_lower = query.lower()
    is_circular_query = "circular" in query_lower or "circulars" in query_lower
    is_rule_query = "rule" in query_lower or "rules" in query_lower
    
    # Filter PDFs by category if query is specific
    filtered_pdfs = pdf_links
    if is_circular_query:
        filtered_pdfs = [pdf for pdf in pdf_links if pdf.get('metadata', {}).get('category') == 'Circulars' 
                        or 'circular' in pdf.get('text', '').lower() 
                        or 'circular' in pdf.get('context', '').lower()]
    elif is_rule_query:
        filtered_pdfs = [pdf for pdf in pdf_links if pdf.get('metadata', {}).get('category') == 'Rules'
                        or 'rule' in pdf.get('text', '').lower()
                        or 'rule' in pdf.get('context', '').lower()]
    
    # If no filtered PDFs, fallback to all PDFs
    if not filtered_pdfs:
        filtered_pdfs = pdf_links
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query)
    
    pdf_texts = []
    for pdf in filtered_pdfs:
        context_text = f"{pdf.get('text', '')} {pdf.get('context', '')}"
        metadata = pdf.get('metadata', {})
        for key, value in metadata.items():
            if isinstance(value, str) and value:
                context_text += f" {key}: {value}"
        pdf_texts.append(context_text)
    
    if not pdf_texts:
        return []
        
    pdf_embeddings = model.encode(pdf_texts)
    
    dimension = pdf_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(pdf_embeddings)
    
    distances, indices = index.search(np.array([query_embedding]), min(top_k, len(pdf_texts)))
    
    results = []
    for idx in indices[0]:
        if idx < len(filtered_pdfs):
            results.append(filtered_pdfs[idx])
    
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
    
    # Add category filters
    if 'initialized' in st.session_state and st.session_state.initialized:
        st.subheader("Filter by Category")
        selected_categories = st.multiselect(
            "Select categories to search", 
            options=list(WEBSITES.keys()), 
            default=None
        )
        st.session_state.selected_categories = selected_categories

if 'status' in st.session_state:
    st.info(st.session_state.status)

if 'initialized' in st.session_state and st.session_state.initialized:
    st.subheader("Ask a question")
    query = st.text_input("What would you like to know?")
    
    if query and st.button("Search"):
        with st.spinner("Searching for information..."):
            # Preprocess query to add context for categories
            enhanced_query = preprocess_query(query)
            
            # Apply category filters if selected
            filtered_docs = None
            if hasattr(st.session_state, 'selected_categories') and st.session_state.selected_categories:
                st.session_state.status = "Applying category filters..."
                # Create a custom retriever that filters by category
                def filter_by_category(docs):
                    return [doc for doc in docs if doc.metadata.get("category") in st.session_state.selected_categories]
                
                base_retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 20}  # Fetch more docs to filter from
                )
                
                # Get docs, filter them, then use them directly
                docs = base_retriever.get_relevant_documents(enhanced_query)
                filtered_docs = filter_by_category(docs)
                
                # Now use these directly in the chain
                result = st.session_state.qa_chain
                # Apply category filters if selected
            filtered_docs = None
            if hasattr(st.session_state, 'selected_categories') and st.session_state.selected_categories:
                st.session_state.status = "Applying category filters..."
                # Create a custom retriever that filters by category
                def filter_by_category(docs):
                    return [doc for doc in docs if doc.metadata.get("category") in st.session_state.selected_categories]
                
                base_retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 20}  # Fetch more docs to filter from
                )
                
                # Get docs, filter them, then use them directly
                docs = base_retriever.get_relevant_documents(enhanced_query)
                filtered_docs = filter_by_category(docs)
                
                # Now use these directly in the chain
                result = st.session_state.qa_chain({"query": enhanced_query, "documents": filtered_docs})
            else:
                # Use the standard retriever
                result = st.session_state.qa_chain({"query": enhanced_query})
            
            # Get relevant PDFs
            relevant_pdfs = find_relevant_pdfs(query, st.session_state.pdf_links)
            
            # Display the answer
            st.subheader("Answer")
            st.write(result["result"])
            
            # Display sources
            if 'source_documents' in result:
                st.subheader("Sources")
                sources = set()
                for doc in result['source_documents']:
                    source = doc.metadata.get('source', 'Unknown Source')
                    category = doc.metadata.get('category', 'Uncategorized')
                    sources.add(f"- [{source}]({source}) (Category: {category})")
                
                for source in sources:
                    st.markdown(source)
            
            # Display relevant PDFs
            if relevant_pdfs:
                st.subheader("Relevant PDF Documents")
                for pdf in relevant_pdfs:
                    pdf_url = pdf.get('url', '#')
                    pdf_text = pdf.get('text', 'PDF Document')
                    context = pdf.get('context', '')
                    metadata = pdf.get('metadata', {})
                    
                    # Format metadata for display
                    metadata_str = ""
                    if 'date' in metadata and metadata['date']:
                        metadata_str += f"Date: {metadata['date']} "
                    if 'id' in metadata and metadata['id']:
                        metadata_str += f"ID: {metadata['id']} "
                    if 'category' in metadata and metadata['category']:
                        metadata_str += f"Category: {metadata['category']} "
                    
                    # Create expandable section for each PDF
                    with st.expander(f"{pdf_text} {metadata_str}"):
                        st.markdown(f"**URL:** [{pdf_url}]({pdf_url})")
                        st.markdown(f"**Context:** {context}")
                        
                        # Display additional metadata if available
                        if metadata:
                            st.markdown("**Additional Information:**")
                            for key, value in metadata.items():
                                if value and key not in ['category', 'date', 'id']:
                                    st.markdown(f"- {key.capitalize()}: {value}")
                        
                        # Add download button for the PDF
                        st.markdown(f"[Download PDF]({pdf_url})")
else:
    st.warning("Please enter your Groq API key and initialize the system first.")

# Add footer with additional information
st.markdown("---")
st.markdown("""
**About this application**: 
This application crawls regulatory websites to provide accurate answers about IRDAI and UIDAI regulations.

**How to use**:
1. Enter your Groq API key in the sidebar
2. Initialize the system (this will crawl websites and create embeddings)
3. Ask questions about regulations, circulars, rules, etc.
4. Optionally filter by categories for more focused results
""")

# Add debug mode
with st.sidebar:
    if st.checkbox("Enable Debug Mode"):
        st.subheader("Debug Information")
        if 'vector_store' in st.session_state:
            st.write(f"Vector Store: {type(st.session_state.vector_store).__name__}")
        if 'pdf_links' in st.session_state:
            st.write(f"PDF Links: {len(st.session_state.pdf_links)}")
        if 'qa_chain' in st.session_state:
            st.write(f"QA Chain: {type(st.session_state.qa_chain).__name__}")
        if query and 'result' in locals():
            st.subheader("Last Query Debug")
            st.write(f"Enhanced Query: {enhanced_query}")
            st.write(f"Number of source documents: {len(result.get('source_documents', []))}")
            
            # Show token usage if available
            if hasattr(result, 'get') and result.get('token_usage'):
                st.write("Token Usage:")
                st.json(result['token_usage'])

# Add caching control
with st.sidebar:
    st.subheader("Cache Management")
    if st.button("Clear Web Cache"):
        try:
            import shutil
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            st.success("Web cache cleared successfully!")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
