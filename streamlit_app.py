import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import time
import re
from datetime import datetime

from urllib.parse import urljoin, urlparse
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.text_splitter import SemanticChunker

HARDCODED_WEBSITES = [
                      "https://irdai.gov.in/acts",
                      "https://irdai.gov.in/rules",
                      "https://irdai.gov.in/notices1",
                      "https://irdai.gov.in/consolidated-gazette-notified-regulations",
                      "https://irdai.gov.in/notifications",
                      "https://irdai.gov.in/consolidated-gazette-notified-regulations",
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
                      "https://uidai.gov.in/en/about-uidai/legal-framework",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/rules",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/notifications",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/regulations",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/circulars",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/judgements",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation",
                      "https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules",
                      "https://enforcementdirectorate.gov.in/pmla",
                      "https://enforcementdirectorate.gov.in/pmla?page=1",
                      "https://enforcementdirectorate.gov.in/fema",
                      "https://enforcementdirectorate.gov.in/fema?page=1",
                      "https://enforcementdirectorate.gov.in/fema?page=2",
                      "https://enforcementdirectorate.gov.in/fema?page=3",
                      "https://enforcementdirectorate.gov.in/bns",
                      "https://enforcementdirectorate.gov.in/bnss",
                      "https://enforcementdirectorate.gov.in/bsa"
]

SYSTEM_PROMPT_TEMPLATE = """
You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.

Answer the question based ONLY on the provided context information.

IMPORTANT INSTRUCTIONS:
- Each chunk in the context starts with "Source URL:" followed by the source website and relevant document links
- Always reference the source URL provided at the beginning of each chunk in your answers
- Answer questions using the provided context from the websites
- Pay special attention to dates, recent updates, and chronological information
- Always Give response in chronological order according to date.if from multiple different sources links then try to use latest  documents linksfrom different source link.
- When asked about "what's new" or recent developments, focus on the most recent information available
- Look for press releases, circulars, guidelines, and policy updates
- Provide specific details about new regulations, policy changes, or announcements
- If you find dated information, mention the specific dates
- When a question like, "Latest guidelines under IRDAI" is asked, follow the 'Last Updated' date and as per the same, respond to the query
- When mentioning any acts, circulars, or regulations, try to reference the available document links that are provided in the context
- If you find any PII data in the question (e.g., PAN card no., AADHAAR no., DOB, Address), respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on, as your query contains PII data"
- Use the document links provided in the context to give more comprehensive answers with proper references
- Always include the source URL in your answer for credibility and reference.
- If any general question is asked like latest updates on legal notices ,then instead of only giving source link also latest document links chronlogically from legals.The document links can be of any category like acts , notices,circulars many more.
-**CRITICAL**: Never miss to give Document link and source link in your answer.
FALLBACK RESPONSE (use ONLY when context is completely irrelevant):
"Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."

Context: {context}

Question: {input}

Provide a comprehensive answer using the available context, including relevant document links and source URLs when available. Be helpful and informative even if the context only partially addresses the question.
"""

RELEVANCE_SCORE_THRESHOLD = 0.3

def filter_urls_by_query(query, urls):
    """Filter URLs based on query relevance to reduce noise"""
    query_lower = query.lower()
    
    # Query-specific URL mapping with priority order
    url_keywords = {
       "acts": [
        "irdai.gov.in/acts",
        "uidai.gov.in/en/about-uidai/legal-framework"
    ],
    "rules": [
        "https://irdai.gov.in/rules",
      

    ],
    "regulations": [
       
        "uidai.gov.in/en/about-uidai/legal-framework/regulations",
        "uidai.gov.in/en/about-uidai/legal-framework/updated-regulation"
    ],
    "notifications": [
        "irdai.gov.in/notifications",
        "uidai.gov.in/en/about-uidai/legal-framework/notifications"
    ],
    "circulars": [
        "irdai.gov.in/circulars",
        "uidai.gov.in/en/about-uidai/legal-framework/circulars"
    ],
    "guidelines": [
        "irdai.gov.in/guidelines"
    ],
    "orders": [
        "irdai.gov.in/orders1",
        "irdai.gov.in/cic-orders"
    ],
    "exposure_drafts": [
        "irdai.gov.in/exposure-drafts"
    ],
    "penalties": [
        "irdai.gov.in/warnings-and-penalties"
    ],
    "anti_money_laundering": [
        "irdai.gov.in/antimoney-laundering"
    ],
    "other_communications": [
        "irdai.gov.in/other-communication"
    ],
    "directory": [
        "irdai.gov.in/directory-of-employees"
    ],
    "programmes_rti": [
        "irdai.gov.in/programmes-to-advance-understanding-of-rti"
    ],
    "judgements": [
        "uidai.gov.in/en/about-uidai/legal-framework/judgements"
    ],
    "enforcement_pmla": [
        "enforcementdirectorate.gov.in/pmla",
		"https://enforcementdirectorate.gov.in/pmla?page=1"
    ],
    "enforcement_fema": [
        "enforcementdirectorate.gov.in/fema",
	    "https://enforcementdirectorate.gov.in/fema?page=1",
        "https://enforcementdirectorate.gov.in/fema?page=2",
        "https://enforcementdirectorate.gov.in/fema?page=3"
    ],
    "enforcement_bns": [
        "enforcementdirectorate.gov.in/bns",
        "enforcementdirectorate.gov.in/bnss"
    ],
    "enforcement_bsa": [
        "enforcementdirectorate.gov.in/bsa"
    ],
    "gazettes": [
        "https://egazette.gov.in/(S(ylhvpcedcc5ooe3drkj1way2))/default.aspx"
    ],
    
    "uidai": ["uidai.gov.in"],
    "aadhaar": ["uidai.gov.in"],
    "enforcement": ["enforcementdirectorate.gov.in"],
    "legal":["irdai.gov.in/acts","uidai.gov.in/en/about-uidai/legal-framework","https://irdai.gov.in/notices1","https://irdai.gov.in/consolidated-gazette-notified-regulations","https://irdai.gov.in/notifications","https://irdai.gov.in/circulars","https://irdai.gov.in/orders1","https://irdai.gov.in/exposure-drafts","https://irdai.gov.in/programmes-to-advance-understanding-of-rti","https://irdai.gov.in/antimoney-laundering","https://irdai.gov.in/other-communication",'irdai.gov.in/guidelines'],

"notices":["irdai.gov.in/notices1"]
    }
    relevant_urls = []
    
    # Find matching patterns with priority for exact matches
    relevant_patterns = []
    priority_patterns = []
    
    # Check for exact keyword matches first (highest priority)
    for keyword, patterns in url_keywords.items():
        if keyword in query_lower:
            if keyword in ['rule', 'rules', 'latest rules']:
                priority_patterns.extend(patterns)
            else:
                relevant_patterns.extend(patterns)
    
    # Prioritize rules patterns if found
    if priority_patterns:
        relevant_patterns = priority_patterns + relevant_patterns
    
    # If no specific patterns found, return all URLs
    if not relevant_patterns:
        return urls
    
    # Filter URLs based on patterns with priority order
    filtered_urls = []
    seen_urls = set()
    
    # First add URLs matching priority patterns
    for pattern in priority_patterns:
        for url in urls:
            if pattern in url and url not in seen_urls:
                filtered_urls.append(url)
                seen_urls.add(url)
    
    # Then add URLs matching other relevant patterns
    for pattern in relevant_patterns:
        if pattern not in priority_patterns:  # Avoid duplicates
            for url in urls:
                if pattern in url and url not in seen_urls:
                    filtered_urls.append(url)
                    seen_urls.add(url)
    
    # If no URLs match patterns, return original URLs to avoid empty results
    return filtered_urls if filtered_urls else urls

def enhanced_relevance_score(query, document, embeddings):
    """Enhanced relevance scoring with multiple factors"""
    try:
        # Base semantic similarity
        query_embedding = embeddings.embed_query(query)
        document_embedding = embeddings.embed_documents([document.page_content])[0]
        similarity = cosine_similarity([query_embedding], [document_embedding])[0][0]
        
        # Keyword matching with weights
        query_keywords = query.lower().split()
        content_lower = document.page_content.lower()
        
        # High-value keywords get more weight
        high_value_keywords = ['act', 'circular', 'guideline', 'regulation', 'amendment', 'notification', 'rule', 'policy']
        keyword_score = 0
        
        for keyword in query_keywords:
            if keyword in content_lower:
                if keyword in high_value_keywords:
                    keyword_score += 0.2
                else:
                    keyword_score += 0.1
        
        # Domain importance bonus
        source_url = document.metadata.get('source', '')
        domain_bonus = 0
        
        # IRDAI gets highest priority for insurance-related queries
        if 'irdai.gov.in' in source_url:
            if any(term in query.lower() for term in ['insurance', 'act', 'circular', 'guideline']):
                domain_bonus += 0.15
        
        # UIDAI for aadhaar-related queries
        if 'uidai.gov.in' in source_url:
            if any(term in query.lower() for term in ['aadhaar', 'uid', 'identity']):
                domain_bonus += 0.15
        
        # Enforcement directorate for FEMA/PMLA queries
        if 'enforcementdirectorate.gov.in' in source_url:
            if any(term in query.lower() for term in ['fema', 'pmla', 'money laundering', 'enforcement']):
                domain_bonus += 0.15
        
        # # Source credibility bonus (acts > circulars > guidelines > others)
        # if '/acts' in source_url:
        #     domain_bonus += 0.1
        # elif '/circulars' in source_url:
        #     domain_bonus += 0.08
        # elif '/guidelines' in source_url:
        #     domain_bonus += 0.06
        
        # Recency bonus for recent years (if mentioned in content)
        current_year = 2024
        for year in range(current_year-2, current_year+1):
            if str(year) in content_lower:
                domain_bonus += 0.05
                break
        
        final_score = similarity + keyword_score + domain_bonus
        return min(final_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        # Fallback to keyword matching
        query_keywords = query.lower().split()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in document.page_content.lower())
        return min(keyword_matches * 0.15, 1.0)

def relevance_score(query, document, embeddings):
    return enhanced_relevance_score(query, document, embeddings)

def re_rank_documents(query, documents, embeddings):
    if not documents:
        return []
    
    if embeddings is None:
        st.warning("Embeddings not available, using original document order")
        return documents
        
    try:
        scored_docs = [(doc, relevance_score(query, doc, embeddings)) for doc in documents]
        
        scored_docs = [(doc, score) for doc, score in scored_docs if score >= RELEVANCE_SCORE_THRESHOLD]
        
        if not scored_docs:
            st.warning("No documents passed relevance threshold, using original documents")
            return documents[:6]
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        top_doc_source = scored_docs[0][0].metadata.get("source", "")
        
        source_groups = {}
        for doc, score in scored_docs:
            source = doc.metadata.get("source", "")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((doc, score))
        
        final_ranked_docs = []
        if top_doc_source in source_groups:
            top_source_docs = sorted(
                source_groups[top_doc_source], 
                key=lambda x: (x[0].metadata.get("page_number", 0), -x[1])
            )
            final_ranked_docs.extend([doc for doc, score in top_source_docs])
            del source_groups[top_doc_source]
        
        other_sources = []
        for source, docs in source_groups.items():
            avg_source_score = sum(score for _, score in docs) / len(docs)
            other_sources.append((source, avg_source_score, docs))
        
        other_sources.sort(key=lambda x: x[1], reverse=True)
        
        for source, avg_score, docs in other_sources:
            sorted_docs = sorted(
                docs, 
                key=lambda x: (x[0].metadata.get("page_number", 0), -x[1])
            )
            final_ranked_docs.extend([doc for doc, score in sorted_docs])
        
        # Enhance chunks with source URL and document links
        enhanced_docs = enhance_chunks_with_links(final_ranked_docs)
        return enhanced_docs
        
    except Exception as e:
        st.error(f"Error in re-ranking documents: {e}")
        st.warning("Falling back to original document order")
        return documents

def enhanced_web_scrape(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text
        
    except Exception as e:
        st.error(f"Enhanced scraping failed for {url}: {e}")
        return None


def extract_document_links(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
    # Pattern 1: Look for tables with "Documents" or "Download" columns
    tables = soup.find_all('table')
    for table in tables:
        # Get table headers to identify column structure
        headers = []
        header_row = table.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True).lower() for cell in header_cells]
        
        # Find Documents or Download column index
        doc_col_index = -1
        title_col_index = -1
        date_col_index = -1
        
        for i, header in enumerate(headers):
            if 'document' in header or 'download' in header:
                doc_col_index = i
            elif 'title' in header or 'name' in header or header == 'sr.no.' or header.startswith('sr'):
                title_col_index = i
            elif 'date' in header or 'updated' in header:
                date_col_index = i
        
        # Process table rows
        rows = table.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            
            # Extract document link from Documents/Download column
            doc_link = None
            doc_title = ""
            doc_date = "No date"
            
            if doc_col_index != -1 and doc_col_index < len(cells):
                # Look for links in Documents/Download column
                doc_cell = cells[doc_col_index]
                link_elem = doc_cell.find('a', href=True)
                
                if link_elem:
                    doc_link = link_elem['href']
                    # If title is in the link text, use it
                    if link_elem.get_text(strip=True):
                        doc_title = link_elem.get_text(strip=True)
                
                # Also check for PDF icons or download buttons
                if not doc_link:
                    # Look for images that might be PDF icons
                    img_links = doc_cell.find_all('img')
                    for img in img_links:
                        parent_link = img.find_parent('a', href=True)
                        if parent_link:
                            doc_link = parent_link['href']
                            break
            
            # If no Documents column, check first few cells for links
            if not doc_link:
                for i, cell in enumerate(cells[:3]):  # Check first 3 columns
                    link_elem = cell.find('a', href=True)
                    if link_elem and link_elem['href']:
                        href = link_elem['href']
                        # Check if it's a document link
                        if (any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']) or
                            'download' in href.lower() or 'document' in href.lower()):
                            doc_link = href
                            if not doc_title:
                                doc_title = link_elem.get_text(strip=True)
                            break
            
            # Extract title from title column or first column
            if not doc_title and title_col_index != -1 and title_col_index < len(cells):
                title_cell = cells[title_col_index]
                # Remove serial numbers
                title_text = title_cell.get_text(strip=True)
                title_text = re.sub(r'^\d+\.?\s*', '', title_text)
                doc_title = title_text
            elif not doc_title and len(cells) > 0:
                # Use first cell as title, removing serial numbers
                title_text = cells[0].get_text(strip=True)
                title_text = re.sub(r'^\d+\.?\s*', '', title_text)
                doc_title = title_text
            
            # Extract date
            if date_col_index != -1 and date_col_index < len(cells):
                doc_date = cells[date_col_index].get_text(strip=True)
            elif len(cells) > 1 and not doc_date:
                # Try to find date in second column
                potential_date = cells[1].get_text(strip=True)
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}', potential_date):
                    doc_date = potential_date
            
            # Clean up and add document link
            if doc_link:
                # Handle relative URLs
                if doc_link.startswith('/'):
                    doc_link = urljoin(url, doc_link)
                elif not doc_link.startswith(('http://', 'https://')):
                    doc_link = urljoin(url, doc_link)
                
                # Clean up date text
                doc_date = re.sub(r'\s+', ' ', doc_date).strip()
                
                # Clean up title
                if not doc_title or len(doc_title) < 3:
                    doc_title = f"Document {len(document_links) + 1}"
                
                document_links.append({
                    'title': doc_title,
                    'link': doc_link,
                    'date': doc_date,
                    'type': 'document'
                })
    
    # Pattern 2: Look for numbered list items with download links (UIDAI pattern)
    if not document_links:
        # Find numbered items in the page
        numbered_items = soup.find_all(['p', 'div', 'li'], string=re.compile(r'^\d+\.'))
        
        for item in numbered_items:
            # Look for download links near this item
            container = item.find_parent(['div', 'section', 'article']) or item.parent
            if container:
                # Find download link in the same container
                download_links = container.find_all('a', href=True)
                for link in download_links:
                    href = link['href']
                    if (any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']) or
                        'download' in href.lower()):
                        
                        # Handle relative URLs
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        elif not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        
                        title_text = item.get_text(strip=True)
                        title_text = re.sub(r'^\d+\.?\s*', '', title_text)
                        
                        document_links.append({
                            'title': title_text or f"Document {len(document_links) + 1}",
                            'link': href,
                            'date': 'No date',
                            'type': 'document'
                        })
                        break
    
    # Pattern 3: Look for "Documents" or "Download" sections if no links found in tables
    if not document_links:
        download_sections = soup.find_all(['div', 'section', 'h2', 'h3'], 
            string=re.compile(r'(?i)(documents?|downloads?|files?|resources?)'))
        
        for section in download_sections:
            # Get the parent container of the section
            container = section.find_parent(['div', 'section', 'article']) or section.parent
            if container:
                # Extract all links in this section
                links = container.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    link_text = link.get_text(strip=True)
                    
                    if not href or len(link_text) < 3:
                        continue
                        
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    elif not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                        
                    # Check if it's a document link
                    if any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']):
                        document_links.append({
                            'title': link_text or "Document",
                            'link': href,
                            'date': 'No date',
                            'type': 'document'
                        })
    
    # Pattern 4: Direct document links in the page (fallback)
    if not document_links:
        for link in soup.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text(strip=True)
            
            # Skip empty or very short links
            if not href or len(link_text) < 3:
                continue
                
            # Handle relative URLs
            if href.startswith('/'):
                href = urljoin(url, href)
            elif not href.startswith(('http://', 'https://')):
                href = urljoin(url, href)
            
            # Check for document file extensions
            is_document = any(href.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
            
            # Check for document detail pages
            is_document_page = any(term in href.lower() for term in ['document-detail', 'documentId=', 'view-document', 'download'])
            
            # Check for download in link text
            has_download_text = any(term in link_text.lower() for term in ['download', 'view', 'pdf', 'doc', 'xls'])
            
            if (is_document or is_document_page or has_download_text) and not any(dl['link'] == href for dl in document_links):
                document_links.append({
                    'title': link_text or f"Document {len(document_links) + 1}",
                    'link': href,
                    'date': 'No date',
                    'type': 'document' if is_document else 'reference'
                })
    
    # Remove duplicates
    seen_links = set()
    unique_links = []
    for link in document_links:
        if link['link'] not in seen_links:
            seen_links.add(link['link'])
            unique_links.append(link)
    
    return unique_links
def format_document_links_for_embedding(document_links):
    if not document_links:
        return ""
    
    formatted_links = "\n\n=== RELEVANT DOCUMENT LINKS ===\n"
    
    content_docs = [link for link in document_links if link['type'] == 'content']
    ref_docs = [link for link in document_links if link['type'] == 'reference']
    
    if content_docs:
        formatted_links += "\n[CONTENT PAGES]\n"
        for i, link_info in enumerate(content_docs, 1):
            formatted_links += f"{i}. {link_info['title']} - {link_info['link']}\n"
    
    if ref_docs:
        formatted_links += "\n[REFERENCE DOCUMENTS]\n"
        for i, link_info in enumerate(ref_docs, 1):
            formatted_links += f"{i}. {link_info['title']} - {link_info['link']}\n"
    
    formatted_links += "=== END DOCUMENT LINKS ===\n\n"
    
    return formatted_links

def extract_structured_content(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content_sections = {}
    
    # Extract news sections
    news_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ['news', 'update', 'recent', 'latest', 'whats-new']
    ))
    
    if news_sections:
        content_sections['news'] = []
        for section in news_sections:
            text = section.get_text(strip=True)
            if len(text) > 50:
                content_sections['news'].append(text)
    
    content_sections['document_links'] = extract_document_links(html_content, url)
    
    main_text = soup.get_text()
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    formatted_links = format_document_links_for_embedding(content_sections['document_links'])
    enhanced_text = clean_text + formatted_links
    
    return enhanced_text, content_sections

def load_hardcoded_websites(query=None):
    loaded_docs = []
    
    # Filter URLs based on query if provided
    urls_to_load = HARDCODED_WEBSITES
    if query:
        urls_to_load = filter_urls_by_query(query, HARDCODED_WEBSITES)
    
    for url in urls_to_load:
        try:
            st.write(f"Loading URL: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                enhanced_text, sections = extract_structured_content(html_content, url)
                
                doc = Document(
                    page_content=enhanced_text,
                    metadata={
                        "source": url, 
                        "sections": sections,
                        "document_links": sections.get('document_links', []),
                        "has_embedded_links": True
                    }
                )
                loaded_docs.append(doc)
                
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                if sections.get('document_links'):
                    with st.expander(f"Document Links found from {url}"):
                        st.write(f"**Total document links found:** {len(sections['document_links'])}")
                        
                        content_docs = [link for link in sections['document_links'] if link['type'] == 'content']
                        ref_docs = [link for link in sections['document_links'] if link['type'] == 'reference']
                        
                        if content_docs:
                            st.write("**Content Pages:**")
                            for i, link_info in enumerate(content_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if ref_docs:
                            st.write("**Reference Documents:**")
                            for i, link_info in enumerate(ref_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        st.success(f"✅ Document links have been embedded into the text content for better retrieval")
                else:
                    st.write(f"No relevant document links found from {url}")
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    return loaded_docs

def is_fallback_response(response_text):
    fallback_phrases = [
        "fall outside the scope of the data I've been trained on",
        "details you've asked for fall outside the scope",
        "outside the scope of the data",
        "However, I've gathered information that closely aligns"
    ]
    
    return any(phrase in response_text for phrase in fallback_phrases)

def display_chunks(chunks, title="Top 3 Retrieved Chunks"):
    st.subheader(title)
    
    for i, chunk in enumerate(chunks[:3], 1):
        with st.expander(f"Chunk {i} - Source: {chunk.metadata.get('source', 'Unknown')}"):
            st.markdown("**Content:**")
            content = chunk.page_content.strip()
            
            has_embedded_links = "=== RELEVANT DOCUMENT LINKS ===" in content
            
            if has_embedded_links:
                text_part, links_part = content.split("=== RELEVANT DOCUMENT LINKS ===", 1)
                
                st.markdown("**Main Content:**")
                if len(text_part) > 1000:
                    st.text_area(f"Chunk {i} Main Content", text_part[:1000] + "...", height=200, disabled=True)
                    st.info(f"Content truncated for display. Full length: {len(text_part)} characters")
                else:
                    st.text_area(f"Chunk {i} Main Content", text_part, height=min(200, max(100, len(text_part)//5)), disabled=True)
                
                st.markdown("**Embedded Document Links:**")
                links_clean = links_part.replace("=== END DOCUMENT LINKS ===", "").strip()
                st.code(links_clean, language="text")
                
            else:
                if len(content) > 1000:
                    st.text_area(f"Chunk {i} Content", content[:1000] + "...", height=200, disabled=True)
                    st.info(f"Content truncated for display. Full length: {len(content)} characters")
                else:
                    st.text_area(f"Chunk {i} Content", content, height=min(200, max(100, len(content)//5)), disabled=True)
            
            st.markdown("**Metadata:**")
            metadata = chunk.metadata
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Source:** {metadata.get('source', 'N/A')}")
                st.write(f"**Has Embedded Links:** {metadata.get('has_embedded_links', False)}")
            
            with col2:
                st.write(f"**Chunk Length:** {len(content)} characters")
                st.write(f"**Document Links Count:** {len(metadata.get('document_links', []))}")

def enhance_documents_before_chunking(documents):
    """Add source URL and document links to documents before chunking"""
    enhanced_documents = []
    
    for doc in documents:
        source_url = doc.metadata.get('source', 'Unknown')
        document_links = doc.metadata.get('document_links', [])
        
        # Create source line with document links
        source_line = f"Source URL: {source_url}"
        
        if document_links:
            links_text = " | Document Links: "
            link_titles = []
            for link in document_links[:5]:  # Limit to first 5 links
                link_titles.append(f"{link['title']} ({link['link']})")
            links_text += "; ".join(link_titles)
            if len(document_links) > 5:
                links_text += f" and {len(document_links) - 5} more..."
            source_line += links_text
        
        # Add source line to the beginning of document content
        enhanced_content = source_line + "\n\n" + doc.page_content
        
        # Create new document with enhanced content
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata=doc.metadata
        )
        enhanced_documents.append(enhanced_doc)
    
    return enhanced_documents

def enhance_chunks_with_links(chunks):
    """Add source URL and document links to the first line of each chunk"""
    enhanced_chunks = []
    
    for chunk in chunks:
        source_url = chunk.metadata.get('source', 'Unknown')
        document_links = chunk.metadata.get('document_links', [])
        
        # Create source line
        source_line = f"Source URL: {source_url}"
        
        # Add document links if available
        if document_links:
            links_text = " | Document Links: "
            link_titles = []
            for link in document_links[:3]:  # Limit to first 3 links to avoid too long first line
                link_titles.append(f"{link['title']} ({link['link']})")
            links_text += "; ".join(link_titles)
            if len(document_links) > 3:
                links_text += f" and {len(document_links) - 3} more..."
            source_line += links_text
        
        # Add source line to the beginning of chunk content
        enhanced_content = source_line + "\n\n" + chunk.page_content
        
        # Create new chunk with enhanced content
        enhanced_chunk = Document(
            page_content=enhanced_content,
            metadata=chunk.metadata
        )
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False
if 'hf_embedding' not in st.session_state:
    st.session_state['hf_embedding'] = None
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None

st.title("Web GEN-ie")

st.subheader("Azure OpenAI Configuration")

col1, col2 = st.columns(2)

with col1:
    azure_endpoint = st.text_input(
        "Azure OpenAI Endpoint:", 
        placeholder="https://your-resource.openai.azure.com/",
        help="Your Azure OpenAI resource endpoint URL"
    )
    
    api_key = st.text_input(
        "Azure OpenAI API Key:", 
        type="password",
        placeholder="Enter your Azure OpenAI API key here...",
        help="Your Azure OpenAI API key"
    )

with col2:
    deployment_name = st.text_input(
        "Deployment Name:", 
        placeholder="gpt-4o",
        help="The name of your deployed model (e.g., gpt-35-turbo, gpt-4)"
    )
    
    api_version = st.selectbox(
        "API Version:",
        ["2025-01-01-preview"],
        index=0,
        help="Azure OpenAI API version"
    )

config_complete = all([azure_endpoint, api_key, deployment_name, api_version])

if not config_complete:
    st.warning("Please fill in all Azure OpenAI configuration fields to proceed.")

if not st.session_state['docs_loaded']:
    if st.button("Load Websites", disabled=not config_complete):
        if not config_complete:
            st.error("Please complete the Azure OpenAI configuration first.")
        else:
            st.session_state['loaded_docs'] = load_hardcoded_websites()
            st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
            
            if config_complete and st.session_state['loaded_docs']:
                with st.spinner("Processing documents with embedded links..."):
                    try:
                        llm = AzureChatOpenAI(
                            azure_endpoint=azure_endpoint,
                            api_key=api_key,
                            azure_deployment=deployment_name,
                            api_version=api_version,
                            temperature=0.0,
                            top_p=0.1
                        )
                        st.session_state['llm'] = llm
                        
                        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-miniLM-L6-v2")
                        st.session_state['hf_embedding'] = hf_embedding
                        
                        try:
                            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                            st.session_state['prompt'] = prompt
                            st.success("Prompt template created successfully")
                        except Exception as prompt_error:
                            st.error(f"Error creating prompt template: {prompt_error}")
                            fallback_template = """Answer the question based on the provided context.
                            
Context: {context}
Question: {input}

Answer:"""
                            prompt = ChatPromptTemplate.from_template(fallback_template)
                            st.session_state['prompt'] = prompt
                            st.warning("Using fallback prompt template")
                        
                        text_splitter = SemanticChunker(
                       embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                       breakpoint_threshold_type="percentile",
                      breakpoint_threshold_amount=95,
                      min_chunk_size=400
                       
                        )
                        
                        enhanced_docs = enhance_documents_before_chunking(st.session_state['loaded_docs'])
                        document_chunks = text_splitter.split_documents(enhanced_docs)
                        st.write(f"Number of chunks created: {len(document_chunks)}")
                        
                        # Count chunks with embedded links
                        chunks_with_links = sum(1 for chunk in document_chunks 
                                              if "Source URL:" in chunk.page_content)
                        st.info(f"{chunks_with_links} chunks contain source URLs and document links")
                        
                        st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                        
                        retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 20})
                        document_chain = create_stuff_documents_chain(llm, st.session_state['prompt'])
                        st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
                        
                        st.session_state['docs_loaded'] = True
                        st.success("Documents processed with embedded links and ready for querying!")
                    
                    except Exception as e:
                        st.error(f"Error initializing Azure OpenAI: {e}")
                        st.error("Please check your Azure OpenAI configuration and try again.")

st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

show_chunks = st.checkbox("Show retrieved chunks used for answer generation", value=True)

if st.button("Get Answer", disabled=not config_complete) and query:
    if not config_complete:
        st.error("Please complete the Azure OpenAI configuration first.")
    elif st.session_state.get('vector_db') and st.session_state.get('llm'):
        with st.spinner("Searching and generating answer..."):
            try:
                if st.session_state.get('hf_embedding') is None:
                    st.info("Initializing embeddings...")
                    st.session_state['hf_embedding'] = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
                
                if st.session_state.get('prompt') is None:
                    st.info("Creating prompt template...")
                    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                    st.session_state['prompt'] = prompt
                
                # Load query-specific documents for better accuracy
                st.info(f"Loading documents relevant to query: {query}")
                query_specific_docs = load_hardcoded_websites(query)
                
                # Create temporary vector store with query-specific documents
                if query_specific_docs:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    
                    # Split documents into chunks
                    query_chunks = text_splitter.split_documents(query_specific_docs)
                    
                    # Create temporary vector store
                    temp_vector_db = FAISS.from_documents(
                        query_chunks, 
                        st.session_state['hf_embedding']
                    )
                    
                    # Retrieve from query-specific vector store
                    raw_retriever = temp_vector_db.as_retriever(search_kwargs={"k": 20})
                    raw_docs = raw_retriever.get_relevant_documents(query)
                else:
                    # Fallback to original vector store
                    raw_retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 20})
                    raw_docs = raw_retriever.get_relevant_documents(query)
                
                # Filter URLs based on query
                filtered_urls = filter_urls_by_query(query, [doc.metadata.get('source', '') for doc in raw_docs])
                filtered_docs = [doc for doc in raw_docs if doc.metadata.get('source', '') in filtered_urls]
                
                # Re-rank documents using enhanced scoring
                reranked_docs = re_rank_documents(query, filtered_docs, st.session_state['hf_embedding'])
                final_docs = reranked_docs[:6]
                
                # Create response using document chain
                document_chain = create_stuff_documents_chain(st.session_state['llm'], st.session_state['prompt'])
                response = document_chain.invoke({"input": query, "context": final_docs})
                
                # Format response to match expected structure
                response = {"answer": response, "context": final_docs}
                
                st.subheader("Response:")
                st.write(response['answer'])
                
                if show_chunks and 'context' in response:
                    retrieved_docs = response['context']
                    if retrieved_docs:
                        display_chunks(retrieved_docs, "Top Chunks Used for Answer Generation")
                        
                        links_used = 0
                        for doc in retrieved_docs:
                            if "=== RELEVANT DOCUMENT LINKS ===" in doc.page_content:
                                links_used += 1
                        
                        if links_used > 0:
                            st.success(f"{links_used} out of {len(retrieved_docs)} chunks contained embedded document links that were sent to the LLM")
                        else:
                            st.info("ℹ️ No chunks with embedded document links were retrieved for this query")
                    else:
                        st.info("No chunks were retrieved for this query.")
                
                if not is_fallback_response(response['answer']):
                    st.write("\n**📍 Information Sources:**")
                    sources = set()
                    retrieved_docs = response.get('context', [])
                    for doc in retrieved_docs:
                        source = doc.metadata.get('source', 'Unknown')
                        sources.add(source)
                    
                    for i, source in enumerate(sources, 1):
                        st.write(f"{i}. [{source}]({source})")
                else:
                    st.info("ℹ️ No specific documents or sources are available for this query as it falls outside the current data scope.")
            
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.error("Please check your Azure OpenAI configuration and try again.")
                st.error("Debug Info:")
                st.write(f"- LLM available: {st.session_state.get('llm') is not None}")
                st.write(f"- Vector DB available: {st.session_state.get('vector_db') is not None}")
                st.write(f"- Prompt available: {st.session_state.get('prompt') is not None}")
                st.write(f"- Embeddings available: {st.session_state.get('hf_embedding') is not None}")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
