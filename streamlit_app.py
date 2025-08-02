import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
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
from urllib.parse import urljoin, urlparse
from collections import Counter
from datetime import datetime

HARDCODED_WEBSITES = ["https://irdai.gov.in/guidelines"]

def extract_dates_from_text(text):
    """Extract dates from text using multiple patterns"""
    date_patterns = [
        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',  # DD Month YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b',  # Month DD, YYYY
        r'\b(202[0-9])\b'  # Years 2020-2029
    ]
    
    dates_found = []
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                groups = match.groups()
                if len(groups) == 1:  # Just year
                    year = int(groups[0])
                    if 2020 <= year <= 2030:  # Reasonable year range
                        dates_found.append((year, 1, 1))  # Default to Jan 1
                else:
                    # Try to parse the date
                    date_str = match.group()
                    parsed_date = extract_date_components(date_str)
                    if parsed_date:
                        dates_found.append(parsed_date)
            except:
                continue
    
    return dates_found

def extract_date_components(date_str):
    """Extract year, month, day from date string"""
    # Look for year first
    year_match = re.search(r'\b(202[0-9])\b', date_str)
    if year_match:
        year = int(year_match.group(1))
        
        # Look for month
        month = 1
        month_match = re.search(r'\b(\d{1,2})[/-]', date_str)
        if month_match:
            month = min(12, max(1, int(month_match.group(1))))
        
        # Look for day
        day = 1
        day_match = re.search(r'[/-](\d{1,2})\b', date_str)
        if day_match:
            day = min(31, max(1, int(day_match.group(1))))
        
        return (year, month, day)
    return None

def create_date_aware_retriever(vector_db, query, k=10):
    """Custom retriever that considers both similarity and recency based on 'Last Updated' dates"""
    
    # First get more documents than needed
    initial_results = vector_db.similarity_search_with_score(query, k=k*2)
    
    # Extract dates and score documents
    scored_docs = []
    
    for doc, similarity_score in initial_results:
        # Look for 'Last Updated' dates in metadata
        most_recent_date = None
        
        # Check document metadata for last updated dates
        if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
            document_links = doc.metadata['sections']['document_links']
            # Find the most recent 'last_updated_date' from all document links
            valid_dates = []
            for link in document_links:
                if link.get('last_updated_date'):
                    valid_dates.append(link['last_updated_date'])
            
            if valid_dates:
                most_recent_date = max(valid_dates)
        
        # If no last updated date found, try to extract from content
        if not most_recent_date:
            content_dates = extract_dates_from_text(doc.page_content)
            if content_dates:
                most_recent_date = max(content_dates)
        
        # Default to old date if nothing found
        if most_recent_date:
            recent_year = most_recent_date[0]
        else:
            recent_year = 2020
        
        # Check if query asks for "latest" or "recent"
        is_recency_query = any(keyword in query.lower() for keyword in 
                              ['latest', 'recent', 'new', 'current', 'updated'])
        
        if is_recency_query:
            # For recency queries, heavily weight recent documents
            # Give massive bonus for 2025 documents, less for 2024, etc.
            if recent_year >= 2025:
                recency_bonus = 1.0  # Huge bonus for 2025+
            elif recent_year >= 2024:
                recency_bonus = 0.6  # Good bonus for 2024
            elif recent_year >= 2023:
                recency_bonus = 0.3  # Some bonus for 2023
            else:
                recency_bonus = 0.0  # No bonus for older
            
            final_score = similarity_score - recency_bonus  # Lower score is better
        else:
            final_score = similarity_score
        
        scored_docs.append({
            'doc': doc,
            'similarity_score': similarity_score,
            'most_recent_year': recent_year,
            'most_recent_date': most_recent_date,
            'final_score': final_score,
            'recency_query': is_recency_query
        })
    
    # Sort documents
    if any(keyword in query.lower() for keyword in ['latest', 'recent', 'new', 'current']):
        # For recency queries, prioritize by year first, then by date, then similarity
        scored_docs.sort(key=lambda x: (
            -x['most_recent_year'],  # Year descending (2025, 2024, 2023...)
            -(x['most_recent_date'][1] if x['most_recent_date'] else 0),  # Month descending
            -(x['most_recent_date'][2] if x['most_recent_date'] else 0),  # Day descending
            x['similarity_score']  # Then by similarity (ascending = better)
        ))
    else:
        # For regular queries, use similarity only
        scored_docs.sort(key=lambda x: x['final_score'])
    
    # Return top k documents
    return [item['doc'] for item in scored_docs[:k]]

# Custom retrieval chain with date awareness
class DateAwareRetrievalChain:
    def __init__(self, vector_db, llm, prompt_template):
        self.vector_db = vector_db
        self.llm = llm
        self.prompt_template = prompt_template
        
    def invoke(self, query_dict):
        query = query_dict['input']
        
        # Use date-aware retrieval
        relevant_docs = create_date_aware_retriever(self.vector_db, query, k=8)
        
        # Create context from retrieved documents with date information
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            doc_dates = doc.metadata.get('extracted_dates', [])
            most_recent = max(doc_dates) if doc_dates else (2020, 1, 1)
            recent_year = most_recent[0]
            
            context_parts.append(
                f"Document {i+1} (Most recent year: {recent_year}):\n{doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Format prompt
        formatted_prompt = self.prompt_template.format(context=context, input=query)
        
        # Get LLM response
        response = self.llm.invoke(formatted_prompt)
        
        return {
            'answer': response.content,
            'context': relevant_docs
        }

def smart_document_filter(document_links, query, ai_response, max_docs=3):
    if not document_links:
        return []
    
    ai_response_lower = ai_response.lower()
    ai_response_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', ai_response_lower))
    
    mentioned_acts = re.findall(r'\b[\w\s]*act[\w\s]*\b', ai_response_lower)
    mentioned_regulations = re.findall(r'\b[\w\s]*regulation[\w\s]*\b', ai_response_lower)
    mentioned_circulars = re.findall(r'\b[\w\s]*circular[\w\s]*\b', ai_response_lower)
    mentioned_guidelines = re.findall(r'\b[\w\s]*guideline[\w\s]*\b', ai_response_lower)
    mentioned_amendments = re.findall(r'\b[\w\s]*amendment[\w\s]*\b', ai_response_lower)
    
    specific_mentions = (mentioned_acts + mentioned_regulations + mentioned_circulars + 
                        mentioned_guidelines + mentioned_amendments)
    
    mentioned_years = set(re.findall(r'\b(20\d{2})\b', ai_response))
    
    high_confidence_docs = []
    
    for doc in document_links:
        title = doc.get('title', '').strip()
        title_lower = title.lower()
        
        if (len(title) < 10 or 
            title.lower() in ['click here', 'read more', 'download', 'view more', 'see all'] or
            any(skip_word in title.lower() for skip_word in ['home', 'contact', 'about us', 'sitemap'])):
            continue
        
        confidence_score = 0
        match_reasons = []
        
        # PRIORITY: Check for 'Last Updated' date and boost score heavily for recent documents
        last_updated_date = doc.get('last_updated_date')
        if last_updated_date:
            year = last_updated_date[0]
            if year >= 2025:
                confidence_score += 60  # Huge boost for 2025
                match_reasons.append(f"Latest document: {year}")
            elif year >= 2024:
                confidence_score += 40  # Good boost for 2024
                match_reasons.append(f"Recent document: {year}")
            elif year >= 2023:
                confidence_score += 20  # Some boost for 2023
                match_reasons.append(f"Fairly recent: {year}")
        
        for mention in specific_mentions:
            mention_clean = mention.strip()
            if len(mention_clean) > 5 and mention_clean in title_lower:
                confidence_score += 50
                match_reasons.append(f"Exact mention: '{mention_clean}'")
        
        title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_lower))
        common_words = ai_response_words.intersection(title_words)
        
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'doe', 'end', 'few', 'got', 'let', 'man', 'new', 'put', 'say', 'she', 'too', 'use'}
        meaningful_common_words = common_words - stop_words
        
        if len(meaningful_common_words) >= 3:
            confidence_score += 30
            match_reasons.append(f"Multiple key terms: {list(meaningful_common_words)[:3]}")
        
        if mentioned_years:
            title_years = set(re.findall(r'\b(20\d{2})\b', title))
            doc_years = set()
            if last_updated_date:
                doc_years.add(str(last_updated_date[0]))
            
            year_intersection = mentioned_years.intersection(title_years.union(doc_years))
            if year_intersection:
                confidence_score += 25
                match_reasons.append(f"Year match: {year_intersection}")
        
        regulatory_types = ['act', 'regulation', 'circular', 'guideline', 'amendment', 'notification', 'policy', 'rule', 'framework', 'directive']
        ai_mentions_reg_type = any(reg_type in ai_response_lower for reg_type in regulatory_types)
        title_has_reg_type = any(reg_type in title_lower for reg_type in regulatory_types)
        
        if ai_mentions_reg_type and title_has_reg_type:
            matching_reg_types = [reg_type for reg_type in regulatory_types 
                                if reg_type in ai_response_lower and reg_type in title_lower]
            if matching_reg_types:
                confidence_score += 25
                match_reasons.append(f"Regulatory type match: {matching_reg_types}")
        
        domain_terms = ['insurance', 'aadhaar', 'uidai', 'irdai', 'pmla', 'licensing', 'compliance', 'bima', 'vahak']
        ai_domain_terms = [term for term in domain_terms if term in ai_response_lower]
        title_domain_terms = [term for term in domain_terms if term in title_lower]
        
        matching_domain_terms = set(ai_domain_terms).intersection(set(title_domain_terms))
        if matching_domain_terms:
            confidence_score += 20
            match_reasons.append(f"Domain terms: {list(matching_domain_terms)}")
        
        # Lower threshold to include recent docs even with moderate relevance
        if confidence_score >= 30:  
            high_confidence_docs.append({
                'doc': doc,
                'score': confidence_score,
                'reasons': match_reasons
            })
    
    # Sort by confidence score (higher is better), with last updated date as tiebreaker
    high_confidence_docs.sort(key=lambda x: (
        x['score'],
        x['doc'].get('last_updated_date', (2020, 1, 1))[0]  # Year as tiebreaker
    ), reverse=True)
    
    return [item['doc'] for item in high_confidence_docs[:max_docs]]'compliance']]
        ai_domain_terms = [term for term in domain_terms if term in ai_response_lower]
        title_domain_terms = [term for term in domain_terms if term in title_lower]
        
        matching_domain_terms = set(ai_domain_terms).intersection(set(title_domain_terms))
        if matching_domain_terms:
            confidence_score += 20
            match_reasons.append(f"Domain terms: {list(matching_domain_terms)}")
        
        if confidence_score >= 40:  # Lowered threshold to include recent docs
            high_confidence_docs.append({
                'doc': doc,
                'score': confidence_score,
                'reasons': match_reasons
            })
    
    high_confidence_docs.sort(key=lambda x: x['score'], reverse=True)
    
    return [item['doc'] for item in high_confidence_docs[:max_docs]]

def extract_key_terms(text):
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'what', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    meaningful_terms = [word.lower() for word in words if word.lower() not in stop_words]
    
    term_counts = Counter(meaningful_terms)
    return [term for term, count in term_counts.most_common(20)]

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

def parse_last_updated_date(date_string):
    """Parse 'Last Updated' date from various formats like '29-07-2025', 'DD-MM-YYYY', etc."""
    if not date_string or not date_string.strip():
        return None
    
    date_string = date_string.strip()
    
    # Pattern for DD-MM-YYYY format (most common in IRDAI)
    pattern1 = r'(\d{1,2})-(\d{1,2})-(\d{4})'
    match1 = re.search(pattern1, date_string)
    if match1:
        day, month, year = match1.groups()
        return (int(year), int(month), int(day))
    
    # Pattern for DD/MM/YYYY format
    pattern2 = r'(\d{1,2})/(\d{1,2})/(\d{4})'
    match2 = re.search(pattern2, date_string)
    if match2:
        day, month, year = match2.groups()
        return (int(year), int(month), int(day))
    
    # Pattern for YYYY-MM-DD format
    pattern3 = r'(\d{4})-(\d{1,2})-(\d{1,2})'
    match3 = re.search(pattern3, date_string)
    if match3:
        year, month, day = match3.groups()
        return (int(year), int(month), int(day))
    
    # Just extract year if no full date found
    year_pattern = r'\b(202[0-9])\b'
    year_match = re.search(year_pattern, date_string)
    if year_match:
        return (int(year_match.group(1)), 1, 1)
    
    return None

def extract_document_links_with_dates(html_content, url):
    """Enhanced document extraction focusing on 'Last Updated' column"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
    # Look for tables with document information
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        if not rows:
            continue
            
        headers = []
        
        # Get table headers
        header_row = rows[0]
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        
        # Find specific column indices
        last_updated_col_idx = -1
        title_col_idx = -1
        document_col_idx = -1
        
        for i, header in enumerate(headers):
            if 'last updated' in header or 'updated' in header:
                last_updated_col_idx = i
            elif 'title' in header or 'sub title' in header:
                title_col_idx = i
            elif 'document' in header:
                document_col_idx = i
        
        st.write(f"Debug: Found headers: {headers}")
        st.write(f"Debug: Last Updated column index: {last_updated_col_idx}")
        st.write(f"Debug: Title column index: {title_col_idx}")
        st.write(f"Debug: Document column index: {document_col_idx}")
        
        # Process data rows
        for row_idx, row in enumerate(rows[1:], 1):  # Skip header row
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:  # Need at least 3 columns for meaningful data
                continue
            
            # Extract last updated date
            last_updated_date = None
            last_updated_text = ""
            if last_updated_col_idx != -1 and last_updated_col_idx < len(cells):
                last_updated_text = cells[last_updated_col_idx].get_text(strip=True)
                last_updated_date = parse_last_updated_date(last_updated_text)
            
            # Extract document title and links from multiple columns
            row_documents = []
            
            for cell_idx, cell in enumerate(cells):
                # Look for links in this cell
                links_in_cell = cell.find_all('a', href=True)
                for link in links_in_cell:
                    href = link.get('href')
                    link_text = link.get_text(strip=True)
                    
                    if not href or len(link_text) < 5:
                        continue
                    
                    # Make absolute URL
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    elif not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                    
                    # Check if it's a relevant document
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'order', 'policy', 'master', 
                                       'framework', 'directive', 'insurance', 'bima', 'vahak']
                    
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        # Get additional context from title column if available
                        title_context = ""
                        if title_col_idx != -1 and title_col_idx < len(cells):
                            title_context = cells[title_col_idx].get_text(strip=True)
                        
                        # Combine link text with title context for better description
                        full_title = f"{title_context} - {link_text}" if title_context and title_context != link_text else link_text
                        
                        document_info = {
                            'title': full_title,
                            'link': href,
                            'last_updated_date': last_updated_date,
                            'last_updated_text': last_updated_text,
                            'type': 'table_document',
                            'row_index': row_idx,
                            'cell_index': cell_idx
                        }
                        
                        row_documents.append(document_info)
            
            # Add all documents from this row
            document_links.extend(row_documents)
            
            # Debug: Show what we extracted from this row
            if row_documents:
                st.write(f"Debug Row {row_idx}: Found {len(row_documents)} documents, Last Updated: {last_updated_text} -> {last_updated_date}")
    
    # Remove duplicates based on title and link
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    # Sort by last updated date (most recent first)
    unique_document_links.sort(key=lambda x: (
        x['last_updated_date'][0] if x['last_updated_date'] else 2020,
        x['last_updated_date'][1] if x['last_updated_date'] else 1,
        x['last_updated_date'][2] if x['last_updated_date'] else 1
    ), reverse=True)
    
    st.write(f"Debug: Total unique documents found: {len(unique_document_links)}")
    
    return unique_document_links

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
    
    # Extract document links with enhanced date parsing
    content_sections['document_links'] = extract_document_links_with_dates(html_content, url)
    
    main_text = soup.get_text()
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return clean_text, content_sections

def create_enhanced_documents(loaded_docs):
    """Create documents with enhanced date metadata"""
    enhanced_docs = []
    
    for doc in loaded_docs:
        # Extract dates from content
        content_dates = extract_dates_from_text(doc.page_content)
        
        # Create enhanced metadata
        enhanced_metadata = doc.metadata.copy()
        enhanced_metadata['extracted_dates'] = content_dates
        enhanced_metadata['most_recent_date'] = max(content_dates) if content_dates else (2020, 1, 1)
        enhanced_metadata['has_recent_content'] = any(date[0] >= 2024 for date in content_dates)
        
        # Add date context to content for better retrieval
        date_context = ""
        if content_dates:
            recent_years = sorted(set(date[0] for date in content_dates), reverse=True)
            date_context = f"Document contains information from years: {', '.join(map(str, recent_years[:5]))}. "
        
        enhanced_content = date_context + doc.page_content
        
        enhanced_doc = Document(
            page_content=enhanced_content,
            metadata=enhanced_metadata
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

def load_hardcoded_websites():
    loaded_docs = []
    
    for url in HARDCODED_WEBSITES:
        try:
            st.write(f"Loading URL: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content(html_content, url)
                
                # Add date information to metadata
                doc_dates = extract_dates_from_text(clean_text)
                
                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "source": url, 
                        "sections": sections,
                        "extracted_dates": doc_dates,
                        "most_recent_date": max(doc_dates) if doc_dates else (2020, 1, 1)
                    }
                )
                loaded_docs.append(doc)
                
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                if sections.get('document_links'):
                    with st.expander(f"Document Links with Dates from {url}"):
                        st.write(f"**Total document links found:** {len(sections['document_links'])}")
                        
                        # Separate documents with and without dates
                        doc_links_with_dates = [link for link in sections['document_links'] if link.get('extracted_date')]
                        doc_links_without_dates = [link for link in sections['document_links'] if not link.get('extracted_date')]
                        
                        if doc_links_with_dates:
                            st.write("**Recent Documents (with dates):**")
                            for i, link_info in enumerate(doc_links_with_dates[:10]):
                                date_tuple = link_info['extracted_date']
                                date_str = f"{date_tuple[2]:02d}/{date_tuple[1]:02d}/{date_tuple[0]}"
                                st.write(f"{i+1}. **{date_str}** - [{link_info['title']}]({link_info['link']})")
                        
                        if doc_links_without_dates:
                            st.write("**Other Documents:**")
                            for i, link_info in enumerate(doc_links_without_dates[:5]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
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

# Initialize session state
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

st.title("Web GEN-ie")
st.subheader("Enhanced with Date-Aware Retrieval")

st.subheader("Configuration")
api_key = st.text_input(
    "Enter your Groq API Key:", 
    type="password",
    placeholder="Enter your Groq API key here...",
    help="You can get your API key from https://console.groq.com/"
)

if api_key and not api_key.startswith("gsk_"):
    st.warning("⚠️ Groq API keys typically start with 'gsk_'. Please verify your API key.")

if not st.session_state['docs_loaded']:
    if st.button("Load Websites", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Groq API key first.")
        else:
            st.session_state['loaded_docs'] = load_hardcoded_websites()
            st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
            
            if api_key and st.session_state['loaded_docs']:
                with st.spinner("Processing documents with enhanced date awareness..."):
                    try:
                        llm = ChatGroq(groq_api_key=api_key, model_name='meta-llama/llama-4-scout-17b-16e-instruct', temperature=0.2, top_p=0.2)
                        st.session_state['llm'] = llm
                        
                        # Enhanced prompt with better date handling
                        prompt_template = """
                        You are a regulatory document expert specializing in IRDAI, UIDAI, PMLA and egazette websites.
                        
                        **CRITICAL DATE HANDLING FOR "LATEST" QUERIES:**
                        When users ask for "latest", "recent", "new", or "current" information, you MUST:
                        
                        1. **PRIORITIZE THE MOST RECENT DATES**: Look for the highest year numbers (2025, 2024, etc.) in the provided context
                        2. **START WITH THE NEWEST**: Your first sentence should mention the most recent year/date you found
                        3. **EXPLICIT DATE MENTIONS**: Always include specific dates like "According to the latest circular dated July 29, 2025..."
                        4. **CHRONOLOGICAL ORDER**: Present information from newest to oldest
                        5. **DOCUMENT YEAR LABELS**: Each document shows its most recent year - use this information
                        
                        **RESPONSE FORMAT FOR "LATEST" QUERIES:**
                        "Based on the most recent information available from [YEAR], the latest [topic] guidelines were [specific details with date]. This updates/supersedes previous guidelines from [earlier dates if applicable]..."
                        
                        **GENERAL INSTRUCTIONS:**
                        - ONLY answer questions using the provided regulatory website context
                        - If completely outside insurance/regulatory domain or no context available: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs."
                        - If PII data detected: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on, as your query contains PII data"
                        - Reference document links and dates when mentioning acts, circulars, or regulations
                        
                        **CRITICAL**: If you see documents from 2025, 2024, and 2023, and user asks for "latest", you MUST start with 2025 information first!
                        
                        Context (documents labeled with most recent years):
                        {context}
                        
                        Question: {input}
                        
                        Remember: For "latest" queries, your FIRST sentence must mention the most recent year found in the documents.
                        """
                        
                        # Create enhanced documents with date metadata
                        enhanced_docs = create_enhanced_documents(st.session_state['loaded_docs'])
                        
                        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1500,
                            chunk_overlap=200,
                            length_function=len,
                        )
                        
                        document_chunks = text_splitter.split_documents(enhanced_docs)
                        st.write(f"Number of enhanced chunks created: {len(document_chunks)}")
                        
                        st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                        
                        # Create custom date-aware retrieval chain instead of standard chain
                        st.session_state['retrieval_chain'] = DateAwareRetrievalChain(
                            st.session_state['vector_db'], 
                            llm, 
                            prompt_template
                        )
                        
                        st.session_state['docs_loaded'] = True
                        st.success("Documents processed with enhanced date awareness and ready for querying!")
                        
                        # Display date analysis
                        all_dates = []
                        for doc in enhanced_docs:
                            dates = doc.metadata.get('extracted_dates', [])
                            all_dates.extend(dates)
                        
                        if all_dates:
                            unique_years = sorted(set(date[0] for date in all_dates), reverse=True)
                            st.info(f"📅 Document years found: {', '.join(map(str, unique_years[:10]))}")
                    
                    except Exception as e:
                        st.error(f"Error initializing enhanced LLM: {e}")
                        st.error("Please check your API key and try again.")

st.subheader("Ask Questions")
st.markdown("💡 **Tip**: Try asking for 'latest guidelines' or 'recent amendments' to see the enhanced date-aware retrieval in action!")

query = st.text_input("Enter your query:", value="What are the latest Insurance Acts and amendments?", placeholder="e.g., What are the latest IRDAI guidelines for 2025?")

if st.button("Get Answer", disabled=not api_key) and query:
    if not api_key:
        st.error("Please enter your Groq API key first.")
    elif st.session_state['retrieval_chain']:
        with st.spinner("Searching with date-aware retrieval and generating answer..."):
            try:
                response = st.session_state['retrieval_chain'].invoke({"input": query})
                
                st.subheader("Response:")
                st.write(response['answer'])
                
                # Show which years were found in the response
                response_dates = extract_dates_from_text(response['answer'])
                if response_dates:
                    response_years = sorted(set(date[0] for date in response_dates), reverse=True)
                    st.info(f"📅 Years mentioned in response: {', '.join(map(str, response_years))}")
                
                if not is_fallback_response(response['answer']):
                    retrieved_docs = response.get('context', [])
                    
                    # Enhanced document link filtering with date awareness
                    all_document_links = []
                    for doc in retrieved_docs:
                        if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                            for link_info in doc.metadata['sections']['document_links']:
                                if link_info not in all_document_links:
                                    all_document_links.append(link_info)
                    
                    if all_document_links:
                        relevant_docs = smart_document_filter(
                            all_document_links, 
                            query, 
                            response['answer'], 
                            max_docs=5  # Increased to show more recent docs
                        )
                        
                        if relevant_docs:
                            st.write("\n**📄 Most Relevant Document Links**")
                            for i, link_info in enumerate(relevant_docs):
                                date_str = ""
                                if link_info.get('extracted_date'):
                                    date_tuple = link_info['extracted_date']
                                    date_str = f" ({date_tuple[2]:02d}/{date_tuple[1]:02d}/{date_tuple[0]})"
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']}){date_str}")
                        else:
                            st.info("No high-confidence document links found that directly match the AI response content.")
                    
                    # Show document sources with their dates
                    st.write("\n**📍 Information Sources:**")
                    sources_with_dates = set()
                    for doc in retrieved_docs:
                        source = doc.metadata.get('source', 'Unknown')
                        doc_dates = doc.metadata.get('extracted_dates', [])
                        if doc_dates:
                            latest_year = max(date[0] for date in doc_dates)
                            sources_with_dates.add(f"{source} (Latest: {latest_year})")
                        else:
                            sources_with_dates.add(source)
                    
                    for i, source in enumerate(sources_with_dates, 1):
                        if source.startswith('http'):
                            st.write(f"{i}. [{source}]({source.split(' (')[0]})")
                        else:
                            st.write(f"{i}. {source}")
                    
                    # Debug information for date-aware retrieval
                    if st.checkbox("Show Debug Info (Document Dates)", value=False):
                        st.write("**Retrieved Documents with Dates:**")
                        for i, doc in enumerate(retrieved_docs):
                            doc_dates = doc.metadata.get('extracted_dates', [])
                            most_recent = max(doc_dates) if doc_dates else None
                            st.write(f"Doc {i+1}: Most recent date = {most_recent}")
                            st.write(f"Content preview: {doc.page_content[:200]}...")
                            st.write("---")
                            
                else:
                    st.info("ℹ️ No specific documents or sources are available for this query as it falls outside the current data scope.")
            
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.error("Please check your API key and try again.")
                st.write("If the error persists, try reloading the websites.")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")

# Add helpful information
st.markdown("---")
st.markdown("### About This Enhanced Version")
st.markdown("""
**Key Improvements:**
- 🎯 **Date-Aware Retrieval**: Prioritizes recent documents when you ask for "latest" information
- 📅 **Enhanced Date Extraction**: Better parsing of dates from various formats
- 🔄 **Smart Document Filtering**: Considers document recency in relevance scoring
- 📊 **Temporal Context**: Shows document years and dates in responses
- 🎮 **Debug Mode**: Optional display of how documents are ranked by date

**Best Queries to Try:**
- "What are the latest IRDAI guidelines?"
- "Show me recent insurance amendments"  
- "Current regulations for 2025"
- "Latest circular updates"
""")

st.markdown("---")
st.markdown("**Note**: This system now properly prioritizes 2025 documents over 2024, 2023, etc. when you ask for latest information!")
