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
from datetime import datetime, timedelta
import dateutil.parser

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts",
                      "https://irdai.gov.in/home",
                      "https://irdai.gov.in/rules",
                      "https://irdai.gov.in/consolidated-gazette-notified-regulations",
                      "https://irdai.gov.in/updated-regulations",
                      "https://irdai.gov.in/notifications",
                      "https://irdai.gov.in/circulars",
                      "https://irdai.gov.in/guidelines",
                      "https://irdai.gov.in/orders1",
                      "https://irdai.gov.in/notices1",
                      "https://irdai.gov.in/exposure-drafts",
                      "https://irdai.gov.in/programmes-to-advance-understanding-of-rti",
                      "https://irdai.gov.in/information-under-section-4-1-b-of-rti-act-2005",
                      "https://irdai.gov.in/information-under-section-4-1-d-of-rti-act-2005",
                      "https://irdai.gov.in/rti-act",
                      "https://irdai.gov.in/cic-orders",
                      "https://irdai.gov.in/rules2",
                      "https://irdai.gov.in/rti-2005/tenders",
                      "https://irdai.gov.in/web/guest/faqs1",
                      "https://irdai.gov.in/other-communication",
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
                      "https://enforcementdirectorate.gov.in/bsa",
                      "https://egazette.gov.in/(S(3di4ni0mu42l0jp35brfok2j))/default.aspx"
                      ]

def extract_dates_from_text(text):
    """Extract various date formats from text content"""
    date_patterns = [
        r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if len(match) == 3:
                    date_str = ' '.join(str(m) for m in match)
                    parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
                    # Only consider dates from 2000 onwards and not in the future
                    if 2000 <= parsed_date.year <= datetime.now().year:
                        dates_found.append(parsed_date)
            except:
                continue
    
    return dates_found

def extract_document_links_with_dates(html_content, url):
    """Enhanced document link extraction with date information"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
    # Look for tables with structured data (common in government websites)
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                row_text = row.get_text()
                row_dates = extract_dates_from_text(row_text)
                
                # Look for date-specific cells
                date_cell_text = ""
                for cell in cells:
                    cell_text = cell.get_text().lower()
                    if any(keyword in cell_text for keyword in ['date', 'updated', 'published', 'issued', 'modified']):
                        date_cell_text = cell.get_text(strip=True)
                        break
                
                for cell in cells:
                    links_in_cell = cell.find_all('a', href=True)
                    for link in links_in_cell:
                        href = link.get('href')
                        link_text = link.get_text(strip=True)
                        
                        if not href or len(link_text) < 5:
                            continue
                            
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        elif not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        
                        # Extract date from the same row
                        latest_date = None
                        if row_dates:
                            latest_date = max(row_dates)
                        
                        document_patterns = [
                            r'guideline', r'circular', r'act.*\d{4}', r'regulation.*\d+',
                            r'amendment.*act', r'insurance.*act', r'master.*direction',
                            r'notification', r'order', r'rule', r'policy', r'framework',
                            r'directive', r'aadhaar'
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        
                        if is_likely_document:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'structured',
                                'date': latest_date,
                                'date_text': date_cell_text,
                                'source_url': url,
                                'context': row_text[:200] + "..." if len(row_text) > 200 else row_text
                            })
    
    # Look for list items with dates
    lists = soup.find_all(['ul', 'ol'])
    for list_elem in lists:
        list_items = list_elem.find_all('li')
        for item in list_items:
            item_text = item.get_text()
            item_dates = extract_dates_from_text(item_text)
            
            links_in_item = item.find_all('a', href=True)
            for link in links_in_item:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if not href or len(link_text) < 5:
                    continue
                
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                latest_date = None
                if item_dates:
                    latest_date = max(item_dates)
                
                document_keywords = ['guideline', 'circular', 'act', 'regulation', 'rule', 
                                   'amendment', 'notification', 'insurance', 'policy', 'aadhaar']
                if any(keyword in link_text.lower() for keyword in document_keywords):
                    document_links.append({
                        'title': link_text,
                        'link': href,
                        'type': 'list',
                        'date': latest_date,
                        'date_text': '',
                        'source_url': url,
                        'context': item_text[:200] + "..." if len(item_text) > 200 else item_text
                    })
    
    # Look for div/section based content with dates
    content_sections = soup.find_all(['div', 'section', 'article'])
    for section in content_sections:
        section_text = section.get_text()
        if any(keyword in section_text.lower() for keyword in ['guideline', 'circular', 'regulation', 'act']):
            section_dates = extract_dates_from_text(section_text)
            
            links_in_section = section.find_all('a', href=True)
            for link in links_in_section:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if not href or len(link_text) < 5:
                    continue
                
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                latest_date = None
                if section_dates:
                    latest_date = max(section_dates)
                
                if 10 < len(link_text) < 200:
                    document_keywords = ['guideline', 'circular', 'act', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy', 'aadhaar']
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        document_links.append({
                            'title': link_text,
                            'link': href,
                            'type': 'content',
                            'date': latest_date,
                            'date_text': '',
                            'source_url': url,
                            'context': section_text[:200] + "..." if len(section_text) > 200 else section_text
                        })
    
    # Remove duplicates and sort by date (most recent first)
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    # Sort by date (most recent first), then by title
    unique_document_links.sort(key=lambda x: (
        x['date'] is None,  # Put items without dates at the end
        -(x['date'].timestamp()) if x['date'] else 0,  # Most recent first
        x['title']
    ))
    
    return unique_document_links

def extract_structured_content_with_metadata(html_content, url):
    """Enhanced content extraction with better metadata"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content_sections = {}
    
    # Extract page metadata
    page_dates = extract_dates_from_text(soup.get_text())
    latest_page_date = max(page_dates) if page_dates else None
    
    # Look for "last updated" information more comprehensively
    last_updated_text = ""
    last_updated_patterns = [
        r'last\s*updated[:\s]*([^<\n]+)',
        r'updated\s*on[:\s]*([^<\n]+)',
        r'modified[:\s]*([^<\n]+)',
        r'published[:\s]*([^<\n]+)',
        r'effective\s*from[:\s]*([^<\n]+)'
    ]
    
    page_text = soup.get_text()
    for pattern in last_updated_patterns:
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            last_updated_text = match.group(1).strip()[:100]
            break
    
    # Also check for meta tags with date information
    meta_tags = soup.find_all('meta')
    for meta in meta_tags:
        name = meta.get('name', '').lower()
        content = meta.get('content', '')
        if any(keyword in name for keyword in ['date', 'modified', 'updated', 'published']):
            if not last_updated_text and content:
                last_updated_text = content[:100]
    
    # Extract document links with dates
    content_sections['document_links'] = extract_document_links_with_dates(html_content, url)
    
    # Categorize content by type with enhanced detection
    guidelines_content = []
    circulars_content = []
    acts_content = []
    regulations_content = []
    notifications_content = []
    
    # Look for specific content sections with better context
    for section in soup.find_all(['div', 'section', 'article', 'p']):
        section_text = section.get_text()
        section_lower = section_text.lower()
        
        if len(section_text.strip()) > 50:  # Only consider substantial content
            if 'guideline' in section_lower:
                guidelines_content.append({
                    'text': section_text.strip(),
                    'dates': extract_dates_from_text(section_text)
                })
            elif 'circular' in section_lower:
                circulars_content.append({
                    'text': section_text.strip(),
                    'dates': extract_dates_from_text(section_text)
                })
            elif 'act' in section_lower and ('insurance' in section_lower or 'aadhaar' in section_lower):
                acts_content.append({
                    'text': section_text.strip(),
                    'dates': extract_dates_from_text(section_text)
                })
            elif 'regulation' in section_lower:
                regulations_content.append({
                    'text': section_text.strip(),
                    'dates': extract_dates_from_text(section_text)
                })
            elif 'notification' in section_lower:
                notifications_content.append({
                    'text': section_text.strip(),
                    'dates': extract_dates_from_text(section_text)
                })
    
    content_sections['guidelines'] = guidelines_content
    content_sections['circulars'] = circulars_content
    content_sections['acts'] = acts_content
    content_sections['regulations'] = regulations_content
    content_sections['notifications'] = notifications_content
    content_sections['page_date'] = latest_page_date
    content_sections['last_updated'] = last_updated_text
    
    # Enhanced main text extraction with date context
    main_text = soup.get_text()
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Add date context to the main text if available
    if latest_page_date or last_updated_text:
        date_context = f"\n--- Page Metadata ---\n"
        if latest_page_date:
            date_context += f"Latest Date Found: {latest_page_date.strftime('%Y-%m-%d')}\n"
        if last_updated_text:
            date_context += f"Last Updated: {last_updated_text}\n"
        date_context += "--- End Metadata ---\n"
        clean_text = date_context + clean_text
    
    return clean_text, content_sections

def smart_document_filter_with_dates(document_links, query, ai_response, max_docs=5):
    """Enhanced filtering that prioritizes recent documents"""
    if not document_links:
        return []
    
    # Check if query is asking for "latest" or "recent"
    is_latest_query = any(keyword in query.lower() for keyword in [
        'latest', 'recent', 'new', 'current', 'updated', 'newest', 'fresh'
    ])
    
    ai_response_lower = ai_response.lower()
    ai_response_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', ai_response_lower))
    
    # Extract mentions from AI response
    mentioned_acts = re.findall(r'\b[\w\s]*act[\w\s]*\b', ai_response_lower)
    mentioned_regulations = re.findall(r'\b[\w\s]*regulation[\w\s]*\b', ai_response_lower)
    mentioned_circulars = re.findall(r'\b[\w\s]*circular[\w\s]*\b', ai_response_lower)
    mentioned_guidelines = re.findall(r'\b[\w\s]*guideline[\w\s]*\b', ai_response_lower)
    mentioned_notifications = re.findall(r'\b[\w\s]*notification[\w\s]*\b', ai_response_lower)
    
    specific_mentions = (mentioned_acts + mentioned_regulations + mentioned_circulars + 
                        mentioned_guidelines + mentioned_notifications)
    mentioned_years = set(re.findall(r'\b(20\d{2})\b', ai_response))
    
    high_confidence_docs = []
    current_year = datetime.now().year
    current_date = datetime.now()
    
    for doc in document_links:
        title = doc.get('title', '').strip()
        title_lower = title.lower()
        doc_date = doc.get('date')
        
        # Skip obviously non-content links
        if (len(title) < 10 or 
            title.lower() in ['click here', 'read more', 'download', 'view more', 'see all'] or
            any(skip_word in title.lower() for skip_word in ['home', 'contact', 'about us', 'sitemap', 'login'])):
            continue
        
        confidence_score = 0
        match_reasons = []
        
        # Enhanced date-based scoring (more important for "latest" queries)
        if doc_date:
            days_old = (current_date - doc_date).days
            if is_latest_query:
                if days_old <= 7:  # Very recent - within a week
                    confidence_score += 60
                    match_reasons.append("Very recent (within 7 days)")
                elif days_old <= 30:  # Recent - within a month
                    confidence_score += 45
                    match_reasons.append("Recent (within 30 days)")
                elif days_old <= 90:  # Fairly recent - within 3 months
                    confidence_score += 35
                    match_reasons.append("Fairly recent (within 90 days)")
                elif days_old <= 365:  # This year
                    confidence_score += 25
                    match_reasons.append("Current year")
                elif doc_date.year >= current_year - 1:  # Last year
                    confidence_score += 15
                    match_reasons.append("Recent (last year)")
                else:  # Older documents get lower scores for latest queries
                    confidence_score += 5
            else:
                # For non-latest queries, still give preference to recent docs but less aggressively
                if days_old <= 365:
                    confidence_score += 15
                    match_reasons.append("Recent content")
                elif days_old <= 365 * 2:
                    confidence_score += 10
                    match_reasons.append("Moderately recent")
        elif is_latest_query:
            # Penalize documents without dates for latest queries
            confidence_score -= 10
        
        # Content matching (enhanced)
        for mention in specific_mentions:
            mention_clean = mention.strip()
            if len(mention_clean) > 5 and mention_clean in title_lower:
                confidence_score += 40
                match_reasons.append(f"Exact mention: '{mention_clean}'")
        
        # Enhanced keyword matching
        title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_lower))
        common_words = ai_response_words.intersection(title_words)
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
                     'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'from'}
        meaningful_common_words = common_words - stop_words
        
        if len(meaningful_common_words) >= 4:
            confidence_score += 30
            match_reasons.append(f"Many key terms: {list(meaningful_common_words)[:4]}")
        elif len(meaningful_common_words) >= 2:
            confidence_score += 20
            match_reasons.append(f"Key terms: {list(meaningful_common_words)[:2]}")
        
        # Year matching (enhanced for latest queries)
        if mentioned_years:
            title_years = set(re.findall(r'\b(20\d{2})\b', title))
            doc_year = {str(doc_date.year)} if doc_date else set()
            
            year_matches = mentioned_years.intersection(title_years.union(doc_year))
            if year_matches:
                year_boost = 20 if is_latest_query and any(int(y) >= current_year - 1 for y in year_matches) else 15
                confidence_score += year_boost
                match_reasons.append(f"Year match: {year_matches}")
        
        # Document type matching (enhanced)
        regulatory_types = ['guideline', 'circular', 'act', 'regulation', 'amendment', 
                          'notification', 'policy', 'rule', 'framework', 'directive', 'order']
        ai_mentions_reg_type = any(reg_type in ai_response_lower for reg_type in regulatory_types)
        title_has_reg_type = any(reg_type in title_lower for reg_type in regulatory_types)
        
        if ai_mentions_reg_type and title_has_reg_type:
            matching_reg_types = [reg_type for reg_type in regulatory_types 
                                if reg_type in ai_response_lower and reg_type in title_lower]
            if matching_reg_types:
                confidence_score += 25
                match_reasons.append(f"Document type match: {matching_reg_types}")
        
        # Domain-specific matching
        domain_terms = ['insurance', 'aadhaar', 'uidai', 'irdai', 'pmla', 'licensing', 
                       'compliance', 'regulatory', 'financial']
        ai_domain_terms = [term for term in domain_terms if term in ai_response_lower]
        title_domain_terms = [term for term in domain_terms if term in title_lower]
        
        matching_domain_terms = set(ai_domain_terms).intersection(set(title_domain_terms))
        if matching_domain_terms:
            confidence_score += 20
            match_reasons.append(f"Domain terms: {list(matching_domain_terms)}")
        
        # Special boost for high-relevance latest queries
        if is_latest_query and confidence_score >= 40:
            confidence_score += 15
            match_reasons.append("Latest query relevance boost")
        
        # Minimum threshold for inclusion (lowered to capture more relevant docs)
        if confidence_score >= 20:
            high_confidence_docs.append({
                'doc': doc,
                'score': confidence_score,
                'reasons': match_reasons,
                'date': doc_date,
                'is_recent': doc_date and (current_date - doc_date).days <= 90 if doc_date else False
            })
    
    # Enhanced sorting: score first, then recency for latest queries
    if is_latest_query:
        high_confidence_docs.sort(key=lambda x: (
            -x['score'],  # Highest score first
            not x['is_recent'],  # Recent docs first within same score
            -(x['date'].timestamp()) if x['date'] else 0  # Most recent first
        ))
    else:
        high_confidence_docs.sort(key=lambda x: (
            -x['score'],  # Highest score first
            -(x['date'].timestamp()) if x['date'] else 0  # Most recent first for same score
        ))
    
    return [item['doc'] for item in high_confidence_docs[:max_docs]]

def enhanced_web_scrape(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=20)
        response.raise_for_status()
        return response.text
        
    except Exception as e:
        st.error(f"Enhanced scraping failed for {url}: {e}")
        return None

def load_hardcoded_websites():
    """Enhanced website loading with better metadata extraction"""
    loaded_docs = []
    
    for i, url in enumerate(HARDCODED_WEBSITES):
        try:
            st.write(f"Loading URL {i+1}/{len(HARDCODED_WEBSITES)}: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content_with_metadata(html_content, url)
                
                # Create enhanced metadata
                metadata = {
                    "source": url,
                    "sections": sections,
                    "page_date": sections.get('page_date'),
                    "last_updated": sections.get('last_updated'),
                    "content_type": "regulatory_content",
                    "load_timestamp": datetime.now().isoformat()
                }
                
                # Determine content type based on URL
                if 'guideline' in url:
                    metadata['primary_type'] = 'guidelines'
                elif 'circular' in url:
                    metadata['primary_type'] = 'circulars'
                elif 'act' in url:
                    metadata['primary_type'] = 'acts'
                elif 'regulation' in url:
                    metadata['primary_type'] = 'regulations'
                elif 'notification' in url:
                    metadata['primary_type'] = 'notifications'
                else:
                    metadata['primary_type'] = 'general'
                
                doc = Document(page_content=clean_text, metadata=metadata)
                loaded_docs.append(doc)
                
                # Enhanced display of information
                if sections.get('document_links'):
                    with st.expander(f"📄 Documents from {url} (sorted by date)", expanded=False):
                        doc_links = sections['document_links']
                        st.write(f"**Total document links found:** {len(doc_links)}")
                        
                        # Show recent documents first (with dates)
                        recent_docs = [link for link in doc_links[:15] if link.get('date')]
                        if recent_docs:
                            st.write("**📅 Recent Documents (with dates):**")
                            for i, link_info in enumerate(recent_docs[:10]):
                                date_str = link_info['date'].strftime('%Y-%m-%d') if link_info['date'] else 'No date'
                                days_ago = (datetime.now() - link_info['date']).days if link_info['date'] else None
                                age_str = f"({days_ago} days ago)" if days_ago is not None else ""
                                st.write(f"{i+1}. **{link_info['title']}** - *{date_str} {age_str}*")
                                st.write(f"   🔗 [Link]({link_info['link']})")
                        
                        # Show other documents without dates
                        other_docs = [link for link in doc_links if not link.get('date')]
                        if other_docs:
                            st.write("**📋 Other Documents:**")
                            for i, link_info in enumerate(other_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                
                # Show last updated information if available
                if sections.get('last_updated'):
                    st.info(f"ℹ️ Last updated information found: {sections['last_updated']}")
                
                # Show content type statistics
                content_stats = []
                if sections.get('guidelines'):
                    content_stats.append(f"Guidelines: {len(sections['guidelines'])}")
                if sections.get('circulars'):
                    content_stats.append(f"Circulars: {len(sections['circulars'])}")
                if sections.get('regulations'):
                    content_stats.append(f"Regulations: {len(sections['regulations'])}")
                if sections.get('notifications'):
                    content_stats.append(f"Notifications: {len(sections['notifications'])}")
                
                if content_stats:
                    st.write(f"📊 Content found: {', '.join(content_stats)}")
            
            st.success(f"✅ Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"❌ Error loading {url}: {e}")
            continue
    
    return loaded_docs

def create_enhanced_prompt():
    """Create enhanced prompt template for better date-aware responses"""
    return ChatPromptTemplate.from_template(
        """
        You are a regulatory expert assistant specializing in IRDAI, UIDAI, PMLA and other regulatory websites.
        
        CRITICAL INSTRUCTIONS FOR DATE-AWARE RESPONSES:
        
        1. **LATEST/RECENT QUERIES**: When asked about "latest", "recent", "new", or "current" content:
           - PRIORITIZE information with the most recent dates
           - Look for "Last Updated", "Published", "Effective from" dates in the context
           - If multiple versions exist, choose the most recent one
           - Clearly mention specific dates when available
           - Sort information chronologically (most recent first)
        
        2. **DATE HANDLING**:
           - Always check the "Page Metadata" sections for latest dates
           - Pay attention to dates in document titles and descriptions
           - When listing items, order by date (newest first) when dates are available
           - If no recent content exists, clearly state the most recent available date
        
        3. **RESPONSE FORMAT FOR LATEST QUERIES**:
           - Start with the most recent items
           - Include publication/update dates in your response
           - Use phrases like "As of [date]", "Updated on [date]", "Published [date]"
           - If asked for "latest guidelines from IRDAI", focus on guidelines section with newest dates
        
        4. **CONTEXT ANALYSIS**:
           - Look for patterns like "Updated on", "Last modified", "Effective from"
           - Check document metadata for date information
           - Consider documents from the last 6 months as "recent"
           - Documents from the last month as "very recent" or "latest"
        
        Current date for reference: {current_date}
        
        Based on the context provided, answer the user's question with strong emphasis on recency and chronological order.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        
        Provide a comprehensive answer prioritizing the most recent and relevant information available, with specific dates mentioned wherever possible.
        """
    )

def is_fallback_response(response_text):
    """Check if the response is a fallback response"""
    fallback_phrases = [
        "fall outside the scope of the data I've been trained on",
        "details you've asked for fall outside the scope",
        "outside the scope of the data",
        "However, I've gathered information that closely aligns",
        "contains PII data"
    ]
    
    return any(phrase in response_text for phrase in fallback_phrases)

def setup_enhanced_retrieval_chain(llm, vector_db):
    """Setup retrieval chain with enhanced prompt"""
    prompt = create_enhanced_prompt()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})  # Increased for better coverage
    return create_retrieval_chain(retriever, document_chain)

def process_query_with_date_context(retrieval_chain, query):
    """Process query with current date context"""
    current_date = datetime.now().strftime("%Y-%m-%d (%A)")
    return retrieval_chain.invoke({
        "input": query,
        "current_date": current_date
    })

# Session state initialization
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

# Main Streamlit App
st.set_page_config(page_title="Web GEN-ie Enhanced", page_icon="🤖", layout="wide")

st.title("🤖 Web GEN-ie - Enhanced Regulatory Assistant")
st.markdown("*Advanced AI assistant with date-aware latest content detection for IRDAI, UIDAI, PMLA and other regulatory websites*")

# Configuration Section
st.subheader("⚙️ Configuration")
col1, col2 = st.columns([2, 1])

with col1:
    api_key = st.text_input(
        "Enter your Groq API Key:", 
        type="password",
        placeholder="Enter your Groq API key here...",
        help="You can get your API key from https://console.groq.com/"
    )

with col2:
    st.markdown("**Model Settings**")
    st.write("• Model: Llama-4-Scout-17B")
    st.write("• Temperature: 0.2")
    st.write("• Enhanced Date Detection: ✅")

if api_key and not api_key.startswith("gsk_"):
    st.warning("⚠️ Groq API keys typically start with 'gsk_'. Please verify your API key.")

# Website Loading Section
st.subheader("📚 Website Data Loading")

if not st.session_state['docs_loaded']:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Load All Websites", disabled=not api_key, type="primary"):
            if not api_key:
                st.error("Please enter your Groq API key first.")
            else:
                with st.spinner("Loading and processing websites... This may take a few minutes."):
                    # Load websites
                    st.session_state['loaded_docs'] = load_hardcoded_websites()
                    st.success(f"✅ Total loaded documents: {len(st.session_state['loaded_docs'])}")
                    
                    if api_key and st.session_state['loaded_docs']:
                        with st.spinner("🔄 Processing documents and creating vector database..."):
                            try:
                                # Initialize LLM
                                llm = ChatGroq(
                                    groq_api_key=api_key, 
                                    model_name='meta-llama/llama-4-scout-17b-16e-instruct', 
                                    temperature=0.2, 
                                    top_p=0.2
                                )
                                st.session_state['llm'] = llm
                                
                                # Initialize embeddings
                                hf_embedding = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                                )
                                
                                # Create text splitter with optimized settings
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1200,  # Slightly smaller for better precision
                                    chunk_overlap=250,  # More overlap for context preservation
                                    length_function=len,
                                    separators=["\n\n", "\n", ". ", " ", ""]
                                )
                                
                                # Split documents
                                document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
                                st.write(f"📄 Number of text chunks created: {len(document_chunks)}")
                                
                                # Create vector database
                                st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                                
                                # Create enhanced retrieval chain
                                st.session_state['retrieval_chain'] = setup_enhanced_retrieval_chain(
                                    llm, st.session_state['vector_db']
                                )
                                
                                st.session_state['docs_loaded'] = True
                                st.success("🎉 Documents processed and ready for intelligent querying!")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"❌ Error initializing LLM: {e}")
                                st.error("Please check your API key and try again.")
    
    with col2:
        st.info("📋 **What this will load:**")
        st.write(f"• {len(HARDCODED_WEBSITES)} regulatory websites")
        st.write("• IRDAI guidelines, circulars, acts")
        st.write("• UIDAI regulations and frameworks")
        st.write("• PMLA compliance documents")
        st.write("• E-Gazette notifications")
        st.write("• **Enhanced date extraction and sorting**")

else:
    st.success("✅ Websites already loaded and processed!")
    if st.button("🔄 Reload All Websites"):
        st.session_state['docs_loaded'] = False
        st.rerun()

# Query Section
st.subheader("💬 Ask Your Questions")

# Sample queries
sample_queries = [
    "Latest guidelines from IRDAI",
    "Recent UIDAI circulars and updates",
    "New insurance regulations in 2024",
    "Current PMLA compliance requirements",
    "Latest amendments to insurance acts",
    "Recent notifications from regulatory authorities"
]

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your query:", 
        value="Latest guidelines from IRDAI",
        placeholder="Ask about latest guidelines, recent circulars, new regulations..."
    )

with col2:
    st.markdown("**Sample Queries:**")
    for i, sample in enumerate(sample_queries[:3]):
        if st.button(f"📝 {sample}", key=f"sample_{i}"):
            query = sample

# Enhanced query processing
if st.button("🔍 Get Answer", disabled=not api_key, type="primary") and query:
    if not api_key:
        st.error("Please enter your Groq API key first.")
    elif st.session_state['retrieval_chain']:
        with st.spinner("🔍 Searching through regulatory documents and generating comprehensive answer..."):
            try:
                # Process query with date context
                response = process_query_with_date_context(st.session_state['retrieval_chain'], query)
                
                # Display response
                st.subheader("📋 Response:")
                st.markdown(response['answer'])
                
                # Enhanced document links display for non-fallback responses
                if not is_fallback_response(response['answer']):
                    retrieved_docs = response.get('context', [])
                    
                    # Collect all document links from retrieved documents
                    all_document_links = []
                    for doc in retrieved_docs:
                        if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                            for link_info in doc.metadata['sections']['document_links']:
                                if link_info not in all_document_links:
                                    all_document_links.append(link_info)
                    
                    # Apply smart filtering with date awareness
                    if all_document_links:
                        relevant_docs = smart_document_filter_with_dates(
                            all_document_links, 
                            query, 
                            response['answer'], 
                            max_docs=5
                        )
                        
                        if relevant_docs:
                            st.markdown("### 📄 **Most Relevant Document Links**")
                            
                            # Check if this is a "latest" query for better formatting
                            is_latest_query = any(keyword in query.lower() for keyword in [
                                'latest', 'recent', 'new', 'current', 'updated'
                            ])
                            
                            for i, link_info in enumerate(relevant_docs):
                                with st.container():
                                    st.markdown(f"**{i+1}. {link_info['title']}**")
                                    
                                    # Enhanced date display
                                    if link_info.get('date'):
                                        date_str = link_info['date'].strftime('%B %d, %Y')
                                        days_ago = (datetime.now() - link_info['date']).days
                                        
                                        if days_ago == 0:
                                            age_str = "📅 Today"
                                        elif days_ago == 1:
                                            age_str = "📅 Yesterday"
                                        elif days_ago <= 7:
                                            age_str = f"📅 {days_ago} days ago"
                                        elif days_ago <= 30:
                                            age_str = f"📅 {days_ago // 7} week(s) ago"
                                        elif days_ago <= 365:
                                            age_str = f"📅 {days_ago // 30} month(s) ago"
                                        else:
                                            age_str = f"📅 {days_ago // 365} year(s) ago"
                                        
                                        st.markdown(f"   *{date_str} ({age_str})*")
                                    else:
                                        st.markdown("   *Date not available*")
                                    
                                    st.markdown(f"   🔗 [Access Document]({link_info['link']})")
                                    
                                    # Show context if available
                                    if link_info.get('context') and len(link_info['context']) > 50:
                                        with st.expander("📖 Context Preview", expanded=False):
                                            st.write(link_info['context'][:300] + "..." if len(link_info['context']) > 300 else link_info['context'])
                                    
                                    st.markdown("---")
                        else:
                            st.info("ℹ️ No high-confidence document links found that directly match the query requirements.")
                    
                    # Display information sources
                    st.markdown("### 📍 **Information Sources**")
                    sources = []
                    for doc in retrieved_docs:
                        source_info = {
                            'url': doc.metadata.get('source', 'Unknown'),
                            'type': doc.metadata.get('primary_type', 'general'),
                            'last_updated': doc.metadata.get('last_updated', ''),
                            'page_date': doc.metadata.get('page_date')
                        }
                        
                        # Avoid duplicates
                        if source_info not in sources:
                            sources.append(source_info)
                    
                    for i, source in enumerate(sources[:5], 1):
                        st.markdown(f"**{i}. [{source['url']}]({source['url']})**")
                        if source['type'] != 'general':
                            st.markdown(f"   📋 Type: {source['type'].title()}")
                        if source['last_updated']:
                            st.markdown(f"   🔄 Last Updated: {source['last_updated']}")
                        if source['page_date']:
                            st.markdown(f"   📅 Page Date: {source['page_date'].strftime('%Y-%m-%d')}")
                
                else:
                    st.info("ℹ️ The query falls outside the current data scope. No specific documents or sources are available.")
                    
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
                st.error("Please check your API key and try again.")
    else:
        st.warning("⚠️ Please load websites first by clicking the 'Load All Websites' button.")

# Footer
st.markdown("---")
st.markdown("*Enhanced Web GEN-ie with intelligent date-aware content detection and latest document prioritization*")
st.markdown("🚀 **Features**: Date extraction • Smart filtering • Latest content prioritization • Comprehensive regulatory coverage")
