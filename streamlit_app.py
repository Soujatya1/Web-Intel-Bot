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
from collections import defaultdict


HARDCODED_WEBSITES = ["https://irdai.gov.in/acts",
                     "https://uidai.gov.in/en/my-aadhaar/downloads/acts-and-rules.html"
                     ]

def get_documents_from_context_chunks(retrieved_docs, query, ai_response, max_docs=3):
    
    relevant_document_links = []
    
    for doc in retrieved_docs:
        doc_metadata = doc.metadata
        doc_content = doc.page_content
        
        if 'sections' in doc_metadata and 'document_links' in doc_metadata['sections']:
            chunk_doc_links = doc_metadata['sections']['document_links']
            
            for link_info in chunk_doc_links:
                enhanced_link = {
                    'title': link_info['title'],
                    'link': link_info['link'],
                    'type': link_info['type'],
                    'source_chunk': doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,
                    'source_url': doc_metadata.get('source', ''),
                    'relevance_context': doc_content,
                    'column_info': link_info.get('column_info', {}),  # New field to store column context
                    'priority': link_info.get('priority', 0)  # New field for link priority
                }
                relevant_document_links.append(enhanced_link)
    
    seen_links = set()
    unique_links = []
    for link in relevant_document_links:
        if link['link'] not in seen_links:
            seen_links.add(link['link'])
            unique_links.append(link)
    
    scored_links = []
    for link in unique_links:
        score = calculate_context_relevance_score(link, query, ai_response)
        if score > 0:
            scored_links.append((link, score))
    
    # Sort by priority first, then by score
    scored_links.sort(key=lambda x: (x[0]['priority'], x[1]), reverse=True)
    
    result_links = []
    
    # Prioritize content links (from subtitle/content columns) over document downloads
    content_links = [(link, score) for link, score in scored_links if link['type'] in ['content', 'reference']]
    document_files = [(link, score) for link, score in scored_links if link['type'] == 'document']
    
    # First add content links (higher priority)
    for link, score in content_links[:max_docs]:
        result_links.append(link)
    
    # Fill remaining slots with document files
    remaining_slots = max_docs - len(result_links)
    for link, score in document_files[:remaining_slots]:
        result_links.append(link)
    
    return result_links

def calculate_context_relevance_score(link_info, query, ai_response):
    
    score = 0
    
    context_text = link_info['relevance_context'].lower()
    link_title = link_info['title'].lower()
    query_lower = query.lower()
    response_lower = ai_response.lower()
    
    # Prioritize content/reference links over direct document downloads
    if link_info['type'] in ['content', 'reference']:
        score += 20  # Higher base score for content links
    elif link_info['type'] == 'document':
        score += 10
    
    # Additional priority boost for links from important columns
    column_info = link_info.get('column_info', {})
    if column_info.get('is_primary_content', False):
        score += 15
    
    query_words = [word for word in query_lower.split() if len(word) > 3]
    for word in query_words:
        if word in context_text:
            score += 8
            if word in link_title:
                score += 5
    
    response_words = [word for word in response_lower.split() if len(word) > 4]
    common_words = set(['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    response_words = [word for word in response_words if word not in common_words]
    
    context_response_matches = 0
    for word in response_words[:20]:
        if word in context_text:
            context_response_matches += 1
    
    if len(response_words) > 0:
        match_ratio = context_response_matches / len(response_words)
        score += int(match_ratio * 20)
    
    regulatory_patterns = [
        r'act.*\d{4}', r'circular.*\d+', r'amendment.*act', r'insurance.*act',
        r'guideline', r'master.*direction', r'regulation.*\d+', r'notification.*\d+'
    ]
    
    for pattern in regulatory_patterns:
        if re.search(pattern, link_title):
            score += 10
            break
    
    years_in_query = re.findall(r'\b(20\d{2})\b', query_lower)
    years_in_response = re.findall(r'\b(20\d{2})\b', response_lower)
    years_in_title = re.findall(r'\b(20\d{2})\b', link_title)
    
    for year in set(years_in_query + years_in_response):
        if year in years_in_title:
            score += 12
    
    poor_titles = ['click here', 'read more', 'download', 'view', 'see more', 'link']
    if any(poor_title in link_title for poor_title in poor_titles) or len(link_info['title'].strip()) < 10:
        score -= 10
    
    return max(0, score)

def enhanced_web_scrape(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        session = requests.Session()
        session.headers.update(headers)
        
        # Try multiple requests to handle dynamic content
        response = session.get(url, timeout=20)
        response.raise_for_status()
        
        # Wait a bit for any JavaScript to execute (though this won't help with client-side JS)
        time.sleep(2)
        
        return response.text
        
    except Exception as e:
        st.error(f"Enhanced scraping failed for {url}: {e}")
        return None

def extract_dynamic_content_patterns(html_content, url):
    """
    Extract content that might be loaded dynamically or through APIs
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    dynamic_links = []
    
    # Look for JSON data embedded in the page
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string:
            script_content = script.string
            
            # Look for JSON arrays that might contain document information
            json_patterns = [
                r'(?:documents|files|items|data)\s*:\s*(\[.*?\])',
                r'var\s+(?:documents|files|items|data)\s*=\s*(\[.*?\]);',
                r'(?:documents|files|items|data)\s*=\s*(\[.*?\]);'
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, script_content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # Try to parse as JSON (this is a simplified approach)
                        # In a real implementation, you'd want more robust JSON parsing
                        if 'title' in match.lower() and 'url' in match.lower():
                            # This indicates structured data that might contain document links
                            # For now, we'll extract text patterns that look like documents
                            doc_titles = re.findall(r'"title"\s*:\s*"([^"]+)"', match)
                            doc_urls = re.findall(r'"(?:url|link|href)"\s*:\s*"([^"]+)"', match)
                            
                            for title, doc_url in zip(doc_titles, doc_urls):
                                if len(title) > 5:
                                    full_url = urljoin(url, doc_url) if not doc_url.startswith(('http://', 'https://')) else doc_url
                                    dynamic_links.append({
                                        'title': title,
                                        'link': full_url,
                                        'type': 'content',
                                        'context': 'Extracted from dynamic content',
                                        'priority': 4,
                                        'column_info': {'is_primary_content': True, 'source': 'dynamic_json'}
                                    })
                    except:
                        continue
    
    return dynamic_links

def detect_content_column(table_rows):
    """
    Intelligently detect which column likely contains the main content links
    """
    if not table_rows:
        return None
    
    # Analyze header row to identify potential content columns
    header_row = table_rows[0] if table_rows else None
    if not header_row:
        return None
    
    headers = header_row.find_all(['th', 'td'])
    content_column_candidates = []
    
    for idx, header in enumerate(headers):
        header_text = header.get_text(strip=True).lower()
        
        # Look for headers that suggest main content
        content_indicators = [
            'title', 'subtitle', 'sub title', 'name', 'description', 'subject',
            'act', 'regulation', 'circular', 'guideline', 'policy', 'document',
            'short description', 'details'
        ]
        
        score = 0
        for indicator in content_indicators:
            if indicator in header_text:
                score += len(indicator)  # Longer matches get higher scores
        
        if score > 0:
            content_column_candidates.append((idx, score, header_text))
    
    # Sort by score and return the best candidate
    if content_column_candidates:
        content_column_candidates.sort(key=lambda x: x[1], reverse=True)
        return content_column_candidates[0][0]  # Return column index
    
    # Fallback: look for columns with most links in data rows
    column_link_counts = defaultdict(int)
    for row in table_rows[1:6]:  # Check first few data rows
        cells = row.find_all(['td', 'th'])
        for idx, cell in enumerate(cells):
            links = cell.find_all('a', href=True)
            column_link_counts[idx] += len(links)
    
    if column_link_counts:
        best_column = max(column_link_counts.items(), key=lambda x: x[1])
        return best_column[0]
    
    return None

def extract_document_links_with_context(html_content, url):

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Don't remove scripts initially - we need to analyze them for dynamic links
    document_links = []
    document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
    
    # First, extract regular links (keeping existing functionality)
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        href = link.get('href')
        link_text = link.get_text(strip=True)
        
        if not href or len(link_text) < 3:
            continue
            
        if href.startswith('/'):
            href = urljoin(url, href)
        elif not href.startswith(('http://', 'https://')):
            href = urljoin(url, href)
        
        context_text = ""
        parent = link.parent
        if parent:
            context_text = parent.get_text(strip=True)
            if len(context_text) < 50 and parent.parent:
                context_text = parent.parent.get_text(strip=True)
        
        is_document_link = any(ext in href.lower() for ext in document_extensions)
        
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive']
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if is_document_link or has_doc_keywords:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'document' if is_document_link else 'content',
                'context': context_text[:300],
                'priority': 1 if is_document_link else 2,  # Content links get higher priority
                'column_info': {'is_primary_content': False}
            })
    
    # Extract JavaScript-based links and dynamic content
    js_links = extract_javascript_links(soup, url)
    document_links.extend(js_links)
    
    # Extract clickable elements that aren't traditional links
    clickable_elements = extract_clickable_elements(soup, url)
    document_links.extend(clickable_elements)
    
    # Now remove scripts and styles for table processing
    for script in soup(["script", "style"]):
        script.decompose()
    
def extract_javascript_links(soup, base_url):
    """
    Extract links from JavaScript code, data attributes, and onclick handlers
    """
    js_links = []
    
    # Look for script tags with URLs
    scripts = soup.find_all('script')
    for script in scripts:
        if script.string:
            script_content = script.string
            
            # Look for URL patterns in JavaScript
            url_patterns = [
                r'["\']([^"\']*\.(?:pdf|doc|docx|xls|xlsx))["\']',  # Direct file references
                r'window\.open\(["\']([^"\']+)["\']',  # window.open calls
                r'location\.href\s*=\s*["\']([^"\']+)["\']',  # location.href assignments
                r'href\s*:\s*["\']([^"\']+)["\']',  # href properties
                r'url\s*:\s*["\']([^"\']+)["\']',  # url properties
            ]
            
            for pattern in url_patterns:
                matches = re.findall(pattern, script_content, re.IGNORECASE)
                for match in matches:
                    if len(match) > 5:  # Basic validation
                        full_url = urljoin(base_url, match) if not match.startswith(('http://', 'https://')) else match
                        js_links.append({
                            'title': match.split('/')[-1] if '/' in match else match,
                            'link': full_url,
                            'type': 'document' if any(ext in match.lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']) else 'content',
                            'context': 'Extracted from JavaScript',
                            'priority': 3,
                            'column_info': {'is_primary_content': False, 'source': 'javascript'}
                        })
    
    return js_links

def extract_clickable_elements(soup, base_url):
    """
    Extract links from clickable elements that aren't traditional <a> tags
    """
    clickable_links = []
    
    # Look for elements with onclick, data-url, data-href, data-link attributes
    clickable_selectors = [
        '[onclick]',
        '[data-url]',
        '[data-href]',
        '[data-link]',
        '[data-file]',
        '[data-download]',
        '.clickable',
        '.download-link',
        '.file-link',
        '.document-link'
    ]
    
    for selector in clickable_selectors:
        elements = soup.select(selector)
        for element in elements:
            link_url = None
            element_text = element.get_text(strip=True)
            
            # Check various attributes for URLs
            for attr in ['data-url', 'data-href', 'data-link', 'data-file', 'data-download']:
                if element.get(attr):
                    link_url = element.get(attr)
                    break
            
            # Parse onclick handlers for URLs
            if not link_url and element.get('onclick'):
                onclick = element.get('onclick')
                url_match = re.search(r'["\']([^"\']*(?:\.(?:pdf|doc|docx|xls|xlsx)|/[^"\']*|http[^"\']*)[^"\']*)["\']', onclick)
                if url_match:
                    link_url = url_match.group(1)
            
            if link_url and len(element_text) > 3:
                if link_url.startswith('/'):
                    link_url = urljoin(base_url, link_url)
                elif not link_url.startswith(('http://', 'https://')):
                    link_url = urljoin(base_url, link_url)
                
                # Get context from parent elements
                context_text = ""
                parent = element.parent
                if parent:
                    context_text = parent.get_text(strip=True)
                    if len(context_text) < 50 and parent.parent:
                        context_text = parent.parent.get_text(strip=True)
                
                is_document = any(ext in link_url.lower() for ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx'])
                
                clickable_links.append({
                    'title': element_text,
                    'link': link_url,
                    'type': 'document' if is_document else 'content',
                    'context': context_text[:300],
                    'priority': 4,  # High priority for explicitly clickable elements
                    'column_info': {'is_primary_content': True, 'source': 'clickable_element'}
                })
    
    # Look for list items that might be clickable documents
    list_items = soup.find_all(['li', 'div'], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ['document', 'file', 'download', 'item', 'entry', 'list-item']
    ))
    
    for item in list_items:
        item_text = item.get_text(strip=True)
        
        # Look for document-like patterns in the text
        doc_patterns = [
            r'(.+?(?:act|amendment|circular|guideline|regulation|rule|notification|order|policy).*?)(?:\d+\.\d+\s*MB|\d+\s*KB|\d{1,2}\s*[A-Za-z]{3}\s*\d{4})',
            r'(\d+\.\s*.+?(?:act|amendment|circular|guideline|regulation|rule|notification|order|policy).*?)(?:\d+\.\d+\s*MB|\d+\s*KB)',
        ]
        
        for pattern in doc_patterns:
            match = re.search(pattern, item_text, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                if len(title) > 10:
                    # Try to construct a reasonable URL based on the title and context
                    # This is a best-guess approach for dynamically loaded content
                    potential_url = f"{base_url}#{title}"  # Placeholder URL
                    
                    clickable_links.append({
                        'title': title,
                        'link': potential_url,
                        'type': 'content',
                        'context': item_text[:300],
                        'priority': 3,
                        'column_info': {'is_primary_content': True, 'source': 'list_item'}
                    })
                    break
    
    return clickable_links
    for table in tables:
        table_context = ""
        prev_sibling = table.find_previous_sibling(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        if prev_sibling:
            table_context = prev_sibling.get_text(strip=True)
        
        rows = table.find_all('tr')
        if not rows:
            continue
        
        # Detect the primary content column
        primary_content_column = detect_content_column(rows)
        
        for row_idx, row in enumerate(rows):
            if row_idx == 0:  # Skip header row
                continue
                
            cells = row.find_all(['td', 'th'])
            row_text = ' '.join(cell.get_text(strip=True) for cell in cells)
            
            for cell_idx, cell in enumerate(cells):
                links_in_cell = cell.find_all('a', href=True)
                
                # Determine if this cell is in the primary content column
                is_primary_content = (primary_content_column is not None and 
                                    cell_idx == primary_content_column)
                
                for link in links_in_cell:
                    href = link.get('href')
                    link_text = link.get_text(strip=True)
                    
                    if not href or len(link_text) < 5:
                        continue
                        
                    if href.startswith('/'):
                        href = urljoin(url, href)
                    elif not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                    
                    combined_context = f"{table_context} {row_text}".strip()
                    
                    document_patterns = [
                        r'act.*\d{4}', r'circular.*\d+', r'amendment.*act', r'insurance.*act',
                        r'guideline', r'master.*direction', r'regulation.*\d+', r'notification.*\d+'
                    ]
                    
                    is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                    is_document_extension = any(ext in href.lower() for ext in document_extensions)
                    
                    # Determine link type and priority
                    if is_document_extension:
                        link_type = 'document'
                        priority = 2
                    elif is_primary_content:
                        link_type = 'content'
                        priority = 5  # Highest priority for primary content column
                    elif is_likely_document:
                        link_type = 'reference'
                        priority = 4
                    else:
                        link_type = 'reference'
                        priority = 3
                    
                    document_links.append({
                        'title': link_text,
                        'link': href,
                        'type': link_type,
                        'context': combined_context[:300],
                        'priority': priority,
                        'column_info': {
                            'is_primary_content': is_primary_content,
                            'column_index': cell_idx,
                            'row_index': row_idx
                        }
                    })
    
    # Remove duplicates while preserving the highest priority version
    seen_links = {}
    unique_document_links = []
    
    for link_info in document_links:
        link_key = (link_info['title'].strip(), link_info['link'])
        
        if link_key not in seen_links or link_info['priority'] > seen_links[link_key]['priority']:
            seen_links[link_key] = link_info
    
    unique_document_links = list(seen_links.values())
    
    # Sort by priority (highest first)
    unique_document_links.sort(key=lambda x: x['priority'], reverse=True)
    
    return unique_document_links

def extract_structured_content(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content_sections = {}
    
    news_sections = soup.find_all(['div', 'section'], class_=lambda x: x and any(
        keyword in x.lower() for keyword in ['news', 'update', 'recent', 'latest', 'whats-new']
    ))
    
    if news_sections:
        content_sections['news'] = []
        for section in news_sections:
            text = section.get_text(strip=True)
            if len(text) > 50:
                content_sections['news'].append(text)
    
    content_sections['document_links'] = extract_document_links_with_context(html_content, url)
    
    main_text = soup.get_text()
    lines = (line.strip() for line in main_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return clean_text, content_sections

def load_hardcoded_websites():
    loaded_docs = []
    
    for url in HARDCODED_WEBSITES:
        try:
            st.write(f"Loading URL: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content(html_content, url)
                
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url, "sections": sections}
                )
                loaded_docs.append(doc)
                
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                if sections.get('document_links'):
                    with st.expander(f"Document Links found from {url} (Debug Info)"):
                        st.write(f"**Total documents found:** {len(sections['document_links'])}")
                        
                        # Debug information
                        sources = {}
                        for link in sections['document_links']:
                            source = link.get('column_info', {}).get('source', 'unknown')
                            sources[source] = sources.get(source, 0) + 1
                        
                        st.write("**Link Sources:**")
                        for source, count in sources.items():
                            st.write(f"- {source}: {count} links")
                        
                        # Group by priority and type
                        content_links = [link for link in sections['document_links'] if link['type'] == 'content']
                        ref_links = [link for link in sections['document_links'] if link['type'] == 'reference']
                        pdf_docs = [link for link in sections['document_links'] if link['type'] == 'document']
                        
                        if content_links:
                            st.write("**Primary Content Links (from main content columns):**")
                            for i, link_info in enumerate(content_links[:15]):
                                priority_indicator = "🔥" if link_info.get('column_info', {}).get('is_primary_content', False) else ""
                                source_indicator = f"[{link_info.get('column_info', {}).get('source', 'unknown')}]"
                                st.write(f"{i+1}. {priority_indicator}{source_indicator} [{link_info['title'][:100]}...]({link_info['link']})")
                        
                        if ref_links:
                            st.write("**Reference Links:**")
                            for i, link_info in enumerate(ref_links[:10]):
                                source_indicator = f"[{link_info.get('column_info', {}).get('source', 'unknown')}]"
                                st.write(f"{i+1}. {source_indicator} [{link_info['title'][:100]}...]({link_info['link']})")
                        
                        if pdf_docs:
                            st.write("**Direct Document Downloads:**")
                            for i, link_info in enumerate(pdf_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        # Show raw HTML sample for debugging
                        st.write("**Raw HTML Sample (first 1000 chars):**")
                        st.code(html_content[:1000], language='html')
                else:
                    st.write(f"No document links found from {url}")
                    st.write("**Raw HTML Sample for debugging:**")
                    st.code(html_content[:1000] if html_content else "No content", language='html')
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    
    return loaded_docs
    
    for url in HARDCODED_WEBSITES:
        try:
            st.write(f"Loading URL: {url}")
            
            html_content = enhanced_web_scrape(url)
            if html_content:
                clean_text, sections = extract_structured_content(html_content, url)
                
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": url, "sections": sections}
                )
                loaded_docs.append(doc)
                
                if sections.get('news'):
                    with st.expander(f"News/Updates found from {url}"):
                        for i, news_item in enumerate(sections['news'][:3]):
                            st.write(f"**Item {i+1}:** {news_item[:200]}...")
                
                if sections.get('document_links'):
                    with st.expander(f"Document Links found from {url}"):
                        st.write(f"**Total documents found:** {len(sections['document_links'])}")
                        
                        # Group by priority and type
                        content_links = [link for link in sections['document_links'] if link['type'] == 'content']
                        ref_links = [link for link in sections['document_links'] if link['type'] == 'reference']
                        pdf_docs = [link for link in sections['document_links'] if link['type'] == 'document']
                        
                        if content_links:
                            st.write("**Primary Content Links (from main content columns):**")
                            for i, link_info in enumerate(content_links[:10]):
                                priority_indicator = "🔥" if link_info.get('column_info', {}).get('is_primary_content', False) else ""
                                st.write(f"{i+1}. {priority_indicator}[{link_info['title']}]({link_info['link']})")
                        
                        if ref_links:
                            st.write("**Reference Links:**")
                            for i, link_info in enumerate(ref_links[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if pdf_docs:
                            st.write("**Direct Document Downloads:**")
                            for i, link_info in enumerate(pdf_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.write(f"No document links found from {url}")
            
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

if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False

st.title("Web GEN-ie")

api_key = "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"

if not st.session_state['docs_loaded']:
    if st.button("Load Websites"):
        st.session_state['loaded_docs'] = load_hardcoded_websites()
        st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
        
        if api_key and st.session_state['loaded_docs']:
            with st.spinner("Processing documents..."):
                llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
                hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.
                    
                    IMPORTANT INSTRUCTIONS:
                    - ONLY answer questions that can be addressed using the provided context ONLY from the provided websites
                    - If a question is completely outside the insurance/regulatory domain or if the information is not available in the provided context, respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."
                    - Pay special attention to dates, recent updates, and chronological information
                    - When asked about "what's new" or recent developments, focus on the most recent information available
                    - Look for press releases, circulars, guidelines, and policy updates
                    - Provide specific details about new regulations, policy changes, or announcements
                    - If you find dated information, mention the specific dates
                    - When mentioning any acts, circulars, or regulations
                    
                    Based on the context provided from the insurance regulatory website(s), answer the user's question accurately and comprehensively.
                    
                    <context>
                    {context}
                    </context>
                    
                    Question: {input}
                    
                    Answer with specific details, dates. If relevant documents are mentioned, note that direct links may be available in the sources section.
                    """
                )
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
                st.write(f"Number of chunks created: {len(document_chunks)}")
                
                st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
                
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 6})
                st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
                
                st.session_state['docs_loaded'] = True
                st.success("Documents processed and ready for querying!")

st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            response = st.session_state['retrieval_chain'].invoke({"input": query})
            
            st.subheader("Response:")
            st.write(response['answer'])
            
            if not is_fallback_response(response['answer']):
                retrieved_docs = response.get('context', [])
                
                relevant_docs = get_documents_from_context_chunks(
                    retrieved_docs, query, response['answer'], max_docs=3
                )
                
                if relevant_docs:
                    st.write("\n**Related Documents (prioritized by relevance and source):**")
                    
                    # Group documents by type with new priority system
                    content_links = [doc for doc in relevant_docs if doc['type'] == 'content']
                    ref_links = [doc for doc in relevant_docs if doc['type'] == 'reference']
                    doc_files = [doc for doc in relevant_docs if doc['type'] == 'document']
                    
                    if content_links:
                        st.write("**📋 Primary Content Links (from main content columns):**")
                        for i, link_info in enumerate(content_links, 1):
                            priority_indicator = "🔥 " if link_info.get('column_info', {}).get('is_primary_content', False) else ""
                            st.write(f"{i}. {priority_indicator}[{link_info['title']}]({link_info['link']})")
                    
                    if ref_links:
                        st.write("**🔗 Reference Links:**")
                        for i, link_info in enumerate(ref_links, 1):
                            st.write(f"{i}. [{link_info['title']}]({link_info['link']})")
                    
                    if doc_files:
                        st.write("**📄 Direct Document Downloads:**")
                        for i, link_info in enumerate(doc_files, 1):
                            st.write(f"{i}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.info("No specific documents found in the context used for this answer.")
                
                st.write("\n**Information Sources:**")
                sources = set()
                for doc in retrieved_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    sources.add(source)
                
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
            else:
                st.info("ℹNo specific documents or sources are available for this query as it falls outside the current data scope.")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
