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
from difflib import SequenceMatcher

# Configuration
GROQ_API_KEY = "your_groq_api_key_here"  # Replace with your actual API key

# Specialized website collections for the four domains
WEBSITE_COLLECTIONS = {
    "IRDAI (Insurance Regulatory)": [
        "https://www.irdai.gov.in/",
        "https://www.irdai.gov.in/regulations-and-circulars",
        "https://www.irdai.gov.in/acts-and-amendments",
        "https://www.irdai.gov.in/whats-new",
        "https://www.irdai.gov.in/guidelines-and-master-directions",
        "https://www.irdai.gov.in/circulars",
        "https://www.irdai.gov.in/notifications"
    ],
    "UIDAI (Unique Identification Authority)": [
        "https://uidai.gov.in/",
        "https://uidai.gov.in/legal-framework/acts-regulations.html",
        "https://uidai.gov.in/my-aadhaar/about-your-aadhaar.html",
        "https://uidai.gov.in/ecosystem/authentication-devices-documents.html",
        "https://uidai.gov.in/contact-support/have-any-question.html",
        "https://uidai.gov.in/about-uidai/unique-identification-authority-of-india/vision-mission.html"
    ],
    "e-Gazette (Government Publications)": [
        "https://egazette.nic.in/",
        "https://egazette.nic.in/SearchGazette.aspx",
        "https://egazette.nic.in/Browse.aspx",
        "https://egazette.nic.in/WriteReadData/2024/",
        "https://egazette.nic.in/WriteReadData/2023/"
    ],
    "PMLA (Prevention of Money Laundering)": [
        "https://finmin.nic.in/divisions/fiu-ind",
        "https://fiuindia.gov.in/",
        "https://fiuindia.gov.in/rules-regulations.php",
        "https://fiuindia.gov.in/notifications.php",
        "https://fiuindia.gov.in/guidelines.php"
    ]
}

class WebScraper:
    """Specialized web scraping for the four target domains"""
    
    @staticmethod
    def get_headers():
        """Return headers optimized for government websites"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    @staticmethod
    def scrape_website(url):
        """Enhanced scraping with retry logic for government websites"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                session = requests.Session()
                session.headers.update(WebScraper.get_headers())
                
                response = session.get(url, timeout=20, verify=True)
                response.raise_for_status()
                
                if response.status_code == 200:
                    return response.text
                    
            except requests.exceptions.SSLError:
                # Retry with SSL verification disabled for problematic sites
                try:
                    response = session.get(url, timeout=20, verify=False)
                    response.raise_for_status()
                    return response.text
                except Exception as ssl_retry_error:
                    st.warning(f"SSL retry failed for {url}: {ssl_retry_error}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to scrape {url} after {max_retries} attempts: {e}")
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return None

class SpecializedDocumentExtractor:
    """Document extraction specialized for IRDAI, UIDAI, e-Gazette, and PMLA"""
    
    # Document extensions common in government/regulatory sites
    DOCUMENT_EXTENSIONS = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt', '.rtf']
    
    # Domain-specific keywords for each target website
    DOMAIN_KEYWORDS = {
        'irdai': [
            'circular', 'guideline', 'regulation', 'notification', 'master', 'direction',
            'amendment', 'insurance', 'policy', 'claim', 'premium', 'coverage',
            'solvency', 'actuarial', 'tariff', 'commission', 'broker', 'agent',
            'life insurance', 'general insurance', 'health insurance', 'reinsurance'
        ],
        'uidai': [
            'aadhaar', 'uid', 'authentication', 'biometric', 'demographic', 'otp',
            'enrollment', 'update', 'correction', 'verification', 'ekyc', 'resident',
            'identity', 'unique', 'identification', 'privacy', 'security', 'api',
            'demographic authentication', 'biometric authentication'
        ],
        'egazette': [
            'gazette', 'notification', 'order', 'act', 'rule', 'amendment', 'ordinance',
            'ministry', 'department', 'government', 'central', 'state', 'publication',
            'official', 'extraordinary', 'part', 'section', 'sub-section'
        ],
        'pmla': [
            'money laundering', 'suspicious transaction', 'cash transaction', 'fiu',
            'financial intelligence', 'aml', 'kyc', 'customer due diligence',
            'beneficial owner', 'wire transfer', 'correspondent banking',
            'suspicious activity report', 'currency transaction report', 'compliance'
        ]
    }
    
    @staticmethod
    def detect_website_domain(url):
        """Detect which of the four domains the URL belongs to"""
        url_lower = url.lower()
        
        if 'irdai.gov.in' in url_lower:
            return 'irdai'
        elif 'uidai.gov.in' in url_lower:
            return 'uidai'
        elif 'egazette.nic.in' in url_lower:
            return 'egazette'
        elif any(indicator in url_lower for indicator in ['fiu', 'finmin.nic.in', 'fiuindia']):
            return 'pmla'
        else:
            # Default fallback - try to detect from content patterns
            return 'general'
    
    @staticmethod
    def extract_document_links(html_content, url):
        """Extract document links with domain-specific logic"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        domain = SpecializedDocumentExtractor.detect_website_domain(url)
        relevant_keywords = SpecializedDocumentExtractor.DOMAIN_KEYWORDS.get(domain, [])
        
        document_links = []
        
        # Method 1: Direct PDF and document links
        document_links.extend(SpecializedDocumentExtractor._extract_direct_documents(soup, url))
        
        # Method 2: Domain-specific extraction
        if domain == 'irdai':
            document_links.extend(SpecializedDocumentExtractor._extract_irdai_documents(soup, url))
        elif domain == 'uidai':
            document_links.extend(SpecializedDocumentExtractor._extract_uidai_documents(soup, url))
        elif domain == 'egazette':
            document_links.extend(SpecializedDocumentExtractor._extract_gazette_documents(soup, url))
        elif domain == 'pmla':
            document_links.extend(SpecializedDocumentExtractor._extract_pmla_documents(soup, url))
        
        # Method 3: Generic keyword-based extraction
        document_links.extend(SpecializedDocumentExtractor._extract_keyword_documents(soup, url, relevant_keywords))
        
        return SpecializedDocumentExtractor._deduplicate_and_rank(document_links)
    
    @staticmethod
    def _extract_direct_documents(soup, url):
        """Extract direct document file links"""
        document_links = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href')
            link_text = link.get_text(strip=True)
            
            if not href or len(link_text) < 3:
                continue
                
            href = SpecializedDocumentExtractor._make_absolute_url(href, url)
            
            # Check for document extensions
            if any(ext in href.lower() for ext in SpecializedDocumentExtractor.DOCUMENT_EXTENSIONS):
                file_ext = href.split('.')[-1].upper()
                document_links.append({
                    'title': link_text or f"Document ({file_ext})",
                    'link': href,
                    'type': 'document',
                    'source': 'direct_file',
                    'confidence': 0.95
                })
        
        return document_links
    
    @staticmethod
    def _extract_irdai_documents(soup, url):
        """IRDAI-specific document extraction"""
        document_links = []
        
        # Look for IRDAI-specific patterns
        irdai_patterns = [
            r'circular\s+no\.?\s*\d+',
            r'guideline\s+no\.?\s*\d+',
            r'master\s+circular',
            r'regulation\s+\d+',
            r'notification\s+no\.?\s*\d+'
        ]
        
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            href = link.get('href')
            
            if not href:
                continue
                
            href = SpecializedDocumentExtractor._make_absolute_url(href, url)
            
            # Check IRDAI patterns
            for pattern in irdai_patterns:
                if re.search(pattern, link_text, re.IGNORECASE):
                    document_links.append({
                        'title': link.get_text(strip=True),
                        'link': href,
                        'type': 'regulation',
                        'source': 'irdai_pattern',
                        'confidence': 0.85
                    })
                    break
        
        return document_links
    
    @staticmethod
    def _extract_uidai_documents(soup, url):
        """UIDAI-specific document extraction"""
        document_links = []
        
        # UIDAI-specific patterns
        uidai_indicators = ['aadhaar', 'uid', 'authentication', 'enrollment', 'api', 'specification']
        
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            href = link.get('href')
            
            if not href or len(link_text) < 5:
                continue
                
            href = SpecializedDocumentExtractor._make_absolute_url(href, url)
            
            # Check for UIDAI-specific content
            if any(indicator in link_text for indicator in uidai_indicators):
                document_links.append({
                    'title': link.get_text(strip=True),
                    'link': href,
                    'type': 'specification',
                    'source': 'uidai_pattern',
                    'confidence': 0.80
                })
        
        return document_links
    
    @staticmethod
    def _extract_gazette_documents(soup, url):
        """e-Gazette specific document extraction"""
        document_links = []
        
        # Look for gazette-specific elements
        gazette_containers = soup.find_all(['div', 'td', 'span'], 
            class_=lambda x: x and any(indicator in str(x).lower() 
                for indicator in ['gazette', 'notification', 'publication']))
        
        for container in gazette_containers:
            links = container.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if href and len(link_text) > 10:
                    href = SpecializedDocumentExtractor._make_absolute_url(href, url)
                    document_links.append({
                        'title': link_text,
                        'link': href,
                        'type': 'gazette',
                        'source': 'gazette_container',
                        'confidence': 0.75
                    })
        
        return document_links
    
    @staticmethod
    def _extract_pmla_documents(soup, url):
        """PMLA/FIU specific document extraction"""
        document_links = []
        
        pmla_indicators = ['suspicious', 'transaction', 'report', 'guidance', 'compliance', 'aml', 'kyc']
        
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            href = link.get('href')
            
            if not href or len(link_text) < 8:
                continue
                
            href = SpecializedDocumentExtractor._make_absolute_url(href, url)
            
            # Check for PMLA-specific content
            if any(indicator in link_text for indicator in pmla_indicators):
                document_links.append({
                    'title': link.get_text(strip=True),
                    'link': href,
                    'type': 'compliance',
                    'source': 'pmla_pattern',
                    'confidence': 0.80
                })
        
        return document_links
    
    @staticmethod
    def _extract_keyword_documents(soup, url, keywords):
        """Generic keyword-based extraction"""
        document_links = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            href = link.get('href')
            
            if not href or len(link_text) < 5:
                continue
                
            href = SpecializedDocumentExtractor._make_absolute_url(href, url)
            
            # Check for keyword matches
            matching_keywords = [kw for kw in keywords if kw in link_text]
            
            if matching_keywords:
                document_links.append({
                    'title': link.get_text(strip=True),
                    'link': href,
                    'type': 'reference',
                    'source': 'keyword_match',
                    'matching_keywords': matching_keywords,
                    'confidence': min(0.7, len(matching_keywords) * 0.2 + 0.3)
                })
        
        return document_links
    
    @staticmethod
    def _make_absolute_url(href, base_url):
        """Convert relative URLs to absolute URLs"""
        if href.startswith('//'):
            return f"https:{href}"
        elif href.startswith('/'):
            return urljoin(base_url, href)
        elif not href.startswith(('http://', 'https://')):
            return urljoin(base_url, href)
        return href
    
    @staticmethod
    def _deduplicate_and_rank(document_links):
        """Remove duplicates and rank documents"""
        seen_urls = set()
        unique_links = []
        
        for link in document_links:
            if link['link'] not in seen_urls:
                seen_urls.add(link['link'])
                unique_links.append(link)
        
        # Sort by confidence and type priority
        type_priority = {'document': 4, 'regulation': 3, 'specification': 3, 'gazette': 2, 'compliance': 2, 'reference': 1}
        
        unique_links.sort(
            key=lambda x: (x['confidence'], type_priority.get(x['type'], 0)), 
            reverse=True
        )
        
        return unique_links[:20]  # Limit to top 20 results

class ContentProcessor:
    """Specialized content processing for the four domains"""
    
    @staticmethod
    def extract_structured_content(html_content, url):
        """Extract and structure content based on domain"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unnecessary elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()
        
        domain = SpecializedDocumentExtractor.detect_website_domain(url)
        
        content_sections = {
            'domain': domain,
            'document_links': SpecializedDocumentExtractor.extract_document_links(html_content, url)
        }
        
        # Domain-specific content extraction
        if domain == 'irdai':
            content_sections.update(ContentProcessor._extract_irdai_content(soup))
        elif domain == 'uidai':
            content_sections.update(ContentProcessor._extract_uidai_content(soup))
        elif domain == 'egazette':
            content_sections.update(ContentProcessor._extract_gazette_content(soup))
        elif domain == 'pmla':
            content_sections.update(ContentProcessor._extract_pmla_content(soup))
        
        # Extract main text content
        main_text = soup.get_text()
        clean_text = ContentProcessor._clean_text(main_text)
        
        return clean_text, content_sections
    
    @staticmethod
    def _extract_irdai_content(soup):
        """Extract IRDAI-specific content sections"""
        content = {}
        
        # Look for circulars and notifications
        circular_sections = soup.find_all(text=re.compile(r'circular|notification', re.IGNORECASE))
        content['circulars'] = [elem.parent.get_text(strip=True) for elem in circular_sections[:5]]
        
        # Look for recent updates
        update_sections = soup.find_all(['div', 'section'], 
            class_=lambda x: x and 'update' in str(x).lower())
        content['updates'] = [section.get_text(strip=True) for section in update_sections[:3]]
        
        return content
    
    @staticmethod
    def _extract_uidai_content(soup):
        """Extract UIDAI-specific content sections"""
        content = {}
        
        # Look for Aadhaar-related sections
        aadhaar_sections = soup.find_all(text=re.compile(r'aadhaar|authentication', re.IGNORECASE))
        content['aadhaar_info'] = [elem.parent.get_text(strip=True) for elem in aadhaar_sections[:5]]
        
        return content
    
    @staticmethod
    def _extract_gazette_content(soup):
        """Extract e-Gazette specific content"""
        content = {}
        
        # Look for publication information
        pub_sections = soup.find_all(['div', 'td'], 
            text=re.compile(r'publication|extraordinary|part', re.IGNORECASE))
        content['publications'] = [section.get_text(strip=True) for section in pub_sections[:5]]
        
        return content
    
    @staticmethod
    def _extract_pmla_content(soup):
        """Extract PMLA/FIU specific content"""
        content = {}
        
        # Look for compliance and reporting information
        compliance_sections = soup.find_all(text=re.compile(r'compliance|report|suspicious', re.IGNORECASE))
        content['compliance_info'] = [elem.parent.get_text(strip=True) for elem in compliance_sections[:5]]
        
        return content
    
    @staticmethod
    def _clean_text(text):
        """Clean and format text content"""
        # Remove extra whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned = '\n'.join(chunk for chunk in chunks if chunk and len(chunk) > 3)
        
        # Remove very long lines (likely to be data/code)
        lines = [line for line in cleaned.split('\n') if len(line) < 500]
        
        return '\n'.join(lines)

class SpecializedProcessor:
    """Main processor for the four specialized domains"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = None
        self.embedding = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize LLM and embedding models"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key, 
                model_name='llama3-70b-8192', 
                temperature=0.1,
                max_tokens=2048
            )
            self.embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            st.error(f"Failed to initialize models: {e}")
    
    def get_specialized_prompt_template(self):
        """Get specialized prompt template for the four domains"""
        return ChatPromptTemplate.from_template(
            """
            You are a specialized AI assistant for Indian regulatory and government websites, 
            specifically IRDAI (Insurance), UIDAI (Aadhaar), e-Gazette (Government Publications), 
            and PMLA/FIU (Financial Intelligence).
            
            DOMAIN-SPECIFIC INSTRUCTIONS:
            
            For IRDAI queries:
            - Focus on insurance regulations, circulars, guidelines, and notifications
            - Mention specific circular numbers, dates, and regulatory changes
            - Emphasize compliance requirements and industry impacts
            
            For UIDAI queries:
            - Focus on Aadhaar authentication, enrollment, and API specifications
            - Mention security, privacy measures, and technical requirements
            - Emphasize resident services and authentication processes
            
            For e-Gazette queries:
            - Focus on government notifications, acts, rules, and amendments
            - Mention publication dates, gazette numbers, and official classifications
            - Emphasize legal and regulatory changes
            
            For PMLA/FIU queries:
            - Focus on anti-money laundering, suspicious transaction reporting
            - Mention compliance requirements, reporting formats, and deadlines
            - Emphasize financial intelligence and regulatory compliance
            
            GENERAL INSTRUCTIONS:
            - Always provide specific dates, numbers, and official references when available
            - Mention if documents or detailed guidelines are available in the source materials
            - Structure your response with clear headings and bullet points when appropriate
            - If the query spans multiple domains, address each relevant domain separately
            
            Context from the websites:
            <context>
            {context}
            </context>
            
            User Question: {input}
            
            Provide a comprehensive, domain-specific answer with official references and specific details.
            """
        )
    
    def process_documents(self, documents):
        """Process documents and create specialized retrieval chain"""
        if not documents:
            return None, 0
            
        try:
            # Enhanced text splitting for regulatory documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            document_chunks = text_splitter.split_documents(documents)
            
            if not document_chunks:
                return None, 0
            
            # Create vector database
            vector_db = FAISS.from_documents(document_chunks, self.embedding)
            
            # Create specialized chains
            prompt = self.get_specialized_prompt_template()
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            return retrieval_chain, len(document_chunks)
            
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return None, 0

class StreamlitUI:
    """Specialized Streamlit UI for the four domains"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'loaded_docs' not in st.session_state:
            st.session_state['loaded_docs'] = []
        if 'retrieval_chain' not in st.session_state:
            st.session_state['retrieval_chain'] = None
        if 'processor' not in st.session_state:
            st.session_state['processor'] = SpecializedProcessor(GROQ_API_KEY)
        if 'current_websites' not in st.session_state:
            st.session_state['current_websites'] = []
        if 'document_links' not in st.session_state:
            st.session_state['document_links'] = []
    
    def render_header(self):
        """Render the application header"""
        st.title("🏛️ Indian Regulatory Intelligence System")
        st.markdown("""
        **Specialized AI Assistant for Indian Regulatory Websites**
        
        📋 **Supported Domains:**
        - **IRDAI**: Insurance regulations, circulars, and guidelines
        - **UIDAI**: Aadhaar authentication and specifications  
        - **e-Gazette**: Government notifications and publications
        - **PMLA/FIU**: Anti-money laundering and financial intelligence
        """)
        
        st.divider()
    
    def render_website_selection(self):
        """Render website selection interface"""
        st.subheader("📡 Select Domain & Load Content")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_collection = st.selectbox(
                "Choose Regulatory Domain:",
                options=list(WEBSITE_COLLECTIONS.keys()),
                help="Select the regulatory domain you want to analyze"
            )
        
        with col2:
            if st.button("🔄 Load Domain Content", type="primary"):
                self.load_website_collection(selected_collection)
        
        # Show custom URL input
        with st.expander("➕ Add Custom URLs"):
            custom_url = st.text_input(
                "Enter additional URL:",
                placeholder="https://example.gov.in/page"
            )
            if st.button("Add URL") and custom_url:
                self.load_single_website(custom_url)
    
    def load_website_collection(self, collection_name):
        """Load a complete website collection"""
        if collection_name not in WEBSITE_COLLECTIONS:
            st.error("Invalid collection selected")
            return
        
        urls = WEBSITE_COLLECTIONS[collection_name]
        
        with st.spinner(f"Loading {collection_name} content..."):
            progress_bar = st.progress(0)
            all_documents = []
            all_doc_links = []
            
            for i, url in enumerate(urls):
                st.info(f"Processing: {url}")
                
                # Scrape website
                html_content = WebScraper.scrape_website(url)
                if html_content:
                    # Process content
                    clean_text, content_sections = ContentProcessor.extract_structured_content(html_content, url)
                    
                    if clean_text and len(clean_text.strip()) > 100:
                        # Create document
                        doc = Document(
                            page_content=clean_text,
                            metadata={
                                'source': url,
                                'domain': content_sections.get('domain', 'unknown'),
                                'collection': collection_name
                            }
                        )
                        all_documents.append(doc)
                        
                        # Collect document links
                        if 'document_links' in content_sections:
                            all_doc_links.extend(content_sections['document_links'])
                
                progress_bar.progress((i + 1) / len(urls))
            
            if all_documents:
                # Process documents
                retrieval_chain, chunk_count = st.session_state['processor'].process_documents(all_documents)
                
                if retrieval_chain:
                    st.session_state['retrieval_chain'] = retrieval_chain
                    st.session_state['loaded_docs'] = all_documents
                    st.session_state['current_websites'] = urls
                    st.session_state['document_links'] = all_doc_links[:15]  # Limit to top 15
                    
                    st.success(f"✅ Successfully loaded {len(all_documents)} pages with {chunk_count} text chunks")
                    
                    # Show document links if available
                    if all_doc_links:
                        with st.expander(f"📄 Found {len(all_doc_links)} Documents & References"):
                            for doc_link in all_doc_links[:10]:
                                st.markdown(f"**{doc_link['title']}**")
                                st.markdown(f"🔗 [{doc_link['link']}]({doc_link['link']})")
                                st.markdown(f"*Type: {doc_link['type']} | Confidence: {doc_link['confidence']:.2f}*")
                                st.divider()
                else:
                    st.error("Failed")
