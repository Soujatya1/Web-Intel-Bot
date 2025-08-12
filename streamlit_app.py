import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from urllib.parse import urljoin, urlparse
from collections import Counter

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts",
                      "https://irdai.gov.in/home",
                      "https://irdai.gov.in/rules",
                      "https://irdai.gov.in/consolidated-gazette-notified-regulations",
                      "https://irdai.gov.in/updated-regulations",
                      "https://irdai.gov.in/notifications",
                      "https://irdai.gov.in/circulars",
                      "https://irdai.gov.in/guidelines",
                      "https://irdai.gov.in/guidelines?p_p_id=com_irdai_document_media_IRDAIDocumentMediaPortlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_com_irdai_document_media_IRDAIDocumentMediaPortlet_delta=20&_com_irdai_document_media_IRDAIDocumentMediaPortlet_resetCur=false&_com_irdai_document_media_IRDAIDocumentMediaPortlet_cur=2"
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

def smart_document_filter(document_links, query, ai_response, max_docs=2):

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
            if mentioned_years.intersection(title_years):
                confidence_score += 20
                match_reasons.append(f"Year match: {mentioned_years.intersection(title_years)}")
        
        regulatory_types = ['act', 'regulation', 'circular', 'guideline', 'amendment', 'notification', 'policy', 'rule', 'framework', 'directive']
        ai_mentions_reg_type = any(reg_type in ai_response_lower for reg_type in regulatory_types)
        title_has_reg_type = any(reg_type in title_lower for reg_type in regulatory_types)
        
        if ai_mentions_reg_type and title_has_reg_type:
            matching_reg_types = [reg_type for reg_type in regulatory_types 
                                if reg_type in ai_response_lower and reg_type in title_lower]
            if matching_reg_types:
                confidence_score += 25
                match_reasons.append(f"Regulatory type match: {matching_reg_types}")
        
        domain_terms = ['insurance', 'aadhaar', 'uidai', 'irdai', 'pmla', 'licensing', 'compliance']
        ai_domain_terms = [term for term in domain_terms if term in ai_response_lower]
        title_domain_terms = [term for term in domain_terms if term in title_lower]
        
        matching_domain_terms = set(ai_domain_terms).intersection(set(title_domain_terms))
        if matching_domain_terms:
            confidence_score += 20
            match_reasons.append(f"Domain terms: {list(matching_domain_terms)}")
        
        if confidence_score >= 60:
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

def extract_document_links(html_content, url):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    document_links = []
    
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
        
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive',
                           'insurance', 'aadhaar', 'compliance', 'licensing']
        
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if has_doc_keywords and len(link_text) > 5:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'content'
            })
    
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
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
                        
                        document_patterns = [
                            r'act.*\d{4}',
                            r'circular.*\d+',
                            r'amendment.*act',
                            r'insurance.*act',
                            r'guideline',
                            r'master.*direction',
                            r'regulation.*\d+',
                            r'aadhaar.*act'
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        
                        if is_likely_document:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'reference'
                            })
    
    content_sections = soup.find_all(['div', 'section', 'article'])
    for section in content_sections:
        section_text = section.get_text().lower()
        if any(keyword in section_text for keyword in ['act', 'circular', 'regulation', 'guideline']):
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
                
                if 10 < len(link_text) < 200:
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy', 'aadhaar']
                    if any(keyword in link_text.lower() for keyword in document_keywords):
                        document_links.append({
                            'title': link_text,
                            'link': href,
                            'type': 'reference'
                        })
    
    seen_links = set()
    unique_document_links = []
    for link_info in document_links:
        link_key = (link_info['title'], link_info['link'])
        if link_key not in seen_links:
            seen_links.add(link_key)
            unique_document_links.append(link_info)
    
    unique_document_links.sort(key=lambda x: (x['type'] != 'content', x['title']))
    
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
    
    content_sections['document_links'] = extract_document_links(html_content, url)
    
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
                with st.spinner("Processing documents..."):
                    try:
                        llm = AzureChatOpenAI(
                            azure_endpoint=azure_endpoint,
                            api_key=api_key,
                            azure_deployment=deployment_name,
                            api_version=api_version,
                            temperature=0.0
                        )
                        st.session_state['llm'] = llm
                        
                        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        
                        prompt = ChatPromptTemplate.from_template(
                              """
    You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.
    
    IMPORTANT INSTRUCTIONS:
    - Answer questions using the provided context from the regulatory websites
    - Pay special attention to dates, recent updates, and chronological information
    - When asked about "what's new" or recent developments, focus on the most recent information available
    - Look for press releases, circulars, guidelines, and policy updates
    - Provide specific details about new regulations, policy changes, or announcements
    - If you find dated information, mention the specific dates
    - When a question like, "Latest guidelines under IRDAI" is asked, follow the 'Last Updated' date and as per the same, respond to the query
    - When mentioning any acts, circulars, or regulations, try to reference the available document links
    - If you find any PII data in the question (e.g., PAN card no., AADHAAR no., DOB, Address), respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on, as your query contains PII data"

    RESPONSE APPROACH:
    - If the provided context contains information that can help answer the user's question (even partially), use that information to provide a helpful response
    - Only use the fallback response below if the context is completely unrelated to the question and contains no relevant information whatsoever
    - Even if you can only provide partial information or related information, do so rather than using the fallback
    
    FALLBACK RESPONSE (use ONLY when context is completely irrelevant):
    "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Provide a comprehensive answer using the available context. Be helpful and informative even if the context only partially addresses the question.
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
                    
                    except Exception as e:
                        st.error(f"Error initializing Azure OpenAI: {e}")
                        st.error("Please check your Azure OpenAI configuration and try again.")

st.subheader("Ask Questions")
query = st.text_input("Enter your query:", value="What are the recent Insurance Acts and amendments?")

if st.button("Get Answer", disabled=not config_complete) and query:
    if not config_complete:
        st.error("Please complete the Azure OpenAI configuration first.")
    elif st.session_state['retrieval_chain']:
        with st.spinner("Searching and generating answer..."):
            try:
                response = st.session_state['retrieval_chain'].invoke({"input": query})
                
                st.subheader("Response:")
                st.write(response['answer'])
                
                if not is_fallback_response(response['answer']):
                    retrieved_docs = response.get('context', [])
                    
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
                            max_docs=3
                        )
                        
                        if relevant_docs:
                            st.write("\n**📄 Relevant Document Links**")
                            for i, link_info in enumerate(relevant_docs):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        else:
                            st.info("No high-confidence document links found that directly match the AI response content.")
                    
                    st.write("\n**📍 Information Sources:**")
                    sources = set()
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
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
