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

HARDCODED_WEBSITES = ["https://irdai.gov.in/acts", "https://irdai.gov.in/rules",
                      "https://irdai.gov.in/consolidated-gazette-notified-regulations", "https://irdai.gov.in/notifications","https://irdai.gov.in/circulars","https://irdai.gov.in/orders1","https://irdai.gov.in/exposure-drafts","https://irdai.gov.in/programmes-to-advance-understanding-of-rti","https://irdai.gov.in/cic-orders","https://irdai.gov.in/antimoney-laundering","https://irdai.gov.in/other-communication","https://irdai.gov.in/directory-of-employees","https://irdai.gov.in/warnings-and-penalties",
            "https://uidai.gov.in/en/","https://uidai.gov.in/en/about-uidai/legal-framework.html","https://uidai.gov.in/en/about-uidai/legal-framework/rules.html","https://uidai.gov.in/en/about-uidai/legal-framework/notifications.html","https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html","https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html","https://uidai.gov.in/en/about-uidai/legal-framework/judgements.html","https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation","https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules","https://enforcementdirectorate.gov.in/pmla", "https://enforcementdirectorate.gov.in/pmla?page=1", "https://enforcementdirectorate.gov.in/fema", "https://enforcementdirectorate.gov.in/fema?page=1", "https://enforcementdirectorate.gov.in/fema?page=2", "https://enforcementdirectorate.gov.in/fema?page=3", "https://enforcementdirectorate.gov.in/bns","https://enforcementdirectorate.gov.in/bnss","https://enforcementdirectorate.gov.in/bsa"
                     ]

def is_query_domain_relevant(query, domain_keywords):
    query_lower = query.lower()
    
    insurance_keywords = [
        'insurance', 'irdai', 'policy', 'premium', 'claim', 'regulation', 'act', 
        'circular', 'guideline', 'amendment', 'notification', 'regulatory', 
        'coverage', 'underwriting', 'reinsurance', 'broker', 'agent', 'policyholder',
        'solvency', 'capital', 'reserve', 'compliance', 'audit', 'financial',
        'motor insurance', 'health insurance', 'life insurance', 'general insurance',
        'micro insurance', 'crop insurance', 'marine insurance', 'fire insurance'
    ]
    
    domain_match_count = sum(1 for keyword in insurance_keywords if keyword in query_lower)
    
    if domain_match_count == 0:
        generic_patterns = [
            r'\bwho is\b', r'\bwhat is.*(?:actor|movie|film|celebrity|person)\b',
            r'\btell me about.*(?:person|people|celebrity|actor|actress)\b',
            r'\b(?:biography|bio|personal life|career|films|movies)\b'
        ]
        
        for pattern in generic_patterns:
            if re.search(pattern, query_lower):
                return False
        
        person_indicators = ['who is', 'biography', 'born in', 'age of', 'actor', 'actress', 'celebrity']
        if any(indicator in query_lower for indicator in person_indicators):
            return False
    
    return domain_match_count > 0

def assess_answer_quality(answer, query):
    answer_lower = answer.lower()
    query_lower = query.lower()
    
    fallback_indicators = [
        "fall outside the scope",
        "not found in the provided context",
        "no specific information",
        "context provided does not contain",
        "information is not available"
    ]
    
    has_fallback = any(indicator in answer_lower for indicator in fallback_indicators)
    
    domain_keywords = [
        'insurance', 'irdai', 'act', 'regulation', 'policy', 'circular',
        'guideline', 'amendment', 'compliance', 'regulatory'
    ]
    
    domain_content_count = sum(1 for keyword in domain_keywords if keyword in answer_lower)
    
    if has_fallback and domain_content_count < 2:
        return False
    
    person_query_indicators = ['who is', 'biography', 'tell me about']
    if any(indicator in query_lower for indicator in person_query_indicators):
        if domain_content_count > 0:
            return False
    
    return True

def filter_relevant_documents(document_links, query, ai_response):
    
    if not is_query_domain_relevant(query, []):
        return []
    
    if not assess_answer_quality(ai_response, query):
        return []
    
    response_lower = ai_response.lower()
    query_lower = query.lower()
    
    response_entities = set()
    
    meaningful_words = re.findall(r'\b[A-Za-z]{4,}\b', ai_response)
    for word in meaningful_words:
        word_lower = word.lower()
        if word_lower not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 
                             'would', 'could', 'should', 'these', 'those', 'which', 'where', 
                             'when', 'what', 'such', 'through', 'under', 'over', 'also', 'information',
                             'provided', 'context', 'question', 'details', 'scope', 'trained']:
            response_entities.add(word_lower)
    
    act_patterns = re.findall(r'[A-Za-z\s]+act\s*\(?(\d{4})?\)?', ai_response, re.IGNORECASE)
    year_patterns = re.findall(r'\b(19|20)\d{2}\b', ai_response)
    
    response_phrases = set()
    sentences = re.split(r'[.!?;]', ai_response)
    for sentence in sentences:
        words = sentence.strip().split()
        for i in range(len(words) - 1):
            if i + 1 < len(words):
                phrase = f"{words[i]} {words[i+1]}".lower().strip()
                if len(phrase) > 6 and not any(stop_word in phrase for stop_word in ['the ', 'and ', 'for ', 'are ', 'this ', 'that ', 'information ', 'context ', 'provided ']):
                    response_phrases.add(phrase)
    
    matched_docs = []
    
    for doc_link in document_links:
        title_lower = doc_link['title'].lower()
        match_score = 0
        match_reasons = []
        
        doc_words = set(re.findall(r'\b[A-Za-z]{4,}\b', title_lower))
        entity_matches = response_entities.intersection(doc_words)
        if entity_matches:
            meaningful_entities = [e for e in entity_matches if e not in ['information', 'context', 'provided', 'question', 'details']]
            if meaningful_entities:
                match_score += len(meaningful_entities) * 10
                match_reasons.append(f"Entity matches: {meaningful_entities}")
        
        phrase_matches = []
        for phrase in response_phrases:
            if phrase in title_lower and len(phrase) > 8:
                match_score += 15
                phrase_matches.append(phrase)
        
        if phrase_matches:
            match_reasons.append(f"Phrase matches: {phrase_matches}")
        
        if year_patterns:
            doc_years = re.findall(r'\b(19|20)\d{2}\b', title_lower)
            year_matches = set(year_patterns).intersection(set(doc_years))
            if year_matches:
                match_score += 20
                match_reasons.append(f"Year matches: {list(year_matches)}")
        
        domain_terms_in_response = ['act', 'regulation', 'circular', 'amendment', 'guideline', 'insurance']
        domain_terms_in_title = ['act', 'regulation', 'circular', 'amendment', 'guideline', 'insurance']
        
        response_has_domain = any(term in response_lower for term in domain_terms_in_response)
        title_has_domain = any(term in title_lower for term in domain_terms_in_title)
        
        if response_has_domain and title_has_domain:
            match_score += 8
            match_reasons.append("Document type match with response")
        
        if match_score > 0:
            similarity = SequenceMatcher(None, title_lower, response_lower).ratio()
            if similarity > 0.3:
                match_score += similarity * 10
                match_reasons.append(f"High semantic similarity: {similarity:.2f}")
        
        if match_score >= 35:
            doc_link['relevance_score'] = match_score
            doc_link['match_reasons'] = match_reasons
            matched_docs.append(doc_link)
    
    matched_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return matched_docs

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
    
    document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
    
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
        
        is_document_link = any(ext in href.lower() for ext in document_extensions)
        
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 'amendment', 
                           'notification', 'order', 'policy', 'master', 'framework', 'directive']
        has_doc_keywords = any(keyword in link_text.lower() for keyword in document_keywords)
        
        if is_document_link or has_doc_keywords:
            document_links.append({
                'title': link_text,
                'link': href,
                'type': 'document' if is_document_link else 'content'
            })
    
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
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
                            r'regulation.*\d+'
                        ]
                        
                        is_likely_document = any(re.search(pattern, link_text.lower()) for pattern in document_patterns)
                        is_document_extension = any(ext in href.lower() for ext in document_extensions)
                        
                        if is_likely_document or is_document_extension:
                            document_links.append({
                                'title': link_text,
                                'link': href,
                                'type': 'document' if is_document_extension else 'reference'
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
                
                if len(link_text) > 10 and len(link_text) < 200:
                    document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                                       'amendment', 'notification', 'insurance', 'policy']
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
    
    unique_document_links.sort(key=lambda x: (x['type'] != 'document', x['title']))
    
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
                        st.write(f"**Total documents found:** {len(sections['document_links'])}")
                        
                        pdf_docs = [link for link in sections['document_links'] if link['type'] == 'document']
                        ref_docs = [link for link in sections['document_links'] if link['type'] in ['reference', 'content']]
                        
                        if pdf_docs:
                            st.write("**Direct Document Downloads:**")
                            for i, link_info in enumerate(pdf_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                        
                        if ref_docs:
                            st.write("**Document References:**")
                            for i, link_info in enumerate(ref_docs[:10]):
                                st.write(f"{i+1}. [{link_info['title']}]({link_info['link']})")
                else:
                    st.write(f"No document links found from {url}")
            
            st.success(f"Successfully loaded content from {url}")
            
        except Exception as e:
            st.error(f"Error loading {url}: {e}")
    
    return loaded_docs

if 'loaded_docs' not in st.session_state:
    st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
    st.session_state['retrieval_chain'] = None
if 'docs_loaded' not in st.session_state:
    st.session_state['docs_loaded'] = False

# Streamlit UI
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
                    You are a website expert assistant specializing in understanding and answering questions asked from IRDAI, UIDAI, PMLA and egazette websites.
                    
                    IMPORTANT INSTRUCTIONS:
                    - Only answer questions related to insurance, regulations, acts, policies, and IRDAI matters
                    - If a question is completely outside the insurance/regulatory domain (like asking about celebrities, movies, general knowledge), respond with: "Thank you for your question about insurance/regulatory matters. While the specific details you've asked for aren't available in my current dataset, I can provide related information that might be helpful based on the available regulatory documents and guidelines."
                    - Pay special attention to dates, recent updates, and chronological information
                    - When asked about "what's new" or recent developments, focus on the most recent information available
                    - Look for press releases, circulars, guidelines, and policy updates
                    - Provide specific details about new regulations, policy changes, or announcements
                    - If you find dated information, mention the specific dates
                    - When mentioning any acts, circulars, or regulations, try to reference the available document links
                    
                    Based on the context provided from the insurance regulatory website(s), answer the user's question accurately and comprehensively.

                    
                    
                    <context>
                    {context}
                    </context>
                    
                    Question: {input}
                    
                    Answer with specific details, dates, and references where available. If relevant documents are mentioned, note that direct links may be available in the sources section.
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
                
                # Create chains
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
            
            show_additional_info = (
                is_query_domain_relevant(query, []) and 
                assess_answer_quality(response['answer'], query)
            )
            
            if show_additional_info:
                retrieved_docs = response.get('context', [])
                all_document_links = []
                
                for doc in retrieved_docs:
                    if 'sections' in doc.metadata and 'document_links' in doc.metadata['sections']:
                        for link_info in doc.metadata['sections']['document_links']:
                            if link_info not in all_document_links:
                                all_document_links.append(link_info)
                
                relevant_docs = filter_relevant_documents(all_document_links, query, response['answer']) if all_document_links else []
                
                if relevant_docs:
                    st.write("\n**📄 Related Documents:**")
                    for i, link_info in enumerate(relevant_docs[:3]):
                        st.write(f"{i+1}. [{link_info['title']}]({link_info['link']}) (Relevance: {link_info['relevance_score']:.1f})")
                
                st.write("\n**📍 Information Sources:**")
                sources = set()
                for doc in retrieved_docs:
                    source = doc.metadata.get('source', 'Unknown')
                    sources.add(source)
                
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
            else:
                st.write("\n*Note: For insurance and regulatory queries, additional document links and sources will be provided.*")
    else:
        st.warning("Please load websites first by clicking the 'Load Websites' button.")
