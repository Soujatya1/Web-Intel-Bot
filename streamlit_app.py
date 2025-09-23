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
from sklearn.metrics.pairwise import cosine_similarity
from langchain_experimental.text_splitter import SemanticChunker

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

SYSTEM_PROMPT_TEMPLATE = """
You are a website expert assistant specializing in understanding and answering questions from IRDAI, UIDAI, PMLA and egazette websites.

Answer the question based ONLY on the provided context information.

IMPORTANT INSTRUCTIONS:
- Each chunk in the context starts with "Source URL:" followed by the source website and relevant document links
- Always reference the source URL provided at the beginning of each chunk in your answers
- Answer questions using the provided context from the websites
- Pay special attention to dates, recent updates, and chronological information
- Always Give response in chronological order according to date.
- When asked about "what's new" or recent developments, focus on the most recent information available
- Look for press releases, circulars, guidelines, and policy updates
- Provide specific details about new regulations, policy changes, or announcements
- If you find dated information, mention the specific dates
- When a question like, "Latest guidelines under IRDAI" is asked, follow the 'Last Updated' date and as per the same, respond to the query
- When mentioning any acts, circulars, or regulations, try to reference the available document links that are provided in the context
- If you find any PII data in the question (e.g., PAN card no., AADHAAR no., DOB, Address), respond with: "Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on, as your query contains PII data"
- Use the document links provided in the context to give more comprehensive answers with proper references
- Always include the source URL in your answer for credibility and reference

FALLBACK RESPONSE (use ONLY when context is completely irrelevant):
"Thank you for your question. The details you've asked for fall outside the scope of the data I've been trained on. However, I've gathered information that closely aligns with your query and may address your needs. Please review the provided details below to ensure they align with your expectations."

Context: {context}

Question: {input}

Provide a comprehensive answer using the available context, including relevant document links and source URLs when available. Be helpful and informative even if the context only partially addresses the question.
"""

QUERY_ENHANCEMENT_PROMPT = """
You are a precise keyword extractor for document retrieval. Your job is to extract ONLY the most essential keywords that are directly related to what the user is asking for.

STRICT RULES:
1. DO NOT add domain context unless explicitly mentioned in the query
2. DO NOT add synonyms or related terms unless they are essential
3. FOCUS on the exact words and concepts the user mentioned
4. If user asks about "recent X", include both "recent" and "X"
5. If user asks about specific document types (gazette, circular, etc.), keep those exact terms
6. Extract the core nouns, adjectives, and specific identifiers from the query
7. Limit to 4-6 keywords maximum

User Query: "{question}"

Extract ONLY the most directly relevant keywords from this query. Do not expand with domain knowledge.

Keywords (comma-separated, max 6):"""

RELEVANCE_SCORE_THRESHOLD = 0.3

def extract_keywords_with_llm(query, llm):
    """Extract keywords from query using LLM for better retrieval"""
    try:
        # Create keyword extraction prompt
        keyword_prompt = ChatPromptTemplate.from_template(QUERY_ENHANCEMENT_PROMPT)
        keyword_chain = keyword_prompt | llm
        
        # Get keywords from LLM
        response = keyword_chain.invoke({"question": query})
        
        # Extract keywords from response
        if hasattr(response, 'content'):
            keywords_text = response.content.strip()
        else:
            keywords_text = str(response).strip()
        
        # Parse keywords
        keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
        
        # Limit to reasonable number
        keywords = keywords[:12]
        
        return keywords, keywords_text
        
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}. Using fallback method.")
        # Fallback to simple keyword extraction
        return extract_keywords_fallback(query), "Fallback method used"

def extract_keywords_fallback(query):
    """Fallback keyword extraction method"""
    import re
    
    # Domain-specific terms to prioritize
    domain_terms = [
        'irdai', 'insurance', 'uidai', 'aadhaar', 'pmla', 'fema', 
        'act', 'regulation', 'circular', 'guideline', 'amendment',
        'notification', 'policy', 'compliance', 'recent', 'latest',
        'new', 'updated', '2024', '2025'
    ]
    
    # Extract words from query
    words = re.findall(r'\b\w+\b', query.lower())
    
    # Prioritize domain terms
    keywords = []
    for term in domain_terms:
        if term in query.lower():
            keywords.append(term)
    
    # Add other significant words
    for word in words:
        if len(word) > 3 and word not in keywords:
            keywords.append(word)
    
    return keywords[:10]

def enhanced_retrieval_with_keywords(query, vector_db, hf_embedding, llm, k=6):
    """Enhanced retrieval using LLM-extracted keywords"""
    try:
        # Step 1: Extract keywords using LLM
        st.info("🔍 Extracting keywords using LLM...")
        keywords, keywords_text = extract_keywords_with_llm(query, llm)
        
        # Display extracted keywords
        with st.expander("📋 Extracted Keywords for Retrieval"):
            st.write("**Keywords extracted by LLM:**")
            st.code(keywords_text)
            st.write("**Processed keywords list:**")
            st.write(", ".join(f"`{kw}`" for kw in keywords))
        
        # Step 2: Create enhanced search queries
        enhanced_queries = []
        
        # Original query
        enhanced_queries.append(query)
        
        # Keywords-only query
        keywords_query = " ".join(keywords)
        enhanced_queries.append(keywords_query)
        
        # Combined query with weights
        combined_query = f"{query} {' '.join(keywords[:6])}"
        enhanced_queries.append(combined_query)
        
        # Step 3: Retrieve documents for each query
        all_retrieved_docs = []
        doc_scores = {}
        
        st.info("🔎 Performing enhanced retrieval...")
        
        for i, search_query in enumerate(enhanced_queries):
            try:
                # Retrieve documents
                docs = vector_db.similarity_search_with_score(search_query, k=k*2)
                
                # Weight documents based on query type
                query_weight = [1.0, 0.8, 0.9][i]  # Original, keywords-only, combined
                
                for doc, score in docs:
                    doc_id = (doc.page_content[:100], doc.metadata.get('source', ''))
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {'doc': doc, 'max_score': 0, 'query_matches': []}
                    
                    weighted_score = (1 / (score + 1e-6)) * query_weight  # Convert distance to similarity
                    doc_scores[doc_id]['max_score'] = max(doc_scores[doc_id]['max_score'], weighted_score)
                    doc_scores[doc_id]['query_matches'].append(f"Query {i+1}: {weighted_score:.3f}")
                    
            except Exception as e:
                st.warning(f"Retrieval failed for query {i+1}: {e}")
                continue
        
        # Step 4: Re-rank and select top documents
        if doc_scores:
            # Sort by combined score
            ranked_docs = sorted(
                doc_scores.values(), 
                key=lambda x: x['max_score'], 
                reverse=True
            )
            
            # Apply additional keyword-based filtering
            final_docs = []
            for doc_info in ranked_docs[:k*2]:
                doc = doc_info['doc']
                
                # Calculate keyword overlap score
                doc_text_lower = doc.page_content.lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in doc_text_lower)
                keyword_score = keyword_matches / len(keywords) if keywords else 0
                
                # Combine with retrieval score
                final_score = doc_info['max_score'] * (1 + keyword_score * 0.5)
                
                final_docs.append({
                    'doc': doc,
                    'final_score': final_score,
                    'keyword_matches': keyword_matches,
                    'query_matches': doc_info['query_matches']
                })
            
            # Final ranking
            final_docs.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Return top k documents
            selected_docs = [doc_info['doc'] for doc_info in final_docs[:k]]
            
            # Display retrieval stats
            with st.expander("📊 Enhanced Retrieval Statistics"):
                st.write(f"**Total unique documents found:** {len(doc_scores)}")
                st.write(f"**Final selected documents:** {len(selected_docs)}")
                
                st.write("**Top 3 selected documents:**")
                for i, doc_info in enumerate(final_docs[:3]):
                    st.write(f"**Document {i+1}:**")
                    st.write(f"- Source: {doc_info['doc'].metadata.get('source', 'Unknown')}")
                    st.write(f"- Final Score: {doc_info['final_score']:.3f}")
                    st.write(f"- Keyword Matches: {doc_info['keyword_matches']}/{len(keywords)}")
                    st.write(f"- Query Match Details: {', '.join(doc_info['query_matches'])}")
            
            return selected_docs
            
        else:
            st.warning("No documents retrieved. Falling back to simple retrieval.")
            return vector_db.similarity_search(query, k=k)
            
    except Exception as e:
        st.error(f"Enhanced retrieval failed: {e}")
        st.warning("Falling back to simple retrieval method.")
        return vector_db.similarity_search(query, k=k)

def create_enhanced_retrieval_chain(llm, vector_db, hf_embedding, prompt):
    """Create a retrieval chain with enhanced keyword-based retrieval"""
    
    class EnhancedRetriever:
        def __init__(self, vector_db, hf_embedding, llm, k=6):
            self.vector_db = vector_db
            self.hf_embedding = hf_embedding
            self.llm = llm
            self.k = k
        
        def get_relevant_documents(self, query):
            return enhanced_retrieval_with_keywords(
                query, self.vector_db, self.hf_embedding, self.llm, self.k
            )
        
        def invoke(self, input_dict):
            query = input_dict.get("input", input_dict.get("query", ""))
            return self.get_relevant_documents(query)
    
    # Create enhanced retriever
    enhanced_retriever = EnhancedRetriever(vector_db, hf_embedding, llm)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create custom retrieval chain
    class EnhancedRetrievalChain:
        def __init__(self, retriever, document_chain):
            self.retriever = retriever
            self.document_chain = document_chain
        
        def invoke(self, input_dict):
            # Get query
            query = input_dict["input"]
            
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Generate answer
            result = self.document_chain.invoke({"input": query, "context": docs})
            
            return {"answer": result, "context": docs}
    
    return EnhancedRetrievalChain(enhanced_retriever, document_chain)

def relevance_score(query, document, embeddings):
    try:
        query_embedding = embeddings.embed_query(query)
        document_embedding = embeddings.embed_documents([document.page_content])[0]
        similarity = cosine_similarity([query_embedding], [document_embedding])[0][0]
        
        keywords = query.lower().split()
        keyword_matches = sum(1 for keyword in keywords if keyword in document.page_content.lower())
        keyword_bonus = keyword_matches * 0.1
        
        return similarity + keyword_bonus
    except Exception as e:
        keywords = query.lower().split()
        keyword_matches = sum(1 for keyword in keywords if keyword in document.page_content.lower())
        return keyword_matches * 0.2

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
        
        document_keywords = ['act', 'circular', 'guideline', 'regulation', 'rule', 
                           'amendment', 'notification', 'insurance', 'policy', 'aadhaar']
        
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

def format_document_links_for_embedding(document_links):
    if not document_links:
        return ""
    
    formatted_links = "\n\n=== RELEVANT DOCUMENT LINKS ===\n"
    
    content_docs = [link for link in document_links if link['type'] == 'content']
    ref_docs = [link for link in document_links if link['type'] == 'reference']
    
    if content_docs:
        formatted_links += "\n[CONTENT PAGES]\n"
        for i, link_info in enumerate(content_docs[:10]):
            formatted_links += f"{i+1}. {link_info['title']} - {link_info['link']}\n"
    
    if ref_docs:
        formatted_links += "\n[REFERENCE DOCUMENTS]\n"
        for i, link_info in enumerate(ref_docs[:10]):
            formatted_links += f"{i+1}. {link_info['title']} - {link_info['link']}\n"
    
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

def load_hardcoded_websites():
    loaded_docs = []
    
    for url in HARDCODED_WEBSITES:
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
    """Add source URL and document links to each document before chunking"""
    enhanced_documents = []
    
    for doc in documents:
        source_url = doc.metadata.get('source', 'Unknown')
        document_links = doc.metadata.get('document_links', [])
        
        # Create source line
        source_line = f"Source URL: {source_url}"
        
        # Add document links if available
        if document_links:
            links_text = " | Document Links: "
            link_titles = []
            for link in document_links[:5]:  # Include more links since we have space
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
if 'hf_embedding' not in st.session_state:
    st.session_state['hf_embedding'] = None
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None

st.title("🤖 Enhanced Web GEN-ie with LLM Keyword Extraction")
st.markdown("*Powered by intelligent keyword extraction and enhanced document retrieval*")

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
                        
                        document_chain = create_stuff_documents_chain(llm, st.session_state['prompt'])
                        retriever = st.session_state['vector_db'].as_retriever(search_kwargs={"k": 6})
                        st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
                        
                        st.session_state['docs_loaded'] = True
                        st.success("🎉 Documents processed with embedded links and ready for enhanced querying!")
                    
                    except Exception as e:
                        st.error(f"Error initializing Azure OpenAI: {e}")
                        st.error("Please check your Azure OpenAI configuration and try again.")

st.subheader("🔍 Ask Questions with Enhanced Retrieval")

# Query input with example suggestions
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Enter your query:", 
        value="What are the recent Insurance Acts and amendments?",
        help="Ask questions about IRDAI regulations, insurance acts, PMLA, FEMA, or related topics"
    )

with col2:
    st.markdown("**Example queries:**")
    example_queries = [
        "Latest IRDAI guidelines",
        "Recent insurance amendments",
        "New PMLA regulations",
        "FEMA compliance updates"
    ]
    
    for example in example_queries:
        if st.button(f"📝 {example}", key=f"example_{example}"):
            query = example
            st.rerun()

# Options
col1, col2 = st.columns(2)
with col1:
    show_chunks = st.checkbox("Show retrieved chunks used for answer generation", value=True)
with col2:
    retrieval_method = st.selectbox(
        "Retrieval Method:",
        ["Enhanced (LLM Keywords)", "Standard"],
        index=0,
        help="Choose between enhanced LLM-based keyword extraction or standard retrieval"
    )

# Query processing
if st.button("🚀 Get Answer", disabled=not config_complete) and query:
    if not config_complete:
        st.error("Please complete the Azure OpenAI configuration first.")
    elif st.session_state.get('vector_db') and st.session_state.get('llm'):
        if retrieval_method == "Enhanced (LLM Keywords)":
            # Enhanced retrieval with LLM keyword extraction
            with st.spinner("🤖 Processing query with enhanced retrieval..."):
                try:
                    # Create enhanced retrieval chain
                    enhanced_chain = create_enhanced_retrieval_chain(
                        st.session_state['llm'],
                        st.session_state['vector_db'],
                        st.session_state['hf_embedding'],
                        st.session_state['prompt']
                    )
                    
                    # Process query
                    response = enhanced_chain.invoke({"input": query})
                    
                    # Display response
                    st.subheader("🎯 Enhanced Response:")
                    st.write(response['answer'])
                    
                    # Show retrieved chunks if enabled
                    if show_chunks and 'context' in response:
                        retrieved_docs = response['context']
                        if retrieved_docs:
                            display_chunks(retrieved_docs, "📚 Documents Used for Enhanced Answer")
                            
                            # Show source URLs
                            sources = set()
                            for doc in retrieved_docs:
                                source = doc.metadata.get('source', 'Unknown')
                                sources.add(source)
                            
                            st.write("\n**📍 Information Sources:**")
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. [{source}]({source})")
                    
                    st.success("✅ Enhanced retrieval completed!")
                    
                except Exception as e:
                    st.error(f"Enhanced processing failed: {e}")
                    # Fallback to original method
                    st.info("Falling back to standard retrieval...")
                    response = st.session_state['retrieval_chain'].invoke({"input": query})
                    st.subheader("📄 Standard Response:")
                    st.write(response['answer'])
        else:
            # Standard retrieval method
            with st.spinner("🔍 Processing query with standard retrieval..."):
                try:
                    if st.session_state.get('hf_embedding') is None:
                        st.info("Initializing embeddings...")
                        st.session_state['hf_embedding'] = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
                    
                    if st.session_state.get('prompt') is None:
                        st.info("Creating prompt template...")
                        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                        st.session_state['prompt'] = prompt
                    
                    response = st.session_state['retrieval_chain'].invoke({"input": query})
                    
                    st.subheader("📄 Standard Response:")
                    st.write(response['answer'])
                    
                    if show_chunks and 'context' in response:
                        retrieved_docs = response['context']
                        if retrieved_docs:
                            display_chunks(retrieved_docs, "📋 Top Chunks Used for Answer Generation")
                            
                            links_used = 0
                            for doc in retrieved_docs:
                                if "Source URL:" in doc.page_content:
                                    links_used += 1
                            
                            if links_used > 0:
                                st.success(f"{links_used} out of {len(retrieved_docs)} chunks contained source URLs and document links")
                            else:
                                st.info("ℹ️ No chunks with source URLs and document links were retrieved for this query")
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

# Sidebar with information
with st.sidebar:
    st.header("📋 About Enhanced Web GEN-ie")
    
    st.markdown("""
    ### 🚀 Key Features:
    - **LLM Keyword Extraction**: Intelligent query processing using GPT
    - **Enhanced Retrieval**: Multi-query search with weighted scoring
    - **Document Link Integration**: Embedded reference links in chunks
    - **Source Attribution**: Always includes source URLs
    - **Regulatory Focus**: Specialized for IRDAI, PMLA, FEMA content
    
    ### 🔍 How Enhanced Retrieval Works:
    1. **Query Analysis**: LLM extracts relevant keywords
    2. **Multi-Query Search**: Original + keywords + combined queries
    3. **Smart Ranking**: Weighted scoring based on relevance
    4. **Context Enhancement**: Source URLs embedded in chunks
    
    ### 📊 Data Sources:
    - IRDAI official website
    - Enforcement Directorate (FEMA)
    - Regulatory notifications & circulars
    - Policy guidelines & amendments
    """)
    
    if st.session_state.get('docs_loaded'):
        st.success("✅ Documents loaded and ready!")
        st.metric("Total Documents", len(st.session_state.get('loaded_docs', [])))
        if st.session_state.get('vector_db'):
            st.metric("Vector Database", "Active")
    else:
        st.warning("⏳ Please load documents first")
    
    st.markdown("---")
    st.markdown("*Made with ❤️ using Streamlit & LangChain*")
