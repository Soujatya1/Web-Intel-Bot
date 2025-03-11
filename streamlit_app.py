import streamlit as st
import os
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import time
import re

st.title("Website Q&A System")

WEBSITES = [
    "https://egazette.gov.in/(S(jjko5lh5lpdta4yyxrdk4lfu))/Default.aspx", "https://irdai.gov.in/web/guest/whats-new", "https://irdai.gov.in/directory-of-employees",
    "https://irdai.gov.in/warnings-and-penalties", "https://uidai.gov.in/en/about-uidai/legal-framework.html",
    "https://irdai.gov.in/rules", "https://irdai.gov.in/consolidated-gazette-notified-regulations", "https://irdai.gov.in/notifications",
    "https://irdai.gov.in/circulars", "https://irdai.gov.in/orders1", "https://irdai.gov.in/exposure-drafts", "https://irdai.gov.in/programmes-to-advance-understanding-of-rti",
    "https://irdai.gov.in/c/portal/layout?p_l_id=1018&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterDepartment=ALL&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterFromDate=&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterToDate=&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterSearchKeyword=&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterEntities=ALL&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterClassification=RTI&_com_irdai_document_media_IRDAIDocumentMediaPortlet_filterTag=ALL&_com_irdai_document_media_IRDAIDocumentMediaPortlet_archiveOn=false&p_p_id=com_irdai_document_media_IRDAIDocumentMediaPortlet&p_p_lifecycle=0",
    "https://irdai.gov.in/cic-orders", "https://irdai.gov.in/rti-2005/tenders", "https://irdai.gov.in/antimoney-laundering", "https://irdai.gov.in/other-communication",
    "https://enforcementdirectorate.gov.in/fema", "https://enforcementdirectorate.gov.in/feoa", "https://enforcementdirectorate.gov.in/bns", "https://enforcementdirectorate.gov.in/bnss",
    "https://enforcementdirectorate.gov.in/bsa", "https://uidai.gov.in/en/about-uidai/legal-framework/rules.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/notification.html", "https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html", "https://uidai.gov.in/en/about-uidai/legal-framework/judgements.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html", "https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules-en.html"
]

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "websites_processed" not in st.session_state:
    st.session_state.websites_processed = False

if "embedding_model_used" not in st.session_state:
    st.session_state.embedding_model_used = None

with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    model_name = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    embedding_model = st.selectbox(
        "Select Embedding Model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    )
    
    st.subheader("Advanced Options")
    with st.expander("Scraping Options"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        max_retries = st.slider("Max Retries per URL", 1, 5, 2)

if st.session_state.embedding_model_used and st.session_state.embedding_model_used != embedding_model:
    st.warning(f"Embedding model changed from {st.session_state.embedding_model_used} to {embedding_model}. Websites need to be reprocessed.")
    st.session_state.websites_processed = False
    st.session_state.vectorstore = None
    
st.header("Websites to Process")
for i, website in enumerate(WEBSITES, 1):
    st.write(f"{i}. {website}")

def classify_website(url):
    """Classify the website type based on URL and content"""
    url_lower = url.lower()
    if "irdai" in url_lower:
        if "tender" in url_lower:
            return "irdai-tender"
        elif "notification" in url_lower or "circular" in url_lower:
            return "irdai-notification"
        elif "rule" in url_lower or "regulation" in url_lower:
            return "irdai-regulation"
        else:
            return "irdai-general"
    elif "uidai" in url_lower:
        if "legal-framework" in url_lower:
            return "uidai-legal"
        else:
            return "uidai-general"
    elif "egazette" in url_lower:
        return "egazette"
    elif "enforcement" in url_lower:
        return "enforcement-directorate"
    else:
        return "general"

def enhanced_scrape_page(url, retries=2):
    website_type = classify_website(url)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    documents = []
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer"]):
                script.extract()
            
            title = soup.title.string if soup.title else "No Title"
            
            if website_type == "egazette":
                main_elements = soup.select('div#pnlSearchData, table.table, div.content_para')
                
                if main_elements:
                    for element in main_elements:
                        content = element.get_text(separator="\n", strip=True)
                        if content:
                            documents.append(Document(
                                page_content=f"eGazette Information:\nTitle: {title}\n{content}",
                                metadata={"source": url, "type": website_type}
                            ))
                
            elif "irdai" in website_type:
                main_content = soup.select_one('main, #content, .content, .portlet-content, .journal-content-article')
                
                if main_content:
                    content = main_content.get_text(separator="\n", strip=True)
                    tables = main_content.select('table')
                    table_contents = []
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            row_content = " | ".join([cell.get_text(strip=True) for cell in cells])
                            if row_content.strip():
                                table_contents.append(row_content)
                    
                    table_text = "\n".join(table_contents)
                    
                    if table_text and not table_text in content:
                        content = f"{content}\n\nTabular Data:\n{table_text}"
                    
                    documents.append(Document(
                        page_content=f"IRDAI {website_type.split('-')[1].title()} Information:\nTitle: {title}\n{content}",
                        metadata={"source": url, "type": website_type}
                    ))
                
                document_listings = soup.select('.document-content, .document-container, .lfr-search-container-wrapper tbody tr')
                
                if document_listings:
                    listing_text = []
                    for item in document_listings:
                        item_text = item.get_text(separator=" | ", strip=True)
                        if item_text:
                            listing_text.append(item_text)
                    
                    if listing_text:
                        documents.append(Document(
                            page_content=f"IRDAI {website_type.split('-')[1].title()} Listings:\nTitle: {title}\n" + "\n".join(listing_text),
                            metadata={"source": url, "type": website_type}
                        ))
            
            elif "uidai" in website_type:
                main_content = soup.select_one('#maincontent, .main-content, article')
                
                if main_content:
                    content = main_content.get_text(separator="\n", strip=True)
                    documents.append(Document(
                        page_content=f"UIDAI Information:\nTitle: {title}\n{content}",
                        metadata={"source": url, "type": website_type}
                    ))
                    
                    links = main_content.select('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"]')
                    if links:
                        link_text = []
                        for link in links:
                            link_text.append(f"Document: {link.get_text(strip=True)} - Link: {link.get('href')}")
                        
                        documents.append(Document(
                            page_content=f"UIDAI Document Links:\nTitle: {title}\n" + "\n".join(link_text),
                            metadata={"source": url, "type": f"{website_type}-documents"}
                        ))
            
            else:
                main_content = soup.select_one('main, #content, .content, article, .container')
                
                if main_content:
                    content = main_content.get_text(separator="\n", strip=True)
                    documents.append(Document(
                        page_content=f"{title}\n{content}",
                        metadata={"source": url, "type": website_type}
                    ))
            
            if not documents:
                body_text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
                
                paragraphs = [p for p in body_text.split("\n\n") if p.strip()]
                
                content = "\n\n".join(paragraphs[:50])
                
                documents.append(Document(
                    page_content=f"{title}\n{content}",
                    metadata={"source": url, "type": website_type}
                ))
            
            if documents:
                break
                
        except Exception as e:
            st.warning(f"Attempt {attempt+1} for {url} failed: {str(e)}")
            time.sleep(2)
    
    if not documents:
        documents.append(Document(
            page_content=f"Unable to extract content from {url}",
            metadata={"source": url, "type": "extraction-failed"}
        ))
    
    return documents

def process_websites(urls_list):
    with st.spinner("Loading and processing websites... This may take a few minutes."):
        try:
            all_documents = []
            progress_bar = st.progress(0)
            
            for i, url in enumerate(urls_list):
                st.write(f"Processing URL ({i+1}/{len(urls_list)}): {url}")
                
                try:
                    loader = WebBaseLoader(url)
                    web_documents = loader.load()
                    
                    if not web_documents or sum(len(doc.page_content) for doc in web_documents) < 500:
                        st.write(f"WebBaseLoader returned insufficient content, trying enhanced scraper for {url}")
                        documents = enhanced_scrape_page(url, retries=max_retries)
                    else:
                        for doc in web_documents:
                            if 'source' not in doc.metadata:
                                doc.metadata['source'] = url
                            doc.metadata['type'] = classify_website(url)
                        documents = web_documents
                
                except Exception as e:
                    st.write(f"WebBaseLoader failed for {url}, using enhanced scraper: {str(e)}")
                    documents = enhanced_scrape_page(url, retries=max_retries)
                
                all_documents.extend(documents)
                progress_bar.progress((i + 1) / len(urls_list))
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            st.write("Splitting documents into chunks...")
            all_chunks = text_splitter.split_documents(all_documents)
            
            # Create vector store
            st.write(f"Creating vector store with {len(all_chunks)} chunks...")
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            
            st.session_state.embedding_model_used = embedding_model
            st.session_state.vectorstore = vectorstore
            st.session_state.websites_processed = True
            st.success(f"Successfully processed all websites with {len(all_chunks)} total chunks. You can now ask questions!")
            return True
        
        except Exception as e:
            st.error(f"Error in processing websites: {str(e)}")
            return False

if st.session_state.websites_processed:
    if st.button("Reprocess Websites"):
        st.session_state.websites_processed = False
        st.session_state.vectorstore = None
        process_websites(WEBSITES)
else:
    if st.button("Process Websites"):
        process_websites(WEBSITES)

def get_relevant_sources(query, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(query, k=k)
    sources = []
    for doc in relevant_docs:
        if 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    return sources, relevant_docs

st.header("Ask Questions")

if groq_api_key and st.session_state.vectorstore is not None:
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.5,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    prompt_template = """
    You are an assistant that helps with information from government websites including IRDAI, UIDAI, and the eGazette.
    
    Question: {question}
    
    Context information to answer the question:
    {context}
    
    Answer the question based on the context provided above. Be specific and include details from the context.
    If the information mentions dates, deadlines, document numbers, or specific details, include those in your answer.
    If you don't have enough information, say so, but try to provide what you know from the context.
    Always structure your answer clearly with paragraphs and bullet points when appropriate.
    """

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
    )
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about the websites:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        query_complexity = len(prompt.split())
        k_docs = min(max(3, query_complexity // 5), 8)
        
        relevant_sources, relevant_docs = get_relevant_sources(prompt, st.session_state.vectorstore, k=k_docs)
        
        with st.spinner("Thinking..."):
            response = conversation_chain.invoke({
                "question": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            source_docs = response.get('source_documents', [])
            
            sources = []
            doc_types = set()
            for doc in source_docs:
                if 'source' in doc.metadata:
                    source_url = doc.metadata['source']
                    if source_url not in sources:
                        sources.append(source_url)
                if 'type' in doc.metadata:
                    doc_types.add(doc.metadata['type'])
            
            if not sources:
                sources = relevant_sources
            
            answer = response['answer']
            
            source_info = ""
            if doc_types:
                source_info = "\n\nInformation sources: " + ", ".join(doc_types)
        
        with st.chat_message("assistant"):
            st.write(answer)
            
            with st.expander("Source Information"):
                st.write("**Document Types Used:**")
                for doc_type in doc_types:
                    st.write(f"- {doc_type}")
                
                st.write("**Sources:**")
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. [{source}]({source})")
        
        response_with_sources = answer
        if sources:
            source_text = "\n\nSources:\n" + "\n".join(sources)
            response_with_sources += source_text
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_sources
        })

elif not st.session_state.websites_processed:
    st.info("Please process the websites to enable Q&A.")
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to enable Q&A functionality.")
