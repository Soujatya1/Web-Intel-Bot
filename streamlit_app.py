import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question concisely.
Question: {question} 
Context: {context} 
Answer:
"""

WEBSITES = [
    "https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html",
    "https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html"
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

model = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def fetch_web_content(url):
    """Fetch content using Selenium (for JavaScript-rendered pages)."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        soup = BeautifulSoup(driver.page_source, "html.parser")
        text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])
        st.write(f"🔍 Extracted Content from {url}:")
        st.write(text[:1000])
        
        driver.quit()
        
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        st.write(f"⚠️ Error fetching {url} with Selenium: {e}")
        driver.quit()
        return None

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = []

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
    st.session_state.pdf_mapping = {}

def load_web_content():
    all_documents = []
    st.session_state.pdf_links_dict = {}

    for url in WEBSITES:
        doc = fetch_web_content(url)
        pdf_links = fetch_pdf_links(url)

        if pdf_links:
            st.session_state.pdf_links_dict[url] = pdf_links

        if doc:
            all_documents.append(doc)

    return all_documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def index_docs(documents):
    if documents:
        st.write(f"✅ Indexing {len(documents)} documents")
        for doc in documents:
            st.write(f"📄 {doc.metadata['source']} → {doc.page_content[:500]}")  # Show first 500 chars
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)=

def retrieve_docs(query):
    if st.session_state.vector_store is None:
        return []

    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=8)

    st.write(f"🔍 Retrieved {len(retrieved_docs)} docs for query: {query}")
    for doc in retrieved_docs:
        st.write(f"📄 {doc.metadata['source']} → {doc.page_content[:500]}")

    return retrieved_docs

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    # Combine retrieved documents into a single context
    context = "\n\n".join([doc.page_content[:1000] for doc in documents])  # Limit context size

    # New improved prompt
    enhanced_template = """
    You are an expert AI assistant trained to answer questions based on provided context. 

    - Read the provided context carefully.
    - Answer **only** based on the given information. 
    - If the context does not provide an answer, **say "The retrieved documents do not contain the required information."** 
    - If a document contains relevant links, **include them** in your response.

    Question: {question}
    Context: {context}
    
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(enhanced_template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    
    answer = response.content if response.content else "I couldn’t generate a proper response."

    return answer

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "web_content_indexed" not in st.session_state:
    all_documents = load_web_content()

    if all_documents:
        chunked_documents = split_text(all_documents)
        index_docs(chunked_documents)
        st.session_state.web_content_indexed = True

question = st.chat_input("Ask a question about IRDAI, e-Gazette, ED PMLA, or UIDAI:")

if question and "web_content_indexed" in st.session_state:
    st.session_state.conversation_history.append({"role": "user", "content": question})

    # Retrieve relevant documents from the vector store
    related_documents = retrieve_docs(question)

    # Let the LLM extract relevant links from the context
    answer = answer_question(question, related_documents)

    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
