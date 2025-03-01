import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.
Question: {question} 
Context: {context} 
Answer:
"""

WEBSITES = [
    "https://irdai.gov.in/rules"
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

model = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def setup_selenium():
    """Set up Selenium with headless Chrome."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def fetch_web_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])
            return Document(page_content=text, metadata={"source": url})
        else:
            st.error(f"Failed to fetch content, status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching content: {e}")
        return None

def load_web_content():
    all_documents = []
    for url in WEBSITES:
        st.write(f"Loading: {url}...")
        doc = fetch_web_content(url)
        if doc:
            all_documents.append(doc)
            st.write(f"Loaded {len(doc.page_content)} chars from {url}")
        else:
            st.write(f"No content loaded from {url}")
    st.write(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_text(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs

def index_docs(documents):
    """Index documents into the vector store."""
    if documents:
        vector_store.add_documents(documents)
        st.write(f"Indexed {len(documents)} documents into vector store")
        # Test retrieval
        test_result = vector_store.similarity_search("insurance rules", k=1)
        if test_result:
            st.write(f"Test retrieval successful: {test_result[0].page_content[:100]}...")
        else:
            st.warning("Test retrieval failed - no documents matched.")
    else:
        st.error("No documents to index")

def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    retrieved = vector_store.similarity_search(query, k=5)
    st.write(f"Retrieved {len(retrieved)} documents for query: {query}")
    return retrieved

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    if not context:
        return "I don’t have enough information to answer this question."
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    return response.content

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "web_content_indexed" not in st.session_state:
    st.write("Loading content from websites, please wait...")
    all_documents = load_web_content()
    if all_documents:
        chunked_documents = split_text(all_documents)
        index_docs(chunked_documents)
        st.session_state.web_content_indexed = True
        st.success(f"Web content loaded and indexed successfully! Loaded {len(all_documents)} pages.")
    else:
        st.error("Failed to load web content.")

question = st.chat_input("Ask a question about IRDAI, e-Gazette, ED PMLA, or UIDAI:")

if question and "web_content_indexed" in st.session_state:
    st.session_state.conversation_history.append({"role": "user", "content": question})
    
    related_documents = retrieve_docs(question)
    
    answer = answer_question(question, related_documents)
    
    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
