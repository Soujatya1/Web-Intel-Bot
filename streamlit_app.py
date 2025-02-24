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
    """Fetch and extract text content from a webpage using requests or Selenium."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ")
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)
        
        st.write(f"Requests fetched {len(cleaned_text)} chars from {url}: {cleaned_text[:100]}...")
        if len(cleaned_text) > 200:
            return Document(page_content=cleaned_text, metadata={"source": url})
    except Exception as e:
        st.warning(f"Requests failed for {url}: {str(e)}. Falling back to Selenium.")

    try:
        driver = setup_selenium()
        driver.get(url)
        time.sleep(5)  # Increased to 5s for more reliable JS loading
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()
        
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ")
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)
        
        st.write(f"Selenium fetched {len(cleaned_text)} chars from {url}: {cleaned_text[:100]}...")
        return Document(page_content=cleaned_text, metadata={"source": url})
    except Exception as e:
        st.error(f"Selenium failed for {url}: {str(e)}")
        return None

def load_web_content():
    """Load content only from the specified websites, no crawling."""
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
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    st.write(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs

def index_docs(documents):
    """Index documents into the vector store."""
    vector_store.add_documents(documents)

def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    retrieved = vector_store.similarity_search(query)
    st.write(f"Retrieved {len(retrieved)} documents for query: {query}")
    return retrieved

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
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
        st.success(f"Web content loaded and indexed successfully! Loaded {len(all_documents)} pages, {len(chunked_documents)} chunks.")
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
