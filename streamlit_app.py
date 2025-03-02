import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Keep responses precise.
Question: {question} 
Context: {context} 
Answer:
"""

WEBSITES = ["https://irdai.gov.in/rules"]

# Initialize embeddings and ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embeddings)

model = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def fetch_web_content(url):
    """Fetch content from a webpage and return a LangChain Document."""
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
    """Fetch and store web content."""
    all_documents = []
    for url in WEBSITES:
        st.write(f"Loading: {url}...")
        doc = fetch_web_content(url)
        if doc:
            all_documents.append(doc)
            st.write(f"Loaded {len(doc.page_content)} characters from {url}")
        else:
            st.write(f"No content loaded from {url}")
    return all_documents

def split_text(documents):
    """Split documents into chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    st.write(f"📌 Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs

def index_docs(documents):
    """Index documents into the Chroma vector store."""
    if documents:
        vector_store.add_documents(documents)
        st.write(f"✅ Indexed {len(documents)} documents into vector store")

        # Test retrieval
        test_retrieval = vector_store.similarity_search("insurance", k=1)
        if test_retrieval:
            st.write("✅ Test retrieval successful! Example chunk:")
            st.write(test_retrieval[0].page_content[:300])
        else:
            st.error("❌ Test retrieval failed - vector store may not be working correctly.")

def retrieve_docs(query):
    """Retrieve relevant documents for a given query."""
    retrieved = vector_store.similarity_search(query, k=3)
    if not retrieved:
        st.error("❌ No relevant documents found. Possible causes:")
        st.write("1️⃣ Documents were not indexed properly.")
        st.write("2️⃣ Embeddings are not working as expected.")
        return []
    
    st.write(f"✅ Retrieved {len(retrieved)} documents for query: {query}")
    return retrieved

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])
    st.write(f"Using context for answering:\n{context[:500]}...") 

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"question": question, "context": context})
    return response.content if response.content else "I couldn’t generate a proper response."

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "web_content_indexed" not in st.session_state:
    st.write("🔄 Reloading content from websites, please wait...")
    all_documents = load_web_content()
    
    if all_documents:
        chunked_documents = split_text(all_documents)

        # Reset ChromaDB and re-index documents
        vector_store = Chroma(embedding_function=embeddings)
        index_docs(chunked_documents)

        st.session_state.web_content_indexed = True
        st.success(f"✅ Web content reloaded and indexed successfully! Loaded {len(all_documents)} pages.")
    else:
        st.error("❌ Failed to load web content.")

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
