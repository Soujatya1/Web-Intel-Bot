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
    model_name="llama3-70b-8192",
    temperature=0
)

def fetch_web_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])

            # Include all hyperlinks found on the page
            links = [a["href"] for a in soup.find_all("a", href=True) if "http" in a["href"]]
            text += "\n\n🔗 Related Links:\n" + "\n".join(links)

            return Document(page_content=text, metadata={"source": url})
    except Exception:
        return None
    return None

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = []

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
    st.session_state.pdf_mapping = {}

def load_web_content():
    all_documents = []

    for url in WEBSITES:
        doc = fetch_web_content(url)
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
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)

def retrieve_docs(query):
    if st.session_state.vector_store is None:
        return []
    return st.session_state.vector_store.similarity_search(query, k=2)

def answer_question(question, documents):
    """Generate an answer and ask the LLM to extract relevant links."""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])

    # New Prompt: Ask LLM to find the best link from the context
    enhanced_template = """
    You are an assistant for question-answering tasks. Use the following retrieved context to answer the question concisely.
    If there is a URL or document reference in the context that is highly relevant to the question, include it in your response.

    Question: {question}
    Context: {context}
    
    Answer (include relevant links if applicable):
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
