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
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def fetch_web_content(url, retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])
                return Document(page_content=text, metadata={"source": url})
            else:
                st.error(f"Failed to fetch content, status code: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            st.warning(f"Timeout error. Retrying {attempt + 1}/{retries}...")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            st.error(f"Error fetching content: {e}")
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
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)

def retrieve_docs(query):
    if st.session_state.vector_store is None:
        return []

    # Retrieve more documents (k=8 instead of k=5)
    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=8)

    # Filter: Only keep documents where the query words are present
    query_lower = query.lower()
    keyword_filtered_docs = [doc for doc in retrieved_docs if any(word in doc.page_content.lower() for word in query_lower.split())]

    # If filtering removes all, use the first 5 documents as fallback
    return keyword_filtered_docs if keyword_filtered_docs else retrieved_docs[:5]

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
