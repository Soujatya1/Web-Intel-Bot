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
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])])
            return Document(page_content=text, metadata={"source": url})
    except Exception:
        return None
    return None

def fetch_pdf_links(url):
    """Extract PDF links from a given website."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            pdf_links = [a["href"] for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
            return [link if link.startswith("http") else url + link for link in pdf_links] if pdf_links else None
    except Exception:
        return None
    return None

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
    return st.session_state.vector_store.similarity_search(query, k=5)

def find_most_relevant_pdf(query):
    """Find the most relevant PDF based on the query."""
    query_lower = query.lower()
    best_match = None
    best_score = 0

    for url, pdf_links in st.session_state.pdf_links_dict.items():
        for pdf_link in pdf_links:
            match_score = sum(keyword in pdf_link.lower() for keyword in query_lower.split())

            if match_score > best_score:
                best_score = match_score
                best_match = pdf_link

    return best_match

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})

    return response.content if response.content else "I couldn’t generate a proper response."

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

    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)

    # Show only the most relevant PDF link if applicable
    pdf_link = find_most_relevant_pdf(question)
    if pdf_link:
        answer += f"\n\n📄 Here is a relevant PDF:\n🔗 [Download PDF]({pdf_link})"

    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
