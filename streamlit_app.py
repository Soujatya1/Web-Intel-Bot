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
    """Extract PDF titles and links for better matching."""
    headers = {"User-Agent": "Mozilla/5.0"}
    pdf_data = []

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for a in soup.find_all("a", href=True):
                link = a["href"]
                if ".pdf" in link.lower():
                    pdf_title = a.text.strip() if a.text else "Unknown Document"
                    full_link = link if link.startswith("http") else url + link
                    pdf_data.append({"title": pdf_title, "link": full_link})
    except Exception:
        return []

    return pdf_data

def index_pdf_links():
    """Indexes PDF links using FAISS for quick similarity search."""
    if not st.session_state.pdf_store:
        return

    pdf_titles = [pdf["title"] for pdf in st.session_state.pdf_store]
    pdf_vectors = embeddings.embed_documents(pdf_titles)

    # Create FAISS index
    d = len(pdf_vectors[0])  # Dimensionality of embeddings
    pdf_index = faiss.IndexFlatL2(d)
    pdf_index.add(np.array(pdf_vectors, dtype=np.float32))

    # Store FAISS index and mapping
    st.session_state.pdf_index = pdf_index
    st.session_state.pdf_mapping = {i: pdf["link"] for i, pdf in enumerate(st.session_state.pdf_store)}

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = []

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None
    st.session_state.pdf_mapping = {}

index_pdf_links()

def find_most_relevant_pdf(answer_context):
    """Finds the most relevant PDF using FAISS vector search."""
    if not st.session_state.pdf_store or "pdf_index" not in st.session_state:
        return None

    query_vector = np.array([embeddings.embed_query(answer_context)], dtype=np.float32)
    
    # Perform similarity search in FAISS
    D, I = st.session_state.pdf_index.search(query_vector, k=1)  # Retrieve top 1 match

    if D[0][0] < 0.5:  # Ensure similarity threshold
        return None

    best_match_index = I[0][0]
    return st.session_state.pdf_mapping.get(best_match_index, None)

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

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    
    answer = response.content if response.content else "I couldn’t generate a proper response."

    # Now find the most relevant PDF **after** the answer is generated
    pdf_link = find_most_relevant_pdf(answer)
    if pdf_link:
        answer += f"\n\n📄 Here is a relevant PDF:\n🔗 [Download PDF]({pdf_link})"

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
