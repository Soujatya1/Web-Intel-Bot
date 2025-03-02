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

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following retrieved context to answer the question concisely.
Question: {question} 
Context: {context} 
Answer:
"""

WEBSITES = [
    "https://uidai.gov.in/en/ecosystem/enrolment-ecosystem/enrolment-agencies.html",
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

def load_web_content():
    all_documents = []
    st.session_state.pdf_store = []

    for url in WEBSITES:
        doc = fetch_web_content(url)
        pdf_links = fetch_pdf_links(url)

        if pdf_links:
            st.session_state.pdf_store.extend(pdf_links)

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
    """Retrieve the most relevant documents using hybrid search (semantic + keyword)."""
    if st.session_state.vector_store is None:
        return []

    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=8)

    query_lower = query.lower()
    keyword_filtered_docs = [doc for doc in retrieved_docs if any(word in doc.page_content.lower() for word in query_lower.split())]

    return keyword_filtered_docs if keyword_filtered_docs else retrieved_docs[:5]

def find_most_relevant_pdf(query):
    """Find the most relevant PDF using embedding similarity."""
    if not st.session_state.pdf_store:
        return None

    pdf_titles = [pdf["title"] for pdf in st.session_state.pdf_store]
    pdf_vectors = embeddings.embed_documents(pdf_titles)

    query_vector = embeddings.embed_query(query)
    
    # Compute cosine similarity
    best_match = None
    highest_similarity = -1

    for idx, vector in enumerate(pdf_vectors):
        similarity = sum(q * v for q, v in zip(query_vector, vector))  # Cosine similarity approximation

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = st.session_state.pdf_store[idx]["link"]

    return best_match if highest_similarity > 0.5 else None  # Ensure minimum relevance threshold

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

    # Show the relevant PDF link only if found
    pdf_link = find_most_relevant_pdf(question)
    if pdf_link:
        answer += f"\n\n📄 Here is a relevant PDF:\n🔗 [Download PDF]({pdf_link})"

    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
