import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # ✅ Use FAISS instead of InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import time

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.
Please do not answer anything which is out of the document/website context.
Question: {question} 
Context: {context} 
Answer:
"""

WEBSITES = [
    "https://enforcementdirectorate.gov.in/pmla", "https://uidai.gov.in/en/", "https://irdai.gov.in/rules", "https://irdai.gov.in/"
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create FAISS Index
vector_store = None  # Initialize vector store as None

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
    
    st.error(f"Failed to fetch content after {retries} attempts.")
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
    """Split documents into overlapping chunks for better context retention."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,   # Increased chunk size
        chunk_overlap=200, # More overlap to retain context
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)

    st.write(f"📌 Split {len(documents)} documents into {len(chunked_docs)} chunks")
    
    # Show first 2 chunks for debugging
    if chunked_docs:
        st.write("🔍 Example chunk preview:")
        st.write(chunked_docs[0].page_content[:300])  # Show first 300 chars
    
    return chunked_docs

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Ensure FAISS is stored persistently

def index_docs(documents):
    if documents:
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
        st.write(f"✅ Indexed {len(documents)} documents into FAISS")
    else:
        st.error("❌ No documents to index")

def retrieve_docs(query):
    if st.session_state.vector_store is None:
        st.error("❌ Vector store is empty. Make sure documents are indexed.")
        return []
    
    return st.session_state.vector_store.similarity_search(query, k=5)

    if not retrieved:
        st.error("❌ No relevant documents found. Possible causes:")
        st.write("1️⃣ Documents were not indexed properly.")
        st.write("2️⃣ Embeddings are not working as expected.")
        st.write("3️⃣ Query does not match indexed content.")
        return []
    
    st.write(f"✅ Retrieved {len(retrieved)} documents for query: {query}")

    # Print first retrieved chunk
    st.write("🔍 Example retrieved chunk:")
    st.write(retrieved[0].page_content[:300])

    return retrieved

def answer_question(question, documents):
    """Generate an answer based on retrieved documents with better handling for missing context."""
    
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])

    # Debug: Print the context being sent
    st.write(f"Using context for answering:\n{context[:500]}...")  # Show first 500 chars

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
        
        # 🔄 Reset vector store before re-indexing
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
