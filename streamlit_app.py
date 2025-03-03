import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

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

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def load_web_content():
    """Fetch content from websites using WebBaseLoader"""
    loader = WebBaseLoader(WEBSITES)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)

def index_docs(documents):
    """Index documents into FAISS"""
    if documents:
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)

def retrieve_docs(query):
    """Retrieve relevant documents"""
    if st.session_state.vector_store is None:
        return []
    return st.session_state.vector_store.similarity_search(query, k=2)

def answer_question(question, documents):
    """Generate an answer"""
    if not documents:
        return "I couldn’t find relevant information to answer this question."

    context = "\n\n".join([doc.page_content for doc in documents])

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

    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)

    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
