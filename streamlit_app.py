import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd

st.title("Document GeN-ie")
st.subheader("Chat with your web documents")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.

Question: {question} 
Context: {context} 
Answer:
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0.3)

def load_web_documents(urls):
    """Load documents from web URLs"""
    loader = WebBaseLoader(urls)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({"question": question, "context": context})
    
    return response.content

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Replace PDF upload with URL input
url_input = st.text_area(
    "Enter URLs (one per line):",
    height=100,
    placeholder="https://example.com\nhttps://another-site.com"
)

if st.button("Load Web Documents") and url_input:
    urls = [url.strip() for url in url_input.split('\n') if url.strip()]
    
    if urls:
        try:
            with st.spinner("Loading web documents..."):
                documents = load_web_documents(urls)
                chunked_documents = split_text(documents)
                index_docs(chunked_documents)
            
            # Display a success message
            st.success(f"Successfully processed {len(urls)} web document(s).")
            
            # Display document preview (optional)
            with st.expander("Preview Loaded Documents"):
                for i, doc in enumerate(documents[:3]):  # Show first 3 documents only
                    st.write(f"**Source: {doc.metadata.get('source', 'Unknown')}**")
                    content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    st.text(content_preview)
                    if i < len(documents[:3]) - 1:
                        st.divider()
                        
        except Exception as e:
            st.error(f"Error loading web documents: {str(e)}")

question = st.chat_input("Ask a question:")
if question:
    st.session_state.conversation_history.append({"role": "user", "content": question})
    
    related_documents = retrieve_docs(question)
    
    answer = answer_question(question, related_documents)
    
    st.session_state.conversation_history.append({"role": "assistant", "content": answer})

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
