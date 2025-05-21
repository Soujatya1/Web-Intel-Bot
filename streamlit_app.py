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
st.subheader("Chat with your websites")

# Create directory if it doesn't exist
websites_directory = '.github/'
os.makedirs(websites_directory, exist_ok=True)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.

The document may contain tables. Tables are formatted as CSV data and preceded by [TABLE] markers.

Question: {question} 
Context: {context} 
Answer:
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0.3)

def load_website(url):
    # Use WebBaseLoader to load content from the website
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # Add source and type metadata
    for doc in documents:
        doc.metadata["source"] = url
        doc.metadata["type"] = "text"
        
    # Process tables in HTML content if they exist
    try:
        # Extract tables from HTML using pandas
        tables = pd.read_html(url)
        
        for i, table in enumerate(tables):
            # Convert to CSV string
            csv_string = table.to_csv(index=False)
            
            # Create a document for each table
            from langchain_core.documents import Document
            table_doc = Document(
                page_content=f"[TABLE] Table {i+1} from {url}:\n{csv_string}",
                metadata={
                    "source": url,
                    "type": "table"
                }
            )
            documents.append(table_doc)
    except:
        # No tables found or error in extraction
        pass
        
    return documents

def split_text(documents):
    # Use a splitter that respects table boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Process text and table documents differently
    split_docs = []
    for doc in documents:
        # Keep tables intact (don't split)
        if doc.metadata["type"] == "table":
            split_docs.append(doc)
        else:
            # Split text documents
            split_docs.extend(text_splitter.split_documents([doc]))
    
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

# Replace file uploader with URL input
website_url = st.text_input("Enter website URL:")

# Add a button to process the URL
process_button = st.button("Process Website")

if website_url and process_button:
    try:
        all_documents = []
        documents = load_website(website_url)
        chunked_documents = split_text(documents)
        all_documents.extend(chunked_documents)
        
        index_docs(all_documents)
        
        # Display a success message
        st.success(f"Successfully processed website: {website_url}")
        
        # Display table preview (optional)
        table_docs = [doc for doc in all_documents if doc.metadata["type"] == "table"]
        if table_docs:
            with st.expander("Preview Extracted Tables"):
                for i, doc in enumerate(table_docs[:3]):  # Show first 3 tables only
                    st.write(f"**{doc.metadata['source']}**")
                    table_content = doc.page_content.replace("[TABLE] ", "")
                    st.text(table_content)
                    if i < len(table_docs[:3]) - 1:
                        st.divider()
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")

# Option to process multiple URLs
with st.expander("Process Multiple URLs"):
    multi_urls = st.text_area("Enter multiple URLs (one per line):")
    process_multi_button = st.button("Process All URLs")
    
    if multi_urls and process_multi_button:
        urls = [url.strip() for url in multi_urls.split("\n") if url.strip()]
        all_documents = []
        
        for url in urls:
            try:
                documents = load_website(url)
                chunked_documents = split_text(documents)
                all_documents.extend(chunked_documents)
                st.write(f"✅ Processed: {url}")
            except Exception as e:
                st.write(f"❌ Failed to process {url}: {str(e)}")
        
        if all_documents:
            index_docs(all_documents)
            st.success(f"Successfully processed {len(urls)} websites.")

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
