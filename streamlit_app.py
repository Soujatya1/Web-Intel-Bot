import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests

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
vector_store = InMemoryVectorStore(embeddings=embeddings)
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
        # Get HTML content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables in the HTML
        html_tables = soup.find_all('table')
        
        from langchain_core.documents import Document
        
        for i, table in enumerate(html_tables):
            try:
                # Convert HTML table to pandas DataFrame
                df_list = pd.read_html(str(table))
                
                for j, df in enumerate(df_list):
                    # Convert to CSV string
                    csv_string = df.to_csv(index=False)
                    
                    # Create a document for each table
                    table_doc = Document(
                        page_content=f"[TABLE] Table {i+1}.{j+1} from {url}:\n{csv_string}",
                        metadata={
                            "source": url,
                            "type": "table"
                        }
                    )
                    documents.append(table_doc)
            except:
                # Skip tables that can't be parsed
                continue
    except Exception as e:
        # No tables found or error in extraction
        print(f"Error extracting tables: {str(e)}")
        
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
            splits = text_splitter.split_documents([doc])
            # Make sure metadata is preserved
            for split in splits:
                split.metadata = doc.metadata.copy()
            split_docs.extend(splits)
    
    return split_docs

def index_docs(documents):
    if documents:
        # Add documents to the vector store
        vector_store.add_documents(documents)
        return True
    return False

def retrieve_docs(query):
    # Add a debug print to check if there are documents in the vector store
    docs = vector_store.similarity_search(query, k=4)  # Retrieve top 4 relevant documents
    if not docs:
        st.warning("No relevant documents found for this query.")
    return docs

def answer_question(question, documents):
    # Combine the contexts from all retrieved documents
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Prepare the prompt with the question and context
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    # Generate the response
    response = chain.invoke({"question": question, "context": context})
    
    return response.content

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = False
if "vector_store_populated" not in st.session_state:
    st.session_state.vector_store_populated = False
if "documents_added" not in st.session_state:
    st.session_state.documents_added = False

# Replace file uploader with URL input
website_url = st.text_input("Enter website URL:")

# Add a button to process the URL
process_button = st.button("Process Website")

if website_url and process_button:
    try:
        # Show progress indicator
        with st.spinner(f"Processing website {website_url}..."):
            # Process the website
            documents = load_website(website_url)
            
            if documents:
                # Split text into chunks
                chunked_documents = split_text(documents)
                
                # Clear existing vector store
                if hasattr(vector_store, 'delete_collection'):
                    try:
                        vector_store.delete_collection()
                    except:
                        # Recreate the vector store if deletion fails
                        vector_store = InMemoryVectorStore(embeddings=embeddings)
                
                # Index the documents
                if index_docs(chunked_documents):
                    # Update session state
                    st.session_state.processed_documents = True
                    st.session_state.vector_store_populated = True
                    st.session_state.documents_added = True
                    
                    # Display a success message with document count
                    st.success(f"Successfully processed website: {website_url} ({len(chunked_documents)} chunks created)")
                    
                    # Show a sample of the extracted content
                    with st.expander("Preview Extracted Content"):
                        for i, doc in enumerate(chunked_documents[:3]):  # Show first 3 chunks only
                            st.write(f"**Chunk {i+1} from {doc.metadata['source']}**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.divider()
                    
                    # Display table preview (optional)
                    table_docs = [doc for doc in chunked_documents if doc.metadata["type"] == "table"]
                    if table_docs:
                        with st.expander("Preview Extracted Tables"):
                            for i, doc in enumerate(table_docs[:3]):  # Show first 3 tables only
                                st.write(f"**Table from {doc.metadata['source']}**")
                                table_content = doc.page_content.replace("[TABLE] ", "")
                                st.text(table_content)
                                if i < len(table_docs[:3]) - 1:
                                    st.divider()
                else:
                    st.warning("No content could be indexed from the website.")
            else:
                st.warning("No content could be extracted from the website.")
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")

# Option to process multiple URLs
with st.expander("Process Multiple URLs"):
    multi_urls = st.text_area("Enter multiple URLs (one per line):")
    process_multi_button = st.button("Process All URLs")
    
    if multi_urls and process_multi_button:
        urls = [url.strip() for url in multi_urls.split("\n") if url.strip()]
        
        # Clear existing vector store
        if hasattr(vector_store, 'delete_collection'):
            try:
                vector_store.delete_collection()
            except:
                # Recreate the vector store if deletion fails
                vector_store = InMemoryVectorStore(embeddings=embeddings)
        
        processed_count = 0
        total_chunks = 0
        all_chunked_documents = []
        
        for url in urls:
            try:
                with st.spinner(f"Processing {url}..."):
                    documents = load_website(url)
                    if documents:
                        chunked_documents = split_text(documents)
                        all_chunked_documents.extend(chunked_documents)
                        processed_count += 1
                        total_chunks += len(chunked_documents)
                        st.write(f"✅ Processed: {url} ({len(chunked_documents)} chunks)")
                    else:
                        st.write(f"⚠️ No content extracted from: {url}")
            except Exception as e:
                st.write(f"❌ Failed to process {url}: {str(e)}")
        
        if all_chunked_documents:
            # Index all documents at once
            if index_docs(all_chunked_documents):
                st.session_state.processed_documents = True
                st.session_state.vector_store_populated = True
                st.session_state.documents_added = True
                st.success(f"Successfully processed {processed_count} websites with a total of {total_chunks} chunks.")
            else:
                st.warning("No content could be indexed from the websites.")

question = st.chat_input("Ask a question:")
if question and st.session_state.vector_store_populated:
    st.session_state.conversation_history.append({"role": "user", "content": question})
    
    # Get related documents
    related_documents = retrieve_docs(question)
    
    # Debug information
    with st.expander("Debug: Retrieved Context"):
        st.write(f"Number of chunks retrieved: {len(related_documents)}")
        for i, doc in enumerate(related_documents):
            st.write(f"**Document {i+1} from {doc.metadata.get('source', 'unknown')}**")
            st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    
    if related_documents:
        answer = answer_question(question, related_documents)
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
    else:
        no_context_msg = "I don't have enough context to answer that question based on the processed websites. Try processing more websites or reformulating your question."
        st.session_state.conversation_history.append({"role": "assistant", "content": no_context_msg})
elif question and not st.session_state.vector_store_populated:
    st.warning("Please process at least one website before asking questions.")

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
