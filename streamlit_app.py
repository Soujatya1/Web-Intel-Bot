import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import FastEmbeddings
import pandas as pd

st.title("Document GeN-ie")
st.subheader("Chat with web content")

# List of hardcoded websites to scrape
WEBSITES = ["https://irdai.gov.in/acts", "https://irdai.gov.in/rules", "https://irdai.gov.in/consolidated-gazette-notified-regulations", "https://irdai.gov.in/notifications","https://irdai.gov.in/circulars","https://irdai.gov.in/orders1","https://irdai.gov.in/exposure-drafts","https://irdai.gov.in/programmes-to-advance-understanding-of-rti","https://irdai.gov.in/cic-orders","https://irdai.gov.in/antimoney-laundering","https://irdai.gov.in/other-communication","https://irdai.gov.in/directory-of-employees","https://irdai.gov.in/warnings-and-penalties",
            "https://uidai.gov.in/en/","https://uidai.gov.in/en/about-uidai/legal-framework.html","https://uidai.gov.in/en/about-uidai/legal-framework/rules.html","https://uidai.gov.in/en/about-uidai/legal-framework/notifications.html","https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html","https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html","https://uidai.gov.in/en/about-uidai/legal-framework/judgements.html","https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation","https://uidai.gov.in/en/about-uidai/legal-framework/updated-rules","https://enforcementdirectorate.gov.in/pmla","https://enforcementdirectorate.gov.in/fema","https://enforcementdirectorate.gov.in/bns","https://enforcementdirectorate.gov.in/bnss","https://enforcementdirectorate.gov.in/bsa"
]

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.

The content may contain tables. Tables are formatted as CSV data and preceded by [TABLE] markers.

Question: {question} 
Context: {context} 
Answer:
"""

embeddings = FastEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)
model = ChatGroq(groq_api_key="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", model_name="llama-3.3-70b-versatile", temperature=0.3)

def extract_tables_from_html(html_content, source_url):
    """Extract tables from HTML content and convert to structured format"""
    import pandas as pd
    from bs4 import BeautifulSoup
    
    # Initialize a list to store all text and tables
    document_content = []
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract tables
    tables = soup.find_all('table')
    for table_num, table in enumerate(tables):
        try:
            # Use pandas to extract table data
            dfs = pd.read_html(str(table))
            for df_num, df in enumerate(dfs):
                # Convert to CSV string
                csv_string = df.to_csv(index=False)
                
                # Add the table as a special content type
                table_content = f"[TABLE] Table {table_num+1}.{df_num+1} from {source_url}:\n{csv_string}"
                document_content.append({
                    "content": table_content,
                    "source": source_url,
                    "type": "table"
                })
        except Exception as e:
            st.error(f"Error extracting table {table_num} from {source_url}: {str(e)}")
    
    return document_content

def load_website(url):
    """Load content from a website URL"""
    try:
        # Use LangChain's WebBaseLoader to get the content
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # Extract the HTML for table processing
        raw_html = documents[0].metadata.get('html', '')
        
        # Extract tables separately
        table_documents = extract_tables_from_html(raw_html, url)
        
        # Convert table documents to LangChain document format
        from langchain_core.documents import Document
        table_docs = []
        
        for item in table_documents:
            # Create a document with the content and metadata
            doc = Document(
                page_content=item["content"],
                metadata={
                    "source": item["source"],
                    "type": item["type"]
                }
            )
            table_docs.append(doc)
        
        # Combine with the original web content
        all_documents = documents + table_docs
        return all_documents, url
    
    except Exception as e:
        st.error(f"Error loading content from {url}: {str(e)}")
        return [], url

def split_text(documents):
    """Split documents into chunks while preserving tables"""
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
        if doc.metadata.get("type") == "table":
            split_docs.append(doc)
        else:
            # Split text documents
            split_docs.extend(text_splitter.split_documents([doc]))
    
    return split_docs

def index_docs(documents):
    """Add documents to the vector store"""
    vector_store.add_documents(documents)

def retrieve_docs(query):
    """Retrieve relevant documents based on query"""
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    """Generate an answer based on retrieved documents"""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({"question": question, "context": context})
    
    return response.content

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Display loading status or control loading
if not st.session_state.documents_loaded:
    if st.button("Load Website Content"):
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_documents = []
        successfully_loaded = []
        
        # Process each website
        for i, website in enumerate(WEBSITES):
            status_text.text(f"Processing {website}...")
            documents, url = load_website(website)
            
            if documents:
                chunked_documents = split_text(documents)
                all_documents.extend(chunked_documents)
                successfully_loaded.append(url)
            
            # Update progress
            progress_bar.progress((i + 1) / len(WEBSITES))
        
        # Index all documents
        if all_documents:
            index_docs(all_documents)
            st.session_state.documents_loaded = True
            
            # Display success message
            st.success(f"Successfully processed {len(successfully_loaded)}/{len(WEBSITES)} websites")
            
            # Display table preview (optional)
            table_docs = [doc for doc in all_documents if doc.metadata.get("type") == "table"]
            if table_docs:
                with st.expander("Preview Extracted Tables"):
                    for i, doc in enumerate(table_docs[:3]):  # Show first 3 tables only
                        st.write(f"**{doc.metadata['source']}**")
                        table_content = doc.page_content.replace("[TABLE] ", "")
                        st.text(table_content)
                        if i < len(table_docs[:3]) - 1:
                            st.divider()
        else:
            st.error("Failed to load any content from the specified websites.")
else:
    # Display list of loaded websites
    st.info("Websites loaded and indexed. Ask questions about their content below.")
    
    # Option to reload
    if st.button("Reload Website Content"):
        st.session_state.documents_loaded = False
        st.experimental_rerun()

# Chat interface (only show after documents are loaded)
if st.session_state.documents_loaded:
    question = st.chat_input("Ask a question about the website content:")
    if question:
        st.session_state.conversation_history.append({"role": "user", "content": question})
        
        related_documents = retrieve_docs(question)
        
        answer = answer_question(question, related_documents)
        
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})

# Display conversation history
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
