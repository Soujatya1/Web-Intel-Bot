import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Hardcoded websites
websites = [
    "https://irdai.gov.in/acts",
    "https://irdai.gov.in/rules",
    "https://irdai.gov.in/consolidated-gazette-notified-regulations",
    "https://irdai.gov.in/notifications",
    "https://irdai.gov.in/circulars",
    "https://irdai.gov.in/orders1",
    "https://irdai.gov.in/exposure-drafts",
    "https://irdai.gov.in/programmes-to-advance-understanding-of-rti",
    "https://irdai.gov.in/cic-orders",
    "https://irdai.gov.in/antimoney-laundering",
    "https://irdai.gov.in/other-communication",
    "https://irdai.gov.in/directory-of-employees",
    "https://irdai.gov.in/warnings-and-penalties"
]

# Function to create vector store from websites
def load_websites_to_vectorstore():
    all_docs = []
    loaded_sites = 0
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    progress_bar = st.progress(0)
    
    for idx, website in enumerate(websites):
        try:
            status_placeholder = st.empty()
            status_placeholder.write(f"Fetching content from: {website}")
            
            # Load documents
            loader = WebBaseLoader(website)
            docs = loader.load()
            
            # Add source metadata to each document
            for doc in docs:
                doc.metadata["source"] = website
            
            # Split documents into chunks
            split_docs = text_splitter.split_documents(docs)
            all_docs.extend(split_docs)
            loaded_sites += 1
            
            # Update progress
            progress_bar.progress((idx + 1) / len(websites))
            status_placeholder.write(f"✅ Loaded: {website}")
            
        except Exception as e:
            st.write(f"❌ Error loading {website}: {e}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create FAISS vector store
    st.write("Creating vector embeddings and FAISS index...")
    vector_store = FAISS.from_documents(all_docs, embeddings)
    st.session_state.vector_store = vector_store
    
    return loaded_sites, len(all_docs)

# LLM Initialization
@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        top_p=0.2,
    )

# Create prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a Website Intelligence specialist who answers questions based on the content from the following websites:
    - IRDAI (Insurance Regulatory and Development Authority of India)
    - eGazette
    - Enforcement Directorate - PMLA
    - UIDAI (Unique Identification Authority of India)

    Consider "rules", "acts" to be keywords from the questions.
    
    Answer the question based only on the information provided in the context. Be precise and accurate.
    If the provided context doesn't contain the answer, clearly state that you don't have enough information.
    Include any relevant hyperlinks from the documents in your response.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    # Load content button
    if st.button("Load Website Content"):
        with st.spinner("Loading website content..."):
            num_sites, num_chunks = load_websites_to_vectorstore()
            st.success(f"Loaded content from {num_sites} websites, created {num_chunks} document chunks")
            st.write("Vector database successfully created!")
            
    # Display status
    if st.session_state.vector_store is not None:
        st.success("Database loaded and ready for queries")
    else:
        st.warning("Please load website content first")

with col2:
    # User query interface
    query = st.text_input("Enter your query:")
    k_docs = st.slider("Number of relevant documents to retrieve", min_value=1, max_value=10, value=4)
    
    if st.button("Get Answer"):
        if query and st.session_state.vector_store is not None:
            with st.spinner("Retrieving relevant documents and generating answer..."):
                # Get the LLM
                llm = get_llm()
                
                # Create a retriever from the vector store
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k_docs}
                )
                
                # Create document chain
                document_chain = create_stuff_documents_chain(llm, prompt)
                
                # Create retrieval chain
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Process the query
                response = retrieval_chain.invoke({"input": query})
                
                # Display retrieved documents
                st.subheader("Retrieved Documents:")
                for i, doc in enumerate(response["context"]):
                    with st.expander(f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')})"):
                        st.write(doc.page_content)
                
                # Display final response
                st.subheader("Response:")
                st.write(response["answer"])
        else:
            st.warning("Please enter a query and load website content first.")
