import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

# Set page title
st.set_page_config(page_title="UIDAI Document QA System")
st.title("UIDAI Document QA System")

# Fixed configuration
urls = [
    'https://uidai.gov.in/en/about-uidai/legal-framework/circulars.html',
    'https://uidai.gov.in/en/about-uidai/legal-framework/updated-regulation.html'
]
api_key = "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"
model_name = "llama-3.3-70b-versatile"
chunk_size = 1000
chunk_overlap = 200
k_value = 4

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'chain' not in st.session_state:
    st.session_state.chain = None

# Function to initialize the system
def initialize_system():
    with st.spinner("Loading documents and initializing the system..."):
        try:
            # Initialize LLM
            llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
            
            # Initialize embeddings
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Load documents
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            
            # Split documents
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(data)
            
            # Create vector store
            vectorstore = FAISS.from_documents(docs, embedding)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
            
            # Create chain
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            
            # Save to session state
            st.session_state.chain = chain
            st.session_state.system_initialized = True
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return False

# Main interface
if not st.session_state.system_initialized:
    st.info("Click the button below to load documents and initialize the system.")
    if st.button("Initialize System", type="primary"):
        initialize_system()
else:
    st.success("System is ready! Ask any question about UIDAI documents.")
    
    # Question input
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Get answer
                result = st.session_state.chain({"question": question})
                
                # Display answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Display sources in a simple way
                st.subheader("Sources")
                for i, source in enumerate(result.get("source_documents", [])):
                    st.write(f"Source {i+1}: {source.metadata.get('source', 'Unknown source')}")
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")

# Footer
st.caption("This application answers questions about UIDAI documents using LangChain and Groq LLM.")
