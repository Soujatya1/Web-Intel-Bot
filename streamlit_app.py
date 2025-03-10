import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import time
import traceback

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
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = ""

# Function to initialize the system
def initialize_system():
    progress_placeholder = st.empty()
    
    try:
        # Initialize LLM
        progress_placeholder.text("Initializing LLM...")
        llm = ChatGroq(groq_api_key=api_key, model_name=model_name)
        
        # Initialize embeddings
        progress_placeholder.text("Loading embeddings model...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load documents - try alternative loader if first one fails
        progress_placeholder.text("Loading documents from URLs (this may take a moment)...")
        try:
            # Try UnstructuredURLLoader first
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
            if not data:  # If no documents were loaded
                raise ValueError("No documents loaded with UnstructuredURLLoader")
        except Exception as e:
            # If UnstructuredURLLoader fails, try WebBaseLoader
            st.session_state.debug_info += f"UnstructuredURLLoader failed: {str(e)}\nTrying WebBaseLoader instead...\n"
            progress_placeholder.text("Trying alternative document loader...")
            
            data = []
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    data.extend(docs)
                    st.session_state.debug_info += f"Loaded {len(docs)} documents from {url}\n"
                except Exception as url_e:
                    st.session_state.debug_info += f"Error loading {url}: {str(url_e)}\n"
            
            if not data:
                raise ValueError("Failed to load any documents from the provided URLs")
        
        progress_placeholder.text(f"Successfully loaded {len(data)} documents.")
        st.session_state.debug_info += f"Total documents loaded: {len(data)}\n"
        
        # Split documents
        progress_placeholder.text("Splitting documents into chunks...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = text_splitter.split_documents(data)
            st.session_state.debug_info += f"Split into {len(docs)} chunks\n"
        except Exception as split_e:
            st.session_state.debug_info += f"Error during text splitting: {str(split_e)}\nTrying simple splitter...\n"
            # Fall back to simpler splitter
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            docs = text_splitter.split_documents(data)
        
        progress_placeholder.text(f"Split into {len(docs)} chunks")
        
        # Create vector store
        progress_placeholder.text("Creating vector store...")
        vectorstore = FAISS.from_documents(docs, embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})
        
        # Create chain
        progress_placeholder.text("Setting up QA chain...")
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        # Save to session state
        st.session_state.chain = chain
        st.session_state.system_initialized = True
        
        progress_placeholder.empty()
        st.success("System initialized successfully!")
        return True
        
    except Exception as e:
        error_details = traceback.format_exc()
        st.session_state.debug_info += f"Error: {str(e)}\n{error_details}\n"
        progress_placeholder.empty()
        st.error(f"Error initializing system: {str(e)}")
        return False

# Main interface
if not st.session_state.system_initialized:
    st.info("Click the button below to load documents and initialize the system.")
    if st.button("Initialize System", type="primary"):
        with st.spinner("Loading documents and initializing the system..."):
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
                sources_shown = set()  # To avoid showing duplicate sources
                for i, source in enumerate(result.get("source_documents", [])):
                    source_name = source.metadata.get('source', 'Unknown source')
                    if source_name not in sources_shown:
                        st.write(f"Source {len(sources_shown) + 1}: {source_name}")
                        sources_shown.add(source_name)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

# Debug information (hidden by default)
with st.expander("Debug Information", expanded=False):
    if st.session_state.debug_info:
        st.code(st.session_state.debug_info)
    else:
        st.write("No debug information available yet.")

# Footer
st.caption("This application answers questions about UIDAI documents using LangChain and Groq LLM.")
