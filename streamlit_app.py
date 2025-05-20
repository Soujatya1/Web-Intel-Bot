import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import WebBaseLoader

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# Hardcoded websites
websites = [
    "https://irdai.gov.in/",
    "https://egazette.gov.in/",
    "https://enforcementdirectorate.gov.in/pmla",
    "https://uidai.gov.in/"
]

loaded_docs = []

# Load content from websites
for website in websites:
    try:
        st.write(f"Fetching content from: {website}")
        loader = WebBaseLoader(website)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = website
        loaded_docs.extend(docs)
    except Exception as e:
        st.write(f"Error loading {website}: {e}")

st.write(f"Loaded documents: {len(loaded_docs)}")

# Store loaded documents in session state
st.session_state.loaded_docs = loaded_docs

# LLM Initialization
llm = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    top_p=0.2,
)

# ChatPrompt Template
prompt = ChatPromptTemplate.from_template(
    """
    You are a Website Intelligence specialist who answers questions as asked from the websites uploaded.
    Please answer precisely and also extract hyperlinks and display, if applicable.
    <context>
    {context}
    </context>
    Question: {input}"""
)

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)

# Create embeddings and vector store for semantic search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(document_chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Stuff Document Chain Creation
document_chain = create_stuff_documents_chain(llm, prompt)

# Save document chain to session state
st.session_state.retrieval_chain = document_chain

query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query and st.session_state.loaded_docs:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Pass retrieved documents to the chain
        response = st.session_state.retrieval_chain.invoke({
            "input": query, 
            "context": "\n".join([doc.page_content for doc in relevant_docs])
        })
        
        # Display response
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a query and ensure documents are loaded.")
