import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import WebBaseLoader
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# Hardcoded websites
websites = ["https://irdai.gov.in/", "https://egazette.gov.in/", "https://enforcementdirectorate.gov.in/pmla", "https://uidai.gov.in/"]

def fetch_website_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            return soup.get_text()
        else:
            return f"Failed to fetch {url}, Status Code: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

loaded_docs = []

for website in websites:
    try:
        st.write(f"Fetching content from: {website}")
        loader = WebBaseLoader(website)
        docs = loader.load()

        # Debugging step: Check what docs contains
        st.write(f"Loaded from {website}: {type(docs)}, Length: {len(docs)}")

        if isinstance(docs, list):  # If WebBaseLoader returns a list
            for content in docs:
                if isinstance(content, Document):  
                    content.metadata["source"] = website  # Ensure metadata is set
                    loaded_docs.append(content)
                elif isinstance(content, str):  
                    loaded_docs.append(Document(page_content=content, metadata={"source": website}))
        elif isinstance(docs, str):  
            loaded_docs.append(Document(page_content=docs, metadata={"source": website}))

    except Exception as e:
        st.write(f"Error loading {website}: {e}")

st.write(f"Total Loaded Documents: {len(loaded_docs)}")
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
    You are a Website Intelligence specialist who answers questions from the websites provided.

    Please answer precisely and include hyperlinks where applicable.

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

if st.session_state.loaded_docs:
    document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)
    st.write(f"Total Document Chunks: {len(document_chunks)}")

# Ensure documents are loaded before processing
if st.session_state.loaded_docs:
    document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Save document chain to session state
    st.session_state.retrieval_chain = document_chain

query = st.text_input("Enter your query:")
if query and st.session_state.loaded_docs:
    context = "\n".join(
        [doc.page_content for doc in st.session_state.loaded_docs if isinstance(doc, Document)]
    )

    response = st.session_state.retrieval_chain.invoke({"input": query, "context": context})

    st.write("Response:")
    if isinstance(response, dict) and "answer" in response:
        st.write(response["answer"])
    else:
        st.write("Here's your response!")
else:
    st.write("No documents loaded. Please check the website fetching.")
