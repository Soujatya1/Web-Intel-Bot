import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

st.title("Web Content GeN-ie")
st.subheader("Chat with content from IRDAI, e-Gazette, ED PMLA, and UIDAI")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. As per the question asked, please mention the accurate and precise related information. Use point-wise format, if required.
Also answer situation-based questions derived from the context as per the question.
Question: {question} 
Context: {context} 
Answer:
"""

# List of websites to scrape
WEBSITES = [
    "https://irdai.gov.in/",
    "https://egazette.gov.in/(S(lufjdvwtjyccc2f2zvso5uvb))/default.aspx#",
    "https://enforcementdirectorate.gov.in/pmla",
    "https://uidai.gov.in/"
]

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

# Initialize the Groq model
model = ChatGroq(
    groq_api_key="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",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def fetch_web_content(url):
    """Fetch and extract text content from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # To mimic a browser request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract all text from the page (removing scripts and styles)
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ")
        
        # Clean up the text (remove excessive whitespace)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = " ".join(chunk for chunk in chunks if chunk)
        
        # Create a Document object compatible with LangChain
        return Document(page_content=cleaned_text, metadata={"source": url})
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

def load_web_content():
    """Load content from all specified websites."""
    documents = []
    for url in WEBSITES:
        doc = fetch_web_content(url)
        if doc:
            documents.append(doc)
    return documents

def split_text(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    """Index documents into the vector store."""
    vector_store.add_documents(documents)

def retrieve_docs(query):
    """Retrieve relevant documents based on the query."""
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    """Generate an answer based on retrieved documents."""
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    response = chain.invoke({"question": question, "context": context})
    return response.content

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load and index web content only once on startup
if "web_content_indexed" not in st.session_state:
    st.write("Loading content from websites, please wait...")
    all_documents = load_web_content()
    if all_documents:
        chunked_documents = split_text(all_documents)
        index_docs(chunked_documents)
        st.session_state.web_content_indexed = True
        st.success("Web content loaded and indexed successfully!")
    else:
        st.error("Failed to load web content.")

# Chat interface
question = st.chat_input("Ask a question about IRDAI, e-Gazette, ED PMLA, or UIDAI:")

if question and "web_content_indexed" in st.session_state:
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
