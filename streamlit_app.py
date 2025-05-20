import streamlit as st
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from bs4 import BeautifulSoup

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

# Custom loader to extract better content from websites
class EnhancedWebLoader(WebBaseLoader):
    def load(self):
        docs = super().load()
        enhanced_docs = []
        
        for doc in docs:
            # Extract the text content
            content = doc.page_content
            source_url = doc.metadata.get("source", "")
            
            # Parse with BeautifulSoup to better extract text and links
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract all links for later reference
            links = {}
            for a_tag in soup.find_all('a', href=True):
                link_text = a_tag.get_text().strip()
                href = a_tag['href']
                if link_text and href:
                    # Make relative URLs absolute
                    if href.startswith('/'):
                        base_url = '/'.join(source_url.split('/')[:3])  # Get domain part
                        href = base_url + href
                    links[link_text] = href
            
            # Extract text content with better formatting
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            clean_text = soup.get_text(separator=' ', strip=True)
            
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # Create a new document with enhanced metadata
            enhanced_doc = doc
            enhanced_doc.page_content = clean_text
            enhanced_doc.metadata.update({
                "source": source_url,
                "links": links,
                "title": soup.title.string if soup.title else "No Title"
            })
            
            enhanced_docs.append(enhanced_doc)
            
        return enhanced_docs

# Function to create vector store from websites
def load_websites_to_vectorstore():
    all_docs = []
    loaded_sites = 0
    
    # Initialize text splitter with smaller chunks and more overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True
    )
    
    progress_bar = st.progress(0)
    
    for idx, website in enumerate(websites):
        try:
            status_placeholder = st.empty()
            status_placeholder.write(f"Fetching content from: {website}")
            
            # Load documents with enhanced loader
            loader = EnhancedWebLoader(website)
            docs = loader.load()
            
            # Split documents into chunks
            split_docs = text_splitter.split_documents(docs)
            all_docs.extend(split_docs)
            loaded_sites += 1
            
            # Update progress
            progress_bar.progress((idx + 1) / len(websites))
            status_placeholder.write(f"✅ Loaded: {website}")
            
        except Exception as e:
            st.write(f"❌ Error loading {website}: {e}")
    
    # Initialize embeddings - using a more powerful model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
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
    k_docs = st.slider("Number of relevant documents to retrieve", min_value=1, max_value=15, value=5)
    
    if st.button("Get Answer"):
        if query and st.session_state.vector_store is not None:
            with st.spinner("Retrieving relevant documents and generating answer..."):
                # Get the LLM
                llm = get_llm()
                
                # Use MMR retrieval for better diversity of results
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={
                        "k": k_docs,
                        "fetch_k": k_docs * 2,  # Fetch more docs initially then select diverse subset
                        "lambda_mult": 0.7  # Balance between relevance and diversity (0-1)
                    }
                )
                
                # Create improved prompt
                improved_prompt = ChatPromptTemplate.from_template(
                    """
                    You're an expert in IRDAI (Insurance Regulatory and Development Authority of India) regulations, 
                    policies, and related information. Your task is to provide accurate information based ONLY on 
                    the context provided.
                    
                    Context information from IRDAI websites:
                    {context}
                    
                    Query: {query}
                    
                    Follow these instructions carefully:
                    1. Answer ONLY from the provided context - do not use prior knowledge
                    2. Be concise but thorough
                    3. If the answer isn't in the context, clearly state that you don't have that information
                    4. Include relevant links from the context in your response
                    5. Format your answer for readability with headings and bullet points if appropriate
                    
                    Your response:
                    """
                )
                
                # Add query rewriting for better retrieval
                rewrite_prompt = ChatPromptTemplate.from_template(
                    """
                    You are an expert in information retrieval from insurance regulatory websites.
                    Your task is to rewrite the following query to make it more effective for retrieving 
                    relevant information from IRDAI (Insurance Regulatory and Development Authority of India) documents.
                    
                    Original query: {query}
                    
                    Rewrite this query to be more specific, including key insurance regulatory terms and 
                    concepts that would appear in official IRDAI documents. Do not change the fundamental 
                    intent of the query, just optimize it for retrieval.
                    
                    Rewritten query:
                    """
                )
                
                # Define query rewriting chain
                query_rewriter = rewrite_prompt | llm | StrOutputParser()
                
                # Create the RAG chain with query rewriting
                rag_chain = (
                    {"query": query_rewriter, "original_query": RunnablePassthrough()}
                    | {"context": retriever, "query": lambda x: x["original_query"]}
                    | improved_prompt
                    | llm
                )
                
                # Process the query
                st.subheader("Query Processing")
                with st.expander("Query Analysis", expanded=True):
                    st.write("Original Query:", query)
                    with st.spinner("Rewriting query for better retrieval..."):
                        rewritten_query = query_rewriter.invoke({"query": query})
                        st.write("Rewritten Query:", rewritten_query)
                
                # Get retrieved documents
                with st.spinner("Retrieving relevant documents..."):
                    retrieved_docs = retriever.invoke(rewritten_query)
                    
                    # Display retrieved documents
                    st.subheader("Retrieved Documents:")
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get('source', 'Unknown')
                        title = doc.metadata.get('title', f'Document {i+1}')
                        
                        with st.expander(f"{i+1}. {title} (Source: {source})"):
                            st.write(doc.page_content)
                            
                            # Display links if available
                            links = doc.metadata.get('links', {})
                            if links:
                                st.write("---")
                                st.write("**Links:**")
                                for link_text, url in links.items():
                                    st.write(f"- [{link_text}]({url})")
                
                # Generate final response
                with st.spinner("Generating comprehensive answer..."):
                    response = rag_chain.invoke({
                        "original_query": query
                    })
                    
                    # Display final response
                    st.subheader("Final Answer:")
                    st.markdown(response.content)
        else:
            st.warning("Please enter a query and load website content first.")
