import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_content" not in st.session_state:
    st.session_state.loaded_content = ""

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

# Function to load website content
def load_websites():
    all_content = []
    
    for website in websites:
        try:
            st.write(f"Fetching content from: {website}")
            loader = WebBaseLoader(website)
            docs = loader.load()
            
            # Extract text content and add source information
            for doc in docs:
                content = doc.page_content
                content += f"\n(Source: {website})"
                all_content.append(content)
                
        except Exception as e:
            st.write(f"Error loading {website}: {e}")
    
    # Join all content into a single string
    combined_content = "\n\n".join(all_content)
    st.session_state.loaded_content = combined_content
    
    return len(all_content)

# Load content button
if st.button("Load Website Content"):
    with st.spinner("Loading website content..."):
        num_docs = load_websites()
        st.success(f"Loaded content from {num_docs} pages")

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
    You are a Website Intelligence specialist who answers questions based on the content from the following websites:
    - IRDAI (Insurance Regulatory and Development Authority of India)
    - eGazette
    - Enforcement Directorate - PMLA
    - UIDAI (Unique Identification Authority of India)
    
    Please answer precisely and also extract hyperlinks and display, if applicable.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    
    Answer the question based only on the information provided in the context.
    """
)

# User query interface
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if query and st.session_state.loaded_content:
        with st.spinner("Generating answer..."):
            # Pass the content directly to the LLM
            response = llm.invoke(
                prompt.format(
                    input=query,
                    context=st.session_state.loaded_content
                )
            )
            
            # Display response
            st.subheader("Response:")
            st.write(response.content)
    else:
        st.warning("Please enter a query and load website content first.")
