import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
 
# Initialize session state variables
if 'loaded_docs' not in st.session_state:
  st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
  st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
  st.session_state['retrieval_chain'] = None
 
# Streamlit UI
st.title("Website Intelligence")
 
api_key = "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"
 
websites_input = st.text_area("Enter website URLs (one per line):")
 
if st.button("Load and Process"):
   website_urls = websites_input.splitlines()
   st.session_state['loaded_docs'] = []
  
   for url in website_urls:
       try:
          #st.write(f"Loading URL: {url}")
          loader = WebBaseLoader(url)
          docs = loader.load()

          for doc in docs:
              doc.metadata["source"] = url

          st.session_state['loaded_docs'].extend(docs)
          #st.write("Successfully loaded document")
       except Exception as e:
          st.write(f"Error loading {url}: {e}")
  
   st.write(f"Loaded documents: {len(st.session_state['loaded_docs'])}")
  
   # LLM and Embeddings Initialization
   if api_key:
       llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
       hf_embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
 
       # Craft ChatPrompt Template
       prompt = ChatPromptTemplate.from_template(
          """
           You are a website expert, who answers questions as per the websites entered and the correct set of documents retrieved.
 
           Do not skip any information from the context. Answer appropriately as per the query asked.
 
           Based on the information above, compare the policies across websites, focusing on key features like premiums, coverage, riders, etc. Structure your comparison clearly in a tabular format if needed.
 
          <context>
          {context}
          </context>
 
          Question: {input}"""
       )
      
       # Text Splitting
       text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=2000,
          chunk_overlap=100,
          length_function=len,
       )
 
       document_chunks = text_splitter.split_documents(st.session_state['loaded_docs'])
       st.write(f"Number of chunks: {len(document_chunks)}")
 
       # Vector database storage
       st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
 
       # Stuff Document Chain Creation
       document_chain = create_stuff_documents_chain(llm, prompt)
 
       # Retriever from Vector store
       retriever = st.session_state['vector_db'].as_retriever()
 
       # Create a retrieval chain
       st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
 
# Query Section
query = st.text_input("Enter your query:")
if st.button("Get Answer") and query:
   if st.session_state['retrieval_chain']:
       response = st.session_state['retrieval_chain'].invoke({"input": query})
       st.write("Response:")
       st.write(response['answer'])
   else:
       st.write("Please load and process documents first.")
