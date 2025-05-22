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
          st.write(f"Loading URL: {url}")
          loader = WebBaseLoader(url)
          docs = loader.load()

          for doc in docs:
              doc.metadata["source"] = url

          st.session_state['loaded_docs'].extend(docs)
          st.write(f"Successfully loaded {len(docs)} document(s) from {url}")
          
          # Display extracted content for each document
          for i, doc in enumerate(docs):
              with st.expander(f"Content from {url} (Document {i+1})"):
                  st.write("**Metadata:**")
                  st.json(doc.metadata)
                  st.write("**Content Preview (first 1000 characters):**")
                  st.text(doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content)
                  st.write(f"**Total Content Length:** {len(doc.page_content)} characters")
                  
       except Exception as e:
          st.error(f"Error loading {url}: {e}")
  
   st.success(f"Total loaded documents: {len(st.session_state['loaded_docs'])}")
  
   # Show summary of all loaded content
   if st.session_state['loaded_docs']:
       st.write("## Content Summary")
       total_chars = sum(len(doc.page_content) for doc in st.session_state['loaded_docs'])
       st.write(f"- Total documents: {len(st.session_state['loaded_docs'])}")
       st.write(f"- Total characters: {total_chars:,}")
       
       # Show content breakdown by source
       source_breakdown = {}
       for doc in st.session_state['loaded_docs']:
           source = doc.metadata.get('source', 'Unknown')
           if source not in source_breakdown:
               source_breakdown[source] = {'docs': 0, 'chars': 0}
           source_breakdown[source]['docs'] += 1
           source_breakdown[source]['chars'] += len(doc.page_content)
       
       st.write("### Content by Source:")
       for source, stats in source_breakdown.items():
           st.write(f"- **{source}**: {stats['docs']} documents, {stats['chars']:,} characters")
  
   # LLM and Embeddings Initialization
   if api_key and st.session_state['loaded_docs']:
       with st.spinner("Processing documents..."):
           llm = ChatGroq(groq_api_key=api_key, model_name='llama3-70b-8192', temperature=0.2, top_p=0.2)
           hf_embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    
           # Craft ChatPrompt Template
           prompt = ChatPromptTemplate.from_template(
              """
               You are a website expert, who answers questions as per the websites entered and the correct set of documents retrieved.
    
               IMPORTANT: Pay attention to the "dates" on the websites for correct answering.
     
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
           st.write(f"Number of chunks created: {len(document_chunks)}")
           
           # Show chunk information
           with st.expander("View Document Chunks"):
               for i, chunk in enumerate(document_chunks[:5]):  # Show first 5 chunks
                   st.write(f"**Chunk {i+1}:**")
                   st.write(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                   st.write(f"Content: {chunk.page_content[:200]}...")
                   st.write("---")
               if len(document_chunks) > 5:
                   st.write(f"... and {len(document_chunks) - 5} more chunks")
     
           # Vector database storage
           st.session_state['vector_db'] = FAISS.from_documents(document_chunks, hf_embedding)
     
           # Stuff Document Chain Creation
           document_chain = create_stuff_documents_chain(llm, prompt)
     
           # Retriever from Vector store
           retriever = st.session_state['vector_db'].as_retriever()
     
           # Create a retrieval chain
           st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)
           
           st.success("Documents processed and ready for querying!")
 
# Query Section
st.write("## Ask Questions")
query = st.text_input("Enter your query:")

# Option to show retrieved documents
show_retrieved = st.checkbox("Show retrieved documents with answer")

if st.button("Get Answer") and query:
   if st.session_state['retrieval_chain']:
       with st.spinner("Searching and generating answer..."):
           response = st.session_state['retrieval_chain'].invoke({"input": query})
           
           st.write("## Response:")
           st.write(response['answer'])
           
           # Show retrieved documents if requested
           if show_retrieved and 'context' in response:
               st.write("## Retrieved Documents:")
               retrieved_docs = response.get('context', [])
               for i, doc in enumerate(retrieved_docs):
                   with st.expander(f"Retrieved Document {i+1} from {doc.metadata.get('source', 'Unknown')}"):
                       st.write(doc.page_content)
   else:
       st.warning("Please load and process documents first.")
