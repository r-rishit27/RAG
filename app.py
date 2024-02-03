import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.readers import PDFReader
import tempfile
import os
import torch


llm = OpenAI(api_key="YOUR API KEY",temperature=0.25 , model="gpt-3.5-turbo")
 
# Function to handle PDF upload
# Streamlit App
st.title("Custom Document Q&A RAG App 👨‍💻")

# PDF Upload Section
st.subheader("Upload Custom PDF")

# Load documents if PDF file is uploaded
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Use the temporary file as the PDF document
    pdf_reader = PDFReader(return_full_document=True)
    documents = pdf_reader.load_data(file=temp_file_path)


    # LLama setup
    system_prompt = """
                           You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
    """

    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    # HuggingFace LLama

    # Embeddings setup
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    # Vector Store Index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # User Query Section
st.header("Ask a Question")
user_query = st.text_input("Enter your question:")
if user_query:
        # Perform the query
        response = query_engine.query(user_query)
        st.write("Answer:", str(response))

# Additional features or UI components can be added as needed
