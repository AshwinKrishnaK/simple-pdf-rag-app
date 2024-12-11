import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = "SIMPLE RAG CHAT APP USING GROQ"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key,model='gemma2-9b-it')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the context only.
    Please provide accurate response based on question.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

def create_vector_embedding(pdf_file):
    if "vector" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
           temp_file.write(pdf_file.getbuffer())
           temp_file_path = temp_file.name
        st.session_state.embedding = OpenAIEmbeddings()
        st.session_state.document_loader = PyPDFLoader(temp_file_path)
        st.session_state.docs = st.session_state.document_loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_docs,st.session_state.embedding)

user_input = st.text_input("Enter your query!")

pdf_file = st.file_uploader("Upload a PDF File",type='pdf')

if pdf_file is not None:
    if st.button("Document Embedding"):
        create_vector_embedding(pdf_file)
        st.write("Embedding done!")

if user_input:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriver = st.session_state.vector_store.as_retriever()
    retrievel_chain = create_retrieval_chain(retriver,document_chain)
    response = retrievel_chain.invoke({'input':user_input})
    st.write(response['answer'])
