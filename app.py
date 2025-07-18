
import streamlit as st
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Setup OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to build vector DB
@st.cache_resource
def build_vector_db():
    loader = UnstructuredWordDocumentLoader("oksnevad-it.docx")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    return db

# Build vector DB
db = build_vector_db()

# Create the QA chain
retriever = db.as_retriever()
llm = ChatOpenAI(openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("📚 Øksnevad IT Chatbot")
st.markdown("Still et spørsmål om IT-tjenester på Øksnevad:")

query = st.text_input("Hva lurer du på?")
if query:
    response = qa_chain.run(query)
    st.write(response)
