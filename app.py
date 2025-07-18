import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

@st.cache_resource
def build_vector_db():
    loader = Docx2txtLoader("faq.docx")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    return db

st.title("📚 IT-hjelp – Øksnevad chatbot")
query = st.text_input("Hva lurer du på?")

if query:
    db = build_vector_db()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
    result = qa_chain.run(query)
    st.write(result)
