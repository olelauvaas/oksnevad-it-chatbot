import streamlit as st
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 🔐 Hent og sett OpenAI-nøkkel
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# 🔮 LLM med OpenAI
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

# 📚 Funksjon for å bygge vektorbasen fra dokumentet
@st.cache_resource
def build_vector_db():
    loader = Docx2txtLoader("faq.docx")  # Bytt evt. filnavn her
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    return db

# 🖼️ Brukergrensesnitt i Streamlit
st.title("📚 IT-hjelp – Øksnevad Chatbot")

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
