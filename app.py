# app.py
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
import os

st.set_page_config(page_title="IT-kontoret på Øksnevad vgs", page_icon="💻")
st.title("💬 IT-kontoret på Øksnevad vgs")
st.write("Spør om alt fra WiFi til programmer – jeg svarer raskt og presist!")

# --- API-nøkkel ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# --- Last inn dokument ---
@st.cache_resource
def build_vector_db():
    loader = Docx2txtLoader("faq.docx")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# --- Spør chatbot ---
def spør_bot(spørsmål, db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    svar = chain.run(spørsmål)
    return svar

# --- Initialiser DB ---
db = build_vector_db()

# --- Vanlige spørsmål ---
st.write("### 🤔 Vanlige spørsmål:")
kspm = st.columns(3)
spørsmål_liste = [
    "Hvordan logger jeg på skolens WiFi?",
    "Hva gjør jeg hvis PC-en ikke virker?",
    "Hvordan får jeg nytt passord til Feide?",
    "Hva er PC-ordningen?",
    "Hvordan bruker jeg Teams på mobilen?",
    "Hva gjør jeg hvis jeg har glemt PC-laderen hjemme?"
]

for i, spm in enumerate(spørsmål_liste):
    if kspm[i % 3].button(spm):
        med_vent = st.spinner("💭 Tenker litt...")
        with med_vent:
            svar = spør_bot(spm, db)
        st.success("Svar:")
        st.write(svar)

# --- Egne spørsmål ---
st.write("### ✍️ Eller skriv ditt eget spørsmål:")
brukerspørsmål = st.text_input("Hva lurer du på?")

if brukerspørsmål:
    med_vent = st.spinner("🔍 Søker etter svar...")
    with med_vent:
        svar = spør_bot(brukerspørsmål, db)
    st.success("Svar:")
    st.write(svar)
