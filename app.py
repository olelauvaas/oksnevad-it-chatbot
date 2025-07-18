import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
import os

# === KONFIG ===
API_NØKKEL = st.secrets["OPENAI_API_KEY"]
FILNAVN = "faq.docx"
CHROMA_DB_DIR = "chroma_db"

# === Leser Word-dokument og bygger vector DB ===
@st.cache_resource
def bygg_vector_db():
    doc = Document(FILNAVN)
    tekst = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    dokumenter = [LCDocument(page_content=chunk) for chunk in splitter.split_text(tekst)]
    embeddings = OpenAIEmbeddings(openai_api_key=API_NØKKEL)
    db = Chroma.from_documents(dokumenter, embeddings, persist_directory=CHROMA_DB_DIR)
    return db

# === Henter svar ===
def spør_bot(spørsmål, db):
    chat = ChatOpenAI(model="gpt-4o", openai_api_key=API_NØKKEL)
    docs_med_score = db.similarity_search_with_score(spørsmål, k=4)
    relevante_docs = [doc for doc, score in docs_med_score if score < 0.75]

    if relevante_docs:
        innhold = "\n".join([doc.page_content for doc in relevante_docs])
        meldinger = [
            SystemMessage(content="""
Du er en smart og entusiastisk assistent som hjelper elever og lærere med digitale spørsmål.
Svar tydelig og engasjerende, gjerne med eksempler og små metaforer for å gjøre det forståelig.
Bruk maks 3–4 avsnitt. Vær gjerne litt morsom hvis det passer, men hold tonen profesjonell og vennlig.
Svar som om du hadde svart direkte i ChatGPT – uten å nevne dokumenter eller kildebegrensninger.
"""),
            HumanMessage(content=f"Denne informasjonen kan hjelpe: \n{innhold}\n\nSpørsmål: {spørsmål}")
        ]
    else:
        meldinger = [
            SystemMessage(content="""
Du er en smart og entusiastisk assistent som hjelper elever og lærere med digitale spørsmål.
Svar tydelig og engasjerende, gjerne med eksempler og små metaforer for å gjøre det forståelig.
Bruk maks 3–4 avsnitt. Vær gjerne litt morsom hvis det passer, men hold tonen profesjonell og vennlig.
Svar som om du hadde svart direkte i ChatGPT – uten å nevne dokumenter eller kildebegrensninger.
"""),
            HumanMessage(content=spørsmål)
        ]

    svar = chat.invoke(meldinger)
    return svar.content

# === Streamlit-grensesnitt ===
st.set_page_config(page_title="IT-kontoret på Øksnevad vgs", page_icon="💻")
st.image("logo.png", width=200)
st.title("💻 IT-kontoret på Øksnevad vgs")
st.write("Stil et spørsmål, og få et klart og godt svar.")

db = bygg_vector_db()

# Ferdige spørsmål
st.subheader("🔎 Vanlige spørsmål")
spørsmål_valg = [
    "Hva er PC-ordningen?",
    "Hvordan kobler jeg til printer?",
    "Hvordan logger jeg inn på Teams?",
    "Hvordan logger jeg på WiFi?",
    "Hva gjør jeg hvis PC-en er ødelagt?"
]

if "valgt_spørsmål" not in st.session_state:
    st.session_state.valgt_spørsmål = ""

for spm in spørsmål_valg:
    if st.button(spm):
        st.session_state.valgt_spørsmål = spm

spørsmål = st.text_input("Eller skriv inn et annet spørsmål:", value=st.session_state.valgt_spørsmål)

if spørsmål:
    with st.spinner("🤔 Tenker litt..."):
        svar = spør_bot(spørsmål, db)
    st.success("💬 Svar:")
    st.write(svar)