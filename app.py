# === app.py ===
import streamlit as st
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === KONFIG ===
API_NOKKEL = "din-openai-api-nokkel-her"
FILNAVN = "faq.docx"

# === Leser Word-dokument ===
@st.cache_resource
def bygg_vector_db():
    doc = Document(FILNAVN)
    tekst = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    dokumenter = [LCDocument(page_content=chunk) for chunk in splitter.split_text(tekst)]
    embedding_model = OpenAIEmbeddings(openai_api_key=API_NOKKEL)
    db = FAISS.from_documents(dokumenter, embedding_model)
    return db

# === Funksjon for å spørre boten ===
def spor_bot(sporsmal, db):
    chat = ChatOpenAI(model="gpt-4o", openai_api_key=API_NOKKEL)
    docs_med_score = db.similarity_search_with_score(sporsmal, k=3)
    relevante_docs = [doc for doc, score in docs_med_score if score < 0.7]

    if relevante_docs:
        innhold = "\n".join([doc.page_content for doc in relevante_docs])
        meldinger = [
            SystemMessage(content="""
            Du er en smart og entusiastisk assistent som hjelper elever og lærere med digitale spørsmål.
            Svar tydelig og engasjerende, gjerne med eksempler og små metaforer for å gjøre det forståelig.
            Vær kort og presis når det passer, men svar utfyllende når det trengs.
            Ikke referer til dokumenter eller at informasjon mangler – bare svar så godt du kan.
            """),
            HumanMessage(content=f"Dokumentinnhold:\n{innhold}\n\nSpørsmål: {sporsmal}")
        ]
    else:
        meldinger = [
            SystemMessage(content="""
            Du er en smart og entusiastisk assistent som hjelper elever og lærere med digitale spørsmål.
            Svar tydelig og engasjerende, gjerne med eksempler og små metaforer.
            Ikke nevn manglende informasjon. Svar på alt du kan som en kunnskapsrik og vennlig hjelper.
            """),
            HumanMessage(content=sporsmal)
        ]

    svar = chat.invoke(meldinger)
    return svar.content

# === Streamlit-grensesnitt ===
st.set_page_config(page_title="IT-kontoret på Øksnevad vgs", page_icon="💻")
st.image("logo.png", width=200)
st.title("💻 IT-kontoret på Øksnevad vgs")
st.write("Stil et spørsmål og få svar med en gang.")

db = bygg_vector_db()

st.subheader("🔎 Vanlige spørsmål")
sporsmal_valg = [
    "Hva er PC-ordningen?",
    "Hvor laster jeg ned Office?",
    "Hvordan kobler jeg til printer?",
    "Hvordan logger jeg inn på Teams?",
    "Hva gjør jeg hvis PC-en er ødelagt?"
]

if "valgt_sporsmal" not in st.session_state:
    st.session_state.valgt_sporsmal = ""

for spm in sporsmal_valg:
    if st.button(spm):
        st.session_state.valgt_sporsmal = spm

sporsmal = st.text_input(
    "Eller skriv inn et annet spørsmål:",
    value=st.session_state.valgt_sporsmal,
    placeholder="Eks: Hvordan får jeg tilgang til e-post?"
)

if sporsmal:
    with st.spinner("🤔 Tenker litt..."):
        svar = spor_bot(sporsmal, db)
    st.success("💬 Svar fra bot:")
    st.write(svar)
