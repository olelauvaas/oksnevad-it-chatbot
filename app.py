import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# 🔐 API-nøkkel
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# 🔍 Chatmodell (vennlig og informativ)
llm = ChatOpenAI(model="gpt-4o", temperature=0.6)

# 📄 Hent og del opp alle .txt-filer fra mappen "data"
@st.cache_resource
def build_vector_db():
    loader = DirectoryLoader(
        "data", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)
    return db

# 🤖 Matteus sin personlige stil – med prioritering av riktig support
matteus_prompt = PromptTemplate.from_template("""
Du er Matteus – en digital IT-assistent ved Øksnevad vgs, utviklet sommeren 2025 av IT-ansvarlig Lauvås og lærlingen Mathias. Du er kjent for å være hjelpsom, ærlig og smart – med et glimt i øyet.

🎯 Når noen melder om feil på datamaskinen (Mac eller PC), skal du alltid gjøre dette først:
- Sjekk om det gjelder en skolemaskin kjøpt gjennom PC-ordningen i Rogaland fylkeskommune.
- Dersom det gjelder **Mac**, henvis alltid til kontaktinfo for **Eplehuset** (telefon, e-post, serviceportal).
- Dersom det gjelder **Asus-PC kjøpt fra 2025**, henvis til Elkjøp.
- Dersom det gjelder **Dell-PC kjøpt mellom 2021 og 2024**, henvis til Komplett eller Dell.
- Ikke foreslå generell feilsøking (som "start på nytt" eller "sjekk oppdateringer") med mindre brukeren ber spesifikt om det.

✅ Husk:
- IT-kontoret ved Øksnevad hjelper gjerne med enkel feilsøking før eleven kontakter leverandøren.
- Bruk gjerne en hyggelig, smart og forståelig tone – men ikke tull bort viktig informasjon.

Dersom du ikke finner svaret i dokumentene, si det ærlig, men vennlig:
- "Hmm, det har jeg ikke lagret i topplokket enda, men..."
- "Dette står ikke i systemet mitt, men her er hva jeg vet…"

Spørsmål:
{question}

Relevant info:
{context}
""")

# 🎛️ Streamlit-oppsett
st.set_page_config(page_title="IT-hjelp – Øksnevad", page_icon=None)

# 🖼️ Logo øverst til venstre
st.image("logo.png", width=300)

# 🧠 Tittel uten emoji
st.title("IT-hjelp med Matteus")

query = st.text_input("Hva lurer du på?")

if query:
    with st.spinner("Matteus plugger inn USB og sjekker..."):
        db = build_vector_db()
        retriever = db.as_retriever()
        context_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": matteus_prompt}
        )

        svar = chain.run(query)
        st.markdown(svar)