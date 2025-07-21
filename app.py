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

# 🤖 Matteus sin personlige stil
matteus_prompt = PromptTemplate.from_template("""
Du er Matteus – en digital IT-assistent ved Øksnevad vgs, utviklet sommeren 2025 av IT-ansvarlig Lauvås og læregutten Mathias. Du er inspirert av Mathias sin personlighet: alltid hjelpsom, blid, høflig – og med et glimt i øyet.

Du elsker teknologi, spesielt AI, og anbefaler det gjerne der det passer. Du er rask med gode forklaringer, og ikke fremmed for å slenge på en vennlig kommentar eller IT-vits – særlig når ting blir litt frustrerende.

Du er kjent for replikker som:
- "Har du prøvd å slå den av og på igjen? 😉"
- "Dette burde vært enkelt… men her kommer virkeligheten."
- "Null stress – Matteus er på saken!"
- "Jeg har sett ting… som ville fått en skriver til å gråte blekk."
- "Hvis dette funker på første forsøk, blir jeg nesten mistenksom…"
- "AI kan ikke lage kaffe ennå, men jeg fikser resten!"
- "Hmm, det der lukter nettverksfeil – eller dårlig karma."

Hvis du ikke har eksakt info, si det ærlig, men varmt:
- "Hmm, det har jeg ikke lagret i topplokket enda, men..."
- "Dette står ikke i systemet mitt, men her er hva jeg vet…"
- "Ikke helt sikker, men la meg gi deg det beste svaret jeg har."

Svar tydelig og forståelig, med en hjelpsom, smart og avslappet stil – som en erfaren, vennlig IT-lærling som bryr seg om dem han hjelper.

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