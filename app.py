import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# API KEY SETUP
# -----------------------------
load_dotenv()

if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

if not os.getenv("MISTRAL_API_KEY"):
    st.error("MISTRAL_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Application using Mistral + Chroma")
st.write("Upload a PDF, create embeddings, and ask questions from the document.")


# -----------------------------
# MODELS
# -----------------------------
embedding_model = MistralAIEmbeddings()
model = ChatMistralAI(model="mistral-small-2506")

PERSIST_DIR = "chroma-db"


# -----------------------------
# PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant.
Use only the provided context to answer the question.
If the answer is not found in the document, say:
"I could not find the answer for the question asked".
"""),
    ("human", """
Context:
{context}

Question:
{question}
""")
])


# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Create / Update Vector Database"):
        with st.spinner("Processing PDF and creating vector database..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=20
            )

            chunks = splitter.split_documents(docs)

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIR
            )

            os.remove(temp_pdf_path)

        st.session_state["db_created"] = True
        st.success("Vector database created successfully!")


# -----------------------------
# QUESTION ANSWERING
# -----------------------------
st.divider()

query = st.text_input("Ask a question from the uploaded PDF")

if query:
    if not os.path.exists(PERSIST_DIR) and not st.session_state.get("db_created"):
        st.warning("Please upload a PDF and click 'Create / Update Vector Database' first.")
    else:
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 8,
                "lambda_mult": 0.5
            }
        )

        with st.spinner("Retrieving relevant context and generating answer..."):
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = model.invoke(final_prompt)

        st.subheader("AI Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)