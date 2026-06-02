import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="RAG PDF/URL/Text Chatbot",
    page_icon="📄",
    layout="centered"
)

load_dotenv()

try:
    if "MISTRAL_API_KEY" in st.secrets:
        os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
except Exception:
    pass

if not os.getenv("MISTRAL_API_KEY"):
    st.error("MISTRAL_API_KEY is missing. Add it in .env locally or Streamlit Secrets.")
    st.stop()


def clear_previous_results():
    st.session_state["answer"] = ""
    st.session_state["retrieved_docs"] = []
    st.session_state["vector_store"] = None
    st.session_state["db_created"] = False


if "answer" not in st.session_state:
    st.session_state["answer"] = ""

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "db_created" not in st.session_state:
    st.session_state["db_created"] = False


st.title("📄 RAG Application using Mistral + Chroma")
st.write("Upload a PDF/Text file or enter a website URL, then ask questions.")


embedding_model = MistralAIEmbeddings()
model = ChatMistralAI(model="mistral-small-2506")


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


st.subheader("Choose Document Source")

source_type = st.radio(
    "Select source type",
    ["PDF", "Website URL", "Text File"],
    on_change=clear_previous_results
)

uploaded_file = None
url_input = None

if source_type == "PDF":
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        on_change=clear_previous_results
    )

elif source_type == "Website URL":
    url_input = st.text_input(
        "Enter website URL",
        on_change=clear_previous_results
    )

elif source_type == "Text File":
    uploaded_file = st.file_uploader(
        "Upload a text file",
        type=["txt"],
        on_change=clear_previous_results
    )


if st.button("Create / Update Vector Database"):
    with st.spinner("Loading document and creating embeddings..."):

        docs = []

        if source_type == "PDF":
            if uploaded_file is None:
                st.warning("Please upload a PDF file.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            os.remove(temp_path)

        elif source_type == "Website URL":
            if not url_input:
                st.warning("Please enter a website URL.")
                st.stop()

            loader = WebBaseLoader(url_input)
            docs = loader.load()

        elif source_type == "Text File":
            if uploaded_file is None:
                st.warning("Please upload a text file.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            loader = TextLoader(temp_path)
            docs = loader.load()
            os.remove(temp_path)

        if not docs:
            st.error("No content was loaded from the selected source.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=20
        )

        chunks = splitter.split_documents(docs)

        chunks = [
            chunk for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not chunks:
            st.error("No valid text chunks found. Please use a readable PDF, URL, or text file.")
            st.stop()

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        st.session_state["vector_store"] = vector_store
        st.session_state["db_created"] = True
        st.session_state["answer"] = ""
        st.session_state["retrieved_docs"] = []

    st.success(f"Vector database created successfully! Total chunks: {len(chunks)}")


st.divider()

query = st.text_input("Ask a question from your document")

if query:
    if st.session_state["vector_store"] is None:
        st.warning("Please create the vector database first.")
    else:
        retriever = st.session_state["vector_store"].as_retriever(
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

        st.session_state["answer"] = response.content
        st.session_state["retrieved_docs"] = docs


if st.session_state.get("answer"):
    st.subheader("AI Answer")
    st.write(st.session_state["answer"])

    with st.expander("Retrieved Context"):
        for i, doc in enumerate(st.session_state["retrieved_docs"], start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)