import os
import shutil
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader
)
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIR = "chroma-db"

embedding_model = MistralAIEmbeddings()

print("\nChoose Document Source")
print("1. PDF")
print("2. Website URL")
print("3. Text File")

choice = input("\nEnter your choice (1/2/3): ")

print("\n"+"="*50)
print("Load Documents")
print("="*50)

if choice == "1":
    pdf_path = input("Enter PDF path or press Enter for default PDF: ")

    if pdf_path.strip() == "":
        pdf_path = "document_loaders/deeplearning.pdf"

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

elif choice == "2":
    url = input("Enter Website URL: ")

    loader = WebBaseLoader(url)
    docs = loader.load()

elif choice == "3":
    txt_path = input("Enter Text File Path: ")

    loader = TextLoader(txt_path)
    docs = loader.load()

else:
    print("Invalid choice")
    exit()

print("\n"+"="*50)
print("Split Documents")
print("="*50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# print(f"\nTotal chunks created: {len(chunks)}")

print("\n"+"="*50)
print("Create Vector Store")
print("="*50)

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR
)

print("\nVector database created successfully!")