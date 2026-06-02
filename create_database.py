#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
embedding_model= MistralAIEmbeddings()

data=PyPDFLoader("document_loaders/deeplearning.pdf")
docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20
)
chunks=splitter.split_documents(docs)

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)