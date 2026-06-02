from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
embedding_model=MistralAIEmbeddings()
model=ChatMistralAI(model="mistral-small-2506")

vector_store=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)
retriever=vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":8,
        "lambda_mult":0.5
    }
)

prompt= ChatPromptTemplate.from_messages([
    ("system","""
    You are an helpful AI asistant.
    Use only the provided context to answer the question.
    If the answer is not found in the document
    Say:"I could not find the answer for the question asked". """),
    ("human", """
    "context":{context},
     "question":{question}""")
])

print("Rag started")
print("press 0 to exit")


while True:
    query=input("you:")
    if query=='0':
        break
    docs = retriever.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })
    response= model.invoke(final_prompt)
    print(f"\nAI\n: {response.content}")
   

