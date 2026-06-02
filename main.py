import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "chroma-db"

if not os.path.exists(PERSIST_DIR):
    print("Vector database not found. Please run create_database.py first.")
    exit()

print("\n"+"="*50)
print("Models")
print("="*50)

embedding_model = MistralAIEmbeddings()
model = ChatMistralAI(model="mistral-small-2506")

print("\n"+"="*50)
print("Load Vector Store")
print("="*50)

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

print("\n"+"="*50)
print("Prompt")
print("="*50)


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

print("\nRAG started")
print("Press 0 to exit\n")

while True:
    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = model.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")