# CourseMate AI RAG Application

A Retrieval-Augmented Generation (RAG) application built with LangChain, Mistral AI, ChromaDB, and Streamlit. The application enables users to upload PDF documents, text files, or provide website URLs and interact with the content through natural language queries.

The system retrieves relevant information using semantic search and generates context-aware responses grounded in the uploaded content.

## Live Demo

🚀 https://coursemateai-2026.streamlit.app

## Features

* PDF document ingestion
* Website URL ingestion
* Text file ingestion
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database integration
* MMR (Max Marginal Relevance) retrieval strategy
* Context-aware response generation
* Interactive Streamlit UI
* Secure API key management

## Tech Stack

* Python
* LangChain
* Mistral AI
* Mistral Embeddings
* ChromaDB
* Streamlit
* BeautifulSoup
* PyPDFLoader
* WebBaseLoader
* TextLoader

## Project Architecture

```text
User Query
    ↓
Document Loader
(PDF / URL / Text)
    ↓
Text Splitter
    ↓
Mistral Embeddings
    ↓
ChromaDB Vector Store
    ↓
Retriever (MMR Search)
    ↓
Mistral LLM
    ↓
Generated Answer
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/lavanyack311/COURSE_MATE_AI_RAG_PROJECT.git
cd COURSE_MATE_AI_RAG_PROJECT
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

### 5. Run the Application

```bash
streamlit run app.py
```

## How It Works

### Document Loader

Supports multiple data sources:

* PDF Documents
* Website URLs
* Text Files

### Text Splitter

Splits large documents into manageable chunks using Recursive Character Text Splitter.

### Embedding Generation

Generates vector embeddings using Mistral Embeddings.

### Vector Database

Stores document embeddings in ChromaDB for efficient semantic retrieval.

### Retriever

Uses Max Marginal Relevance (MMR) search to retrieve the most relevant document chunks.

### LLM Generation

Passes the retrieved context and user query to the Mistral language model to generate grounded responses.

## Example Workflow

1. User uploads a PDF, text file, or enters a website URL.
2. The application extracts and processes the content.
3. The content is split into chunks.
4. Mistral Embeddings generate vector representations.
5. Chunks are stored in ChromaDB.
6. User submits a question.
7. Relevant chunks are retrieved using MMR search.
8. Mistral AI generates a context-aware response.
9. Results are displayed through the Streamlit interface.

## Repository Structure

```text
COURSE_MATE_AI_RAG_PROJECT/
│
├── app.py
├── create_database.py
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── document_loaders/
    └── deeplearning.pdf
```

## Author
Lavanya C K

## GitHub Repository
💻 https://github.com/lavanyack311/COURSE_MATE_AI_RAG_PROJECT

