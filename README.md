# Document-Q-A-System-with-Hybrid-Search

1. Project Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline for answering questions from a knowledge-heavy document.

The system combines two retrieval strategies:

BM25 for exact keyword matching

Vector search for semantic similarity

The two result lists are combined using Reciprocal Rank Fusion (RRF). The top 10 candidates are then reranked using a cross-encoder, and the best 3 chunks are passed as context to a local LLM through LangChain.

2. Pipeline

Document
   ↓
RecursiveCharacterTextSplitter
(chunk size = 300, overlap = 50)
   ↓
 ┌───────────────────────┐
 │                       │
 ↓                       ↓
Vector Embeddings       BM25
all-MiniLM-L6-v2        Keyword Search
 │                       │
 ↓                       ↓
ChromaDB                BM25 Results
 └───────────┬───────────┘
             ↓
       Reciprocal Rank Fusion
             ↓
          Top 10
             ↓
       Cross-Encoder
             ↓
           Top 3
             ↓
       Context + LLM
             ↓
          Answer

3. Technologies Used

Component

Technology

Language

Python

Notebook

Jupyter Notebook

Chunking

RecursiveCharacterTextSplitter

Embeddings

sentence-transformers / all-MiniLM-L6-v2

Vector Database

ChromaDB

Keyword Retrieval

rank-bm25 / BM25Okapi

Hybrid Retrieval

Reciprocal Rank Fusion

Reranking

cross-encoder/ms-marco-MiniLM-L-6-v2

LLM Framework

LangChain

Local LLM

Ollama

Example Model

qwen3:1.7b

4. Repository Structure

rag-hybrid-search/
│
├── RAG_Hybrid_Search_Assignment.ipynb
├── data/
│   └── document.txt
├── chroma_db/
├── requirements.txt
└── README.md

5. Requirements

Python 3.10 or newer recommended

Jupyter Notebook or JupyterLab

Ollama installed and running

Internet connection for the first download of embedding/reranker models

6. Installation

Create a virtual environment:

Windows

python -m venv .venv
.venv\Scripts\activate

macOS/Linux

python3 -m venv .venv
source .venv/bin/activate

Install the Python dependencies:

pip install -r requirements.txt

If Jupyter is not installed:

pip install notebook

Start Jupyter:

jupyter notebook

Then open:

RAG_Hybrid_Search_Assignment.ipynb

7. Prepare the Document

Create the directory:

data/

Place a knowledge-heavy document containing at least two pages of content inside it:

data/document.txt

The notebook reads this file automatically.

For example, the document can be a Wikipedia article, textbook chapter, product manual, company policy, or another substantial knowledge-heavy text.

8. Install and Configure Ollama

Install Ollama and make sure it is running.

Pull the model used by the notebook:

ollama pull qwen3:1.7b

The notebook contains:

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0
)

If another Ollama model is installed, change the model name in the notebook.

9. Run the Notebook

Run the cells in order.

The notebook performs these steps:

Load the document.

Split it into 300-character chunks with 50-character overlap.

Generate embeddings using all-MiniLM-L6-v2.

Store embeddings and chunks in ChromaDB.

Build a BM25 index.

Perform vector search.

Perform BM25 search.

Combine results using Reciprocal Rank Fusion.

Keep the top 10 hybrid results.

Rerank them using cross-encoder/ms-marco-MiniLM-L-6-v2.

Select the best 3 chunks.

Pass those chunks to the LLM.

Generate a context-grounded answer.

10. RAG Prompt

The system uses the following instruction:

You are a document question-answering assistant.

Answer using ONLY the context provided below.

If the answer cannot be found in the context, say exactly:
Not in context.

Do not use outside knowledge.

This is designed to reduce unsupported answers from the LLM.

11. Testing

The notebook includes five test cases.

Test 1 — Basic document understanding

Tests whether the system can answer a direct question about the document.

Test 2 — Exact keyword matching

Uses an exact or distinctive term from the document to demonstrate the strength of BM25.

Test 3 — Semantic understanding

Uses different wording from the source to demonstrate the strength of vector retrieval.

Test 4 — Multi-concept question

Tests whether the retrieved context contains enough information for a broader question.

Test 5 — Not in context

Asks for information that is absent from the document and checks whether the system avoids using outside knowledge.

12. Why Hybrid Search?

BM25 and vector search have different strengths.

BM25:

Excellent for exact keywords

Useful for names, product codes, technical terms, and rare phrases

Does not require semantic similarity

Vector Search:

Understands semantic similarity

Can retrieve relevant information when the wording is different

Useful for natural-language questions

Using both gives the system better retrieval coverage.

13. Why RRF?

The BM25 and vector search scores are different types of scores and should not simply be added together.

Reciprocal Rank Fusion combines their rankings:

RRF Score = Σ 1 / (k + rank)

The notebook uses:

k = 60

A chunk appearing near the top of both retrieval lists receives a stronger combined ranking.

14. Why Cross-Encoder Reranking?

Initial retrieval is optimized for speed and recall.

The cross-encoder performs a more detailed relevance comparison between:

Question + Retrieved Chunk

The system reranks the top 10 candidates and keeps the best 3 before sending context to the LLM.

This reduces irrelevant context and can improve answer quality.

15. Edge Case Awareness

A potential failure point is missing or malformed input data.

If data/document.txt is missing, the notebook raises a clear FileNotFoundError.

Other possible issues include:

Ollama is not running.

The selected Ollama model is not installed.

Embedding or reranker models cannot be downloaded.

The query contains no useful BM25 keywords.

The requested information is absent from the document.

The document contains too little content.

There are fewer chunks than the requested top-k value.

The implementation limits retrieval to the number of available chunks and instructs the LLM to return Not in context. when the answer is not supported by the retrieved context.

16. Real-World Use Case

This project can be used to build an internal document assistant.

For example, a company could use it to answer questions about:

Product manuals

Company policies

Employee training documents

Pricing documents

Technical documentation

Operating procedures

Internal knowledge bases

Instead of manually searching through many pages, a user can ask a natural-language question and receive an answer grounded in the organization's documents.

17. Expected Learning Outcomes

This project demonstrates understanding of:

Document chunking

Embeddings

Vector databases

Semantic search

Keyword search

Hybrid retrieval

Reciprocal Rank Fusion

Cross-encoder reranking

RAG architecture

Prompt grounding

Local LLM inference

LangChain integration
