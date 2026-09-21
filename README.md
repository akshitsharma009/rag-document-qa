# RAG Document QA

A local Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

The application retrieves the most relevant document sections using semantic search and generates grounded answers using a local language model. It also provides source pages and uses a relevance check to avoid answering questions that are outside the uploaded document.

## Live Demo

https://rag-document-app-29ii7bafaqbsqzcjxhzzhj.streamlit.app/

## GitHub

https://github.com/akshitsharma009/rag-document-qa

## Author

**Akshit Sharma**

- GitHub: https://github.com/akshitsharma009
- LinkedIn: https://www.linkedin.com/in/akshit-sharma-dev/

---

## Project Overview

Traditional document search requires users to manually find information inside long documents.

This project provides a simple question-answering interface where users can:

1. Upload a PDF document
2. Convert the document into searchable text chunks
3. Generate vector embeddings for the chunks
4. Store the embeddings in a FAISS vector database
5. Retrieve the most relevant chunks for a question
6. Generate an answer using the retrieved context
7. View the source pages used for the answer

The application is designed to run locally and does not require a paid LLM API.

---

## Key Features

- PDF document upload
- Automatic text extraction
- Text chunking for efficient retrieval
- Semantic search using vector embeddings
- FAISS-based vector storage
- Persistent document-specific vector databases
- Top-K relevant chunk retrieval
- Relevance filtering for out-of-scope questions
- Local FLAN-T5 model for answer generation
- Source page references
- Retrieval scores and document excerpts
- Streamlit web interface
- CPU-compatible inference

---

## Architecture

```text
PDF Document
     |
     v
PyPDFLoader
     |
     v
Text Cleaning & Chunking
     |
     v
Sentence Transformers
(all-MiniLM-L6-v2)
     |
     v
FAISS Vector Store
     |
     v
Semantic Retrieval
     |
     v
Relevance Check
     |
     +------ Not Relevant ------> Fallback Response
     |
     v
Relevant Context
     |
     v
FLAN-T5
     |
     v
Grounded Answer + Sources
Tech Stack
Programming Language
Python
RAG / NLP
LangChain
Sentence Transformers
Hugging Face Transformers
Vector Database
FAISS
Document Processing
PyPDF
Machine Learning
PyTorch
Frontend
Streamlit
Development Tools
Git
GitHub
VS Code
How It Works
1. Document Upload

The user uploads a PDF through the Streamlit interface.

2. Text Extraction

The application extracts text from the uploaded document using PyPDFLoader.

3. Text Chunking

The extracted text is divided into smaller overlapping chunks using a recursive text splitter.

Current configuration:

Chunk Size: 500
Chunk Overlap: 100

Chunking makes it easier to retrieve only the relevant sections instead of processing the complete document.

4. Embedding Generation

Each text chunk is converted into a numerical vector using:

all-MiniLM-L6-v2

These embeddings represent the semantic meaning of the document chunks.

5. FAISS Vector Store

The generated embeddings are stored in a FAISS vector index.

The application maintains a document-specific vector store so that an existing processed document can be loaded again instead of rebuilding the index every time.

6. Semantic Retrieval

When a user asks a question, the question is converted into an embedding and compared against the document embeddings.

The most relevant chunks are retrieved using semantic similarity.

7. Relevance Check

The retrieved results are checked against a distance threshold.

If the retrieved content is not sufficiently relevant, the application returns:

I couldn't find sufficient information in the uploaded documents to answer this question.

This helps prevent the system from generating answers for questions that are unrelated to the uploaded document.

8. Answer Generation

For relevant questions, the retrieved document context is passed to the local:

google/flan-t5-base

model.

The model generates an answer using the retrieved document context.

9. Sources

The application displays the document pages and retrieved excerpts so the user can see where the answer came from.

Evaluation

A small manually labeled validation set was created to test the retrieval and relevance filtering behavior.

Validation Set
Total Questions: 8
Relevant Questions: 5
Out-of-Scope Questions: 3
Results
True Positives : 5
True Negatives  : 3
False Positives : 0
False Negatives : 0
Accuracy         : 1.00
Precision        : 1.00
Recall           : 1.00
Fallback Accuracy: 1.00

These results are based on the small manually labeled validation set and should not be interpreted as a general measure of RAG accuracy.

Example Questions

Using a research paper as the uploaded document:

Relevant Question
What is multi-head attention?

The system retrieves relevant sections from the document and generates a grounded answer with source pages.

Out-of-Scope Question
What is the capital of France?

The system detects that the question is not sufficiently supported by the uploaded document and returns the fallback response instead of generating an unsupported answer.

Project Structure
rag-document-qa/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── src/
│   └── vector_store.py
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   └── evaluate_answers.py
│
└── data/
    ├── documents/
    └── vectorstore/
Installation
1. Clone the repository
git clone https://github.com/akshitsharma009/rag-document-qa.git
cd rag-document-qa
2. Create a virtual environment
python -m venv .venv
3. Activate the environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run the application
streamlit run app.py

The application will open in your browser.

Usage
Open the application.
Upload a PDF document.
Wait for the document to be processed.
Enter a question related to the document.
Review the generated answer.
Check the retrieved sources and page references.
Limitations
The current application is designed primarily for PDF documents.
Answer quality depends on the quality of document extraction and retrieval.
The local FLAN-T5 model has limitations compared with larger hosted language models.
The evaluation dataset is small and manually labeled.
CPU inference can be slower for larger documents.
Future Improvements

Possible improvements include:

Better retrieval strategies
Improved answer generation models
Support for additional document formats
Larger evaluation datasets
Improved document processing
More advanced retrieval and ranking techniques
License

This repository contains code adapted from an existing open-source project and retains the applicable license and attribution requirements of that source.

Additional modifications and project-specific implementation were made for this project.


### One thing I deliberately changed

I **didn't put "100% accuracy" as a project claim**. Instead:

> "Accuracy 1.00 on a small manually labeled validation set"

That's much safer in an interview because an interviewer can ask about your evaluation set and you can explain it honestly.

Also, I didn't write that you built the entire RAG framework from scratch. You can confidently explain **what you implemented/modified**: persistent FAISS storage, document-specific collections, relevance filtering, source display, evaluation, and the Streamlit workflow.

### Now update it

In VS Code:

1. Open `README.md`
2. `Ctrl + A`
3. Delete everything
4. Paste the README above
5. Save with `Ctrl + S`

**Don't commit/push yet.**

After saving, send me a screenshot or the `git diff -- README.md` output. We'll do a quick final check, then commit and pus