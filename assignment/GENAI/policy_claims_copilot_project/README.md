# Policy & Claims Copilot (LLM + RAG)

## Features
- Ask questions about policy coverage
- Retrieves relevant policy clauses
- Provides source page references
- Claim scenario pre-check

## Tech Stack
Python, LangChain, FAISS, OpenAI, FastAPI

- venv\Scripts\activate 

# Step 1: Build vectorstore (only once or when policy.pdf changes)
python ingest.py

# Step 2: Start the API server
- uvicorn app:app --host 0.0.0.0 --port 8000
or
- \venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000

# Step 3: Query the API
curl "http://localhost:8000/ask?query=What+are+the+exclusions?"

# Url
http://localhost:8000
