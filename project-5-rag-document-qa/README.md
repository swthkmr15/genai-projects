# Enterprise Document QA System (RAG)
Project 5 - Tata AIA Life Insurance. Upload insurance documents and ask
questions; answers are grounded in the documents with sources.
Pipeline: loaders/splitters -> chunking -> embeddings -> FAISS vector store
-> retrieval -> reranking -> query rewriting -> grounded answer.

## Run locally
1. `pip install -r requirements.txt`
2. `export OPENAI_API_KEY=sk-...`   (Windows: `set OPENAI_API_KEY=...`)
3. `python app.py`  then open http://localhost:7860

## Deploy
- **Hugging Face Spaces** (Gradio SDK): add `OPENAI_API_KEY` as a Secret, upload the files.
- **Render / Cloud Run**: set the env var; start command `python app.py`.