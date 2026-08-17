# Project 6 - Advanced RAG with Reranking
Legal Research & Document Q&A Assistant. **Client:** Cyril Amarchand Mangaldas (Legal Services).

Compares **basic RAG** against an **advanced** pipeline: hybrid search (BM25 + dense),
cross-encoder reranking, and multi-query transformation, with a before/after evaluation.

## Run locally
1. `pip install -r requirements.txt`
2. `export OPENAI_API_KEY=sk-...`   (Windows: `set OPENAI_API_KEY=...`)
3. `python legal_rag.py`

This project is a **script/demo** (it prints the basic-vs-advanced comparison to the
console), not a web app. Swap in the firm's real PDFs where the sample documents are
defined near the top of `legal_rag.py`.
