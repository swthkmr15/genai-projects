# GenAI Projects — PROITBRIDGE (Enterprise AI)

Four hands-on Generative-AI projects, each in its own folder with its own
`README.md`, `requirements.txt`, and code. Every app reads the OpenAI key from
the environment variable `OPENAI_API_KEY` — no key is stored in the code.

| # | Folder | What it is | Client |
|---|--------|-----------|--------|
| 1 | `project-1-llm-playground` | LLM Playground + Token Explorer (Gradio app) | Amazon |
| 5 | `project-5-rag-document-qa` | RAG document Q&A with sources (Gradio app) | Tata AIA |
| 6 | `project-6-advanced-rag-reranking` | Advanced RAG: hybrid search + reranking (script) | Cyril Amarchand Mangaldas |
| 10 | `project-10-mcp-desktop-assistant` | MCP-powered desktop assistant (Gradio app) | Infosys |

## How to run any project
1. `cd` into the project folder.
2. `pip install -r requirements.txt`
3. Set your key: `export OPENAI_API_KEY=sk-...`  (Windows: `set OPENAI_API_KEY=sk-...`)
4. Run the app / script (see each folder's `README.md`).

## Deploying the web apps (1, 5, 10)
Push this repo to GitHub, then deploy on **Hugging Face Spaces** (easiest for
Gradio) or **Render** — add `OPENAI_API_KEY` as a Secret / environment variable
on the host. Never commit your real key.
