# LLM Playground & Token Explorer
Project 1 - Amazon (E-Commerce). An LLM Playground (adjust decoding
parameters and see the response) and a Token Explorer (see tokens and cost).

## Run locally
1. `pip install -r requirements.txt`
2. `export OPENAI_API_KEY=sk-...`   (Windows: `set OPENAI_API_KEY=...`)
3. `python app.py`  then open http://localhost:7860

## Deploy
- **Hugging Face Spaces** (easiest): create a Gradio Space, add
  `OPENAI_API_KEY` as a Secret, upload these files.
- **Render**: New Web Service from this repo, add the env var,
  start command `python app.py`.
- **Docker**: `docker build -t app . && docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... app`