# Project 10 - MCP-Powered Desktop Assistant

**Client:** Infosys | **Industry:** IT Services
**Stack:** Python - OpenAI - Model Context Protocol (MCP) - Gradio

One AI brain (OpenAI) reaches many desktop tools through one standard protocol (MCP),
with no custom integration per tool.

## Architecture
```
Host (app + OpenAI brain)  ->  MCP Client  ->  MCP Server  ->  Tools
```
- The Host decides *which* tool to use (OpenAI function calling).
- The MCP Server exposes the tools and runs them.
- Add a new `@mcp.tool()` and it is instantly available to the Host.

## Tools
- get_current_time
- calculate
- list_files
- write_note
- read_note

## Files
- `app.py` - Gradio web UI + OpenAI<->MCP bridge (entry point)
- `mcp_desktop_server.py` - the MCP server (tools)
- `requirements.txt` - dependencies
- `.env.example` - copy to `.env` and add your key

## Run locally
```
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."     # Windows: set OPENAI_API_KEY=sk-...
python app.py
```
Then open the local URL that Gradio prints.

## Push to GitHub
```
git init
git add .
git commit -m "Project 10 - MCP-Powered Desktop Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
Never commit your real key - `.gitignore` already excludes `.env`.

## Notes
- MCP is model-agnostic; this build uses OpenAI as the Host brain.
- Default model is `gpt-4o-mini`; change it in `app.py` if needed.