"""
Project 10 - MCP-Powered Desktop Assistant (standalone Gradio app)
Client: Infosys | Industry: IT Services
Architecture: Host (OpenAI brain OR built-in router) -> MCP Client -> MCP Server -> Tools

Works with OR without an OpenAI key:
  - If OPENAI_API_KEY is a valid, funded key, OpenAI decides the tool.
  - Otherwise a built-in rule router picks the tool, so the assistant still replies.

Run locally:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."      # optional; Windows: set OPENAI_API_KEY=sk-...
    python app.py
"""
import os
import re
import json
import asyncio
import threading

import gradio as gr
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = OpenAI()  # reads OPENAI_API_KEY from the environment (may be empty)
SERVER = StdioServerParameters(command="python", args=["mcp_desktop_server.py"])
ERRLOG = open("mcp_server.log", "w")

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()
def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def mcp_to_openai(tools):
    return [{
        "type": "function",
        "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
    } for t in tools]


# ---------------- built-in router (used when OpenAI is unavailable) ----------------
def _find_filename(text):
    m = re.search(r'([\w\-]+\.txt)', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:called|named)\s+([\w\-\.]+)', text, re.I)
    if m:
        name = m.group(1).strip('.:,')
        return name if '.' in name else name + '.txt'
    return None

def _find_content(text):
    m = re.search(r'(?:saying|that says|says|:)\s*(.+)$', text, re.I)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None

def _find_expression(text):
    t = text.lower().replace('into', '*').replace('times', '*').replace('plus', '+').replace('minus', '-').replace('divided by', '/')
    t = t.replace('x', '*').replace('\u00d7', '*')
    m = re.search(r'[0-9][0-9\s+\-*/().%]*', t)
    if m:
        e = m.group().strip()
        if re.search(r'[0-9]', e) and re.search(r'[+\-*/]', e):
            return e
    return None

def local_brain(text):
    t = text.lower().strip()
    if any(w in t for w in ['time', 'date', 'day', 'today', 'now', 'clock']):
        return 'get_current_time', {}
    expr = _find_expression(text)
    if expr and (any(w in t for w in ['calc', 'what is', 'how much', 'sum', 'multiply', 'add', 'subtract', 'divide']) or re.search(r'[0-9]\s*[+\-*/]', expr)):
        return 'calculate', {'expression': expr}
    if 'note' in t and any(w in t for w in ['save', 'write', 'create', 'add', 'make']):
        return 'write_note', {'filename': _find_filename(text) or 'note.txt',
                              'content': _find_content(text) or '(empty note)'}
    if 'note' in t and any(w in t for w in ['read', 'open', 'show', 'get', 'view']):
        return 'read_note', {'filename': _find_filename(text) or 'note.txt'}
    if ('list' in t or 'all' in t or 'my notes' in t or 'what notes' in t) and ('note' in t or 'file' in t):
        return 'list_files', {}
    return None, None

def _phrase(tool, args, output):
    if tool == 'get_current_time':
        return f"It's currently {output}."
    if tool == 'calculate':
        return f"{args.get('expression', '')} = {output}"
    if tool == 'write_note':
        return output
    if tool == 'read_note':
        return f"Here is your note:\n\n{output}"
    if tool == 'list_files':
        return f"Your saved notes:\n{output}"
    return str(output)

HELP = ("I'm your desktop assistant. Try: 'what time is it?', "
        "'calculate 1250 * 12 + 300', 'save a note called plan.txt saying Deliver Project 10', "
        "'read note plan.txt', or 'list my notes'.")

_openai_state = "unknown"


async def _engine_async(user_query, model):
    global _openai_state
    steps = []
    async with stdio_client(SERVER, errlog=ERRLOG) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            steps.append(f"1. Host connected to the MCP Server and discovered {len(tools)} tools.")

            key = os.environ.get("OPENAI_API_KEY", "").strip()
            use_openai = key not in ("", "your-api-key-here") and _openai_state != "bad"

            if use_openai:
                try:
                    oai_tools = mcp_to_openai(tools)
                    messages = [
                        {"role": "system", "content": "You are a helpful desktop assistant. Use a tool when it helps answer the user."},
                        {"role": "user", "content": user_query},
                    ]
                    steps.append("2. Sent your prompt + the tool list to OpenAI (the Host brain).")
                    first = client.chat.completions.create(model=model, messages=messages, tools=oai_tools)
                    _openai_state = "ok"
                    msg = first.choices[0].message
                    if not msg.tool_calls:
                        steps.append("3. OpenAI answered directly (no tool needed).")
                        return msg.content, steps
                    messages.append(msg)
                    for call in msg.tool_calls:
                        args = json.loads(call.function.arguments or "{}")
                        steps.append(f"3. OpenAI chose the tool: {call.function.name}({args})")
                        result = await session.call_tool(call.function.name, args)
                        output = result.content[0].text if result.content else ""
                        steps.append(f"4. MCP Server ran it &rarr; {output}")
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                    final = client.chat.completions.create(model=model, messages=messages)
                    steps.append("5. OpenAI wrote the final answer from the tool result.")
                    return final.choices[0].message.content, steps
                except Exception as e:
                    _openai_state = "bad"
                    steps.append(f"2. OpenAI not available ({type(e).__name__}) &mdash; switching to the built-in router.")
            else:
                steps.append("2. OpenAI key not active &mdash; using the built-in router (no API needed).")

            tool, args = local_brain(user_query)
            if not tool:
                steps.append("3. Router: no tool matched, showing help.")
                return HELP, steps
            steps.append(f"3. Router chose the tool: {tool}({args})")
            result = await session.call_tool(tool, args)
            output = result.content[0].text if result.content else ""
            steps.append(f"4. MCP Server ran it &rarr; {output}")
            steps.append("5. Answer composed from the tool result.")
            return _phrase(tool, args, output), steps


def ask_assistant_traced(user_query, model="gpt-4o-mini"):
    try:
        return run_async(_engine_async(user_query, model))
    except BaseException as e:
        while hasattr(e, "exceptions") and getattr(e, "exceptions", None):
            e = e.exceptions[0]
        return f"[ERROR] {type(e).__name__}: {e}", [f"Error: {type(e).__name__}: {e}"]


PB_CSS = """
.gradio-container{max-width:1180px !important;}
#pb-hero{position:relative;overflow:hidden;background:linear-gradient(120deg,#0a1f44 0%,#0e2a55 55%,#123a73 100%);
  border-radius:16px;padding:26px 30px;color:#fff;margin-bottom:16px;}
#pb-hero .kick{font-size:11px;letter-spacing:.28em;color:#7fd3ff;font-weight:700;text-transform:uppercase;}
#pb-hero h1{font-size:30px;margin:6px 0 6px 0;font-weight:800;color:#fff;}
#pb-hero p{margin:0;color:#cfe0f5;font-size:14px;max-width:640px;}
#pb-hero .globe{position:absolute;right:-30px;top:-16px;opacity:.22;}
.pb-brand{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.pb-brand .wm{font-weight:800;font-size:20px;letter-spacing:.04em;color:#fff;}
.pb-brand .wm i{color:#1aa3e8;font-style:normal;}
.pb-brand .tag{font-size:9px;letter-spacing:.24em;color:#9fb6d6;text-transform:uppercase;}
#pb-cards{display:flex;gap:14px;margin-bottom:6px;}
.pb-card{flex:1;background:#fff;border:1px solid #dbe4f0;border-radius:12px;padding:14px 16px;}
.pb-card.hot{background:linear-gradient(120deg,#1aa3e8,#38bdf8);border:none;}
.pb-card .lbl{font-size:10px;letter-spacing:.14em;color:#6b7a90;text-transform:uppercase;font-weight:700;}
.pb-card.hot .lbl{color:#eaf6ff;}
.pb-card .val{font-size:26px;font-weight:800;color:#0a1f44;margin-top:4px;}
.pb-card.hot .val{color:#fff;}
.pb-card .sub{font-size:11px;color:#6b7a90;margin-top:2px;}
.pb-card.hot .sub{color:#eaf6ff;}
#pb-trace{background:#0a1f44;border-radius:12px;padding:14px 16px;color:#dbe7f7;min-height:320px;}
#pb-trace h3{color:#7fd3ff;font-size:13px;letter-spacing:.12em;text-transform:uppercase;margin:0 0 10px 0;}
#pb-trace ol{margin:0;padding-left:18px;}
#pb-trace li{margin:6px 0;font-size:13px;line-height:1.45;color:#cfe0f5;}
#pb-flow{background:#f4f8fc;border:1px solid #dbe4f0;border-radius:12px;padding:12px 16px;
  font-family:monospace;font-size:12px;color:#0a1f44;white-space:pre;overflow:auto;}
footer{display:none !important;}
"""
PB_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

GLOBE = ("<svg class='globe' width='240' height='240' viewBox='0 0 260 260' fill='none' stroke='#7fd3ff' stroke-width='1'>"
         "<circle cx='130' cy='130' r='96'/><ellipse cx='130' cy='130' rx='96' ry='40'/>"
         "<ellipse cx='130' cy='130' rx='96' ry='72'/><ellipse cx='130' cy='130' rx='40' ry='96'/>"
         "<ellipse cx='130' cy='130' rx='72' ry='96'/><line x1='34' y1='130' x2='226' y2='130'/></svg>")

HERO = f"""
<div id='pb-hero'>
  {GLOBE}
  <div class='pb-brand'><span class='wm'>PRO<i>IT</i>BRIDGE</span>
    <span class='tag'>Strive for a better future</span></div>
  <div class='kick'>Automated Client Intelligence</div>
  <h1>MCP-Powered Desktop Assistant</h1>
  <p>Give a prompt in plain language &mdash; one AI brain reaches five desktop tools through one
     protocol (MCP). Watch the backend work, step by step, on the right.</p>
</div>"""

CARDS = """
<div id='pb-cards'>
  <div class='pb-card'><div class='lbl'>Active Tools</div><div class='val'>5</div><div class='sub'>via MCP server</div></div>
  <div class='pb-card'><div class='lbl'>Protocol</div><div class='val'>MCP</div><div class='sub'>model-agnostic</div></div>
  <div class='pb-card hot'><div class='lbl'>Host Brain</div><div class='val'>gpt-4o-mini</div><div class='sub'>or built-in router</div></div>
  <div class='pb-card'><div class='lbl'>Transport</div><div class='val'>stdio</div><div class='sub'>client &harr; server</div></div>
</div>"""

FLOW = chr(10).join([
    "Employee",
    "   |  prompt",
    "   v",
    "HOST (OpenAI brain OR built-in router)  --choose tool-->  MCP CLIENT",
    "                                                              |  stdio",
    "                                                              v",
    "                                                         MCP SERVER",
    "                                                              |",
    "                                                              v",
    "                         time . calculate . list . write . read",
])


def trace_html(steps):
    items = "".join(f"<li>{s}</li>" for s in steps)
    return f"<div id='pb-trace'><h3>Backend process (live)</h3><ol>{items}</ol></div>"


INIT_TRACE = trace_html(["Send a prompt to see the Host &rarr; Server &rarr; Tool &rarr; Result journey here."])


def respond(message, history):
    if not message:
        return history, INIT_TRACE, ""
    answer, steps = ask_assistant_traced(message)
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    return history, trace_html(steps), ""


def run_example(example, history):
    answer, steps = ask_assistant_traced(example)
    history = history + [{"role": "user", "content": example},
                         {"role": "assistant", "content": answer}]
    return history, trace_html(steps)


with gr.Blocks(title="PROITBRIDGE - MCP Desktop Assistant") as demo:
    gr.HTML(HERO)
    gr.HTML(CARDS)
    with gr.Tabs():
        with gr.Tab("Assistant"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=380, show_label=False)
                    with gr.Row():
                        msg = gr.Textbox(show_label=False, scale=8, container=False,
                                         placeholder="Ask: what time is it? / calculate 1250 * 12 + 300 / save a note / read plan.txt")
                        send = gr.Button("Send", variant="primary", scale=1, min_width=90)
                    with gr.Row():
                        b1 = gr.Button("Time", size="sm"); b2 = gr.Button("Calculate", size="sm")
                        b3 = gr.Button("Save note", size="sm"); b4 = gr.Button("Read note", size="sm")
                    clear = gr.Button("Clear chat", size="sm")
                with gr.Column(scale=2):
                    trace = gr.HTML(INIT_TRACE)
        with gr.Tab("How it works (backend)"):
            gr.Markdown("### The request journey")
            gr.HTML(f"<div id='pb-flow'>{FLOW}</div>")
            gr.Markdown(
                "**1. Host** takes your prompt to the brain (OpenAI if a key is active, else a built-in router).  \n"
                "**2. Brain** decides whether a tool is needed and which one.  \n"
                "**3. MCP client** calls that tool on the **MCP server** over stdio.  \n"
                "**4. Server** runs the tool and returns the result.  \n"
                "**5.** The final answer is composed from that result.\n\n"
                "**Tools:** get_current_time, calculate, list_files, write_note, read_note.")

    send.click(respond, [msg, chatbot], [chatbot, trace, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, trace, msg])
    b1.click(lambda h: run_example("What time is it right now?", h), chatbot, [chatbot, trace])
    b2.click(lambda h: run_example("Calculate 1250 * 12 + 300", h), chatbot, [chatbot, trace])
    b3.click(lambda h: run_example("Save a note called plan.txt saying: Deliver Project 10 to Infosys", h), chatbot, [chatbot, trace])
    b4.click(lambda h: run_example("Read note plan.txt", h), chatbot, [chatbot, trace])
    clear.click(lambda: ([], INIT_TRACE), None, [chatbot, trace])


if __name__ == "__main__":
    demo.launch(theme=PB_THEME, css=PB_CSS)