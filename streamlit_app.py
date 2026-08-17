"""
GenAI Projects — Combined Demo Hub (Streamlit)
==============================================
One Streamlit app with a section for each of four projects. Deploy free on
Streamlit Community Cloud straight from your GitHub repo → one permanent public link.

Projects
  1  LLM Playground & Token Explorer      (Amazon)
  5  Enterprise Document QA — RAG         (Tata AIA Life Insurance)
  6  Advanced RAG: Hybrid + Rerank        (Cyril Amarchand Mangaldas — legal)
 10  MCP-style Desktop Assistant          (Infosys)

Set your key in Streamlit Cloud → app → Settings → Secrets, as:
    OPENAI_API_KEY = "sk-..."
The full original version of each project lives in the GitHub repo (one folder each).
"""

import os, re, json
from datetime import datetime
import numpy as np
import streamlit as st

MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Make the key from Streamlit "Secrets" available as an env var for the OpenAI client.
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ------------------------------------------------------------------ shared helpers
@st.cache_resource
def get_client():
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "No OPENAI_API_KEY found. In Streamlit Cloud: open your app → "
            "Settings (⋮) → Secrets → add  OPENAI_API_KEY = \"sk-...\"")
    return OpenAI()

def ask(prompt, system="You are a helpful assistant.", temperature=0.2, max_tokens=400):
    r = get_client().chat.completions.create(
        model=MODEL, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}])
    return r.choices[0].message.content

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    r = get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")

@st.cache_resource
def get_enc():
    import tiktoken
    try:
        return tiktoken.encoding_for_model(MODEL)
    except Exception:
        return tiktoken.get_encoding("o200k_base")

def chunk_text(t, size=500, overlap=80):
    parts = [p for p in re.split(r"(\n\n+|(?<=[.!?])\s+)", t or "") if p and not p.isspace()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= size:
            cur += p
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = (cur[-overlap:] if overlap and cur else "") + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

class Store:
    def __init__(self, d=1536):
        import faiss
        self.faiss = faiss
        self.i = faiss.IndexFlatIP(d)
        self.m = []
    def add(self, v, meta):
        v = np.ascontiguousarray(v, dtype="float32"); self.faiss.normalize_L2(v)
        self.i.add(v); self.m += meta
    def search(self, q, k=5):
        q = np.ascontiguousarray(q, dtype="float32"); self.faiss.normalize_L2(q)
        if self.i.ntotal == 0:
            return []
        D, I = self.i.search(q, min(k, self.i.ntotal))
        return [(self.m[i], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]

PIN, POUT = 0.15, 0.60  # $ per 1M input / output tokens


# ============================================================ data
INSURANCE_DOCS = [
    {"title": "Premium & Grace Period", "text": "Policyholders must pay the premium by the due date. A grace period of 30 days (15 days for monthly mode) is allowed after the due date without losing benefits. If the premium is not paid within the grace period, the policy may lapse."},
    {"title": "Free-Look Period", "text": "If you disagree with any policy terms, you may return the policy within the free-look period of 15 days (30 days if bought through distance marketing) from the date of receipt. The premium is refunded after deducting stamp duty, medical, and proportionate risk charges."},
    {"title": "Death Claim Process", "text": "In case of the life assured's death, the nominee must intimate the company and submit the claim form, original policy document, death certificate, and nominee identity/address proof. Once documents are verified, the death benefit is paid to the nominee."},
    {"title": "Maturity Claim", "text": "On the maturity date, the maturity benefit is paid to the policyholder, who should submit the discharge form and original policy document. Payment is made to the registered bank account."},
    {"title": "Policy Surrender", "text": "A policy acquires a surrender value after paying premiums for the required minimum years. On surrender, the surrender value is paid and the policy terminates. Surrendering early may result in lower returns."},
    {"title": "Policy Revival", "text": "A lapsed policy can be revived within the revival period (usually 5 years) by paying all due premiums with interest and submitting a health declaration, subject to underwriting approval."},
    {"title": "Tax Benefits", "text": "Premiums paid may qualify for tax deduction under Section 80C, and the maturity/death benefit may be exempt under Section 10(10D), subject to prevailing tax laws. Please consult a tax advisor."},
    {"title": "Riders", "text": "Optional riders such as Accidental Death Benefit, Critical Illness, and Waiver of Premium can be added for extra protection by paying an additional rider premium."},
]

LEGAL_DOCS = [
    {"source": "MSA.pdf", "text": "MASTER SERVICES AGREEMENT (Acme Corp and Vendor). Clause 9.1 Limitation of Liability: the total aggregate liability of either party under this Agreement shall not exceed the fees paid in the twelve (12) months preceding the event giving rise to the claim. Clause 12.3 Termination for Convenience: either party may terminate this Agreement for convenience by giving thirty (30) days' prior written notice to the other party. Clause 15 Indemnity: the Vendor shall defend, indemnify and hold harmless the Client against all claims arising from any allegation that the Services infringe a third party's intellectual property rights. Clause 18 Governing Law and Dispute Resolution: this Agreement is governed by the laws of India. Any dispute shall be referred to arbitration seated in Mumbai under the Arbitration and Conciliation Act, 1996."},
    {"source": "NDA.pdf", "text": "NON-DISCLOSURE AGREEMENT. The Receiving Party shall keep Confidential Information secret and shall not use it for any purpose other than the Permitted Purpose for a period of five (5) years from the date of disclosure. Permitted disclosure: Confidential Information may be disclosed to employees and professional advisers strictly on a need-to-know basis."},
    {"source": "Employment.pdf", "text": "EMPLOYMENT AGREEMENT. Notice period: either party may end the employment by giving ninety (90) days' written notice. Clause 7 Non-Compete: for twelve (12) months after termination, the Employee shall not join a direct competitor operating within India."},
    {"source": "DPA.pdf", "text": "DATA PROTECTION ADDENDUM. Personal Data breach notification: the Processor shall notify the Controller without undue delay and in any event within seventy-two (72) hours of becoming aware of a Personal Data breach. Sub-processors: the Processor may engage sub-processors only with the Controller's prior written authorisation."},
    {"source": "CaseLawNote.pdf", "text": "CASE-LAW NOTE — Liquidated Damages vs Penalty. Section 74 of the Indian Contract Act, 1872 allows a party to recover reasonable compensation not exceeding the amount named in the contract, whether or not actual loss is proven. Courts distinguish a genuine pre-estimate of loss (enforceable) from a penalty (not enforceable)."},
]


# ============================================================ cached indexes
@st.cache_resource
def p5_store():
    st_ = Store(1536)
    ch, me = [], []
    for d in INSURANCE_DOCS:
        for c in chunk_text(d["text"]):
            ch.append(c); me.append({"text": c, "source": d["title"]})
    st_.add(embed(ch), me)
    return st_

@st.cache_resource
def p6_index():
    from rank_bm25 import BM25Okapi
    chunks = []
    for d in LEGAL_DOCS:
        for c in chunk_text(d["text"], size=300, overlap=50):
            chunks.append({"text": c, "source": d["source"]})
    store = Store(1536)
    store.add(embed([c["text"] for c in chunks]), chunks)
    bm25 = BM25Okapi([re.findall(r"[a-z0-9.]+", c["text"].lower()) for c in chunks])
    return {"chunks": chunks, "store": store, "bm25": bm25}


# ============================================================ Project 10 tools
WORKSPACE = "/tmp/desktop_workspace"
os.makedirs(WORKSPACE, exist_ok=True)

def t_time():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
def t_calc(expression):
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) . % are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"
def t_list():
    fs = os.listdir(WORKSPACE)
    return "\n".join(fs) if fs else "(workspace is empty)"
def t_write(filename, content):
    with open(os.path.join(WORKSPACE, filename), "w") as f:
        f.write(content)
    return f"Saved note '{filename}'."
def t_read(filename):
    p = os.path.join(WORKSPACE, filename)
    if not os.path.exists(p):
        return f"Error: '{filename}' not found."
    return open(p).read()

TOOLS = {"get_current_time": t_time, "calculate": t_calc,
         "list_files": t_list, "write_note": t_write, "read_note": t_read}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_current_time",
        "description": "Return the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "calculate",
        "description": "Evaluate a simple math expression like '12*7+5'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "list_files",
        "description": "List files saved in the workspace.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "write_note",
        "description": "Save a text note into the workspace.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filename", "content"]}}},
    {"type": "function", "function": {"name": "read_note",
        "description": "Read a text note back from the workspace.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}}, "required": ["filename"]}}},
]

def router(message):
    m = message.lower()
    if "time" in m or "date" in m:
        return "get_current_time", {}
    if any(c.isdigit() for c in m) and any(op in m for op in "+-*/x"):
        expr = re.sub(r"[^0-9+\-*/(). %]", "", m.replace("x", "*"))
        return "calculate", {"expression": expr}
    if "read" in m:
        f = re.search(r"([\w.\-]+\.txt)", m)
        return "read_note", {"filename": f.group(1) if f else "note.txt"}
    if "save" in m or "write" in m or "note" in m:
        f = re.search(r"([\w.\-]+\.txt)", m)
        return "write_note", {"filename": f.group(1) if f else "note.txt", "content": message}
    if "list" in m or "files" in m:
        return "list_files", {}
    return None, {}

def mcp_assistant(message):
    try:
        r = get_client().chat.completions.create(
            model=MODEL, tools=TOOL_SCHEMAS, tool_choice="auto",
            messages=[{"role": "system", "content": "You are a desktop assistant. Use a tool when helpful."},
                      {"role": "user", "content": message}])
        choice = r.choices[0].message
        if choice.tool_calls:
            tc = choice.tool_calls[0]
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = TOOLS[name](**args)
            final = ask(f"The user asked: {message}\nTool '{name}' returned: {result}\n"
                        f"Give a short, friendly reply using that result.",
                        "You compose the final answer from a tool result.")
            trace = f"1. Host brain (OpenAI) chose: {name}\n2. Arguments: {args}\n3. Tool result: {result}"
            return final, trace
        return choice.content or "(no tool needed)", "Host brain answered directly (no tool)."
    except Exception:
        name, args = router(message)
        if not name:
            return ("I can tell the time, do maths, and read/write notes.",
                    "Built-in router: no matching tool.")
        result = TOOLS[name](**args)
        return result, f"1. Built-in router chose: {name}\n2. Arguments: {args}\n3. Tool result: {result}"


# ============================================================ UI
st.set_page_config(page_title="GenAI Projects — Demo Hub", page_icon="🚀", layout="wide")
st.title("🚀 GenAI Projects — Demo Hub")

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("No OpenAI key set yet. Add it in **Settings → Secrets** as "
               "`OPENAI_API_KEY = \"sk-...\"` for the live answers to work. "
               "The MCP tools below still work without a key.")

home, t1, t5, t6, t10 = st.tabs([
    "🏠 Home", "1 · LLM Playground", "5 · Document QA (RAG)",
    "6 · Advanced RAG (Legal)", "10 · MCP Assistant"])

with home:
    st.markdown("""
Four enterprise Generative-AI projects, each in its own tab above.

| # | Project | What it does | Client |
|---|---------|--------------|--------|
| 1 | LLM Playground & Token Explorer | Control how an LLM answers; see tokens & cost | Amazon |
| 5 | Enterprise Document QA (RAG) | Answers grounded in documents, with sources | Tata AIA |
| 6 | Advanced RAG (Hybrid + Rerank) | Basic vs advanced retrieval on legal contracts | Cyril Amarchand Mangaldas |
| 10 | MCP-style Desktop Assistant | One AI brain picks the right tool | Infosys |

*Full source for each project is in the GitHub repo (one folder each). This is a combined demo.
The first click on the RAG tabs takes a few seconds to build the search index.*
""")

# ---- Project 1 ----
with t1:
    st.subheader("LLM Playground & Token Explorer — Amazon")
    sub_play, sub_tok = st.tabs(["Playground", "Token Explorer"])
    with sub_play:
        system = st.text_input("System prompt", "You are a helpful Amazon e-commerce assistant.")
        prompt = st.text_area("Your prompt", "Write a short product description for wireless earbuds.")
        c1, c2, c3 = st.columns(3)
        temperature = c1.slider("temperature", 0.0, 2.0, 0.7, 0.1)
        top_p = c2.slider("top_p", 0.1, 1.0, 1.0, 0.05)
        max_tokens = c3.slider("max_tokens", 50, 800, 300, 50)
        c4, c5 = st.columns(2)
        fpen = c4.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1)
        ppen = c5.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1)
        if st.button("Generate", type="primary", key="p1gen"):
            if not prompt.strip():
                st.info("Please enter a prompt.")
            else:
                try:
                    with st.spinner("Generating…"):
                        r = get_client().chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": prompt}],
                            temperature=temperature, top_p=top_p, max_tokens=int(max_tokens),
                            frequency_penalty=fpen, presence_penalty=ppen)
                    u = r.usage
                    cost = u.prompt_tokens / 1e6 * PIN + u.completion_tokens / 1e6 * POUT
                    st.write(r.choices[0].message.content)
                    st.caption(f"total tokens: {u.total_tokens} | cost: ${cost:.6f}")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
    with sub_tok:
        text = st.text_area("Text to tokenize", "Amazon Prime delivers unbelievably fast!", key="p1tok")
        if st.button("Explore tokens", key="p1tokbtn"):
            try:
                enc = get_enc()
                ids = enc.encode(text or "")
                pieces = [enc.decode([i]) for i in ids]
                st.write(f"**Token count:** {len(ids)} &nbsp; | &nbsp; **Words:** {len((text or '').split())} "
                         f"&nbsp; | &nbsp; **Cost as input:** ${len(ids)/1e6*PIN:.6f}")
                st.code("|".join(pieces))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

# ---- Project 5 ----
with t5:
    st.subheader("Enterprise Document QA — Tata AIA Life Insurance")
    st.caption("Ask about the sample insurance documents, or upload your own PDFs/TXT.")
    question = st.text_input("Question", "Is there a grace period to pay my premium?", key="p5q")
    k5 = st.slider("top_k", 1, 8, 5, key="p5k")
    if st.button("Answer", type="primary", key="p5btn"):
        if not question.strip():
            st.info("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving and answering…"):
                    hits = p5_store().search(embed([question]), k=int(k5))
                    ctx = "\n\n".join(f"- {h[0]['text']}" for h in hits) or "(none)"
                    ans = ask(f"Context:\n{ctx}\n\nQuestion: {question}",
                              "You are a Tata AIA Life Insurance assistant. Answer only from the "
                              "context; if it isn't there, say you don't have it. Be concise.")
                st.write(ans)
                srcs = list(dict.fromkeys(h[0]["source"] for h in hits))
                st.caption("Sources: " + ", ".join(srcs))
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
    st.divider()
    st.markdown("**Add your own documents**")
    files = st.file_uploader("Upload .pdf / .txt", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Index uploaded documents", key="p5up"):
        if not files:
            st.info("No files selected (sample documents are already indexed).")
        else:
            try:
                added = 0
                for f in files:
                    if f.name.lower().endswith(".pdf"):
                        from pypdf import PdfReader
                        import io
                        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(f.read())).pages)
                    else:
                        text = f.read().decode("utf-8", errors="ignore")
                    ch = chunk_text(text)
                    if ch:
                        p5_store().add(embed(ch), [{"text": c, "source": f.name} for c in ch])
                        added += len(ch)
                st.success(f"Indexed {added} new chunks. Ask about them above.")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

# ---- Project 6 ----
with t6:
    st.subheader("Advanced RAG: Hybrid Search + Rerank — Cyril Amarchand Mangaldas")
    st.caption("See how advanced retrieval beats basic on the same legal question. "
               "Try: *What does Clause 9.1 say?* or *Who pays if a third party sues us for using their patent?*")
    q6 = st.text_input("Legal question", "What does Clause 9.1 say?", key="p6q")
    k6 = st.slider("top_k", 2, 6, 4, key="p6k")
    if st.button("Compare basic vs advanced", type="primary", key="p6btn"):
        if not q6.strip():
            st.info("Please enter a legal question.")
        else:
            try:
                with st.spinner("Running basic and advanced retrieval…"):
                    idx = p6_index()
                    chunks, store, bm25 = idx["chunks"], idx["store"], idx["bm25"]
                    dense = [h[0] for h in store.search(embed([q6]), k=int(k6))]
                    def answer_from(hits):
                        ctx = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits) or "(none)"
                        a = ask(f"Context:\n{ctx}\n\nQuestion: {q6}",
                                "You are a legal research assistant for Cyril Amarchand Mangaldas. "
                                "Answer using ONLY the context. Cite the clause number or document "
                                "name where possible. If it isn't in the context, say so. Be concise.")
                        return a, list(dict.fromkeys(h["source"] for h in hits))
                    basic_ans, basic_src = answer_from(dense)
                    scores = bm25.get_scores(re.findall(r"[a-z0-9.]+", q6.lower()))
                    top_bm = [chunks[i] for i in np.argsort(scores)[::-1][:int(k6)]]
                    seen, hybrid = set(), []
                    for h in top_bm + dense:
                        key = h["text"][:40]
                        if key not in seen:
                            seen.add(key); hybrid.append(h)
                    adv_ans, adv_src = answer_from(hybrid[:int(k6) + 2])
                col_b, col_a = st.columns(2)
                with col_b:
                    st.markdown("**BASIC RAG (dense only)**")
                    st.write(basic_ans)
                    st.caption("from: " + ", ".join(basic_src))
                with col_a:
                    st.markdown("**ADVANCED RAG (hybrid)**")
                    st.write(adv_ans)
                    st.caption("from: " + ", ".join(adv_src))
                st.info("Advanced = hybrid retrieval (BM25 keyword + dense embeddings). It reliably catches "
                        "exact references like 'Clause 9.1' or 'Section 74' that pure embedding search often misses.")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

# ---- Project 10 ----
with t10:
    st.subheader("MCP-style Desktop Assistant — Infosys")
    st.caption("One AI brain picks the right tool: time · calculate · list/read/write notes. "
               "Try: *what time is it?* · *calculate 1250 * 12 + 300* · "
               "*save a note called plan.txt saying Deliver Project 10* · *read note plan.txt*")
    q10 = st.text_input("Ask the assistant", "Calculate 1250 * 12 + 300", key="p10q")
    if st.button("Send", type="primary", key="p10btn"):
        if not q10.strip():
            st.info("Type a request.")
        else:
            with st.spinner("Thinking…"):
                answer, trace = mcp_assistant(q10)
            st.write(answer)
            st.caption("Backend trace:")
            st.code(trace)
