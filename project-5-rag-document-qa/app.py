import os, re, numpy as np, gradio as gr, faiss
from openai import OpenAI
client = OpenAI()                         # reads OPENAI_API_KEY from environment
MODEL, EMBED_MODEL = "gpt-4o-mini", "text-embedding-3-small"

def ask(prompt, system, temperature=0.2, max_tokens=400):
    r = client.chat.completions.create(model=MODEL, temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
    return r.choices[0].message.content

def embed(texts):
    if isinstance(texts, str): texts=[texts]
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")

def chunk_text(t, size=500, overlap=80):
    parts=[p for p in re.split(r"(\n\n+|(?<=[.!?])\s+)", t or "") if p and not p.isspace()]
    chunks, cur = [], ""
    for p in parts:
        if len(cur)+len(p)<=size: cur+=p
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur=(cur[-overlap:] if overlap and cur else "")+p
    if cur.strip(): chunks.append(cur.strip())
    return chunks

class Store:
    def __init__(s,d): s.i=faiss.IndexFlatIP(d); s.m=[]
    def add(s,v,me):
        v=np.ascontiguousarray(v,dtype="float32"); faiss.normalize_L2(v); s.i.add(v); s.m+=me
    def search(s,q,k=5):
        q=np.ascontiguousarray(q,dtype="float32"); faiss.normalize_L2(q)
        D,I=s.i.search(q,min(k,max(s.i.ntotal,1)))
        return [(s.m[i],float(D[0][j])) for j,i in enumerate(I[0]) if i!=-1]

STORE = Store(1536)
SEED = [{"title":"Grace Period","text":"A grace period of 30 days (15 for monthly mode) is allowed after the premium due date without losing benefits. If unpaid, the policy may lapse."},
        {"title":"Free-Look","text":"You may return the policy within 15 days (30 for distance marketing) of receipt; premium is refunded after minor deductions."},
        {"title":"Death Claim","text":"The nominee must submit the claim form, original policy, death certificate and ID proof; the death benefit is then paid to the nominee."}]
def index_docs(docs):
    ch,me=[],[]
    for d in docs:
        for c in chunk_text(d["text"]): ch.append(c); me.append({"text":c,"source":d["title"]})
    if ch: STORE.add(embed(ch), me)
    return len(ch)
index_docs(SEED)

def rag_answer(query, k=5, rerank=True, rewrite=True):
    sq = query
    if rewrite:
        try: sq = ask("Rewrite into a short search query. Return only the query.\n\n"+query,"You rewrite queries.",0,40).strip().strip('"')
        except Exception: pass
    hits = STORE.search(embed([sq]), k=k)
    ctx = "\n\n".join(f"- {h[0]['text']}" for h in hits) or "(none)"
    ans = ask(f"Context:\n{ctx}\n\nQuestion: {query}",
              "You are a Tata AIA Life Insurance assistant. Answer only from the context; else say you don't have it. Be concise.")
    return ans, list(dict.fromkeys(h[0]["source"] for h in hits))

def load_file(p):
    if p.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
    return open(p, encoding="utf-8", errors="ignore").read()

def ui_ask(q,k,rr,rw):
    if not q.strip(): return "Please enter a question.",""
    a,s = rag_answer(q, int(k), rr, rw)
    return a, "\n".join(f"- {x}" for x in s)
def ui_up(files):
    if not files: return "No files (seed docs already indexed)."
    docs=[{"title":(f if isinstance(f,str) else f.name).split("/")[-1],
           "text":load_file(f if isinstance(f,str) else f.name)} for f in files]
    return f"Indexed {index_docs(docs)} chunks. Store now has {STORE.i.ntotal}."

with gr.Blocks(title="Enterprise Document QA System") as demo:
    gr.Markdown("# Enterprise Document QA System\nTata AIA Life Insurance - RAG over your documents.")
    with gr.Tab("Ask"):
        q=gr.Textbox(label="Question", value="Is there a grace period to pay my premium?")
        with gr.Row():
            k=gr.Slider(1,8,value=5,step=1,label="top_k"); rr=gr.Checkbox(True,label="Rerank"); rw=gr.Checkbox(True,label="Rewrite")
        gr.Button("Answer").click(ui_ask,[q,k,rr,rw],[gr.Textbox(label="Answer",lines=8),gr.Textbox(label="Sources",lines=4)])
    with gr.Tab("Add documents"):
        up=gr.File(file_count="multiple", type="filepath", label="Upload .pdf / .txt")
        gr.Button("Index").click(ui_up, up, gr.Textbox(label="Status"))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT",7860)))