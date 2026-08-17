"""
Project 6 - Advanced RAG with Reranking
Legal Research & Document Q&A Assistant | Client: Cyril Amarchand Mangaldas

Runnable script exported from the Colab notebook. Reads OPENAI_API_KEY from the
environment (or prompts once if unset). Install deps: pip install -r requirements.txt
Run: python legal_rag.py
"""


import os, getpass

# OpenAI — required. Get a key at platform.openai.com
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# Cohere — OPTIONAL (only for section 3.1b)
# os.environ["COHERE_API_KEY"] = getpass.getpass("Enter your Cohere API key: ")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# One embedding model + one chat model, reused everywhere (same choices as Project 5)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Models ready.")

from langchain_core.documents import Document

documents = [
    Document(
        page_content=(
            "MASTER SERVICES AGREEMENT (Acme Corp and Vendor).\n"
            "Clause 9.1 Limitation of Liability: the total aggregate liability of either party "
            "under this Agreement shall not exceed the fees paid in the twelve (12) months "
            "preceding the event giving rise to the claim.\n"
            "Clause 12.3 Termination for Convenience: either party may terminate this Agreement "
            "for convenience by giving thirty (30) days' prior written notice to the other party.\n"
            "Clause 15 Indemnity: the Vendor shall defend, indemnify and hold harmless the Client "
            "against all claims arising from any allegation that the Services infringe a third "
            "party's intellectual property rights.\n"
            "Clause 18 Governing Law and Dispute Resolution: this Agreement is governed by the laws "
            "of India. Any dispute shall be referred to arbitration seated in Mumbai under the "
            "Arbitration and Conciliation Act, 1996."
        ),
        metadata={"source": "MSA.pdf", "doc_type": "contract"},
    ),
    Document(
        page_content=(
            "NON-DISCLOSURE AGREEMENT.\n"
            "The Receiving Party shall keep Confidential Information secret and shall not use it "
            "for any purpose other than the Permitted Purpose for a period of five (5) years from "
            "the date of disclosure.\n"
            "Permitted disclosure: Confidential Information may be disclosed to employees and "
            "professional advisers strictly on a need-to-know basis."
        ),
        metadata={"source": "NDA.pdf", "doc_type": "contract"},
    ),
    Document(
        page_content=(
            "EMPLOYMENT AGREEMENT.\n"
            "Notice period: either party may end the employment by giving ninety (90) days' "
            "written notice.\n"
            "Clause 7 Non-Compete: for twelve (12) months after termination, the Employee shall "
            "not join a direct competitor operating within India."
        ),
        metadata={"source": "Employment.pdf", "doc_type": "contract"},
    ),
    Document(
        page_content=(
            "DATA PROTECTION ADDENDUM.\n"
            "Personal Data breach notification: the Processor shall notify the Controller without "
            "undue delay and in any event within seventy-two (72) hours of becoming aware of a "
            "Personal Data breach.\n"
            "Sub-processors: the Processor may engage sub-processors only with the Controller's "
            "prior written authorisation."
        ),
        metadata={"source": "DPA.pdf", "doc_type": "contract"},
    ),
    Document(
        page_content=(
            "CASE-LAW NOTE — Liquidated Damages vs Penalty.\n"
            "Section 74 of the Indian Contract Act, 1872 allows a party to recover reasonable "
            "compensation not exceeding the amount named in the contract, whether or not actual "
            "loss or damage is proven. Courts distinguish a genuine pre-estimate of loss "
            "(enforceable) from a penalty (not enforceable)."
        ),
        metadata={"source": "CaseLawNote.pdf", "doc_type": "memo"},
    ),
]
print(f"Loaded {len(documents)} documents.")

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Small chunks so each clause is retrievable on its own
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"{len(documents)} documents -> {len(chunks)} chunks")

# To use the firm's REAL documents instead of the sample set, replace the block above with:
#
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# loader = DirectoryLoader("contracts/", glob="**/*.pdf", loader_cls=PyPDFLoader)
# documents = loader.load()
# chunks = splitter.split_documents(documents)

from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Vector index over the chunks
vectorstore = FAISS.from_documents(chunks, embeddings)

# Basic retriever: top-4 by embedding similarity (the Project 5 setting)
basic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Context-only prompt: grounded, with a citation instruction
answer_prompt = ChatPromptTemplate.from_template(
    "You are a legal research assistant for Cyril Amarchand Mangaldas.\n"
    "Answer the question using ONLY the context below.\n"
    "If the answer is not in the context, say you cannot find it in the provided documents.\n"
    "Where possible, cite the clause number or document name.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}\n"
)

document_chain = create_stuff_documents_chain(llm, answer_prompt)
basic_rag = create_retrieval_chain(basic_retriever, document_chain)
print("Basic RAG ready.")

# Small helper to print retrieved chunks in rank order
def show_chunks(docs, title="Retrieved chunks"):
    print(f"\n{title} ({len(docs)}):")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        preview = " ".join(d.page_content.split())
        print(f"  {i}. [{src}] {preview[:90]}...")

# Four questions, each chosen to stress a different weakness of basic retrieval
Q_SEMANTIC   = "If we want to exit the master services agreement early, how much notice must we give?"
Q_EXACT      = "What does Clause 9.1 say?"
Q_PARAPHRASE = "Who pays if a third party sues us for using their patent?"
Q_JURIS      = "Which country's law governs the contract and where are disputes resolved?"

for q in [Q_SEMANTIC, Q_EXACT, Q_PARAPHRASE]:
    print("=" * 80)
    print("Q:", q)
    resp = basic_rag.invoke({"input": q})
    print("\nBASIC answer:\n", resp["answer"])
    show_chunks(resp["context"], "Basic retrieved")

from sentence_transformers import CrossEncoder

# Reads (question, chunk) together and scores the match. Downloads once (~90 MB).
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, k_fetch=15, k_keep=4):
    # 1) fetch a big shortlist with fast embedding search
    candidates = vectorstore.similarity_search(query, k=k_fetch)
    # 2) score every (query, chunk) pair with the cross-encoder
    pairs = [[query, d.page_content] for d in candidates]
    scores = cross_encoder.predict(pairs)
    # 3) sort by score (high = better) and keep the best few
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:k_keep]]

# Compare ordering for the semantic query: embedding order vs reranked order
print("Q:", Q_SEMANTIC)
show_chunks(vectorstore.similarity_search(Q_SEMANTIC, k=4), "BEFORE rerank (embedding order)")
show_chunks(rerank(Q_SEMANTIC), "AFTER rerank (cross-encoder order)")

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

ce_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=ce_model, top_n=4)

# base fetches 15 by embedding; the reranker keeps the best 4
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
)
show_chunks(rerank_retriever.invoke(Q_SEMANTIC), "Reranked (LangChain retriever)")

# OPTIONAL — hosted alternative to the local cross-encoder.
# !pip install -q langchain-cohere cohere
# from langchain_cohere import CohereRerank
# from langchain.retrievers import ContextualCompressionRetriever
# cohere_reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
# cohere_retriever = ContextualCompressionRetriever(
#     base_compressor=cohere_reranker,
#     base_retriever=vectorstore.as_retriever(search_kwargs={"k": 15}),
# )
# show_chunks(cohere_retriever.invoke(Q_SEMANTIC), "Reranked (Cohere)")

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 = keyword search (nails "Clause 9.1", "Section 74", party names)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 15

# dense = embedding search (nails meaning / paraphrase)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],
)

# The exact-reference query that pure embeddings often fumble:
print("Q:", Q_EXACT)
show_chunks(vectorstore.similarity_search(Q_EXACT, k=4), "Dense only")
show_chunks(hybrid_retriever.invoke(Q_EXACT)[:4], "Hybrid (BM25 + dense)")

import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)  # show generated variants

from langchain.retrievers.multi_query import MultiQueryRetriever

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
)
print("Q:", Q_PARAPHRASE)
docs = multiquery_retriever.invoke(Q_PARAPHRASE)
show_chunks(docs, "Multi-query results")

# Advanced retriever = Hybrid (BM25 + dense, 15 each)  ->  cross-encoder rerank (keep best 4)
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,          # from 3.1
    base_retriever=hybrid_retriever,   # from 3.2
)

advanced_rag = create_retrieval_chain(advanced_retriever, document_chain)
print("Advanced RAG ready.")

# (Optional) add query transformation in front of everything:
# advanced_retriever_mq = ContextualCompressionRetriever(
#     base_compressor=reranker,
#     base_retriever=MultiQueryRetriever.from_llm(retriever=hybrid_retriever, llm=llm),
# )

# Basic vs Advanced answers, side by side
for q in [Q_SEMANTIC, Q_EXACT, Q_PARAPHRASE, Q_JURIS]:
    print("=" * 80)
    print("Q:", q)
    b = basic_rag.invoke({"input": q})["answer"]
    a = advanced_rag.invoke({"input": q})["answer"]
    print("\n[BASIC]   ", b)
    print("\n[ADVANCED]", a)

# For each question we know the "gold" text that a good chunk MUST contain.
gold = {
    Q_SEMANTIC:   "Clause 12.3",   # termination for convenience
    Q_EXACT:      "Clause 9.1",    # limitation of liability
    Q_PARAPHRASE: "indemnif",      # Clause 15 indemnity
    Q_JURIS:      "arbitration",   # governing law / Mumbai arbitration
}

def rank_of_gold(docs, needle):
    for i, d in enumerate(docs, 1):
        if needle.lower() in d.page_content.lower():
            return i
    return None

def basic_docs(q):  return basic_retriever.invoke(q)      # top-4 dense
def adv_docs(q):    return advanced_retriever.invoke(q)    # hybrid + rerank -> best 4

print(f"{'Question':<45}{'Basic':<10}{'Advanced'}")
print("-" * 70)
for q, needle in gold.items():
    rb, ra = rank_of_gold(basic_docs(q), needle), rank_of_gold(adv_docs(q), needle)
    fmt = lambda r: "miss" if r is None else f"#{r}"
    print(f"{q[:43]:<45}{fmt(rb):<10}{fmt(ra)}")

print("\nLower rank is better;  '#1' = the correct clause came first;  'miss' = not retrieved.")

from langchain_core.output_parsers import StrOutputParser

judge_prompt = ChatPromptTemplate.from_template(
    "You are a strict evaluator. Given the CONTEXT and an ANSWER, reply with ONE word: "
    "GROUNDED if every claim in the answer is supported by the context, or UNSUPPORTED otherwise.\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n"
)
judge = judge_prompt | llm | StrOutputParser()

def faithfulness(q, rag):
    out = rag.invoke({"input": q})
    ctx = "\n\n".join(d.page_content for d in out["context"])
    verdict = judge.invoke({"context": ctx, "answer": out["answer"]}).strip()
    return verdict, out["answer"]

for q in [Q_PARAPHRASE, Q_JURIS]:
    v, ans = faithfulness(q, advanced_rag)
    print(f"Q: {q}\n  verdict: {v}\n  answer : {ans}\n")

# OPTIONAL — RAGAS gives faithfulness, answer relevancy, context precision & recall.
# The RAGAS API changes across versions; if this errors, check the current RAGAS docs.
try:
    # !pip install -q ragas datasets
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness as r_faith, answer_relevancy, context_precision, context_recall,
    )

    eval_questions = [Q_SEMANTIC, Q_EXACT, Q_PARAPHRASE, Q_JURIS]
    ground_truth = [
        "Either party may terminate for convenience on 30 days' written notice (Clause 12.3).",
        "Clause 9.1 caps each party's total liability at the fees paid in the prior 12 months.",
        "The Vendor must indemnify the Client against third-party IP infringement claims (Clause 15).",
        "Indian law governs; disputes go to arbitration seated in Mumbai (Clause 18).",
    ]
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q, gt in zip(eval_questions, ground_truth):
        out = advanced_rag.invoke({"input": q})
        rows["question"].append(q)
        rows["answer"].append(out["answer"])
        rows["contexts"].append([d.page_content for d in out["context"]])
        rows["ground_truth"].append(gt)

    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[r_faith, answer_relevancy, context_precision, context_recall],
    )
    print(result)
except Exception as e:
    print("RAGAS optional step skipped:", type(e).__name__, e)
    print("The retrieval-quality table (4.1) and faithfulness check (4.2) already give a solid before/after.")