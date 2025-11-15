import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import csv
from datetime import datetime

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="SaaS Support Assistant", page_icon="🤖")

st.title("🤖 SaaS Support Assistant")
st.caption("Ask anything about billing, onboarding, API, security, and troubleshooting.")

# Load Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-2.5-flash")


# Load FAISS and metadata
try:
    index = faiss.read_index("saas_index.idx")
except:
    st.error("❌ Could not load FAISS index. Make sure saas_index.idx is in the repo root.")
    st.stop()

try:
    with open("saas_meta.pkl", "rb") as f:
        meta = pickle.load(f)
except:
    st.error("❌ Could not load metadata file: saas_meta.pkl")
    st.stop()

# Embedding model
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except:
    st.error("❌ Could not load SentenceTransformer. Check requirements.txt.")
    st.stop()

# ---------------------------
# Logging Setup
# ---------------------------
LOG_FILE = "query_logs.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "query", "answer", "is_relevant", "top_sources"])

def log_query(query, answer, is_relevant, top_docs):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            query,
            answer[:500],
            is_relevant,
            ", ".join([d["title"] for d in top_docs])
        ])

# ---------------------------
# RAG Retrieval
# ---------------------------
def retrieve(query, top_k=10):
    q = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        doc = meta[idx]
        results.append({
            "title": doc["title"],
            "source": doc["source"],
            "content": doc["content"],
            "score": float(score),
            "rerank": float(np.dot(embed_model.encode([doc["content"]])[0], q[0]))
        })

    return sorted(results, key=lambda x: x["rerank"], reverse=True)[:3]

# ---------------------------
# Irrelevance Handling
# ---------------------------
def is_irrelevant(docs, threshold=0.32):
    return all(d["rerank"] < threshold for d in docs)

def fallback_message():
    return (
        "I’m sorry, but I don't have information related to your query.\n\n"
        "Here are some things I *can* help you with:\n"
        "• Billing & Payments\n"
        "• API keys & Rate Limits\n"
        "• Login or Email Issues\n"
        "• Onboarding & Team Management\n"
        "• Integrations (Google Workspace, etc.)\n\n"
        "Please ask something related to these topics."
    )

# ---------------------------
# RAG + Gemini Answer
# ---------------------------
def rag_gemini(query):
    docs = retrieve(query)

    # If irrelevant → fallback response
    if is_irrelevant(docs):
        fb = fallback_message()
        log_query(query, fb, False, docs)
        return fb, docs

    context = "\n\n".join([d["content"] for d in docs])

    prompt = f"""
You are a SaaS customer-support assistant.
ONLY answer using the context below:

{context}

If the answer is not present in the context, say so. Do NOT hallucinate.

User question:
{query}
"""

    try:
        answer = gemini.generate_content(prompt).text
    except Exception as e:
        answer = "⚠️ Gemini API error. Please try again."

    log_query(query, answer, True, docs)
    return answer, docs

# ---------------------------
# UI Chat Input
# ---------------------------
query = st.chat_input("Ask your question...")

if query:
    st.chat_message("user").markdown(query)

    answer, docs = rag_gemini(query)

    st.chat_message("assistant").markdown(answer)

    with st.expander("📚 Sources Used"):
        for d in docs:
            st.markdown(f"- **{d['title']}** (Category: {d['source']})")
