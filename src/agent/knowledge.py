from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Index location (absolute, resolved from this file) ────────────────────────
# src/agent/knowledge.py → src/agent → src → project root → data/faiss_index
INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "faiss_index"

# ── Load the prebuilt FAISS index ─────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

# ── Retriever (top-5 semantic matches) ────────────────────────────────────────
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ── Helper: search and return formatted context ──────────────────────────────
def search_knowledge(query: str) -> str:
    """Search the ATS knowledge base. Returns top-5 relevant docs as text."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = search_knowledge("how to add missing cloud skills to resume")
    print(result)