import os, hashlib
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-demo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------- Utils ----------
def pdf_to_text_from_path(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def pdf_to_text_from_bytes(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

def embed_batch(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

def stable_doc_id(text: str, user_id: str) -> str:
    return hashlib.sha1((user_id + "|" + text).encode("utf-8", errors="ignore")).hexdigest()[:16]

def upsert_chunks(doc_id: str, title: str, user_id: str, chunks: list[str]):
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch_texts = chunks[i:i+BATCH]
        vectors = embed_batch(batch_texts)
        payload = []
        for j, vec in enumerate(vectors):
            cid = f"{doc_id}#c{i+j:06d}"
            payload.append({
                "id": cid,
                "values": vec,
                "metadata": {
                    "user_id": user_id,
                    "document_id": doc_id,
                    "document_title": title,
                    "chunk_number": i + j,
                    "chunk_text": batch_texts[j],
                    "source": title,
                },
            })
        index.upsert(vectors=payload)

# ---------- Public API (réutilisable par FastAPI) ----------
def ingest_pdf_bytes(filename: str, data: bytes, user_id: str) -> int:
    """Ingest un PDF en mémoire. Retourne le nombre de chunks."""
    text = pdf_to_text_from_bytes(data)
    if not text or not text.strip():
        return 0
    chunks = [c.strip() for c in splitter.split_text(text) if c and c.strip()]
    if not chunks:
        return 0

    # supprimer anciennes versions (même user + même titre)
    index.delete(filter={"user_id": user_id, "document_title": filename})

    doc_id = stable_doc_id(text, user_id)
    upsert_chunks(doc_id, filename, user_id, chunks)
    return len(chunks)

def ingest_path(path: Path, user_id: str) -> int:
    text = pdf_to_text_from_path(path)
    if not text or not text.strip():
        return 0
    chunks = [c.strip() for c in splitter.split_text(text) if c and c.strip()]
    if not chunks:
        return 0
    index.delete(filter={"user_id": user_id, "document_title": path.name})
    doc_id = stable_doc_id(text, user_id)
    upsert_chunks(doc_id, path.name, user_id, chunks)
    return len(chunks)

if __name__ == "__main__":
    # Ingestion en local pour un user "local"
    user = "local"
    data_dir = Path("data")
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print("Place des PDF dans /data puis relance.")
    for p in tqdm(pdfs):
        n = ingest_path(p, user)
        print(f"✅ {p.name}: {n} chunks")
