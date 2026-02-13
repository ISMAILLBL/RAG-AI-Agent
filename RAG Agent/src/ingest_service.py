# src/ingest_service.py
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- ENV ---
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = os.getenv("PINECONE_INDEX", "rag-demo")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Clients ---
client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(INDEX_NAME)

# --- Splitter ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def _embed_batch(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]

def _upsert_chunks(doc_id: str, title: str, chunks: list[str]):
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        batch_texts = chunks[i:i + BATCH]
        vectors = _embed_batch(batch_texts)

        payload = []
        for j, vec in enumerate(vectors):
            cid = f"{doc_id}#c{i+j:06d}"
            payload.append({
                "id": cid,
                "values": vec,
                "metadata": {
                    "document_id": doc_id,
                    "document_title": title,
                    "chunk_number": i + j,
                    "chunk_text": batch_texts[j],
                    "source": title
                },
            })
        index.upsert(vectors=payload)

def ingest_pdf_path(path: Path) -> int:
    """
    Ingestion d'un PDF (fichier local) dans Pinecone.
    Retourne le nombre de chunks ingérés.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    text = _pdf_to_text(path)
    if not text.strip():
        raise ValueError(f"Aucun texte lisible dans: {path.name}")

    chunks = splitter.split_text(text)
    doc_id = uuid.uuid4().hex[:12]
    _upsert_chunks(doc_id, path.name, chunks)
    print(f"✅ Ingestion terminée: {path.name} ({len(chunks)} chunks)")
    return len(chunks)
