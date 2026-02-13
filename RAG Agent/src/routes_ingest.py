# src/routes_ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil
import mimetypes

from src.ingest_service import ingest_pdf_path  # service d’ingestion

router = APIRouter()

def _save_tmp(file: UploadFile) -> Path:
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .pdf sont acceptés.")

    mime = file.content_type or mimetypes.guess_type(filename)[0] or ""
    if "pdf" not in mime and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Fichier non-PDF.")

    tmp_dir = Path("tmp_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / filename
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return tmp_path

@router.post("/upload")
async def ingest_upload(file: UploadFile = File(...)):
    print("📥 [ingest] Reçu 1 fichier:", file.filename)
    try:
        tmp_path = _save_tmp(file)
    finally:
        await file.close()

    try:
        chunks = ingest_pdf_path(tmp_path)
    except Exception as e:
        try: tmp_path.unlink(missing_ok=True)
        except Exception: pass
        print("❌ [ingest] Erreur:", e)
        raise HTTPException(status_code=500, detail=f"Ingestion échouée: {e}")

    try: tmp_path.unlink(missing_ok=True)
    except Exception: pass

    print(f"✅ [ingest] OK {file.filename} → {chunks} chunks")
    return {"status":"ok","filename":file.filename,"message":"Fichier ingéré dans Pinecone.","chunks":chunks}

@router.post("/upload-multi")
async def ingest_upload_multi(files: List[UploadFile] = File(...)):
    print("📥 [ingest] Reçu", len(files), "fichiers")
    results = []
    for uf in files:
        try:
            tmp_path = _save_tmp(uf)
            chunks = ingest_pdf_path(tmp_path)
            results.append({"status":"ok","filename":uf.filename,"chunks":chunks})
        except HTTPException as he:
            results.append({"status":"error","filename":uf.filename,"detail":he.detail})
        except Exception as e:
            results.append({"status":"error","filename":uf.filename,"detail":str(e)})
        finally:
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass
            try: await uf.close()
            except Exception: pass
    print("✅ [ingest] Terminé multi")
    return {"items": results}
