#!/usr/bin/env python3
import os
import sys
import importlib
from pathlib import Path
from typing import Tuple

def section(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def check_python_version(min_major=3, min_minor=10) -> bool:
    ok = (sys.version_info.major, sys.version_info.minor) >= (min_major, min_minor)
    print(f"Python version: {sys.version.split()[0]}  ->  {'OK' if ok else 'Upgrade required (>=3.10)'}")
    return ok

def check_packages(pkgs) -> bool:
    ok_all = True
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            ver = getattr(m, '__version__', '?')
            print(f"Import {p:12s} -> OK (version {ver})")
        except Exception as e:
            ok_all = False
            print(f"Import {p:12s} -> FAILED: {e}")
    return ok_all

def load_env(env_path: Path) -> Tuple[bool, dict]:
    if not env_path.exists():
        print(f".env not found at: {env_path.resolve()}")
        return False, {}
    env = {}
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    print(f"Loaded {len(env)} vars from .env")
    return True, env

def sanity_env(env: dict) -> bool:
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_REGION", "PINECONE_CLOUD"]
    ok = True
    for k in required:
        if not env.get(k):
            print(f"Missing env var: {k}")
            ok = False
    emb = env.get("EMBED_MODEL", "text-embedding-3-small").strip()
    gen = env.get("GENERATION_MODEL", "gpt-4o").strip()
    print(f"EMBED_MODEL={emb}")
    print(f"GENERATION_MODEL={gen}")
    return ok

def tiny_live_test(env: dict) -> bool:
    try:
        from openai import OpenAI
        from pinecone import Pinecone
    except Exception as e:
        print("OpenAI or Pinecone package missing:", e)
        return False

    OPENAI_API_KEY = env["OPENAI_API_KEY"]
    PINECONE_API_KEY = env["PINECONE_API_KEY"]
    INDEX = env["PINECONE_INDEX"]
    EMBED_MODEL = env.get("EMBED_MODEL", "text-embedding-3-small").strip()

    print("\nCreating a test embedding with OpenAI ...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    emb = client.embeddings.create(model=EMBED_MODEL, input=["preflight test"]).data[0].embedding
    print(f"Embedding length: {len(emb)} (should be 1536 for text-embedding-3-small, 3072 for -large)")

    print("Connecting to Pinecone and listing indexes ...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    try:
        indexes = [it.name for it in pc.list_indexes()]
    except Exception:
        indexes = pc.list_indexes()
    print("Indexes:", indexes)
    if INDEX not in indexes:
        print(f"❌ Index '{INDEX}' not found in your project. Create it or fix PINECONE_INDEX.")
        return False

    print(f"Querying index '{INDEX}' with top_k=1 ...")
    index = pc.Index(INDEX)
    try:
        res = index.query(vector=emb, top_k=1, include_metadata=True)
        matches = getattr(res, 'matches', []) if hasattr(res, 'matches') else res.get('matches', [])
        print(f"Query OK. Received {len(matches)} matches (expected 0 before ingestion).")
        print("✅ Live test succeeded — API keys & index connectivity look good.")
        return True
    except Exception as e:
        print("❌ Query failed. Common reason: vector dimension mismatch with index.")
        print("Error:", e)
        return False

def check_project_layout():
    print("\nProject layout quick check:")
    expected = ["src", "data", "requirements.txt"]
    for name in expected:
        from pathlib import Path
        path = Path(name)
        print(f"- {name:16s}", "OK" if path.exists() else "MISSING")
    from pathlib import Path
    pdfs = list(Path("data").glob("*.pdf"))
    print(f"PDFs in data/: {len(pdfs)} found. (OK if 0 — just a heads-up)")

if __name__ == "__main__":
    section("1) Python & packages")
    ok_py = check_python_version()
    ok_pkgs = check_packages(["openai", "pinecone", "fastapi", "uvicorn", "pydantic", "pypdf", "tiktoken", "dotenv"])

    section("2) .env presence & variables")
    ok_env, env = load_env(Path(".env"))
    ok_env_vars = sanity_env(env) if ok_env else False

    section("3) Project layout")
    check_project_layout()

    section("4) Live test (OpenAI + Pinecone)")
    ok_live = False
    if ok_env_vars and ok_pkgs:
        try:
            ok_live = tiny_live_test(env)
        except Exception as e:
            print("Live test raised an exception:", e)
    else:
        print("Skipping live test due to previous failures.")

    section("SUMMARY")
    print(f"Python OK:         {ok_py}")
    print(f"Packages OK:       {ok_pkgs}")
    print(f".env & vars OK:    {ok_env_vars}")
    print(f"Live test OK:      {ok_live}")
    print("\nIf 'Live test' fails with a dimension error, recreate your Pinecone index with the correct dimension (1536 for text-embedding-3-small).")
