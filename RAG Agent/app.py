import os
import requests
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="RAG Agent", layout="centered")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")  # ton API FastAPI

# ---------- UI ----------
st.title("🧠 RAG Agent")
st.caption("Pose une question — l’agent va chercher dans Pinecone et répondre en se basant uniquement sur le PDF ingéré.")

with st.sidebar:
    st.subheader("Paramètres")
    st.write("Le backend lit TOP_K & MIN_SCORE depuis `.env`.")
    st.code("uvicorn src.api:app --reload --port 8000", language="bash")
    st.write("Backend URL:")
    backend_input = st.text_input("Backend URL", BACKEND_URL)
    if backend_input:
        BACKEND_URL = backend_input

# Historique (session)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher historique
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s.get('title','?')}** · score={s.get('score')} · chunk={s.get('chunk')}")

# Zone d’entrée
query = st.chat_input("Écris ta question…")

# Envoi
if query:
    # afficher question côté UI
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # appel backend
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json={"query": query}, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "").strip()
            sources = data.get("sources", [])

            # afficher réponse
            with st.chat_message("assistant"):
                st.markdown(answer if answer else "_(Réponse vide)_")
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- **{s.get('title','?')}** · score={s.get('score')} · chunk={s.get('chunk')}")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        else:
            with st.chat_message("assistant"):
                st.error(f"Erreur backend {resp.status_code}: {resp.text}")
            st.session_state.messages.append({"role": "assistant", "content": f"Erreur backend {resp.status_code}."})
    except requests.exceptions.RequestException as e:
        with st.chat_message("assistant"):
            st.error(f"Impossible de contacter le backend à {BACKEND_URL}\n\n{e}")
        st.session_state.messages.append({"role": "assistant", "content": "Backend injoignable."})
