"""Streamlit web interface for the PSB Legal Assistant.

Run with: streamlit run src/ui/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8000/answer")

st.set_page_config(page_title="PSB Legal Assistant", layout="wide")

# ----- Session state -----
if "history" not in st.session_state:
    st.session_state.history = []
if "sources" not in st.session_state:
    st.session_state.sources = []


def call_api(question: str) -> tuple[str, list]:
    payload = {"question": question, "history": st.session_state.history[-8:]}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", ""), data.get("sources", [])


# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Настройки")
    st.caption("API: " + API_URL)
    if st.button("🗑️ Очистить историю"):
        st.session_state.history = []
        st.session_state.sources = []
        st.rerun()

    st.divider()
    st.header("📚 База знаний")
    st.caption("Документы загружены в векторную базу.")
    st.caption("Используются: BAAI/bge-m3, Qwen3 Reranker, DeepSeek V4.1 Flash")

# ----- Main chat area -----
st.title("⚖️ PSB Legal Assistant")
st.caption("Ответы только на основе предоставленной базы знаний")

# Display history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show latest sources below assistant message
if st.session_state.sources:
    with st.expander("📄 Источники последнего ответа", expanded=False):
        for i, src in enumerate(st.session_state.sources, 1):
            doc = src.get("document", "—")
            clause = src.get("clause", "—")
            revision = src.get("revision_date") or "не установлена"
            score = src.get("score", 0.0)
            snippet = src.get("text", "")[:500]
            st.markdown(f"**[{i}]** {doc} — {clause} (редакция: {revision}, score={score:.3f})")
            st.code(snippet + ("…" if len(src.get("text", "")) > 500 else ""), language="text")

# Chat input
user_input = st.chat_input("Ваш вопрос…")
if user_input:
    # Add user message
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Ищу в базе знаний…"):
            try:
                answer_text, sources = call_api(user_input)
            except Exception as exc:
                answer_text = f"❌ Ошибка обращения к API: {exc}"
                sources = []

        st.markdown(answer_text)
        st.session_state.history.append({"role": "assistant", "content": answer_text})
        st.session_state.sources = sources

    st.rerun()