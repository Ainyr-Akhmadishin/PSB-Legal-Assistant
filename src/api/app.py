"""FastAPI wrapper for the LegalRAG assistant.

POST /answer
    JSON body: {"question": str, "history": [{"role": "user"|"assistant", "content": str}, ...]}
    Returns: {"answer": str, "sources": [...]}
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.legal.core import LegalRAG  # noqa: E402

app = FastAPI(title="PSB Legal Assistant")


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class Query(BaseModel):
    question: str = Field(..., description="User question")
    history: list[Message] | None = Field(default_factory=list, description="Previous dialogue turns")


KB_PATH = None
for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge", "documents"):
    candidate = Path(os.getenv("DATA_DIR", "data")) / name
    if candidate.is_file():
        KB_PATH = candidate
        break
if not KB_PATH:
    for name in ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge", "documents"):
        candidate = ROOT / name
        if candidate.is_file():
            KB_PATH = candidate
            break
if not KB_PATH:
    raise RuntimeError("Knowledge base not found")

the_rag = LegalRAG(KB_PATH)


@app.post("/answer")
def answer(query: Query):
    try:
        result = the_rag.answer(
            query.question,
            [{"role": m.role, "content": m.content} for m in (query.history or [])]
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(the_rag.chunks)}
