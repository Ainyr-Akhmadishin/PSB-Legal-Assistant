"""Единая точка входа: JSON stdin -> JSON stdout или questions.csv -> submission.csv.

Платформенный режим: python solution.py (DATA_DIR/data/questions.csv + kb.zip).
JSON режим: python solution.py --json  (на stdin передаётся {"question": ..., "history": [...]})
"""
from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.legal.core import LegalRAG, REFUSAL  # noqa: E402

KB_NAMES = ("kb.zip", "knowledge-base.zip", "knowledge_base.zip", "knowledge", "documents")


def find_kb(directory: Path) -> Path:
    """Ищет архив базы знаний в каталоге данных, затем в корне проекта."""
    for name in KB_NAMES:
        path = directory / name
        if path.is_file():
            return path
    for name in KB_NAMES:
        path = ROOT.parent / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Не найден kb.zip / knowledge-base.zip в %s или в корне проекта" % directory
    )


def data_dir() -> Path:
    path = Path(os.getenv("DATA_DIR", "data"))
    return path if path.is_absolute() else ROOT / path


def respond(payload: dict, rag: LegalRAG) -> dict:
    """Приводит разные форматы входа к единому виду и возвращает ответ с источниками."""
    if not isinstance(payload, dict):
        raise ValueError("Ожидается JSON-объект")
    history = payload.get("history") or payload.get("messages") or []
    if not isinstance(history, list):
        raise ValueError("history должен быть массивом")
    question = payload.get("question") or payload.get("text")
    if not question and history:
        last = history[-1]
        if isinstance(last, dict):
            question = last.get("content") or last.get("text")
            history = history[:-1]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Нужен непустой question или последняя реплика пользователя в history")
    clean = []
    for item in history[-8:]:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            content = item.get("content", item.get("text"))
            if isinstance(content, str):
                clean.append({"role": item["role"], "content": content})
    return rag.answer(question.strip(), clean)


def run_json() -> None:
    rag = LegalRAG(find_kb(data_dir()))
    payload = json.load(sys.stdin)
    print(json.dumps(respond(payload, rag), ensure_ascii=False))


def run_csv() -> None:
    directory = data_dir()
    path = directory / "questions.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("Ожидаются колонки id,question")
        questions = [(r["id"], r["question"]) for r in reader]
    rag = LegalRAG(find_kb(directory))
    answers = [REFUSAL] * len(questions)

    def work(index: int, question: str):
        try:
            return index, respond({"question": question}, rag)["answer"]
        except Exception as exc:  # noqa: BLE001 - один сбойный вопрос не роняет прогон
            print("Ошибка вопроса %d: %s" % (index, exc), file=sys.stderr)
            return index, REFUSAL

    workers = max(1, min(8, int(os.getenv("MAX_WORKERS", "2") or 2)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, i, q) for i, (_, q) in enumerate(questions)]
        for future in as_completed(futures):
            index, answer = future.result()
            answers[index] = answer
    with Path("submission.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for index, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[index]])
    print("Создан submission.csv: %d ответов" % len(questions), file=sys.stderr)


if __name__ == "__main__":
    if "--json" in sys.argv:
        run_json()
    elif not sys.stdin.isatty() and not (data_dir() / "questions.csv").exists():
        run_json()
    else:
        run_csv()
