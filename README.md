# PSB Legal Assistant — RAG-система для юридических сотрудников банка

ИИ-ассистент отвечает на вопросы **только** на основе предоставленной базы знаний (договоры, регламенты, нормативные акты). Каждое утверждение сопровождается ссылкой на источник. Если информации недостаточно — ассистент вежливо отказывается и указывает, чего не хватает.

## Возможности
- **Гибридный поиск**: BM25 (Okapi) + векторный поиск (BAAI/bge-m3) с RRF-фузией.
- **Реранкинг**: Qwen3 Reranker 8B (через API платформы).
- **LLM**: DeepSeek V4.1 Flash (через API платформы).
- **Жёсткий Anti-hallucination**: системный промпт запрещает внешние знания, требует цитирования `[Документ: ..., Статья N]`, защита от prompt-injection.
- **Учёт дат**: поиск применимой на указанную дату редакции документа.
- **Веб-интерфейс**: Streamlit-чат с просмотром источников и историей диалога.
- **Точка входа для автопроверки**: `solution.py` (JSON stdin/stdout или CSV batch).
- **Docker / Docker Compose**: один командой поднимает API и UI.

## Структура проекта
```
PSB-Legal-Assistant/
├── data/                       # База знаний (kb.zip) и questions.csv (монтируется в контейнер)
├── src/
│   ├── api/app.py              # FastAPI /answer endpoint
│   ├── legal/
│   │   ├── __init__.py
│   │   ├── core.py             # LegalRAG: поиск, реранкинг, генерация ответа
│   │   └── ingest.py           # Загрузка TXT/MD/DOCX/ZIP, чанкинг, метаданные
│   └── ui/streamlit_app.py     # Streamlit чат-интерфейс
├── solution.py                 # Единая точка входа для платформы
├── requirements.txt            # Зафиксированные зависимости
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # API (8000) + UI (8501)
├── .env.example                # Пример переменных окружения
└── README.md
```

## Быстрый старт (Docker Compose)

```bash
# 1. Клонируйте репозиторий и перейдите в папку
cd PSB-Legal-Assistant

# 2. Положите базу знаний в data/kb.zip (или knowledge-base.zip)
#    и (опционально) data/questions.csv для автопроверки.

# 3. Создайте .env на основе примера и укажите ключи API
cp .env.example .env
# отредактируйте .env: LLM_BASE_URL, LLM_API_KEY, EMB_MODEL, RERANKER_MODEL, RERANKER_TOP_N

# 4. Запустите
docker compose up --build -d

# 5. Откройте интерфейсы
#    API:  http://localhost:8000/docs   (Swagger)
#    UI:   http://localhost:8501
```

## Переменные окружения (`.env`)

| Переменная | Описание | Пример |
|------------|----------|--------|
| `LLM_BASE_URL` | Базовый URL DeepSeek API | `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | Ключ DeepSeek | `sk-...` |
| `EMB_MODEL` | Имя модели эмбеддингов | `BAAI/bge-m3` |
| `RERANKER_MODEL` | Имя реранкера | `Qwen3-Reranker-8B` |
| `RERANKER_TOP_N` | Сколько чанков реранжировать | `10` |
| `DATA_DIR` | Путь к данным внутри контейнера | `/app/data` |
| `API_URL` | Для UI — адрес API | `http://api:8000/answer` |

## API — ручной тест

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Каков порядок согласования договора аренды?", "history": []}'
```

Ответ:
```json
{
  "answer": "Согласно Регламенту согласования договоров ред 2025, договор аренды проходит согласование у юриста, комплаенс и руководителя направления... [Документ: Регламент согласования договоров ред 2025, п. 3.2]",
  "sources": [
    {"document": "Регламент согласования договоров ред 2025", "clause": "п. 3.2", "revision_date": "2025-01-01", ...}
  ]
}
```

## solution.py — режим автопроверки

Платформа вызывает:
```bash
python solution.py          # читает data/questions.csv, пишет submission.csv
python solution.py --json   # JSON stdin -> JSON stdout
```

Пример JSON stdin:
```json
{
  "question": "Какая сумма штрафа за просрочку в договоре П-17-2024?",
  "history": [
    {"role": "user", "content": "Есть ли у нас договор с Альфа-Трейд?"},
    {"role": "assistant", "content": "Да, Договор поставки П-17-2024 Альфа-Трейд от 15.03.2024..."}
  ]
}
```

JSON stdout:
```json
{
  "answer": "Штраф 0,1% от суммы просрочки за каждый день... [Документ: Договор поставки П-17-2024 Альфа-Трейд, п. 5.3]",
  "sources": [...]
}
```

## Разработка без Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(cat .env | xargs)   # загрузить переменные
python solution.py --json    # или uvicorn src.api.app:app --reload
streamlit run src/ui/streamlit_app.py
```

## Особенности реализации

1. **Чанкинг** сохраняет структуру: заголовок статьи/пункта (`clause`), название документа, дата редакции, флаг `obsolete`.
2. **Поиск**: BM25 + cosine similarity → RRF → реранкер → топ-N.
3. **Отказ (Refusal)**: если нет пересечения терминов (`MIN_TERM_OVERLAP=1`) или векторный скор `< 0.20` и BM25 ≤ 0.
4. **Цитирование**: LLM обязана ставить `[Документ: точное название, Статья/пункт N]`. Ответ проверяется: все упомянутые документы должны быть среди переданных чанков.
5. **Даты**: из вопроса извлекается дата (`на 15.10.2024`, `в 2024 году`, `по состоянию на...`). В поиск попадают только чанки с `revision_date ≤ дата вопроса` (если дата известна).
6. **Prompt Injection защита**: системный промпт помечает выдержки как «недоверенные данные», запрещает исполнять инструкции из них и менять роль.

## Лицензия
MIT — используйте и адаптируйте под задачи банка.