# Runbook — Climate Research Intelligence System

Setup, configuration, and operational guide for running locally or on AWS.

---

## Prerequisites

| Item | Notes |
|------|--------|
| Python | 3.12 |
| PostgreSQL | Running locally (pgAdmin4 or any Postgres install) with pgvector extension |
| Google Gemini API | API key → https://aistudio.google.com/app/apikey |

---

## 1. Clone and environment

```bash
git clone <repository-url>
cd climate-rag
python3.12 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate   # Windows
```

---

## 2. Dependencies

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 3. Configuration

```bash
cp .env.example .env
```

Edit `.env`. Required variables:

| Variable | Purpose |
|----------|---------|
| `DB_HOST` | Postgres host (e.g. `localhost` for local, RDS endpoint for AWS) |
| `DB_PORT` | Postgres port (default `5432`) |
| `DB_NAME` | Database name |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |
| `GEMINI_API_KEY` | LLM for RAG query answering |
| `BACKEND_URL` | FastAPI base URL used by Streamlit (no trailing slash); use your ALB URL in AWS |
| `HF_TOKEN` | Optional — HuggingFace token if you hit rate limits streaming [`ShayManor/Labeled-arXiv`](https://huggingface.co/datasets/ShayManor/Labeled-arXiv) during ingestion |

---

## 4. Database schema

**Default:** Every `python data/ingestion.py …` run calls `setup_schema()` first and executes `sql/01_create_schema.sql` from the repo (see `data/ingestion.py`). You can still apply the same SQL yourself first if you prefer.

**Manual:** Open pgAdmin4, connect to your database, and run `sql/01_create_schema.sql`.

Or run the file via Python (from repo root):

```bash
python3 -c "
from pathlib import Path
from scripts.db_connect import get_conn
sql_path = Path('sql/01_create_schema.sql')
conn = get_conn()
cur = conn.cursor()
with open(sql_path, encoding='utf-8') as f:
    cur.execute(f.read())
conn.close()
print('Schema ready.')
"
```

---

## 5. Data ingestion

> **Note:** Ingesting 3,000 papers can take a long time (HF streaming, embedding, KG). Use `--resume` if interrupted.

```bash
# Full run — 3,000 climate papers
python3 data/ingestion.py --n 3000

# Resume if interrupted
python3 data/ingestion.py --n 3000 --resume

# Quick test run — 20 papers
python3 data/ingestion.py --n 20
```

### Vector index

On a full successful pipeline, **Stage 6 (verify)** also runs `sql/02_create_index.sql` when tables look healthy (IVFFlat on chunk embeddings). You can run that file manually in pgAdmin anytime. Search works without it but is slower.

Ingestion is **not** run by `reproduce.sh` — run it manually once before using the app (after Postgres is up).

---

## 6. Backend (FastAPI)

```bash
uvicorn backend.app:app --reload --port 3001
```

| Endpoint | Use |
|----------|-----|
| `GET /health` | Liveness check |
| `GET /health/db` | Postgres connectivity + table row counts |
| `POST /query` | Run RAG query |
| `GET /history` | Query history |
| `GET /metrics` | Aggregated performance stats |
| `GET /metrics/history` | Per-query metrics (for analytics / external tools) |
| `GET /papers` | List all papers in corpus |

---

## 7. Frontend (Streamlit)

```bash
streamlit run frontend/app.py --server.port 3000
```

Open http://localhost:3000. Set `BACKEND_URL` in `.env` to the API the browser can reach (e.g. `http://localhost:3001` locally, or your ALB URL with port **3001** for the backend listener).

---

## 8. Smoke tests

Requires the backend running first. Tests call `BACKEND_URL` from `.env`
(default `http://localhost:3001`).

```bash
pytest tests/smoke_test.py -v
```

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Missing env var errors | Confirm `.env` exists and all required keys are set |
| `psycopg2` connection refused | Check DB_HOST/PORT match your running Postgres instance |
| `vector` type not found | Run `CREATE EXTENSION IF NOT EXISTS vector;` in your database |
| Frontend can't reach API | Confirm backend is running and `BACKEND_URL` matches host/port |
| Gemini `429` quota error | You're hitting rate limits — wait a moment and retry |
| Slow first query | Normal — embedding model loads into memory on first request |

---

*See root `README.md` for full project overview and architecture.*