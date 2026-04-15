from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.retrieval import get_top_chunks, graph_search
from evaluation.evaluate import log_metrics_to_postgres
import os, time, json, uuid, csv
from google import genai
from google.genai import types
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from scripts.db_connect import get_conn
from backend.logger import logger, query_id_var, latency_var

load_dotenv()

MODEL_ID = "gemini-2.5-flash-lite"
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Climate RAG API")

_GLOBAL_CONN = None

def get_active_conn():
    global _GLOBAL_CONN
    if _GLOBAL_CONN is not None and _GLOBAL_CONN.closed == 0:
        return _GLOBAL_CONN
    _GLOBAL_CONN = get_conn()
    return _GLOBAL_CONN


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    chat_id: Optional[str] = None
    chat_history: List[Dict[str, Any]] = []


def save_to_csv_log(question: str, result: dict):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "query_log.csv"
    fieldnames = ["timestamp", "question", "answer_preview", "confidence", "latency_ms"]
    file_exists = log_file.exists()
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":      datetime.now().isoformat(),
            "question":       question,
            "answer_preview": result["answer"][:100].replace("\n", " ") + "...",
            "confidence":     result["confidence"],
            "latency_ms":     result["latency_ms"],
        })


def save_to_history(query_text: str, answer: str, citations: list,
                    confidence: float = 0.0, latency_ms: int = 0,
                    tool_calls: list = None, num_iterations: int = 0,
                    chat_id: str = None):
    history_path = Path("backend/history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                history_data = []
    else:
        history_data = []

    msg_packet = {
        "timestamp":      datetime.now().isoformat(),
        "query":          query_text,
        "answer":         answer,
        "chunks":         citations,
        "confidence":     confidence,
        "latency_ms":     latency_ms,
        "tool_calls":     tool_calls or [],
        "num_iterations": num_iterations,
    }

    found = False
    if chat_id:
        for chat in history_data:
            if chat.get("chat_id") == chat_id:
                chat["messages"].append(msg_packet)
                chat["updated_at"] = datetime.now().isoformat()
                found = True
                break

    if not found:
        if not chat_id:
            chat_id = uuid.uuid4().hex
        history_data.append({
            "chat_id":    chat_id,
            "title":      query_text[:50] + "..." if len(query_text) > 50 else query_text,
            "updated_at": datetime.now().isoformat(),
            "messages":   [msg_packet],
        })

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def _query_logic(req: QueryRequest):
    start          = time.time()
    conn           = get_active_conn()
    log_id         = uuid.uuid4().hex
    chat_id        = req.chat_id or uuid.uuid4().hex

    query_id_var.set(log_id)
    latency_var.set("N/A")

    logger.info(f"Running vector search for: '{req.question}'")
    chunks = get_top_chunks(conn, req.question, top_k=req.top_k)

    citations = []
    context_parts = []
    max_confidence = 0.0

    for i, chunk in enumerate(chunks, start=1):
        score, chunk_id, paper_id, title, section, text = chunk
        if score > max_confidence:
            max_confidence = score
        context_parts.append(f"[{i}] Paper: {title}\nSection: {section}\nText: {text}")
        citations.append({
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "title":    title,
            "section":  section,
            "text":     text[:200],
            "score":    score,
        })

    logger.info("Running knowledge graph search...")
    graph_data = graph_search(conn, req.question)
    kg_context = ""
    if graph_data:
        kg_lines   = [
            f"{r['source']} -[{r['relation']}]-> {r['target']} (weight: {r['weight']})"
            for r in graph_data[:20]
        ]
        kg_context = "\nKnowledge Graph Relationships:\n" + "\n".join(kg_lines)

    context = "\n\n".join(context_parts) + kg_context

    if req.chat_history:
        history_text = "Prior conversation:\n"
        for msg in req.chat_history:
            role = "Assistant" if msg.get("role") in ["assistant", "model"] else "User"
            history_text += f"{role}: {msg.get('content')}\n\n"
        user_prompt = f"{history_text}\nContext from research papers:\n{context}\n\nQuestion: {req.question}"
    else:
        user_prompt = f"Context from research papers:\n{context}\n\nQuestion: {req.question}"

    system_prompt = (
        "You are a climate science research assistant. "
        "Answer the user's question using ONLY the provided context from academic papers. "
        "Cite sources using [1], [2], etc. matching the paper numbers in the context. "
        "If the answer is not in the provided context, say so clearly. "
        "Never invent facts or citations."
    )

    logger.info("Calling Gemini...")
    response = None
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=1200,
                ),
            )
            break
        except Exception as e:
            if attempt < 2:
                logger.warning(
                    "Gemini generate_content failed (%s/3): %s; retrying in 5s...",
                    attempt + 1,
                    e,
                )
                time.sleep(5)
                continue
            raise
    answer = response.text
    logger.info("Gemini responded.")

    final_latency = int((time.time() - start) * 1000)
    latency_var.set(f"{final_latency}ms")

    result = {
        "chat_id":        chat_id,
        "answer":         answer,
        "citations":      citations,
        "confidence":     round(max_confidence, 3),
        "retrieval_mode": "vector+kg",
        "latency_ms":     final_latency,
        "tool_calls":     ["get_top_chunks", "graph_search"],
        "num_iterations": 1,
    }

    save_to_csv_log(req.question, result)
    save_to_history(
        query_text=req.question,
        answer=answer,
        citations=citations,
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        tool_calls=result["tool_calls"],
        num_iterations=result["num_iterations"],
        chat_id=chat_id,
    )

    log_data = {
        "log_id":         log_id,
        "question":       req.question,
        "answer":         answer,
        "context_used":   context[:5000],
        "retrieval_mode": result["retrieval_mode"],
        "confidence":     result["confidence"],
        "latency_ms":     result["latency_ms"],
        "tool_calls":     result["tool_calls"],
        "num_iterations": result["num_iterations"],
    }
    try:
        log_metrics_to_postgres(log_data, conn=conn)
        logger.info("Metrics logged to Postgres.")
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")

    return result


@app.get("/")
def read_root():
    return {"message": "Climate RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
def health_db():
    try:
        conn = get_active_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT table_schema, table_name,
                   (xpath('/row/cnt/text()',
                    query_to_xml(
                        format('SELECT COUNT(*) AS cnt FROM %I.%I', table_schema, table_name),
                        false, true, ''))
                   )[1]::text::int AS row_count
            FROM information_schema.tables
            WHERE table_schema IN ('raw', 'graph', 'app')
              AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """)
        rows = cur.fetchall()
        cur.close()
        tables = [
            {"schema": r[0], "name": r[1], "row_count": r[2]}
            for r in rows
        ]
        return {"status": "ok", "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB health check failed: {e}")


@app.post("/query")
def query(req: QueryRequest):
    try:
        return _query_logic(req)
    except Exception as e:
        logger.error("Error in /query", exc_info=True)
        raise e


@app.get("/papers")
def papers():
    conn = get_active_conn()
    cur  = conn.cursor()
    cur.execute("SELECT paper_id, title, abstract, categories FROM raw.papers")
    rows = cur.fetchall()
    cur.close()
    return [
        {"paper_id": r[0], "title": r[1], "abstract": r[2][:200], "categories": r[3]}
        for r in rows
    ]


@app.get("/history")
def history():
    history_path = Path("backend/history.json")
    if not history_path.exists():
        return []
    with open(history_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


@app.get("/metrics")
def get_metrics():
    conn = get_active_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(log_id)       AS total_queries,
                AVG(latency_ms)     AS avg_latency,
                AVG(confidence)     AS avg_confidence,
                AVG(num_iterations) AS avg_iterations
            FROM app.eval_metrics
        """)
        row = cur.fetchone()
        cur.execute("""
            SELECT retrieval_mode, COUNT(*) AS count
            FROM app.eval_metrics
            GROUP BY retrieval_mode
        """)
        modes = {r[0]: r[1] for r in cur.fetchall()}
        cur.close()
        return {
            "total_queries":  row[0] or 0,
            "avg_latency_ms": round(float(row[1] or 0), 2),
            "avg_confidence": round(float(row[2] or 0), 3),
            "avg_iterations": round(float(row[3] or 0), 2),
            "retrieval_modes": modes,
        }
    except Exception as e:
        logger.error(f"Metrics fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")


@app.get("/metrics/history")
def get_metrics_history(limit: int = 100):
    conn = get_active_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT log_id, question, confidence, latency_ms,
                   retrieval_mode, num_iterations, tool_calls, created_at
            FROM app.eval_metrics
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        columns = [desc[0] for desc in cur.description]
        rows    = [dict(zip(columns, row)) for row in cur.fetchall()]
        cur.close()
        return rows
    except Exception as e:
        logger.error(f"Metrics history fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics history")