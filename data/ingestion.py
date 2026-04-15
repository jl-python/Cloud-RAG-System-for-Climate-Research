"""
Ingestion pipeline for the Climate RAG system.

Stages:
    1. load_and_clean_dataset()   - Stream ShayManor/Labeled-arXiv (category + keyword)
    2. chunk_documents()          - Split papers into text chunks
    3. generate_embeddings()      - Embed chunks with all-mpnet-base-v2
    4. extract_knowledge_graph()  - Extract KG nodes/edges via scispaCy
    5. upload_to_postgres()       - Push all data to local Postgres / AWS RDS
    6. verify_ingestion()         - Sanity check row counts + build vector index

Usage:
    python data/ingestion.py              # full run
    python data/ingestion.py --n 50       # small test run
    python data/ingestion.py --resume     # skip completed stages
    python data/ingestion.py --stage load # run one stage only
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import psycopg2
import psycopg2.extras

load_dotenv()

random.seed(100)
np.random.seed(100)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.config import (
    CHECKPOINT_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_CHECKPOINT,
    CLIMATE_ARXIV_CATEGORY_MARKERS,
    CLIMATE_KEYWORDS_REQUIRED,
    EDGES_CHECKPOINT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KG_MIN_NAME_LENGTH,
    MAP_CHECKPOINT,
    MIN_CHUNK_WORDS,
    NODES_CHECKPOINT,
    NUM_PAPERS,
    PAPERS_CHECKPOINT,
    SPACY_MODEL,
    INGEST_SOURCE_TAG,
)
from scripts.db_connect import get_conn

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# SETUP — Run Schema SQL
# ════════════════════════════════════════════════════════════

def setup_schema():
    """Run 01_create_schema.sql to create all tables if they don't exist."""
    sql_path = Path(__file__).resolve().parent.parent / "sql" / "01_create_schema.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"Schema file not found: {sql_path}")

    print("[Setup] Running schema setup...")
    conn = get_conn()
    cur  = conn.cursor()
    with open(sql_path, encoding="utf-8") as f:
        cur.execute(f.read())
    conn.commit()
    conn.close()
    print("[Setup] Schema ready.")


# ════════════════════════════════════════════════════════════
# STAGE 1 — Load and Clean Dataset
# ════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[a-z]+\d*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_arxiv_id(raw: str) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().lower()
    if s.startswith("arxiv:"):
        s = s[6:].strip()
    s = re.sub(r"v\d+$", "", s, flags=re.I)
    return s or None


def _paper_id_from_arxiv_id(aid: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", aid)
    return f"arxiv_{safe}"


def _arxiv_categories_allow(categories: str) -> bool:
    if not categories or not isinstance(categories, str):
        return False
    c = categories.lower()
    return any(m.lower() in c for m in CLIMATE_ARXIV_CATEGORY_MARKERS)


def _year_from_labeled_item(item: dict) -> int | None:
    u = item.get("update_date")
    if u is None:
        return None
    if hasattr(u, "year"):
        return int(u.year)
    s = str(u)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def _is_climate_paper(abstract: str) -> bool:
    abstract_lower = abstract.lower()
    return any(
        re.search(r"\b" + re.escape(kw.lower()) + r"\b", abstract_lower)
        for kw in CLIMATE_KEYWORDS_REQUIRED
    )


def load_and_clean_dataset(n: int = NUM_PAPERS, resume: bool = False) -> pd.DataFrame:
    if resume and PAPERS_CHECKPOINT.exists():
        prev = pd.read_parquet(PAPERS_CHECKPOINT)
        if "ingest_source" in prev.columns and prev["ingest_source"].eq(INGEST_SOURCE_TAG).all():
            print(f"[Stage 1] Resuming from checkpoint: {PAPERS_CHECKPOINT}")
            print(f"[Stage 1] Loaded {len(prev)} papers from checkpoint.")
            return prev
        print(
            "[Stage 1] Checkpoint is from an older ingest; ignoring --resume for Stage 1 "
            f"(expected ingest_source={INGEST_SOURCE_TAG!r})."
        )

    print(
        f"[Stage 1] Loading up to {n} climate papers from HuggingFace "
        "(ShayManor/Labeled-arXiv, papers)..."
    )
    print(
        "[Stage 1] Filter: arXiv categories, then >=1 keyword in abstract (word boundaries) "
        f"({len(CLIMATE_ARXIV_CATEGORY_MARKERS)} category markers, "
        f"{len(CLIMATE_KEYWORDS_REQUIRED)} keywords)."
    )
    print(
        "[Stage 1] Note: full PDFs are not in this dataset; chunking uses title+abstract."
    )

    dataset = load_dataset(
        "ShayManor/Labeled-arXiv",
        "papers",
        split="train",
        streaming=True,
        token=os.getenv("HF_TOKEN") or None,
    )

    records = []
    skipped_deleted   = 0
    skipped_category  = 0
    skipped_domain    = 0
    skipped_quality   = 0
    scanned           = 0

    for item in tqdm(dataset, desc="Scanning arXiv metadata"):
        if len(records) >= n:
            break

        scanned += 1

        if item.get("deleted"):
            skipped_deleted += 1
            continue

        categories = item.get("categories") or ""
        if not _arxiv_categories_allow(categories):
            skipped_category += 1
            continue

        raw_id = item.get("id", "")
        aid = _normalize_arxiv_id(str(raw_id))
        if not aid:
            skipped_quality += 1
            continue

        title = _clean_text(item.get("title", "") or "")
        abstract = _clean_text(item.get("abstract", "") or "")
        authors = (item.get("authors") or "").strip() if isinstance(item.get("authors"), str) else ""

        body = _clean_text(f"{title}\n\n{abstract}" if title else abstract)

        if not abstract or len(abstract.split()) < 20:
            skipped_quality += 1
            continue
        if len(body.split()) < 40:
            skipped_quality += 1
            continue

        if not _is_climate_paper(abstract):
            skipped_domain += 1
            continue

        paper_id = _paper_id_from_arxiv_id(aid)
        year = _year_from_labeled_item(item)

        records.append({
            "paper_id":         paper_id,
            "title":            title or f"arXiv {aid}",
            "authors":          authors,
            "abstract":         abstract,
            "publication_year": year,
            "source":           "arxiv",
            "source_url":       f"https://arxiv.org/abs/{aid}",
            "categories":       categories.strip(),
            "section_names":    ["abstract", "body"],
            "sections":         [abstract, body],
            "ingest_source":    INGEST_SOURCE_TAG,
        })

    df = pd.DataFrame(records)
    print(f"\n[Stage 1] Scanned {scanned} metadata rows.")
    print(f"[Stage 1] Kept {len(df)} climate papers.")
    print(
        "[Stage 1] Skipped "
        f"{skipped_deleted} (deleted), {skipped_category} (wrong arXiv category), "
        f"{skipped_domain} (keyword filter), {skipped_quality} (too short / bad id)."
    )
    df.to_parquet(PAPERS_CHECKPOINT, index=False)
    print(f"[Stage 1] Checkpoint saved → {PAPERS_CHECKPOINT}")
    return df


# ════════════════════════════════════════════════════════════
# STAGE 2 — Chunk Documents
# ════════════════════════════════════════════════════════════

def chunk_documents(papers_df: pd.DataFrame, resume: bool = False) -> pd.DataFrame:
    if resume and CHUNKS_CHECKPOINT.exists():
        papers_newer = (
            PAPERS_CHECKPOINT.exists()
            and PAPERS_CHECKPOINT.stat().st_mtime > CHUNKS_CHECKPOINT.stat().st_mtime
        )
        if not papers_newer:
            print(f"[Stage 2] Resuming from checkpoint: {CHUNKS_CHECKPOINT}")
            df = pd.read_parquet(CHUNKS_CHECKPOINT)
            print(f"[Stage 2] Loaded {len(df)} chunks from checkpoint.")
            return df
        print(
            "[Stage 2] papers.parquet is newer than chunks checkpoint; "
            "re-chunking instead of --resume."
        )

    print(f"[Stage 2] Chunking {len(papers_df)} papers...")

    def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end   = start + size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += size - overlap
        return chunks

    records            = []
    chunk_index_global = 0

    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Chunking papers"):
        paper_id      = row["paper_id"]
        section_names = row["section_names"]
        sections      = row["sections"]

        for sec_name, sec_text in zip(section_names, sections):
            if not sec_text or len(sec_text.split()) < MIN_CHUNK_WORDS:
                continue

            slug = re.sub(r"[^a-z0-9]", "_", sec_name.lower())

            if sec_name == "abstract":
                chunks = [sec_text]
            else:
                chunks = split_into_chunks(sec_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

            for i, chunk_text in enumerate(chunks):
                word_count = len(chunk_text.split())
                if word_count < MIN_CHUNK_WORDS:
                    continue

                chunk_id = f"{paper_id}_{slug}_c{i:03d}"
                records.append({
                    "chunk_id":     chunk_id,
                    "paper_id":     paper_id,
                    "chunk_index":  chunk_index_global,
                    "section_name": sec_name,
                    "text_content": chunk_text,
                    "word_count":   word_count,
                })
                chunk_index_global += 1

    df = pd.DataFrame(records)
    print(f"[Stage 2] Created {len(df)} chunks from {len(papers_df)} papers.")
    df.to_parquet(CHUNKS_CHECKPOINT, index=False)
    print(f"[Stage 2] Checkpoint saved → {CHUNKS_CHECKPOINT}")
    return df


# ════════════════════════════════════════════════════════════
# STAGE 3 — Generate Embeddings
# ════════════════════════════════════════════════════════════

def generate_embeddings(chunks_df: pd.DataFrame, resume: bool = False) -> pd.DataFrame:
    if resume and CHUNKS_CHECKPOINT.exists():
        df = pd.read_parquet(CHUNKS_CHECKPOINT)
        if "embedding" in df.columns:
            print(f"[Stage 3] Resuming — {len(df)} chunks already embedded.")
            return df

    print(f"[Stage 3] Embedding {len(chunks_df)} chunks with {EMBEDDING_MODEL}...")

    model          = SentenceTransformer(EMBEDDING_MODEL)
    texts          = chunks_df["text_content"].tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding chunks"):
        batch      = texts[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())

    chunks_df            = chunks_df.copy()
    chunks_df["embedding"] = all_embeddings

    assert len(all_embeddings[0]) == EMBEDDING_DIM, \
        f"Expected {EMBEDDING_DIM}-dim, got {len(all_embeddings[0])}"

    chunks_df.to_parquet(CHUNKS_CHECKPOINT, index=False)
    print(f"[Stage 3] Done. Dim={EMBEDDING_DIM}")
    return chunks_df


# ════════════════════════════════════════════════════════════
# STAGE 4 — Extract Knowledge Graph
# ════════════════════════════════════════════════════════════

def extract_knowledge_graph(chunks_df: pd.DataFrame, resume: bool = False):
    if resume and NODES_CHECKPOINT.exists() and EDGES_CHECKPOINT.exists() and MAP_CHECKPOINT.exists():
        chunks_newer = (
            CHUNKS_CHECKPOINT.exists()
            and CHUNKS_CHECKPOINT.stat().st_mtime > NODES_CHECKPOINT.stat().st_mtime
        )
        if not chunks_newer:
            print(f"[Stage 4] Resuming from checkpoints.")
            return (
                pd.read_parquet(NODES_CHECKPOINT),
                pd.read_parquet(EDGES_CHECKPOINT),
                pd.read_parquet(MAP_CHECKPOINT),
            )
        print(
            "[Stage 4] chunks.parquet newer than KG checkpoints; "
            "re-extracting instead of --resume."
        )

    print(f"[Stage 4] Extracting knowledge graph from {len(chunks_df)} chunks...")
    nlp = spacy.load(SPACY_MODEL)

    node_registry = {}
    map_records   = []
    edge_pairs    = {}

    for _, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Extracting entities"):
        chunk_id = row["chunk_id"]
        paper_id = row["paper_id"]
        text     = row["text_content"]

        doc            = nlp(text)
        chunk_entities = []

        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < KG_MIN_NAME_LENGTH:
                continue

            normalized = re.sub(r"\s+", " ", name.lower()).strip()
            normalized = re.sub(r"[^a-z0-9 ]", "", normalized).strip()
            if not normalized:
                continue

            if normalized not in node_registry:
                node_id = "node_" + re.sub(r"\s+", "_", normalized)[:60]
                node_registry[normalized] = {
                    "node_id":         node_id,
                    "name":            name,
                    "name_normalized": normalized,
                    "label":           "Entity",
                    "paper_ids":       set(),
                }
            node_registry[normalized]["paper_ids"].add(paper_id)
            chunk_entities.append(normalized)

            map_records.append({
                "map_id":     str(uuid.uuid4()),
                "chunk_id":   chunk_id,
                "node_id":    node_registry[normalized]["node_id"],
                "confidence": 1.0,
            })

        seen = list(dict.fromkeys(chunk_entities))
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                a   = node_registry[seen[i]]["node_id"]
                b   = node_registry[seen[j]]["node_id"]
                key = (min(a, b), max(a, b), paper_id)
                edge_pairs[key] = edge_pairs.get(key, 0) + 1

    node_records = [
        {
            "node_id":         info["node_id"],
            "label":           info["label"],
            "name":            info["name"],
            "name_normalized": normalized,
            "paper_count":     len(info["paper_ids"]),
        }
        for normalized, info in node_registry.items()
    ]
    nodes_df = pd.DataFrame(node_records)

    edge_records = [
        {
            "edge_id":        str(uuid.uuid4()),
            "source_node_id": src,
            "target_node_id": tgt,
            "relation_type":  "CO_OCCURS",
            "paper_id":       paper_id,
            "weight":         float(weight),
        }
        for (src, tgt, paper_id), weight in edge_pairs.items()
    ]
    edges_df = pd.DataFrame(edge_records)
    map_df   = pd.DataFrame(map_records)

    before   = len(edges_df)
    edges_df = edges_df[edges_df["weight"] >= 2].reset_index(drop=True)
    print(f"[Stage 4] Edge filtering: {before:,} → {len(edges_df):,} (weight >= 2)")

    print(f"[Stage 4] {len(nodes_df)} nodes, {len(edges_df)} edges, {len(map_df)} mappings.")

    nodes_df.to_parquet(NODES_CHECKPOINT, index=False)
    edges_df.to_parquet(EDGES_CHECKPOINT, index=False)
    map_df.to_parquet(MAP_CHECKPOINT,     index=False)
    print(f"[Stage 4] Checkpoints saved.")
    return nodes_df, edges_df, map_df


# ════════════════════════════════════════════════════════════
# STAGE 5 — Upload to Postgres
# ════════════════════════════════════════════════════════════

def truncate_tables(conn):
    """Truncate all tables in correct FK order."""
    cur = conn.cursor()
    tables = [
        "graph.chunk_entity_map",
        "graph.knowledge_edges",
        "graph.knowledge_nodes",
        "raw.chunks",
        "raw.papers",
    ]
    print("[Stage 5] Truncating existing tables...")
    for table in tables:
        cur.execute(f"TRUNCATE TABLE {table} CASCADE")
        print(f"  -> Truncated {table}")
    conn.commit()
    print("[Stage 5] Tables cleared.")


def upload_to_postgres(papers_df, chunks_df, nodes_df, edges_df, map_df):
    print("[Stage 5] Connecting to Postgres...")
    conn = get_conn()
    cur  = conn.cursor()

    truncate_tables(conn)

    # ── 1. raw.papers ─────────────────────────────────────────
    print(f"[Stage 5] Uploading {len(papers_df)} rows → raw.papers...")
    papers_data = [
        (
            row["paper_id"],
            row["title"],
            row["authors"],
            row["abstract"],
            row.get("publication_year"),
            row["source"],
            row["source_url"],
            row["categories"],
        )
        for _, row in papers_df.iterrows()
    ]
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO raw.papers
            (paper_id, title, authors, abstract, publication_year,
             source, source_url, categories)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (paper_id) DO NOTHING
        """,
        papers_data,
        page_size=500,
    )
    conn.commit()
    print("[Stage 5] raw.papers done.")

    # ── 2. raw.chunks (pgvector embedding) ────────────────────
    print(f"[Stage 5] Uploading {len(chunks_df)} rows → raw.chunks...")
    BATCH_SIZE = 500
    rows = chunks_df[[
        "chunk_id", "paper_id", "chunk_index",
        "section_name", "text_content", "word_count", "embedding"
    ]].values.tolist()

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Uploading chunks"):
        batch      = rows[i : i + BATCH_SIZE]
        chunk_data = [
            (r[0], r[1], int(r[2]), r[3], r[4], int(r[5]), json.dumps(list(r[6])))
            for r in batch
        ]
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO raw.chunks
                (chunk_id, paper_id, chunk_index, section_name,
                 text_content, word_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            chunk_data,
            page_size=500,
        )
    conn.commit()
    print("[Stage 5] raw.chunks done.")

    # ── 3. graph.knowledge_nodes ──────────────────────────────
    print(f"[Stage 5] Uploading {len(nodes_df)} rows → graph.knowledge_nodes...")
    nodes_data = [
        (
            row["node_id"], row["label"], row["name"],
            row["name_normalized"], int(row["paper_count"])
        )
        for _, row in nodes_df.iterrows()
    ]
    psycopg2.extras.execute_batch(
        cur,
        """
        INSERT INTO graph.knowledge_nodes
            (node_id, label, name, name_normalized, paper_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (node_id) DO NOTHING
        """,
        nodes_data,
        page_size=500,
    )
    conn.commit()
    print("[Stage 5] graph.knowledge_nodes done.")

    # ── 4. graph.knowledge_edges ─────────
    print(f"[Stage 5] Uploading {len(edges_df)} rows → graph.knowledge_edges...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, dir="/tmp") as f:
        tmp_edges_path = f.name
        edges_df[["edge_id", "source_node_id", "target_node_id",
                  "relation_type", "paper_id", "weight"]].to_csv(
            f, sep="\t", header=False, index=False
        )
    print(f"[Stage 5] Edges buffer written to disk → {tmp_edges_path}")

    cur.execute("SET search_path TO graph, public")
    with open(tmp_edges_path, encoding="utf-8") as f:
        cur.copy_from(
            f,
            "knowledge_edges",
            columns=("edge_id", "source_node_id", "target_node_id",
                     "relation_type", "paper_id", "weight"),
        )
    os.unlink(tmp_edges_path)
    conn.commit()
    print("[Stage 5] graph.knowledge_edges done.")

    # ── 5. graph.chunk_entity_map ────────
    print(f"[Stage 5] Uploading {len(map_df)} rows → graph.chunk_entity_map...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False, dir="/tmp") as f:
        tmp_map_path = f.name
        map_df[["map_id", "chunk_id", "node_id", "confidence"]].to_csv(
            f, sep="\t", header=False, index=False
        )
    print(f"[Stage 5] Map buffer written to disk → {tmp_map_path}")

    with open(tmp_map_path, encoding="utf-8") as f:
        cur.copy_from(
            f,
            "chunk_entity_map",
            columns=("map_id", "chunk_id", "node_id", "confidence"),
        )
    os.unlink(tmp_map_path)
    conn.commit()
    print("[Stage 5] graph.chunk_entity_map done.")

    print("\n[Stage 5] All tables uploaded successfully.")


# ════════════════════════════════════════════════════════════
# STAGE 6 — Verify Ingestion + Build Vector Index
# ════════════════════════════════════════════════════════════

def verify_ingestion():
    print("[Stage 6] Verifying Postgres tables...")
    conn = get_conn()
    cur  = conn.cursor()

    tables = [
        ("raw",   "papers"),
        ("raw",   "chunks"),
        ("graph", "knowledge_nodes"),
        ("graph", "knowledge_edges"),
        ("graph", "chunk_entity_map"),
        ("app",   "eval_metrics"),
    ]

    print(f"\n{'Schema':<10} {'Table':<25} {'Row Count':>10}")
    print("-" * 48)
    all_good = True
    for schema, table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        count  = cur.fetchone()[0]
        status = "OK" if count > 0 or table == "eval_metrics" else "EMPTY (warn)"
        print(f"{schema:<10} {table:<25} {count:>10,}  {status}")
        if count == 0 and table != "eval_metrics":
            all_good = False

    print("-" * 48)

    if all_good:
        print("\n[Stage 6] All tables populated. Ingestion successful.")

        # ── Build pgvector IVFFlat index ──────────────────────
        print("\n[Stage 6] Building pgvector IVFFlat index on raw.chunks...")
        print("[Stage 6] This may take a few minutes on large tables...")
        index_path = Path(__file__).resolve().parent.parent / "sql" / "02_create_index.sql"
        with open(index_path, encoding="utf-8") as f:
            index_sql = f.read()
        cur.execute(index_sql)
        conn.commit()
        print("[Stage 6] Vector index created.")
    else:
        print("\n[Stage 6] Some tables are empty. Check Stage 5 logs.")

    conn.close()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Climate RAG Ingestion Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "load", "chunk", "embed", "kg", "upload", "verify"],
        default="all",
    )
    parser.add_argument("--n",      type=int,            default=NUM_PAPERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_schema()

    papers_df = chunks_df = nodes_df = edges_df = map_df = None

    if args.stage in ("all", "load"):
        papers_df = load_and_clean_dataset(n=args.n, resume=args.resume)

    if args.stage in ("all", "chunk"):
        if args.stage == "chunk":
            papers_df = pd.read_parquet(PAPERS_CHECKPOINT)
        chunks_df = chunk_documents(papers_df, resume=args.resume)

    if args.stage in ("all", "embed"):
        if args.stage == "embed":
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
        chunks_df = generate_embeddings(chunks_df, resume=args.resume)

    if args.stage in ("all", "kg"):
        if args.stage == "kg":
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
        nodes_df, edges_df, map_df = extract_knowledge_graph(chunks_df, resume=args.resume)

    if args.stage in ("all", "upload"):
        if args.stage == "upload":
            papers_df = pd.read_parquet(PAPERS_CHECKPOINT)
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
            nodes_df  = pd.read_parquet(NODES_CHECKPOINT)
            edges_df  = pd.read_parquet(EDGES_CHECKPOINT)
            map_df    = pd.read_parquet(MAP_CHECKPOINT)
        upload_to_postgres(papers_df, chunks_df, nodes_df, edges_df, map_df)

    if args.stage in ("all", "verify"):
        verify_ingestion()


if __name__ == "__main__":
    main()