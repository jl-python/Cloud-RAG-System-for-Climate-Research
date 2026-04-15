-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Schemas ──────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS graph;
CREATE SCHEMA IF NOT EXISTS app;

-- ── raw.papers ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.papers (
    paper_id         TEXT PRIMARY KEY,
    title            TEXT,
    authors          TEXT,
    abstract         TEXT,
    publication_year INT,
    source           TEXT DEFAULT 'arxiv',
    source_url       TEXT,
    categories       TEXT,
    ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── raw.chunks ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw.chunks (
    chunk_id         TEXT PRIMARY KEY,
    paper_id         TEXT REFERENCES raw.papers(paper_id),
    chunk_index      INT,
    section_name     TEXT,
    text_content     TEXT,
    word_count       INT,
    embedding        vector(768),
    ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── graph.knowledge_nodes ────────────────────────────────────
CREATE TABLE IF NOT EXISTS graph.knowledge_nodes (
    node_id          TEXT PRIMARY KEY,
    label            TEXT,
    name             TEXT,
    name_normalized  TEXT,
    paper_count      INT DEFAULT 0
);

-- ── graph.knowledge_edges ────────────────────────────────────
CREATE TABLE IF NOT EXISTS graph.knowledge_edges (
    edge_id          TEXT PRIMARY KEY,
    source_node_id   TEXT REFERENCES graph.knowledge_nodes(node_id),
    target_node_id   TEXT REFERENCES graph.knowledge_nodes(node_id),
    relation_type    TEXT,
    paper_id         TEXT REFERENCES raw.papers(paper_id),
    weight           FLOAT DEFAULT 1.0,
    ingested_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── graph.chunk_entity_map ───────────────────────────────────
CREATE TABLE IF NOT EXISTS graph.chunk_entity_map (
    map_id           TEXT PRIMARY KEY,
    chunk_id         TEXT REFERENCES raw.chunks(chunk_id),
    node_id          TEXT REFERENCES graph.knowledge_nodes(node_id),
    confidence       FLOAT DEFAULT 1.0
);

-- ── app.chunks_v ─────────────────────────────────────────────
CREATE OR REPLACE VIEW app.chunks_v AS
SELECT
    c.chunk_id,
    c.paper_id,
    c.chunk_index,
    c.section_name,
    c.text_content,
    c.word_count,
    c.embedding,
    p.title,
    p.authors,
    p.publication_year,
    p.categories,
    p.source_url
FROM raw.chunks c
JOIN raw.papers p ON c.paper_id = p.paper_id;

-- ── app.eval_metrics ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS app.eval_metrics (
    log_id           TEXT PRIMARY KEY,
    question         TEXT,
    generated_response TEXT,
    context_used     TEXT,
    retrieval_mode   TEXT,
    confidence       FLOAT,
    latency_ms       INT,
    tool_calls       JSONB,
    num_iterations   INT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT 'Schema setup complete' AS status;