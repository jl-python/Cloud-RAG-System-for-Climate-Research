-- Run this ONCE after ingestion is complete.
-- Requires raw.chunks to be populated (at least 1 row).

CREATE INDEX ON raw.chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);