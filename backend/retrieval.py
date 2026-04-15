from typing import List, Dict
from sentence_transformers import SentenceTransformer
import spacy
import data.config as config

# ── Load models once at startup ───────────────────────────────
print("Loading embedding model...")
EMBEDDING_MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
print("Embedding model loaded.")

nlp_ner = spacy.load(config.SPACY_MODEL)


def extract_query_entities(query: str) -> List[str]:
    """Extract and normalize scientific entities from a query."""
    if not query:
        return []

    doc = nlp_ner(query)
    entities = []
    for ent in doc.ents:
        cleaned = ent.text.strip()
        if len(cleaned) >= config.KG_MIN_NAME_LENGTH:
            entities.append(cleaned.upper())

    return list(set(entities))


def get_top_chunks(conn, query_text: str, top_k: int = 10):
    """
    Encode the query and run pgvector cosine similarity search
    against app.chunks_v.
    """
    query_vec = EMBEDDING_MODEL.encode(
        [query_text], normalize_embeddings=True
    )[0]

    # pgvector expects a string like '[0.1, 0.2, ...]'
    vec_str = "[" + ",".join(str(float(v)) for v in query_vec) + "]"

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            1 - (embedding <=> %s::vector)  AS score,
            chunk_id,
            paper_id,
            title,
            section_name,
            text_content
        FROM app.chunks_v
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec_str, vec_str, int(top_k))
    )
    rows = cur.fetchall()
    cur.close()

    return [
        (float(score), chunk_id, paper_id, title, section, text)
        for score, chunk_id, paper_id, title, section, text in rows
    ]


def graph_search(conn, query: str) -> List[Dict]:
    """
    Look up extracted entities in the knowledge graph
    and return their CO_OCCURS relationships.
    """
    entities = extract_query_entities(query)
    if not entities:
        return []

    placeholders = ", ".join(["%s"] * len(entities))
    normalized   = [e.strip().upper() for e in entities]

    cur = conn.cursor()
    cur.execute(
        f"""
        WITH target_nodes AS (
            SELECT node_id, name
            FROM graph.knowledge_nodes
            WHERE UPPER(name_normalized) IN ({placeholders})
        )
        SELECT
            tn.name        AS source_entity,
            e.relation_type,
            n2.name        AS target_entity,
            e.weight
        FROM target_nodes tn
        JOIN graph.knowledge_edges e  ON tn.node_id = e.source_node_id
        JOIN graph.knowledge_nodes n2 ON e.target_node_id = n2.node_id

        UNION ALL

        SELECT
            n2.name        AS source_entity,
            e.relation_type,
            tn.name        AS target_entity,
            e.weight
        FROM target_nodes tn
        JOIN graph.knowledge_edges e  ON tn.node_id = e.target_node_id
        JOIN graph.knowledge_nodes n2 ON e.source_node_id = n2.node_id
        """,
        normalized
    )
    rows = cur.fetchall()
    cur.close()

    return [
        {
            "source":   src,
            "relation": rel,
            "target":   tgt,
            "weight":   float(w) if w else 1.0,
        }
        for src, rel, tgt, w in rows
    ]