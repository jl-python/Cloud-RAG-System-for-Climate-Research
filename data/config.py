from pathlib import Path

ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
CHECKPOINT_DIR  = DATA_DIR / "checkpoints"

PAPERS_CHECKPOINT = CHECKPOINT_DIR / "papers.parquet"
CHUNKS_CHECKPOINT = CHECKPOINT_DIR / "chunks.parquet"
NODES_CHECKPOINT  = CHECKPOINT_DIR / "nodes.parquet"
EDGES_CHECKPOINT  = CHECKPOINT_DIR / "edges.parquet"
MAP_CHECKPOINT    = CHECKPOINT_DIR / "chunk_entity_map.parquet"

NUM_PAPERS = 3000

INGEST_SOURCE_TAG = "labeled_arxiv_v2"

CLIMATE_ARXIV_CATEGORY_MARKERS = [
    "physics.ao-ph",
]

CLIMATE_KEYWORDS_REQUIRED = [
    "climate change", "global warming", "climate model",
    "climate variability", "climate feedback", "climate projection",
    "climate simulation", "climate sensitivity", "climate system",
    "climate tipping", "climate forcing",
    "greenhouse gas", "greenhouse effect",
    "carbon dioxide emission", "co2 concentration", "atmospheric co2",
    "co2 emission", "carbon emission", "carbon sink", "carbon cycle",
    "fossil fuel emission", "radiative forcing",
    "sea level rise", "sea level change",
    "glacier melt", "permafrost",
    "arctic warming", "antarctic ice", "polar ice",
    "interglacial",
    "ocean warming", "ocean acidification", "thermohaline circulation",
    "atlantic meridional overturning", "el nino", "enso",
    "global temperature", "temperature anomaly",
    "ozone depletion", "ozone hole",
    "tropical cyclone", "hurricane intensity",
    "ipcc",
    "climate change impact", "paleoclimate",
    "dansgaard", "monsoon rainfall",
]

CHUNK_SIZE_WORDS    = 200
CHUNK_OVERLAP_WORDS = 30
MIN_CHUNK_WORDS     = 30

EMBEDDING_MODEL      = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM        = 768
EMBEDDING_BATCH_SIZE = 64

SPACY_MODEL        = "en_core_sci_sm"
KG_MIN_NAME_LENGTH = 3
