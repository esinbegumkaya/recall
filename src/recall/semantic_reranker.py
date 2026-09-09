from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from foundry_local_sdk.exception import (
    FoundryLocalException,
)

from recall.database import (
    get_connection,
    search_chunks_fts,
)

from recall.query_planner import (
    normalize_text,
)


MODEL_NAME = "qwen3-embedding-0.6b"


# =========================================================
# Vector helpers
# =========================================================

def cosine_similarity(
    a: List[float],
    b: List[float],
) -> float:
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(
            x * x
            for x in a
        )
    )

    norm_b = math.sqrt(
        sum(
            y * y
            for y in b
        )
    )

    if (
        norm_a == 0.0
        or norm_b == 0.0
    ):
        return 0.0

    return dot / (
        norm_a * norm_b
    )


def normalize_similarity(
    score: float,
) -> float:
    """
    Convert cosine similarity into a stable 0..1 range.
    """

    return max(
        0.0,
        min(
            1.0,
            (score + 1.0) / 2.0,
        ),
    )


# =========================================================
# Foundry Local embedding model
# =========================================================

class LocalEmbeddingModel:
    """
    Small wrapper around the local Foundry embedding model.

    FoundryLocalManager is process-wide and behaves as a
    singleton, so repeated searches must reuse the existing
    manager instead of initializing a second one.
    """

    def __init__(self):
        self.model = None
        self.client = None

    def load(self):
        if self.client is not None:
            return

        try:
            FoundryLocalManager.initialize(
                Configuration(
                    app_name="Recall"
                )
            )

        except FoundryLocalException as exc:
            message = str(exc)

            # A previous search in the same Python process
            # may already have initialized the singleton.
            if (
                "already been initialized"
                not in message
            ):
                raise

        manager = (
            FoundryLocalManager.instance
        )

        self.model = (
            manager.catalog.get_model(
                MODEL_NAME
            )
        )

        self.model.load()

        self.client = (
            self.model
            .get_embedding_client()
        )

    def unload(self):
        """
        Unload the embedding model itself.

        The FoundryLocalManager singleton intentionally
        remains initialized for the lifetime of the process.
        """

        if self.model is not None:
            try:
                self.model.unload()
            except Exception:
                pass

        self.model = None
        self.client = None

    def embed(
        self,
        text: str,
    ) -> List[float]:
        self.load()

        response = (
            self.client
            .generate_embedding(
                text
            )
        )

        return list(
            response
            .data[0]
            .embedding
        )


# =========================================================
# Existing embedding cache
# =========================================================

def get_cached_embedding(
    file_path: str,
    chunk_index: int,
) -> Optional[List[float]]:
    """
    Return an already-generated chunk embedding when one is
    present in Recall's SQLite cache.
    """

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT embedding
            FROM embeddings
            WHERE file_path = ?
              AND chunk_index = ?
              AND model_name = ?
            LIMIT 1
            """,
            (
                file_path,
                chunk_index,
                MODEL_NAME,
            ),
        ).fetchone()

    if row is None:
        return None

    raw = (
        row["embedding"]
        if hasattr(row, "keys")
        else row[0]
    )

    if not raw:
        return None

    try:
        parsed = json.loads(
            raw
        )

        return [
            float(value)
            for value in parsed
        ]

    except Exception:
        return None


def cache_embedding(
    file_path: str,
    chunk_index: int,
    embedding: List[float],
):
    """
    Save a lazily generated embedding.

    The embedding table is used as a persistent cache so
    future queries do not need to regenerate the same
    candidate chunk embedding.
    """

    serialized = json.dumps(
        embedding
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO embeddings (
                file_path,
                chunk_index,
                model_name,
                embedding
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(
                file_path,
                chunk_index,
                model_name
            )
            DO UPDATE SET
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                file_path,
                chunk_index,
                MODEL_NAME,
                serialized,
            ),
        )

        connection.commit()


# =========================================================
# Chunk loading helpers
# =========================================================

def get_chunk(
    file_path: str,
    chunk_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Fetch one exact indexed chunk.
    """

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                chunk_index,
                chunk_text,
                page_number,
                section_name
            FROM chunks
            WHERE file_path = ?
              AND chunk_index = ?
            LIMIT 1
            """,
            (
                file_path,
                chunk_index,
            ),
        ).fetchone()

    if row is None:
        return None

    return {
        "chunk_index": (
            row["chunk_index"]
            if hasattr(row, "keys")
            else row[0]
        ),
        "chunk_text": (
            row["chunk_text"]
            if hasattr(row, "keys")
            else row[1]
        ),
        "page_number": (
            row["page_number"]
            if hasattr(row, "keys")
            else row[2]
        ),
        "section_name": (
            row["section_name"]
            if hasattr(row, "keys")
            else row[3]
        ),
    }


def get_fallback_chunks(
    file_path: str,
    max_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Fallback used only when lexical query-aware chunk
    discovery cannot find a usable chunk in the file.
    """

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                chunk_index,
                chunk_text,
                page_number,
                section_name
            FROM chunks
            WHERE file_path = ?
            ORDER BY chunk_index ASC
            LIMIT ?
            """,
            (
                file_path,
                max_chunks,
            ),
        ).fetchall()

    results = []

    for row in rows:
        results.append(
            {
                "chunk_index": (
                    row["chunk_index"]
                    if hasattr(row, "keys")
                    else row[0]
                ),
                "chunk_text": (
                    row["chunk_text"]
                    if hasattr(row, "keys")
                    else row[1]
                ),
                "page_number": (
                    row["page_number"]
                    if hasattr(row, "keys")
                    else row[2]
                ),
                "section_name": (
                    row["section_name"]
                    if hasattr(row, "keys")
                    else row[3]
                ),
            }
        )

    return results


# =========================================================
# Query-aware lexical chunk discovery
# =========================================================

SEMANTIC_QUERY_STOPWORDS = {
    # English
    "where",
    "what",
    "which",
    "who",
    "when",
    "how",
    "is",
    "are",
    "was",
    "were",
    "my",
    "me",
    "mine",
    "the",
    "a",
    "an",
    "about",
    "find",
    "show",
    "get",
    "most",
    "recent",
    "latest",
    "newest",
    "file",
    "document",

    # Turkish normalized
    "en",
    "son",
    "sonki",
    "guncel",
    "yeni",
    "nerede",
    "hangi",
    "ne",
    "nasil",
    "bana",
    "benim",
    "bul",
    "getir",
    "goster",
    "hakkinda",
    "ilgili",
    "olan",
    "olani",
    "yazdigim",
    "dosya",
    "dosyam",
    "belge",
    "belgem",
}


def extract_query_terms(
    query: str,
) -> List[str]:
    """
    Extract useful lexical terms from the natural-language
    query for candidate chunk discovery.
    """

    normalized = normalize_text(
        query
    )

    terms = []

    for token in normalized.split():
        if len(token) <= 1:
            continue

        if (
            token
            in SEMANTIC_QUERY_STOPWORDS
        ):
            continue

        terms.append(
            token
        )

    return list(
        dict.fromkeys(
            terms
        )
    )


def build_query_variants(
    query: str,
) -> List[str]:
    """
    Build multiple lexical variants.

    Example:
        rag hakkında yazdığım belge

    approximately becomes:
        rag

    Example:
        presentation about ai

    approximately becomes:
        presentation ai
        presentation
        ai
    """

    terms = extract_query_terms(
        query
    )

    if not terms:
        return []

    variants = []

    # Full combination.
    if len(terms) >= 2:
        variants.append(
            " ".join(
                terms
            )
        )

    # Individual concept terms.
    variants.extend(
        terms
    )

    # Adjacent concept pairs.
    if len(terms) >= 2:
        for index in range(
            len(terms) - 1
        ):
            variants.append(
                " ".join(
                    terms[
                        index:
                        index + 2
                    ]
                )
            )

    return list(
        dict.fromkeys(
            variant
            for variant in variants
            if variant.strip()
        )
    )


def get_query_aware_chunks(
    query: str,
    file_path: str,
    max_chunks: int = 6,
) -> List[Dict[str, Any]]:
    """
    Find chunks inside one candidate file that are most
    lexically related to the user's query.

    This prevents the semantic reranker from blindly
    embedding only the first few chunks of every document.
    """

    variants = build_query_variants(
        query
    )

    candidates: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for variant_index, variant in enumerate(
        variants
    ):
        try:
            rows = search_chunks_fts(
                variant,
                limit=120,
            )

        except Exception:
            continue

        for rank, row in enumerate(
            rows
        ):
            try:
                row_file_path = row[
                    "file_path"
                ]

            except Exception:
                continue

            if row_file_path != file_path:
                continue

            try:
                chunk_index = int(
                    row[
                        "chunk_index"
                    ]
                )

            except Exception:
                continue

            try:
                raw_bm25 = float(
                    row[
                        "lexical_score"
                    ]
                )

            except Exception:
                raw_bm25 = 0.0

            bm25_strength = abs(
                min(
                    raw_bm25,
                    0.0,
                )
            )

            bm25_score = (
                bm25_strength
                / (
                    bm25_strength
                    + 8.0
                )
            )

            rank_score = (
                1.0
                / math.sqrt(
                    rank + 1
                )
            )

            variant_weight = (
                1.0
                - (
                    0.30
                    * variant_index
                    / max(
                        len(variants) - 1,
                        1,
                    )
                )
            )

            lexical_candidate_score = (
                variant_weight
                * (
                    0.75
                    * bm25_score
                    + 0.25
                    * rank_score
                )
            )

            existing = candidates.get(
                chunk_index
            )

            if (
                existing is None
                or lexical_candidate_score
                > existing[
                    "candidate_score"
                ]
            ):
                candidates[
                    chunk_index
                ] = {
                    "chunk_index": (
                        chunk_index
                    ),
                    "chunk_text": (
                        row[
                            "chunk_text"
                        ]
                    ),
                    "page_number": (
                        row[
                            "page_number"
                        ]
                    ),
                    "section_name": (
                        row[
                            "section_name"
                        ]
                    ),
                    "candidate_score": (
                        lexical_candidate_score
                    ),
                    "matched_variant": (
                        variant
                    ),
                }

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item[
                "candidate_score"
            ],
            item[
                "chunk_index"
            ],
        ),
    )

    if ranked:
        return ranked[
            :max_chunks
        ]

    return get_fallback_chunks(
        file_path,
        max_chunks=max_chunks,
    )


# =========================================================
# File semantic score
# =========================================================

def semantic_score_file(
    query: str,
    query_embedding: List[float],
    file_path: str,
    embedding_model: LocalEmbeddingModel,
    max_chunks: int = 6,
) -> Dict[str, Any]:
    """
    Calculate semantic relevance for a candidate file.

    The score is based on the best query-aware chunk in the
    document, rather than the first chunk in the file.
    """

    chunks = get_query_aware_chunks(
        query=query,
        file_path=file_path,
        max_chunks=max_chunks,
    )

    best_score = 0.0
    best_chunk = None

    chunk_scores = []

    for chunk in chunks:
        chunk_index = int(
            chunk[
                "chunk_index"
            ]
        )

        text = (
            chunk[
                "chunk_text"
            ]
            or ""
        ).strip()

        if not text:
            continue

        embedding = (
            get_cached_embedding(
                file_path,
                chunk_index,
            )
        )

        if embedding is None:
            embedding = (
                embedding_model.embed(
                    text
                )
            )

            try:
                cache_embedding(
                    file_path,
                    chunk_index,
                    embedding,
                )

            except Exception:
                # Search should remain usable even if a
                # cache write fails.
                pass

        raw_score = cosine_similarity(
            query_embedding,
            embedding,
        )

        score = normalize_similarity(
            raw_score
        )

        chunk_scores.append(
            {
                "chunk_index": (
                    chunk_index
                ),
                "score": (
                    score
                ),
                "page_number": (
                    chunk.get(
                        "page_number"
                    )
                ),
                "section_name": (
                    chunk.get(
                        "section_name"
                    )
                ),
                "chunk_text": (
                    text
                ),
                "matched_variant": (
                    chunk.get(
                        "matched_variant"
                    )
                ),
            }
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk

    chunk_scores.sort(
        key=lambda item: (
            -item[
                "score"
            ],
            item[
                "chunk_index"
            ],
        )
    )

    return {
        "semantic_score": (
            best_score
        ),
        "semantic_chunk": (
            best_chunk
        ),
        "semantic_chunk_scores": (
            chunk_scores
        ),
    }


# =========================================================
# Candidate semantic reranking
# =========================================================

def rerank_candidates_semantically(
    query: str,
    candidates: List[Dict[str, Any]],
    candidate_limit: int = 20,
    max_chunks_per_file: int = 6,
) -> List[Dict[str, Any]]:
    """
    Semantically rerank only first-stage candidates.

    This keeps Recall performant by avoiding global
    embedding over every indexed chunk.
    """

    if not query.strip():
        return candidates

    if not candidates:
        return candidates

    selected = candidates[
        :candidate_limit
    ]

    embedding_model = (
        LocalEmbeddingModel()
    )

    try:
        # Generate the query embedding once.
        query_embedding = (
            embedding_model.embed(
                query
            )
        )

        for candidate in selected:
            result = (
                semantic_score_file(
                    query=query,
                    query_embedding=(
                        query_embedding
                    ),
                    file_path=(
                        candidate[
                            "file_path"
                        ]
                    ),
                    embedding_model=(
                        embedding_model
                    ),
                    max_chunks=(
                        max_chunks_per_file
                    ),
                )
            )

            candidate[
                "semantic_score"
            ] = result[
                "semantic_score"
            ]

            candidate[
                "semantic_chunk"
            ] = result[
                "semantic_chunk"
            ]

            candidate[
                "semantic_chunk_scores"
            ] = result[
                "semantic_chunk_scores"
            ]

    finally:
        embedding_model.unload()

    return candidates