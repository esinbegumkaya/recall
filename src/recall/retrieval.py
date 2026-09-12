from collections import defaultdict
import json
import math
import re

from recall.database import (
    get_chunks_with_embeddings,
    search_chunks_fts,
)

from recall.semantic_reranker import (
    LocalEmbeddingModel,
)


DEFAULT_CANDIDATE_LIMIT = 100
DEFAULT_RESULT_LIMIT = 10

MIN_EVIDENCE_CHARACTERS = 40
MIN_EVIDENCE_WORDS = 5

MAX_CHUNKS_PER_FILE = 3

EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"


STOPWORDS = {
    # English
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",

    # Turkish
    "ve",
    "veya",
    "ile",
    "bu",
    "şu",
    "bir",
    "ne",
    "nerede",
    "hangi",
    "kim",
    "için",
    "de",
    "da",
}


CODE_NOISE_MARKERS = (
    "def ",
    "assert ",
    "from recall.",
    "import recall.",
    "pytest",
    "fakechatclient",
    "fake_response",
    "test_",
    "judge_claim(",
    "filter_by_topic(",
    "evidence_units",
)


# =========================================================
# Text helpers
# =========================================================

def normalize_text(
    text: str,
) -> str:
    return " ".join(
        (text or "").split()
    )


def tokenize(
    text: str,
) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜü0-9]+",
            text or "",
            flags=re.UNICODE,
        )
    ]


def query_terms(
    query: str,
) -> list[str]:
    terms = []

    for token in tokenize(
        query
    ):
        if token in STOPWORDS:
            continue

        if len(token) < 2:
            continue

        if token not in terms:
            terms.append(
                token
            )

    return terms


def meaningful_word_count(
    text: str,
) -> int:
    return len(
        tokenize(
            text
        )
    )


def canonicalize_for_duplicate_check(
    text: str,
) -> str:
    text = normalize_text(
        text
    ).casefold()

    text = re.sub(
        r"\W+",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    )


# =========================================================
# Candidate quality
# =========================================================

def is_low_quality_chunk(
    text: str,
) -> bool:
    normalized = normalize_text(
        text
    )

    if not normalized:
        return True

    if (
        len(normalized)
        < MIN_EVIDENCE_CHARACTERS
    ):
        return True

    if (
        meaningful_word_count(
            normalized
        )
        < MIN_EVIDENCE_WORDS
    ):
        return True

    return False


def filter_candidate_quality(
    candidates: list,
):
    return [
        candidate
        for candidate in candidates
        if not is_low_quality_chunk(
            candidate.get(
                "text",
                "",
            )
        )
    ]


# =========================================================
# FTS retrieval
# =========================================================

def lexical_candidates(
    query: str,
    candidate_limit: int = (
        DEFAULT_CANDIDATE_LIMIT
    ),
):
    rows = search_chunks_fts(
        query=query,
        limit=candidate_limit,
    )

    candidates = []

    for lexical_rank, row in enumerate(
        rows,
        start=1,
    ):
        candidates.append(
            {
                "file_path": (
                    row["file_path"]
                ),
                "file_name": (
                    row["file_name"]
                ),
                "chunk_index": (
                    row["chunk_index"]
                ),
                "text": (
                    row["chunk_text"]
                ),
                "page_number": (
                    row["page_number"]
                ),
                "section_name": (
                    row["section_name"]
                ),
                "lexical_score": (
                    row["lexical_score"]
                ),
                "lexical_rank": (
                    lexical_rank
                ),
                "snippet": (
                    row["snippet"]
                ),
                "semantic_score": 0.0,
                "retrieval_method": (
                    "FTS5"
                ),
            }
        )

    return candidates


# =========================================================
# Dense semantic retrieval
# =========================================================

def cosine_similarity(
    a: list[float],
    b: list[float],
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

    return (
        dot
        / (
            norm_a
            * norm_b
        )
    )


def normalize_similarity(
    score: float,
) -> float:
    """
    Convert cosine similarity from -1..1
    into a stable 0..1 range.
    """

    return max(
        0.0,
        min(
            1.0,
            (
                score + 1.0
            ) / 2.0,
        ),
    )


def dense_candidates(
    query: str,
    candidate_limit: int = (
        DEFAULT_CANDIDATE_LIMIT
    ),
):
    """
    Global dense chunk retrieval.

    Only embeddings already stored in SQLite are searched.
    Missing embeddings are deliberately NOT generated here.
    """

    if not query.strip():
        return []

    embedding_model = (
        LocalEmbeddingModel()
    )

    try:
        query_embedding = (
            embedding_model.embed(
                query
            )
        )

        rows = (
            get_chunks_with_embeddings(
                EMBEDDING_MODEL_NAME
            )
        )

        candidates = []

        for row in rows:
            try:
                raw_embedding = (
                    row["embedding"]
                )

                embedding = json.loads(
                    raw_embedding
                )

                embedding = [
                    float(value)
                    for value
                    in embedding
                ]

            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
                KeyError,
            ):
                continue

            raw_similarity = (
                cosine_similarity(
                    query_embedding,
                    embedding,
                )
            )

            semantic_score = (
                normalize_similarity(
                    raw_similarity
                )
            )

            file_path = (
                row["file_path"]
            )

            file_name = (
                file_path
                .replace("\\", "/")
                .rsplit("/", 1)[-1]
            )

            candidates.append(
                {
                    "file_path": (
                        file_path
                    ),
                    "file_name": (
                        file_name
                    ),
                    "chunk_index": (
                        row["chunk_index"]
                    ),
                    "text": (
                        row["chunk_text"]
                    ),
                    "page_number": (
                        row["page_number"]
                    ),
                    "section_name": (
                        row["section_name"]
                    ),
                    "lexical_score": 0.0,
                    "lexical_rank": None,
                    "snippet": "",
                    "semantic_score": (
                        semantic_score
                    ),
                    "retrieval_score": (
                        semantic_score
                    ),
                    "retrieval_method": (
                        "DENSE"
                    ),
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate[
                    "semantic_score"
                ]
            )
        )

        return candidates[
            :candidate_limit
        ]

    finally:
        embedding_model.unload()


# =========================================================
# Candidate fusion
# =========================================================

def fuse_candidates(
    lexical: list,
    dense: list,
):
    """
    Merge lexical and dense candidates using:

        (file_path, chunk_index)

    as the chunk identity.

    Chunks appearing in both channels become HYBRID.
    """

    merged = {}

    for candidate in lexical:
        key = (
            candidate["file_path"],
            candidate["chunk_index"],
        )

        item = dict(
            candidate
        )

        item.setdefault(
            "semantic_score",
            0.0,
        )

        item["retrieval_method"] = (
            "FTS5"
        )

        merged[key] = item

    for candidate in dense:
        key = (
            candidate["file_path"],
            candidate["chunk_index"],
        )

        existing = merged.get(
            key
        )

        if existing is None:
            merged[key] = dict(
                candidate
            )
            continue

        existing[
            "semantic_score"
        ] = candidate.get(
            "semantic_score",
            0.0,
        )

        existing[
            "retrieval_method"
        ] = "HYBRID"

    return list(
        merged.values()
    )


# =========================================================
# Reranking
# =========================================================

def term_coverage(
    terms: list[str],
    text: str,
) -> float:
    if not terms:
        return 0.0

    tokens = set(
        tokenize(
            text
        )
    )

    matched = sum(
        1
        for term in terms
        if term in tokens
    )

    return (
        matched
        / len(terms)
    )


def code_noise_score(
    text: str,
) -> float:
    lowered = (
        text or ""
    ).casefold()

    matches = sum(
        1
        for marker
        in CODE_NOISE_MARKERS
        if marker.casefold()
        in lowered
    )

    if matches == 0:
        return 0.0

    return min(
        1.0,
        matches / 4.0,
    )


def calculate_retrieval_score(
    query: str,
    candidate: dict,
) -> float:
    terms = query_terms(
        query
    )

    text_coverage = (
        term_coverage(
            terms,
            candidate.get(
                "text",
                "",
            ),
        )
    )

    filename_coverage = (
        term_coverage(
            terms,
            candidate.get(
                "file_name",
                "",
            ),
        )
    )

    section_coverage = (
        term_coverage(
            terms,
            candidate.get(
                "section_name",
                "",
            )
            or "",
        )
    )

    raw_lexical_rank = (
        candidate.get(
            "lexical_rank"
        )
    )

    if raw_lexical_rank is None:
        lexical_rank_bonus = 0.0

    else:
        try:
            lexical_rank = max(
                1,
                int(
                    raw_lexical_rank
                ),
            )

            lexical_rank_bonus = (
                1.0
                / lexical_rank
            )

        except (
            TypeError,
            ValueError,
        ):
            lexical_rank_bonus = 0.0

    noise_penalty = (
        code_noise_score(
            candidate.get(
                "text",
                "",
            )
        )
    )

    text = normalize_text(
        candidate.get(
            "text",
            "",
        )
    )

    length_bonus = min(
        1.0,
        len(text) / 500.0,
    )

    semantic_score = float(
        candidate.get(
            "semantic_score",
            0.0,
        )
        or 0.0
    )

    #
    # Hybrid retrieval score
    #
    # Semantic similarity is intentionally the strongest
    # single component, while lexical evidence still retains
    # substantial influence.
    #
    score = (
        (0.42 * text_coverage)
        + (
            0.10
            * filename_coverage
        )
        + (
            0.04
            * section_coverage
        )
        + (
            0.08
            * lexical_rank_bonus
        )
        + (
            0.06
            * length_bonus
        )
        + (
            0.50
            * semantic_score
        )
        - (
            0.20
            * noise_penalty
        )
    )

    candidate[
        "query_term_coverage"
    ] = text_coverage

    candidate[
        "filename_term_coverage"
    ] = filename_coverage

    candidate[
        "retrieval_score"
    ] = score

    return score


def rerank_candidates(
    query: str,
    candidates: list,
):
    for candidate in candidates:
        calculate_retrieval_score(
            query,
            candidate,
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.get(
                "retrieval_score",
                0.0,
            ),
            -candidate.get(
                "semantic_score",
                0.0,
            ),
            candidate.get(
                "lexical_rank"
            )
            if candidate.get(
                "lexical_rank"
            )
            is not None
            else float("inf"),
        ),
    )


# =========================================================
# Deduplication
# =========================================================

def remove_exact_duplicates(
    candidates: list,
):
    """
    Remove duplicate chunk text while preserving the
    highest-ranked occurrence.

    Important:
    This should run AFTER hybrid scoring so that the best
    copy survives.
    """

    seen = set()
    result = []

    for candidate in candidates:
        key = (
            canonicalize_for_duplicate_check(
                candidate.get(
                    "text",
                    "",
                )
            )
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            candidate
        )

    return result


# =========================================================
# File diversity
# =========================================================

def diversify_by_file(
    candidates: list,
    max_chunks_per_file: int = (
        MAX_CHUNKS_PER_FILE
    ),
):
    """
    Prevent one repetitive file from monopolizing the final
    evidence candidate set.
    """

    file_counts = defaultdict(
        int
    )

    diversified = []

    for candidate in candidates:
        file_path = candidate.get(
            "file_path",
            "",
        )

        if (
            file_counts[
                file_path
            ]
            >= max_chunks_per_file
        ):
            continue

        diversified.append(
            candidate
        )

        file_counts[
            file_path
        ] += 1

    return diversified


# =========================================================
# Final retrieval entry point
# =========================================================

def retrieve_candidates(
    query: str,
    candidate_limit: int = (
        DEFAULT_CANDIDATE_LIMIT
    ),
    result_limit: int = (
        DEFAULT_RESULT_LIMIT
    ),
):
    """
    Recall hybrid evidence retrieval pipeline.

    Pipeline:

        FTS5 lexical retrieval
            +
        global dense retrieval over cached embeddings
            ↓
        candidate fusion
            ↓
        minimum evidence quality
            ↓
        hybrid scoring
            ↓
        duplicate removal
            ↓
        per-file diversity
            ↓
        final chunk candidates

    Dense retrieval searches only embeddings that already
    exist in SQLite. It does not generate missing document
    embeddings during a user query.
    """

    if not query:
        return []

    query = query.strip()

    if not query:
        return []

    candidate_limit = max(
        candidate_limit,
        result_limit,
    )

    # -----------------------------------------------------
    # Lexical channel
    # -----------------------------------------------------

    lexical = lexical_candidates(
        query=query,
        candidate_limit=(
            candidate_limit
        ),
    )

    # -----------------------------------------------------
    # Dense semantic channel
    # -----------------------------------------------------

    dense = dense_candidates(
        query=query,
        candidate_limit=(
            candidate_limit
        ),
    )

    # -----------------------------------------------------
    # Fusion
    # -----------------------------------------------------

    candidates = fuse_candidates(
        lexical=lexical,
        dense=dense,
    )

    # -----------------------------------------------------
    # Quality filtering
    # -----------------------------------------------------

    candidates = (
        filter_candidate_quality(
            candidates
        )
    )

    # -----------------------------------------------------
    # Hybrid scoring BEFORE deduplication
    #
    # This ensures that when duplicate CV copies exist,
    # the strongest semantic/lexical version survives.
    # -----------------------------------------------------

    candidates = (
        rerank_candidates(
            query,
            candidates,
        )
    )

    # -----------------------------------------------------
    # Exact duplicate removal
    # -----------------------------------------------------

    candidates = (
        remove_exact_duplicates(
            candidates
        )
    )

    # -----------------------------------------------------
    # File diversity
    # -----------------------------------------------------

    candidates = (
        diversify_by_file(
            candidates
        )
    )

    return candidates[
        :result_limit
    ]