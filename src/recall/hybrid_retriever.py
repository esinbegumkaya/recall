from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rapidfuzz import fuzz

from recall.database import (
    get_connection,
    search_chunks_fts,
)

from recall.query_planner import (
    SearchPlan,
    normalize_text,
)

from recall.semantic_reranker import (
    rerank_candidates_semantically,
)


# =========================================================
# User-search source hygiene
# =========================================================

INTERNAL_FILE_NAMES = {
    "recall_review.txt",
    "nli_bug_review.txt",
    "engine_inputs.txt",
    "project_tree.txt",
}

INTERNAL_PATH_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}


def is_user_searchable_file(file_path: str) -> bool:
    """Exclude Recall implementation/debug artifacts from normal user search."""
    path = Path(file_path)

    if path.name.casefold() in INTERNAL_FILE_NAMES:
        return False

    if {part.casefold() for part in path.parts} & INTERNAL_PATH_PARTS:
        return False

    normalized = str(path).replace("/", "\\").casefold()
    if "\\documents\\recall\\" in normalized:
        return False

    return True


# =========================================================
# Indexed files
# =========================================================

def get_indexed_files() -> List[Dict[str, Any]]:
    """
    Read all indexed file paths from SQLite.

    Filesystem timestamps are derived from the actual file.
    """

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT file_path
            FROM files
            """
        ).fetchall()

    finally:
        connection.close()

    files = []

    for row in rows:
        try:
            file_path = (
                row["file_path"]
                if hasattr(row, "keys")
                else row[0]
            )

        except Exception:
            continue

        if not file_path:
            continue

        if not is_user_searchable_file(file_path):
            continue

        path = Path(file_path)

        try:
            stat = path.stat()

            modified_at = float(
                stat.st_mtime
            )

            created_at = float(
                stat.st_ctime
            )

            exists = True

        except OSError:
            modified_at = 0.0
            created_at = 0.0
            exists = False

        files.append(
            {
                "file_path": str(path),
                "file_name": path.name,
                "extension": path.suffix.casefold(),
                "modified_at": modified_at,
                "created_at": created_at,
                "exists": exists,
            }
        )

    return files


# =========================================================
# Filename fuzzy matching
# =========================================================

def token_similarity(
    query_token: str,
    candidate_token: str,
) -> float:
    """
    Compare two individual tokens.

    Exact matches receive 1.0.
    Fuzzy matching is used for spelling mistakes.
    """

    query_token = normalize_text(
        query_token
    )

    candidate_token = normalize_text(
        candidate_token
    )

    if (
        not query_token
        or not candidate_token
    ):
        return 0.0

    if (
        query_token
        == candidate_token
    ):
        return 1.0

    ratio = fuzz.ratio(
        query_token,
        candidate_token,
    )

    partial = fuzz.partial_ratio(
        query_token,
        candidate_token,
    )

    return max(
        ratio,
        partial,
    ) / 100.0


def tokenize_filename(
    file_name: str,
) -> List[str]:
    """
    Convert a filename into normalized searchable tokens.
    """

    stem = Path(
        file_name
    ).stem

    normalized = normalize_text(
        stem
    )

    normalized = (
        normalized
        .replace("_", " ")
        .replace("-", " ")
        .replace(".", " ")
    )

    return [
        token
        for token in normalized.split()
        if token
    ]


FILENAME_GENERIC_TERMS = {
    "file",
    "document",
    "doc",
    "dosya",
    "dosyam",
    "belge",
    "belgem",
    "cv",
    "resume",
    "curriculum vitae",
}


def filename_relevance(
    plan: SearchPlan,
    file_name: str,
) -> float:
    """
    Score how strongly query terms match the filename.

    Distinctive filename clues (for example "Harvard") are
    deliberately more important than generic type words such
    as "CV", "resume", "file", or "document".
    """

    query_terms = (
        plan.filename_terms
        or plan.keywords
        or plan.concepts
    )

    if not query_terms:
        return 0.0

    filename_tokens = tokenize_filename(
        file_name
    )

    if not filename_tokens:
        return 0.0

    term_scores = []
    distinctive_scores = []

    for query_term in query_terms:
        normalized_term = normalize_text(
            query_term
        )

        best = max(
            (
                token_similarity(
                    normalized_term,
                    filename_token,
                )
                for filename_token
                in filename_tokens
            ),
            default=0.0,
        )

        term_scores.append(
            best
        )

        if (
            normalized_term
            and normalized_term
            not in FILENAME_GENERIC_TERMS
        ):
            distinctive_scores.append(
                best
            )

    if not term_scores:
        return 0.0

    strongest = max(
        term_scores
    )

    average = sum(
        term_scores
    ) / len(
        term_scores
    )

    if distinctive_scores:
        distinctive_average = sum(
            distinctive_scores
        ) / len(
            distinctive_scores
        )

        distinctive_strongest = max(
            distinctive_scores
        )

        return min(
            1.0,
            (
                0.55 * distinctive_average
                + 0.25 * distinctive_strongest
                + 0.20 * average
            ),
        )

    return min(
        1.0,
        (
            0.70 * strongest
            + 0.30 * average
        ),
    )


# =========================================================
# Path / location relevance
# =========================================================

def path_relevance(
    plan: SearchPlan,
    file_path: str,
) -> float:
    """
    Match location hints such as Downloads, Desktop,
    Documents, OneDrive, etc.
    """

    terms = (
        plan.location_terms
    )

    if not terms:
        return 0.0

    normalized_path = normalize_text(
        file_path
    )

    scores = []

    for term in terms:
        normalized_term = normalize_text(
            term
        )

        if (
            normalized_term
            in normalized_path
        ):
            scores.append(
                1.0
            )

        else:
            scores.append(
                fuzz.partial_ratio(
                    normalized_term,
                    normalized_path,
                )
                / 100.0
            )

    return max(
        scores,
        default=0.0,
    )


# =========================================================
# Extension relevance
# =========================================================

def extension_relevance(
    plan: SearchPlan,
    extension: str,
) -> float:
    """
    Score file-extension constraints when the user
    explicitly mentions a type such as PDF or DOCX.
    """

    if not plan.extensions:
        return 0.0

    normalized_extension = (
        extension.casefold()
    )

    allowed_extensions = {
        item.casefold()
        for item in plan.extensions
    }

    return (
        1.0
        if normalized_extension
        in allowed_extensions
        else 0.0
    )


# =========================================================
# Temporal metadata relevance
# =========================================================

def _iso_to_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError, OSError):
        return None


def date_window_relevance(
    plan: SearchPlan,
    item: Dict[str, Any],
) -> float:
    """
    Soft score for explicit planner date windows.

    Date phrases such as "last week" must not become hard
    filters because filesystem modified/created timestamps are
    not necessarily the same thing as when a document was
    received, discussed, or assigned.

    1.0 means the file timestamp is inside the requested
    window. Outside the window, the score decays smoothly.
    """

    modified_at = float(
        item.get(
            "modified_at",
            0.0,
        )
        or 0.0
    )

    created_at = float(
        item.get(
            "created_at",
            0.0,
        )
        or 0.0
    )

    modified_after = _iso_to_timestamp(
        plan.modified_after
    )
    modified_before = _iso_to_timestamp(
        plan.modified_before
    )
    created_after = _iso_to_timestamp(
        plan.created_after
    )
    created_before = _iso_to_timestamp(
        plan.created_before
    )

    def score_timestamp(
        timestamp: float,
        after: float | None,
        before: float | None,
    ) -> float:
        if (
            timestamp <= 0
            or (
                after is None
                and before is None
            )
        ):
            return 0.0

        if (
            (after is None or timestamp >= after)
            and
            (before is None or timestamp <= before)
        ):
            return 1.0

        distances = []

        if (
            after is not None
            and timestamp < after
        ):
            distances.append(
                after - timestamp
            )

        if (
            before is not None
            and timestamp > before
        ):
            distances.append(
                timestamp - before
            )

        if not distances:
            return 0.0

        days_away = min(
            distances
        ) / 86400.0

        # Smooth decay: ~0.50 at 14 days away,
        # ~0.25 at 42 days away.
        return 1.0 / (
            1.0
            + days_away / 14.0
        )

    modified_score = score_timestamp(
        modified_at,
        modified_after,
        modified_before,
    )

    created_score = score_timestamp(
        created_at,
        created_after,
        created_before,
    )

    return max(
        modified_score,
        created_score,
    )


def has_explicit_date_window(
    plan: SearchPlan,
) -> bool:
    return any(
        (
            plan.modified_after,
            plan.modified_before,
            plan.created_after,
            plan.created_before,
        )
    )


# =========================================================
# Recency
# =========================================================

def calculate_recency_scores(
    files: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Convert modification times into percentile-like
    recency scores.

    Newest file approaches 1.0.
    Oldest file approaches 0.0.
    """

    existing = [
        item
        for item in files
        if item[
            "modified_at"
        ] > 0
    ]

    existing.sort(
        key=lambda item: item[
            "modified_at"
        ],
        reverse=True,
    )

    if not existing:
        return {}

    total = len(
        existing
    )

    scores = {}

    for rank, item in enumerate(
        existing
    ):
        if total == 1:
            score = 1.0

        else:
            percentile = (
                rank
                / (
                    total - 1
                )
            )

            score = (
                1.0
                - percentile
            )

        scores[
            item["file_path"]
        ] = score

    return scores


# =========================================================
# Multi-query FTS candidate generation
# =========================================================

GENERIC_SEARCH_TERMS = {
    # English
    "where",
    "what",
    "which",
    "who",
    "when",
    "how",
    "about",
    "most",
    "recent",
    "latest",
    "newest",
    "find",
    "show",
    "get",
    "file",
    "document",
    "my",

    # Turkish - normalized
    "nerede",
    "hangi",
    "ne",
    "nasil",
    "hakkinda",
    "ilgili",
    "olani",
    "olan",
    "yazdigim",
    "bul",
    "getir",
    "goster",
    "dosya",
    "dosyam",
    "belge",
    "belgem",
    "en",
    "son",
    "sonki",
    "guncel",
    "yeni",

    # Temporal/context language — useful for metadata/planning,
    # but usually poor topical FTS terms.
    "gecen",
    "gecen hafta",
    "hafta",
    "bugun",
    "dun",
    "yarin",

    # Relational/source wording — describes how the user
    # remembers the file, not what the file is about.
    "hocadan",
    "hoca",
    "gelen",
    "gonderilen",
    "aldigim",

    # Generic artifact/task words. These remain available to
    # filename fuzzy matching and planner metadata, but should
    # not dominate content FTS when a stronger topic term exists.
    "odev",
    "ödev",
    "assignment",
    "task",
    "not",
    "notlar",
    "notes",
}


def get_core_search_terms(
    plan: SearchPlan,
) -> List[str]:
    """
    Extract topical lexical search terms.

    Natural-language queries often contain three different
    signal classes:
    - topic/entity terms: "network", "RAG", "Harvard"
    - artifact words: "assignment", "notes", "document"
    - context/temporal words: "from my teacher", "last week"

    FTS should primarily search the first class. Artifact and
    temporal/context signals are still used elsewhere by
    filename fuzzy matching, metadata, and date scoring.
    """

    source_terms = (
        plan.content_terms
        or plan.keywords
        or plan.concepts
    )

    core_terms = []

    for term in source_terms:
        normalized = normalize_text(
            term
        ).strip()

        if not normalized:
            continue

        if normalized in GENERIC_SEARCH_TERMS:
            continue

        # Multi-word planner output may contain context phrases.
        # Remove generic words inside the phrase and retain only
        # informative components.
        parts = [
            part
            for part in normalized.split()
            if (
                part
                and part not in GENERIC_SEARCH_TERMS
                and len(part) > 1
            )
        ]

        if not parts:
            continue

        cleaned = " ".join(
            parts
        )

        if cleaned:
            core_terms.append(
                cleaned
            )

    return list(
        dict.fromkeys(
            core_terms
        )
    )


def build_fts_queries(
    plan: SearchPlan,
) -> List[str]:
    """
    Build multiple FTS searches instead of sending the
    whole natural-language sentence into one MATCH query.
    """

    core_terms = get_core_search_terms(
        plan
    )

    if not core_terms:
        return []

    queries = []

    # Combined query.
    if len(core_terms) >= 2:
        queries.append(
            " ".join(
                core_terms
            )
        )

    # Individual terms.
    queries.extend(
        core_terms
    )

    # Adjacent pairs.
    if len(core_terms) >= 2:
        for index in range(
            len(core_terms) - 1
        ):
            queries.append(
                " ".join(
                    core_terms[
                        index:
                        index + 2
                    ]
                )
            )

    return list(
        dict.fromkeys(
            query
            for query in queries
            if query.strip()
        )
    )


def collect_fts_scores(
    plan: SearchPlan,
    limit_per_query: int = 80,
) -> Dict[str, Dict[str, Any]]:
    """
    Union results from multiple FTS searches.

    Files matching several query variants receive an
    agreement bonus.
    """

    queries = build_fts_queries(
        plan
    )

    if not queries:
        return {}

    per_file: Dict[
        str,
        Dict[str, Any],
    ] = {}

    total_queries = len(
        queries
    )

    for query_index, query in enumerate(
        queries
    ):
        try:
            rows = search_chunks_fts(
                query,
                limit=limit_per_query,
            )

        except Exception:
            continue

        if not rows:
            continue

        query_specificity = (
            1.0
            - (
                0.35
                * query_index
                / max(
                    total_queries - 1,
                    1,
                )
            )
        )

        for rank, row in enumerate(
            rows
        ):
            try:
                file_path = row[
                    "file_path"
                ]

            except Exception:
                continue

            if not file_path:
                continue

            try:
                raw_bm25 = float(
                    row[
                        "lexical_score"
                    ]
                )

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

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                bm25_score = 0.0

            rank_score = (
                1.0
                / math.sqrt(
                    rank + 1
                )
            )

            result_score = (
                query_specificity
                * (
                    0.70
                    * bm25_score
                    + 0.30
                    * rank_score
                )
            )

            current = per_file.get(
                file_path
            )

            if current is None:
                current = {
                    "lexical_score": 0.0,
                    "lexical_hits": 0,
                    "matched_queries": [],
                    "best_chunk_text": "",
                    "section_name": None,
                    "page_number": None,
                }

                per_file[
                    file_path
                ] = current

            current[
                "lexical_hits"
            ] += 1

            current[
                "matched_queries"
            ].append(
                query
            )

            if (
                result_score
                > current[
                    "lexical_score"
                ]
            ):
                current[
                    "lexical_score"
                ] = result_score

                try:
                    current[
                        "best_chunk_text"
                    ] = (
                        row[
                            "chunk_text"
                        ]
                        or ""
                    )
                except Exception:
                    pass

                try:
                    current[
                        "section_name"
                    ] = row[
                        "section_name"
                    ]
                except Exception:
                    pass

                try:
                    current[
                        "page_number"
                    ] = row[
                        "page_number"
                    ]
                except Exception:
                    pass

    # Multi-query agreement bonus.
    for data in per_file.values():
        unique_hits = len(
            set(
                data[
                    "matched_queries"
                ]
            )
        )

        agreement = min(
            1.0,
            unique_hits
            / max(
                min(
                    total_queries,
                    3,
                ),
                1,
            )
        )

        data[
            "lexical_score"
        ] = min(
            1.0,
            (
                0.85
                * data[
                    "lexical_score"
                ]
                + 0.15
                * agreement
            ),
        )

    return per_file


# =========================================================
# Query → file direct fuzzy relevance
# =========================================================

def direct_query_relevance(
    plan: SearchPlan,
    file_name: str,
    file_path: str,
) -> float:
    """
    Directly compare the full normalized query against
    filename and path.
    """

    query = (
        plan.normalized_query
    )

    if not query:
        return 0.0

    filename = normalize_text(
        Path(
            file_name
        ).stem
    )

    path = normalize_text(
        file_path
    )

    filename_score = (
        fuzz.token_set_ratio(
            query,
            filename,
        )
        / 100.0
    )

    path_score = (
        fuzz.token_set_ratio(
            query,
            path,
        )
        / 100.0
    )

    return max(
        filename_score,
        0.65 * path_score,
    )


# =========================================================
# Hybrid scoring
# =========================================================

def hybrid_search(
    plan: SearchPlan,
    limit: int | None = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    First-stage candidate retrieval using:

    - filename fuzzy similarity
    - FTS/BM25 content matching
    - whole-query filename/path similarity
    - location metadata
    - extension metadata
    - recency

    Semantic reranking happens later in search_query().
    """

    requested_limit = (
        limit
        if limit is not None
        else plan.limit
    )

    indexed_files = get_indexed_files()

    if not indexed_files:
        return []

    recency_scores = (
        calculate_recency_scores(
            indexed_files
        )
    )

    fts_results = (
        collect_fts_scores(
            plan
        )
    )

    results = []
    total_indexed_files = len(indexed_files)

    if progress_callback is not None:
        progress_callback(
            "scanning",
            0,
            total_indexed_files,
            "Scanning indexed files",
        )

    for position, item in enumerate(
        indexed_files,
        start=1,
    ):
        file_path = item[
            "file_path"
        ]

        file_name = item[
            "file_name"
        ]

        filename_score = (
            filename_relevance(
                plan,
                file_name,
            )
        )

        direct_score = (
            direct_query_relevance(
                plan,
                file_name,
                file_path,
            )
        )

        lexical_data = (
            fts_results.get(
                file_path,
                {},
            )
        )

        lexical_score = float(
            lexical_data.get(
                "lexical_score",
                0.0,
            )
        )

        location_score = (
            path_relevance(
                plan,
                file_path,
            )
        )

        extension_score = (
            extension_relevance(
                plan,
                item[
                    "extension"
                ],
            )
        )

        date_window_score = (
            date_window_relevance(
                plan,
                item,
            )
        )

        recency_score = (
            recency_scores.get(
                file_path,
                0.0,
            )
            if (
                plan.date_preference
                == "most_recent"
            )
            else 0.0
        )

        # Semantic is not calculated in the first stage.
        semantic_score = 0.0

        filename_weight = max(
            plan.filename_weight,
            0.0,
        )

        lexical_weight = max(
            plan.lexical_weight,
            0.0,
        )

        recency_weight = max(
            plan.recency_weight,
            0.0,
        )

        semantic_weight = max(
            plan.semantic_weight,
            0.0,
        )

        direct_weight = 0.35

        location_weight = (
            0.60
            if plan.location_terms
            else 0.0
        )

        extension_weight = (
            0.60
            if plan.extensions
            else 0.0
        )

        date_window_weight = (
            max(
                plan.metadata_weight,
                0.0,
            )
            if has_explicit_date_window(
                plan
            )
            else 0.0
        )

        relevance_signal = max(
            filename_score,
            lexical_score,
            direct_score,
            semantic_score,
        )

        gated_recency = (
            recency_score
            * (
                0.25
                + 0.75
                * relevance_signal
            )
        )

        weighted_values = [
            (
                filename_score,
                filename_weight,
            ),
            (
                lexical_score,
                lexical_weight,
            ),
            (
                semantic_score,
                semantic_weight,
            ),
            (
                direct_score,
                direct_weight,
            ),
            (
                location_score,
                location_weight,
            ),
            (
                extension_score,
                extension_weight,
            ),
            (
                date_window_score,
                date_window_weight,
            ),
            (
                gated_recency,
                recency_weight,
            ),
        ]

        numerator = sum(
            score * weight
            for (
                score,
                weight,
            )
            in weighted_values
        )

        denominator = sum(
            weight
            for (
                _,
                weight,
            )
            in weighted_values
            if weight > 0
        )

        hybrid_score = (
            numerator / denominator
            if denominator
            else 0.0
        )

        if (
            relevance_signal
            < 0.15
            and location_score
            <= 0
            and extension_score
            <= 0
        ):
            continue

        results.append(
            {
                **item,

                "score": (
                    hybrid_score
                ),

                "filename_score": (
                    filename_score
                ),

                "lexical_score": (
                    lexical_score
                ),

                "lexical_hits": (
                    lexical_data.get(
                        "lexical_hits",
                        0,
                    )
                ),

                "matched_queries": (
                    lexical_data.get(
                        "matched_queries",
                        [],
                    )
                ),

                "semantic_score": (
                    semantic_score
                ),

                "direct_score": (
                    direct_score
                ),

                "location_score": (
                    location_score
                ),

                "extension_score": (
                    extension_score
                ),

                "date_window_score": (
                    date_window_score
                ),

                "recency_score": (
                    recency_score
                ),

                "best_chunk_text": (
                    lexical_data.get(
                        "best_chunk_text",
                        "",
                    )
                ),

                "section_name": (
                    lexical_data.get(
                        "section_name"
                    )
                ),

                "page_number": (
                    lexical_data.get(
                        "page_number"
                    )
                ),
            }
        )


        if (
            progress_callback is not None
            and (
                position == 1
                or position % 25 == 0
                or position == total_indexed_files
            )
        ):
            progress_callback(
                "scanning",
                position,
                total_indexed_files,
                item.get(
                    "file_name",
                    "",
                ),
            )

    results.sort(
        key=lambda item: (
            -item[
                "score"
            ],
            -item[
                "modified_at"
            ],
        )
    )

    return results[
        :requested_limit
    ]


# =========================================================
# Full search pipeline
# =========================================================

def search_query(
    query: str,
    limit: int = 10,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Natural-language search pipeline:

    1. Query planning
    2. Lexical/fuzzy/metadata candidate retrieval
    3. Semantic reranking on a limited candidate set
    4. Final score recomputation
    """

    from recall.query_planner import (
        plan_query_with_llm,
    )

    if progress_callback is not None:
        progress_callback(
            "planning",
            0,
            1,
            "Understanding your request",
        )

    plan = plan_query_with_llm(
        query
    )

    if progress_callback is not None:
        progress_callback(
            "planning",
            1,
            1,
            "Search plan ready",
        )

    # Retrieve more candidates than the final requested
    # result count so semantic reranking can reorder them.
    candidate_limit = max(
        20,
        limit * 4,
    )

    candidates = hybrid_search(
        plan,
        limit=candidate_limit,
        progress_callback=progress_callback,
    )

    if not candidates:
        return []

    # =====================================================
    # Semantic reranking
    # =====================================================

    if progress_callback is not None:
        progress_callback(
            "semantic",
            0,
            max(len(candidates), 1),
            "Semantic reranking",
        )

    candidates = (
        rerank_candidates_semantically(
            query=query,
            candidates=candidates,
            candidate_limit=candidate_limit,
            max_chunks_per_file=6,
        )
    )

    if progress_callback is not None:
        progress_callback(
            "semantic",
            len(candidates),
            max(len(candidates), 1),
            "Semantic reranking complete",
        )

    # =====================================================
    # Final score — driven by SearchPlan
    # =====================================================

    for candidate in candidates:
        semantic_score = float(candidate.get("semantic_score", 0.0))
        lexical_score = float(candidate.get("lexical_score", 0.0))
        filename_score = float(candidate.get("filename_score", 0.0))
        direct_score = float(candidate.get("direct_score", 0.0))
        location_score = float(candidate.get("location_score", 0.0))
        extension_score = float(candidate.get("extension_score", 0.0))
        date_window_score = float(candidate.get("date_window_score", 0.0))
        recency_score = float(candidate.get("recency_score", 0.0))

        semantic_weight = max(float(plan.semantic_weight), 0.0)
        lexical_weight = max(float(plan.lexical_weight), 0.0)
        filename_weight = max(float(plan.filename_weight), 0.0)
        metadata_weight = max(float(plan.metadata_weight), 0.0)
        recency_weight = max(float(plan.recency_weight), 0.0)

        filename_intent_boost = (
            1.55
            if filename_weight >= 0.90
            and filename_weight >= semantic_weight + 0.25
            and filename_weight >= lexical_weight + 0.25
            else 1.0
        )
        effective_filename_weight = filename_weight * filename_intent_boost

        # Full-query fuzzy matching supports typo-heavy filename requests.
        direct_weight = 0.40 * effective_filename_weight

        location_weight = (
            metadata_weight
            if plan.location_terms
            else 0.0
        )

        extension_weight = (
            metadata_weight
            if plan.extensions
            else 0.0
        )

        date_window_weight = (
            0.70 * metadata_weight
            if has_explicit_date_window(plan)
            else 0.0
        )

        # A recent file should win only if it is also relevant.
        relevance_signal = max(
            semantic_score,
            lexical_score,
            filename_score,
            direct_score,
        )

        gated_recency = (
            recency_score
            * (
                0.15
                + 0.85 * relevance_signal
            )
        )

        weighted_values = [
            (semantic_score, semantic_weight),
            (lexical_score, lexical_weight),
            (filename_score, effective_filename_weight),
            (direct_score, direct_weight),
            (location_score, location_weight),
            (extension_score, extension_weight),
            (date_window_score, date_window_weight),
            (gated_recency, recency_weight),
        ]

        numerator = sum(
            score * weight
            for score, weight in weighted_values
            if weight > 0
        )

        denominator = sum(
            weight
            for _, weight in weighted_values
            if weight > 0
        )

        final_score = (
            numerator / denominator
            if denominator
            else 0.0
        )

        candidate["score"] = final_score
        candidate["gated_recency_score"] = gated_recency
        candidate["planner"] = plan.planner

    candidates.sort(
        key=lambda item: (
            -item.get(
                "score",
                0.0,
            ),
            -item.get(
                "modified_at",
                0.0,
            ),
        )
    )

    return candidates[
        :limit
    ]