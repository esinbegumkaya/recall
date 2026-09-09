from collections import defaultdict
import re

from recall.database import (
    search_chunks_fts,
)


DEFAULT_CANDIDATE_LIMIT = 100
DEFAULT_RESULT_LIMIT = 10

MIN_EVIDENCE_CHARACTERS = 40
MIN_EVIDENCE_WORDS = 5

MAX_CHUNKS_PER_FILE = 3


STOPWORDS = {
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
                "retrieval_method": (
                    "FTS5"
                ),
            }
        )

    return candidates


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

    lexical_rank = max(
        1,
        int(
            candidate.get(
                "lexical_rank",
                1,
            )
        ),
    )

    lexical_rank_bonus = (
        1.0
        / lexical_rank
    )

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

    score = (
        (0.60 * text_coverage)
        + (0.16 * filename_coverage)
        + (0.05 * section_coverage)
        + (0.12 * lexical_rank_bonus)
        + (0.07 * length_bonus)
        - (0.20 * noise_penalty)
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
            -candidate[
                "retrieval_score"
            ],
            candidate[
                "lexical_score"
            ],
        ),
    )


# =========================================================
# Deduplication
# =========================================================

def remove_exact_duplicates(
    candidates: list,
):
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
    Recall v1 retrieval pipeline.

    FTS5 broad retrieval
        -> quality filtering
        -> duplicate removal
        -> content-aware reranking
        -> file diversity
        -> final candidates
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

    candidates = lexical_candidates(
        query=query,
        candidate_limit=(
            candidate_limit
        ),
    )

    candidates = (
        filter_candidate_quality(
            candidates
        )
    )

    candidates = (
        remove_exact_duplicates(
            candidates
        )
    )

    candidates = (
        rerank_candidates(
            query,
            candidates,
        )
    )

    candidates = (
        diversify_by_file(
            candidates
        )
    )

    return candidates[
        :result_limit
    ]