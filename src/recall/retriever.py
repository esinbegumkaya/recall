from collections import defaultdict
import re

from recall.database import (
    search_chunks_fts,
)


DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_RESULT_LIMIT = 10

MIN_EVIDENCE_CHARACTERS = 40
MIN_EVIDENCE_WORDS = 5

MAX_CHUNKS_PER_FILE = 3


def normalize_text(text: str) -> str:
    return " ".join(
        (text or "").split()
    )


def meaningful_word_count(
    text: str,
) -> int:
    return len(
        re.findall(
            r"\b[\wÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜü]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def is_low_quality_chunk(
    text: str,
) -> bool:
    """
    Reject fragments that are too small to function as
    useful evidence.

    Examples:
        "65"
        "Hours Limit"
        "Macro F1= 1"
    """

    normalized = normalize_text(
        text
    )

    if len(
        normalized
    ) < MIN_EVIDENCE_CHARACTERS:
        return True

    if (
        meaningful_word_count(
            normalized
        )
        < MIN_EVIDENCE_WORDS
    ):
        return True

    return False


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


def lexical_candidates(
    query: str,
    candidate_limit: int = (
        DEFAULT_CANDIDATE_LIMIT
    ),
):
    """
    Broad FTS candidate generation.

    FTS is intentionally allowed to retrieve more results
    than we eventually expose.
    """

    rows = search_chunks_fts(
        query,
        limit=candidate_limit,
    )

    candidates = []

    for row in rows:
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
                "retrieval_method": (
                    "FTS5"
                ),
            }
        )

    return candidates


def filter_candidate_quality(
    candidates: list,
):
    return [
        candidate
        for candidate in candidates
        if not is_low_quality_chunk(
            candidate["text"]
        )
    ]


def remove_exact_duplicates(
    candidates: list,
):
    """
    Remove identical text duplicated across copied files or
    repeated chunks while preserving the best-ranked copy.
    """

    seen = set()
    result = []

    for candidate in candidates:
        key = (
            canonicalize_for_duplicate_check(
                candidate["text"]
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


def diversify_by_file(
    candidates: list,
    max_chunks_per_file: int = (
        MAX_CHUNKS_PER_FILE
    ),
):
    """
    Prevent one highly repetitive file from monopolizing the
    candidate set.
    """

    file_counts = defaultdict(
        int
    )

    diversified = []

    for candidate in candidates:
        file_path = candidate[
            "file_path"
        ]

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
    Recall v1 lexical candidate retrieval.

    Pipeline:
        FTS5
        -> minimum evidence quality
        -> exact duplicate removal
        -> per-file diversity
        -> final candidate set
    """

    candidates = lexical_candidates(
        query,
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
        diversify_by_file(
            candidates
        )
    )

    return candidates[
        :result_limit
    ]