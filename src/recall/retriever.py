import json
import math
import re

from rank_bm25 import BM25Okapi


SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30


STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "does",
    "do",
    "did",
    "this",
    "that",
    "these",
    "those",
    "person",
    "have",
    "has",
    "had",
    "with",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
}


def cosine_similarity(a, b):
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


def tokenize(text):
    tokens = re.findall(
        r"\b\w+\b",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    ]


def normalize_scores(scores):
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [
            0.0
            for _ in scores
        ]

    return [
        (score - minimum)
        / (maximum - minimum)
        for score in scores
    ]


def dense_retrieve(
    query_embedding,
    rows,
    top_k=10,
):
    rows = list(rows)

    if not rows:
        return []

    results = []

    for row in rows:
        embedding = json.loads(
            row["embedding"]
        )

        semantic_score = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append(
            {
                "file_path": row["file_path"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "page_number": row["page_number"],
                "section_name": row["section_name"],
                "semantic_score": semantic_score,
            }
        )

    results.sort(
        key=lambda result: result[
            "semantic_score"
        ],
        reverse=True,
    )

    return results[:top_k]


def hybrid_retrieve(
    query,
    query_embedding,
    rows,
    top_k=10,
):
    rows = list(rows)

    if not rows:
        return []

    semantic_scores = []

    for row in rows:
        embedding = json.loads(
            row["embedding"]
        )

        semantic_scores.append(
            cosine_similarity(
                query_embedding,
                embedding,
            )
        )

    corpus = [
        tokenize(
            row["chunk_text"]
        )
        for row in rows
    ]

    bm25 = BM25Okapi(
        corpus
    )

    bm25_scores = list(
        bm25.get_scores(
            tokenize(query)
        )
    )

    normalized_semantic = normalize_scores(
        semantic_scores
    )

    normalized_bm25 = normalize_scores(
        bm25_scores
    )

    results = []

    for index, row in enumerate(rows):
        hybrid_score = (
            SEMANTIC_WEIGHT
            * normalized_semantic[index]
            + BM25_WEIGHT
            * normalized_bm25[index]
        )

        results.append(
            {
                "file_path": row["file_path"],
                "chunk_index": row["chunk_index"],
                "chunk_text": row["chunk_text"],
                "page_number": row["page_number"],
                "section_name": row["section_name"],
                "semantic_score": semantic_scores[index],
                "bm25_score": bm25_scores[index],
                "hybrid_score": hybrid_score,
            }
        )

    results.sort(
        key=lambda result: result[
            "hybrid_score"
        ],
        reverse=True,
    )

    return results[:top_k]


def section_aware_dense_retrieve(
    query,
    query_embedding,
    rows,
    detect_section_intent,
    detect_query_mode,
    top_k=10,
    section_boost=0.075,
    heading_penalty=0.05,
):
    rows = list(rows)

    if not rows:
        return []

    dense_results = dense_retrieve(
        query_embedding=query_embedding,
        rows=rows,
        top_k=len(rows),
    )

    intended_section = (
        detect_section_intent(
            query
        )
    )

    query_mode = (
        detect_query_mode(
            query
        )
    )

    results = []

    for result in dense_results:
        updated_result = dict(
            result
        )

        section_match = (
            intended_section is not None
            and result.get(
                "section_name"
            ) == intended_section
        )

        chunk_text = (
            result["chunk_text"]
            .strip()
        )

        heading_only = (
            chunk_text.startswith("#")
            and "\n" not in chunk_text
        )

        low_information_penalty = (
            heading_penalty
            if (
                query_mode == "CONTENT_SEARCH"
                and heading_only
            )
            else 0.0
        )

        updated_result[
            "section_match"
        ] = section_match

        updated_result[
            "intended_section"
        ] = intended_section

        updated_result[
            "query_mode"
        ] = query_mode

        updated_result[
            "heading_only"
        ] = heading_only

        updated_result[
            "heading_penalty"
        ] = low_information_penalty

        updated_result[
            "final_score"
        ] = (
            result[
                "semantic_score"
            ]
            + (
                section_boost
                if section_match
                else 0.0
            )
            - low_information_penalty
        )

        results.append(
            updated_result
        )

    results.sort(
        key=lambda result: result[
            "final_score"
        ],
        reverse=True,
    )

    return results[:top_k]