import re
from typing import List, Dict


MIN_EVIDENCE_LENGTH = 25
MIN_EVIDENCE_WORDS = 4


def normalize_evidence_text(
    text: str,
) -> str:
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def is_low_information_unit(
    text: str,
) -> bool:
    normalized = normalize_evidence_text(
        text
    )

    if len(normalized) < MIN_EVIDENCE_LENGTH:
        return True

    words = re.findall(
        r"\b\w+\b",
        normalized,
    )

    if len(words) < MIN_EVIDENCE_WORDS:
        return True

    return False


def split_evidence_units(
    text: str,
) -> List[str]:
    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    lines = [
        normalize_evidence_text(line)
        for line in text.splitlines()
        if line.strip()
    ]

    units = []

    for line in lines:
        cleaned = re.sub(
            r"^[\s•●▪■\-–—*]+",
            "",
            line,
        ).strip()

        if (
            cleaned
            and not is_low_information_unit(
                cleaned
            )
        ):
            units.append(
                cleaned
            )

    if len(units) > 1:
        return units

    parts = re.split(
        r"\s+[•●▪■]\s+",
        text,
    )

    parts = [
        normalize_evidence_text(
            part
        )
        for part in parts
        if part.strip()
    ]

    parts = [
        part
        for part in parts
        if not is_low_information_unit(
            part
        )
    ]

    if len(parts) > 1:
        return parts

    normalized_text = (
        normalize_evidence_text(
            text
        )
    )

    if is_low_information_unit(
        normalized_text
    ):
        return []

    return [
        normalized_text
    ]


def cosine_similarity(
    a: List[float],
    b: List[float],
) -> float:
    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = sum(
        x * x
        for x in a
    ) ** 0.5

    magnitude_b = sum(
        y * y
        for y in b
    ) ** 0.5

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return dot_product / (
        magnitude_a
        * magnitude_b
    )


def evidence_dedup_key(
    text: str,
) -> str:
    normalized = (
        normalize_evidence_text(
            text
        )
        .lower()
    )

    normalized = re.sub(
        r"[^\w\s]",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def build_evidence_units(
    results: List[Dict],
) -> List[Dict]:
    evidence_units = []

    seen_exact = set()

    for source_number, result in enumerate(
        results,
        start=1,
    ):
        units = split_evidence_units(
            result["chunk_text"]
        )

        for unit_index, unit_text in enumerate(
            units
        ):
            dedup_key = evidence_dedup_key(
                unit_text
            )

            if dedup_key in seen_exact:
                continue

            seen_exact.add(
                dedup_key
            )

            evidence_units.append(
                {
                    "source_number": (
                        source_number
                    ),
                    "unit_index": (
                        unit_index
                    ),
                    "text": (
                        unit_text
                    ),
                    "file_path": (
                        result["file_path"]
                    ),
                    "chunk_index": (
                        result["chunk_index"]
                    ),
                    "page_number": (
                        result.get(
                            "page_number"
                        )
                    ),
                    "section_name": (
                        result.get(
                            "section_name"
                        )
                    ),
                }
            )

    return evidence_units


def rank_evidence_units(
    query: str,
    results: List[Dict],
    embedding_client,
    top_k: int = 5,
) -> List[Dict]:
    evidence_units = (
        build_evidence_units(
            results
        )
    )

    if not evidence_units:
        return []

    query_response = (
        embedding_client
        .generate_embedding(
            query
        )
    )

    query_embedding = (
        query_response
        .data[0]
        .embedding
    )

    ranked_units = []

    for unit in evidence_units:
        unit_response = (
            embedding_client
            .generate_embedding(
                unit["text"]
            )
        )

        unit_embedding = (
            unit_response
            .data[0]
            .embedding
        )

        score = cosine_similarity(
            query_embedding,
            unit_embedding,
        )

        ranked_unit = dict(
            unit
        )

        ranked_unit[
            "evidence_score"
        ] = score

        ranked_units.append(
            ranked_unit
        )

    ranked_units.sort(
        key=lambda item: item[
            "evidence_score"
        ],
        reverse=True,
    )

    return ranked_units[:top_k]