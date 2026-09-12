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



# A "mixed evidence" unit looks like "Label: value | Label2: value2" --
# common in resume/CV-style skill lines ("Technical: Python, SQL | Cloud:
# AWS, Azure"), but the same shape shows up in any structured document
# (specs, FAQs, product sheets: "Version: 2.1 | Released: 2026-03-01").
#
# The previous implementation only recognized this shape when EVERY label
# was a member of a small, hardcoded vocabulary (MIXED_EVIDENCE_LABELS:
# "technical", "cloud", "ai", "data science", "machine learning", ...) --
# tuned to look like the labels used in one specific CV. Any other document
# with a perfectly well-formed "Label: value | Label: value" structure, but
# using different label names, would fail this whitelist check and fall
# through to coarser, less structure-aware splitting further down in
# split_evidence_units().
#
# Fixed by validating the *shape* of a label instead of matching it against
# a fixed vocabulary: short, mostly-alphabetic, a handful of words at most.
# This still rejects things that aren't really labels (long sentences,
# numeric-heavy fragments) without requiring the label text itself to be
# known in advance.
_PLAUSIBLE_LABEL_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9 /&+\-]*$"
)

MAX_LABEL_WORDS = 5


def is_plausible_label(label: str) -> bool:
    label = label.strip()

    if not label or len(label) > 40:
        return False

    if not _PLAUSIBLE_LABEL_RE.match(label):
        return False

    if len(label.split()) > MAX_LABEL_WORDS:
        return False

    return True


def split_mixed_evidence_units(
    text: str,
) -> List[str]:
    normalized = normalize_evidence_text(
        text
    )

    if not normalized:
        return []

    parts = [
        normalize_evidence_text(part)
        for part in normalized.split("|")
        if part.strip()
    ]

    if len(parts) < 2:
        return []

    labeled_parts = []

    for part in parts:
        match = re.match(
            r"^([^:]{1,40}):\s*(.+)$",
            part,
        )

        if not match:
            return []

        label = match.group(1).strip()
        value = match.group(2).strip()

        if not is_plausible_label(label):
            return []

        if not value:
            return []

        labeled_parts.append(
            f"{label}: {value}"
        )

    if len(labeled_parts) < 2:
        return []

    return labeled_parts

def split_evidence_units(
    text: str,
) -> List[str]:
    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    mixed_units = split_mixed_evidence_units(
        text
    )

    if mixed_units:
        return mixed_units

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

    pipe_parts = [
        normalize_evidence_text(part)
        for part in text.split("|")
        if part.strip()
    ]

    pipe_parts = [
        part
        for part in pipe_parts
        if not is_low_information_unit(
            part
        )
    ]

    if len(pipe_parts) > 1:
        return pipe_parts
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



def evidence_token_set(
    text: str,
) -> set:
    normalized = evidence_dedup_key(
        text
    )

    return {
        token
        for token in normalized.split()
        if token
    }


def evidence_overlap_score(
    first_text: str,
    second_text: str,
) -> float:
    """
    Measure how much the smaller evidence unit is
    covered by the larger one.

    This is intended for chunk-overlap fragments,
    not general semantic similarity.
    """
    first_tokens = evidence_token_set(
        first_text
    )

    second_tokens = evidence_token_set(
        second_text
    )

    if (
        not first_tokens
        or not second_tokens
    ):
        return 0.0

    shared = (
        first_tokens
        & second_tokens
    )

    smaller_size = min(
        len(first_tokens),
        len(second_tokens),
    )

    if smaller_size == 0:
        return 0.0

    return (
        len(shared)
        / smaller_size
    )


def evidence_quality_score(
    unit: Dict,
) -> tuple:
    """
    Prefer evidence that carries provenance and looks
    like a complete textual unit.
    """
    text = normalize_evidence_text(
        unit.get(
            "text",
            "",
        )
    )

    has_parent = bool(
        unit.get(
            "parent_text"
        )
    )

    looks_complete = bool(
        text
        and not text.endswith(
            (
                ",",
                ";",
                ":",
            )
        )
    )

    starts_cleanly = bool(
        text
        and (
            text.startswith("?")
            or text[0].isupper()
        )
    )

    return (
        int(has_parent),
        int(looks_complete),
        int(starts_cleanly),
        len(text),
    )


def deduplicate_overlapping_evidence_units(
    evidence_units: List[Dict],
    overlap_threshold: float = 0.80,
) -> List[Dict]:
    """
    Remove near-duplicate evidence caused by adjacent
    chunk overlap.

    Units are compared only when their provenance scope
    is compatible:
    - same file
    - same page when page metadata exists
    - same section when section metadata exists
    - same supporting subquery when present
    - adjacent/same chunks when chunk indices exist

    For overlapping units, preserve the higher-quality
    unit rather than weakening downstream validation.
    """
    kept = []

    for candidate in evidence_units:
        duplicate_index = None

        for index, existing in enumerate(
            kept
        ):
            if (
                candidate.get("file_path")
                != existing.get("file_path")
            ):
                continue

            candidate_page = candidate.get(
                "page_number"
            )
            existing_page = existing.get(
                "page_number"
            )

            if (
                candidate_page is not None
                and existing_page is not None
                and candidate_page != existing_page
            ):
                continue

            candidate_section = candidate.get(
                "section_name"
            )
            existing_section = existing.get(
                "section_name"
            )

            if (
                candidate_section
                and existing_section
                and candidate_section
                != existing_section
            ):
                continue

            candidate_subquery = candidate.get(
                "supporting_subquery"
            )
            existing_subquery = existing.get(
                "supporting_subquery"
            )

            if (
                candidate_subquery
                and existing_subquery
                and candidate_subquery
                != existing_subquery
            ):
                continue

            candidate_chunk = candidate.get(
                "chunk_index"
            )
            existing_chunk = existing.get(
                "chunk_index"
            )

            if (
                candidate_chunk is not None
                and existing_chunk is not None
                and abs(
                    candidate_chunk
                    - existing_chunk
                ) > 1
            ):
                continue

            overlap = evidence_overlap_score(
                candidate.get(
                    "text",
                    "",
                ),
                existing.get(
                    "text",
                    "",
                ),
            )

            if overlap >= overlap_threshold:
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(
                candidate
            )
            continue

        existing = kept[
            duplicate_index
        ]

        if evidence_quality_score(
            candidate
        ) > evidence_quality_score(
            existing
        ):
            kept[
                duplicate_index
            ] = candidate

    return kept


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
        attach_parent_context(
            build_evidence_units(
                results
            )
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
def is_likely_heading(
    text: str,
) -> bool:
    if not text:
        return False

    normalized = normalize_evidence_text(
        text
    )

    if not normalized:
        return False

    if len(normalized) > 120:
        return False

    if normalized.endswith(
        (".", ";", ":")
    ):
        return False

    word_count = len(
        normalized.split()
    )

    if word_count > 14:
        return False

    return True

def attach_parent_context(
    evidence_units: List[Dict],
) -> List[Dict]:
    grouped = {}

    for unit in evidence_units:
        key = (
            unit.get("source_number"),
            unit.get("chunk_index"),
        )

        grouped.setdefault(
            key,
            []
        ).append(unit)

    updated_units = []

    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: item.get(
                "unit_index",
                0,
            ),
        )

        parent_text = None

        for unit in ordered:
            updated = dict(unit)

            unit_text = unit.get(
                "text",
                "",
            )

            if is_likely_heading(
                unit_text
            ):
                parent_text = unit_text
            elif parent_text:
                updated[
                    "parent_text"
                ] = parent_text

            updated_units.append(
                updated
            )

    return updated_units