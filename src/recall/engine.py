from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from recall.database import (
    get_index_stats,
    initialize_database,
)

from recall.evidence import (
    attach_parent_context,
    build_evidence_units,
    deduplicate_overlapping_evidence_units,
    normalize_evidence_text,
)

from recall.evidence_gate import (
    detect_required_evidence_type,
    filter_by_evidence_type,
    filter_by_topic,
    infer_evidence_type,
)

from recall.generator import (
    ABSTENTION_MESSAGE,
    generate_grounded_answer_from_evidence,
)

from recall.nli_judge import LocalNLIJudge

from recall.retrieval import (
    retrieve_candidates,
)


# =========================================================
# Configuration
# =========================================================

DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_RESULT_LIMIT = 15
DEFAULT_SOURCE_LIMIT = 8

# Structural + topical + provenance gates do most of the
# precision work. NLI is an admission check, therefore
# intentionally permissive.
DEFAULT_NLI_THRESHOLD = 0.15

DEFAULT_CHAT_MODEL_NAME = "qwen3.5-2b"


# =========================================================
# Production source hygiene
# =========================================================

INTERNAL_ARTIFACT_FILENAMES = {
    "recall_review.txt",
    "nli_bug_review.txt",
    "engine_inputs.txt",
}


def is_internal_artifact_candidate(
    candidate: Dict[str, Any],
) -> bool:
    """
    Reject Recall's own development / regression artifacts
    from the production answer path.

    This prevents test statements such as
    "Build AI agents and RAG pipelines."
    from being treated as real user evidence.
    """

    file_path = (
        candidate.get("file_path")
        or ""
    )

    file_name = (
        candidate.get("file_name")
        or (
            Path(file_path).name
            if file_path
            else ""
        )
    )

    normalized_name = (
        file_name
        .strip()
        .casefold()
    )

    normalized_internal_names = {
        name.casefold()
        for name in INTERNAL_ARTIFACT_FILENAMES
    }

    if (
        normalized_name
        in normalized_internal_names
    ):
        return True

    text = (
        candidate.get(
            "text",
            "",
        )
        or ""
    ).casefold()

    internal_markers = (
        "from recall.",
        "def test_",
        "pytest",
        "judge_claim(",
        "filter_by_topic(",
        "test_real_agentic_evidence",
        "supporting_hypothesis",
        "assert result[",
    )

    marker_count = sum(
        1
        for marker in internal_markers
        if marker in text
    )

    # Avoid rejecting ordinary documents because of one
    # accidental keyword.
    return marker_count >= 2


def filter_internal_artifacts(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if not is_internal_artifact_candidate(
            candidate
        )
    ]


# =========================================================
# Result models
# =========================================================

@dataclass
class RecallSource:
    source_number: int

    file_name: str
    file_path: str

    chunk_index: int
    text: str

    page_number: Optional[int] = None
    section_name: Optional[str] = None
    parent_text: Optional[str] = None

    lexical_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    retrieval_method: Optional[str] = None

    evidence_type: Optional[str] = None
    topic_match: Optional[str] = None

    nli_score: Optional[float] = None
    supporting_hypothesis: Optional[str] = None

    requested_organization: Optional[str] = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "source_number": (
                self.source_number
            ),
            "file_name": (
                self.file_name
            ),
            "file_path": (
                self.file_path
            ),
            "chunk_index": (
                self.chunk_index
            ),
            "text": (
                self.text
            ),
            "page_number": (
                self.page_number
            ),
            "section_name": (
                self.section_name
            ),
            "parent_text": (
                self.parent_text
            ),
            "lexical_score": (
                self.lexical_score
            ),
            "retrieval_score": (
                self.retrieval_score
            ),
            "retrieval_method": (
                self.retrieval_method
            ),
            "evidence_type": (
                self.evidence_type
            ),
            "topic_match": (
                self.topic_match
            ),
            "nli_score": (
                self.nli_score
            ),
            "supporting_hypothesis": (
                self.supporting_hypothesis
            ),
            "requested_organization": (
                self.requested_organization
            ),
        }


@dataclass
class RecallResult:
    query: str
    answer: str

    sources: List[RecallSource] = field(
        default_factory=list
    )

    other_relevant_files: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    abstained: bool = False
    abstention_reason: Optional[str] = None

    candidate_count: int = 0
    evidence_count: int = 0

    elapsed_seconds: float = 0.0

    index_stats: Dict[str, int] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "query": (
                self.query
            ),
            "answer": (
                self.answer
            ),
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "other_relevant_files": (
                self.other_relevant_files
            ),
            "abstained": (
                self.abstained
            ),
            "abstention_reason": (
                self.abstention_reason
            ),
            "candidate_count": (
                self.candidate_count
            ),
            "evidence_count": (
                self.evidence_count
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "index_stats": (
                self.index_stats
            ),
        }


# =========================================================
# Query helpers
# =========================================================

def normalize_query(
    query: str,
) -> str:
    return " ".join(
        (query or "")
        .strip()
        .split()
    )


def extract_requested_organization(
    query: str,
) -> Optional[str]:
    """
    Extract explicit organization constraints.

    Examples:
        Did this person work with AI agents at Cubewise?
            -> Cubewise

        Did this person use RAG at Microsoft?
            -> Microsoft

        Where did this person work with AI agents?
            -> None

    Only explicit:
        at
        for
        during

    constraints are interpreted here.
    """

    normalized = normalize_query(
        query
    )

    patterns = [
        (
            r"\b(?:at|for)\s+"
            r"([A-Z][A-Za-z0-9&.+\- ]{1,80}?)"
            r"(?:\?|$)"
        ),
        (
            r"\bduring\s+(?:the\s+)?"
            r"([A-Z][A-Za-z0-9&.+\- ]{1,80}?)"
            r"(?:\?|$)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
        )

        if not match:
            continue

        organization = (
            match.group(1)
            .strip()
            .rstrip(
                "?.!,"
            )
            .strip()
        )

        if organization:
            return organization

    return None


def normalize_entity_text(
    text: str,
) -> str:
    normalized = (
        text or ""
    ).casefold()

    normalized = re.sub(
        r"[^\w\s]",
        " ",
        normalized,
        flags=re.UNICODE,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def provenance_text(
    unit: Dict[str, Any],
) -> str:
    file_path = (
        unit.get("file_path")
        or ""
    )

    parts = [
        unit.get(
            "parent_text"
        )
        or "",
        unit.get(
            "text"
        )
        or "",
        unit.get(
            "section_name"
        )
        or "",
        (
            Path(file_path).stem
            if file_path
            else ""
        ),
    ]

    return normalize_entity_text(
        " ".join(
            parts
        )
    )


def organization_matches_unit(
    organization: str,
    unit: Dict[str, Any],
) -> bool:
    organization_normalized = (
        normalize_entity_text(
            organization
        )
    )

    if not organization_normalized:
        return True

    haystack = provenance_text(
        unit
    )

    if (
        organization_normalized
        in haystack
    ):
        return True

    organization_tokens = [
        token
        for token
        in organization_normalized.split()
        if len(token) >= 3
    ]

    if not organization_tokens:
        return False

    matched_tokens = sum(
        1
        for token
        in organization_tokens
        if token in haystack
    )

    return (
        matched_tokens
        == len(
            organization_tokens
        )
    )


def filter_by_query_provenance(
    query: str,
    evidence_units: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    requested_organization = (
        extract_requested_organization(
            query
        )
    )

    if not requested_organization:
        return evidence_units

    matching = []

    for unit in evidence_units:
        if not organization_matches_unit(
            requested_organization,
            unit,
        ):
            continue

        updated = dict(
            unit
        )

        updated[
            "requested_organization"
        ] = requested_organization

        matching.append(
            updated
        )

    return matching


# =========================================================
# Retrieval -> Evidence adapter
# =========================================================

def adapt_candidates_for_evidence(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    retrieval.py currently exposes the chunk body as:

        candidate["text"]

    evidence.build_evidence_units() expects:

        result["chunk_text"]

    This adapter maintains backwards compatibility without
    changing the tested evidence module.
    """

    adapted_candidates = []

    for candidate in candidates:
        adapted = dict(
            candidate
        )

        adapted[
            "chunk_text"
        ] = candidate.get(
            "text",
            candidate.get(
                "chunk_text",
                "",
            ),
        )

        adapted_candidates.append(
            adapted
        )

    return adapted_candidates


# =========================================================
# Preserve retrieval metadata on evidence
# =========================================================

def make_candidate_lookup_key(
    file_path: str,
    chunk_index: Any,
) -> tuple:
    return (
        str(
            file_path
            or ""
        ),
        int(
            chunk_index
            or 0
        ),
    )


def attach_retrieval_metadata(
    evidence_units: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    build_evidence_units preserves basic provenance but not
    every retrieval-ranking field. Restore those fields here.
    """

    candidate_lookup = {}

    for candidate in candidates:
        key = make_candidate_lookup_key(
            candidate.get(
                "file_path",
                "",
            ),
            candidate.get(
                "chunk_index",
                0,
            ),
        )

        candidate_lookup[
            key
        ] = candidate

    updated_units = []

    for unit in evidence_units:
        key = make_candidate_lookup_key(
            unit.get(
                "file_path",
                "",
            ),
            unit.get(
                "chunk_index",
                0,
            ),
        )

        candidate = (
            candidate_lookup.get(
                key,
                {}
            )
        )

        updated = dict(
            unit
        )

        updated[
            "file_name"
        ] = (
            candidate.get(
                "file_name"
            )
            or (
                Path(
                    unit.get(
                        "file_path",
                        "",
                    )
                ).name
                if unit.get(
                    "file_path"
                )
                else ""
            )
        )

        updated[
            "lexical_score"
        ] = candidate.get(
            "lexical_score"
        )

        updated[
            "retrieval_score"
        ] = candidate.get(
            "retrieval_score"
        )

        updated[
            "retrieval_method"
        ] = candidate.get(
            "retrieval_method"
        )

        updated_units.append(
            updated
        )

    return updated_units


# =========================================================
# Evidence preparation
# =========================================================

def prepare_evidence_units(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evidence_candidates = (
        adapt_candidates_for_evidence(
            candidates
        )
    )

    units = build_evidence_units(
        evidence_candidates
    )

    units = attach_retrieval_metadata(
        units,
        candidates,
    )

    units = attach_parent_context(
        units
    )

    # ---------------------------------------------
    # Structural gate
    # ---------------------------------------------

    required_type = detect_required_evidence_type(
        query
    )

    units = filter_by_evidence_type(
        query,
        units,
    )

    # Explicit professional/work questions must not
    # silently fall back to certificates or skills.
    if required_type == "EXPERIENCE":
        strict_experience_units = []

        for unit in units:
            evidence_type = infer_evidence_type(
                unit
            )

            parent_text = (
                unit.get("parent_text")
                or ""
            ).casefold()

            looks_like_experience_parent = any(
                marker in parent_text
                for marker in (
                    "intern",
                    "engineer",
                    "developer",
                    "consultant",
                    "specialist",
                    "analyst",
                    "manager",
                )
            )

            if (
                evidence_type == "EXPERIENCE"
                or looks_like_experience_parent
            ):
                updated = dict(
                    unit
                )

                updated[
                    "evidence_type"
                ] = "EXPERIENCE"

                strict_experience_units.append(
                    updated
                )

        units = strict_experience_units

    # ---------------------------------------------
    # Topical gate
    # ---------------------------------------------

    topical_units = filter_by_topic(
        query,
        units,
    )

    # filter_by_topic returns [] if it cannot infer a
    # concrete topic. Preserve the structurally valid
    # evidence in that case.
    if topical_units:
        units = topical_units

    # ---------------------------------------------
    # Query provenance / organization gate
    # ---------------------------------------------

    units = filter_by_query_provenance(
        query,
        units,
    )

    # ---------------------------------------------
    # Chunk overlap deduplication
    # ---------------------------------------------

    units = (
        deduplicate_overlapping_evidence_units(
            units
        )
    )

    return units


# =========================================================
# NLI admission
# =========================================================

def validate_evidence_with_nli(
    query: str,
    evidence_units: List[Dict[str, Any]],
    judge: LocalNLIJudge,
    threshold: float = DEFAULT_NLI_THRESHOLD,
) -> List[Dict[str, Any]]:
    validated = []

    hypothesis = (
        judge.build_hypothesis(
            query
        )
    )

    for unit in evidence_units:
        text = (
            unit.get(
                "text",
                "",
            )
            .strip()
        )

        if not text:
            continue

        result = judge.judge(
            query,
            text,
        )

        score = float(
            result.get(
                "entailment_score",
                0.0,
            )
        )

        if score < threshold:
            continue

        updated = dict(
            unit
        )

        updated[
            "nli_score"
        ] = score

        updated[
            "supporting_subquery"
        ] = query

        updated[
            "supporting_hypothesis"
        ] = result.get(
            "hypothesis",
            hypothesis,
        )

        validated.append(
            updated
        )

    validated.sort(
        key=lambda item: (
            float(
                item.get(
                    "nli_score",
                    0.0,
                )
                or 0.0
            ),
            float(
                item.get(
                    "retrieval_score",
                    0.0,
                )
                or 0.0
            ),
        ),
        reverse=True,
    )

    return validated


# =========================================================
# Deterministic grounded answer
# =========================================================

def clean_evidence_for_answer(
    text: str,
) -> str:
    cleaned = (
        normalize_evidence_text(
            text
        )
    )

    cleaned = re.sub(
        r"^[\s\u2022\u25cf\u25aa\u25a0\-–—*]+",
        "",
        cleaned,
    ).strip()

    return cleaned


def source_label(
    source_number: int,
) -> str:
    return (
        f"[Source {source_number}]"
    )


def select_parent_label(
    unit: Dict[str, Any],
) -> str:
    parent_text = (
        unit.get(
            "parent_text"
        )
    )

    if parent_text:
        return (
            normalize_evidence_text(
                str(
                    parent_text
                )
            )
        )

    section_name = (
        unit.get(
            "section_name"
        )
    )

    if section_name:
        return (
            normalize_evidence_text(
                str(
                    section_name
                )
            )
        )

    file_name = (
        unit.get(
            "file_name"
        )
    )

    if file_name:
        return file_name

    file_path = (
        unit.get(
            "file_path"
        )
        or ""
    )

    if file_path:
        return (
            Path(
                file_path
            ).name
        )

    return "Relevant evidence"


def compose_grounded_answer(
    evidence_units: List[Dict[str, Any]],
    max_sources: int = DEFAULT_SOURCE_LIMIT,
) -> str:
    """
    Deterministic composer.

    No factual paraphrasing is delegated to the LLM.
    The answer is composed only from already validated
    evidence.
    """

    if not evidence_units:
        return (
            "I could not find sufficiently supported "
            "evidence for this question in the indexed "
            "files."
        )

    selected = evidence_units[
        :max_sources
    ]

    grouped: Dict[
        str,
        List[tuple],
    ] = {}

    for source_number, unit in enumerate(
        selected,
        start=1,
    ):
        parent = select_parent_label(
            unit
        )

        text = clean_evidence_for_answer(
            unit.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        grouped.setdefault(
            parent,
            [],
        ).append(
            (
                source_number,
                text,
            )
        )

    if not grouped:
        return (
            "I could not find sufficiently supported "
            "evidence for this question in the indexed "
            "files."
        )

    lines = []

    for parent, items in grouped.items():
        lines.append(
            f"{parent}:"
        )

        for (
            source_number,
            text,
        ) in items:
            lines.append(
                "- "
                + text
                + " "
                + source_label(
                    source_number
                )
            )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    ).strip()


# =========================================================
# Source conversion
# =========================================================

def build_sources(
    evidence_units: List[Dict[str, Any]],
    max_sources: int = DEFAULT_SOURCE_LIMIT,
) -> List[RecallSource]:
    sources = []

    selected = evidence_units[
        :max_sources
    ]

    for source_number, unit in enumerate(
        selected,
        start=1,
    ):
        file_path = str(
            unit.get(
                "file_path",
                "",
            )
        )

        file_name = (
            unit.get(
                "file_name"
            )
            or (
                Path(
                    file_path
                ).name
                if file_path
                else "Unknown file"
            )
        )

        sources.append(
            RecallSource(
                source_number=(
                    source_number
                ),
                file_name=(
                    file_name
                ),
                file_path=(
                    file_path
                ),
                chunk_index=int(
                    unit.get(
                        "chunk_index",
                        0,
                    )
                    or 0
                ),
                text=(
                    unit.get(
                        "text",
                        "",
                    )
                ),
                page_number=(
                    unit.get(
                        "page_number"
                    )
                ),
                section_name=(
                    unit.get(
                        "section_name"
                    )
                ),
                parent_text=(
                    unit.get(
                        "parent_text"
                    )
                ),
                lexical_score=(
                    unit.get(
                        "lexical_score"
                    )
                ),
                retrieval_score=(
                    unit.get(
                        "retrieval_score"
                    )
                ),
                retrieval_method=(
                    unit.get(
                        "retrieval_method"
                    )
                ),
                evidence_type=(
                    unit.get(
                        "evidence_type"
                    )
                ),
                topic_match=(
                    unit.get(
                        "topic_match"
                    )
                ),
                nli_score=(
                    unit.get(
                        "nli_score"
                    )
                ),
                supporting_hypothesis=(
                    unit.get(
                        "supporting_hypothesis"
                    )
                ),
                requested_organization=(
                    unit.get(
                        "requested_organization"
                    )
                ),
            )
        )

    return sources


def build_other_relevant_files(
    candidates: List[Dict[str, Any]],
    selected_sources: List[RecallSource],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    selected_paths = {
        source.file_path
        for source
        in selected_sources
    }

    seen_paths = set()

    other_files = []

    for candidate in candidates:
        file_path = str(
            candidate.get(
                "file_path",
                "",
            )
        )

        if not file_path:
            continue

        if (
            file_path
            in selected_paths
        ):
            continue

        if (
            file_path
            in seen_paths
        ):
            continue

        seen_paths.add(
            file_path
        )

        other_files.append(
            {
                "file_name": (
                    candidate.get(
                        "file_name"
                    )
                    or Path(
                        file_path
                    ).name
                ),
                "file_path": (
                    file_path
                ),
                "page_number": (
                    candidate.get(
                        "page_number"
                    )
                ),
                "section_name": (
                    candidate.get(
                        "section_name"
                    )
                ),
                "chunk_index": (
                    candidate.get(
                        "chunk_index"
                    )
                ),
                "lexical_score": (
                    candidate.get(
                        "lexical_score"
                    )
                ),
                "retrieval_score": (
                    candidate.get(
                        "retrieval_score"
                    )
                ),
                "retrieval_method": (
                    candidate.get(
                        "retrieval_method"
                    )
                ),
            }
        )

        if (
            len(
                other_files
            )
            >= limit
        ):
            break

    return other_files


# =========================================================
# Generator-abstention safety check
# =========================================================

_QUERY_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "person", "people", "does", "do", "did", "has", "have", "had",
    "is", "are", "was", "were", "what", "where", "which", "who",
    "when", "why", "how", "find", "show", "tell", "give", "list",
    "work", "worked", "working", "experience", "experienced",
    "involving", "involve", "involves", "with", "using", "used",
    "use", "at", "for", "from", "of", "to", "in", "on", "and",
    "or", "their", "his", "her", "its",
}


def _normalize_support_token(token: str) -> str:
    # The tokenizer deliberately accepts characters such as ``.`` and ``-``
    # so technical terms can survive extraction, but sentence-final
    # punctuation must never become part of the semantic token.  Without
    # this, a query ending in ``classification.`` produced the literal token
    # ``classification.`` and failed an otherwise exact evidence match.
    token = (token or "").casefold().strip()
    token = token.strip(".,;:!?()[]{}\"'`-_")

    irregular = {
        "images": "image",
        "imaging": "image",
        "agents": "agent",
        "workflows": "workflow",
        "projects": "project",
        "restaurants": "restaurant",
    }

    if token in irregular:
        return irregular[token]

    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("s"):
        return token[:-1]

    return token


def _query_support_tokens(query: str) -> set[str]:
    tokens = set()

    for raw in re.findall(r"[A-Za-z0-9+#.\-]+", query or ""):
        token = _normalize_support_token(raw)

        if not token:
            continue

        if token in _QUERY_STOPWORDS:
            continue

        if len(token) < 2:
            continue

        tokens.add(token)

    return tokens


def _evidence_support_tokens(
    evidence_units: List[Dict[str, Any]],
) -> set[str]:
    parts: List[str] = []

    for unit in evidence_units:
        parts.extend(
            [
                str(unit.get("text") or ""),
                str(unit.get("parent_text") or ""),
                str(unit.get("section_name") or ""),
                str(unit.get("file_name") or ""),
            ]
        )

    tokens = set()

    for raw in re.findall(
        r"[A-Za-z0-9+#.\-]+",
        " ".join(parts),
    ):
        token = _normalize_support_token(raw)

        if token:
            tokens.add(token)

    return tokens


def has_direct_query_support(
    query: str,
    evidence_units: List[Dict[str, Any]],
) -> bool:
    """
    Decide whether an explicit model abstention should be overridden by
    Recall's deterministic grounded composer.

    This check is intentionally evidence-only: it never invents an answer.
    It asks whether the validated evidence explicitly covers the important
    concepts in the user's query.  Concept aliases are used for ordinary
    morphological/domain variants such as ``image``/``imaging`` and
    ``medical``/``dermoscopic``.

    The rule is strict enough to keep unsupported preference questions such
    as ``favorite restaurant`` as abstentions: seeing the word ``restaurant``
    alone is not enough because the independent concept ``favorite`` is not
    supported.
    """
    query_tokens = _query_support_tokens(query)

    if not query_tokens:
        return False

    evidence_tokens = _evidence_support_tokens(
        evidence_units
    )

    if not evidence_tokens:
        return False

    evidence_parts: List[str] = []

    for unit in evidence_units:
        evidence_parts.extend(
            [
                str(unit.get("text") or ""),
                str(unit.get("parent_text") or ""),
                str(unit.get("section_name") or ""),
                str(unit.get("file_name") or ""),
            ]
        )

    evidence_text = normalize_entity_text(
        " ".join(evidence_parts)
    )

    # Domain-equivalent concept groups.  These are not answer templates;
    # they only recognize explicit terminology already present in validated
    # evidence.
    concept_aliases = {
        "ai": (
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "llm",
            "rag",
            "agentic ai",
            "ai agent",
        ),
        "agent": (
            "agent",
            "agents",
            "ai agent",
            "ai agents",
            "agentic ai",
        ),
        "medical": (
            "medical",
            "clinical",
            "melanoma",
            "skin cancer",
            "dermoscopic",
        ),
        "image": (
            "image",
            "images",
            "imaging",
            "dermoscopic",
            "computer vision",
        ),
        "classification": (
            "classification",
            "classifier",
            "classify",
            "classified",
            "skin lesion",
            "11 class",
        ),
        "ibm": ("ibm",),
        "planning": ("planning",),
        "analytics": ("analytics",),
        "favorite": (
            "favorite",
            "favourite",
            "preferred",
        ),
        "restaurant": (
            "restaurant",
            "restaurants",
        ),
    }

    def concept_is_supported(token: str) -> bool:
        # Exact normalized token support remains the strongest signal.
        if token in evidence_tokens:
            return True

        aliases = concept_aliases.get(token)

        if not aliases:
            return False

        return any(
            normalize_entity_text(alias) in evidence_text
            for alias in aliases
        )

    matched_tokens = {
        token
        for token in query_tokens
        if concept_is_supported(token)
    }

    # One meaningful concept is enough only for genuinely one-concept
    # questions such as broad AI-experience queries after stopword removal.
    if len(query_tokens) == 1:
        return len(matched_tokens) == 1

    coverage = (
        len(matched_tokens)
        / len(query_tokens)
    )

    # Multi-concept questions need independent support for at least two
    # concepts and >= 2/3 of the meaningful query content.
    return (
        len(matched_tokens) >= 2
        and coverage >= 0.67
    )


# =========================================================
# Recall engine
# =========================================================

class RecallEngine:
    def __init__(
        self,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        result_limit: int = DEFAULT_RESULT_LIMIT,
        source_limit: int = DEFAULT_SOURCE_LIMIT,
        nli_threshold: float = DEFAULT_NLI_THRESHOLD,
        load_nli: bool = True,
        use_generation: bool = True,
        chat_model_name: str = DEFAULT_CHAT_MODEL_NAME,
        chat_client: Any = None,
    ):
        initialize_database()

        self.candidate_limit = (
            candidate_limit
        )

        self.result_limit = (
            result_limit
        )

        self.source_limit = (
            source_limit
        )

        self.nli_threshold = (
            nli_threshold
        )

        self.use_generation = (
            use_generation
        )

        self.chat_model_name = (
            chat_model_name
        )

        self.judge: Optional[
            LocalNLIJudge
        ] = None

        if load_nli:
            self.judge = (
                LocalNLIJudge()
            )

        # The chat model is intentionally lazy-loaded.
        # Retrieval-only or failing searches should not pay
        # the model-load cost.
        self.chat_client = chat_client
        self._chat_model = None

        # Kept for diagnostics without changing the public
        # RecallResult schema.
        self.last_generation_error: Optional[
            str
        ] = None

    def _get_judge(
        self,
    ) -> LocalNLIJudge:
        if self.judge is None:
            self.judge = (
                LocalNLIJudge()
            )

        return self.judge

    def _get_chat_client(
        self,
    ):
        """
        Lazily load the local Foundry chat model once and
        reuse its client across searches.

        Tests can inject a fake chat_client through __init__
        so they do not need to load Foundry Local.
        """

        if self.chat_client is not None:
            return self.chat_client

        if not self.use_generation:
            return None

        # Lazy import keeps the deterministic engine path
        # importable even when generation is disabled.
        from foundry_local_sdk import (
            Configuration,
            FoundryLocalManager,
        )

        try:
            FoundryLocalManager.initialize(
                Configuration(
                    app_name="Recall"
                )
            )
        except Exception as exc:
            # Foundry Local uses a process-wide manager.
            # Re-initialization is harmless if another Recall
            # component initialized it first.
            message = str(exc).casefold()

            if (
                "already"
                not in message
                or "initial"
                not in message
            ):
                raise

        manager = (
            FoundryLocalManager.instance
        )

        chat_model = (
            manager.catalog.get_model(
                self.chat_model_name
            )
        )

        chat_model.load()

        self._chat_model = chat_model
        self.chat_client = (
            chat_model.get_chat_client()
        )

        return self.chat_client

    def unload_chat_model(
        self,
    ) -> None:
        """
        Explicitly release the local chat model when the
        caller is finished with the engine.

        Normal searches intentionally keep it loaded so
        repeated queries do not reload the model.
        """

        if self._chat_model is not None:
            try:
                self._chat_model.unload()
            except Exception:
                pass

        self._chat_model = None
        self.chat_client = None

    def _generate_answer_or_fallback(
        self,
        query: str,
        validated: List[Dict[str, Any]],
        judge: LocalNLIJudge,
    ) -> str:
        """
        Try local LLM generation only after evidence has
        passed structural, topical, provenance and NLI gates.

        generator.py performs citation + claim validation.
        Any generation/model/validation failure falls back to
        the deterministic evidence composer.
        """

        deterministic_answer = (
            compose_grounded_answer(
                validated,
                max_sources=(
                    self.source_limit
                ),
            )
        )

        if not self.use_generation:
            return deterministic_answer

        self.last_generation_error = None

        try:
            chat_client = (
                self._get_chat_client()
            )

            if chat_client is None:
                return deterministic_answer

            generated_answer = (
                generate_grounded_answer_from_evidence(
                    chat_client=(
                        chat_client
                    ),
                    query=query,
                    evidence_units=(
                        validated[
                            :self.source_limit
                        ]
                    ),
                    claim_judge=judge,
                )
            )

            if (
                generated_answer
                and generated_answer.strip()
            ):
                normalized_answer = (
                    generated_answer.strip()
                )

                # Preserve an explicit model abstention.
                #
                # Previously an abstention was discarded here and the
                # deterministic evidence composer was used instead. That
                # could turn a correct "not enough evidence" decision into
                # an unsupported answer built from merely related chunks.
                if (
                    normalized_answer
                    == ABSTENTION_MESSAGE
                ):
                    # A small local model can abstain even when the
                    # validated evidence explicitly matches the important
                    # query concepts. In that case use the deterministic
                    # grounded composer instead of converting a supported
                    # query into a false abstention.
                    #
                    # If the validated evidence does NOT directly support
                    # the query (for example a "favorite restaurant"
                    # question matched only to generic restaurant text),
                    # preserve the abstention.
                    if has_direct_query_support(
                        query,
                        validated,
                    ):
                        self.last_generation_error = (
                            "MODEL_ABSTAINED_WITH_DIRECT_EVIDENCE; "
                            "used deterministic grounded fallback"
                        )
                        return deterministic_answer

                    return ABSTENTION_MESSAGE

                return normalized_answer

        except Exception as exc:
            self.last_generation_error = (
                f"{type(exc).__name__}: {exc}"
            )

        return deterministic_answer

    def search(
        self,
        query: str,
    ) -> RecallResult:
        started = (
            time.perf_counter()
        )

        query = normalize_query(
            query
        )

        index_stats = (
            get_index_stats()
        )

        # ---------------------------------------------
        # Empty query
        # ---------------------------------------------

        if not query:
            return RecallResult(
                query=query,
                answer=(
                    "Please enter a question."
                ),
                abstained=True,
                abstention_reason=(
                    "EMPTY_QUERY"
                ),
                elapsed_seconds=(
                    time.perf_counter()
                    - started
                ),
                index_stats=(
                    index_stats
                ),
            )

        # ---------------------------------------------
        # Hybrid retrieval
        # ---------------------------------------------

        candidates = retrieve_candidates(
            query,
            candidate_limit=(
                self.candidate_limit
            ),
            result_limit=(
                self.result_limit
            ),
        )

        # ---------------------------------------------
        # Remove Recall development artifacts
        # ---------------------------------------------

        candidates = (
            filter_internal_artifacts(
                candidates
            )
        )

        if not candidates:
            return RecallResult(
                query=query,
                answer=(
                    "I could not find relevant "
                    "information in the indexed files."
                ),
                abstained=True,
                abstention_reason=(
                    "NO_RETRIEVAL_RESULTS"
                ),
                candidate_count=0,
                evidence_count=0,
                elapsed_seconds=(
                    time.perf_counter()
                    - started
                ),
                index_stats=(
                    index_stats
                ),
            )

        # ---------------------------------------------
        # Evidence extraction + gates
        # ---------------------------------------------

        evidence_units = (
            prepare_evidence_units(
                query,
                candidates,
            )
        )

        if not evidence_units:
            return RecallResult(
                query=query,
                answer=(
                    "I found related files, but none "
                    "contained evidence that directly "
                    "matched the question."
                ),
                other_relevant_files=(
                    build_other_relevant_files(
                        candidates,
                        [],
                    )
                ),
                abstained=True,
                abstention_reason=(
                    "NO_VALID_EVIDENCE"
                ),
                candidate_count=(
                    len(
                        candidates
                    )
                ),
                evidence_count=0,
                elapsed_seconds=(
                    time.perf_counter()
                    - started
                ),
                index_stats=(
                    index_stats
                ),
            )

        # ---------------------------------------------
        # NLI admission
        # ---------------------------------------------

        judge = (
            self._get_judge()
        )

        validated = (
            validate_evidence_with_nli(
                query=query,
                evidence_units=(
                    evidence_units
                ),
                judge=(
                    judge
                ),
                threshold=(
                    self.nli_threshold
                ),
            )
        )

        if not validated:
            return RecallResult(
                query=query,
                answer=(
                    "I found related files, but the "
                    "available evidence was not strong "
                    "enough to support an answer."
                ),
                other_relevant_files=(
                    build_other_relevant_files(
                        candidates,
                        [],
                    )
                ),
                abstained=True,
                abstention_reason=(
                    "NLI_REJECTED_EVIDENCE"
                ),
                candidate_count=(
                    len(
                        candidates
                    )
                ),
                evidence_count=0,
                elapsed_seconds=(
                    time.perf_counter()
                    - started
                ),
                index_stats=(
                    index_stats
                ),
            )

        # ---------------------------------------------
        # Grounded generation with deterministic fallback
        # ---------------------------------------------

        selected_validated = (
            validated[
                :self.source_limit
            ]
        )

        sources = build_sources(
            selected_validated,
            max_sources=(
                self.source_limit
            ),
        )

        answer = (
            self._generate_answer_or_fallback(
                query=query,
                validated=(
                    selected_validated
                ),
                judge=judge,
            )
        )

        other_files = (
            build_other_relevant_files(
                candidates,
                sources,
            )
        )

        # ---------------------------------------------
        # Final result
        # ---------------------------------------------

        generation_abstained = (
            answer.strip()
            == ABSTENTION_MESSAGE
        )

        return RecallResult(
            query=(
                query
            ),
            answer=(
                answer
            ),
            sources=(
                sources
            ),
            other_relevant_files=(
                other_files
            ),
            abstained=(
                generation_abstained
            ),
            abstention_reason=(
                "GENERATOR_ABSTAINED"
                if generation_abstained
                else None
            ),
            candidate_count=(
                len(
                    candidates
                )
            ),
            evidence_count=(
                len(
                    validated
                )
            ),
            elapsed_seconds=(
                time.perf_counter()
                - started
            ),
            index_stats=(
                index_stats
            ),
        )


# =========================================================
# Convenience API
# =========================================================

_default_engine: Optional[
    RecallEngine
] = None


def get_default_engine(
) -> RecallEngine:
    global _default_engine

    if (
        _default_engine
        is None
    ):
        _default_engine = (
            RecallEngine()
        )

    return _default_engine


def search_recall(
    query: str,
) -> Dict[str, Any]:
    engine = (
        get_default_engine()
    )

    result = (
        engine.search(
            query
        )
    )

    return (
        result.to_dict()
    )
