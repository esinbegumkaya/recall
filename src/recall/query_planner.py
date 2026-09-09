from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException


# =========================================================
# Search plan
# =========================================================

@dataclass
class SearchPlan:
    original_query: str
    normalized_query: str

    # What should Recall do?
    action: str = "search"

    # Semantic ideas extracted from the request.
    concepts: List[str] = field(
        default_factory=list
    )

    # Strong lexical terms.
    keywords: List[str] = field(
        default_factory=list
    )

    # Terms likely related to filename.
    filename_terms: List[str] = field(
        default_factory=list
    )

    # Terms expected inside file contents.
    content_terms: List[str] = field(
        default_factory=list
    )

    # Optional metadata hints.
    extensions: List[str] = field(
        default_factory=list
    )

    location_terms: List[str] = field(
        default_factory=list
    )

    date_preference: Optional[str] = None

    created_after: Optional[str] = None
    created_before: Optional[str] = None

    modified_after: Optional[str] = None
    modified_before: Optional[str] = None

    # Retrieval preferences rather than rigid routes.
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0
    filename_weight: float = 1.0
    metadata_weight: float = 0.5
    recency_weight: float = 0.0

    limit: int = 10

    language: Optional[str] = None

    confidence: float = 0.0

    planner: str = "heuristic"

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return asdict(
            self
        )


# =========================================================
# Normalization
# =========================================================

TURKISH_TRANSLATION_TABLE = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "İ": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)


def normalize_text(
    text: str,
) -> str:
    text = (
        text
        or ""
    ).strip()

    text = text.translate(
        TURKISH_TRANSLATION_TABLE
    )

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(
            char
        )
    )

    text = text.casefold()

    text = re.sub(
        r"[^a-z0-9+#._\-\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# =========================================================
# Language
# =========================================================

def detect_language(
    query: str,
) -> str:
    raw = (
        query
        or ""
    ).casefold()

    if any(
        char in raw
        for char in (
            "ç",
            "ğ",
            "ı",
            "ö",
            "ş",
            "ü",
        )
    ):
        return "tr"

    normalized = normalize_text(
        raw
    )

    tokens = set(
        normalized.split()
    )

    turkish_tokens = {
        "en",
        "son",
        "guncel",
        "nerede",
        "hangi",
        "bul",
        "getir",
        "goster",
        "dosya",
        "dosyam",
        "not",
        "notlar",
        "notlarim",
        "ozgecmis",
        "sunum",
        "proje",
        "turkce",
    }

    typo_tokens = {
        "gtr",
        "gncel",
        "gncelni",
        "nrde",
        "notlarm",
        "cvm",
        "cvmn",
    }

    if (
        tokens & turkish_tokens
        or tokens & typo_tokens
    ):
        return "tr"

    return "en"


# =========================================================
# Token utilities
# =========================================================

STOPWORDS = {
    # English
    "a",
    "an",
    "the",
    "my",
    "me",
    "mine",
    "please",
    "can",
    "could",
    "would",
    "you",
    "i",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "and",

    # Turkish
    "benim",
    "bana",
    "beni",
    "bir",
    "bu",
    "su",
    "şu",
    "ve",
    "ile",
    "icin",
    "için",
    "mi",
    "mı",
    "mu",
    "mü",
}


ACTION_WORDS = {
    "find",
    "get",
    "bring",
    "show",
    "locate",
    "search",
    "open",

    "bul",
    "getir",
    "goster",
    "gtr",
    "ac",
}


RECENCY_WORDS = {
    "latest",
    "newest",
    "recent",
    "last",

    "guncel",
    "gncel",
    "gncelni",
    "son",
    "yeni",
    "en",
}


FILE_TYPE_HINTS = {
    "pdf": ".pdf",
    "docx": ".docx",
    "word": ".docx",
    "txt": ".txt",
    "text": ".txt",
    "markdown": ".md",
    "md": ".md",
    "python": ".py",
    "py": ".py",
}


LOCATION_HINTS = {
    "downloads",
    "download",
    "desktop",
    "documents",
    "onedrive",

    "indirilenler",
    "masaustu",
    "belgeler",
}


# =========================================================
# Fuzzy concept utilities
# =========================================================

def fuzzy_token_match(
    token: str,
    candidates: set[str],
    threshold: float = 78.0,
) -> Optional[str]:

    best_candidate = None
    best_score = 0.0

    for candidate in candidates:
        score = fuzz.ratio(
            token,
            candidate,
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_score >= threshold:
        return best_candidate

    return None


def clean_tokens(
    query: str,
) -> List[str]:

    normalized = normalize_text(
        query
    )

    tokens = []

    for token in normalized.split():

        if token in STOPWORDS:
            continue

        if len(token) <= 1:
            continue

        tokens.append(
            token
        )

    return tokens


# =========================================================
# Search signal extraction
# =========================================================

def extract_extensions(
    tokens: List[str],
) -> List[str]:

    extensions = []

    for token in tokens:
        extension = FILE_TYPE_HINTS.get(
            token
        )

        if extension:
            extensions.append(
                extension
            )

    return list(
        dict.fromkeys(
            extensions
        )
    )


def extract_locations(
    tokens: List[str],
) -> List[str]:

    locations = []

    for token in tokens:

        direct = (
            token
            if token
            in LOCATION_HINTS
            else None
        )

        if direct:
            locations.append(
                direct
            )
            continue

        fuzzy = fuzzy_token_match(
            token,
            LOCATION_HINTS,
            threshold=82.0,
        )

        if fuzzy:
            locations.append(
                fuzzy
            )

    return list(
        dict.fromkeys(
            locations
        )
    )


def detect_recency(
    tokens: List[str],
) -> bool:

    for token in tokens:

        if token in RECENCY_WORDS:
            return True

        fuzzy = fuzzy_token_match(
            token,
            RECENCY_WORDS,
            threshold=80.0,
        )

        if fuzzy:
            return True

    return False


# =========================================================
# Deterministic relative-date understanding
# =========================================================

def extract_relative_date_window(
    query: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve common relative date phrases into local ISO bounds."""
    normalized = normalize_text(query)
    now = datetime.now().astimezone()

    def day_start(value: datetime) -> datetime:
        return value.replace(hour=0, minute=0, second=0, microsecond=0)

    if "gecen hafta" in normalized or "last week" in normalized:
        return (now - timedelta(days=7)).isoformat(), now.isoformat()

    if "dun" in normalized or "yesterday" in normalized:
        return day_start(now - timedelta(days=1)).isoformat(), day_start(now).isoformat()

    if "bugun" in normalized or "today" in normalized:
        return day_start(now).isoformat(), now.isoformat()

    return None, None


# =========================================================
# Generic heuristic fallback planner
# =========================================================

def plan_query(
    query: str,
) -> SearchPlan:

    original_query = (
        " ".join(
            (
                query
                or ""
            ).strip().split()
        )
    )

    normalized_query = normalize_text(
        original_query
    )

    language = detect_language(
        original_query
    )

    tokens = clean_tokens(
        original_query
    )

    extensions = extract_extensions(
        tokens
    )

    locations = extract_locations(
        tokens
    )

    recency_requested = (
        detect_recency(
            tokens
        )
    )

    # ---------------------------------------------
    # Remove purely operational words.
    # Remaining terms become search signals.
    # ---------------------------------------------

    meaningful_tokens = []

    for token in tokens:

        if token in ACTION_WORDS:
            continue

        action_match = fuzzy_token_match(
            token,
            ACTION_WORDS,
            threshold=84.0,
        )

        if action_match:
            continue

        if token in RECENCY_WORDS:
            continue

        if (
            token
            in FILE_TYPE_HINTS
        ):
            continue

        if token in LOCATION_HINTS:
            continue

        meaningful_tokens.append(
            token
        )

    meaningful_tokens = list(
        dict.fromkeys(
            meaningful_tokens
        )
    )

    # ---------------------------------------------
    # Generic retrieval weights
    # ---------------------------------------------

    semantic_weight = 1.0
    lexical_weight = 0.9
    filename_weight = 0.9
    metadata_weight = 0.5
    recency_weight = 0.0

    if recency_requested:
        metadata_weight = 1.0
        recency_weight = 1.0

    if extensions:
        metadata_weight = max(
            metadata_weight,
            0.9,
        )

    if locations:
        metadata_weight = max(
            metadata_weight,
            0.9,
        )

    # ---------------------------------------------
    # Do NOT decide which exact file category
    # the query belongs to here.
    #
    # All meaningful terms become candidate
    # search concepts.
    # ---------------------------------------------

    return SearchPlan(
        original_query=original_query,
        normalized_query=normalized_query,

        action="search",

        concepts=meaningful_tokens,

        keywords=meaningful_tokens,

        filename_terms=meaningful_tokens,

        content_terms=meaningful_tokens,

        extensions=extensions,

        location_terms=locations,

        date_preference=(
            "most_recent"
            if recency_requested
            else None
        ),

        semantic_weight=(
            semantic_weight
        ),

        lexical_weight=(
            lexical_weight
        ),

        filename_weight=(
            filename_weight
        ),

        metadata_weight=(
            metadata_weight
        ),

        recency_weight=(
            recency_weight
        ),

        limit=10,

        language=language,

        confidence=0.60,

        planner="heuristic_fallback",
    )

# =========================================================
# Local LLM planner
# =========================================================

PLANNER_MODEL_NAME = "qwen3.5-2b"

ALLOWED_DATE_PREFERENCES = {
    None,
    "most_recent",
    "oldest",
}

ALLOWED_ACTIONS = {
    "search",
    "open",
}


def _dedupe_strings(
    values: Any,
    max_items: int = 12,
) -> List[str]:
    if not isinstance(values, list):
        return []

    cleaned: List[str] = []

    for value in values:
        if not isinstance(value, str):
            continue

        value = " ".join(value.strip().split())
        if not value:
            continue

        if value not in cleaned:
            cleaned.append(value)

        if len(cleaned) >= max_items:
            break

    return cleaned


def _clamp_weight(
    value: Any,
    default: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return max(0.0, min(1.0, number))


def _safe_optional_string(
    value: Any,
) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _extract_json_object(
    text: str,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _validate_llm_plan(
    query: str,
    payload: Dict[str, Any],
    fallback: SearchPlan,
) -> Optional[SearchPlan]:
    if not isinstance(payload, dict):
        return None

    concepts = _dedupe_strings(payload.get("concepts"))
    keywords = _dedupe_strings(payload.get("keywords"))
    filename_terms = _dedupe_strings(payload.get("filename_terms"))
    content_terms = _dedupe_strings(payload.get("content_terms"))
    extensions = _dedupe_strings(payload.get("extensions"), max_items=6)
    location_terms = _dedupe_strings(payload.get("location_terms"), max_items=6)

    normalized_extensions: List[str] = []
    for extension in extensions:
        extension = extension.casefold().strip()
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        if extension:
            normalized_extensions.append(extension)

    date_preference = payload.get("date_preference")
    if date_preference not in ALLOWED_DATE_PREFERENCES:
        date_preference = fallback.date_preference

    action = payload.get("action", "search")
    if action not in ALLOWED_ACTIONS:
        action = "search"

    language = payload.get("language")
    if language not in {"tr", "en", "mixed", None}:
        language = fallback.language

    try:
        confidence = float(payload.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75
    confidence = max(0.0, min(1.0, confidence))

    has_signal = any((
        concepts,
        keywords,
        filename_terms,
        content_terms,
        normalized_extensions,
        location_terms,
        date_preference,
    ))
    if not has_signal:
        return None

    try:
        limit = int(payload.get("limit", fallback.limit))
    except (TypeError, ValueError):
        limit = fallback.limit

    return SearchPlan(
        original_query=fallback.original_query,
        normalized_query=fallback.normalized_query,
        action=action,
        concepts=concepts or fallback.concepts,
        keywords=keywords or fallback.keywords,
        filename_terms=filename_terms or fallback.filename_terms,
        content_terms=content_terms or fallback.content_terms,
        extensions=normalized_extensions or fallback.extensions,
        location_terms=location_terms or fallback.location_terms,
        date_preference=date_preference,
        created_after=_safe_optional_string(payload.get("created_after")),
        created_before=_safe_optional_string(payload.get("created_before")),
        modified_after=_safe_optional_string(payload.get("modified_after")),
        modified_before=_safe_optional_string(payload.get("modified_before")),
        semantic_weight=_clamp_weight(payload.get("semantic_weight"), fallback.semantic_weight),
        lexical_weight=_clamp_weight(payload.get("lexical_weight"), fallback.lexical_weight),
        filename_weight=_clamp_weight(payload.get("filename_weight"), fallback.filename_weight),
        metadata_weight=_clamp_weight(payload.get("metadata_weight"), fallback.metadata_weight),
        recency_weight=_clamp_weight(payload.get("recency_weight"), fallback.recency_weight),
        limit=max(1, min(limit, 50)),
        language=language or fallback.language,
        confidence=confidence,
        planner="qwen3.5-2b",
    )


def _planner_system_prompt() -> str:
    return """You are Recall's local query planner.

Your only job is to convert a natural-language request for files into JSON search signals. Do not answer the request, choose a file, invent a filename, or claim a file exists.

Queries may be Turkish, English, mixed-language, abbreviated, misspelled, or informal. Infer intended concepts conservatively. Add useful English semantic equivalents for Turkish concepts when that can help search English documents.

Return exactly one JSON object and nothing else.

Schema:
{
  "action": "search" | "open",
  "concepts": [string],
  "keywords": [string],
  "filename_terms": [string],
  "content_terms": [string],
  "extensions": [string],
  "location_terms": [string],
  "date_preference": "most_recent" | "oldest" | null,
  "created_after": string | null,
  "created_before": string | null,
  "modified_after": string | null,
  "modified_before": string | null,
  "semantic_weight": number,
  "lexical_weight": number,
  "filename_weight": number,
  "metadata_weight": number,
  "recency_weight": number,
  "language": "tr" | "en" | "mixed",
  "confidence": number
}

Rules:
- All weights and confidence are 0..1.
- Preserve distinctive proper nouns and technical terms.
- Correct obvious typos conceptually, but never invent a specific filename.
- Filename-like clues go in filename_terms.
- Concepts likely inside contents go in content_terms.
- Generic words like file, document, belge, dosya, getir, bul are weak or omitted.
- For newest/latest/most recent requests, use date_preference=most_recent and high recency_weight.
- Content questions: semantic_weight and lexical_weight should dominate.
- Filename/entity clues: filename_weight should dominate.
- Do not hard-code a closed set of document categories.
- Only emit date bounds when explicitly supported by the query.

Example input: cvmn en gncelni gtr
Example output: {"action":"search","concepts":["cv","resume","curriculum vitae"],"keywords":["cv","resume"],"filename_terms":["cv","resume"],"content_terms":[],"extensions":[],"location_terms":[],"date_preference":"most_recent","created_after":null,"created_before":null,"modified_after":null,"modified_before":null,"semantic_weight":0.35,"lexical_weight":0.35,"filename_weight":1.0,"metadata_weight":0.9,"recency_weight":1.0,"language":"tr","confidence":0.95}

Example input: rag hakkında yazdığım belge
Example output: {"action":"search","concepts":["RAG","retrieval augmented generation"],"keywords":["RAG"],"filename_terms":[],"content_terms":["RAG","retrieval augmented generation"],"extensions":[],"location_terms":[],"date_preference":null,"created_after":null,"created_before":null,"modified_after":null,"modified_before":null,"semantic_weight":1.0,"lexical_weight":1.0,"filename_weight":0.2,"metadata_weight":0.2,"recency_weight":0.0,"language":"tr","confidence":0.95}

Example input: Harvad cv dosyam
Example output: {"action":"search","concepts":["Harvard","CV","resume"],"keywords":["Harvard","CV"],"filename_terms":["Harvard","CV"],"content_terms":["Harvard"],"extensions":[],"location_terms":[],"date_preference":null,"created_after":null,"created_before":null,"modified_after":null,"modified_before":null,"semantic_weight":0.45,"lexical_weight":0.5,"filename_weight":1.0,"metadata_weight":0.2,"recency_weight":0.0,"language":"tr","confidence":0.95}

Example input: Where is my most recent presentation about AI?
Example output: {"action":"search","concepts":["presentation","artificial intelligence","AI"],"keywords":["presentation","AI"],"filename_terms":["presentation","AI"],"content_terms":["artificial intelligence","AI"],"extensions":[],"location_terms":[],"date_preference":"most_recent","created_after":null,"created_before":null,"modified_after":null,"modified_before":null,"semantic_weight":0.85,"lexical_weight":0.8,"filename_weight":0.8,"metadata_weight":0.8,"recency_weight":0.85,"language":"en","confidence":0.95}"""


def plan_query_with_llm(
    query: str,
    use_llm: bool = True,
) -> SearchPlan:
    """Build a validated SearchPlan with local Qwen, falling back safely."""

    fallback = plan_query(query)

    relative_after, relative_before = extract_relative_date_window(query)
    if relative_after is not None:
        fallback.modified_after = relative_after
    if relative_before is not None:
        fallback.modified_before = relative_before

    if not use_llm or not fallback.original_query:
        return fallback

    model = None

    try:
        try:
            FoundryLocalManager.initialize(
                Configuration(app_name="Recall")
            )
        except FoundryLocalException as exc:
            if "already been initialized" not in str(exc):
                raise

        manager = FoundryLocalManager.instance
        model = manager.catalog.get_model(PLANNER_MODEL_NAME)
        model.load()
        client = model.get_chat_client()

        if hasattr(client, "settings"):
            try:
                client.settings.temperature = 0
                client.settings.max_tokens = 900
            except Exception:
                pass

        response = client.complete_chat([
            {
                "role": "system",
                "content": _planner_system_prompt(),
            },
            {
                "role": "user",
                "content": fallback.original_query,
            },
        ])

        text = response.choices[0].message.content or ""
        payload = _extract_json_object(text)

        if payload is None:
            retry_response = client.complete_chat([
                {
                    "role": "system",
                    "content": _planner_system_prompt(),
                },
                {
                    "role": "user",
                    "content": (
                        "Return STRICT JSON only. "
                        "Interpret abbreviations and obvious typos. "
                        f"Query: {fallback.original_query}"
                    ),
                },
            ])

            retry_text = (
                retry_response
                .choices[0]
                .message
                .content
                or ""
            )

            payload = _extract_json_object(retry_text)

        if payload is None:
            return fallback

        validated = _validate_llm_plan(
            query=query,
            payload=payload,
            fallback=fallback,
        )
        if validated is None or validated.confidence < 0.45:
            return fallback

        if relative_after is not None:
            validated.modified_after = relative_after
        if relative_before is not None:
            validated.modified_before = relative_before

        return validated

    except Exception:
        return fallback

    finally:
        if model is not None:
            try:
                model.unload()
            except Exception:
                pass
