from typing import List, Dict, Optional
import re


SECTION_TYPE_MAP = {
    "PROJECTS": "PROJECT",
    "EXPERIENCE": "EXPERIENCE",
    "WORK EXPERIENCE": "EXPERIENCE",
    "PROFESSIONAL EXPERIENCE": "EXPERIENCE",

    "CERTIFICATES": "CERTIFICATE",
    "CERTIFICATIONS": "CERTIFICATE",

    "SKILLS": "SKILL",
    "TECHNICAL SKILLS": "SKILL",
    "SKILLS, LANGUAGES & INTERESTS": "SKILL",
}


PROGRAMMING_LANGUAGE_TERMS = {
    "python",
    "java",
    "c++",
    "cpp",
    "c",
    "c#",
    "javascript",
    "typescript",
    "sql",
    "kotlin",
    "swift",
    "go",
    "golang",
    "rust",
    "php",
    "ruby",
    "scala",
    "r",
}


CLOUD_PLATFORM_TERMS = {
    "google cloud",
    "gcp",
    "azure",
    "azure ai foundry",
    "microsoft azure",
    "huawei cloud",
    "aws",
    "amazon web services",
}


AI_TECHNOLOGY_TERMS = {
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "llms",
    "rag",
    "rag pipelines",
    "ai agents",
    "agentic ai",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
}


def normalize_section_type(
    section_name: Optional[str],
) -> Optional[str]:
    if not section_name:
        return None

    normalized = (
        section_name
        .strip()
        .upper()
    )

    return SECTION_TYPE_MAP.get(
        normalized
    )


def contains_term(
    text: str,
    terms: set,
) -> bool:
    lower_text = text.lower()

    for term in terms:
        normalized_term = term.lower()

        pattern = (
            r"(?<![\w+#])"
            + re.escape(normalized_term)
            + r"(?![\w+#])"
        )

        if re.search(
            pattern,
            lower_text,
        ):
            return True

    return False


def infer_evidence_type(
    unit: Dict,
) -> Optional[str]:
    section_type = normalize_section_type(
        unit.get("section_name")
    )

    text = (
        unit.get("text", "")
        .strip()
    )

    lower_text = text.lower()

    if section_type == "PROJECT":
        return "PROJECT"

    if section_type == "EXPERIENCE":
        return "EXPERIENCE"

    if section_type == "CERTIFICATE":
        return "CERTIFICATE"

    if section_type == "SKILL":

        if (
            "languages:" in lower_text
            and "technical:" not in lower_text
        ):
            return "NATURAL_LANGUAGE"

        if contains_term(
            text,
            PROGRAMMING_LANGUAGE_TERMS,
        ):
            return "PROGRAMMING_LANGUAGE"

        if contains_term(
            text,
            CLOUD_PLATFORM_TERMS,
        ):
            return "CLOUD_PLATFORM"

        if contains_term(
            text,
            AI_TECHNOLOGY_TERMS,
        ):
            return "AI_TECHNOLOGY"

        return "OTHER_SKILL"

    return section_type


def detect_required_evidence_type(
    query: str,
) -> Optional[str]:
    query_lower = (
        query
        .lower()
        .strip()
    )

    if (
        "programming language"
        in query_lower
        or "programming languages"
        in query_lower
    ):
        return "PROGRAMMING_LANGUAGE"

    if (
        "cloud platform"
        in query_lower
        or "cloud platforms"
        in query_lower
    ):
        return "CLOUD_PLATFORM"

    if (
        "project"
        in query_lower
        or "projects"
        in query_lower
    ):
        return "PROJECT"

    if (
        "professionally"
        in query_lower
        or "professional experience"
        in query_lower
        or "work with"
        in query_lower
        or "worked with"
        in query_lower
        or "where did"
        in query_lower
        or "internship"
        in query_lower
        or "intern"
        in query_lower
    ):
        return "EXPERIENCE"

    if (
        "certificate"
        in query_lower
        or "certificates"
        in query_lower
        or "certification"
        in query_lower
        or "certifications"
        in query_lower
        or "training"
        in query_lower
    ):
        return "CERTIFICATE"

    if (
        "ai technology"
        in query_lower
        or "ai technologies"
        in query_lower
        or "machine learning"
        in query_lower
        or "deep learning"
        in query_lower
        or "nlp"
        in query_lower
        or "rag"
        in query_lower
    ):
        return "AI_TECHNOLOGY"

    return None


def filter_by_evidence_type(
    query: str,
    evidence_units: List[Dict],
) -> List[Dict]:
    required_type = (
        detect_required_evidence_type(
            query
        )
    )

    if required_type is None:
        return evidence_units

    matching_units = []

    for unit in evidence_units:
        evidence_type = (
            infer_evidence_type(
                unit
            )
        )

        updated_unit = dict(
            unit
        )

        updated_unit[
            "evidence_type"
        ] = evidence_type

        updated_unit[
            "required_evidence_type"
        ] = required_type

        if evidence_type == required_type:
            matching_units.append(
                updated_unit
            )

    if matching_units:
        return matching_units

    # Conservative fallback:
    # If no evidence matches the inferred type,
    # do not destroy retrieval completely.
    return evidence_units
AI_AGENT_TOPIC_TERMS = {
    "ai agent",
    "ai agents",
    "agentic ai",
    "agentic workflow",
    "agentic workflows",
    "autonomous agent",
    "autonomous agents",
    "multi-agent",
    "multi-agent system",
    "multi-agent systems",
}

RAG_TOPIC_TERMS = {
    "rag",
    "rag pipeline",
    "rag pipelines",
    "retrieval augmented generation",
    "retrieval-augmented generation",
}

QUANTUM_TOPIC_TERMS = {
    "quantum computing",
    "quantum computer",
    "quantum computers",
    "quantum computing researcher",
}

TOPIC_GROUPS = {
    "AI_AGENT": AI_AGENT_TOPIC_TERMS,
    "RAG": RAG_TOPIC_TERMS,
    "QUANTUM": QUANTUM_TOPIC_TERMS,
}


def requires_ai_agent_topic(
    query: str,
) -> bool:
    return contains_term(
        query,
        AI_AGENT_TOPIC_TERMS,
    )


def has_ai_agent_topic(
    text: str,
) -> bool:
    return contains_term(
        text,
        AI_AGENT_TOPIC_TERMS,
    )


def extract_dynamic_topic(
    query: str,
) -> Optional[str]:
    """
    Extract a concrete topic from common factual
    questions without requiring a hard-coded taxonomy.

    Examples:
    - "Where did this person use Kubernetes?"
      -> "Kubernetes"
    - "Did this person use IBM Planning Analytics?"
      -> "IBM Planning Analytics"
    - "Which document discusses GDPR compliance?"
      -> "GDPR compliance"
    """
    normalized = (
        query
        .strip()
    )

    patterns = [
        (
            r"^(?:where\s+)?did\s+this\s+person\s+"
            r"(?:use|build|develop|create|implement|"
            r"work\s+with|work\s+on|have\s+experience\s+with)"
            r"\s+(.+?)\??$"
        ),
        (
            r"^does\s+this\s+person\s+know\s+"
            r"(.+?)\??$"
        ),
        (
            r"^where\s+is\s+(.+?)\s+mentioned\??$"
        ),
        (
            r"^which\s+(?:document|file|source)\s+"
            r"(?:mentions|discusses|covers)\s+"
            r"(.+?)\??$"
        ),
    ]

    topic = None

    for pattern in patterns:
        match = re.match(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:
            topic = (
                match.group(1)
                .strip()
                .rstrip("?")
                .strip()
            )
            break

    if not topic:
        return None

    # Remove explicit provenance attribution from
    # the topic itself. Provenance is validated later.
    topic = re.split(
        r"\s+(?:at|for|during)\s+",
        topic,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if not topic:
        return None

    return topic


def detect_topic_requirement(
    query: str,
) -> Optional[str]:
    query_text = (
        query
        .lower()
        .strip()
    )

    if contains_term(
        query_text,
        AI_AGENT_TOPIC_TERMS,
    ):
        return "AI_AGENT"

    if contains_term(
        query_text,
        RAG_TOPIC_TERMS,
    ):
        return "RAG"

    if contains_term(
        query_text,
        QUANTUM_TOPIC_TERMS,
    ):
        return "QUANTUM"

    dynamic_topic = extract_dynamic_topic(
        query
    )

    if dynamic_topic:
        return (
            "DYNAMIC::"
            + dynamic_topic
        )

    return None


def has_topic(
    text: str,
    topic: str,
) -> bool:
    if topic.startswith(
        "DYNAMIC::"
    ):
        dynamic_topic = topic.split(
            "::",
            1,
        )[1].strip()

        if not dynamic_topic:
            return False

        return contains_term(
            text,
            {
                dynamic_topic,
            },
        )

    terms = TOPIC_GROUPS.get(
        topic,
        set(),
    )

    if not terms:
        return False

    return contains_term(
        text,
        terms,
    )


def filter_by_topic(
    query: str,
    evidence_units: List[Dict],
) -> List[Dict]:
    required_topic = detect_topic_requirement(
        query
    )

    if required_topic is None:
        return []

    matching_units = []

    for unit in evidence_units:
        text = (
            unit.get("text", "")
            .strip()
        )

        if has_topic(
            text,
            required_topic,
        ):
            updated_unit = dict(unit)

            updated_unit[
                "topic_match"
            ] = required_topic

            matching_units.append(
                updated_unit
            )

    return matching_units

