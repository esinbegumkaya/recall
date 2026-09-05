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