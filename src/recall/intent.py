import re


STRONG_SECTION_INTENTS = {
    "PROJECTS": {
        "project",
        "projects",
        "built",
        "build",
        "created",
        "developed",
        "implementation",
    },

    "EXPERIENCE": {
        "professionally",
        "intern",
        "internship",
        "job",
        "employment",
        "employer",
        "company",
    },

    "CERTIFICATES": {
        "certificate",
        "certificates",
        "certification",
        "certifications",
        "training",
        "course",
        "courses",
        "program",
        "programs",
    },

    "SKILLS, LANGUAGES & INTERESTS": {
        "skill",
        "skills",
        "technology",
        "technologies",
        "language",
        "languages",
        "proficiency",
    },
}


WEAK_SECTION_INTENTS = {
    "EXPERIENCE": {
        "experience",
        "work",
        "worked",
    },

    "SKILLS, LANGUAGES & INTERESTS": {
        "know",
        "knows",
    },
}


FILE_LOOKUP_PATTERNS = [
    r"\bwhich file\b",
    r"\bwhat file\b",
    r"\bfind the file\b",
    r"\bfind the document\b",
    r"\bwhich document\b",
    r"\bwhat document\b",
    r"\bwhich note\b",
    r"\bwhich script\b",
    r"\bwhich code file\b",
    r"\bwhere is it\b",
    r"\bwhere is the file\b",
    r"\bwhere is the document\b",
    r"\bfile contains\b",
    r"\bdocument contains\b",
    r"\bfile has\b",
]


def tokenize_query(query):
    return set(
        re.findall(
            r"\b\w+\b",
            query.lower(),
        )
    )


def detect_query_mode(query):
    normalized_query = (
        query
        .lower()
        .strip()
    )

    for pattern in FILE_LOOKUP_PATTERNS:
        if re.search(
            pattern,
            normalized_query,
        ):
            return "FILE_LOOKUP"

    return "CONTENT_SEARCH"


def detect_section_intent(query):
    query_tokens = tokenize_query(
        query
    )

    strong_scores = {}

    for section, keywords in (
        STRONG_SECTION_INTENTS.items()
    ):
        matches = (
            query_tokens
            & keywords
        )

        strong_scores[section] = len(
            matches
        )

    best_strong_score = max(
        strong_scores.values(),
        default=0,
    )

    if best_strong_score > 0:
        best_sections = [
            section
            for section, score
            in strong_scores.items()
            if score == best_strong_score
        ]

        if len(best_sections) == 1:
            return best_sections[0]

        return None

    weak_scores = {}

    for section, keywords in (
        WEAK_SECTION_INTENTS.items()
    ):
        matches = (
            query_tokens
            & keywords
        )

        weak_scores[section] = len(
            matches
        )

    best_weak_score = max(
        weak_scores.values(),
        default=0,
    )

    if best_weak_score == 0:
        return None

    best_sections = [
        section
        for section, score
        in weak_scores.items()
        if score == best_weak_score
    ]

    if len(best_sections) != 1:
        return None

    return best_sections[0]