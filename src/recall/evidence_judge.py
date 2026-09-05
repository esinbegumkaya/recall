import json
from typing import List, Dict


JUDGE_SYSTEM_PROMPT = """
You are an evidence classification component for a local RAG system.

Your task is NOT to answer the user's question.

Your task is to determine whether one evidence unit directly supports
including that evidence in an answer to the user's question.

Use only the supplied question and evidence.

Return only valid JSON.

Classification rules:

SUPPORTS:
The evidence directly provides information that satisfies the user's
requested fact or category.

DOES_NOT_SUPPORT:
The evidence may be related, similar, or useful background, but it does
not directly satisfy the requested fact or category.

For category questions, semantic similarity is not enough.

Examples:

Question:
What formal AI training has this person completed?

Evidence:
Microsoft AI Innovators Program — AI agents, RAG, local LLMs.

Result:
SUPPORTS

Question:
What formal AI training has this person completed?

Evidence:
Data Analyst Boostcamp — data analysis using Google Cloud.

Result:
DOES_NOT_SUPPORT

Question:
What formal AI training has this person completed?

Evidence:
Python programming certificate.

Result:
DOES_NOT_SUPPORT

Question:
What formal AI training has this person completed?

Evidence:
Yapay Zeka Programı — Machine Learning and Data Science & AI.

Result:
SUPPORTS

Do not infer unsupported category membership.
"""


def build_judge_prompt(
    query: str,
    evidence_text: str,
) -> str:
    return f"""
QUESTION:
{query}

EVIDENCE:
{evidence_text}

Return exactly one JSON object using this schema:

{{
    "label": "SUPPORTS" or "DOES_NOT_SUPPORT"
}}
""".strip()


def parse_judge_response(
    content: str,
) -> str:
    if not content:
        return "DOES_NOT_SUPPORT"

    content = content.strip()

    try:
        parsed = json.loads(
            content
        )

        label = parsed.get(
            "label"
        )

        if label in {
            "SUPPORTS",
            "DOES_NOT_SUPPORT",
        }:
            return label

    except json.JSONDecodeError:
        pass

    upper_content = (
        content.upper()
    )

    if (
        "SUPPORTS"
        in upper_content
        and "DOES_NOT_SUPPORT"
        not in upper_content
    ):
        return "SUPPORTS"

    return "DOES_NOT_SUPPORT"


def judge_evidence_unit(
    chat_client,
    query: str,
    evidence_text: str,
) -> str:
    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 50

    response = (
        chat_client.complete_chat(
            [
                {
                    "role": "system",
                    "content": (
                        JUDGE_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        build_judge_prompt(
                            query,
                            evidence_text,
                        )
                    ),
                },
            ]
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    return parse_judge_response(
        content
    )


def filter_supported_evidence(
    chat_client,
    query: str,
    evidence_units: List[Dict],
) -> List[Dict]:
    supported = []

    for unit in evidence_units:
        label = judge_evidence_unit(
            chat_client=chat_client,
            query=query,
            evidence_text=unit[
                "text"
            ],
        )

        judged_unit = dict(
            unit
        )

        judged_unit[
            "entailment_label"
        ] = label

        if label == "SUPPORTS":
            supported.append(
                judged_unit
            )

    return supported