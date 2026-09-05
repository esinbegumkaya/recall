from typing import List, Dict


ABSTENTION_MESSAGE = (
    "The available files do not provide enough evidence "
    "to answer this question."
)


SYSTEM_PROMPT = """
You are Recall, a private local retrieval-grounded assistant.

Answer using ONLY the retrieved evidence.

CORE RULES

1. Every factual statement must be supported by the retrieved evidence.

2. Answer only the user's exact question.

3. Do not add related facts that were not requested.

4. Never invent information or qualifiers.

5. Cite every included factual item with [Source N].

6. If no retrieved evidence directly supports an answer, output exactly:

The available files do not provide enough evidence to answer this question.


CATEGORY QUESTIONS

When the user asks for items belonging to a category, evaluate each item
independently.

Include an item when either its NAME or its DESCRIPTION explicitly shows
that it belongs to the requested category.

For AI training, explicit evidence includes terms or content such as:

- AI
- Artificial Intelligence
- Machine Learning
- LLM
- Large Language Model
- RAG
- AI agents
- agentic AI
- deep learning
- NLP
- computer vision
- explicitly AI-focused tools or coursework

Examples:

INCLUDE:
- "Microsoft AI Innovators Program"
- a program explicitly covering AI agents and RAG
- "HCCDA-AI"
- a course explicitly teaching Machine Learning
- a program explicitly named "Yapay Zeka Programı"

EXCLUDE:
- a generic data visualization course
- a generic data analytics course
- a generic Python certificate
- a cloud course with no explicit AI content

Do NOT abstain if at least one item clearly satisfies the requested
category.


PRECISION RULE

For each candidate item, ask:

"Does the retrieved text explicitly connect this item to what the user
asked for?"

If yes, include it.

If no, exclude it.

Do not require the exact phrase "formal AI training" to appear in the
source. A program or certificate whose title or description explicitly
contains AI-related training is sufficient evidence.


OUTPUT RULES

For a list question:
- return only qualifying items;
- use one bullet per item;
- cite the source immediately after that item;
- do not add unrelated items;
- do not add a concluding paragraph unless necessary.

For an unsupported question:
- return only the exact abstention sentence.
"""


def build_source_label(
    source_number: int,
    result: Dict,
) -> str:
    parts = [
        f"Source {source_number}",
        f"File: {result['file_path']}",
    ]

    if result.get("page_number") is not None:
        parts.append(
            f"Page: {result['page_number']}"
        )

    if result.get("section_name"):
        parts.append(
            f"Section: {result['section_name']}"
        )

    parts.append(
        f"Chunk: {result['chunk_index']}"
    )

    return "\n".join(parts)


def build_context(
    results: List[Dict],
) -> str:
    context_blocks = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        source_label = build_source_label(
            index,
            result,
        )

        block = (
            f"[Source {index}]\n"
            f"{source_label}\n\n"
            f"{result['chunk_text']}"
        )

        context_blocks.append(
            block
        )

    return "\n\n---\n\n".join(
        context_blocks
    )


def build_user_prompt(
    query: str,
    results: List[Dict],
) -> str:
    context = build_context(
        results
    )

    return f"""
QUESTION

{query}


RETRIEVED EVIDENCE

{context}


TASK

Answer the question using only the retrieved evidence.

For every candidate item:

1. Check whether its title or description explicitly supports its
   membership in the category requested by the user.

2. Include it only if that evidence exists.

3. Exclude merely related items.

4. Cite the supporting source immediately after the item.

If at least one qualifying item exists, answer with those items.

If none exists, use the exact abstention sentence.
""".strip()


def generate_grounded_answer(
    chat_client,
    query: str,
    results: List[Dict],
) -> str:
    if not results:
        return ABSTENTION_MESSAGE

    user_prompt = build_user_prompt(
        query,
        results,
    )

    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 250

    response = chat_client.complete_chat(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        return ABSTENTION_MESSAGE

    return content.strip()