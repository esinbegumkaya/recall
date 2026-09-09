from typing import List, Dict
import re


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
- a program explicitly named "Yapay Zeka ProgramÄ±"

EXCLUDE:
- a generic data visualization course
- a generic data analytics course
- a generic Python certificate
- a cloud course with no explicit AI content

Do NOT abstain if at least one item clearly satisfies the requested
category.


EVIDENCE TERMINOLOGY

Preserve the terminology used in the validated evidence whenever possible.

- Prefer wording directly supported by the evidence.
- Do not replace a specific validated phrase with a broader or different
  paraphrase if that changes the factual wording.
- For example, if the evidence says "agentic AI workflows", prefer
  "worked on agentic AI workflows" or "implemented agentic AI workflows"
  rather than rewriting it as "worked with AI agents".
- Parent context may be used for attribution such as organization, role,
  project, or location, but the factual action itself should remain close
  to the evidence wording.

COMPOUND QUESTIONS

A question may contain multiple independent subquestions.

When validated evidence explicitly states which subquestion it supports:

- evaluate each subquestion independently;
- use only evidence mapped to that subquestion;
- use Parent context when the question asks where, at which company,
  under which role, or in which project the evidence belongs;
- answer all supported subquestions in the same response;
- cite each answer with the source that supports it;
- do not abstain merely because the evidence for different subquestions
  comes from different sources.

If every requested subquestion has validated supporting evidence, answer
the question.

If a requested subquestion has no validated supporting evidence, do not
invent an answer for that subquestion.


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



def build_evidence_context(
    evidence_units: List[Dict],
) -> str:
    context_blocks = []

    for source_number, unit in enumerate(
        evidence_units,
        start=1,
    ):
        source_number = unit.get(
            "_source_number",
            source_number,
        )

        source_label = build_source_label(
            source_number,
            unit,
        )

        parent_text = unit.get(
            "parent_text"
        )

        evidence_text = unit.get(
            "text",
            "",
        )

        details = []

        supporting_subquery = unit.get(
            "supporting_subquery"
        )

        if supporting_subquery:
            details.append(
                f"Supports subquestion: "
                f"{supporting_subquery}"
            )

        if parent_text:
            details.append(
                f"Parent context: {parent_text}"
            )

        details.append(
            f"Evidence: {evidence_text}"
        )

        block = (
            f"[Source {source_number}]\n"
            f"{source_label}\n\n"
            + "\n".join(details)
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




def extract_citation_numbers(
    answer: str,
) -> List[int]:
    return [
        int(number)
        for number in re.findall(
            r"\[Source (\d+)\]",
            answer,
        )
    ]


def extract_cited_claims(
    answer: str,
):
    """
    Extract complete claims and all citations attached
    to each claim.

    Citation blocks are treated as anchors rather than
    trying to infer the sentence with overlapping regexes.

    Supported:
    - sentence. [Source 1]
    - sentence [Source 1].
    - sentence. [Source 2] [Source 3]
    - sentence [Source 2] [Source 3].
    """
    claims = []

    compact_answer = re.sub(
        r"\s+",
        " ",
        answer.strip(),
    )

    citation_block_pattern = re.compile(
        r"(?:\[Source\s+\d+\]\s*)+"
    )

    previous_end = 0

    for match in citation_block_pattern.finditer(
        compact_answer
    ):
        segment = (
            compact_answer[
                previous_end:match.start()
            ]
            .strip()
        )

        # A citation may appear before the final
        # sentence punctuation:
        #
        # "claim [Source 1]."
        #
        # Remove punctuation left behind by the
        # previous citation before processing the
        # next segment.
        segment = segment.lstrip(
            ".!? "
        )

        if not segment:
            previous_end = match.end()
            continue

        # Find the start of the current factual
        # sentence. Ignore a final punctuation mark
        # belonging to this same sentence.
        search_text = segment

        if (
            search_text
            and search_text[-1] in ".!?"
        ):
            boundary_search = (
                search_text[:-1]
            )
        else:
            boundary_search = (
                search_text
            )

        boundary_positions = [
            boundary_search.rfind("."),
            boundary_search.rfind("!"),
            boundary_search.rfind("?"),
        ]

        sentence_boundary = max(
            boundary_positions
        )

        if sentence_boundary >= 0:
            claim_text = (
                segment[
                    sentence_boundary + 1:
                ]
                .strip()
            )
        else:
            claim_text = segment.strip()

        # If citation comes before punctuation,
        # attach that punctuation to the claim.
        after_citation = (
            compact_answer[
                match.end():
            ]
        )

        punctuation_match = re.match(
            r"\s*([.!?])",
            after_citation,
        )

        if (
            claim_text
            and claim_text[-1] not in ".!?"
            and punctuation_match
        ):
            claim_text += (
                punctuation_match.group(1)
            )

        source_numbers = [
            int(number)
            for number in re.findall(
                r"\[Source\s+(\d+)\]",
                match.group(0),
            )
        ]

        for source_number in source_numbers:
            claims.append(
                {
                    "claim": claim_text,
                    "source_number": source_number,
                }
            )

        previous_end = match.end()

        # Consume punctuation following citations so
        # it does not contaminate the next claim.
        trailing = re.match(
            r"\s*[.!?]",
            compact_answer[
                previous_end:
            ],
        )

        if trailing:
            previous_end += (
                trailing.end()
            )

    return claims


def normalize_claim_for_nli(
    claim: str,
) -> str:
    normalized = claim.strip()

    normalized = re.sub(
        r"^(yes|no)\s*,\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"^(yes|no)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = normalized.strip()

    if normalized and not normalized.endswith("."):
        normalized += "."

    return normalized



def strip_provenance_for_nli(
    claim: str,
) -> str:
    """
    Remove explicit provenance attribution after it has
    already been validated deterministically.

    Examples:
    - "The person worked with AI agents at Microsoft."
      -> "The person worked with AI agents."
    - "The person built the system for Company X."
      -> "The person built the system."

    Only explicit attribution markers handled by
    provenance_supports_claim are stripped.
    Other semantic uses such as "with Watson",
    "during projects", or "in Python" are preserved.
    """
    normalized = claim.strip()

    trailing_punctuation = ""

    if (
        normalized
        and normalized[-1] in ".!?"
    ):
        trailing_punctuation = normalized[-1]
        normalized = normalized[:-1].rstrip()

    provenance_pattern = re.compile(
        r"\s+(?:at|for)\s+.+$",
        flags=re.IGNORECASE,
    )

    stripped = provenance_pattern.sub(
        "",
        normalized,
    ).strip()

    if not stripped:
        stripped = normalized

    if (
        stripped
        and stripped[-1] not in ".!?"
    ):
        stripped += (
            trailing_punctuation
            or "."
        )

    return stripped


def build_source_evidence_map(
    evidence_units,
):
    """
    Map each citation source number to exactly one
    validated evidence unit.

    Source identity is intentionally unit-level rather
    than only file/page/section-level. Multiple evidence
    units can originate from the same chunk and still
    represent different claims/provenance.
    """
    source_map = {}

    for index, unit in enumerate(
        evidence_units,
        start=1,
    ):
        source_number = unit.get(
            "_source_number",
            index,
        )

        source_map[source_number] = [
            unit
        ]

    return source_map


def extract_source_unit(
    source_units,
):
    if not source_units:
        return None

    if len(source_units) == 1:
        return source_units[0]

    return None


def normalize_provenance_text(
    text,
):
    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip().lower()


def provenance_supports_claim(
    claim,
    evidence_unit,
):
    """
    Conservative deterministic provenance validation.

    Only explicit attribution constructions such as
    "at <organization>" or "for <organization>" trigger
    deterministic parent-context validation.

    General semantic phrases such as:
    - "with Watson"
    - "during financial planning projects"
    - "in Python"
    are left to NLI rather than being misclassified
    as organization provenance.
    """
    if not evidence_unit:
        return False

    parent_text = normalize_provenance_text(
        evidence_unit.get(
            "parent_text",
            "",
        )
    )

    if not parent_text:
        return True

    normalized_claim = normalize_provenance_text(
        claim
    )

    provenance_markers = (
        " at ",
        " for ",
    )

    claim_context = ""

    for marker in provenance_markers:
        if marker in normalized_claim:
            claim_context = (
                normalized_claim.split(
                    marker,
                    1,
                )[1]
            )
            break

    if not claim_context:
        return True

    parent_tokens = set(
        re.findall(
            r"\b[\w+#.-]+\b",
            parent_text,
        )
    )

    context_tokens = set(
        re.findall(
            r"\b[\w+#.-]+\b",
            claim_context,
        )
    )

    ignored_tokens = {
        "the",
        "person",
        "this",
        "company",
        "organization",
        "programme",
        "program",
        "role",
        "work",
        "worked",
        "working",
        "project",
        "projects",
        "team",
    }

    attribution_tokens = [
        token
        for token in context_tokens
        if (
            len(token) >= 3
            and token not in ignored_tokens
        )
    ]

    if not attribution_tokens:
        return True

    return any(
        token in parent_tokens
        for token in attribution_tokens
    )

def validate_generated_answer(
    answer: str,
    evidence_units: List[Dict],
    claim_judge=None,
) -> bool:
    if not answer:
        return False

    normalized = answer.strip()

    if normalized == ABSTENTION_MESSAGE:
        return True

    citation_numbers = (
        extract_citation_numbers(
            normalized
        )
    )

    if not citation_numbers:
        return False

    source_map = (
        build_source_evidence_map(
            evidence_units
        )
    )

    if not source_map:
        return False

    if not all(
        number in source_map
        for number in citation_numbers
    ):
        return False

    # Backward-compatible behavior:
    # citation validation alone remains sufficient
    # when no claim judge is supplied.
    if claim_judge is None:
        return True

    cited_claims = (
        extract_cited_claims(
            normalized
        )
    )

    if not cited_claims:
        return False

    for item in cited_claims:
        source_number = (
            item["source_number"]
        )

        claim = normalize_claim_for_nli(
            item["claim"]
        )

        source_units = source_map.get(
            source_number,
            [],
        )

        source_unit = extract_source_unit(
            source_units
        )

        if source_unit is None:
            return False

        # First perform deterministic provenance
        # validation. This prevents NLI from accepting
        # semantically plausible but incorrectly attributed
        # organizations.
        if not provenance_supports_claim(
            claim,
            source_unit,
        ):
            return False

        evidence_text = (
            source_unit.get(
                "text",
                "",
            )
        )

        parent_text = (
            source_unit.get(
                "parent_text",
                "",
            )
        )

        if parent_text:
            evidence_for_judge = (
                f"Parent context: "
                f"{parent_text}\n"
                f"Evidence: "
                f"{evidence_text}"
            )
        else:
            evidence_for_judge = (
                evidence_text
            )

        semantic_claim = (
            strip_provenance_for_nli(
                claim
            )
        )

        result = (
            claim_judge.judge_claim(
                semantic_claim,
                evidence_for_judge,
            )
        )

        score = result.get(
            "entailment_score",
            0.0,
        )

        if score >= 0.90:
            continue

        # The generated wording may be a faithful
        # restatement of the semantic hypothesis that
        # this evidence already passed upstream, even
        # when direct evidence-to-generated-claim NLI
        # is sensitive to paraphrasing.
        #
        # Provenance has already been validated above,
        # so this fallback checks semantic consistency
        # only. The same strict threshold is retained.
        supporting_hypothesis = (
            source_unit.get(
                "supporting_hypothesis"
            )
        )

        if not supporting_hypothesis:
            return False

        contract_result = (
            claim_judge.judge_claim(
                semantic_claim,
                supporting_hypothesis,
            )
        )

        contract_score = (
            contract_result.get(
                "entailment_score",
                0.0,
            )
        )

        if contract_score < 0.90:
            return False

    return True


def build_evidence_user_prompt(
    query: str,
    evidence_units: List[Dict],
) -> str:
    context = build_evidence_context(
        evidence_units
    )

    return f"""
QUESTION

{query}


VALIDATED EVIDENCE

{context}

TASK

Answer the question using only the validated evidence above.

Write atomic factual claims.

Each citation must support every factual detail in the claim immediately
associated with it.

If different sources support different details, split those details into
separate sentences or clauses and cite each one separately.

Do not place multiple citations after a combined claim unless every cited
source independently supports the entire claim.

Preserve the terminology used in the evidence whenever possible.

Parent context identifies where or under which role the evidence belongs.
Use it when needed to answer questions such as where the work was performed.

Cite every factual claim with the corresponding [Source N].

If the validated evidence does not support an answer, use the exact
abstention sentence.
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

def generate_validated_subanswer(
    chat_client,
    subquery: str,
    evidence_units: List[Dict],
    claim_judge=None,
) -> str:
    if not evidence_units:
        return ABSTENTION_MESSAGE

    user_prompt = build_evidence_user_prompt(
        subquery,
        evidence_units,
    )

    if len(evidence_units) == 1:
        user_prompt += """

SINGLE-SOURCE PRECISION RULES

You are answering from exactly one validated evidence unit.

Do not restate the full wording of the question as a factual claim.

Include only facts explicitly stated in this evidence unit or its Parent context.

If the question asks about multiple concepts but this evidence supports only
one of them, answer only the supported concept.

Use wording as close as possible to the Evidence text.

Do not introduce broader synonyms or qualifiers such as "professionally"
unless they are explicitly supported by the evidence.

The Parent context may identify where the evidence belongs.

Cite only this source.
"""

    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 180

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

    content = content.strip()

    print(
        "\nSUBQUERY:"
    )
    print(
        subquery
    )

    print(
        "\nSUBQUERY CANDIDATE:"
    )
    print(
        content
    )

    if content == ABSTENTION_MESSAGE:
        print(
            "Subquery result: MODEL ABSTAINED"
        )
        return ABSTENTION_MESSAGE

    validation_result = (
        validate_generated_answer(
            content,
            evidence_units,
            claim_judge=claim_judge,
        )
    )

    print(
        f"Subquery validation: "
        f"{'PASS' if validation_result else 'FAIL'}"
    )

    if not validation_result:
        return ABSTENTION_MESSAGE

    return content


def generate_grounded_answer_from_evidence(
    chat_client,
    query: str,
    evidence_units: List[Dict],
    claim_judge=None,
) -> str:
    if not evidence_units:
        return ABSTENTION_MESSAGE

    numbered_evidence = []

    for source_number, unit in enumerate(
        evidence_units,
        start=1,
    ):
        numbered_unit = dict(unit)

        numbered_unit[
            "_source_number"
        ] = source_number

        numbered_evidence.append(
            numbered_unit
        )

    subqueries = []

    for unit in numbered_evidence:
        subquery = unit.get(
            "supporting_subquery"
        )

        if (
            subquery
            and subquery not in subqueries
        ):
            subqueries.append(
                subquery
            )

    if len(subqueries) > 1:
        subanswers = []

        for subquery in subqueries:
            subquery_evidence = [
                unit
                for unit in numbered_evidence
                if unit.get(
                    "supporting_subquery"
                ) == subquery
            ]

            subanswer = (
                generate_validated_subanswer(
                    chat_client,
                    subquery,
                    subquery_evidence,
                    claim_judge=claim_judge,
                )
            )

            if (
                subanswer
                != ABSTENTION_MESSAGE
            ):
                subanswers.append(
                    subanswer
                )

        if not subanswers:
            return ABSTENTION_MESSAGE

        combined_answer = "\n".join(
            subanswers
        )

        print(
            "\nGenerated compound candidate answer:"
        )
        print(
            combined_answer
        )

        validation_result = (
            validate_generated_answer(
                combined_answer,
                numbered_evidence,
                claim_judge=claim_judge,
            )
        )

        print(
            f"\nCitation/claim validation: "
            f"{'PASS' if validation_result else 'FAIL'}"
        )

        if not validation_result:
            return ABSTENTION_MESSAGE

        return combined_answer

    # Single subquery with multiple evidence units:
    # generate and validate one atomic answer per source.
    if (
        len(subqueries) == 1
        and len(numbered_evidence) > 1
    ):
        atomic_answers = []

        subquery = subqueries[0]

        for unit in numbered_evidence:
            atomic_answer = (
                generate_validated_subanswer(
                    chat_client,
                    subquery,
                    [unit],
                    claim_judge=claim_judge,
                )
            )

            if (
                atomic_answer
                != ABSTENTION_MESSAGE
            ):
                atomic_answers.append(
                    atomic_answer
                )

        if not atomic_answers:
            return ABSTENTION_MESSAGE

        combined_answer = "\n".join(
            atomic_answers
        )

        print(
            "\nGenerated atomic candidate answer:"
        )
        print(
            combined_answer
        )

        validation_result = (
            validate_generated_answer(
                combined_answer,
                numbered_evidence,
                claim_judge=claim_judge,
            )
        )

        print(
            f"\nCitation/claim validation: "
            f"{'PASS' if validation_result else 'FAIL'}"
        )

        if not validation_result:
            return ABSTENTION_MESSAGE

        return combined_answer
    evidence_units = numbered_evidence

    user_prompt = build_evidence_user_prompt(
        query,
        evidence_units,
    )

    print(
        "\nGENERATOR USER PROMPT:"
    )
    print(
        "=" * 70
    )
    print(
        user_prompt
    )
    print(
        "=" * 70
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

    content = content.strip()

    print(
        "\nGenerated candidate answer:"
    )
    print(content)

    validation_result = validate_generated_answer(
        content,
        evidence_units,
        claim_judge=claim_judge,
    )

    print(
        f"\nCitation/claim validation: "
        f"{'PASS' if validation_result else 'FAIL'}"
    )

    if not validation_result:
        return ABSTENTION_MESSAGE

    return content


