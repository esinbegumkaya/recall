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



def _extract_explicit_provenance_suffix(
    claim: str,
):
    """
    Return an explicit organization-like ``at/for ...`` suffix.

    The generator can legitimately use semantic phrases such as
    ``for real-world use cases``. Those must not be mistaken for
    organization attribution. We therefore treat a suffix as provenance
    only when the text after ``at`` or ``for`` looks entity-like, e.g.
    ``at Microsoft`` or ``for Cubewise``.
    """
    text = (claim or "").strip()

    match = re.search(
        r"\s+(at|for)\s+([^.!?]+?)(?:[.!?])?$",
        text,
    )

    if not match:
        return None

    candidate = match.group(2).strip()

    if not candidate:
        return None

    raw_tokens = re.findall(
        r"[A-Za-z0-9&+#.\-]+",
        candidate,
    )

    if not raw_tokens:
        return None

    # Organization names produced from retrieved evidence normally retain
    # proper-noun/acronym casing. Lower-case semantic complements such as
    # "real-world use cases" therefore remain part of the factual claim.
    looks_entity_like = any(
        (
            token[:1].isupper()
            or (len(token) >= 2 and token.isupper())
        )
        for token in raw_tokens
    )

    if not looks_entity_like:
        return None

    return {
        "marker": match.group(1).casefold(),
        "candidate": candidate,
        "start": match.start(),
        "end": match.end(),
    }


def strip_provenance_for_nli(
    claim: str,
) -> str:
    """
    Remove explicit organization attribution after deterministic provenance
    validation, without stripping ordinary semantic ``for`` phrases.

    Examples:
    - "The person worked with AI agents at Microsoft."
      -> "The person worked with AI agents."
    - "The person built the system for Company X."
      -> "The person built the system."
    - "Built RAG pipelines for real-world use cases."
      -> unchanged
    """
    normalized = (claim or "").strip()

    provenance = _extract_explicit_provenance_suffix(
        normalized
    )

    if provenance is None:
        if normalized and normalized[-1] not in ".!?":
            normalized += "."
        return normalized

    stripped = normalized[
        :provenance["start"]
    ].rstrip()

    if not stripped:
        stripped = normalized

    if stripped and stripped[-1] not in ".!?":
        stripped += "."

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
    Conservative deterministic organization-attribution validation.

    Only entity-like suffixes such as ``at Microsoft`` or ``for Cubewise``
    are interpreted as provenance. Semantic phrases such as
    ``for real-world use cases`` are not provenance and remain available
    to lexical/NLI validation.
    """
    if not evidence_unit:
        return False

    provenance = _extract_explicit_provenance_suffix(
        claim
    )

    if provenance is None:
        return True

    parent_text = normalize_provenance_text(
        evidence_unit.get(
            "parent_text",
            "",
        )
    )

    # If there is no parent context, leave semantic consistency to the
    # lexical/NLI validator rather than inventing a provenance relationship.
    if not parent_text:
        return True

    claim_context = normalize_provenance_text(
        provenance["candidate"]
    )

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


def _normalize_support_token(token: str) -> str:
    token = token.casefold().strip()

    irregular = {
        "built": "build",
        "building": "build",
        "developed": "develop",
        "developing": "develop",
        "implemented": "implement",
        "implementing": "implement",
        "worked": "work",
        "working": "work",
        "used": "use",
        "using": "use",
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


def _support_tokens(text: str) -> set:
    """Return factual content tokens for conservative lexical support."""
    stopwords = {
        "a", "an", "the", "this", "that", "these", "those",
        "person", "has", "have", "had", "is", "are", "was", "were",
        "with", "of", "to", "in", "on", "at", "for", "from", "and",
        "or", "as", "by", "their", "his", "her", "its", "experience",
        "experienced",
    }

    raw_tokens = re.findall(
        r"[A-Za-z0-9+#.\-]+",
        text or "",
    )

    tokens = set()

    for raw_token in raw_tokens:
        token = _normalize_support_token(raw_token)

        if not token or token in stopwords:
            continue

        tokens.add(token)

    return tokens


def claim_has_strong_lexical_support(
    claim: str,
    evidence_text: str,
) -> bool:
    """
    Accept near-extractive restatements before semantic NLI validation.

    This is intentionally conservative: the claim must preserve nearly all
    of its factual content tokens in the validated evidence. It mainly
    prevents false negatives when the generator changes grammar such as
    "develop" -> "developed" or "build" -> "built".
    """
    claim_tokens = _support_tokens(claim)
    evidence_tokens = _support_tokens(evidence_text)

    if len(claim_tokens) < 3 or not evidence_tokens:
        return False

    # Never allow a new number/year that does not occur in the evidence.
    claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim or ""))
    evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text or ""))

    if not claim_numbers.issubset(evidence_numbers):
        return False

    supported = claim_tokens & evidence_tokens
    coverage = len(supported) / len(claim_tokens)

    return coverage >= 0.85



def _lexical_support_coverage(
    claim: str,
    evidence_text: str,
) -> float:
    """
    Return factual-token coverage of a claim by evidence.

    Used only to choose among already validated evidence units when
    repairing a missing citation. A citation is never added unless the
    conservative near-extractive support rule also passes.
    """
    claim_tokens = _support_tokens(claim)
    evidence_tokens = _support_tokens(evidence_text)

    if not claim_tokens or not evidence_tokens:
        return 0.0

    supported = claim_tokens & evidence_tokens
    return len(supported) / len(claim_tokens)


def repair_missing_citations(
    answer: str,
    evidence_units: List[Dict],
) -> str:
    """
    Conservatively attach citations omitted by the local chat model.

    The small local model can occasionally follow the factual constraints
    but omit ``[Source N]`` even when the prompt explicitly requests them.
    We repair only near-extractive claims that map unambiguously to one
    validated evidence unit.

    Safety rules:
    - Existing citations are never changed.
    - Direct evidence text is preferred over parent context.
    - A source is eligible only if the same conservative lexical-support
      rule used by the validator passes.
    - If no unique supported source can be identified, the claim is left
      uncited. Final validation will then reject the answer and the engine
      will use its deterministic fallback.
    """
    if not answer:
        return answer

    normalized = answer.strip()

    if normalized == ABSTENTION_MESSAGE:
        return normalized

    repaired_lines = []

    for raw_line in normalized.splitlines():
        line = raw_line.strip()

        if not line:
            repaired_lines.append("")
            continue

        # Preserve already cited lines exactly.
        if re.search(
            r"\[Source\s+\d+\]",
            line,
            flags=re.IGNORECASE,
        ):
            repaired_lines.append(line)
            continue

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            line,
        )

        repaired_parts = []

        for part in parts:
            claim = part.strip()

            if not claim:
                continue

            if re.search(
                r"\[Source\s+\d+\]",
                claim,
                flags=re.IGNORECASE,
            ):
                repaired_parts.append(claim)
                continue

            semantic_claim = normalize_claim_for_nli(
                claim
            )

            direct_matches = []

            for index, unit in enumerate(
                evidence_units,
                start=1,
            ):
                source_number = unit.get(
                    "_source_number",
                    index,
                )

                if not provenance_supports_claim(
                    semantic_claim,
                    unit,
                ):
                    continue

                evidence_text = unit.get(
                    "text",
                    "",
                )

                if not claim_has_strong_lexical_support(
                    semantic_claim,
                    evidence_text,
                ):
                    continue

                direct_matches.append(
                    (
                        _lexical_support_coverage(
                            semantic_claim,
                            evidence_text,
                        ),
                        source_number,
                    )
                )

            chosen_source = None

            if direct_matches:
                direct_matches.sort(
                    reverse=True
                )

                best_score = direct_matches[0][0]

                best_sources = [
                    source_number
                    for score, source_number
                    in direct_matches
                    if abs(score - best_score) < 1e-9
                ]

                if len(best_sources) == 1:
                    chosen_source = best_sources[0]

            # If the direct evidence itself is not enough, parent context
            # may legitimately establish role/organization provenance.
            if chosen_source is None:
                contextual_matches = []

                for index, unit in enumerate(
                    evidence_units,
                    start=1,
                ):
                    source_number = unit.get(
                        "_source_number",
                        index,
                    )

                    if not provenance_supports_claim(
                        semantic_claim,
                        unit,
                    ):
                        continue

                    evidence_text = unit.get(
                        "text",
                        "",
                    )

                    parent_text = unit.get(
                        "parent_text",
                        "",
                    )

                    if parent_text:
                        context = (
                            f"{parent_text}\n"
                            f"{evidence_text}"
                        )
                    else:
                        context = evidence_text

                    if not claim_has_strong_lexical_support(
                        semantic_claim,
                        context,
                    ):
                        continue

                    contextual_matches.append(
                        (
                            _lexical_support_coverage(
                                semantic_claim,
                                context,
                            ),
                            source_number,
                        )
                    )

                if contextual_matches:
                    contextual_matches.sort(
                        reverse=True
                    )

                    best_score = (
                        contextual_matches[0][0]
                    )

                    best_sources = [
                        source_number
                        for score, source_number
                        in contextual_matches
                        if abs(
                            score - best_score
                        ) < 1e-9
                    ]

                    if len(best_sources) == 1:
                        chosen_source = (
                            best_sources[0]
                        )

            if chosen_source is None:
                repaired_parts.append(
                    claim
                )
            else:
                repaired_parts.append(
                    f"{claim} "
                    f"[Source {chosen_source}]"
                )

        repaired_lines.append(
            " ".join(repaired_parts)
        )

    return "\n".join(
        repaired_lines
    ).strip()


def remove_standalone_citation_lines(
    answer: str,
) -> str:
    """
    Remove model artifacts that consist only of citation labels.

    Example:
        [Source 1]
        [Source 2]

    A citation must be attached to a factual claim; a citation-only line
    contributes no answer content and can confuse later validation.
    """
    cleaned_lines = []

    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()

        if (
            line
            and re.fullmatch(
                r"(?:\[Source\s+\d+\]\s*)+",
                line,
                flags=re.IGNORECASE,
            )
        ):
            continue

        cleaned_lines.append(raw_line.rstrip())

    return "\n".join(cleaned_lines).strip()


def deduplicate_generated_lines(
    answer: str,
) -> str:
    """
    Remove exact repeated non-empty answer lines while preserving order.

    Small local models occasionally repeat a complete generated paragraph.
    This post-processing is deliberately conservative: only exact normalized
    duplicates are removed.
    """
    if not answer:
        return answer

    seen = set()
    output = []

    for raw_line in answer.splitlines():
        line = raw_line.strip()

        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        key = re.sub(
            r"\s+",
            " ",
            line,
        ).casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(line)

    while output and output[-1] == "":
        output.pop()

    return "\n".join(output).strip()


def _meaningful_claim_count(
    line: str,
) -> int:
    """
    Estimate the number of factual sentence-like claims on one output line.

    This is used only for citation completeness. It intentionally ignores
    citation-only material and short heading-like labels.
    """
    without_citations = re.sub(
        r"\[Source\s+\d+\]",
        "",
        line or "",
        flags=re.IGNORECASE,
    ).strip()

    without_citations = re.sub(
        r"^\s*(?:[-*•]|\d+[.)])\s*",
        "",
        without_citations,
    ).strip()

    if not without_citations:
        return 0

    # Allow short heading labels such as "Experience:".
    if (
        without_citations.endswith(":")
        and len(
            re.findall(
                r"\b[\w+#.-]+\b",
                without_citations,
            )
        ) <= 8
    ):
        return 0

    parts = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            without_citations,
        )
        if part.strip()
    ]

    count = 0

    for part in parts:
        words = re.findall(
            r"\b[\w+#.-]+\b",
            part,
        )

        if len(words) >= 3:
            count += 1

    return count


def all_factual_claims_have_citations(
    answer: str,
) -> bool:
    """
    Require citation coverage for every factual sentence-like claim.

    The older validator verified only claims that already had citations.
    Uncited factual sentences were therefore invisible to validation. This
    closes that gap by comparing meaningful claim count with citation count
    line by line.
    """
    if not answer:
        return False

    normalized = answer.strip()

    if normalized == ABSTENTION_MESSAGE:
        return True

    for raw_line in normalized.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        meaningful_claims = _meaningful_claim_count(
            line
        )

        if meaningful_claims == 0:
            continue

        citations = len(
            re.findall(
                r"\[Source\s+\d+\]",
                line,
                flags=re.IGNORECASE,
            )
        )

        if citations < meaningful_claims:
            return False

    return True


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

    if not all_factual_claims_have_citations(
        normalized
    ):
        return False

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

        # Prefer an exact/near-extractive support check before NLI.
        # Local NLI can be over-sensitive to harmless grammatical changes
        # such as "develop" -> "developed" or "build" -> "built".
        # This lexical path remains conservative and uses only the already
        # validated source unit (plus its validated parent context).
        if claim_has_strong_lexical_support(
            semantic_claim,
            evidence_for_judge,
        ):
            continue

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


# ---------------------------------------------------------------------------
# Chat-template turn leakage.
#
# Some locally-loaded models (notably reasoning/chat models using a ChatML-
# style template, identifiable by "<think>...</think>" blocks) don't stop
# cleanly at the end of their own turn -- the small quantized model here
# went on to hallucinate an entire fake continued conversation ("user\n{the
# same question}\nassistant\n<think>...") and repeated THAT instead of a
# single sentence. This happened even after adding frequency_penalty /
# presence_penalty, which fixed the earlier verbatim-sentence loop but
# doesn't stop a model from inventing structurally-different-but-still-
# repetitive continuations.
#
# chat_client.settings has no `stop` sequence parameter in this SDK version
# (confirmed via dir(chat_client.settings)), so we can't ask the runtime to
# stop generation early. The next best thing: truncate the raw output
# ourselves at the first sign the model has left its own answer and started
# simulating a new turn, before any citation extraction/repair/validation
# runs on it. This can only ever shorten a malformed response -- it never
# touches a normal, well-behaved answer, since none of these markers appear
# in a real one-turn factual answer.
# ---------------------------------------------------------------------------

_TURN_LEAKAGE_RE = re.compile(
    r"""
    \n\s*(?:user|assistant|system)\s*\n   # a bare role line on its own,
                                          # e.g. "\nuser\n" or "\nassistant\n"
    |
    <\|im_(?:start|end)\|>               # ChatML special tokens, if the
                                          # runtime ever leaks them literally
    """,
    re.IGNORECASE | re.VERBOSE,
)


def truncate_at_turn_leakage(content: str) -> str:
    if not content:
        return content

    match = _TURN_LEAKAGE_RE.search(content)

    if not match:
        return content

    return content[: match.start()].strip()


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

    # frequency_penalty / presence_penalty: without these, this small
    # local model (temperature=0.0, i.e. greedy decoding) can get stuck
    # repeating the same sentence verbatim until max_tokens is exhausted --
    # observed directly in production (the same factual sentence repeated
    # ~13 times for a single-fact answer). frequency_penalty discourages
    # reusing tokens in proportion to how often they've already appeared;
    # presence_penalty discourages reusing ANY token that has appeared at
    # all, which is the more direct lever against verbatim-loop repetition
    # specifically. Values are conservative starting points (max range is
    # -2.0 to 2.0) -- raise them further if repetition still appears, but
    # avoid setting them so high that the answer degrades into rambling
    # padding just to avoid repeating factual terms (source citations like
    # "[Source 1]" and evidence terminology legitimately recur).
    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 250
    chat_client.settings.frequency_penalty = 0.5
    chat_client.settings.presence_penalty = 0.3

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

    return truncate_at_turn_leakage(content.strip())

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

    # See generate_grounded_answer() above for why frequency_penalty /
    # presence_penalty are needed here.
    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 180
    chat_client.settings.frequency_penalty = 0.5
    chat_client.settings.presence_penalty = 0.3

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

    content = truncate_at_turn_leakage(content.strip())

    # A provenance question can have clearly validated parent context while the
    # small local model still chooses to abstain. Treat that as a generation
    # failure, not as a knowledge-base abstention, so RecallEngine can use the
    # deterministic grounded composer. This is intentionally narrow and only
    # applies to explicit provenance questions with validated parent context.
    if (
        content == ABSTENTION_MESSAGE
        and query_requests_provenance(query)
        and has_explicit_provenance_evidence(evidence_units)
    ):
        return ""

    content = repair_missing_citations(
        content,
        evidence_units,
    )

    content = remove_standalone_citation_lines(
        content
    )

    content = deduplicate_generated_lines(
        content
    )

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
        # Validation failure means the model output is unusable; it does NOT mean
        # the validated evidence is absent. Return an empty sentinel so the engine
        # falls back to its deterministic evidence composer instead of reporting a
        # false GENERATOR_ABSTAINED result.
        return ""

    return content




def query_requests_provenance(query: str) -> bool:
    """Return True when the question explicitly asks where/under whom evidence belongs.

    When validated evidence already contains parent/provenance context, a small local
    model may still over-abstain on these questions. In that case the engine should
    use its deterministic grounded composer rather than turning the whole result into
    a false abstention.
    """
    normalized = " ".join((query or "").casefold().split())

    provenance_markers = (
        "where ",
        "where was ",
        "where did ",
        "which company",
        "which organization",
        "which organisation",
        "at which company",
        "at which organization",
        "under which role",
        "which role",
    )

    return any(
        marker in normalized
        for marker in provenance_markers
    )


def has_explicit_provenance_evidence(
    evidence_units: List[Dict],
) -> bool:
    """Return True when at least one validated unit carries useful parent context."""
    for unit in evidence_units:
        parent_text = (
            unit.get("parent_text")
            or ""
        ).strip()

        if parent_text:
            return True

    return False

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

    # Single-subquery questions are generated in one model call,
    # even when several validated evidence units support the answer.
    #
    # The prompt already requires atomic factual claims and per-claim
    # citations. The final validator still checks every cited claim
    # against its exact source, so we keep grounding guarantees while
    # avoiding one expensive chat completion per evidence unit.
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

    # See generate_grounded_answer() above for why frequency_penalty /
    # presence_penalty are needed here.
    chat_client.settings.temperature = 0.0
    chat_client.settings.max_tokens = 250
    chat_client.settings.frequency_penalty = 0.5
    chat_client.settings.presence_penalty = 0.3

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

    content = truncate_at_turn_leakage(content.strip())

    content = repair_missing_citations(
        content,
        evidence_units,
    )

    content = remove_standalone_citation_lines(
        content
    )

    content = deduplicate_generated_lines(
        content
    )

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