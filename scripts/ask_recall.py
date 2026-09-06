import re
from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
    )
)


from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from recall.database import (
    initialize_database,
    get_chunks_with_embeddings,
)

from recall.retriever import (
    section_aware_dense_retrieve,
)

from recall.intent import (
    detect_section_intent,
    detect_query_mode,
)

from recall.evidence import (
    deduplicate_overlapping_evidence_units,
    rank_evidence_units,
)

from recall.evidence_gate import (
    filter_by_evidence_type,
    filter_by_topic,
)

from recall.nli_judge import (
    LocalNLIJudge,
)

from recall.generator import (
    generate_grounded_answer_from_evidence,
)


EMBEDDING_MODEL_NAME = (
    "qwen3-embedding-0.6b"
)

CHAT_MODEL_NAME = (
    "qwen3.5-2b"
)

TOP_K = 5

NLI_THRESHOLD = 0.15

SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05


def build_nli_hypothesis(query):
    normalized = query.strip()

    replacements = [
        (
            "Did this person build ",
            "This person built ",
        ),
        (
            "Did this person work with ",
            "This person worked with ",
        ),
        (
            "Did this person work ",
            "This person worked ",
        ),
        (
            "Did this person use ",
            "This person used ",
        ),
        (
            "Did this person develop ",
            "This person developed ",
        ),
        (
            "Did this person create ",
            "This person created ",
        ),
        (
            "Did this person have ",
            "This person had ",
        ),
        (
            "Does this person know ",
            "This person knows ",
        ),
        (
            "Where did this person work with ",
            "This person worked with ",
        ),
        (
            "Where did this person use ",
            "This person used ",
        ),
        (
            "Where did this person build ",
            "This person built ",
        ),
        (
            "Where did this person develop ",
            "This person developed ",
        ),
    ]

    for prefix, replacement in replacements:
        if normalized.lower().startswith(
            prefix.lower()
        ):
            hypothesis = (
                replacement
                + normalized[len(prefix):]
            )

            if hypothesis.endswith("?"):
                hypothesis = hypothesis[:-1]

            return hypothesis.strip() + "."

    if normalized.endswith("?"):
        normalized = normalized[:-1]

    return normalized.strip() + "."



def split_compound_query(query):
    """
    Split only clearly repeated question clauses.

    Examples:
    - "... AI agents, and where did ... IBM Planning Analytics?"
      -> two independent subqueries

    Ordinary conjunctions such as
    "AI agents and RAG systems" remain intact.
    """
    parts = re.split(
        r"\s*,?\s+and\s+"
        r"(?=(?:where|did|does|what|which|how)\b)",
        query.strip(),
        flags=re.IGNORECASE,
    )

    cleaned = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    return cleaned or [query.strip()]

def main():
    initialize_database()

    query = input(
        "Ask Recall: "
    ).strip()

    if not query:
        print(
            "Question cannot be empty."
        )
        return

    rows = list(
        get_chunks_with_embeddings(
            EMBEDDING_MODEL_NAME
        )
    )

    if not rows:
        print(
            "No indexed chunks found."
        )
        return

    print(
        "\nLoading local models..."
    )

    FoundryLocalManager.initialize(
        Configuration(
            app_name="Recall"
        )
    )

    manager = (
        FoundryLocalManager.instance
    )

    embedding_model = (
        manager.catalog.get_model(
            EMBEDDING_MODEL_NAME
        )
    )

    chat_model = (
        manager.catalog.get_model(
            CHAT_MODEL_NAME
        )
    )

    embedding_model.load()

    if not chat_model.is_cached:
        print(
            f"\nDownloading "
            f"{CHAT_MODEL_NAME}..."
        )

        def show_progress(progress):
            print(
                f"\rDownload: "
                f"{progress:.1f}%",
                end="",
                flush=True,
            )

        chat_model.download(
            show_progress
        )

        print(
            "\nDownload complete."
        )

    chat_model.load()

    embedding_client = (
        embedding_model
        .get_embedding_client()
    )

    chat_client = (
        chat_model
        .get_chat_client()
    )

    embedding_response = (
        embedding_client
        .generate_embedding(
            query
        )
    )

    query_embedding = (
        embedding_response
        .data[0]
        .embedding
    )

    results = (
        section_aware_dense_retrieve(
            query=query,
            query_embedding=query_embedding,
            rows=rows,
            detect_section_intent=(
                detect_section_intent
            ),
            detect_query_mode=(
                detect_query_mode
            ),
            top_k=TOP_K,
            section_boost=SECTION_BOOST,
            heading_penalty=(
                HEADING_PENALTY
            ),
        )
    )

    query_mode = (
        detect_query_mode(
            query
        )
    )

    intended_section = (
        detect_section_intent(
            query
        )
    )

    print(
        "\nQuery mode:"
    )

    print(
        query_mode
    )

    print(
        "\nDetected section intent:"
    )

    print(
        intended_section
    )

    print(
        "\nRetrieved sources:"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()

        print(
            f"[Source {index}]"
        )

        print(
            f"File: "
            f"{Path(result['file_path']).name}"
        )

        if result.get(
            "page_number"
        ) is not None:
            print(
                f"Page: "
                f"{result['page_number']}"
            )

        if result.get(
            "section_name"
        ):
            print(
                f"Section: "
                f"{result['section_name']}"
            )

        print(
            f"Score: "
            f"{result['final_score']:.4f}"
        )

    generation_results = results

    if intended_section is not None:
        matching_section_results = [
            result
            for result in results
            if result.get(
                "section_name"
            ) == intended_section
        ]

        if matching_section_results:
            generation_results = (
                matching_section_results
            )

    ranked_evidence = (
        rank_evidence_units(
            query=query,
            results=generation_results,
            embedding_client=embedding_client,
            top_k=10,
        )
    )

    structural_evidence = (
        filter_by_evidence_type(
            query,
            ranked_evidence,
        )
    )

    subqueries = split_compound_query(
        query
    )

    print(
        "\nQuery subclaims:"
    )

    for index, subquery in enumerate(
        subqueries,
        start=1,
    ):
        print(
            f"{index}. {subquery}"
        )

    topical_evidence = []
    topical_seen = set()

    for subquery in subqueries:
        subquery_evidence = (
            filter_by_topic(
                subquery,
                structural_evidence,
            )
        )

        for unit in subquery_evidence:
            evidence_key = (
                unit.get("file_path"),
                unit.get("chunk_index"),
                unit.get("unit_index"),
                unit.get("text"),
                unit.get("parent_text"),
            )

            if evidence_key in topical_seen:
                continue

            topical_seen.add(
                evidence_key
            )

            topical_evidence.append(
                unit
            )

    print(
        "\nStructural evidence units:"
    )

    for index, unit in enumerate(
        structural_evidence,
        start=1,
    ):
        print(
            f"{index}. {unit['text']}"
        )

    print(
        "\nTopical evidence units:"
    )

    for index, unit in enumerate(
        topical_evidence,
        start=1,
    ):
        print(
            f"{index}. {unit['text']}"
        )

        if unit.get("parent_text"):
            print(
                f"   Parent: "
                f"{unit['parent_text']}"
            )

    judge = LocalNLIJudge()

    accepted_evidence = []
    accepted_seen = set()

    print(
        "\nNLI validation:"
    )

    validation_index = 1

    for subquery in subqueries:
        subquery_evidence = (
            filter_by_topic(
                subquery,
                structural_evidence,
            )
        )

        hypothesis = build_nli_hypothesis(
            subquery
        )

        print(
            f"\nSubclaim: {subquery}"
        )

        print(
            f"Hypothesis: {hypothesis}"
        )

        for unit in subquery_evidence:
            result = judge.judge_claim(
                hypothesis,
                unit["text"],
            )

            score = result[
                "entailment_score"
            ]

            accepted = (
                score >= NLI_THRESHOLD
            )

            print(
                f"{validation_index}. Entailment: "
                f"{score:.4f} | "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            )

            print(
                f"   {unit['text']}"
            )

            validation_index += 1

            if not accepted:
                continue

            evidence_key = (
                unit.get("file_path"),
                unit.get("chunk_index"),
                unit.get("unit_index"),
                unit.get("text"),
                unit.get("parent_text"),
            )

            if evidence_key in accepted_seen:
                continue

            accepted_seen.add(
                evidence_key
            )

            accepted_unit = dict(
                unit
            )

            accepted_unit[
                "entailment_score"
            ] = score

            accepted_unit[
                "supporting_subquery"
            ] = subquery

            accepted_unit[
                "supporting_hypothesis"
            ] = hypothesis

            accepted_evidence.append(
                accepted_unit
            )

    accepted_evidence = (
        deduplicate_overlapping_evidence_units(
            accepted_evidence
        )
    )

    print(
        "\nAccepted evidence:"
    )

    if not accepted_evidence:
        print(
            "No supported evidence."
        )
    else:
        for index, unit in enumerate(
            accepted_evidence,
            start=1,
        ):
            print(
                f"{index}. {unit['text']}"
            )

            if unit.get("parent_text"):
                print(
                    f"   Parent: "
                    f"{unit['parent_text']}"
                )

    print(
        "\nGeneration sources:"
    )

    for index, unit in enumerate(
        accepted_evidence,
        start=1,
    ):
        print()

        print(
            f"[Generation Source {index}]"
        )

        print(
            f"File: "
            f"{Path(unit['file_path']).name}"
        )

        if unit.get(
            "page_number"
        ) is not None:
            print(
                f"Page: "
                f"{unit['page_number']}"
            )

        if unit.get(
            "section_name"
        ):
            print(
                f"Section: "
                f"{unit['section_name']}"
            )

        if unit.get(
            "chunk_index"
        ) is not None:
            print(
                f"Chunk: "
                f"{unit['chunk_index']}"
            )

        if unit.get(
            "parent_text"
        ):
            print(
                f"Parent: "
                f"{unit['parent_text']}"
            )

        print(
            f"Evidence: "
            f"{unit['text']}"
        )

        if unit.get(
            "entailment_score"
        ) is not None:
            print(
                f"Entailment: "
                f"{unit['entailment_score']:.4f}"
            )

    print(
        "\nGenerating grounded answer..."
    )

    answer = (
        generate_grounded_answer_from_evidence(
            chat_client=chat_client,
            query=query,
            evidence_units=accepted_evidence,
            claim_judge=judge,
        )
    )

    print(
        "\nANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    print(
        "=" * 70
    )

    chat_model.unload()
    embedding_model.unload()


if __name__ == "__main__":
    main()