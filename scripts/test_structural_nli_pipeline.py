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
    rank_evidence_units,
)

from recall.evidence_gate import (
    filter_by_evidence_type,
    filter_by_topic,
)

from recall.nli_judge import (
    LocalNLIJudge,
)


EMBEDDING_MODEL_NAME = (
    "qwen3-embedding-0.6b"
)

SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05

# Experimental only.
# Not yet a production threshold.
NLI_THRESHOLD = 0.15


def main():
    initialize_database()

    query = input(
        "Pipeline query: "
    ).strip()

    if not query:
        print(
            "Query cannot be empty."
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

    embedding_model.load()

    embedding_client = (
        embedding_model
        .get_embedding_client()
    )

    query_response = (
        embedding_client
        .generate_embedding(
            query
        )
    )

    query_embedding = (
        query_response
        .data[0]
        .embedding
    )

    retrieval_results = (
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
            top_k=5,
            section_boost=(
                SECTION_BOOST
            ),
            heading_penalty=(
                HEADING_PENALTY
            ),
        )
    )

    print(
        "\n1. RETRIEVED CHUNKS\n"
    )

    for index, result in enumerate(
        retrieval_results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result.get('section_name')} "
            f"| {result['final_score']:.4f}"
        )

    evidence_units = (
        rank_evidence_units(
            query=query,
            results=retrieval_results,
            embedding_client=(
                embedding_client
            ),
            top_k=15,
        )
    )

    print(
        "\n2. MICRO-EVIDENCE BEFORE GATE\n"
    )

    for index, unit in enumerate(
        evidence_units,
        start=1,
    ):
        print(
            f"{index}. "
            f"[{unit.get('section_name')}] "
            f"{unit['evidence_score']:.4f}"
        )

        print(
            unit["text"]
        )

        print(
            "-" * 70
        )

    gated_units = (
        filter_by_evidence_type(
            query=query,
            evidence_units=(
                evidence_units
            ),
        )
    )

    print(
        "\n3. AFTER STRUCTURAL GATE\n"
    )

    for index, unit in enumerate(
        gated_units,
        start=1,
    ):
        print(
            f"{index}. "
            f"[{unit.get('section_name')}]"
        )

        print(
            unit["text"]
        )

        print(
            "-" * 70
        )

    topical_units = (
        filter_by_topic(
            query=query,
            evidence_units=(
                gated_units
            ),
        )
    )

    print(
        "\n4. AFTER TOPICAL GATE\n"
    )

    for index, unit in enumerate(
        topical_units,
        start=1,
    ):
        print(
            f"{index}. "
            f"[{unit.get('section_name')}]"
        )

        print(
            unit["text"]
        )

        print(
            "-" * 70
        )
    judge = LocalNLIJudge()

    print(
        "\n5. NLI VALIDATION\n"
    )

    accepted_units = []

    for index, unit in enumerate(
        topical_units,
        start=1,
    ):
        result = judge.judge(
            query=query,
            evidence_text=unit[
                "text"
            ],
        )

        score = result[
            "entailment_score"
        ]

        accepted = (
            score >= NLI_THRESHOLD
        )

        print(
            f"{index}. "
            f"Entailment: "
            f"{score:.4f} "
            f"| "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )

        print(
            unit["text"]
        )

        print(
            "-" * 70
        )

        if accepted:
            accepted_unit = dict(
                unit
            )

            accepted_unit[
                "entailment_score"
            ] = score

            accepted_units.append(
                accepted_unit
            )

    print(
        "\n6. FINAL EVIDENCE\n"
    )

    if not accepted_units:
        print(
            "No supported evidence."
        )

    else:
        for index, unit in enumerate(
            accepted_units,
            start=1,
        ):
            print(
                f"{index}. "
                f"{unit['text']}"
            )

            if unit.get("parent_text"):
                print(
                    f"   Parent: "
                    f"{unit['parent_text']}"
                )

            print(
                f"   Semantic: "
                f"{unit['evidence_score']:.4f}"
            )

            print(
                f"   Entailment: "
                f"{unit['entailment_score']:.4f}"
            )

    embedding_model.unload()


if __name__ == "__main__":
    main()




