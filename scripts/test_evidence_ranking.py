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


MODEL_NAME = (
    "qwen3-embedding-0.6b"
)

SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05


def main():
    initialize_database()

    query = input(
        "Evidence query: "
    ).strip()

    if not query:
        print(
            "Query cannot be empty."
        )
        return

    rows = list(
        get_chunks_with_embeddings(
            MODEL_NAME
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
            MODEL_NAME
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
            top_k=5,
            section_boost=(
                SECTION_BOOST
            ),
            heading_penalty=(
                HEADING_PENALTY
            ),
        )
    )

    intended_section = (
        detect_section_intent(
            query
        )
    )

    if intended_section is not None:
        matching_results = [
            result
            for result in results
            if result.get(
                "section_name"
            ) == intended_section
        ]

        if matching_results:
            results = (
                matching_results
            )

    evidence_units = (
        rank_evidence_units(
            query=query,
            results=results,
            embedding_client=(
                embedding_client
            ),
            top_k=10,
        )
    )

    print(
        "\nEVIDENCE RANKING\n"
    )

    for rank, unit in enumerate(
        evidence_units,
        start=1,
    ):
        print(
            f"Rank: {rank}"
        )

        print(
            f"Score: "
            f"{unit['evidence_score']:.4f}"
        )

        print(
            f"Source: "
            f"{unit['source_number']}"
        )

        print(
            f"Section: "
            f"{unit['section_name']}"
        )

        print(
            "Text:"
        )

        print(
            unit["text"]
        )

        print(
            "-" * 70
        )

    embedding_model.unload()


if __name__ == "__main__":
    main()