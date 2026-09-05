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


MODEL_NAME = "qwen3-embedding-0.6b"

FINAL_TOP_K = 5
SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05


def main():
    initialize_database()

    query = input(
        "Search query: "
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

    print(
        "\nLoading embedding model..."
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
            MODEL_NAME
        )
    )

    embedding_model.load()

    embedding_client = (
        embedding_model
        .get_embedding_client()
    )

    response = (
        embedding_client
        .generate_embedding(
            query
        )
    )

    query_embedding = (
        response
        .data[0]
        .embedding
    )

    intended_section = (
        detect_section_intent(
            query
        )
    )

    query_mode = (
        detect_query_mode(
            query
        )
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
            top_k=FINAL_TOP_K,
            section_boost=SECTION_BOOST,
            heading_penalty=HEADING_PENALTY,
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
        "\nSEARCH RESULTS\n"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"Rank: {rank}"
        )

        print(
            f"Final Score: "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Section Match: "
            f"{result['section_match']}"
        )

        print(
            f"Heading Only: "
            f"{result['heading_only']}"
        )

        print(
            f"Heading Penalty: "
            f"{result['heading_penalty']:.4f}"
        )

        print(
            f"File: "
            f"{Path(result['file_path']).name}"
        )

        if result["page_number"] is not None:
            print(
                f"Page: "
                f"{result['page_number']}"
            )

        if result["section_name"]:
            print(
                f"Section: "
                f"{result['section_name']}"
            )

        print(
            f"Chunk: "
            f"{result['chunk_index']}"
        )

        print(
            "Text:"
        )

        print(
            result["chunk_text"]
        )

        print(
            "-" * 60
        )

    embedding_model.unload()


if __name__ == "__main__":
    main()