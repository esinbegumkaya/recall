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
    dense_retrieve,
)


MODEL_NAME = "qwen3-embedding-0.6b"


FAILURE_QUERIES = [
    {
        "id": "v01",
        "query": (
            "What hands-on artificial "
            "intelligence projects are described?"
        ),
    },
    {
        "id": "v02",
        "query": (
            "Where did this person work with "
            "AI agents and RAG systems professionally?"
        ),
    },
    {
        "id": "v05",
        "query": (
            "What formal training has this person "
            "completed in artificial intelligence?"
        ),
    },
]


def main():
    initialize_database()

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
        f"Loaded {len(rows)} indexed chunks."
    )

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

    for item in FAILURE_QUERIES:
        query = item["query"]

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

        results = dense_retrieve(
            query_embedding=(
                query_embedding
            ),
            rows=rows,
            top_k=10,
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"{item['id']}: {query}"
        )

        print(
            "=" * 80
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"\nRank: {rank}"
            )

            print(
                f"Score: "
                f"{result['semantic_score']:.4f}"
            )

            print(
                f"File: "
                f"{Path(result['file_path']).name}"
            )

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
                "-" * 80
            )

    embedding_model.unload()


if __name__ == "__main__":
    main()