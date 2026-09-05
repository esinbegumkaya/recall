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

from recall.generator import (
    generate_grounded_answer,
)


EMBEDDING_MODEL_NAME = (
    "qwen3-embedding-0.6b"
)

CHAT_MODEL_NAME = (
    "qwen3.5-2b"
)

TOP_K = 5

SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05


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

    print(
        "\nGeneration sources:"
    )

    for index, result in enumerate(
        generation_results,
        start=1,
    ):
        print()

        print(
            f"[Generation Source {index}]"
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

    print(
        "\nGenerating grounded answer..."
    )

    answer = (
        generate_grounded_answer(
            chat_client=chat_client,
            query=query,
            results=generation_results,
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