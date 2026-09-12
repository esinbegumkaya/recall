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


from recall.engine import search_recall


def print_sources(sources):
    if not sources:
        return

    print("\nSOURCES")
    print("=" * 70)

    for source in sources:
        source_number = source.get(
            "source_number"
        )

        file_name = source.get(
            "file_name"
        )

        print(
            f"\n[Source {source_number}] "
            f"{file_name}"
        )

        page_number = source.get(
            "page_number"
        )

        if page_number is not None:
            print(
                f"Page: {page_number}"
            )

        section_name = source.get(
            "section_name"
        )

        if section_name:
            print(
                f"Section: {section_name}"
            )

        chunk_index = source.get(
            "chunk_index"
        )

        if chunk_index is not None:
            print(
                f"Chunk: {chunk_index}"
            )

        parent_text = source.get(
            "parent_text"
        )

        if parent_text:
            print(
                f"Parent: {parent_text}"
            )

        evidence_text = source.get(
            "text"
        )

        if evidence_text:
            print(
                f"Evidence: {evidence_text}"
            )


def print_diagnostics(result):
    print("\nDIAGNOSTICS")
    print("=" * 70)

    print(
        f"Candidates: "
        f"{result.get('candidate_count', 0)}"
    )

    print(
        f"Accepted evidence: "
        f"{result.get('evidence_count', 0)}"
    )

    print(
        f"Abstained: "
        f"{result.get('abstained', False)}"
    )

    abstention_reason = result.get(
        "abstention_reason"
    )

    if abstention_reason:
        print(
            f"Abstention reason: "
            f"{abstention_reason}"
        )

    elapsed_seconds = result.get(
        "elapsed_seconds"
    )

    if elapsed_seconds is not None:
        print(
            f"Elapsed: "
            f"{elapsed_seconds:.2f}s"
        )


def main():
    print("Recall — Local RAG Assistant")
    print("Type 'exit' or 'quit' to close.")

    while True:
        try:
            query = input(
                "\nAsk Recall: "
            ).strip()
        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print("\nGoodbye.")
            break

        if not query:
            print(
                "Question cannot be empty."
            )
            continue

        if query.lower() in {
            "exit",
            "quit",
        }:
            print(
                "Goodbye."
            )
            break

        try:
            result = search_recall(
                query
            )
        except Exception as exc:
            print(
                "\nRecall could not process "
                "the question."
            )

            print(
                f"Error: {exc}"
            )

            continue

        print("\nANSWER")
        print("=" * 70)

        print(
            result.get(
                "answer",
                (
                    "The available files do not "
                    "provide enough evidence to "
                    "answer this question."
                ),
            )
        )

        print_sources(
            result.get(
                "sources",
                [],
            )
        )

        print_diagnostics(
            result
        )


if __name__ == "__main__":
    main()