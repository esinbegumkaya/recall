from pathlib import Path
import sys
import re


sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
    )
)


from recall.database import (
    initialize_database,
    get_chunks_with_embeddings,
)


def count_words(text):
    return len(
        re.findall(
            r"\b\w+\b",
            text
        )
    )


def is_heading_like(text):
    stripped = text.strip()

    if not stripped:
        return False

    lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip()
    ]

    if len(lines) != 1:
        return False

    line = lines[0]

    if line.startswith("#"):
        return True

    word_count = count_words(
        line
    )

    return word_count <= 5


def main():
    initialize_database()

    rows = list(
        get_chunks_with_embeddings(
            "qwen3-embedding-0.6b"
        )
    )

    print(
        f"Indexed chunks: {len(rows)}"
    )

    print(
        "\nCHUNK QUALITY ANALYSIS"
    )

    print(
        "=" * 70
    )

    sorted_rows = sorted(
        rows,
        key=lambda row: count_words(
            row["chunk_text"]
        ),
    )

    for row in sorted_rows:
        text = row[
            "chunk_text"
        ]

        word_count = count_words(
            text
        )

        heading_like = (
            is_heading_like(
                text
            )
        )

        print()

        print(
            f"File: "
            f"{Path(row['file_path']).name}"
        )

        print(
            f"Chunk: "
            f"{row['chunk_index']}"
        )

        print(
            f"Section: "
            f"{row['section_name']}"
        )

        print(
            f"Words: "
            f"{word_count}"
        )

        print(
            f"Heading-like: "
            f"{heading_like}"
        )

        print(
            f"Text: "
            f"{text[:200]}"
        )

        print(
            "-" * 70
        )


if __name__ == "__main__":
    main()