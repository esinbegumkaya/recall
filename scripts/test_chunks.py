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

from recall.crawler import scan_directory
from recall.database import (
    initialize_database,
    replace_chunks,
)
from recall.parser import read_text_file
from recall.chunker import chunk_text


def main():
    initialize_database()

    files = scan_directory(
        "data/samples"
    )

    total_chunks = 0

    for file in files:

        try:
            text = read_text_file(
                file["file_path"]
            )
        except ValueError:
            continue

        chunks = chunk_text(
            text,
            chunk_size=100,
            overlap=20,
        )

        replace_chunks(
            file["file_path"],
            chunks,
        )

        total_chunks += len(chunks)

        print(
            f"{file['file_name']}: "
            f"{len(chunks)} chunks"
        )

    print(
        f"\nCreated "
        f"{total_chunks} chunks."
    )


if __name__ == "__main__":
    main()