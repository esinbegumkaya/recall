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

from foundry_local_sdk import Configuration, FoundryLocalManager

from recall.crawler import scan_directory
from recall.parser import (
    read_text_file,
    read_pdf_pages,
)
from recall.chunker import chunk_text
from recall.database import (
    initialize_database,
    get_file_by_path,
    upsert_file,
    replace_chunks,
    delete_embeddings_for_file,
    upsert_embedding,
)


MODEL_NAME = "qwen3-embedding-0.6b"


def build_chunks(file):
    extension = file["extension"]

    if extension == ".pdf":
        pdf_pages = read_pdf_pages(
            file["file_path"]
        )

        chunks = []

        for page in pdf_pages:
            sections = page.get(
                "sections",
                [],
            )

            if sections:
                for section in sections:
                    section_chunks = chunk_text(
                        section["text"],
                        chunk_size=500,
                        overlap=100,
                    )

                    for chunk in section_chunks:
                        chunks.append(
                            {
                                "text": chunk,
                                "page_number": page[
                                    "page_number"
                                ],
                                "section_name": section[
                                    "section_name"
                                ],
                            }
                        )

            else:
                page_chunks = chunk_text(
                    page["text"],
                    chunk_size=500,
                    overlap=100,
                )

                for chunk in page_chunks:
                    chunks.append(
                        {
                            "text": chunk,
                            "page_number": page[
                                "page_number"
                            ],
                            "section_name": None,
                        }
                    )

        return chunks

    text = read_text_file(
        file["file_path"]
    )

    text_chunks = chunk_text(
        text,
        chunk_size=500,
        overlap=100,
    )

    return [
        {
            "text": chunk,
            "page_number": None,
            "section_name": None,
        }
        for chunk in text_chunks
    ]


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python scripts\\index_folder.py '
            '"C:\\path\\to\\folder"'
        )
        return

    folder = sys.argv[1]

    print(f"Scanning folder: {folder}")

    initialize_database()

    files = scan_directory(folder)

    print(
        f"Found {len(files)} supported files.\n"
    )

    if not files:
        return

    files_to_index = []
    skipped_files = 0

    for file in files:
        existing_file = get_file_by_path(
            file["file_path"]
        )

        if (
            existing_file is not None
            and existing_file["content_hash"]
            == file["content_hash"]
        ):
            print(
                f"Unchanged: "
                f"{file['file_name']}"
            )

            skipped_files += 1
            continue

        files_to_index.append(file)

    if not files_to_index:
        print("\nINDEXING COMPLETE")
        print("No files changed.")
        print(
            f"Files skipped: "
            f"{skipped_files}"
        )
        return

    print(
        f"\nFiles requiring indexing: "
        f"{len(files_to_index)}"
    )

    print("Loading embedding model...")

    FoundryLocalManager.initialize(
        Configuration(app_name="Recall")
    )

    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(
        MODEL_NAME
    )

    model.load()

    embedding_client = (
        model.get_embedding_client()
    )

    total_chunks = 0
    indexed_files = 0

    for file in files_to_index:
        print(
            f"\nProcessing: "
            f"{file['file_name']}"
        )

        try:
            chunks = build_chunks(file)

        except ValueError as error:
            print(
                f"Skipped: {error}"
            )
            continue

        except Exception as error:
            print(
                f"Failed to read file: "
                f"{error}"
            )
            continue

        if not chunks:
            print(
                "Skipped: no text content."
            )
            continue

        replace_chunks(
            file["file_path"],
            chunks,
        )

        delete_embeddings_for_file(
            file["file_path"],
            MODEL_NAME,
        )

        for chunk_index, chunk in enumerate(
            chunks
        ):
            response = (
                embedding_client
                .generate_embedding(
                    chunk["text"]
                )
            )

            embedding = (
                response
                .data[0]
                .embedding
            )

            upsert_embedding(
                file_path=file["file_path"],
                chunk_index=chunk_index,
                model_name=MODEL_NAME,
                embedding=embedding,
            )

        upsert_file(file)

        indexed_files += 1
        total_chunks += len(chunks)

        print(
            f"Indexed: "
            f"{len(chunks)} chunks"
        )

    model.unload()

    print("\nINDEXING COMPLETE")
    print(
        f"Files indexed: "
        f"{indexed_files}"
    )
    print(
        f"Files skipped: "
        f"{skipped_files}"
    )
    print(
        f"Chunks created: "
        f"{total_chunks}"
    )


if __name__ == "__main__":
    main()