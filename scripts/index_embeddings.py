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

from recall.database import (
    initialize_database,
    get_connection,
    upsert_embedding,
)


MODEL_NAME = "qwen3-embedding-0.6b"


def main():
    initialize_database()

    FoundryLocalManager.initialize(
        Configuration(app_name="Recall")
    )

    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(MODEL_NAME)

    print("Loading embedding model...")
    model.load()

    client = model.get_embedding_client()

    with get_connection() as connection:
        chunks = connection.execute(
            """
            SELECT file_path, chunk_index, chunk_text
            FROM chunks
            ORDER BY file_path, chunk_index
            """
        ).fetchall()

    print(f"Embedding {len(chunks)} chunks...\n")

    for row in chunks:
        response = client.generate_embedding(
            row["chunk_text"]
        )

        embedding = response.data[0].embedding

        upsert_embedding(
            file_path=row["file_path"],
            chunk_index=row["chunk_index"],
            model_name=MODEL_NAME,
            embedding=embedding,
        )

        print(
            f"Embedded: "
            f"{Path(row['file_path']).name} "
            f"chunk {row['chunk_index']}"
        )

    model.unload()

    print("\nEmbedding index complete.")


if __name__ == "__main__":
    main()