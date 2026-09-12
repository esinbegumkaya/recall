from pathlib import Path
import argparse
import json
import sys
import time


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
    get_connection,
    initialize_database,
)


MODEL_NAME = "qwen3-embedding-0.6b"


def get_missing_chunks(
    limit=None,
    path_contains=None,
):
    with get_connection() as connection:
        query = """
            SELECT
                c.file_path,
                c.chunk_index,
                c.chunk_text
            FROM chunks AS c

            LEFT JOIN embeddings AS e
                ON e.file_path = c.file_path
                AND e.chunk_index = c.chunk_index
                AND e.model_name = ?

            WHERE e.id IS NULL
        """

        parameters = [MODEL_NAME]

        if path_contains:
            query += """
                AND lower(c.file_path)
                    LIKE lower(?)
            """

            parameters.append(
                f"%{path_contains}%"
            )

        query += """
            ORDER BY
                c.file_path,
                c.chunk_index
        """

        if limit is not None:
            query += "\nLIMIT ?"
            parameters.append(limit)

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

        return rows


def count_missing_chunks(
    path_contains=None,
):
    with get_connection() as connection:
        query = """
            SELECT COUNT(*)
            FROM chunks AS c

            LEFT JOIN embeddings AS e
                ON e.file_path = c.file_path
                AND e.chunk_index = c.chunk_index
                AND e.model_name = ?

            WHERE e.id IS NULL
        """

        parameters = [MODEL_NAME]

        if path_contains:
            query += """
                AND lower(c.file_path)
                    LIKE lower(?)
            """

            parameters.append(
                f"%{path_contains}%"
            )

        row = connection.execute(
            query,
            parameters,
        ).fetchone()

        return int(
            row[0]
        )


def save_embedding(
    connection,
    file_path,
    chunk_index,
    embedding,
):
    connection.execute(
        """
        INSERT INTO embeddings (
            file_path,
            chunk_index,
            model_name,
            embedding
        )
        VALUES (?, ?, ?, ?)

        ON CONFLICT(
            file_path,
            chunk_index,
            model_name
        )
        DO UPDATE SET
            embedding = excluded.embedding,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            file_path,
            chunk_index,
            MODEL_NAME,
            json.dumps(
                embedding
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill missing Recall embeddings "
            "without re-parsing files."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most N missing chunks. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--path-contains",
        type=str,
        default=None,
        help=(
            "Only backfill chunks whose file path "
            "contains this text."
        ),
    )

    parser.add_argument(
        "--commit-every",
        type=int,
        default=50,
        help=(
            "Commit progress every N embeddings."
        ),
    )

    args = parser.parse_args()

    initialize_database()

    missing_before = count_missing_chunks(
        path_contains=args.path_contains
    )

    print(
        f"Missing embeddings before: "
        f"{missing_before}"
    )

    if missing_before == 0:
        print(
            "Nothing to backfill."
        )
        return

    rows = get_missing_chunks(
        limit=args.limit,
        path_contains=args.path_contains,
    )

    print(
        f"Chunks selected: "
        f"{len(rows)}"
    )

    if not rows:
        return

    print(
        "\nLoading Foundry Local embedding model..."
    )

    FoundryLocalManager.initialize(
        Configuration(
            app_name="Recall"
        )
    )

    manager = (
        FoundryLocalManager.instance
    )

    model = manager.catalog.get_model(
        MODEL_NAME
    )

    model.load()

    embedding_client = (
        model.get_embedding_client()
    )

    completed = 0
    failed = 0

    started = (
        time.perf_counter()
    )

    connection = (
        get_connection()
    )

    try:
        for index, row in enumerate(
            rows,
            start=1,
        ):
            file_path = (
                row["file_path"]
            )

            chunk_index = (
                row["chunk_index"]
            )

            chunk_text = (
                row["chunk_text"]
            )

            try:
                response = (
                    embedding_client
                    .generate_embedding(
                        chunk_text
                    )
                )

                embedding = (
                    response
                    .data[0]
                    .embedding
                )

                save_embedding(
                    connection=connection,
                    file_path=file_path,
                    chunk_index=chunk_index,
                    embedding=embedding,
                )

                completed += 1

                if (
                    completed
                    % args.commit_every
                    == 0
                ):
                    connection.commit()

                if (
                    index == 1
                    or index % 25 == 0
                    or index == len(rows)
                ):
                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    rate = (
                        completed / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    print(
                        f"[{index}/{len(rows)}] "
                        f"completed={completed} "
                        f"failed={failed} "
                        f"rate={rate:.2f} chunks/s"
                    )

            except KeyboardInterrupt:
                connection.commit()

                print(
                    "\nInterrupted by user."
                )

                print(
                    "Completed progress was saved."
                )

                break

            except Exception as error:
                failed += 1

                print(
                    f"\nFailed:"
                    f"\n  file: {file_path}"
                    f"\n  chunk: {chunk_index}"
                    f"\n  error: {error}"
                )

        connection.commit()

    finally:
        connection.close()

        try:
            model.unload()
        except Exception:
            pass

    missing_after = count_missing_chunks(
        path_contains=args.path_contains
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        "\nBACKFILL COMPLETE"
    )

    print(
        f"Completed: {completed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Missing before: {missing_before}"
    )

    print(
        f"Missing after: {missing_after}"
    )

    print(
        f"Elapsed seconds: {elapsed:.1f}"
    )


if __name__ == "__main__":
    main()