from pathlib import Path
import sqlite3
from typing import Dict
import json


DEFAULT_DB_PATH = Path("data/index/recall.db")


def get_connection(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    return connection


def column_exists(
    connection,
    table_name: str,
    column_name: str,
) -> bool:
    columns = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return any(
        column["name"] == column_name
        for column in columns
    )


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT,
                size_bytes INTEGER,
                created_at TEXT,
                modified_at TEXT,
                content_hash TEXT NOT NULL,
                indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                page_number INTEGER,
                section_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(file_path, chunk_index)
            )
            """
        )

        if not column_exists(
            connection,
            "chunks",
            "page_number",
        ):
            connection.execute(
                """
                ALTER TABLE chunks
                ADD COLUMN page_number INTEGER
                """
            )

        if not column_exists(
            connection,
            "chunks",
            "section_name",
        ):
            connection.execute(
                """
                ALTER TABLE chunks
                ADD COLUMN section_name TEXT
                """
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(file_path, chunk_index, model_name)
            )
            """
        )

        connection.commit()


def get_file_by_path(file_path: str):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM files
            WHERE file_path = ?
            """,
            (file_path,),
        ).fetchone()

        return row


def upsert_file(file_data: Dict):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO files (
                file_path,
                file_name,
                extension,
                size_bytes,
                created_at,
                modified_at,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(file_path)
            DO UPDATE SET
                file_name = excluded.file_name,
                extension = excluded.extension,
                size_bytes = excluded.size_bytes,
                created_at = excluded.created_at,
                modified_at = excluded.modified_at,
                content_hash = excluded.content_hash,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (
                file_data["file_path"],
                file_data["file_name"],
                file_data["extension"],
                file_data["size_bytes"],
                file_data["created_at"],
                file_data["modified_at"],
                file_data["content_hash"],
            ),
        )

        connection.commit()


def replace_chunks(
    file_path: str,
    chunks: list,
):
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM chunks
            WHERE file_path = ?
            """,
            (file_path,),
        )

        for index, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                chunk_text = chunk
                page_number = None
                section_name = None
            else:
                chunk_text = chunk["text"]
                page_number = chunk.get(
                    "page_number"
                )
                section_name = chunk.get(
                    "section_name"
                )

            connection.execute(
                """
                INSERT INTO chunks (
                    file_path,
                    chunk_index,
                    chunk_text,
                    page_number,
                    section_name
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    index,
                    chunk_text,
                    page_number,
                    section_name,
                ),
            )

        connection.commit()


def delete_embeddings_for_file(
    file_path: str,
    model_name: str,
):
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM embeddings
            WHERE file_path = ?
              AND model_name = ?
            """,
            (
                file_path,
                model_name,
            ),
        )

        connection.commit()


def upsert_embedding(
    file_path: str,
    chunk_index: int,
    model_name: str,
    embedding: list[float],
):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO embeddings (
                file_path,
                chunk_index,
                model_name,
                embedding
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(file_path, chunk_index, model_name)
            DO UPDATE SET
                embedding = excluded.embedding,
                created_at = CURRENT_TIMESTAMP
            """,
            (
                file_path,
                chunk_index,
                model_name,
                json.dumps(embedding),
            ),
        )

        connection.commit()


def get_chunks_with_embeddings(
    model_name: str,
):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.file_path,
                c.chunk_index,
                c.chunk_text,
                c.page_number,
                c.section_name,
                e.embedding
            FROM chunks c
            JOIN embeddings e
                ON c.file_path = e.file_path
                AND c.chunk_index = e.chunk_index
            WHERE e.model_name = ?
            """,
            (model_name,),
        ).fetchall()

        return rows