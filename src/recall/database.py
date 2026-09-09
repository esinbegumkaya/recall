from pathlib import Path
import sqlite3
from typing import Dict
import json
import re


DEFAULT_DB_PATH = Path(
    "data/index/recall.db"
)


# =========================================================
# Connection
# =========================================================

def get_connection(
    db_path: Path = DEFAULT_DB_PATH,
):
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        db_path
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# =========================================================
# Schema helpers
# =========================================================

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


def table_exists(
    connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


# =========================================================
# FTS query preparation
# =========================================================

FTS_STOPWORDS = {
    # English
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",

    # Turkish
    "acaba",
    "ama",
    "ben",
    "beni",
    "benim",
    "bir",
    "bu",
    "da",
    "de",
    "hangi",
    "ile",
    "icin",
    "için",
    "mi",
    "mı",
    "mu",
    "mü",
    "ne",
    "nerede",
    "nerden",
    "nereden",
    "ve",
}


def build_fts_query(
    query: str,
) -> str:
    """
    Convert a natural-language query into a safe SQLite
    FTS5 OR query.

    Example:

        Where did I work with AI agents and RAG systems?

    becomes approximately:

        "work" OR "AI" OR "agents" OR "RAG" OR "systems"
    """

    tokens = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜü0-9_+#.-]+",
        query,
    )

    cleaned = []
    seen = set()

    for token in tokens:
        token = token.strip(
            "._-"
        )

        if not token:
            continue

        normalized = token.casefold()

        if normalized in FTS_STOPWORDS:
            continue

        if (
            len(normalized) < 2
            and not normalized.isdigit()
        ):
            continue

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        escaped = token.replace(
            '"',
            '""',
        )

        cleaned.append(
            f'"{escaped}"'
        )

    if not cleaned:
        return ""

    return " OR ".join(
        cleaned
    )


# =========================================================
# Database initialization
# =========================================================

def initialize_database():
    with get_connection() as connection:

        # -------------------------------------------------
        # Files
        # -------------------------------------------------

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
                indexed_at DATETIME
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # -------------------------------------------------
        # Chunks
        # -------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                page_number INTEGER,
                section_name TEXT,
                created_at DATETIME
                    DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(
                    file_path,
                    chunk_index
                )
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

        # -------------------------------------------------
        # Existing semantic embeddings
        # -------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at DATETIME
                    DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(
                    file_path,
                    chunk_index,
                    model_name
                )
            )
            """
        )

        # -------------------------------------------------
        # Full-text search
        # -------------------------------------------------

        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS
                chunks_fts
                USING fts5(
                    file_path UNINDEXED,
                    chunk_index UNINDEXED,
                    file_name,
                    chunk_text,
                    section_name
                )
                """
            )

        except sqlite3.OperationalError as error:
            raise RuntimeError(
                f"Could not initialize SQLite FTS5: "
                f"{error}"
            ) from error

        # -------------------------------------------------
        # Helpful indexes
        # -------------------------------------------------

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_chunks_file_path
            ON chunks(file_path)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_embeddings_file_model
            ON embeddings(
                file_path,
                model_name
            )
            """
        )

        connection.commit()

        # -------------------------------------------------
        # Existing DB migration
        # -------------------------------------------------

        chunk_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        ).fetchone()[0]

        fts_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks_fts
            """
        ).fetchone()[0]

        if chunk_count != fts_count:
            rebuild_fts_index(
                connection=connection
            )


# =========================================================
# FTS rebuilding
# =========================================================

def rebuild_fts_index(
    connection=None,
):
    """
    Rebuild lexical search index from canonical chunks.
    """

    owns_connection = (
        connection is None
    )

    if owns_connection:
        connection = (
            get_connection()
        )

    try:
        connection.execute(
            """
            DELETE FROM chunks_fts
            """
        )

        connection.execute(
            """
            INSERT INTO chunks_fts (
                file_path,
                chunk_index,
                file_name,
                chunk_text,
                section_name
            )
            SELECT
                c.file_path,
                c.chunk_index,
                COALESCE(
                    f.file_name,
                    ''
                ),
                c.chunk_text,
                COALESCE(
                    c.section_name,
                    ''
                )
            FROM chunks c
            LEFT JOIN files f
                ON f.file_path =
                   c.file_path
            """
        )

        connection.commit()

    finally:
        if owns_connection:
            connection.close()


# =========================================================
# File metadata
# =========================================================

def get_file_by_path(
    file_path: str,
):
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


def upsert_file(
    file_data: Dict,
):
    with get_connection() as connection:
        _upsert_file_with_connection(
            connection,
            file_data,
        )

        connection.commit()


def _upsert_file_with_connection(
    connection,
    file_data: Dict,
):
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
            file_name =
                excluded.file_name,
            extension =
                excluded.extension,
            size_bytes =
                excluded.size_bytes,
            created_at =
                excluded.created_at,
            modified_at =
                excluded.modified_at,
            content_hash =
                excluded.content_hash,
            indexed_at =
                CURRENT_TIMESTAMP
        """,
        (
            file_data[
                "file_path"
            ],
            file_data[
                "file_name"
            ],
            file_data[
                "extension"
            ],
            file_data[
                "size_bytes"
            ],
            file_data[
                "created_at"
            ],
            file_data[
                "modified_at"
            ],
            file_data[
                "content_hash"
            ],
        ),
    )


# =========================================================
# Chunk storage
# =========================================================

def replace_chunks(
    file_path: str,
    chunks: list,
):
    """
    Compatibility helper for the existing semantic
    indexing pipeline.

    Also keeps chunks_fts synchronized.
    """

    with get_connection() as connection:

        file_row = connection.execute(
            """
            SELECT file_name
            FROM files
            WHERE file_path = ?
            """,
            (file_path,),
        ).fetchone()

        file_name = (
            file_row["file_name"]
            if file_row
            else Path(
                file_path
            ).name
        )

        _replace_chunks_with_connection(
            connection,
            file_path=file_path,
            file_name=file_name,
            chunks=chunks,
        )

        connection.commit()


def _replace_chunks_with_connection(
    connection,
    file_path: str,
    file_name: str,
    chunks: list,
):
    connection.execute(
        """
        DELETE FROM chunks
        WHERE file_path = ?
        """,
        (file_path,),
    )

    connection.execute(
        """
        DELETE FROM chunks_fts
        WHERE file_path = ?
        """,
        (file_path,),
    )

    for index, chunk in enumerate(
        chunks
    ):
        if isinstance(
            chunk,
            str,
        ):
            chunk_text = chunk
            page_number = None
            section_name = None

        else:
            chunk_text = (
                chunk["text"]
            )

            page_number = (
                chunk.get(
                    "page_number"
                )
            )

            section_name = (
                chunk.get(
                    "section_name"
                )
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

        connection.execute(
            """
            INSERT INTO chunks_fts (
                file_path,
                chunk_index,
                file_name,
                chunk_text,
                section_name
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_path,
                index,
                file_name,
                chunk_text,
                section_name or "",
            ),
        )


# =========================================================
# Fast text-only indexing
# =========================================================

def index_file_text(
    file_data: Dict,
    chunks: list,
):
    """
    Atomically update metadata, chunks and FTS rows.

    Old embeddings for this file are deleted because
    re-chunking would make them stale.
    """

    with get_connection() as connection:
        try:
            connection.execute(
                "BEGIN"
            )

            _upsert_file_with_connection(
                connection,
                file_data,
            )

            _replace_chunks_with_connection(
                connection,
                file_path=file_data[
                    "file_path"
                ],
                file_name=file_data[
                    "file_name"
                ],
                chunks=chunks,
            )

            connection.execute(
                """
                DELETE FROM embeddings
                WHERE file_path = ?
                """,
                (
                    file_data[
                        "file_path"
                    ],
                ),
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise


# =========================================================
# Embeddings
# =========================================================

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


def delete_all_embeddings_for_file(
    file_path: str,
):
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM embeddings
            WHERE file_path = ?
            """,
            (
                file_path,
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

            ON CONFLICT(
                file_path,
                chunk_index,
                model_name
            )
            DO UPDATE SET
                embedding =
                    excluded.embedding,
                created_at =
                    CURRENT_TIMESTAMP
            """,
            (
                file_path,
                chunk_index,
                model_name,
                json.dumps(
                    embedding
                ),
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
                ON c.file_path =
                   e.file_path
                AND c.chunk_index =
                    e.chunk_index
            WHERE e.model_name = ?
            """,
            (
                model_name,
            ),
        ).fetchall()

        return rows


# =========================================================
# Full-text retrieval
# =========================================================

def search_chunks_fts(
    query: str,
    limit: int = 50,
):
    """
    Search Recall's lexical index.

    Lower BM25 score is better.
    """

    prepared_query = (
        build_fts_query(
            query
        )
    )

    if not prepared_query:
        return []

    limit = max(
        1,
        min(
            int(limit),
            200,
        ),
    )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                fts.file_path,

                CAST(
                    fts.chunk_index
                    AS INTEGER
                ) AS chunk_index,

                fts.file_name,

                c.chunk_text,
                c.page_number,
                c.section_name,

                bm25(
                    chunks_fts,
                    0.0,
                    0.0,
                    2.0,
                    1.0,
                    1.5
                ) AS lexical_score,

                snippet(
                    chunks_fts,
                    3,
                    '[[',
                    ']]',
                    ' ... ',
                    24
                ) AS snippet

            FROM chunks_fts AS fts

            JOIN chunks AS c
                ON c.file_path =
                   fts.file_path
                AND c.chunk_index =
                    CAST(
                        fts.chunk_index
                        AS INTEGER
                    )

            WHERE chunks_fts
                MATCH ?

            ORDER BY
                lexical_score ASC

            LIMIT ?
            """,
            (
                prepared_query,
                limit,
            ),
        ).fetchall()

        return rows


# =========================================================
# Index statistics
# =========================================================

def get_index_stats():
    with get_connection() as connection:

        files = connection.execute(
            """
            SELECT COUNT(*)
            FROM files
            """
        ).fetchone()[0]

        chunks = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            """
        ).fetchone()[0]

        fts_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks_fts
            """
        ).fetchone()[0]

        embeddings = connection.execute(
            """
            SELECT COUNT(*)
            FROM embeddings
            """
        ).fetchone()[0]

        return {
            "files": files,
            "chunks": chunks,
            "fts_rows": fts_rows,
            "embeddings": embeddings,
        }