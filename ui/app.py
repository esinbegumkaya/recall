from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


# =========================================================
# Project path
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from recall.database import (  # noqa: E402
    get_index_stats,
    initialize_database,
)
from recall.hybrid_retriever import (  # noqa: E402
    search_query,
)


# =========================================================
# App configuration
# =========================================================

APP_TITLE = "Recall"
APP_SUBTITLE = (
    "Find the right file on your computer using natural language."
)

DEFAULT_RESULT_LIMIT = 8
MAX_RESULT_LIMIT = 20

INDEX_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "index_folder.py"
)


st.set_page_config(
    page_title="Recall",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Helpers
# =========================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def safe_float(
    value: Any,
) -> float:
    try:
        return float(
            value
            or 0.0
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def format_score(
    value: Any,
) -> str:
    return (
        f"{clamp(safe_float(value)) * 100:.0f}%"
    )


def format_timestamp(
    value: Any,
) -> str:
    timestamp = safe_float(
        value
    )

    if timestamp <= 0:
        return "Unknown"

    try:
        return datetime.fromtimestamp(
            timestamp
        ).strftime(
            "%d %b %Y · %H:%M"
        )
    except (
        OSError,
        OverflowError,
        ValueError,
    ):
        return "Unknown"


def format_file_size(
    path: Path,
) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "Unknown size"

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    value = float(
        size
    )

    for unit in units:
        if value < 1024.0:
            if unit == "B":
                return (
                    f"{int(value)} {unit}"
                )
            return (
                f"{value:.1f} {unit}"
            )

        value /= 1024.0

    return (
        f"{value:.1f} TB"
    )


def clean_snippet(
    candidate: Dict[str, Any],
    limit: int = 650,
) -> str:
    text = (
        candidate.get(
            "best_chunk_text"
        )
        or candidate.get(
            "chunk_text"
        )
        or candidate.get(
            "text"
        )
        or ""
    )

    text = " ".join(
        str(text).split()
    ).strip()

    if not text:
        return (
            "No matching text preview is available "
            "for this result."
        )

    if len(text) <= limit:
        return text

    return (
        text[:limit].rstrip()
        + "…"
    )


def result_page_section(
    candidate: Dict[str, Any],
) -> str:
    pieces = []

    page_number = candidate.get(
        "page_number"
    )

    section_name = candidate.get(
        "section_name"
    )

    if page_number not in (
        None,
        "",
    ):
        pieces.append(
            f"Page {page_number}"
        )

    if section_name:
        pieces.append(
            str(section_name)
        )

    return (
        " · ".join(
            pieces
        )
        if pieces
        else ""
    )


def open_file(
    file_path: str,
) -> tuple[bool, str]:
    path = Path(
        file_path
    )

    if not path.exists():
        return (
            False,
            "The file no longer exists at its indexed path.",
        )

    try:
        if os.name == "nt":
            os.startfile(
                str(path)
            )
        elif sys.platform == "darwin":
            subprocess.Popen(
                [
                    "open",
                    str(path),
                ]
            )
        else:
            subprocess.Popen(
                [
                    "xdg-open",
                    str(path),
                ]
            )

        return (
            True,
            f"Opened {path.name}",
        )

    except Exception as exc:
        return (
            False,
            f"Could not open file: {exc}",
        )


def open_folder(
    file_path: str,
) -> tuple[bool, str]:
    path = Path(
        file_path
    )

    folder = (
        path.parent
        if path.suffix
        else path
    )

    if not folder.exists():
        return (
            False,
            "The folder no longer exists at its indexed path.",
        )

    try:
        if os.name == "nt":
            subprocess.Popen(
                [
                    "explorer",
                    str(folder),
                ]
            )
        elif sys.platform == "darwin":
            subprocess.Popen(
                [
                    "open",
                    str(folder),
                ]
            )
        else:
            subprocess.Popen(
                [
                    "xdg-open",
                    str(folder),
                ]
            )

        return (
            True,
            f"Opened {folder}",
        )

    except Exception as exc:
        return (
            False,
            f"Could not open folder: {exc}",
        )


def notify_action(
    success: bool,
    message: str,
) -> None:
    if success:
        st.toast(
            message,
            icon="✅",
        )
    else:
        st.error(
            message
        )


def normalize_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deduplicate exact indexed paths while preserving retrieval order.
    """

    deduplicated = []
    seen_paths = set()

    for item in results:
        file_path = str(
            item.get(
                "file_path",
                "",
            )
        ).strip()

        if not file_path:
            continue

        normalized = os.path.normcase(
            os.path.normpath(
                file_path
            )
        )

        if normalized in seen_paths:
            continue

        seen_paths.add(
            normalized
        )

        deduplicated.append(
            item
        )

    return deduplicated



def run_index_update(
    include_code: bool = False,
) -> tuple[bool, str]:
    """
    Run Recall's incremental computer-wide indexer and stream
    its console progress into the Streamlit UI.
    """

    if not INDEX_SCRIPT.exists():
        return (
            False,
            f"Index script not found: {INDEX_SCRIPT}",
        )

    command = [
        sys.executable,
        "-u",
        str(INDEX_SCRIPT),
        "--computer",
    ]

    if include_code:
        command.append(
            "--include-code"
        )

    status_box = st.status(
        "Scanning your computer…",
        expanded=True,
    )

    phase_text = status_box.empty()
    current_file_text = status_box.empty()
    detail_text = status_box.empty()
    progress_bar = status_box.progress(
        0.0,
        text="Starting scan…",
    )

    log_lines: List[str] = []

    found_total = None
    metadata_current = 0
    metadata_total = 0
    indexing_current = 0
    indexing_total = 0
    files_indexed = None
    chunks_created = None
    failures = None
    up_to_date = False

    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:
        status_box.update(
            label="Could not start scan",
            state="error",
            expanded=True,
        )

        return (
            False,
            str(exc),
        )

    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.strip()

        if not line:
            continue

        log_lines.append(
            line
        )

        # Keep the in-memory log bounded.
        if len(log_lines) > 300:
            log_lines = log_lines[-300:]

        found_match = re.search(
            r"Found\s+(\d+)\s+supported files",
            line,
            flags=re.IGNORECASE,
        )

        if found_match:
            found_total = int(
                found_match.group(1)
            )

            phase_text.markdown(
                "**Phase:** File discovery complete"
            )

            detail_text.caption(
                f"Found {found_total:,} supported files."
            )

            progress_bar.progress(
                0.20,
                text=(
                    f"Found {found_total:,} supported files"
                ),
            )

            continue

        metadata_match = re.search(
            r"Checking metadata:\s*(\d+)\s*/\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if metadata_match:
            metadata_current = int(
                metadata_match.group(1)
            )

            metadata_total = max(
                int(
                    metadata_match.group(2)
                ),
                1,
            )

            fraction = (
                metadata_current
                / metadata_total
            )

            # Discovery/classification occupies the first 55%
            # of the visual progress bar.
            visual_progress = (
                0.20
                + 0.35 * fraction
            )

            phase_text.markdown(
                "**Phase:** Comparing files with the local index"
            )

            current_file_text.caption(
                "Checking file metadata and detecting changes…"
            )

            progress_bar.progress(
                min(
                    visual_progress,
                    0.55,
                ),
                text=(
                    f"Checked {metadata_current:,} "
                    f"of {metadata_total:,} files"
                ),
            )

            continue

        requires_match = re.search(
            r"Files requiring indexing:\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if requires_match:
            indexing_total = int(
                requires_match.group(1)
            )

            detail_text.caption(
                f"{indexing_total:,} changed/new files require indexing."
            )

            if indexing_total == 0:
                progress_bar.progress(
                    1.0,
                    text="Local index is already up to date",
                )

            continue

        processing_match = re.search(
            r"\[(\d+)\s*/\s*(\d+)\]\s*Processing:\s*(.+)",
            line,
            flags=re.IGNORECASE,
        )

        if processing_match:
            indexing_current = int(
                processing_match.group(1)
            )

            indexing_total = max(
                int(
                    processing_match.group(2)
                ),
                1,
            )

            current_name = (
                processing_match
                .group(3)
                .strip()
            )

            fraction = (
                indexing_current
                / indexing_total
            )

            # Actual changed-file indexing occupies the final 45%.
            visual_progress = (
                0.55
                + 0.45 * fraction
            )

            phase_text.markdown(
                "**Phase:** Updating searchable index"
            )

            current_file_text.markdown(
                f"**Current file:** `{current_name}`"
            )

            detail_text.caption(
                f"Indexing {indexing_current:,} "
                f"of {indexing_total:,} changed files"
            )

            progress_bar.progress(
                min(
                    visual_progress,
                    0.99,
                ),
                text=(
                    f"Indexing {indexing_current:,} "
                    f"of {indexing_total:,}"
                ),
            )

            continue

        files_indexed_match = re.search(
            r"Files indexed:\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if files_indexed_match:
            files_indexed = int(
                files_indexed_match.group(1)
            )
            continue

        chunks_match = re.search(
            r"Chunks created:\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if chunks_match:
            chunks_created = int(
                chunks_match.group(1)
            )
            continue

        failures_match = re.search(
            r"Failures:\s*(\d+)",
            line,
            flags=re.IGNORECASE,
        )

        if failures_match:
            failures = int(
                failures_match.group(1)
            )
            continue

        if (
            "Local index is already up to date"
            in line
        ):
            up_to_date = True

    return_code = process.wait()

    if return_code != 0:
        status_box.update(
            label="Index update failed",
            state="error",
            expanded=True,
        )

        with status_box.expander(
            "Indexer output",
            expanded=False,
        ):
            st.code(
                "\n".join(
                    log_lines[-100:]
                ),
                language=None,
            )

        return (
            False,
            f"Indexer exited with code {return_code}.",
        )

    progress_bar.progress(
        1.0,
        text="Index update complete",
    )

    current_file_text.empty()

    if up_to_date:
        final_message = (
            "Your local index is already up to date."
        )
    else:
        pieces = []

        if found_total is not None:
            pieces.append(
                f"{found_total:,} files scanned"
            )

        if files_indexed is not None:
            pieces.append(
                f"{files_indexed:,} files indexed"
            )

        if chunks_created is not None:
            pieces.append(
                f"{chunks_created:,} chunks created"
            )

        if failures is not None:
            pieces.append(
                f"{failures:,} failures"
            )

        final_message = (
            " · ".join(pieces)
            if pieces
            else "Index update complete."
        )

    phase_text.markdown(
        "**Phase:** Complete"
    )

    detail_text.caption(
        final_message
    )

    status_box.update(
        label="Local index updated",
        state="complete",
        expanded=True,
    )

    return (
        True,
        final_message,
    )

def render_result_card(
    candidate: Dict[str, Any],
    rank: int,
    diagnostics: bool,
) -> None:
    file_path = str(
        candidate.get(
            "file_path",
            "",
        )
    )

    path = Path(
        file_path
    )

    file_name = (
        candidate.get(
            "file_name"
        )
        or path.name
        or "Unknown file"
    )

    score = safe_float(
        candidate.get(
            "score"
        )
    )

    modified_at = candidate.get(
        "modified_at"
    )

    created_at = candidate.get(
        "created_at"
    )

    extension = (
        candidate.get(
            "extension"
        )
        or path.suffix
        or ""
    )

    with st.container(
        border=True
    ):
        header_col, score_col = st.columns(
            [
                5,
                1,
            ],
            vertical_alignment="center",
        )

        with header_col:
            st.markdown(
                f"### {rank}. {file_name}"
            )

            metadata = []

            if extension:
                metadata.append(
                    extension.upper().lstrip(
                        "."
                    )
                )

            metadata.append(
                format_file_size(
                    path
                )
            )

            page_section = (
                result_page_section(
                    candidate
                )
            )

            if page_section:
                metadata.append(
                    page_section
                )

            st.caption(
                " · ".join(
                    metadata
                )
            )

        with score_col:
            st.metric(
                "Match",
                format_score(
                    score
                ),
            )

        st.write(
            clean_snippet(
                candidate
            )
        )

        date_col_1, date_col_2 = st.columns(
            2
        )

        with date_col_1:
            st.caption(
                "Modified · "
                + format_timestamp(
                    modified_at
                )
            )

        with date_col_2:
            st.caption(
                "Created · "
                + format_timestamp(
                    created_at
                )
            )

        with st.expander(
            "Location",
            expanded=False,
        ):
            st.code(
                file_path,
                language=None,
                wrap_lines=True,
            )

        button_col_1, button_col_2, _ = st.columns(
            [
                1,
                1,
                3,
            ]
        )

        with button_col_1:
            if st.button(
                "Open file",
                key=f"open_file_{rank}_{file_path}",
                use_container_width=True,
            ):
                success, message = (
                    open_file(
                        file_path
                    )
                )

                notify_action(
                    success,
                    message,
                )

        with button_col_2:
            if st.button(
                "Open folder",
                key=f"open_folder_{rank}_{file_path}",
                use_container_width=True,
            ):
                success, message = (
                    open_folder(
                        file_path
                    )
                )

                notify_action(
                    success,
                    message,
                )

        if diagnostics:
            with st.expander(
                "Developer diagnostics",
                expanded=False,
            ):
                diagnostic_rows = {
                    "final_score": candidate.get(
                        "score"
                    ),
                    "semantic_score": candidate.get(
                        "semantic_score"
                    ),
                    "lexical_score": candidate.get(
                        "lexical_score"
                    ),
                    "filename_score": candidate.get(
                        "filename_score"
                    ),
                    "direct_score": candidate.get(
                        "direct_score"
                    ),
                    "recency_score": candidate.get(
                        "recency_score"
                    ),
                    "date_window_score": candidate.get(
                        "date_window_score"
                    ),
                    "location_score": candidate.get(
                        "location_score"
                    ),
                    "extension_score": candidate.get(
                        "extension_score"
                    ),
                    "planner": candidate.get(
                        "planner"
                    ),
                    "chunk_index": candidate.get(
                        "chunk_index"
                    ),
                }

                st.json(
                    diagnostic_rows
                )


# =========================================================
# Initialize index
# =========================================================

initialize_database()


# =========================================================
# Session state
# =========================================================

if "recall_results" not in st.session_state:
    st.session_state.recall_results = []

if "recall_query" not in st.session_state:
    st.session_state.recall_query = ""

if "recall_error" not in st.session_state:
    st.session_state.recall_error = None


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title(
        "Recall"
    )

    st.caption(
        "Private local file discovery"
    )

    st.caption(
        "Search runs over Recall's private local index."
    )

    st.divider()


    try:
        stats = get_index_stats()
    except Exception:
        stats = {}

    file_count = (
        stats.get(
            "files"
        )
        or stats.get(
            "file_count"
        )
        or 0
    )

    chunk_count = (
        stats.get(
            "chunks"
        )
        or stats.get(
            "chunk_count"
        )
        or 0
    )

    embedding_count = (
        stats.get(
            "embeddings"
        )
        or stats.get(
            "embedding_count"
        )
        or 0
    )

    st.metric(
        "Indexed files",
        f"{int(file_count):,}",
    )

    st.metric(
        "Searchable chunks",
        f"{int(chunk_count):,}",
    )

    st.metric(
        "Cached embeddings",
        f"{int(embedding_count):,}",
    )

    st.divider()

    result_limit = st.slider(
        "Results",
        min_value=3,
        max_value=MAX_RESULT_LIMIT,
        value=DEFAULT_RESULT_LIMIT,
        step=1,
    )

    diagnostics = st.toggle(
        "Developer diagnostics",
        value=False,
    )

    st.caption(
        "Queries are processed locally by Recall's "
        "query planner and hybrid retrieval pipeline."
    )


# =========================================================
# Main view
# =========================================================

st.title(
    APP_TITLE
)

st.subheader(
    APP_SUBTITLE
)

st.caption(
    "Search by topic, filename, meaning, date, or a rough memory "
    "of what the file contains. Turkish, English, mixed language, "
    "and common typos are supported."
)

query = st.text_input(
    "What are you looking for?",
    placeholder=(
        "Describe the file or information you remember…"
    ),
    value=st.session_state.recall_query,
)

search_clicked = st.button(
    "Search my computer",
    type="primary",
    use_container_width=True,
)

if search_clicked:
    query = (
        query
        or ""
    ).strip()

    st.session_state.recall_query = (
        query
    )

    st.session_state.recall_error = (
        None
    )

    if not query:
        st.session_state.recall_results = []

        st.warning(
            "Enter a search request first."
        )

    else:
        search_status = st.status(
            "Searching your computer…",
            expanded=True,
        )

        phase_text = search_status.empty()
        file_text = search_status.empty()
        progress_text = search_status.empty()
        progress_bar = search_status.progress(
            0.02,
            text="Starting search…",
        )

        def on_search_progress(
            phase: str,
            current: int,
            total: int,
            detail: str,
        ) -> None:
            total = max(
                int(total or 0),
                1,
            )

            current = max(
                int(current or 0),
                0,
            )

            if phase == "planning":
                fraction = (
                    current
                    / total
                )

                progress_bar.progress(
                    min(
                        0.12,
                        0.03
                        + 0.09 * fraction,
                    ),
                    text="Understanding your request…",
                )

                phase_text.markdown(
                    "**Understanding your request**"
                )

                progress_text.caption(
                    detail
                )

            elif phase == "scanning":
                fraction = min(
                    current
                    / total,
                    1.0,
                )

                progress_bar.progress(
                    0.12
                    + 0.63 * fraction,
                    text=(
                        f"Scanned {current:,} "
                        f"of {total:,} indexed files"
                    ),
                )

                phase_text.markdown(
                    "**Searching indexed files**"
                )

                progress_text.caption(
                    f"{current:,} / {total:,} files checked"
                )

                if detail:
                    file_text.markdown(
                        f"**Current file:** `{detail}`"
                    )

            elif phase == "semantic":
                fraction = min(
                    current
                    / total,
                    1.0,
                )

                progress_bar.progress(
                    0.75
                    + 0.20 * fraction,
                    text="Ranking the strongest matches…",
                )

                phase_text.markdown(
                    "**Semantic reranking**"
                )

                file_text.empty()

                progress_text.caption(
                    detail
                )

            elif phase == "complete":
                progress_bar.progress(
                    1.0,
                    text="Search complete",
                )

                phase_text.markdown(
                    "**Complete**"
                )

                file_text.empty()

        try:
            raw_results = search_query(
                query,
                limit=result_limit,
                progress_callback=on_search_progress,
            )

            results = normalize_results(
                list(
                    raw_results
                    or []
                )
            )

            st.session_state.recall_results = (
                results
            )

            progress_bar.progress(
                1.0,
                text=(
                    f"Search complete · "
                    f"{len(results)} relevant files found"
                ),
            )

            progress_text.caption(
                f"{len(results)} relevant files found."
            )

            search_status.update(
                label=(
                    f"Search complete · "
                    f"{len(results)} relevant files found"
                ),
                state="complete",
                expanded=False,
            )

        except Exception as exc:
            st.session_state.recall_results = []

            st.session_state.recall_error = (
                str(exc)
            )

            search_status.update(
                label="Search failed",
                state="error",
                expanded=True,
            )


if st.session_state.recall_error:
    st.error(
        "Recall could not complete the search."
    )

    with st.expander(
        "Error details",
        expanded=False,
    ):
        st.code(
            st.session_state.recall_error,
            language=None,
            wrap_lines=True,
        )


results = (
    st.session_state.recall_results
)

if results:
    st.divider()

    top_score = safe_float(
        results[0].get(
            "score"
        )
    )

    summary_col_1, summary_col_2 = st.columns(
        [
            3,
            1,
        ]
    )

    with summary_col_1:
        st.subheader(
            f"Found {len(results)} relevant files"
        )

        st.caption(
            f'Results for “{st.session_state.recall_query}”'
        )

    with summary_col_2:
        st.metric(
            "Best match",
            format_score(
                top_score
            ),
        )

    for rank, candidate in enumerate(
        results,
        start=1,
    ):
        render_result_card(
            candidate,
            rank,
            diagnostics,
        )

elif (
    search_clicked
    and st.session_state.recall_query
    and not st.session_state.recall_error
):
    st.info(
        "No relevant files were found in the current local index."
    )

else:
    st.divider()

    st.info(
        "Enter what you remember about the file, then choose Search my computer. "
        "Recall will search the files already indexed from your computer."
    )
