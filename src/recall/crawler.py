from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib
import os


# =========================================================
# Supported file types
# =========================================================

# Default personal-document profile.
DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
}


# Optional structured-data profile.
STRUCTURED_EXTENSIONS = {
    ".json",
    ".yaml",
    ".yml",
    ".csv",
}


# Optional source-code profile.
CODE_EXTENSIONS = {
    ".py",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".js",
    ".ts",
}


ALLOWED_EXTENSIONS = (
    DOCUMENT_EXTENSIONS
    | STRUCTURED_EXTENSIONS
    | CODE_EXTENSIONS
)


# =========================================================
# Scan limits
# =========================================================

# Skip very large files in computer-wide discovery.
# Explicit folder indexing can still be handled separately.
DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


# =========================================================
# Ignored directories
# =========================================================

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    "AppData",
    ".cache",
    ".pytest_cache",
    ".npm",
    ".gradle",
    ".idea",
    ".vscode",
    "$Recycle.Bin",
    "System Volume Information",
}


# =========================================================
# File hashing
# =========================================================

def calculate_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# Ignore rules
# =========================================================

def should_ignore_path(path: Path) -> bool:
    ignored_lower = {
        directory.lower()
        for directory in IGNORED_DIRECTORIES
    }

    return any(
        part.lower() in ignored_lower
        for part in path.parts
    )


def is_inside_recall_project(path: Path) -> bool:
    """
    Prevent Recall from indexing its own source/debug files
    during default computer-wide discovery.
    """

    try:
        recall_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        resolved_path = path.resolve()

        return (
            resolved_path == recall_root
            or recall_root in resolved_path.parents
        )

    except OSError:
        return False


# =========================================================
# File metadata
# =========================================================

def build_file_record(
    path: Path,
    calculate_hash: bool = True,
) -> Dict:
    stat = path.stat()

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_at": datetime.fromtimestamp(
            stat.st_ctime
        ).isoformat(),
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(),
        "content_hash": (
            calculate_file_hash(path)
            if calculate_hash
            else None
        ),
    }


# =========================================================
# Single-directory scanner
# =========================================================

def scan_directory(
    root_path: str,
    calculate_hash: bool = True,
) -> List[Dict]:
    """
    Explicit folder indexing.

    Supports all Recall file types.
    """

    root = Path(root_path).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {root}"
        )

    files = []

    for current_root, directories, filenames in os.walk(
        root,
        topdown=True,
    ):
        current_path = Path(current_root)

        directories[:] = [
            directory
            for directory in directories
            if not should_ignore_path(
                current_path / directory
            )
        ]

        for filename in filenames:
            path = current_path / filename

            if should_ignore_path(path):
                continue

            if (
                path.suffix.lower()
                not in ALLOWED_EXTENSIONS
            ):
                continue

            try:
                files.append(
                    build_file_record(
                        path,
                        calculate_hash=calculate_hash,
                    )
                )

            except (
                PermissionError,
                OSError,
                FileNotFoundError,
            ):
                continue

    return files


# =========================================================
# Default computer locations
# =========================================================

def get_default_scan_locations() -> List[Path]:
    home = Path.home()

    candidate_locations = [
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
        home / "OneDrive",
    ]

    locations = []
    seen_locations = set()

    for location in candidate_locations:
        if not location.exists():
            continue

        if not location.is_dir():
            continue

        try:
            resolved = location.resolve()

        except OSError:
            continue

        normalized_location = str(
            resolved
        ).lower()

        if normalized_location in seen_locations:
            continue

        seen_locations.add(
            normalized_location
        )

        locations.append(
            resolved
        )

    return locations


# =========================================================
# Computer-wide scanner
# =========================================================

def scan_computer(
    calculate_hash: bool = False,
    include_code: bool = False,
    include_structured: bool = False,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> List[Dict]:
    """
    Discover useful user files across Recall's default
    computer locations.

    Default:
        pdf, docx, txt, md

    Optional:
        structured files
        source code

    Recall's own project directory and oversized files are
    excluded from computer-wide discovery.
    """

    files = []
    seen_paths = set()

    allowed_extensions = set(
        DOCUMENT_EXTENSIONS
    )

    if include_structured:
        allowed_extensions.update(
            STRUCTURED_EXTENSIONS
        )

    if include_code:
        allowed_extensions.update(
            CODE_EXTENSIONS
        )

    for location in get_default_scan_locations():
        location_files = scan_directory(
            str(location),
            calculate_hash=calculate_hash,
        )

        for file_data in location_files:
            path = Path(
                file_data["file_path"]
            )

            if is_inside_recall_project(path):
                continue

            if (
                file_data["extension"]
                not in allowed_extensions
            ):
                continue

            if (
                file_data["size_bytes"]
                > max_file_size_bytes
            ):
                continue

            normalized_path = (
                file_data["file_path"].lower()
            )

            if normalized_path in seen_paths:
                continue

            seen_paths.add(
                normalized_path
            )

            files.append(
                file_data
            )

    return files