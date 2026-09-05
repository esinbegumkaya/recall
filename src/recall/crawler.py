from pathlib import Path
from typing import List, Dict
from datetime import datetime
import hashlib


ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".py",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "AppData",
}


def calculate_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def scan_directory(root_path: str) -> List[Dict]:
    root = Path(root_path)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in path.parts):
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        stat = path.stat()

        files.append(
            {
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
                "content_hash": calculate_file_hash(path),
            }
        )

    return files