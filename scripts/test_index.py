from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from recall.crawler import scan_directory
from recall.database import initialize_database, upsert_file


def main():
    initialize_database()

    files = scan_directory("data/samples")

    for file in files:
        upsert_file(file)

        print(
            f"Indexed: {file['file_name']} "
            f"{file['content_hash'][:12]}..."
        )

    print(f"\nIndexed {len(files)} files successfully.")


if __name__ == "__main__":
    main()