from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from recall.crawler import scan_directory


def main():
    folder = input("Folder to scan: ").strip()

    files = scan_directory(folder)

    print(f"\nFound {len(files)} supported files.\n")

    for file in files[:20]:
        print(
            f"{file['extension']:8} "
            f"{file['size_bytes']:10} bytes  "
            f"{file['file_name']}"
        )


if __name__ == "__main__":
    main()