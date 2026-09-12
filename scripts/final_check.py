"""Final delivery smoke-check for Recall / MemoRAG.

Run from the repository root:
    python scripts/final_check.py

This verifies the Python environment, project imports, local database presence,
and then executes the regression suite. It intentionally does not fake Foundry
Local model availability: model-backed tests must pass on the delivery machine.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_IMPORTS = {
    "foundry_local_sdk": "foundry-local-sdk-winml",
    "pypdf": "pypdf",
    "docx": "python-docx",
    "rapidfuzz": "RapidFuzz",
    "streamlit": "streamlit",
    "torch": "torch",
    "transformers": "transformers",
}


def main() -> int:
    print("Recall final delivery check")
    print("=" * 27)

    missing: list[str] = []
    for module, package in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module)
            print(f"[OK] dependency: {package}")
        except Exception as exc:
            missing.append(package)
            print(f"[FAIL] dependency: {package} ({type(exc).__name__}: {exc})")

    if missing:
        print("\nInstall dependencies first:")
        print("  python -m pip install -r requirements.txt")
        return 2

    try:
        from recall.engine import RecallEngine  # noqa: F401
        print("[OK] recall.engine import")
    except Exception as exc:
        print(f"[FAIL] recall.engine import ({type(exc).__name__}: {exc})")
        return 3

    db_path = ROOT / "data" / "index" / "recall.db"
    if db_path.exists():
        print(f"[OK] local index: {db_path.relative_to(ROOT)}")
    else:
        print("[WARN] no local index found; index sample/user files before the demo")

    print("\nRunning regression tests...")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        check=False,
    )

    if completed.returncode != 0:
        print("\n[FAIL] Regression suite did not pass.")
        return completed.returncode

    print("\n[OK] Final delivery smoke-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
