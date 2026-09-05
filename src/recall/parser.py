from pathlib import Path
import re

from pypdf import PdfReader
from docx import Document


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
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


COMMON_SECTION_TITLES = {
    "EDUCATION",
    "EXPERIENCE",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "PROJECTS",
    "SKILLS",
    "TECHNICAL SKILLS",
    "SKILLS, LANGUAGES & INTERESTS",
    "CERTIFICATIONS",
    "CERTIFICATES",
    "LANGUAGES",
    "INTERESTS",
    "PUBLICATIONS",
    "ACHIEVEMENTS",
    "AWARDS",
    "VOLUNTEERING",
    "SUMMARY",
    "PROFILE",
}


def read_plain_text_file(path: Path) -> str:
    raw = path.read_bytes()

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")

    try:
        return raw.decode("utf-8")

    except UnicodeDecodeError:
        try:
            return raw.decode("cp1254")

        except UnicodeDecodeError:
            return raw.decode(
                "utf-8",
                errors="replace",
            )


def looks_like_section_title(line: str) -> bool:
    cleaned = line.strip()

    if not cleaned:
        return False

    normalized = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    normalized_without_colon = (
        normalized.rstrip(":").strip()
    )

    upper_version = (
        normalized_without_colon.upper()
    )

    # Known section names are the strongest signal.
    if upper_version in COMMON_SECTION_TITLES:
        return True

    # Avoid treating long content lines as headings.
    if len(normalized_without_colon) > 60:
        return False

    words = normalized_without_colon.split()

    if not (1 <= len(words) <= 5):
        return False

    # There must actually be letters.
    if not any(
        character.isalpha()
        for character in normalized_without_colon
    ):
        return False

    # Important:
    # Check the ORIGINAL text, not a version
    # that was already converted to uppercase.
    letters = "".join(
        character
        for character in normalized_without_colon
        if character.isalpha()
    )

    if not letters:
        return False

    return letters == letters.upper()


def detect_sections(text: str):
    lines = text.splitlines()

    sections = []

    current_section = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if looks_like_section_title(stripped):
            if current_lines:
                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:
                    sections.append(
                        {
                            "section_name": current_section,
                            "text": section_text,
                        }
                    )

            current_section = stripped.rstrip(":")
            current_lines = []
            continue

        current_lines.append(line)

    if current_lines:
        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:
            sections.append(
                {
                    "section_name": current_section,
                    "text": section_text,
                }
            )

    return sections


def read_pdf_pages(file_path: str):
    path = Path(file_path)

    reader = PdfReader(path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        sections = detect_sections(text)

        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "sections": sections,
            }
        )

    return pages


def read_pdf_file(path: Path) -> str:
    pages = read_pdf_pages(str(path))

    return "\n\n".join(
        page["text"]
        for page in pages
    )


def read_docx_file(path: Path) -> str:
    document = Document(path)

    paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def read_text_file(file_path: str) -> str:
    path = Path(file_path)

    extension = path.suffix.lower()

    if extension in TEXT_EXTENSIONS:
        return read_plain_text_file(path)

    if extension == ".pdf":
        return read_pdf_file(path)

    if extension == ".docx":
        return read_docx_file(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )