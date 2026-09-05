from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__)
        .resolve()
        .parents[1]
        / "src"
    )
)


from recall.evidence_gate import (
    detect_required_evidence_type,
    filter_by_evidence_type,
)


TEST_UNITS = [
    {
        "text": (
            "Microsoft AI Engineering Intern — "
            "worked with AI agents and RAG."
        ),
        "section_name": "EXPERIENCE",
    },
    {
        "text": (
            "Microsoft AI Innovators Program — "
            "AI agents and RAG."
        ),
        "section_name": "CERTIFICATES",
    },
    {
        "text": (
            "Skin Cancer Detection — "
            "CNN image classification."
        ),
        "section_name": "PROJECTS",
    },
    {
        "text": (
            "Programming Languages: "
            "Python, Java, C++."
        ),
        "section_name": (
            "SKILLS, LANGUAGES & INTERESTS"
        ),
    },
]


TEST_QUERIES = [
    (
        "Where did this person work "
        "with AI agents?"
    ),
    (
        "What formal AI training "
        "has this person completed?"
    ),
    (
        "What projects involved "
        "machine learning?"
    ),
    (
        "What programming languages "
        "does this person know?"
    ),
]


def main():
    for query in TEST_QUERIES:
        required_type = (
            detect_required_evidence_type(
                query
            )
        )

        filtered = (
            filter_by_evidence_type(
                query=query,
                evidence_units=TEST_UNITS,
            )
        )

        print()
        print("=" * 70)

        print(
            f"Query: {query}"
        )

        print(
            f"Required type: "
            f"{required_type}"
        )

        print(
            "Accepted evidence:"
        )

        for unit in filtered:
            print(
                f"- [{unit['section_name']}] "
                f"{unit['text']}"
            )


if __name__ == "__main__":
    main()