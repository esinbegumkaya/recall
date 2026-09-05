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


from recall.nli_judge import (
    LocalNLIJudge,
)


QUERY = (
    "What formal AI training "
    "has this person completed?"
)


TEST_ITEMS = [
    (
        "Microsoft AI Innovators Program "
        "(2026) — Azure AI Foundry, "
        "local LLMs, AI agents, "
        "RAG pipelines."
    ),
    (
        "Data Analyst Boostcamp — "
        "Istanbul Data Science Academy "
        "& Google Cloud (2024) — "
        "Data analysis techniques focused "
        "on Google Cloud."
    ),
    (
        "Huawei Cloud HCCDA–AI "
        "(2025) — fine-tuning LLMs, "
        "RAG-based search."
    ),
    (
        "Turkcell Geleceği Yazan Kadınlar "
        "Yapay Zeka Programı (2024) — "
        "Machine Learning and "
        "Data Science & AI."
    ),
    (
        "NTT DATA Data Visualization "
        "Program (2024) — "
        "Data visualization using Power BI."
    ),
    (
        "PCAP: Programming Essentials "
        "in Python — OpenEDG Python "
        "Institute (2020)."
    ),
]


def main():
    judge = LocalNLIJudge()

    for index, evidence in enumerate(
        TEST_ITEMS,
        start=1,
    ):
        result = judge.judge(
            query=QUERY,
            evidence_text=evidence,
        )

        print()
        print(
            f"Item {index}"
        )

        print(
            f"Label: "
            f"{result['label']}"
        )

        print(
            f"Entailment: "
            f"{result['entailment_score']:.4f}"
        )

        print(
            f"All scores: "
            f"{result['scores']}"
        )

        print(
            f"Text: {evidence}"
        )

        print(
            "-" * 70
        )


if __name__ == "__main__":
    main()