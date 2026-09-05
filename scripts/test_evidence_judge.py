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


from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from recall.evidence_judge import (
    judge_evidence_unit,
)


CHAT_MODEL_NAME = (
    "qwen3.5-2b"
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
    query = (
        "What formal AI training "
        "has this person completed?"
    )

    FoundryLocalManager.initialize(
        Configuration(
            app_name="Recall"
        )
    )

    manager = (
        FoundryLocalManager.instance
    )

    chat_model = (
        manager.catalog.get_model(
            CHAT_MODEL_NAME
        )
    )

    chat_model.load()

    chat_client = (
        chat_model
        .get_chat_client()
    )

    for index, text in enumerate(
        TEST_ITEMS,
        start=1,
    ):
        label = judge_evidence_unit(
            chat_client=chat_client,
            query=query,
            evidence_text=text,
        )

        print(
            f"\nItem {index}"
        )

        print(
            f"Label: {label}"
        )

        print(
            f"Text: {text}"
        )

        print(
            "-" * 70
        )

    chat_model.unload()


if __name__ == "__main__":
    main()