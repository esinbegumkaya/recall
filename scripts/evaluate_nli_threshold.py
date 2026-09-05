from pathlib import Path
import sys
import json


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


DATASET_PATH = (
    Path("data")
    / "evaluation"
    / "evidence_judge_queries.json"
)


THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


def calculate_metrics(
    examples,
    threshold,
):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for example in examples:
        predicted = (
            "SUPPORTS"
            if example[
                "entailment_score"
            ] >= threshold
            else "DOES_NOT_SUPPORT"
        )

        expected = example[
            "label"
        ]

        if (
            predicted == "SUPPORTS"
            and expected == "SUPPORTS"
        ):
            tp += 1

        elif (
            predicted == "DOES_NOT_SUPPORT"
            and expected == "DOES_NOT_SUPPORT"
        ):
            tn += 1

        elif (
            predicted == "SUPPORTS"
            and expected == "DOES_NOT_SUPPORT"
        ):
            fp += 1

        elif (
            predicted == "DOES_NOT_SUPPORT"
            and expected == "SUPPORTS"
        ):
            fn += 1

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(
            file
        )

    judge = LocalNLIJudge()

    scored_examples = []

    print(
        "\nSCORING EVIDENCE\n"
    )

    for example in dataset:
        result = judge.judge(
            query=example["query"],
            evidence_text=(
                example["evidence"]
            ),
        )

        scored_example = dict(
            example
        )

        scored_example[
            "entailment_score"
        ] = result[
            "entailment_score"
        ]

        scored_examples.append(
            scored_example
        )

        print(
            f"{example['id']} | "
            f"Expected: "
            f"{example['label']} | "
            f"Entailment: "
            f"{result['entailment_score']:.4f}"
        )

    print(
        "\nTHRESHOLD SWEEP\n"
    )

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<6}"
        f"{'FN':<6}"
    )

    print(
        "-" * 72
    )

    for threshold in THRESHOLDS:
        metrics = (
            calculate_metrics(
                scored_examples,
                threshold,
            )
        )

        print(
            f"{threshold:<12.2f}"
            f"{metrics['accuracy']:<12.3f}"
            f"{metrics['precision']:<12.3f}"
            f"{metrics['recall']:<12.3f}"
            f"{metrics['f1']:<12.3f}"
            f"{metrics['fp']:<6}"
            f"{metrics['fn']:<6}"
        )


if __name__ == "__main__":
    main()