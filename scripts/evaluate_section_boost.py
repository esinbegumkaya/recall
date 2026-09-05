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


from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from recall.database import (
    initialize_database,
    get_chunks_with_embeddings,
)

from recall.retriever import (
    dense_retrieve,
)

from recall.intent import (
    detect_section_intent,
)


MODEL_NAME = "qwen3-embedding-0.6b"

DEVELOPMENT_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)

VALIDATION_FILE = Path(
    "data/evaluation/validation_queries.json"
)

TOP_K = 10


BOOST_VALUES = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
]


def load_queries(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def is_relevant(
    result,
    relevant_items,
):
    result_file_name = Path(
        result["file_path"]
    ).name

    result_section = result.get(
        "section_name"
    )

    for relevant in relevant_items:
        expected_file = relevant[
            "file_name"
        ]

        expected_section = relevant.get(
            "section_name"
        )

        if result_file_name != expected_file:
            continue

        if expected_section is None:
            return True

        if result_section == expected_section:
            return True

    return False


def reciprocal_rank(
    results,
    relevant_items,
):
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if is_relevant(
            result,
            relevant_items,
        ):
            return 1.0 / rank

    return 0.0


def hit_at_k(
    results,
    relevant_items,
    k,
):
    return int(
        any(
            is_relevant(
                result,
                relevant_items,
            )
            for result in results[:k]
        )
    )


def evaluate_results(
    results,
    relevant_items,
):
    return {
        "hit@1": hit_at_k(
            results,
            relevant_items,
            1,
        ),
        "hit@3": hit_at_k(
            results,
            relevant_items,
            3,
        ),
        "hit@5": hit_at_k(
            results,
            relevant_items,
            5,
        ),
        "rr": reciprocal_rank(
            results,
            relevant_items,
        ),
    }


def create_metrics():
    return {
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
        "mrr": 0.0,
    }


def add_metrics(
    totals,
    current,
):
    totals["hit@1"] += current[
        "hit@1"
    ]

    totals["hit@3"] += current[
        "hit@3"
    ]

    totals["hit@5"] += current[
        "hit@5"
    ]

    totals["mrr"] += current[
        "rr"
    ]


def average_metrics(
    metrics,
    query_count,
):
    return {
        key: value / query_count
        for key, value in metrics.items()
    }


def apply_section_boost(
    query,
    results,
    boost,
):
    intended_section = (
        detect_section_intent(
            query
        )
    )

    boosted_results = []

    for result in results:
        updated = dict(
            result
        )

        section_match = (
            intended_section is not None
            and result.get(
                "section_name"
            ) == intended_section
        )

        updated[
            "section_match"
        ] = section_match

        updated[
            "final_score"
        ] = (
            result[
                "semantic_score"
            ]
            + (
                boost
                if section_match
                else 0.0
            )
        )

        boosted_results.append(
            updated
        )

    boosted_results.sort(
        key=lambda item: item[
            "final_score"
        ],
        reverse=True,
    )

    return boosted_results


def get_query_embedding(
    embedding_client,
    query,
):
    response = (
        embedding_client
        .generate_embedding(
            query
        )
    )

    return (
        response
        .data[0]
        .embedding
    )


def evaluate_boost_values(
    queries,
    rows,
    embedding_client,
):
    metrics_by_boost = {
        boost: create_metrics()
        for boost in BOOST_VALUES
    }

    for item in queries:
        query = item["query"]

        relevant_items = item[
            "relevant"
        ]

        query_embedding = (
            get_query_embedding(
                embedding_client,
                query,
            )
        )

        dense_results = dense_retrieve(
            query_embedding=(
                query_embedding
            ),
            rows=rows,
            top_k=len(rows),
        )

        for boost in BOOST_VALUES:
            boosted_results = (
                apply_section_boost(
                    query,
                    dense_results,
                    boost,
                )
            )

            metrics = evaluate_results(
                boosted_results[:TOP_K],
                relevant_items,
            )

            add_metrics(
                metrics_by_boost[
                    boost
                ],
                metrics,
            )

    query_count = len(
        queries
    )

    return {
        boost: average_metrics(
            metrics,
            query_count,
        )
        for boost, metrics
        in metrics_by_boost.items()
    }


def print_results(
    title,
    results,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"{'Boost':>10}"
        f"{'Hit@1':>12}"
        f"{'Hit@3':>12}"
        f"{'Hit@5':>12}"
        f"{'MRR':>12}"
    )

    print(
        "-" * 58
    )

    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1]["mrr"],
            item[1]["hit@1"],
            item[1]["hit@3"],
            item[1]["hit@5"],
        ),
        reverse=True,
    )

    for boost, metrics in ranked:
        print(
            f"{boost:>10.3f}"
            f"{metrics['hit@1']:>12.3f}"
            f"{metrics['hit@3']:>12.3f}"
            f"{metrics['hit@5']:>12.3f}"
            f"{metrics['mrr']:>12.3f}"
        )

    return ranked


def inspect_validation_queries(
    queries,
    rows,
    embedding_client,
    selected_boost,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION PER-QUERY ANALYSIS"
    )

    print(
        "=" * 70
    )

    for item in queries:
        query = item["query"]

        query_embedding = (
            get_query_embedding(
                embedding_client,
                query,
            )
        )

        dense_results = dense_retrieve(
            query_embedding=(
                query_embedding
            ),
            rows=rows,
            top_k=len(rows),
        )

        boosted_results = (
            apply_section_boost(
                query,
                dense_results,
                selected_boost,
            )
        )

        dense_rr = reciprocal_rank(
            dense_results,
            item["relevant"],
        )

        boosted_rr = reciprocal_rank(
            boosted_results,
            item["relevant"],
        )

        intent = detect_section_intent(
            query
        )

        print(
            f"\n{item['id']}: "
            f"{query}"
        )

        print(
            f"  Intent: {intent}"
        )

        print(
            f"  Dense RR: "
            f"{dense_rr:.3f}"
        )

        print(
            f"  Boosted RR: "
            f"{boosted_rr:.3f}"
        )

        if boosted_results:
            top = boosted_results[0]

            print(
                f"  Top result: "
                f"{Path(top['file_path']).name}"
                f" | "
                f"{top['section_name']}"
            )


def main():
    initialize_database()

    development_queries = (
        load_queries(
            DEVELOPMENT_FILE
        )
    )

    validation_queries = (
        load_queries(
            VALIDATION_FILE
        )
    )

    rows = list(
        get_chunks_with_embeddings(
            MODEL_NAME
        )
    )

    if not rows:
        print(
            "No indexed chunks found."
        )
        return

    print(
        f"Loaded "
        f"{len(development_queries)} "
        f"development queries."
    )

    print(
        f"Loaded "
        f"{len(validation_queries)} "
        f"validation queries."
    )

    print(
        f"Loaded "
        f"{len(rows)} indexed chunks."
    )

    print(
        "\nLoading embedding model..."
    )

    FoundryLocalManager.initialize(
        Configuration(
            app_name="Recall"
        )
    )

    manager = (
        FoundryLocalManager.instance
    )

    embedding_model = (
        manager.catalog.get_model(
            MODEL_NAME
        )
    )

    embedding_model.load()

    embedding_client = (
        embedding_model
        .get_embedding_client()
    )

    development_results = (
        evaluate_boost_values(
            development_queries,
            rows,
            embedding_client,
        )
    )

    ranked_development = (
        print_results(
            "DEVELOPMENT - SECTION BOOST SWEEP",
            development_results,
        )
    )

    selected_boost = (
        ranked_development[0][0]
    )

    print(
        "\nSELECTED BOOST FROM DEVELOPMENT"
    )

    print(
        f"Boost: {selected_boost:.3f}"
    )

    validation_results = (
        evaluate_boost_values(
            validation_queries,
            rows,
            embedding_client,
        )
    )

    print_results(
        "VALIDATION - ALL BOOSTS "
        "(DIAGNOSTIC ONLY)",
        validation_results,
    )

    inspect_validation_queries(
        validation_queries,
        rows,
        embedding_client,
        selected_boost,
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Validation sweep is diagnostic only. "
        "Do not choose a new boost from "
        "validation results."
    )

    embedding_model.unload()


if __name__ == "__main__":
    main()