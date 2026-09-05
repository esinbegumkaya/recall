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
    section_aware_dense_retrieve,
)

from recall.intent import (
    detect_section_intent,
    detect_query_mode,
)


MODEL_NAME = "qwen3-embedding-0.6b"

DEVELOPMENT_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)

VALIDATION_FILE = Path(
    "data/evaluation/validation_queries.json"
)

SECTION_BOOST = 0.075
HEADING_PENALTY = 0.05


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
    count,
):
    return {
        key: value / count
        for key, value in metrics.items()
    }


def evaluate_pipeline(
    queries,
    rows,
    embedding_client,
):
    dense_totals = create_metrics()
    structural_totals = create_metrics()

    per_query = []

    for item in queries:
        query = item["query"]

        response = (
            embedding_client
            .generate_embedding(
                query
            )
        )

        query_embedding = (
            response
            .data[0]
            .embedding
        )

        dense_results = dense_retrieve(
            query_embedding=query_embedding,
            rows=rows,
            top_k=len(rows),
        )

        structural_results = (
            section_aware_dense_retrieve(
                query=query,
                query_embedding=query_embedding,
                rows=rows,
                detect_section_intent=(
                    detect_section_intent
                ),
                detect_query_mode=(
                    detect_query_mode
                ),
                top_k=len(rows),
                section_boost=SECTION_BOOST,
                heading_penalty=(
                    HEADING_PENALTY
                ),
            )
        )

        dense_current = {
            "hit@1": hit_at_k(
                dense_results,
                item["relevant"],
                1,
            ),
            "hit@3": hit_at_k(
                dense_results,
                item["relevant"],
                3,
            ),
            "hit@5": hit_at_k(
                dense_results,
                item["relevant"],
                5,
            ),
            "rr": reciprocal_rank(
                dense_results,
                item["relevant"],
            ),
        }

        structural_current = {
            "hit@1": hit_at_k(
                structural_results,
                item["relevant"],
                1,
            ),
            "hit@3": hit_at_k(
                structural_results,
                item["relevant"],
                3,
            ),
            "hit@5": hit_at_k(
                structural_results,
                item["relevant"],
                5,
            ),
            "rr": reciprocal_rank(
                structural_results,
                item["relevant"],
            ),
        }

        add_metrics(
            dense_totals,
            dense_current,
        )

        add_metrics(
            structural_totals,
            structural_current,
        )

        per_query.append(
            {
                "id": item["id"],
                "query": query,
                "mode": detect_query_mode(
                    query
                ),
                "intent": detect_section_intent(
                    query
                ),
                "dense_rr": dense_current[
                    "rr"
                ],
                "structural_rr": (
                    structural_current[
                        "rr"
                    ]
                ),
                "dense_top": (
                    dense_results[0]
                    if dense_results
                    else None
                ),
                "structural_top": (
                    structural_results[0]
                    if structural_results
                    else None
                ),
            }
        )

    count = len(
        queries
    )

    return (
        average_metrics(
            dense_totals,
            count,
        ),
        average_metrics(
            structural_totals,
            count,
        ),
        per_query,
    )


def print_metrics(
    title,
    dense_metrics,
    structural_metrics,
):
    print(
        "\n"
        + "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )

    print(
        f"{'Pipeline':<32}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR':>10}"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Dense':<32}"
        f"{dense_metrics['hit@1']:>10.3f}"
        f"{dense_metrics['hit@3']:>10.3f}"
        f"{dense_metrics['hit@5']:>10.3f}"
        f"{dense_metrics['mrr']:>10.3f}"
    )

    print(
        f"{'Structural Dense':<32}"
        f"{structural_metrics['hit@1']:>10.3f}"
        f"{structural_metrics['hit@3']:>10.3f}"
        f"{structural_metrics['hit@5']:>10.3f}"
        f"{structural_metrics['mrr']:>10.3f}"
    )


def print_per_query(
    title,
    per_query,
):
    print(
        "\n"
        + "=" * 76
    )

    print(
        title
    )

    print(
        "=" * 76
    )

    for item in per_query:
        print(
            f"\n{item['id']}: "
            f"{item['query']}"
        )

        print(
            f"  Mode: "
            f"{item['mode']}"
        )

        print(
            f"  Intent: "
            f"{item['intent']}"
        )

        print(
            f"  Dense RR: "
            f"{item['dense_rr']:.3f}"
        )

        print(
            f"  Structural RR: "
            f"{item['structural_rr']:.3f}"
        )

        if item["dense_top"]:
            dense_top = item[
                "dense_top"
            ]

            print(
                f"  Dense top: "
                f"{Path(dense_top['file_path']).name}"
                f" | "
                f"{dense_top['section_name']}"
            )

        if item["structural_top"]:
            structural_top = item[
                "structural_top"
            ]

            print(
                f"  Structural top: "
                f"{Path(structural_top['file_path']).name}"
                f" | "
                f"{structural_top['section_name']}"
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

    (
        dev_dense,
        dev_structural,
        dev_per_query,
    ) = evaluate_pipeline(
        development_queries,
        rows,
        embedding_client,
    )

    print_metrics(
        "DEVELOPMENT RESULTS",
        dev_dense,
        dev_structural,
    )

    (
        val_dense,
        val_structural,
        val_per_query,
    ) = evaluate_pipeline(
        validation_queries,
        rows,
        embedding_client,
    )

    print_metrics(
        "HELD-OUT VALIDATION RESULTS",
        val_dense,
        val_structural,
    )

    print_per_query(
        "VALIDATION PER-QUERY ANALYSIS",
        val_per_query,
    )

    embedding_model.unload()


if __name__ == "__main__":
    main()