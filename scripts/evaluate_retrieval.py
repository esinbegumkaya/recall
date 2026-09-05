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
    hybrid_retrieve,
    normalize_scores,
)


MODEL_NAME = "qwen3-embedding-0.6b"

DEVELOPMENT_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)

VALIDATION_FILE = Path(
    "data/evaluation/validation_queries.json"
)

TOP_K = 10


WEIGHT_CONFIGURATIONS = [
    (1.00, 0.00),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.30, 0.70),
]


def load_queries(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}"
        )

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
    for result in results[:k]:
        if is_relevant(
            result,
            relevant_items,
        ):
            return 1

    return 0


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
    if query_count == 0:
        return create_metrics()

    return {
        key: value / query_count
        for key, value in metrics.items()
    }


def apply_weights(
    hybrid_results,
    semantic_weight,
    bm25_weight,
):
    if not hybrid_results:
        return []

    semantic_scores = [
        result["semantic_score"]
        for result in hybrid_results
    ]

    bm25_scores = [
        result["bm25_score"]
        for result in hybrid_results
    ]

    normalized_semantic = (
        normalize_scores(
            semantic_scores
        )
    )

    normalized_bm25 = (
        normalize_scores(
            bm25_scores
        )
    )

    weighted_results = []

    for index, result in enumerate(
        hybrid_results
    ):
        updated = dict(
            result
        )

        updated[
            "experimental_score"
        ] = (
            semantic_weight
            * normalized_semantic[index]
            + bm25_weight
            * normalized_bm25[index]
        )

        weighted_results.append(
            updated
        )

    weighted_results.sort(
        key=lambda item: item[
            "experimental_score"
        ],
        reverse=True,
    )

    return weighted_results


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


def build_candidate_results(
    query,
    query_embedding,
    rows,
):
    dense_results = dense_retrieve(
        query_embedding=query_embedding,
        rows=rows,
        top_k=len(rows),
    )

    hybrid_results = hybrid_retrieve(
        query=query,
        query_embedding=query_embedding,
        rows=rows,
        top_k=len(rows),
    )

    return (
        dense_results,
        hybrid_results,
    )


def run_development_sweep(
    queries,
    rows,
    embedding_client,
):
    metrics_by_weight = {
        configuration: create_metrics()
        for configuration
        in WEIGHT_CONFIGURATIONS
    }

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DEVELOPMENT SET - WEIGHT SWEEP"
    )

    print(
        "=" * 78
    )

    for item in queries:
        query_id = item["id"]
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

        (
            dense_results,
            hybrid_results,
        ) = build_candidate_results(
            query,
            query_embedding,
            rows,
        )

        print(
            f"\n{query_id}: {query}"
        )

        for (
            semantic_weight,
            bm25_weight,
        ) in WEIGHT_CONFIGURATIONS:

            if bm25_weight == 0.0:
                results = dense_results[
                    :TOP_K
                ]

            else:
                results = apply_weights(
                    hybrid_results,
                    semantic_weight,
                    bm25_weight,
                )[:TOP_K]

            current_metrics = (
                evaluate_results(
                    results,
                    relevant_items,
                )
            )

            configuration = (
                semantic_weight,
                bm25_weight,
            )

            add_metrics(
                metrics_by_weight[
                    configuration
                ],
                current_metrics,
            )

            print(
                f"  "
                f"{semantic_weight:.2f}/"
                f"{bm25_weight:.2f}"
                f" -> RR "
                f"{current_metrics['rr']:.3f}"
            )

    query_count = len(
        queries
    )

    averaged_results = {
        configuration: average_metrics(
            metrics,
            query_count,
        )
        for configuration, metrics
        in metrics_by_weight.items()
    }

    ranked_configurations = sorted(
        averaged_results.items(),
        key=lambda item: (
            item[1]["mrr"],
            item[1]["hit@1"],
            item[1]["hit@3"],
            item[1]["hit@5"],
        ),
        reverse=True,
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "DEVELOPMENT RESULTS"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"{'Semantic':>10}"
        f"{'BM25':>10}"
        f"{'Hit@1':>12}"
        f"{'Hit@3':>12}"
        f"{'Hit@5':>12}"
        f"{'MRR':>12}"
    )

    print(
        "-" * 78
    )

    for (
        semantic_weight,
        bm25_weight,
    ), metrics in ranked_configurations:

        print(
            f"{semantic_weight:>10.2f}"
            f"{bm25_weight:>10.2f}"
            f"{metrics['hit@1']:>12.3f}"
            f"{metrics['hit@3']:>12.3f}"
            f"{metrics['hit@5']:>12.3f}"
            f"{metrics['mrr']:>12.3f}"
        )

    best_configuration = (
        ranked_configurations[0][0]
    )

    return best_configuration


def run_validation(
    queries,
    rows,
    embedding_client,
    best_configuration,
):
    (
        semantic_weight,
        bm25_weight,
    ) = best_configuration

    dense_metrics = create_metrics()

    selected_hybrid_metrics = (
        create_metrics()
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "VALIDATION SET"
    )

    print(
        "=" * 78
    )

    print(
        "\nSelected development configuration:"
    )

    print(
        f"Semantic = {semantic_weight:.2f}, "
        f"BM25 = {bm25_weight:.2f}\n"
    )

    for item in queries:
        query_id = item["id"]
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

        (
            dense_results,
            hybrid_results,
        ) = build_candidate_results(
            query,
            query_embedding,
            rows,
        )

        dense_results = dense_results[
            :TOP_K
        ]

        if bm25_weight == 0.0:
            selected_results = (
                dense_results
            )

        else:
            selected_results = (
                apply_weights(
                    hybrid_results,
                    semantic_weight,
                    bm25_weight,
                )[:TOP_K]
            )

        dense_current = (
            evaluate_results(
                dense_results,
                relevant_items,
            )
        )

        hybrid_current = (
            evaluate_results(
                selected_results,
                relevant_items,
            )
        )

        add_metrics(
            dense_metrics,
            dense_current,
        )

        add_metrics(
            selected_hybrid_metrics,
            hybrid_current,
        )

        print(
            f"{query_id}: {query}"
        )

        print(
            f"  Dense RR: "
            f"{dense_current['rr']:.3f}"
        )

        print(
            f"  Selected hybrid RR: "
            f"{hybrid_current['rr']:.3f}"
        )

        print()

    query_count = len(
        queries
    )

    dense_metrics = average_metrics(
        dense_metrics,
        query_count,
    )

    selected_hybrid_metrics = (
        average_metrics(
            selected_hybrid_metrics,
            query_count,
        )
    )

    print(
        "=" * 78
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 78
    )

    print()

    print(
        f"{'Pipeline':<30}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'MRR':>10}"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Dense':<30}"
        f"{dense_metrics['hit@1']:>10.3f}"
        f"{dense_metrics['hit@3']:>10.3f}"
        f"{dense_metrics['hit@5']:>10.3f}"
        f"{dense_metrics['mrr']:>10.3f}"
    )

    hybrid_label = (
        f"Hybrid "
        f"{semantic_weight:.2f}/"
        f"{bm25_weight:.2f}"
    )

    print(
        f"{hybrid_label:<30}"
        f"{selected_hybrid_metrics['hit@1']:>10.3f}"
        f"{selected_hybrid_metrics['hit@3']:>10.3f}"
        f"{selected_hybrid_metrics['hit@5']:>10.3f}"
        f"{selected_hybrid_metrics['mrr']:>10.3f}"
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

    best_configuration = (
        run_development_sweep(
            development_queries,
            rows,
            embedding_client,
        )
    )

    print(
        "\nBEST DEVELOPMENT CONFIGURATION"
    )

    print(
        f"Semantic: "
        f"{best_configuration[0]:.2f}"
    )

    print(
        f"BM25: "
        f"{best_configuration[1]:.2f}"
    )

    run_validation(
        validation_queries,
        rows,
        embedding_client,
        best_configuration,
    )

    embedding_model.unload()


if __name__ == "__main__":
    main()