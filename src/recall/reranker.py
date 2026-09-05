from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "BAAI/bge-reranker-base"
)


class LocalReranker:
    def __init__(
        self,
        model_name=DEFAULT_RERANKER_MODEL,
    ):
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name
        )


    def rerank(
        self,
        query,
        results,
        top_k=5,
    ):
        if not results:
            return []

        pairs = [
            [
                query,
                result["chunk_text"],
            ]
            for result in results
        ]

        scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for result, score in zip(
            results,
            scores,
        ):
            updated_result = dict(
                result
            )

            updated_result[
                "reranker_score"
            ] = float(score)

            reranked_results.append(
                updated_result
            )

        reranked_results.sort(
            key=lambda item: item[
                "reranker_score"
            ],
            reverse=True,
        )

        return reranked_results[
            :top_k
        ]