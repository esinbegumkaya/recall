from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_NLI_MODEL = (
    "cross-encoder/nli-deberta-v3-small"
)


class LocalNLIJudge:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
    ):
        self.model_name = model_name

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.eval()

        self.id2label = {
            int(index): label.lower()
            for index, label
            in self.model.config.id2label.items()
        }

        print(
            "NLI labels:",
            self.id2label,
        )


    def build_hypothesis(
        self,
        query: str,
    ) -> str:
        normalized = (
            query
            .strip()
            .rstrip("?")
        )

        lower_query = normalized.lower()

        # AI training
        if (
            "formal ai training"
            in lower_query
            or "artificial intelligence training"
            in lower_query
            or "ai training"
            in lower_query
        ):
            return (
                "This item is artificial "
                "intelligence training."
            )

        # Cloud platforms
        if (
            "cloud platform"
            in lower_query
            or "cloud platforms"
            in lower_query
        ):
            return (
                "This item involves "
                "a cloud platform."
            )

        # Machine learning projects
        if (
            "project"
            in lower_query
            and "machine learning"
            in lower_query
        ):
            return (
                "This item involves "
                "machine learning."
            )

        # AI agents
        if (
            "ai agent"
            in lower_query
            or "ai agents"
            in lower_query
            or "agentic ai"
            in lower_query
        ):
            return (
                "This item involves AI agents."
            )

        # Programming languages
        if (
            "programming language"
            in lower_query
            or "programming languages"
            in lower_query
        ):
            return (
                "This item contains "
                "programming languages."
            )

        # Data visualization
        if (
            "data visualization"
            in lower_query
        ):
            return (
                "This item involves "
                "data visualization."
            )

        # Professional AI experience
        if (
            "professional ai experience"
            in lower_query
            or (
                "professional"
                in lower_query
                and "ai experience"
                in lower_query
            )
        ):
            return (
                "This item involves "
                "artificial intelligence."
            )

        return (
            "This evidence directly supports "
            "the topic requested "
            "by the question."
        )


    def judge(
        self,
        query: str,
        evidence_text: str,
    ) -> Dict:
        hypothesis = (
            self.build_hypothesis(
                query
            )
        )

        inputs = (
            self.tokenizer(
                evidence_text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        )

        with torch.no_grad():
            output = self.model(
                **inputs
            )

        probabilities = (
            torch.softmax(
                output.logits,
                dim=-1,
            )[0]
        )

        scores = {}

        for index, probability in enumerate(
            probabilities
        ):
            label = self.id2label.get(
                index,
                str(index),
            )

            scores[label] = float(
                probability
            )

        entailment_score = 0.0

        for label, score in scores.items():
            if "entail" in label:
                entailment_score = score
                break

        return {
            "label": (
                "SUPPORTS"
                if entailment_score >= 0.5
                else "DOES_NOT_SUPPORT"
            ),
            "entailment_score": (
                entailment_score
            ),
            "scores": scores,
            "hypothesis": hypothesis,
        }

    def judge_claim(
        self,
        claim: str,
        evidence_text: str,
    ) -> Dict:
        inputs = (
            self.tokenizer(
                evidence_text,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        )

        with torch.no_grad():
            output = self.model(
                **inputs
            )

        probabilities = (
            torch.softmax(
                output.logits,
                dim=-1,
            )[0]
        )

        scores = {}

        for index, probability in enumerate(
            probabilities
        ):
            label = self.id2label.get(
                index,
                str(index),
            )

            scores[label] = float(
                probability
            )

        entailment_score = 0.0

        for label, score in scores.items():
            if "entail" in label:
                entailment_score = score
                break

        return {
            "label": (
                "SUPPORTS"
                if entailment_score >= 0.5
                else "DOES_NOT_SUPPORT"
            ),
            "entailment_score": (
                entailment_score
            ),
            "scores": scores,
            "hypothesis": claim,
        }

