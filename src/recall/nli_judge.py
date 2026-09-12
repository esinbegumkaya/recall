import re
from typing import Dict

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


_LEADING_INTERROGATIVE_RE = re.compile(
    r"^(?:what|which|where|who|whom|when|how|why)\b\s*",
    re.IGNORECASE,
)

_LEADING_AUX_RE = re.compile(
    r"^(?:is|are|was|were|does|do|did|has|have|had|"
    r"can|could|will|would|should)\b\s+",
    re.IGNORECASE,
)

_LEADING_SUBJECT_RE = re.compile(
    r"^(?:this person's|this person|the candidate's|the candidate|"
    r"the person's|the person|the document|the documents|the text|"
    r"this document|it|this)\b\s*",
    re.IGNORECASE,
)

_TRAILING_BOILERPLATE_RE = re.compile(
    r"\s+(?:is|are)?\s*"
    r"(?:mentioned|discussed|covered|used|involved|present|listed)"
    r"(?:\s+(?:in|anywhere|here|above))?\s*$",
    re.IGNORECASE,
)


_TRAILING_ADVERB_RE = re.compile(r"\s+\w+ly$", re.IGNORECASE)


_SPECIFIC_TOPIC_PATTERNS = [
    r"^what\s+(.+?)\s+does\s+this\s+person\s+know\s*$",
    r"^what\s+(.+?)\s+does\s+this\s+person\s+(?:use|have)\s*$",
    r"^which\s+document\s+(?:mentions|discusses|covers)\s+(.+)$",
    r"^where\s+is\s+(.+?)\s+mentioned\s*$",
]


def extract_topic_from_query(query: str) -> str:
    """
    Strip generic question scaffolding from a query and return whatever
    substantive topic is left, without relying on a fixed topic vocabulary.

    Examples:
        "What programming languages are mentioned?" -> "programming languages"
        "Did this person work with Kubernetes?"      -> "work with Kubernetes"
        "Is GDPR compliance discussed?"               -> "GDPR compliance"
        "What programming languages does this person know?" -> "programming languages"

    This is intentionally simple (regex-based scaffolding removal, not a
    parser) -- it doesn't need to be perfect, it just needs to keep the
    query's real vocabulary in the hypothesis instead of discarding it.
    """
    normalized = query.strip().rstrip("?").strip()

    if not normalized:
        return normalized

    for pattern in _SPECIFIC_TOPIC_PATTERNS:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            if topic:
                return topic

    topic = normalized
    topic = _LEADING_INTERROGATIVE_RE.sub("", topic)
    topic = _LEADING_AUX_RE.sub("", topic)
    topic = _LEADING_SUBJECT_RE.sub("", topic)
    topic = _TRAILING_BOILERPLATE_RE.sub("", topic)
    topic = _TRAILING_ADVERB_RE.sub("", topic)
    topic = topic.strip()

    return topic or normalized


class LocalNLIJudge:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

        self.id2label = {
            int(index): label.lower()
            for index, label in self.model.config.id2label.items()
        }

        print("NLI labels:", self.id2label)

    def build_hypothesis(self, query: str) -> str:
        """
        Turn a user question into a short declarative NLI hypothesis.

        No topic is special-cased here on purpose -- see the module-level
        docstring above for why a hardcoded keyword list silently degrades
        every other topic to a near-content-free fallback hypothesis. This
        always builds the hypothesis from the query's own words so it
        generalizes to any document collection, not just the one it was
        tuned against.

        Phrasing matters more than it looks: cross-encoder NLI models are
        trained on strict textual entailment (does the premise logically
        assert the hypothesis), not topical relevance. An early version of
        this method used "This evidence directly relates to {topic}.",
        which real-world testing showed scores as NEUTRAL (entailment
        scores near 0) even for evidence that obviously discusses the
        topic -- "relates to" reads as a vague association claim, not
        something the premise text actually asserts. "This item involves
        {topic}." reads as a direct claim the premise can actually entail,
        matching the phrasing style that scored correctly (>=0.90
        entailment) in this project's own test suite.
        """
        topic = extract_topic_from_query(query)

        if not topic:
            return (
                "This evidence directly supports "
                "the topic requested by the question."
            )

        return f"This item involves {topic}."

    def _score_entailment(self, premise: str, hypothesis: str) -> Dict:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = self.model(**inputs)

        probabilities = torch.softmax(output.logits, dim=-1)[0]

        scores = {}
        for index, probability in enumerate(probabilities):
            label = self.id2label.get(index, str(index))
            scores[label] = float(probability)

        entailment_score = 0.0
        for label, score in scores.items():
            if "entail" in label:
                entailment_score = score
                break

        return {
            "label": (
                "SUPPORTS" if entailment_score >= 0.5 else "DOES_NOT_SUPPORT"
            ),
            "entailment_score": entailment_score,
            "scores": scores,
        }

    def judge(self, query: str, evidence_text: str) -> Dict:
        hypothesis = self.build_hypothesis(query)

        result = self._score_entailment(evidence_text, hypothesis)
        result["hypothesis"] = hypothesis

        return result

    def judge_claim(self, claim: str, evidence_text: str) -> Dict:
        result = self._score_entailment(evidence_text, claim)
        result["hypothesis"] = claim

        return result