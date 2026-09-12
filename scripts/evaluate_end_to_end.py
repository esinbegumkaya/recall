from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recall.engine import RecallEngine


@dataclass
class EvaluationCase:
    case_id: str
    query: str
    description: str
    validator: Callable[[dict], tuple[bool, str]]


def normalize(text: str) -> str:
    return " ".join(
        (text or "")
        .casefold()
        .strip()
        .split()
    )


def source_text(result: dict) -> str:
    parts: list[str] = []

    for source in result.get("sources", []):
        parts.extend(
            [
                str(source.get("file_name") or ""),
                str(source.get("file_path") or ""),
                str(source.get("section_name") or ""),
                str(source.get("parent_text") or ""),
                str(source.get("text") or ""),
            ]
        )

    return normalize(" ".join(parts))


def validate_supported_answer(
    result: dict,
) -> tuple[bool, str]:
    if result.get("abstained"):
        return (
            False,
            f"Unexpected abstention: "
            f"{result.get('abstention_reason')}",
        )

    if not result.get("sources"):
        return False, "No validated sources returned."

    answer = (result.get("answer") or "").strip()

    if not answer:
        return False, "Answer is empty."

    return True, "Supported answer returned with source evidence."


def validate_unanswerable(
    result: dict,
) -> tuple[bool, str]:
    if result.get("abstained"):
        return (
            True,
            "Correctly abstained: "
            f"{result.get('abstention_reason')}",
        )

    answer = normalize(
        result.get("answer", "")
    )

    abstention_phrases = (
        "could not find",
        "not enough information",
        "do not have enough information",
        "don't have enough information",
        "insufficient evidence",
        "cannot determine",
        "can't determine",
    )

    if any(
        phrase in answer
        for phrase in abstention_phrases
    ):
        return True, "Safe no-evidence answer returned."

    return (
        False,
        "System answered an unsupported question "
        "instead of abstaining.",
    )


def validate_ibm_provenance(
    result: dict,
) -> tuple[bool, str]:
    if result.get("abstained"):
        return (
            False,
            f"Unexpected abstention: "
            f"{result.get('abstention_reason')}",
        )

    evidence = source_text(result)

    if "cubewise" not in evidence:
        return (
            False,
            "IBM Planning Analytics evidence was not "
            "attributed to Cubewise.",
        )

    if (
        "ibm planning analytics" not in evidence
        and "tm1" not in evidence
    ):
        return (
            False,
            "Expected IBM Planning Analytics/TM1 evidence "
            "was not present.",
        )

    return (
        True,
        "IBM Planning Analytics evidence remained "
        "associated with Cubewise.",
    )


def validate_ai_agents(
    result: dict,
) -> tuple[bool, str]:
    if result.get("abstained"):
        return (
            False,
            f"Unexpected abstention: "
            f"{result.get('abstention_reason')}",
        )

    evidence = source_text(result)

    topic_markers = (
        "ai agent",
        "ai agents",
        "agentic ai",
        "agentic",
    )

    if not any(
        marker in evidence
        for marker in topic_markers
    ):
        return (
            False,
            "Returned sources do not contain explicit "
            "AI-agent evidence.",
        )

    return (
        True,
        "Topic-specific AI-agent evidence survived "
        "the validation pipeline.",
    )


def validate_medical_ml(
    result: dict,
) -> tuple[bool, str]:
    if result.get("abstained"):
        return (
            False,
            f"Unexpected abstention: "
            f"{result.get('abstention_reason')}",
        )

    evidence = source_text(result)

    medical_markers = (
        "skin cancer",
        "dermoscopic",
        "clinical image",
        "efficientnet",
        "cnn",
    )

    if not any(
        marker in evidence
        for marker in medical_markers
    ):
        return (
            False,
            "Expected medical-image / skin-cancer "
            "project evidence was not returned.",
        )

    return (
        True,
        "Relevant medical-image project evidence "
        "was returned.",
    )


CASES = [
    EvaluationCase(
        case_id="E2E-01",
        query="What AI experience does this person have?",
        description=(
            "Answerable question: should return a "
            "supported answer with validated sources."
        ),
        validator=validate_supported_answer,
    ),
    EvaluationCase(
        case_id="E2E-02",
        query="What is this person's favorite restaurant?",
        description=(
            "Unsupported question: should abstain safely."
        ),
        validator=validate_unanswerable,
    ),
    EvaluationCase(
        case_id="E2E-03",
        query="Where was IBM Planning Analytics used?",
        description=(
            "Provenance test: IBM Planning Analytics "
            "should remain associated with Cubewise."
        ),
        validator=validate_ibm_provenance,
    ),
    EvaluationCase(
        case_id="E2E-04",
        query="What experience does this person have with AI agents?",
        description=(
            "Topic precision test: AI-agent-specific "
            "evidence should survive validation."
        ),
        validator=validate_ai_agents,
    ),
    EvaluationCase(
        case_id="E2E-05",
        query="Find the work involving medical image classification.",
        description=(
            "Answerable project query: should retrieve "
            "the skin-cancer / medical-image project."
        ),
        validator=validate_medical_ml,
    ),
]


def print_result_details(
    result: dict,
) -> None:
    print(
        f"Abstained      : "
        f"{result.get('abstained')}"
    )
    print(
        f"Abstention     : "
        f"{result.get('abstention_reason')}"
    )
    print(
        f"Candidates     : "
        f"{result.get('candidate_count')}"
    )
    print(
        f"Evidence       : "
        f"{result.get('evidence_count')}"
    )
    print(
        f"Sources        : "
        f"{len(result.get('sources', []))}"
    )
    print(
        f"Elapsed        : "
        f"{result.get('elapsed_seconds', 0):.2f}s"
    )

    answer = (
        result.get("answer")
        or ""
    ).strip()

    print("\nAnswer:")
    print(answer or "<empty>")

    sources = result.get(
        "sources",
        [],
    )

    if sources:
        print("\nSources:")

        for source in sources:
            source_number = source.get(
                "source_number"
            )
            file_name = source.get(
                "file_name"
            )
            section_name = source.get(
                "section_name"
            )
            page_number = source.get(
                "page_number"
            )
            nli_score = source.get(
                "nli_score"
            )

            location_parts = []

            if section_name:
                location_parts.append(
                    f"section={section_name}"
                )

            if page_number is not None:
                location_parts.append(
                    f"page={page_number}"
                )

            location = (
                ", ".join(
                    location_parts
                )
                or "no location metadata"
            )

            if isinstance(
                nli_score,
                (int, float),
            ):
                nli_text = (
                    f"{nli_score:.3f}"
                )
            else:
                nli_text = "n/a"

            print(
                f"  [{source_number}] "
                f"{file_name} "
                f"({location}, "
                f"NLI={nli_text})"
            )


def main() -> int:
    print()
    print("=" * 72)
    print("RECALL END-TO-END EVALUATION")
    print("=" * 72)
    print(
        "Pipeline: retrieval -> evidence gates -> NLI -> "
        "Foundry Local generation -> citation validation"
    )
    print(
        f"Cases: {len(CASES)}"
    )
    print()

    engine = RecallEngine(
        use_generation=True,
    )

    passed = 0
    failed = 0
    total_started = time.perf_counter()

    case_results: list[
        tuple[str, bool, str, float]
    ] = []

    for index, case in enumerate(
        CASES,
        start=1,
    ):
        print("-" * 72)
        print(
            f"[{index}/{len(CASES)}] "
            f"{case.case_id}"
        )
        print(
            f"Query       : {case.query}"
        )
        print(
            f"Expectation : {case.description}"
        )
        print("-" * 72)

        started = time.perf_counter()

        try:
            result_obj = engine.search(
                case.query
            )

            result = (
                result_obj.to_dict()
            )

            case_elapsed = (
                time.perf_counter()
                - started
            )

            is_pass, reason = (
                case.validator(
                    result
                )
            )

            if is_pass:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            case_results.append(
                (
                    case.case_id,
                    is_pass,
                    reason,
                    case_elapsed,
                )
            )

            print()
            print_result_details(
                result
            )

            print()
            print(
                f"RESULT         : {status}"
            )
            print(
                f"VALIDATION     : {reason}"
            )
            print(
                f"TOTAL CASE TIME: "
                f"{case_elapsed:.2f}s"
            )

            generation_error = getattr(
                engine,
                "last_generation_error",
                None,
            )

            if generation_error:
                print(
                    "GENERATION NOTE : "
                    f"{generation_error}"
                )

        except Exception as exc:
            failed += 1

            case_elapsed = (
                time.perf_counter()
                - started
            )

            reason = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            case_results.append(
                (
                    case.case_id,
                    False,
                    reason,
                    case_elapsed,
                )
            )

            print()
            print("RESULT         : ERROR")
            print(
                f"ERROR          : {reason}"
            )

        print()

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    total = len(CASES)

    pass_rate = (
        (passed / total) * 100
        if total
        else 0.0
    )

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for (
        case_id,
        is_pass,
        reason,
        elapsed,
    ) in case_results:
        status = (
            "PASS"
            if is_pass
            else "FAIL"
        )

        print(
            f"{case_id}: "
            f"{status:<4} | "
            f"{elapsed:>6.2f}s | "
            f"{reason}"
        )

    print()
    print(
        f"Queries   : {total}"
    )
    print(
        f"Passed    : {passed}"
    )
    print(
        f"Failed    : {failed}"
    )
    print(
        f"Pass Rate : {pass_rate:.1f}%"
    )
    print(
        f"Total Time: {total_elapsed:.2f}s"
    )
    print("=" * 72)

    if failed == 0:
        print(
            "\n[OK] End-to-end evaluation passed."
        )
        return 0

    print(
        "\n[WARN] One or more end-to-end "
        "evaluation cases failed."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )