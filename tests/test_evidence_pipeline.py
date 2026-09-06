import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "src"
    )
)

from recall.evidence import (
    attach_parent_context,
    split_evidence_units,
)
from recall.evidence_gate import (
    filter_by_topic,
    has_ai_agent_topic,
)


def test_mixed_skills_are_split():
    text = (
        "Technical: Python, C++, Java, SQL, TM1, SAP | "
        "Cloud: Google Cloud, Huawei Cloud, Microsoft Azure AI Foundry | "
        "AI: LLMs, RAG pipelines, prompt engineering, Machine Learning & NLP"
    )

    units = split_evidence_units(text)

    assert len(units) == 3
    assert units[0].startswith("Technical:")
    assert units[1].startswith("Cloud:")
    assert units[2].startswith("AI:")


def test_ai_agent_topic_accepts_real_agentic_evidence():
    assert has_ai_agent_topic(
        "Implemented agentic AI workflows using Python."
    )

    assert has_ai_agent_topic(
        "Built AI agents and RAG pipelines."
    )


def test_ai_agent_topic_rejects_unrelated_experience():
    assert not has_ai_agent_topic(
        "Supported financial planning projects using IBM Planning Analytics."
    )

    assert not has_ai_agent_topic(
        "Worked with ERP and infrastructure systems."
    )


def test_rag_alone_is_not_ai_agent_topic():
    assert not has_ai_agent_topic(
        "Built production RAG pipelines."
    )


def test_topical_gate_filters_false_positive():
    query = (
        "Where did this person work with "
        "AI agents and RAG systems professionally?"
    )

    units = [
        {
            "text": (
                "Implemented Python-based AI solutions "
                "and agentic AI workflows."
            )
        },
        {
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics."
            )
        },
    ]

    filtered = filter_by_topic(
        query,
        units,
    )

    assert len(filtered) == 1
    assert "agentic AI" in filtered[0]["text"]


def test_parent_context_attaches_heading():
    units = [
        {
            "source_number": 1,
            "chunk_index": 2,
            "unit_index": 0,
            "text": (
                "Microsoft AI Innovators Program "
                "— AI Engineering Intern 2026"
            ),
        },
        {
            "source_number": 1,
            "chunk_index": 2,
            "unit_index": 1,
            "text": (
                "Built AI agents and RAG pipelines."
            ),
        },
    ]

    result = attach_parent_context(units)

    assert result[1]["parent_text"].startswith(
        "Microsoft AI Innovators Program"
    )


def test_normal_sentence_is_not_parent():
    units = [
        {
            "source_number": 2,
            "chunk_index": 3,
            "unit_index": 0,
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics."
            ),
        },
        {
            "source_number": 2,
            "chunk_index": 3,
            "unit_index": 1,
            "text": (
                "Built multidimensional data models."
            ),
        },
    ]

    result = attach_parent_context(units)

    assert "parent_text" not in result[1]

def test_ai_agent_hypothesis():
    from recall.nli_judge import LocalNLIJudge

    judge = LocalNLIJudge.__new__(
        LocalNLIJudge
    )

    hypothesis = judge.build_hypothesis(
        "Where did this person work with "
        "AI agents and RAG systems professionally?"
    )

    assert hypothesis == (
        "This item involves AI agents."
    )

def test_real_agentic_evidence_gets_high_entailment():
    from recall.nli_judge import LocalNLIJudge

    judge = LocalNLIJudge()

    result = judge.judge(
        "Where did this person work with "
        "AI agents and RAG systems professionally?",
        (
            "Develop AI applications using Azure AI Foundry "
            "and local LLMs; build AI agents and RAG pipelines "
            "for real-world use cases."
        ),
    )

    assert result["hypothesis"] == (
        "This item involves AI agents."
    )

    assert result["entailment_score"] >= 0.90
def test_evidence_context_preserves_source_provenance():
    from recall.generator import build_evidence_context

    units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "text": "Implement AI agents.",
            "parent_text": "Microsoft AI Innovators Program",
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "text": "Build RAG pipelines.",
            "parent_text": "Microsoft AI Innovators Program",
        },
        {
            "file_path": "other.pdf",
            "page_number": 3,
            "section_name": "CERTIFICATES",
            "chunk_index": 1,
            "text": "Completed AI training.",
        },
    ]

    context = build_evidence_context(
        units
    )

    assert context.count(
        "[Source 1]"
    ) == 1

    assert context.count(
        "[Source 2]"
    ) == 1

    assert context.count(
        "[Source 3]"
    ) == 1

    assert context.count(
        "[Source 2]"
    ) == 1

    assert (
        "Implement AI agents."
        in context
    )

    assert (
        "Build RAG pipelines."
        in context
    )

    assert (
        "Completed AI training."
        in context
    )


def test_evidence_prompt_contains_only_validated_evidence():
    from recall.generator import (
        build_evidence_user_prompt,
    )

    units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "text": (
                "Build AI agents and RAG pipelines."
            ),
            "parent_text": (
                "Microsoft AI Innovators Program"
            ),
        },
    ]

    prompt = build_evidence_user_prompt(
        "Where did this person work with AI agents and RAG systems professionally?",
        units,
    )

    assert (
        "Build AI agents and RAG pipelines."
        in prompt
    )

    assert (
        "Microsoft AI Innovators Program"
        in prompt
    )

    assert (
        "IBM Planning Analytics"
        not in prompt
    )

    assert (
        "FLO Group"
        not in prompt
    )
def test_citation_numbers_are_extracted():
    from recall.generator import (
        extract_citation_numbers,
    )

    assert (
        extract_citation_numbers(
            "Worked at Microsoft [Source 1]."
        )
        == [1]
    )

    assert (
        extract_citation_numbers(
            "Worked at Microsoft [Source 1] "
            "and completed training [Source 2]."
        )
        == [1, 2]
    )


def test_invalid_citation_is_rejected():
    from recall.generator import (
        validate_generated_answer,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": "Built AI agents.",
        },
    ]

    assert not validate_generated_answer(
        "Worked at Microsoft [Source 2].",
        evidence,
    )


def test_valid_citation_is_accepted():
    from recall.generator import (
        validate_generated_answer,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": "Built AI agents.",
        },
    ]

    assert validate_generated_answer(
        "Worked at Microsoft [Source 1].",
        evidence,
    )


def test_abstention_is_valid_without_citation():
    from recall.generator import (
        ABSTENTION_MESSAGE,
        validate_generated_answer,
    )

    assert validate_generated_answer(
        ABSTENTION_MESSAGE,
        [],
    )
def test_generator_rejects_invalid_citation():
    from recall.generator import (
        ABSTENTION_MESSAGE,
        generate_grounded_answer_from_evidence,
    )

    class FakeSettings:
        temperature = None
        max_tokens = None

    class FakeMessage:
        content = "Worked at Microsoft [Source 2]."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeChatClient:
        settings = FakeSettings()

        def complete_chat(self, messages):
            return FakeResponse()

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "text": "Built AI agents.",
        },
    ]

    result = generate_grounded_answer_from_evidence(
        chat_client=FakeChatClient(),
        query="Where did this person work with AI agents?",
        evidence_units=evidence,
    )

    assert result == ABSTENTION_MESSAGE


def test_generator_accepts_valid_citation():
    from recall.generator import (
        generate_grounded_answer_from_evidence,
    )

    class FakeSettings:
        temperature = None
        max_tokens = None

    class FakeMessage:
        content = "Worked at Microsoft [Source 1]."

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeChatClient:
        settings = FakeSettings()

        def complete_chat(self, messages):
            return FakeResponse()

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "text": "Built AI agents.",
        },
    ]

    result = generate_grounded_answer_from_evidence(
        chat_client=FakeChatClient(),
        query="Where did this person work with AI agents?",
        evidence_units=evidence,
    )

    assert result == (
        "Worked at Microsoft [Source 1]."
    )
def test_claim_nli_distinguishes_supported_claim():
    from recall.nli_judge import LocalNLIJudge

    judge = LocalNLIJudge()

    evidence = (
        "Develop AI applications using Azure AI Foundry "
        "and local LLMs; build AI agents and RAG pipelines "
        "for real-world use cases."
    )

    supported = judge.judge_claim(
        "This person built AI agents and RAG pipelines.",
        evidence,
    )

    unsupported = judge.judge_claim(
        "This person worked as a quantum computing researcher.",
        evidence,
    )

    assert (
        supported["entailment_score"]
        >= 0.90
    )

    assert (
        unsupported["entailment_score"]
        < 0.15
    )

def test_mixed_evidence_is_decomposed_by_category():
    from recall.evidence import (
        split_evidence_units,
    )

    units = split_evidence_units(
        "Technical: Python, Java | "
        "Cloud: Azure, AWS | "
        "AI: RAG, AI agents"
    )

    assert units == [
        "Technical: Python, Java",
        "Cloud: Azure, AWS",
        "AI: RAG, AI agents",
    ]

def test_normal_ai_rag_evidence_is_not_mixed_decomposed():
    from recall.evidence import (
        split_evidence_units,
    )

    text = (
        "Develop AI applications using Azure AI Foundry "
        "and local LLMs; build AI agents and RAG pipelines "
        "for real-world use cases."
    )

    units = split_evidence_units(
        text
    )

    assert units == [
        text
    ]

def test_mixed_evidence_maps_to_structural_types():
    from recall.evidence import (
        split_evidence_units,
    )
    from recall.evidence_gate import (
        infer_evidence_type,
    )

    units = split_evidence_units(
        "Technical: Python, Java | "
        "Cloud: Azure, AWS | "
        "AI: RAG, AI agents"
    )

    types = [
        infer_evidence_type(
            {
                "section_name": "SKILLS",
                "text": unit,
            }
        )
        for unit in units
    ]

    assert types == [
        "PROGRAMMING_LANGUAGE",
        "CLOUD_PLATFORM",
        "AI_TECHNOLOGY",
    ]

def test_mixed_evidence_ai_agent_query_keeps_only_ai_topic():
    from recall.evidence import (
        split_evidence_units,
    )
    from recall.evidence_gate import (
        filter_by_topic,
    )

    units = split_evidence_units(
        "Technical: Python, Java | "
        "Cloud: Azure, AWS | "
        "AI: RAG, AI agents"
    )

    evidence_units = [
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "chunk_index": 1,
            "unit_index": index,
            "text": unit,
        }
        for index, unit in enumerate(units)
    ]

    query = (
        "Where did this person work with "
        "AI agents and RAG systems?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert filtered[0]["text"] == (
        "AI: RAG, AI agents"
    )

    assert filtered[0]["topic_match"] == (
        "AI_AGENT"
    )

def test_programming_language_query_keeps_programming_language_evidence():
    from recall.evidence_gate import (
        filter_by_evidence_type,
    )

    evidence_units = [
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "Technical: Python, Java",
        },
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "Cloud: Azure, AWS",
        },
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "AI: RAG, AI agents",
        },
    ]

    query = (
        "What programming languages does this person know?"
    )

    filtered = filter_by_evidence_type(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert filtered[0]["text"] == (
        "Technical: Python, Java"
    )

    assert filtered[0]["evidence_type"] == (
        "PROGRAMMING_LANGUAGE"
    )


def test_cloud_platform_query_keeps_cloud_evidence():
    from recall.evidence_gate import (
        filter_by_evidence_type,
    )

    evidence_units = [
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "Technical: Python, Java",
        },
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "Cloud: Azure, AWS",
        },
        {
            "file_path": "skills.txt",
            "page_number": 1,
            "section_name": "SKILLS",
            "text": "AI: RAG, AI agents",
        },
    ]

    query = (
        "Which cloud platforms does this person know?"
    )

    filtered = filter_by_evidence_type(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert filtered[0]["text"] == (
        "Cloud: Azure, AWS"
    )

    assert filtered[0]["evidence_type"] == (
        "CLOUD_PLATFORM"
    )


def test_ai_agent_topic_gate_rejects_unrelated_experience():
    from recall.evidence_gate import (
        filter_by_topic,
    )

    evidence_units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Develop AI applications using Azure AI Foundry "
                "and local LLMs; build AI agents and RAG pipelines "
                "for real-world use cases."
            ),
        },
    ]

    query = (
        "Where did this person work with AI agents "
        "and RAG systems professionally?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert filtered[0]["text"].startswith(
        "Develop AI applications using Azure AI Foundry"
    )

    assert filtered[0]["topic_match"] == (
        "AI_AGENT"
    )

def test_unrelated_query_is_rejected_by_ai_agent_topic_gate():
    from recall.evidence_gate import (
        filter_by_topic,
    )

    evidence_units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Develop AI applications using Azure AI Foundry "
                "and local LLMs; build AI agents and RAG pipelines "
                "for real-world use cases."
            ),
        },
    ]

    query = (
        "Did this person work with quantum computing?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert filtered == []


def test_ai_agent_query_does_not_accept_cubewise_evidence():
    from recall.evidence_gate import (
        filter_by_topic,
    )

    evidence_units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Develop AI applications using Azure AI Foundry "
                "and local LLMs; build AI agents and RAG pipelines "
                "for real-world use cases."
            ),
        },
    ]

    query = (
        "Did this person work with AI agents at Cubewise?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert filtered[0]["text"].startswith(
        "Develop AI applications using Azure AI Foundry"
    )

    assert filtered[0]["topic_match"] == (
        "AI_AGENT"
    )


def test_rag_query_keeps_rag_evidence():
    from recall.evidence_gate import (
        filter_by_topic,
    )

    evidence_units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Develop AI applications using Azure AI Foundry "
                "and local LLMs; build AI agents and RAG pipelines "
                "for real-world use cases."
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
        },
    ]

    query = (
        "Did this person build RAG pipelines?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert "RAG pipelines" in filtered[0]["text"]


def test_ai_agent_query_keeps_agent_evidence():
    from recall.evidence_gate import (
        filter_by_topic,
    )

    evidence_units = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Develop AI applications using Azure AI Foundry "
                "and local LLMs; build AI agents and RAG pipelines "
                "for real-world use cases."
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
        },
    ]

    query = (
        "Where did this person work with AI agents?"
    )

    filtered = filter_by_topic(
        query,
        evidence_units,
    )

    assert len(filtered) == 1

    assert "AI agents" in filtered[0]["text"]

def test_rag_query_does_not_get_project_section_from_build():
    from recall.intent import detect_section_intent

    assert detect_section_intent(
        "Did this person build RAG pipelines?"
    ) == "EXPERIENCE"


def test_ai_agent_work_query_prefers_experience():
    from recall.intent import detect_section_intent

    assert detect_section_intent(
        "Where did this person work with AI agents?"
    ) == "EXPERIENCE"


def test_explicit_project_query_still_gets_projects():
    from recall.intent import detect_section_intent

    assert detect_section_intent(
        "What projects did this person build?"
    ) == "PROJECTS"


def test_multi_claim_correct_sources_are_accepted():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": (
                "Built AI agents and RAG pipelines."
            ),
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 4,
            "unit_index": 0,
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
            "parent_text": (
                "Cubewise, Turkiye - "
                "Software Development Intern 2025 - 2026"
            ),
        },
    ]

    answer = (
        "The person worked with AI agents at Microsoft "
        "[Source 1]. "
        "The person worked with IBM Planning Analytics "
        "at Cubewise [Source 2]."
    )

    judge = LocalNLIJudge()

    assert validate_generated_answer(
        answer,
        evidence,
        claim_judge=judge,
    )


def test_multi_claim_swapped_sources_are_rejected():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": "Built AI agents and RAG pipelines.",
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 4,
            "unit_index": 0,
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
            "parent_text": (
                "Cubewise, Turkiye - "
                "Software Development Intern 2025 - 2026"
            ),
        },
    ]

    answer = (
        "The person worked with AI agents at Microsoft "
        "[Source 2]. "
        "The person worked with IBM Planning Analytics "
        "at Cubewise [Source 1]."
    )

    judge = LocalNLIJudge()

    assert not validate_generated_answer(
        answer,
        evidence,
        claim_judge=judge,
    )


def test_ai_agent_claim_at_wrong_parent_is_rejected():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": "Built AI agents and RAG pipelines.",
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
        },
    ]

    judge = LocalNLIJudge()

    assert not validate_generated_answer(
        (
            "The person worked with AI agents "
            "at Cubewise [Source 1]."
        ),
        evidence,
        claim_judge=judge,
    )


def test_multi_claim_unsupported_second_claim_rejects_answer():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": "Built AI agents and RAG pipelines.",
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
        },
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 4,
            "unit_index": 0,
            "text": (
                "Supported financial planning projects "
                "using IBM Planning Analytics with Watson (TM1)."
            ),
            "parent_text": (
                "Cubewise, Turkiye - "
                "Software Development Intern 2025 - 2026"
            ),
        },
    ]

    answer = (
        "The person worked with AI agents at Microsoft "
        "[Source 1]. "
        "The person worked as a quantum computing researcher "
        "at Cubewise [Source 2]."
    )

    judge = LocalNLIJudge()

    assert not validate_generated_answer(
        answer,
        evidence,
        claim_judge=judge,
    )


def test_dynamic_topic_extracts_kubernetes():
    from recall.evidence_gate import (
        extract_dynamic_topic,
        filter_by_topic,
    )

    query = (
        "Where did this person use Kubernetes?"
    )

    assert extract_dynamic_topic(
        query
    ) == "Kubernetes"

    units = [
        {
            "text": (
                "Deployed services using Kubernetes "
                "and Docker."
            )
        },
        {
            "text": (
                "Built reporting dashboards in Power BI."
            )
        },
    ]

    result = filter_by_topic(
        query,
        units,
    )

    assert len(result) == 1

    assert "Kubernetes" in result[0]["text"]


def test_dynamic_topic_extracts_postgresql():
    from recall.evidence_gate import (
        extract_dynamic_topic,
        filter_by_topic,
    )

    query = (
        "Did this person use PostgreSQL?"
    )

    assert extract_dynamic_topic(
        query
    ) == "PostgreSQL"

    units = [
        {
            "text": (
                "Designed backend services using "
                "PostgreSQL and FastAPI."
            )
        },
        {
            "text": (
                "Created TensorFlow image classifiers."
            )
        },
    ]

    result = filter_by_topic(
        query,
        units,
    )

    assert len(result) == 1

    assert "PostgreSQL" in result[0]["text"]


def test_dynamic_topic_extracts_gdpr_compliance():
    from recall.evidence_gate import (
        extract_dynamic_topic,
        filter_by_topic,
    )

    query = (
        "Which document discusses GDPR compliance?"
    )

    assert extract_dynamic_topic(
        query
    ) == "GDPR compliance"

    units = [
        {
            "text": (
                "The policy describes GDPR compliance "
                "requirements and data retention rules."
            )
        },
        {
            "text": (
                "The infrastructure guide describes "
                "container deployment."
            )
        },
    ]

    result = filter_by_topic(
        query,
        units,
    )

    assert len(result) == 1

    assert (
        "GDPR compliance"
        in result[0]["text"]
    )


def test_dynamic_topic_extracts_unregistered_product():
    from recall.evidence_gate import (
        detect_topic_requirement,
        filter_by_topic,
    )

    query = (
        "Where did this person use Apache Airflow?"
    )

    topic = detect_topic_requirement(
        query
    )

    assert topic == (
        "DYNAMIC::Apache Airflow"
    )

    units = [
        {
            "text": (
                "Built scheduled data pipelines "
                "with Apache Airflow."
            )
        },
        {
            "text": (
                "Developed REST APIs with FastAPI."
            )
        },
    ]

    result = filter_by_topic(
        query,
        units,
    )

    assert len(result) == 1

    assert (
        "Apache Airflow"
        in result[0]["text"]
    )

def test_semantic_contract_fallback_accepts_validated_claim():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": (
                "Implement Python-based AI solutions "
                "and agentic AI workflows, applying "
                "prompt engineering with GitHub-based "
                "collaboration."
            ),
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
            "supporting_subquery": (
                "Where did this person work with AI agents"
            ),
            "supporting_hypothesis": (
                "This person worked with AI agents."
            ),
            "_source_number": 1,
        },
    ]

    judge = LocalNLIJudge()

    answer = (
        "The person worked with AI agents at the "
        "Microsoft AI Innovators Program - "
        "AI Engineering Intern 2026. "
        "[Source 1]"
    )

    assert validate_generated_answer(
        answer,
        evidence,
        claim_judge=judge,
    )


def test_semantic_contract_fallback_rejects_topic_drift():
    from recall.generator import (
        validate_generated_answer,
    )
    from recall.nli_judge import (
        LocalNLIJudge,
    )

    evidence = [
        {
            "file_path": "cv.pdf",
            "page_number": 1,
            "section_name": "EXPERIENCE",
            "chunk_index": 2,
            "unit_index": 0,
            "text": (
                "Implement Python-based AI solutions "
                "and agentic AI workflows, applying "
                "prompt engineering with GitHub-based "
                "collaboration."
            ),
            "parent_text": (
                "Microsoft AI Innovators Program - "
                "AI Engineering Intern 2026"
            ),
            "supporting_subquery": (
                "Where did this person work with AI agents"
            ),
            "supporting_hypothesis": (
                "This person worked with AI agents."
            ),
            "_source_number": 1,
        },
    ]

    judge = LocalNLIJudge()

    answer = (
        "The person worked with quantum computing "
        "at the Microsoft AI Innovators Program - "
        "AI Engineering Intern 2026. "
        "[Source 1]"
    )

    assert not validate_generated_answer(
        answer,
        evidence,
        claim_judge=judge,
    )
