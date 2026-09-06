# MemoRAG

**MemoRAG** is a local retrieval-grounded AI assistant designed to answer questions from documents while keeping retrieval, evidence validation, generation, and citations grounded in the original source material.

The project is being developed as a **general-purpose local RAG system**, rather than a document-specific question-answering application. The goal is to support different document types such as CVs, reports, technical documents, contracts, and personal notes.

> **Current development status: ~95% complete**

---

## Project Goals

MemoRAG aims to provide:

* Fully local document retrieval and question answering
* Semantic search over indexed documents
* Evidence-level validation before generation
* Hallucination-resistant grounded answers
* Source-aware citations
* Multi-part question handling
* Provenance-aware evidence association
* Support for multiple document types
* A modular architecture that can be extended beyond a single use case

---

# Current Architecture

The current MemoRAG evidence pipeline follows approximately:

```text
Documents
    ↓
Parsing & Chunking
    ↓
Local Embeddings
    ↓
SQLite Vector Storage
    ↓
Semantic Retrieval
    ↓
Section Intent Detection
    ↓
Evidence Unit Construction
    ↓
Structural Gate
    ↓
Topical Gate
    ↓
NLI Validation
    ↓
Overlap Deduplication
    ↓
Parent / Provenance Association
    ↓
Grounded Generation
    ↓
Citation & Claim Validation
    ↓
Final Answer
```

---

# Implemented

## 1. Local Document Processing

* [x] Local document ingestion
* [x] TXT parsing
* [x] Markdown parsing
* [x] Python/source-text parsing
* [x] PDF parsing
* [x] Chunk-based document processing
* [x] Configurable chunk size and overlap
* [x] Page metadata preservation
* [x] Section metadata preservation
* [x] Incremental indexing
* [x] SQLite-backed document and chunk storage
* [x] Local embedding generation

---

## 2. Semantic Retrieval

* [x] Semantic document search
* [x] Retrieval scoring
* [x] Top-k source retrieval
* [x] File-level provenance
* [x] Chunk-level provenance
* [x] Page-level provenance
* [x] Section-aware retrieval metadata

---

## 3. Section Intent Detection

MemoRAG detects the likely document section required by a question.

Examples:

```text
"What projects did this person build?"
→ PROJECTS

"Where did this person work with AI agents?"
→ EXPERIENCE
```

Implemented:

* [x] Section intent detection
* [x] EXPERIENCE detection
* [x] PROJECT detection
* [x] SKILL detection
* [x] CERTIFICATE detection
* [x] Conservative fallback when section filtering would remove all evidence

---

## 4. Evidence Unit Construction

Retrieved chunks are converted into smaller evidence units before validation.

Implemented:

* [x] Evidence-unit extraction
* [x] Source metadata preservation
* [x] Page metadata preservation
* [x] Section metadata preservation
* [x] Chunk provenance preservation
* [x] Unit-level indexing
* [x] Heading detection
* [x] Parent-child evidence association
* [x] Mixed evidence splitting

This allows MemoRAG to reason over smaller factual units instead of passing entire retrieved chunks directly to the generator.

---

## 5. Structural Evidence Gate

The structural gate determines whether an evidence unit has the correct structural type for the question.

Supported evidence categories currently include:

* EXPERIENCE
* PROJECT
* SKILL
* CERTIFICATE

Additional structural filtering exists for areas such as:

* Programming languages
* Cloud technologies
* AI technologies

Implemented:

* [x] Structural evidence classification
* [x] Boundary-aware term matching
* [x] False-positive reduction for overlapping terms
* [x] Conservative structural fallback

---

## 6. Topical Evidence Gate

MemoRAG performs topic filtering separately from structural filtering.

Known synonym families currently include concepts such as:

* AI agents
* Agentic AI
* Agentic workflows
* RAG
* Retrieval-Augmented Generation
* Quantum computing

The system does **not** rely exclusively on a hard-coded topic taxonomy.

Implemented:

* [x] Known topic synonym groups
* [x] Boundary-aware topic matching
* [x] Dynamic topic extraction
* [x] Generic topic detection for previously unseen technologies
* [x] Topic filtering before NLI

Examples:

```text
"Where did this person use Kubernetes?"
→ Dynamic topic: Kubernetes

"Did this person use IBM Planning Analytics?"
→ Dynamic topic: IBM Planning Analytics
```

---

## 7. Dynamic Topic Extraction

MemoRAG can extract concrete topics directly from common factual questions without requiring every possible technology or subject to be predefined.

Examples include:

```text
Kubernetes
PostgreSQL
IBM Planning Analytics
Apache Airflow
GDPR compliance
```

This prevents the evidence gate from becoming document-specific or dependent on a large hard-coded taxonomy.

---

## 8. NLI Evidence Validation

MemoRAG uses a local **Natural Language Inference (NLI)** model to determine whether retrieved evidence actually supports a query-derived hypothesis.

Current model:

```text
cross-encoder/nli-deberta-v3-small
```

Implemented:

* [x] Local NLI inference
* [x] Entailment scoring
* [x] Query-to-hypothesis generation
* [x] Evidence acceptance/rejection
* [x] Independent NLI validation for compound questions
* [x] Unsupported evidence rejection

NLI thresholds are intentionally treated separately across different validation stages rather than being adjusted to hide retrieval or structural problems.

---

## 9. Parent / Provenance Association

MemoRAG preserves relationships between evidence and its surrounding structural context.

Example:

```text
Parent:
Microsoft AI Innovators Program — AI Engineering Intern 2026

Evidence:
Implement Python-based AI solutions and agentic AI workflows...
```

This allows answers to correctly attribute evidence to organizations, roles, projects, or other parent contexts.

Implemented:

* [x] Heading detection
* [x] Parent-child association
* [x] Parent metadata preservation
* [x] Provenance-aware generation
* [x] Deterministic provenance validation
* [x] Wrong-parent attribution rejection

---

## 10. Overlap-Aware Evidence Deduplication

Chunk overlap can produce partial duplicate evidence.

MemoRAG detects overlapping evidence units and keeps the higher-quality version.

The deduplication logic considers:

* File
* Page
* Section
* Chunk proximity
* Supporting subquery
* Token overlap
* Evidence completeness
* Parent context availability

Implemented:

* [x] Exact evidence deduplication
* [x] Token-overlap scoring
* [x] Near-duplicate detection
* [x] Evidence quality scoring
* [x] Adjacent chunk overlap suppression

This prevents incomplete chunk fragments from becoming separate generation sources.

---

## 11. Compound Query Handling

MemoRAG can decompose multi-part questions and validate each subquestion independently.

Example:

```text
Where did this person work with AI agents,
and where did this person use IBM Planning Analytics?
```

Processing flow:

```text
Subquery 1
    ↓
Topical Filtering
    ↓
NLI
    ↓
Evidence Selection
    ↓
Generation
    ↓
Validation

Subquery 2
    ↓
Topical Filtering
    ↓
NLI
    ↓
Evidence Selection
    ↓
Generation
    ↓
Validation

Validated Subanswers
    ↓
Final Composition
```

Implemented:

* [x] Compound-query decomposition
* [x] Independent topical filtering
* [x] Independent NLI validation
* [x] Evidence-to-subquery mapping
* [x] Independent subanswer generation
* [x] Independent subanswer validation
* [x] Validated answer composition

---

## 12. Grounded Answer Generation

The generator receives validated evidence rather than unrestricted retrieved context.

Generation context can contain:

* Source number
* File
* Page
* Section
* Chunk
* Parent context
* Evidence text
* Supporting subquery

Implemented:

* [x] Evidence-grounded prompting
* [x] Deterministic generation settings
* [x] Evidence terminology guidance
* [x] Compound answer generation
* [x] Exact abstention behavior when evidence is insufficient

---

## 13. Citation Validation

Generated factual claims must reference valid evidence sources.

Example:

```text
The person worked with AI agents at the Microsoft AI Innovators Program. [Source 1]
```

Implemented:

* [x] Source-number validation
* [x] Claim-to-source extraction
* [x] Invalid citation rejection
* [x] Multi-claim citation validation
* [x] Swapped-source rejection
* [x] Unsupported secondary-claim rejection

---

## 14. Post-Generation Claim Validation

MemoRAG does not assume that grounded prompting alone is sufficient to prevent hallucinations.

Generated claims are validated after generation.

Current validation flow:

```text
Generated Claim
      ↓
Citation Validation
      ↓
Provenance Validation
      ↓
Semantic Evidence Validation
      ↓
Semantic Contract Validation
```

Implemented:

* [x] Evidence-to-claim NLI validation
* [x] Strict claim-validation threshold
* [x] Separation of provenance and semantic validation
* [x] Semantic claim extraction
* [x] Supporting hypothesis preservation
* [x] Semantic-contract fallback
* [x] Topic-drift rejection

The semantic-contract mechanism handles cases where a generated claim faithfully expresses an already validated query hypothesis but direct evidence-to-generated-claim NLI is overly sensitive to paraphrasing.

---

# Hallucination Protection

MemoRAG uses multiple independent safeguards rather than relying on retrieval similarity alone.

```text
Semantic Retrieval
        ↓
Structural Gate
        ↓
Topical Gate
        ↓
Evidence NLI
        ↓
Provenance Association
        ↓
Grounded Generation
        ↓
Citation Validation
        ↓
Provenance Validation
        ↓
Claim NLI / Semantic Contract
```

This layered architecture is designed to reduce:

* Semantically similar but irrelevant evidence
* Incorrect topic matches
* Wrong organization/project attribution
* Unsupported generated claims
* Incorrect citations
* Semantic topic drift
* Duplicate evidence caused by chunk overlap

---

# Testing

A regression suite has been introduced for the evidence pipeline.

Current coverage includes:

* [x] Structural evidence filtering
* [x] AI-agent evidence
* [x] RAG evidence
* [x] Unsupported topics
* [x] Dynamic topic extraction
* [x] Parent/provenance preservation
* [x] Citation validation
* [x] Multi-claim answers
* [x] Swapped citations
* [x] Wrong-parent attribution
* [x] Unsupported secondary claims
* [x] Semantic-contract fallback
* [x] Semantic topic-drift rejection
* [x] Evidence overlap deduplication

Current regression suite:

```text
42 tests passing
```

---

# Example

### Query

```text
Where did this person work with AI agents,
and where did this person use IBM Planning Analytics?
```

MemoRAG independently identifies and validates evidence for both parts of the question.

### Grounded Output

```text
The person worked with AI agents at the Microsoft AI Innovators Program
— AI Engineering Intern 2026. [Source 1]

The person used IBM Planning Analytics with Watson (TM1) to support
financial planning projects during their internship at Cubewise,
Türkiye, from 2025 to 2026. [Source 2]
```

Both claims are independently validated against their corresponding evidence before being returned.

---

# Roadmap

## Evidence Pipeline Hardening

* [ ] Expand regression coverage across additional query formulations
* [ ] Test ambiguous queries
* [ ] Test partially supported compound queries
* [ ] Test larger compound questions
* [ ] Stress-test dynamic topic extraction
* [ ] Test additional topic synonyms and paraphrases
* [ ] Expand negative evidence tests
* [ ] Expand provenance edge-case tests

---

## Multi-Document Validation

* [ ] Test MemoRAG on documents other than CVs
* [ ] Test technical documentation
* [ ] Test reports
* [ ] Test contracts
* [ ] Test personal notes
* [ ] Test multiple documents containing conflicting or overlapping evidence
* [ ] Test cross-document source attribution

---

## Retrieval Evaluation

* [ ] Build a small evaluation dataset
* [ ] Measure retrieval precision
* [ ] Measure evidence acceptance precision
* [ ] Measure unsupported-answer rejection rate
* [ ] Measure citation correctness
* [ ] Evaluate compound-query performance

---

## Code Cleanup

* [ ] Remove temporary debug output
* [ ] Remove temporary diagnostic scripts
* [ ] Consolidate configuration constants
* [ ] Review module boundaries
* [ ] Improve type hints
* [ ] Improve internal documentation
* [ ] Review error handling

---

## Developer Experience

* [ ] Finalize installation instructions
* [ ] Document environment setup
* [ ] Document indexing workflow
* [ ] Document query workflow
* [ ] Add example commands
* [ ] Add architecture diagram
* [ ] Add troubleshooting section

---

## Final Demo

* [ ] Prepare clean demo documents
* [ ] Define representative supported queries
* [ ] Define hallucination/adversarial queries
* [ ] Demonstrate compound-query handling
* [ ] Demonstrate provenance validation
* [ ] Demonstrate unsupported-query abstention
* [ ] Demonstrate dynamic topic extraction
* [ ] Record final example outputs

---

# Development Status

```text
███████████████████░ 95%
```

### Completed

Core local RAG pipeline, evidence construction, structural filtering, topical filtering, dynamic topic extraction, NLI validation, provenance association, overlap deduplication, compound-query processing, grounded generation, citations, and post-generation claim validation.

### Remaining

The remaining work primarily consists of:

* Regression and stress testing
* Multi-document validation
* Retrieval evaluation
* Code cleanup
* Documentation
* Demo preparation

At this stage, development is primarily focused on **hardening and evaluating the existing architecture rather than introducing another major architectural layer**.

---

# Core Design Principle

> **Retrieval is not evidence, and generated text is not automatically a supported answer.**

A retrieved chunk must pass progressively stronger evidence checks before it can support generation. Generated claims must then be validated against their cited evidence and provenance before they are returned to the user.

This separation between **retrieval**, **evidence qualification**, **generation**, and **claim validation** is the central architectural principle behind MemoRAG.
