# MemoRAG

**MemoRAG** is a local knowledge retrieval assistant designed to help users find information across files on their computer without requiring them to remember **where a file is stored, what it is called, or which version contains the information they need**.

MemoRAG is being built as a **local-first knowledge layer** over user-authorized files. It combines document discovery, semantic retrieval, evidence validation, provenance tracking, and grounded generation so that answers are not only relevant, but traceable to their exact supporting sources.

> **Full Product Status: ~70–75%**  
> **Core RAG & Evidence Engine: ~95%**

---

## Why MemoRAG?

Personal knowledge is often fragmented across:

- Documents
- Downloads
- Desktop folders
- Reports
- Contracts
- Notes
- Source files
- Multiple revisions of the same document

The difficult part is often not knowing *what* to search for, but remembering:

- Where the file was saved
- What the file was called
- Which copy is current
- Which version contains a specific fact
- Which document originally mentioned something

MemoRAG is being built to remove that burden.

> **Ask for the information. MemoRAG finds the file, the relevant version, the evidence, and the source.**

---

# Product Goals

MemoRAG aims to provide:

- [x] Fully local document retrieval and question answering
- [x] Semantic search over indexed content
- [x] Evidence-level validation before generation
- [x] Hallucination-resistant grounded answers
- [x] Source-aware citations
- [x] Multi-part question handling
- [x] Provenance-aware evidence association
- [x] Incremental document indexing
- [ ] Automatic discovery across user-authorized folders and drives
- [ ] File-change monitoring and background re-indexing
- [ ] Duplicate and near-duplicate document detection
- [ ] Document version-family detection
- [ ] Latest/relevant version resolution
- [ ] Cross-document and cross-version retrieval
- [ ] A polished local knowledge workspace UI

---

# Target Architecture

```text
Local Computer
      ↓
User-Authorized Folders / Drives
      ↓
File Discovery & Filesystem Watcher
      ↓
Metadata + Hash Extraction
      ↓
Duplicate / Version Resolution
      ↓
Parsing & Chunking
      ↓
Local Embeddings + Knowledge Index
      ↓
Semantic + Metadata Retrieval
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
Citation + Claim Validation
      ↓
Answer + Exact Source + Version
```

The evidence-validation portion of this architecture is already largely implemented. File-system intelligence, version awareness, product UI, and broader evaluation remain under development.

---

# Implemented

## 1. Local Document Processing

- [x] Local document ingestion
- [x] TXT parsing
- [x] Markdown parsing
- [x] Python/source-text parsing
- [x] PDF parsing
- [x] Chunk-based document processing
- [x] Configurable chunk size and overlap
- [x] Page metadata preservation
- [x] Section metadata preservation
- [x] Incremental indexing
- [x] SQLite-backed document and chunk storage
- [x] Local embedding generation

MemoRAG can already convert supported local documents into searchable chunks while retaining provenance metadata needed later in the evidence pipeline.

---

## 2. Semantic Retrieval

- [x] Semantic document search
- [x] Retrieval scoring
- [x] Top-k source retrieval
- [x] File-level provenance
- [x] Chunk-level provenance
- [x] Page-level provenance
- [x] Section-aware retrieval metadata

Retrieval is intentionally treated as the beginning of the evidence pipeline rather than proof that a retrieved passage actually answers the question.

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

- [x] Section intent detection
- [x] EXPERIENCE detection
- [x] PROJECT detection
- [x] SKILL detection
- [x] CERTIFICATE detection
- [x] Conservative fallback when section filtering would remove all evidence

---

## 4. Evidence Unit Construction

Retrieved chunks are converted into smaller evidence units before validation.

Implemented:

- [x] Evidence-unit extraction
- [x] Source metadata preservation
- [x] Page metadata preservation
- [x] Section metadata preservation
- [x] Chunk provenance preservation
- [x] Unit-level indexing
- [x] Heading detection
- [x] Parent-child evidence association
- [x] Mixed evidence splitting

This allows MemoRAG to reason over smaller factual units instead of treating an entire retrieved chunk as a single piece of evidence.

---

## 5. Structural Evidence Gate

The structural gate checks whether an evidence unit has the appropriate structural type for the question.

Supported evidence categories currently include:

- EXPERIENCE
- PROJECT
- SKILL
- CERTIFICATE

Additional structural filtering exists for areas such as:

- Programming languages
- Cloud technologies
- AI technologies

Implemented:

- [x] Structural evidence classification
- [x] Boundary-aware term matching
- [x] False-positive reduction for overlapping terms
- [x] Conservative structural fallback

---

## 6. Topical Evidence Gate

MemoRAG separates **topic relevance** from structural relevance.

Known synonym families currently cover concepts such as:

- AI agents
- Agentic AI
- Agentic workflows
- RAG
- Retrieval-Augmented Generation
- Quantum computing

The system does **not** rely exclusively on a hard-coded topic taxonomy.

Implemented:

- [x] Known topic synonym groups
- [x] Boundary-aware topic matching
- [x] Dynamic topic extraction
- [x] Generic topic detection for previously unseen technologies
- [x] Topic filtering before NLI

Examples:

```text
"Where did this person use Kubernetes?"
→ Dynamic topic: Kubernetes

"Did this person use IBM Planning Analytics?"
→ Dynamic topic: IBM Planning Analytics
```

---

## 7. Dynamic Topic Extraction

MemoRAG can extract concrete topics from common factual questions without requiring every possible subject to be predefined.

Tested examples include:

```text
Kubernetes
PostgreSQL
IBM Planning Analytics
Apache Airflow
GDPR compliance
```

This keeps the system general-purpose instead of turning the evidence gate into a document-specific taxonomy.

---

## 8. NLI Evidence Validation

MemoRAG uses a local **Natural Language Inference (NLI)** model to determine whether candidate evidence actually supports a query-derived hypothesis.

Current model:

```text
cross-encoder/nli-deberta-v3-small
```

Implemented:

- [x] Local NLI inference
- [x] Entailment scoring
- [x] Query-to-hypothesis generation
- [x] Evidence acceptance/rejection
- [x] Independent NLI validation for compound questions
- [x] Unsupported evidence rejection

Different validation stages intentionally retain separate semantics instead of changing thresholds merely to compensate for retrieval or wording problems.

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

This allows answers to attribute evidence to the correct organization, role, project, or other parent context.

Implemented:

- [x] Heading detection
- [x] Parent-child association
- [x] Parent metadata preservation
- [x] Provenance-aware generation
- [x] Deterministic provenance validation
- [x] Wrong-parent attribution rejection

---

## 10. Overlap-Aware Evidence Deduplication

Chunk overlap can produce partial duplicates of the same evidence.

MemoRAG detects these overlaps and preserves the higher-quality evidence unit.

The deduplication logic considers:

- File
- Page
- Section
- Chunk proximity
- Supporting subquery
- Token overlap
- Evidence completeness
- Parent context availability

Implemented:

- [x] Exact evidence deduplication
- [x] Token-overlap scoring
- [x] Near-duplicate detection
- [x] Evidence quality scoring
- [x] Adjacent chunk overlap suppression

This prevents incomplete overlap fragments from becoming independent generation sources.

---

## 11. Compound Query Handling

MemoRAG can decompose multi-part questions and process each subquestion independently.

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

- [x] Compound-query decomposition
- [x] Independent topical filtering
- [x] Independent NLI validation
- [x] Evidence-to-subquery mapping
- [x] Independent subanswer generation
- [x] Independent subanswer validation
- [x] Validated answer composition

---

## 12. Grounded Answer Generation

The generator receives validated evidence rather than unrestricted retrieved context.

Generation context can contain:

- Source number
- File
- Page
- Section
- Chunk
- Parent context
- Evidence text
- Supporting subquery

Implemented:

- [x] Evidence-grounded prompting
- [x] Deterministic generation settings
- [x] Evidence terminology guidance
- [x] Compound answer generation
- [x] Exact abstention behavior when evidence is insufficient

---

## 13. Citation Validation

Generated factual claims must reference valid evidence sources.

Example:

```text
The person worked with AI agents at the Microsoft AI Innovators Program. [Source 1]
```

Implemented:

- [x] Source-number validation
- [x] Claim-to-source extraction
- [x] Invalid citation rejection
- [x] Multi-claim citation validation
- [x] Swapped-source rejection
- [x] Unsupported secondary-claim rejection

---

## 14. Post-Generation Claim Validation

MemoRAG does not assume that grounded prompting alone prevents hallucinations.

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

- [x] Evidence-to-claim NLI validation
- [x] Strict claim-validation threshold
- [x] Separation of provenance and semantic validation
- [x] Semantic claim extraction
- [x] Supporting hypothesis preservation
- [x] Semantic-contract fallback
- [x] Topic-drift rejection

The semantic-contract mechanism handles cases where generated wording faithfully expresses an already validated query hypothesis but direct evidence-to-generated-claim NLI is overly sensitive to paraphrasing.

---

# Hallucination Protection

MemoRAG uses several independent safeguards instead of relying on retrieval similarity alone.

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

- Semantically similar but irrelevant evidence
- Incorrect topic matches
- Wrong organization/project attribution
- Unsupported generated claims
- Incorrect citations
- Semantic topic drift
- Duplicate evidence caused by chunk overlap

---

# Testing

A regression suite is in place for the evidence pipeline.

Current coverage includes:

- [x] Structural evidence filtering
- [x] AI-agent evidence
- [x] RAG evidence
- [x] Unsupported topics
- [x] Dynamic topic extraction
- [x] Parent/provenance preservation
- [x] Citation validation
- [x] Multi-claim answers
- [x] Swapped citations
- [x] Wrong-parent attribution
- [x] Unsupported secondary claims
- [x] Semantic-contract fallback
- [x] Semantic topic-drift rejection
- [x] Evidence overlap deduplication

Current regression checkpoint:

```text
42 tests passing
```

---

# Current Example

### Query

```text
Where did this person work with AI agents,
and where did this person use IBM Planning Analytics?
```

### Grounded Output

```text
The person worked with AI agents at the Microsoft AI Innovators Program
— AI Engineering Intern 2026. [Source 1]

The person used IBM Planning Analytics with Watson (TM1) to support
financial planning projects during their internship at Cubewise,
Türkiye, from 2025 to 2026. [Source 2]
```

Each subanswer is independently grounded and validated against its corresponding evidence.

> This CV-based example is currently used as a development fixture. **MemoRAG itself is not intended to be CV-specific.**

---

# Roadmap

## 1. Evidence Pipeline Hardening

- [ ] Expand regression coverage across additional query formulations
- [ ] Test ambiguous queries
- [ ] Test partially supported compound queries
- [ ] Test larger compound questions
- [ ] Stress-test dynamic topic extraction
- [ ] Test additional topic synonyms and paraphrases
- [ ] Expand negative evidence tests
- [ ] Expand provenance edge-case tests

---

## 2. Local Knowledge Library & File Discovery

MemoRAG should eventually discover knowledge without requiring the user to manually upload or locate every file.

- [ ] Select user-authorized indexing locations
- [ ] Scan selected folders and drives recursively
- [ ] Automatic supported-file discovery
- [ ] Preserve exact file paths
- [ ] Extract filesystem metadata
- [ ] Extract document metadata where available
- [ ] Content hashing
- [ ] Detect unchanged files during subsequent scans
- [ ] Detect newly created files
- [ ] Detect modified files
- [ ] Detect deleted files
- [ ] Detect moved or renamed files
- [ ] Background incremental indexing
- [ ] Filesystem watcher
- [ ] Ignore rules for system/cache/build directories
- [ ] Large-library indexing controls
- [ ] User privacy and indexing permissions

Initial target locations may include user-authorized areas such as:

```text
Documents
Desktop
Downloads
OneDrive / synchronized folders
Custom folders
Optional additional drives
```

MemoRAG should not require unrestricted disk access by default. The user controls which locations become part of the local knowledge index.

---

## 3. Duplicate & Version Intelligence

A central product goal is to answer questions even when users do not remember **which version** contains the information.

For example:

```text
contract.docx
contract_final.docx
contract_final_v2.docx
contract_FINAL_revised.pdf
```

These should not necessarily be treated as four unrelated documents.

Planned:

- [ ] Exact duplicate detection using hashes
- [ ] Near-duplicate document detection
- [ ] Document similarity scoring
- [ ] Version-family grouping
- [ ] Filename-based version signals
- [ ] Modification-time signals
- [ ] Content-based version relationships
- [ ] Latest-version inference
- [ ] Relevant-version retrieval
- [ ] Cross-version evidence comparison
- [ ] Surface conflicting facts across versions
- [ ] Preserve exact version provenance
- [ ] Allow users to inspect a version family

The goal is **not simply to select the newest file**. MemoRAG should retrieve the version that actually contains the requested evidence and clearly identify that version.

---

## 4. Cross-Document Retrieval

- [ ] Search across the entire indexed knowledge library
- [ ] Retrieve evidence from multiple documents
- [ ] Cross-document source attribution
- [ ] Handle duplicate evidence across files
- [ ] Handle contradictory evidence
- [ ] Rank evidence using semantic and metadata signals
- [ ] Support questions whose answers are distributed across multiple files
- [ ] Test large heterogeneous document collections

---

## 5. Multi-Document & Format Validation

- [ ] Test MemoRAG on documents other than CVs
- [ ] Test technical documentation
- [ ] Test reports
- [ ] Test contracts
- [ ] Test personal notes
- [ ] Test source-code collections
- [ ] Test multiple documents containing overlapping evidence
- [ ] Test multiple document versions
- [ ] Expand supported document formats where useful

---

## 6. Retrieval Evaluation

- [ ] Build a small evaluation dataset
- [ ] Measure retrieval precision
- [ ] Measure evidence acceptance precision
- [ ] Measure unsupported-answer rejection rate
- [ ] Measure citation correctness
- [ ] Evaluate compound-query performance
- [ ] Evaluate version-resolution accuracy
- [ ] Evaluate cross-document retrieval
- [ ] Measure false-attribution rate

---

## 7. UI & Product Layer

MemoRAG should become a **local knowledge workspace** rather than remain a terminal-only demo.

Planned interface:

- [ ] Professional local web/desktop UI
- [ ] Global **Ask MemoRAG** interface
- [ ] Knowledge Library
- [ ] Indexed-location management
- [ ] Drag-and-drop/manual indexing when desired
- [ ] Indexing progress and status
- [ ] Recently changed files
- [ ] Searchable file browser
- [ ] Duplicate families
- [ ] Version families
- [ ] Source cards
- [ ] Clickable citations
- [ ] Expandable evidence inspector
- [ ] Exact file/page/section/chunk provenance
- [ ] Validation indicators
- [ ] Local/private status indicator
- [ ] Indexing and privacy settings

A citation should eventually allow the user to inspect:

```text
Source
  File
  Exact path
  Version
  Page / section
  Evidence text
  Parent context

Validation
  Structural Gate
  Topical Gate
  NLI
  Provenance
  Claim Validation
```

The validation internals should remain inspectable without overwhelming the normal answer experience.

---

## 8. Code Cleanup

- [ ] Remove temporary debug output
- [ ] Remove temporary diagnostic scripts
- [ ] Consolidate configuration constants
- [ ] Review module boundaries
- [ ] Improve type hints
- [ ] Improve internal documentation
- [ ] Review error handling
- [ ] Separate development diagnostics from production logging

---

## 9. Developer Experience

- [ ] Finalize installation instructions
- [ ] Document environment setup
- [ ] Document indexing workflow
- [ ] Document query workflow
- [ ] Document local model requirements
- [ ] Add example commands
- [ ] Add architecture diagram
- [ ] Add troubleshooting section
- [ ] Document privacy model and indexed-location behavior

---

## 10. Final Demo

- [ ] Prepare a heterogeneous local document library
- [ ] Include multiple versions of selected documents
- [ ] Define representative supported queries
- [ ] Define hallucination/adversarial queries
- [ ] Demonstrate "I don't know where the file is" retrieval
- [ ] Demonstrate version resolution
- [ ] Demonstrate cross-document retrieval
- [ ] Demonstrate compound-query handling
- [ ] Demonstrate provenance validation
- [ ] Demonstrate unsupported-query abstention
- [ ] Demonstrate dynamic topic extraction
- [ ] Demonstrate evidence inspection through the UI
- [ ] Record final example outputs

---

# Development Status

MemoRAG has two useful progress measures because the evidence engine is substantially further along than the complete product.

## Core RAG & Evidence Engine

```text
███████████████████░  ~95%
```

### Completed

- Local parsing and indexing
- Semantic retrieval
- Evidence construction
- Structural filtering
- Topical filtering
- Dynamic topic extraction
- NLI validation
- Parent/provenance association
- Evidence overlap deduplication
- Compound-query processing
- Grounded generation
- Citation validation
- Post-generation claim validation
- Regression coverage for key evidence behaviors

Remaining work in this layer is primarily hardening, broader evaluation, cleanup, and edge-case testing.

## Full MemoRAG Product

```text
███████████████░░░░░  ~70–75%
```

### Major Remaining Work

- File-system discovery
- Filesystem monitoring
- Full local knowledge-library indexing
- Duplicate intelligence
- Version-family detection
- Version-aware retrieval
- Cross-document retrieval validation
- Broader document-format testing
- Product UI
- Large-library evaluation
- Documentation
- Final demo preparation

> These percentages are approximate development checkpoints rather than formal engineering metrics and will be revised as the remaining product layers are implemented.

---

# Privacy Direction

MemoRAG is designed around a **local-first architecture**.

The intended product model is:

- Files remain on the user's machine
- Indexing locations are explicitly user-authorized
- Retrieval operates over the local knowledge index
- Local models are preferred for embeddings, NLI, and generation
- Exact source provenance remains available to the user
- Unrelated/system directories are excluded from indexing by default

As the filesystem layer is implemented, permission boundaries and privacy controls will be treated as first-class product requirements.

---

# Core Design Principles

## 1. Retrieval is not evidence

A semantically similar chunk is only a candidate. It must pass stronger evidence checks before supporting an answer.

## 2. Generated text is not automatically supported

Generation is followed by citation, provenance, and semantic claim validation.

## 3. The user should not need to remember the file

MemoRAG should locate information across the authorized local knowledge space rather than require the user to identify the source beforehand.

## 4. The user should not need to remember the version

When several revisions exist, MemoRAG should identify the relevant version and preserve exact provenance instead of silently treating all copies as equivalent.

## 5. Provenance must survive the entire pipeline

File, version, page, section, chunk, and parent context should remain traceable from indexing through the final answer.

## 6. Local-first should remain meaningful

Local execution is not only an implementation detail. Privacy, user-controlled indexing scope, and inspectable sources are core product properties.

---

# Vision

> ## Your files should behave like searchable memory.

Instead of manually navigating folders, remembering filenames, opening several versions, and searching documents individually, the user should be able to ask for the information directly.

MemoRAG's job is to determine:

```text
What information is being requested?
        ↓
Which files may contain it?
        ↓
Which version is relevant?
        ↓
Which exact evidence supports it?
        ↓
Can the generated claim be verified?
        ↓
Where did the answer come from?
```

The end goal is a **private local knowledge assistant** that can find, validate, and explain information across a user's own files while keeping every answer traceable to its source.
