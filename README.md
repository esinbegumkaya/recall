# Recall

**A privacy-first, local AI assistant for finding files across your computer using natural language.**

Recall indexes files on your computer and lets you search for them the way you would ask another person.

Instead of remembering exact filenames, folders, or keywords, you can ask:

```text
Find my most recent CV.
Where is the document I wrote about RAG?
Find the presentation about artificial intelligence.
Which file mentions Microsoft and AI agents?
Find the network assignment from last week.
```

Recall combines local query understanding, metadata search, full-text retrieval, fuzzy filename matching, semantic similarity, and evidence validation to identify the most relevant files while keeping the search pipeline local.

---

## Why Recall?

Traditional desktop search works well when you already know what a file is called.

Real searches are often much less precise.

You may remember:

* what the document was about,
* roughly when you worked on it,
* a company or person mentioned inside it,
* the type of document,
* part of its filename,
* or simply the idea you were looking for.

Recall turns those incomplete memories into structured search signals and searches across both **file metadata and file contents**.

The goal is simple:

> **You should not need to remember where a file is stored in order to find it.**

---

## Key Features

### Natural-Language File Search

Search using conversational queries instead of exact filenames.

Examples:

```text
Find my latest CV.
Find the document where I wrote about RAG.
Where are my Turkish notes?
Find my AI presentation.
```

The query planner extracts signals such as:

* semantic concepts
* lexical keywords
* filename clues
* content clues
* file-type hints
* location hints
* date preferences
* requested actions

---

### Multilingual & Typo-Tolerant Queries

Recall is designed to handle flexible queries rather than requiring a rigid command syntax.

Queries can contain:

* English
* Turkish
* incomplete descriptions
* approximate filenames
* spelling mistakes
* mixed semantic and metadata constraints

For example:

```text
en güncel cvmi getir
cvmn en gncelni getir
rag hakkında yazdığım belge
Where is my most recent presentation about AI?
```

---

### Hybrid Retrieval

Recall does not depend on a single retrieval method.

It combines multiple signals:

```text
Natural-language query
        │
        ▼
Query Understanding
        │
        ▼
Search Signal Extraction
        │
        ├── Filename / path matching
        ├── Full-text search
        ├── Metadata / date relevance
        ├── Fuzzy matching
        └── Semantic similarity
        │
        ▼
Hybrid Ranking
        │
        ▼
Semantic Reranking
        │
        ▼
Evidence Validation
        │
        ▼
Relevant Files
```

This allows Recall to handle both precise searches such as:

```text
Harvard CV
```

and semantic searches such as:

```text
the document where I described building AI agents
```

---

## Privacy-First Architecture

Recall is designed around **local file discovery**.

The system indexes local files into a local SQLite database and performs retrieval using local components.

The current architecture uses:

* local file crawling
* local parsing and chunking
* SQLite metadata storage
* SQLite FTS5 full-text search
* local embeddings
* local semantic reranking
* local NLI-based evidence validation
* local query planning

This architecture avoids requiring users to upload their personal documents to a remote search service.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     User Query      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Query Planner    │
                         │   Local Qwen Model  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │      Search Signal Extraction   │
                  │                                 │
                  │ concepts · keywords · filename  │
                  │ content · type · location · date│
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │        Hybrid Retriever         │
                  │                                 │
                  │ SQLite FTS5                     │
                  │ Filename / Path Fuzzy Matching  │
                  │ Metadata & Recency              │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ Semantic Reranker   │
                         │ Local Embeddings    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Evidence Pipeline  │
                         │ Structural + Topic  │
                         │ Gates + Local NLI   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Grounded Result   │
                         │ File · Path · Page  │
                         │ Section · Evidence  │
                         └─────────────────────┘
```

---

## Retrieval Pipeline

### 1. Local File Crawling

Recall scans configured directories and discovers supported documents while ignoring development and system directories such as virtual environments, Git metadata, caches, and other irrelevant locations.

The crawler collects metadata including:

* file path
* filename
* extension
* file size
* creation time
* modification time

---

### 2. Parsing & Chunking

Supported files are parsed into searchable text chunks.

Where available, Recall preserves provenance information such as:

* source file
* page number
* section name
* chunk index

This information is later used to explain why a result was retrieved.

---

### 3. Full-Text Index

Parsed chunks are stored in SQLite and indexed using **FTS5**.

This provides fast lexical retrieval without requiring every file to be embedded during the initial indexing process.

---

### 4. Query Planning

A local language model converts the user's request into structured search signals.

A query can produce signals for:

```text
concepts
keywords
filename_terms
content_terms
extensions
location_terms
date preferences
retrieval weights
```

Deterministic heuristics are available as a fallback if model-based planning is unavailable.

---

### 5. Hybrid Candidate Retrieval

Recall combines:

* FTS lexical relevance
* filename similarity
* path similarity
* fuzzy matching
* file metadata
* modification/creation dates
* query-specific retrieval weights

Different queries therefore produce different ranking strategies.

A filename-oriented query can prioritize filenames, while a conceptual query can give semantic evidence more influence.

---

### 6. Lazy Semantic Reranking

Recall deliberately avoids embedding the entire computer during initial indexing.

Instead:

1. lexical and metadata retrieval creates a candidate set,
2. relevant candidate chunks are selected,
3. embeddings are generated only when required,
4. embeddings are cached locally,
5. semantic similarity reranks the candidates.

This keeps initial indexing substantially lighter while still providing semantic retrieval.

---

### 7. Evidence Validation

For evidence-sensitive queries, Recall applies multiple validation layers.

The evidence pipeline includes:

```text
Structural Gate
      ↓
Topical Gate
      ↓
Provenance Constraints
      ↓
NLI Validation
      ↓
Grounded Evidence
```

This reduces false-positive evidence caused by semantically similar but unrelated text.

For example, a query asking whether AI-agent work occurred at a specific organization should not accept AI-agent evidence originating from a different organization.

---

## Grounded Results

Recall is designed to return more than a filename.

Results can include:

* filename
* full local path
* relevance score
* modification date
* page number
* section
* supporting text
* evidence diagnostics

The interface can also open the selected file or its containing folder directly.

---

## User Interface

Recall includes a Streamlit interface focused on file discovery.

The main workflow is intentionally simple:

```text
Ask a question
      ↓
Search my computer
      ↓
Understand query
      ↓
Search indexed files
      ↓
Semantic reranking
      ↓
Display ranked files
```

Search progress is tied to actual retrieval stages rather than a simulated loading animation.

---

## Tech Stack

| Component           | Technology      |
| ------------------- | --------------- |
| Language            | Python 3.11     |
| UI                  | Streamlit       |
| Database            | SQLite          |
| Lexical Search      | SQLite FTS5     |
| Fuzzy Matching      | RapidFuzz       |
| Local AI Runtime    | Foundry Local   |
| Query Planner       | Qwen 3.5        |
| Embeddings          | Qwen3 Embedding |
| Evidence Validation | DeBERTa NLI     |
| Testing             | pytest          |

---

## Project Structure

```text
Recall/
│
├── src/
│   └── recall/
│       ├── crawler.py
│       ├── parser.py
│       ├── chunker.py
│       ├── database.py
│       ├── query_planner.py
│       ├── hybrid_retriever.py
│       ├── semantic_reranker.py
│       ├── evidence.py
│       ├── evidence_gate.py
│       ├── nli_judge.py
│       └── engine.py
│
├── ui/
│   └── app.py
│
├── scripts/
│   ├── index_folder.py
│   └── evaluate_*.py
│
├── tests/
│   └── test_evidence_pipeline.py
│
├── data/
│   ├── evaluation/
│   └── samples/
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd memorag
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Recall

Make the source package available:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Start the interface:

```powershell
streamlit run .\ui\app.py
```

Then open the local Streamlit address shown in the terminal.

---

## Testing

Run the regression suite with:

```bash
pytest -q
```

Current regression status:

```text
42 passed
```

The suite covers key parts of the evidence and retrieval pipeline, including structural filtering, topical validation, provenance handling, citation validation, and NLI-based evidence admission.

---

## Example Queries

```text
en güncel cvmi getir
```

```text
rag hakkında yazdığım belge
```

```text
geçen hafta hocadan gelen network ödevi
```

```text
Microsoft geçen son cvyi bul
```

```text
Where is my most recent presentation about AI?
```

```text
Find the document where I worked with AI agents and RAG systems professionally.
```

---

## Design Principles

### Local First

Personal documents should remain under the user's control.

### Retrieval Before Generation

Finding the correct evidence is more important than generating a fluent but unsupported answer.

### Evidence Over Guessing

When valid evidence cannot be found, Recall should abstain rather than fabricate a result.

### Hybrid Search Over Single-Method Search

Filename matching, lexical retrieval, metadata, and semantics solve different parts of the file-discovery problem.

### Explainable Results

A useful search system should be able to show where a result came from and why it was considered relevant.

---

## Current Status

Recall is under active development.

The current version includes:

* computer-wide local file indexing
* incremental metadata-aware indexing
* SQLite FTS5 search
* multilingual natural-language query planning
* typo-tolerant filename matching
* hybrid retrieval
* lazy semantic reranking
* evidence extraction
* structural and topical evidence gates
* local NLI validation
* provenance-aware grounding
* Streamlit file-discovery interface
* local file and folder opening
* regression testing

Planned improvements include further indexing optimization, broader parser coverage, retrieval evaluation, UI refinement, and packaging the application for easier installation.

---

## Motivation

Modern computers can contain thousands of documents spread across Downloads, Desktop, Documents, cloud-synced folders, university materials, work files, and personal archives.

The information is there.

The difficult part is remembering **where**.

Recall explores a different interaction model:

> Instead of navigating folders, describe what you remember.

---

## Author

**Esin Begüm Kaya**

Computer Engineering · AI · Data · Software Engineering

---

## Disclaimer

Recall is an independent local file-search and retrieval project. It is not affiliated with other projects or products that use the names “Recall” or “MemoRAG”.
