<div align="center">

# Recall

### Offline Local RAG Assistant powered by Microsoft Foundry Local

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Foundry Local](https://img.shields.io/badge/Microsoft-Foundry%20Local-0078D4?logo=microsoft&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?logo=huggingface&logoColor=black)

![License](https://img.shields.io/github/license/esinbegumkaya/recall)
![Release](https://img.shields.io/github/v/release/esinbegumkaya/recall)
![Last Commit](https://img.shields.io/github/last-commit/esinbegumkaya/recall)
![Repo Size](https://img.shields.io/github/repo-size/esinbegumkaya/recall)

![Tests](https://img.shields.io/badge/Tests-42%2F42%20Passing-brightgreen)
![Compile](https://img.shields.io/badge/compileall-Passing-success)
![Status](https://img.shields.io/badge/Status-v1.0.0-success)
![Offline](https://img.shields.io/badge/Offline-Yes-success)
![RAG](https://img.shields.io/badge/RAG-Hybrid-blueviolet)
[![CI](https://github.com/esinbegumkaya/recall/actions/workflows/ci.yml/badge.svg)](https://github.com/esinbegumkaya/recall/actions/workflows/ci.yml)
</div>


<div align="center">

### Offline Local RAG Assistant powered by Microsoft Foundry Local

A local-first Retrieval-Augmented Generation (RAG) assistant for indexing, searching, validating, and answering questions from documents on a user's computer.

**Python 3.11 · Microsoft Foundry Local · Streamlit · SQLite · Hybrid Retrieval · NLI**

**Status:** Completed · **Unit tests:** 42/42 passing · **End-to-end evaluation:** 5/5 passing (100%)

</div>

---

## Overview

Recall is a local document question-answering assistant built around Retrieval-Augmented Generation (RAG). Documents are parsed and indexed on the user's machine, embeddings are stored locally in SQLite, relevant chunks are retrieved with a hybrid search pipeline, evidence is validated before generation, and answers are generated with Microsoft Foundry Local.

The project began from the Microsoft Foundry Local summer-school RAG plan and extends the basic reference pipeline with hybrid retrieval, structural reranking, evidence validation, provenance tracking, citation validation, responsible abstention, incremental indexing, Firebase authentication, and a Streamlit interface.

> **Local-first note:** document parsing, indexing, retrieval, evidence processing, embeddings, and LLM inference are designed to run locally. Firebase Authentication is an optional online authentication layer. Initial model/dependency acquisition may also require network access before offline runtime.

---

## Key Features

- **Local LLM inference** with Microsoft Foundry Local
- **Local embeddings** and persistent SQLite storage
- **PDF, DOCX, TXT, Markdown, and Python** document ingestion
- Recursive folder discovery and incremental indexing
- Dense semantic retrieval
- SQLite **FTS5** keyword retrieval
- Hybrid retrieval with **Reciprocal Rank Fusion (RRF)**
- Structural and section-aware reranking
- Parent-child context recovery and provenance tracking
- Topic-aware evidence filtering
- Natural Language Inference (**NLI**) evidence validation
- Citation-aware grounded generation
- Deterministic grounded fallback when generated output cannot be safely validated but direct evidence is available
- Responsible abstention when the available files do not support an answer
- Streamlit UI with source visualization and theme support
- Optional Firebase Authentication
- Automated regression and end-to-end evaluation

---

## Architecture

```text
                         +----------------------+
                         |     Streamlit UI     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Recall Engine     |
                         +----------+-----------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
     +----------------+    +----------------+    +------------------+
     | Hybrid Search  |    | Evidence Gate  |    | Foundry Local    |
     | Dense + FTS5   |    | Topic + NLI    |    | Local LLM        |
     +----------------+    +----------------+    +------------------+
              |                     |                     |
              +---------------------+---------------------+
                                    |
                                    v
                         +----------------------+
                         |   SQLite Database    |
                         +----------+-----------+
                                    ^
                                    |
                         +----------------------+
                         |  Document Indexer    |
                         +----------+-----------+
                                    ^
                                    |
                         PDF / DOCX / TXT / MD / PY
```

The central `RecallEngine` coordinates retrieval, evidence validation, local generation, citation validation, abstention, and final response construction.

---

## Query Workflow

```text
User Question
     |
     v
Hybrid Retrieval
     |
     v
Candidate Chunks
     |
     v
Structural / Topic Filtering
     |
     v
Evidence Splitting + Parent Context Recovery
     |
     v
NLI Validation
     |
     v
Validated Evidence
     |
     v
Foundry Local Generation
     |
     v
Citation / Claim Validation
     |
     +---- valid generated answer ------> Final Answer
     |
     +---- direct evidence available ---> Grounded Deterministic Fallback
     |
     +---- insufficient support --------> Responsible Abstention
```

Recall does not blindly send retrieved chunks to the LLM. Candidate evidence is filtered and validated first. Generated factual claims are then checked before the answer is returned.

---

## Supported Files

| Format | Extension | Supported |
|---|---|:---:|
| PDF | `.pdf` | ✅ |
| Microsoft Word | `.docx` | ✅ |
| Plain text | `.txt` | ✅ |
| Markdown | `.md` | ✅ |
| Python source | `.py` | ✅ |

Scanned/image-only PDFs are not currently OCR-processed.

---

## Project Structure

The exact repository can evolve, but the main components are organized as follows:

```text
Recall/
├── src/
│   └── recall/
│       ├── engine.py
│       ├── generator.py
│       ├── retriever.py
│       ├── hybrid_retriever.py
│       ├── retrieval.py
│       ├── semantic_reranker.py
│       ├── evidence.py
│       ├── evidence_gate.py
│       ├── evidence_judge.py
│       ├── nli_judge.py
│       ├── query_planner.py
│       ├── intent.py
│       ├── parser.py
│       ├── chunker.py
│       ├── database.py
│       └── crawler.py
├── ui/
│   ├── app.py
│   └── firebase_auth.py
├── scripts/
│   ├── index_folder.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_end_to_end.py
│   ├── final_check.py
│   └── ...
├── tests/
│   └── test_evidence_pipeline.py
├── data/
│   ├── evaluation/
│   ├── index/
│   └── samples/
├── .streamlit/
│   └── secrets.toml.example
├── requirements.txt
├── pytest.ini
├── FINAL_VALIDATION.md
└── README.md
```

---

## Requirements

Recommended environment:

- Windows 11
- Python 3.11
- Microsoft Foundry Local
- Sufficient local storage/RAM for the selected local models

The validated development environment used Python 3.11 and Microsoft Foundry Local. The project requirements include the Foundry Local SDK, Streamlit, PyTorch/Transformers-related components, PDF/DOCX parsing libraries, and Pytest.

---

## Installation

### 1. Clone the repository

```powershell
git clone <YOUR_REPOSITORY_URL>
cd Recall
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, adjust the execution policy according to your local security requirements.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Install / configure Microsoft Foundry Local

Install Microsoft Foundry Local and ensure its runtime/SDK is available before starting generation-enabled Recall.

The application loads the local model lazily, so the model is initialized when generation is first required rather than at application import time.

---

## Firebase Authentication

Firebase provides the optional login layer for the Streamlit application.

Create:

```text
.streamlit/secrets.toml
```

Use the repository's example file as the template:

```text
.streamlit/secrets.toml.example
```

Example structure:

```toml
[firebase]
apiKey = "..."
authDomain = "..."
projectId = "..."
storageBucket = "..."
messagingSenderId = "..."
appId = "..."
```

### Security

Do **not** commit your real `.streamlit/secrets.toml` file.

Keep it in `.gitignore` and commit only the example configuration.

Firebase Authentication requires network connectivity. It is separate from Recall's local RAG inference pipeline.

---

## Running the Application

From the repository root:

```powershell
streamlit run ui/app.py
```

The UI provides document indexing/search controls, grounded answers, source information, and application navigation.

---

## Indexing Documents

### Index default computer locations

```powershell
python .\scripts\index_folder.py --computer
```

Recall discovers supported documents and compares them with the existing local index.

The incremental indexing pipeline distinguishes between:

- new files,
- modified files,
- unchanged files.

Unchanged files can be skipped rather than unnecessarily parsed and embedded again.

### Index a custom folder

```powershell
python .\scripts\index_folder.py --folder "C:\path\to\documents"
```

The local index is persisted in SQLite.

---

## Retrieval Pipeline

Recall uses a hybrid retrieval architecture rather than relying on a single search strategy.

### Dense retrieval

Local embeddings provide semantic similarity search and improve retrieval for paraphrased or conceptually related queries.

### SQLite FTS5

FTS5 complements semantic retrieval with lexical matching. It is especially useful for exact names, identifiers, filenames, technical terminology, and programming-related tokens.

### Reciprocal Rank Fusion

Dense and lexical rankings are combined using Reciprocal Rank Fusion.

### Structural reranking

Additional ranking signals can use document structure, including:

- section information,
- headings,
- parent context,
- query intent.

This is particularly useful for structured documents such as CVs and reports.

---

## Evidence Validation

Retrieval is treated as candidate discovery, not final proof.

Before generation, Recall processes evidence through multiple stages:

1. Structural evidence splitting
2. Topic filtering
3. Evidence normalization
4. Parent-context association
5. Provenance preservation
6. NLI-based validation

This reduces the chance that a semantically similar but unsupported chunk is treated as evidence for the user's question.

---

## Grounded Generation

Validated evidence is formatted into a constrained prompt for Microsoft Foundry Local.

The generation instructions require the model to:

- use only validated evidence,
- write grounded factual claims,
- cite supporting sources,
- preserve evidence terminology where appropriate,
- abstain when the evidence does not support an answer.

The generation layer is therefore downstream of retrieval and evidence validation rather than acting as an unrestricted chatbot.

---

## Citation Validation and Safe Fallback

Generated output is checked before it is accepted.

If a generated answer contains unsupported or inadequately cited factual claims, Recall does not blindly return it. When the engine has sufficiently direct validated evidence, it can construct a deterministic, source-grounded fallback response from that evidence.

If the available files do not actually support the requested information, Recall returns the abstention message instead of fabricating an answer:

```text
The available files do not provide enough evidence to answer this question.
```

This distinction is important: **generation failure is not automatically the same as evidence failure**.

---

## Testing

### Full delivery smoke-check

Run:

```powershell
python .\scripts\final_check.py
```

Latest validated result:

```text
42 passed, 1 warning

[OK] Final delivery smoke-check passed.
```

**Unit/regression status: 42/42 passing.**

The remaining PyTorch deprecation warning does not cause a test failure.

---

## End-to-End Evaluation

Run:

```powershell
python .\scripts\evaluate_end_to_end.py
```

The current benchmark contains five cases covering:

- answerable AI-experience retrieval,
- unsupported-query abstention,
- provenance preservation,
- AI-agent topic precision,
- medical-image project retrieval.

### Final result

| Case | Purpose | Result |
|---|---|:---:|
| E2E-01 | Supported AI-experience question | PASS |
| E2E-02 | Unsupported favorite-restaurant question | PASS |
| E2E-03 | IBM Planning Analytics / Cubewise provenance | PASS |
| E2E-04 | AI-agent-specific evidence | PASS |
| E2E-05 | Medical-image classification project | PASS |
| **Total** | **5 queries** | **5/5 — 100%** |

Latest validated run:

```text
Queries   : 5
Passed    : 5
Failed    : 0
Pass Rate : 100.0%

[OK] End-to-end evaluation passed.
```

The end-to-end test exercises the complete path:

```text
retrieval
→ evidence gates
→ NLI
→ Foundry Local generation
→ citation validation
→ final response / safe fallback
```

---

## Retrieval Evaluation

The production hybrid retrieval benchmark uses Hit@K and Mean Reciprocal Rank (MRR).

### Production Hybrid Retrieval

| Metric | Development | Held-out |
|---|---:|---:|
| Hit@1 | 0.50 | 0.40 |
| Hit@3 | 0.60 | 0.60 |
| Hit@5 | 0.60 | 0.60 |
| MRR | 0.55 | 0.467 |

These metrics describe retrieval ranking performance and should be interpreted separately from the final 5/5 end-to-end functional benchmark.

### Structural Retrieval Experiment

A separate structural-ranking experiment achieved:

| Metric | Held-out Result |
|---|---:|
| Hit@1 | 0.90 |
| Hit@3 | 1.00 |
| Hit@5 | 1.00 |
| MRR | 0.95 |

These experimental results demonstrate the value of document structure and section-aware signals, but they are not presented as the production hybrid benchmark.

---

## Performance

Performance depends heavily on:

- CPU/NPU/GPU availability,
- RAM,
- storage speed,
- local model selection,
- model cache state,
- query/evidence complexity,
- number and size of indexed documents.

The final five-case end-to-end validation run completed in approximately **270.51 seconds total** on the tested environment, with individual cases taking roughly **43–72 seconds**.

This end-to-end timing includes multiple pipeline stages and local model work. It should not be confused with retrieval-only latency.

Performance optimization remains an area for future work.

---

## Privacy and Offline Behavior

Recall follows a local-first architecture:

- documents are parsed locally,
- embeddings are generated locally,
- the SQLite knowledge base is local,
- retrieval runs locally,
- evidence validation runs locally,
- Foundry Local performs local LLM inference.

No cloud-hosted LLM is required for the RAG pipeline.

However, two practical distinctions are important:

1. **Firebase Authentication is optional and online.**
2. Model/dependency setup or first-time acquisition may require Internet access before models are available locally.

Once the required local components are installed/cached and optional online authentication is not required, the core RAG workflow is designed for local operation.

---

## Known Limitations

- Initial indexing of large document collections can take significant time.
- End-to-end generation latency depends strongly on local hardware and model choice.
- Retrieval quality still depends on document structure and chunking.
- Broad queries can retrieve semantically related but non-essential evidence.
- Local LLM output can occasionally fail strict citation validation; Recall therefore includes grounded fallback behavior.
- NLI scores are useful validation signals but are not infallible.
- OCR is not implemented for scanned/image-only PDFs.
- Image understanding and image retrieval are not currently implemented.
- Parent-context quality depends on the structure of the source document.
- Firebase login introduces an optional network dependency.

---

## Screenshots

Add final project screenshots under a repository directory such as:

```text
docs/screenshots/
```

Recommended captures:

1. Login screen
2. Main dashboard
3. Document indexing
4. Search / question interface
5. Generated answer with citations
6. Source viewer
7. Evaluation result showing `5/5` and `100%`

Example Markdown once the images are added:

```markdown
![Recall Dashboard](docs/screenshots/dashboard.png)
![Grounded Answer](docs/screenshots/grounded-answer.png)
![End-to-End Evaluation](docs/screenshots/e2e-100.png)
```

---

## Final Validation Status

| Component | Status |
|---|:---:|
| Document ingestion | ✅ PASS |
| Chunking and metadata | ✅ PASS |
| Local embeddings | ✅ PASS |
| SQLite persistence | ✅ PASS |
| Dense retrieval | ✅ PASS |
| FTS5 retrieval | ✅ PASS |
| Hybrid retrieval | ✅ PASS |
| Structural reranking | ✅ PASS |
| Evidence filtering | ✅ PASS |
| Parent/provenance handling | ✅ PASS |
| NLI validation | ✅ PASS |
| Foundry Local integration | ✅ PASS |
| Citation validation | ✅ PASS |
| Responsible abstention | ✅ PASS |
| Deterministic grounded fallback | ✅ PASS |
| Streamlit UI | ✅ PASS |
| Firebase login | ✅ PASS |
| Unit/regression tests | ✅ 42/42 |
| End-to-end evaluation | ✅ 5/5 (100%) |

---

## Future Work

Potential improvements include:

- faster local generation,
- retrieval and model caching optimizations,
- adaptive chunk sizing,
- cross-encoder reranking,
- learning-to-rank,
- OCR,
- table extraction,
- image understanding,
- richer citation visualization,
- conversation memory,
- multi-user document libraries,
- retrieval analytics,
- improved parent-context extraction,
- broader end-to-end benchmark suites.

---

## Acknowledgements

Recall was developed around the Microsoft Foundry Local learning project and the local RAG architecture described in the summer-school project plan, then extended with additional retrieval, evidence-validation, provenance, citation, testing, authentication, and UI functionality.

Technologies used include Microsoft Foundry Local, Python, Streamlit, SQLite, PyTorch/Transformers components, and Firebase Authentication.

---

## License

If this repository is distributed under the MIT License, include a top-level `LICENSE` file containing the MIT License text.

Do not advertise an MIT license badge until that file is actually present in the repository.

---

<div align="center">

### Recall

**Local documents. Local retrieval. Local inference. Grounded answers.**

</div>
