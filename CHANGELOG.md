# Changelog

All notable changes to this project are documented in this file.

The format is inspired by **Keep a Changelog**, and this project follows **Semantic Versioning** where applicable.

---

## [1.0.0] - 2026-09-12

### 🎉 Initial Stable Release

This release marks the first stable version of **Recall**, an offline Retrieval-Augmented Generation (RAG) assistant powered by **Microsoft Foundry Local**.

---

### ✨ Added

#### Core RAG Pipeline

- Local Retrieval-Augmented Generation (RAG) architecture
- Microsoft Foundry Local integration
- Hybrid semantic + lexical retrieval pipeline
- Local document indexing
- Incremental indexing
- SQLite-based document storage
- Automatic chunk generation
- Metadata extraction
- Source-aware retrieval

---

#### Evidence Validation

- Multi-stage evidence filtering
- Structural evidence gate
- Topic-aware evidence gate
- Natural Language Inference (NLI) validation
- Parent-child context association
- Citation-aware answer generation
- Hallucination reduction pipeline

---

#### Search

- Hybrid retrieval
- Semantic reranking
- Duplicate removal
- File diversification
- Candidate quality filtering
- Query planning
- Intent detection

---

#### Document Support

Supported document types:

- PDF
- DOCX
- TXT
- Markdown
- Python source files

---

#### User Interface

- Streamlit desktop interface
- Dark / Light theme
- Search interface
- Indexed document explorer
- Source viewer
- Answer panel
- Evidence visualization

---

#### Authentication

- Firebase Authentication
- Google Sign-In
- Protected application access

---

#### Evaluation

Implemented evaluation pipelines for:

- Retrieval quality
- Structural ranking
- End-to-end generation
- Evidence validation
- NLI validation

---

#### Testing

- 42/42 unit tests passing
- Static syntax validation using compileall

---

#### Documentation

Added:

- README
- FINAL_VALIDATION
- MIT License
- GitHub Actions CI
- CITATION metadata
- Project architecture
- Installation guide
- Usage guide

---

### ⚡ Performance

- Offline inference
- Local document processing
- Incremental indexing
- Hybrid retrieval optimization
- Citation-based response generation

---

### 🔒 Security

- Local-first architecture
- No external LLM API required
- No API keys committed
- Sensitive configuration excluded from repository
- MIT licensed

---

### 🛠 Developer Experience

- GitHub Actions CI
- Automated compilation checks
- Automated unit testing
- Clean repository structure
- Version tagging (v1.0.0)

---

### 📦 Dependencies

Primary runtime components:

- Microsoft Foundry Local SDK
- PyTorch
- Transformers
- Streamlit
- RapidFuzz
- PyPDF
- python-docx

---

### 🚀 Future Roadmap

Potential improvements for future releases:

- OCR support
- Image understanding
- Cross-encoder reranking
- Multi-user support
- Additional document formats
- Retrieval performance optimization
- Expanded evaluation datasets
