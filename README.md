# AI Document Assistant (RAG)

A local, privacy-friendly Q&A app for PDF and TXT documents — narrative prose, tutorials with headings and lists, and PDFs with real row/column tables. Built with Streamlit, LangChain, ChromaDB, sentence-transformers, unstructured.io, and Ollama — **no API keys, no cloud calls, no cost**.

## Architecture

```
+-----------+    +------------------+    +-----------------+    +-----------+    +----------+
| Upload    | -> | Layout-aware     | -> | Element-aware   | -> | BGE-small | -> |  Chroma  |
| (PDF/TXT) |    | extraction       |    | chunking        |    | embed     |    |  vector  |
|           |    | (unstructured +  |    | (chunks +       |    | (384-dim) |    |  store   |
|           |    |  PyMuPDF tables) |    |  table rows)    |    |           |    |          |
+-----------+    +------------------+    +-----------------+    +-----------+    +----------+
                                                                                       |
                                                                                       v
   +---------+   +---------------+   +-------------+   +-----------------+   +-------------------+
   | Answer  | <-| Ollama LLM    | <-| Grounded    | <-| Cross-encoder   | <-| Hybrid retrieval  |
   | (UI)    |   |(llama3.2:3b   |   | prompt with |   | reranker        |   | BM25 + dense      |
   |         |   | default,      |   | strict      |   | (bge-reranker-  |   | (RRF, pool of 20) |
   |         |   | swappable)    |   | citations   |   |  base, top 3)   |   |                   |
   +---------+   +---------------+   +-------------+   +-----------------+   +-------------------+
```

1. **Upload**: PDF or TXT via the Streamlit file uploader.
2. **Extract**: `unstructured.io` partitions the file into typed elements (`Title`, `NarrativeText`, `ListItem`, `Header`, `Footer`, ...). Headers, footers, and page numbers are dropped before indexing. For PDFs, a secondary PyMuPDF pass detects real row/column tables and emits both a markdown-table chunk and one "fact" chunk per row.
3. **Chunk**: elements are grouped into title-bounded chunks via `chunk_by_title` (atomic for tables, list items, and rows; recursive backstop for oversized elements).
4. **Embed**: each chunk is embedded with `BAAI/bge-small-en-v1.5` (384-dim, normalized).
5. **Index**: chunks are stored in a per-document Chroma collection persisted to `./chroma_db` (regenerable cache — gitignored).
6. **Retrieve**: hybrid BM25 (lexical) + Chroma dense (semantic), fused via RRF. A pool of 20 candidates is reranked by `BAAI/bge-reranker-base` cross-encoder down to top-3.
7. **Answer**: top-3 chunks go into a strict grounded prompt for a local Ollama model. The prompt forbids preamble, requires exact quoting of numbers/codes, and forces the canonical refusal string `"I don't know based on the provided document."` whenever the answer isn't in the context.

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running locally:
  - macOS: `brew install ollama` (or download from https://ollama.com)
  - Pull the default model: `ollama pull llama3.2:3b`
- Roughly **5 GB free disk** total:
  - ~2 GB for the Ollama 3B model
  - ~150 MB for the BGE-small embedding model
  - ~100 MB for the cross-encoder reranker
  - Pulled lazily on first run.
- **Internet on the first run** to fetch the embedding and reranker models from Hugging Face and the NLTK tokenizer/POS data used by `unstructured`. Subsequent runs work fully offline (the eval harness automatically flips `HF_HUB_OFFLINE=1` once the cache is populated).
- For 8 GB RAM machines, `llama3.2:3b` is the practical ceiling. For 16 GB+, you can swap to `llama3.1:8b` or `qwen2.5:7b` for a noticeable quality lift (see [Configuration](#configuration)).

## Setup

```bash
git clone <this-repo-url>
cd ai-document-assistant

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

In one terminal, start Ollama (skip if it's already running):

```bash
ollama serve
```

In another terminal, launch the app:

```bash
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser.

## Usage

1. **Upload** a PDF or TXT in the main pane.
2. Wait a few seconds while it's chunked, embedded, and indexed (you'll see spinners).
3. **Ask** any question in the chat box at the bottom.
4. Expand **Sources** under each answer to see which chunks were retrieved.
5. Use **Clear index & chat** in the sidebar to start fresh.

## Configuration

All settings are read from environment variables and have sensible defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model used for answer generation. Any pulled model name works (`llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, etc.). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server is reachable. |
| `CHROMA_DIR` | `chroma_db` | Local directory for the persisted Chroma index. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder used to rerank retrieval candidates. |
| `USE_RERANKER` | `1` | Set to `0` to disable reranking and return top-k from hybrid retrieval directly. |
| `SEMANTIC_CHUNKING` | `0` | Set to `1` to replace recursive splitting with embedding-aware semantic chunking. Helps prose; can regress on structured docs. |
| `HF_HUB_OFFLINE` | unset (eval auto-sets `1` once cache exists) | Force offline mode for Hugging Face downloads. Useful on flaky networks or to guarantee reproducibility. |

You can override the Ollama model and base URL live from the sidebar.

## Project Layout

```
.
├── app.py                       # Streamlit UI
├── rag_pipeline.py              # Load -> chunk -> embed -> retrieve -> rerank -> answer
├── requirements.txt
├── README.md
├── .gitignore
└── eval/
    ├── run_eval.py              # End-to-end eval harness (deterministic scorer)
    ├── dataset.jsonl            # 35 questions across TXT and PDF fixtures
    ├── baseline.json            # Captured reference scores (regenerate locally)
    └── sample_docs/
        ├── nova_handbook.txt    # Narrative TXT (HR handbook)
        ├── sentry_x3_manual.txt # Structured TXT (gas detector manual)
        ├── football_tutorial.pdf# Narrative PDF (headings + lists)
        └── sku_price_list.pdf   # Tabular PDF (row/column SKU table)
```

## How the Mandatory Requirements Are Met

| Requirement | Where |
| --- | --- |
| Document upload | `app.py` — `st.file_uploader` |
| Text extraction & chunking | `rag_pipeline.load_document` (unstructured.io + PyMuPDF table pass) + `chunk_documents` |
| Embedding generation | `rag_pipeline.get_embeddings` (`BAAI/bge-small-en-v1.5`) |
| Vector DB integration | `rag_pipeline.build_vectorstore` (Chroma, single-client to avoid double-init deadlock) |
| Retrieval | `rag_pipeline.build_hybrid_retriever` (BM25 + dense, RRF) + `wrap_with_reranker` (cross-encoder) |
| Question answering with LLM | `rag_pipeline.build_qa_chain` (Ollama + LCEL retrieval chain) |
| Simple UI | Streamlit chat interface in `app.py` |

## Evaluation Harness

The `eval/` directory contains a deterministic evaluation harness so you can measure retrieval and answer quality (and prove that future changes actually help). No LLM-as-judge — scores are reproducible across runs and the whole scoring step takes seconds.

### What's in it

- `eval/sample_docs/` — four fixtures covering the document shapes the pipeline targets: narrative TXT (`nova_handbook.txt`), structured TXT (`sentry_x3_manual.txt`), narrative PDF with headings and lists (`football_tutorial.pdf`), and a tabular PDF with a real row/column SKU table (`sku_price_list.pdf`).
- `eval/dataset.jsonl` — 35 questions over the fixtures: factual lookups, multi-chunk synthesis, numeric specifics, table cell lookups, and out-of-scope (expecting `"I don't know based on the provided document."`).
- `eval/run_eval.py` — runs every question through the live pipeline and scores predictions with token F1, embedding-based context P/R (with a token-overlap fallback for short ground truths), and refusal exact-match.

### Capture a baseline

Before changing any pipeline code, run:

```bash
python eval/run_eval.py --output eval/baseline.json
```

The script reports five metrics, each on a 0–1 scale (higher is better):

| Metric | What it measures |
| --- | --- |
| `answer_correctness` | Per-row composite: token F1 vs `ground_truth` for in-scope rows, refusal exact-match for out-of-scope rows. The single number to track. |
| `answer_f1` | SQuAD-style token-overlap F1 vs `ground_truth`, averaged over in-scope rows only. |
| `refusal_accuracy` | Fraction of out-of-scope rows that emitted the canonical `"I don't know based on the provided document."` refusal. |
| `context_precision` | Mean fraction of retrieved chunks that are relevant to `ground_truth` — either cosine ≥ 0.35 in BGE space, or ≥ 60% of `ground_truth` tokens appear in the chunk (rescues short ground truths from being drowned out by long chunks). Skipped for out-of-scope rows. |
| `context_recall` | Mean fraction of `ground_truth` sentences covered by at least one retrieved chunk under the same rule. Skipped for out-of-scope rows. |

### Re-run after each improvement

```bash
python eval/run_eval.py --output eval/last_run.json
```

Then diff the `aggregates` block in `last_run.json` against `baseline.json`. A change that regresses any metric by more than 5% should be investigated before continuing.

### Tips

- Use `--smoke` to run only the first 2 questions (<30s) for a quick sanity check after a change.
- Use `--skip-scoring` to dump predictions without scoring them.
- Scoring is deterministic (token math + cosine on pre-computed embeddings) so re-running the same predictions always yields the same numbers — making real regressions easy to spot.
- After models are downloaded once, `eval/run_eval.py` automatically sets `HF_HUB_OFFLINE=1` so subsequent runs never round-trip to Hugging Face. On a fresh clone the cache is empty, the check is skipped, and the first run downloads normally. To force a fresh model fetch any time, run `HF_HUB_OFFLINE=0 python eval/run_eval.py ...`.

## What This Pipeline Handles Well

- **Narrative TXT and PDF** — handbooks, manuals, tutorials, articles. Page headers, footers, and page numbers are stripped automatically so they don't pollute embeddings.
- **PDFs with real row/column tables** — PyMuPDF detects genuine tables, emits a markdown-formatted table chunk plus one fact chunk per row for keyword-level cell lookups. A heuristic filter skips false-positive table detections in narrative-heavy PDFs.
- **Hybrid lexical + semantic retrieval** — BM25 catches exact codes/SKUs/numbers; BGE-small dense catches paraphrased and conceptual queries. RRF fusion + cross-encoder reranking surfaces the most relevant chunks.
- **Strict grounded answers** — the prompt forbids preamble, requires exact quoting, and forces a canonical refusal string when the answer isn't in the document.
- **Clean failure modes** — missing files, corrupted PDFs, encrypted PDFs, scanned-only PDFs (no extractable text), Ollama unreachable, and Ollama model not pulled all surface as user-friendly errors, not tracebacks.

## Limitations & Possible Extensions

- **One document at a time.** Multi-doc support would mean joining collections at query time.
- **No OCR for scanned PDFs.** A scanned-only file is rejected with a clean error rather than producing garbage. Adding `pytesseract` as an opt-in fallback is a possible extension.
- **Equation-heavy PDFs.** Math symbols often come out garbled — no good local extractor for this. Documented as out-of-scope.
- **Diagram-internal text and PDF form fields** are not extracted.
- **No streaming token output** — answers appear all at once.
- **Source citations** show chunk text and page number but don't deep-link into the PDF.
- **Answer quality is bounded by the local LLM** — `llama3.2:3b` is fine for extractive QA but limits multi-step reasoning. On 16 GB+ machines, swap `OLLAMA_MODEL` to `llama3.1:8b` or `qwen2.5:7b` for a substantial lift — the rest of the pipeline is model-agnostic and scales with the LLM.
