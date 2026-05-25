"""RAG pipeline: load -> chunk -> embed -> store -> retrieve -> answer.

Stack: LangChain + ChromaDB + sentence-transformers + Ollama. Fully local.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
SEMANTIC_CHUNKING = os.getenv("SEMANTIC_CHUNKING", "0") == "1"
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"

_RERANKER_CACHE: HuggingFaceCrossEncoder | None = None


class RagPipelineError(Exception):
    """User-facing pipeline error with a friendly, actionable message."""


# Elements that are page-level boilerplate and pollute retrieval if indexed.
_IGNORED_ELEMENT_CATEGORIES = frozenset({"Header", "Footer", "PageNumber"})
# Element types that should stay atomic through chunking (do not split mid-element).
_ATOMIC_ELEMENT_CATEGORIES = frozenset({"Table", "TableRow", "ListItem"})


def _looks_like_real_table(rows: list[list]) -> bool:
    """Filter out PyMuPDF's false-positive table detections in narrative PDFs.

    A real table has a short header row, multiple data rows, and mostly-filled
    cells. False positives (two-column text reflow, sidebars) usually have a
    long-sentence header or sparse cells.
    """
    if len(rows) < 3 or not rows[0] or len(rows[0]) < 2:
        return False
    header = [c for c in rows[0] if c]
    if not header:
        return False
    avg_header_len = sum(len(str(c)) for c in header) / len(header)
    if avg_header_len > 30:
        return False
    data_cells = [c for row in rows[1:] for c in row]
    filled = sum(1 for c in data_cells if c not in (None, ""))
    if not data_cells or filled / len(data_cells) < 0.5:
        return False
    return True


def _table_to_markdown(rows: list[list]) -> str:
    """Render rows as a github-flavored markdown table."""
    def cell(c) -> str:
        return str(c).replace("|", "\\|").replace("\n", " ").strip() if c else ""

    width = max(len(r) for r in rows)
    norm = [[cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    header = norm[0]
    sep = ["---"] * width
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for r in norm[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _row_to_fact(header: list[str], row: list) -> str | None:
    """Render one table row as a natural-language fact for keyword retrieval."""
    pairs = []
    for h, v in zip(header, row):
        h_str, v_str = (h or "").strip(), ("" if v is None else str(v)).strip()
        if h_str and v_str:
            pairs.append(f"{h_str}: {v_str}")
    return "; ".join(pairs) if pairs else None


def _extract_pdf_tables(file_path: str) -> list[Document]:
    """Detect real tables in a PDF via PyMuPDF and emit:
       - one Document holding the table as markdown (for context-heavy queries)
       - one Document per row in natural-language form (for keyword lookups)
    """
    import pymupdf

    out: list[Document] = []
    try:
        doc = pymupdf.open(file_path)
    except Exception:
        return out

    for page_idx, page in enumerate(doc):
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        for t_idx, t in enumerate(tables):
            try:
                rows = t.extract()
            except Exception:
                continue
            if not _looks_like_real_table(rows):
                continue
            header = [str(c or "").strip() for c in rows[0]]
            md = _table_to_markdown(rows)
            out.append(Document(
                page_content=md,
                metadata={"page": page_idx + 1, "category": "Table", "table_idx": t_idx},
            ))
            for r_idx, row in enumerate(rows[1:], start=1):
                fact = _row_to_fact(header, row)
                if fact:
                    out.append(Document(
                        page_content=fact,
                        metadata={
                            "page": page_idx + 1,
                            "category": "TableRow",
                            "table_idx": t_idx,
                            "row_idx": r_idx,
                        },
                    ))
    return out


def _ensure_nltk_data() -> None:
    """Download unstructured's NLTK dependencies once and cache them.

    Workaround for macOS Python builds shipping without system CA roots; if the
    initial HTTPS request fails, fall back to unverified context for the download.
    """
    import nltk

    needed = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    missing = []
    for resource, pkg in needed:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(pkg)
    if not missing:
        return

    import ssl

    orig_ctx = getattr(ssl, "_create_default_https_context", None)
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        for pkg in missing:
            nltk.download(pkg, quiet=True)
    finally:
        if orig_ctx is not None:
            ssl._create_default_https_context = orig_ctx

QA_SYSTEM_PROMPT = (
    "You answer questions about a document the user uploaded. "
    "Use ONLY the information in the context below.\n\n"
    "Rules:\n"
    "1. If the answer is not in the context, reply exactly: "
    "\"I don't know based on the provided document.\" Nothing else.\n"
    "2. Be terse. Aim for one sentence. Never preface with phrases like "
    "\"According to the document\", \"The document states\", \"Based on the context\", "
    "\"Section X says\", or \"It is mentioned that\". Just answer.\n"
    "3. Quote numbers, dates, dollar amounts, codes, and proper nouns exactly as they "
    "appear in the context. Do not round, paraphrase, or summarize them.\n"
    "4. If the question asks to compare or list, give only the items - no commentary.\n\n"
    "Context:\n{context}"
)


def load_document(file_path: str) -> list[Document]:
    """Layout-aware extraction for PDF/TXT, returning chunk-sized Documents.

    Uses unstructured.io to partition the file into typed elements (Title,
    NarrativeText, Table, ListItem, Header, Footer, ...). Headers, footers, and
    page numbers are dropped. Remaining elements are grouped into title-bounded
    chunks via chunk_by_title; tables and list items stay atomic.

    Each returned Document carries metadata {category, page} for downstream
    citation and adaptive handling.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in {".pdf", ".txt"}:
        raise RagPipelineError(
            f"Unsupported file type: {ext or '(none)'}. Upload a .pdf or .txt file."
        )
    if not path.exists():
        raise RagPipelineError(f"File not found: {path.name}")

    _ensure_nltk_data()

    try:
        if ext == ".pdf":
            from unstructured.partition.pdf import partition_pdf

            elements = partition_pdf(
                filename=str(path),
                strategy="fast",
                infer_table_structure=False,
            )
        else:
            from unstructured.partition.text import partition_text

            elements = partition_text(filename=str(path))
    except UnicodeDecodeError as exc:
        raise RagPipelineError(
            "Could not decode the text file as UTF-8. Re-save it as UTF-8 and try again."
        ) from exc
    except Exception as exc:
        raise RagPipelineError(
            f"Failed to read {path.name}. The file may be corrupted, "
            f"encrypted, or scanned-only (no extractable text). Details: {exc}"
        ) from exc

    # Drop page-level boilerplate before chunking so it doesn't pollute embeddings.
    elements = [el for el in elements if el.category not in _IGNORED_ELEMENT_CATEGORIES]
    if not elements:
        raise RagPipelineError(
            f"No extractable text in {path.name}. If it's a scanned PDF, OCR it first."
        )

    from unstructured.chunking.title import chunk_by_title

    chunked = chunk_by_title(
        elements,
        max_characters=1000,
        combine_text_under_n_chars=200,
        new_after_n_chars=900,
    )

    docs: list[Document] = []
    for ch in chunked:
        text = (ch.text or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "page": getattr(ch.metadata, "page_number", None),
                "category": ch.category,
            },
        ))

    if not docs:
        raise RagPipelineError(
            f"No extractable text in {path.name}. If it's a scanned PDF, OCR it first."
        )

    if ext == ".pdf":
        # Structured-table pass: PyMuPDF reliably extracts row/column structure
        # that unstructured's "fast" strategy flattens. Tables are added as a
        # whole-table markdown chunk plus per-row fact chunks for keyword hits.
        docs.extend(_extract_pdf_tables(str(path)))

    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    semantic: bool | None = None,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> list[Document]:
    """Element-aware chunking.

    load_document already produces title-bounded chunks via unstructured, so
    this is mostly a pass-through. Documents whose category is Table or
    ListItem are kept atomic. Any document exceeding chunk_size is re-split
    with RecursiveCharacterTextSplitter so a stray giant element doesn't blow
    past the model's context window.

    With `semantic=True` (or env `SEMANTIC_CHUNKING=1`), falls back to
    embedding-aware splitting over the joined text - useful for pure prose
    docs, but discards element boundaries.
    """
    use_semantic = semantic if semantic is not None else SEMANTIC_CHUNKING
    if use_semantic:
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(
            embeddings or get_embeddings(),
            breakpoint_threshold_type="percentile",
        )
        return splitter.split_documents(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out: list[Document] = []
    for doc in docs:
        category = doc.metadata.get("category")
        if category in _ATOMIC_ELEMENT_CATEGORIES or len(doc.page_content) <= chunk_size:
            out.append(doc)
        else:
            out.extend(splitter.split_documents([doc]))
    return out


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: list[Document],
    collection_name: str,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Embed `chunks` and store them in a Chroma collection.

    Re-uploading the same file would duplicate vectors, so we reset the
    named collection in a single client to avoid a known Chroma-rust
    deadlock that happens when two PersistentClients share a directory.
    """
    embeddings = embeddings or get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    try:
        existing_ids = store.get().get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)
    except Exception:
        pass

    store.add_documents(chunks)
    return store


def check_ollama_ready(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    timeout: float = 3.0,
) -> None:
    """Verify Ollama is reachable and `model_name` is pulled. Raises RagPipelineError otherwise."""
    tags_url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RagPipelineError(
            f"Could not reach Ollama at {base_url}. "
            "Start it with `ollama serve` (or check OLLAMA_BASE_URL)."
        ) from exc
    except Exception as exc:
        raise RagPipelineError(
            f"Unexpected response from Ollama at {base_url}: {exc}"
        ) from exc

    installed = {m.get("name", "").split(":")[0] for m in payload.get("models", [])}
    installed |= {m.get("name", "") for m in payload.get("models", [])}
    if model_name not in installed and model_name.split(":")[0] not in installed:
        raise RagPipelineError(
            f"Ollama model '{model_name}' is not pulled. "
            f"Run `ollama pull {model_name}` and try again."
        )


def build_hybrid_retriever(
    vectorstore: Chroma,
    chunks: list[Document],
    k: int = 4,
    bm25_weight: float = 0.5,
) -> EnsembleRetriever:
    """Combine BM25 (lexical) with the Chroma dense retriever via RRF fusion."""
    dense = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    dense_weight = max(0.0, 1.0 - bm25_weight)
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight],
    )


def get_reranker(model_name: str = RERANKER_MODEL) -> HuggingFaceCrossEncoder:
    """Load (and cache) the cross-encoder used for reranking."""
    global _RERANKER_CACHE
    if _RERANKER_CACHE is None:
        _RERANKER_CACHE = HuggingFaceCrossEncoder(model_name=model_name)
    return _RERANKER_CACHE


def wrap_with_reranker(base_retriever, top_n: int = 4):
    """Wrap a retriever so its candidates are reordered by a cross-encoder."""
    compressor = CrossEncoderReranker(model=get_reranker(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


def build_qa_chain(
    vectorstore: Chroma,
    chunks: list[Document] | None = None,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    k: int = 3,
    bm25_weight: float = 0.5,
    use_reranker: bool | None = None,
    rerank_pool: int = 20,
):
    """Build a retrieval-augmented QA chain backed by Ollama.

    Retrieval stack:
    - If `chunks` is provided, hybrid (BM25 + dense, RRF-fused).
    - If `use_reranker` (default `USE_RERANKER` env), a cross-encoder reorders
      a larger candidate pool down to top-k before the LLM sees it.
    """
    check_ollama_ready(base_url=base_url, model_name=model_name)
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)

    rerank = USE_RERANKER if use_reranker is None else use_reranker
    pool = max(k, rerank_pool) if rerank else k

    if chunks:
        retriever = build_hybrid_retriever(
            vectorstore, chunks, k=pool, bm25_weight=bm25_weight
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": pool})

    if rerank:
        retriever = wrap_with_reranker(retriever, top_n=k)

    prompt = ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), ("human", "{input}")]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


def answer_question(chain, question: str) -> dict:
    """Invoke the chain and return {'answer': str, 'context': list[Document]}."""
    if not question or not question.strip():
        raise RagPipelineError("Question is empty.")
    try:
        return chain.invoke({"input": question})
    except urllib.error.URLError as exc:
        raise RagPipelineError(
            "Lost connection to Ollama while answering. "
            "Is `ollama serve` still running?"
        ) from exc
    except Exception as exc:
        # ChatOllama raises various httpx/ConnectionError types; surface a clean message.
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "timeout" in msg:
            raise RagPipelineError(
                "Lost connection to Ollama while answering. "
                "Is `ollama serve` still running?"
            ) from exc
        raise RagPipelineError(f"Answering failed: {exc}") from exc
