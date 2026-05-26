"""End-to-end evaluation harness for the RAG pipeline.

Reads `eval/dataset.jsonl`, runs every question through the live pipeline,
then scores the predictions with a deterministic, embedding-based scorer.
No LLM-as-judge - results are reproducible across runs.

Usage:
    # First time -> capture a baseline:
    python eval/run_eval.py --output eval/baseline.json

    # After each improvement -> compare against baseline:
    python eval/run_eval.py --output eval/last_run.json

    # Just dump predictions without scoring (fast smoke test):
    python eval/run_eval.py --skip-scoring

Metrics (all 0-1, higher is better):
- answer_correctness  : composite per-row score.
                        In-scope rows: SQuAD-style token F1 vs ground_truth.
                        Out-of-scope rows: 1.0 iff answer is the canonical refusal.
- answer_f1           : token F1, averaged over in-scope rows only.
- refusal_accuracy    : fraction of out-of-scope rows answered with the refusal string.
- context_precision   : mean fraction of retrieved chunks whose embedding is
                        cosine-similar (>= threshold) to ground_truth.
- context_recall      : mean fraction of ground_truth "info units" (sentences)
                        each covered by at least one retrieved chunk.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import re
import sys
import time

faulthandler.enable()


def _hf_cache_has_models() -> bool:
    """True if the HF hub cache already contains downloaded models.

    We only flip offline mode when we know the models are local. On a fresh
    clone the cache is empty and the first run needs to actually download from
    the Hub - forcing offline would block that with a confusing error.
    """
    cache_root = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub_dir = os.path.join(cache_root, "hub")
    if not os.path.isdir(hub_dir):
        return False
    return any(name.startswith("models--") for name in os.listdir(hub_dir))


# After the first download, avoid HF Hub round-trips - unauthenticated requests
# get rate-limited and can stall model loads for minutes. Skip this on a fresh
# clone so the initial download actually happens. Override with HF_HUB_OFFLINE=0
# at any time to force a refresh.
if _hf_cache_has_models():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline import (  # noqa: E402
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    build_qa_chain,
    build_vectorstore,
    chunk_documents,
    get_embeddings,
    load_document,
)

REFUSAL_STRING = "I don't know based on the provided document."
CTX_SIM_THRESHOLD = 0.35  # cosine threshold for chunk/sentence relevance
TOKEN_OVERLAP_THRESHOLD = 0.6  # fallback: fraction of gold tokens present in chunk


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def build_chain_for_doc(doc_relpath: str, embeddings):
    doc_path = str(PROJECT_ROOT / doc_relpath)
    docs = load_document(doc_path)
    chunks = chunk_documents(docs)
    collection = f"eval_{abs(hash(doc_relpath))}"
    store = build_vectorstore(
        chunks,
        collection_name=collection,
        embeddings=embeddings,
    )
    return build_qa_chain(store, chunks=chunks), len(chunks)


def run_pipeline(rows: list[dict]) -> list[dict]:
    embeddings = get_embeddings()
    chains: dict[str, Any] = {}
    records: list[dict] = []

    for i, row in enumerate(rows, 1):
        doc = row["doc"]
        if doc not in chains:
            print(f"[{i}/{len(rows)}] Indexing {doc}...", flush=True)
            faulthandler.dump_traceback_later(30, repeat=True)
            chain, n_chunks = build_chain_for_doc(doc, embeddings)
            faulthandler.cancel_dump_traceback_later()
            chains[doc] = chain
            print(f"    -> {n_chunks} chunks indexed", flush=True)

        chain = chains[doc]
        question = row["question"]
        print(f"[{i}/{len(rows)}] Q: {question[:90]}", flush=True)
        t0 = time.perf_counter()
        out = chain.invoke({"input": question})
        latency = time.perf_counter() - t0
        answer = (out.get("answer") or "").strip()
        contexts = [d.page_content for d in out.get("context", [])]
        print(f"    A: {answer[:120]}", flush=True)
        print(f"    latency: {latency:.2f}s, retrieved {len(contexts)} chunks", flush=True)

        records.append({
            "id": row.get("id"),
            "type": row.get("type"),
            "format_type": row.get("format_type"),
            "doc": doc,
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row.get("ground_truth", ""),
            "latency_sec": round(latency, 3),
        })
    return records


# ---------- deterministic scorer ----------


def _normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_refusal(answer: str) -> bool:
    """True if the answer starts with the canonical refusal string.

    The system prompt asks for the exact refusal verbatim, but the LLM often
    tacks on a helpful sentence explaining what's missing. Both are valid
    refusals - we only require the canonical prefix.
    """
    norm_answer = _normalize(answer)
    norm_refusal = _normalize(REFUSAL_STRING)
    return norm_answer == norm_refusal or norm_answer.startswith(norm_refusal + " ")


def _token_set(text: str) -> set[str]:
    return set(_normalize(text).split())


def _chunk_is_relevant(gold: str, gold_vec: np.ndarray, chunk_text: str, chunk_vec: np.ndarray) -> bool:
    """A chunk counts as relevant if either:
    - cosine(chunk, gold) >= CTX_SIM_THRESHOLD, OR
    - >= TOKEN_OVERLAP_THRESHOLD of gold's content tokens appear in chunk.
    The second rule rescues short ground truths (addresses, codes) from being
    drowned out by long chunks that literally contain them.
    """
    if float(chunk_vec @ gold_vec) >= CTX_SIM_THRESHOLD:
        return True
    gold_tokens = _token_set(gold)
    if not gold_tokens:
        return False
    chunk_tokens = _token_set(chunk_text)
    overlap = len(gold_tokens & chunk_tokens) / len(gold_tokens)
    return overlap >= TOKEN_OVERLAP_THRESHOLD


def _split_sentences(text: str) -> list[str]:
    """Crude sentence splitter that won't break on $0.67 / 22.5 etc."""
    # Split on . ! ? followed by whitespace + capital letter, or end of string.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


# Format checks for the question-type-aware prompt. Each returns True if the
# answer shape matches what the FORMAT_RULES directive asked for. These are
# intentionally loose - we want to catch flagrant misses (no bullets in a LIST
# answer, no YES/NO prefix), not nitpick stylistic choices.
_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*(?:step\s+)?\d+[.)]\s+\S", re.I)
_BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+\S")
_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
    re.I,
)
_CAUSE_EFFECT_RE = re.compile(r"->|→|\bbecause\b|\bso that\b|\bresults? in\b|\bleads? to\b|\bcauses?\b", re.I)
_WHY_RE = re.compile(r"\bbecause\b|\bdue to\b|\bin order to\b|\bto ensure\b|\bso that\b|\bthe reason\b", re.I)
_CONTRAST_RE = re.compile(
    r"\bwhereas\b|\bwhile\b|\bin contrast\b|\bby contrast\b|\bdiffers?\b|"
    r"\b(?:vs|versus)\.?\b|\bcompared to\b|\bunlike\b|\bon the other hand\b",
    re.I,
)
_YES_NO_RE = re.compile(r"^\s*[*_`'\"]*\s*(yes|no|partially)\b[\s\-:—]", re.I)
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z.]+)+")


def check_format(answer: str, fmt: str) -> bool | None:
    """Return True/False if `answer` matches the expected shape for `fmt`, or None if no check."""
    if not answer:
        return False
    if fmt == "YES_NO":
        return bool(_YES_NO_RE.match(answer))
    if fmt == "LIST":
        return bool(_BULLET_RE.search(answer))
    if fmt == "SUMMARY":
        n = len(_split_sentences(answer))
        return 4 <= n <= 9
    if fmt == "WHEN":
        return bool(_DATE_RE.search(answer))
    if fmt == "HOW":
        return bool(_NUMBERED_STEP_RE.search(answer) or _CAUSE_EFFECT_RE.search(answer))
    if fmt == "WHY":
        return bool(_WHY_RE.search(answer))
    if fmt == "COMPARE":
        return bool(_CONTRAST_RE.search(answer))
    if fmt == "WHAT":
        n = len(_split_sentences(answer))
        return 1 <= n <= 6
    if fmt == "DEFINE":
        # Looks like a definition: under 4 sentences, not a refusal, has content.
        return len(_split_sentences(answer)) <= 4 and not is_refusal(answer)
    if fmt == "WHO":
        # At minimum names a person (multi-word capitalized noun).
        return bool(_PROPER_NAME_RE.search(answer))
    if fmt == "WHERE":
        # Non-empty, non-refusal. Locations are too varied to pattern-match cheaply.
        return bool(answer.strip()) and not is_refusal(answer)
    return None


def score_deterministic(records: list[dict], embeddings) -> dict:
    """Score records with deterministic + embedding-based metrics. No LLM judge."""
    # Collect all strings we need embeddings for, in one batch per category.
    gold_per_row = [r["ground_truth"] for r in records]
    gold_sentences_per_row = [_split_sentences(g) for g in gold_per_row]
    flat_gold_sentences = [s for row in gold_sentences_per_row for s in row]
    flat_contexts = [c for r in records for c in r["contexts"]]

    print("Embedding ground truths, sentences, and retrieved contexts...")
    gold_vecs = np.array(embeddings.embed_documents(gold_per_row))
    sent_vecs = (
        np.array(embeddings.embed_documents(flat_gold_sentences))
        if flat_gold_sentences else np.empty((0, gold_vecs.shape[1]))
    )
    ctx_vecs = (
        np.array(embeddings.embed_documents(flat_contexts))
        if flat_contexts else np.empty((0, gold_vecs.shape[1]))
    )

    # Index back into flat arrays.
    sent_offset = 0
    ctx_offset = 0
    per_row: list[dict] = []
    f1_scores: list[float] = []
    refusal_correct: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    correctness: list[float] = []
    format_scores: list[float] = []

    for i, r in enumerate(records):
        gold = r["ground_truth"]
        answer = r["answer"]
        is_oos = r.get("type") == "out_of_scope"
        n_sents = len(gold_sentences_per_row[i])
        n_ctxs = len(r["contexts"])
        ctx_block = ctx_vecs[ctx_offset:ctx_offset + n_ctxs]
        sent_block = sent_vecs[sent_offset:sent_offset + n_sents]

        # Answer correctness.
        if is_oos:
            refused = is_refusal(answer)
            refusal_correct.append(1.0 if refused else 0.0)
            row_correct = 1.0 if refused else 0.0
            row_f1 = float("nan")
        else:
            row_f1 = token_f1(answer, gold)
            f1_scores.append(row_f1)
            row_correct = row_f1
        correctness.append(row_correct)

        # Context P/R only meaningful when ground_truth has real content.
        # For out-of-scope rows, the "ground truth" is the refusal string, so
        # comparing chunks against it produces a misleading 0.
        if is_oos or n_ctxs == 0:
            row_precision = float("nan")
        else:
            relevant = [
                _chunk_is_relevant(gold, gold_vecs[i], r["contexts"][j], ctx_block[j])
                for j in range(n_ctxs)
            ]
            row_precision = float(sum(relevant) / n_ctxs)
            precisions.append(row_precision)

        # Recall: each ground-truth sentence is "covered" if any chunk is
        # relevant to it (cosine OR token-overlap fallback).
        if is_oos or n_sents == 0 or n_ctxs == 0:
            row_recall = float("nan")
        else:
            covered = 0
            sent_sims = sent_block @ ctx_block.T  # (n_sents, n_ctxs)
            for s_idx in range(n_sents):
                if (sent_sims[s_idx] >= CTX_SIM_THRESHOLD).any():
                    covered += 1
                    continue
                sent_tokens = _token_set(gold_sentences_per_row[i][s_idx])
                if not sent_tokens:
                    continue
                for c_idx in range(n_ctxs):
                    chunk_tokens = _token_set(r["contexts"][c_idx])
                    if len(sent_tokens & chunk_tokens) / len(sent_tokens) >= TOKEN_OVERLAP_THRESHOLD:
                        covered += 1
                        break
            row_recall = covered / n_sents
            recalls.append(row_recall)

        # Format-shape compliance for rows tagged with format_type.
        fmt = r.get("format_type")
        if fmt:
            fmt_pass = check_format(answer, fmt)
            row_format = None if fmt_pass is None else (1.0 if fmt_pass else 0.0)
            if row_format is not None:
                format_scores.append(row_format)
        else:
            row_format = None

        per_row.append({
            "id": r["id"],
            "type": r["type"],
            "format_type": fmt,
            "answer_correctness": row_correct,
            "answer_f1": row_f1,
            "refusal_correct": (None if not is_oos else bool(is_refusal(answer))),
            "format_compliance": row_format,
            "context_precision": row_precision,
            "context_recall": row_recall,
        })

        sent_offset += n_sents
        ctx_offset += n_ctxs

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    aggregates = {
        "answer_correctness": _mean(correctness),
        "answer_f1": _mean(f1_scores),
        "refusal_accuracy": _mean(refusal_correct),
        "format_compliance": _mean(format_scores),
        "context_precision": _mean(precisions),
        "context_recall": _mean(recalls),
    }
    return {"aggregates": aggregates, "per_row": per_row}


def print_aggregates(aggregates: dict) -> None:
    print("\n=== Aggregate scores (higher is better, range 0-1) ===")
    for k, v in aggregates.items():
        bar = "#" * int(v * 20) if v == v else ""  # NaN-safe
        print(f"  {k:22s} {v:.3f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset.jsonl"),
        help="Path to eval dataset (JSONL).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "last_run.json"),
        help="Where to write predictions + scores.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama base URL (used for the QA chain, not for scoring).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model used by the QA chain.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Run the pipeline and dump predictions; skip scoring.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 2 questions for a fast sanity check (<30s).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_dataset(dataset_path)
    if args.smoke:
        rows = rows[:2]
    print(f"Loaded {len(rows)} eval rows from {dataset_path}")

    records = run_pipeline(rows)
    payload: dict = {
        "dataset": str(dataset_path),
        "n_rows": len(records),
        "scorer": "deterministic_v1",
        "records": records,
    }

    if not args.skip_scoring:
        print("\nScoring (deterministic, embedding-based)...")
        scoring = score_deterministic(records, get_embeddings())
        payload["aggregates"] = scoring["aggregates"]
        payload["per_row_scores"] = scoring["per_row"]
        print_aggregates(scoring["aggregates"])

    Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
