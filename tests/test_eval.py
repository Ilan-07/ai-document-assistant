"""Tests for eval/run_eval.py — dataset loader, deterministic scorers, pipeline."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from eval import run_eval as ev


# ---------------------------------------------------------------------------
# _hf_cache_has_models
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hf_cache_has_models_no_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nope"))
    assert ev._hf_cache_has_models() is False


@pytest.mark.unit
def test_hf_cache_has_models_empty_hub(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert ev._hf_cache_has_models() is False


@pytest.mark.unit
def test_hf_cache_has_models_with_model(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "models--something--here").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert ev._hf_cache_has_models() is True


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_dataset_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '# this is a comment\n'
        '\n'
        '   \n'
        '{"id": "a", "question": "q1"}\n'
        '# another comment\n'
        '{"id": "b", "question": "q2"}\n',
        encoding="utf-8",
    )
    rows = ev.load_dataset(p)
    assert len(rows) == 2
    assert [r["id"] for r in rows] == ["a", "b"]


@pytest.mark.unit
def test_load_dataset_invalid_json_raises(tmp_path):
    p = tmp_path / "ds.jsonl"
    p.write_text('{"id": "ok"}\n{ this is not json }\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        ev.load_dataset(p)


# ---------------------------------------------------------------------------
# _normalize and token_f1
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_normalize_lowercases_strips_articles_punctuation():
    assert ev._normalize("The Quick, Brown Fox!") == "quick brown fox"
    assert ev._normalize("An apple a day") == "apple day"


@pytest.mark.unit
def test_token_f1_exact_match():
    assert ev.token_f1("hello world", "hello world") == 1.0


@pytest.mark.unit
def test_token_f1_no_overlap():
    assert ev.token_f1("foo bar", "baz qux") == 0.0


@pytest.mark.unit
def test_token_f1_partial_overlap():
    # pred=['quick','fox'], gold=['quick','brown','fox'] -> P=1.0, R=2/3 -> F1=0.8
    assert abs(ev.token_f1("quick fox", "quick brown fox") - 0.8) < 1e-6


@pytest.mark.unit
def test_token_f1_normalization_kills_articles():
    # After normalization both reduce to "policy" -> F1=1.0
    assert ev.token_f1("The policy", "a policy") == 1.0


@pytest.mark.unit
def test_token_f1_empty_strings():
    assert ev.token_f1("", "") == 1.0
    assert ev.token_f1("hello", "") == 0.0


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_is_refusal_canonical():
    assert ev.is_refusal("I don't know based on the provided document.") is True


@pytest.mark.unit
def test_is_refusal_with_trailing_explanation():
    assert ev.is_refusal(
        "I don't know based on the provided document. The handbook does not cover this."
    ) is True


@pytest.mark.unit
def test_is_refusal_false_for_real_answer():
    assert ev.is_refusal("The orientation is held on Monday.") is False


@pytest.mark.unit
def test_is_refusal_empty():
    assert ev.is_refusal("") is False


# ---------------------------------------------------------------------------
# check_format
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("answer, expected", [
    ("YES - the lab is open daily.", True),
    ("NO - access requires a badge.", True),
    ("PARTIALLY - only during business hours.", True),
    ("It depends on the day.", False),
    ("Maybe yes maybe no.", False),
])
def test_check_format_yes_no(answer, expected):
    assert ev.check_format(answer, "YES_NO") is expected


@pytest.mark.unit
def test_check_format_list_pass():
    assert ev.check_format("- item one\n- item two\n- item three", "LIST") is True


@pytest.mark.unit
def test_check_format_list_fail():
    assert ev.check_format("Just some prose about things.", "LIST") is False


@pytest.mark.unit
def test_check_format_summary_pass():
    answer = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    assert ev.check_format(answer, "SUMMARY") is True


@pytest.mark.unit
def test_check_format_summary_too_short():
    assert ev.check_format("One sentence only.", "SUMMARY") is False


@pytest.mark.unit
def test_check_format_summary_too_long():
    long = " ".join(f"Sentence {i}." for i in range(20))
    assert ev.check_format(long, "SUMMARY") is False


@pytest.mark.unit
def test_check_format_when_pass():
    assert ev.check_format("The policy was updated in 2023.", "WHEN") is True
    assert ev.check_format("It happened in March.", "WHEN") is True


@pytest.mark.unit
def test_check_format_when_fail():
    assert ev.check_format("It happened sometime ago.", "WHEN") is False


@pytest.mark.unit
def test_check_format_how_steps():
    assert ev.check_format("1. Open the app\n2. Click submit\n3. Wait", "HOW") is True


@pytest.mark.unit
def test_check_format_how_cause_effect():
    assert ev.check_format("Pressure rises because heat builds up.", "HOW") is True


@pytest.mark.unit
def test_check_format_why_pass():
    assert ev.check_format("Because the regulation requires it.", "WHY") is True


@pytest.mark.unit
def test_check_format_why_fail():
    assert ev.check_format("The reason is unclear from the text.", "WHY") is True
    assert ev.check_format("It just is.", "WHY") is False


@pytest.mark.unit
def test_check_format_compare_pass():
    assert ev.check_format("A is fast whereas B is slow.", "COMPARE") is True
    assert ev.check_format("X versus Y on speed.", "COMPARE") is True


@pytest.mark.unit
def test_check_format_compare_fail():
    assert ev.check_format("They are both options.", "COMPARE") is False


@pytest.mark.unit
def test_check_format_what_pass_and_fail():
    assert ev.check_format("It's a widget.", "WHAT") is True
    # 7+ sentences is too many for WHAT
    long = " ".join(f"S{i}." for i in range(8))
    assert ev.check_format(long, "WHAT") is False


@pytest.mark.unit
def test_check_format_define_pass():
    assert ev.check_format("Latency is the time delay between request and response.", "DEFINE") is True


@pytest.mark.unit
def test_check_format_define_refusal_fails():
    assert ev.check_format("I don't know based on the provided document.", "DEFINE") is False


@pytest.mark.unit
def test_check_format_who_pass_fail():
    assert ev.check_format("Maria Lopez heads HR.", "WHO") is True
    assert ev.check_format("unknown person.", "WHO") is False


@pytest.mark.unit
def test_check_format_where_pass_fail():
    assert ev.check_format("In the lobby on floor 2.", "WHERE") is True
    assert ev.check_format("I don't know based on the provided document.", "WHERE") is False
    assert ev.check_format("   ", "WHERE") is False


@pytest.mark.unit
def test_check_format_unknown_returns_none():
    assert ev.check_format("anything", "UNKNOWN_FORMAT") is None


@pytest.mark.unit
def test_check_format_empty_answer():
    assert ev.check_format("", "LIST") is False


# ---------------------------------------------------------------------------
# _chunk_is_relevant
# ---------------------------------------------------------------------------

def _unit(*vals):
    v = np.array(vals, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


@pytest.mark.unit
def test_chunk_is_relevant_high_cosine():
    v = _unit(1, 0, 0)
    assert ev._chunk_is_relevant("gold", v, "chunk text", v) is True


@pytest.mark.unit
def test_chunk_is_relevant_token_overlap_fallback():
    """Low cosine but high token overlap should pass."""
    gold = "address 123 main street"
    chunk = "delivery address: 123 main street, suite 4"
    orthog1 = _unit(1, 0, 0)
    orthog2 = _unit(0, 1, 0)
    assert ev._chunk_is_relevant(gold, orthog1, chunk, orthog2) is True


@pytest.mark.unit
def test_chunk_is_relevant_negative():
    gold = "totally unrelated words here"
    chunk = "completely different topic entirely"
    assert ev._chunk_is_relevant(gold, _unit(1, 0, 0), chunk, _unit(0, 1, 0)) is False


@pytest.mark.unit
def test_chunk_is_relevant_empty_gold_tokens():
    """Pure punctuation gold + no cosine match -> False."""
    assert ev._chunk_is_relevant("...", _unit(1, 0, 0), "chunk", _unit(0, 1, 0)) is False


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_split_sentences_basic():
    s = ev._split_sentences("First sentence. Second sentence! Third one?")
    # All start with uppercase after the boundary so they should split.
    assert len(s) >= 2


@pytest.mark.unit
def test_split_sentences_keeps_decimal_numbers():
    """$0.67 / 22.5 shouldn't be split."""
    s = ev._split_sentences("The price is $0.67 per unit. Volume is 22.5 liters.")
    assert len(s) == 2


@pytest.mark.unit
def test_split_sentences_empty():
    assert ev._split_sentences("") == []


# ---------------------------------------------------------------------------
# build_chain_for_doc
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_build_chain_for_doc(monkeypatch):
    monkeypatch.setattr(ev, "load_document", MagicMock(return_value=[Document(page_content="x")]))
    monkeypatch.setattr(ev, "chunk_documents", MagicMock(return_value=[
        Document(page_content="c1"), Document(page_content="c2"),
    ]))
    monkeypatch.setattr(ev, "build_vectorstore", MagicMock(return_value="STORE"))
    monkeypatch.setattr(ev, "build_qa_chain", MagicMock(return_value="CHAIN"))

    chain, n = ev.build_chain_for_doc("eval/sample_docs/x.txt", MagicMock(name="emb"))
    assert chain == "CHAIN"
    assert n == 2


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

@pytest.mark.functional
def test_run_pipeline_collects_predictions(monkeypatch):
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "answer": "An answer.",
        "context": [Document(page_content="ctx", metadata={"page": 1})],
    }
    monkeypatch.setattr(ev, "get_embeddings", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ev, "build_chain_for_doc", MagicMock(return_value=(fake_chain, 3)))

    rows = [
        {"id": "q1", "type": "in_scope", "doc": "x.txt", "question": "Q1?", "ground_truth": "A1"},
        {"id": "q2", "type": "in_scope", "doc": "x.txt", "question": "Q2?", "ground_truth": "A2"},
    ]
    out = ev.run_pipeline(rows)
    assert len(out) == 2
    assert out[0]["answer"] == "An answer."
    assert out[0]["contexts"] == ["ctx"]
    assert out[0]["id"] == "q1"
    assert "latency_sec" in out[0]


@pytest.mark.functional
def test_run_pipeline_reuses_chain_per_doc(monkeypatch):
    """build_chain_for_doc should be called once per unique doc."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {"answer": "x", "context": []}

    build_mock = MagicMock(return_value=(fake_chain, 1))
    monkeypatch.setattr(ev, "get_embeddings", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ev, "build_chain_for_doc", build_mock)

    rows = [
        {"id": "a", "type": "in_scope", "doc": "d1.txt", "question": "Q?", "ground_truth": ""},
        {"id": "b", "type": "in_scope", "doc": "d1.txt", "question": "Q?", "ground_truth": ""},
        {"id": "c", "type": "in_scope", "doc": "d2.txt", "question": "Q?", "ground_truth": ""},
    ]
    ev.run_pipeline(rows)
    # 2 unique docs -> 2 build calls
    assert build_mock.call_count == 2


# ---------------------------------------------------------------------------
# score_deterministic
# ---------------------------------------------------------------------------

class _ScoringEmb:
    """Deterministic 8-d unit embeddings keyed by text identity."""

    def embed_documents(self, texts):
        out = []
        for t in texts:
            # Just use hash to seed a vector
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(8)
            v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v.tolist())
        return out


@pytest.mark.unit
def test_score_deterministic_aggregates_have_all_keys():
    records = [
        {
            "id": "q1", "type": "in_scope", "format_type": "YES_NO",
            "doc": "x", "question": "Is it open?",
            "answer": "YES - the lab is open.",
            "contexts": ["The lab is open daily from 8am to 6pm."],
            "ground_truth": "The lab is open daily.",
        },
        {
            "id": "q2", "type": "out_of_scope", "format_type": None,
            "doc": "x", "question": "What is the moon made of?",
            "answer": "I don't know based on the provided document.",
            "contexts": ["unrelated chunk"],
            "ground_truth": "I don't know based on the provided document.",
        },
    ]
    result = ev.score_deterministic(records, _ScoringEmb())
    agg = result["aggregates"]
    assert set(agg.keys()) == {
        "answer_correctness", "answer_f1", "refusal_accuracy",
        "format_compliance", "context_precision", "context_recall",
    }
    # Per-row dicts present
    assert len(result["per_row"]) == 2
    # The out-of-scope row with a refusal should score 1.0 on refusal accuracy.
    assert agg["refusal_accuracy"] == 1.0


@pytest.mark.unit
def test_score_deterministic_empty_records():
    """Empty records list -> all aggregates NaN, no per-row entries, no crash."""
    result = ev.score_deterministic([], _ScoringEmb())
    assert result["per_row"] == []
    agg = result["aggregates"]
    expected_keys = {
        "answer_correctness", "answer_f1", "refusal_accuracy",
        "format_compliance", "context_precision", "context_recall",
    }
    assert set(agg.keys()) == expected_keys
    for v in agg.values():
        assert np.isnan(v)


@pytest.mark.unit
def test_score_deterministic_handles_empty_contexts():
    records = [{
        "id": "q1", "type": "in_scope", "format_type": None,
        "doc": "x", "question": "Q?",
        "answer": "An answer.",
        "contexts": [],
        "ground_truth": "An answer.",
    }]
    result = ev.score_deterministic(records, _ScoringEmb())
    row = result["per_row"][0]
    assert np.isnan(row["context_precision"])
    assert np.isnan(row["context_recall"])


@pytest.mark.unit
def test_score_deterministic_format_compliance_recorded():
    records = [{
        "id": "q1", "type": "in_scope", "format_type": "LIST",
        "doc": "x", "question": "List things.",
        "answer": "- item one\n- item two",
        "contexts": ["item one and item two are here."],
        "ground_truth": "item one and item two",
    }]
    result = ev.score_deterministic(records, _ScoringEmb())
    assert result["per_row"][0]["format_compliance"] == 1.0


# ---------------------------------------------------------------------------
# print_aggregates (smoke test)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_print_aggregates_runs(capsys):
    ev.print_aggregates({"answer_correctness": 0.5, "answer_f1": float("nan")})
    out = capsys.readouterr().out
    assert "answer_correctness" in out
    assert "0.500" in out


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

@pytest.mark.functional
def test_main_smoke_skip_scoring(monkeypatch, tmp_path):
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        '{"id":"a","type":"in_scope","doc":"x","question":"Q1?","ground_truth":"A"}\n'
        '{"id":"b","type":"in_scope","doc":"x","question":"Q2?","ground_truth":"A"}\n'
        '{"id":"c","type":"in_scope","doc":"x","question":"Q3?","ground_truth":"A"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out.json"

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {"answer": "ok", "context": []}
    monkeypatch.setattr(ev, "get_embeddings", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ev, "build_chain_for_doc", MagicMock(return_value=(fake_chain, 1)))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py",
        "--dataset", str(dataset),
        "--output", str(output),
        "--skip-scoring",
        "--smoke",
    ])

    ev.main()

    payload = json.loads(output.read_text())
    # --smoke trims to 2 rows
    assert payload["n_rows"] == 2
    # --skip-scoring -> no aggregates
    assert "aggregates" not in payload


@pytest.mark.functional
def test_main_full_run_with_scoring(monkeypatch, tmp_path):
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        '{"id":"a","type":"in_scope","format_type":null,"doc":"x","question":"Q?","ground_truth":"text"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out.json"

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "answer": "text",
        "context": [Document(page_content="text", metadata={"page": 1})],
    }
    monkeypatch.setattr(ev, "get_embeddings", MagicMock(return_value=_ScoringEmb()))
    monkeypatch.setattr(ev, "build_chain_for_doc", MagicMock(return_value=(fake_chain, 1)))

    monkeypatch.setattr(sys, "argv", [
        "run_eval.py",
        "--dataset", str(dataset),
        "--output", str(output),
    ])

    ev.main()

    payload = json.loads(output.read_text())
    assert "aggregates" in payload
    assert payload["n_rows"] == 1
