"""Tests for rag_pipeline private helpers: table detection, markdown, NLTK setup."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import rag_pipeline as rp
from rag_pipeline import (
    _looks_like_real_table,
    _row_to_fact,
    _table_to_markdown,
)


# ---------- _looks_like_real_table ----------

@pytest.mark.unit
def test_looks_like_real_table_true():
    rows = [
        ["SKU", "Product", "Price"],
        ["X100", "Widget", "$25"],
        ["X200", "Gadget", "$50"],
        ["X300", "Doohickey", "$75"],
    ]
    assert _looks_like_real_table(rows) is True


@pytest.mark.unit
def test_looks_like_real_table_too_few_rows():
    assert _looks_like_real_table([["a", "b"], ["1", "2"]]) is False


@pytest.mark.unit
def test_looks_like_real_table_single_column():
    assert _looks_like_real_table([["a"], ["1"], ["2"]]) is False


@pytest.mark.unit
def test_looks_like_real_table_empty_header():
    rows = [["", "", ""], ["1", "2", "3"], ["4", "5", "6"]]
    assert _looks_like_real_table(rows) is False


@pytest.mark.unit
def test_looks_like_real_table_long_header_rejected():
    """Headers averaging >30 chars are sentence-like, not a real table."""
    long_header = "This is clearly a paragraph that got misidentified as a table header"
    rows = [[long_header, long_header], ["1", "2"], ["3", "4"]]
    assert _looks_like_real_table(rows) is False


@pytest.mark.unit
def test_looks_like_real_table_sparse_cells_rejected():
    """Less than 50% filled cells -> probably not a real table."""
    rows = [
        ["A", "B", "C"],
        [None, None, None],
        ["", "", None],
        [None, "x", None],
    ]
    assert _looks_like_real_table(rows) is False


@pytest.mark.unit
def test_looks_like_real_table_empty_rows():
    """Header row with all empties after filtering."""
    rows = [[None, None, None], ["a", "b", "c"], ["d", "e", "f"]]
    assert _looks_like_real_table(rows) is False


# ---------- _table_to_markdown ----------

@pytest.mark.unit
def test_table_to_markdown_basic():
    rows = [["a", "b"], ["1", "2"]]
    md = _table_to_markdown(rows)
    assert "| a | b |" in md
    assert "| --- | --- |" in md
    assert "| 1 | 2 |" in md


@pytest.mark.unit
def test_table_to_markdown_escapes_pipes_and_newlines():
    rows = [["x|y", "ok"], ["a\nb", "c"]]
    md = _table_to_markdown(rows)
    assert "x\\|y" in md
    assert "a b" in md  # newline replaced with space


@pytest.mark.unit
def test_table_to_markdown_pads_uneven_rows():
    rows = [["a", "b", "c"], ["1"]]
    md = _table_to_markdown(rows)
    lines = md.splitlines()
    # Each row should have the same number of pipes
    pipe_count = lines[0].count("|")
    assert all(line.count("|") == pipe_count for line in lines)


@pytest.mark.unit
def test_table_to_markdown_handles_none_cells():
    rows = [["a", "b"], [None, "x"]]
    md = _table_to_markdown(rows)
    assert "| | x |" in md or "|  | x |" in md


# ---------- _row_to_fact ----------

@pytest.mark.unit
def test_row_to_fact_happy_path():
    fact = _row_to_fact(["Name", "Age"], ["Alice", "30"])
    assert "Name: Alice" in fact
    assert "Age: 30" in fact
    assert "; " in fact


@pytest.mark.unit
def test_row_to_fact_skips_empty_pairs():
    fact = _row_to_fact(["Name", "Age", "City"], ["Alice", "", None])
    assert fact == "Name: Alice"


@pytest.mark.unit
def test_row_to_fact_all_empty_returns_none():
    assert _row_to_fact(["", ""], [None, ""]) is None


@pytest.mark.unit
def test_row_to_fact_handles_numeric_values():
    fact = _row_to_fact(["SKU", "Price"], ["X100", 42])
    assert "Price: 42" in fact


# ---------- _ensure_nltk_data ----------

@pytest.mark.unit
def test_ensure_nltk_data_no_download_when_present(monkeypatch):
    fake_nltk = MagicMock()
    fake_nltk.data.find = MagicMock(return_value=True)
    fake_nltk.download = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "nltk", fake_nltk)

    rp._ensure_nltk_data()
    fake_nltk.download.assert_not_called()


@pytest.mark.unit
def test_ensure_nltk_data_downloads_when_missing(monkeypatch):
    fake_nltk = MagicMock()
    fake_nltk.data.find = MagicMock(side_effect=LookupError("missing"))
    fake_nltk.download = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "nltk", fake_nltk)

    rp._ensure_nltk_data()
    # Two resources requested -> two downloads
    assert fake_nltk.download.call_count == 2


# ---------- _extract_pdf_tables (functional, uses real PyMuPDF) ----------

@pytest.mark.functional
def test_extract_pdf_tables_real_table(tabular_pdf_path):
    """Tabular fixture should yield markdown table + row facts."""
    out = rp._extract_pdf_tables(str(tabular_pdf_path))
    assert len(out) >= 1
    categories = [d.metadata.get("category") for d in out]
    assert "Table" in categories
    # At least three TableRow docs from the three data rows
    assert categories.count("TableRow") >= 3


@pytest.mark.functional
def test_extract_pdf_tables_no_tables_returns_empty(sample_pdf_path):
    out = rp._extract_pdf_tables(str(sample_pdf_path))
    assert out == []


@pytest.mark.unit
def test_extract_pdf_tables_pymupdf_open_fails():
    """Broken file path -> pymupdf.open raises -> returns empty list."""
    out = rp._extract_pdf_tables("/nonexistent/missing.pdf")
    assert out == []
