"""Tests for rag_pipeline.load_document — TXT/PDF extraction, error paths."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import rag_pipeline as rp
from rag_pipeline import RagPipelineError, load_document


@pytest.mark.functional
def test_load_txt_happy_path(sample_txt_path, monkeypatch):
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)
    docs = load_document(str(sample_txt_path))
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)
    assert all("category" in d.metadata for d in docs)
    combined = " ".join(d.page_content for d in docs)
    assert "Nova Robotics" in combined
    assert "Orientation" in combined


@pytest.mark.functional
def test_load_pdf_happy_path(sample_pdf_path, monkeypatch):
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)
    docs = load_document(str(sample_pdf_path))
    assert len(docs) >= 1
    combined = " ".join(d.page_content for d in docs)
    assert "Nova" in combined or "Orientation" in combined


@pytest.mark.functional
def test_load_pdf_with_tables(tabular_pdf_path, monkeypatch):
    """Tabular PDF should yield both narrative chunks and table+row chunks."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)
    docs = load_document(str(tabular_pdf_path))
    categories = [d.metadata.get("category") for d in docs]
    # The PyMuPDF table pass should add Table + TableRow entries.
    assert "Table" in categories
    assert "TableRow" in categories
    # Verify at least one row fact contains both header + value
    row_facts = [d.page_content for d in docs if d.metadata.get("category") == "TableRow"]
    assert any("SKU" in f and "X100" in f for f in row_facts)


@pytest.mark.unit
def test_unsupported_extension(tmp_path):
    p = tmp_path / "doc.docx"
    p.write_bytes(b"not a real docx")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    assert ".docx" in str(exc.value) or "Unsupported" in str(exc.value)


@pytest.mark.unit
def test_no_extension_raises(tmp_path):
    p = tmp_path / "noextension"
    p.write_text("hello")
    with pytest.raises(RagPipelineError):
        load_document(str(p))


@pytest.mark.unit
def test_missing_file_raises():
    with pytest.raises(RagPipelineError) as exc:
        load_document("/nonexistent/path/to/file.txt")
    assert "not found" in str(exc.value).lower()


@pytest.mark.unit
def test_corrupted_pdf_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a pdf at all")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    # Either "Failed to read" path or the no-extractable-text path
    assert "broken.pdf" in str(exc.value) or "extractable" in str(exc.value).lower()


@pytest.mark.unit
def test_partition_pdf_raises_propagates(tmp_path, monkeypatch):
    """Exception from unstructured.partition_pdf gets wrapped in RagPipelineError."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("encrypted document")

    import unstructured.partition.pdf as upp
    monkeypatch.setattr(upp, "partition_pdf", boom)

    p = tmp_path / "encrypted.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    msg = str(exc.value)
    assert "encrypted.pdf" in msg
    assert "encrypted" in msg.lower() or "corrupted" in msg.lower()


@pytest.mark.unit
def test_unicode_decode_error_propagates(tmp_path, monkeypatch):
    """UnicodeDecodeError should be wrapped in a friendly RagPipelineError."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    def boom(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")

    import unstructured.partition.text as upt
    monkeypatch.setattr(upt, "partition_text", boom)

    p = tmp_path / "garbled.txt"
    p.write_bytes(b"\xff\xfe garbled")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    assert "UTF-8" in str(exc.value) or "decode" in str(exc.value).lower()


@pytest.mark.unit
def test_empty_text_raises(tmp_path, monkeypatch):
    """An empty TXT should hit the 'no extractable text' branch."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    import unstructured.partition.text as upt
    monkeypatch.setattr(upt, "partition_text", lambda **kw: [])

    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    assert "extractable" in str(exc.value).lower()


@pytest.mark.unit
def test_filters_header_footer_pagenumber(tmp_path, monkeypatch):
    """Header/Footer/PageNumber elements should be dropped before chunking."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    class FakeElement:
        def __init__(self, text, category):
            self.text = text
            self.category = category
            self.metadata = type("M", (), {"page_number": 1})()

    elements = [
        FakeElement("PAGE HEADER", "Header"),
        FakeElement("PAGE 1", "PageNumber"),
        FakeElement("Real body text about widgets.", "NarrativeText"),
        FakeElement("FOOTER COPYRIGHT", "Footer"),
    ]

    import unstructured.partition.text as upt
    monkeypatch.setattr(upt, "partition_text", lambda **kw: elements)

    # Stub chunk_by_title to be identity-ish so we can inspect what survives
    import unstructured.chunking.title as uct

    def fake_chunk_by_title(els, **kw):
        return els

    monkeypatch.setattr(uct, "chunk_by_title", fake_chunk_by_title)

    p = tmp_path / "doc.txt"
    p.write_text("body")
    docs = load_document(str(p))

    texts = [d.page_content for d in docs]
    assert any("widgets" in t for t in texts)
    assert not any("HEADER" in t for t in texts)
    assert not any("FOOTER" in t for t in texts)
    assert not any("PAGE 1" in t for t in texts)


@pytest.mark.unit
def test_chunk_by_title_empty_text_filtered(tmp_path, monkeypatch):
    """Chunks with empty/whitespace text are skipped; if all empty, raise."""
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    class FakeChunk:
        def __init__(self, text):
            self.text = text
            self.category = "CompositeElement"
            self.metadata = type("M", (), {"page_number": 1})()

    class FakeElement:
        text = "x"
        category = "NarrativeText"
        metadata = type("M", (), {"page_number": 1})()

    import unstructured.partition.text as upt
    monkeypatch.setattr(upt, "partition_text", lambda **kw: [FakeElement()])

    import unstructured.chunking.title as uct
    monkeypatch.setattr(uct, "chunk_by_title", lambda els, **kw: [FakeChunk(""), FakeChunk("   ")])

    p = tmp_path / "all_blank.txt"
    p.write_text("x")
    with pytest.raises(RagPipelineError) as exc:
        load_document(str(p))
    assert "extractable" in str(exc.value).lower()
