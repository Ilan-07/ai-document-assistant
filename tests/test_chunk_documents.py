"""Tests for rag_pipeline.chunk_documents — element-aware + semantic chunking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import rag_pipeline as rp
from rag_pipeline import chunk_documents


@pytest.mark.unit
def test_default_chunking_returns_documents(sample_documents):
    out = chunk_documents(sample_documents)
    assert len(out) >= len(sample_documents)
    assert all(isinstance(d, Document) for d in out)


@pytest.mark.unit
def test_atomic_table_not_split():
    """Documents with category=Table must remain a single chunk no matter what."""
    big_table_text = "| col |\n| --- |\n" + "\n".join(f"| {i} |" for i in range(500))
    doc = Document(page_content=big_table_text, metadata={"category": "Table", "page": 1})
    out = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    # Should still be exactly one Document (the Table is atomic).
    assert len(out) == 1
    assert out[0].page_content == big_table_text


@pytest.mark.unit
def test_atomic_listitem_not_split():
    big_list_text = "- item " * 1000
    doc = Document(page_content=big_list_text, metadata={"category": "ListItem", "page": 1})
    out = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(out) == 1


@pytest.mark.unit
def test_oversized_narrative_resplit(oversized_document):
    """Narrative docs larger than chunk_size get split by RecursiveCharacterTextSplitter."""
    out = chunk_documents(oversized_document, chunk_size=500, chunk_overlap=50)
    assert len(out) > 1
    assert all(len(d.page_content) <= 700 for d in out)  # some slack for splitter behavior


@pytest.mark.unit
def test_small_narrative_passthrough():
    """Narrative docs at or below chunk_size pass through unchanged."""
    small = Document(page_content="Just a short note.", metadata={"category": "NarrativeText"})
    out = chunk_documents([small], chunk_size=1000)
    assert len(out) == 1
    assert out[0] is small


@pytest.mark.unit
def test_semantic_chunking_branch(monkeypatch, fake_embeddings):
    """semantic=True should route through SemanticChunker.split_documents."""
    fake_splitter = MagicMock()
    fake_splitter.split_documents.return_value = [
        Document(page_content="semantic chunk 1", metadata={}),
        Document(page_content="semantic chunk 2", metadata={}),
    ]
    fake_chunker_cls = MagicMock(return_value=fake_splitter)

    import langchain_experimental.text_splitter as lets
    monkeypatch.setattr(lets, "SemanticChunker", fake_chunker_cls)

    docs = [Document(page_content="anything", metadata={"category": "NarrativeText"})]
    out = chunk_documents(docs, semantic=True, embeddings=fake_embeddings)

    fake_chunker_cls.assert_called_once()
    fake_splitter.split_documents.assert_called_once_with(docs)
    assert len(out) == 2


@pytest.mark.unit
def test_semantic_uses_get_embeddings_when_not_provided(monkeypatch):
    """When semantic=True and no embeddings passed, get_embeddings() is called."""
    fake_splitter = MagicMock()
    fake_splitter.split_documents.return_value = []

    import langchain_experimental.text_splitter as lets
    monkeypatch.setattr(lets, "SemanticChunker", MagicMock(return_value=fake_splitter))

    get_emb_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(rp, "get_embeddings", get_emb_mock)

    chunk_documents([Document(page_content="x")], semantic=True)
    get_emb_mock.assert_called_once()


@pytest.mark.unit
def test_env_flag_semantic_chunking_default(monkeypatch):
    """SEMANTIC_CHUNKING module-level flag drives default behavior."""
    monkeypatch.setattr(rp, "SEMANTIC_CHUNKING", True)
    fake_splitter = MagicMock()
    fake_splitter.split_documents.return_value = []
    import langchain_experimental.text_splitter as lets
    monkeypatch.setattr(lets, "SemanticChunker", MagicMock(return_value=fake_splitter))
    monkeypatch.setattr(rp, "get_embeddings", MagicMock(return_value=MagicMock()))

    chunk_documents([Document(page_content="x")])
    fake_splitter.split_documents.assert_called_once()
