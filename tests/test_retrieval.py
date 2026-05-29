"""Tests for build_hybrid_retriever, get_reranker, wrap_with_reranker."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import rag_pipeline as rp
from rag_pipeline import (
    build_hybrid_retriever,
    get_reranker,
    wrap_with_reranker,
)


class _FakeRetriever(BaseRetriever):
    """Minimal BaseRetriever for EnsembleRetriever's pydantic validation."""

    label: str = "fake"
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return [Document(page_content=f"{self.label}:{query}")]


@pytest.mark.unit
def test_build_hybrid_retriever_combines_bm25_and_dense(monkeypatch):
    fake_bm25 = _FakeRetriever(label="bm25")
    bm25_factory = MagicMock(return_value=fake_bm25)
    monkeypatch.setattr(rp.BM25Retriever, "from_documents", bm25_factory)

    fake_dense = _FakeRetriever(label="dense")
    fake_vs = MagicMock(name="chroma")
    fake_vs.as_retriever.return_value = fake_dense

    chunks = [Document(page_content="a"), Document(page_content="b")]
    retr = build_hybrid_retriever(fake_vs, chunks, k=4, bm25_weight=0.5)

    assert isinstance(retr, EnsembleRetriever)
    bm25_factory.assert_called_once_with(chunks)
    assert fake_bm25.k == 4
    fake_vs.as_retriever.assert_called_once_with(search_kwargs={"k": 4})
    # Order: [bm25, dense] per the function definition
    assert retr.retrievers == [fake_bm25, fake_dense]
    assert retr.weights == [0.5, 0.5]


@pytest.mark.unit
def test_build_hybrid_retriever_dense_only_weight(monkeypatch):
    """bm25_weight=0 -> dense weight = 1.0."""
    monkeypatch.setattr(
        rp.BM25Retriever, "from_documents", MagicMock(return_value=_FakeRetriever(label="bm25"))
    )
    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = _FakeRetriever(label="dense")
    retr = build_hybrid_retriever(fake_vs, [Document(page_content="x")], bm25_weight=0.0)
    assert retr.weights == [0.0, 1.0]


@pytest.mark.unit
def test_build_hybrid_retriever_bm25_only_weight(monkeypatch):
    """bm25_weight=1.0 -> dense weight = 0.0."""
    monkeypatch.setattr(
        rp.BM25Retriever, "from_documents", MagicMock(return_value=_FakeRetriever(label="bm25"))
    )
    fake_vs = MagicMock()
    fake_vs.as_retriever.return_value = _FakeRetriever(label="dense")
    retr = build_hybrid_retriever(fake_vs, [Document(page_content="x")], bm25_weight=1.0)
    assert retr.weights == [1.0, 0.0]


@pytest.mark.unit
def test_get_reranker_caches_singleton(monkeypatch):
    """First call constructs, second call reuses the cached instance."""
    ctor = MagicMock(return_value=MagicMock(name="cross_encoder"))
    monkeypatch.setattr(rp, "HuggingFaceCrossEncoder", ctor)
    monkeypatch.setattr(rp, "_RERANKER_CACHE", None)

    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2
    assert ctor.call_count == 1


@pytest.mark.unit
def test_get_reranker_passes_model_name(monkeypatch):
    ctor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(rp, "HuggingFaceCrossEncoder", ctor)
    monkeypatch.setattr(rp, "_RERANKER_CACHE", None)

    get_reranker(model_name="custom/reranker")
    ctor.assert_called_once_with(model_name="custom/reranker")


@pytest.mark.unit
def test_wrap_with_reranker_returns_compression_retriever(monkeypatch):
    from langchain_community.cross_encoders.base import BaseCrossEncoder

    class _FakeCrossEncoder(BaseCrossEncoder):
        def score(self, text_pairs):
            return [1.0 - i * 0.1 for i in range(len(text_pairs))]

    monkeypatch.setattr(rp, "HuggingFaceCrossEncoder", lambda **kw: _FakeCrossEncoder())
    monkeypatch.setattr(rp, "_RERANKER_CACHE", None)

    base_retr = _FakeRetriever(label="base")
    wrapped = wrap_with_reranker(base_retr, top_n=3)

    assert isinstance(wrapped, ContextualCompressionRetriever)
    assert wrapped.base_retriever is base_retr
    assert getattr(wrapped.base_compressor, "top_n", None) == 3
