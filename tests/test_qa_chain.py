"""Tests for build_qa_chain and answer_question."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import rag_pipeline as rp
from rag_pipeline import RagPipelineError, answer_question, build_qa_chain


@pytest.mark.unit
def test_build_qa_chain_without_chunks_uses_vectorstore_retriever(
    monkeypatch, fake_vectorstore, patch_external_models,
):
    """No chunks passed -> vectorstore.as_retriever is used directly."""
    monkeypatch.setattr(rp, "USE_RERANKER", False)
    chain = build_qa_chain(fake_vectorstore, chunks=None, use_reranker=False)
    assert chain is not None
    fake_vectorstore.as_retriever.assert_called()


@pytest.mark.unit
def test_build_qa_chain_with_chunks_enables_hybrid(
    monkeypatch, fake_vectorstore, patch_external_models,
):
    """chunks list -> build_hybrid_retriever path is taken."""
    hybrid_mock = MagicMock(return_value=fake_vectorstore.as_retriever.return_value)
    monkeypatch.setattr(rp, "build_hybrid_retriever", hybrid_mock)

    chunks = [Document(page_content="x")]
    chain = build_qa_chain(fake_vectorstore, chunks=chunks, use_reranker=False)
    assert chain is not None
    hybrid_mock.assert_called_once()
    assert hybrid_mock.call_args.args[1] == chunks


@pytest.mark.unit
def test_build_qa_chain_with_reranker_wraps(
    monkeypatch, fake_vectorstore, patch_external_models,
):
    """use_reranker=True -> wrap_with_reranker is invoked."""
    wrap_mock = MagicMock(return_value=MagicMock(name="wrapped"))
    monkeypatch.setattr(rp, "wrap_with_reranker", wrap_mock)

    build_qa_chain(fake_vectorstore, use_reranker=True, k=3, rerank_pool=10)
    wrap_mock.assert_called_once()
    assert wrap_mock.call_args.kwargs["top_n"] == 3


@pytest.mark.unit
def test_build_qa_chain_disables_reranker_when_flag_false(
    monkeypatch, fake_vectorstore, patch_external_models,
):
    wrap_mock = MagicMock()
    monkeypatch.setattr(rp, "wrap_with_reranker", wrap_mock)
    build_qa_chain(fake_vectorstore, use_reranker=False)
    wrap_mock.assert_not_called()


@pytest.mark.unit
def test_build_qa_chain_uses_use_reranker_env_default(
    monkeypatch, fake_vectorstore, patch_external_models,
):
    """When use_reranker arg is None, module-level USE_RERANKER decides."""
    monkeypatch.setattr(rp, "USE_RERANKER", True)
    wrap_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(rp, "wrap_with_reranker", wrap_mock)

    build_qa_chain(fake_vectorstore, use_reranker=None)
    wrap_mock.assert_called_once()


@pytest.mark.unit
def test_answer_question_success():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "answer": "The answer.",
        "context": [Document(page_content="c1", metadata={"page": 1})],
    }
    out = answer_question(fake_chain, "What is X?")
    assert out["answer"] == "The answer."
    assert len(out["context"]) == 1
    fake_chain.invoke.assert_called_once_with({"input": "What is X?"})


@pytest.mark.unit
def test_answer_question_empty_question_raises():
    fake_chain = MagicMock()
    with pytest.raises(RagPipelineError) as exc:
        answer_question(fake_chain, "")
    assert "empty" in str(exc.value).lower()


@pytest.mark.unit
def test_answer_question_whitespace_question_raises():
    fake_chain = MagicMock()
    with pytest.raises(RagPipelineError):
        answer_question(fake_chain, "   \n  ")


@pytest.mark.unit
def test_answer_question_urlerror_wrapped():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = urllib.error.URLError("lost connection")
    with pytest.raises(RagPipelineError) as exc:
        answer_question(fake_chain, "Q?")
    assert "Lost connection to Ollama" in str(exc.value)


@pytest.mark.unit
def test_answer_question_connection_keyword_wrapped():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("Connection refused by host")
    with pytest.raises(RagPipelineError) as exc:
        answer_question(fake_chain, "Q?")
    assert "Lost connection to Ollama" in str(exc.value)


@pytest.mark.unit
def test_answer_question_timeout_keyword_wrapped():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("Request timeout")
    with pytest.raises(RagPipelineError) as exc:
        answer_question(fake_chain, "Q?")
    assert "Lost connection to Ollama" in str(exc.value)


@pytest.mark.unit
def test_answer_question_generic_error_wrapped():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = ValueError("some other failure")
    with pytest.raises(RagPipelineError) as exc:
        answer_question(fake_chain, "Q?")
    assert "Answering failed" in str(exc.value)
