"""Tests for rag_pipeline.get_embeddings and build_vectorstore."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import rag_pipeline as rp
from rag_pipeline import build_vectorstore, get_embeddings


@pytest.mark.unit
def test_get_embeddings_constructs_with_model_name(monkeypatch):
    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="fake_emb")

    monkeypatch.setattr(rp, "HuggingFaceEmbeddings", fake_ctor)
    get_embeddings()

    assert captured.get("model_name") == rp.EMBEDDING_MODEL
    assert captured.get("encode_kwargs", {}).get("normalize_embeddings") is True


@pytest.mark.unit
def test_build_vectorstore_creates_collection(monkeypatch, fake_embeddings):
    """build_vectorstore wraps the chunks in Chroma + add_documents."""
    fake_store = MagicMock(name="fake_chroma")
    fake_store.get.return_value = {"ids": []}
    chroma_ctor = MagicMock(return_value=fake_store)
    monkeypatch.setattr(rp, "Chroma", chroma_ctor)

    chunks = [Document(page_content="a"), Document(page_content="b")]
    store = build_vectorstore(
        chunks, collection_name="test_coll", persist_dir="/tmp/x",
        embeddings=fake_embeddings,
    )

    assert store is fake_store
    chroma_ctor.assert_called_once()
    call_kwargs = chroma_ctor.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_coll"
    assert call_kwargs["persist_directory"] == "/tmp/x"
    assert call_kwargs["embedding_function"] is fake_embeddings
    fake_store.add_documents.assert_called_once_with(chunks)


@pytest.mark.unit
def test_build_vectorstore_resets_existing_collection(monkeypatch, fake_embeddings):
    """If the collection already has ids, they're deleted before re-adding."""
    fake_store = MagicMock(name="fake_chroma")
    fake_store.get.return_value = {"ids": ["id1", "id2", "id3"]}
    monkeypatch.setattr(rp, "Chroma", MagicMock(return_value=fake_store))

    build_vectorstore(
        [Document(page_content="x")], collection_name="reuse",
        embeddings=fake_embeddings,
    )

    fake_store.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
    fake_store.add_documents.assert_called_once()


@pytest.mark.unit
def test_build_vectorstore_handles_get_failure(monkeypatch, fake_embeddings):
    """If store.get() raises (Chroma quirk), continue with add_documents anyway."""
    fake_store = MagicMock(name="fake_chroma")
    fake_store.get.side_effect = RuntimeError("Chroma boom")
    monkeypatch.setattr(rp, "Chroma", MagicMock(return_value=fake_store))

    # Should not raise.
    build_vectorstore(
        [Document(page_content="x")], collection_name="resilient",
        embeddings=fake_embeddings,
    )
    fake_store.delete.assert_not_called()
    fake_store.add_documents.assert_called_once()


@pytest.mark.unit
def test_build_vectorstore_uses_default_embeddings_when_none(monkeypatch):
    """When embeddings=None, get_embeddings is called."""
    fake_store = MagicMock(name="fake_chroma")
    fake_store.get.return_value = {"ids": []}
    monkeypatch.setattr(rp, "Chroma", MagicMock(return_value=fake_store))

    get_emb_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(rp, "get_embeddings", get_emb_mock)

    build_vectorstore([Document(page_content="x")], collection_name="default")
    get_emb_mock.assert_called_once()


@pytest.mark.unit
def test_build_vectorstore_uses_default_persist_dir(monkeypatch, fake_embeddings):
    """When persist_dir is not passed, DEFAULT_CHROMA_DIR is used."""
    fake_store = MagicMock(name="fake_chroma")
    fake_store.get.return_value = {"ids": []}
    chroma_ctor = MagicMock(return_value=fake_store)
    monkeypatch.setattr(rp, "Chroma", chroma_ctor)

    build_vectorstore(
        [Document(page_content="x")], collection_name="c",
        embeddings=fake_embeddings,
    )
    assert chroma_ctor.call_args.kwargs["persist_directory"] == rp.DEFAULT_CHROMA_DIR
