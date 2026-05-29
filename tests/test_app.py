"""Tests for app.py (Streamlit UI) with the streamlit module mocked.

Every test uses `mock_streamlit` (replaces st.* with stubs) and
`patch_external_models` indirectly via fixture-level patches.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import app
from rag_pipeline import RagPipelineError


# ---------------------------------------------------------------------------
# file_fingerprint
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_file_fingerprint_stable():
    fp1 = app.file_fingerprint(b"hello world", "doc.txt")
    fp2 = app.file_fingerprint(b"hello world", "doc.txt")
    assert fp1 == fp2


@pytest.mark.unit
def test_file_fingerprint_length_16():
    assert len(app.file_fingerprint(b"x", "name")) == 16


@pytest.mark.unit
def test_file_fingerprint_changes_with_name():
    fp1 = app.file_fingerprint(b"same", "a.txt")
    fp2 = app.file_fingerprint(b"same", "b.txt")
    assert fp1 != fp2


@pytest.mark.unit
def test_file_fingerprint_changes_with_content():
    fp1 = app.file_fingerprint(b"one", "n.txt")
    fp2 = app.file_fingerprint(b"two", "n.txt")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# cached_embeddings
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cached_embeddings_calls_get_embeddings(mock_streamlit, monkeypatch):
    fake = MagicMock(name="fake_embeddings")
    monkeypatch.setattr(app, "get_embeddings", MagicMock(return_value=fake))
    result = app.cached_embeddings()
    assert result is fake


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@pytest.mark.functional
def test_ingest_happy_path(mock_streamlit, monkeypatch, uploaded_file_factory):
    monkeypatch.setattr(app, "load_document", MagicMock(return_value=[
        Document(page_content="page text", metadata={"page": 1})
    ]))
    monkeypatch.setattr(app, "chunk_documents", MagicMock(return_value=[
        Document(page_content="chunk1"),
        Document(page_content="chunk2"),
    ]))
    monkeypatch.setattr(app, "build_vectorstore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(app, "build_qa_chain", MagicMock(return_value="FAKE_CHAIN"))
    monkeypatch.setattr(app, "cached_embeddings", MagicMock(return_value=MagicMock()))

    uploaded = uploaded_file_factory(name="myfile.txt", data=b"hello content")
    result = app.ingest(uploaded, model_name="llama3.2:3b", base_url="http://x")

    assert result["chain"] == "FAKE_CHAIN"
    assert result["filename"] == "myfile.txt"
    assert result["chunk_count"] == 2
    assert result["page_count"] == 1
    assert len(result["fingerprint"]) == 16


@pytest.mark.functional
def test_ingest_cleans_temp_file_on_success(mock_streamlit, monkeypatch, uploaded_file_factory):
    """The temp file should be unlinked after ingest, even on success."""
    captured_paths = []

    def capture_load(path):
        captured_paths.append(path)
        return [Document(page_content="x")]

    monkeypatch.setattr(app, "load_document", capture_load)
    monkeypatch.setattr(app, "chunk_documents", MagicMock(return_value=[Document(page_content="x")]))
    monkeypatch.setattr(app, "build_vectorstore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(app, "build_qa_chain", MagicMock(return_value="C"))
    monkeypatch.setattr(app, "cached_embeddings", MagicMock(return_value=MagicMock()))

    uploaded = uploaded_file_factory(name="x.txt", data=b"x")
    app.ingest(uploaded, "m", "http://x")
    assert captured_paths
    assert not os.path.exists(captured_paths[0])


@pytest.mark.functional
def test_ingest_cleans_temp_file_on_error(mock_streamlit, monkeypatch, uploaded_file_factory):
    captured_paths = []

    def boom(path):
        captured_paths.append(path)
        raise RagPipelineError("boom")

    monkeypatch.setattr(app, "load_document", boom)

    uploaded = uploaded_file_factory(name="bad.txt", data=b"x")
    with pytest.raises(RagPipelineError):
        app.ingest(uploaded, "m", "http://x")
    assert captured_paths
    assert not os.path.exists(captured_paths[0])


@pytest.mark.unit
def test_ingest_swallows_unlink_oserror(mock_streamlit, monkeypatch, uploaded_file_factory):
    """If os.unlink raises OSError in cleanup, ingest should not blow up."""
    monkeypatch.setattr(app, "load_document", MagicMock(return_value=[Document(page_content="x")]))
    monkeypatch.setattr(app, "chunk_documents", MagicMock(return_value=[Document(page_content="x")]))
    monkeypatch.setattr(app, "build_vectorstore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(app, "build_qa_chain", MagicMock(return_value="C"))
    monkeypatch.setattr(app, "cached_embeddings", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(app.os, "unlink", MagicMock(side_effect=OSError("locked")))

    uploaded = uploaded_file_factory(name="x.txt", data=b"x")
    result = app.ingest(uploaded, "m", "http://x")
    assert result["chain"] == "C"


# ---------------------------------------------------------------------------
# render_sidebar
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_render_sidebar_writes_session_state(mock_streamlit, monkeypatch):
    import streamlit as st
    # text_input returns the value param (per our mock side_effect)
    st.text_input.side_effect = lambda label, value=None, **kw: value or "default-val"
    model, base = app.render_sidebar()
    assert st.session_state["model_name"] == model
    assert st.session_state["base_url"] == base


@pytest.mark.unit
def test_render_sidebar_uses_existing_session_state(mock_streamlit, monkeypatch):
    import streamlit as st
    st.session_state["model_name"] = "phi:latest"
    st.session_state["base_url"] = "http://other:99"
    captured = {}

    def fake_text_input(label, value=None, **kw):
        captured.setdefault(label, value)
        return value

    st.text_input.side_effect = fake_text_input
    app.render_sidebar()
    assert captured["Ollama model"] == "phi:latest"
    assert captured["Ollama base URL"] == "http://other:99"


@pytest.mark.unit
def test_render_sidebar_clear_button(mock_streamlit, monkeypatch):
    import streamlit as st
    import shutil as real_shutil

    rmtree_mock = MagicMock()
    monkeypatch.setattr(real_shutil, "rmtree", rmtree_mock)

    st.session_state["doc"] = {"x": 1}
    st.session_state["messages"] = [{"role": "user"}]
    st.button.return_value = True
    st.text_input.side_effect = lambda label, value=None, **kw: value or "v"

    app.render_sidebar()
    rmtree_mock.assert_called_once()
    assert "doc" not in st.session_state
    assert "messages" not in st.session_state
    st.rerun.assert_called_once()


# ---------------------------------------------------------------------------
# render_chat
# ---------------------------------------------------------------------------

def _doc_info(chain=None):
    return {
        "chain": chain or MagicMock(),
        "fingerprint": "abc123",
        "filename": "test.txt",
        "chunk_count": 5,
        "page_count": 2,
    }


@pytest.mark.unit
def test_render_chat_no_question_returns_early(mock_streamlit):
    import streamlit as st
    st.chat_input.return_value = None
    st.session_state["messages"] = []
    app.render_chat(_doc_info())
    assert st.session_state["messages"] == []


@pytest.mark.unit
def test_render_chat_happy_path(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = "What is X?"

    fake_chain = MagicMock()
    monkeypatch.setattr(app, "answer_question", MagicMock(return_value={
        "answer": "The answer is 42.",
        "context": [
            Document(page_content="ctx 1", metadata={"page": 0}),
            Document(page_content="ctx 2", metadata={"page": 1}),
        ],
    }))
    app.render_chat(_doc_info(chain=fake_chain))

    msgs = st.session_state["messages"]
    # user message + assistant message
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "What is X?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "The answer is 42."
    assert len(msgs[1]["sources"]) == 2


@pytest.mark.unit
def test_render_chat_replays_existing_messages(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = None
    st.session_state["messages"] = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer",
         "sources": [{"page": 0, "snippet": "src snippet"}]},
    ]
    # No new question -> no new appends, but the replay loop runs.
    app.render_chat(_doc_info())
    assert len(st.session_state["messages"]) == 2


@pytest.mark.unit
def test_render_chat_replays_message_without_sources(mock_streamlit):
    import streamlit as st
    st.chat_input.return_value = None
    st.session_state["messages"] = [
        {"role": "assistant", "content": "no-source answer"},  # no "sources" key
    ]
    # Should not crash on missing sources key.
    app.render_chat(_doc_info())


@pytest.mark.unit
def test_render_chat_rag_error(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = "Q?"
    monkeypatch.setattr(app, "answer_question",
                        MagicMock(side_effect=RagPipelineError("Ollama is down")))

    app.render_chat(_doc_info())
    msgs = st.session_state["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert "Ollama is down" in msgs[-1]["content"]
    st.error.assert_called()


@pytest.mark.unit
def test_render_chat_unexpected_error(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = "Q?"
    monkeypatch.setattr(app, "answer_question",
                        MagicMock(side_effect=ValueError("weird")))

    app.render_chat(_doc_info())
    msgs = st.session_state["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert "weird" in msgs[-1]["content"]


@pytest.mark.unit
def test_render_chat_truncates_long_snippet(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = "Q?"
    long_text = "x" * 600
    monkeypatch.setattr(app, "answer_question", MagicMock(return_value={
        "answer": "ok",
        "context": [Document(page_content=long_text, metadata={"page": 5})],
    }))

    app.render_chat(_doc_info())
    snippet = st.session_state["messages"][-1]["sources"][0]["snippet"]
    assert snippet.endswith("...")
    assert len(snippet) == 403  # 400 chars + "..."


@pytest.mark.unit
def test_render_chat_empty_context_no_expander(mock_streamlit, monkeypatch):
    import streamlit as st
    st.chat_input.return_value = "Q?"
    monkeypatch.setattr(app, "answer_question",
                        MagicMock(return_value={"answer": "no sources", "context": []}))
    app.render_chat(_doc_info())
    assert st.session_state["messages"][-1]["sources"] == []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_main_no_upload_shows_info(mock_streamlit, monkeypatch):
    import streamlit as st
    monkeypatch.setattr(app, "render_sidebar", MagicMock(return_value=("m", "http://x")))
    st.file_uploader.return_value = None

    app.main()
    st.info.assert_called()


@pytest.mark.unit
def test_main_new_file_triggers_ingest(mock_streamlit, monkeypatch, uploaded_file_factory):
    import streamlit as st
    monkeypatch.setattr(app, "render_sidebar", MagicMock(return_value=("m", "http://x")))
    uploaded = uploaded_file_factory(name="new.txt", data=b"abc")
    st.file_uploader.return_value = uploaded

    ingest_mock = MagicMock(return_value={
        "chain": "C", "fingerprint": "fp1", "filename": "new.txt",
        "chunk_count": 1, "page_count": 1,
    })
    monkeypatch.setattr(app, "ingest", ingest_mock)
    monkeypatch.setattr(app, "render_chat", MagicMock())

    app.main()

    ingest_mock.assert_called_once()
    assert st.session_state["doc"]["chain"] == "C"
    st.toast.assert_called_once()


@pytest.mark.unit
def test_main_same_fingerprint_skips_ingest(mock_streamlit, monkeypatch, uploaded_file_factory):
    import streamlit as st
    monkeypatch.setattr(app, "render_sidebar", MagicMock(return_value=("m", "http://x")))
    uploaded = uploaded_file_factory(name="same.txt", data=b"abc")
    st.file_uploader.return_value = uploaded

    # Pre-seed session with matching fingerprint
    fp = app.file_fingerprint(b"abc", "same.txt")
    st.session_state["doc"] = {
        "chain": "OLD", "fingerprint": fp, "filename": "same.txt",
        "chunk_count": 1, "page_count": 1,
    }

    ingest_mock = MagicMock()
    monkeypatch.setattr(app, "ingest", ingest_mock)
    monkeypatch.setattr(app, "render_chat", MagicMock())

    app.main()
    ingest_mock.assert_not_called()


@pytest.mark.unit
def test_main_ingest_rag_error(mock_streamlit, monkeypatch, uploaded_file_factory):
    import streamlit as st
    monkeypatch.setattr(app, "render_sidebar", MagicMock(return_value=("m", "http://x")))
    st.file_uploader.return_value = uploaded_file_factory(name="bad.txt", data=b"x")
    monkeypatch.setattr(app, "ingest", MagicMock(side_effect=RagPipelineError("cant load")))
    st.session_state["doc"] = {"fingerprint": "different", "chain": None}

    app.main()
    st.error.assert_called()
    assert "doc" not in st.session_state


@pytest.mark.unit
def test_main_ingest_unexpected_error(mock_streamlit, monkeypatch, uploaded_file_factory):
    import streamlit as st
    monkeypatch.setattr(app, "render_sidebar", MagicMock(return_value=("m", "http://x")))
    st.file_uploader.return_value = uploaded_file_factory(name="bad.txt", data=b"x")
    monkeypatch.setattr(app, "ingest", MagicMock(side_effect=ValueError("oops")))
    st.session_state["doc"] = None  # so existing is falsy

    app.main()
    st.error.assert_called()
