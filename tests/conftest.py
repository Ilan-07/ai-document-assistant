"""Shared pytest fixtures for the RAG test suite.

All external services are mocked here so tests run fully offline:
- HuggingFace embedding model -> FakeEmbeddings (deterministic 384-d vectors)
- ChatOllama LLM -> MagicMock returning canned AIMessage
- HuggingFaceCrossEncoder -> MagicMock returning descending scores
- Streamlit -> module-level patch with widget stubs + dict-like session_state
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (no model download)
# ---------------------------------------------------------------------------

class FakeEmbeddings:
    """Deterministic 384-d unit-norm embeddings derived from a hash of the text.

    Same text -> same vector across calls. Different texts -> different vectors.
    Vectors are L2-normalized so cosine similarity == dot product.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.tolist()

    def embed_documents(self, texts):
        self.embed_documents_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        self.embed_query_calls.append(text)
        return self._vec(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


# ---------------------------------------------------------------------------
# Fake ChatOllama (no Ollama needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_chat_ollama():
    """A mock ChatOllama-like object. Tests can override `.invoke.return_value`."""
    from langchain_core.messages import AIMessage

    mock = MagicMock(name="FakeChatOllama")
    mock.invoke.return_value = AIMessage(content="Mock answer.")
    return mock


# ---------------------------------------------------------------------------
# Patch external model constructors in rag_pipeline namespace
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_external_models(monkeypatch, fake_embeddings, fake_chat_ollama):
    """Patch HuggingFaceEmbeddings, ChatOllama, CrossEncoder, NLTK, Ollama HTTP."""
    import rag_pipeline as rp

    monkeypatch.setattr(rp, "HuggingFaceEmbeddings", lambda **kw: fake_embeddings)
    monkeypatch.setattr(rp, "ChatOllama", lambda **kw: fake_chat_ollama)

    fake_reranker = MagicMock(name="FakeCrossEncoder")
    fake_reranker.score = MagicMock(side_effect=lambda pairs: [1.0 - i * 0.1 for i in range(len(pairs))])
    monkeypatch.setattr(rp, "HuggingFaceCrossEncoder", lambda **kw: fake_reranker)
    monkeypatch.setattr(rp, "_RERANKER_CACHE", None)
    monkeypatch.setattr(rp, "_ensure_nltk_data", lambda: None)

    # Stub Ollama readiness check so build_qa_chain succeeds without a server.
    monkeypatch.setattr(rp, "check_ollama_ready", lambda **kw: None)

    return {
        "embeddings": fake_embeddings,
        "chat": fake_chat_ollama,
        "reranker": fake_reranker,
    }


# ---------------------------------------------------------------------------
# Sample documents
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """Mixed-category Document list for chunking / retrieval tests."""
    return [
        Document(
            page_content="The Nova handbook explains onboarding procedures for new hires. "
                         "All new hires must complete orientation in the first week.",
            metadata={"category": "NarrativeText", "page": 1},
        ),
        Document(
            page_content="- First aid kit\n- Fire extinguisher\n- Emergency exit map",
            metadata={"category": "ListItem", "page": 2},
        ),
        Document(
            page_content="| Model | Year | Price |\n| --- | --- | --- |\n| X3 | 2023 | $400 |",
            metadata={"category": "Table", "page": 3},
        ),
        Document(
            page_content="Refer to section 4.2 for safety protocols.",
            metadata={"category": "NarrativeText", "page": 4},
        ),
    ]


@pytest.fixture
def oversized_document():
    """A single >5000-char narrative doc that should be re-split by chunk_documents."""
    para = "This is a sentence about widgets that goes on for a while. " * 100
    return [Document(page_content=para, metadata={"category": "NarrativeText", "page": 1})]


# ---------------------------------------------------------------------------
# Chroma vectorstore mock
# ---------------------------------------------------------------------------

class _FakeRetriever:
    """Minimal BaseRetriever-compatible Runnable for tests.

    Subclasses BaseRetriever so pydantic validators in EnsembleRetriever /
    create_retrieval_chain accept it without a real Chroma backing.
    """

    def __new__(cls, docs=None):
        from langchain_core.retrievers import BaseRetriever

        class _Impl(BaseRetriever):
            _docs: list = []

            def _get_relevant_documents(self, query, *, run_manager=None):
                return self._docs

        inst = _Impl()
        inst._docs = docs or [
            Document(page_content="retrieved chunk", metadata={"page": 1})
        ]
        return inst


@pytest.fixture
def fake_vectorstore():
    """Mock Chroma store. as_retriever() returns a real BaseRetriever."""
    store = MagicMock(name="FakeChroma")
    store.get.return_value = {"ids": []}
    store.add_documents = MagicMock()
    store.delete = MagicMock()
    store.as_retriever.return_value = _FakeRetriever()
    return store


# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt_path():
    return FIXTURES_DIR / "sample.txt"


@pytest.fixture
def sample_pdf_path():
    return FIXTURES_DIR / "sample.pdf"


@pytest.fixture
def tabular_pdf_path():
    return FIXTURES_DIR / "tabular.pdf"


# ---------------------------------------------------------------------------
# Streamlit module mock (for app.py tests)
# ---------------------------------------------------------------------------

class _SessionState(dict):
    """Mimics st.session_state - attribute and item access, with setdefault/pop."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


class _ContextManagerMock(MagicMock):
    """A MagicMock that also works as a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_context_callable(name="ctx"):
    """Returns a callable that, when called, returns a context manager mock."""
    cm = _ContextManagerMock(name=name)
    fn = MagicMock(name=f"{name}_factory", return_value=cm)
    return fn


@pytest.fixture
def mock_streamlit(monkeypatch):
    """Replace the `streamlit` module with stubs for everything app.py touches."""
    import streamlit as st

    session_state = _SessionState()
    monkeypatch.setattr(st, "session_state", session_state, raising=False)

    # Widgets that return values
    monkeypatch.setattr(st, "file_uploader", MagicMock(return_value=None))
    monkeypatch.setattr(st, "text_input", MagicMock(side_effect=lambda label, value=None, **kw: value))
    monkeypatch.setattr(st, "button", MagicMock(return_value=False))
    monkeypatch.setattr(st, "chat_input", MagicMock(return_value=None))

    # Context managers
    monkeypatch.setattr(st, "spinner", _make_context_callable("spinner"))
    monkeypatch.setattr(st, "chat_message", _make_context_callable("chat_message"))
    monkeypatch.setattr(st, "sidebar", _ContextManagerMock(name="sidebar"))
    monkeypatch.setattr(st, "expander", _make_context_callable("expander"))

    # Display calls (no return value)
    for fn_name in (
        "markdown", "error", "toast", "rerun", "info", "set_page_config",
        "header", "subheader", "caption", "divider", "title", "write",
    ):
        monkeypatch.setattr(st, fn_name, MagicMock(name=fn_name))

    # cache_resource: passthrough decorator that ignores its kwargs
    def _passthrough_decorator(*dargs, **dkwargs):
        # Called as @st.cache_resource or @st.cache_resource(...)
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def _wrap(fn):
            return fn

        return _wrap

    monkeypatch.setattr(st, "cache_resource", _passthrough_decorator)

    return st


# ---------------------------------------------------------------------------
# Uploaded-file factory for Streamlit ingest tests
# ---------------------------------------------------------------------------

@pytest.fixture
def uploaded_file_factory():
    """Build something that looks like Streamlit's UploadedFile."""

    def _make(name: str = "test.txt", data: bytes = b"hello world"):
        mock = MagicMock(name=f"UploadedFile({name})")
        mock.name = name
        mock.getvalue = MagicMock(return_value=data)
        return mock

    return _make
