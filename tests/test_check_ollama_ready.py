"""Tests for rag_pipeline.check_ollama_ready — Ollama readiness probe."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock

import pytest

import rag_pipeline as rp
from rag_pipeline import RagPipelineError, check_ollama_ready


def _fake_response(payload: dict | str | None):
    """Build a context-manager response that .read()s the given payload."""
    cm = MagicMock()
    if isinstance(payload, dict):
        body = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = b""
    cm.__enter__ = MagicMock(return_value=BytesIO(body))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


@pytest.mark.unit
def test_ready_when_model_pulled(monkeypatch):
    """Model present in tags list -> no exception."""
    fake = MagicMock(return_value=_fake_response({"models": [{"name": "llama3.2:3b"}]}))
    monkeypatch.setattr(rp.urllib.request, "urlopen", fake)
    check_ollama_ready(base_url="http://x:11434", model_name="llama3.2:3b")


@pytest.mark.unit
def test_ready_when_model_base_name_matches(monkeypatch):
    """`llama3.2:3b` should match `llama3.2:1b` in tags (base name 'llama3.2')."""
    fake = MagicMock(return_value=_fake_response({"models": [{"name": "llama3.2:1b"}]}))
    monkeypatch.setattr(rp.urllib.request, "urlopen", fake)
    check_ollama_ready(model_name="llama3.2:3b")


@pytest.mark.unit
def test_raises_when_unreachable(monkeypatch):
    def boom(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(rp.urllib.request, "urlopen", boom)
    with pytest.raises(RagPipelineError) as exc:
        check_ollama_ready(base_url="http://nope:11434")
    assert "Could not reach Ollama" in str(exc.value)
    assert "nope" in str(exc.value)


@pytest.mark.unit
def test_raises_on_bad_json(monkeypatch):
    fake = MagicMock(return_value=_fake_response("not json at all"))
    monkeypatch.setattr(rp.urllib.request, "urlopen", fake)
    with pytest.raises(RagPipelineError) as exc:
        check_ollama_ready()
    assert "Unexpected response" in str(exc.value)


@pytest.mark.unit
def test_raises_when_model_not_pulled(monkeypatch):
    fake = MagicMock(return_value=_fake_response({"models": [{"name": "phi:latest"}]}))
    monkeypatch.setattr(rp.urllib.request, "urlopen", fake)
    with pytest.raises(RagPipelineError) as exc:
        check_ollama_ready(model_name="llama3.2:3b")
    assert "not pulled" in str(exc.value)
    assert "ollama pull" in str(exc.value)


@pytest.mark.unit
def test_raises_when_models_field_missing(monkeypatch):
    """Payload without 'models' key -> empty installed set -> not-pulled error."""
    fake = MagicMock(return_value=_fake_response({}))
    monkeypatch.setattr(rp.urllib.request, "urlopen", fake)
    with pytest.raises(RagPipelineError):
        check_ollama_ready(model_name="llama3.2:3b")
