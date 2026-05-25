"""Streamlit UI for the AI Document Assistant (RAG)."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    RagPipelineError,
    answer_question,
    build_qa_chain,
    build_vectorstore,
    chunk_documents,
    get_embeddings,
    load_document,
)

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings():
    return get_embeddings()


def file_fingerprint(data: bytes, name: str) -> str:
    h = hashlib.sha1()
    h.update(name.encode("utf-8"))
    h.update(data)
    return h.hexdigest()[:16]


def ingest(uploaded_file, model_name: str, base_url: str):
    data = uploaded_file.getvalue()
    fp = file_fingerprint(data, uploaded_file.name)
    collection = f"doc_{fp}"

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with st.spinner("Extracting text..."):
            docs = load_document(tmp_path)
        with st.spinner("Splitting into chunks..."):
            chunks = chunk_documents(docs)
        with st.spinner(f"Embedding {len(chunks)} chunks and indexing..."):
            store = build_vectorstore(
                chunks,
                collection_name=collection,
                embeddings=cached_embeddings(),
            )
        with st.spinner("Connecting to Ollama..."):
            chain = build_qa_chain(
                store, chunks=chunks, model_name=model_name, base_url=base_url
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {
        "chain": chain,
        "fingerprint": fp,
        "filename": uploaded_file.name,
        "chunk_count": len(chunks),
        "page_count": len(docs),
    }


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.title("📚 AI Document Assistant")
        st.caption("Upload a PDF or TXT, then ask questions about it.")
        st.divider()

        st.subheader("Model")
        model_name = st.text_input(
            "Ollama model",
            value=st.session_state.get("model_name", DEFAULT_OLLAMA_MODEL),
            help="Must be pulled locally via `ollama pull <model>`.",
        )
        base_url = st.text_input(
            "Ollama base URL",
            value=st.session_state.get("base_url", DEFAULT_OLLAMA_BASE_URL),
        )
        st.session_state["model_name"] = model_name
        st.session_state["base_url"] = base_url

        st.divider()
        if st.button("Clear index & chat", use_container_width=True):
            shutil.rmtree(DEFAULT_CHROMA_DIR, ignore_errors=True)
            for key in ("doc", "messages"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.caption("Stack: Streamlit + LangChain + ChromaDB + sentence-transformers + Ollama.")
    return model_name, base_url


def render_chat(doc_info: dict):
    st.subheader(f"Chat about: {doc_info['filename']}")
    st.caption(
        f"{doc_info['page_count']} pages/sections - {doc_info['chunk_count']} chunks indexed"
    )

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.get("page")
                        header = f"**Chunk {i}**"
                        if page is not None:
                            header += f" - page {page + 1}"
                        st.markdown(header)
                        st.markdown(f"> {src['snippet']}")

    question = st.chat_input("Ask a question about your document")
    if not question:
        return

    st.session_state.setdefault("messages", []).append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = answer_question(doc_info["chain"], question)
        except RagPipelineError as exc:
            st.error(str(exc))
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"⚠️ {exc}", "sources": []}
            )
            return
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"⚠️ {exc}", "sources": []}
            )
            return
        answer = result.get("answer", "").strip()
        st.markdown(answer)

        sources = []
        for d in result.get("context", []):
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            sources.append({"page": d.metadata.get("page"), "snippet": snippet})

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, src in enumerate(sources, 1):
                    page = src["page"]
                    header = f"**Chunk {i}**"
                    if page is not None:
                        header += f" - page {page + 1}"
                    st.markdown(header)
                    st.markdown(f"> {src['snippet']}")

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main():
    model_name, base_url = render_sidebar()

    st.header("Upload a document")
    uploaded = st.file_uploader(
        "PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Upload a PDF or TXT file in the panel above to get started.")
        return

    data = uploaded.getvalue()
    fp = file_fingerprint(data, uploaded.name)
    existing = st.session_state.get("doc")

    if existing is None or existing["fingerprint"] != fp:
        try:
            st.session_state["doc"] = ingest(uploaded, model_name, base_url)
        except RagPipelineError as exc:
            st.session_state.pop("doc", None)
            st.error(str(exc))
            return
        except Exception as exc:
            st.session_state.pop("doc", None)
            st.error(f"Unexpected error while indexing {uploaded.name}: {exc}")
            return
        st.session_state["messages"] = []
        st.toast(
            f"Indexed {st.session_state['doc']['chunk_count']} chunks from {uploaded.name}",
            icon="✅",
        )

    render_chat(st.session_state["doc"])


if __name__ == "__main__":
    main()
