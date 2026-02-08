#!/usr/bin/env python3
"""ClinXplain chatbot: upload documents (RAG ingest) and chat via WebSocket with streaming."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import streamlit as st

st.set_page_config(page_title="ClinXplain Chat", layout="wide", initial_sidebar_state="expanded")

# Session state: chat history, API base, selected document for scoped chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_base" not in st.session_state:
    st.session_state.api_base = "http://localhost:8000"
if "selected_document_source" not in st.session_state:
    st.session_state.selected_document_source = None
if "selected_document_name" not in st.session_state:
    st.session_state.selected_document_name = None


def ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = parsed.path.rstrip("/") + "/ws/chat"
    return f"{scheme}://{netloc}{path}"


# ---------------------------------------------------------------------------
# Sidebar: API URL, document upload, clear chat
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    with st.expander("How to run", expanded=False):
        st.markdown("""
1. **Qdrant:** `docker run -p 6333:6333 qdrant/qdrant`
2. **API:** `uv run api` (default port 8000)
3. **This UI:** `uv run streamlit run streamlit_app.py`

Set **API base URL** below to match (e.g. http://localhost:8000).
        """.strip())
    base = st.text_input(
        "API base URL",
        value=st.session_state.api_base,
        help="ClinXplain API URL (no path). Default: http://localhost:8000",
        key="api_base_input",
    ).rstrip("/")
    st.session_state.api_base = base

    # Show whether the API is reachable
    api_ok = False
    if base:
        try:
            import httpx
            r = httpx.get(f"{base}/health", timeout=3.0)
            api_ok = r.status_code == 200
        except Exception:
            pass
    if base and not api_ok:
        st.error("API not reachable. Start it in a terminal:")
        st.code("uv run api", language="text")
        st.caption("Default: http://localhost:8000. Or: uv run api --port 8001")
    elif base:
        st.success("API connected")
        st.markdown(f"[Open API docs]({base}/docs)")

    st.divider()
    st.subheader("Upload documents")
    st.caption("Upload PDFs or text files to ingest into the RAG index (Qdrant). Then ask questions in the chat.")
    uploaded = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="uploader",
    )
    if uploaded and st.button("Ingest into RAG", key="ingest_btn"):
        try:
            import httpx
            files = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in uploaded
            ]
            # Try /ingest first, then /api/ingest if 404 (handles both server configs)
            for path in ("/ingest", "/api/ingest"):
                ingest_url = f"{base.rstrip('/')}{path}"
                r = httpx.post(ingest_url, files=files, timeout=120.0)
                if r.status_code == 200:
                    data = r.json()
                    st.success(
                        f"Ingested {data.get('chunks_ingested', 0)} chunks from {data.get('files_received', 0)} file(s). "
                        "Select a document below to chat (or use All documents)."
                    )
                    st.rerun()  # Refetch document list
                    break
                if r.status_code != 404:
                    st.error(f"Error {r.status_code}: {r.text[:200]}")
                    break
            else:
                # Both returned 404 — API unreachable or wrong base URL
                health_url = f"{base.rstrip('/')}/health"
                try:
                    h = httpx.get(health_url, timeout=5.0)
                    if h.status_code == 200:
                        st.error("Ingest endpoint not found (404) but API is up. Try opening the API docs to see routes: " + f"{base.rstrip('/')}/docs")
                    else:
                        st.error(f"API at {base} returned {h.status_code}. Start the API with: uv run api (in project root).")
                except httpx.ConnectError:
                    st.error(f"Cannot reach API at {base}. Start it with: uv run api (default port 8000).")
                except Exception:
                    st.error(f"Cannot reach API at {base}. Start it with: uv run api (default port 8000).")
        except Exception as e:
            st.error(str(e))

    st.divider()
    if st.button("Clear chat", key="clear_btn"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main: document selector, then chat
# ---------------------------------------------------------------------------
st.title("ClinXplain")
st.caption("Chat over your documents. Upload files in the sidebar, then select a document and ask questions below.")

# Document selector: fetch list from API, show dropdown (All documents + one per doc)
documents_list: list[dict[str, str]] = []
if base:
    try:
        import httpx
        for path in ("/documents", "/api/documents"):
            r_docs = httpx.get(f"{base.rstrip('/')}{path}", timeout=5.0)
            if r_docs.status_code == 200:
                documents_list = r_docs.json()
                break
    except Exception:
        pass

# Build options: "All documents" (value None) + one per document (value=source)
doc_options = [("All documents", None)]
for it in documents_list:
    src = it.get("source") or ""
    name = it.get("document_name") or src or "Unknown"
    doc_options.append((name, src))

# Selectbox: show document_name, store source in session state
if doc_options:
    labels = [o[0] for o in doc_options]
    idx = 0
    if st.session_state.selected_document_source is not None:
        for i, (_, src) in enumerate(doc_options):
            if src == st.session_state.selected_document_source:
                idx = i
                break
    selected_label = st.selectbox(
        "Chat with document",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        index=idx,
        key="doc_selector",
    )
    chosen_source = doc_options[selected_label][1]
    chosen_name = doc_options[selected_label][0]
    st.session_state.selected_document_source = chosen_source
    st.session_state.selected_document_name = chosen_name if chosen_source else None
else:
    st.session_state.selected_document_source = None
    st.session_state.selected_document_name = None
    if base:
        st.caption("No documents in the index yet. Upload files in the sidebar and ingest.")

st.divider()

# Remind to connect API if not reachable
if base:
    try:
        import httpx
        _h = httpx.get(f"{base}/health", timeout=2.0)
        if _h.status_code != 200:
            st.warning("API did not respond. Set **API base URL** in the sidebar (e.g. http://localhost:8000) and start the API: **uv run api**")
    except Exception:
        st.warning("API not reachable. In the sidebar set **API base URL** (e.g. http://localhost:8000) and run: **uv run api**")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and send
prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant reply via WebSocket (streaming from API; we show full response when done)
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        stream_placeholder.markdown("Thinking…")
        # Build conversation_history from prior user/assistant pairs
        conversation_history = []
        for i, m in enumerate(st.session_state.messages[:-1]):
            if m["role"] == "user" and i + 1 < len(st.session_state.messages):
                next_m = st.session_state.messages[i + 1]
                if next_m["role"] == "assistant":
                    conversation_history.append({"query": m["content"], "response": next_m["content"]})

        try:
            import websockets
        except ImportError:
            stream_placeholder.markdown("Install websockets: `uv add websockets`")
            st.session_state.messages.append({"role": "assistant", "content": ""})
            st.stop()

        async def run_ws():
            uri = ws_url(base)
            accumulated = []
            error_detail = None
            async with websockets.connect(uri, open_timeout=10, close_timeout=60) as ws:
                await ws.send(json.dumps({
                    "message": prompt,
                    "patient_id": None,
                    "document_source": st.session_state.get("selected_document_source"),
                    "conversation_history": conversation_history,
                }))
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    msg_type = data.get("type")
                    if msg_type == "token":
                        content = data.get("content", "")
                        accumulated.append(content)
                    elif msg_type == "done":
                        return data.get("response", "") or "".join(accumulated), None
                    elif msg_type == "error":
                        error_detail = data.get("detail", "Unknown error")
                        return "", error_detail
            return "".join(accumulated), error_detail

        full_response = ""
        try:
            full_response, err = asyncio.run(run_ws())
            if err:
                stream_placeholder.error(err)
                full_response = ""
            else:
                stream_placeholder.markdown(full_response or "_No response._")
        except Exception as e:
            err_msg = str(e)
            if "connect" in err_msg.lower() or "refused" in err_msg.lower() or "11001" in err_msg:
                stream_placeholder.error(
                    f"Cannot connect to API at {base}. Start the API: **uv run api** (default http://localhost:8000). "
                    "Set the same URL in the sidebar."
                )
            else:
                stream_placeholder.error(err_msg)
            full_response = ""

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()
