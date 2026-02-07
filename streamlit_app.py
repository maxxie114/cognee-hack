import os
import requests
import streamlit as st

# API backend URL (FastAPI running on same container or host)
API_URL = os.getenv("COGNEE_API_URL", "http://localhost:8000")


def run_query(query: str) -> str:
    """Call the FastAPI /query endpoint."""
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": query},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


# --- UI ---
st.set_page_config(page_title="Cognee Q&A", page_icon="🧠", layout="wide")
st.title("🧠 Cognee Knowledge Graph Q&A")
st.caption("Ask questions about invoices, transactions, and vendors")

# Sidebar with example questions
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "Can you check whether all payments to Vendor 2 are correct?",
        "Did we ever pay for a laptop from Vendor 3?",
        "Which vendors consistently give us discounts?",
        "Have we paid for any storage devices recently?",
        "Do we spend more with Vendor 4 or Vendor 2?",
        "Did we clear all bills for high-value equipment orders?",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_query"] = ex
            st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question about the data...")

# Handle sidebar button click
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge graph... (this may take ~60s)"):
            try:
                answer = run_query(query)
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
