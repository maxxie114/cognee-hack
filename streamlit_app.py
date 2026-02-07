import os
import asyncio
import resource
import pathlib
import threading
import concurrent.futures
from dotenv import load_dotenv

# Load Qdrant credentials
load_dotenv()
os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
os.environ.setdefault("VECTOR_DB_URL", os.getenv("QDRANT_URL", ""))
os.environ.setdefault("VECTOR_DB_KEY", os.getenv("QDRANT_API_KEY", ""))

# Ollama host: use OLLAMA_HOST env var if set (e.g. http://ollama:11434 in Docker)
_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

os.environ.setdefault("LLM_API_KEY", ".")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "cognee-distillabs-model-gguf-quantized")
os.environ.setdefault("LLM_ENDPOINT", f"{_ollama_host}/v1")
os.environ.setdefault("LLM_MAX_TOKENS", "16384")

os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("EMBEDDING_ENDPOINT", f"{_ollama_host}/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")

# Register Qdrant adapter BEFORE importing cognee
import cognee_community_vector_adapter_qdrant.register  # noqa: F401

import streamlit as st
from custom_retriever import GraphCompletionRetrieverWithUserPrompt

# Bump file descriptor limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(10000, hard), hard))


@st.cache_resource
def get_retriever():
    system_prompt_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, "prompts/system_prompt.txt")
        ).resolve()
    )
    return GraphCompletionRetrieverWithUserPrompt(
        user_prompt_filename="user_prompt.txt",
        system_prompt_path=system_prompt_path,
        top_k=10,
    )


def run_query(retriever, query: str) -> str:
    """Run async cognee query in a separate thread to avoid event loop conflicts."""
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(retriever.get_completion(query=query))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_run)
        result = future.result(timeout=300)
    return result[0]


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
                retriever = get_retriever()
                answer = run_query(retriever, query)
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
