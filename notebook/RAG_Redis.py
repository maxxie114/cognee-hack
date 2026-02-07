import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # RAG Pipeline

    Verify the RAG pipeline (Redis + RedisVL + LangChain + LangGraph) and **Redis LLM cache** for fast repeated queries.

    - Config & Redis health
    - **Redis cache** (LLM response cache so repeated/similar questions are fast)
    - Ingest (optional), Retrieve, Query
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup: path and env
    """)
    return


@app.cell
def _():
    import os
    import sys

    # Project root: cwd if it has src/, else parent (when run from notebook/)
    project_root = os.getcwd()
    if not os.path.isdir(os.path.join(project_root, "src")):
        project_root = os.path.abspath(os.path.join(project_root, ".."))
    src = os.path.join(project_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from dotenv import load_dotenv
    load_dotenv(os.path.join(project_root, ".env"))

    print("Project root:", project_root)
    print("src on path:", src in sys.path)
    return os, project_root


@app.cell
def _():
    from RAG import RAGConfig, RAGPipeline

    config = RAGConfig.from_env()
    redis_display = config.redis_url[:50] + "..." if len(config.redis_url) > 50 else config.redis_url
    print("Redis URL:", redis_display)
    print("Index:", config.index_name)
    print("LLM model:", config.llm_model)
    print("Embedding model:", config.embedding_model)
    print("Top-K:", config.top_k)
    return RAGPipeline, config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Redis LLM cache (makes RAG fast)

    Enable Redis-backed cache for LLM responses. Repeated or identical questions return cached answers instead of calling the LLM again.
    """)
    return


@app.cell
def _(config):
    from redis import Redis
    from langchain_community.cache import RedisCache
    from langchain_core.globals import set_llm_cache

    redis_client = Redis.from_url(config.redis_url)
    cache = RedisCache(redis_=redis_client, ttl=3600)  # 1 hour TTL
    set_llm_cache(cache)
    print("Redis LLM cache enabled (TTL=3600s). Repeated queries will be fast.")
    return (redis_client,)


@app.cell
def _(config, redis_client):
    from redisvl.index import SearchIndex
    from RAG.ingestion import get_schema_dict

    try:
        ok = redis_client.ping()
        print("Redis PING:", "OK" if ok else "FAIL")
        schema = get_schema_dict(config)
        index = SearchIndex.from_dict(schema, redis_url=config.redis_url)
        print("Index exists:", index.exists())
    except Exception as e:
        print("Health check failed:", e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ingest documents (path → Redis)

    Load PDFs/text from a folder, chunk and embed them, and **store in Redis**. After this, Retrieve and Query read **from Redis** (vector search + cached LLM).

    - **Document path:** `docs/` at project root or `notebook/` (e.g. 66_10.pdf).
    - Run once to index; re-run to add more documents.
    """)
    return


@app.cell
def _(RAGPipeline, config, os, project_root):
    # Document path: prefer docs/, fallback to notebook/ (e.g. 66_10.pdf)
    docs_dir = os.path.join(project_root, "docs")
    notebook_dir = os.path.join(project_root, "notebook")
    if os.path.isdir(docs_dir) and any(f.endswith((".pdf", ".txt", ".md")) for f in os.listdir(docs_dir)):
        docs_path = docs_dir
    elif os.path.isdir(notebook_dir) and any(f.endswith((".pdf", ".txt", ".md")) for f in os.listdir(notebook_dir)):
        docs_path = notebook_dir
    else:
        docs_path = docs_dir
    os.makedirs(docs_path, exist_ok=True)
    print("Document path:", docs_path)

    pipeline = RAGPipeline(config)
    try:
        keys = pipeline.ingest(docs_path, loader_type="auto", glob="**/*.pdf")
        print("Ingested chunks into Redis:", len(keys))
    except Exception as e:
        print("Ingest failed (add PDFs to docs/ or notebook/ and re-run):", e)
    return (pipeline,)


@app.cell
def _(RAGPipeline, config):
    pipeline = RAGPipeline(config)
    docs = pipeline.retrieve("What is this about?", top_k=3)
    print("Retrieved chunks:", len(docs))
    for i, d in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(d.page_content[:250] + "..." if len(d.page_content) > 250 else d.page_content)
    return (pipeline,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Query (RAG: retrieve + generate)

    First run may call the LLM. **Run the same question again** to see cache hit (fast response).
    """)
    return


@app.cell
def _(pipeline):
    import time

    question = "What is the main topic?"

    start = time.perf_counter()
    answer = pipeline.query(question)
    elapsed = time.perf_counter() - start

    print("Question:", question)
    print("Answer:", answer)
    print(f"Time: {elapsed:.2f}s (run again for cache hit → much faster)")
    return question, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cache hit test

    Run the same question again. With Redis cache enabled, the second run should be much faster (cached LLM response).
    """)
    return


@app.cell
def _(pipeline, question, time):
    start2 = time.perf_counter()
    answer2 = pipeline.query(question)
    elapsed2 = time.perf_counter() - start2

    print("Question (same):", question)
    print("Answer:", answer2)
    print(f"Time: {elapsed2:.2f}s (expect faster if cache hit)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using from the main app

    The RAG pipeline is used by the **supervisor agent** directly (no HTTP API). Import `RAGPipeline` and call `ingest()`, `retrieve()`, `query()`.
    """)
    return


@app.cell
def _():
    # Supervisor agent uses RAGPipeline directly:
    # from RAG import RAGPipeline, RAGConfig
    # pipeline = RAGPipeline(); pipeline.ingest(...); pipeline.retrieve(...); pipeline.query(...)
    print("RAG pipeline ready for supervisor agent (no API).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RAG Evaluation (Weave)

    Run RAG evaluation with **Weave** (tracing + LLM judge). For each question we:
    1. **Predict**: retrieve context and generate an answer (traced in Weave).
    2. **Score**: LLM judge checks if the context was useful for the answer (context precision).

    Use your own questions below to match your ingested documents, or use the default set.
    """)
    return


@app.cell
async def _(os):
    # RAG evaluation: Weave tracing + LLM judge (context precision)
    # (Use await in this cell — Jupyter already has a running event loop, so we avoid asyncio.run().)
    from RAG.weave_eval import (
        init_weave,
        RAGModel,
        context_precision_score,
        DEFAULT_EVAL_QUESTIONS,
    )

    # Questions to evaluate (customize to match your ingested docs, or use default)
    eval_questions = [
        {"question": "What is the main topic of the document?"},
        {"question": "What is openBIM and how is it defined?"},
        {"question": "Which tools are used for the analysis (Blender, add-on)?"},
        {"question": "What is the buildingSMART organization?"},
    ]
    # Or use the full default set: eval_questions = DEFAULT_EVAL_QUESTIONS

    # Initialize Weave (set WEAVE_PROJECT in .env for eval project)
    init_weave(project=os.getenv("WEAVE_PROJECT") or None)

    # Run evaluation: predict then score each row (async — safe in Jupyter's event loop)
    async def run_eval():
        model = RAGModel()
        results = []
        for row in eval_questions:
            q = row["question"]
            out = model.predict(q)
            score = await context_precision_score(q, out)
            results.append({
                "question": q,
                "answer": (out.get("answer") or "")[:300],
                "context_preview": (out.get("context") or "")[:150],
                "verdict": score.get("verdict", False),
            })
        return results

    eval_results = await run_eval()
    print(f"Evaluated {len(eval_results)} questions. See chart below.")
    return (eval_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation results chart

    Summary and per-question **context precision** (was the retrieved context useful for the answer?). Green = useful, red = not useful.
    """)
    return


@app.cell
def _(eval_results):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Summary stats
    n = len(eval_results)
    passed = sum(1 for r in eval_results if r["verdict"])
    pct = (passed / n * 100) if n else 0

    # Figure: summary card + horizontal bar chart
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [0.35, 0.65]})

    # Summary card
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    bg = "#1a1a2e"
    fig.patch.set_facecolor(bg)
    ax0.set_facecolor(bg)
    ax1.set_facecolor(bg)
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("white")
    ax1.spines["left"].set_color("white")
    ax1.xaxis.label.set_color("white")
    ax1.yaxis.label.set_color("white")

    # Big score text
    ax0.text(0.5, 0.6, "Context precision", ha="center", fontsize=14, color="#a0a0b0")
    ax0.text(0.5, 0.25, f"{passed} / {n}  ({pct:.0f}%)", ha="center", fontsize=28, color="#00d4aa", fontweight="bold")
    ax0.text(0.5, -0.05, "questions with useful context for the answer", ha="center", fontsize=11, color="#808090")

    # Bar chart: one bar per question
    labels = [r["question"][:50] + ("..." if len(r["question"]) > 50 else "") for r in eval_results]
    y_pos = range(len(labels))
    colors = ["#00d4aa" if r["verdict"] else "#ff6b6b" for r in eval_results]
    bars = ax1.barh(y_pos, [1] * len(labels), color=colors, height=0.6, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10, color="white")
    ax1.set_xlim(0, 1.2)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels(["", "Score", "Pass"])
    ax1.set_xlabel("")
    legend = [mpatches.Patch(color="#00d4aa", label="Useful context"), mpatches.Patch(color="#ff6b6b", label="Not useful")]
    ax1.legend(handles=legend, loc="lower right", facecolor="#2a2a4e", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(eval_results):
    # Table: question, answer snippet, verdict
    from IPython.display import display, HTML

    rows = []
    for r in eval_results:
        verdict_label = "✓ Useful" if r["verdict"] else "✗ Not useful"
        verdict_color = "#00d4aa" if r["verdict"] else "#ff6b6b"
        rows.append(
            f"<tr><td style='padding:8px;border-bottom:1px solid #333'>{r['question']}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #333;max-width:400px'>{r['answer'][:200]}…</td>"
            f"<td style='padding:8px;border-bottom:1px solid #333;color:{verdict_color};font-weight:bold'>{verdict_label}</td></tr>"
        )
    table = (
        "<table style='width:100%; border-collapse:collapse; color:#e0e0e0; font-size:13px'>"
        "<thead><tr><th style='text-align:left;padding:8px'>Question</th><th style='text-align:left;padding:8px'>Answer (snippet)</th><th style='text-align:left;padding:8px'>Context precision</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    display(HTML(table))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
