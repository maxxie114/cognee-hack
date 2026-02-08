"""
RAG evaluation and tracing with Weave.

- Wrap retrieval with weave.op() for tracing
- RAGModel with predict(question) -> {answer, context} for evaluation
- LLM judge (context precision) scorer; optional RAGAS metrics as LLM judge
- Evaluation runner with configurable dataset

RAGAS (optional): install with `uv add "clinxplain[ragas]"` or `pip install ragas`.
Use scorers like ragas_faithfulness_score, ragas_answer_relevancy_score in Weave evaluations.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import RAGConfig
from .retrieval import get_retriever

# Optional Weave/Model imports for eval; allow running without weave for CLI help
try:
    import weave
    from weave import Model as WeaveModel
    _WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    WeaveModel = None
    _WEAVE_AVAILABLE = False

# Default evaluation questions (customize per knowledge base)
DEFAULT_EVAL_QUESTIONS: list[dict[str, str]] = [
    {"question": "What significant result was reported about Zealand Pharma's obesity trial?"},
    {"question": "How much did Berkshire Hathaway's cash levels increase in the fourth quarter?"},
    {"question": "What is the goal of Highmark Health's integration of Google Cloud and Epic Systems technology?"},
    {"question": "What were Rivian and Lucid's vehicle production forecasts for 2024?"},
    {"question": "Why was the Norwegian Dawn cruise ship denied access to Mauritius?"},
    {"question": "Which company achieved the first U.S. moon landing since 1972?"},
    {"question": "What issue did Intuitive Machines' lunar lander encounter upon landing on the moon?"},
]


def _get_retrieved_context(
    question: str,
    config: RAGConfig | None = None,
    top_k: int | None = None,
) -> str:
    """Retrieve context for a question and return as a single string. Used for tracing."""
    cfg = config or RAGConfig.from_env()
    k = top_k if top_k is not None else cfg.top_k
    retriever = get_retriever(cfg)
    docs = retriever(question, config=cfg, top_k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs) or "No relevant context found."


def _generate_answer(question: str, context_str: str, config: RAGConfig) -> str:
    """Generate answer from context and question using the same prompt as the RAG graph."""
    llm = ChatOpenAI(
        model=config.llm_model,
        temperature=config.llm_temperature,
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions based only on the following context. If the context does not contain enough information, say so. Do not make up information.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})
    return getattr(response, "content", None) or str(response)


# --- Weave-backed components (only when weave is available) ---

if _WEAVE_AVAILABLE:

    @weave.op()
    def get_retrieved_context(question: str) -> str:
        """
        Retrieval step wrapped with weave.op() for tracing.
        Returns concatenated context from vector search (Qdrant).
        """
        return _get_retrieved_context(question)

    class RAGModel(WeaveModel):
        """
        RAG model interface for Weave evaluation and tracing.

        predict(question) calls get_retrieved_context (traced) then generates an answer.
        Returns {"answer": ..., "context": ...} so scorers (e.g. context precision) can use both.
        """

        system_message: str = (
            "You answer questions based only on the provided context. "
            "If the context does not contain enough information, say so. Do not make up information."
        )
        model_name: str = "gpt-4o-mini"  # display only; actual model from RAGConfig

        def __init__(
            self,
            system_message: str | None = None,
            model_name: str | None = None,
            config: RAGConfig | None = None,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            if system_message is not None:
                self.system_message = system_message
            if model_name is not None:
                self.model_name = model_name
            self._config = config or RAGConfig.from_env()

        @weave.op()
        def predict(self, question: str) -> dict[str, Any]:
            """Run RAG: traced retrieval then generate; return answer and context for evaluation."""
            context_str = get_retrieved_context(question)
            answer = _generate_answer(question, context_str, self._config)
            return {"answer": answer, "context": context_str}

    @weave.op()
    async def context_precision_score(question: str, output: dict[str, Any]) -> dict[str, bool]:
        """
        LLM judge: was the retrieved context useful for the given answer?
        Prompt adapted from RAGAS-style context precision (see WandB Weave RAG tutorial).
        """
        from openai import OpenAI

        context_precision_prompt = """Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.
Output in only valid JSON format.

question: {question}
context: {context}
answer: {answer}
verdict: """
        client = OpenAI()
        ctx = output.get("context") or ""
        ans = output.get("answer") or ""
        prompt = context_precision_prompt.format(
            question=question,
            context=ctx,
            answer=ans,
        )
        response = client.chat.completions.create(
            model=os.getenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw) if raw else {}
        verdict = int(parsed.get("verdict", 0)) == 1
        return {"verdict": verdict}

    # --- Optional RAGAS metrics as Weave scorers (LLM judge) ---
    # See https://docs.ragas.io/
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics._faithfulness import faithfulness as _ragas_faithfulness
        from ragas.metrics._answer_relevance import answer_relevancy as _ragas_answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI as _LCOpenAI
        _RAGAS_AVAILABLE = True
    except ImportError:
        _RAGAS_AVAILABLE = False

    def _output_to_ragas_context(output: dict[str, Any]) -> list[str]:
        """Turn model output context string into list of chunks for RAGAS."""
        ctx = output.get("context") or ""
        return [s.strip() for s in ctx.split("\n\n---\n\n") if s.strip()] or [""]

    if _RAGAS_AVAILABLE:

        def _ragas_llm():
            model = os.getenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o-mini")
            return LangchainLLMWrapper(_LCOpenAI(model=model))

        def _ragas_one_row_sync(question: str, output: dict[str, Any], metrics_list: list, metric_key: str) -> float:
            """Run RAGAS evaluate on a single row (sync; run in executor from async)."""
            dataset = EvaluationDataset.from_list([{
                "user_input": question,
                "retrieved_contexts": _output_to_ragas_context(output),
                "response": output.get("answer") or "",
            }])
            result = evaluate(dataset, metrics=metrics_list, llm=_ragas_llm())
            if hasattr(result, "scores") and result.scores:
                return float(result.scores[0].get(metric_key, 0.0))
            return float(result.get(metric_key, 0.0))

        @weave.op()
        async def ragas_faithfulness_score(question: str, output: dict[str, Any]) -> dict[str, float]:
            """RAGAS Faithfulness: factual consistency of answer with retrieved context (0–1)."""
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(
                None,
                lambda: _ragas_one_row_sync(question, output, [_ragas_faithfulness], "faithfulness"),
            )
            return {"faithfulness": score}

        @weave.op()
        async def ragas_answer_relevancy_score(question: str, output: dict[str, Any]) -> dict[str, float]:
            """RAGAS Answer Relevancy: how relevant the answer is to the question (0–1)."""
            loop = asyncio.get_event_loop()
            score = await loop.run_in_executor(
                None,
                lambda: _ragas_one_row_sync(question, output, [_ragas_answer_relevancy], "answer_relevancy"),
            )
            return {"answer_relevancy": score}

        RAGAS_SCORERS = [ragas_faithfulness_score, ragas_answer_relevancy_score]
    else:
        ragas_faithfulness_score = None  # type: ignore[assignment]
        ragas_answer_relevancy_score = None  # type: ignore[assignment]
        RAGAS_SCORERS = []  # type: ignore[assignment]

    def _default_weave_project() -> str:
        """Use WEAVE_PROJECT env (required for Weave init)."""
        return os.getenv("WEAVE_PROJECT", "").strip()

    def init_weave(project: str | None = None) -> None:
        """Initialize Weave for tracing and evaluation. Call before predict or evaluate."""
        entity_project = (project or os.getenv("WEAVE_PROJECT") or _default_weave_project()).strip()
        if not entity_project:
            raise ValueError(
                "Weave project (entity/project) is required. Set WEAVE_PROJECT in .env, e.g.:\n"
                "  WEAVE_PROJECT=your-username/rag-eval"
            )
        weave.init(entity_project)

    def run_evaluation(
        project: str | None = None,
        dataset: list[dict[str, str]] | None = None,
        system_message: str | None = None,
        parallelism: int | None = None,
        scorers: list | None = None,
        use_ragas: bool = False,
    ) -> Any:
        """
        Run RAG evaluation with Weave over the dataset.

        Args:
            project: Weave project (e.g. 'team/rag-eval'). Uses WEAVE_PROJECT env if unset.
            dataset: List of {"question": "..."}. Uses DEFAULT_EVAL_QUESTIONS if unset.
            system_message: Override default RAG system message.
            parallelism: Max parallel workers (e.g. WEAVE_PARALLELISM=3 to avoid rate limits).
            scorers: List of Weave scorer functions (question, output) -> dict. If None, uses context_precision_score.
            use_ragas: If True and ragas is installed, add RAGAS scorers (faithfulness, answer_relevancy).

        Returns:
            Result of evaluation.evaluate(model).
        """
        if not _WEAVE_AVAILABLE:
            raise ImportError("weave is required for evaluation. Install with: pip install weave")
        if parallelism is not None:
            os.environ["WEAVE_PARALLELISM"] = str(parallelism)
        init_weave(project)
        questions = dataset if dataset is not None else DEFAULT_EVAL_QUESTIONS
        model = RAGModel(system_message=system_message or RAGModel.system_message)
        if scorers is not None:
            scorer_list = scorers
        else:
            scorer_list = [context_precision_score]
            if use_ragas and _RAGAS_AVAILABLE and RAGAS_SCORERS:
                scorer_list = scorer_list + list(RAGAS_SCORERS)
        evaluation = weave.Evaluation(dataset=questions, scorers=scorer_list)
        return asyncio.run(evaluation.evaluate(model))

else:
    RAGModel = None  # type: ignore[misc, assignment]
    get_retrieved_context = None  # type: ignore[assignment]
    context_precision_score = None  # type: ignore[assignment]
    init_weave = None  # type: ignore[assignment]
    run_evaluation = None  # type: ignore[assignment]
