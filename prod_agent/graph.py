"""The LangGraph agent under observation.

A small tool-using graph (answer -> maybe recheck with tools -> finalize)
instrumented with spans, metrics, and alerts. LangGraph supplies the
graph runtime; the observability layer is framework-free and would wrap
any executor the same way.

Nodes update an AgentState; the graph retries the tool step at most
`max_tool_steps` times (loop guard) before degrading gracefully.
"""

from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from .alerts import AlertEngine
from .metrics import MetricsRegistry
from .tracing import Tracer

LLM = Callable[[str, str], str]  # (instruction, input) -> text


class AgentState(TypedDict, total=False):
    prompt: str
    tool_steps: int
    draft: str
    answer: str
    error: str


def build_agent_graph(llm: LLM, tools: dict[str, Callable[[str], str]] | None = None,
                      tracer: Tracer | None = None,
                      metrics: MetricsRegistry | None = None,
                      alert_engine: AlertEngine | None = None,
                      max_tool_steps: int = 2):
    """Return a compiled LangGraph StateGraph wrapped in observability.

    Falls back to a pure-Python executor when LangGraph is unavailable,
    so the observability layer runs anywhere (that fallback is also what
    the unit tests exercise)."""
    tracer = tracer or Tracer()
    metrics = metrics or MetricsRegistry()
    alert_engine = alert_engine or AlertEngine()
    tools = tools or {}

    def plan(state: AgentState) -> AgentState:
        with tracer.span("plan", prompt=state["prompt"][:80]):
            t0 = time.time()
            try:
                draft = llm("Draft a short answer to the user's question.",
                            state["prompt"])
                metrics.observe("plan", (time.time() - t0) * 1000, "ok")
                return {**state, "draft": draft, "tool_steps": 0}
            except Exception as exc:
                metrics.observe("plan", (time.time() - t0) * 1000, "error")
                return {**state, "error": f"plan failed: {exc}"}

    def use_tool(state: AgentState) -> AgentState:
        step = f"tool:{list(tools)[0]}" if tools else "tool:none"
        with tracer.span(step):
            t0 = time.time()
            try:
                if not tools or state.get("tool_steps", 0) >= max_tool_steps:
                    metrics.observe(step, (time.time() - t0) * 1000, "ok")
                    return state  # nothing to add; finalize
                name = next(iter(tools))
                extra = tools[name](state.get("draft", state["prompt"]))
                out = {**state, "draft": f"{state.get('draft', '')}\n{extra}",
                       "tool_steps": state.get("tool_steps", 0) + 1}
                metrics.observe(step, (time.time() - t0) * 1000, "ok")
                return out
            except Exception as exc:
                metrics.observe(step, (time.time() - t0) * 1000, "error")
                return {**state, "error": f"tool failed: {exc}"}

    def finalize(state: AgentState) -> AgentState:
        with tracer.span("finalize"):
            t0 = time.time()
            if state.get("error"):
                metrics.observe("finalize", (time.time() - t0) * 1000, "error")
                return {**state, "answer": "", "error": state["error"]}
            answer = llm("Polish this draft into a final 2-sentence answer.",
                         state.get("draft", state["prompt"]))
            metrics.observe("finalize", (time.time() - t0) * 1000, "ok",
                            tokens_in=150, tokens_out=80)
            return {**state, "answer": answer}

    def needs_tool(state: AgentState) -> str:
        if state.get("error"):
            return "finalize"
        if not tools:
            return "finalize"
        if state.get("tool_steps", 0) >= 1:  # one tool pass is enough here
            return "finalize"
        return "tool"

    # ---- LangGraph wiring (with stdlib fallback) ----------------------
    executor: Callable[[str], dict[str, Any]]
    try:
        from langgraph.graph import END, StateGraph

        g = StateGraph(AgentState)
        g.add_node("plan", plan)
        g.add_node("tool", use_tool)
        g.add_node("finalize", finalize)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", needs_tool,
                                {"tool": "tool", "finalize": "finalize"})
        g.add_edge("tool", "finalize")
        g.add_edge("finalize", END)
        compiled = g.compile()

        def executor(prompt: str) -> dict[str, Any]:
            alerts = alert_engine.evaluate(metrics)
            for a in alerts:
                tracer.span  # noop touch; alerts land in fired list
            result = compiled.invoke({"prompt": prompt})
            return dict(result)
    except ImportError:
        def executor(prompt: str) -> dict[str, Any]:
            state: AgentState = {"prompt": prompt}
            state = plan(state)
            if needs_tool(state) == "tool":
                state = use_tool(state)
            return dict(finalize(state))

    executor.metrics = metrics  # type: ignore[attr-defined]
    executor.tracer = tracer  # type: ignore[attr-defined]
    executor.alerts = alert_engine  # type: ignore[attr-defined]
    return executor
