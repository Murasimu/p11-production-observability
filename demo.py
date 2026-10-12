"""Demo: a productionized agent with the full observability stack.

Shows:
1. Tracing: nested spans per request, JSONL export
2. Metrics: latency/error/cost dashboard after live traffic
3. Alerting: a traffic spike trips loop detection; errors trip the error rule
4. Canary: a buggy candidate is auto-rolled-back by measured regression

Usage:
  python demo.py            # fully offline, deterministic-ish
  python demo.py --live     # real LLM behind the graph (tracing on)
"""

from __future__ import annotations

import random
import sys
import tempfile
import os

from prod_agent import (
    AlertEngine, CanaryRelease, MetricsRegistry, Tracer, build_agent_graph,
)

random.seed(42)


def banner(t):
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def main():
    trace_path = os.path.join(tempfile.mkdtemp(), "traces.jsonl")
    tracer = Tracer(export_path=trace_path)
    metrics = MetricsRegistry()
    alerts = AlertEngine()

    # -- the agent (LangGraph when installed, stdlib fallback otherwise)
    if "--live" in sys.argv:
        from prod_agent.llm import make_llm
        try:
            live = make_llm()
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        llm = lambda instruction, text: live(
            [{"role": "system", "content": instruction},
             {"role": "user", "content": text}])
    else:
        def llm(instruction, text):
            return f"[{instruction[:12]}] response to: {text[:40]}"

    tools = {"kb_lookup": lambda draft: "(kb) Aurora Robotics policy: 25 vacation days"}
    agent = build_agent_graph(llm, tools, tracer=tracer, metrics=metrics,
                              alert_engine=alerts)

    banner("1) Normal traffic -> traces + metrics")
    for q in ["What is the vacation policy?", "How do I file expenses?",
              "Who do I call for a security incident?"]:
        r = agent(q)
        print(f"  q: {q[:45]:47} -> answer[:40]={r['answer'][:40]!r}")
    print(f"\n  spans recorded: {len(tracer.spans)} "
          f"(names: {sorted({s.name for s in tracer.spans})})")
    print(f"  exported to: {trace_path}")

    banner("2) Dashboard")
    for step, d in metrics.dashboard()["steps"].items():
        print(f"  {step:12} calls={d['calls']} p50={d['p50_ms']}ms "
              f"p95={d['p95_ms']}ms err={d['error_rate']}")
    print(f"  cost: ${metrics.dashboard()['total_cost_usd']}")

    banner("3) Alerting: loop + error rules trip on bad traffic")
    now = __import__("time").time()
    for _ in range(25):  # a stuck loop hammering one tool
        metrics.observe("tool:kb_lookup", 5, "ok")
    for _ in range(8):   # a failing deploy
        metrics.observe("plan", 10, "error")
    fired = alerts.evaluate(metrics)
    for a in fired:
        print(f"  [{a.severity.value:8}] {a.rule}: {a.message}")

    banner("4) Canary: buggy candidate -> measured auto-rollback")
    def baseline(p):
        return "stable answer"
    state = {"calls": 0}
    def flaky_candidate(p):
        state["calls"] += 1
        if state["calls"] % 2 == 0:
            raise RuntimeError("candidate regression: tool binding error")
        return "candidate answer"
    canary = CanaryRelease(baseline, flaky_candidate, traffic_share=1.0,
                           min_samples=5)
    for _ in range(10):
        try:
            canary.route("q")
        except RuntimeError:
            pass  # errors are recorded even when the request fails
    for _ in range(10):
        canary.baseline_metrics.observe("answer", 10, "ok")
    decision = canary.evaluate()
    print(f"  canary decision: {decision}")
    print(f"  reason: {canary.decision_reason}")
    print(f"  after rollback, routes go to baseline only: "
          f"{canary.route('q') == 'stable answer'}")

    print("\ndone.")


if __name__ == "__main__":
    main()
