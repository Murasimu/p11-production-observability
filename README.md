# Production Agent with Observability

**Project 11 / 12 — Agentic AI portfolio.**
*Shows: you can ship to production, not just localhost.*

A LangGraph-based agent wrapped in a production observability stack:
hierarchical tracing with JSONL export (and an optional LangSmith
adapter), latency/cost/token dashboards, rule-based alerting on loops
and failure rates, and canary releases with measured auto-rollback.

## Why this matters

The gap between a demo and production is not the agent logic -- it's
everything around it. When (not if) the model API degrades, a tool
loops, or a new version regresses, you need:

1. **Tracing** — hierarchical spans (`plan` → `tool:lookup` → `finalize`)
   with status, duration, and attributes, exported to JSONL. A
   LangSmith adapter forwards root spans when `LANGSMITH_API_KEY` is
   set, and never raises (observability must not take the agent down).
2. **Metrics** — per-step p50/p95 latency, error rates, calls/min, and
   token cost in a rolling window; `dashboard()` is the snapshot
   dashboards read.
3. **Alerting** — rules evaluated against those metrics: loop detection
   (a step at >10 calls/min smells like a stuck agent), error rate
   >25%, p95 latency >8s, cost burn >$5. Cooldown dedupes flapping.
4. **Canary + rollback** — route a traffic share to a candidate graph,
   record both sides' metrics, and decide on *measured* deltas
   (error-rate delta, latency ratio): promote, keep sampling, or roll
   back automatically.

## Architecture

```
prompt ──> LangGraph StateGraph ──┬─ plan node ──── llm draft
                  │               ├─ tool node ──── kb_lookup (bounded passes)
                  │               └─ finalize ───── polished answer
                  │
        every node wrapped in Tracer.span() and MetricsRegistry.observe()
                  │
        AlertEngine.evaluate(metrics) ──> fired alerts (deduped by cooldown)
                  │
   releases: CanaryRelease(baseline, candidate, share)
        route() observes BOTH sides ──> evaluate()
        err_delta > 5% or latency > 1.5x ──> rolled_back (automatic)
        healthy + enough samples ──> promoted
```

- `prod_agent/tracing.py` — span tree, JSONL export, LangSmith adapter
- `prod_agent/metrics.py` — latency/cost/token registry, dashboard
- `prod_agent/alerts.py` — rules + cooldown dedupe
- `prod_agent/canary.py` — traffic splitting, comparison, auto-rollback
- `prod_agent/graph.py` — the LangGraph agent (stdlib fallback so the
  observability layer runs even without LangGraph installed)

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # full stack offline: traces, dashboard, alerts, canary
python demo.py --live     # real LLM behind the graph
```

Optional: set `LANGSMITH_API_KEY` to forward root spans to LangSmith.

## Tests

```bash
pytest -q
```

25 tests cover: nested spans (parent/trace ids, status, error
attributes, duration), JSONL export, LangSmith export never raising
under network failure; metrics (dashboard shape, per-step error rates,
p50/p95 math, token cost, empty registry); alerts (error-rate rule,
loop detection on simulated call storms, latency rule, cooldown dedupe,
healthy silence); canary (traffic-share routing, rollback on error-rate
delta with reason, promotion on healthy candidates, insufficient
samples, report shape); graph end-to-end (tool inclusion, metrics
population, LLM failure surfacing as error metric, trace recording).

## Design decisions

- **Observability must never take the agent down** — the LangSmith
  adapter swallows network failures; a dead telemetry endpoint degrades
  to local-only tracing, never an exception in the request path.
- **Framework-free core, framework-ready wiring** — spans/metrics/alerts
  are plain Python; LangGraph is the runtime in `graph.py` with a
  stdlib fallback, so the stack observes *any* executor.
- **Canary decisions are measured** — promote/rollback hinge on
  error-rate delta and latency ratio with explicit thresholds; "the
  canary felt worse" is not a deploy gate.
- **Loop detection is calls/min, not vibes** — stuck agents hammer one
  node; that signature is directly observable and alertable.
- **Rolling windows** — metrics age out (default 5 min) so dashboards
  and alerts reflect *current* state, not all-time history.

## Portfolio context

Project 11 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
