"""Production agent with observability.

A LangGraph-based agent deployed behind production guardrails:
- OpenTelemetry-style tracing spans (and optional LangSmith export)
- latency/cost metrics with a live dashboard
- alerting on loop patterns and failure rates
- canary releases: shadow a candidate graph, compare metrics, auto-rollback

Modules:
- tracing.py    span tree, JSONL export, LangSmith adapter (optional)
- metrics.py    latency/cost/token counters, dashboard snapshot
- alerts.py     rule engine: loop detection, error-rate, latency, cost
- canary.py     canary deploy + comparison + automatic rollback
- graph.py      the LangGraph agent under observation
"""

from .alerts import Alert, AlertEngine, Rule
from .canary import CanaryRelease
from .graph import build_agent_graph, AgentState
from .metrics import MetricsRegistry
from .tracing import Span, Tracer

__all__ = [
    "Alert", "AlertEngine", "Rule",
    "CanaryRelease",
    "build_agent_graph", "AgentState",
    "MetricsRegistry",
    "Span", "Tracer",
]
__version__ = "1.0.0"
