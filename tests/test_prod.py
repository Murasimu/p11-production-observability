"""Tests for the production observability stack. Offline, deterministic."""

import json
import time

import pytest

from prod_agent import (
    AlertEngine, CanaryRelease, MetricsRegistry, Rule, Tracer,
    build_agent_graph,
)
from prod_agent.alerts import Severity, default_rules


# ---------------------------------------------------------------- tracing
class TestTracer:
    def test_nested_spans(self):
        t = Tracer()
        with t.span("root", prompt="hi"):
            with t.span("child"):
                pass
        assert len(t.spans) == 2
        child = t.spans[1]
        assert child.parent_id == t.spans[0].span_id
        assert child.trace_id == t.spans[0].trace_id
        assert all(s.status == "ok" for s in t.spans)

    def test_error_span_records_exception(self):
        t = Tracer()
        with pytest.raises(ValueError):
            with t.span("boom"):
                raise ValueError("kaput")
        s = t.spans[0]
        assert s.status == "error" and "kaput" in s.attributes["error"]

    def test_duration_measured(self):
        t = Tracer()
        with t.span("slow"):
            time.sleep(0.02)
        assert t.spans[0].duration_ms >= 15

    def test_jsonl_export(self, tmp_path):
        p = str(tmp_path / "trace.jsonl")
        t = Tracer(export_path=p)
        with t.span("root"):
            with t.span("child"):
                pass
        lines = open(p).read().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "root"

    def test_langsmith_export_never_raises(self, monkeypatch):
        # simulate total network failure
        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
        t = Tracer(langsmith_api_key="fake-key-for-test")
        with t.span("root"):  # must not raise despite the broken network
            pass


# ---------------------------------------------------------------- metrics
class TestMetrics:
    def test_observe_and_dashboard(self):
        m = MetricsRegistry()
        m.observe("plan", 120, "ok")
        m.observe("plan", 140, "ok")
        m.observe("plan", 900, "error")
        d = m.dashboard()
        step = d["steps"]["plan"]
        assert step["calls"] == 3 and step["errors"] == 1
        assert step["error_rate"] == pytest.approx(1 / 3, abs=0.001)

    def test_error_rate_scoped_to_step(self):
        m = MetricsRegistry()
        m.observe("good", 10, "ok")
        m.observe("bad", 10, "error")
        m.observe("bad", 10, "error")
        assert m.error_rate("good") == 0.0
        assert m.error_rate("bad") == 1.0

    def test_p95_latency(self):
        m = MetricsRegistry()
        for i in range(100):
            m.observe("x", float(i))
        assert m.p_latency("x", 0.5) == pytest.approx(50, abs=1)
        assert m.p_latency("x", 0.95) >= 90

    def test_cost_from_tokens(self):
        m = MetricsRegistry()
        m.observe("finalize", 5, "ok", tokens_in=1_000_000, tokens_out=0)
        assert m.cost_usd() == pytest.approx(0.80)

    def test_empty_metrics(self):
        d = MetricsRegistry().dashboard()
        assert d["steps"] == {} and d["total_cost_usd"] == 0


# ---------------------------------------------------------------- alerts
class TestAlerts:
    def test_error_rate_rule_fires(self):
        m = MetricsRegistry()
        for _ in range(6):
            m.observe("plan", 10, "error")
        for _ in range(2):
            m.observe("plan", 10, "ok")
        engine = AlertEngine([Rule("err", lambda mm: "too many errors"
                                   if mm.error_rate() > 0.25 else None,
                                   Severity.CRITICAL)])
        fired = engine.evaluate(m)
        assert len(fired) == 1 and fired[0].rule == "err"
        assert fired[0].severity == Severity.CRITICAL

    def test_loop_detection_fires_on_high_cpm(self):
        m = MetricsRegistry()
        now = time.time()
        # simulate 15 calls within the last minute
        for _ in range(15):
            m._event_log.append((now, "tool:lookup", "ok"))
            m._calls["tool:lookup"] += 1
        engine = AlertEngine(default_rules())
        fired = engine.evaluate(m)
        assert any(a.rule == "loop_detection" for a in fired)

    def test_latency_rule(self):
        m = MetricsRegistry()
        for _ in range(10):
            m.observe("slow_step", 9000, "ok")
        engine = AlertEngine(default_rules())
        fired = engine.evaluate(m)
        assert any(a.rule == "latency_p95" for a in fired)

    def test_cooldown_dedupes(self):
        m = MetricsRegistry()
        for _ in range(10):
            m.observe("x", 10, "error")
        rule = Rule("err", lambda mm: "bad" if mm.error_rate() > 0.25 else None)
        engine = AlertEngine([rule], cooldown_seconds=60)
        assert len(engine.evaluate(m)) == 1
        assert engine.evaluate(m) == []  # within cooldown -> deduped

    def test_healthy_metrics_no_alerts(self):
        m = MetricsRegistry()
        m.observe("plan", 100, "ok")
        assert AlertEngine(default_rules()).evaluate(m) == []


# ---------------------------------------------------------------- canary
class TestCanary:
    def test_routing_respects_state(self):
        baseline = lambda p: f"base:{p}"
        candidate = lambda p: f"cand:{p}"
        c = CanaryRelease(baseline, candidate, traffic_share=0.0)
        assert c.route("q").startswith("base:")
        c.state = "rolled_back"
        # even with share=1.0, rolled back routes everything to baseline
        c.traffic_share = 1.0
        assert c.route("q").startswith("base:")

    def test_rollback_on_error_rate(self):
        def good(p):
            return "ok"
        def bad(p):
            raise RuntimeError("candidate bug")
        c = CanaryRelease(good, bad, traffic_share=1.0, min_samples=3)
        for _ in range(5):
            try:
                c.route("q")
            except RuntimeError:
                pass
        for _ in range(5):
            c.baseline_metrics.observe("answer", 10, "ok")
        assert c.evaluate() == "rolled_back"
        assert "error rate" in c.decision_reason

    def test_promote_on_healthy_candidate(self):
        c = CanaryRelease(lambda p: "b", lambda p: "c",
                          traffic_share=1.0, min_samples=5)
        for _ in range(6):
            c.route("q")
        for _ in range(6):
            c.baseline_metrics.observe("answer", 10, "ok")
        assert c.evaluate() == "promoted"

    def test_insufficient_samples_keeps_running(self):
        c = CanaryRelease(lambda p: "b", lambda p: "c", min_samples=10)
        c.route("q")
        assert c.evaluate() == "running"
        assert "insufficient" in c.decision_reason

    def test_report_shape(self):
        c = CanaryRelease(lambda p: "b", lambda p: "c")
        r = c.report()
        assert set(r) == {"state", "reason", "baseline", "candidate"}


# ---------------------------------------------------------------- graph
class TestGraph:
    def test_graph_end_to_end(self):
        calls = []
        def llm(instruction, text):
            calls.append(instruction[:10])
            return f"answer about {text[:20]}"
        tools = {"lookup": lambda d: "extra facts"}
        agent = build_agent_graph(llm, tools)
        result = agent("What is quantum tunneling?")
        assert "answer about" in result["answer"]
        assert calls  # llm was invoked

    def test_tool_step_included(self):
        seen = {}
        def llm(i, t):
            return "draft"
        tools = {"lookup": lambda d: seen.update({"called": True}) or "facts"}
        build_agent_graph(llm, tools)("q")
        assert seen.get("called") is True

    def test_metrics_populated(self):
        agent = build_agent_graph(lambda i, t: "x", {"t": lambda d: "y"})
        agent("q1")
        agent("q2")
        d = agent.metrics.dashboard()
        assert "plan" in d["steps"] and "finalize" in d["steps"]
        assert d["steps"]["finalize"]["calls"] == 2

    def test_llm_failure_surfaces_error(self):
        def boom(i, t):
            raise RuntimeError("model down")
        agent = build_agent_graph(boom)
        result = agent("q")
        assert "model down" in result.get("error", "")
        assert agent.metrics.error_rate("plan") == 1.0

    def test_traces_recorded(self):
        agent = build_agent_graph(lambda i, t: "x")
        agent("hello")
        names = [s.name for s in agent.tracer.spans]
        assert "plan" in names and "finalize" in names
