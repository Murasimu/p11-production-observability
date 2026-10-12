"""Canary release: route a share of traffic to a candidate graph,
compare metrics against the baseline, auto-rollback on regression.

The comparison uses the same MetricsRegistry the dashboards read, so
"canary is worse" is a measured judgment (error rate delta, latency
delta), not a guess.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .metrics import MetricsRegistry

GraphFn = Callable[[str], str]  # prompt -> answer


@dataclass
class CanaryRelease:
    baseline: GraphFn
    candidate: GraphFn
    traffic_share: float = 0.1       # fraction routed to the candidate
    min_samples: int = 10
    max_error_rate_delta: float = 0.05
    max_latency_ratio: float = 1.5

    baseline_metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    candidate_metrics: MetricsRegistry = field(default_factory=MetricsRegistry)
    state: str = "running"  # running | promoted | rolled_back
    decision_reason: str = ""

    def route(self, prompt: str) -> str:
        """Send traffic, observe both sides (candidate errors recorded even
        when it fails)."""
        if self.state != "running":
            return self._answer(self.baseline, self.baseline_metrics, prompt)

        if random.random() < self.traffic_share:
            return self._answer(self.candidate, self.candidate_metrics, prompt)
        return self._answer(self.baseline, self.baseline_metrics, prompt)

    def _answer(self, fn: GraphFn, metrics: MetricsRegistry, prompt: str) -> str:
        import time
        t0 = time.time()
        try:
            out = fn(prompt)
            metrics.observe("answer", (time.time() - t0) * 1000, "ok")
            return out
        except Exception:
            metrics.observe("answer", (time.time() - t0) * 1000, "error")
            raise

    # ------------------------------------------------------------------
    def evaluate(self) -> str:
        """Compare candidate vs baseline; promote, keep running, or rollback."""
        c_calls = self.candidate_metrics._calls.get("answer", 0)
        if c_calls < self.min_samples:
            self.decision_reason = f"insufficient samples ({c_calls}/{self.min_samples})"
            return self.state

        err_delta = (self.candidate_metrics.error_rate("answer")
                     - self.baseline_metrics.error_rate("answer"))
        lat_ratio = 1.0
        b_p95 = self.baseline_metrics.p_latency("answer", 0.95)
        c_p95 = self.candidate_metrics.p_latency("answer", 0.95)
        if b_p95 > 0:
            lat_ratio = c_p95 / b_p95

        if err_delta > self.max_error_rate_delta:
            self.state = "rolled_back"
            self.decision_reason = (f"candidate error rate {err_delta:+.0%} "
                                    f"vs baseline (allowed {self.max_error_rate_delta:+.0%})")
        elif lat_ratio > self.max_latency_ratio:
            self.state = "rolled_back"
            self.decision_reason = (f"candidate p95 latency {lat_ratio:.1f}x "
                                    "of baseline")
        elif c_calls >= self.min_samples and err_delta <= 0 and lat_ratio <= 1.1:
            self.state = "promoted"
            self.decision_reason = (f"candidate healthy: err {err_delta:+.0%}, "
                                    f"latency {lat_ratio:.2f}x")
        else:
            self.decision_reason = (f"ok so far: err {err_delta:+.0%}, "
                                    f"latency {lat_ratio:.2f}x; collecting samples")
        return self.state

    def report(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.decision_reason,
            "baseline": self.baseline_metrics.dashboard()["steps"].get("answer"),
            "candidate": self.candidate_metrics.dashboard()["steps"].get("answer"),
        }
