"""Metrics: latency, cost, tokens, and per-step counters -- the numbers
the dashboard and alerts read."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens (demo pricing; see P7 for a fuller treatment)
PRICE_IN, PRICE_OUT = 0.80, 3.20


@dataclass
class MetricsRegistry:
    _latencies_ms: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list))
    _errors: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _tokens: dict[str, tuple[int, int]] = field(default_factory=dict)  # step -> (in, out)
    window_seconds: float = 300.0
    _event_log: list[tuple[float, str, str]] = field(default_factory=list)  # (ts, step, status)

    def observe(self, step: str, latency_ms: float, status: str = "ok",
                tokens_in: int = 0, tokens_out: int = 0) -> None:
        now = time.time()
        self._latencies_ms[step].append(latency_ms)
        self._event_log.append((now, step, status))
        self._calls[step] += 1
        if status != "ok":
            self._errors[step] += 1
        if tokens_in or tokens_out:
            i, o = self._tokens.get(step, (0, 0))
            self._tokens[step] = (i + tokens_in, o + tokens_out)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        if len(self._event_log) > 10_000:
            self._event_log = [e for e in self._event_log if e[0] >= cutoff]

    # ------------------------------------------------------------------
    def error_rate(self, step: str | None = None) -> float:
        """Errors / calls within the window (all steps, or one)."""
        cutoff = time.time() - self.window_seconds
        events = [e for e in self._event_log if e[0] >= cutoff]
        if step:
            events = [e for e in events if e[1] == step]
        if not events:
            return 0.0
        errs = sum(1 for e in events if e[2] != "ok")
        return errs / len(events)

    def p_latency(self, step: str, q: float = 0.95) -> float:
        vals = self._latencies_ms.get(step) or []
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = min(len(vals) - 1, int(q * len(vals)))
        return vals[idx]

    def cost_usd(self) -> float:
        total_in = sum(i for i, _ in self._tokens.values())
        total_out = sum(o for _, o in self._tokens.values())
        return total_in / 1e6 * PRICE_IN + total_out / 1e6 * PRICE_OUT

    def calls_per_minute(self, step: str | None = None) -> float:
        cutoff = time.time() - 60
        events = [e for e in self._event_log if e[0] >= cutoff]
        if step:
            events = [e for e in events if e[1] == step]
        return len(events)  # per rolling minute

    def dashboard(self) -> dict[str, Any]:
        steps = sorted(self._calls)
        return {
            "steps": {
                s: {
                    "calls": self._calls[s],
                    "errors": self._errors.get(s, 0),
                    "error_rate": round(self.error_rate(s), 3),
                    "p50_ms": round(self.p_latency(s, 0.5), 1),
                    "p95_ms": round(self.p_latency(s, 0.95), 1),
                }
                for s in steps
            },
            "total_cost_usd": round(self.cost_usd(), 6),
            "requests_per_minute": self.calls_per_minute(),
            "window_seconds": self.window_seconds,
        }
