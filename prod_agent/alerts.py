"""Alerting: rules evaluated against metrics -- loop detection, error
rate, latency, cost burn.

Rules return Alert objects with severity and a dedupe key so a flapping
condition fires once, not every evaluation cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .metrics import MetricsRegistry


class Severity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    rule: str
    severity: Severity
    message: str
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Rule:
    name: str
    check: Callable[[MetricsRegistry], str | None]
    """Return an alert message, or None when healthy."""
    severity: Severity = Severity.WARNING


class AlertEngine:
    def __init__(self, rules: list[Rule] | None = None,
                 cooldown_seconds: float = 60.0) -> None:
        self.rules = rules or default_rules()
        self.cooldown = cooldown_seconds
        self._last_fired: dict[str, float] = {}
        self.fired: list[Alert] = []

    def evaluate(self, metrics: MetricsRegistry) -> list[Alert]:
        now = time.time()
        new: list[Alert] = []
        for rule in self.rules:
            msg = rule.check(metrics)
            if msg is None:
                continue
            if now - self._last_fired.get(rule.name, 0) < self.cooldown:
                continue  # dedupe within cooldown
            self._last_fired[rule.name] = now
            alert = Alert(rule=rule.name, severity=rule.severity, message=msg)
            self.fired.append(alert)
            new.append(alert)
        return new


# ---------------------------------------------------------------- rules
def _loop_check(m: MetricsRegistry) -> str | None:
    """A step running >10 calls per minute per user request smells like a
    loop (agent stuck calling the same node)."""
    for step, calls in m._calls.items():
        cpm = m.calls_per_minute(step)
        if cpm > 10:
            return f"possible loop: step '{step}' at {cpm:.0f} calls/min"
    return None


def _error_rate_check(m: MetricsRegistry) -> str | None:
    rate = m.error_rate()
    if rate > 0.25:
        return f"error rate {rate:.0%} exceeds 25%"
    return None


def _latency_check(m: MetricsRegistry) -> str | None:
    for step in m._calls:
        p95 = m.p_latency(step, 0.95)
        if p95 > 8000:
            return f"p95 latency for '{step}' is {p95:.0f}ms (>8s)"
    return None


def _cost_check(m: MetricsRegistry) -> str | None:
    cost = m.cost_usd()
    if cost > 5.0:
        return f"token spend ${cost:.2f} exceeds $5.00 budget"
    return None


def default_rules() -> list[Rule]:
    return [
        Rule("loop_detection", _loop_check, Severity.CRITICAL),
        Rule("error_rate", _error_rate_check, Severity.CRITICAL),
        Rule("latency_p95", _latency_check, Severity.WARNING),
        Rule("cost_burn", _cost_check, Severity.WARNING),
    ]
