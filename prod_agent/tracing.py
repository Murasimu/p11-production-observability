"""Tracing: hierarchical spans for every agent step.

Framework-free span tree (OpenTelemetry-style naming). Spans record
name, duration, status, and attributes; they nest via context, serialize
to JSONL, and an optional LangSmith adapter forwards completed root
spans (kept behind an env flag so tests never ship data anywhere).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start: float = 0.0
    end: float | None = None
    status: str = "in_progress"  # in_progress | ok | error
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_id": self.parent_id, "name": self.name,
            "start": self.start, "end": self.end,
            "duration_ms": round(self.duration_ms, 2), "status": self.status,
            "attributes": self.attributes,
        }


class Tracer:
    """Context-managed span tree with JSONL export + optional LangSmith."""

    def __init__(self, export_path: str | None = None,
                 langsmith_api_key: str | None = None) -> None:
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.export_path = export_path
        # only forward to LangSmith when explicitly configured
        self.langsmith_key = langsmith_api_key or os.environ.get("LANGSMITH_API_KEY")

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        s = Span(
            name=name,
            parent_id=parent.span_id if parent else None,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:16],
            start=time.time(),
            attributes=attributes,
        )
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
            s.status = "ok"
        except Exception as exc:
            s.status = "error"
            s.attributes["error"] = str(exc)[:300]
            raise
        finally:
            s.end = time.time()
            self._stack.pop()
            if not self._stack:  # root span closed -> maybe export
                self._maybe_export(s)

    def _maybe_export(self, root: Span) -> None:
        if self.export_path:
            with open(self.export_path, "a", encoding="utf-8") as fh:
                for s in self.spans:
                    fh.write(json.dumps(s.to_dict()) + "\n")
        if self.langsmith_key:
            self._export_langsmith(root)

    def _export_langsmith(self, root: Span) -> None:
        """Forward the root span summary to LangSmith runs API.

        Intentionally thin: production setups use the official SDK's
        RunTree; this adapter shows the integration point without making
        tests depend on network or credentials.
        """
        try:
            import urllib.request
            import urllib.error

            body = json.dumps({
                "name": root.name,
                "run_type": "chain",
                "inputs": {"prompt": root.attributes.get("prompt", "")},
                "outputs": {"status": root.status},
                "extra": {"duration_ms": root.duration_ms},
            }).encode()
            req = urllib.request.Request(
                "https://api.smith.langchain.com/runs/create",
                data=body,
                headers={"x-api-key": self.langsmith_key,
                         "Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            # observability must never take the agent down
            pass

    def trace_of(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    def dump(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]
