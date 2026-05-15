from __future__ import annotations

import sys
from typing import Any, TextIO

from spice.runtime.streaming import (
    SpiceStreamEvent,
    stream_error_event,
    stream_response_delta_event,
    stream_response_done_event,
    stream_status_event,
)
from spice.runtime.tui.surfaces.progress import TUIStatusFlow


class TUIStreamWriter:
    """Render Spice stream events without changing runtime facts."""

    def __init__(
        self,
        *,
        console: Any = None,
        output_stream: TextIO | None = None,
        status_title: str = "SPICE",
        use_status: bool = True,
    ) -> None:
        self.console = console
        self.output_stream = output_stream
        self.status_title = status_title
        self.use_status = use_status
        self.events: list[SpiceStreamEvent] = []
        self.started = False
        self.finished = False
        self.failed = False
        self._status_flow: TUIStatusFlow | None = None
        self._response_parts: list[str] = []
        self._chunk_count = 0

    def __enter__(self) -> "TUIStreamWriter":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.fail(str(exc or "Streaming failed."))
        elif not self.finished:
            self.finish()

    def start(self) -> "TUIStreamWriter":
        self.started = True
        return self

    def status(self, label: str, detail: str = "") -> SpiceStreamEvent:
        self.start()
        event = stream_status_event(label, detail=detail)
        self.events.append(event)
        if self.use_status and self.console is not None:
            if self._status_flow is None:
                self._status_flow = TUIStatusFlow(
                    console=self.console,
                    title=self.status_title,
                    label=label,
                    detail=detail,
                )
                self._status_flow.__enter__()
            else:
                self._status_flow.update(label, detail)
        return event

    def write(self, text: str) -> SpiceStreamEvent:
        self.start()
        self._close_status()
        normalized = str(text or "")
        event = stream_response_delta_event(
            normalized,
            unit="text",
            index=self._chunk_count,
        )
        self.events.append(event)
        self._response_parts.append(normalized)
        self._chunk_count += 1
        self._emit(normalized, end="")
        return event

    def write_block(self, text: str) -> SpiceStreamEvent:
        self.start()
        self._close_status()
        normalized = str(text or "").rstrip()
        event = stream_response_delta_event(
            normalized,
            unit="block",
            index=self._chunk_count,
        )
        self.events.append(event)
        self._response_parts.append(f"{normalized}\n")
        self._chunk_count += 1
        self._emit(normalized, end="\n")
        return event

    def finish(self, label: str = "Ready.", detail: str = "") -> SpiceStreamEvent:
        self.start()
        self._close_status(label=label, detail=detail)
        self.finished = True
        event = stream_response_done_event(
            "".join(self._response_parts),
            chunk_count=self._chunk_count,
            status="failed" if self.failed else "finished",
        )
        self.events.append(event)
        self._flush()
        return event

    def fail(self, fallback_text: str = "") -> SpiceStreamEvent:
        self.start()
        self.failed = True
        if self._status_flow is not None:
            try:
                self._status_flow.fail("Failed.")
                self._status_flow.__exit__(None, None, None)
            finally:
                self._status_flow = None
        error = stream_error_event(str(fallback_text or "Streaming failed."))
        self.events.append(error)
        if fallback_text:
            self.write_block(fallback_text)
        self.finish("Failed.")
        return error

    def _close_status(self, label: str = "Ready.", detail: str = "") -> None:
        if self._status_flow is None:
            return
        try:
            self._status_flow.finish(label, detail)
            self._status_flow.__exit__(None, None, None)
        finally:
            self._status_flow = None

    def _emit(self, text: str, *, end: str) -> None:
        if self.console is not None:
            try:
                self.console.print(text, end=end, markup=False, highlight=False)
            except TypeError:
                self.console.print(text, end=end)
            return
        stream = self.output_stream or sys.stdout
        stream.write(f"{text}{end}")
        stream.flush()

    def _flush(self) -> None:
        if self.console is not None:
            file = getattr(self.console, "file", None)
            if file is not None:
                try:
                    file.flush()
                except Exception:
                    pass
            return
        stream = self.output_stream or sys.stdout
        stream.flush()
