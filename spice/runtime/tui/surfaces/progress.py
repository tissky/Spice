from __future__ import annotations

from typing import Any

from spice.runtime.tui.theme import SpiceTheme


class TUIStatusFlow:
    """A truthful loading surface driven by explicit runtime state updates."""

    def __init__(
        self,
        *,
        console: Any,
        title: str,
        label: str = "Working...",
        detail: str = "",
    ) -> None:
        self.console = console
        self.title = title
        self.label = label
        self.detail = detail
        self.status = "running"
        self._live: Any = None

    def __enter__(self) -> "TUIStatusFlow":
        if self.console is None:
            return self
        try:
            from rich.live import Live
        except ImportError:
            return self
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            detail = str(exc) if exc is not None else ""
            self.fail("Failed.", detail)
        elif self.status == "running":
            self.finish("Ready.")
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()

    def update(self, label: str, detail: str = "") -> None:
        self.status = "running"
        self.label = label
        self.detail = detail
        self._refresh()

    def finish(self, label: str = "Ready.", detail: str = "") -> None:
        self.status = "finished"
        self.label = label
        self.detail = detail
        self._refresh()

    def fail(self, label: str = "Failed.", detail: str = "") -> None:
        self.status = "failed"
        self.label = label
        self.detail = detail
        self._refresh()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Any:
        try:
            from rich import box
            from rich.console import Group
            from rich.panel import Panel
            from rich.spinner import Spinner
            from rich.text import Text
        except ImportError:
            return self.label

        if self.status == "running":
            body: Any = Spinner(
                "dots",
                text=self._status_text(),
                style=SpiceTheme.WARNING,
            )
        else:
            icon = "✓" if self.status == "finished" else "x"
            style = SpiceTheme.SUCCESS if self.status == "finished" else SpiceTheme.ERROR
            body = Text(f"{icon} {self._status_text()}", style=style)
        return Panel(
            Group(body),
            title=f"[bold red]{self.title}[/bold red]",
            border_style=SpiceTheme.PANEL_BORDER,
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _status_text(self) -> str:
        label = self.label.strip() or "Working..."
        detail = self.detail.strip()
        return f"{label}  {detail}" if detail else label
