from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from spice.runtime.tui.surfaces.stream import TUIStreamWriter


class RuntimeTUIStreamWriterTests(unittest.TestCase):
    def test_plain_stream_writer_prints_blocks_and_records_events(self) -> None:
        output = io.StringIO()
        writer = TUIStreamWriter(console=None, output_stream=output)

        writer.start()
        writer.status("Thinking through the decision...", "deterministic runtime")
        writer.write_block("I would start with A.")
        writer.write("Use /details for the audit trail.")
        done = writer.finish()

        self.assertEqual(output.getvalue(), "I would start with A.\nUse /details for the audit trail.")
        self.assertEqual(
            [event.event_type for event in writer.events],
            ["status", "response_delta", "response_delta", "response_done"],
        )
        self.assertEqual(writer.events[0].metadata["label"], "Thinking through the decision...")
        self.assertEqual(writer.events[1].metadata["unit"], "block")
        self.assertEqual(writer.events[2].metadata["unit"], "text")
        self.assertEqual(done.metadata["chunk_count"], 2)

    def test_plain_stream_writer_does_not_require_rich_console(self) -> None:
        output = io.StringIO()
        writer = TUIStreamWriter(console=None, output_stream=output)

        writer.start()
        writer.status("Reading the active decision...")
        writer.write("Plain response.")
        done = writer.finish()

        self.assertEqual(output.getvalue(), "Plain response.")
        self.assertEqual(done.event_type, "response_done")
        self.assertEqual(done.metadata["status"], "finished")
        self.assertEqual([event.event_type for event in writer.events], ["status", "response_delta", "response_done"])

    def test_status_flow_stops_before_response_output(self) -> None:
        status_events: list[tuple[str, str, str]] = []

        class FakeStatusFlow:
            def __init__(self, *, console: object, title: str, label: str, detail: str = "") -> None:
                status_events.append(("init", label, detail))

            def __enter__(self) -> "FakeStatusFlow":
                status_events.append(("enter", "", ""))
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                status_events.append(("exit", "", ""))

            def update(self, label: str, detail: str = "") -> None:
                status_events.append(("update", label, detail))

            def finish(self, label: str = "Ready.", detail: str = "") -> None:
                status_events.append(("finish", label, detail))

            def fail(self, label: str = "Failed.", detail: str = "") -> None:
                status_events.append(("fail", label, detail))

        class FakeConsole:
            def __init__(self) -> None:
                self.output = io.StringIO()

            def print(self, text: str, *, end: str = "\n", **_: object) -> None:
                self.output.write(f"{text}{end}")

        console = FakeConsole()
        writer = TUIStreamWriter(console=console)

        with patch("spice.runtime.tui.surfaces.stream.TUIStatusFlow", FakeStatusFlow):
            writer.status("Thinking...", "model")
            writer.status("Composing...", "model")
            writer.write_block("Response body.")

        self.assertEqual(console.output.getvalue(), "Response body.\n")
        self.assertEqual(
            status_events,
            [
                ("init", "Thinking...", "model"),
                ("enter", "", ""),
                ("update", "Composing...", "model"),
                ("finish", "Ready.", ""),
                ("exit", "", ""),
            ],
        )

    def test_fail_closes_status_and_prints_fallback(self) -> None:
        status_events: list[str] = []

        class FakeStatusFlow:
            def __init__(self, *, console: object, title: str, label: str, detail: str = "") -> None:
                status_events.append("init")

            def __enter__(self) -> "FakeStatusFlow":
                status_events.append("enter")
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                status_events.append("exit")

            def update(self, label: str, detail: str = "") -> None:
                status_events.append("update")

            def finish(self, label: str = "Ready.", detail: str = "") -> None:
                status_events.append("finish")

            def fail(self, label: str = "Failed.", detail: str = "") -> None:
                status_events.append("fail")

        class FakeConsole:
            def __init__(self) -> None:
                self.output = io.StringIO()

            def print(self, text: str, *, end: str = "\n", **_: object) -> None:
                self.output.write(f"{text}{end}")

        console = FakeConsole()
        writer = TUIStreamWriter(console=console)

        with patch("spice.runtime.tui.surfaces.stream.TUIStatusFlow", FakeStatusFlow):
            writer.status("Working...")
            error = writer.fail("Fallback response.")

        self.assertEqual(console.output.getvalue(), "Fallback response.\n")
        self.assertEqual(status_events, ["init", "enter", "fail", "exit"])
        self.assertEqual(error.event_type, "error")
        self.assertIn("error", [event.event_type for event in writer.events])
        self.assertEqual(writer.events[-1].event_type, "response_done")
        self.assertEqual(writer.events[-1].metadata["status"], "failed")


if __name__ == "__main__":
    unittest.main()
