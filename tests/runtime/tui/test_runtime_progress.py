from __future__ import annotations

import unittest
from unittest.mock import patch

from spice.runtime.tui.surfaces.progress import TUIStatusFlow


class RuntimeProgressTests(unittest.TestCase):
    def test_status_flow_updates_explicit_runtime_state(self) -> None:
        flow = TUIStatusFlow(console=None, title="SPICE STATUS", label="Starting...")

        with flow as active:
            active.update("Thinking through the decision...", "model configured")

        self.assertEqual(flow.status, "finished")
        self.assertEqual(flow.label, "Ready.")
        self.assertEqual(flow.detail, "")

    def test_status_flow_does_not_overwrite_explicit_finish(self) -> None:
        flow = TUIStatusFlow(console=None, title="SPICE STATUS")

        with flow as active:
            active.finish("Decision ready.", "brief composed")

        self.assertEqual(flow.status, "finished")
        self.assertEqual(flow.label, "Decision ready.")
        self.assertEqual(flow.detail, "brief composed")

    def test_status_flow_marks_exception_as_failed(self) -> None:
        flow = TUIStatusFlow(console=None, title="SPICE STATUS")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with flow:
                raise RuntimeError("boom")

        self.assertEqual(flow.status, "failed")
        self.assertEqual(flow.label, "Failed.")
        self.assertEqual(flow.detail, "boom")

    def test_status_flow_falls_back_without_rich(self) -> None:
        flow = TUIStatusFlow(console=None, title="SPICE STATUS", label="Reading state...")

        def blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "rich" or name.startswith("rich."):
                raise ImportError("blocked")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            rendered = flow._render()

        self.assertEqual(rendered, "Reading state...")


if __name__ == "__main__":
    unittest.main()
