from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from spice.llm.core import LLMRequest, LLMResponse
from spice.perception import build_workspace_perception_artifact, workspace_context_from_perception
from spice.runtime import LocalJsonStore, run_once, setup_workspace
from spice.runtime.context_debug import compile_workspace_decision_context_payload
from spice.runtime.context_debug import compile_sources_debug_payload, render_sources_debug_text
from spice.runtime.refine import refine_decision
from spice.runtime.workspace import load_workspace_memory_provider
from spice.runtime.workspace_perception import run_runtime_workspace_perception_step


NOW = datetime(2026, 5, 12, 6, 0, tzinfo=timezone.utc)


class _FakeWorkspaceClient:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        output = self.outputs.pop(0) if self.outputs else {"done": True, "tool_calls": []}
        return LLMResponse(
            provider_id="fake",
            model_id="fake-model",
            output_text=json.dumps(output),
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=1,
            request_id=f"workspace.{len(self.requests)}",
        )


class RuntimeWorkspacePerceptionStepTests(unittest.TestCase):
    def test_runtime_workspace_perception_writes_artifact_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            root = Path(tmp_dir)
            (root / "spice").mkdir(exist_ok=True)
            (root / "spice" / "runtime.py").write_text(
                "class Runtime:\n    workspace_context = {}\n",
                encoding="utf-8",
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            client = _FakeWorkspaceClient(
                [
                    {
                        "summary": "Find runtime workspace context.",
                        "tool_calls": [
                            {
                                "tool": "search",
                                "args": {"pattern": "workspace_context", "file_glob": "*.py"},
                            },
                            {
                                "tool": "read_file",
                                "args": {"path": "spice/runtime.py", "offset": 1, "limit": 3},
                            },
                        ],
                    },
                    {"done": True, "summary": "Runtime context found.", "tool_calls": []},
                ]
            )

            with patch(
                "spice.runtime.workspace_perception.build_candidate_expander_client",
                return_value=client,
            ):
                result = run_runtime_workspace_perception_step(
                    project_root=tmp_dir,
                    query="current workspace context implementation",
                    config={"llm_provider": "fake", "llm_model": "fake-model"},
                    trigger="new_decision",
                    store=store,
                    now=NOW,
                )

            self.assertEqual(result.status, "written")
            self.assertTrue(result.path and result.path.exists())
            self.assertEqual(result.context["source"], "workspace_perception")
            self.assertEqual(result.context["perception_id"], result.artifact["perception_id"])
            self.assertEqual(result.artifact["query"], "current workspace context implementation")
            self.assertEqual(len(result.artifact["tool_calls"]), 2)
            self.assertEqual(result.artifact["files_read"][0]["path"], "spice/runtime.py")
            saved = store.load_perception(result.artifact["perception_id"])
            self.assertEqual(saved["perception_id"], result.artifact["perception_id"])
            self.assertEqual(result.memory_writeback["namespace"], "general.workspace_perception")
            provider = load_workspace_memory_provider(tmp_dir)
            memory_records = provider.query(namespace="general.workspace_perception", limit=-1)
            self.assertEqual(len(memory_records), 1)
            memory_record = memory_records[0]
            self.assertEqual(memory_record["perception_id"], result.artifact["perception_id"])
            self.assertEqual(memory_record["files"]["read"][0]["path"], "spice/runtime.py")
            self.assertNotIn("content_preview", json.dumps(memory_record))
            self.assertNotIn('"text": "class Runtime', json.dumps(memory_record))
            self.assertEqual(memory_record["snippet_refs"][0]["path"], "spice/runtime.py")

    def test_runtime_workspace_perception_uses_depth_config_for_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            root = Path(tmp_dir)
            (root / "README.md").write_text("hello\n", encoding="utf-8")
            client = _FakeWorkspaceClient(
                [
                    {
                        "summary": "Read the repo readme.",
                        "tool_calls": [{"tool": "read_file", "args": {"path": "README.md"}}],
                    }
                ]
            )

            with patch(
                "spice.runtime.workspace_perception.build_candidate_expander_client",
                return_value=client,
            ):
                result = run_runtime_workspace_perception_step(
                    project_root=tmp_dir,
                    query="current implementation report",
                    config={
                        "llm_provider": "fake",
                        "llm_model": "fake-model",
                        "workspace_perception": {
                            "depth": "deep",
                            "max_rounds": 12,
                            "max_tool_calls": 90,
                        },
                    },
                    trigger="new_decision",
                    store=LocalJsonStore.from_project_root(tmp_dir),
                    now=NOW,
                )

            limits = result.artifact["budget"]["limits"]
            self.assertEqual(limits["depth"], "deep")
            self.assertEqual(limits["max_rounds"], 12)
            self.assertEqual(limits["max_tool_calls"], 90)
            self.assertEqual(limits["max_tool_calls_per_round"], 16)
            self.assertEqual(client.requests[0].max_tokens, 8000)

    def test_run_once_accepts_workspace_context_for_decision_and_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            workspace_context = {
                "source": "workspace_perception",
                "perception_id": "workspace.test",
                "summary": "The repo already has a workspace_context field.",
                "facts": [{"text": "DecisionContext carries workspace_context."}],
            }
            workspace_perception = {
                "perception_id": "workspace.test",
                "summary": "The repo already has a workspace_context field.",
            }

            result = run_once(
                "Plan based on the current repo.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                workspace_context=workspace_context,
                workspace_perception=workspace_perception,
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"]["workspace_context"]
            simulation_ctx = result.artifact["compiled_context"]["simulation_context"]["workspace_context"]
            self.assertEqual(decision_ctx["source"], "workspace_perception")
            self.assertEqual(simulation_ctx["perception_id"], "workspace.test")
            self.assertEqual(result.artifact["workspace_context"]["perception_id"], "workspace.test")
            self.assertEqual(result.artifact["store_paths"]["workspace_perception"], ".spice/perceptions/workspace.test.json")
            turn = LocalJsonStore.from_project_root(tmp_dir).load_conversation_turn(
                result.artifact["conversation_turn_id"]
            )
            self.assertEqual(turn["artifact_refs"]["workspace_perception"], ".spice/perceptions/workspace.test.json")
            debug_context = compile_workspace_decision_context_payload(project_root=tmp_dir)
            self.assertEqual(debug_context["workspace_context"]["perception_id"], "workspace.test")
            self.assertEqual(debug_context["workspace_context"]["source"], "workspace_perception")

    def test_sources_debug_payload_includes_workspace_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            perception = build_workspace_perception_artifact(
                workspace_root=tmp_dir,
                trigger="test",
                query="workspace source audit",
                tool_calls=[
                    {
                        "call_id": "tool.1",
                        "round_index": 1,
                        "tool": "search",
                        "args": {"pattern": "workspace_context"},
                        "status": "executed",
                        "result": {
                            "ok": True,
                            "matches": [
                                {
                                    "path": "spice/runtime/run_once.py",
                                    "line": 42,
                                    "text": "workspace_context is passed into run_once.",
                                }
                            ],
                        },
                    },
                    {
                        "call_id": "tool.2",
                        "round_index": 1,
                        "tool": "git_diff",
                        "args": {"mode": "stat"},
                        "status": "executed",
                        "result": {
                            "ok": True,
                            "path": "git.diff",
                            "mode": "stat",
                            "content_preview": "spice/runtime/run_once.py | 3 +++",
                        },
                    },
                    {
                        "call_id": "tool.3",
                        "round_index": 1,
                        "tool": "git_log",
                        "args": {"limit": 1},
                        "status": "executed",
                        "result": {
                            "ok": True,
                            "entries": [{"sha": "abc123", "subject": "Add workspace perception"}],
                        },
                    },
                    {
                        "call_id": "tool.4",
                        "round_index": 1,
                        "tool": "python_symbol_index",
                        "args": {"path": "spice/runtime"},
                        "status": "executed",
                        "result": {
                            "ok": True,
                            "path": "spice/runtime",
                            "symbols": [
                                {
                                    "path": "spice/runtime/run_once.py",
                                    "name": "run_once",
                                    "qualified_name": "run_once",
                                    "kind": "function",
                                    "line_start": 10,
                                    "line_end": 80,
                                }
                            ],
                            "imports": [{"path": "spice/runtime/run_once.py", "module": "typing"}],
                            "modules": [
                                {
                                    "path": "spice/runtime/run_once.py",
                                    "module": "spice.runtime.run_once",
                                    "symbol_count": 1,
                                    "import_count": 1,
                                }
                            ],
                        },
                    },
                    {
                        "call_id": "tool.5",
                        "round_index": 1,
                        "tool": "read_python_symbol",
                        "args": {
                            "path": "spice/runtime/run_once.py",
                            "qualified_name": "run_once",
                        },
                        "status": "executed",
                        "result": {
                            "ok": True,
                            "path": "spice/runtime/run_once.py",
                            "qualified_name": "run_once",
                            "name": "run_once",
                            "kind": "function",
                            "content_preview": "def run_once(...): ...",
                            "line_start": 10,
                            "line_end": 80,
                            "chars_read": 100,
                            "content_hash": "hash-symbol",
                        },
                    },
                ],
                files_read=[{"path": "spice/runtime/run_once.py", "chars_read": 1200}],
                snippets=[
                    {
                        "path": "spice/runtime/run_once.py",
                        "text": "def run_once(..., workspace_context=None): ...",
                        "source": "read_file",
                        "line_start": 10,
                        "line_end": 20,
                    }
                ],
                facts=[
                    {
                        "text": "run_once accepts workspace_context.",
                        "source_path": "spice/runtime/run_once.py",
                    }
                ],
                summary="Workspace context is wired into run_once.",
            ).to_payload()
            store.save_perception(str(perception["perception_id"]), perception)
            run_once(
                "Plan based on current repo sources.",
                project_root=tmp_dir,
                workspace_context=workspace_context_from_perception(perception),
                workspace_perception=perception,
                full_loop_preview=False,
            )

            payload = compile_sources_debug_payload(project_root=tmp_dir)
            rendered = render_sources_debug_text(payload)

            self.assertEqual(payload["schema_version"], "spice.sources_debug.v1")
            self.assertEqual(payload["status"], "available")
            self.assertEqual(payload["workspace"]["perception_id"], perception["perception_id"])
            self.assertEqual(payload["workspace"]["files_read"][0]["path"], "spice/runtime/run_once.py")
            self.assertEqual(payload["workspace"]["search_matches"][0]["path"], "spice/runtime/run_once.py")
            self.assertEqual(payload["workspace"]["git_diff"][0]["path"], "git.diff")
            self.assertEqual(payload["workspace"]["git_log"][0]["entries"][0]["sha"], "abc123")
            self.assertEqual(
                payload["workspace"]["python_symbol_index"][0]["symbols"][0]["qualified_name"],
                "run_once",
            )
            self.assertEqual(
                payload["workspace"]["python_symbol_reads"][0]["qualified_name"],
                "run_once",
            )
            self.assertIn("SOURCES", rendered)
            self.assertIn("Workspace sources: Spice inspected directly", rendered)
            self.assertIn("Files read:", rendered)
            self.assertIn("Search matches:", rendered)
            self.assertIn("Git diff:", rendered)
            self.assertIn("Git log:", rendered)
            self.assertIn("Python symbol index:", rendered)
            self.assertIn("Python symbols read:", rendered)

    def test_refine_accepts_workspace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW, full_loop_preview=False)
            workspace_context = {
                "source": "workspace_perception",
                "perception_id": "workspace.refine",
                "summary": "Refine should consider current repo facts.",
            }

            result = refine_decision(
                "Adjust this based on current implementation.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                workspace_context=workspace_context,
                workspace_perception={"perception_id": "workspace.refine"},
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"]["workspace_context"]
            self.assertEqual(decision_ctx["perception_id"], "workspace.refine")
            self.assertEqual(result.artifact["store_paths"]["workspace_perception"], ".spice/perceptions/workspace.refine.json")


if __name__ == "__main__":
    unittest.main()
