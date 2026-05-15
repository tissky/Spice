from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spice.llm.core import LLMRequest, LLMResponse
from spice.perception import (
    ControlledWorkspacePerceptionLimits,
    WorkspaceInspector,
    build_workspace_perception_artifact_from_loop,
    run_controlled_workspace_perception_loop,
    workspace_context_from_perception,
)


class _FakeWorkspaceLoopClient:
    def __init__(self, outputs: list[dict[str, object] | str]) -> None:
        self.outputs = list(outputs)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.outputs:
            output = self.outputs.pop(0)
        else:
            output = {"done": True, "tool_calls": [], "summary": "done"}
        output_text = output if isinstance(output, str) else json.dumps(output)
        return LLMResponse(
            provider_id="test",
            model_id="workspace-loop-test",
            output_text=output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=1,
            request_id=f"workspace-loop-{len(self.requests)}",
        )


class ControlledWorkspacePerceptionLoopTests(unittest.TestCase):
    def test_executes_allowed_read_only_search_and_read_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "class Runtime:\n    pass\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Find runtime code.",
                        "tool_calls": [
                            {
                                "tool": "search",
                                "args": {
                                    "pattern": "Runtime",
                                    "file_glob": "*.py",
                                    "limit": 5,
                                },
                            },
                            {
                                "tool": "read_file",
                                "args": {
                                    "path": "spice/runtime.py",
                                    "offset": 1,
                                    "limit": 2,
                                },
                            },
                        ],
                    },
                    {"done": True, "summary": "Runtime code found.", "tool_calls": []},
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Inspect runtime implementation.",
            )

            self.assertEqual(len(client.requests), 2)
            self.assertEqual(result.rounds_used, 2)
            self.assertEqual(len(result.executed_tool_calls), 2)
            self.assertEqual(result.executed_tool_calls[0].tool, "search")
            self.assertEqual(
                result.executed_tool_calls[0].result["matches"][0]["path"],
                "spice/runtime.py",
            )
            self.assertEqual(result.executed_tool_calls[1].tool, "read_file")
            self.assertIn("class Runtime", result.executed_tool_calls[1].result["content"])
            self.assertEqual(len(result.round_batches), 1)
            self.assertEqual(result.round_batches[0]["round"], 1)
            self.assertEqual(len(result.round_batches[0]["requested_tool_calls"]), 2)
            self.assertEqual(len(result.round_batches[0]["executed_tool_calls"]), 2)
            self.assertEqual(result.round_batches[0]["executed_tool_calls"][0]["tool"], "search")
            self.assertEqual(result.round_batches[0]["blocked_tool_calls"], [])
            self.assertEqual(result.budget["tool_calls_executed"], 2)
            self.assertEqual(result.budget["round_batches_recorded"], 1)
            second_prompt = json.loads(client.requests[1].input_text)
            self.assertEqual(second_prompt["budget_state"]["remaining_tool_calls"], 78)
            self.assertEqual(second_prompt["budget_state"]["remaining_files"], 59)
            self.assertLess(second_prompt["budget_state"]["remaining_chars"], 500_000)
            self.assertEqual(second_prompt["budget_state"]["files_already_read"], ["spice/runtime.py"])

    def test_default_limits_use_normal_workspace_perception_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [{"done": True, "summary": "No more evidence needed.", "tool_calls": []}]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Inspect current repo.",
            )

            limits = result.budget["limits"]
            self.assertEqual(limits["depth"], "normal")
            self.assertEqual(limits["max_rounds"], 10)
            self.assertEqual(limits["max_tool_calls"], 80)
            self.assertEqual(limits["max_tool_calls_per_round"], 10)
            self.assertEqual(limits["max_blocked_tool_calls"], 20)
            self.assertEqual(limits["max_blocked_tool_calls_per_round"], 5)
            self.assertEqual(limits["max_files_read"], 60)
            self.assertEqual(limits["max_chars_per_file"], 12_000)
            self.assertEqual(limits["total_char_budget"], 500_000)
            self.assertEqual(limits["planner_max_tokens"], 4_000)
            self.assertEqual(client.requests[0].max_tokens, 4_000)
            prompt = json.loads(client.requests[0].input_text)
            self.assertEqual(prompt["limits"]["max_tool_calls_per_round"], 10)
            self.assertEqual(prompt["budget_state"]["depth"], "normal")
            self.assertEqual(prompt["budget_state"]["remaining_tool_calls"], 80)
            self.assertEqual(prompt["budget_state"]["remaining_files"], 60)
            self.assertEqual(prompt["budget_state"]["remaining_chars"], 500_000)
            self.assertEqual(prompt["budget_state"]["blocked_call_budget"]["remaining"], 20)
            self.assertEqual(prompt["budget_state"]["budget_pressure"], "low")
            self.assertEqual(prompt["budget_state"]["files_already_read"], [])
            self.assertIn(
                "Request at most 10 read-only tool calls in this round.",
                prompt["rules"],
            )
            self.assertIn("investigation_strategy", prompt)
            self.assertIn("repo_map", " ".join(prompt["investigation_strategy"]["orient"]))
            self.assertIn("read_package_metadata", " ".join(prompt["investigation_strategy"]["orient"]))
            self.assertIn("read_test_structure", " ".join(prompt["investigation_strategy"]["orient"]))
            self.assertIn("git_status", " ".join(prompt["investigation_strategy"]["orient"]))
            self.assertIn("search", " ".join(prompt["investigation_strategy"]["locate"]))
            self.assertIn("python_symbol_index", " ".join(prompt["investigation_strategy"]["locate"]))
            self.assertIn("read_file", " ".join(prompt["investigation_strategy"]["inspect"]))
            self.assertIn("read_python_symbol", " ".join(prompt["investigation_strategy"]["inspect"]))
            self.assertIn("tests", " ".join(prompt["investigation_strategy"]["verify"]))
            self.assertIn("docs", " ".join(prompt["investigation_strategy"]["verify"]))
            self.assertIn("imports", " ".join(prompt["investigation_strategy"]["verify"]))
            self.assertIn("config", " ".join(prompt["investigation_strategy"]["verify"]))
            self.assertIn("Do not read random large files.", prompt["investigation_strategy"]["avoid"])
            self.assertIn(
                "Follow the investigation_strategy: first orient, then locate, then inspect, then verify.",
                prompt["rules"],
            )
            self.assertNotIn("Request at most 3 tool calls in one round.", prompt["rules"])

    def test_rolling_investigation_state_is_carried_to_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "class Runtime:\n"
                "    workspace_context = 'state'\n"
                "    " + ("x" * 2000) + "\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Inspect runtime context.",
                        "investigation_state": {
                            "investigation_summary": "Looking for workspace context wiring.",
                            "key_findings": ["runtime context is likely in spice/runtime.py"],
                            "open_questions": ["Need to read the implementation."],
                            "next_leads": ["spice/runtime.py"],
                            "source_map": [
                                {
                                    "source": "search:workspace_context",
                                    "supports": "points at runtime context wiring",
                                }
                            ],
                        },
                        "tool_calls": [
                            {
                                "tool": "read_file",
                                "args": {"path": "spice/runtime.py", "offset": 1, "limit": 3},
                            }
                        ],
                    },
                    {
                        "done": True,
                        "summary": "Runtime context source read.",
                        "investigation_state": {
                            "investigation_summary": "Runtime context source was read.",
                            "key_findings": ["spice/runtime.py contains workspace_context wiring"],
                            "open_questions": [],
                            "next_leads": [],
                        },
                        "tool_calls": [],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Inspect runtime context.",
            )

            second_prompt = json.loads(client.requests[1].input_text)
            state = second_prompt["investigation_state"]
            self.assertEqual(state["investigation_summary"], "Looking for workspace context wiring.")
            self.assertIn("runtime context is likely in spice/runtime.py", state["key_findings"])
            self.assertEqual(state["files_read"], ["spice/runtime.py"])
            self.assertEqual(state["source_map"][0]["source"], "search:workspace_context")
            self.assertEqual(second_prompt["recent_observations"][0]["tool"], "read_file")
            content = second_prompt["recent_observations"][0]["result"]["content"]
            self.assertLessEqual(len(content), 1200)
            self.assertEqual(
                result.investigation_state["investigation_summary"],
                "Runtime context source was read.",
            )
            self.assertIn("spice/runtime.py", result.investigation_state["files_read"])
            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=result,
            )
            artifact_payload = artifact.to_payload()
            self.assertEqual(
                artifact_payload["metadata"]["loop"]["investigation_state"][
                    "investigation_summary"
                ],
                "Runtime context source was read.",
            )

    def test_sufficiency_check_stops_early_with_complete_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "def run_once():\n    return 'decision'\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Read the key runtime file.",
                        "tool_calls": [
                            {
                                "tool": "read_file",
                                "args": {"path": "spice/runtime.py", "offset": 1, "limit": 4},
                            }
                        ],
                    },
                    {
                        "summary": "Enough source-backed evidence was read.",
                        "sufficiency_check": {
                            "sufficient_evidence": True,
                            "can_answer_user_question": True,
                            "remaining_gaps": [],
                            "reason": "read key implementation files and supporting docs",
                        },
                        "tool_calls": [
                            {"tool": "search", "args": {"pattern": "should_not_run"}}
                        ],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Inspect current implementation.",
            )

            self.assertEqual(len(client.requests), 2)
            self.assertEqual(len(result.executed_tool_calls), 1)
            self.assertTrue(result.done)
            self.assertEqual(result.exploration_status, "complete")
            self.assertTrue(result.sufficiency_check["sufficient_evidence"])
            self.assertTrue(result.sufficiency_check["can_answer_user_question"])
            self.assertEqual(
                result.sufficiency_check["reason"],
                "read key implementation files and supporting docs",
            )

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=result,
            )
            loop_metadata = artifact.to_payload()["metadata"]["loop"]
            self.assertEqual(loop_metadata["exploration_status"], "complete")
            self.assertTrue(loop_metadata["sufficiency_check"]["sufficient_evidence"])

    def test_partial_status_when_model_can_answer_with_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Can answer with caveats.",
                        "sufficiency_check": {
                            "sufficient_evidence": False,
                            "can_answer_user_question": True,
                            "remaining_gaps": ["did not read tests"],
                            "reason": "implementation file was read but tests were not checked",
                        },
                        "tool_calls": [],
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(tmp),
                query="Inspect repo with caveats.",
            )

            self.assertTrue(result.done)
            self.assertEqual(result.exploration_status, "partial")
            self.assertFalse(result.sufficiency_check["sufficient_evidence"])
            self.assertTrue(result.sufficiency_check["can_answer_user_question"])
            self.assertEqual(result.sufficiency_check["remaining_gaps"], ["did not read tests"])

    def test_hard_repo_evidence_empty_first_round_runs_minimum_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\n",
                encoding="utf-8",
            )
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "STATE_AS_CONTEXT = True\n"
                "def compile_workspace_context():\n"
                "    return {'state_as_context': STATE_AS_CONTEXT}\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_runtime.py").write_text(
                "def test_state_as_context(): pass\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "No tool calls needed.",
                        "sufficiency_check": {
                            "sufficient_evidence": True,
                            "can_answer_user_question": True,
                            "remaining_gaps": [],
                            "reason": "claimed enough without sources",
                        },
                        "tool_calls": [],
                    },
                    {
                        "summary": "Fallback evidence was gathered.",
                        "sufficiency_check": {
                            "sufficient_evidence": True,
                            "can_answer_user_question": True,
                            "remaining_gaps": [],
                            "reason": "source-backed fallback read repo evidence",
                        },
                        "tool_calls": [],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="基于当前 repo 实现判断 state-as-context 优先级",
                initial_context={
                    "evidence_requirement": {
                        "requires_evidence": True,
                        "evidence_domain": "repo",
                    }
                },
            )

            tools = [call.tool for call in result.executed_tool_calls]
            self.assertIn("repo_map", tools)
            self.assertIn("read_package_metadata", tools)
            self.assertIn("read_test_structure", tools)
            self.assertIn("git_status", [call.tool for call in result.tool_calls])
            self.assertIn("search", tools)
            self.assertIn("read_file", tools)
            self.assertTrue(result.budget["minimum_investigation_fallback"]["triggered"])
            self.assertTrue(
                result.budget["minimum_investigation_fallback"]["source_backed_evidence"]
            )
            self.assertTrue(
                any(batch.get("minimum_investigation_fallback") for batch in result.round_batches)
            )

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=result,
            )
            payload = artifact.to_payload()
            self.assertTrue(payload["files_read"])
            self.assertTrue(payload["facts"])
            self.assertTrue(payload["snippets"])
            self.assertTrue(
                any(
                    batch.get("minimum_investigation_fallback")
                    for batch in payload["metadata"]["round_batches"]
                )
            )

    def test_minimum_fallback_reserves_read_slots_for_multi_keyword_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\n",
                encoding="utf-8",
            )
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "class SpiceRuntime:\n"
                "    def state_as_context(self): return True\n"
                "    def open_chronicle_perception(self): return 'planned'\n"
                "    def executor_handoff(self): return 'hermes'\n",
                encoding="utf-8",
            )
            (root / "docs").mkdir()
            (root / "docs" / "current_state.md").write_text(
                "state-as-context OpenChronicle proactive perception executor handoff\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {"summary": "No planner calls.", "tool_calls": []},
                    {
                        "summary": "Fallback evidence collected.",
                        "sufficiency_check": {
                            "sufficient_evidence": True,
                            "can_answer_user_question": True,
                        },
                        "tool_calls": [],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query=(
                    "Spice state-as-context OpenChronicle proactive perception executor "
                    "handoff 当前实现 代码 证据"
                ),
                initial_context={
                    "evidence_requirement": {
                        "requires_evidence": True,
                        "evidence_domain": "repo",
                    }
                },
                limits=ControlledWorkspacePerceptionLimits(max_tool_calls_per_round=10),
            )

            fallback_batches = [
                batch for batch in result.round_batches if batch.get("minimum_investigation_fallback")
            ]
            self.assertTrue(fallback_batches)
            requested_tools = [
                call["tool"]
                for batch in fallback_batches
                for call in batch["requested_tool_calls"]
            ]
            self.assertLessEqual(requested_tools.count("search"), 3)
            self.assertGreaterEqual(requested_tools.count("read_file"), 1)
            self.assertGreaterEqual(
                len([call for call in result.executed_tool_calls if call.tool == "read_file"]),
                1,
            )
            self.assertFalse(
                any(
                    call.tool == "read_file"
                    and call.status == "blocked"
                    and call.reason == "max_tool_calls_per_round_exceeded"
                    for call in result.tool_calls
                )
            )

    def test_hard_repo_evidence_blocked_first_round_runs_minimum_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "README.md"
            target.write_text("state-as-context implementation notes\n", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {
                                "tool": "write_file",
                                "args": {"path": "README.md", "content": "bad"},
                            }
                        ]
                    },
                    {
                        "summary": "Fallback evidence exists.",
                        "sufficiency_check": {
                            "sufficient_evidence": True,
                            "can_answer_user_question": True,
                        },
                        "tool_calls": [],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="读取当前代码后分析 state-as-context",
                initial_context={
                    "route_merge_policy": {
                        "needs_workspace_context": True,
                        "forced_by": ["repo_evidence_requirement"],
                        "evidence_requirement": {
                            "requires_evidence": True,
                            "evidence_domain": "repo",
                        },
                    }
                },
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "state-as-context implementation notes\n")
            self.assertEqual(result.blocked_tool_calls[0].tool, "write_file")
            self.assertTrue(result.budget["minimum_investigation_fallback"]["triggered"])
            self.assertIn("read_file", [call.tool for call in result.executed_tool_calls])

    def test_non_hard_repo_evidence_empty_first_round_does_not_run_minimum_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = _FakeWorkspaceLoopClient(
                [{"summary": "No evidence needed.", "tool_calls": []}]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(tmp),
                query="General architecture advice.",
            )

            self.assertTrue(result.done)
            self.assertEqual(result.tool_calls, [])
            self.assertFalse(result.budget["minimum_investigation_fallback"]["triggered"])

    def test_hard_repo_evidence_fallback_blocks_when_it_collects_no_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [{"summary": "No calls.", "tool_calls": []}]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="基于当前 repo 代码判断下一步",
                initial_context={
                    "evidence_requirement": {
                        "requires_evidence": True,
                        "evidence_domain": "repo",
                    }
                },
                limits=ControlledWorkspacePerceptionLimits(
                    max_tool_calls=80,
                    max_tool_calls_per_round=0,
                    max_blocked_tool_calls=40,
                    max_blocked_tool_calls_per_round=40,
                ),
            )

            self.assertTrue(result.done)
            self.assertEqual(result.exploration_status, "blocked")
            self.assertFalse(result.sufficiency_check["can_answer_user_question"])
            self.assertIn("source-backed repo evidence", result.sufficiency_check["reason"])
            self.assertTrue(result.budget["minimum_investigation_fallback"]["triggered"])
            self.assertFalse(
                result.budget["minimum_investigation_fallback"]["source_backed_evidence"]
            )

    def test_budget_exhausted_status_when_max_rounds_are_used_without_sufficiency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Still investigating.",
                        "sufficiency_check": {
                            "sufficient_evidence": False,
                            "can_answer_user_question": False,
                            "remaining_gaps": ["need implementation files"],
                            "reason": "only orientation completed",
                        },
                        "tool_calls": [{"tool": "file_index", "args": {"path": "."}}],
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Keep investigating.",
                limits=ControlledWorkspacePerceptionLimits(max_rounds=1),
            )

            self.assertFalse(result.done)
            self.assertEqual(result.rounds_used, 1)
            self.assertEqual(result.exploration_status, "budget_exhausted")
            self.assertEqual(result.sufficiency_check["remaining_gaps"], ["need implementation files"])

    def test_medium_budget_pressure_guides_planner_to_converge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Use most of the search budget.",
                        "tool_calls": [
                            {"tool": "file_index", "args": {"path": ".", "limit": 1}}
                            for _ in range(7)
                        ],
                    },
                    {"done": True, "summary": "Stop after medium pressure.", "tool_calls": []},
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Investigate under pressure.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_rounds=2,
                    max_tool_calls=10,
                    max_tool_calls_per_round=10,
                ),
            )

            second_prompt = json.loads(client.requests[1].input_text)
            self.assertEqual(second_prompt["budget_state"]["budget_pressure"], "medium")
            self.assertIn(
                "Converge on key sources",
                second_prompt["budget_pressure_guidance"],
            )
            self.assertEqual(result.budget["budget_pressure"], "medium")
            self.assertEqual(result.budget["budget_pressure_events"][0]["budget_pressure"], "medium")

    def test_high_budget_pressure_guides_planner_to_stop_broad_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Use nearly all search budget.",
                        "tool_calls": [
                            {"tool": "file_index", "args": {"path": ".", "limit": 1}}
                            for _ in range(9)
                        ],
                    },
                    {
                        "summary": "Answer with gaps.",
                        "sufficiency_check": {
                            "sufficient_evidence": False,
                            "can_answer_user_question": True,
                            "remaining_gaps": ["one source still unchecked"],
                            "reason": "budget is high",
                        },
                        "tool_calls": [],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Investigate under high pressure.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_rounds=2,
                    max_tool_calls=10,
                    max_tool_calls_per_round=10,
                ),
            )

            second_prompt = json.loads(client.requests[1].input_text)
            self.assertEqual(second_prompt["budget_state"]["budget_pressure"], "high")
            self.assertIn(
                "Stop broad search",
                second_prompt["budget_pressure_guidance"],
            )
            self.assertEqual(result.exploration_status, "partial")
            self.assertEqual(result.budget["budget_pressure"], "high")
            self.assertEqual(result.budget["budget_pressure_events"][0]["budget_pressure"], "high")

    def test_exhausted_budget_stops_loop_and_records_pressure_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Use final available tool call.",
                        "tool_calls": [{"tool": "file_index", "args": {"path": "."}}],
                    },
                    {
                        "summary": "This round should not run.",
                        "tool_calls": [{"tool": "search", "args": {"pattern": "unused"}}],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Exhaust tool budget.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_rounds=5,
                    max_tool_calls=1,
                    max_tool_calls_per_round=1,
                ),
            )

            self.assertEqual(len(client.requests), 1)
            self.assertEqual(result.exploration_status, "budget_exhausted")
            self.assertEqual(result.budget["budget_pressure"], "exhausted")
            self.assertEqual(
                result.budget["budget_pressure_events"][0]["budget_pressure"],
                "exhausted",
            )
            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=result,
            )
            self.assertEqual(
                artifact.to_payload()["metadata"]["loop"]["exploration_status"],
                "budget_exhausted",
            )
            payload = artifact.to_payload()
            self.assertEqual(payload["exploration_status"], "budget_exhausted")
            self.assertEqual(payload["budget_used"]["budget_pressure"], "exhausted")
            self.assertEqual(payload["metadata"]["exploration_status"], "budget_exhausted")
            self.assertEqual(payload["metadata"]["budget_used"]["budget_pressure"], "exhausted")
            self.assertEqual(
                payload["metadata"]["budget_pressure_events"][0]["budget_pressure"],
                "exhausted",
            )
            self.assertTrue(
                any("budget was exhausted" in item for item in payload["metadata"]["limitations"])
            )

    def test_executes_extended_local_workspace_read_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\ndependencies = [\"rich\"]\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_runtime.py").write_text("def test_x(): pass\n", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Orient around repo state.",
                        "tool_calls": [
                            {"tool": "repo_map", "args": {"path": ".", "max_depth": 2}},
                            {"tool": "read_package_metadata", "args": {"path": "."}},
                            {"tool": "read_test_structure", "args": {"path": "."}},
                        ],
                    },
                    {
                        "summary": "Inspect git state.",
                        "tool_calls": [
                            {"tool": "git_status", "args": {"limit": 20}},
                            {"tool": "git_diff", "args": {"mode": "stat"}},
                            {"tool": "git_log", "args": {"limit": 3}},
                        ],
                    },
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Understand the current repo.",
            )

            tools = [call.tool for call in result.tool_calls]
            self.assertEqual(
                tools,
                [
                    "repo_map",
                    "read_package_metadata",
                    "read_test_structure",
                    "git_status",
                    "git_diff",
                    "git_log",
                ],
            )
            self.assertEqual(result.tool_calls[0].status, "executed")
            self.assertEqual(result.tool_calls[1].status, "executed")
            self.assertEqual(result.tool_calls[2].status, "executed")
            self.assertIn("files", result.tool_calls[1].result)
            self.assertIn("test_files", result.tool_calls[2].result)
            self.assertIn(result.tool_calls[3].status, {"executed", "failed"})
            self.assertIn(result.tool_calls[4].status, {"executed", "failed"})
            self.assertIn(result.tool_calls[5].status, {"executed", "failed"})

    def test_executes_python_symbol_index_and_symbol_read_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "service.py").write_text(
                "class Service:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Find Python structure.",
                        "tool_calls": [
                            {"tool": "python_symbol_index", "args": {"path": "pkg"}},
                            {
                                "tool": "read_python_symbol",
                                "args": {
                                    "path": "pkg/service.py",
                                    "qualified_name": "Service.run",
                                },
                            },
                        ],
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Understand Service.run.",
            )

            self.assertEqual(len(result.executed_tool_calls), 2)
            self.assertEqual(result.executed_tool_calls[0].tool, "python_symbol_index")
            self.assertEqual(
                result.executed_tool_calls[0].result["symbols"][0]["qualified_name"],
                "Service",
            )
            self.assertEqual(result.executed_tool_calls[1].tool, "read_python_symbol")
            self.assertIn("def run", result.executed_tool_calls[1].result["content"])

    def test_blocks_disallowed_write_tool_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "README.md"
            target.write_text("original", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {
                                "tool": "write_file",
                                "args": {"path": "README.md", "content": "changed"},
                            }
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Try to write.",
            )

            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].tool, "write_file")
            self.assertEqual(result.blocked_tool_calls[0].reason, "tool_not_allowed")

    def test_blocks_non_write_illegal_tool_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {
                                "tool": "terminal_command",
                                "args": {"cmd": "python -m pytest"},
                            }
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Try to run tests.",
            )

            self.assertEqual(len(result.executed_tool_calls), 0)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].tool, "terminal_command")
            self.assertEqual(result.blocked_tool_calls[0].reason, "tool_not_allowed")

    def test_read_only_boundary_blocks_patch_and_test_run_tools(self) -> None:
        for tool_name, args in (
            ("patch", {"path": "a.txt", "patch": "@@ -1 +1 @@"}),
            ("test_run", {"command": "python -m unittest"}),
        ):
            with self.subTest(tool=tool_name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "a.txt").write_text("alpha", encoding="utf-8")
                    client = _FakeWorkspaceLoopClient(
                        [{"tool_calls": [{"tool": tool_name, "args": args}]}]
                    )

                    result = run_controlled_workspace_perception_loop(
                        client=client,
                        inspector=WorkspaceInspector(root),
                        query=f"Try illegal {tool_name}.",
                    )

                    self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "alpha")
                    self.assertEqual(len(result.executed_tool_calls), 0)
                    self.assertEqual(len(result.blocked_tool_calls), 1)
                    self.assertEqual(result.blocked_tool_calls[0].tool, tool_name)
                    self.assertEqual(result.blocked_tool_calls[0].reason, "tool_not_allowed")

    def test_blocked_calls_do_not_consume_normal_tool_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "write_file", "args": {"path": "a.txt", "content": "bad"}},
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Blocked call should not burn the read budget.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_tool_calls=1,
                    max_blocked_tool_calls=5,
                    max_blocked_tool_calls_per_round=5,
                ),
            )

            self.assertEqual(len(result.executed_tool_calls), 1)
            self.assertEqual(result.executed_tool_calls[0].tool, "read_file")
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "tool_not_allowed")
            self.assertEqual(result.budget["normal_tool_calls_used"], 1)
            self.assertEqual(result.budget["tool_calls_blocked"], 1)
            self.assertEqual(len(result.round_batches), 1)
            batch = result.round_batches[0]
            self.assertEqual(batch["round"], 1)
            self.assertEqual(len(batch["requested_tool_calls"]), 2)
            self.assertEqual(len(batch["executed_tool_calls"]), 1)
            self.assertEqual(batch["executed_tool_calls"][0]["tool"], "read_file")
            self.assertEqual(len(batch["blocked_tool_calls"]), 1)
            self.assertEqual(batch["blocked_tool_calls"][0]["tool"], "write_file")

    def test_blocked_budget_exhaustion_stops_loop_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "terminal_command", "args": {"cmd": "pytest"}},
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Stop after blocked budget is exhausted.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_tool_calls=5,
                    max_blocked_tool_calls=1,
                    max_blocked_tool_calls_per_round=1,
                ),
            )

            self.assertEqual(len(result.executed_tool_calls), 0)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "tool_not_allowed")
            self.assertEqual(result.exploration_status, "blocked")
            self.assertTrue(result.budget["blocked_budget_exhausted"])

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=result,
            )
            payload = artifact.to_payload()
            self.assertEqual(payload["exploration_status"], "blocked")
            self.assertTrue(
                any("Blocked tool-call budget was exhausted" in item for item in payload["limitations"])
            )

    def test_blocked_calls_per_round_exhaustion_stops_before_later_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "write_file", "args": {"path": "a.txt", "content": "bad"}},
                            {"tool": "terminal_command", "args": {"cmd": "pytest"}},
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Stop after blocked per-round budget is exhausted.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_tool_calls=5,
                    max_blocked_tool_calls=10,
                    max_blocked_tool_calls_per_round=1,
                ),
            )

            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "alpha")
            self.assertEqual(len(result.executed_tool_calls), 0)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].tool, "write_file")
            self.assertEqual(result.exploration_status, "blocked")
            self.assertTrue(result.budget["blocked_budget_exhausted"])
            self.assertEqual(result.budget["normal_tool_calls_used"], 0)

    def test_blocks_calls_over_max_tool_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                            {"tool": "read_file", "args": {"path": "b.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Read files.",
                limits=ControlledWorkspacePerceptionLimits(max_tool_calls=1),
            )

            self.assertEqual(len(result.executed_tool_calls), 1)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "max_tool_calls_exceeded")

    def test_blocks_calls_over_per_round_tool_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "file_index", "args": {"path": "."}},
                            {"tool": "search", "args": {"pattern": "alpha"}},
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Read files.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_tool_calls=6,
                    max_tool_calls_per_round=2,
                ),
            )

            self.assertEqual(len(result.executed_tool_calls), 2)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(
                result.blocked_tool_calls[0].reason,
                "max_tool_calls_per_round_exceeded",
            )

    def test_loop_clamps_inspector_read_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                            {"tool": "read_file", "args": {"path": "b.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Read files.",
                limits=ControlledWorkspacePerceptionLimits(max_files_read=1),
            )

            self.assertEqual(len(result.executed_tool_calls), 1)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "max_files_read_exceeded")
            self.assertEqual(result.budget["inspector"]["files_read_count"], 1)

    def test_loop_stops_reading_when_total_char_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            (root / "b.txt").write_text("beta", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "read_file", "args": {"path": "a.txt"}},
                            {"tool": "read_file", "args": {"path": "b.txt"}},
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Read files.",
                limits=ControlledWorkspacePerceptionLimits(
                    max_chars_per_file=20,
                    total_char_budget=4,
                ),
            )

            self.assertEqual(len(result.executed_tool_calls), 1)
            self.assertEqual(result.executed_tool_calls[0].result["content"], "alph")
            self.assertTrue(result.executed_tool_calls[0].result["truncated"])
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "total_char_budget_exceeded")
            self.assertEqual(result.budget["inspector"]["chars_used"], 4)

    def test_primitive_guardrail_failures_are_recorded_as_blocked_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".spice").mkdir()
            (root / ".spice" / "state.json").write_text("{}", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "tool_calls": [
                            {"tool": "read_file", "args": {"path": ".spice/state.json"}}
                        ]
                    }
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Read hidden state.",
            )

            self.assertEqual(len(result.executed_tool_calls), 0)
            self.assertEqual(len(result.blocked_tool_calls), 1)
            self.assertEqual(result.blocked_tool_calls[0].reason, "deny_dir")

    def test_stops_after_max_rounds_even_if_model_keeps_requesting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {"tool_calls": [{"tool": "file_index", "args": {"path": "."}}]},
                    {"tool_calls": [{"tool": "search", "args": {"pattern": "alpha"}}]},
                    {"tool_calls": [{"tool": "read_file", "args": {"path": "a.txt"}}]},
                ]
            )

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Keep going.",
                limits=ControlledWorkspacePerceptionLimits(max_rounds=2),
            )

            self.assertEqual(result.rounds_used, 2)
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(len(result.executed_tool_calls), 2)

    def test_malformed_model_output_stops_without_tool_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = _FakeWorkspaceLoopClient(["not json"])

            result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(tmp),
                query="Inspect.",
            )

            self.assertEqual(result.rounds_used, 1)
            self.assertEqual(result.tool_calls, [])
            self.assertTrue(result.done)

    def test_legal_search_and_read_loop_result_builds_auditable_artifact_and_compact_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice").mkdir()
            (root / "spice" / "runtime.py").write_text(
                "class Runtime:\n    workspace_context = {}\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Runtime workspace context found.",
                        "tool_calls": [
                            {
                                "tool": "search",
                                "args": {"pattern": "workspace_context", "file_glob": "*.py"},
                            },
                            {
                                "tool": "read_file",
                                "args": {"path": "spice/runtime.py", "offset": 1, "limit": 2},
                            },
                        ],
                    }
                ]
            )
            loop_result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Find workspace context implementation.",
            )

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=loop_result,
            )
            payload = artifact.to_payload()
            context = workspace_context_from_perception(payload)

            self.assertEqual(len(payload["tool_calls"]), 2)
            self.assertEqual(payload["tool_calls"][0]["tool"], "search")
            self.assertEqual(payload["tool_calls"][1]["tool"], "read_file")
            self.assertEqual(payload["files_read"][0]["path"], "spice/runtime.py")
            self.assertEqual(payload["facts"][0]["source_path"], "spice/runtime.py")
            self.assertNotIn("content", payload["tool_calls"][1]["result"])
            self.assertIn("content_preview", payload["tool_calls"][1]["result"])
            self.assertEqual(len(payload["metadata"]["round_batches"]), 1)
            self.assertEqual(len(payload["metadata"]["round_batches"][0]["requested_tool_calls"]), 2)
            self.assertEqual(
                payload["metadata"]["round_batches"][0]["executed_tool_calls"][1]["tool"],
                "read_file",
            )
            self.assertNotIn(
                "content",
                payload["metadata"]["round_batches"][0]["executed_tool_calls"][1]["result"],
            )
            self.assertIn(
                "content_preview",
                payload["metadata"]["round_batches"][0]["executed_tool_calls"][1]["result"],
            )
            self.assertEqual(
                payload["metadata"]["loop"]["round_batches"],
                payload["metadata"]["round_batches"],
            )
            self.assertEqual(context["source"], "workspace_perception")
            self.assertEqual(context["perception_id"], payload["perception_id"])
            self.assertEqual(context["files_read"][0]["path"], "spice/runtime.py")
            self.assertNotIn("tool_calls", context)

    def test_extended_read_tools_build_compact_facts_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\ndependencies = [\"rich\"]\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_runtime.py").write_text("def test_x(): pass\n", encoding="utf-8")
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Read local repo orientation.",
                        "tool_calls": [
                            {"tool": "repo_map", "args": {"path": "."}},
                            {"tool": "read_package_metadata", "args": {"path": "."}},
                            {"tool": "read_test_structure", "args": {"path": "."}},
                        ],
                    }
                ]
            )
            loop_result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Repo orientation.",
            )

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=loop_result,
            )
            payload = artifact.to_payload()
            context = workspace_context_from_perception(payload)
            fact_text = "\n".join(item["text"] for item in payload["facts"])

            self.assertIn("Repo map inspected", fact_text)
            self.assertIn("Package metadata inspected", fact_text)
            self.assertIn("Test structure inspected", fact_text)
            self.assertEqual(payload["files_read"][0]["path"], "pyproject.toml")
            self.assertEqual(context["files_read"][0]["path"], "pyproject.toml")
            self.assertNotIn("tool_calls", context)

    def test_python_symbol_tools_build_compact_facts_and_snippets_in_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "service.py").write_text(
                "class Service:\n"
                "    def run(self):\n"
                "        return 'ok'\n",
                encoding="utf-8",
            )
            client = _FakeWorkspaceLoopClient(
                [
                    {
                        "summary": "Read Python symbols.",
                        "tool_calls": [
                            {"tool": "python_symbol_index", "args": {"path": "pkg"}},
                            {
                                "tool": "read_python_symbol",
                                "args": {
                                    "path": "pkg/service.py",
                                    "qualified_name": "Service.run",
                                },
                            },
                        ],
                    }
                ]
            )
            loop_result = run_controlled_workspace_perception_loop(
                client=client,
                inspector=WorkspaceInspector(root),
                query="Service.run implementation.",
            )

            artifact = build_workspace_perception_artifact_from_loop(
                workspace_root=root,
                trigger="test",
                loop_result=loop_result,
            )
            payload = artifact.to_payload()
            context = workspace_context_from_perception(payload)
            fact_text = "\n".join(item["text"] for item in payload["facts"])

            self.assertIn("Python symbol index inspected", fact_text)
            self.assertIn("Read Python symbol Service.run", fact_text)
            self.assertEqual(payload["snippets"][0]["source"], "read_python_symbol")
            self.assertEqual(payload["files_read"][0]["path"], "pkg/service.py")
            self.assertEqual(context["snippets"][0]["source"], "read_python_symbol")


if __name__ == "__main__":
    unittest.main()
