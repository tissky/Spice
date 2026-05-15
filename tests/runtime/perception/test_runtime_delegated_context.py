from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from spice.llm.core import LLMRequest, LLMResponse
from spice.runtime import LocalJsonStore, run_once, setup_workspace
from spice.runtime.composer_context import compact_composer_context
from spice.runtime.context_debug import (
    compile_sources_debug_payload,
    compile_workspace_debug_payload,
    compile_workspace_decision_context_payload,
    render_sources_debug_text,
    render_workspace_debug_text,
)
from spice.runtime.follow_up import answer_general_follow_up, answer_general_follow_up_with_llm
from spice.runtime.refine import refine_decision
from spice.runtime.workspace import load_workspace_memory_provider


NOW = datetime(2026, 5, 13, 8, 0, tzinfo=timezone.utc)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider_id="fake",
            model_id="fake-model",
            output_text=self.output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=1,
            request_id="req.delegated.context",
        )


class RuntimeDelegatedContextTests(unittest.TestCase):
    def test_run_once_accepts_delegated_context_for_decision_and_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            delegated_context = _delegated_context()
            delegated_perception = {
                **delegated_context,
                "status": "completed",
                "executor_run_ref": "hermes.run.1",
            }

            result = run_once(
                "Based on Hermes investigation, what should we do?",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                delegated_perception_context=delegated_context,
                delegated_perception=delegated_perception,
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"][
                "delegated_perception_context"
            ]
            simulation_ctx = result.artifact["compiled_context"]["simulation_context"][
                "delegated_perception_context"
            ]
            self.assertEqual(decision_ctx["source"], "delegated_perception")
            self.assertEqual(simulation_ctx["perception_id"], "delegated.test")
            self.assertEqual(result.artifact["delegated_perception_context"]["perception_id"], "delegated.test")
            self.assertTrue(result.artifact["evidence_context"]["delegated"]["present"])
            self.assertEqual(
                result.artifact["evidence_context"]["delegated"]["perception_id"],
                "delegated.test",
            )
            self.assertEqual(
                result.artifact["store_paths"]["delegated_perception"],
                ".spice/perceptions/delegated.test.json",
            )

            store = LocalJsonStore.from_project_root(tmp_dir)
            turn = store.load_conversation_turn(result.artifact["conversation_turn_id"])
            self.assertEqual(
                turn["artifact_refs"]["delegated_perception"],
                ".spice/perceptions/delegated.test.json",
            )
            self.assertEqual(
                turn["metadata"]["delegated_perception_context"]["executor_id"],
                "hermes",
            )
            self.assertEqual(
                turn["metadata"]["evidence_context"]["delegated"]["executor_id"],
                "hermes",
            )

            debug_context = compile_workspace_decision_context_payload(project_root=tmp_dir)
            self.assertEqual(
                debug_context["delegated_perception_context"]["perception_id"],
                "delegated.test",
            )
            debug_payload = compile_workspace_debug_payload(project_root=tmp_dir)
            self.assertEqual(debug_payload["delegated_perception_context"]["executor_id"], "hermes")
            rendered_debug = render_workspace_debug_text(debug_payload)
            self.assertIn("Delegated perception:", rendered_debug)
            self.assertIn("delegated.test", rendered_debug)

    def test_sources_debug_payload_includes_delegated_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            delegated_context = _delegated_context()
            delegated_perception = {
                **delegated_context,
                "status": "completed",
                "consent_id": "investigation.sources",
                "executor_report_ref": "executor_report.sources",
                "executor_run_ref": "hermes.run.sources",
            }
            store.save_perception("delegated.test", delegated_perception)

            run_once(
                "Use the delegated investigation.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                delegated_perception_context=delegated_context,
                delegated_perception=delegated_perception,
            )

            payload = compile_sources_debug_payload(project_root=tmp_dir)
            rendered = render_sources_debug_text(payload)

            self.assertEqual(payload["status"], "available")
            self.assertEqual(payload["delegated"]["perception_id"], "delegated.test")
            self.assertEqual(payload["delegated"]["executor_id"], "hermes")
            self.assertEqual(payload["delegated"]["consent_id"], "investigation.sources")
            self.assertEqual(payload["delegated"]["executor_report_ref"], "executor_report.sources")
            self.assertEqual(payload["delegated"]["executor_run_ref"], "hermes.run.sources")
            self.assertEqual(payload["delegated"]["sources"][0]["uri"], "https://example.com/agent-doc")
            self.assertTrue(payload["evidence_context"]["delegated"]["present"])
            self.assertEqual(
                payload["evidence_context"]["delegated"]["perception_id"],
                "delegated.test",
            )
            self.assertEqual(
                payload["delegated"]["sources"][0]["verification_status"],
                "reported_by_executor",
            )
            self.assertIn("Delegated sources: Hermes reported", rendered)
            self.assertIn("consent_id: investigation.sources", rendered)
            self.assertIn("executor_report_ref: executor_report.sources", rendered)
            self.assertIn("executor_run_ref: hermes.run.sources", rendered)
            self.assertIn("Hermes-reported source refs:", rendered)
            self.assertIn("verification_status=reported_by_executor", rendered)
            self.assertIn("Hermes found", rendered)

    def test_compact_composer_context_keeps_delegated_findings_without_raw_report(self) -> None:
        context = compact_composer_context(
            {
                "delegated_perception_context": {
                    **_delegated_context(),
                    "raw_executor_report": "drop this",
                    "sources": [
                        {
                            **_delegated_context()["sources"][0],
                            "raw_payload": "drop this too",
                        }
                    ],
                }
            }
        )

        delegated = context["delegated_perception_context"]
        self.assertEqual(delegated["source"], "delegated_perception")
        self.assertEqual(delegated["perception_id"], "delegated.test")
        self.assertEqual(delegated["findings"][0]["source_refs"], ["source.1"])
        self.assertEqual(delegated["sources"][0]["observed_by"], "hermes")
        self.assertEqual(context["evidence_context"]["delegated"]["executor_id"], "hermes")
        serialized = repr(delegated)
        self.assertNotIn("raw_executor_report", serialized)
        self.assertNotIn("raw_payload", serialized)

    def test_followup_links_delegated_context_to_turn_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            memory = load_workspace_memory_provider(tmp_dir)

            result = answer_general_follow_up(
                store=store,
                session_id="session.default",
                user_input="基于 Hermes 调查，两周内怎么做？",
                source_run=_decision_artifact(),
                action="answer_from_decision",
                config={"llm_provider": "deterministic"},
                context_payload={"delegated_perception_context": _delegated_context()},
                memory_provider=memory,
                now=NOW,
            )

            self.assertEqual(
                result.artifact["delegated_perception_context"]["perception_id"],
                "delegated.test",
            )
            turn = store.load_conversation_turn(result.artifact["turn_id"])
            self.assertEqual(
                turn["artifact_refs"]["delegated_perception"],
                ".spice/perceptions/delegated.test.json",
            )
            self.assertEqual(
                turn["metadata"]["delegated_perception_context"]["executor_id"],
                "hermes",
            )
            records = memory.query(namespace="general.evolution", limit=-1)
            self.assertEqual(
                records[0]["delegated_perception_context"]["perception_id"],
                "delegated.test",
            )
            self.assertEqual(
                records[0]["evidence_context"]["delegated"]["perception_id"],
                "delegated.test",
            )

    def test_llm_followup_composer_receives_delegated_recent_context(self) -> None:
        client = _FakeClient(json.dumps({"response": "Hermes reported this as external evidence."}))

        rendered, evidence = answer_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="What did Hermes find?",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
            context_payload={"delegated_perception_context": _delegated_context()},
        )

        self.assertIn("Hermes reported", rendered)
        prompt_payload = json.loads(client.requests[0].input_text)
        delegated = prompt_payload["facts"]["recent_context"]["delegated_perception_context"]
        self.assertEqual(delegated["perception_id"], "delegated.test")
        self.assertEqual(delegated["findings"][0]["source_refs"], ["source.1"])
        self.assertEqual(
            prompt_payload["facts"]["recent_context"]["evidence_context"]["delegated"]["perception_id"],
            "delegated.test",
        )
        self.assertEqual(
            evidence["composer_result"]["facts"]["decision_context"][
                "delegated_perception_context"
            ]["executor_id"],
            "hermes",
        )
        self.assertEqual(
            evidence["composer_result"]["facts"]["decision_context"]["evidence_context"]["delegated"][
                "executor_id"
            ],
            "hermes",
        )

    def test_refine_accepts_delegated_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW, full_loop_preview=False)

            result = refine_decision(
                "Adjust this using the delegated investigation.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                delegated_perception_context=_delegated_context(),
                delegated_perception={"perception_id": "delegated.test"},
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"][
                "delegated_perception_context"
            ]
            self.assertEqual(decision_ctx["perception_id"], "delegated.test")
            self.assertEqual(
                result.artifact["evidence_context"]["delegated"]["perception_id"],
                "delegated.test",
            )
            self.assertEqual(
                result.artifact["store_paths"]["delegated_perception"],
                ".spice/perceptions/delegated.test.json",
            )


def _delegated_context() -> dict[str, object]:
    return {
        "schema_version": "spice.delegated_perception_context.v1",
        "source": "delegated_perception",
        "perception_id": "delegated.test",
        "delegation_id": "delegation.test",
        "executor_id": "hermes",
        "scope": "read_only_investigation",
        "permission_mode": "read_only",
        "query": "Investigate external agent architecture.",
        "summary": "Hermes found external evidence about agent handoff boundaries.",
        "confidence": "medium",
        "consent_id": "investigation.test",
        "executor_report_ref": "executor_report.test",
        "executor_run_ref": "hermes.run.test",
        "findings": [
            {
                "finding_id": "finding.1",
                "text": "Hermes found that delegated investigation should stay read-only.",
                "confidence": 0.78,
                "source_refs": ["source.1"],
                "limitations": [],
            }
        ],
        "sources": [
            {
                "source_id": "source.1",
                "source_type": "url",
                "title": "Agent Doc",
                "uri": "https://example.com/agent-doc",
                "excerpt": "Delegated investigation is read-only.",
                "observed_by": "hermes",
                "accessed_at": "2026-05-13T08:00:00Z",
                "verification_status": "reported_by_executor",
            }
        ],
        "limitations": ["reported by executor; not directly inspected by Spice"],
    }


def _decision_artifact() -> dict[str, object]:
    return {
        "run_id": "run.test",
        "decision_id": "decision.test",
        "trace_ref": "trace.test",
        "selected_candidate_id": "candidate.a",
        "display_language": "zh",
        "decision_brief": {
            "display_language": "zh",
            "selected": {
                "candidate_id": "candidate.a",
                "title": "State-as-context",
                "recommendation": "Make runtime state first-class context.",
            },
            "why_this_won": ["It improves every later decision."],
            "execution": {"status": "advisory"},
            "next_actions": ["details", "execute", "refine"],
        },
        "compare_payload": {
            "display_language": "zh",
            "selected_recommendation": {
                "candidate_id": "candidate.a",
                "title": "State-as-context",
                "human_summary": "Make runtime state first-class context.",
            },
            "candidate_decisions": [
                {
                    "candidate_id": "candidate.a",
                    "title": "State-as-context",
                    "recommended_action": "Use active frames as context.",
                    "expected_result": "Decisions cite prior state.",
                    "execution_affordance": {"candidate_executable": False},
                }
            ],
            "why_not_the_others": [],
        },
    }


if __name__ == "__main__":
    unittest.main()
