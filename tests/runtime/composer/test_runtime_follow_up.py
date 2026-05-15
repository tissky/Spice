from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from spice.llm.core import LLMRequest, LLMResponse, LLMStreamChunk
from spice.runtime import setup_workspace
from spice.runtime.follow_up import (
    answer_general_follow_up,
    answer_general_follow_up_with_llm,
    answer_why_not_candidate,
    compose_general_follow_up_with_llm,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import load_workspace_memory_provider


NOW = datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc)


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
            latency_ms=0,
            request_id=f"req.{uuid4().hex}",
        )


class _FakeStreamingClient(_FakeClient):
    def __init__(self, chunks: list[str]) -> None:
        super().__init__("")
        self.chunks = chunks

    def stream(self, request: LLMRequest):
        self.requests.append(request)
        for index, text in enumerate(self.chunks):
            yield LLMStreamChunk(
                text=text,
                finish_reason="stop" if index == len(self.chunks) - 1 else "",
                raw_event={"id": "req.stream", "model": "fake-stream-model"},
            )


class RuntimeFollowUpTests(unittest.TestCase):
    def test_general_followup_uses_deterministic_answer_and_records_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            memory = load_workspace_memory_provider(tmp_dir)

            result = answer_general_follow_up(
                store=store,
                session_id="session.default",
                user_input="两周内怎么做？",
                source_run=_decision_artifact(display_language="zh"),
                action="answer_from_decision",
                config={"llm_provider": "deterministic"},
                memory_provider=memory,
                now=NOW,
            )

            self.assertIn("基于当前 Decision Card", result.rendered_text)
            self.assertEqual(result.artifact["action"], "answer_from_decision")
            turn = store.load_conversation_turn(result.artifact["turn_id"])
            self.assertEqual(turn["route"], "follow_up")
            self.assertEqual(turn["source_decision_id"], "decision.test")
            self.assertEqual(turn["source_candidate_id"], "candidate.a")
            self.assertEqual(turn["metadata"]["follow_up_action"], "answer_from_decision")
            composer_result = result.artifact["evidence"]["composer_result"]
            self.assertEqual(composer_result["schema_version"], "spice.composer_result.v1")
            self.assertEqual(composer_result["composer_kind"], "follow_up_response")
            self.assertEqual(composer_result["status"], "disabled")
            records = memory.query(namespace="general.evolution", limit=-1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["follow_up_type"], "answer_from_decision")
            self.assertEqual(records[0]["user_input"], "两周内怎么做？")

    def test_general_followup_links_workspace_context_to_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            workspace_context = {
                "source": "workspace_perception",
                "perception_id": "workspace.followup",
                "summary": "The repo has runtime workspace perception.",
                "files_read": [{"path": "spice/runtime/workspace_perception.py"}],
            }

            result = answer_general_follow_up(
                store=store,
                session_id="session.default",
                user_input="基于代码，两周内怎么做？",
                source_run=_decision_artifact(display_language="zh"),
                action="answer_from_decision",
                config={"llm_provider": "deterministic"},
                context_payload={"workspace_context": workspace_context},
                now=NOW,
            )

            self.assertEqual(result.artifact["workspace_context"]["perception_id"], "workspace.followup")
            turn = store.load_conversation_turn(result.artifact["turn_id"])
            self.assertEqual(
                turn["artifact_refs"]["workspace_perception"],
                ".spice/perceptions/workspace.followup.json",
            )
            self.assertEqual(
                turn["metadata"]["workspace_context"]["summary"],
                "The repo has runtime workspace perception.",
            )

    def test_llm_general_followup_composer_answers_from_decision_facts(self) -> None:
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "I would make the smallest two-week version: wire active frames into "
                        "candidate context first, then verify a follow-up can cite prior state."
                    )
                }
            )
        )

        rendered, evidence = answer_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="What is the two-week version?",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "recent_conversation_turns": [
                    {
                        "turn_id": "turn.previous",
                        "route": "follow_up",
                        "user_input": "Keep it solo-founder sized.",
                    }
                ],
                "session_summary": {"summary_text": "The user is optimizing for a two-week solo plan."},
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.followup.prompt",
                    "summary": "The repo has follow-up composer workspace injection.",
                    "facts": [{"text": "Follow-up composer receives workspace_context."}],
                },
            },
        )

        self.assertIn("smallest two-week version", rendered)
        self.assertEqual(evidence["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(evidence["llm"]["status"], "composed")
        self.assertEqual(evidence["composer_result"]["schema_version"], "spice.composer_result.v1")
        self.assertEqual(evidence["composer_result"]["composer_kind"], "follow_up_response")
        self.assertEqual(evidence["composer_result"]["facts"]["selected"]["candidate_id"], "candidate.a")
        self.assertIn("solo-founder", evidence["decision_context"]["recent_conversation_turns"][0]["user_input"])
        self.assertEqual(client.requests[0].task_hook.value, "response_compose")
        self.assertEqual(client.requests[0].max_tokens, 6000)
        self.assertIn("Do not change the winner", client.requests[0].system_text)
        self.assertIn("two-week solo plan", client.requests[0].input_text)
        prompt_payload = json.loads(client.requests[0].input_text)
        self.assertIn("selected_candidate", prompt_payload["facts"])
        self.assertIn("target_candidate", prompt_payload["facts"])
        self.assertIn("why_won", prompt_payload["facts"])
        self.assertIn("why_not", prompt_payload["facts"])
        self.assertIn("simulation", prompt_payload["facts"])
        self.assertIn("execution_affordance", prompt_payload["facts"])
        self.assertIn("recent_context", prompt_payload["facts"])
        self.assertEqual(prompt_payload["facts"]["response_depth"]["answer_mode"], "detailed")
        self.assertEqual(
            prompt_payload["facts"]["recent_context"]["workspace_context"]["perception_id"],
            "workspace.followup.prompt",
        )
        self.assertNotIn("allowed_actions", prompt_payload)
        self.assertNotIn("response_schema", prompt_payload)

    def test_why_not_followup_uses_llm_composer_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            client = _FakeClient(
                "Executor handoff is not wrong; it is just riskier before state context is stable."
            )

            with patch("spice.runtime.follow_up.build_candidate_expander_client", return_value=client):
                result = answer_why_not_candidate(
                    store=store,
                    session_id="session.default",
                    user_input="Why not executor handoff?",
                    source_run=_decision_artifact(),
                    candidate_id="candidate.b",
                    config={"llm_provider": "fake", "llm_model": "fake-model"},
                    now=NOW,
                )

            self.assertIn("not wrong", result.rendered_text)
            self.assertNotIn("I did not pick", result.rendered_text)
            evidence = result.artifact["evidence"]
            self.assertEqual(evidence["composer_result"]["status"], "composed")
            self.assertEqual(evidence["composer_result"]["facts"]["action"], "explain_why_not")
            self.assertEqual(evidence["target_candidate"]["candidate_id"], "candidate.b")
            prompt_payload = json.loads(client.requests[0].input_text)
            self.assertEqual(prompt_payload["facts"]["follow_up_action"], "explain_why_not")
            self.assertTrue(any("natural language" in item for item in prompt_payload["constraints"]))

    def test_general_followup_prompt_uses_top_candidate_summaries_only(self) -> None:
        source_run = _decision_artifact()
        source_run["compare_payload"]["candidate_decisions"] = [
            {
                "candidate_id": f"candidate.{index}",
                "title": f"Option {index}",
                "recommended_action": f"Do option {index}",
                "expected_result": f"Result {index}",
                "simulation": {
                    "expected_outcome": f"Outcome {index}",
                    "downside": f"Downside {index}",
                    "success_signal": f"Signal {index}",
                    "confidence": 0.5,
                    "raw_model_output": "raw simulation dump",
                },
                "execution_affordance": {"candidate_executable": False, "raw": "drop"},
            }
            for index in range(5)
        ]
        client = _FakeClient(json.dumps({"response": "I would keep the selected option and use the first tradeoff."}))

        answer_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="What is the smallest version?",
            source_run=source_run,
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )

        prompt_payload = json.loads(client.requests[0].input_text)
        visible = prompt_payload["facts"]["visible_candidates"]
        self.assertEqual(len(visible), 3)
        self.assertEqual(visible[0]["candidate_id"], "candidate.0")
        self.assertEqual(set(visible[0]["simulation"]), {"expected_outcome", "downside", "success_signal", "confidence"})
        serialized = repr(prompt_payload)
        self.assertNotIn("raw simulation dump", serialized)
        self.assertNotIn("raw_model_output", serialized)

    def test_general_followup_composer_returns_shared_contract(self) -> None:
        client = _FakeClient(json.dumps({"response": "Use the current decision as the base and keep it small."}))

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Make it smaller.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "composed")
        self.assertEqual(result.composer_kind, "follow_up_response")
        self.assertEqual(result.facts["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(result.to_payload()["schema_version"], "spice.composer_result.v1")

    def test_general_followup_composer_streams_then_validates_response(self) -> None:
        streamed: list[str] = []
        client = _FakeStreamingClient(
            [
                "Use the current decision ",
                "as the base and keep it small.",
            ]
        )

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Make it smaller.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("Use the current decision", result.response_text)
        self.assertEqual("".join(streamed), result.raw_output)
        self.assertEqual(client.requests[0].response_format_hint, "")
        streaming = result.metadata["streaming"]
        self.assertTrue(streaming["valid"])
        self.assertEqual(streaming["mode"], "provider_token_stream")

    def test_general_followup_composer_accepts_alias_and_plain_text(self) -> None:
        alias_result = compose_general_follow_up_with_llm(
            client=_FakeClient(json.dumps({"content": "Keep A, but make the first step smaller."})),  # type: ignore[arg-type]
            user_input="Make it smaller.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )
        plain_result = compose_general_follow_up_with_llm(
            client=_FakeClient("Keep A, but make the first step smaller."),  # type: ignore[arg-type]
            user_input="Make it smaller.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(alias_result.status, "composed")
        self.assertEqual(plain_result.status, "composed")

    def test_general_followup_composer_falls_back_when_target_candidate_is_missing(self) -> None:
        client = _FakeClient(json.dumps({"response": "Use the missing option."}))

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Give me Z.",
            source_run=_decision_artifact(),
            action="plan_candidate",
            candidate_id="candidate.z",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.fallback_reason, "missing_target_candidate")
        self.assertIn("cannot find that option", result.response_text)
        self.assertEqual(client.requests, [])

    def test_general_followup_composer_rejects_answering_why_not_with_selected_plan(self) -> None:
        client = _FakeClient(json.dumps({"response": "Plan for State-as-context: first step is to wire memory."}))

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Why not B?",
            source_run=_decision_artifact(),
            action="compare_alternative",
            candidate_id="candidate.b",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("why-not/compare", result.error)

    def test_general_followup_composer_rejects_fabricated_candidate_id(self) -> None:
        client = _FakeClient(json.dumps({"response": "I would create candidate.z for this."}))

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="What else?",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("invented artifact id", result.error)

    def test_general_followup_composer_fallback_records_raw_output(self) -> None:
        client = _FakeClient('{"not_response": true}')

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Make it smaller.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.raw_output, '{"not_response": true}')
        self.assertEqual(result.fallback_reason, "invalid_composed_response")
        self.assertIn("Based on the current Decision Card", result.response_text)

    def test_general_followup_composer_rejects_executable_claim_for_advisory_candidate(self) -> None:
        client = _FakeClient(json.dumps({"response": "Executor handoff is ready for approval and can execute now."}))

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Can we execute B?",
            source_run=_decision_artifact(),
            action="plan_candidate",
            candidate_id="candidate.b",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("advisory candidate", result.error)

    def test_general_followup_composer_rejects_unbacked_workspace_symbol_claim(self) -> None:
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "The smallest plan is to update function `compile_magic_context` first, "
                        "then keep State-as-context as the selected path."
                    )
                }
            )
        )

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="What is the smallest version?",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.followup.symbol",
                    "summary": "The repo has follow-up composer workspace injection.",
                    "files_read": [{"path": "spice/runtime/follow_up.py"}],
                    "facts": [
                        {
                            "text": "follow_up composer receives workspace_context.",
                            "source_path": "spice/runtime/follow_up.py",
                        }
                    ],
                }
            },
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("workspace symbol", result.error)

    def test_general_followup_composer_rejects_invented_workspace_file_claim(self) -> None:
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "I checked `spice/runtime/missing.py`; use that file as the first step "
                        "for State-as-context."
                    )
                }
            )
        )

        result = compose_general_follow_up_with_llm(
            client=client,  # type: ignore[arg-type]
            user_input="Give me the smallest repo-aware plan.",
            source_run=_decision_artifact(),
            action="answer_from_decision",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.followup.file",
                    "summary": "The repo has follow-up composer workspace injection.",
                    "files_read": [{"path": "spice/runtime/follow_up.py"}],
                    "facts": [
                        {
                            "text": "follow_up composer receives workspace_context.",
                            "source_path": "spice/runtime/follow_up.py",
                        }
                    ],
                }
            },
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("workspace file", result.error)

    def test_compare_alternative_followup_targets_visible_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            result = answer_general_follow_up(
                store=store,
                session_id="session.default",
                user_input="Could B be better?",
                source_run=_decision_artifact(),
                action="compare_alternative",
                candidate_id="candidate.b",
                config={"llm_provider": "deterministic"},
                now=NOW,
            )

            self.assertIn("Executor handoff could be better", result.rendered_text)
            self.assertEqual(result.artifact["source_candidate_id"], "candidate.b")
            self.assertEqual(result.artifact["evidence"]["target_candidate"]["candidate_id"], "candidate.b")


def _decision_artifact(*, display_language: str = "en") -> dict[str, object]:
    return {
        "run_id": "run.test",
        "decision_id": "decision.test",
        "trace_ref": "trace.test",
        "selected_candidate_id": "candidate.a",
        "display_language": display_language,
        "decision_brief": {
            "display_language": display_language,
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
            "display_language": display_language,
            "selected_recommendation": {
                "candidate_id": "candidate.a",
                "title": "State-as-context",
                "human_summary": "Make runtime state first-class context.",
            },
            "candidate_decisions": [
                {
                    "candidate_id": "candidate.a",
                    "title": "State-as-context",
                    "recommended_action": "Use active frames, summaries, and open loops as decision context.",
                    "expected_result": "Decisions cite prior state instead of starting from scratch.",
                    "execution_affordance": {"candidate_executable": False},
                    "simulation": {
                        "expected_outcome": "Follow-ups can cite prior state.",
                        "downside": "Less visible than a flashy feature.",
                        "success_signal": "A follow-up references the previous decision.",
                        "confidence": 0.7,
                    },
                },
                {
                    "candidate_id": "candidate.b",
                    "title": "Executor handoff",
                    "recommended_action": "Improve handoff to execution agents.",
                    "expected_result": "Decisions turn into approved execution.",
                    "execution_affordance": {"candidate_executable": False},
                    "simulation": {
                        "expected_outcome": "Handoffs become reliable.",
                        "downside": "Higher blast radius.",
                        "success_signal": "A task executes through approval.",
                        "confidence": 0.6,
                    },
                },
            ],
            "why_not_the_others": [
                {
                    "candidate_id": "candidate.b",
                    "title": "Executor handoff",
                    "reasons": [{"message": "Higher risk before state context is stable."}],
                }
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
