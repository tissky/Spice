from __future__ import annotations

import json
import tempfile
import unittest
from uuid import uuid4

from spice.llm.core import LLMRequest, LLMResponse, LLMStreamChunk
from spice.runtime import run_once, setup_workspace
from spice.runtime.response_composer import (
    compose_decision_response_from_runtime_config,
    compose_decision_response_with_llm,
    response_composer_facts,
)


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


class RuntimeResponseComposerTests(unittest.TestCase):
    def test_response_composer_facts_include_selected_simulation_and_execution(self) -> None:
        artifact = _decision_artifact()

        facts = response_composer_facts(artifact)

        self.assertEqual(facts["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(facts["simulation"]["expected_outcome"], "A outcome")
        self.assertEqual(facts["execution"]["status"], "advisory")
        self.assertIn("details", facts["allowed_next_actions"][0])
        self.assertEqual(facts["why_not_others"][0]["candidate_id"], "candidate.b")

    def test_response_composer_facts_include_compact_decision_context(self) -> None:
        artifact = _decision_artifact()

        facts = response_composer_facts(
            artifact,
            context_payload={
                "active_decision_frame": {
                    "decision_id": "decision.previous",
                    "selected_candidate_id": "candidate.previous",
                    "selected": {"candidate_id": "candidate.previous", "title": "Previous choice"},
                },
                "recent_conversation_turns": [
                    {
                        "turn_id": "turn.previous",
                        "route": "follow_up",
                        "user_input": "Make it smaller.",
                    }
                ],
                "session_summary": {"summary_text": "The user prefers minimal scoped plans."},
                "executor_affordance": {"executor": "hermes", "status": "available"},
            },
        )

        context = facts["decision_context"]
        self.assertEqual(context["active_decision_frame"]["decision_id"], "decision.previous")
        self.assertEqual(context["recent_conversation_turns"][0]["user_input"], "Make it smaller.")
        self.assertIn("minimal scoped plans", context["memory_summary"]["summary_text"])
        self.assertEqual(context["executor_affordance"]["executor"], "hermes")

    def test_llm_response_composer_returns_natural_response_without_mutating_facts(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "I would start with Option A.\n\n"
                        "It is the safest foundation, while B is useful but less urgent.\n\n"
                        "Execution stays advisory for now. Use details or execute if you want to continue."
                    )
                }
            )
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "session_summary": {"summary_text": "Keep continuity from the prior turn."},
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.response",
                    "summary": "The repo has response composer workspace injection.",
                    "facts": [{"text": "Response composer receives workspace_context."}],
                },
            },
        )

        self.assertEqual(result.status, "composed")
        self.assertEqual(result.composer_kind, "decision_response")
        self.assertIn("I would start with Option A", result.response_text)
        self.assertEqual(result.facts["selected"]["candidate_id"], "candidate.a")
        payload = result.to_payload()
        self.assertEqual(payload["schema_version"], "spice.composer_result.v1")
        self.assertEqual(payload["composer_kind"], "decision_response")
        self.assertEqual(payload["facts"]["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(artifact["decision_brief"]["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(client.requests[0].task_hook.value, "response_compose")
        self.assertIn("Do not change the selected option", client.requests[0].system_text)
        self.assertIn("Keep continuity from the prior turn", client.requests[0].input_text)
        prompt_payload = json.loads(client.requests[0].input_text)
        self.assertEqual(
            prompt_payload["output"],
            "Write only the user-facing natural-language response as plain text. Do not wrap it in JSON.",
        )
        self.assertEqual(client.requests[0].response_format_hint, "")
        self.assertIn("selected_candidate", prompt_payload["facts"])
        self.assertIn("why_won", prompt_payload["facts"])
        self.assertIn("why_not", prompt_payload["facts"])
        self.assertIn("simulation", prompt_payload["facts"])
        self.assertIn("execution_affordance", prompt_payload["facts"])
        self.assertIn("recent_context", prompt_payload["facts"])
        self.assertEqual(
            prompt_payload["facts"]["recent_context"]["workspace_context"]["perception_id"],
            "workspace.response",
        )
        self.assertNotIn("response_schema", prompt_payload)
        self.assertNotIn("schema_version", prompt_payload["facts"])

    def test_llm_response_composer_streams_then_validates_response(self) -> None:
        streamed: list[str] = []
        client = _FakeStreamingClient(
            [
                "I would start ",
                "with Option A. Execution stays advisory.",
            ]
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=_decision_artifact(),
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("I would start with Option A", result.response_text)
        self.assertEqual("".join(streamed), result.raw_output)
        streaming = result.metadata["streaming"]
        self.assertEqual(streaming["mode"], "provider_token_stream")
        self.assertTrue(streaming["valid"])
        self.assertTrue(streaming["displayed_to_user"])
        self.assertEqual(streaming["source"], "validated_streamed_composer_result")

    def test_llm_response_composer_suppresses_streamed_json_wrapper_until_parsed(self) -> None:
        streamed: list[str] = []
        client = _FakeStreamingClient(
            [
                '{"response": "I would start ',
                'with Option A. Execution stays advisory."}',
            ]
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=_decision_artifact(),
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("I would start with Option A", result.response_text)
        self.assertEqual(streamed, [])
        self.assertTrue(result.raw_output.startswith('{"response"'))
        streaming = result.metadata["streaming"]
        self.assertTrue(streaming["valid"])
        self.assertFalse(streaming["displayed_to_user"])
        self.assertEqual(streaming["raw_text_chunk_count"], 2)
        self.assertEqual(streaming["text_chunk_count"], 0)

    def test_llm_response_composer_stream_invalid_output_falls_back_with_raw_output(self) -> None:
        streamed: list[str] = []
        client = _FakeStreamingClient(["{\"response\": \"I recommend Option B instead.\"}"])

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=_decision_artifact(),
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.response_text, "I'd choose Option A.")
        self.assertEqual(result.fallback_reason, "invalid_composed_response")
        self.assertEqual(streamed, [])
        self.assertTrue(result.raw_output)
        streaming = result.metadata["streaming"]
        self.assertFalse(streaming["valid"])
        self.assertFalse(streaming["displayed_to_user"])
        self.assertEqual(streaming["fallback_reason"], "invalid_composed_response")

    def test_streaming_validation_failure_does_not_mutate_decision_facts(self) -> None:
        artifact = _decision_artifact()
        artifact["approval_id"] = "approval.a"
        artifact["decision_brief"]["execution"] = {  # type: ignore[index]
            "status": "approval_required",
            "approval_id": "approval.a",
            "summary": "approval required before execution",
        }
        selected = artifact["compare_payload"]["candidate_decisions"][0]  # type: ignore[index]
        selected["score"] = 0.91  # type: ignore[index]
        selected["execution_affordance"] = {  # type: ignore[index]
            "candidate_executable": True,
            "required_capability": "code_edit",
            "executor_capability_source": "static_baseline",
            "capability": {
                "matched_capability": "code_edit",
                "executor_has_required_capability": True,
            },
        }
        original_artifact = json.loads(json.dumps(artifact, sort_keys=True))
        streamed: list[str] = []
        client = _FakeStreamingClient(['{"response": "I recommend Option B and already executed approval.fake."}'])

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-stream-model",
            stream_callback=streamed.append,
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.response_text, "I'd choose Option A.")
        self.assertTrue(result.raw_output)
        self.assertEqual(streamed, [])
        self.assertEqual(artifact, original_artifact)
        self.assertEqual(result.facts["selected"]["candidate_id"], "candidate.a")
        self.assertEqual(result.facts["selected"]["title"], "Option A")
        self.assertEqual(result.facts["execution"]["status"], "approval_required")
        self.assertEqual(result.facts["execution"]["approval_id"], "approval.a")
        self.assertEqual(result.facts["execution_affordance"]["required_capability"], "code_edit")
        streaming = result.metadata["streaming"]
        self.assertFalse(streaming["valid"])
        self.assertEqual(streaming["fallback_reason"], "invalid_composed_response")

    def test_llm_response_composer_prompt_uses_compact_capability_facts(self) -> None:
        artifact = _decision_artifact()
        selected = artifact["compare_payload"]["candidate_decisions"][0]  # type: ignore[index]
        selected["execution_affordance"] = {  # type: ignore[index]
            "candidate_executable": True,
            "executable": True,
            "blocked": False,
            "required_capability": "code_edit",
            "executor_capability_source": "static_baseline",
            "capability": {
                "required_capability": "code_edit",
                "executor_has_required_capability": True,
                "source": "static_baseline",
                "status": "available",
                "available_capability_ids": [
                    "repo_read",
                    "code_edit",
                    "test_run",
                    "workspace_write",
                    "terminal_command",
                    "raw_capability_that_should_not_enter_prompt",
                ],
                "limitations": ["Static baseline, not live tool inventory."],
                "matched_capability": "code_edit",
                "simulates_required_capability": False,
            },
            "executor": {
                "executor_id": "codex",
                "status": "ready",
                "real_executor": True,
            },
            "permission": {
                "required": "workspace_write",
                "configured": "workspace_write",
                "escalation_required": False,
                "escalation_supported": True,
            },
            "approval": {
                "required": True,
                "eligible_for_approval": True,
                "status": "approval_required_on_selection",
            },
        }
        client = _FakeClient(json.dumps({"response": "I recommend Option A first."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "composed")
        prompt_payload = json.loads(client.requests[0].input_text)
        affordance = prompt_payload["facts"]["execution_affordance"]
        self.assertEqual(affordance["required_capability"], "code_edit")
        self.assertEqual(affordance["capability"]["matched_capability"], "code_edit")
        self.assertNotIn("available_capability_ids", affordance["capability"])
        self.assertNotIn("limitations", affordance["capability"])
        self.assertNotIn("raw_capability_that_should_not_enter_prompt", repr(prompt_payload))

    def test_llm_response_composer_falls_back_on_invalid_output(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient('{"not_response": "missing"}')

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.response_text, "I'd choose Option A.")
        self.assertEqual(result.fallback_reason, "invalid_composed_response")
        self.assertEqual(result.to_payload()["facts"]["selected"]["candidate_id"], "candidate.a")
        self.assertIn("missing response text", result.error)
        self.assertTrue(result.raw_output)

    def test_llm_response_composer_rejects_non_selected_recommendation(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(json.dumps({"response": "I recommend Option B as the next move."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("non-selected candidate", result.error)

    def test_llm_response_composer_rejects_approval_state_change(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(json.dumps({"response": "Approval is ready for this decision."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("approval state", result.error)

    def test_llm_response_composer_rejects_fabricated_artifact_id(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(json.dumps({"response": "This is recorded under decision.fake."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("invented artifact id", result.error)

    def test_llm_response_composer_accepts_alias_and_plain_text(self) -> None:
        artifact = _decision_artifact()
        alias_result = compose_decision_response_with_llm(
            client=_FakeClient(json.dumps({"message": "Option A is still the right first move."})),  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )
        plain_result = compose_decision_response_with_llm(
            client=_FakeClient("Option A is still the right first move."),  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(alias_result.status, "composed")
        self.assertEqual(plain_result.status, "composed")
        self.assertIn("Option A", alias_result.response_text)
        self.assertIn("Option A", plain_result.response_text)

    def test_llm_response_composer_accepts_fenced_json_and_chinese_response(self) -> None:
        artifact = _decision_artifact()
        fenced_result = compose_decision_response_with_llm(
            client=_FakeClient('```json\n{"response": "Option A is still the right first move."}\n```'),  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )
        chinese_result = compose_decision_response_with_llm(
            client=_FakeClient("我建议先做 Option A，因为它能先稳定决策上下文。"),  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(fenced_result.status, "composed")
        self.assertIn("Option A", fenced_result.response_text)
        self.assertEqual(chinese_result.status, "composed")
        self.assertIn("先做 Option A", chinese_result.response_text)

    def test_llm_response_composer_allows_supported_workspace_fact_reference(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "I checked `spice/runtime/run_once.py`; the workspace facts say "
                        "function `run_once` already receives workspace context, so Option A stays the right move."
                    )
                }
            )
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={"workspace_context": _workspace_context()},
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("run_once.py", result.response_text)
        prompt_payload = json.loads(client.requests[0].input_text)
        self.assertTrue(
            any("workspace/repo facts" in item for item in prompt_payload["constraints"])
        )

    def test_llm_response_composer_rejects_unread_workspace_file_claim(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(
            json.dumps(
                {
                    "response": (
                        "I checked `spice/runtime/missing.py`; it already implements the feature, "
                        "so Option A is ready."
                    )
                }
            )
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={"workspace_context": _workspace_context()},
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("workspace file", result.error)

    def test_llm_response_composer_rejects_repo_inspection_without_workspace_context(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(
            json.dumps({"response": "I checked the repo and the code already implements this path."})
        )

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "fallback")
        self.assertIn("workspace inspection", result.error)

    def test_llm_response_composer_falls_back_on_malformed_empty_and_long_output(self) -> None:
        artifact = _decision_artifact()
        cases = [
            ('{"response": "unterminated"', "structured data"),
            ("   ", "empty"),
            (json.dumps({"response": "x" * 8001}), "too long"),
        ]

        for raw_output, error_fragment in cases:
            with self.subTest(error_fragment=error_fragment):
                result = compose_decision_response_with_llm(
                    client=_FakeClient(raw_output),  # type: ignore[arg-type]
                    artifact=artifact,
                    deterministic_text="I'd choose Option A.",
                    model_provider="fake",
                    model_id="fake-model",
                )

                self.assertEqual(result.status, "fallback")
                self.assertEqual(result.response_text, "I'd choose Option A.")
                self.assertEqual(result.fallback_reason, "invalid_composed_response")
                self.assertEqual(result.raw_output, raw_output)
                self.assertIn(error_fragment, result.error)

    def test_llm_response_composer_uses_normal_response_depth_by_default(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(json.dumps({"response": "I would start with Option A."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
        )

        self.assertEqual(result.status, "composed")
        self.assertEqual(client.requests[0].max_tokens, 3000)
        self.assertEqual(client.requests[0].timeout_sec, 75.0)
        self.assertEqual(result.metadata["response_depth"]["answer_mode"], "normal")
        self.assertEqual(result.metadata["response_depth"]["max_chars"], 8000)

    def test_llm_response_composer_uses_report_depth_for_evidence_context(self) -> None:
        artifact = _decision_artifact()
        long_response = "I would start with Option A.\n\n" + ("Evidence detail. " * 260)
        client = _FakeClient(json.dumps({"response": long_response}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
            context_payload={
                "evidence_context": {
                    "requirements": {
                        "answer_mode": "report",
                        "evidence_domain": "external",
                    },
                    "delegated": {"present": True, "source_count": 2},
                    "sources": [{"source_id": "source.1", "source_type": "url"}],
                }
            },
        )

        self.assertEqual(result.status, "composed")
        self.assertIn("Evidence detail", result.response_text)
        self.assertEqual(client.requests[0].max_tokens, 12000)
        self.assertEqual(client.requests[0].timeout_sec, 180.0)
        self.assertEqual(result.metadata["response_depth"]["answer_mode"], "report")
        prompt_payload = json.loads(client.requests[0].input_text)
        self.assertEqual(prompt_payload["facts"]["response_depth"]["answer_mode"], "report")

    def test_llm_response_composer_native_depth_leaves_provider_token_ceiling(self) -> None:
        artifact = _decision_artifact()
        client = _FakeClient(json.dumps({"response": "I would start with Option A."}))

        result = compose_decision_response_with_llm(
            client=client,  # type: ignore[arg-type]
            artifact=artifact,
            deterministic_text="I'd choose Option A.",
            model_provider="fake",
            model_id="fake-model",
            config={"response_depth": "native"},
        )

        self.assertEqual(result.status, "composed")
        self.assertIsNone(client.requests[0].max_tokens)
        self.assertTrue(result.metadata["response_depth"]["native"])

    def test_runtime_config_uses_deterministic_brief_when_provider_is_deterministic(self) -> None:
        artifact = _decision_artifact()

        result = compose_decision_response_from_runtime_config(
            config={"llm_provider": "deterministic", "llm_model": ""},
            artifact=artifact,
        )

        self.assertFalse(result.enabled)
        self.assertEqual(result.status, "disabled")
        self.assertEqual(result.composer_kind, "decision_response")
        self.assertEqual(result.facts["selected"]["candidate_id"], "candidate.a")
        self.assertIn("I'd choose Option A", result.response_text)

    def test_real_run_once_artifact_can_feed_response_composer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_once(
                "Compare state-as-context, proactive perception, and executor handoff.",
                project_root=tmp_dir,
                use_bars=False,
                full_loop_preview=False,
            )

        composed = compose_decision_response_from_runtime_config(
            config={"llm_provider": "deterministic", "llm_model": ""},
            artifact=result.artifact,
        )

        self.assertIn("I'd choose", composed.response_text)
        self.assertIn("selected", response_composer_facts(result.artifact))


def _decision_artifact() -> dict[str, object]:
    return {
        "run_id": "run.test",
        "decision_id": "decision.test",
        "selected_candidate_id": "candidate.a",
        "display_language": "en",
        "decision_brief": {
            "display_language": "en",
            "run_id": "run.test",
            "decision_id": "decision.test",
            "selected": {
                "candidate_id": "candidate.a",
                "title": "Option A",
                "recommendation": "Choose Option A first.",
            },
            "why_this_won": ["Outcome value carried meaningful weight."],
            "alternatives": [
                {
                    "candidate_id": "candidate.b",
                    "title": "Option B",
                    "summary": "Useful later.",
                }
            ],
            "execution": {
                "status": "advisory",
                "summary": "advisory only; no executor handoff requested",
            },
            "next_actions": [
                "details  expand the full Decision Card",
                "why      show why-not comparison",
                "execute  request approval only when executable",
            ],
            "warnings": [],
        },
        "compare_payload": {
            "candidate_decisions": [
                {
                    "candidate_id": "candidate.a",
                    "title": "Option A",
                    "simulation": {
                        "expected_outcome": "A outcome",
                        "downside": "A downside",
                        "success_signal": "A signal",
                        "confidence": 0.7,
                    },
                    "execution_affordance": {"candidate_executable": False},
                },
                {
                    "candidate_id": "candidate.b",
                    "title": "Option B",
                    "simulation": {
                        "expected_outcome": "B outcome",
                        "downside": "B downside",
                        "success_signal": "B signal",
                        "confidence": 0.6,
                    },
                    "execution_affordance": {"candidate_executable": False},
                },
            ],
            "why_not_the_others": [
                {
                    "candidate_id": "candidate.b",
                    "title": "Option B",
                    "reasons": [{"message": "Lower risk reduction."}],
                }
            ],
        },
    }


def _workspace_context() -> dict[str, object]:
    return {
        "source": "workspace_perception",
        "perception_id": "workspace.response.test",
        "summary": "run_once implements workspace_context injection for decision context.",
        "files_read": [{"path": "spice/runtime/run_once.py"}],
        "facts": [
            {
                "text": "function run_once already receives workspace context and passes it into compiled decision context.",
                "source_path": "spice/runtime/run_once.py",
            }
        ],
        "snippets": [
            {
                "path": "spice/runtime/run_once.py",
                "text": "def run_once(..., workspace_context=None): ...",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
