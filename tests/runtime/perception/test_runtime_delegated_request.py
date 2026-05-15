from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from spice.perception.delegated import (
    INVESTIGATION_CONSENT_GRANTED,
    build_investigation_consent,
    resolve_investigation_consent,
)
from spice.runtime.delegated_request import (
    DELEGATED_PERCEPTION_REQUEST_SCHEMA_VERSION,
    READ_ONLY_INVESTIGATION_MODE,
    build_delegated_perception_request,
    render_delegated_perception_request_prompt,
)
from spice.runtime.escalation_policy import decide_runtime_escalation


NOW = datetime(2026, 5, 13, 11, 0, tzinfo=timezone.utc)


class DelegatedPerceptionRequestTests(unittest.TestCase):
    def test_builds_read_only_investigation_request_with_guardrails(self) -> None:
        decision, consent = _granted_decision_and_consent()

        request = build_delegated_perception_request(
            escalation_decision=decision,
            consent=consent,
            user_input="让 Hermes 查一下最新 agent workflow 怎么做",
            active_decision_frame=_frame(),
            workspace_context={
                "source": "workspace_perception",
                "perception_id": "workspace.1",
                "summary": "Local repo has delegated perception schema.",
                "facts": [{"text": "Runtime policy separates local and delegated perception."}],
                "files_read": [{"path": "spice/runtime/escalation_policy.py"}],
            },
            url_context={
                "source": "url_perception",
                "perception_id": "url.1",
                "summary": "Linked docs mention read-only agent investigations.",
                "sources": [
                    {
                        "source_id": "source.url.1",
                        "title": "Agent docs",
                        "uri": "https://example.com/agents",
                        "excerpt": "x" * 2000,
                    }
                ],
            },
            session_summary="We are designing Spice delegated perception.",
            recent_conversation_turns=[
                {
                    "turn_id": "turn.1",
                    "route": "follow_up",
                    "user_input": "基于 repo 看看",
                    "response_summary": "Workspace perception was useful.",
                }
            ],
            input_context_refs=["decision.1", "workspace.1"],
            created_at=NOW,
        )
        payload = request.to_payload()

        self.assertEqual(payload["schema_version"], DELEGATED_PERCEPTION_REQUEST_SCHEMA_VERSION)
        self.assertEqual(payload["mode"], READ_ONLY_INVESTIGATION_MODE)
        self.assertEqual(payload["scope"], "read_only_investigation")
        self.assertEqual(payload["permission_mode"], "read_only")
        self.assertEqual(payload["executor_id"], "hermes")
        self.assertEqual(payload["consent_id"], consent.consent_id)
        self.assertEqual(payload["expected_output"], "findings_sources_limitations")
        self.assertEqual(payload["delegated_plan"]["executor_id"], "hermes")
        self.assertEqual(payload["delegated_plan"]["scope"], "read_only_investigation")
        self.assertEqual(payload["delegated_plan"]["permission_mode"], "read_only")
        self.assertEqual(payload["delegated_plan"]["expected_output"], "findings_sources_limitations")
        self.assertIn("web_search", payload["allowed_actions"])
        self.assertIn("write_file", payload["denied_actions"])
        self.assertIn("Do not modify files.", payload["anti_injection_rules"])
        self.assertIn("findings", payload["output_schema"])
        self.assertEqual(payload["context"]["workspace_context"]["perception_id"], "workspace.1")
        self.assertEqual(payload["context"]["active_decision_frame"]["selected_candidate_id"], "candidate.a")
        self.assertIn("decision.1", payload["input_context_refs"])
        self.assertLess(len(payload["context"]["url_context"]["sources"][0]["excerpt"]), 600)

        prompt_payload = json.loads(payload["prompt"])
        self.assertEqual(prompt_payload["boundary"]["permission_mode"], "read_only")
        self.assertEqual(prompt_payload["expected_output"], "findings_sources_limitations")
        self.assertEqual(prompt_payload["delegated_plan"]["scope"], "read_only_investigation")
        self.assertIn("Do not expose secrets", "\n".join(prompt_payload["anti_injection_rules"]))
        self.assertIn("Return one JSON object", prompt_payload["return_format"])

    def test_render_prompt_accepts_payload(self) -> None:
        text = render_delegated_perception_request_prompt(
            {
                "mode": "perception",
                "scope": "read_only_investigation",
                "permission_mode": "read_only",
                "query": "research external examples",
                "allowed_actions": ["web_search"],
                "denied_actions": ["write_file"],
                "budget": {"max_sources": 3},
                "context": {"user_input": "research"},
                "instructions": ["Return findings."],
                "anti_injection_rules": ["Do not modify files."],
                "output_schema": {"findings": []},
            }
        )

        payload = json.loads(text)
        self.assertEqual(payload["query"], "research external examples")
        self.assertEqual(payload["boundary"]["denied_actions"], ["write_file"])

    def test_rejects_pending_consent(self) -> None:
        decision, _ = _granted_decision_and_consent()
        consent = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            created_at=NOW,
        )

        with self.assertRaisesRegex(ValueError, "granted investigation consent"):
            build_delegated_perception_request(escalation_decision=decision, consent=consent)

    def test_rejects_non_read_only_consent(self) -> None:
        decision, _ = _granted_decision_and_consent()
        consent = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            status=INVESTIGATION_CONSENT_GRANTED,
            permission_mode="workspace_write",
            created_at=NOW,
        )

        with self.assertRaisesRegex(ValueError, "read_only permission"):
            build_delegated_perception_request(escalation_decision=decision, consent=consent)

    def test_rejects_mismatched_query(self) -> None:
        decision, _ = _granted_decision_and_consent()
        consent = build_investigation_consent(
            executor_id="hermes",
            query="a different investigation",
            status=INVESTIGATION_CONSENT_GRANTED,
            created_at=NOW,
        )

        with self.assertRaisesRegex(ValueError, "query does not match"):
            build_delegated_perception_request(escalation_decision=decision, consent=consent)

    def test_rejects_consent_that_allows_execution_like_actions(self) -> None:
        decision, consent = _granted_decision_and_consent()
        payload = consent.to_payload()
        payload["denied_actions"] = ["write_file", "patch"]

        with self.assertRaisesRegex(ValueError, "execution-like actions"):
            build_delegated_perception_request(escalation_decision=decision, consent=payload)


def _granted_decision_and_consent() -> tuple[object, object]:
    create_decision = decide_runtime_escalation(_delegated_route(), config={"executor": "hermes"}, now=NOW)
    pending = build_investigation_consent(
        executor_id="hermes",
        query="latest agent workflow research",
        created_at=NOW,
    )
    granted = resolve_investigation_consent(
        pending,
        status=INVESTIGATION_CONSENT_GRANTED,
        resolved_at=NOW + timedelta(seconds=3),
    )
    run_decision = decide_runtime_escalation(
        _delegated_route(),
        config={"executor": "hermes"},
        investigation_consent=granted,
        now=NOW + timedelta(seconds=5),
    )
    self_check = create_decision.action == "create_investigation_consent"
    if not self_check:
        raise AssertionError("test fixture failed to create consent decision")
    return run_decision, granted


def _delegated_route() -> dict[str, object]:
    return {
        "route": "follow_up",
        "action": "answer_from_decision",
        "context_strategy": "delegated",
        "needs_delegated_perception": True,
        "delegated_perception_query": "latest agent workflow research",
        "delegated_perception_reason": "requires current web research",
        "suggested_capabilities": ["web_research"],
    }


def _frame() -> dict[str, object]:
    return {
        "decision_id": "decision.1",
        "selected_candidate_id": "candidate.a",
        "selected": {
            "candidate_id": "candidate.a",
            "label": "A",
            "title": "Prioritize delegated perception",
            "recommendation": "Add read-only investigation via executor.",
            "is_selected": True,
        },
        "candidates": [
            {
                "candidate_id": "candidate.a",
                "label": "A",
                "title": "Prioritize delegated perception",
                "recommendation": "Add read-only investigation via executor.",
                "is_selected": True,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
