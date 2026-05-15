from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from spice.perception.delegated import (
    INVESTIGATION_CONSENT_GRANTED,
    INVESTIGATION_CONSENT_PENDING,
    INVESTIGATION_CONSENT_REJECTED,
    build_investigation_consent,
    resolve_investigation_consent,
)
from spice.runtime.escalation_policy import (
    CONSENT_DELEGATED,
    ESCALATION_AWAIT_INVESTIGATION_CONSENT,
    ESCALATION_BLOCKED,
    ESCALATION_CONTINUE,
    ESCALATION_CREATE_INVESTIGATION_CONSENT,
    ESCALATION_REQUEST_EXECUTION_APPROVAL,
    ESCALATION_RUN_DELEGATED_PERCEPTION,
    ESCALATION_RUN_URL_PERCEPTION,
    ESCALATION_RUN_WORKSPACE_PERCEPTION,
    FINAL_STRATEGY_DELEGATED,
    FINAL_STRATEGY_EXECUTION_APPROVAL,
    FINAL_STRATEGY_LOCAL_THEN_DELEGATED,
    FINAL_STRATEGY_LOCAL_WORKSPACE,
    FINAL_STRATEGY_URL,
    STEP_DELEGATED,
    STEP_EXECUTION_APPROVAL,
    STEP_URL,
    STEP_WORKSPACE,
    build_investigation_consent_for_escalation,
    decide_runtime_escalation,
)
from spice.runtime.executor_capabilities import static_executor_capability_snapshot


NOW = datetime(2026, 5, 13, 10, 0, tzinfo=timezone.utc)


class RuntimeEscalationPolicyTests(unittest.TestCase):
    def test_local_workspace_strategy_runs_workspace_perception_first(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "context_strategy": "local_workspace",
                "workspace_query": "current state-as-context implementation",
            },
            config={"executor": "hermes"},
        )

        self.assertEqual(decision.action, ESCALATION_RUN_WORKSPACE_PERCEPTION)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_WORKSPACE)
        self.assertEqual(decision.steps, [STEP_WORKSPACE])
        self.assertEqual(decision.requires_consent, [])
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertEqual(decision.workspace_query, "current state-as-context implementation")
        self.assertFalse(decision.should_create_investigation_consent)

    def test_local_workspace_strategy_continues_when_context_exists(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "context_strategy": "local_workspace",
                "workspace_query": "current implementation",
            },
            config={"executor": "hermes"},
            workspace_context={
                "source": "workspace_perception",
                "summary": "Workspace perception found the implementation.",
                "facts": [{"text": "run_once injects workspace_context.", "confidence": 0.9}],
            },
        )

        self.assertEqual(decision.action, ESCALATION_CONTINUE)
        self.assertFalse(decision.should_run_workspace_perception)

    def test_url_strategy_runs_url_perception_before_answering(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "new_decision",
                "action": "new_intent",
                "context_strategy": "url",
                "url_query": "read the linked spec",
                "urls": ["https://example.com/spec"],
            },
            config={"executor": "hermes"},
        )

        self.assertEqual(decision.action, ESCALATION_RUN_URL_PERCEPTION)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_URL)
        self.assertEqual(decision.steps, [STEP_URL])
        self.assertTrue(decision.should_run_url_perception)
        self.assertEqual(decision.urls, ["https://example.com/spec"])

    def test_delegated_strategy_creates_investigation_consent_without_existing_consent(self) -> None:
        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_DELEGATED)
        self.assertEqual(decision.steps, [STEP_DELEGATED])
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertEqual(decision.delegated_scope, "read_only_investigation")
        self.assertEqual(decision.permission_mode, "read_only")
        self.assertTrue(decision.should_create_investigation_consent)
        self.assertEqual(decision.executor_id, "hermes")
        self.assertEqual(decision.missing_capability_ids, [])
        self.assertEqual(decision.delegated_plan["scope"], "read_only_investigation")
        self.assertEqual(decision.delegated_plan["permission_mode"], "read_only")
        self.assertEqual(decision.delegated_plan["expected_output"], "findings_sources_limitations")

        consent = build_investigation_consent_for_escalation(decision, created_at=NOW)
        self.assertEqual(consent.status, INVESTIGATION_CONSENT_PENDING)
        self.assertEqual(consent.executor_id, "hermes")
        self.assertEqual(consent.permission_mode, "read_only")
        self.assertIn("write_file", consent.denied_actions)
        self.assertEqual(consent.metadata.get("final_strategy"), FINAL_STRATEGY_DELEGATED)
        self.assertEqual(consent.metadata.get("requires_consent"), [CONSENT_DELEGATED])
        self.assertEqual(consent.metadata.get("expected_output"), "findings_sources_limitations")
        self.assertEqual(consent.metadata.get("delegated_plan", {}).get("scope"), "read_only_investigation")

    def test_delegated_strategy_waits_for_pending_consent(self) -> None:
        consent = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            created_at=NOW,
        )

        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=consent,
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_AWAIT_INVESTIGATION_CONSENT)
        self.assertEqual(decision.consent_id, consent.consent_id)
        self.assertEqual(decision.consent_status, INVESTIGATION_CONSENT_PENDING)
        self.assertFalse(decision.should_run_delegated_perception)

    def test_delegated_strategy_runs_after_granted_consent(self) -> None:
        pending = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            created_at=NOW,
        )
        granted = resolve_investigation_consent(
            pending,
            status=INVESTIGATION_CONSENT_GRANTED,
            resolved_at=NOW + timedelta(seconds=5),
        )

        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=granted,
            now=NOW + timedelta(seconds=10),
        )

        self.assertEqual(decision.action, ESCALATION_RUN_DELEGATED_PERCEPTION)
        self.assertTrue(decision.should_run_delegated_perception)
        self.assertEqual(decision.consent_status, INVESTIGATION_CONSENT_GRANTED)

    def test_delegated_strategy_blocks_rejected_consent(self) -> None:
        pending = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            created_at=NOW,
        )
        rejected = resolve_investigation_consent(
            pending,
            status=INVESTIGATION_CONSENT_REJECTED,
            resolved_at=NOW + timedelta(seconds=5),
        )

        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=rejected,
            now=NOW + timedelta(seconds=10),
        )

        self.assertEqual(decision.action, ESCALATION_BLOCKED)
        self.assertIn("rejected", decision.blocked_reason)
        self.assertFalse(decision.should_run_delegated_perception)

    def test_delegated_strategy_blocks_expired_consent(self) -> None:
        pending = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            created_at=NOW,
            expires_in_sec=1,
        )
        granted = resolve_investigation_consent(
            pending,
            status=INVESTIGATION_CONSENT_GRANTED,
            resolved_at=NOW + timedelta(seconds=1),
        )

        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=granted,
            now=NOW + timedelta(seconds=30),
        )

        self.assertEqual(decision.action, ESCALATION_BLOCKED)
        self.assertIn("expired", decision.blocked_reason)

    def test_delegated_strategy_blocks_non_read_only_consent(self) -> None:
        consent = build_investigation_consent(
            executor_id="hermes",
            query="latest agent workflow research",
            permission_mode="workspace_write",
            status=INVESTIGATION_CONSENT_GRANTED,
            created_at=NOW,
        )

        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=consent,
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_BLOCKED)
        self.assertIn("permission mode", decision.blocked_reason)

    def test_delegated_strategy_blocks_missing_executor_capability(self) -> None:
        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "dry_run"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_BLOCKED)
        self.assertIn("lacks delegated perception capability", decision.blocked_reason)
        self.assertEqual(decision.missing_capability_ids, ["web_research"])

    def test_repo_inspection_can_match_codex_static_baseline(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "delegated",
            "delegated_perception_query": "inspect repo architecture deeply",
            "suggested_capabilities": ["repo_inspection"],
        }

        decision = decide_runtime_escalation(route, config={"executor": "codex"}, now=NOW)

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.missing_capability_ids, [])
        self.assertTrue(any("repo_inspection:" in item for item in decision.matched_capability_ids))

    def test_local_then_delegated_runs_local_first(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "local_then_delegated_if_insufficient",
            "workspace_query": "current implementation",
            "delegated_perception_query": "research external patterns if local evidence is insufficient",
            "suggested_capabilities": ["repo_inspection"],
        }

        decision = decide_runtime_escalation(route, config={"executor": "codex"}, now=NOW)

        self.assertEqual(decision.action, ESCALATION_RUN_WORKSPACE_PERCEPTION)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(decision.steps, [STEP_WORKSPACE, STEP_DELEGATED])
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertEqual(decision.permission_mode, "read_only")
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertFalse(decision.should_create_investigation_consent)

    def test_local_then_delegated_continues_when_local_context_is_sufficient(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "local_then_delegated_if_insufficient",
            "workspace_query": "current implementation",
            "delegated_perception_query": "research external patterns if local evidence is insufficient",
            "suggested_capabilities": ["repo_inspection"],
        }

        decision = decide_runtime_escalation(
            route,
            config={"executor": "codex"},
            workspace_context={
                "summary": "Implementation evidence found.",
                "facts": [{"text": "Runtime policy exists.", "confidence": 0.9}],
                "files_read": [{"path": "spice/runtime/escalation_policy.py"}],
            },
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CONTINUE)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_WORKSPACE)
        self.assertEqual(decision.steps, [STEP_WORKSPACE])
        self.assertEqual(decision.requires_consent, [])
        self.assertIn("sufficient", decision.reason)

    def test_external_requirement_in_nested_route_policy_still_escalates_after_local_context(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "local_then_delegated_if_insufficient",
            "workspace_query": "current implementation",
            "delegated_perception_query": "research current external examples",
            "suggested_capabilities": ["web_research"],
            "raw": {
                "route_merge_policy": {
                    "forced_by": ["repo_evidence_requirement", "external_evidence_requirement"],
                },
            },
        }

        decision = decide_runtime_escalation(
            route,
            config={"executor": "hermes"},
            workspace_context={
                "summary": "Implementation evidence found.",
                "facts": [{"text": "Runtime policy exists.", "confidence": 0.9}],
                "files_read": [{"path": "spice/runtime/escalation_policy.py"}],
            },
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(decision.steps, [STEP_WORKSPACE, STEP_DELEGATED])
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertEqual(decision.forced_by, ["explicit_repo", "external_comparison_requested"])
        self.assertTrue(decision.should_create_investigation_consent)

    def test_top_level_route_merge_forced_by_still_escalates_after_local_context(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "local_then_delegated_if_insufficient",
            "workspace_query": "current implementation",
            "delegated_perception_query": "research current external examples",
            "suggested_capabilities": ["web_research"],
            "forced_by": ["repo_evidence_requirement", "external_evidence_requirement"],
        }

        decision = decide_runtime_escalation(
            route,
            config={"executor": "hermes"},
            workspace_context={
                "summary": "Implementation evidence found.",
                "facts": [{"text": "Runtime policy exists.", "confidence": 0.9}],
                "files_read": [{"path": "spice/runtime/escalation_policy.py"}],
            },
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(decision.forced_by, ["explicit_repo", "external_comparison_requested"])

    def test_local_then_delegated_escalates_after_insufficient_local_context(self) -> None:
        route = {
            "route": "follow_up",
            "action": "answer_from_decision",
            "context_strategy": "local_then_delegated_if_insufficient",
            "workspace_query": "current implementation",
            "delegated_perception_query": "research external patterns if local evidence is insufficient",
            "suggested_capabilities": ["repo_inspection"],
        }

        decision = decide_runtime_escalation(
            route,
            config={"executor": "codex"},
            workspace_context={
                "summary": "Workspace perception skipped: insufficient evidence.",
                "facts": [{"text": "No files inspected.", "confidence": 0.0}],
            },
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertEqual(decision.permission_mode, "read_only")
        self.assertTrue(decision.should_create_investigation_consent)

    def test_execution_request_always_uses_execution_approval_boundary(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "execution_request",
                "action": "execute_selected",
                "context_strategy": "delegated",
                "needs_delegated_perception": True,
                "delegated_perception_query": "do the thing",
            },
            config={"executor": "hermes"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_REQUEST_EXECUTION_APPROVAL)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_EXECUTION_APPROVAL)
        self.assertEqual(decision.steps, [STEP_EXECUTION_APPROVAL])
        self.assertEqual(decision.requires_consent, [])
        self.assertEqual(decision.permission_mode, "")
        self.assertTrue(decision.requires_execution_approval)
        self.assertFalse(decision.should_create_investigation_consent)
        self.assertFalse(decision.should_run_delegated_perception)

    def test_planner_none_cannot_override_explicit_workspace_context_flag(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "needs_workspace_context": True,
                "workspace_query": "current repo implementation",
                "perception_plan": {
                    "needs_perception": False,
                    "perception_strategy": "none",
                },
            },
            config={"executor": "hermes"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_RUN_WORKSPACE_PERCEPTION)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_WORKSPACE)
        self.assertEqual(decision.steps, [STEP_WORKSPACE])
        self.assertEqual(decision.workspace_query, "current repo implementation")

    def test_nested_planner_delegated_strategy_creates_read_only_consent(self) -> None:
        decision = decide_runtime_escalation(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "perception_plan": {
                    "needs_perception": True,
                    "perception_strategy": "delegated",
                    "reason": "requires external comparison",
                    "delegated_plan": {
                        "query": "research how mature agents handle read-only routing",
                        "requested_capabilities": ["web_research", "repo_inspection"],
                    },
                },
            },
            config={"executor": "hermes"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_DELEGATED)
        self.assertEqual(decision.steps, [STEP_DELEGATED])
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertEqual(decision.delegated_perception_query, "research how mature agents handle read-only routing")
        self.assertEqual(decision.suggested_capabilities, ["web_research", "repo_inspection"])
        self.assertEqual(decision.delegated_plan["query"], "research how mature agents handle read-only routing")
        self.assertEqual(decision.delegated_plan["requested_capabilities"], ["web_research", "repo_inspection"])
        self.assertEqual(decision.delegated_plan["scope"], "read_only_investigation")
        self.assertEqual(decision.delegated_plan["permission_mode"], "read_only")
        self.assertEqual(decision.delegated_plan["expected_output"], "findings_sources_limitations")
        self.assertEqual(decision.delegated_scope, "read_only_investigation")
        self.assertEqual(decision.permission_mode, "read_only")

    def test_malformed_capability_snapshot_blocks_delegated_perception(self) -> None:
        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            executor_capabilities={"executor_id": "hermes", "source": "bogus"},
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_BLOCKED)
        self.assertIn("malformed", " ".join(decision.limitations))

    def test_direct_capability_snapshot_can_authorize_delegated_perception(self) -> None:
        decision = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "dry_run"},
            executor_capabilities=static_executor_capability_snapshot("hermes"),
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.executor_id, "hermes")


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


if __name__ == "__main__":
    unittest.main()
