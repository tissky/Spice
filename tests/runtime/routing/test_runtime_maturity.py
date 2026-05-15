from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from spice.perception import (
    INVESTIGATION_CONSENT_REJECTED,
    build_delegated_perception_artifact,
    build_evidence_context,
    build_investigation_consent,
    resolve_investigation_consent,
)
from spice.runtime.escalation_policy import (
    CONSENT_DELEGATED,
    ESCALATION_BLOCKED,
    ESCALATION_CREATE_INVESTIGATION_CONSENT,
    ESCALATION_RUN_DELEGATED_PERCEPTION,
    FINAL_STRATEGY_LOCAL_THEN_DELEGATED,
    STEP_DELEGATED,
    STEP_WORKSPACE,
    decide_runtime_escalation,
)
from spice.runtime.intent_perception_planner import (
    PERCEPTION_STRATEGY_DELEGATED,
    PERCEPTION_STRATEGY_LOCAL_WORKSPACE,
    PERCEPTION_STRATEGY_MIXED,
    PERCEPTION_STRATEGY_URL,
    planner_result_from_semantic_payload,
    runtime_context_strategy_for_perception_strategy,
)
from spice.runtime.response_depth import resolve_response_depth_budget


NOW = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)


class RuntimeMaturityTests(unittest.TestCase):
    def test_planner_normalizes_workspace_url_delegated_and_mixed_strategies(self) -> None:
        cases = [
            ("local_workspace", PERCEPTION_STRATEGY_LOCAL_WORKSPACE, "local_workspace"),
            ("url", PERCEPTION_STRATEGY_URL, "url"),
            ("delegated", PERCEPTION_STRATEGY_DELEGATED, "delegated"),
            ("mixed", PERCEPTION_STRATEGY_MIXED, "local_then_delegated_if_insufficient"),
        ]

        for raw_strategy, expected_strategy, expected_runtime_strategy in cases:
            with self.subTest(strategy=raw_strategy):
                result = planner_result_from_semantic_payload(
                    {
                        "intent": {"intent_kind": "follow_up", "answer_mode": "report"},
                        "perception_plan": {
                            "needs_perception": True,
                            "perception_strategy": raw_strategy,
                            "evidence_requirement": "required",
                            "workspace_plan": {"query": "current implementation"},
                            "url_plan": {"query": "linked spec"},
                            "delegated_plan": {"query": "external investigation"},
                            "reason": "maturity test",
                        },
                    }
                )

                self.assertTrue(result.perception_plan.needs_perception)
                self.assertEqual(result.perception_plan.perception_strategy, expected_strategy)
                self.assertEqual(
                    runtime_context_strategy_for_perception_strategy(
                        result.perception_plan.perception_strategy
                    ),
                    expected_runtime_strategy,
                )

    def test_local_insufficient_escalates_to_investigation_consent(self) -> None:
        decision = decide_runtime_escalation(
            _local_then_delegated_route(),
            config={"executor": "hermes"},
            workspace_context={
                "source": "workspace_perception",
                "summary": "Workspace perception could not find enough evidence.",
                "facts": [],
                "limitations": ["insufficient local evidence"],
            },
            now=NOW,
        )

        self.assertEqual(decision.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertEqual(decision.final_strategy, FINAL_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(decision.steps, [STEP_WORKSPACE, STEP_DELEGATED])
        self.assertEqual(decision.requires_consent, [CONSENT_DELEGATED])
        self.assertTrue(decision.should_create_investigation_consent)
        self.assertFalse(decision.requires_execution_approval)

    def test_rejected_investigation_consent_never_handoffs(self) -> None:
        pending = build_investigation_consent(
            executor_id="hermes",
            query="research external agent routing",
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
        self.assertFalse(decision.should_run_delegated_perception)
        self.assertFalse(decision.requires_execution_approval)
        self.assertIn("rejected", decision.blocked_reason)

    def test_delegated_finding_without_source_is_low_confidence(self) -> None:
        artifact = build_delegated_perception_artifact(
            executor_id="hermes",
            query="research decision runtime maturity",
            findings=[
                {
                    "finding_id": "finding.unsourced",
                    "text": "Unsourced delegated finding should not be trusted strongly.",
                    "confidence": 0.92,
                    "source_refs": [],
                }
            ],
            sources=[],
            created_at=NOW,
        ).to_payload()

        self.assertEqual(artifact["findings"][0]["confidence"], 0.35)
        self.assertIn("no_source_refs", artifact["findings"][0]["limitations"])

    def test_evidence_context_merges_workspace_url_and_delegated_sources(self) -> None:
        context = build_evidence_context(
            requirements={
                "requires_evidence": True,
                "evidence_domain": "mixed",
                "answer_mode": "report",
            },
            workspace_context={
                "source": "workspace_perception",
                "perception_id": "workspace.maturity",
                "summary": "Workspace read runtime files.",
                "files_read": [{"path": "spice/runtime/run_once.py", "chars_read": 900}],
                "facts": [{"text": "run_once accepts evidence context."}],
            },
            url_context={
                "source": "url_perception",
                "perception_id": "url.maturity",
                "documents": [{"url": "https://example.com/spec", "title": "Spec"}],
                "facts": [{"text": "The linked spec requires source tracking."}],
            },
            delegated_perception_context={
                "source": "delegated_perception",
                "perception_id": "delegated.maturity",
                "executor_id": "hermes",
                "confidence": "medium",
                "findings": [
                    {
                        "finding_id": "finding.hermes.1",
                        "text": "Hermes reported mature agents separate read-only investigation from execution.",
                        "source_refs": ["source.hermes.1"],
                    }
                ],
                "sources": [
                    {
                        "source_id": "source.hermes.1",
                        "uri": "https://example.com/hermes",
                        "observed_by": "hermes",
                    }
                ],
            },
        )

        self.assertTrue(context["workspace"]["present"])
        self.assertTrue(context["url"]["present"])
        self.assertTrue(context["delegated"]["present"])
        self.assertEqual(context["requirements"]["evidence_domain"], "mixed")
        self.assertEqual(context["confidence"], "medium")
        self.assertEqual(
            {source["observed_by"] for source in context["sources"]},
            {"spice", "hermes"},
        )
        self.assertEqual(len(context["sources"]), 3)
        self.assertEqual(
            {finding["observed_by"] for finding in context["findings"]},
            {"spice", "hermes"},
        )

    def test_response_depth_gets_longer_for_repo_report_context(self) -> None:
        normal = resolve_response_depth_budget(answer_mode="normal")
        repo_report = resolve_response_depth_budget(
            evidence_context={
                "requirements": {"evidence_domain": "repo", "answer_mode": "report"},
                "workspace": {"present": True, "source_count": 4},
            }
        )

        self.assertEqual(repo_report.answer_mode, "report")
        self.assertGreater(repo_report.max_tokens or 0, normal.max_tokens or 0)
        self.assertGreater(repo_report.max_chars, normal.max_chars)

    def test_delegated_perception_boundary_does_not_create_execution_approval(self) -> None:
        without_consent = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            now=NOW,
        )
        pending = build_investigation_consent(
            executor_id="hermes",
            query="research external agent routing",
            created_at=NOW,
        )
        granted = resolve_investigation_consent(
            pending,
            status="granted",
            resolved_at=NOW + timedelta(seconds=3),
        )
        with_consent = decide_runtime_escalation(
            _delegated_route(),
            config={"executor": "hermes"},
            investigation_consent=granted,
            now=NOW + timedelta(seconds=5),
        )

        self.assertEqual(without_consent.action, ESCALATION_CREATE_INVESTIGATION_CONSENT)
        self.assertFalse(without_consent.requires_execution_approval)
        self.assertEqual(with_consent.action, ESCALATION_RUN_DELEGATED_PERCEPTION)
        self.assertTrue(with_consent.should_run_delegated_perception)
        self.assertFalse(with_consent.requires_execution_approval)


def _delegated_route() -> dict[str, object]:
    return {
        "route": "follow_up",
        "action": "answer_from_decision",
        "context_strategy": "delegated",
        "delegated_perception_query": "research external agent routing",
        "delegated_perception_reason": "requires external read-only investigation",
        "suggested_capabilities": ["web_research"],
        "delegated_plan": {
            "executor_id": "hermes",
            "scope": "read_only_investigation",
            "permission_mode": "read_only",
            "query": "research external agent routing",
            "requested_capabilities": ["web_research"],
            "expected_output": "findings_sources_limitations",
        },
    }


def _local_then_delegated_route() -> dict[str, object]:
    return {
        "route": "follow_up",
        "action": "answer_from_decision",
        "context_strategy": "local_then_delegated_if_insufficient",
        "workspace_query": "inspect current implementation",
        "delegated_perception_query": "research external agent routing if local evidence is insufficient",
        "delegated_perception_reason": "local evidence may be insufficient",
        "suggested_capabilities": ["web_research"],
    }


if __name__ == "__main__":
    unittest.main()
