from __future__ import annotations

import unittest

from spice.runtime.intent_perception_planner import (
    EVIDENCE_REQUIREMENT_HELPFUL,
    EVIDENCE_REQUIREMENT_REQUIRED,
    INTENT_KIND_FOLLOW_UP,
    INTENT_KIND_INVESTIGATION_REQUEST,
    PERCEPTION_STRATEGY_DELEGATED,
    PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED,
    PERCEPTION_STRATEGY_NONE,
    planner_result_from_semantic_payload,
    runtime_context_strategy_for_perception_strategy,
)


class RuntimeIntentPerceptionPlannerTests(unittest.TestCase):
    def test_parses_nested_combined_planner_schema(self) -> None:
        result = planner_result_from_semantic_payload(
            {
                "intent": {"intent_kind": "follow_up", "answer_mode": "report"},
                "perception_plan": {
                    "needs_perception": True,
                    "perception_strategy": "local_then_delegated",
                    "evidence_requirement": "required",
                    "workspace_plan": {"query": "inspect current implementation"},
                    "delegated_plan": {
                        "executor_id": "hermes",
                        "scope": "read_only_investigation",
                        "query": "compare external examples",
                        "requested_capabilities": ["web_research"],
                    },
                    "reason": "local code and external examples are both useful",
                },
            }
        )

        self.assertEqual(result.intent.intent_kind, INTENT_KIND_FOLLOW_UP)
        self.assertEqual(result.intent.answer_mode, "report")
        self.assertTrue(result.perception_plan.needs_perception)
        self.assertEqual(result.perception_plan.perception_strategy, PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED)
        self.assertEqual(result.perception_plan.evidence_requirement, EVIDENCE_REQUIREMENT_REQUIRED)
        self.assertEqual(result.perception_plan.workspace_plan["query"], "inspect current implementation")
        self.assertEqual(result.perception_plan.delegated_plan["executor_id"], "hermes")
        self.assertEqual(result.perception_plan.delegated_plan["scope"], "read_only_investigation")
        self.assertEqual(result.perception_plan.delegated_plan["permission_mode"], "read_only")
        self.assertEqual(result.perception_plan.delegated_plan["expected_output"], "findings_sources_limitations")
        self.assertEqual(result.perception_plan.delegated_plan["requested_capabilities"], ["web_research"])
        self.assertEqual(
            runtime_context_strategy_for_perception_strategy(result.perception_plan.perception_strategy),
            "local_then_delegated_if_insufficient",
        )

    def test_legacy_route_fields_fall_back_to_planner_view(self) -> None:
        result = planner_result_from_semantic_payload(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "context_strategy": "delegated",
                "needs_delegated_perception": True,
                "delegated_perception_query": "research latest agent routing",
                "suggested_capabilities": ["web_research", "docs_review"],
            }
        )

        self.assertEqual(result.intent.intent_kind, INTENT_KIND_FOLLOW_UP)
        self.assertTrue(result.perception_plan.needs_perception)
        self.assertEqual(result.perception_plan.perception_strategy, PERCEPTION_STRATEGY_DELEGATED)
        self.assertEqual(result.perception_plan.evidence_requirement, EVIDENCE_REQUIREMENT_HELPFUL)
        self.assertEqual(result.perception_plan.delegated_plan["executor_id"], "hermes")
        self.assertEqual(result.perception_plan.delegated_plan["scope"], "read_only_investigation")
        self.assertEqual(result.perception_plan.delegated_plan["permission_mode"], "read_only")
        self.assertEqual(result.perception_plan.delegated_plan["expected_output"], "findings_sources_limitations")
        self.assertEqual(result.perception_plan.delegated_plan["query"], "research latest agent routing")
        self.assertEqual(result.perception_plan.delegated_plan["requested_capabilities"], ["web_research", "docs_review"])

    def test_investigation_intent_is_normalized_without_legacy_route(self) -> None:
        result = planner_result_from_semantic_payload(
            {
                "intent": {"intent_kind": "investigation_request", "answer_mode": "detailed"},
                "perception_plan": {
                    "needs_perception": True,
                    "perception_strategy": "delegated",
                    "evidence_requirement": "required",
                    "delegated_plan": {"query": "research Hermes read-only routing"},
                },
            }
        )

        self.assertEqual(result.intent.intent_kind, INTENT_KIND_INVESTIGATION_REQUEST)
        self.assertEqual(result.intent.answer_mode, "detailed")
        self.assertEqual(result.perception_plan.perception_strategy, PERCEPTION_STRATEGY_DELEGATED)
        self.assertEqual(result.perception_plan.delegated_plan["executor_id"], "hermes")
        self.assertEqual(result.perception_plan.delegated_plan["scope"], "read_only_investigation")
        self.assertEqual(result.perception_plan.delegated_plan["permission_mode"], "read_only")
        self.assertEqual(result.perception_plan.delegated_plan["expected_output"], "findings_sources_limitations")

    def test_empty_payload_defaults_to_no_perception(self) -> None:
        result = planner_result_from_semantic_payload({})

        self.assertFalse(result.perception_plan.needs_perception)
        self.assertEqual(result.perception_plan.perception_strategy, PERCEPTION_STRATEGY_NONE)


if __name__ == "__main__":
    unittest.main()
