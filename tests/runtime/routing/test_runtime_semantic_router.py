from __future__ import annotations

import unittest
from unittest.mock import patch

from spice.llm.core import LLMRequest, LLMResponse
from spice.runtime.semantic_router import (
    route_semantic_input,
    route_semantic_input_from_runtime_config,
    route_semantic_input_with_llm,
    semantic_route_to_continuation,
)


class _FakeSemanticRouterClient:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            provider_id="openai",
            model_id="gpt-test",
            output_text=self.output_text,
            raw_payload={},
            finish_reason="stop",
            usage={},
            latency_ms=1,
            request_id="semantic-router-test",
        )


class SemanticRouterTests(unittest.TestCase):
    def test_no_llm_route_does_not_interpret_natural_followup(self) -> None:
        route = route_semantic_input("go with B", _frame())

        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertFalse(route.is_continuation)
        self.assertEqual(route.source, "none")
        self.assertIn("LLM semantic router", route.reason)

    def test_no_llm_route_does_not_interpret_followup_actions(self) -> None:
        why_not = route_semantic_input("why not B?", _frame())
        plan = route_semantic_input("give me A's plan", _frame())
        execute = route_semantic_input("execute B", _frame())

        for route in (why_not, plan, execute):
            self.assertEqual(route.route, "new_decision")
            self.assertEqual(route.action, "new_intent")
            self.assertFalse(route.is_continuation)

    def test_no_active_frame_routes_as_new_decision(self) -> None:
        route = route_semantic_input("What should I do about onboarding?", None)

        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertFalse(route.is_continuation)

    def test_llm_routes_natural_execution_request(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "execution_request", "action": "execute_selected", '
            '"is_continuation": true, "candidate_label": "B", "confidence": 0.86, '
            '"reason": "User asked to start implementing the selected decision."}'
        )

        route = route_semantic_input_with_llm("那就去干吧", _frame(), client=client)

        self.assertEqual(route.route, "execution_request")
        self.assertEqual(route.action, "execute_selected")
        self.assertTrue(route.is_continuation)
        self.assertEqual(route.candidate_id, "candidate.b")
        self.assertEqual(route.label, "B")
        self.assertEqual(route.source, "llm")
        self.assertAlmostEqual(route.confidence, 0.86)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0].metadata["purpose"], "semantic_route")

    def test_llm_routes_semantic_candidate_choice(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "choose_option", '
            '"is_continuation": true, "candidate_label": "B", '
            '"confidence": 0.8, "reason": "User chose the demo option."}'
        )

        route = route_semantic_input_with_llm("就选改 demo 那个", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "choose_option")
        self.assertEqual(route.candidate_id, "candidate.b")
        self.assertEqual(route.label, "B")

    def test_runtime_config_uses_llm_before_any_natural_shortcut(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "new_decision", "action": "new_intent", '
            '"is_continuation": false, "confidence": 0.9, '
            '"reason": "Treat this as a fresh decision."}'
        )

        with patch(
            "spice.runtime.semantic_router.build_candidate_expander_client",
            return_value=client,
        ) as build:
            route = route_semantic_input_from_runtime_config(
                "B",
                _frame(),
                config={"llm_provider": "openrouter", "llm_model": "test/model"},
            )

        build.assert_called_once()
        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertEqual(route.source, "llm")
        self.assertEqual(route.reason, "Treat this as a fresh decision.")

    def test_llm_routes_candidate_plan_followup(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "plan_candidate", '
            '"is_continuation": true, "candidate_label": "B", '
            '"confidence": 0.8, "reason": "User asked for a plan."}'
        )

        route = route_semantic_input_with_llm("给我 demo 那个方案的计划", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "plan_candidate")
        self.assertEqual(route.candidate_id, "candidate.b")

    def test_llm_routes_general_followup_answer_from_decision(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "confidence": 0.8, '
            '"reason": "User asked how to apply the selected decision in two weeks."}'
        )

        route = route_semantic_input_with_llm("两周内怎么做？", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "answer_from_decision")
        self.assertEqual(route.candidate_id, "candidate.a")
        self.assertEqual(route.source, "llm")

    def test_llm_routes_workspace_context_need_without_deciding_files(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "context_strategy": "local_workspace", '
            '"needs_workspace_context": true, '
            '"workspace_query": "current state-as-context implementation", '
            '"confidence": 0.82, "reason": "Needs current repo facts."}'
        )

        route = route_semantic_input_with_llm(
            "基于当前 repo 看看 state-as-context 实现到哪了",
            _frame(),
            client=client,
        )

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "answer_from_decision")
        self.assertEqual(route.context_strategy, "local_workspace")
        self.assertTrue(route.needs_workspace_context)
        self.assertEqual(route.workspace_query, "current state-as-context implementation")
        prompt = client.requests[0].input_text
        self.assertIn("context_strategy", prompt)
        self.assertIn("needs_workspace_context", prompt)
        self.assertIn("workspace_query", prompt)
        self.assertIn("Do not output file paths, tool calls", prompt)

    def test_llm_routes_nested_combined_intent_perception_planner_schema(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"intent": {"intent_kind": "follow_up", "answer_mode": "report"}, '
            '"perception_plan": {'
            '"needs_perception": true, '
            '"perception_strategy": "local_then_delegated", '
            '"evidence_requirement": "required", '
            '"workspace_plan": {"query": "current workspace perception implementation"}, '
            '"delegated_plan": {"query": "external examples if local evidence is insufficient", '
            '"requested_capabilities": ["web_research"]}, '
            '"reason": "needs local code and possible external comparison"}, '
            '"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "confidence": 0.83}'
        )

        route = route_semantic_input_with_llm(
            "基于当前实现，如果不够再让 Hermes 调查",
            _frame(),
            client=client,
        )

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "answer_from_decision")
        self.assertEqual(route.intent_kind, "follow_up")
        self.assertEqual(route.answer_mode, "report")
        self.assertEqual(route.context_strategy, "local_then_delegated_if_insufficient")
        self.assertTrue(route.needs_workspace_context)
        self.assertTrue(route.needs_delegated_perception)
        self.assertEqual(route.workspace_query, "current workspace perception implementation")
        self.assertEqual(route.delegated_perception_query, "external examples if local evidence is insufficient")
        self.assertEqual(route.suggested_capabilities, ["web_research"])
        self.assertEqual(route.perception_plan["perception_strategy"], "local_then_delegated")
        prompt = client.requests[0].input_text
        self.assertIn("intent_kind", prompt)
        self.assertIn("perception_plan", prompt)

    def test_nested_investigation_intent_without_legacy_route_uses_delegated_context(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"intent": {"intent_kind": "investigation_request", "answer_mode": "detailed"}, '
            '"perception_plan": {'
            '"needs_perception": true, '
            '"perception_strategy": "delegated", '
            '"evidence_requirement": "required", '
            '"delegated_plan": {"query": "research current Hermes read-only routing", '
            '"requested_capabilities": ["web_research", "repo_inspection"]}, '
            '"reason": "requires external investigation"}, '
            '"confidence": 0.8}'
        )

        route = route_semantic_input_with_llm(
            "让 Hermes 查一下 read-only routing 怎么做",
            _frame(),
            client=client,
        )

        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertEqual(route.intent_kind, "investigation_request")
        self.assertEqual(route.context_strategy, "delegated")
        self.assertTrue(route.needs_delegated_perception)
        self.assertEqual(route.delegated_perception_query, "research current Hermes read-only routing")
        self.assertEqual(route.suggested_capabilities, ["web_research", "repo_inspection"])

    def test_llm_workspace_context_need_for_new_decision_without_active_frame(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "new_decision", "action": "new_intent", '
            '"is_continuation": false, "needs_workspace_context": true, '
            '"workspace_query": "current runtime implementation", '
            '"confidence": 0.8, "reason": "Fresh repo-based decision."}'
        )

        route = route_semantic_input_with_llm(
            "看一下当前实现然后告诉我下一步做什么",
            None,
            client=client,
        )

        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertFalse(route.is_continuation)
        self.assertTrue(route.needs_workspace_context)
        self.assertEqual(route.workspace_query, "current runtime implementation")

    def test_llm_ignores_workspace_tool_or_file_plans_from_router_payload(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "needs_workspace_context": true, '
            '"workspace_query": "current runtime implementation", '
            '"file_path": "spice/runtime/run_once.py", '
            '"tool_calls": [{"tool": "read_file", "args": {"path": "spice/runtime/run_once.py"}}], '
            '"confidence": 0.8}'
        )

        route = route_semantic_input_with_llm("基于代码回答", _frame(), client=client)
        payload = route.to_payload()

        self.assertTrue(route.needs_workspace_context)
        self.assertEqual(route.workspace_query, "current runtime implementation")
        self.assertNotIn("file_path", payload)
        self.assertNotIn("tool_calls", payload)

    def test_deterministic_route_extracts_explicit_urls_without_llm(self) -> None:
        route = route_semantic_input(
            "基于 https://example.com/spec 和当前情况给我建议",
            _frame(),
        )

        self.assertEqual(route.route, "new_decision")
        self.assertTrue(route.needs_url_context)
        self.assertEqual(route.context_strategy, "url")
        self.assertEqual(route.url_query, "基于 https://example.com/spec 和当前情况给我建议")
        self.assertEqual(route.urls, ["https://example.com/spec"])

    def test_llm_route_preserves_explicit_urls_even_if_payload_omits_url_context(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "confidence": 0.8, '
            '"reason": "Answer from active decision plus linked docs."}'
        )

        route = route_semantic_input_with_llm(
            "结合 https://example.com/spec 看这个决策还对吗？",
            _frame(),
            client=client,
        )

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.context_strategy, "url")
        self.assertTrue(route.needs_url_context)
        self.assertEqual(route.urls, ["https://example.com/spec"])
        self.assertIn("https://example.com/spec", route.url_query)
        prompt = client.requests[0].input_text
        self.assertIn("needs_url_context", prompt)
        self.assertIn("runtime URL perception", prompt)

    def test_route_converts_url_context_request_to_continuation_resolution(self) -> None:
        route = route_semantic_input_with_llm(
            "based on https://example.com/spec, give me the plan for B",
            _frame(),
            client=_FakeSemanticRouterClient(
                '{"route": "follow_up", "action": "plan_candidate", '
                '"is_continuation": true, "candidate_label": "B", '
                '"needs_url_context": true, '
                '"url_query": "linked spec constraints"}'
            ),
        )

        resolution = semantic_route_to_continuation(route)

        self.assertTrue(resolution.is_continuation)
        self.assertEqual(resolution.action, "plan_candidate")
        self.assertEqual(resolution.candidate_id, "candidate.b")
        self.assertTrue(resolution.needs_url_context)
        self.assertEqual(resolution.context_strategy, "url")
        self.assertEqual(resolution.url_query, "linked spec constraints")
        self.assertEqual(resolution.urls, ["https://example.com/spec"])

    def test_llm_routes_delegated_perception_need(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "context_strategy": "delegated", '
            '"needs_delegated_perception": true, '
            '"delegated_perception_query": "latest agent workflow research", '
            '"delegated_perception_reason": "requires current web research", '
            '"suggested_capabilities": ["web_research", "docs_review"], '
            '"confidence": 0.84}'
        )

        route = route_semantic_input_with_llm(
            "让 Hermes 查一下最新 agent workflow 怎么做",
            _frame(),
            client=client,
        )

        self.assertEqual(route.context_strategy, "delegated")
        self.assertFalse(route.needs_workspace_context)
        self.assertFalse(route.needs_url_context)
        self.assertTrue(route.needs_delegated_perception)
        self.assertEqual(route.delegated_perception_query, "latest agent workflow research")
        self.assertEqual(route.delegated_perception_reason, "requires current web research")
        self.assertEqual(route.suggested_capabilities, ["web_research", "docs_review"])

    def test_llm_routes_local_then_delegated_escalation_strategy(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, '
            '"context_strategy": "local_then_delegated_if_insufficient", '
            '"workspace_query": "current implementation of workspace perception", '
            '"delegated_perception_query": "investigate external examples if local evidence is insufficient", '
            '"delegated_perception_reason": "local implementation may not answer best practices", '
            '"suggested_capabilities": ["repo_inspection"], '
            '"confidence": 0.8}'
        )

        route = route_semantic_input_with_llm(
            "先看 repo，如果不够再让 Hermes 调查",
            _frame(),
            client=client,
        )
        resolution = semantic_route_to_continuation(route)

        self.assertEqual(route.context_strategy, "local_then_delegated_if_insufficient")
        self.assertTrue(route.needs_workspace_context)
        self.assertTrue(route.needs_delegated_perception)
        self.assertEqual(route.workspace_query, "current implementation of workspace perception")
        self.assertEqual(
            route.delegated_perception_query,
            "investigate external examples if local evidence is insufficient",
        )
        self.assertEqual(resolution.context_strategy, "local_then_delegated_if_insufficient")
        self.assertTrue(resolution.needs_delegated_perception)

    def test_explicit_urls_take_url_strategy_over_delegated_payload(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "answer_from_decision", '
            '"is_continuation": true, "context_strategy": "delegated", '
            '"needs_delegated_perception": true, '
            '"delegated_perception_query": "read linked page through Hermes", '
            '"confidence": 0.8}'
        )

        route = route_semantic_input_with_llm(
            "基于 https://example.com/spec 回答",
            _frame(),
            client=client,
        )

        self.assertEqual(route.context_strategy, "url")
        self.assertTrue(route.needs_url_context)
        self.assertFalse(route.needs_delegated_perception)
        self.assertEqual(route.urls, ["https://example.com/spec"])

    def test_llm_routes_compare_alternative_followup_to_visible_candidate(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "compare_alternative", '
            '"is_continuation": true, "candidate_label": "B", '
            '"confidence": 0.8, "reason": "User asked if B could be better."}'
        )

        route = route_semantic_input_with_llm("那 B 有没有可能更适合？", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "compare_alternative")
        self.assertEqual(route.candidate_id, "candidate.b")

    def test_llm_routes_clarifying_question_followup(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "ask_clarifying_question", '
            '"is_continuation": true, "confidence": 0.7, '
            '"reason": "The requested answer needs one missing constraint."}'
        )

        route = route_semantic_input_with_llm("如果条件不一样呢？", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "ask_clarifying_question")
        self.assertEqual(route.candidate_id, "candidate.a")

    def test_llm_routes_refine_decision_followup(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "refine_decision", '
            '"is_continuation": true, "refinement": "make it lower risk", '
            '"confidence": 0.7, "reason": "User asked to adjust the card."}'
        )

        route = route_semantic_input_with_llm("把这个改成低风险版本", _frame(), client=client)

        self.assertEqual(route.route, "follow_up")
        self.assertEqual(route.action, "refine_decision")
        self.assertEqual(route.text, "make it lower risk")

    def test_llm_new_topic_stays_new_decision(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "new_decision", "action": "new_intent", '
            '"is_continuation": false, "confidence": 0.7, '
            '"reason": "Fresh question."}'
        )

        route = route_semantic_input_with_llm("My CLI has low DAU. What next?", _frame(), client=client)

        self.assertEqual(route.route, "new_decision")
        self.assertEqual(route.action, "new_intent")
        self.assertFalse(route.is_continuation)

    def test_unknown_llm_candidate_choice_does_not_continue(self) -> None:
        client = _FakeSemanticRouterClient(
            '{"route": "follow_up", "action": "choose_option", '
            '"is_continuation": true, "candidate_id": "candidate.missing"}'
        )

        route = route_semantic_input_with_llm("选不存在的那个", _frame(), client=client)

        self.assertEqual(route.route, "new_decision")
        self.assertFalse(route.is_continuation)
        self.assertIn("unknown candidate", route.reason)

    def test_route_converts_back_to_continuation_resolution(self) -> None:
        route = route_semantic_input_with_llm(
            "choose B",
            _frame(),
            client=_FakeSemanticRouterClient(
                '{"route": "follow_up", "action": "choose_option", '
                '"is_continuation": true, "candidate_label": "B"}'
            ),
        )

        resolution = semantic_route_to_continuation(route)

        self.assertTrue(resolution.is_continuation)
        self.assertEqual(resolution.action, "choose_option")
        self.assertEqual(resolution.candidate_id, "candidate.b")

    def test_route_converts_workspace_context_request_to_continuation_resolution(self) -> None:
        route = route_semantic_input_with_llm(
            "based on the repo, give me the plan for B",
            _frame(),
            client=_FakeSemanticRouterClient(
                '{"route": "follow_up", "action": "plan_candidate", '
                '"is_continuation": true, "candidate_label": "B", '
                '"needs_workspace_context": true, '
                '"workspace_query": "current demo implementation"}'
            ),
        )

        resolution = semantic_route_to_continuation(route)

        self.assertTrue(resolution.is_continuation)
        self.assertEqual(resolution.action, "plan_candidate")
        self.assertEqual(resolution.candidate_id, "candidate.b")
        self.assertTrue(resolution.needs_workspace_context)
        self.assertEqual(resolution.workspace_query, "current demo implementation")


def _frame() -> dict[str, object]:
    return {
        "decision_id": "decision.test",
        "selected_candidate_id": "candidate.a",
        "selected": {
            "label": "A",
            "candidate_id": "candidate.a",
            "title": "Fix onboarding",
            "executor_task": "Create the onboarding fix plan.",
            "recommended_action": "Prioritize onboarding.",
            "intent": "Fix first-run onboarding.",
            "is_selected": True,
        },
        "candidates": [
            {
                "label": "A",
                "candidate_id": "candidate.a",
                "title": "Fix onboarding",
                "executor_task": "Create the onboarding fix plan.",
                "is_selected": True,
            },
            {
                "label": "B",
                "candidate_id": "candidate.b",
                "title": "Improve demo",
                "executor_task": "Create the demo polish plan.",
                "is_selected": False,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
