from __future__ import annotations

import unittest

from spice.runtime.response_depth import resolve_response_depth_budget


class RuntimeResponseDepthPolicyTests(unittest.TestCase):
    def test_explicit_answer_modes_resolve_to_configured_budgets(self) -> None:
        brief = resolve_response_depth_budget(answer_mode="brief")
        normal = resolve_response_depth_budget(answer_mode="normal")
        detailed = resolve_response_depth_budget(answer_mode="detailed")
        report = resolve_response_depth_budget(answer_mode="report")

        self.assertEqual(brief.max_tokens, 1200)
        self.assertEqual(brief.max_chars, 3500)
        self.assertEqual(normal.max_tokens, 3000)
        self.assertEqual(normal.max_chars, 8000)
        self.assertEqual(detailed.max_tokens, 6000)
        self.assertEqual(detailed.max_chars, 14000)
        self.assertEqual(report.max_tokens, 12000)
        self.assertEqual(report.max_chars, 24000)

    def test_user_wording_can_request_detailed_or_report_depth(self) -> None:
        detailed = resolve_response_depth_budget(user_input="给我一个两周内怎么做的具体计划")
        report = resolve_response_depth_budget(user_input="基于当前实现给我一份证据报告")

        self.assertEqual(detailed.answer_mode, "detailed")
        self.assertEqual(report.answer_mode, "report")

    def test_evidence_context_strengthens_response_depth(self) -> None:
        workspace = resolve_response_depth_budget(
            evidence_context={
                "requirements": {"evidence_domain": "repo", "answer_mode": "normal"},
                "workspace": {"present": True, "source_count": 4},
            }
        )
        delegated = resolve_response_depth_budget(
            evidence_context={
                "requirements": {"evidence_domain": "external", "answer_mode": "normal"},
                "delegated": {"present": True, "source_count": 3},
            }
        )

        self.assertEqual(workspace.answer_mode, "detailed")
        self.assertEqual(delegated.answer_mode, "report")

    def test_config_can_opt_into_native_ceiling(self) -> None:
        budget = resolve_response_depth_budget(config={"response_depth": "native"})

        self.assertEqual(budget.answer_mode, "native")
        self.assertIsNone(budget.max_tokens)
        self.assertTrue(budget.native)
        self.assertEqual(budget.max_chars, 24000)

    def test_config_can_request_brief_when_no_hard_evidence_depth_exists(self) -> None:
        budget = resolve_response_depth_budget(config={"response_depth": "brief"})

        self.assertEqual(budget.answer_mode, "brief")
        self.assertEqual(budget.max_tokens, 1200)

    def test_config_cannot_downgrade_hard_evidence_depth(self) -> None:
        budget = resolve_response_depth_budget(
            evidence_context={
                "requirements": {"evidence_domain": "external", "answer_mode": "report"},
                "delegated": {"present": True, "source_count": 2},
            },
            config={"response_depth": "brief"},
        )

        self.assertEqual(budget.answer_mode, "report")


if __name__ == "__main__":
    unittest.main()
