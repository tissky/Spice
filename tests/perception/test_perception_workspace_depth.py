from __future__ import annotations

import unittest

from spice.perception import (
    WORKSPACE_PERCEPTION_DEPTH_DEEP,
    WORKSPACE_PERCEPTION_DEPTH_NATIVE,
    WORKSPACE_PERCEPTION_DEPTH_NORMAL,
    resolve_workspace_perception_depth,
    workspace_perception_depth_from_payload,
)


class WorkspacePerceptionDepthTests(unittest.TestCase):
    def test_auto_defaults_to_normal_budget(self) -> None:
        budget = resolve_workspace_perception_depth()

        self.assertEqual(budget.depth, WORKSPACE_PERCEPTION_DEPTH_NORMAL)
        self.assertEqual(budget.max_rounds, 10)
        self.assertEqual(budget.max_tool_calls, 80)
        self.assertEqual(budget.max_tool_calls_per_round, 10)
        self.assertEqual(budget.max_blocked_tool_calls, 20)
        self.assertEqual(budget.max_blocked_tool_calls_per_round, 5)
        self.assertEqual(budget.max_files_read, 60)
        self.assertEqual(budget.max_chars_per_file, 12_000)
        self.assertEqual(budget.total_char_budget, 500_000)
        self.assertEqual(budget.planner_max_tokens, 4_000)

    def test_repo_evidence_uses_normal_budget(self) -> None:
        budget = resolve_workspace_perception_depth(
            evidence_domain="repo",
            user_input="请基于当前实现判断下一步。",
        )

        self.assertEqual(budget.depth, WORKSPACE_PERCEPTION_DEPTH_NORMAL)
        self.assertIn("repo", budget.reason)

    def test_current_implementation_wording_uses_normal_budget(self) -> None:
        budget = resolve_workspace_perception_depth(
            user_input="请读取当前实现，然后基于实际代码判断下一步。",
        )

        self.assertEqual(budget.depth, WORKSPACE_PERCEPTION_DEPTH_NORMAL)
        self.assertIn("current implementation", budget.reason)

    def test_report_or_deep_review_uses_deep_budget(self) -> None:
        report = resolve_workspace_perception_depth(answer_mode="report")
        wording = resolve_workspace_perception_depth(user_input="请做一次完整 review 并给证据。")

        self.assertEqual(report.depth, WORKSPACE_PERCEPTION_DEPTH_DEEP)
        self.assertEqual(report.max_rounds, 25)
        self.assertEqual(report.max_tool_calls, 200)
        self.assertEqual(report.max_files_read, 120)
        self.assertEqual(report.total_char_budget, 1_500_000)
        self.assertEqual(report.planner_max_tokens, 8_000)
        self.assertEqual(wording.depth, WORKSPACE_PERCEPTION_DEPTH_DEEP)

    def test_native_requires_explicit_opt_in(self) -> None:
        capped = resolve_workspace_perception_depth(requested_depth="native")
        native = resolve_workspace_perception_depth(requested_depth="native", native_opt_in=True)

        self.assertEqual(capped.depth, WORKSPACE_PERCEPTION_DEPTH_DEEP)
        self.assertFalse(capped.native)
        self.assertIn("requires explicit opt-in", capped.reason)
        self.assertEqual(native.depth, WORKSPACE_PERCEPTION_DEPTH_NATIVE)
        self.assertTrue(native.native)
        self.assertIsNone(native.planner_max_tokens)
        self.assertEqual(native.max_rounds, 90)
        self.assertEqual(native.max_tool_calls, 500)

    def test_configured_native_counts_as_explicit_opt_in(self) -> None:
        budget = resolve_workspace_perception_depth(
            config={"workspace_perception": {"depth": "native"}}
        )

        self.assertEqual(budget.depth, WORKSPACE_PERCEPTION_DEPTH_NATIVE)
        self.assertTrue(budget.native)
        self.assertTrue(budget.explicit_opt_in)

    def test_workspace_perception_config_overrides_budget_values(self) -> None:
        budget = resolve_workspace_perception_depth(
            config={
                "workspace_perception": {
                    "depth": "deep",
                    "max_rounds": 12,
                    "max_tool_calls": 90,
                    "max_tool_calls_per_round": 9,
                    "max_blocked_tool_calls": 11,
                    "max_blocked_tool_calls_per_round": 3,
                    "max_files_read": 70,
                    "max_chars_per_file": 13_000,
                    "total_char_budget": 600_000,
                    "planner_max_tokens": 5_000,
                }
            },
        )

        self.assertEqual(budget.depth, WORKSPACE_PERCEPTION_DEPTH_DEEP)
        self.assertEqual(budget.max_rounds, 12)
        self.assertEqual(budget.max_tool_calls, 90)
        self.assertEqual(budget.max_tool_calls_per_round, 9)
        self.assertEqual(budget.max_blocked_tool_calls, 11)
        self.assertEqual(budget.max_blocked_tool_calls_per_round, 3)
        self.assertEqual(budget.max_files_read, 70)
        self.assertEqual(budget.max_chars_per_file, 13_000)
        self.assertEqual(budget.total_char_budget, 600_000)
        self.assertEqual(budget.planner_max_tokens, 5_000)

    def test_payload_round_trip_keeps_native_none_planner_tokens(self) -> None:
        original = resolve_workspace_perception_depth(requested_depth="native", native_opt_in=True)
        restored = workspace_perception_depth_from_payload(original.to_payload())

        self.assertEqual(restored.depth, WORKSPACE_PERCEPTION_DEPTH_NATIVE)
        self.assertIsNone(restored.planner_max_tokens)
        self.assertTrue(restored.native)


if __name__ == "__main__":
    unittest.main()
