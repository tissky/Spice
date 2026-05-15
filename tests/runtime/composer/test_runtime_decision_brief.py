from __future__ import annotations

import tempfile
import unittest

from spice.runtime import run_once, setup_workspace
from spice.runtime.decision_brief import (
    DECISION_BRIEF_SCHEMA_VERSION,
    compose_decision_brief,
    render_decision_brief,
)


class RuntimeDecisionBriefTests(unittest.TestCase):
    def test_compose_decision_brief_from_compare_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_once(
                "Compare state-as-context, proactive perception, and executor handoff.",
                project_root=tmp_dir,
                use_bars=False,
                full_loop_preview=False,
            )

            brief = compose_decision_brief(
                result.artifact["compare_payload"],
                run_id=result.artifact["run_id"],
                decision_id=result.artifact["decision_id"],
            )

        self.assertEqual(brief["schema_version"], DECISION_BRIEF_SCHEMA_VERSION)
        self.assertEqual(brief["run_id"], result.artifact["run_id"])
        self.assertEqual(brief["decision_id"], result.artifact["decision_id"])
        self.assertTrue(brief["selected"]["candidate_id"])
        self.assertTrue(brief["selected"]["title"])
        self.assertTrue(brief["next_actions"])

        rendered = render_decision_brief(brief)
        self.assertIn("I'd choose", rendered)
        self.assertIn("details  expand the full Decision Card", rendered)
        self.assertIn("why      show why-not comparison", rendered)

    def test_run_once_artifact_contains_decision_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_once(
                "What should Spice prioritize next?",
                project_root=tmp_dir,
                use_bars=False,
                full_loop_preview=False,
            )

        self.assertIn("decision_brief", result.artifact)
        self.assertEqual(
            result.artifact["decision_brief"]["schema_version"],
            DECISION_BRIEF_SCHEMA_VERSION,
        )
        self.assertIn("I'd choose", result.rendered_text)
        self.assertIn("SPICE RUN ONCE", result.rendered_text)
        self.assertIn("DECISION COMPARISON", result.rendered_text)

    def test_decision_brief_mentions_execution_capability_lightly(self) -> None:
        brief = compose_decision_brief(
            _compare_payload_with_affordance(
                {
                    "candidate_executable": True,
                    "executable": True,
                    "blocked": False,
                    "required_capability": "code_edit",
                    "executor_capability_source": "static_baseline",
                    "capability": {
                        "required_capability": "code_edit",
                        "executor_has_required_capability": True,
                        "source": "static_baseline",
                        "matched_capability": "code_edit",
                    },
                    "executor": {"executor_id": "codex"},
                    "permission": {"configured": "workspace_write", "required": "workspace_write"},
                    "approval": {"required": True},
                }
            )
        )

        rendered = render_decision_brief(brief)

        self.assertIn("needs code_edit", rendered)
        self.assertIn("codex static_baseline supports it", rendered)
        self.assertIn("enter approval", rendered)

    def test_decision_brief_mentions_dry_run_simulation_without_real_execution_claim(self) -> None:
        brief = compose_decision_brief(
            _compare_payload_with_affordance(
                {
                    "candidate_executable": True,
                    "executable": True,
                    "blocked": False,
                    "required_capability": "code_edit",
                    "executor_capability_source": "static_baseline",
                    "capability": {
                        "required_capability": "code_edit",
                        "executor_has_required_capability": True,
                        "source": "static_baseline",
                        "matched_capability": "simulate_execution",
                        "simulates_required_capability": True,
                    },
                    "executor": {"executor_id": "dry_run"},
                    "permission": {"configured": "workspace_write", "required": "workspace_write"},
                    "approval": {"required": True},
                }
            )
        )

        rendered = render_decision_brief(brief)

        self.assertIn("only simulates", rendered)
        self.assertIn("not a real executor run", rendered)


def _compare_payload_with_affordance(affordance: dict[str, object]) -> dict[str, object]:
    return {
        "decision_id": "decision.brief.capability",
        "trace_ref": "trace.brief.capability",
        "decision_relevant_state_summary": {
            "active_commitments": [],
            "open_work_items": [],
            "active_conflicts": [],
            "executor_available": True,
        },
        "candidate_decisions": [
            {
                "candidate_id": "candidate.a",
                "title": "Implement the change",
                "action": "intent.execute",
                "intent": "Implement the state-as-context change.",
                "recommended_action": "Implement the state-as-context change.",
                "enabled_reason": "available",
                "requires_confirmation": True,
                "expected_effect": {},
                "execution_affordance": affordance,
                "is_selected": True,
            }
        ],
        "score_breakdown": {
            "candidates": {
                "candidate.a": {
                    "score_total": 0.8,
                    "dimensions": [],
                    "constraints": [],
                    "vetoes": [],
                    "tradeoff_rules": [],
                }
            }
        },
        "selected_recommendation": {
            "candidate_id": "candidate.a",
            "title": "Implement the change",
            "human_summary": "Implement the state-as-context change.",
            "decision_basis": [],
            "reason_summary": ["It closes the gap."],
        },
        "why_not_the_others": [],
        "expected_outcome_or_risk": {},
    }


if __name__ == "__main__":
    unittest.main()
