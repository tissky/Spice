from __future__ import annotations

import unittest

from spice.runtime.runtime_guardrails import render_guardrail_message, validate_active_frame_route


class RuntimeGuardrailTests(unittest.TestCase):
    def test_blocks_execution_without_active_frame(self) -> None:
        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=None,
            config={"executor": "dry_run"},
        )

        self.assertFalse(result.allowed)
        self.assertIn("missing_active_decision_frame", result.blockers)
        self.assertIn("no active Decision Card", result.message)

    def test_blocks_missing_target_candidate_for_choice(self) -> None:
        result = validate_active_frame_route(
            action="choose_option",
            active_frame=_frame(_executable_candidate()),
            candidate_id="candidate.missing",
            config={"executor": "dry_run"},
        )

        self.assertFalse(result.allowed)
        self.assertIn("target_candidate_not_found", result.blockers)

    def test_blocks_advisory_only_candidate_execution(self) -> None:
        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=_frame(_advisory_candidate()),
            config={"executor": "dry_run"},
        )

        self.assertFalse(result.allowed)
        self.assertIn("candidate_advisory_only", result.blockers)
        self.assertIn("advisory-only", result.message)
        self.assertIn("advisory-only", render_guardrail_message(result))

    def test_blocks_missing_artifact_source(self) -> None:
        frame = _frame(_executable_candidate())
        frame["run_id"] = ""

        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=frame,
            config={"executor": "dry_run"},
        )

        self.assertFalse(result.allowed)
        self.assertIn("missing_artifact_source", result.blockers)

    def test_blocks_executor_not_ready(self) -> None:
        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=_frame(_executable_candidate()),
            config={"executor": "missing_executor"},
        )

        self.assertFalse(result.allowed)
        self.assertTrue(any("Unsupported executor" in blocker for blocker in result.blockers))

    def test_blocks_permission_escalation_for_natural_execution(self) -> None:
        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=_frame(_executable_candidate(escalation_required=True)),
            config={"executor": "dry_run"},
        )

        self.assertFalse(result.allowed)
        self.assertTrue(any(blocker.startswith("permission_insufficient") for blocker in result.blockers))

    def test_allows_executable_candidate_with_artifact_source_and_executor(self) -> None:
        result = validate_active_frame_route(
            action="execute_selected",
            active_frame=_frame(_executable_candidate()),
            config={"executor": "dry_run"},
        )

        self.assertTrue(result.allowed)
        self.assertEqual(result.candidate_id, "candidate.a")


def _frame(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": "run.test",
        "decision_id": "decision.test",
        "approval_id": "",
        "selected_candidate_id": "candidate.a",
        "selected": dict(candidate),
        "candidates": [dict(candidate)],
    }


def _advisory_candidate() -> dict[str, object]:
    return {
        "candidate_id": "candidate.a",
        "label": "A",
        "title": "Prioritize state-as-context",
        "executor_task": "Plan state-as-context improvements.",
        "execution_affordance": {
            "candidate_executable": False,
            "executor_available": True,
            "executable": False,
            "blocked": True,
            "blockers": [
                "Candidate is advisory; execution_intent.intent_class is not execution_requested.",
            ],
            "approval": {
                "required": False,
                "eligible_for_approval": False,
            },
            "permission": {
                "required": "read_only",
                "configured": "workspace_write",
                "escalation_required": False,
            },
        },
    }


def _executable_candidate(*, escalation_required: bool = False) -> dict[str, object]:
    return {
        "candidate_id": "candidate.a",
        "label": "A",
        "title": "Implement selected plan",
        "executor_task": "Implement the selected plan.",
        "execution_affordance": {
            "candidate_executable": True,
            "executor_available": True,
            "executable": True,
            "blocked": False,
            "approval": {
                "required": True,
                "eligible_for_approval": True,
            },
            "permission": {
                "required": "workspace_write",
                "configured": "read_only" if escalation_required else "workspace_write",
                "escalation_required": escalation_required,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
