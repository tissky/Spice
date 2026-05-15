from __future__ import annotations

import unittest

from spice.decision.general.candidates import (
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    GenericExecutionIntent,
)
from spice.runtime import (
    REQUIRED_CAPABILITY_INFERENCE_VERSION,
    annotate_required_capabilities,
    infer_required_capability,
)


class RuntimeCandidateCapabilityInferenceTests(unittest.TestCase):
    def test_intent_execute_defaults_to_general_execution(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            intent="Run the approved task.",
        )

        self.assertEqual(infer_required_capability(candidate), "general_execution")

    def test_code_change_execution_infers_code_edit(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            intent="Fix the failing test by changing the repo files.",
            handoff_task="Modify the code and update the test.",
        )

        annotated = annotate_required_capabilities([candidate])[0]

        self.assertEqual(annotated.required_capability, "code_edit")
        self.assertEqual(annotated.execution_boundary.required_capability, "code_edit")
        self.assertEqual(
            annotated.metadata["required_capability_inference"]["schema_version"],
            REQUIRED_CAPABILITY_INFERENCE_VERSION,
        )
        self.assertEqual(
            annotated.metadata["required_capability_inference"]["source"],
            "runtime_inference",
        )

    def test_repo_read_execution_infers_repo_read(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            intent="Review the repo and summarize the risky files.",
            handoff_task="Inspect the codebase and summarize architecture.",
        )

        self.assertEqual(infer_required_capability(candidate), "repo_read")

    def test_github_work_takes_precedence_for_pr_and_issue_text(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            intent="Review GitHub PR #42 and triage the linked issue.",
        )

        self.assertEqual(infer_required_capability(candidate), "github_work")

    def test_browser_or_research_work_infers_external_tools(self) -> None:
        candidate = _candidate(
            action_type="intent.execute",
            intent="Research the web and compare browser evidence.",
        )

        self.assertEqual(infer_required_capability(candidate), "browser_or_external_tools")

    def test_advisory_and_guardrail_candidates_do_not_require_execution_capability(self) -> None:
        candidate = _candidate(
            action_type="item.triage",
            intent="Plan a code change and decide whether to edit files.",
            execution_requested=False,
        )

        annotated = annotate_required_capabilities([candidate])[0]

        self.assertEqual(annotated.required_capability, "")
        self.assertEqual(
            annotated.metadata["required_capability_inference"]["reason"],
            "Candidate is advisory, planning, or runtime guardrail; no execution capability required.",
        )

    def test_existing_required_capability_is_preserved(self) -> None:
        candidate = _candidate(
            action_type="capability.use",
            intent="Delegate to existing review capability.",
            required_capability="cap.review",
        )

        annotated = annotate_required_capabilities([candidate])[0]

        self.assertEqual(annotated.required_capability, "cap.review")
        self.assertEqual(
            annotated.metadata["required_capability_inference"]["source"],
            "existing_candidate_field",
        )

    def test_capability_use_without_specific_match_defaults_to_general_execution(self) -> None:
        candidate = _candidate(
            action_type="capability.use",
            intent="Delegate the task to the configured executor.",
        )

        self.assertEqual(infer_required_capability(candidate), "general_execution")


def _candidate(
    *,
    action_type: str,
    intent: str,
    handoff_task: str = "",
    required_capability: str = "",
    execution_requested: bool = True,
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=f"candidate.test.{action_type.replace('.', '_')}",
        action_type=action_type,
        intent=intent,
        required_capability=required_capability,
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested" if execution_requested else "advisory",
            requested=execution_requested,
            handoff_task=handoff_task or intent,
            side_effect_class="external_effect" if execution_requested else "read_only",
        ),
        expected_state_delta=ExpectedStateDelta(summary="Candidate may update state."),
        execution_boundary=ExecutionBoundary(
            mode="execution_intent" if execution_requested else "none",
            requires_confirmation=execution_requested,
            side_effect_class="external_effect" if execution_requested else "read_only",
        ),
        requires_confirmation=execution_requested,
        side_effect_class="external_effect" if execution_requested else "read_only",
        metadata={"user_facing_title": intent},
    )


if __name__ == "__main__":
    unittest.main()
