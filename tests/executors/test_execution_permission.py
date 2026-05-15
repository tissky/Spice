from __future__ import annotations

import unittest

from spice.decision.general.candidates import (
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    GenericExecutionIntent,
)
from spice.decision.general.permissions import (
    infer_executor_permission_requirement,
    permission_exceeds,
)
from spice.runtime.execution_affordance import build_execution_affordance
from spice.runtime.executor_capabilities import static_executor_capability_snapshot
from spice.runtime.executor_runtime import ResolvedExecutorRuntime


class ExecutionPermissionTests(unittest.TestCase):
    def test_file_creation_requires_workspace_write(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.smoke",
            action_type="intent.execute",
            intent="Create .spice-smoke/codex_executor_smoke.txt with exact text.",
            target_refs=[".spice-smoke/codex_executor_smoke.txt"],
            expected_state_delta=ExpectedStateDelta(
                summary="A file is created in the workspace."
            ),
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Create the smoke file.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
            side_effect_class="external_effect",
        )

        requirement = infer_executor_permission_requirement(candidate)

        self.assertEqual(requirement.required_permission, "workspace_write")
        self.assertTrue(permission_exceeds(requirement.required_permission, "read_only"))

    def test_destructive_intent_requires_full_access(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.delete",
            action_type="intent.execute",
            intent="Delete /Users/me/tmp/data.",
            target_refs=["/Users/me/tmp/data"],
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
            side_effect_class="external_effect",
        )

        requirement = infer_executor_permission_requirement(candidate)

        self.assertEqual(requirement.required_permission, "danger_full_access")

    def test_read_only_candidate_stays_read_only(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.context.prepare",
            action_type="context.prepare",
            intent="Prepare context before acting.",
            requires_confirmation=False,
            execution_boundary=ExecutionBoundary(
                mode="read_or_prepare",
                requires_confirmation=False,
                side_effect_class="none",
            ),
            side_effect_class="none",
        )

        requirement = infer_executor_permission_requirement(candidate)

        self.assertEqual(requirement.required_permission, "read_only")

    def test_executable_affordance_records_executor_permission_and_approval(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.smoke",
            action_type="intent.execute",
            intent="Create .spice-smoke/codex_executor_smoke.txt with exact text.",
            target_refs=[".spice-smoke/codex_executor_smoke.txt"],
            expected_state_delta=ExpectedStateDelta(
                summary="A file is created in the workspace."
            ),
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Create the smoke file.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="codex",
            executor_id="codex",
            transport="sdep_subprocess_wrapper",
            command="codex exec --skip-git-repo-check --sandbox read-only -",
            permission_mode="read_only",
            permission_enforcement="command_flag",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(candidate, executor_runtime=runtime)

        self.assertEqual(affordance["generated_by"], "spice.runtime.execution_affordance")
        self.assertTrue(affordance["candidate_executable"])
        self.assertTrue(affordance["executable"])
        self.assertFalse(affordance["blocked"])
        self.assertEqual(affordance["executor"]["executor_id"], "codex")
        self.assertEqual(affordance["permission"]["configured"], "read_only")
        self.assertEqual(affordance["permission"]["required"], "workspace_write")
        self.assertTrue(affordance["permission"]["escalation_required"])
        self.assertTrue(affordance["permission"]["escalation_supported"])
        self.assertTrue(affordance["approval"]["required"])
        self.assertTrue(affordance["approval"]["eligible_for_approval"])

    def test_planning_affordance_is_blocked_and_not_approval_eligible(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.context.prepare",
            action_type="context.prepare",
            intent="Prepare context before acting.",
            requires_confirmation=False,
            execution_boundary=ExecutionBoundary(
                mode="read_or_prepare",
                requires_confirmation=False,
                side_effect_class="none",
            ),
            side_effect_class="none",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="dry_run",
            executor_id="dry_run",
            transport="local_dry_run",
            command="",
            permission_mode="workspace_write",
            status="ready",
            approval_required=True,
            real_executor=False,
            sends_sdep_request=False,
        )

        affordance = build_execution_affordance(candidate, executor_runtime=runtime)

        self.assertFalse(affordance["candidate_executable"])
        self.assertFalse(affordance["executable"])
        self.assertTrue(affordance["blocked"])
        self.assertIn("execution_intent.intent_class", affordance["blocked_reason"])
        self.assertFalse(affordance["approval"]["required"])
        self.assertFalse(affordance["approval"]["eligible_for_approval"])

    def test_missing_executor_capability_blocks_execution_candidate(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.github",
            action_type="intent.execute",
            intent="Review the GitHub PR and leave a comment.",
            target_refs=["https://github.com/example/repo/pull/12"],
            required_capability="github_work",
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Review the GitHub PR and leave a comment.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
                required_capability="github_work",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="codex",
            executor_id="codex",
            transport="sdep_subprocess_wrapper",
            command="codex exec --skip-git-repo-check --sandbox workspace-write -",
            permission_mode="workspace_write",
            permission_enforcement="command_flag",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("codex"),
        )

        self.assertFalse(affordance["candidate_executable"])
        self.assertFalse(affordance["executable"])
        self.assertTrue(affordance["blocked"])
        self.assertEqual(affordance["required_capability"], "github_work")
        self.assertEqual(affordance["executor_capability_source"], "static_baseline")
        self.assertEqual(
            affordance["blocked_reason"],
            "Executor lacks required capability: github_work",
        )
        self.assertTrue(affordance["approval"]["required"])
        self.assertFalse(affordance["approval"]["eligible_for_approval"])

    def test_permission_and_capability_blockers_are_both_retained(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.delete_github",
            action_type="intent.execute",
            intent="Delete /Users/me/tmp/data after reviewing the GitHub PR.",
            target_refs=["/Users/me/tmp/data", "https://github.com/example/repo/pull/12"],
            required_capability="github_work",
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Delete the local data and update the GitHub PR.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
                required_capability="github_work",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="codex",
            executor_id="codex",
            transport="sdep_subprocess_wrapper",
            command="codex exec --skip-git-repo-check --sandbox read-only -",
            permission_mode="read_only",
            permission_enforcement="not_applicable",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("codex"),
        )

        self.assertFalse(affordance["candidate_executable"])
        self.assertFalse(affordance["executable"])
        self.assertTrue(affordance["blocked"])
        self.assertEqual(affordance["required_capability"], "github_work")
        self.assertEqual(affordance["permission"]["required"], "danger_full_access")
        self.assertTrue(affordance["permission"]["escalation_required"])
        self.assertFalse(affordance["permission"]["escalation_supported"])
        self.assertIn(
            "Executor lacks required capability: github_work",
            affordance["blockers"],
        )
        self.assertIn(
            "Executor permission escalation to danger_full_access is not automated.",
            affordance["blockers"],
        )
        self.assertTrue(affordance["approval"]["required"])
        self.assertFalse(affordance["approval"]["eligible_for_approval"])

    def test_general_execution_capability_covers_specific_execution_candidate(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.code",
            action_type="intent.execute",
            intent="Implement the state-as-context change in the repo.",
            target_refs=["spice/runtime/run_once.py"],
            required_capability="code_edit",
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Implement the state-as-context change.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
                required_capability="code_edit",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="hermes",
            executor_id="hermes",
            transport="sdep_subprocess_wrapper",
            command="hermes",
            permission_mode="workspace_write",
            permission_enforcement="command_flag",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("hermes"),
        )

        self.assertTrue(affordance["candidate_executable"])
        self.assertTrue(affordance["executable"])
        self.assertFalse(affordance["blocked"])
        self.assertEqual(affordance["required_capability"], "code_edit")
        self.assertEqual(affordance["capability"]["matched_capability"], "general_execution")

    def test_matching_executor_capability_keeps_execution_candidate_executable(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.code",
            action_type="intent.execute",
            intent="Implement the state-as-context change in the repo.",
            target_refs=["spice/runtime/run_once.py"],
            required_capability="code_edit",
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Implement the state-as-context change.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
                required_capability="code_edit",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="codex",
            executor_id="codex",
            transport="sdep_subprocess_wrapper",
            command="codex exec --skip-git-repo-check --sandbox workspace-write -",
            permission_mode="workspace_write",
            permission_enforcement="command_flag",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("codex"),
        )

        self.assertTrue(affordance["candidate_executable"])
        self.assertTrue(affordance["executable"])
        self.assertFalse(affordance["blocked"])
        self.assertEqual(affordance["required_capability"], "code_edit")
        self.assertTrue(affordance["capability"]["executor_has_required_capability"])

    def test_dry_run_simulate_execution_can_preview_required_capability(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.intent.execute.code",
            action_type="intent.execute",
            intent="Implement the state-as-context change in the repo.",
            target_refs=["spice/runtime/run_once.py"],
            required_capability="code_edit",
            execution_intent=GenericExecutionIntent(
                intent_class="execution_requested",
                requested=True,
                handoff_task="Preview implementing the state-as-context change.",
                side_effect_class="external_effect",
            ),
            execution_boundary=ExecutionBoundary(
                mode="execution_intent",
                target="executor",
                requires_confirmation=True,
                side_effect_class="external_effect",
                required_capability="code_edit",
            ),
            side_effect_class="external_effect",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="dry_run",
            executor_id="dry_run",
            transport="local_dry_run",
            command="",
            permission_mode="workspace_write",
            status="ready",
            approval_required=True,
            real_executor=False,
            sends_sdep_request=False,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("dry_run"),
        )

        self.assertTrue(affordance["candidate_executable"])
        self.assertTrue(affordance["executable"])
        self.assertFalse(affordance["blocked"])
        self.assertEqual(affordance["required_capability"], "code_edit")
        self.assertTrue(affordance["capability"]["simulates_required_capability"])

    def test_matching_executor_capability_does_not_make_advisory_candidate_executable(self) -> None:
        candidate = GenericCandidate(
            candidate_id="candidate.item.triage",
            action_type="item.triage",
            intent="Plan a code edit without requesting execution.",
            required_capability="code_edit",
            requires_confirmation=False,
            execution_boundary=ExecutionBoundary(
                mode="read_or_prepare",
                requires_confirmation=False,
                side_effect_class="none",
                required_capability="code_edit",
            ),
            side_effect_class="none",
        )
        runtime = ResolvedExecutorRuntime(
            requested_executor_id="codex",
            executor_id="codex",
            transport="sdep_subprocess_wrapper",
            command="codex exec --skip-git-repo-check --sandbox workspace-write -",
            permission_mode="workspace_write",
            permission_enforcement="command_flag",
            command_required=True,
            command_found=True,
            status="ready",
            approval_required=True,
            real_executor=True,
            sends_sdep_request=True,
        )

        affordance = build_execution_affordance(
            candidate,
            executor_runtime=runtime,
            executor_capabilities=static_executor_capability_snapshot("codex"),
        )

        self.assertFalse(affordance["candidate_executable"])
        self.assertFalse(affordance["executable"])
        self.assertTrue(affordance["blocked"])
        self.assertIn("advisory", affordance["blocked_reason"])
        self.assertTrue(affordance["capability"]["executor_has_required_capability"])


if __name__ == "__main__":
    unittest.main()
