from __future__ import annotations

import unittest

from spice.decision.general.candidates import (
    ExecutionBoundary,
    GenericCandidate,
    GenericExecutionIntent,
    crosses_execution_approval_boundary,
    is_approval_eligible_executable_candidate,
)
from spice.runtime.execution_affordance import build_execution_affordance
from spice.runtime.executor_runtime import ResolvedExecutorRuntime


class RuntimeApprovalBoundaryTests(unittest.TestCase):
    def test_read_file_candidate_never_enters_execution_approval(self) -> None:
        candidate = _execution_candidate(
            intent="Use read_file and git_status to inspect current repo evidence.",
            handoff_task="read_file spice/runtime/run_once.py and git_status, then return findings/sources.",
            metadata={
                "executor_task": "read_file spice/runtime/run_once.py and git_status.",
                "execution_affordance": {
                    "candidate_execution_requested": True,
                    "approval": {"required": True, "eligible_for_approval": True},
                    "permission": {"required": "workspace_write"},
                },
                "required_permission": "workspace_write",
                "required_capability_inference": {"required_capability": "code_edit"},
            },
        )

        self.assertFalse(crosses_execution_approval_boundary(candidate))
        self.assertFalse(is_approval_eligible_executable_candidate(candidate))

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertFalse(affordance["candidate_execution_requested"])
        self.assertFalse(affordance["candidate_executable"])
        self.assertFalse(affordance["approval"]["required"])
        self.assertEqual(affordance["approval"]["status"], "not_approval_eligible")
        self.assertIn("read-only perception", affordance["blockers"][0])

    def test_delegated_read_only_investigation_uses_investigation_consent_not_approval(self) -> None:
        candidate = _execution_candidate(
            intent="Ask Hermes for a read-only investigation and return findings and sources.",
            handoff_task=(
                "Run read_only_investigation with web_search/read_web_page only; "
                "return findings and sources."
            ),
        )

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertFalse(is_approval_eligible_executable_candidate(candidate))
        self.assertFalse(affordance["candidate_execution_requested"])
        self.assertFalse(affordance["approval"]["required"])

    def test_chinese_current_implementation_read_is_not_treated_as_implementation_work(self) -> None:
        candidate = _execution_candidate(
            intent="查看当前实现，读取本地 repo 后基于实际代码判断。",
            handoff_task="读取本地仓库并返回 sources，不要修改文件。",
        )

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertFalse(crosses_execution_approval_boundary(candidate))
        self.assertFalse(affordance["candidate_execution_requested"])
        self.assertFalse(affordance["approval"]["required"])

    def test_write_file_candidate_still_crosses_execution_approval_boundary(self) -> None:
        candidate = _execution_candidate(
            intent="write_file .spice-smoke/output.txt with the selected summary.",
            handoff_task="write_file .spice-smoke/output.txt with exact text.",
            target_refs=[".spice-smoke/output.txt"],
        )

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertTrue(crosses_execution_approval_boundary(candidate))
        self.assertTrue(is_approval_eligible_executable_candidate(candidate))
        self.assertTrue(affordance["candidate_execution_requested"])
        self.assertTrue(affordance["approval"]["required"])
        self.assertTrue(affordance["approval"]["eligible_for_approval"])

    def test_malformed_write_file_candidate_still_crosses_approval_boundary(self) -> None:
        candidate = _execution_candidate(
            intent="write_file .spice-smoke/output.txt with the selected summary.",
            handoff_task="write_file .spice-smoke/output.txt with exact text.",
            target_refs=[".spice-smoke/output.txt"],
            side_effect_class="none",
            execution_side_effect_class="none",
            boundary_side_effect_class="none",
        )

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertTrue(crosses_execution_approval_boundary(candidate))
        self.assertTrue(is_approval_eligible_executable_candidate(candidate))
        self.assertTrue(affordance["candidate_execution_requested"])
        self.assertTrue(affordance["approval"]["required"])

    def test_read_file_with_negated_modify_instruction_stays_read_only(self) -> None:
        candidate = _execution_candidate(
            intent="read_file spice/runtime/run_once.py; do not modify files.",
            handoff_task="read_file spice/runtime/run_once.py; do not modify files.",
        )

        affordance = build_execution_affordance(candidate, executor_runtime=_runtime())

        self.assertFalse(crosses_execution_approval_boundary(candidate))
        self.assertFalse(affordance["candidate_execution_requested"])
        self.assertFalse(affordance["approval"]["required"])

    def test_search_and_patch_is_side_effectful_when_patch_is_explicit(self) -> None:
        candidate = _execution_candidate(
            intent="Search for the bug and patch the matching file.",
            handoff_task="search for the bug, then patch spice/runtime/run_once.py.",
            target_refs=["spice/runtime/run_once.py"],
        )

        self.assertTrue(crosses_execution_approval_boundary(candidate))
        self.assertTrue(is_approval_eligible_executable_candidate(candidate))


def _execution_candidate(
    *,
    intent: str,
    handoff_task: str,
    target_refs: list[str] | None = None,
    metadata: dict[str, object] | None = None,
    side_effect_class: str = "external_effect",
    execution_side_effect_class: str = "external_effect",
    boundary_side_effect_class: str = "external_effect",
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id="candidate.intent.execute.test",
        action_type="intent.execute",
        intent=intent,
        candidate_kind="decision",
        target_refs=list(target_refs or ["workspace"]),
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested",
            requested=True,
            handoff_task=handoff_task,
            required_permission_hint="workspace_write",
            side_effect_class=execution_side_effect_class,
        ),
        requires_confirmation=True,
        execution_boundary=ExecutionBoundary(
            mode="execution_intent",
            target="executor",
            protocol="sdep",
            requires_confirmation=True,
            side_effect_class=boundary_side_effect_class,
        ),
        side_effect_class=side_effect_class,
        metadata=dict(metadata or {}),
    )


def _runtime() -> ResolvedExecutorRuntime:
    return ResolvedExecutorRuntime(
        requested_executor_id="hermes",
        executor_id="hermes",
        transport="sdep_subprocess_wrapper",
        command="hermes chat -Q",
        permission_mode="workspace_write",
        permission_enforcement="command_flag",
        command_required=True,
        command_found=True,
        status="ready",
        approval_required=True,
        real_executor=True,
        sends_sdep_request=True,
    )


if __name__ == "__main__":
    unittest.main()
