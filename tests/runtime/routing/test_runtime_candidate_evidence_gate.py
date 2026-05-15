from __future__ import annotations

import unittest

from spice.decision.general.candidates import GenericCandidate, GenericExecutionIntent
from spice.runtime.candidate_evidence_gate import (
    HARD_REPO_EVIDENCE_CONSTRAINT_ID,
    apply_candidate_evidence_gate,
)
from spice.runtime.evidence_requirement import detect_evidence_requirement
from spice.runtime.run_once import _candidate_selection_for_run_mode


class RuntimeCandidateEvidenceGateTests(unittest.TestCase):
    def test_document_quick_judgment_is_ineligible_under_hard_repo_evidence(self) -> None:
        candidate = _candidate(
            "candidate.llm.doc",
            "基于文档快速判断（不执行完整代码读取）",
        )
        result = apply_candidate_evidence_gate(
            [candidate],
            intent_text="请基于当前实现和实际代码判断下一步。",
            evidence_requirement=detect_evidence_requirement("请基于当前实现和实际代码判断下一步。"),
            workspace_context={
                "perception_id": "workspace.empty",
                "summary": "Perception ran but did not read source-backed evidence.",
                "files_read": [],
                "facts": [],
                "snippets": [],
            },
        )

        self.assertTrue(result.applied)
        self.assertFalse(result.source_backed_workspace_evidence)
        self.assertIn("candidate.llm.doc", result.ineligible_candidate_ids)
        self.assertEqual(candidate.availability_status, "blocked")
        self.assertIn(HARD_REPO_EVIDENCE_CONSTRAINT_ID, {
            item["constraint_id"] for item in candidate.constraints_triggered
        })
        self.assertTrue(result.added_candidate_ids)

        selection = _candidate_selection_for_run_mode(result.candidates, mode="auto")
        self.assertNotIn("candidate.llm.doc", selection["selection_candidate_ids"])
        self.assertIn(result.added_candidate_ids[0], selection["selection_candidate_ids"])

    def test_source_backed_code_candidate_remains_eligible(self) -> None:
        candidate = _candidate(
            "candidate.llm.code",
            "基于实际读到的代码 source 判断 state-as-context 优先。",
        )
        result = apply_candidate_evidence_gate(
            [candidate],
            intent_text="请基于实际代码判断下一步。",
            evidence_requirement=detect_evidence_requirement("请基于实际代码判断下一步。"),
            workspace_context={
                "perception_id": "workspace.1",
                "files_read": [{"path": "spice/runtime/run_once.py"}],
                "facts": [
                    {
                        "text": "run_once consumes workspace_context.",
                        "source_refs": ["workspace:spice/runtime/run_once.py"],
                    }
                ],
            },
        )

        self.assertTrue(result.source_backed_workspace_evidence)
        self.assertEqual(candidate.availability_status, "available")
        self.assertIn("candidate.llm.code", result.eligible_candidate_ids)
        self.assertEqual(result.added_candidate_ids, [])

    def test_missing_evidence_blocks_final_answer_but_allows_evidence_gathering_candidate(self) -> None:
        final = _candidate("candidate.llm.final", "先基于架构直觉判断 state-as-context 优先。")
        gather = _candidate(
            "candidate.llm.gather",
            "执行完整代码感知，读取关键实现文件后再判断。",
            action_type="intent.execute",
            execution_requested=True,
        )

        result = apply_candidate_evidence_gate(
            [final, gather],
            intent_text="请读取当前 repo 后基于实际代码判断。",
            evidence_requirement=detect_evidence_requirement("请读取当前 repo 后基于实际代码判断。"),
            workspace_context={
                "perception_id": "workspace.empty",
                "summary": "No files were read.",
            },
        )

        self.assertEqual(final.availability_status, "blocked")
        self.assertEqual(gather.availability_status, "available")
        self.assertEqual(gather.action_type, "context.prepare")
        self.assertFalse(gather.execution_intent.requested)
        self.assertFalse(gather.requires_confirmation)
        self.assertIn("candidate.llm.gather", result.eligible_candidate_ids)
        self.assertEqual(result.added_candidate_ids, [])

    def test_doc_only_candidate_is_blocked_even_when_other_workspace_sources_exist(self) -> None:
        doc = _candidate("candidate.llm.doc", "基于文档快速判断，节省时间和预算。")
        code = _candidate("candidate.llm.code", "基于 workspace perception 的 files/facts/snippets 判断。")

        result = apply_candidate_evidence_gate(
            [doc, code],
            intent_text="基于实际代码判断。",
            evidence_requirement=detect_evidence_requirement("基于实际代码判断。"),
            workspace_context={
                "perception_id": "workspace.1",
                "snippets": [{"path": "spice/runtime/workspace_perception.py", "text": "loop"}],
            },
        )

        self.assertEqual(doc.availability_status, "blocked")
        self.assertEqual(code.availability_status, "available")
        self.assertIn("candidate.llm.doc", result.ineligible_candidate_ids)
        self.assertIn("candidate.llm.code", result.eligible_candidate_ids)

    def test_read_only_evidence_candidate_clears_stale_execution_metadata(self) -> None:
        candidate = _candidate(
            "candidate.llm.stale_execution",
            "读取关键实现文件后基于实际代码判断。",
            action_type="intent.execute",
            execution_requested=True,
            metadata={
                "executor_task": "Run workspace investigation.",
                "execution_affordance": {
                    "candidate_execution_requested": True,
                    "approval": {"required": True, "eligible_for_approval": True},
                    "permission": {"required": "workspace_write"},
                },
                "permission": {"required": "workspace_write"},
                "required_permission": "workspace_write",
                "required_capability_inference": {"required_capability": "code_edit"},
                "skill_resolution": {"status": "resolved"},
                "resolved_skill": {"skill_id": "spice.hermes.execute"},
            },
        )

        result = apply_candidate_evidence_gate(
            [candidate],
            intent_text="请读取当前 repo 后基于实际代码判断。",
            evidence_requirement=detect_evidence_requirement("请读取当前 repo 后基于实际代码判断。"),
            workspace_context={"perception_id": "workspace.empty"},
        )

        self.assertIn(candidate.candidate_id, result.eligible_candidate_ids)
        self.assertEqual(candidate.action_type, "context.prepare")
        self.assertEqual(candidate.side_effect_class, "read_only")
        self.assertFalse(candidate.execution_intent.requested)
        self.assertEqual(candidate.execution_intent.intent_class, "advisory")
        self.assertEqual(candidate.execution_boundary.mode, "none")
        self.assertFalse(candidate.requires_confirmation)
        self.assertEqual(candidate.metadata["executor_task"], "")
        self.assertTrue(candidate.metadata["read_only_intent_boundary_applied"])
        self.assertNotIn("execution_affordance", candidate.metadata)
        self.assertNotIn("permission", candidate.metadata)
        self.assertNotIn("required_permission", candidate.metadata)
        self.assertNotIn("required_capability_inference", candidate.metadata)
        self.assertNotIn("skill_resolution", candidate.metadata)
        self.assertNotIn("resolved_skill", candidate.metadata)


def _candidate(
    candidate_id: str,
    intent: str,
    *,
    action_type: str = "item.triage",
    execution_requested: bool = False,
    metadata: dict[str, object] | None = None,
) -> GenericCandidate:
    payload_metadata = {
        "source": "llm_candidate_expander",
        "candidate_source": "llm_generator",
        "user_facing_title": intent,
        "recommendation": intent,
    }
    payload_metadata.update(metadata or {})
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=intent,
        candidate_kind="decision",
        execution_intent=GenericExecutionIntent(
            intent_class="execution_requested" if execution_requested else "advisory",
            requested=execution_requested,
            handoff_task="Run workspace investigation." if execution_requested else "",
            side_effect_class="external_effect" if execution_requested else "none",
        ),
        requires_confirmation=False,
        metadata=payload_metadata,
    )


if __name__ == "__main__":
    unittest.main()
