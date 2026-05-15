from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from spice.runtime.evidence_requirement import detect_evidence_requirement
from spice.runtime.pre_run_evidence_gate import (
    PRE_RUN_EVIDENCE_BLOCK,
    PRE_RUN_EVIDENCE_CONTINUE,
    PRE_RUN_EVIDENCE_CREATE_INVESTIGATION_CONSENT,
    PRE_RUN_EVIDENCE_RUN_URL_PERCEPTION,
    PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION,
    evaluate_pre_run_evidence_gate,
)
from spice.runtime.resource_extractor import extract_resources
from spice.runtime.route_merge_policy import merge_route_context_policy
from spice.runtime.semantic_router import SemanticRoute
from spice.runtime.workspace_scope import resolve_workspace_scope


class RuntimePreRunEvidenceGateTests(unittest.TestCase):
    def test_repo_evidence_required_runs_workspace_perception_without_artifact(self) -> None:
        policy = merge_route_context_policy(
            SemanticRoute(
                route="new_decision",
                action="new_intent",
                text="基于当前实现判断下一步。",
                context_strategy="none",
                needs_workspace_context=False,
                source="llm",
            ),
            user_input="基于当前实现判断下一步。",
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertFalse(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, ["repo"])

    def test_empty_workspace_artifact_does_not_satisfy_required_repo_evidence(self) -> None:
        policy = merge_route_context_policy(
            SemanticRoute(
                route="new_decision",
                action="new_intent",
                text="基于当前实现判断下一步。",
                context_strategy="none",
                needs_workspace_context=False,
                source="llm",
            ),
            user_input="基于当前实现判断下一步。",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "present": True,
                "perception_id": "workspace.empty",
                "summary": "Workspace perception ran but did not read source-backed evidence.",
                "files_read": [],
                "facts": [],
                "snippets": [],
                "source_count": 0,
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION)
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertFalse(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, ["repo"])

    def test_workspace_summary_without_source_backing_does_not_satisfy_repo_evidence(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "current implementation"},
            user_input="current implementation",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "present": True,
                "perception_id": "workspace.summary-only",
                "summary": "The workspace probably has relevant code.",
                "sources": [],
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION)
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertEqual(decision.missing_source_domains, ["repo"])

    def test_explicit_repo_path_runs_workspace_perception_even_when_llm_route_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            text = f"请读取本地 {root} 这个 repo 的当前实现，再基于实际代码判断。"
            resources = extract_resources(text)
            evidence = detect_evidence_requirement(text, resource_extraction=resources)
            scope = resolve_workspace_scope(project_root=root, resource_extraction=resources)
            policy = merge_route_context_policy(
                SemanticRoute(
                    route="new_decision",
                    action="new_intent",
                    text=text,
                    context_strategy="none",
                    needs_workspace_context=False,
                    source="llm",
                ),
                user_input=text,
                resource_extraction=resources,
                evidence_requirement=evidence,
                workspace_scope=scope,
            )

            decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION)
        self.assertTrue(decision.should_run_workspace_perception)
        self.assertEqual(decision.missing_source_domains, ["repo"])

    def test_repo_evidence_present_allows_high_confidence_evidence_claims(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "current implementation"},
            user_input="current implementation",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "perception_id": "workspace.1",
                "files_read": [{"path": "spice/runtime/run_once.py"}],
                "summary": "Workspace evidence exists.",
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_CONTINUE)
        self.assertTrue(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, [])

    def test_workspace_source_refs_allow_required_repo_evidence(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "current implementation"},
            user_input="current implementation",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "perception_id": "workspace.1",
                "facts": [
                    {
                        "text": "run_once accepts workspace_context.",
                        "source_refs": ["workspace:spice/runtime/run_once.py"],
                    }
                ],
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_CONTINUE)
        self.assertTrue(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, [])

    def test_workspace_source_count_allows_required_repo_evidence(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "current implementation"},
            user_input="current implementation",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "perception_id": "workspace.1",
                "source_count": 2,
                "summary": "Workspace evidence exists.",
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_CONTINUE)
        self.assertTrue(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, [])

    def test_workspace_sources_allow_required_repo_evidence(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "current implementation"},
            user_input="current implementation",
        )

        decision = evaluate_pre_run_evidence_gate(
            policy,
            workspace_context={
                "perception_id": "workspace.1",
                "sources": [{"source_id": "workspace:spice/runtime/run_once.py"}],
                "summary": "Workspace evidence exists.",
            },
        )

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_CONTINUE)
        self.assertTrue(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, [])

    def test_url_evidence_required_runs_url_perception_when_url_exists(self) -> None:
        policy = merge_route_context_policy(
            {"route": "follow_up", "action": "answer_from_decision", "text": "结合链接回答"},
            user_input="结合 https://example.com/spec 回答。",
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_URL_PERCEPTION)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.should_run_url_perception)
        self.assertFalse(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, ["url"])

    def test_explicit_url_runs_url_perception_even_when_llm_route_returns_none(self) -> None:
        text = "结合 https://example.com/spec 再回答。"
        policy = merge_route_context_policy(
            SemanticRoute(
                route="follow_up",
                action="answer_from_decision",
                is_continuation=True,
                text=text,
                context_strategy="none",
                needs_url_context=False,
                source="llm",
            ),
            user_input=text,
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_URL_PERCEPTION)
        self.assertTrue(decision.should_run_url_perception)
        self.assertEqual(decision.missing_source_domains, ["url"])

    def test_url_evidence_required_without_url_blocks(self) -> None:
        policy = merge_route_context_policy(
            {
                "route": "follow_up",
                "action": "answer_from_decision",
                "context_strategy": "url",
                "needs_url_context": True,
                "url_query": "read the linked spec",
                "text": "read the linked spec",
            },
            user_input="read the linked spec",
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_BLOCK)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.missing_source_domains, ["url"])

    def test_external_evidence_required_creates_investigation_consent(self) -> None:
        policy = merge_route_context_policy(
            {
                "route": "new_decision",
                "action": "new_intent",
                "context_strategy": "none",
                "needs_delegated_perception": False,
                "text": "联网查一下最新 Hermes 是怎么处理 read-only routing 的。",
            },
            user_input="联网查一下最新 Hermes 是怎么处理 read-only routing 的。",
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_CREATE_INVESTIGATION_CONSENT)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.should_create_investigation_consent)
        self.assertFalse(decision.can_make_high_confidence_evidence_claims)
        self.assertEqual(decision.missing_source_domains, ["external"])

    def test_mixed_evidence_runs_missing_workspace_before_url_or_external(self) -> None:
        policy = merge_route_context_policy(
            {
                "route": "new_decision",
                "action": "new_intent",
                "context_strategy": "none",
                "text": "基于当前实现和 https://example.com/spec，再联网查一下最新方案。",
            },
            user_input="基于当前实现和 https://example.com/spec，再联网查一下最新方案。",
        )

        decision = evaluate_pre_run_evidence_gate(policy)

        self.assertEqual(decision.action, PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION)
        self.assertEqual(decision.missing_source_domains, ["repo", "url", "external"])


if __name__ == "__main__":
    unittest.main()
