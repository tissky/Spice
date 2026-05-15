from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from spice.runtime.evidence_requirement import detect_evidence_requirement
from spice.runtime.resource_extractor import extract_resources
from spice.runtime.route_merge_policy import (
    CONTEXT_STRATEGY_DELEGATED,
    CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED,
    CONTEXT_STRATEGY_LOCAL_WORKSPACE,
    CONTEXT_STRATEGY_URL,
    FORCED_BY_EXPLICIT_URL,
    FORCED_BY_EXTERNAL_EVIDENCE_REQUIREMENT,
    FORCED_BY_REPO_EVIDENCE_REQUIREMENT,
    FORCED_BY_WORKSPACE_SCOPE_BLOCKED,
    FORCED_BY_WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
    ROUTE_MERGE_POLICY_SCHEMA_VERSION,
    merge_route_context_policy,
)
from spice.runtime.semantic_router import SemanticRoute
from spice.runtime.workspace_scope import (
    WORKSPACE_SCOPE_BLOCKED,
    WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
    resolve_workspace_scope,
)


class RuntimeRouteMergePolicyTests(unittest.TestCase):
    def test_explicit_workspace_path_forces_workspace_context_when_llm_route_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            text = f"请读取本地 {root} 这个 repo 的当前实现再判断下一步。"
            resources = extract_resources(text)
            evidence = detect_evidence_requirement(text, resource_extraction=resources)
            scope = resolve_workspace_scope(project_root=root, resource_extraction=resources)
            route = SemanticRoute(
                route="new_decision",
                action="new_intent",
                text=text,
                context_strategy="none",
                needs_workspace_context=False,
                source="llm",
            )

            merged = merge_route_context_policy(
                route,
                user_input=text,
                resource_extraction=resources,
                evidence_requirement=evidence,
                workspace_scope=scope,
            )

            self.assertEqual(merged.context_strategy, CONTEXT_STRATEGY_LOCAL_WORKSPACE)
            self.assertTrue(merged.needs_workspace_context)
            self.assertIn(FORCED_BY_REPO_EVIDENCE_REQUIREMENT, merged.forced_by)
            self.assertEqual(merged.to_route_payload()["route"], "new_decision")
            self.assertEqual(merged.to_route_payload()["action"], "new_intent")

    def test_explicit_url_forces_url_context_when_llm_route_misses(self) -> None:
        text = "结合 https://example.com/spec 再回答。"
        route = SemanticRoute(
            route="follow_up",
            action="answer_from_decision",
            is_continuation=True,
            text=text,
            context_strategy="none",
            needs_url_context=False,
            source="llm",
        )

        merged = merge_route_context_policy(route, user_input=text)

        self.assertEqual(merged.context_strategy, CONTEXT_STRATEGY_URL)
        self.assertTrue(merged.needs_url_context)
        self.assertEqual(merged.urls, ["https://example.com/spec"])
        self.assertIn(FORCED_BY_EXPLICIT_URL, merged.forced_by)

    def test_external_research_requirement_forces_delegated_perception(self) -> None:
        text = "联网查一下最新 Hermes 和 OpenClaw 是怎么做 read-only routing 的。"
        route = SemanticRoute(
            route="new_decision",
            action="new_intent",
            text=text,
            context_strategy="none",
            needs_delegated_perception=False,
            source="llm",
        )

        merged = merge_route_context_policy(route, user_input=text)

        self.assertEqual(merged.context_strategy, CONTEXT_STRATEGY_DELEGATED)
        self.assertTrue(merged.needs_delegated_perception)
        self.assertIn("web_research", merged.suggested_capabilities)
        self.assertIn(FORCED_BY_EXTERNAL_EVIDENCE_REQUIREMENT, merged.forced_by)

    def test_repo_requirement_plus_delegated_route_runs_local_first_then_delegates_if_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            text = f"先读 {root} 当前实现，如果不够再让 Hermes 深度调查。"
            resources = extract_resources(text)
            evidence = detect_evidence_requirement(text, resource_extraction=resources)
            scope = resolve_workspace_scope(project_root=root, resource_extraction=resources)
            route = SemanticRoute(
                route="new_decision",
                action="new_intent",
                text=text,
                context_strategy="delegated",
                needs_delegated_perception=True,
                delegated_perception_query="investigate external examples",
                suggested_capabilities=["web_research"],
                source="llm",
            )

            merged = merge_route_context_policy(
                route,
                user_input=text,
                resource_extraction=resources,
                evidence_requirement=evidence,
                workspace_scope=scope,
            )

            self.assertEqual(merged.context_strategy, CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED)
            self.assertTrue(merged.needs_workspace_context)
            self.assertTrue(merged.needs_delegated_perception)
            self.assertEqual(merged.delegated_perception_query, "investigate external examples")

    def test_external_repo_scope_needs_confirmation_is_not_silently_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir)
            (external / "pyproject.toml").write_text("[project]\nname='external'\n", encoding="utf-8")
            text = f"请读取本地 {external} 这个 repo 的当前实现。"
            resources = extract_resources(text)
            scope = resolve_workspace_scope(
                project_root=current_dir,
                resource_extraction=resources,
                interactive=True,
            )

            merged = merge_route_context_policy(
                {"route": "new_decision", "action": "new_intent", "text": text},
                user_input=text,
                resource_extraction=resources,
                workspace_scope=scope,
            )

            self.assertEqual(scope.status, WORKSPACE_SCOPE_NEEDS_CONFIRMATION)
            self.assertTrue(merged.needs_workspace_context)
            self.assertEqual(merged.context_strategy, CONTEXT_STRATEGY_LOCAL_WORKSPACE)
            self.assertIn(FORCED_BY_WORKSPACE_SCOPE_NEEDS_CONFIRMATION, merged.forced_by)
            self.assertTrue(merged.limitations)

    def test_blocked_scope_keeps_workspace_requirement_for_evidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir)
            (external / "package.json").write_text('{"name":"external"}\n', encoding="utf-8")
            text = f"基于实际代码读取 {external} 再回答。"
            resources = extract_resources(text)
            scope = resolve_workspace_scope(
                project_root=current_dir,
                resource_extraction=resources,
                interactive=False,
            )

            merged = merge_route_context_policy(
                {"route": "new_decision", "action": "new_intent", "text": text},
                user_input=text,
                resource_extraction=resources,
                workspace_scope=scope,
            )

            self.assertEqual(scope.status, WORKSPACE_SCOPE_BLOCKED)
            self.assertTrue(merged.needs_workspace_context)
            self.assertIn(FORCED_BY_WORKSPACE_SCOPE_BLOCKED, merged.forced_by)
            self.assertTrue(any("blocked" in item for item in merged.limitations))

    def test_payload_contains_merged_route_and_stable_schema(self) -> None:
        text = "结合 https://example.com/spec 给我一个计划。"
        merged = merge_route_context_policy(
            {
                "route": "follow_up",
                "action": "plan_candidate",
                "candidate_id": "candidate.a",
                "text": text,
            },
            user_input=text,
        )

        payload = merged.to_payload()

        self.assertEqual(payload["schema_version"], ROUTE_MERGE_POLICY_SCHEMA_VERSION)
        self.assertEqual(payload["merged_route"]["candidate_id"], "candidate.a")
        self.assertTrue(payload["merged_route"]["needs_url_context"])
        self.assertIn("route_merge_policy", payload["merged_route"])


if __name__ == "__main__":
    unittest.main()
