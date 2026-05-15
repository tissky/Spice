from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from spice.perception import build_delegated_perception_artifact
from spice.runtime import (
    LocalJsonStore,
    build_general_delegated_perception_memory_record,
    finalize_runtime_delegated_perception_result,
    setup_workspace,
)
from spice.runtime.memory_writeback import (
    GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE,
    write_general_delegated_perception_memory,
)
from spice.runtime.workspace import load_workspace_memory_provider


NOW = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)


class RuntimeDelegatedPerceptionMemoryTests(unittest.TestCase):
    def test_delegated_perception_memory_record_is_compact_and_linked(self) -> None:
        artifact = _artifact()

        record = build_general_delegated_perception_memory_record(
            artifact=artifact,
            user_input="让 Hermes 查一下最新材料。",
            route_result={
                "route": "follow_up",
                "context_strategy": "delegated",
                "needs_delegated_perception": True,
            },
            linked_decision_id="decision.test",
            linked_run_id="run.test",
            conversation_turn_id="turn.test",
        )

        self.assertEqual(record["record_type"], "general.delegated_perception")
        self.assertEqual(record["perception_id"], "delegated.test")
        self.assertEqual(record["delegation_id"], "delegation.test")
        self.assertEqual(record["context_strategy"], "delegated")
        self.assertEqual(record["consent_id"], "investigation.test")
        self.assertEqual(record["linked"]["decision_id"], "decision.test")
        self.assertEqual(record["linked"]["run_id"], "run.test")
        self.assertEqual(record["linked"]["conversation_turn_id"], "turn.test")
        self.assertEqual(record["route_result"]["context_strategy"], "delegated")
        self.assertEqual(record["findings"][0]["source_refs"], ["source.1"])
        self.assertEqual(record["source_refs"][0]["uri"], "https://example.com/report")
        self.assertIn("raw_output_retained_in_executor_report", record["metadata"])
        serialized = json.dumps(record)
        self.assertNotIn("raw-output-secret", serialized)
        self.assertNotIn("structured-output-secret", serialized)
        self.assertNotIn("raw_executor_report", serialized)

    def test_write_general_delegated_perception_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            provider = load_workspace_memory_provider(tmp_dir)

            result = write_general_delegated_perception_memory(
                provider,
                artifact=_artifact(),
                user_input="investigate with Hermes",
                route_result={"route": "new_decision", "context_strategy": "delegated"},
                linked_decision_id="decision.mem",
                linked_run_id="run.mem",
                conversation_turn_id="turn.mem",
            )

            self.assertEqual(result["namespace"], GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE)
            records = provider.query(namespace="general.delegated_perception", limit=-1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["perception_id"], "delegated.test")
            self.assertEqual(records[0]["user_input"], "investigate with Hermes")
            self.assertEqual(records[0]["linked"]["conversation_turn_id"], "turn.mem")

    def test_finalize_runtime_delegated_perception_result_persists_artifact_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            result = finalize_runtime_delegated_perception_result(
                project_root=tmp_dir,
                store=store,
                artifact_payload=_artifact(),
                requested=True,
                status="written",
                user_input="based on delegated investigation",
                route_result={"route": "follow_up", "context_strategy": "delegated"},
                linked_decision_id="decision.finalize",
                linked_run_id="run.finalize",
                conversation_turn_id="turn.finalize",
            )

            self.assertEqual(result.status, "written")
            self.assertTrue(result.path and result.path.exists())
            self.assertEqual(result.context["source"], "delegated_perception")
            self.assertEqual(result.context["perception_id"], "delegated.test")
            self.assertEqual(result.memory_writeback["namespace"], "general.delegated_perception")
            saved = store.load_perception("delegated.test")
            self.assertEqual(saved["memory_writeback"]["namespace"], "general.delegated_perception")
            provider = load_workspace_memory_provider(tmp_dir)
            records = provider.query(namespace="general.delegated_perception", limit=-1)
            self.assertEqual(records[0]["linked"]["decision_id"], "decision.finalize")

    def test_finalize_runtime_delegated_perception_result_respects_preview_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            result = finalize_runtime_delegated_perception_result(
                project_root=tmp_dir,
                store=store,
                artifact_payload=_artifact(),
                status="preview",
                persist=False,
            )

            self.assertEqual(result.status, "preview")
            self.assertIsNone(result.path)
            self.assertEqual(result.memory_writeback["status"], "skipped")
            self.assertEqual(result.memory_writeback["reason"], "not_persisted")
            provider = load_workspace_memory_provider(tmp_dir)
            self.assertEqual(
                provider.query(namespace="general.delegated_perception", limit=-1),
                [],
            )


def _artifact() -> dict[str, object]:
    return build_delegated_perception_artifact(
        executor_id="hermes",
        query="research delegated perception",
        perception_id="delegated.test",
        delegation_id="delegation.test",
        consent_id="investigation.test",
        request_ref="request.test",
        executor_report_ref="executor_report.test",
        executor_run_ref="hermes.run.test",
        status="completed",
        context_strategy="delegated",
        input_context_refs=["decision.test", "turn.test"],
        findings=[
            {
                "finding_id": "finding.1",
                "text": "Hermes reported that delegated perception should stay read-only.",
                "confidence": 0.78,
                "source_refs": ["source.1"],
                "limitations": ["reported_by_executor"],
                "metadata": {"raw_executor_report": "raw-output-secret"},
            }
        ],
        sources=[
            {
                "source_id": "source.1",
                "source_type": "url",
                "title": "Delegated report",
                "uri": "https://example.com/report",
                "excerpt": "A short source excerpt.",
                "observed_by": "hermes",
                "accessed_at": "2026-05-13T09:00:00Z",
                "verification_status": "reported_by_executor",
                "metadata": {"structured_output": "structured-output-secret"},
            }
        ],
        limitations=["not verified directly by Spice"],
        confidence="medium",
        summary="Hermes returned one sourced delegated finding.",
        created_at=NOW,
        metadata={
            "source": "test",
            "parser_status": "parsed",
            "raw_output_retained_in_executor_report": True,
            "raw_output": "raw-output-secret",
            "structured_output": "structured-output-secret",
        },
    ).to_payload()
