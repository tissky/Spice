from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest

from spice.perception import (
    INVESTIGATION_CONSENT_GRANTED,
    build_investigation_consent,
    resolve_investigation_consent,
)
from spice.runtime import LocalJsonStore, setup_workspace, update_workspace_config
from spice.runtime.delegated_perception import run_delegated_perception_handoff


class RuntimeDelegatedPerceptionHandoffTests(unittest.TestCase):
    def test_handoff_runs_read_only_executor_and_records_report_and_perception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "sdep_subprocess")
            update_workspace_config(tmp_dir, "executor_command", _json_report_command())
            store = LocalJsonStore.from_project_root(tmp_dir)
            consent = _granted_consent("sdep_subprocess", "research routing patterns")
            store.save_investigation_consent(consent.consent_id, consent.to_payload())

            result = run_delegated_perception_handoff(
                project_root=tmp_dir,
                store=store,
                consent=consent,
                escalation_decision=_escalation("sdep_subprocess", consent),
                route_payload={"context_strategy": "delegated"},
                user_input="Ask the executor to investigate routing patterns.",
            )

            self.assertEqual(result.status, "completed")
            self.assertEqual(store.list_record_ids("approvals"), [])
            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertIn(result.executor_report["report_id"], store.list_record_ids("perceptions"))
            perception_id = result.perception.artifact["perception_id"]
            self.assertIn(perception_id, store.list_record_ids("perceptions"))
            perception = store.load_perception(perception_id)
            self.assertEqual(perception["status"], "completed")
            self.assertEqual(perception["permission_mode"], "read_only")
            self.assertEqual(perception["executor_report_ref"], result.executor_report["report_id"])
            self.assertEqual(perception["findings"][0]["source_refs"], ["source.1"])
            self.assertEqual(perception["sources"][0]["observed_by"], "sdep_subprocess")
            self.assertEqual(result.request["mode"], "perception")
            self.assertEqual(result.request["permission_mode"], "read_only")
            self.assertEqual(result.request["expected_output"], "findings_sources_limitations")
            self.assertEqual(result.executor_report["metadata"]["expected_output"], "findings_sources_limitations")
            self.assertEqual(perception["metadata"]["expected_output"], "findings_sources_limitations")
            self.assertEqual(perception["metadata"]["finding_source_binding"]["status"], "complete")

    def test_unsupported_executor_writes_failed_report_and_perception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "dry_run")
            store = LocalJsonStore.from_project_root(tmp_dir)
            consent = _granted_consent("dry_run", "research routing patterns")
            store.save_investigation_consent(consent.consent_id, consent.to_payload())

            result = run_delegated_perception_handoff(
                project_root=tmp_dir,
                store=store,
                consent=consent,
                escalation_decision=_escalation("dry_run", consent),
                route_payload={"context_strategy": "delegated"},
                user_input="Ask the executor to investigate routing patterns.",
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(store.list_record_ids("approvals"), [])
            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertIn(result.executor_report["report_id"], store.list_record_ids("perceptions"))
            perception = store.load_perception(result.perception.artifact["perception_id"])
            self.assertEqual(perception["status"], "failed")
            self.assertIn("does not support delegated perception", " ".join(perception["limitations"]))

    def test_executor_failure_records_failed_report_without_execution_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "sdep_subprocess")
            update_workspace_config(tmp_dir, "executor_command", _failing_command())
            store = LocalJsonStore.from_project_root(tmp_dir)
            consent = _granted_consent("sdep_subprocess", "research routing patterns")
            store.save_investigation_consent(consent.consent_id, consent.to_payload())

            result = run_delegated_perception_handoff(
                project_root=tmp_dir,
                store=store,
                consent=consent,
                escalation_decision=_escalation("sdep_subprocess", consent),
                route_payload={"context_strategy": "delegated"},
                user_input="Ask the executor to investigate routing patterns.",
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertEqual(result.executor_report["status"], "failed")
            self.assertEqual(result.executor_report["metadata"]["command_result"]["exit_code"], 3)
            perception = store.load_perception(result.perception.artifact["perception_id"])
            self.assertEqual(perception["status"], "failed")


def _granted_consent(executor_id: str, query: str):
    pending = build_investigation_consent(executor_id=executor_id, query=query)
    return resolve_investigation_consent(
        pending,
        status=INVESTIGATION_CONSENT_GRANTED,
    )


def _escalation(executor_id: str, consent) -> dict[str, object]:
    return {
        "action": "run_delegated_perception",
        "context_strategy": "delegated",
        "executor_id": executor_id,
        "delegated_perception_query": consent.query,
        "delegated_perception_reason": "requires external read-only investigation",
        "delegated_plan": {
            "executor_id": executor_id,
            "scope": "read_only_investigation",
            "permission_mode": "read_only",
            "query": consent.query,
            "requested_capabilities": ["web_research"],
            "expected_output": "findings_sources_limitations",
        },
        "suggested_capabilities": ["web_research"],
        "consent_id": consent.consent_id,
    }


def _json_report_command() -> str:
    code = """
import json, sys
request = json.loads(sys.stdin.read())
query = request.get("query", "")
print(json.dumps({
    "status": "completed",
    "summary": "Executor investigated the requested topic.",
    "findings": [{
        "finding_id": "finding.1",
        "text": "Read-only investigation returned a supported source.",
        "confidence": 0.8,
        "source_refs": ["source.1"],
        "limitations": []
    }],
    "sources": [{
        "source_id": "source.1",
        "source_type": "executor_report",
        "title": "Synthetic investigation report",
        "uri": "executor://synthetic",
        "excerpt": query or "read-only investigation",
        "observed_by": "sdep_subprocess",
        "verification_status": "reported_by_executor"
    }],
    "limitations": [],
    "confidence": "medium"
}))
""".strip()
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _failing_command() -> str:
    code = "import sys; sys.stderr.write('executor failed'); sys.exit(3)"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


if __name__ == "__main__":
    unittest.main()
