from __future__ import annotations

from datetime import datetime, timezone
import unittest

from spice.perception import (
    DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION,
    DELEGATED_PERCEPTION_SCHEMA_VERSION,
    EXECUTOR_REPORT_SCHEMA_VERSION,
    INVESTIGATION_CONSENT_GRANTED,
    INVESTIGATION_CONSENT_PENDING,
    INVESTIGATION_CONSENT_SCHEMA_VERSION,
    DelegatedPerceptionArtifact,
    ExecutorReportArtifact,
    InvestigationConsent,
    InvestigationConsentBudget,
    build_delegated_perception_artifact,
    build_executor_report_artifact,
    build_investigation_consent,
    delegated_perception_context_from_artifact,
    resolve_investigation_consent,
)


class DelegatedPerceptionArtifactTests(unittest.TestCase):
    def test_builds_investigation_consent_with_read_only_boundary(self) -> None:
        consent = build_investigation_consent(
            executor_id="hermes",
            query="research current OpenChronicle integration options",
            input_context_refs=["decision.1"],
            budget=InvestigationConsentBudget(
                max_duration_sec=90,
                max_sources=6,
                max_repo_files=8,
                max_tokens=12000,
            ),
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        payload = consent.to_payload()
        self.assertEqual(payload["schema_version"], INVESTIGATION_CONSENT_SCHEMA_VERSION)
        self.assertTrue(payload["consent_id"].startswith("investigation."))
        self.assertEqual(payload["scope"], "read_only_investigation")
        self.assertEqual(payload["permission_mode"], "read_only")
        self.assertEqual(payload["status"], INVESTIGATION_CONSENT_PENDING)
        self.assertIn("web_search", payload["allowed_actions"])
        self.assertIn("repo_inspection", payload["allowed_actions"])
        self.assertIn("write_file", payload["denied_actions"])
        self.assertIn("terminal_command", payload["denied_actions"])
        self.assertEqual(payload["budget"]["max_duration_sec"], 90)
        self.assertEqual(payload["input_context_refs"], ["decision.1"])
        self.assertTrue(payload["expires_at"])

        restored = InvestigationConsent.from_payload(payload)
        self.assertEqual(restored.consent_id, consent.consent_id)
        self.assertEqual(restored.budget.max_sources, 6)

    def test_resolves_investigation_consent_without_execution_approval_semantics(self) -> None:
        consent = build_investigation_consent(
            executor_id="codex",
            query="inspect repo read-only",
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        resolved = resolve_investigation_consent(
            consent,
            status=INVESTIGATION_CONSENT_GRANTED,
            reason="Read-only investigation is fine.",
            resolved_at=datetime(2026, 5, 13, 0, 1, tzinfo=timezone.utc),
        )
        payload = resolved.to_payload()

        self.assertEqual(payload["status"], INVESTIGATION_CONSENT_GRANTED)
        self.assertEqual(payload["response"], INVESTIGATION_CONSENT_GRANTED)
        self.assertEqual(payload["reason"], "Read-only investigation is fine.")
        self.assertEqual(payload["metadata"]["resolved_by"], "spice.perception.delegated")
        self.assertNotIn("execution_allowed", payload)
        self.assertNotIn("approval_id", payload)

        with self.assertRaisesRegex(ValueError, "not pending"):
            resolve_investigation_consent(resolved, status=INVESTIGATION_CONSENT_GRANTED)

    def test_builds_executor_report_artifact(self) -> None:
        artifact = build_executor_report_artifact(
            executor_id="hermes",
            query="investigate repo architecture",
            raw_output="Hermes raw report with sources",
            structured_output={"findings": [{"text": "runtime owns guardrails"}]},
            request_ref="request.1",
            executor_run_ref="hermes.run.1",
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        payload = artifact.to_payload()
        self.assertEqual(payload["schema_version"], EXECUTOR_REPORT_SCHEMA_VERSION)
        self.assertTrue(payload["report_id"].startswith("executor_report."))
        self.assertEqual(payload["executor_id"], "hermes")
        self.assertEqual(payload["scope"], "read_only_investigation")
        self.assertEqual(payload["permission_mode"], "read_only")
        self.assertEqual(payload["raw_output"], "Hermes raw report with sources")
        self.assertEqual(payload["structured_output"]["findings"][0]["text"], "runtime owns guardrails")
        self.assertEqual(payload["request_ref"], "request.1")
        self.assertEqual(payload["executor_run_ref"], "hermes.run.1")

        restored = ExecutorReportArtifact.from_payload(payload)
        self.assertEqual(restored.report_id, artifact.report_id)
        self.assertEqual(restored.structured_output["findings"][0]["text"], "runtime owns guardrails")

    def test_builds_delegated_perception_artifact_and_lowers_unsourced_finding_confidence(self) -> None:
        artifact = build_delegated_perception_artifact(
            executor_id="hermes",
            query="research whether state-as-context should come first",
            status="completed",
            consent_id="investigation.1",
            request_ref="request.1",
            executor_report_ref="executor_report.1",
            executor_run_ref="hermes.run.1",
            findings=[
                {
                    "finding_id": "finding.1",
                    "text": "State-as-context is a prerequisite for reliable proactive perception.",
                    "confidence": 0.82,
                    "source_refs": ["source.1"],
                },
                {
                    "finding_id": "finding.2",
                    "text": "A second claim without sources should stay low confidence.",
                    "confidence": 0.91,
                    "source_refs": [],
                },
            ],
            sources=[
                {
                    "source_id": "source.1",
                    "source_type": "repo",
                    "title": "Architecture note",
                    "uri": "file://docs/architecture.md",
                    "excerpt": "State feeds decisions.",
                    "observed_by": "hermes",
                }
            ],
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        payload = artifact.to_payload()
        self.assertEqual(payload["schema_version"], DELEGATED_PERCEPTION_SCHEMA_VERSION)
        self.assertTrue(payload["perception_id"].startswith("delegated."))
        self.assertTrue(payload["delegation_id"].startswith("delegation."))
        self.assertEqual(payload["permission_mode"], "read_only")
        self.assertEqual(payload["sources"][0]["verification_status"], "reported_by_executor")
        self.assertEqual(payload["findings"][0]["confidence"], 0.82)
        self.assertEqual(payload["findings"][1]["confidence"], 0.35)
        self.assertIn("no_source_refs", payload["findings"][1]["limitations"])

    def test_delegated_perception_payload_round_trips(self) -> None:
        artifact = build_delegated_perception_artifact(
            executor_id="codex",
            query="inspect docs and repo",
            input_context_refs=["decision.1", "workspace.1"],
            findings=[
                {
                    "finding_id": "finding.1",
                    "text": "The repo already has a local workspace inspector.",
                    "confidence": 0.7,
                    "source_refs": ["source.1"],
                }
            ],
            sources=[
                {
                    "source_id": "source.1",
                    "source_type": "file",
                    "uri": "spice/perception/workspace_inspector.py",
                    "excerpt": "class WorkspaceInspector",
                    "observed_by": "codex",
                    "accessed_at": "2026-05-13T00:00:00Z",
                    "verification_status": "reported_by_executor",
                }
            ],
            limitations=["not verified by Spice direct inspection"],
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        restored = DelegatedPerceptionArtifact.from_payload(artifact.to_payload())

        self.assertEqual(restored.perception_id, artifact.perception_id)
        self.assertEqual(restored.input_context_refs, ["decision.1", "workspace.1"])
        self.assertEqual(restored.findings[0].source_refs, ["source.1"])
        self.assertEqual(restored.sources[0].uri, "spice/perception/workspace_inspector.py")
        self.assertEqual(restored.limitations, ["not verified by Spice direct inspection"])

    def test_delegated_perception_context_is_compact_and_omits_raw_report(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="deep external investigation",
            raw_output="raw executor transcript that should not enter composer context",
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )
        artifact = build_delegated_perception_artifact(
            executor_id="hermes",
            query="deep external investigation",
            status="completed",
            executor_report_ref=report.report_id,
            findings=[
                {
                    "finding_id": "finding.1",
                    "text": "Hermes reported that executor handoff needs clearer result feedback.",
                    "confidence": 0.76,
                    "source_refs": ["source.1"],
                }
            ],
            sources=[
                {
                    "source_id": "source.1",
                    "source_type": "executor_report",
                    "title": "Hermes investigation report",
                    "uri": report.report_id,
                    "excerpt": "executor handoff needs clearer result feedback",
                    "observed_by": "hermes",
                }
            ],
            created_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
        )

        context = delegated_perception_context_from_artifact(artifact)

        self.assertEqual(context["schema_version"], DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(context["source"], "delegated_perception")
        self.assertEqual(context["executor_id"], "hermes")
        self.assertEqual(context["executor_report_ref"], report.report_id)
        self.assertEqual(context["findings"][0]["source_refs"], ["source.1"])
        self.assertEqual(context["sources"][0]["observed_by"], "hermes")
        self.assertNotIn("raw_output", context)
        self.assertNotIn("structured_output", context)


if __name__ == "__main__":
    unittest.main()
