from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from spice.perception.delegated import build_executor_report_artifact
from spice.runtime.delegated_result import (
    DELEGATED_PERCEPTION_NORMALIZER_SCHEMA_VERSION,
    normalize_delegated_perception_result,
)


NOW = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


class DelegatedPerceptionResultNormalizerTests(unittest.TestCase):
    def test_normalizes_structured_executor_report(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="latest agent workflow research",
            request_ref="delegated_request.1",
            executor_run_ref="hermes.run.1",
            structured_output={
                "status": "completed",
                "summary": "Agent workflow examples favor read-only research before execution.",
                "confidence": "high",
                "sources": [
                    {
                        "source_id": "source.1",
                        "source_type": "url",
                        "title": "Agent workflow note",
                        "uri": "https://example.com/agent-workflow",
                        "excerpt": "Research should be separated from execution.",
                        "observed_by": "hermes",
                    }
                ],
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "Read-only investigation should be separated from execution approval.",
                        "confidence": 0.82,
                        "source_refs": ["source.1"],
                    }
                ],
                "limitations": ["web examples were not cross-checked by Spice"],
            },
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(
            executor_report=report,
            request={
                "request_id": "delegated_request.1",
                "consent_id": "investigation.1",
                "query": "latest agent workflow research",
                "context_strategy": "delegated",
                "input_context_refs": ["decision.1"],
                "delegated_plan": {
                    "executor_id": "hermes",
                    "scope": "read_only_investigation",
                    "permission_mode": "read_only",
                    "query": "latest agent workflow research",
                    "requested_capabilities": ["web_research"],
                    "expected_output": "findings_sources_limitations",
                },
                "expected_output": "findings_sources_limitations",
            },
            created_at=NOW,
        )
        payload = result.to_payload()
        artifact = payload["artifact"]

        self.assertEqual(payload["schema_version"], DELEGATED_PERCEPTION_NORMALIZER_SCHEMA_VERSION)
        self.assertEqual(result.parser_status, "parsed")
        self.assertEqual(artifact["status"], "completed")
        self.assertEqual(artifact["executor_id"], "hermes")
        self.assertEqual(artifact["request_ref"], "delegated_request.1")
        self.assertEqual(artifact["executor_report_ref"], report.report_id)
        self.assertEqual(artifact["executor_run_ref"], "hermes.run.1")
        self.assertEqual(artifact["consent_id"], "investigation.1")
        self.assertEqual(artifact["findings"][0]["source_refs"], ["source.1"])
        self.assertEqual(artifact["findings"][0]["confidence"], 0.82)
        self.assertEqual(artifact["sources"][0]["verification_status"], "reported_by_executor")
        self.assertEqual(artifact["metadata"]["expected_output"], "findings_sources_limitations")
        self.assertEqual(artifact["metadata"]["delegated_plan"]["executor_id"], "hermes")
        self.assertEqual(artifact["metadata"]["finding_source_binding"]["status"], "complete")
        self.assertEqual(artifact["metadata"]["finding_source_binding"]["sourced_finding_count"], 1)
        self.assertEqual(result.context["source"], "delegated_perception")
        self.assertNotIn("raw_output", json.dumps(result.context))
        self.assertNotIn("structured_output", json.dumps(result.context))

    def test_parses_markdown_fenced_raw_json(self) -> None:
        report = build_executor_report_artifact(
            executor_id="codex",
            query="inspect repo architecture",
            raw_output="""```json
{
  "status": "completed",
  "summary": "Repo has a runtime policy layer.",
  "sources": [
    {
      "source_id": "source.1",
      "source_type": "file",
      "uri": "spice/runtime/escalation_policy.py",
      "excerpt": "decide_runtime_escalation"
    }
  ],
  "findings": [
    {
      "text": "Runtime policy chooses local, URL, delegated, or execution boundaries.",
      "confidence": 0.74,
      "source_refs": ["source.1"]
    }
  ]
}
```""",
            request_ref="delegated_request.2",
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(
            executor_report=report,
            request={"request_id": "delegated_request.2", "consent_id": "investigation.2"},
            created_at=NOW,
        )

        self.assertEqual(result.parser_status, "parsed")
        self.assertEqual(result.artifact.status, "completed")
        self.assertEqual(result.artifact.sources[0].uri, "spice/runtime/escalation_policy.py")
        self.assertEqual(result.artifact.findings[0].source_refs, ["source.1"])

    def test_malformed_raw_output_becomes_failed_artifact_without_raw_context(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="deep research",
            raw_output="not json, just a long raw executor transcript",
            request_ref="delegated_request.3",
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(
            executor_report=report,
            request={"request_id": "delegated_request.3", "consent_id": "investigation.3"},
            created_at=NOW,
        )

        self.assertEqual(result.parser_status, "malformed")
        self.assertEqual(result.fallback_reason, "executor_report_output_malformed")
        self.assertEqual(result.artifact.status, "failed")
        self.assertIn("executor_report_output_malformed", result.artifact.limitations)
        self.assertEqual(result.context["findings"], [])
        self.assertNotIn("long raw executor transcript", json.dumps(result.context))

    def test_unsourced_finding_is_low_confidence(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="research",
            structured_output={
                "status": "completed",
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "This claim has no cited source.",
                        "confidence": 0.91,
                    }
                ],
            },
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(executor_report=report, created_at=NOW)

        finding = result.artifact.to_payload()["findings"][0]
        self.assertEqual(finding["confidence"], 0.35)
        self.assertIn("no_source_refs", finding["limitations"])
        self.assertIn("finding.1.no_source_refs", result.warnings)
        self.assertEqual(
            result.artifact.to_payload()["metadata"]["finding_source_binding"]["unsourced_finding_count"],
            1,
        )

    def test_missing_source_refs_are_removed_and_limited(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="research",
            structured_output={
                "sources": [{"source_id": "source.1", "uri": "https://example.com", "excerpt": "ok"}],
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "This claim cites one missing source.",
                        "confidence": 0.8,
                        "source_refs": ["source.1", "source.missing"],
                    }
                ],
            },
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(executor_report=report, created_at=NOW)
        finding = result.artifact.to_payload()["findings"][0]

        self.assertEqual(finding["source_refs"], ["source.1"])
        self.assertEqual(finding["confidence"], 0.35)
        self.assertIn("missing_source_refs:source.missing", finding["limitations"])
        self.assertIn("finding.1.missing_source_refs", result.warnings)

    def test_incomplete_source_is_marked_unverified(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="research",
            structured_output={
                "sources": [{"source_id": "source.1", "title": "No URI or excerpt"}],
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "Finding references incomplete source.",
                        "source_refs": ["source.1"],
                    }
                ],
            },
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(executor_report=report, created_at=NOW)
        source = result.artifact.to_payload()["sources"][0]

        self.assertEqual(source["verification_status"], "unverified")
        self.assertTrue(source["metadata"]["incomplete_source"])
        self.assertEqual(source["metadata"]["missing_fields"], ["uri", "excerpt"])
        self.assertIn("source.1.incomplete_source:uri,excerpt", result.artifact.limitations)

    def test_executor_decision_fields_are_ignored(self) -> None:
        report = build_executor_report_artifact(
            executor_id="hermes",
            query="research",
            structured_output={
                "recommendation": "Execute B now.",
                "selected_candidate_id": "candidate.b",
                "sources": [{"source_id": "source.1", "uri": "https://example.com", "excerpt": "ok"}],
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "The executor reported a finding.",
                        "source_refs": ["source.1"],
                    }
                ],
            },
            created_at=NOW,
        )

        result = normalize_delegated_perception_result(executor_report=report, created_at=NOW)
        payload = result.artifact.to_payload()

        self.assertIn("executor_decision_fields_ignored", payload["limitations"])
        self.assertIn("recommendation", payload["metadata"]["ignored_executor_fields"])
        self.assertNotIn("selected_candidate_id", result.context)
        self.assertNotIn("Execute B now", json.dumps(result.context))


if __name__ == "__main__":
    unittest.main()
