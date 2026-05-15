from __future__ import annotations

import unittest

from spice.perception import build_evidence_context, compact_evidence_context


class PerceptionEvidenceContextTests(unittest.TestCase):
    def test_builds_normalized_view_across_workspace_url_and_delegated_sources(self) -> None:
        context = build_evidence_context(
            requirements={
                "requires_evidence": True,
                "evidence_domain": "mixed",
                "reason": "repo and external evidence requested",
            },
            workspace_context={
                "source": "workspace_perception",
                "perception_id": "workspace.1",
                "summary": "Workspace inspection read run_once.",
                "files_read": [{"path": "spice/runtime/run_once.py", "chars_read": 1200}],
                "exploration_status": "partial",
                "depth": "normal",
                "budget_used": {
                    "tool_calls_executed": 4,
                    "chars_used": 1200,
                    "total_char_budget": 500000,
                    "budget_pressure": "low",
                },
                "budget_pressure_events": [{"budget_pressure": "medium"}],
                "limitations": ["did not read tests"],
                "facts": [
                    {
                        "text": "run_once accepts workspace context.",
                        "source_path": "spice/runtime/run_once.py",
                    }
                ],
            },
            url_context={
                "source": "url_perception",
                "perception_id": "url.1",
                "urls": ["https://example.com/design"],
                "documents": [{"url": "https://example.com/design", "title": "Design"}],
            },
            delegated_perception_context={
                "source": "delegated_perception",
                "perception_id": "delegated.1",
                "executor_id": "hermes",
                "consent_id": "investigation.1",
                "confidence": "medium",
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "Hermes reported a sourced finding.",
                        "source_refs": ["source.1"],
                    }
                ],
                "sources": [{"source_id": "source.1", "uri": "https://example.com/report"}],
            },
        )

        self.assertTrue(context["workspace"]["present"])
        self.assertEqual(context["workspace"]["observed_by"], "spice")
        self.assertEqual(context["workspace"]["source_count"], 1)
        self.assertEqual(context["workspace"]["exploration_status"], "partial")
        self.assertEqual(context["workspace"]["depth"], "normal")
        self.assertEqual(context["workspace"]["budget_used"]["tool_calls_executed"], 4)
        self.assertEqual(context["workspace"]["budget_pressure_event_count"], 1)
        self.assertIn("did not read tests", context["limitations"])
        self.assertTrue(context["url"]["present"])
        self.assertEqual(context["url"]["source_count"], 1)
        self.assertTrue(context["delegated"]["present"])
        self.assertEqual(context["delegated"]["observed_by"], "hermes")
        self.assertEqual(context["delegated"]["consent_id"], "investigation.1")
        self.assertEqual(context["requirements"]["evidence_domain"], "mixed")
        self.assertEqual(context["confidence"], "medium")
        self.assertEqual(
            {source["source_id"] for source in context["sources"]},
            {
                "workspace:spice/runtime/run_once.py",
                "url:https://example.com/design",
                "source.1",
            },
        )
        self.assertEqual(
            {source["observed_by"] for source in context["sources"]},
            {"spice", "hermes"},
        )
        workspace_finding = next(
            item
            for item in context["findings"]
            if item["finding_id"] == "workspace.finding.1"
        )
        self.assertEqual(workspace_finding["source_refs"], ["workspace:spice/runtime/run_once.py"])
        self.assertEqual(workspace_finding["observed_by"], "spice")
        delegated_finding = next(
            item
            for item in context["findings"]
            if item["finding_id"] == "finding.1"
        )
        self.assertEqual(delegated_finding["source_refs"], ["source.1"])
        self.assertEqual(delegated_finding["observed_by"], "hermes")

    def test_empty_context_is_explicitly_no_evidence(self) -> None:
        context = build_evidence_context()

        self.assertFalse(context["workspace"]["present"])
        self.assertFalse(context["url"]["present"])
        self.assertFalse(context["delegated"]["present"])
        self.assertEqual(context["confidence"], "none")

    def test_compact_context_keeps_only_source_counts_and_refs(self) -> None:
        context = build_evidence_context(
            workspace_context={
                "source": "workspace_perception",
                "perception_id": "workspace.compact",
                "summary": "A" * 1000,
                "files_read": [{"path": "spice/runtime/run_once.py"}],
                "raw_file_contents": "drop",
            }
        )

        compact = compact_evidence_context(context)

        self.assertEqual(compact["workspace"]["perception_id"], "workspace.compact")
        self.assertEqual(compact["workspace"]["source_count"], 1)
        self.assertEqual(compact["sources"][0]["source_id"], "workspace:spice/runtime/run_once.py")
        self.assertEqual(compact["sources"][0]["observed_by"], "spice")
        self.assertLessEqual(len(compact["workspace"]["summary"]), 283)
        self.assertNotIn("raw_file_contents", repr(compact))

    def test_finding_without_backing_source_is_limited(self) -> None:
        context = build_evidence_context(
            delegated_perception_context={
                "source": "delegated_perception",
                "perception_id": "delegated.no-source",
                "executor_id": "hermes",
                "findings": [
                    {
                        "finding_id": "finding.unsourced",
                        "text": "Hermes inferred something without a source.",
                        "source_refs": ["missing.source"],
                        "confidence": 0.8,
                    }
                ],
                "sources": [],
            },
        )

        self.assertEqual(context["findings"][0]["source_refs"], [])
        self.assertIn("missing_source_ref", context["findings"][0]["limitations"])
        self.assertIn("no_source_refs", context["findings"][0]["limitations"])
        self.assertIn("missing_source_ref", context["limitations"])


if __name__ == "__main__":
    unittest.main()
