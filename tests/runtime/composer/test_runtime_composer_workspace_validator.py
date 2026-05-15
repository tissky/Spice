from __future__ import annotations

import unittest

from spice.runtime.composer_workspace_validator import (
    build_citation_evidence_index,
    validate_workspace_claims,
)


class RuntimeComposerWorkspaceValidatorTests(unittest.TestCase):
    def test_delegated_report_phrasing_with_known_source_passes(self) -> None:
        facts = _delegated_facts()

        validate_workspace_claims(
            "Hermes 的只读调查报告里提到 source.1：read-only investigation should stay separate from execution.",
            facts,
            composer_kind="response composer",
        )

    def test_delegated_source_without_executor_attribution_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "without executor attribution"):
            validate_workspace_claims(
                "source.1 says this boundary should stay separate.",
                _delegated_facts(),
                composer_kind="response composer",
            )

    def test_delegated_report_without_delegated_context_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "without delegated_perception_context"):
            validate_workspace_claims(
                "Hermes reported source.1 says read-only investigation should stay separate from execution.",
                {"recent_context": {}},
                composer_kind="response composer",
            )

    def test_delegated_source_is_indexed_with_url_and_verification_status(self) -> None:
        facts = _delegated_facts(verification_status="cross_checked")

        index = build_citation_evidence_index(facts)
        source = next(item for item in index.delegated_items if item.source_id == "source.1")

        self.assertIn("https://example.com/agent-workflow", index.urls)
        self.assertEqual(source.verification_status, "cross_checked")

    def test_direct_spice_inspection_claim_for_delegated_context_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "directly inspected delegated sources"):
            validate_workspace_claims(
                "我检查了这些网页，Spice 已经确认 source.1 里的说法是事实。",
                _delegated_facts(),
                composer_kind="response composer",
            )

    def test_direct_url_inspection_claim_for_delegated_source_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "direct URL inspection for delegated source"):
            validate_workspace_claims(
                "I checked the docs at https://example.com/agent-workflow and confirmed the finding.",
                _delegated_facts(),
                composer_kind="response composer",
            )

    def test_nonexistent_delegated_source_ref_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonexistent delegated source"):
            validate_workspace_claims(
                "Hermes reported source.99 says this should be delegated.",
                _delegated_facts(),
                composer_kind="response composer",
            )

    def test_delegated_finding_as_final_decision_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "final decision"):
            validate_workspace_claims(
                "Hermes reported source.1, so this is the final decision.",
                _delegated_facts(),
                composer_kind="response composer",
            )

    def test_delegated_perception_as_execution_completed_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "completed execution"):
            validate_workspace_claims(
                "Hermes 的只读调查说明这个已经执行完成。",
                _delegated_facts(),
                composer_kind="execution response composer",
            )

    def test_cross_checked_claim_requires_cross_checked_status(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross_checked verification_status"):
            validate_workspace_claims(
                "Hermes reported source.1 and it was cross-checked.",
                _delegated_facts(verification_status="reported_by_executor"),
            composer_kind="response composer",
        )

    def test_reported_by_executor_source_cannot_be_called_verified_by_spice(self) -> None:
        with self.assertRaisesRegex(ValueError, "cross_checked verification_status"):
            validate_workspace_claims(
                "Hermes reported source.1 as verified_by_spice.",
                _delegated_facts(verification_status="reported_by_executor"),
                composer_kind="response composer",
            )

    def test_verified_by_spice_status_passes_when_source_is_cross_checked(self) -> None:
        validate_workspace_claims(
            "Hermes reported source.1 with verification_status=cross_checked.",
            _delegated_facts(verification_status="cross_checked"),
            composer_kind="response composer",
        )

    def test_cross_checked_claim_passes_with_cross_checked_status(self) -> None:
        validate_workspace_claims(
            "Hermes reported source.1 and source.1 was cross-checked.",
            _delegated_facts(verification_status="cross_checked"),
            composer_kind="response composer",
        )

    def test_negative_cross_checked_limitation_passes(self) -> None:
        validate_workspace_claims(
            "Hermes reported source.1, but it was not cross-checked by Spice.",
            _delegated_facts(),
            composer_kind="response composer",
        )

    def test_repo_read_claim_without_workspace_context_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "workspace inspection claim"):
            validate_workspace_claims(
                "我读了 repo，代码里当前已经实现了 workspace perception。",
                {"recent_context": {}},
                composer_kind="response composer",
            )

    def test_file_read_claim_without_workspace_source_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "invented workspace file|workspace inspection claim"):
            validate_workspace_claims(
                "I read `spice/runtime/run_once.py` and it already implements the full evidence gate.",
                {"recent_context": {}},
                composer_kind="response composer",
            )

    def test_implementation_claim_without_any_evidence_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "workspace implementation claim"):
            validate_workspace_claims(
                "已经实现了 delegated perception。",
                {"recent_context": {}},
                composer_kind="response composer",
            )

    def test_url_inspection_claim_without_url_context_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "URL inspection claim"):
            validate_workspace_claims(
                "I read the linked PR and it says executor handoff should wait.",
                {"recent_context": {}},
                composer_kind="response composer",
            )

    def test_invented_explicit_url_fails(self) -> None:
        facts = {
            "recent_context": {
                "url_context": {
                    "source": "url_perception",
                    "perception_id": "url.known",
                    "urls": ["https://example.com/spec"],
                    "facts": [
                        {
                            "text": "The spec says URL perception is read-only.",
                            "source_url": "https://example.com/spec",
                        }
                    ],
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "invented external URL"):
            validate_workspace_claims(
                "According to https://example.com/other, we should execute now.",
                facts,
                composer_kind="response composer",
            )

    def test_known_url_context_allows_linked_doc_claim(self) -> None:
        facts = {
            "recent_context": {
                "url_context": {
                    "source": "url_perception",
                    "perception_id": "url.known",
                    "documents": [
                        {
                            "url": "https://example.com/spec",
                            "title": "Spec",
                        }
                    ],
                    "facts": [
                        {
                            "text": "The spec says URL perception is read-only.",
                            "source_url": "https://example.com/spec",
                        }
                    ],
                }
            }
        }

        validate_workspace_claims(
            "According to https://example.com/spec, this should stay read-only perception.",
            facts,
            composer_kind="response composer",
        )

    def test_citation_index_extracts_workspace_and_url_sources(self) -> None:
        facts = {
            "recent_context": {
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.known",
                    "files_read": [{"path": "spice/runtime/run_once.py"}],
                    "facts": [
                        {
                            "text": "function run_once already receives workspace context.",
                            "source_path": "spice/runtime/run_once.py",
                        }
                    ],
                },
                "url_context": {
                    "source": "url_perception",
                    "perception_id": "url.known",
                    "documents": [{"url": "https://example.com/spec"}],
                    "facts": [{"text": "The spec is read-only.", "source_url": "https://example.com/spec"}],
                },
            }
        }

        index = build_citation_evidence_index(facts)

        self.assertIn("spice/runtime/run_once.py", index.workspace_paths)
        self.assertIn("run_once", index.workspace_symbols)
        self.assertIn("https://example.com/spec", index.urls)

    def test_supported_workspace_implementation_claim_passes(self) -> None:
        facts = {
            "recent_context": {
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.known",
                    "files_read": [{"path": "spice/runtime/run_once.py"}],
                    "facts": [
                        {
                            "text": "function run_once already receives workspace context and passes it into compiled decision context.",
                            "source_path": "spice/runtime/run_once.py",
                        }
                    ],
                }
            }
        }

        validate_workspace_claims(
            "I checked `spice/runtime/run_once.py`; function `run_once` already receives workspace context.",
            facts,
            composer_kind="response composer",
        )

    def test_partial_workspace_context_requires_limitation_when_used(self) -> None:
        validate_workspace_claims(
            "I checked `spice/runtime/run_once.py`; this was a partial workspace perception, so tests remain a limitation.",
            _partial_workspace_facts(),
            composer_kind="response composer",
        )

    def test_partial_workspace_context_rejects_complete_exploration_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete workspace exploration"):
            validate_workspace_claims(
                "I checked `spice/runtime/run_once.py`; I fully inspected the repo and there are no remaining gaps.",
                _partial_workspace_facts(),
                composer_kind="response composer",
            )

    def test_partial_workspace_context_rejects_high_confidence_complete_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete workspace exploration"):
            validate_workspace_claims(
                "我看了 `spice/runtime/run_once.py`，已经完整确认当前实现没有缺口。",
                _partial_workspace_facts(),
                composer_kind="response composer",
            )

    def test_budget_exhausted_workspace_context_rejects_unqualified_repo_claim(self) -> None:
        facts = _partial_workspace_facts()
        facts["recent_context"]["workspace_context"]["exploration_status"] = "budget_exhausted"
        with self.assertRaisesRegex(ValueError, "omitted workspace exploration limitation"):
            validate_workspace_claims(
                "I checked `spice/runtime/run_once.py`; the code shows run_once accepts workspace context.",
                facts,
                composer_kind="response composer",
            )

    def test_budget_exhausted_workspace_context_allows_qualified_repo_claim(self) -> None:
        facts = _partial_workspace_facts()
        facts["recent_context"]["workspace_context"]["exploration_status"] = "budget_exhausted"

        validate_workspace_claims(
            "I checked `spice/runtime/run_once.py`; the code shows run_once accepts workspace context, but the workspace perception hit a budget exhausted limitation and tests remain a gap.",
            facts,
            composer_kind="response composer",
        )

    def test_budget_exhausted_workspace_context_rejects_evidence_claim_without_limitation(self) -> None:
        facts = _partial_workspace_facts()
        facts["recent_context"]["workspace_context"]["exploration_status"] = "budget_exhausted"
        with self.assertRaisesRegex(ValueError, "omitted workspace exploration limitation"):
            validate_workspace_claims(
                "The evidence shows run_once accepts workspace context.",
                facts,
                composer_kind="response composer",
            )

    def test_files_read_alone_cannot_support_implementation_judgment(self) -> None:
        facts = {
            "recent_context": {
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.known",
                    "files_read": [{"path": "spice/runtime/run_once.py"}],
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "implementation claim"):
            validate_workspace_claims(
                "I checked `spice/runtime/run_once.py`; it already implements workspace context injection.",
                facts,
                composer_kind="response composer",
            )

    def test_workspace_summary_alone_cannot_support_implementation_judgment(self) -> None:
        facts = {
            "recent_context": {
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.summary",
                    "summary": "Python symbol index is already implemented.",
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "implementation claim"):
            validate_workspace_claims(
                "The repo shows Python symbol index is already implemented.",
                facts,
                composer_kind="response composer",
            )

    def test_unrelated_workspace_implementation_claim_fails(self) -> None:
        facts = {
            "recent_context": {
                "workspace_context": {
                    "source": "workspace_perception",
                    "perception_id": "workspace.known",
                    "files_read": [{"path": "spice/runtime/run_once.py"}],
                    "facts": [
                        {
                            "text": "function run_once already receives workspace context.",
                            "source_path": "spice/runtime/run_once.py",
                        }
                    ],
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "implementation claim"):
            validate_workspace_claims(
                "The repo shows Python symbol index is already implemented.",
                facts,
                composer_kind="response composer",
            )

    def test_url_claim_with_unrelated_url_evidence_fails(self) -> None:
        facts = {
            "recent_context": {
                "url_context": {
                    "source": "url_perception",
                    "perception_id": "url.known",
                    "documents": [{"url": "https://example.com/spec"}],
                    "facts": [
                        {
                            "text": "The spec says URL perception is read-only.",
                            "source_url": "https://example.com/spec",
                        }
                    ],
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "URL claim"):
            validate_workspace_claims(
                "I read the linked docs, and they say executor handoff is production-ready.",
                facts,
                composer_kind="response composer",
            )


def _delegated_facts(*, verification_status: str = "reported_by_executor") -> dict[str, object]:
    return {
        "recent_context": {
            "delegated_perception_context": {
                "source": "delegated_perception",
                "perception_id": "delegated.known",
                "executor_id": "hermes",
                "summary": "Hermes reported that read-only investigation should stay separate from execution.",
                "findings": [
                    {
                        "finding_id": "finding.1",
                        "text": "Read-only investigation should be separated from execution approval.",
                        "confidence": 0.82,
                        "source_refs": ["source.1"],
                    }
                ],
                "sources": [
                    {
                        "source_id": "source.1",
                        "source_type": "url",
                        "title": "Agent workflow note",
                        "uri": "https://example.com/agent-workflow",
                        "excerpt": "Research should be separated from execution.",
                        "observed_by": "hermes",
                        "verification_status": verification_status,
                    }
                ],
            }
        }
    }


def _partial_workspace_facts() -> dict[str, object]:
    return {
        "recent_context": {
            "workspace_context": {
                "source": "workspace_perception",
                "perception_id": "workspace.partial",
                "exploration_status": "partial",
                "limitations": ["did not read tests"],
                "files_read": [{"path": "spice/runtime/run_once.py"}],
                "facts": [
                    {
                        "text": "function run_once accepts workspace context.",
                        "source_path": "spice/runtime/run_once.py",
                    }
                ],
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
