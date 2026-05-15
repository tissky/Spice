from __future__ import annotations

import unittest

from spice.runtime.evidence_requirement import (
    ANSWER_MODE_BRIEF,
    ANSWER_MODE_DETAILED,
    ANSWER_MODE_NORMAL,
    ANSWER_MODE_REPORT,
    EVIDENCE_DOMAIN_EXTERNAL,
    EVIDENCE_DOMAIN_MIXED,
    EVIDENCE_DOMAIN_NONE,
    EVIDENCE_DOMAIN_REPO,
    EVIDENCE_DOMAIN_URL,
    EVIDENCE_REQUIREMENT_SCHEMA_VERSION,
    detect_evidence_requirement,
    evidence_requirement_from_payload,
    strengthen_evidence_requirement,
)
from spice.runtime.resource_extractor import extract_resources


class RuntimeEvidenceRequirementTests(unittest.TestCase):
    def test_local_path_requires_repo_evidence(self) -> None:
        requirement = detect_evidence_requirement(
            "请读取本地 /Users/jiadongyu/Desktop/spice_update/Spice-main 这个 repo 的当前实现。"
        )

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_REPORT)
        self.assertIn("local path", requirement.reason)

    def test_repo_phrases_require_evidence_without_explicit_path(self) -> None:
        requirement = detect_evidence_requirement("基于当前实现看一下我们 repo 现在做到哪了。")

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_REPORT)
        self.assertIn("current implementation", requirement.reason)

    def test_common_repo_status_phrases_require_repo_evidence(self) -> None:
        cases = [
            "基于当前实现给我判断。",
            "基于当前代码给我判断。",
            "repo 现在做到哪了？",
        ]

        for text in cases:
            with self.subTest(text=text):
                requirement = detect_evidence_requirement(text)
                self.assertTrue(requirement.requires_evidence)
                self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_REPO)

    def test_url_requires_url_evidence(self) -> None:
        requirement = detect_evidence_requirement("基于这个链接判断下一步：https://example.com/spec")

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_URL)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_DETAILED)
        self.assertIn("URL", requirement.reason)

    def test_external_research_requires_external_evidence(self) -> None:
        requirement = detect_evidence_requirement("查一下最新 Hermes 和 OpenClaw 的设计，再给我报告。")

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_EXTERNAL)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_REPORT)
        self.assertIn("external", requirement.reason)

    def test_mixed_evidence_when_repo_and_url_are_both_present(self) -> None:
        requirement = detect_evidence_requirement(
            "读取 /Users/me/project，然后结合 https://example.com/spec 给完整分析。"
        )

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_MIXED)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_REPORT)

    def test_no_evidence_for_abstract_advice(self) -> None:
        requirement = detect_evidence_requirement("我们下一步应该优先 state-as-context 还是 executor handoff？")

        self.assertFalse(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_NONE)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_NORMAL)

    def test_brief_mode_can_still_require_evidence(self) -> None:
        requirement = detect_evidence_requirement("简短说一下当前代码里这个设计做到哪了。")

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(requirement.answer_mode, ANSWER_MODE_BRIEF)

    def test_uses_existing_resource_extraction(self) -> None:
        extraction = extract_resources("看 README.md 和 pyproject.toml，基于这些文件判断。")
        requirement = detect_evidence_requirement(
            "看 README.md 和 pyproject.toml，基于这些文件判断。",
            resource_extraction=extraction,
        )

        self.assertTrue(requirement.requires_evidence)
        self.assertEqual(requirement.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(requirement.resource_extraction, extraction)

    def test_strengthen_cannot_downgrade_hard_requirement(self) -> None:
        base = detect_evidence_requirement("基于当前实现给我一个报告。")

        strengthened = strengthen_evidence_requirement(
            base,
            requires_evidence=False,
            evidence_domain=EVIDENCE_DOMAIN_NONE,
            answer_mode=ANSWER_MODE_NORMAL,
            reason="planner suggested no evidence",
        )

        self.assertTrue(strengthened.requires_evidence)
        self.assertEqual(strengthened.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(strengthened.answer_mode, ANSWER_MODE_REPORT)
        self.assertIn("planner suggested no evidence", strengthened.reason)

    def test_strengthen_can_add_domain_and_answer_depth(self) -> None:
        base = detect_evidence_requirement("给我一个普通建议。")

        strengthened = strengthen_evidence_requirement(
            base,
            requires_evidence=True,
            evidence_domain=EVIDENCE_DOMAIN_EXTERNAL,
            answer_mode=ANSWER_MODE_DETAILED,
            reason="planner requested external research",
        )

        self.assertTrue(strengthened.requires_evidence)
        self.assertEqual(strengthened.evidence_domain, EVIDENCE_DOMAIN_EXTERNAL)
        self.assertEqual(strengthened.answer_mode, ANSWER_MODE_DETAILED)

    def test_payload_roundtrip_for_stable_fields(self) -> None:
        requirement = detect_evidence_requirement("基于当前实现给我一个报告。")
        payload = requirement.to_payload()

        self.assertEqual(payload["schema_version"], EVIDENCE_REQUIREMENT_SCHEMA_VERSION)
        restored = evidence_requirement_from_payload(payload)
        self.assertTrue(restored.requires_evidence)
        self.assertEqual(restored.evidence_domain, EVIDENCE_DOMAIN_REPO)
        self.assertEqual(restored.answer_mode, ANSWER_MODE_REPORT)


if __name__ == "__main__":
    unittest.main()
