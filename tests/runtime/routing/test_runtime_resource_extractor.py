from __future__ import annotations

import unittest

from spice.runtime.resource_extractor import (
    RESOURCE_EXTRACTION_SCHEMA_VERSION,
    extract_resources,
)


class RuntimeResourceExtractorTests(unittest.TestCase):
    def test_extracts_absolute_repo_path_without_semantic_routing(self) -> None:
        extraction = extract_resources(
            "请读取本地 /Users/jiadongyu/Desktop/spice_update/Spice-main 这个 repo 的当前实现。"
        )

        self.assertEqual(extraction.local_paths, ["/Users/jiadongyu/Desktop/spice_update/Spice-main"])
        self.assertIn("repo", extraction.repo_hints)
        self.assertIn("local_workspace", extraction.repo_hints)
        self.assertIn("current_implementation", extraction.repo_hints)
        self.assertTrue(extraction.has_repo_signal)

    def test_extracts_users_repo_path_and_https_url_for_correctness_gate(self) -> None:
        extraction = extract_resources(
            "基于 /Users/example/work/Spice-main 和 https://example.com/spec，判断当前实现。"
        )

        self.assertEqual(extraction.local_paths, ["/Users/example/work/Spice-main"])
        self.assertEqual(extraction.urls, ["https://example.com/spec"])
        self.assertTrue(extraction.has_repo_signal)
        self.assertTrue(extraction.has_external_signal)

    def test_extracts_urls_and_external_research_hints(self) -> None:
        extraction = extract_resources(
            "查一下 https://github.com/NousResearch/hermes-agent 和 https://openclawdoc.com/docs/agents/tools/，对比 Hermes 最新设计。"
        )

        self.assertEqual(
            extraction.urls,
            [
                "https://github.com/NousResearch/hermes-agent",
                "https://openclawdoc.com/docs/agents/tools/",
            ],
        )
        self.assertIn("lookup", extraction.external_research_hints)
        self.assertIn("latest", extraction.external_research_hints)
        self.assertIn("external_compare", extraction.external_research_hints)
        self.assertTrue(extraction.has_external_signal)

    def test_extracts_relative_paths_file_refs_and_symbols(self) -> None:
        extraction = extract_resources(
            "看一下 ./spice/runtime/tui/shell.py、spice/runtime/run_once.py、pyproject.toml，"
            "重点是 `_run_intent()` 和 `WorkspaceInspector`。"
        )

        self.assertIn("./spice/runtime/tui/shell.py", extraction.relative_paths)
        self.assertIn("spice/runtime/run_once.py", extraction.relative_paths)
        self.assertIn("pyproject.toml", extraction.file_refs)
        self.assertIn("_run_intent", extraction.symbols)
        self.assertIn("WorkspaceInspector", extraction.symbols)

    def test_payload_contains_stable_schema_and_flags(self) -> None:
        payload = extract_resources("基于当前代码和 README.md 给我判断。").to_payload()

        self.assertEqual(payload["schema_version"], RESOURCE_EXTRACTION_SCHEMA_VERSION)
        self.assertEqual(payload["file_refs"], ["README.md"])
        self.assertIn("current_implementation", payload["repo_hints"])
        self.assertTrue(payload["has_resources"])
        self.assertTrue(payload["has_repo_signal"])
        self.assertFalse(payload["has_external_signal"])

    def test_deduplicates_and_does_not_extract_url_paths_as_local_paths(self) -> None:
        extraction = extract_resources(
            "看 https://example.com/a/b.py 和 https://example.com/a/b.py，再看 /tmp/project 和 /tmp/project。"
        )

        self.assertEqual(extraction.urls, ["https://example.com/a/b.py"])
        self.assertEqual(extraction.local_paths, ["/tmp/project"])
        self.assertNotIn("/a/b.py", extraction.local_paths)


if __name__ == "__main__":
    unittest.main()
