from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from spice.perception import (
    WORKSPACE_CONTEXT_SCHEMA_VERSION,
    WORKSPACE_PERCEPTION_SCHEMA_VERSION,
    WorkspaceFact,
    WorkspaceFileRead,
    WorkspaceFileSkipped,
    WorkspaceInspector,
    WorkspaceInspectorLimits,
    WorkspacePerceptionLimits,
    WorkspacePerceptionQuery,
    WorkspaceSnippet,
    WorkspaceToolCall,
    build_workspace_perception_artifact,
    build_workspace_perception_artifact_from_loop,
    build_workspace_summary_cache,
    compact_workspace_summary_cache,
    load_or_refresh_workspace_summary_cache,
    workspace_context_from_perception,
)


class WorkspacePerceptionArtifactTests(unittest.TestCase):
    def test_builds_workspace_perception_artifact_and_context(self) -> None:
        artifact = build_workspace_perception_artifact(
            workspace_root="/repo",
            trigger="user_follow_up",
            created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
            queries=[
                WorkspacePerceptionQuery(
                    query="state as context implementation",
                    query_type="semantic",
                    path="spice/runtime",
                    file_glob="*.py",
                    limit=5,
                )
            ],
            files_read=[
                WorkspaceFileRead(
                    path="spice/runtime/run_once.py",
                    chars_read=1200,
                    line_start=1,
                    line_end=80,
                    truncated=False,
                    content_hash="abc",
                )
            ],
            files_skipped=[
                WorkspaceFileSkipped(path=".spice/state/state.json", reason="deny_dir")
            ],
            facts=[
                WorkspaceFact(
                    text="run_once compiles decision context before candidate generation.",
                    source_path="spice/runtime/run_once.py",
                    line_start=177,
                    line_end=190,
                    confidence=0.9,
                )
            ],
            summary="Runtime already compiles context before decision generation.",
            limits=WorkspacePerceptionLimits(max_files=10, max_chars_per_file=4000, total_char_budget=12000),
        )

        payload = artifact.to_payload()
        self.assertEqual(payload["schema_version"], WORKSPACE_PERCEPTION_SCHEMA_VERSION)
        self.assertTrue(payload["perception_id"].startswith("workspace."))
        self.assertEqual(payload["workspace_root"], "/repo")
        self.assertEqual(payload["trigger"], "user_follow_up")
        self.assertEqual(payload["queries"][0]["query"], "state as context implementation")
        self.assertEqual(payload["files_read"][0]["path"], "spice/runtime/run_once.py")
        self.assertEqual(payload["files_skipped"][0]["reason"], "deny_dir")
        self.assertEqual(payload["facts"][0]["confidence"], 0.9)
        self.assertEqual(payload["limits"]["max_files"], 10)

        context = workspace_context_from_perception(artifact)
        self.assertEqual(context["schema_version"], WORKSPACE_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(context["source"], "workspace_perception")
        self.assertEqual(context["perception_id"], payload["perception_id"])
        self.assertEqual(context["summary"], payload["summary"])
        self.assertEqual(context["facts"][0]["source_path"], "spice/runtime/run_once.py")

    def test_builds_workspace_perception_artifact_from_controlled_loop(self) -> None:
        loop_payload = {
            "schema_version": "spice.workspace_perception_loop.v1",
            "query": "current state-as-context implementation",
            "summary": "Inspected runtime state context files.",
            "done": True,
            "rounds_used": 1,
            "tool_calls": [
                {
                    "call_id": "tool.1",
                    "round_index": 1,
                    "tool": "search",
                    "args": {"pattern": "workspace_context", "file_glob": "*.py"},
                    "status": "executed",
                    "result": {
                        "ok": True,
                        "pattern": "workspace_context",
                        "matches": [
                            {
                                "path": "spice/memory/context.py",
                                "line_number": 49,
                                "line": "workspace_context: dict[str, Any]",
                                "content_hash": "hash-search",
                            }
                        ],
                    },
                },
                {
                    "call_id": "tool.2",
                    "round_index": 1,
                    "tool": "read_file",
                    "args": {"path": "spice/memory/context.py", "offset": 45, "limit": 10},
                    "status": "executed",
                    "result": {
                        "ok": True,
                        "path": "spice/memory/context.py",
                        "content": "class DecisionContext:\n    workspace_context: dict[str, Any]",
                        "line_start": 45,
                        "line_end": 50,
                        "chars_read": 66,
                        "truncated": False,
                        "content_hash": "hash-read",
                    },
                },
                {
                    "call_id": "tool.3",
                    "round_index": 1,
                    "tool": "write_file",
                    "args": {"path": "README.md", "content": "bad"},
                    "status": "blocked",
                    "reason": "tool_not_allowed",
                    "result": {},
                },
            ],
            "budget": {
                "limits": {
                    "max_rounds": 10,
                    "max_tool_calls": 80,
                    "max_tool_calls_per_round": 10,
                    "max_blocked_tool_calls": 20,
                    "max_blocked_tool_calls_per_round": 5,
                    "max_files_read": 60,
                    "max_chars_per_file": 12000,
                    "total_char_budget": 500000,
                    "planner_max_tokens": 4000,
                    "depth": "normal",
                },
                "tool_calls_recorded": 3,
                "tool_calls_executed": 2,
                "tool_calls_blocked": 1,
                "budget_state": {
                    "chars_used": 66,
                    "files_read_count": 1,
                    "budget_pressure": "medium",
                },
                "budget_pressure_events": [
                    {
                        "round_index": 1,
                        "budget_pressure": "medium",
                        "stage": "after_round",
                    }
                ],
            },
            "sufficiency_check": {
                "sufficient_evidence": False,
                "can_answer_user_question": True,
                "remaining_gaps": ["tests not inspected"],
                "reason": "Read the runtime hook but not tests.",
            },
        }

        artifact = build_workspace_perception_artifact_from_loop(
            workspace_root="/repo",
            trigger="user_follow_up",
            loop_result=loop_payload,
            created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        )
        payload = artifact.to_payload()

        self.assertEqual(payload["query"], "current state-as-context implementation")
        self.assertEqual(len(payload["tool_calls"]), 3)
        self.assertEqual(len(payload["blocked_tool_calls"]), 1)
        self.assertEqual(payload["blocked_tool_calls"][0]["reason"], "tool_not_allowed")
        self.assertEqual(payload["files_read"][0]["path"], "spice/memory/context.py")
        self.assertEqual(payload["snippets"][0]["source"], "search")
        self.assertEqual(payload["snippets"][1]["source"], "read_file")
        self.assertEqual(payload["facts"][0]["source_path"], "spice/memory/context.py")
        self.assertEqual(payload["budget"]["tool_calls_blocked"], 1)
        self.assertNotIn("content", payload["tool_calls"][1]["result"])
        self.assertIn("content_preview", payload["tool_calls"][1]["result"])
        self.assertEqual(payload["exploration_status"], "partial")
        self.assertEqual(payload["depth"], "normal")
        self.assertEqual(payload["budget_used"]["tool_calls_executed"], 2)
        self.assertTrue(any("Blocked write_file" in item for item in payload["limitations"]))
        self.assertEqual(payload["metadata"]["exploration_status"], "partial")
        self.assertEqual(payload["metadata"]["depth"], "normal")
        self.assertEqual(payload["metadata"]["budget_used"]["tool_calls_executed"], 2)
        self.assertEqual(payload["metadata"]["budget_used"]["tool_calls_blocked"], 1)
        self.assertEqual(payload["metadata"]["budget_used"]["max_files_read"], 60)
        self.assertEqual(payload["budget_used"]["chars_used"], 66)
        self.assertEqual(payload["budget_used"]["total_char_budget"], 500000)
        self.assertEqual(payload["budget_pressure_events"][0]["budget_pressure"], "medium")
        self.assertEqual(
            payload["metadata"]["loop"]["sufficiency_check"]["remaining_gaps"],
            ["tests not inspected"],
        )
        self.assertTrue(
            any("Blocked write_file" in item for item in payload["metadata"]["limitations"])
        )

        context = workspace_context_from_perception(artifact)
        self.assertEqual(context["perception_id"], payload["perception_id"])
        self.assertEqual(context["query"], "current state-as-context implementation")
        self.assertEqual(context["files_read"][0]["path"], "spice/memory/context.py")
        self.assertEqual(context["snippets"][1]["source"], "read_file")
        self.assertEqual(context["exploration_status"], "partial")
        self.assertEqual(context["depth"], "normal")
        self.assertEqual(context["budget_used"]["tool_calls_executed"], 2)
        self.assertEqual(context["budget_used"]["chars_used"], 66)
        self.assertEqual(context["budget_pressure_events"][0]["budget_pressure"], "medium")
        self.assertEqual(context["sufficiency_check"]["remaining_gaps"], ["tests not inspected"])
        self.assertTrue(any("Blocked write_file" in item for item in context["limitations"]))
        self.assertNotIn("tool_calls", context)
        self.assertNotIn("blocked_tool_calls", context)
        self.assertNotIn("budget", context)

    def test_workspace_perception_payload_round_trips_new_artifact_fields(self) -> None:
        artifact = build_workspace_perception_artifact(
            workspace_root="/repo",
            trigger="new_decision",
            query="repo state",
            tool_calls=[
                WorkspaceToolCall(
                    call_id="tool.1",
                    round_index=1,
                    tool="git_status",
                    status="executed",
                    result={"ok": True, "entries": []},
                )
            ],
            blocked_tool_calls=[
                WorkspaceToolCall(
                    call_id="tool.2",
                    round_index=1,
                    tool="write_file",
                    status="blocked",
                    reason="tool_not_allowed",
                )
            ],
            snippets=[
                WorkspaceSnippet(
                    path="README.md",
                    text="Spice is a decision brain.",
                    line_start=1,
                    line_end=1,
                    source="read_file",
                )
            ],
            budget={"tool_calls_recorded": 2},
            exploration_status="partial",
            depth="normal",
            budget_used={"tool_calls_recorded": 2, "budget_pressure": "low"},
            budget_pressure_events=[{"budget_pressure": "medium"}],
            limitations=["did not read tests"],
        )

        round_trip = type(artifact).from_payload(artifact.to_payload())

        self.assertEqual(round_trip.query, "repo state")
        self.assertEqual(round_trip.tool_calls[0].tool, "git_status")
        self.assertEqual(round_trip.blocked_tool_calls[0].reason, "tool_not_allowed")
        self.assertEqual(round_trip.snippets[0].path, "README.md")
        self.assertEqual(round_trip.budget["tool_calls_recorded"], 2)
        self.assertEqual(round_trip.exploration_status, "partial")
        self.assertEqual(round_trip.depth, "normal")
        self.assertEqual(round_trip.budget_used["tool_calls_recorded"], 2)
        self.assertEqual(round_trip.budget_pressure_events[0]["budget_pressure"], "medium")
        self.assertEqual(round_trip.limitations, ["did not read tests"])

    def test_workspace_context_carries_summary_cache_metadata(self) -> None:
        cache = {
            "status": "hit",
            "source": "workspace_summary_cache",
            "summary": "Workspace map cached 2 directories and 3 files.",
            "directory_summaries": [{"path": "spice/runtime", "purpose": "runtime orchestration"}],
            "file_summaries": [{"path": "spice/runtime/run_once.py", "purpose": "main runtime decision loop"}],
        }
        artifact = build_workspace_perception_artifact(
            workspace_root="/repo",
            trigger="new_decision",
            query="repo state",
            summary="Repo-aware perception.",
            metadata={"workspace_summary_cache": cache},
        )

        context = workspace_context_from_perception(artifact)

        self.assertEqual(context["workspace_summary_cache"]["status"], "hit")
        self.assertEqual(
            context["workspace_summary_cache"]["directory_summaries"][0]["path"],
            "spice/runtime",
        )


class WorkspaceSummaryCacheTests(unittest.TestCase):
    def test_build_workspace_summary_cache_maps_directories_and_file_purposes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice" / "runtime").mkdir(parents=True)
            (root / "spice" / "runtime" / "run_once.py").write_text("pass\n", encoding="utf-8")
            (root / "spice" / "perception").mkdir(parents=True)
            (root / "spice" / "perception" / "workspace_inspector.py").write_text("pass\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_runtime.py").write_text("def test_x(): pass\n", encoding="utf-8")
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\nversion = \"0.1.0\"\n",
                encoding="utf-8",
            )

            cache = build_workspace_summary_cache(WorkspaceInspector(root), max_depth=4)
            compact = compact_workspace_summary_cache(cache)

            self.assertIn("Workspace map cached", cache.summary)
            directories = {item.path: item for item in cache.directory_summaries}
            self.assertIn("spice/runtime", directories)
            self.assertIn("runtime orchestration", directories["spice/runtime"].purpose)
            files = {item.path: item for item in cache.file_summaries}
            self.assertEqual(files["spice/runtime/run_once.py"].purpose, "main runtime decision loop")
            self.assertEqual(files["tests/test_runtime.py"].metadata["role"], "test")
            self.assertEqual(cache.package_metadata["files"][0]["name"], "spice")
            self.assertIn("pytest_or_unittest", cache.test_structure["framework_hints"])
            self.assertEqual(compact["source"], "workspace_summary_cache")

    def test_load_or_refresh_workspace_summary_cache_hits_fresh_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".spice").mkdir()
            (root / "README.md").write_text("hello\n", encoding="utf-8")

            first = load_or_refresh_workspace_summary_cache(project_root=root)
            second = load_or_refresh_workspace_summary_cache(project_root=root)

            self.assertEqual(first.status, "created")
            self.assertEqual(second.status, "hit")
            self.assertTrue(second.path.exists())
            self.assertEqual(first.cache.cache_key, second.cache.cache_key)


class WorkspaceInspectorTests(unittest.TestCase):
    def test_file_index_lists_workspace_entries_and_skips_denied_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / ".spice").mkdir()
            (root / "README.md").write_text("hello\n", encoding="utf-8")

            inspector = WorkspaceInspector(root)
            result = inspector.file_index()

            self.assertTrue(result.ok)
            entries = {entry.path: entry for entry in result.entries}
            self.assertEqual(entries["README.md"].kind, "file")
            self.assertEqual(entries["src"].kind, "dir")
            self.assertEqual(entries[".spice"].kind, "skipped")
            self.assertEqual(entries[".spice"].skipped_reason, "deny_dir")
            self.assertEqual(inspector.files_skipped[0].path, ".spice")

    def test_read_file_reads_paginated_text_redacts_and_records_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "config.txt"
            target.write_text(
                "one\napi_key = sk-secret-value\nthree\n",
                encoding="utf-8",
            )

            inspector = WorkspaceInspector(root)
            result = inspector.read_file("config.txt", offset=2, limit=1)

            self.assertTrue(result.ok)
            self.assertEqual(result.path, "config.txt")
            self.assertEqual(result.line_start, 2)
            self.assertEqual(result.line_end, 2)
            self.assertIn("api_key=<redacted>", result.content)
            self.assertNotIn("sk-secret-value", result.content)
            self.assertEqual(inspector.files_read[0].path, "config.txt")
            self.assertEqual(inspector.files_read[0].content_hash, result.content_hash)

            duplicate = inspector.read_file("config.txt", offset=2, limit=1)
            self.assertTrue(duplicate.ok)
            self.assertTrue(duplicate.dedup)
            self.assertEqual(duplicate.reason, "already_read")
            self.assertEqual(len(inspector.files_read), 1)

    def test_read_file_enforces_budget_and_binary_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "big.txt").write_text("abcdef\n", encoding="utf-8")
            (root / "image.png").write_bytes(b"\x89PNG\x00binary")

            inspector = WorkspaceInspector(
                root,
                limits=WorkspaceInspectorLimits(
                    max_files_read=2,
                    max_chars_per_file=3,
                    total_char_budget=3,
                ),
            )
            first = inspector.read_file("big.txt")
            self.assertTrue(first.ok)
            self.assertEqual(first.content, "abc")
            self.assertTrue(first.truncated)

            budget_block = inspector.read_file("big.txt", offset=2)
            self.assertFalse(budget_block.ok)
            self.assertEqual(budget_block.reason, "total_char_budget_exceeded")

            binary = WorkspaceInspector(root).read_file("image.png")
            self.assertFalse(binary.ok)
            self.assertEqual(binary.reason, "binary_file")

    def test_read_file_blocks_denied_dirs_and_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".spice").mkdir()
            (root / ".spice" / "state.json").write_text("{}", encoding="utf-8")

            inspector = WorkspaceInspector(root)
            denied = inspector.read_file(".spice/state.json")
            self.assertFalse(denied.ok)
            self.assertEqual(denied.reason, "deny_dir")

            escaped = inspector.read_file("../outside.txt")
            self.assertFalse(escaped.ok)
            self.assertIn("escapes workspace root", escaped.reason)

    def test_read_file_blocks_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / f"{root.name}-outside.txt"
            outside.write_text("outside", encoding="utf-8")
            try:
                (root / "outside-link").symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            try:
                result = WorkspaceInspector(root).read_file("outside-link")
                self.assertFalse(result.ok)
                self.assertEqual(result.reason, "symlink_escape")
            finally:
                outside.unlink(missing_ok=True)

    def test_search_finds_text_with_glob_and_records_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
            (root / "b.md").write_text("Alpha in docs\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("Alpha hidden\n", encoding="utf-8")

            inspector = WorkspaceInspector(root)
            result = inspector.search("Alpha", file_glob="*.py", limit=10)

            self.assertTrue(result.ok)
            self.assertEqual(len(result.matches), 1)
            self.assertEqual(result.matches[0].path, "a.py")
            self.assertEqual(result.matches[0].line_number, 1)
            self.assertTrue(any(item.path == ".git" for item in inspector.files_skipped))

    def test_search_prefers_rg_backend_when_available(self) -> None:
        rg_output = json.dumps(
            {
                "type": "match",
                "data": {
                    "path": {"text": "src/app.py"},
                    "lines": {"text": "class Alpha:\n"},
                    "line_number": 7,
                },
            }
        )
        completed = subprocess.CompletedProcess(
            args=["rg"],
            returncode=0,
            stdout=f"{rg_output}\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("class Alpha:\n    pass\n", encoding="utf-8")
            with patch("spice.perception.workspace_inspector.shutil.which", return_value="/usr/bin/rg"):
                with patch("spice.perception.workspace_inspector.subprocess.run", return_value=completed) as run:
                    result = WorkspaceInspector(root).search("Alpha", file_glob="*.py", limit=5)

            self.assertTrue(result.ok)
            self.assertEqual(result.backend, "rg")
            self.assertEqual(result.matches[0].path, "src/app.py")
            self.assertEqual(result.matches[0].line_number, 7)
            command = run.call_args.args[0]
            self.assertEqual(command[0], "rg")
            self.assertIn("--json", command)
            self.assertIn("--hidden", command)
            self.assertIn("*.py", command)
            self.assertIn("!**/.git/**", command)
            self.assertEqual(command[-3:], ["--", "Alpha", str(root.resolve())])

    def test_search_falls_back_to_python_when_rg_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("class Alpha:\n", encoding="utf-8")

            with patch("spice.perception.workspace_inspector.shutil.which", return_value=None):
                result = WorkspaceInspector(root).search("Alpha", file_glob="*.py", limit=5)

            self.assertTrue(result.ok)
            self.assertEqual(result.backend, "python")
            self.assertEqual(result.reason, "rg_unavailable")
            self.assertEqual(result.matches[0].path, "src/app.py")

    def test_search_falls_back_to_python_when_rg_fails(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["rg"],
            returncode=2,
            stdout="",
            stderr="rg failed",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src.py").write_text("Alpha\n", encoding="utf-8")

            with patch("spice.perception.workspace_inspector.shutil.which", return_value="/usr/bin/rg"):
                with patch("spice.perception.workspace_inspector.subprocess.run", return_value=completed):
                    result = WorkspaceInspector(root).search("Alpha", file_glob="*.py", limit=5)

            self.assertTrue(result.ok)
            self.assertEqual(result.backend, "python")
            self.assertEqual(result.reason, "rg_failed")
            self.assertEqual(result.matches[0].path, "src.py")

    def test_search_enforces_file_scan_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "b.txt").write_text("beta\n", encoding="utf-8")

            inspector = WorkspaceInspector(
                root,
                limits=WorkspaceInspectorLimits(max_search_files_scanned=1),
            )
            with patch("spice.perception.workspace_inspector.shutil.which", return_value=None):
                result = inspector.search("missing", file_glob="*.txt", limit=10)

            self.assertTrue(result.ok)
            self.assertTrue(result.truncated)
            self.assertEqual(result.reason, "max_search_files_scanned_exceeded")

    def test_git_status_uses_read_only_porcelain_command(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="## main...origin/main\n M README.md\n?? new.py\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("spice.perception.workspace_inspector.subprocess.run", return_value=completed) as run:
                result = WorkspaceInspector(root).git_status()

            self.assertTrue(result.ok)
            self.assertEqual(result.branch, "main...origin/main")
            self.assertEqual(result.entries[0].path, "README.md")
            self.assertEqual(result.entries[0].status, "M")
            self.assertEqual(result.entries[1].path, "new.py")
            self.assertEqual(result.entries[1].status, "??")
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["git", "-C", str(root.resolve())])
            self.assertIn("status", command)
            self.assertIn("--porcelain=v1", command)

    def test_git_diff_uses_read_only_git_diff_and_truncates_content(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="diff --git a/README.md b/README.md\n+token=secret-value\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello\n", encoding="utf-8")
            with patch("spice.perception.workspace_inspector.subprocess.run", return_value=completed) as run:
                result = WorkspaceInspector(
                    root,
                    limits=WorkspaceInspectorLimits(max_git_diff_chars=20),
                ).git_diff(path="README.md", mode="patch")

            self.assertTrue(result.ok)
            self.assertEqual(result.path, "README.md")
            self.assertEqual(result.mode, "patch")
            self.assertTrue(result.truncated)
            self.assertNotIn("secret-value", result.content)
            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["git", "-C", str(root.resolve()), "diff"])
            self.assertEqual(command[-2:], ["--", "README.md"])

    def test_git_log_reads_recent_commits_without_shell(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="abc123\x1fHEAD -> main\x1fInitial commit\ndef456\x1f\x1fSecond commit\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("spice.perception.workspace_inspector.subprocess.run", return_value=completed) as run:
                result = WorkspaceInspector(root).git_log(limit=2)

            self.assertTrue(result.ok)
            self.assertEqual(result.entries[0].commit, "abc123")
            self.assertEqual(result.entries[0].refs, "HEAD -> main")
            self.assertEqual(result.entries[1].subject, "Second commit")
            command = run.call_args.args[0]
            self.assertIn("log", command)
            self.assertIn("--max-count=2", command)

    def test_repo_map_summarizes_workspace_and_skips_denied_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spice" / "runtime").mkdir(parents=True)
            (root / "spice" / "runtime" / "run_once.py").write_text("pass\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("hidden\n", encoding="utf-8")

            result = WorkspaceInspector(root).repo_map(max_depth=3)

            self.assertTrue(result.ok)
            entries = {entry.path: entry for entry in result.entries}
            self.assertEqual(entries["spice"].kind, "dir")
            self.assertEqual(entries["spice/runtime/run_once.py"].kind, "file")
            self.assertEqual(entries[".git"].kind, "skipped")
            self.assertEqual(entries[".git"].skipped_reason, "deny_dir")

    def test_read_package_metadata_extracts_common_manifest_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text(
                "[project]\nname = \"spice\"\nversion = \"0.1.0\"\ndependencies = [\"rich\"]\n",
                encoding="utf-8",
            )
            (root / "package.json").write_text(
                '{"name":"web","version":"1.2.3","scripts":{"test":"vitest"},"dependencies":{"vite":"latest"}}',
                encoding="utf-8",
            )

            inspector = WorkspaceInspector(root)
            result = inspector.read_package_metadata()

            self.assertTrue(result.ok)
            by_path = {item.path: item for item in result.files}
            self.assertEqual(by_path["pyproject.toml"].name, "spice")
            self.assertIn("rich", by_path["pyproject.toml"].dependencies)
            self.assertIn("test", by_path["package.json"].scripts)
            self.assertIn("vite", by_path["package.json"].dependencies)
            self.assertEqual(len(result.files_read), 2)

    def test_read_test_structure_detects_test_files_without_running_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_runtime.py").write_text("def test_x(): pass\n", encoding="utf-8")
            (root / "web").mkdir()
            (root / "web" / "app.test.ts").write_text("test('x', () => {})\n", encoding="utf-8")

            result = WorkspaceInspector(root).read_test_structure()

            self.assertTrue(result.ok)
            paths = {item.path for item in result.test_files}
            self.assertIn("tests/test_runtime.py", paths)
            self.assertIn("web/app.test.ts", paths)
            self.assertIn("pytest_or_unittest", result.framework_hints)
            self.assertIn("javascript_test_runner", result.framework_hints)

    def test_python_symbol_index_finds_classes_functions_and_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "service.py").write_text(
                "import os\n"
                "from .helpers import build\n\n"
                "class Service:\n"
                "    def run(self):\n"
                "        return build(os.getcwd())\n\n"
                "async def load():\n"
                "    return Service()\n",
                encoding="utf-8",
            )

            result = WorkspaceInspector(root).python_symbol_index("pkg")

            self.assertTrue(result.ok)
            by_qualified = {item.qualified_name: item for item in result.symbols}
            self.assertEqual(by_qualified["Service"].kind, "class")
            self.assertEqual(by_qualified["Service.run"].kind, "method")
            self.assertEqual(by_qualified["load"].kind, "async_function")
            imports = {(item.kind, item.module, tuple(item.names), item.level) for item in result.imports}
            self.assertIn(("import", "os", (), 0), imports)
            self.assertIn(("from_import", "helpers", ("build",), 1), imports)
            modules = {item.path: item for item in result.modules}
            self.assertEqual(modules["pkg/service.py"].module, "pkg.service")
            self.assertEqual(modules["pkg/service.py"].symbol_count, 3)

    def test_read_python_symbol_reads_exact_class_or_function_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "service.py").write_text(
                "class Service:\n"
                "    def run(self):\n"
                "        return 'ok'\n\n"
                "def helper():\n"
                "    return Service()\n",
                encoding="utf-8",
            )

            inspector = WorkspaceInspector(root)
            result = inspector.read_python_symbol(
                path="pkg/service.py",
                qualified_name="Service.run",
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.path, "pkg/service.py")
            self.assertEqual(result.qualified_name, "Service.run")
            self.assertEqual(result.kind, "method")
            self.assertEqual(result.line_start, 2)
            self.assertEqual(result.line_end, 3)
            self.assertIn("def run", result.content)
            self.assertEqual(inspector.files_read[0].path, "pkg/service.py")

    def test_python_symbol_index_respects_denied_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".spice").mkdir()
            (root / ".spice" / "hidden.py").write_text("class Hidden: pass\n", encoding="utf-8")

            result = WorkspaceInspector(root).python_symbol_index(".spice")

            self.assertFalse(result.ok)
            self.assertEqual(result.reason, "deny_dir")

    def test_summarize_workspace_reports_read_and_budget_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")

            inspector = WorkspaceInspector(root)
            inspector.read_file("README.md")
            summary = inspector.summarize_workspace()

            self.assertEqual(summary["workspace_root"], str(root.resolve()))
            self.assertEqual(summary["files_read"][0]["path"], "README.md")
            self.assertEqual(summary["budget"]["files_read_count"], 1)
            self.assertGreaterEqual(summary["budget"]["chars_remaining"], 0)


if __name__ == "__main__":
    unittest.main()
