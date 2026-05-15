from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from spice.runtime.resource_extractor import extract_resources
from spice.runtime.workspace_scope import (
    WORKSPACE_SCOPE_ALLOWED,
    WORKSPACE_SCOPE_BLOCKED,
    WORKSPACE_SCOPE_CURRENT_ROOT,
    WORKSPACE_SCOPE_EXTERNAL_ROOT,
    WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
    WORKSPACE_SCOPE_NEEDS_SELECTION,
    WORKSPACE_SCOPE_RESOLUTION_SCHEMA_VERSION,
    WORKSPACE_SCOPE_SUBPATH,
    resolve_workspace_scope,
)


class RuntimeWorkspaceScopeTests(unittest.TestCase):
    def test_current_workspace_path_is_allowed_as_scope_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "spice" / "runtime").mkdir(parents=True)
            target = root / "spice" / "runtime"

            resolution = resolve_workspace_scope(
                project_root=root,
                resource_extraction=extract_resources(f"读取 {target} 的实现。"),
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_ALLOWED)
            self.assertEqual(resolution.selected.scope_type, WORKSPACE_SCOPE_SUBPATH)  # type: ignore[union-attr]
            self.assertEqual(resolution.workspace_root, str(root.resolve()))
            self.assertEqual(resolution.scope_path, str(target.resolve()))

    def test_current_root_path_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

            resolution = resolve_workspace_scope(
                project_root=root,
                resource_extraction=extract_resources(f"读取 {root} 这个 repo。"),
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_ALLOWED)
            self.assertEqual(resolution.selected.scope_type, WORKSPACE_SCOPE_CURRENT_ROOT)  # type: ignore[union-attr]
            self.assertTrue(resolution.selected.is_repo_root)  # type: ignore[union-attr]

    def test_external_repo_root_needs_confirmation_in_interactive_mode(self) -> None:
        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir)
            (external / ".git").mkdir()
            (external / "README.md").write_text("demo\n", encoding="utf-8")

            resolution = resolve_workspace_scope(
                project_root=current_dir,
                resource_extraction=extract_resources(f"请读取 {external} 的当前实现。"),
                interactive=True,
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_NEEDS_CONFIRMATION)
            self.assertTrue(resolution.requires_confirmation)
            self.assertEqual(resolution.selected.scope_type, WORKSPACE_SCOPE_EXTERNAL_ROOT)  # type: ignore[union-attr]
            self.assertEqual(resolution.workspace_root, str(external.resolve()))

    def test_external_repo_root_blocked_in_non_interactive_mode(self) -> None:
        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir)
            (external / "pyproject.toml").write_text("[project]\nname='external'\n", encoding="utf-8")

            resolution = resolve_workspace_scope(
                project_root=current_dir,
                resource_extraction=extract_resources(f"请读取 {external}。"),
                interactive=False,
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_BLOCKED)
            self.assertTrue(resolution.blocked)
            self.assertIn("external repo root blocked", resolution.reason)

    def test_external_repo_root_can_be_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as current_dir, tempfile.TemporaryDirectory() as external_dir:
            external = Path(external_dir)
            (external / "package.json").write_text('{"name":"external"}\n', encoding="utf-8")

            resolution = resolve_workspace_scope(
                project_root=current_dir,
                resource_extraction=extract_resources(f"读取 {external}。"),
                allow_external_roots=True,
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_ALLOWED)
            self.assertEqual(resolution.selected.scope_type, WORKSPACE_SCOPE_EXTERNAL_ROOT)  # type: ignore[union-attr]
            self.assertEqual(resolution.workspace_root, str(external.resolve()))

    def test_missing_and_sensitive_paths_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing"

            missing_resolution = resolve_workspace_scope(
                project_root=tmp_dir,
                resource_extraction=extract_resources(f"读取 {missing}。"),
            )
            sensitive_resolution = resolve_workspace_scope(
                project_root=tmp_dir,
                resource_extraction=extract_resources("读取 /Users。"),
            )

            self.assertEqual(missing_resolution.status, WORKSPACE_SCOPE_BLOCKED)
            self.assertIn("path_not_found", missing_resolution.reason)
            self.assertEqual(sensitive_resolution.status, WORKSPACE_SCOPE_BLOCKED)
            self.assertIn("sensitive_root", sensitive_resolution.reason)

    def test_deny_dir_path_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".spice").mkdir()
            (root / ".spice" / "state.json").write_text("{}", encoding="utf-8")

            resolution = resolve_workspace_scope(
                project_root=root,
                resource_extraction=extract_resources(f"读取 {root / '.spice' / 'state.json'}。"),
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_BLOCKED)
            self.assertIn("deny_dir", resolution.reason)

    def test_multiple_allowed_scopes_need_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "pkg").mkdir()
            (root / "tests").mkdir()

            resolution = resolve_workspace_scope(
                project_root=root,
                resource_extraction=extract_resources(f"读取 {root / 'pkg'} 和 {root / 'tests'}。"),
            )

            self.assertEqual(resolution.status, WORKSPACE_SCOPE_NEEDS_SELECTION)
            self.assertEqual(len(resolution.candidates), 2)

    def test_payload_contains_stable_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "README.md").write_text("demo\n", encoding="utf-8")
            resolution = resolve_workspace_scope(
                project_root=root,
                resource_extraction=extract_resources(f"读取 {root / 'README.md'}。"),
            )

            payload = resolution.to_payload()

            self.assertEqual(payload["schema_version"], WORKSPACE_SCOPE_RESOLUTION_SCHEMA_VERSION)
            self.assertEqual(payload["status"], WORKSPACE_SCOPE_ALLOWED)
            self.assertIn("selected", payload)
            self.assertEqual(payload["selected"]["scope_path"], str((root / "README.md").resolve()))


if __name__ == "__main__":
    unittest.main()
