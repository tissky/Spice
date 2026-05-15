from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from typing import Any, Iterable

from spice.decision.general.types import payload_value
from spice.perception.workspace import WorkspaceFileRead, WorkspaceFileSkipped


DEFAULT_DENY_DIRS = frozenset(
    {
        ".git",
        ".spice",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
    }
)

DEFAULT_BINARY_EXTENSIONS = frozenset(
    {
        ".7z",
        ".a",
        ".avi",
        ".bin",
        ".bmp",
        ".class",
        ".dll",
        ".dmg",
        ".doc",
        ".docx",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".pdf",
        ".png",
        ".pyc",
        ".pyo",
        ".rar",
        ".so",
        ".tar",
        ".tiff",
        ".webp",
        ".xls",
        ".xlsx",
        ".zip",
    }
)


@dataclass(frozen=True, slots=True)
class WorkspaceInspectorLimits:
    max_files_read: int = 20
    max_chars_per_file: int = 12_000
    total_char_budget: int = 50_000
    max_index_entries: int = 200
    max_search_results: int = 50
    max_search_files_scanned: int = 500
    max_file_size_bytes: int = 1_000_000
    max_line_limit: int = 2_000
    max_git_diff_chars: int = 12_000
    max_git_log_entries: int = 20
    max_repo_map_entries: int = 400
    max_package_metadata_files: int = 20
    max_test_structure_entries: int = 300
    max_python_symbol_entries: int = 500
    max_python_import_entries: int = 500

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFileIndexEntry:
    path: str
    kind: str
    size_bytes: int | None = None
    skipped_reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFileIndexResult:
    ok: bool
    path: str
    entries: list[WorkspaceFileIndexEntry] = field(default_factory=list)
    truncated: bool = False
    reason: str = ""
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceSearchMatch:
    path: str
    line_number: int
    line: str
    content_hash: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceSearchResult:
    ok: bool
    pattern: str
    matches: list[WorkspaceSearchMatch] = field(default_factory=list)
    truncated: bool = False
    reason: str = ""
    backend: str = "python"
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceReadResult:
    ok: bool
    path: str
    content: str = ""
    line_start: int | None = None
    line_end: int | None = None
    chars_read: int = 0
    truncated: bool = False
    content_hash: str = ""
    reason: str = ""
    dedup: bool = False
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceGitStatusEntry:
    path: str
    status: str

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceGitStatusResult:
    ok: bool
    branch: str = ""
    entries: list[WorkspaceGitStatusEntry] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceGitDiffResult:
    ok: bool
    path: str = ""
    mode: str = "stat"
    content: str = ""
    chars_read: int = 0
    content_hash: str = ""
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceGitLogEntry:
    commit: str
    subject: str
    refs: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceGitLogResult:
    ok: bool
    entries: list[WorkspaceGitLogEntry] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceRepoMapEntry:
    path: str
    kind: str
    depth: int
    size_bytes: int | None = None
    skipped_reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceRepoMapResult:
    ok: bool
    path: str
    entries: list[WorkspaceRepoMapEntry] = field(default_factory=list)
    truncated: bool = False
    reason: str = ""
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePackageMetadataFile:
    path: str
    kind: str
    name: str = ""
    version: str = ""
    scripts: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    content_preview: str = ""
    content_hash: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePackageMetadataResult:
    ok: bool
    files: list[WorkspacePackageMetadataFile] = field(default_factory=list)
    files_read: list[WorkspaceFileRead] = field(default_factory=list)
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceTestFileEntry:
    path: str
    kind: str
    framework_hint: str = ""

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceTestStructureResult:
    ok: bool
    test_files: list[WorkspaceTestFileEntry] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePythonSymbolEntry:
    path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    parent: str = ""
    decorators: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePythonImportEntry:
    path: str
    module: str
    names: list[str] = field(default_factory=list)
    level: int = 0
    line_number: int = 0
    kind: str = "import"

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePythonModuleEntry:
    path: str
    module: str
    symbol_count: int = 0
    import_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePythonSymbolIndexResult:
    ok: bool
    path: str
    symbols: list[WorkspacePythonSymbolEntry] = field(default_factory=list)
    imports: list[WorkspacePythonImportEntry] = field(default_factory=list)
    modules: list[WorkspacePythonModuleEntry] = field(default_factory=list)
    files_inspected: int = 0
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)
    reason: str = ""
    truncated: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePythonSymbolReadResult:
    ok: bool
    path: str
    qualified_name: str = ""
    name: str = ""
    kind: str = ""
    content: str = ""
    line_start: int | None = None
    line_end: int | None = None
    chars_read: int = 0
    truncated: bool = False
    content_hash: str = ""
    reason: str = ""
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


class WorkspaceInspector:
    """Read-only workspace inspection primitives.

    The inspector deliberately exposes no write, patch, shell, install, or test-run
    primitive. Path and budget checks live here so future callers cannot bypass
    guardrails by skipping the controlled perception loop.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        limits: WorkspaceInspectorLimits | None = None,
        deny_dirs: Iterable[str] | None = None,
        binary_extensions: Iterable[str] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.limits = limits or WorkspaceInspectorLimits()
        self.deny_dirs = frozenset(deny_dirs or DEFAULT_DENY_DIRS)
        self.binary_extensions = frozenset(
            ext.lower() for ext in (binary_extensions or DEFAULT_BINARY_EXTENSIONS)
        )
        self.files_read: list[WorkspaceFileRead] = []
        self.files_skipped: list[WorkspaceFileSkipped] = []
        self._read_cache: dict[tuple[str, int, int], float | None] = {}
        self._chars_used = 0
        self._files_read_count = 0

    def file_index(self, path: str = ".", *, limit: int | None = None) -> WorkspaceFileIndexResult:
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspaceFileIndexResult(ok=False, path=str(path or "."), reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspaceFileIndexResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )
        if not resolved.is_dir():
            return WorkspaceFileIndexResult(
                ok=True,
                path=self._display_path(resolved),
                entries=[
                    WorkspaceFileIndexEntry(
                        path=self._display_path(resolved),
                        kind="file",
                        size_bytes=_safe_size(resolved),
                    )
                ],
            )

        max_entries = _positive_limit(limit, self.limits.max_index_entries)
        entries: list[WorkspaceFileIndexEntry] = []
        local_skipped: list[WorkspaceFileSkipped] = []
        for child in sorted(resolved.iterdir(), key=lambda item: item.name.lower()):
            child_skip = self._skip_for_path(child, require_file=False)
            if child_skip:
                local_skipped.append(child_skip)
                entries.append(
                    WorkspaceFileIndexEntry(
                        path=self._display_path(child),
                        kind="skipped",
                        skipped_reason=child_skip.reason,
                    )
                )
                continue
            kind = "dir" if child.is_dir() else "file"
            entries.append(
                WorkspaceFileIndexEntry(
                    path=self._display_path(child),
                    kind=kind,
                    size_bytes=_safe_size(child) if kind == "file" else None,
                )
            )
            if len(entries) >= max_entries:
                break
        self._record_skips(local_skipped)
        return WorkspaceFileIndexResult(
            ok=True,
            path=self._display_path(resolved),
            entries=entries[:max_entries],
            truncated=len(entries) >= max_entries,
            files_skipped=local_skipped,
        )

    def search(
        self,
        pattern: str,
        *,
        file_glob: str | None = None,
        limit: int = 50,
        path: str = ".",
    ) -> WorkspaceSearchResult:
        if not pattern:
            return WorkspaceSearchResult(ok=False, pattern=pattern, reason="empty_pattern")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return WorkspaceSearchResult(
                ok=False,
                pattern=pattern,
                reason=f"invalid_regex: {exc}",
            )

        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspaceSearchResult(ok=False, pattern=pattern, reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspaceSearchResult(
                ok=False,
                pattern=pattern,
                reason=skipped.reason,
                backend="none",
                files_skipped=[skipped],
            )

        max_results = min(_positive_limit(limit, 50), self.limits.max_search_results)
        if shutil.which("rg"):
            return self._search_with_rg(
                pattern,
                resolved=resolved,
                file_glob=file_glob,
                max_results=max_results,
            )
        return self._search_with_python(
            regex,
            pattern=pattern,
            resolved=resolved,
            file_glob=file_glob,
            max_results=max_results,
            fallback_reason="rg_unavailable",
        )

    def _search_with_rg(
        self,
        pattern: str,
        *,
        resolved: Path,
        file_glob: str | None,
        max_results: int,
    ) -> WorkspaceSearchResult:
        root_skip = None
        if resolved.is_file():
            root_skip = self._skip_for_path(resolved, require_file=True, for_search=True)
        if root_skip:
            return WorkspaceSearchResult(
                ok=False,
                pattern=pattern,
                reason=root_skip.reason,
                backend="rg",
                files_skipped=[root_skip],
            )

        local_skipped = self._collect_denied_dir_skips(resolved)
        command = [
            "rg",
            "--json",
            "--line-number",
            "--color",
            "never",
            "--hidden",
        ]
        for deny_dir in sorted(self.deny_dirs):
            command.extend(["--glob", f"!**/{deny_dir}/**"])
            command.extend(["--glob", f"!{deny_dir}/**"])
        if file_glob:
            command.extend(["--glob", file_glob])
        command.extend(["--", pattern, str(resolved)])
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=6,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._search_with_python(
                re.compile(pattern),
                pattern=pattern,
                resolved=resolved,
                file_glob=file_glob,
                max_results=max_results,
                fallback_reason=f"rg_failed: {exc}",
                initial_skipped=local_skipped,
            )

        if completed.returncode == 1:
            self._record_skips(local_skipped)
            return WorkspaceSearchResult(
                ok=True,
                pattern=pattern,
                matches=[],
                backend="rg",
                files_skipped=local_skipped,
            )
        if completed.returncode != 0:
            return self._search_with_python(
                re.compile(pattern),
                pattern=pattern,
                resolved=resolved,
                file_glob=file_glob,
                max_results=max_results,
                fallback_reason="rg_failed",
                initial_skipped=local_skipped,
            )

        matches: list[WorkspaceSearchMatch] = []
        truncated = False
        for raw_line in completed.stdout.splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            match = self._match_from_rg_event(data)
            if match is None:
                continue
            matches.append(match)
            if len(matches) >= max_results:
                truncated = True
                break
        self._record_skips(local_skipped)
        return WorkspaceSearchResult(
            ok=True,
            pattern=pattern,
            matches=matches,
            truncated=truncated,
            backend="rg",
            files_skipped=local_skipped,
        )

    def _search_with_python(
        self,
        regex: re.Pattern[str],
        *,
        pattern: str,
        resolved: Path,
        file_glob: str | None,
        max_results: int,
        fallback_reason: str = "",
        initial_skipped: list[WorkspaceFileSkipped] | None = None,
    ) -> WorkspaceSearchResult:
        matches: list[WorkspaceSearchMatch] = []
        local_skipped: list[WorkspaceFileSkipped] = list(initial_skipped or [])
        files_scanned = 0
        for file_path in self._iter_candidate_files(resolved, file_glob=file_glob):
            file_skip = self._skip_for_path(file_path, require_file=True, for_search=True)
            if file_skip:
                local_skipped.append(file_skip)
                continue
            if files_scanned >= self.limits.max_search_files_scanned:
                self._record_skips(local_skipped)
                return WorkspaceSearchResult(
                    ok=True,
                    pattern=pattern,
                    matches=matches,
                    truncated=True,
                    reason="max_search_files_scanned_exceeded",
                    backend="python",
                    files_skipped=local_skipped,
                )
            files_scanned += 1
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        line = raw_line.rstrip("\n")
                        if regex.search(line):
                            safe_line = _redact_sensitive_text(line)[:500]
                            matches.append(
                                WorkspaceSearchMatch(
                                    path=self._display_path(file_path),
                                    line_number=line_number,
                                    line=safe_line,
                                    content_hash=_content_hash(safe_line),
                                )
                            )
                            if len(matches) >= max_results:
                                self._record_skips(local_skipped)
                                return WorkspaceSearchResult(
                                    ok=True,
                                    pattern=pattern,
                                    matches=matches,
                                    truncated=True,
                                    reason=fallback_reason,
                                    backend="python",
                                    files_skipped=local_skipped,
                                )
            except OSError as exc:
                local_skipped.append(
                    self._skip(file_path, "read_error", {"error": str(exc)})
                )
        self._record_skips(local_skipped)
        return WorkspaceSearchResult(
            ok=True,
            pattern=pattern,
            matches=matches,
            reason=fallback_reason,
            backend="python",
            files_skipped=local_skipped,
        )

    def _match_from_rg_event(self, data: dict[str, Any]) -> WorkspaceSearchMatch | None:
        path_payload = data.get("path")
        line_payload = data.get("lines")
        if not isinstance(path_payload, dict) or not isinstance(line_payload, dict):
            return None
        raw_path = path_payload.get("text")
        raw_line = line_payload.get("text")
        if not isinstance(raw_path, str) or not isinstance(raw_line, str):
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            self._ensure_lexically_inside_workspace(Path(os.path.abspath(candidate)))
        except ValueError:
            return None
        line_number = data.get("line_number")
        try:
            parsed_line_number = int(line_number)
        except (TypeError, ValueError):
            parsed_line_number = 0
        safe_line = _redact_sensitive_text(raw_line.rstrip("\n"))[:500]
        return WorkspaceSearchMatch(
            path=self._display_path(candidate),
            line_number=parsed_line_number,
            line=safe_line,
            content_hash=_content_hash(safe_line),
        )

    def _collect_denied_dir_skips(
        self,
        root: Path,
        *,
        limit: int = 50,
    ) -> list[WorkspaceFileSkipped]:
        if root.is_file():
            return []
        local_skipped: list[WorkspaceFileSkipped] = []
        try:
            children = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError:
            return local_skipped
        for child in children:
            if not child.is_dir():
                continue
            display_parts = Path(self._display_path(child)).parts
            if any(part in self.deny_dirs for part in display_parts):
                local_skipped.append(self._skip(child, "deny_dir"))
                if len(local_skipped) >= limit:
                    return local_skipped
        return local_skipped

    def read_file(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int = 300,
    ) -> WorkspaceReadResult:
        line_start = max(int(offset or 1), 1)
        line_limit = min(_positive_limit(limit, 300), self.limits.max_line_limit)
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspaceReadResult(ok=False, path=str(path or ""), reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=True)
        if skipped:
            return WorkspaceReadResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )
        if self._files_read_count >= self.limits.max_files_read:
            skipped = self._skip(resolved, "max_files_read_exceeded")
            return WorkspaceReadResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )
        remaining_budget = self.limits.total_char_budget - self._chars_used
        if remaining_budget <= 0:
            skipped = self._skip(resolved, "total_char_budget_exceeded")
            return WorkspaceReadResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )

        dedup_key = (self._display_path(resolved), line_start, line_limit)
        mtime = _safe_mtime(resolved)
        if self._read_cache.get(dedup_key) == mtime:
            return WorkspaceReadResult(
                ok=True,
                path=self._display_path(resolved),
                reason="already_read",
                dedup=True,
            )

        char_limit = min(self.limits.max_chars_per_file, remaining_budget)
        lines: list[str] = []
        line_end: int | None = None
        truncated = False
        try:
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if line_number < line_start:
                        continue
                    if len(lines) >= line_limit:
                        truncated = True
                        break
                    safe_line = _redact_sensitive_text(raw_line.rstrip("\n"))
                    next_content = "\n".join([*lines, safe_line]) if lines else safe_line
                    if len(next_content) > char_limit:
                        remaining = max(char_limit - (len("\n".join(lines)) + (1 if lines else 0)), 0)
                        if remaining > 0:
                            lines.append(safe_line[:remaining])
                            line_end = line_number
                        truncated = True
                        break
                    lines.append(safe_line)
                    line_end = line_number
        except OSError as exc:
            skipped = self._skip(resolved, "read_error", {"error": str(exc)})
            return WorkspaceReadResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )

        content = "\n".join(lines)
        content_hash = _content_hash(content)
        chars_read = len(content)
        self._chars_used += chars_read
        self._files_read_count += 1
        self._read_cache[dedup_key] = mtime
        file_read = WorkspaceFileRead(
            path=self._display_path(resolved),
            chars_read=chars_read,
            line_start=line_start if lines else None,
            line_end=line_end,
            truncated=truncated,
            content_hash=content_hash,
        )
        self.files_read.append(file_read)
        return WorkspaceReadResult(
            ok=True,
            path=file_read.path,
            content=content,
            line_start=file_read.line_start,
            line_end=file_read.line_end,
            chars_read=chars_read,
            truncated=truncated,
            content_hash=content_hash,
        )

    def git_status(self, *, limit: int = 100) -> WorkspaceGitStatusResult:
        command = ["git", "-C", str(self.workspace_root), "status", "--porcelain=v1", "--branch"]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return WorkspaceGitStatusResult(ok=False, reason=str(exc))
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "not_git_repository").strip()
            return WorkspaceGitStatusResult(ok=False, reason=reason)

        branch = ""
        entries: list[WorkspaceGitStatusEntry] = []
        max_entries = _positive_limit(limit, 100)
        truncated = False
        for raw_line in completed.stdout.splitlines():
            if raw_line.startswith("## "):
                branch = raw_line[3:]
                continue
            if not raw_line:
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            status = raw_line[:2].strip() or "?"
            path = raw_line[3:] if len(raw_line) > 3 else raw_line[2:].strip()
            entries.append(WorkspaceGitStatusEntry(path=path, status=status))
        return WorkspaceGitStatusResult(
            ok=True,
            branch=branch,
            entries=entries,
            truncated=truncated,
        )

    def git_diff(
        self,
        *,
        path: str = "",
        mode: str = "stat",
        max_chars: int | None = None,
    ) -> WorkspaceGitDiffResult:
        normalized_mode = "patch" if str(mode or "").strip().lower() == "patch" else "stat"
        display_path = ""
        git_path = ""
        if path:
            try:
                git_path, display_path = self._git_path(path)
            except ValueError as exc:
                return WorkspaceGitDiffResult(ok=False, path=str(path), mode=normalized_mode, reason=str(exc))
        command = ["git", "-C", str(self.workspace_root), "diff"]
        if normalized_mode == "stat":
            command.append("--stat")
        if git_path:
            command.extend(["--", git_path])
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return WorkspaceGitDiffResult(ok=False, path=display_path, mode=normalized_mode, reason=str(exc))
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "git_diff_failed").strip()
            return WorkspaceGitDiffResult(ok=False, path=display_path, mode=normalized_mode, reason=reason)
        content, truncated = _truncate_text(
            _redact_sensitive_text(completed.stdout),
            _positive_limit(max_chars, self.limits.max_git_diff_chars),
        )
        return WorkspaceGitDiffResult(
            ok=True,
            path=display_path,
            mode=normalized_mode,
            content=content,
            chars_read=len(content),
            content_hash=_content_hash(content),
            truncated=truncated,
        )

    def git_log(self, *, limit: int = 10) -> WorkspaceGitLogResult:
        max_entries = min(_positive_limit(limit, 10), self.limits.max_git_log_entries)
        command = [
            "git",
            "-C",
            str(self.workspace_root),
            "log",
            f"--max-count={max_entries}",
            "--pretty=format:%H%x1f%D%x1f%s",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return WorkspaceGitLogResult(ok=False, reason=str(exc))
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "git_log_failed").strip()
            return WorkspaceGitLogResult(ok=False, reason=reason)
        entries: list[WorkspaceGitLogEntry] = []
        for raw_line in completed.stdout.splitlines():
            commit, refs, subject = _split_git_log_line(raw_line)
            if commit:
                entries.append(
                    WorkspaceGitLogEntry(
                        commit=commit,
                        refs=refs,
                        subject=_redact_sensitive_text(subject),
                    )
                )
        return WorkspaceGitLogResult(ok=True, entries=entries, truncated=len(entries) >= max_entries)

    def repo_map(
        self,
        path: str = ".",
        *,
        max_depth: int = 3,
        limit: int | None = None,
    ) -> WorkspaceRepoMapResult:
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspaceRepoMapResult(ok=False, path=str(path or "."), reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspaceRepoMapResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )
        max_entries = min(
            _positive_limit(limit, self.limits.max_repo_map_entries),
            self.limits.max_repo_map_entries,
        )
        depth_limit = max(int(max_depth or 1), 1)
        entries: list[WorkspaceRepoMapEntry] = []
        local_skipped: list[WorkspaceFileSkipped] = []
        if resolved.is_file():
            entries.append(
                WorkspaceRepoMapEntry(
                    path=self._display_path(resolved),
                    kind="file",
                    depth=0,
                    size_bytes=_safe_size(resolved),
                )
            )
            return WorkspaceRepoMapResult(ok=True, path=self._display_path(resolved), entries=entries)

        base_parts = Path(self._display_path(resolved)).parts
        for dirpath, dirnames, filenames in os.walk(resolved, followlinks=False):
            current = Path(dirpath)
            current_display = self._display_path(current)
            current_parts = Path(current_display).parts if current_display != "." else ()
            depth = max(len(current_parts) - len(base_parts), 0)
            if depth > depth_limit:
                dirnames[:] = []
                continue
            allowed_dirs: list[str] = []
            for dirname in sorted(dirnames):
                child = current / dirname
                child_skip = self._skip_for_path(child, require_file=False)
                if child_skip:
                    local_skipped.append(child_skip)
                    if len(entries) < max_entries:
                        entries.append(
                            WorkspaceRepoMapEntry(
                                path=self._display_path(child),
                                kind="skipped",
                                depth=depth + 1,
                                skipped_reason=child_skip.reason,
                            )
                        )
                    continue
                allowed_dirs.append(dirname)
                if len(entries) < max_entries:
                    entries.append(
                        WorkspaceRepoMapEntry(
                            path=self._display_path(child),
                            kind="dir",
                            depth=depth + 1,
                        )
                    )
            dirnames[:] = allowed_dirs if depth + 1 < depth_limit else []
            for filename in sorted(filenames):
                if len(entries) >= max_entries:
                    self._record_skips(local_skipped)
                    return WorkspaceRepoMapResult(
                        ok=True,
                        path=self._display_path(resolved),
                        entries=entries,
                        truncated=True,
                        files_skipped=local_skipped,
                    )
                file_path = current / filename
                file_skip = self._skip_for_path(file_path, require_file=True, for_search=True)
                if file_skip:
                    local_skipped.append(file_skip)
                    continue
                entries.append(
                    WorkspaceRepoMapEntry(
                        path=self._display_path(file_path),
                        kind="file",
                        depth=depth + 1,
                        size_bytes=_safe_size(file_path),
                    )
                )
        self._record_skips(local_skipped)
        return WorkspaceRepoMapResult(
            ok=True,
            path=self._display_path(resolved),
            entries=entries,
            files_skipped=local_skipped,
        )

    def read_package_metadata(
        self,
        path: str = ".",
        *,
        limit: int | None = None,
    ) -> WorkspacePackageMetadataResult:
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspacePackageMetadataResult(ok=False, reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspacePackageMetadataResult(ok=False, reason=skipped.reason, files_skipped=[skipped])
        max_files = min(
            _positive_limit(limit, self.limits.max_package_metadata_files),
            self.limits.max_package_metadata_files,
        )
        candidates = _package_metadata_candidates(resolved if resolved.is_dir() else resolved.parent)
        files: list[WorkspacePackageMetadataFile] = []
        files_read_before = len(self.files_read)
        local_skipped: list[WorkspaceFileSkipped] = []
        for candidate in candidates:
            if len(files) >= max_files:
                break
            rel = self._display_path(candidate)
            read_result = self.read_file(rel, offset=1, limit=250)
            if not read_result.ok:
                local_skipped.extend(read_result.files_skipped)
                continue
            files.append(_package_metadata_file_from_read(read_result))
        return WorkspacePackageMetadataResult(
            ok=True,
            files=files,
            files_read=self.files_read[files_read_before:],
            files_skipped=local_skipped,
            truncated=len(candidates) > len(files),
            reason="" if files else "no_package_metadata_found",
        )

    def read_test_structure(
        self,
        path: str = ".",
        *,
        limit: int | None = None,
    ) -> WorkspaceTestStructureResult:
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspaceTestStructureResult(ok=False, reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspaceTestStructureResult(ok=False, reason=skipped.reason, files_skipped=[skipped])
        max_entries = min(
            _positive_limit(limit, self.limits.max_test_structure_entries),
            self.limits.max_test_structure_entries,
        )
        test_files: list[WorkspaceTestFileEntry] = []
        test_dirs: set[str] = set()
        framework_hints: set[str] = set()
        local_skipped: list[WorkspaceFileSkipped] = []
        for file_path in self._iter_candidate_files(resolved, file_glob=None):
            file_skip = self._skip_for_path(file_path, require_file=True, for_search=True)
            if file_skip:
                local_skipped.append(file_skip)
                continue
            rel = self._display_path(file_path)
            kind = _test_file_kind(rel)
            if not kind:
                continue
            hint = _framework_hint(rel)
            if hint:
                framework_hints.add(hint)
            parent = Path(rel).parent.as_posix()
            if parent and parent != ".":
                test_dirs.add(parent)
            test_files.append(WorkspaceTestFileEntry(path=rel, kind=kind, framework_hint=hint))
            if len(test_files) >= max_entries:
                self._record_skips(local_skipped)
                return WorkspaceTestStructureResult(
                    ok=True,
                    test_files=test_files,
                    test_dirs=sorted(test_dirs),
                    framework_hints=sorted(framework_hints),
                    files_skipped=local_skipped,
                    truncated=True,
                )
        self._record_skips(local_skipped)
        return WorkspaceTestStructureResult(
            ok=True,
            test_files=test_files,
            test_dirs=sorted(test_dirs),
            framework_hints=sorted(framework_hints),
            files_skipped=local_skipped,
            reason="" if test_files else "no_tests_found",
        )

    def python_symbol_index(
        self,
        path: str = ".",
        *,
        file_glob: str | None = None,
        limit: int | None = None,
    ) -> WorkspacePythonSymbolIndexResult:
        try:
            resolved = self._resolve_existing(path)
        except ValueError as exc:
            return WorkspacePythonSymbolIndexResult(ok=False, path=str(path or "."), reason=str(exc))
        skipped = self._skip_for_path(resolved, require_file=False)
        if skipped:
            return WorkspacePythonSymbolIndexResult(
                ok=False,
                path=self._display_path(resolved),
                reason=skipped.reason,
                files_skipped=[skipped],
            )

        max_symbols = min(
            _positive_limit(limit, self.limits.max_python_symbol_entries),
            self.limits.max_python_symbol_entries,
        )
        max_imports = self.limits.max_python_import_entries
        symbols: list[WorkspacePythonSymbolEntry] = []
        imports: list[WorkspacePythonImportEntry] = []
        modules: list[WorkspacePythonModuleEntry] = []
        local_skipped: list[WorkspaceFileSkipped] = []
        files_inspected = 0
        truncated = False

        for file_path in self._iter_candidate_files(resolved, file_glob=file_glob or "*.py"):
            file_skip = self._skip_for_path(file_path, require_file=True, for_search=True)
            if file_skip:
                local_skipped.append(file_skip)
                continue
            if file_path.suffix.lower() != ".py":
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(content, filename=self._display_path(file_path))
            except (OSError, SyntaxError) as exc:
                local_skipped.append(
                    self._skip(
                        file_path,
                        "python_parse_error",
                        {"error": _redact_sensitive_text(str(exc))[:240]},
                    )
                )
                continue

            files_inspected += 1
            module_symbols = _python_symbols_from_ast(tree, path=self._display_path(file_path))
            module_imports = _python_imports_from_ast(tree, path=self._display_path(file_path))
            remaining_symbols = max(max_symbols - len(symbols), 0)
            remaining_imports = max(max_imports - len(imports), 0)
            symbols.extend(module_symbols[:remaining_symbols])
            imports.extend(module_imports[:remaining_imports])
            modules.append(
                WorkspacePythonModuleEntry(
                    path=self._display_path(file_path),
                    module=_module_name_for_python_path(self._display_path(file_path)),
                    symbol_count=len(module_symbols),
                    import_count=len(module_imports),
                )
            )
            if len(module_symbols) > remaining_symbols or len(module_imports) > remaining_imports:
                truncated = True
                break

        self._record_skips(local_skipped)
        return WorkspacePythonSymbolIndexResult(
            ok=True,
            path=self._display_path(resolved),
            symbols=symbols,
            imports=imports,
            modules=modules,
            files_inspected=files_inspected,
            files_skipped=local_skipped,
            reason="" if files_inspected else "no_python_files_found",
            truncated=truncated,
        )

    def read_python_symbol(
        self,
        *,
        path: str = ".",
        qualified_name: str = "",
        name: str = "",
        kind: str = "",
    ) -> WorkspacePythonSymbolReadResult:
        target = str(qualified_name or name or "").strip()
        if not target:
            return WorkspacePythonSymbolReadResult(ok=False, path=str(path or "."), reason="missing_symbol_query")
        index = self.python_symbol_index(path=path, limit=self.limits.max_python_symbol_entries)
        if not index.ok:
            return WorkspacePythonSymbolReadResult(
                ok=False,
                path=index.path,
                qualified_name=str(qualified_name or ""),
                name=str(name or ""),
                kind=str(kind or ""),
                reason=index.reason,
                files_skipped=index.files_skipped,
            )
        match = _select_python_symbol(
            index.symbols,
            qualified_name=str(qualified_name or ""),
            name=str(name or ""),
            kind=str(kind or ""),
        )
        if match is None:
            return WorkspacePythonSymbolReadResult(
                ok=False,
                path=index.path,
                qualified_name=str(qualified_name or ""),
                name=str(name or ""),
                kind=str(kind or ""),
                reason="symbol_not_found",
                files_skipped=index.files_skipped,
            )
        line_limit = max(match.line_end - match.line_start + 1, 1)
        read_result = self.read_file(match.path, offset=match.line_start, limit=line_limit)
        if not read_result.ok:
            return WorkspacePythonSymbolReadResult(
                ok=False,
                path=match.path,
                qualified_name=match.qualified_name,
                name=match.name,
                kind=match.kind,
                reason=read_result.reason,
                files_skipped=read_result.files_skipped,
            )
        return WorkspacePythonSymbolReadResult(
            ok=True,
            path=match.path,
            qualified_name=match.qualified_name,
            name=match.name,
            kind=match.kind,
            content=read_result.content,
            line_start=read_result.line_start,
            line_end=read_result.line_end,
            chars_read=read_result.chars_read,
            truncated=read_result.truncated,
            content_hash=read_result.content_hash,
            reason=read_result.reason,
            files_skipped=read_result.files_skipped,
        )

    def summarize_workspace(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "files_read": [item.to_payload() for item in self.files_read],
            "files_skipped": [item.to_payload() for item in self.files_skipped],
            "limits": self.limits.to_payload(),
            "budget": {
                "files_read_count": self._files_read_count,
                "chars_used": self._chars_used,
                "chars_remaining": max(self.limits.total_char_budget - self._chars_used, 0),
            },
        }

    def _resolve_existing(self, path: str) -> Path:
        if not path:
            path = "."
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        lexical_path = Path(os.path.abspath(candidate))
        self._ensure_lexically_inside_workspace(lexical_path)
        if not lexical_path.exists():
            raise ValueError(f"Path does not exist: {self._display_path(lexical_path)}")
        return lexical_path

    def _git_path(self, path: str) -> tuple[str, str]:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            lexical_path = Path(os.path.abspath(candidate))
        else:
            lexical_path = Path(os.path.abspath(self.workspace_root / candidate))
        self._ensure_lexically_inside_workspace(lexical_path)
        display = self._display_path(lexical_path)
        for part in Path(display).parts:
            if part in self.deny_dirs:
                raise ValueError(f"Path is in denied directory: {display}")
        if lexical_path.exists():
            try:
                self._ensure_inside_workspace(lexical_path.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Path escapes workspace root: {lexical_path}") from exc
        return display, display

    def _ensure_lexically_inside_workspace(self, path: Path) -> None:
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace root: {path}") from exc

    def _ensure_inside_workspace(self, resolved: Path) -> None:
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(
                f"Path escapes workspace root: {resolved}"
            ) from exc

    def _skip_for_path(
        self,
        path: Path,
        *,
        require_file: bool,
        for_search: bool = False,
    ) -> WorkspaceFileSkipped | None:
        display_path = self._display_path(path)
        rel_parts = Path(display_path).parts
        for part in rel_parts:
            if part in self.deny_dirs:
                return self._skip(path, "deny_dir")
        try:
            real_path = path.resolve(strict=True)
            self._ensure_inside_workspace(real_path)
        except (OSError, ValueError):
            return self._skip(path, "symlink_escape")
        if require_file and not path.is_file():
            return self._skip(path, "not_file")
        if require_file and self._is_binary(path):
            return self._skip(path, "binary_file")
        if for_search and _safe_size(path) > self.limits.max_file_size_bytes:
            return self._skip(
                path,
                "file_too_large",
                {"size_bytes": _safe_size(path), "max_file_size_bytes": self.limits.max_file_size_bytes},
            )
        return None

    def _iter_candidate_files(self, root: Path, *, file_glob: str | None) -> Iterable[Path]:
        if root.is_file():
            if not file_glob or _matches_glob(self._display_path(root), file_glob):
                yield root
            return
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            current = Path(dirpath)
            allowed_dirs: list[str] = []
            for dirname in dirnames:
                child = current / dirname
                if self._skip_for_path(child, require_file=False):
                    continue
                allowed_dirs.append(dirname)
            dirnames[:] = allowed_dirs
            for filename in sorted(filenames):
                file_path = current / filename
                rel = self._display_path(file_path)
                if file_glob and not _matches_glob(rel, file_glob):
                    continue
                yield file_path

    def _is_binary(self, path: Path) -> bool:
        if path.suffix.lower() in self.binary_extensions:
            return True
        try:
            with path.open("rb") as handle:
                sample = handle.read(4096)
        except OSError:
            return False
        return b"\x00" in sample

    def _display_path(self, path: Path) -> str:
        try:
            return Path(os.path.abspath(path)).relative_to(self.workspace_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _skip(
        self,
        path: Path,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceFileSkipped:
        item = WorkspaceFileSkipped(
            path=self._display_path(path),
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self._record_skips([item])
        return item

    def _record_skips(self, items: Iterable[WorkspaceFileSkipped]) -> None:
        existing = {(item.path, item.reason) for item in self.files_skipped}
        for item in items:
            key = (item.path, item.reason)
            if key not in existing:
                self.files_skipped.append(item)
                existing.add(key)


def _python_symbols_from_ast(tree: ast.AST, *, path: str) -> list[WorkspacePythonSymbolEntry]:
    symbols: list[WorkspacePythonSymbolEntry] = []

    def visit_body(body: list[ast.stmt], parents: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = ".".join([*parents, node.name]) if parents else node.name
                symbols.append(
                    WorkspacePythonSymbolEntry(
                        path=path,
                        name=node.name,
                        qualified_name=qualified,
                        kind="class",
                        line_start=_node_start_line(node),
                        line_end=_node_end_line(node),
                        parent=".".join(parents),
                        decorators=_decorator_names(node),
                    )
                )
                visit_body(list(node.body), [*parents, node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join([*parents, node.name]) if parents else node.name
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                if parents:
                    kind = "async_method" if isinstance(node, ast.AsyncFunctionDef) else "method"
                symbols.append(
                    WorkspacePythonSymbolEntry(
                        path=path,
                        name=node.name,
                        qualified_name=qualified,
                        kind=kind,
                        line_start=_node_start_line(node),
                        line_end=_node_end_line(node),
                        parent=".".join(parents),
                        decorators=_decorator_names(node),
                    )
                )
                visit_body(list(node.body), [*parents, node.name])

    visit_body(list(getattr(tree, "body", [])), [])
    return sorted(symbols, key=lambda item: (item.path, item.line_start, item.qualified_name))


def _python_imports_from_ast(tree: ast.AST, *, path: str) -> list[WorkspacePythonImportEntry]:
    imports: list[WorkspacePythonImportEntry] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    WorkspacePythonImportEntry(
                        path=path,
                        module=alias.name,
                        names=[alias.asname] if alias.asname else [],
                        line_number=int(getattr(node, "lineno", 0) or 0),
                        kind="import",
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                WorkspacePythonImportEntry(
                    path=path,
                    module=str(node.module or ""),
                    names=[alias.name for alias in node.names],
                    level=int(node.level or 0),
                    line_number=int(getattr(node, "lineno", 0) or 0),
                    kind="from_import",
                )
            )
    return sorted(imports, key=lambda item: (item.path, item.line_number, item.module))


def _node_start_line(node: ast.AST) -> int:
    lineno = int(getattr(node, "lineno", 1) or 1)
    decorators = getattr(node, "decorator_list", [])
    decorator_lines = [
        int(getattr(decorator, "lineno", lineno) or lineno)
        for decorator in decorators
    ]
    return min([lineno, *decorator_lines]) if decorator_lines else lineno


def _node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", None) or getattr(node, "lineno", 1) or 1)


def _decorator_names(node: ast.AST) -> list[str]:
    decorators = getattr(node, "decorator_list", [])
    names: list[str] = []
    for decorator in decorators:
        try:
            names.append(ast.unparse(decorator))
        except Exception:
            names.append(type(decorator).__name__)
    return names[:12]


def _module_name_for_python_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.endswith("/__init__.py"):
        normalized = normalized[: -len("/__init__.py")]
    elif normalized.endswith(".py"):
        normalized = normalized[:-3]
    parts = [part for part in Path(normalized).parts if part not in {".", ""}]
    return ".".join(parts)


def _select_python_symbol(
    symbols: list[WorkspacePythonSymbolEntry],
    *,
    qualified_name: str = "",
    name: str = "",
    kind: str = "",
) -> WorkspacePythonSymbolEntry | None:
    target_qualified = qualified_name.strip()
    target_name = name.strip()
    target_kind = kind.strip()
    if target_qualified:
        for symbol in symbols:
            if symbol.qualified_name == target_qualified and _symbol_kind_matches(symbol, target_kind):
                return symbol
    if target_name:
        for symbol in symbols:
            if symbol.name == target_name and _symbol_kind_matches(symbol, target_kind):
                return symbol
        for symbol in symbols:
            if symbol.qualified_name.endswith(f".{target_name}") and _symbol_kind_matches(symbol, target_kind):
                return symbol
    return None


def _symbol_kind_matches(symbol: WorkspacePythonSymbolEntry, kind: str) -> bool:
    if not kind:
        return True
    normalized = kind.strip()
    if normalized == "function":
        return symbol.kind in {"function", "async_function", "method", "async_method"}
    if normalized == "method":
        return symbol.kind in {"method", "async_method"}
    return symbol.kind == normalized


def _positive_limit(value: int | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else int(default)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(parsed, 1)


def _matches_glob(path: str, pattern: str) -> bool:
    return fnmatch(path, pattern) or fnmatch(Path(path).name, pattern)


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    normalized = str(text or "")
    if len(normalized) <= limit:
        return normalized, False
    return normalized[: max(0, limit)].rstrip(), True


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _safe_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _content_hash(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()[:16]


def _redact_sensitive_text(text: str) -> str:
    patterns = [
        re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?([^'\"\s]+)"),
        re.compile(r"(?i)(bearer)\s+([a-z0-9._\-]+)"),
    ]
    redacted = text
    for pattern in patterns:
        redacted = pattern.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    return redacted


def _split_git_log_line(raw_line: str) -> tuple[str, str, str]:
    parts = raw_line.split("\x1f", 2)
    if len(parts) != 3:
        return "", "", raw_line
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _package_metadata_candidates(root: Path) -> list[Path]:
    names = {
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "package.json",
        "pnpm-workspace.yaml",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "Gemfile",
    }
    candidates: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if name not in DEFAULT_DENY_DIRS]
        for filename in sorted(filenames):
            if filename in names:
                candidates.append(current / filename)
    return sorted(candidates, key=lambda item: item.as_posix())


def _package_metadata_file_from_read(read_result: WorkspaceReadResult) -> WorkspacePackageMetadataFile:
    content = read_result.content
    path = read_result.path
    kind = Path(path).name
    name = ""
    version = ""
    scripts: list[str] = []
    dependencies: list[str] = []
    if kind == "package.json":
        name, version, scripts, dependencies = _metadata_from_package_json(content)
    elif kind == "pyproject.toml":
        name, version, scripts, dependencies = _metadata_from_pyproject(content)
    elif kind == "Cargo.toml":
        name, version, scripts, dependencies = _metadata_from_cargo_toml(content)
    elif kind == "go.mod":
        name = _metadata_from_go_mod(content)
    elif kind.startswith("requirements"):
        dependencies = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ][:40]
    preview, _ = _truncate_text(content, 1200)
    return WorkspacePackageMetadataFile(
        path=path,
        kind=kind,
        name=name,
        version=version,
        scripts=scripts[:40],
        dependencies=dependencies[:80],
        content_preview=preview,
        content_hash=read_result.content_hash,
        truncated=read_result.truncated,
    )


def _metadata_from_package_json(content: str) -> tuple[str, str, list[str], list[str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "", "", [], []
    if not isinstance(payload, dict):
        return "", "", [], []
    scripts = sorted(_mapping_keys(payload.get("scripts")))
    dependencies = sorted(
        {
            *_mapping_keys(payload.get("dependencies")),
            *_mapping_keys(payload.get("devDependencies")),
            *_mapping_keys(payload.get("peerDependencies")),
        }
    )
    return str(payload.get("name") or ""), str(payload.get("version") or ""), scripts, dependencies


def _metadata_from_pyproject(content: str) -> tuple[str, str, list[str], list[str]]:
    try:
        payload = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return "", "", [], []
    project = payload.get("project") if isinstance(payload, dict) else {}
    if not isinstance(project, dict):
        project = {}
    dependencies = [
        str(item)
        for item in project.get("dependencies", [])
        if isinstance(item, str)
    ]
    optional = project.get("optional-dependencies")
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                dependencies.extend(str(item) for item in values if isinstance(item, str))
    scripts = sorted(_mapping_keys(project.get("scripts")))
    return str(project.get("name") or ""), str(project.get("version") or ""), scripts, dependencies


def _metadata_from_cargo_toml(content: str) -> tuple[str, str, list[str], list[str]]:
    try:
        payload = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return "", "", [], []
    package = payload.get("package") if isinstance(payload, dict) else {}
    dependencies = sorted(_mapping_keys(payload.get("dependencies")) if isinstance(payload, dict) else [])
    if not isinstance(package, dict):
        package = {}
    return str(package.get("name") or ""), str(package.get("version") or ""), [], dependencies


def _metadata_from_go_mod(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("module "):
            return stripped.removeprefix("module ").strip()
    return ""


def _mapping_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    return []


def _test_file_kind(path: str) -> str:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    lower_name = name.lower()
    parts = normalized.lower().split("/")
    if "tests" in parts or "test" in parts:
        if Path(normalized).suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb"}:
            return "test_file"
    if lower_name.startswith("test_") and Path(normalized).suffix.lower() == ".py":
        return "test_file"
    if lower_name.endswith("_test.py"):
        return "test_file"
    if any(lower_name.endswith(suffix) for suffix in (".test.js", ".test.jsx", ".test.ts", ".test.tsx", ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx")):
        return "test_file"
    if lower_name.endswith("_test.go") or lower_name.endswith("_test.rs"):
        return "test_file"
    return ""


def _framework_hint(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".py"):
        return "pytest_or_unittest"
    if any(lower.endswith(suffix) for suffix in (".test.js", ".test.jsx", ".test.ts", ".test.tsx", ".spec.js", ".spec.jsx", ".spec.ts", ".spec.tsx")):
        return "javascript_test_runner"
    if lower.endswith("_test.go"):
        return "go_test"
    if lower.endswith("_test.rs"):
        return "rust_test"
    if lower.endswith(".java"):
        return "java_test"
    if lower.endswith(".rb"):
        return "ruby_test"
    return ""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
