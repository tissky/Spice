from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from spice.decision.general.types import payload_value, safe_dataclass_from_payload
from spice.perception.workspace_inspector import (
    WorkspaceInspector,
    WorkspacePackageMetadataResult,
    WorkspaceRepoMapEntry,
    WorkspaceRepoMapResult,
    WorkspaceTestStructureResult,
)


WORKSPACE_SUMMARY_CACHE_SCHEMA_VERSION = "spice.workspace_summary_cache.v1"
DEFAULT_WORKSPACE_SUMMARY_CACHE_MAX_AGE_SECONDS = 300


@dataclass(frozen=True, slots=True)
class WorkspaceDirectorySummary:
    path: str
    purpose: str = ""
    file_count: int = 0
    child_dirs: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFileSummary:
    path: str
    purpose: str = ""
    language: str = ""
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceSummaryCache:
    workspace_root: str
    created_at: str
    refreshed_at: str
    summary: str = ""
    cache_key: str = ""
    repo_map: list[WorkspaceRepoMapEntry] = field(default_factory=list)
    directory_summaries: list[WorkspaceDirectorySummary] = field(default_factory=list)
    file_summaries: list[WorkspaceFileSummary] = field(default_factory=list)
    package_metadata: dict[str, Any] = field(default_factory=dict)
    test_structure: dict[str, Any] = field(default_factory=dict)
    schema_version: str = WORKSPACE_SUMMARY_CACHE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WorkspaceSummaryCache":
        if not isinstance(payload, Mapping):
            raise ValueError("Workspace summary cache payload must be a mapping.")
        return cls(
            workspace_root=str(payload.get("workspace_root") or ""),
            created_at=str(payload.get("created_at") or ""),
            refreshed_at=str(payload.get("refreshed_at") or ""),
            summary=str(payload.get("summary") or ""),
            cache_key=str(payload.get("cache_key") or ""),
            repo_map=[
                safe_dataclass_from_payload(WorkspaceRepoMapEntry, item)
                for item in _mappings(payload.get("repo_map"))
            ],
            directory_summaries=[
                safe_dataclass_from_payload(WorkspaceDirectorySummary, item)
                for item in _mappings(payload.get("directory_summaries"))
            ],
            file_summaries=[
                safe_dataclass_from_payload(WorkspaceFileSummary, item)
                for item in _mappings(payload.get("file_summaries"))
            ],
            package_metadata=(
                dict(payload.get("package_metadata"))
                if isinstance(payload.get("package_metadata"), Mapping)
                else {}
            ),
            test_structure=(
                dict(payload.get("test_structure"))
                if isinstance(payload.get("test_structure"), Mapping)
                else {}
            ),
            schema_version=str(
                payload.get("schema_version") or WORKSPACE_SUMMARY_CACHE_SCHEMA_VERSION
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceSummaryCacheLoadResult:
    cache: WorkspaceSummaryCache
    status: str
    path: Path
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload = self.cache.to_payload()
        return {
            "status": self.status,
            "path": str(self.path),
            "reason": self.reason,
            "cache": payload,
        }


def workspace_summary_cache_path(project_root: str | Path) -> Path:
    return Path(project_root) / ".spice" / "cache" / "workspace_summary.json"


def load_workspace_summary_cache(path: str | Path) -> WorkspaceSummaryCache | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Workspace summary cache must be a JSON object: {cache_path}")
    return WorkspaceSummaryCache.from_payload(payload)


def save_workspace_summary_cache(
    cache: WorkspaceSummaryCache,
    path: str | Path,
) -> Path:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache.to_payload(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return cache_path


def load_or_refresh_workspace_summary_cache(
    *,
    project_root: str | Path,
    inspector: WorkspaceInspector | None = None,
    max_age_seconds: int = DEFAULT_WORKSPACE_SUMMARY_CACHE_MAX_AGE_SECONDS,
    max_depth: int = 4,
    limit: int | None = None,
    persist: bool = True,
) -> WorkspaceSummaryCacheLoadResult:
    cache_path = workspace_summary_cache_path(project_root)
    existing = load_workspace_summary_cache(cache_path)
    if existing is not None:
        age = _cache_age_seconds(existing)
        if age is not None and age <= max(max_age_seconds, 0):
            return WorkspaceSummaryCacheLoadResult(cache=existing, status="hit", path=cache_path)
    active_inspector = inspector or WorkspaceInspector(project_root)
    cache = build_workspace_summary_cache(
        active_inspector,
        max_depth=max_depth,
        limit=limit,
        created_at=existing.created_at if existing else None,
    )
    if persist:
        save_workspace_summary_cache(cache, cache_path)
    return WorkspaceSummaryCacheLoadResult(
        cache=cache,
        status="preview" if not persist else ("refreshed" if existing else "created"),
        path=cache_path,
        reason="stale" if existing else "missing",
    )


def build_workspace_summary_cache(
    inspector: WorkspaceInspector,
    *,
    max_depth: int = 4,
    limit: int | None = None,
    created_at: str | None = None,
) -> WorkspaceSummaryCache:
    refreshed = _timestamp()
    repo_map = inspector.repo_map(max_depth=max_depth, limit=limit)
    package_metadata = inspector.read_package_metadata()
    test_structure = inspector.read_test_structure()
    entries = list(repo_map.entries if repo_map.ok else [])
    directories = _directory_summaries(entries)
    files = _file_summaries(
        entries=entries,
        package_metadata=package_metadata,
        test_structure=test_structure,
    )
    summary = _cache_summary(
        directories=directories,
        files=files,
        package_metadata=package_metadata,
        test_structure=test_structure,
    )
    return WorkspaceSummaryCache(
        workspace_root=str(inspector.workspace_root),
        created_at=created_at or refreshed,
        refreshed_at=refreshed,
        summary=summary,
        cache_key=_cache_key(entries),
        repo_map=entries,
        directory_summaries=directories,
        file_summaries=files,
        package_metadata=_compact_package_metadata(package_metadata),
        test_structure=_compact_test_structure(test_structure),
    )


def compact_workspace_summary_cache(
    cache: WorkspaceSummaryCache | Mapping[str, Any] | None,
    *,
    max_dirs: int = 24,
    max_files: int = 40,
    max_repo_entries: int = 80,
) -> dict[str, Any]:
    if cache is None:
        return {}
    payload = cache.to_payload() if isinstance(cache, WorkspaceSummaryCache) else dict(cache)
    return {
        "schema_version": str(payload.get("schema_version") or WORKSPACE_SUMMARY_CACHE_SCHEMA_VERSION),
        "source": "workspace_summary_cache",
        "workspace_root": str(payload.get("workspace_root") or ""),
        "refreshed_at": str(payload.get("refreshed_at") or ""),
        "cache_key": str(payload.get("cache_key") or ""),
        "summary": _shorten(str(payload.get("summary") or ""), 700),
        "directory_summaries": _mappings(payload.get("directory_summaries"))[:max_dirs],
        "file_summaries": _mappings(payload.get("file_summaries"))[:max_files],
        "repo_map": _mappings(payload.get("repo_map"))[:max_repo_entries],
        "package_metadata": _mapping(payload.get("package_metadata")),
        "test_structure": _mapping(payload.get("test_structure")),
    }


def _directory_summaries(entries: list[WorkspaceRepoMapEntry]) -> list[WorkspaceDirectorySummary]:
    direct_children: dict[str, set[str]] = {}
    file_counts: dict[str, int] = {}
    directories: set[str] = set()
    for entry in entries:
        if entry.kind == "dir":
            directories.add(entry.path)
        if entry.kind != "file":
            continue
        parent = Path(entry.path).parent.as_posix()
        if parent == ".":
            continue
        parts = Path(parent).parts
        for index in range(1, len(parts) + 1):
            directory = Path(*parts[:index]).as_posix()
            directories.add(directory)
            file_counts[directory] = file_counts.get(directory, 0) + 1
            if index < len(parts):
                direct_children.setdefault(directory, set()).add(Path(*parts[: index + 1]).as_posix())
    summaries = [
        WorkspaceDirectorySummary(
            path=path,
            purpose=_directory_purpose(path),
            file_count=file_counts.get(path, 0),
            child_dirs=sorted(direct_children.get(path, set()))[:12],
        )
        for path in sorted(directories)
    ]
    return summaries[:120]


def _file_summaries(
    *,
    entries: list[WorkspaceRepoMapEntry],
    package_metadata: WorkspacePackageMetadataResult,
    test_structure: WorkspaceTestStructureResult,
) -> list[WorkspaceFileSummary]:
    package_paths = {
        str(item.get("path") or "")
        for item in _mappings(package_metadata.to_payload().get("files"))
    }
    test_paths = {
        str(item.get("path") or "")
        for item in _mappings(test_structure.to_payload().get("test_files"))
    }
    summaries: list[WorkspaceFileSummary] = []
    for entry in entries:
        if entry.kind != "file":
            continue
        metadata: dict[str, Any] = {}
        if entry.path in package_paths:
            metadata["role"] = "package_metadata"
        if entry.path in test_paths:
            metadata["role"] = "test"
        purpose = _file_purpose(entry.path, metadata=metadata)
        summaries.append(
            WorkspaceFileSummary(
                path=entry.path,
                purpose=purpose,
                language=_language_for_path(entry.path),
                size_bytes=entry.size_bytes,
                metadata=metadata,
            )
        )
    return summaries[:240]


def _directory_purpose(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    rules = [
        ("spice/runtime", "runtime orchestration, TUI, approvals, execution, and workspace decision loop"),
        ("spice/llm", "LLM providers, prompts, proposal normalization, and simulation/composer adapters"),
        ("spice/perception", "read-only perception sources, workspace inspection, and perception artifacts"),
        ("spice/memory", "memory provider, context compiler, and decision-relevant state summaries"),
        ("spice/decision", "candidate generation, scoring, policy, and decision domain types"),
        ("spice/executors", "executor adapters, skill resolution, and handoff context"),
        ("tests", "automated test suite"),
        ("docs", "project documentation and architecture notes"),
        ("examples", "demos and example integrations"),
        ("schemas", "protocol and payload schemas"),
    ]
    for prefix, purpose in rules:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return purpose
    name = Path(normalized).name
    if name in {"runtime", "llm", "perception", "memory", "decision", "executors"}:
        return f"{name} implementation area"
    if name in {"tests", "test"}:
        return "test code"
    if name == "docs":
        return "documentation"
    return "workspace directory"


def _file_purpose(path: str, *, metadata: Mapping[str, Any]) -> str:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    if metadata.get("role") == "package_metadata":
        return "package metadata and project dependency/configuration file"
    if metadata.get("role") == "test":
        return "test file"
    if name in {"README.md", "README_zh.md"}:
        return "project overview documentation"
    if name == "workspace_inspector.py":
        return "workspace read-only primitives and local guardrails"
    if name == "workspace_loop.py":
        return "controlled read-only workspace perception loop"
    if name == "workspace.py":
        return "workspace perception artifact and compact context shaping"
    if name == "run_once.py":
        return "main runtime decision loop"
    if name == "interactive_shell.py" or normalized.endswith("/tui/shell.py"):
        return "interactive shell or TUI conversation flow"
    if name.endswith("_composer.py") or "composer" in name:
        return "natural language response composer"
    if normalized.startswith("docs/"):
        return "documentation"
    return "source file" if Path(normalized).suffix in {".py", ".js", ".ts", ".tsx", ".go", ".rs"} else "workspace file"


def _language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".go": "go",
        ".rs": "rust",
    }.get(suffix, "")


def _compact_package_metadata(result: WorkspacePackageMetadataResult) -> dict[str, Any]:
    payload = result.to_payload()
    return {
        "ok": bool(payload.get("ok")),
        "reason": str(payload.get("reason") or ""),
        "files": [
            {
                "path": str(item.get("path") or ""),
                "kind": str(item.get("kind") or ""),
                "name": str(item.get("name") or ""),
                "version": str(item.get("version") or ""),
                "scripts": list(item.get("scripts") or [])[:20],
                "dependencies": list(item.get("dependencies") or [])[:40],
            }
            for item in _mappings(payload.get("files"))[:20]
        ],
    }


def _compact_test_structure(result: WorkspaceTestStructureResult) -> dict[str, Any]:
    payload = result.to_payload()
    return {
        "ok": bool(payload.get("ok")),
        "reason": str(payload.get("reason") or ""),
        "test_dirs": list(payload.get("test_dirs") or [])[:40],
        "framework_hints": list(payload.get("framework_hints") or [])[:20],
        "test_files": _mappings(payload.get("test_files"))[:80],
    }


def _cache_summary(
    *,
    directories: list[WorkspaceDirectorySummary],
    files: list[WorkspaceFileSummary],
    package_metadata: WorkspacePackageMetadataResult,
    test_structure: WorkspaceTestStructureResult,
) -> str:
    package_payload = _compact_package_metadata(package_metadata)
    test_payload = _compact_test_structure(test_structure)
    package_names = [
        str(item.get("name") or item.get("path") or "")
        for item in package_payload.get("files", [])
        if item.get("name") or item.get("path")
    ][:4]
    frameworks = [str(item) for item in test_payload.get("framework_hints", []) if item][:4]
    parts = [
        f"Workspace map cached {len(directories)} directories and {len(files)} files.",
    ]
    if package_names:
        parts.append(f"Package metadata: {', '.join(package_names)}.")
    if frameworks:
        parts.append(f"Test hints: {', '.join(frameworks)}.")
    return " ".join(parts)


def _cache_key(entries: list[WorkspaceRepoMapEntry]) -> str:
    material = "\n".join(
        f"{entry.path}\t{entry.kind}\t{entry.size_bytes or ''}\t{entry.skipped_reason}"
        for entry in entries
    )
    return sha256(material.encode("utf-8")).hexdigest()[:16]


def _cache_age_seconds(cache: WorkspaceSummaryCache) -> float | None:
    try:
        refreshed = datetime.fromisoformat(cache.refreshed_at)
    except ValueError:
        return None
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - refreshed.astimezone(timezone.utc)).total_seconds()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _shorten(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "..."
