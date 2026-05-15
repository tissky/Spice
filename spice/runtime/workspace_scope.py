from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from spice.perception.workspace_inspector import DEFAULT_DENY_DIRS
from spice.runtime.resource_extractor import ResourceExtraction


WORKSPACE_SCOPE_RESOLUTION_SCHEMA_VERSION = "spice.workspace_scope_resolution.v1"

WORKSPACE_SCOPE_ALLOWED = "allowed"
WORKSPACE_SCOPE_NEEDS_CONFIRMATION = "needs_confirmation"
WORKSPACE_SCOPE_NEEDS_SELECTION = "needs_selection"
WORKSPACE_SCOPE_BLOCKED = "blocked"
WORKSPACE_SCOPE_NONE = "none"

WORKSPACE_SCOPE_CURRENT_ROOT = "current_workspace"
WORKSPACE_SCOPE_EXTERNAL_ROOT = "external_workspace"
WORKSPACE_SCOPE_SUBPATH = "workspace_subpath"


_REPO_MARKERS = (".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "README.md")
_SENSITIVE_ROOTS = {"/", "/Users", "/private", "/private/tmp", "/tmp", "/var", "/usr", "/etc", "/opt", "/Volumes"}


@dataclass(frozen=True, slots=True)
class WorkspaceScopeCandidate:
    raw_path: str
    resolved_path: str
    status: str
    scope_type: str = ""
    reason: str = ""
    workspace_root: str = ""
    scope_path: str = ""
    is_repo_root: bool = False
    repo_markers: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "raw_path": self.raw_path,
            "resolved_path": self.resolved_path,
            "status": self.status,
            "scope_type": self.scope_type,
            "reason": self.reason,
            "workspace_root": self.workspace_root,
            "scope_path": self.scope_path,
            "is_repo_root": self.is_repo_root,
            "repo_markers": list(self.repo_markers),
        }


@dataclass(frozen=True, slots=True)
class WorkspaceScopeResolution:
    status: str
    candidates: list[WorkspaceScopeCandidate] = field(default_factory=list)
    selected: WorkspaceScopeCandidate | None = None
    reason: str = ""

    @property
    def workspace_root(self) -> str:
        return self.selected.workspace_root if self.selected is not None else ""

    @property
    def scope_path(self) -> str:
        return self.selected.scope_path if self.selected is not None else ""

    @property
    def requires_confirmation(self) -> bool:
        return self.status == WORKSPACE_SCOPE_NEEDS_CONFIRMATION

    @property
    def blocked(self) -> bool:
        return self.status == WORKSPACE_SCOPE_BLOCKED

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_SCOPE_RESOLUTION_SCHEMA_VERSION,
            "status": self.status,
            "reason": self.reason,
            "workspace_root": self.workspace_root,
            "scope_path": self.scope_path,
            "requires_confirmation": self.requires_confirmation,
            "blocked": self.blocked,
            "selected": self.selected.to_payload() if self.selected is not None else {},
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


def resolve_workspace_scope(
    *,
    project_root: str | Path,
    resource_extraction: ResourceExtraction,
    interactive: bool = False,
    allow_external_roots: bool = False,
) -> WorkspaceScopeResolution:
    current_root = Path(project_root).expanduser().resolve()
    candidates = [
        _resolve_candidate(
            raw_path=path,
            current_root=current_root,
            interactive=interactive,
            allow_external_roots=allow_external_roots,
        )
        for path in [*resource_extraction.local_paths, *resource_extraction.relative_paths]
    ]
    if not candidates:
        return WorkspaceScopeResolution(status=WORKSPACE_SCOPE_NONE, reason="no workspace path references")

    allowed = [candidate for candidate in candidates if candidate.status == WORKSPACE_SCOPE_ALLOWED]
    confirmations = [
        candidate for candidate in candidates if candidate.status == WORKSPACE_SCOPE_NEEDS_CONFIRMATION
    ]
    blocked = [candidate for candidate in candidates if candidate.status == WORKSPACE_SCOPE_BLOCKED]

    if len(allowed) == 1 and not confirmations:
        return WorkspaceScopeResolution(
            status=WORKSPACE_SCOPE_ALLOWED,
            selected=allowed[0],
            candidates=candidates,
            reason=allowed[0].reason,
        )
    if len(allowed) > 1:
        return WorkspaceScopeResolution(
            status=WORKSPACE_SCOPE_NEEDS_SELECTION,
            candidates=candidates,
            reason="multiple workspace scopes matched; select one before reading",
        )
    if len(confirmations) == 1:
        return WorkspaceScopeResolution(
            status=WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
            selected=confirmations[0],
            candidates=candidates,
            reason=confirmations[0].reason,
        )
    if len(confirmations) > 1:
        return WorkspaceScopeResolution(
            status=WORKSPACE_SCOPE_NEEDS_SELECTION,
            candidates=candidates,
            reason="multiple external workspace scopes need confirmation",
        )
    return WorkspaceScopeResolution(
        status=WORKSPACE_SCOPE_BLOCKED,
        candidates=candidates,
        reason=_blocked_reason(blocked),
    )


def _resolve_candidate(
    *,
    raw_path: str,
    current_root: Path,
    interactive: bool,
    allow_external_roots: bool,
) -> WorkspaceScopeCandidate:
    try:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = current_root / path
        resolved = path.resolve(strict=False)
    except Exception as exc:
        return WorkspaceScopeCandidate(
            raw_path=raw_path,
            resolved_path="",
            status=WORKSPACE_SCOPE_BLOCKED,
            reason=f"path_resolution_failed:{exc}",
        )

    if _is_sensitive_root(resolved):
        return _candidate(raw_path, resolved, WORKSPACE_SCOPE_BLOCKED, reason="sensitive_root")
    if _contains_deny_dir(resolved):
        return _candidate(raw_path, resolved, WORKSPACE_SCOPE_BLOCKED, reason="deny_dir")
    if not resolved.exists():
        return _candidate(raw_path, resolved, WORKSPACE_SCOPE_BLOCKED, reason="path_not_found")

    try:
        resolved.relative_to(current_root)
        markers = _repo_markers(resolved if resolved.is_dir() else resolved.parent)
        return _candidate(
            raw_path,
            resolved,
            WORKSPACE_SCOPE_ALLOWED,
            scope_type=WORKSPACE_SCOPE_CURRENT_ROOT if resolved == current_root else WORKSPACE_SCOPE_SUBPATH,
            reason="path is inside current workspace",
            workspace_root=current_root,
            scope_path=resolved,
            repo_markers=markers,
        )
    except ValueError:
        pass

    external_root = _external_workspace_root(resolved)
    if external_root is None:
        return _candidate(
            raw_path,
            resolved,
            WORKSPACE_SCOPE_BLOCKED,
            reason="external path is not a recognizable repo root",
        )

    markers = _repo_markers(external_root)
    if allow_external_roots:
        return _candidate(
            raw_path,
            resolved,
            WORKSPACE_SCOPE_ALLOWED,
            scope_type=WORKSPACE_SCOPE_EXTERNAL_ROOT,
            reason="external repo root explicitly allowed",
            workspace_root=external_root,
            scope_path=resolved,
            repo_markers=markers,
        )
    if interactive:
        return _candidate(
            raw_path,
            resolved,
            WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
            scope_type=WORKSPACE_SCOPE_EXTERNAL_ROOT,
            reason="external repo root needs user confirmation",
            workspace_root=external_root,
            scope_path=resolved,
            repo_markers=markers,
        )
    return _candidate(
        raw_path,
        resolved,
        WORKSPACE_SCOPE_BLOCKED,
        scope_type=WORKSPACE_SCOPE_EXTERNAL_ROOT,
        reason="external repo root blocked in non-interactive mode",
        workspace_root=external_root,
        scope_path=resolved,
        repo_markers=markers,
    )


def _candidate(
    raw_path: str,
    resolved: Path,
    status: str,
    *,
    scope_type: str = "",
    reason: str = "",
    workspace_root: Path | None = None,
    scope_path: Path | None = None,
    repo_markers: Iterable[str] = (),
) -> WorkspaceScopeCandidate:
    markers = list(repo_markers)
    return WorkspaceScopeCandidate(
        raw_path=raw_path,
        resolved_path=str(resolved),
        status=status,
        scope_type=scope_type,
        reason=reason,
        workspace_root=str((workspace_root or Path()).resolve()) if workspace_root is not None else "",
        scope_path=str((scope_path or resolved).resolve()),
        is_repo_root=bool(markers) and (workspace_root is None or (scope_path or resolved) == workspace_root),
        repo_markers=markers,
    )


def _external_workspace_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in [start, *start.parents]:
        if _is_sensitive_root(candidate) or _contains_deny_dir(candidate):
            return None
        if _repo_markers(candidate):
            return candidate
    return None


def _repo_markers(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    markers: list[str] = []
    for marker in _REPO_MARKERS:
        if (path / marker).exists():
            markers.append(marker)
    return markers


def _contains_deny_dir(path: Path) -> bool:
    return any(part in DEFAULT_DENY_DIRS for part in path.parts)


def _is_sensitive_root(path: Path) -> bool:
    return str(path) in _SENSITIVE_ROOTS


def _blocked_reason(candidates: list[WorkspaceScopeCandidate]) -> str:
    reasons = [candidate.reason for candidate in candidates if candidate.reason]
    if not reasons:
        return "no readable workspace scope"
    unique: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        unique.append(reason)
    return "; ".join(unique)
