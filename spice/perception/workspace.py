from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from spice.decision.general.types import payload_value, safe_dataclass_from_payload


WORKSPACE_PERCEPTION_SCHEMA_VERSION = "spice.workspace_perception.v1"
WORKSPACE_CONTEXT_SCHEMA_VERSION = "spice.workspace_context.v1"

WORKSPACE_EXPLORATION_COMPLETE = "complete"
WORKSPACE_EXPLORATION_PARTIAL = "partial"
WORKSPACE_EXPLORATION_BUDGET_EXHAUSTED = "budget_exhausted"
WORKSPACE_EXPLORATION_BLOCKED = "blocked"
WORKSPACE_EXPLORATION_FAILED = "failed"

DEFAULT_WORKSPACE_PERCEPTION_MAX_FILES = 20
DEFAULT_WORKSPACE_PERCEPTION_MAX_CHARS_PER_FILE = 12_000
DEFAULT_WORKSPACE_PERCEPTION_TOTAL_CHAR_BUDGET = 50_000


@dataclass(frozen=True, slots=True)
class WorkspacePerceptionLimits:
    max_files: int = DEFAULT_WORKSPACE_PERCEPTION_MAX_FILES
    max_chars_per_file: int = DEFAULT_WORKSPACE_PERCEPTION_MAX_CHARS_PER_FILE
    total_char_budget: int = DEFAULT_WORKSPACE_PERCEPTION_TOTAL_CHAR_BUDGET

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePerceptionQuery:
    query: str
    query_type: str = "semantic"
    path: str = "."
    file_glob: str = ""
    limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFileRead:
    path: str
    chars_read: int = 0
    line_start: int | None = None
    line_end: int | None = None
    truncated: bool = False
    content_hash: str = ""
    reason: str = ""
    snippets: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFileSkipped:
    path: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceFact:
    text: str
    source_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceSnippet:
    path: str
    text: str
    line_start: int | None = None
    line_end: int | None = None
    content_hash: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspaceToolCall:
    call_id: str
    round_index: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "executed"
    reason: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class WorkspacePerceptionArtifact:
    perception_id: str
    workspace_root: str
    trigger: str
    created_at: str
    query: str = ""
    queries: list[WorkspacePerceptionQuery] = field(default_factory=list)
    tool_calls: list[WorkspaceToolCall] = field(default_factory=list)
    blocked_tool_calls: list[WorkspaceToolCall] = field(default_factory=list)
    files_read: list[WorkspaceFileRead] = field(default_factory=list)
    files_skipped: list[WorkspaceFileSkipped] = field(default_factory=list)
    facts: list[WorkspaceFact] = field(default_factory=list)
    snippets: list[WorkspaceSnippet] = field(default_factory=list)
    summary: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    limits: WorkspacePerceptionLimits = field(default_factory=WorkspacePerceptionLimits)
    exploration_status: str = ""
    depth: str = ""
    budget_used: dict[str, Any] = field(default_factory=dict)
    budget_pressure_events: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    schema_version: str = WORKSPACE_PERCEPTION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WorkspacePerceptionArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("Workspace perception payload must be a mapping.")
        limits = payload.get("limits")
        return cls(
            perception_id=_required_string(payload, "perception_id"),
            workspace_root=str(payload.get("workspace_root") or ""),
            trigger=str(payload.get("trigger") or "command"),
            created_at=str(payload.get("created_at") or ""),
            query=str(payload.get("query") or ""),
            queries=[
                safe_dataclass_from_payload(WorkspacePerceptionQuery, item)
                for item in _mappings(payload.get("queries"))
            ],
            tool_calls=[
                safe_dataclass_from_payload(WorkspaceToolCall, item)
                for item in _mappings(payload.get("tool_calls"))
            ],
            blocked_tool_calls=[
                safe_dataclass_from_payload(WorkspaceToolCall, item)
                for item in _mappings(payload.get("blocked_tool_calls"))
            ],
            files_read=[
                safe_dataclass_from_payload(WorkspaceFileRead, item)
                for item in _mappings(payload.get("files_read"))
            ],
            files_skipped=[
                safe_dataclass_from_payload(WorkspaceFileSkipped, item)
                for item in _mappings(payload.get("files_skipped"))
            ],
            facts=[
                safe_dataclass_from_payload(WorkspaceFact, item)
                for item in _mappings(payload.get("facts"))
            ],
            snippets=[
                safe_dataclass_from_payload(WorkspaceSnippet, item)
                for item in _mappings(payload.get("snippets"))
            ],
            summary=str(payload.get("summary") or ""),
            budget=dict(payload.get("budget"))
            if isinstance(payload.get("budget"), dict)
            else {},
            limits=(
                safe_dataclass_from_payload(WorkspacePerceptionLimits, limits)
                if isinstance(limits, Mapping)
                else WorkspacePerceptionLimits()
            ),
            exploration_status=str(payload.get("exploration_status") or ""),
            depth=str(payload.get("depth") or ""),
            budget_used=dict(payload.get("budget_used"))
            if isinstance(payload.get("budget_used"), dict)
            else {},
            budget_pressure_events=_mappings(payload.get("budget_pressure_events")),
            limitations=_strings(payload.get("limitations")),
            schema_version=str(
                payload.get("schema_version") or WORKSPACE_PERCEPTION_SCHEMA_VERSION
            ),
            metadata=dict(payload.get("metadata"))
            if isinstance(payload.get("metadata"), dict)
            else {},
        )


def build_workspace_perception_artifact(
    *,
    workspace_root: str | Path,
    trigger: str,
    query: str = "",
    queries: list[WorkspacePerceptionQuery | Mapping[str, Any]] | None = None,
    tool_calls: list[WorkspaceToolCall | Mapping[str, Any]] | None = None,
    blocked_tool_calls: list[WorkspaceToolCall | Mapping[str, Any]] | None = None,
    files_read: list[WorkspaceFileRead | Mapping[str, Any]] | None = None,
    files_skipped: list[WorkspaceFileSkipped | Mapping[str, Any]] | None = None,
    facts: list[WorkspaceFact | Mapping[str, Any]] | None = None,
    snippets: list[WorkspaceSnippet | Mapping[str, Any]] | None = None,
    summary: str = "",
    budget: dict[str, Any] | None = None,
    limits: WorkspacePerceptionLimits | Mapping[str, Any] | None = None,
    exploration_status: str = "",
    depth: str = "",
    budget_used: dict[str, Any] | None = None,
    budget_pressure_events: list[dict[str, Any]] | None = None,
    limitations: list[str] | None = None,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkspacePerceptionArtifact:
    created = _timestamp(created_at)
    root = str(Path(workspace_root))
    normalized_queries = [_query(item) for item in queries or []]
    normalized_query = str(query or "").strip()
    if not normalized_query and normalized_queries:
        normalized_query = normalized_queries[0].query
    perception_id = _workspace_perception_id(
        workspace_root=root,
        trigger=trigger,
        created_at=created,
        summary=summary or normalized_query,
    )
    return WorkspacePerceptionArtifact(
        perception_id=perception_id,
        workspace_root=root,
        trigger=str(trigger or "command"),
        created_at=created,
        query=normalized_query,
        queries=normalized_queries,
        tool_calls=[_tool_call(item) for item in tool_calls or []],
        blocked_tool_calls=[_tool_call(item) for item in blocked_tool_calls or []],
        files_read=[_file_read(item) for item in files_read or []],
        files_skipped=[_file_skipped(item) for item in files_skipped or []],
        facts=[_fact(item) for item in facts or []],
        snippets=[_snippet(item) for item in snippets or []],
        summary=str(summary or ""),
        budget=dict(budget or {}),
        limits=_limits(limits),
        exploration_status=str(exploration_status or ""),
        depth=str(depth or ""),
        budget_used=dict(budget_used or {}),
        budget_pressure_events=_mappings(budget_pressure_events),
        limitations=_strings(limitations),
        metadata=dict(metadata or {}),
    )


def build_workspace_perception_artifact_from_loop(
    *,
    workspace_root: str | Path,
    trigger: str,
    loop_result: Any,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkspacePerceptionArtifact:
    """Build the auditable workspace perception artifact from a controlled loop.

    The artifact records all tool calls, but tool call results are compacted so
    read content lives in snippets instead of being duplicated as raw file bodies.
    """

    loop_payload = _payload(loop_result)
    query = str(loop_payload.get("query") or "")
    tool_call_payloads = [
        _compact_tool_call_payload(item)
        for item in _mappings(loop_payload.get("tool_calls"))
    ]
    round_batches = _compact_round_batches(loop_payload.get("round_batches"))
    blocked_payloads = [
        item for item in tool_call_payloads if str(item.get("status") or "") == "blocked"
    ]
    snippets = _snippets_from_tool_calls(tool_call_payloads)
    files_read = _files_read_from_tool_calls(tool_call_payloads)
    files_skipped = _files_skipped_from_tool_calls(tool_call_payloads)
    facts = _facts_from_tool_calls(tool_call_payloads, snippets)
    summary = str(loop_payload.get("summary") or "").strip()
    investigation_state = (
        dict(loop_payload.get("investigation_state"))
        if isinstance(loop_payload.get("investigation_state"), Mapping)
        else {}
    )
    sufficiency_check = (
        dict(loop_payload.get("sufficiency_check"))
        if isinstance(loop_payload.get("sufficiency_check"), Mapping)
        else {}
    )
    if not summary:
        summary = str(investigation_state.get("investigation_summary") or "").strip()
    if not summary:
        summary = _default_loop_summary(files_read=files_read, snippets=snippets, tool_calls=tool_call_payloads)
    budget = dict(loop_payload.get("budget")) if isinstance(loop_payload.get("budget"), dict) else {}
    if "rounds_used" not in budget and loop_payload.get("rounds_used") is not None:
        budget["rounds_used"] = loop_payload.get("rounds_used")
    limits = _limits_from_loop_budget(budget)
    exploration_status = _normalize_exploration_status(
        loop_payload.get("exploration_status"),
        tool_calls=tool_call_payloads,
    )
    budget_used = _workspace_budget_used(budget, limits=limits)
    budget_pressure_events = _mappings(budget.get("budget_pressure_events"))
    limitations = _workspace_exploration_limitations(
        exploration_status=exploration_status,
        sufficiency_check=sufficiency_check,
        budget=budget,
        files_skipped=files_skipped,
        blocked_tool_calls=blocked_payloads,
    )
    depth = str(_mapping(budget.get("limits")).get("depth") or "")
    loop_metadata = {
        "exploration_status": exploration_status,
        "depth": depth,
        "budget_used": budget_used,
        "budget_pressure_events": budget_pressure_events,
        "limitations": limitations,
        "round_batches": round_batches,
        "loop": {
            "schema_version": str(loop_payload.get("schema_version") or ""),
            "done": bool(loop_payload.get("done")),
            "rounds_used": loop_payload.get("rounds_used"),
            "exploration_status": exploration_status,
            "depth": depth,
            "budget_used": budget_used,
            "budget_pressure_events": budget_pressure_events,
            "limitations": limitations,
            "round_batches": round_batches,
            "sufficiency_check": sufficiency_check,
            "investigation_state": investigation_state,
        }
    }
    loop_metadata.update(dict(metadata or {}))
    return build_workspace_perception_artifact(
        workspace_root=workspace_root,
        trigger=trigger,
        query=query,
        queries=[WorkspacePerceptionQuery(query=query)] if query else [],
        tool_calls=tool_call_payloads,
        blocked_tool_calls=blocked_payloads,
        files_read=files_read,
        files_skipped=files_skipped,
        facts=facts,
        snippets=snippets,
        summary=summary,
        budget=budget,
        limits=limits,
        exploration_status=exploration_status,
        depth=depth,
        budget_used=budget_used,
        budget_pressure_events=budget_pressure_events,
        limitations=limitations,
        created_at=created_at,
        metadata=loop_metadata,
    )


def workspace_context_from_perception(
    artifact: WorkspacePerceptionArtifact | Mapping[str, Any],
) -> dict[str, Any]:
    payload = artifact.to_payload() if isinstance(artifact, WorkspacePerceptionArtifact) else dict(artifact)
    context = {
        "schema_version": WORKSPACE_CONTEXT_SCHEMA_VERSION,
        "source": "workspace_perception",
        "perception_id": str(payload.get("perception_id") or ""),
        "workspace_root": str(payload.get("workspace_root") or ""),
        "trigger": str(payload.get("trigger") or ""),
        "query": str(payload.get("query") or ""),
        "queries": list(payload.get("queries")) if isinstance(payload.get("queries"), list) else [],
        "files_read": list(payload.get("files_read")) if isinstance(payload.get("files_read"), list) else [],
        "files_skipped": (
            list(payload.get("files_skipped"))
            if isinstance(payload.get("files_skipped"), list)
            else []
        ),
        "facts": list(payload.get("facts")) if isinstance(payload.get("facts"), list) else [],
        "snippets": list(payload.get("snippets")) if isinstance(payload.get("snippets"), list) else [],
        "summary": str(payload.get("summary") or ""),
        "limits": dict(payload.get("limits")) if isinstance(payload.get("limits"), dict) else {},
    }
    metadata = dict(payload.get("metadata")) if isinstance(payload.get("metadata"), Mapping) else {}
    loop_metadata = _mapping(metadata.get("loop"))
    exploration_status = str(
        payload.get("exploration_status")
        or metadata.get("exploration_status")
        or loop_metadata.get("exploration_status")
        or ""
    )
    context["exploration_status"] = exploration_status
    context["depth"] = str(payload.get("depth") or metadata.get("depth") or loop_metadata.get("depth") or "")
    context["budget_used"] = (
        dict(payload.get("budget_used"))
        if isinstance(payload.get("budget_used"), Mapping)
        else
        dict(metadata.get("budget_used"))
        if isinstance(metadata.get("budget_used"), Mapping)
        else dict(loop_metadata.get("budget_used"))
        if isinstance(loop_metadata.get("budget_used"), Mapping)
        else {}
    )
    context["budget_pressure_events"] = (
        _mappings(payload.get("budget_pressure_events"))
        or _mappings(metadata.get("budget_pressure_events"))
        or _mappings(loop_metadata.get("budget_pressure_events"))
    )
    context["sufficiency_check"] = (
        dict(payload.get("sufficiency_check"))
        if isinstance(payload.get("sufficiency_check"), Mapping)
        else dict(metadata.get("sufficiency_check"))
        if isinstance(metadata.get("sufficiency_check"), Mapping)
        else dict(loop_metadata.get("sufficiency_check"))
        if isinstance(loop_metadata.get("sufficiency_check"), Mapping)
        else {}
    )
    context["limitations"] = _workspace_context_limitations(
        payload=payload,
        metadata=metadata,
        loop_metadata=loop_metadata,
        files_skipped=context["files_skipped"],
    )
    workspace_cache = metadata.get("workspace_summary_cache")
    if isinstance(workspace_cache, Mapping):
        context["workspace_summary_cache"] = dict(workspace_cache)
    return context


def _workspace_perception_id(
    *,
    workspace_root: str,
    trigger: str,
    created_at: str,
    summary: str,
) -> str:
    digest = sha256(
        "\n".join([workspace_root, trigger, created_at, summary]).encode("utf-8")
    ).hexdigest()[:12]
    compact_time = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"workspace.{compact_time}.{digest}"


def _timestamp(value: datetime | None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _query(value: WorkspacePerceptionQuery | Mapping[str, Any]) -> WorkspacePerceptionQuery:
    if isinstance(value, WorkspacePerceptionQuery):
        return value
    return safe_dataclass_from_payload(WorkspacePerceptionQuery, value)


def _file_read(value: WorkspaceFileRead | Mapping[str, Any]) -> WorkspaceFileRead:
    if isinstance(value, WorkspaceFileRead):
        return value
    return safe_dataclass_from_payload(WorkspaceFileRead, value)


def _file_skipped(value: WorkspaceFileSkipped | Mapping[str, Any]) -> WorkspaceFileSkipped:
    if isinstance(value, WorkspaceFileSkipped):
        return value
    return safe_dataclass_from_payload(WorkspaceFileSkipped, value)


def _fact(value: WorkspaceFact | Mapping[str, Any]) -> WorkspaceFact:
    if isinstance(value, WorkspaceFact):
        return value
    return safe_dataclass_from_payload(WorkspaceFact, value)


def _snippet(value: WorkspaceSnippet | Mapping[str, Any]) -> WorkspaceSnippet:
    if isinstance(value, WorkspaceSnippet):
        return value
    return safe_dataclass_from_payload(WorkspaceSnippet, value)


def _tool_call(value: WorkspaceToolCall | Mapping[str, Any]) -> WorkspaceToolCall:
    if isinstance(value, WorkspaceToolCall):
        return value
    return safe_dataclass_from_payload(WorkspaceToolCall, value)


def _limits(value: WorkspacePerceptionLimits | Mapping[str, Any] | None) -> WorkspacePerceptionLimits:
    if isinstance(value, WorkspacePerceptionLimits):
        return value
    if isinstance(value, Mapping):
        return safe_dataclass_from_payload(WorkspacePerceptionLimits, value)
    return WorkspacePerceptionLimits()


def _mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workspace perception payload missing required string: {key}")
    return value


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _compact_tool_call_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["args"] = dict(payload.get("args")) if isinstance(payload.get("args"), Mapping) else {}
    payload["result"] = _compact_tool_result(
        dict(payload.get("result")) if isinstance(payload.get("result"), Mapping) else {}
    )
    return payload


def _compact_round_batches(value: Any) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for item in _mappings(value):
        requested = [
            {
                "tool": str(call.get("tool") or call.get("name") or ""),
                "args": dict(call.get("args")) if isinstance(call.get("args"), Mapping) else {},
            }
            for call in _mappings(item.get("requested_tool_calls"))
        ]
        batch = {
            "round": item.get("round"),
            "requested_tool_calls": requested,
            "executed_tool_calls": [
                _compact_tool_call_payload(call)
                for call in _mappings(item.get("executed_tool_calls"))
            ],
            "blocked_tool_calls": [
                _compact_tool_call_payload(call)
                for call in _mappings(item.get("blocked_tool_calls"))
            ],
            "failed_tool_calls": [
                _compact_tool_call_payload(call)
                for call in _mappings(item.get("failed_tool_calls"))
            ],
        }
        if bool(item.get("minimum_investigation_fallback")):
            batch["minimum_investigation_fallback"] = True
            reason = str(item.get("fallback_reason") or "").strip()
            if reason:
                batch["fallback_reason"] = reason
        batches.append(batch)
    return batches


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    if "content" in compact:
        content = str(compact.pop("content") or "")
        compact["content_preview"] = _shorten(content, 1200)
        compact["content_omitted"] = len(content) > 1200
    if isinstance(compact.get("matches"), list):
        compact["matches"] = [dict(item) for item in compact["matches"][:30] if isinstance(item, Mapping)]
    if isinstance(compact.get("entries"), list):
        compact["entries"] = [dict(item) for item in compact["entries"][:80] if isinstance(item, Mapping)]
    if isinstance(compact.get("test_files"), list):
        compact["test_files"] = [
            dict(item) for item in compact["test_files"][:80] if isinstance(item, Mapping)
        ]
    if isinstance(compact.get("test_dirs"), list):
        compact["test_dirs"] = list(compact["test_dirs"][:80])
    if isinstance(compact.get("framework_hints"), list):
        compact["framework_hints"] = list(compact["framework_hints"][:40])
    if isinstance(compact.get("files"), list):
        compact["files"] = [dict(item) for item in compact["files"][:40] if isinstance(item, Mapping)]
    if isinstance(compact.get("files_read"), list):
        compact["files_read"] = [
            dict(item) for item in compact["files_read"][:40] if isinstance(item, Mapping)
        ]
    if isinstance(compact.get("symbols"), list):
        compact["symbols"] = [
            dict(item) for item in compact["symbols"][:80] if isinstance(item, Mapping)
        ]
    if isinstance(compact.get("imports"), list):
        compact["imports"] = [
            dict(item) for item in compact["imports"][:80] if isinstance(item, Mapping)
        ]
    if isinstance(compact.get("modules"), list):
        compact["modules"] = [
            dict(item) for item in compact["modules"][:60] if isinstance(item, Mapping)
        ]
    if isinstance(compact.get("files_skipped"), list):
        compact["files_skipped"] = [
            dict(item) for item in compact["files_skipped"][:30] if isinstance(item, Mapping)
        ]
    return compact


def _snippets_from_tool_calls(tool_calls: list[Mapping[str, Any]]) -> list[WorkspaceSnippet]:
    snippets: list[WorkspaceSnippet] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for call in tool_calls:
        if str(call.get("status") or "") != "executed":
            continue
        tool = str(call.get("tool") or "")
        result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
        if tool == "search":
            for match in _mappings(result.get("matches")):
                snippet = WorkspaceSnippet(
                    path=str(match.get("path") or ""),
                    text=_shorten(str(match.get("line") or ""), 800),
                    line_start=_optional_int(match.get("line_number")),
                    line_end=_optional_int(match.get("line_number")),
                    content_hash=str(match.get("content_hash") or ""),
                    source="search",
                    metadata={"call_id": str(call.get("call_id") or "")},
                )
                _append_unique_snippet(snippets, seen, snippet)
        elif tool == "read_file":
            text = str(result.get("content_preview") or "")
            if not text:
                continue
            snippet = WorkspaceSnippet(
                path=str(result.get("path") or ""),
                text=_shorten(text, 1500),
                line_start=_optional_int(result.get("line_start")),
                line_end=_optional_int(result.get("line_end")),
                content_hash=str(result.get("content_hash") or ""),
                source="read_file",
                metadata={
                    "call_id": str(call.get("call_id") or ""),
                    "truncated": bool(result.get("truncated")),
                    "content_omitted": bool(result.get("content_omitted")),
                },
            )
            _append_unique_snippet(snippets, seen, snippet)
        elif tool == "read_python_symbol":
            text = str(result.get("content_preview") or "")
            if not text:
                continue
            snippet = WorkspaceSnippet(
                path=str(result.get("path") or ""),
                text=_shorten(text, 1500),
                line_start=_optional_int(result.get("line_start")),
                line_end=_optional_int(result.get("line_end")),
                content_hash=str(result.get("content_hash") or ""),
                source="read_python_symbol",
                metadata={
                    "call_id": str(call.get("call_id") or ""),
                    "qualified_name": str(result.get("qualified_name") or ""),
                    "kind": str(result.get("kind") or ""),
                    "truncated": bool(result.get("truncated")),
                    "content_omitted": bool(result.get("content_omitted")),
                },
            )
            _append_unique_snippet(snippets, seen, snippet)
        elif tool == "git_diff":
            text = str(result.get("content_preview") or "")
            if not text:
                continue
            snippet = WorkspaceSnippet(
                path=str(result.get("path") or "git.diff"),
                text=_shorten(text, 1500),
                content_hash=str(result.get("content_hash") or ""),
                source="git_diff",
                metadata={
                    "call_id": str(call.get("call_id") or ""),
                    "mode": str(result.get("mode") or ""),
                    "truncated": bool(result.get("truncated")),
                    "content_omitted": bool(result.get("content_omitted")),
                },
            )
            _append_unique_snippet(snippets, seen, snippet)
    return snippets[:24]


def _append_unique_snippet(
    snippets: list[WorkspaceSnippet],
    seen: set[tuple[str, int | None, int | None, str]],
    snippet: WorkspaceSnippet,
) -> None:
    key = (snippet.path, snippet.line_start, snippet.line_end, snippet.content_hash or snippet.text)
    if snippet.path and snippet.text and key not in seen:
        snippets.append(snippet)
        seen.add(key)


def _files_read_from_tool_calls(tool_calls: list[Mapping[str, Any]]) -> list[WorkspaceFileRead]:
    files: list[WorkspaceFileRead] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for call in tool_calls:
        if str(call.get("status") or "") != "executed":
            continue
        if str(call.get("tool") or "") == "read_package_metadata":
            result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
            for item in _mappings(result.get("files_read")):
                path = str(item.get("path") or "")
                if not path:
                    continue
                file_read = WorkspaceFileRead(
                    path=path,
                    chars_read=int(item.get("chars_read") or 0),
                    line_start=_optional_int(item.get("line_start")),
                    line_end=_optional_int(item.get("line_end")),
                    truncated=bool(item.get("truncated")),
                    content_hash=str(item.get("content_hash") or ""),
                    reason=str(item.get("reason") or ""),
                    metadata={"call_id": str(call.get("call_id") or ""), "source": "read_package_metadata"},
                )
                key = (file_read.path, file_read.line_start, file_read.line_end)
                if key not in seen:
                    files.append(file_read)
                    seen.add(key)
            continue
        if str(call.get("tool") or "") not in {"read_file", "read_python_symbol"}:
            continue
        result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
        path = str(result.get("path") or "")
        if not path:
            continue
        file_read = WorkspaceFileRead(
            path=path,
            chars_read=int(result.get("chars_read") or 0),
            line_start=_optional_int(result.get("line_start")),
            line_end=_optional_int(result.get("line_end")),
            truncated=bool(result.get("truncated")),
            content_hash=str(result.get("content_hash") or ""),
            reason=str(result.get("reason") or ""),
            metadata={
                "call_id": str(call.get("call_id") or ""),
                "source": str(call.get("tool") or ""),
                "qualified_name": str(result.get("qualified_name") or ""),
            },
        )
        key = (file_read.path, file_read.line_start, file_read.line_end)
        if key not in seen:
            files.append(file_read)
            seen.add(key)
    return files


def _files_skipped_from_tool_calls(tool_calls: list[Mapping[str, Any]]) -> list[WorkspaceFileSkipped]:
    skipped: list[WorkspaceFileSkipped] = []
    seen: set[tuple[str, str]] = set()
    for call in tool_calls:
        result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
        local_items = _mappings(result.get("files_skipped"))
        if str(call.get("status") or "") == "blocked" and not local_items:
            path = str(_mapping(call.get("args")).get("path") or result.get("path") or "")
            reason = str(call.get("reason") or result.get("reason") or "blocked")
            if path:
                local_items = [{"path": path, "reason": reason}]
        for item in local_items:
            value = WorkspaceFileSkipped(
                path=str(item.get("path") or ""),
                reason=str(item.get("reason") or call.get("reason") or "skipped"),
                metadata=dict(item.get("metadata")) if isinstance(item.get("metadata"), dict) else {},
            )
            key = (value.path, value.reason)
            if value.path and key not in seen:
                skipped.append(value)
                seen.add(key)
    return skipped


def _facts_from_tool_calls(
    tool_calls: list[Mapping[str, Any]],
    snippets: list[WorkspaceSnippet],
) -> list[WorkspaceFact]:
    facts: list[WorkspaceFact] = []
    for snippet in snippets[:12]:
        if snippet.source == "search":
            text = f"Search matched {snippet.path}"
            if snippet.line_start:
                text += f":{snippet.line_start}"
            if snippet.text:
                text += f" - {snippet.text}"
            facts.append(
                WorkspaceFact(
                    text=_shorten(text, 420),
                    source_path=snippet.path,
                    line_start=snippet.line_start,
                    line_end=snippet.line_end,
                    confidence=0.75,
                    metadata={"source": "search"},
                )
            )
        elif snippet.source == "read_file":
            line_range = ""
            if snippet.line_start is not None and snippet.line_end is not None:
                line_range = f" lines {snippet.line_start}-{snippet.line_end}"
            facts.append(
                WorkspaceFact(
                    text=f"Read {snippet.path}{line_range}.",
                    source_path=snippet.path,
                    line_start=snippet.line_start,
                    line_end=snippet.line_end,
                    confidence=0.8,
                    metadata={"source": "read_file"},
                )
            )
    for call in tool_calls:
        if str(call.get("status") or "") != "executed" or str(call.get("tool") or "") != "git_status":
            tool = str(call.get("tool") or "")
            if str(call.get("status") or "") != "executed":
                continue
            result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
            if tool == "git_diff":
                path = str(result.get("path") or "")
                mode = str(result.get("mode") or "stat")
                chars = int(result.get("chars_read") or 0)
                facts.append(
                    WorkspaceFact(
                        text=f"Git diff {mode} inspected{f' for {path}' if path else ''}; chars={chars}.",
                        source_path=path,
                        confidence=0.75,
                        metadata={"source": "git_diff", "mode": mode},
                    )
                )
            elif tool == "git_log":
                entries = _mappings(result.get("entries"))
                latest = str(entries[0].get("subject") or "") if entries else ""
                facts.append(
                    WorkspaceFact(
                        text=f"Git log inspected {len(entries)} commits{f'; latest: {latest}' if latest else ''}.",
                        confidence=0.7,
                        metadata={"source": "git_log"},
                    )
                )
            elif tool == "repo_map":
                entries = _mappings(result.get("entries"))
                dirs = len([item for item in entries if str(item.get("kind") or "") == "dir"])
                files_count = len([item for item in entries if str(item.get("kind") or "") == "file"])
                facts.append(
                    WorkspaceFact(
                        text=f"Repo map inspected {dirs} directories and {files_count} files under {result.get('path') or '.'}.",
                        confidence=0.75,
                        metadata={"source": "repo_map"},
                    )
                )
            elif tool == "read_package_metadata":
                files_payload = _mappings(result.get("files"))
                names = [str(item.get("name") or item.get("path") or "") for item in files_payload[:5]]
                facts.append(
                    WorkspaceFact(
                        text=f"Package metadata inspected {len(files_payload)} files{f': {', '.join(name for name in names if name)}' if names else ''}.",
                        confidence=0.78,
                        metadata={"source": "read_package_metadata"},
                    )
                )
            elif tool == "read_test_structure":
                test_files = _mappings(result.get("test_files"))
                hints = [str(item) for item in result.get("framework_hints", []) if item]
                facts.append(
                    WorkspaceFact(
                        text=f"Test structure inspected {len(test_files)} test files{f'; frameworks: {', '.join(hints)}' if hints else ''}.",
                        confidence=0.78,
                        metadata={"source": "read_test_structure"},
                    )
                )
            elif tool == "python_symbol_index":
                modules = _mappings(result.get("modules"))
                symbols = _mappings(result.get("symbols"))
                imports = _mappings(result.get("imports"))
                module_names = [
                    str(item.get("module") or item.get("path") or "")
                    for item in modules[:5]
                ]
                symbol_names = [
                    str(item.get("qualified_name") or item.get("name") or "")
                    for item in symbols[:8]
                ]
                facts.append(
                    WorkspaceFact(
                        text=(
                            f"Python symbol index inspected {len(modules)} modules, "
                            f"{len(symbols)} symbols, and {len(imports)} imports"
                            + (f"; modules: {', '.join(name for name in module_names if name)}" if module_names else "")
                            + (f"; symbols: {', '.join(name for name in symbol_names if name)}" if symbol_names else "")
                            + "."
                        ),
                        confidence=0.8,
                        metadata={"source": "python_symbol_index"},
                    )
                )
            elif tool == "read_python_symbol":
                qualified_name = str(result.get("qualified_name") or result.get("name") or "")
                path = str(result.get("path") or "")
                line_start = _optional_int(result.get("line_start"))
                line_end = _optional_int(result.get("line_end"))
                line_range = ""
                if line_start and line_end:
                    line_range = f" lines {line_start}-{line_end}"
                facts.append(
                    WorkspaceFact(
                        text=f"Read Python symbol {qualified_name} in {path}{line_range}.",
                        source_path=path,
                        line_start=line_start,
                        line_end=line_end,
                        confidence=0.82,
                        metadata={
                            "source": "read_python_symbol",
                            "qualified_name": qualified_name,
                            "kind": str(result.get("kind") or ""),
                        },
                    )
                )
            continue
        result = dict(call.get("result")) if isinstance(call.get("result"), Mapping) else {}
        entries = _mappings(result.get("entries"))
        branch = str(result.get("branch") or "")
        facts.append(
            WorkspaceFact(
                text=f"Git status branch={branch or 'unknown'} changed_entries={len(entries)}.",
                confidence=0.75,
                metadata={"source": "git_status"},
            )
        )
    return facts[:24]


def _default_loop_summary(
    *,
    files_read: list[WorkspaceFileRead],
    snippets: list[WorkspaceSnippet],
    tool_calls: list[Mapping[str, Any]],
) -> str:
    executed = len([call for call in tool_calls if str(call.get("status") or "") == "executed"])
    blocked = len([call for call in tool_calls if str(call.get("status") or "") == "blocked"])
    return (
        f"Workspace perception executed {executed} read-only tool calls, "
        f"read {len(files_read)} files, recorded {len(snippets)} snippets"
        + (f", and blocked {blocked} unsafe or invalid calls." if blocked else ".")
    )


def _limits_from_loop_budget(budget: Mapping[str, Any]) -> WorkspacePerceptionLimits:
    limits = _mapping(budget.get("limits"))
    return WorkspacePerceptionLimits(
        max_files=_optional_int(limits.get("max_files_read")) or DEFAULT_WORKSPACE_PERCEPTION_MAX_FILES,
        max_chars_per_file=(
            _optional_int(limits.get("max_chars_per_file"))
            or DEFAULT_WORKSPACE_PERCEPTION_MAX_CHARS_PER_FILE
        ),
        total_char_budget=(
            _optional_int(limits.get("total_char_budget"))
            or DEFAULT_WORKSPACE_PERCEPTION_TOTAL_CHAR_BUDGET
        ),
    )


def _normalize_exploration_status(
    value: Any,
    *,
    tool_calls: list[Mapping[str, Any]],
) -> str:
    status = str(value or "").strip().lower()
    allowed = {
        WORKSPACE_EXPLORATION_COMPLETE,
        WORKSPACE_EXPLORATION_PARTIAL,
        WORKSPACE_EXPLORATION_BUDGET_EXHAUSTED,
        WORKSPACE_EXPLORATION_BLOCKED,
        WORKSPACE_EXPLORATION_FAILED,
    }
    if status in allowed:
        return status
    executed = any(str(call.get("status") or "") == "executed" for call in tool_calls)
    blocked = any(str(call.get("status") or "") == "blocked" for call in tool_calls)
    failed = any(str(call.get("status") or "") == "failed" for call in tool_calls)
    if failed and not executed:
        return WORKSPACE_EXPLORATION_FAILED
    if blocked and not executed:
        return WORKSPACE_EXPLORATION_BLOCKED
    if executed:
        return WORKSPACE_EXPLORATION_PARTIAL
    return ""


def _workspace_budget_used(
    budget: Mapping[str, Any],
    *,
    limits: WorkspacePerceptionLimits,
) -> dict[str, Any]:
    state = _mapping(budget.get("budget_state"))
    inspector = _mapping(budget.get("inspector"))
    limit_payload = _mapping(budget.get("limits"))
    chars_used = (
        _optional_int(state.get("chars_used"))
        or _optional_int(inspector.get("chars_used"))
        or 0
    )
    files_read_count = (
        _optional_int(state.get("files_read_count"))
        or _optional_int(inspector.get("files_read_count"))
        or 0
    )
    total_char_budget = (
        _optional_int(limit_payload.get("total_char_budget"))
        or limits.total_char_budget
    )
    return {
        "depth": str(limit_payload.get("depth") or ""),
        "rounds_used": _optional_int(budget.get("rounds_used")) or _optional_int(state.get("round_index")) or 0,
        "tool_calls_recorded": _optional_int(budget.get("tool_calls_recorded")) or 0,
        "tool_calls_executed": _optional_int(budget.get("tool_calls_executed")) or 0,
        "tool_calls_blocked": _optional_int(budget.get("tool_calls_blocked")) or 0,
        "files_read_count": files_read_count,
        "max_files_read": _optional_int(limit_payload.get("max_files_read")) or limits.max_files,
        "chars_used": chars_used,
        "total_char_budget": total_char_budget,
        "budget_pressure": str(budget.get("budget_pressure") or state.get("budget_pressure") or ""),
        "blocked_budget_exhausted": bool(budget.get("blocked_budget_exhausted")),
    }


def _workspace_exploration_limitations(
    *,
    exploration_status: str,
    sufficiency_check: Mapping[str, Any],
    budget: Mapping[str, Any],
    files_skipped: list[WorkspaceFileSkipped],
    blocked_tool_calls: list[Mapping[str, Any]],
) -> list[str]:
    limitations: list[str] = []
    if exploration_status == WORKSPACE_EXPLORATION_PARTIAL:
        limitations.append("Workspace perception was partial; do not treat the repo evidence as exhaustive.")
    elif exploration_status == WORKSPACE_EXPLORATION_BUDGET_EXHAUSTED:
        limitations.append("Workspace perception stopped because the investigation budget was exhausted.")
    elif exploration_status == WORKSPACE_EXPLORATION_BLOCKED:
        limitations.append("Workspace perception was blocked before enough read-only evidence was collected.")
    elif exploration_status == WORKSPACE_EXPLORATION_FAILED:
        limitations.append("Workspace perception failed before enough read-only evidence was collected.")

    for gap in _strings(sufficiency_check.get("remaining_gaps"))[:6]:
        limitations.append(f"Remaining gap: {gap}")

    pressure = str(budget.get("budget_pressure") or "").strip().lower()
    if pressure in {"high", "exhausted"}:
        limitations.append(f"Workspace perception ended under {pressure} budget pressure.")
    if bool(budget.get("blocked_budget_exhausted")):
        limitations.append("Blocked tool-call budget was exhausted; unsafe or invalid tool requests stopped the loop.")

    skipped_reasons = [
        item.reason
        for item in files_skipped[:4]
        if item.reason
    ]
    for reason in skipped_reasons:
        limitations.append(f"Skipped file reason: {reason}")

    for call in blocked_tool_calls[:4]:
        reason = str(call.get("reason") or "blocked")
        tool = str(call.get("tool") or "tool")
        limitations.append(f"Blocked {tool}: {reason}")
    return _unique_strings(limitations)


def _workspace_context_limitations(
    *,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    loop_metadata: Mapping[str, Any],
    files_skipped: list[Any],
) -> list[str]:
    limitations = [
        *_strings(payload.get("limitations")),
        *_strings(metadata.get("limitations")),
        *_strings(loop_metadata.get("limitations")),
    ]
    for item in files_skipped[:6]:
        payload = _mapping(item)
        reason = str(payload.get("reason") or "")
        if reason:
            limitations.append(f"Skipped file reason: {reason}")
    return _unique_strings(limitations)


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value if str(item or "")] if isinstance(value, list) else []


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _shorten(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"
