from __future__ import annotations

from typing import Any, Mapping


EVIDENCE_CONTEXT_SCHEMA_VERSION = "spice.evidence_context.v2"

_SOURCE_LIMIT = 24
_FINDING_LIMIT = 24


def build_evidence_context(
    *,
    requirements: Mapping[str, Any] | None = None,
    workspace_context: Mapping[str, Any] | None = None,
    url_context: Mapping[str, Any] | None = None,
    delegated_perception_context: Mapping[str, Any] | None = None,
    limitations: list[str] | None = None,
    confidence: str = "",
) -> dict[str, Any]:
    """Build a compact normalized view over all read-only evidence sources.

    This intentionally does not replace workspace/url/delegated artifacts. It is
    a small index that lets gates, composers, /sources, and memory writeback ask
    the same first question: what evidence is present and where did it come from?
    """

    workspace = _workspace_view(_mapping(workspace_context))
    url = _url_view(_mapping(url_context))
    delegated = _delegated_view(_mapping(delegated_perception_context))
    sources = _evidence_sources(
        workspace_context=_mapping(workspace_context),
        url_context=_mapping(url_context),
        delegated_context=_mapping(delegated_perception_context),
    )
    findings = _evidence_findings(
        workspace_context=_mapping(workspace_context),
        url_context=_mapping(url_context),
        delegated_context=_mapping(delegated_perception_context),
        source_ids={str(item.get("source_id") or "") for item in sources if item.get("source_id")},
    )
    all_limitations = [
        *(_strings(limitations)),
        *workspace["limitations"],
        *url["limitations"],
        *delegated["limitations"],
        *[
            limitation
            for item in findings
            for limitation in _strings(item.get("limitations"))
        ],
    ]
    return {
        "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
        "requirements": dict(requirements) if isinstance(requirements, Mapping) else {},
        "sources": sources,
        "findings": findings,
        "workspace": workspace,
        "url": url,
        "delegated": delegated,
        "limitations": _unique(all_limitations),
        "confidence": confidence or _confidence(workspace, url, delegated),
    }


def compact_evidence_context(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        "schema_version": str(payload.get("schema_version") or EVIDENCE_CONTEXT_SCHEMA_VERSION),
        "requirements": _mapping(payload.get("requirements")),
        "sources": [
            _compact_evidence_source(_mapping(item))
            for item in _list(payload.get("sources"))[:_SOURCE_LIMIT]
        ],
        "findings": [
            _compact_evidence_finding(_mapping(item))
            for item in _list(payload.get("findings"))[:_FINDING_LIMIT]
        ],
        "workspace": _compact_source_view(_mapping(payload.get("workspace"))),
        "url": _compact_source_view(_mapping(payload.get("url"))),
        "delegated": _compact_source_view(_mapping(payload.get("delegated"))),
        "limitations": _strings(payload.get("limitations"))[:12],
        "confidence": str(payload.get("confidence") or ""),
    }


def _workspace_view(context: dict[str, Any]) -> dict[str, Any]:
    files_read = _list(context.get("files_read"))
    facts = _list(context.get("facts"))
    snippets = _list(context.get("snippets"))
    perception_id = str(context.get("perception_id") or "")
    present = bool(perception_id or files_read or facts or snippets)
    source_count = _unique_count(
        [
            *(_source_path(item) for item in files_read),
            *(_source_path(item) for item in facts),
            *(_source_path(item) for item in snippets),
        ]
    )
    if present and source_count == 0:
        source_count = len(files_read) + len(facts) + len(snippets)
    return {
        "present": present,
        "source_type": "workspace",
        "observed_by": "spice",
        "perception_id": perception_id,
        "source_count": source_count,
        "fact_count": len(facts),
        "snippet_count": len(snippets),
        "summary": _shorten(str(context.get("summary") or ""), 360),
        "exploration_status": str(context.get("exploration_status") or ""),
        "depth": str(context.get("depth") or ""),
        "budget_used": _mapping(context.get("budget_used")),
        "budget_pressure_event_count": len(_list(context.get("budget_pressure_events"))),
        "limitations": _strings(context.get("limitations")),
    }


def _url_view(context: dict[str, Any]) -> dict[str, Any]:
    documents = _list(context.get("documents"))
    urls = _strings(context.get("urls"))
    facts = _list(context.get("facts"))
    snippets = _list(context.get("snippets"))
    skipped = _list(context.get("urls_skipped"))
    perception_id = str(context.get("perception_id") or "")
    present = bool(perception_id or documents or urls or facts or snippets)
    source_count = _unique_count(
        [
            *urls,
            *(_source_url(item) for item in documents),
            *(_source_url(item) for item in facts),
            *(_source_url(item) for item in snippets),
        ]
    )
    if present and source_count == 0:
        source_count = len(documents) + len(facts) + len(snippets)
    return {
        "present": present,
        "source_type": "url",
        "observed_by": "spice",
        "perception_id": perception_id,
        "source_count": source_count,
        "fact_count": len(facts),
        "snippet_count": len(snippets),
        "summary": _shorten(str(context.get("summary") or ""), 360),
        "limitations": _strings(context.get("limitations")) + [
            str(_mapping(item).get("reason") or "")
            for item in skipped[:6]
            if str(_mapping(item).get("reason") or "")
        ],
    }


def _delegated_view(context: dict[str, Any]) -> dict[str, Any]:
    findings = _list(context.get("findings"))
    sources = _list(context.get("sources"))
    limitations = _strings(context.get("limitations"))
    perception_id = str(context.get("perception_id") or "")
    executor_id = str(context.get("executor_id") or "")
    present = bool(perception_id or findings or sources)
    source_count = _unique_count(
        [
            str(_mapping(item).get("source_id") or _mapping(item).get("uri") or "")
            for item in sources
        ]
    )
    if present and source_count == 0:
        source_count = len(sources)
    return {
        "present": present,
        "source_type": "delegated",
        "observed_by": executor_id or "executor",
        "perception_id": perception_id,
        "executor_id": executor_id,
        "consent_id": str(context.get("consent_id") or ""),
        "source_count": source_count,
        "finding_count": len(findings),
        "summary": _shorten(str(context.get("summary") or ""), 360),
        "confidence": str(context.get("confidence") or ""),
        "limitations": limitations,
    }


def _compact_source_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(payload.get("present")),
        "source_type": str(payload.get("source_type") or ""),
        "observed_by": str(payload.get("observed_by") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "executor_id": str(payload.get("executor_id") or ""),
        "consent_id": str(payload.get("consent_id") or ""),
        "source_count": int(payload.get("source_count") or 0),
        "fact_count": int(payload.get("fact_count") or 0),
        "snippet_count": int(payload.get("snippet_count") or 0),
        "finding_count": int(payload.get("finding_count") or 0),
        "summary": _shorten(str(payload.get("summary") or ""), 280),
        "exploration_status": str(payload.get("exploration_status") or ""),
        "depth": str(payload.get("depth") or ""),
        "budget_used": _compact_budget_used(_mapping(payload.get("budget_used"))),
        "budget_pressure_event_count": int(payload.get("budget_pressure_event_count") or 0),
        "confidence": str(payload.get("confidence") or ""),
        "limitations": _strings(payload.get("limitations"))[:6],
    }


def _evidence_sources(
    *,
    workspace_context: dict[str, Any],
    url_context: dict[str, Any],
    delegated_context: dict[str, Any],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    workspace_perception_id = str(workspace_context.get("perception_id") or "")
    workspace_sources_by_id: dict[str, dict[str, Any]] = {}
    for item in _list(workspace_context.get("files_read")):
        payload = _mapping(item)
        path = _source_path(payload)
        if not path:
            continue
        source = _workspace_source(
            path=path,
            perception_id=workspace_perception_id,
            excerpt="",
            line_start=payload.get("line_start"),
            line_end=payload.get("line_end"),
        )
        workspace_sources_by_id[source["source_id"]] = source
    for item in [*_list(workspace_context.get("facts")), *_list(workspace_context.get("snippets"))]:
        payload = _mapping(item)
        path = _source_path(payload)
        if not path:
            continue
        source = _workspace_source(
            path=path,
            perception_id=workspace_perception_id,
            excerpt=str(payload.get("text") or ""),
            line_start=payload.get("line_start"),
            line_end=payload.get("line_end"),
        )
        existing = workspace_sources_by_id.get(source["source_id"])
        if existing and not existing.get("excerpt") and source.get("excerpt"):
            existing["excerpt"] = source["excerpt"]
            existing["line_start"] = source.get("line_start")
            existing["line_end"] = source.get("line_end")
        else:
            workspace_sources_by_id.setdefault(source["source_id"], source)
    sources.extend(workspace_sources_by_id.values())

    url_perception_id = str(url_context.get("perception_id") or "")
    url_sources_by_id: dict[str, dict[str, Any]] = {}
    for raw_url in _strings(url_context.get("urls")):
        source = _url_source(
            uri=raw_url,
            title="",
            perception_id=url_perception_id,
            excerpt="",
        )
        url_sources_by_id[source["source_id"]] = source
    for item in _list(url_context.get("documents")):
        payload = _mapping(item)
        uri = _source_url(payload)
        if not uri:
            continue
        source = _url_source(
            uri=uri,
            title=str(payload.get("title") or ""),
            perception_id=url_perception_id,
            excerpt="",
        )
        url_sources_by_id[source["source_id"]] = source
    for item in [*_list(url_context.get("facts")), *_list(url_context.get("snippets"))]:
        payload = _mapping(item)
        uri = _source_url(payload)
        if not uri:
            continue
        source = _url_source(
            uri=uri,
            title=str(payload.get("title") or ""),
            perception_id=url_perception_id,
            excerpt=str(payload.get("text") or ""),
        )
        existing = url_sources_by_id.get(source["source_id"])
        if existing:
            if not existing.get("title") and source.get("title"):
                existing["title"] = source["title"]
            if not existing.get("excerpt") and source.get("excerpt"):
                existing["excerpt"] = source["excerpt"]
        else:
            url_sources_by_id[source["source_id"]] = source
    sources.extend(url_sources_by_id.values())

    delegated_perception_id = str(delegated_context.get("perception_id") or "")
    executor_id = str(delegated_context.get("executor_id") or "")
    for item in _list(delegated_context.get("sources")):
        payload = _mapping(item)
        source_id = str(payload.get("source_id") or payload.get("uri") or payload.get("title") or "")
        if not source_id:
            continue
        sources.append(
            {
                "source_id": source_id,
                "source_type": str(payload.get("source_type") or "executor_report"),
                "title": _shorten(str(payload.get("title") or ""), 220),
                "uri": _shorten(str(payload.get("uri") or ""), 500),
                "path": "",
                "excerpt": _shorten(str(payload.get("excerpt") or ""), 700),
                "observed_by": str(payload.get("observed_by") or executor_id or "executor"),
                "perception_id": delegated_perception_id,
                "verification_status": str(payload.get("verification_status") or "reported_by_executor"),
                "accessed_at": str(payload.get("accessed_at") or ""),
            }
        )
    return _dedupe_sources(sources)[:_SOURCE_LIMIT]


def _evidence_findings(
    *,
    workspace_context: dict[str, Any],
    url_context: dict[str, Any],
    delegated_context: dict[str, Any],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    workspace_perception_id = str(workspace_context.get("perception_id") or "")
    for index, item in enumerate(_list(workspace_context.get("facts")), start=1):
        payload = _mapping(item)
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        source_ref = _workspace_source_id(_source_path(payload))
        findings.append(
            _finding_payload(
                finding_id=str(payload.get("finding_id") or f"workspace.finding.{index}"),
                text=text,
                confidence=payload.get("confidence"),
                source_refs=[source_ref] if source_ref else [],
                source_type="workspace",
                observed_by="spice",
                perception_id=workspace_perception_id,
                limitations=[],
                valid_source_ids=source_ids,
            )
        )

    url_perception_id = str(url_context.get("perception_id") or "")
    for index, item in enumerate(_list(url_context.get("facts")), start=1):
        payload = _mapping(item)
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        source_ref = _url_source_id(_source_url(payload))
        findings.append(
            _finding_payload(
                finding_id=str(payload.get("finding_id") or f"url.finding.{index}"),
                text=text,
                confidence=payload.get("confidence"),
                source_refs=[source_ref] if source_ref else [],
                source_type="url",
                observed_by="spice",
                perception_id=url_perception_id,
                limitations=[],
                valid_source_ids=source_ids,
            )
        )

    delegated_perception_id = str(delegated_context.get("perception_id") or "")
    executor_id = str(delegated_context.get("executor_id") or "")
    for index, item in enumerate(_list(delegated_context.get("findings")), start=1):
        payload = _mapping(item)
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        findings.append(
            _finding_payload(
                finding_id=str(payload.get("finding_id") or f"delegated.finding.{index}"),
                text=text,
                confidence=payload.get("confidence"),
                source_refs=_strings(payload.get("source_refs")),
                source_type="delegated",
                observed_by=executor_id or "executor",
                perception_id=delegated_perception_id,
                limitations=_strings(payload.get("limitations")),
                valid_source_ids=source_ids,
            )
        )
    return findings[:_FINDING_LIMIT]


def _workspace_source(
    *,
    path: str,
    perception_id: str,
    excerpt: str,
    line_start: Any = None,
    line_end: Any = None,
) -> dict[str, Any]:
    return {
        "source_id": _workspace_source_id(path),
        "source_type": "file",
        "title": path,
        "uri": "",
        "path": _shorten(path, 500),
        "excerpt": _shorten(excerpt, 700),
        "observed_by": "spice",
        "perception_id": perception_id,
        "verification_status": "verified_by_spice",
        "accessed_at": "",
        "line_start": line_start,
        "line_end": line_end,
    }


def _url_source(*, uri: str, title: str, perception_id: str, excerpt: str) -> dict[str, Any]:
    return {
        "source_id": _url_source_id(uri),
        "source_type": "url",
        "title": _shorten(title or uri, 220),
        "uri": _shorten(uri, 500),
        "path": "",
        "excerpt": _shorten(excerpt, 700),
        "observed_by": "spice",
        "perception_id": perception_id,
        "verification_status": "verified_by_spice",
        "accessed_at": "",
    }


def _finding_payload(
    *,
    finding_id: str,
    text: str,
    confidence: Any,
    source_refs: list[str],
    source_type: str,
    observed_by: str,
    perception_id: str,
    limitations: list[str],
    valid_source_ids: set[str],
) -> dict[str, Any]:
    valid_refs = [item for item in source_refs if item in valid_source_ids]
    missing_refs = [item for item in source_refs if item and item not in valid_source_ids]
    all_limitations = list(limitations)
    if missing_refs:
        all_limitations.append("missing_source_ref")
    if not valid_refs:
        all_limitations.append("no_source_refs")
    return {
        "finding_id": finding_id,
        "text": _shorten(text, 700),
        "confidence": confidence,
        "source_refs": valid_refs,
        "source_type": source_type,
        "observed_by": observed_by,
        "perception_id": perception_id,
        "limitations": _unique(all_limitations),
    }


def _compact_evidence_source(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(payload.get("source_id") or ""),
        "source_type": str(payload.get("source_type") or ""),
        "title": _shorten(str(payload.get("title") or ""), 180),
        "uri": _shorten(str(payload.get("uri") or ""), 360),
        "path": _shorten(str(payload.get("path") or ""), 260),
        "excerpt": _shorten(str(payload.get("excerpt") or ""), 420),
        "observed_by": str(payload.get("observed_by") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "verification_status": str(payload.get("verification_status") or ""),
        "accessed_at": str(payload.get("accessed_at") or ""),
        "line_start": payload.get("line_start"),
        "line_end": payload.get("line_end"),
    }


def _compact_evidence_finding(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": str(payload.get("finding_id") or ""),
        "text": _shorten(str(payload.get("text") or ""), 420),
        "confidence": payload.get("confidence"),
        "source_refs": _strings(payload.get("source_refs"))[:6],
        "source_type": str(payload.get("source_type") or ""),
        "observed_by": str(payload.get("observed_by") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "limitations": _strings(payload.get("limitations"))[:4],
    }


def _dedupe_sources(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        source_id = str(item.get("source_id") or "").strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        result.append(item)
    return result


def _workspace_source_id(path: str) -> str:
    return f"workspace:{path}" if path else ""


def _url_source_id(url: str) -> str:
    return f"url:{url}" if url else ""


def _confidence(workspace: Mapping[str, Any], url: Mapping[str, Any], delegated: Mapping[str, Any]) -> str:
    if delegated.get("present") and delegated.get("confidence"):
        return str(delegated.get("confidence"))
    if workspace.get("present") or url.get("present") or delegated.get("present"):
        return "medium"
    return "none"


def _source_path(value: Any) -> str:
    payload = _mapping(value)
    return str(payload.get("path") or payload.get("source_path") or "")


def _source_url(value: Any) -> str:
    payload = _mapping(value)
    return str(payload.get("url") or payload.get("final_url") or payload.get("source_url") or "")


def _compact_budget_used(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    keys = (
        "depth",
        "rounds_used",
        "tool_calls_recorded",
        "tool_calls_executed",
        "tool_calls_blocked",
        "files_read_count",
        "max_files_read",
        "chars_used",
        "total_char_budget",
        "budget_pressure",
    )
    return {key: payload[key] for key in keys if key in payload}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in _list(value) if str(item).strip()]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique_count(values: list[str]) -> int:
    return len(_unique(values))


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."
