from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from spice.perception import (
    URLPerceptionLimits,
    build_url_perception_artifact,
    extract_urls,
    run_url_perception,
    url_context_from_perception,
)
from spice.runtime.memory_writeback import (
    skipped_general_url_perception_memory_writeback,
    write_general_url_perception_memory,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import load_workspace_memory_provider


@dataclass(frozen=True, slots=True)
class RuntimeURLPerceptionResult:
    requested: bool
    status: str
    query: str = ""
    urls: list[str] | None = None
    context: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    path: Path | None = None
    memory_writeback: dict[str, Any] | None = None
    error: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "status": self.status,
            "query": self.query,
            "urls": list(self.urls or []),
            "context": dict(self.context or {}),
            "artifact": dict(self.artifact or {}),
            "path": str(self.path) if self.path is not None else "",
            "memory_writeback": dict(self.memory_writeback or {}),
            "error": self.error,
        }


def run_runtime_url_perception_step(
    *,
    project_root: str | Path,
    text: str = "",
    urls: list[str] | None = None,
    query: str = "",
    trigger: str,
    store: LocalJsonStore | None = None,
    now: datetime | None = None,
    persist: bool = True,
    limits: URLPerceptionLimits | None = None,
) -> RuntimeURLPerceptionResult:
    """Read explicit external URLs into a compact decision-relevant context."""

    normalized_urls = extract_urls(text)
    for url in urls or []:
        if url and url not in normalized_urls:
            normalized_urls.append(url)
    normalized_query = (query or text).strip()
    if not normalized_urls:
        return RuntimeURLPerceptionResult(
            requested=False,
            status="skipped",
            query=normalized_query,
            urls=[],
            error="no_urls_found",
        )

    active_store = store or LocalJsonStore.from_project_root(project_root)
    created = now or datetime.now(timezone.utc)
    try:
        result = run_url_perception(
            urls=normalized_urls,
            query=normalized_query,
            limits=limits or URLPerceptionLimits(),
        )
        artifact = build_url_perception_artifact(
            trigger=trigger,
            result=result,
            created_at=created,
            metadata={"runtime_step": "url_perception", "status": "read"},
        )
        status = "written" if persist else "preview"
        error = ""
    except Exception as exc:
        artifact = _skipped_url_perception_artifact(
            urls=normalized_urls,
            query=normalized_query,
            trigger=trigger,
            reason=f"url_perception_failed:{exc}",
            created_at=created,
        )
        status = "failed"
        error = str(exc)
    return _finalize_runtime_url_perception_result(
        project_root=project_root,
        store=active_store,
        artifact_payload=artifact.to_payload(),
        requested=True,
        status=status,
        query=normalized_query,
        persist=persist,
        error=error,
    )


def _skipped_url_perception_artifact(
    *,
    urls: list[str],
    query: str,
    trigger: str,
    reason: str,
    created_at: datetime,
) -> Any:
    result = run_url_perception(
        urls=[],
        query=query,
        limits=URLPerceptionLimits(max_urls=0),
    )
    payload = result.to_payload()
    payload["urls"] = list(urls)
    payload["urls_skipped"] = [
        {"url": url, "reason": reason, "metadata": {"source": "runtime_url_perception"}}
        for url in urls
    ]
    payload["budget"] = {
        "requested_url_count": len(urls),
        "normalized_url_count": len(urls),
        "document_count": 0,
        "skipped_count": len(urls),
        "total_chars_read": 0,
        "total_char_budget": 0,
        "remaining_char_budget": 0,
    }
    return build_url_perception_artifact(
        trigger=trigger,
        result=payload,
        created_at=created_at,
        metadata={
            "runtime_step": "url_perception",
            "status": "skipped",
            "reason": reason,
        },
    )


def _finalize_runtime_url_perception_result(
    *,
    project_root: str | Path,
    store: LocalJsonStore,
    artifact_payload: dict[str, Any],
    requested: bool,
    status: str,
    query: str,
    persist: bool,
    error: str = "",
) -> RuntimeURLPerceptionResult:
    memory_writeback = _write_url_perception_memory(
        project_root=project_root,
        artifact_payload=artifact_payload,
        persist=persist,
    )
    artifact_payload["memory_writeback"] = memory_writeback
    perception_id = str(artifact_payload.get("perception_id") or "")
    saved_path = store.save_perception(perception_id, artifact_payload) if persist else None
    return RuntimeURLPerceptionResult(
        requested=requested,
        status=status,
        query=query,
        urls=[str(item) for item in _list(artifact_payload.get("urls"))],
        context=url_context_from_perception(artifact_payload),
        artifact=artifact_payload,
        path=saved_path,
        memory_writeback=memory_writeback,
        error=error,
    )


def _write_url_perception_memory(
    *,
    project_root: str | Path,
    artifact_payload: dict[str, Any],
    persist: bool,
) -> dict[str, Any]:
    if not persist:
        return skipped_general_url_perception_memory_writeback(reason="not_persisted")
    try:
        provider = load_workspace_memory_provider(project_root)
    except Exception as exc:
        return skipped_general_url_perception_memory_writeback(
            reason=f"memory_provider_unavailable:{exc}"
        )
    try:
        return write_general_url_perception_memory(provider, artifact=artifact_payload)
    except Exception as exc:
        return skipped_general_url_perception_memory_writeback(reason=f"write_failed:{exc}")


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
