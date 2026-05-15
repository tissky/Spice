from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from spice.llm.candidate_expander import build_candidate_expander_client
from spice.perception import (
    ControlledWorkspacePerceptionLimits,
    WorkspaceFact,
    WorkspaceInspector,
    build_workspace_perception_artifact,
    build_workspace_perception_artifact_from_loop,
    compact_workspace_summary_cache,
    load_or_refresh_workspace_summary_cache,
    run_controlled_workspace_perception_loop,
    workspace_context_from_perception,
)
from spice.runtime.memory_writeback import (
    skipped_general_workspace_perception_memory_writeback,
    write_general_workspace_perception_memory,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import load_workspace_memory_provider


@dataclass(frozen=True, slots=True)
class RuntimeWorkspacePerceptionResult:
    requested: bool
    status: str
    query: str = ""
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
            "context": dict(self.context or {}),
            "artifact": dict(self.artifact or {}),
            "path": str(self.path) if self.path is not None else "",
            "memory_writeback": dict(self.memory_writeback or {}),
            "error": self.error,
        }


def run_runtime_workspace_perception_step(
    *,
    project_root: str | Path,
    query: str,
    config: Mapping[str, Any] | None,
    trigger: str,
    store: LocalJsonStore | None = None,
    now: datetime | None = None,
    initial_context: Mapping[str, Any] | None = None,
    persist: bool = True,
    limits: ControlledWorkspacePerceptionLimits | None = None,
) -> RuntimeWorkspacePerceptionResult:
    """Run the bounded read-only workspace perception step used by runtime paths."""

    normalized_query = query.strip()
    if not normalized_query:
        return RuntimeWorkspacePerceptionResult(
            requested=False,
            status="skipped",
            error="empty_workspace_query",
        )

    payload = dict(config or {})
    provider_id = str(payload.get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(payload)
    active_store = store or LocalJsonStore.from_project_root(project_root)
    created = now or datetime.now(timezone.utc)

    if not model_id or provider_id == "deterministic":
        artifact = _skipped_workspace_perception_artifact(
            project_root=project_root,
            query=normalized_query,
            trigger=trigger,
            reason="llm_provider_not_configured",
            created_at=created,
        )
        return _finalize_runtime_workspace_perception_result(
            project_root=project_root,
            store=active_store,
            artifact_payload=artifact.to_payload(),
            requested=True,
            status="skipped",
            query=normalized_query,
            persist=persist,
            error="llm_provider_not_configured",
        )

    try:
        client = build_candidate_expander_client(provider_id=provider_id, model_id=model_id)
        cache_result = load_or_refresh_workspace_summary_cache(
            project_root=project_root,
            inspector=WorkspaceInspector(project_root),
            persist=persist,
        )
        workspace_cache_context = {
            "status": cache_result.status,
            "path": str(cache_result.path),
            **compact_workspace_summary_cache(cache_result.cache),
        }
        enriched_initial_context = dict(initial_context or {})
        enriched_initial_context["workspace_summary_cache"] = workspace_cache_context
        inspector = WorkspaceInspector(project_root)
        resolved_limits = limits or ControlledWorkspacePerceptionLimits.from_depth(
            config=payload,
            evidence_domain=_initial_context_evidence_domain(enriched_initial_context),
            answer_mode=_initial_context_answer_mode(enriched_initial_context),
            user_input=normalized_query,
            route_policy=_initial_context_route_policy(enriched_initial_context),
        )
        loop_result = run_controlled_workspace_perception_loop(
            client=client,
            inspector=inspector,
            query=normalized_query,
            limits=resolved_limits,
            initial_context=enriched_initial_context,
        )
        artifact = build_workspace_perception_artifact_from_loop(
            workspace_root=project_root,
            trigger=trigger,
            loop_result=loop_result,
            created_at=created,
            metadata={
                "runtime_step": "workspace_perception",
                "model_provider": provider_id,
                "model_id": model_id,
                "workspace_summary_cache": workspace_cache_context,
            },
        )
        return _finalize_runtime_workspace_perception_result(
            project_root=project_root,
            store=active_store,
            artifact_payload=artifact.to_payload(),
            requested=True,
            status="written" if persist else "preview",
            query=normalized_query,
            persist=persist,
        )
    except Exception as exc:
        artifact = _skipped_workspace_perception_artifact(
            project_root=project_root,
            query=normalized_query,
            trigger=trigger,
            reason=f"workspace_perception_failed:{exc}",
            created_at=created,
        )
        return _finalize_runtime_workspace_perception_result(
            project_root=project_root,
            store=active_store,
            artifact_payload=artifact.to_payload(),
            requested=True,
            status="failed",
            query=normalized_query,
            persist=persist,
            error=str(exc),
        )


def _skipped_workspace_perception_artifact(
    *,
    project_root: str | Path,
    query: str,
    trigger: str,
    reason: str,
    created_at: datetime,
) -> Any:
    return build_workspace_perception_artifact(
        workspace_root=project_root,
        trigger=trigger,
        query=query,
        facts=[
            WorkspaceFact(
                text=f"Workspace perception did not inspect files: {reason}.",
                confidence=0.0,
                metadata={"source": "runtime_workspace_perception", "reason": reason},
            )
        ],
        summary=f"Workspace perception skipped: {reason}.",
        created_at=created_at,
        metadata={
            "runtime_step": "workspace_perception",
            "status": "skipped",
            "reason": reason,
        },
    )


def _finalize_runtime_workspace_perception_result(
    *,
    project_root: str | Path,
    store: LocalJsonStore,
    artifact_payload: dict[str, Any],
    requested: bool,
    status: str,
    query: str,
    persist: bool,
    error: str = "",
) -> RuntimeWorkspacePerceptionResult:
    memory_writeback = _write_workspace_perception_memory(
        project_root=project_root,
        artifact_payload=artifact_payload,
        persist=persist,
    )
    artifact_payload["memory_writeback"] = memory_writeback
    perception_id = str(artifact_payload.get("perception_id") or "")
    saved_path = store.save_perception(perception_id, artifact_payload) if persist else None
    return RuntimeWorkspacePerceptionResult(
        requested=requested,
        status=status,
        query=query,
        context=workspace_context_from_perception(artifact_payload),
        artifact=artifact_payload,
        path=saved_path,
        memory_writeback=memory_writeback,
        error=error,
    )


def _write_workspace_perception_memory(
    *,
    project_root: str | Path,
    artifact_payload: dict[str, Any],
    persist: bool,
) -> dict[str, Any]:
    if not persist:
        return skipped_general_workspace_perception_memory_writeback(reason="not_persisted")
    try:
        provider = load_workspace_memory_provider(project_root)
    except Exception as exc:
        return skipped_general_workspace_perception_memory_writeback(
            reason=f"memory_provider_unavailable:{exc}"
        )
    try:
        return write_general_workspace_perception_memory(
            provider,
            artifact=artifact_payload,
        )
    except Exception as exc:
        return skipped_general_workspace_perception_memory_writeback(
            reason=f"write_failed:{exc}"
        )


def _runtime_model_id(config: Mapping[str, Any]) -> str:
    return str(
        config.get("llm_model")
        or config.get("model")
        or config.get("model_id")
        or ""
    ).strip()


def _initial_context_evidence_domain(initial_context: Mapping[str, Any]) -> str:
    evidence = _mapping(initial_context.get("evidence_requirement"))
    if evidence.get("evidence_domain"):
        return str(evidence.get("evidence_domain") or "")
    context = _mapping(initial_context.get("evidence_context"))
    requirements = _mapping(context.get("requirements"))
    if requirements.get("evidence_domain"):
        return str(requirements.get("evidence_domain") or "")
    policy = _mapping(initial_context.get("route_merge_policy"))
    policy_requirement = _mapping(policy.get("evidence_requirement"))
    return str(policy_requirement.get("evidence_domain") or "")


def _initial_context_answer_mode(initial_context: Mapping[str, Any]) -> str:
    evidence = _mapping(initial_context.get("evidence_requirement"))
    if evidence.get("answer_mode"):
        return str(evidence.get("answer_mode") or "")
    context = _mapping(initial_context.get("evidence_context"))
    requirements = _mapping(context.get("requirements"))
    if requirements.get("answer_mode"):
        return str(requirements.get("answer_mode") or "")
    policy = _mapping(initial_context.get("route_merge_policy"))
    policy_requirement = _mapping(policy.get("evidence_requirement"))
    return str(policy_requirement.get("answer_mode") or "")


def _initial_context_route_policy(initial_context: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "route_merge_policy",
        "escalation_decision",
        "route_policy",
        "planner_result",
        "perception_plan",
    ):
        value = initial_context.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
