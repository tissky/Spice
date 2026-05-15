from __future__ import annotations

from typing import Any

from spice.decision.general.candidates import (
    GenericExecutionIntent,
    GenericCandidate,
    crosses_execution_approval_boundary,
    is_approval_eligible_executable_candidate,
)
from spice.decision.general.permissions import (
    infer_executor_permission_requirement,
    permission_exceeds,
)
from spice.runtime.executor_runtime import (
    ResolvedExecutorRuntime,
    resolve_executor_runtime_from_config,
)
from spice.runtime.executor_capabilities import (
    ExecutorCapabilitySnapshot,
    unavailable_executor_capability_snapshot,
)
from spice.runtime.workspace import SpiceWorkspaceConfig


def annotate_execution_affordances(
    candidates: list[GenericCandidate],
    *,
    config: SpiceWorkspaceConfig | dict[str, Any],
) -> list[GenericCandidate]:
    config_payload = _config_payload(config)
    runtime = (
        resolve_executor_runtime_from_config(config)
        if isinstance(config, SpiceWorkspaceConfig)
        else resolve_executor_runtime_from_config(SpiceWorkspaceConfig.from_payload(config))
    )
    executor_capabilities = _executor_capabilities_from_config(config_payload, runtime=runtime)
    return [
        _annotate_candidate(
            candidate,
            runtime=runtime,
            executor_capabilities=executor_capabilities,
        )
        for candidate in candidates
    ]


def build_execution_affordance(
    candidate: GenericCandidate,
    *,
    executor_runtime: ResolvedExecutorRuntime,
    executor_capabilities: ExecutorCapabilitySnapshot | dict[str, Any] | None = None,
) -> dict[str, Any]:
    requirement = infer_executor_permission_requirement(candidate)
    candidate_eligible_without_capability = is_approval_eligible_executable_candidate(candidate)
    capability_check = _capability_check(
        candidate,
        executor_capabilities,
        runtime=executor_runtime,
    )
    candidate_eligible = bool(
        candidate_eligible_without_capability
        and not capability_check["blocked"]
    )
    executor_ready = executor_runtime.status == "ready"
    escalation_required = permission_exceeds(
        requirement.required_permission,
        executor_runtime.permission_mode,
    )
    escalation_supported = (
        not escalation_required
        or executor_runtime.permission_enforcement == "command_flag"
    )
    blockers = _candidate_execution_blockers(candidate)
    if candidate_eligible_without_capability:
        blockers.extend(capability_check["blockers"])
    if not executor_ready:
        blockers.append(executor_runtime.detail or f"{executor_runtime.executor_id} is not ready.")
    if escalation_required and not escalation_supported:
        blockers.append(
            f"Executor permission escalation to {requirement.required_permission} is not automated."
        )
    approval_required = bool(
        candidate_eligible_without_capability
        and (candidate.requires_confirmation or executor_runtime.approval_required)
    )
    executable = bool(candidate_eligible and executor_ready and escalation_supported)
    return {
        "schema_version": "0.1",
        "generated_by": "spice.runtime.execution_affordance",
        "candidate_executable": candidate_eligible,
        "candidate_execution_requested": candidate_eligible_without_capability,
        "executor_available": executor_ready,
        "executable": executable,
        "blocked": bool(blockers),
        "blocked_reason": blockers[0] if blockers else "",
        "blockers": blockers,
        "required_capability": capability_check["required_capability"],
        "executor_capability_source": capability_check["executor_capability_source"],
        "capability": {
            "required_capability": capability_check["required_capability"],
            "executor_has_required_capability": capability_check["executor_has_required_capability"],
            "source": capability_check["executor_capability_source"],
            "status": capability_check["executor_capability_status"],
            "available_capability_ids": capability_check["available_capability_ids"],
            "limitations": capability_check["executor_capability_limitations"],
            "simulates_required_capability": capability_check["simulates_required_capability"],
            "matched_capability": capability_check["matched_capability"],
        },
        "executor": {
            "executor_id": executor_runtime.executor_id,
            "requested_executor_id": executor_runtime.requested_executor_id,
            "transport": executor_runtime.transport,
            "status": executor_runtime.status,
            "detail": executor_runtime.detail,
            "command": executor_runtime.command,
            "command_source": executor_runtime.command_source,
            "command_found": executor_runtime.command_found,
            "command_path": executor_runtime.command_path,
            "real_executor": executor_runtime.real_executor,
            "sends_sdep_request": executor_runtime.sends_sdep_request,
        },
        "permission": {
            "required": requirement.required_permission,
            "configured": executor_runtime.permission_mode,
            "reason": requirement.reason,
            "source": requirement.source,
            "side_effect_class": requirement.side_effect_class,
            "escalation_required": escalation_required,
            "escalation_supported": escalation_supported,
            "enforcement": executor_runtime.permission_enforcement,
        },
        "approval": {
            "required": approval_required,
            "candidate_requires_confirmation": bool(candidate.requires_confirmation),
            "executor_approval_required": bool(executor_runtime.approval_required),
            "eligible_for_approval": candidate_eligible,
            "status": "approval_required_on_selection" if candidate_eligible else "not_approval_eligible",
        },
    }


def _annotate_candidate(
    candidate: GenericCandidate,
    *,
    runtime: ResolvedExecutorRuntime,
    executor_capabilities: ExecutorCapabilitySnapshot,
) -> GenericCandidate:
    metadata = dict(candidate.metadata or {})
    metadata["execution_affordance"] = build_execution_affordance(
        candidate,
        executor_runtime=runtime,
        executor_capabilities=executor_capabilities,
    )
    candidate.metadata = metadata
    return candidate


def _capability_check(
    candidate: GenericCandidate,
    executor_capabilities: ExecutorCapabilitySnapshot | dict[str, Any] | None,
    *,
    runtime: ResolvedExecutorRuntime,
) -> dict[str, Any]:
    snapshot = _coerce_capability_snapshot(executor_capabilities, runtime=runtime)
    required_capability = str(getattr(candidate, "required_capability", "") or "").strip()
    capability_ids = [str(item).strip() for item in snapshot.capability_ids if str(item).strip()]
    capability_id_set = set(capability_ids)
    execution_requested = _candidate_requests_execution(candidate)
    exact_match = bool(required_capability and required_capability in capability_id_set)
    general_match = bool(
        required_capability
        and not exact_match
        and "general_execution" in capability_id_set
    )
    simulates_required = bool(
        required_capability
        and not runtime.real_executor
        and "simulate_execution" in capability_id_set
    )
    has_required = (
        not required_capability
        or exact_match
        or general_match
        or simulates_required
    )
    blockers: list[str] = []
    if execution_requested and required_capability and not has_required:
        blockers.append(f"Executor lacks required capability: {required_capability}")
    return {
        "required_capability": required_capability,
        "executor_capability_source": snapshot.source,
        "executor_capability_status": snapshot.status,
        "available_capability_ids": capability_ids,
        "executor_capability_limitations": list(snapshot.limitations),
        "executor_has_required_capability": bool(has_required),
        "simulates_required_capability": simulates_required,
        "matched_capability": _matched_capability(
            required_capability,
            exact_match=exact_match,
            general_match=general_match,
            simulates_required=simulates_required,
        ),
        "blocked": bool(blockers),
        "blockers": blockers,
    }


def _matched_capability(
    required_capability: str,
    *,
    exact_match: bool,
    general_match: bool,
    simulates_required: bool,
) -> str:
    if not required_capability:
        return ""
    if exact_match:
        return required_capability
    if general_match:
        return "general_execution"
    if simulates_required:
        return "simulate_execution"
    return ""


def _candidate_requests_execution(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    return bool(
        execution_intent.intent_class == "execution_requested"
        and execution_intent.requested
    )


def _executor_capabilities_from_config(
    payload: dict[str, Any],
    *,
    runtime: ResolvedExecutorRuntime,
) -> ExecutorCapabilitySnapshot:
    raw = payload.get("executor_capabilities")
    return _coerce_capability_snapshot(raw, runtime=runtime)


def _coerce_capability_snapshot(
    raw: ExecutorCapabilitySnapshot | dict[str, Any] | None,
    *,
    runtime: ResolvedExecutorRuntime,
) -> ExecutorCapabilitySnapshot:
    if isinstance(raw, ExecutorCapabilitySnapshot):
        return raw
    if isinstance(raw, dict) and raw:
        try:
            return ExecutorCapabilitySnapshot.from_payload(raw)
        except (TypeError, ValueError):
            return unavailable_executor_capability_snapshot(
                runtime.executor_id,
                provider=runtime.executor_id,
                reason="Executor capability snapshot is malformed.",
            )
    return unavailable_executor_capability_snapshot(
        runtime.executor_id,
        provider=runtime.executor_id,
        reason="Executor capability snapshot is unavailable.",
    )


def _config_payload(config: SpiceWorkspaceConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, SpiceWorkspaceConfig):
        return dict(config.to_payload())
    return dict(config or {})


def _candidate_execution_blockers(candidate: GenericCandidate) -> list[str]:
    blockers: list[str] = []
    if candidate.availability_status == "blocked":
        reason = "; ".join(candidate.why_blocked) or "Candidate availability is blocked."
        blockers.append(reason)
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    if execution_intent.intent_class != "execution_requested":
        blockers.append(
            "Candidate is advisory; execution_intent.intent_class is not execution_requested."
        )
    if not crosses_execution_approval_boundary(candidate):
        blockers.append(
            "Candidate is read-only perception or does not cross an execution approval boundary."
        )
    if not execution_intent.requested:
        blockers.append("Candidate execution_intent.requested is false.")
    if not str(execution_intent.handoff_task or "").strip():
        blockers.append("Candidate execution_intent.handoff_task is empty.")
    if not candidate.requires_confirmation:
        blockers.append("Candidate does not request an approval-gated executor handoff.")
    boundary = candidate.execution_boundary
    if boundary is not None and boundary.requires_confirmation is False:
        blockers.append("Candidate execution boundary does not require confirmation.")
    if not _has_handoff_anchor(candidate):
        blockers.append("Candidate has no concrete executor handoff target.")
    if candidate.action_type == "capability.use" and not candidate.required_capability:
        blockers.append("Candidate uses a capability action but does not name a required capability.")
    return _dedupe(blockers)


def _has_handoff_anchor(candidate: GenericCandidate) -> bool:
    execution_intent = getattr(candidate, "execution_intent", GenericExecutionIntent())
    if str(execution_intent.handoff_task or "").strip():
        return True
    boundary = candidate.execution_boundary
    if candidate.target_refs:
        return True
    if boundary is None:
        return False
    if boundary.target or boundary.protocol:
        return True
    return boundary.mode in {"execution_intent", "capability", "sdep"}


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
