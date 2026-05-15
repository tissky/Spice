from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from spice.runtime.executor_runtime import resolve_executor_runtime_from_config
from spice.runtime.workspace import SpiceWorkspaceConfig


RUNTIME_GUARDRAIL_SCHEMA_VERSION = "spice.runtime_guardrail.v1"


@dataclass(frozen=True, slots=True)
class RuntimeGuardrailResult:
    allowed: bool
    action: str
    message: str = ""
    blockers: tuple[str, ...] = ()
    candidate_id: str = ""
    approval_id: str = ""
    schema_version: str = RUNTIME_GUARDRAIL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "action": self.action,
            "message": self.message,
            "blockers": list(self.blockers),
            "candidate_id": self.candidate_id,
            "approval_id": self.approval_id,
            "metadata": dict(self.metadata),
        }


def validate_active_frame_route(
    *,
    action: str,
    active_frame: Mapping[str, Any] | None,
    candidate_id: str = "",
    config: Mapping[str, Any] | SpiceWorkspaceConfig | None = None,
) -> RuntimeGuardrailResult:
    normalized_action = str(action or "").strip().lower()
    frame = _mapping(active_frame)
    if normalized_action in {"new_intent"}:
        return _allow(normalized_action)
    if not frame:
        return _block(
            normalized_action,
            "There is no active Decision Card to continue from.",
            "missing_active_decision_frame",
        )

    if normalized_action == "choose_option":
        candidate = _candidate_for_route(frame, candidate_id)
        if not candidate:
            return _block(
                normalized_action,
                "I could not find that option on the active Decision Card.",
                "target_candidate_not_found",
                candidate_id=candidate_id,
            )
        return _allow(normalized_action, candidate_id=str(candidate.get("candidate_id") or ""))

    if normalized_action in {"refine", "show_details", "skip"}:
        return _allow(normalized_action, candidate_id=_selected_candidate_id(frame))

    if normalized_action == "approve_only":
        approval_id = _approval_id(frame)
        if not approval_id:
            return _block(
                normalized_action,
                "There is no approval attached to the active Decision Card.",
                "missing_approval",
                candidate_id=_selected_candidate_id(frame),
            )
        return _allow(
            normalized_action,
            candidate_id=_selected_candidate_id(frame),
            approval_id=approval_id,
        )

    if normalized_action in {"execute_selected", "approve_execute"}:
        return _validate_execution_route(
            action=normalized_action,
            frame=frame,
            candidate_id=candidate_id,
            config=config,
        )

    return _block(
        normalized_action,
        f"Unsupported routed action: {normalized_action or 'unknown'}.",
        "unsupported_action",
    )


def render_guardrail_message(result: RuntimeGuardrailResult) -> str:
    if result.allowed:
        return ""
    lines = ["I can't continue into execution from that message."]
    if result.message:
        lines.append(result.message)
    if result.blockers:
        lines.append("Runtime guardrails:")
        lines.extend(f"- {blocker}" for blocker in result.blockers)
    if result.candidate_id:
        lines.append(f"candidate_id: {result.candidate_id}")
    if result.approval_id:
        lines.append(f"approval_id: {result.approval_id}")
    lines.append("Next: choose another option, refine the decision, or create an explicit `/act ...` request.")
    return "\n".join(lines)


def _validate_execution_route(
    *,
    action: str,
    frame: Mapping[str, Any],
    candidate_id: str,
    config: Mapping[str, Any] | SpiceWorkspaceConfig | None,
) -> RuntimeGuardrailResult:
    candidate = _candidate_for_route(frame, candidate_id) or _selected_candidate(frame)
    selected_id = str(candidate.get("candidate_id") or _selected_candidate_id(frame))
    approval_id = _approval_id(frame)
    blockers: list[str] = []

    if not candidate:
        blockers.append("target_candidate_not_found")
    if not _has_artifact_source(frame):
        blockers.append("missing_artifact_source")

    if action == "approve_execute" and not approval_id:
        blockers.append("missing_approval")

    affordance = _mapping(candidate.get("execution_affordance"))
    approval = _mapping(affordance.get("approval"))
    permission = _mapping(affordance.get("permission"))
    if not approval_id:
        if not _candidate_execution_eligible(affordance):
            blockers.append("candidate_advisory_only")
        if not approval.get("required"):
            blockers.append("approval_not_required_or_not_available")
    if affordance.get("blocked"):
        blockers.extend(str(item) for item in _list(affordance.get("blockers"))[:3])

    executor_blocker = _executor_blocker(config)
    if executor_blocker:
        blockers.append(executor_blocker)
    if permission.get("escalation_required"):
        blockers.append(
            "permission_insufficient: "
            f"requires {permission.get('required') or 'higher permission'}; "
            f"configured {permission.get('configured') or 'unknown'}"
        )

    blockers = _dedupe(blockers)
    if blockers:
        return _block(
            action,
            _execution_block_message(blockers, approval_id=approval_id),
            *blockers,
            candidate_id=selected_id,
            approval_id=approval_id,
            metadata={"execution_affordance": affordance},
        )
    return _allow(
        action,
        candidate_id=selected_id,
        approval_id=approval_id,
        metadata={"execution_affordance": affordance},
    )


def _execution_block_message(blockers: list[str], *, approval_id: str) -> str:
    if "candidate_advisory_only" in blockers:
        return (
            "The selected option is advisory-only. It was recommended for decision support, "
            "but it is not an executor handoff candidate."
        )
    if "missing_approval" in blockers:
        return "The active Decision Card has no approval artifact to execute."
    if "missing_artifact_source" in blockers:
        return "The active Decision Card is missing the run or decision artifact source."
    if any(blocker.startswith("permission_insufficient") for blocker in blockers):
        return "The selected option needs higher executor permission than the current runtime has."
    if approval_id:
        return "The active approval cannot be executed until runtime guardrails pass."
    return "The selected option is not ready for an approval-gated executor handoff."


def _candidate_execution_eligible(affordance: Mapping[str, Any]) -> bool:
    approval = _mapping(affordance.get("approval"))
    return bool(
        affordance.get("candidate_executable")
        and affordance.get("executor_available")
        and affordance.get("executable")
        and approval.get("eligible_for_approval")
    )


def _executor_blocker(config: Mapping[str, Any] | SpiceWorkspaceConfig | None) -> str:
    try:
        if isinstance(config, SpiceWorkspaceConfig):
            runtime = resolve_executor_runtime_from_config(config)
        else:
            runtime = resolve_executor_runtime_from_config(
                SpiceWorkspaceConfig.from_payload(_mapping(config))
            )
    except Exception as exc:
        return f"executor_config_error: {exc}"
    if runtime.status != "ready":
        return runtime.detail or f"executor_not_ready: {runtime.executor_id}"
    return ""


def _candidate_for_route(frame: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    if candidate_id:
        for candidate in _list(frame.get("candidates")):
            item = _mapping(candidate)
            if str(item.get("candidate_id") or "") == candidate_id:
                return item
    return {}


def _selected_candidate(frame: Mapping[str, Any]) -> dict[str, Any]:
    selected = _mapping(frame.get("selected"))
    if selected:
        return selected
    return _candidate_for_route(frame, _selected_candidate_id(frame))


def _selected_candidate_id(frame: Mapping[str, Any]) -> str:
    selected = _mapping(frame.get("selected"))
    return str(selected.get("candidate_id") or frame.get("selected_candidate_id") or "")


def _approval_id(frame: Mapping[str, Any]) -> str:
    return str(frame.get("approval_id") or "").strip()


def _has_artifact_source(frame: Mapping[str, Any]) -> bool:
    return bool(str(frame.get("run_id") or "").strip() and str(frame.get("decision_id") or "").strip())


def _allow(
    action: str,
    *,
    candidate_id: str = "",
    approval_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> RuntimeGuardrailResult:
    return RuntimeGuardrailResult(
        allowed=True,
        action=action,
        candidate_id=candidate_id,
        approval_id=approval_id,
        metadata=dict(metadata or {}),
    )


def _block(
    action: str,
    message: str,
    *blockers: str,
    candidate_id: str = "",
    approval_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> RuntimeGuardrailResult:
    return RuntimeGuardrailResult(
        allowed=False,
        action=action,
        message=message,
        blockers=tuple(_dedupe(list(blockers))),
        candidate_id=candidate_id,
        approval_id=approval_id,
        metadata=dict(metadata or {}),
    )


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
