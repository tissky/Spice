from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from spice.runtime.evidence_qualification import has_source_backed_evidence
from spice.runtime.route_merge_policy import RouteMergePolicy
from spice.runtime.workspace_scope import (
    WORKSPACE_SCOPE_BLOCKED,
    WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
    WORKSPACE_SCOPE_NEEDS_SELECTION,
)


PRE_RUN_EVIDENCE_GATE_SCHEMA_VERSION = "spice.pre_run_evidence_gate.v1"

PRE_RUN_EVIDENCE_CONTINUE = "continue"
PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION = "run_workspace_perception"
PRE_RUN_EVIDENCE_RUN_URL_PERCEPTION = "run_url_perception"
PRE_RUN_EVIDENCE_CREATE_INVESTIGATION_CONSENT = "create_investigation_consent"
PRE_RUN_EVIDENCE_ANSWER_WITH_LIMITATION = "answer_with_limitation"
PRE_RUN_EVIDENCE_BLOCK = "block"
PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_CONFIRMATION = "needs_workspace_confirmation"
PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_SELECTION = "needs_workspace_selection"


@dataclass(frozen=True, slots=True)
class PreRunEvidenceGateDecision:
    allowed: bool
    action: str = PRE_RUN_EVIDENCE_CONTINUE
    reason: str = ""
    limitations: list[str] = field(default_factory=list)
    should_run_workspace_perception: bool = False
    should_run_url_perception: bool = False
    should_create_investigation_consent: bool = False
    can_make_high_confidence_evidence_claims: bool = True
    missing_source_domains: list[str] = field(default_factory=list)
    route_merge_policy: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PRE_RUN_EVIDENCE_GATE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "limitations": list(self.limitations),
            "should_run_workspace_perception": self.should_run_workspace_perception,
            "should_run_url_perception": self.should_run_url_perception,
            "should_create_investigation_consent": self.should_create_investigation_consent,
            "can_make_high_confidence_evidence_claims": self.can_make_high_confidence_evidence_claims,
            "missing_source_domains": list(self.missing_source_domains),
            "route_merge_policy": dict(self.route_merge_policy),
        }


def evaluate_pre_run_evidence_gate(
    policy: RouteMergePolicy,
    *,
    workspace_context: Mapping[str, Any] | None = None,
    url_context: Mapping[str, Any] | None = None,
    delegated_perception_context: Mapping[str, Any] | None = None,
) -> PreRunEvidenceGateDecision:
    """Choose the evidence action needed before answer generation.

    The gate is deliberately earlier and narrower than composer validation:
    it decides whether Spice must gather evidence, ask for investigation
    consent, or block a wrong-scope answer. Composer/source validators later
    enforce that final prose does not overclaim beyond those sources.
    """

    workspace_scope = policy.workspace_scope
    if policy.needs_workspace_context and workspace_scope is not None:
        scope_payload = workspace_scope.to_payload()
        if workspace_scope.status == WORKSPACE_SCOPE_BLOCKED:
            return _decision(
                policy,
                allowed=False,
                action=PRE_RUN_EVIDENCE_BLOCK,
                reason=f"workspace evidence is required, but the requested workspace scope is blocked: {workspace_scope.reason}",
                limitations=list(policy.limitations),
                missing_source_domains=["repo"],
                scope_payload=scope_payload,
            )
        if workspace_scope.status == WORKSPACE_SCOPE_NEEDS_CONFIRMATION:
            return _decision(
                policy,
                allowed=False,
                action=PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_CONFIRMATION,
                reason=(
                    "workspace evidence is required from an external repo path; "
                    "confirm that workspace before Spice reads it"
                ),
                limitations=list(policy.limitations),
                missing_source_domains=["repo"],
                scope_payload=scope_payload,
            )
        if workspace_scope.status == WORKSPACE_SCOPE_NEEDS_SELECTION:
            return _decision(
                policy,
                allowed=False,
                action=PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_SELECTION,
                reason="multiple workspace scopes matched; choose one before Spice reads code evidence",
                limitations=list(policy.limitations),
                missing_source_domains=["repo"],
                scope_payload=scope_payload,
            )

    missing = _missing_source_domains(
        policy,
        workspace_context=workspace_context,
        url_context=url_context,
        delegated_perception_context=delegated_perception_context,
    )
    if policy.needs_workspace_context and "repo" in missing:
        return _decision(
            policy,
            action=PRE_RUN_EVIDENCE_RUN_WORKSPACE_PERCEPTION,
            reason="repo evidence is required, but no workspace perception artifact is available yet",
            should_run_workspace_perception=True,
            missing_source_domains=missing,
            can_make_high_confidence_evidence_claims=False,
        )
    if policy.needs_url_context and "url" in missing:
        if not policy.urls:
            return _decision(
                policy,
                allowed=False,
                action=PRE_RUN_EVIDENCE_BLOCK,
                reason="URL evidence is required, but no URL was found to read",
                limitations=list(policy.limitations),
                missing_source_domains=missing,
            )
        return _decision(
            policy,
            action=PRE_RUN_EVIDENCE_RUN_URL_PERCEPTION,
            reason="URL evidence is required, but no URL perception artifact is available yet",
            should_run_url_perception=True,
            missing_source_domains=missing,
            can_make_high_confidence_evidence_claims=False,
        )
    if policy.needs_delegated_perception and "external" in missing:
        return _decision(
            policy,
            action=PRE_RUN_EVIDENCE_CREATE_INVESTIGATION_CONSENT,
            reason="external evidence is required, but no delegated perception artifact is available yet",
            should_create_investigation_consent=True,
            missing_source_domains=missing,
            can_make_high_confidence_evidence_claims=False,
        )

    return _decision(
        policy,
        action=PRE_RUN_EVIDENCE_CONTINUE,
        reason="required evidence is present or not required",
        missing_source_domains=missing,
        can_make_high_confidence_evidence_claims=not missing,
    )


def render_pre_run_evidence_gate_message(decision: PreRunEvidenceGateDecision) -> str:
    if decision.allowed:
        return ""
    lines = ["I need evidence before answering this safely.", f"Reason: {decision.reason}"]
    policy = _mapping(decision.route_merge_policy)
    scope = _mapping(policy.get("workspace_scope"))
    scope_path = str(scope.get("scope_path") or scope.get("workspace_root") or "").strip()
    if scope_path:
        lines.append(f"Requested scope: {scope_path}")
    if decision.action == PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_CONFIRMATION:
        lines.append("Start Spice from that workspace or configure it as the workspace root, then ask again.")
    elif decision.action == PRE_RUN_EVIDENCE_NEEDS_WORKSPACE_SELECTION:
        lines.append("Ask again with one specific repo/path so Spice does not read the wrong workspace.")
    elif decision.action == PRE_RUN_EVIDENCE_BLOCK and decision.missing_source_domains:
        lines.append("Missing source domains: " + ", ".join(decision.missing_source_domains))
    return "\n".join(lines)


def _decision(
    policy: RouteMergePolicy,
    *,
    allowed: bool = True,
    action: str,
    reason: str,
    limitations: list[str] | None = None,
    should_run_workspace_perception: bool = False,
    should_run_url_perception: bool = False,
    should_create_investigation_consent: bool = False,
    can_make_high_confidence_evidence_claims: bool = True,
    missing_source_domains: list[str] | None = None,
    scope_payload: Mapping[str, Any] | None = None,
) -> PreRunEvidenceGateDecision:
    return PreRunEvidenceGateDecision(
        allowed=allowed,
        action=action,
        reason=reason,
        limitations=list(limitations or []),
        should_run_workspace_perception=should_run_workspace_perception,
        should_run_url_perception=should_run_url_perception,
        should_create_investigation_consent=should_create_investigation_consent,
        can_make_high_confidence_evidence_claims=can_make_high_confidence_evidence_claims,
        missing_source_domains=_unique(missing_source_domains or []),
        route_merge_policy=_compact_policy(policy, scope_payload=scope_payload),
    )


def _missing_source_domains(
    policy: RouteMergePolicy,
    *,
    workspace_context: Mapping[str, Any] | None,
    url_context: Mapping[str, Any] | None,
    delegated_perception_context: Mapping[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if policy.needs_workspace_context and not _workspace_present(workspace_context):
        missing.append("repo")
    if policy.needs_url_context and not _url_present(url_context):
        missing.append("url")
    if policy.needs_delegated_perception and not _delegated_present(delegated_perception_context):
        missing.append("external")
    return missing


def _compact_policy(
    policy: RouteMergePolicy,
    *,
    scope_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = policy.to_payload()
    return {
        "schema_version": payload.get("schema_version"),
        "context_strategy": payload.get("context_strategy"),
        "needs_workspace_context": payload.get("needs_workspace_context"),
        "needs_url_context": payload.get("needs_url_context"),
        "needs_delegated_perception": payload.get("needs_delegated_perception"),
        "forced_by": list(payload.get("forced_by") or []),
        "limitations": list(payload.get("limitations") or []),
        "reason": str(payload.get("reason") or ""),
        "resource_extraction": _mapping(payload.get("resource_extraction")),
        "evidence_requirement": _mapping(payload.get("evidence_requirement")),
        "workspace_scope": dict(scope_payload or _mapping(payload.get("workspace_scope"))),
    }


def _workspace_present(payload: Mapping[str, Any] | None) -> bool:
    return has_source_backed_evidence(payload)


def _url_present(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping) or not payload:
        return False
    if payload.get("present") is False:
        return False
    return bool(
        payload.get("perception_id")
        or payload.get("documents")
        or payload.get("urls")
        or payload.get("facts")
        or payload.get("snippets")
        or payload.get("summary")
    )


def _delegated_present(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping) or not payload:
        return False
    if payload.get("present") is False:
        return False
    return bool(
        payload.get("perception_id")
        or payload.get("findings")
        or payload.get("sources")
        or payload.get("summary")
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
