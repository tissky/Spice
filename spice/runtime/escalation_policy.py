from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from spice.perception.delegated import (
    INVESTIGATION_CONSENT_EXPIRED,
    INVESTIGATION_CONSENT_GRANTED,
    INVESTIGATION_CONSENT_PENDING,
    INVESTIGATION_CONSENT_REJECTED,
    InvestigationConsent,
    InvestigationConsentBudget,
    build_investigation_consent,
)
from spice.runtime.executor_capabilities import (
    ExecutorCapabilitySnapshot,
    static_executor_capability_snapshot,
    unavailable_executor_capability_snapshot,
)


RUNTIME_ESCALATION_POLICY_SCHEMA_VERSION = "spice.runtime_escalation_policy.v2"

ESCALATION_CONTINUE = "continue"
ESCALATION_RUN_WORKSPACE_PERCEPTION = "run_workspace_perception"
ESCALATION_RUN_URL_PERCEPTION = "run_url_perception"
ESCALATION_CREATE_INVESTIGATION_CONSENT = "create_investigation_consent"
ESCALATION_AWAIT_INVESTIGATION_CONSENT = "await_investigation_consent"
ESCALATION_RUN_DELEGATED_PERCEPTION = "run_delegated_perception"
ESCALATION_REQUEST_EXECUTION_APPROVAL = "request_execution_approval"
ESCALATION_BLOCKED = "blocked"

FINAL_STRATEGY_NONE = "none"
FINAL_STRATEGY_LOCAL_WORKSPACE = "local_workspace"
FINAL_STRATEGY_URL = "url"
FINAL_STRATEGY_DELEGATED = "delegated"
FINAL_STRATEGY_LOCAL_THEN_DELEGATED = "local_then_delegated"
FINAL_STRATEGY_EXECUTION_APPROVAL = "execution_approval"

STEP_WORKSPACE = "workspace"
STEP_URL = "url"
STEP_DELEGATED = "delegated"
STEP_EXECUTION_APPROVAL = "execution_approval"

CONSENT_DELEGATED = "delegated"

_EXECUTION_ACTIONS = frozenset({"execute_selected", "approve_execute"})
_DELEGATED_STRATEGIES = frozenset({"delegated", "local_then_delegated_if_insufficient"})
_WORKSPACE_STRATEGIES = frozenset({"local_workspace", "local_then_delegated_if_insufficient"})
_DELEGATED_SCOPE = "read_only_investigation"
_DELEGATED_PERMISSION_MODE = "read_only"

_WEB_CAPABILITY_IDS = frozenset(
    {
        "web_research",
        "external_search",
        "browser_research",
        "read_web_page",
        "browser_or_external_tools",
        "tool_use",
        "general_execution",
    }
)
_REPO_CAPABILITY_IDS = frozenset(
    {
        "repo_inspection",
        "repo_read",
        "docs_review",
        "tool_use",
        "general_execution",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeEscalationDecision:
    action: str
    context_strategy: str = "none"
    final_strategy: str = FINAL_STRATEGY_NONE
    steps: list[str] = field(default_factory=list)
    forced_by: list[str] = field(default_factory=list)
    requires_consent: list[str] = field(default_factory=list)
    route: str = ""
    semantic_action: str = ""
    reason: str = ""
    workspace_query: str = ""
    url_query: str = ""
    urls: list[str] = field(default_factory=list)
    delegated_perception_query: str = ""
    delegated_perception_reason: str = ""
    delegated_plan: dict[str, Any] = field(default_factory=dict)
    suggested_capabilities: list[str] = field(default_factory=list)
    executor_id: str = ""
    executor_capability_source: str = ""
    executor_capability_status: str = ""
    available_capability_ids: list[str] = field(default_factory=list)
    matched_capability_ids: list[str] = field(default_factory=list)
    missing_capability_ids: list[str] = field(default_factory=list)
    delegated_scope: str = ""
    permission_mode: str = ""
    consent_id: str = ""
    consent_status: str = ""
    should_run_workspace_perception: bool = False
    should_run_url_perception: bool = False
    should_create_investigation_consent: bool = False
    should_run_delegated_perception: bool = False
    requires_execution_approval: bool = False
    blocked_reason: str = ""
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RUNTIME_ESCALATION_POLICY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "context_strategy": self.context_strategy,
            "final_strategy": self.final_strategy,
            "steps": list(self.steps),
            "forced_by": list(self.forced_by),
            "requires_consent": list(self.requires_consent),
            "route": self.route,
            "semantic_action": self.semantic_action,
            "reason": self.reason,
            "workspace_query": self.workspace_query,
            "url_query": self.url_query,
            "urls": list(self.urls),
            "delegated_perception_query": self.delegated_perception_query,
            "delegated_perception_reason": self.delegated_perception_reason,
            "delegated_plan": dict(self.delegated_plan),
            "suggested_capabilities": list(self.suggested_capabilities),
            "executor_id": self.executor_id,
            "executor_capability_source": self.executor_capability_source,
            "executor_capability_status": self.executor_capability_status,
            "available_capability_ids": list(self.available_capability_ids),
            "matched_capability_ids": list(self.matched_capability_ids),
            "missing_capability_ids": list(self.missing_capability_ids),
            "delegated_scope": self.delegated_scope,
            "permission_mode": self.permission_mode,
            "consent_id": self.consent_id,
            "consent_status": self.consent_status,
            "should_run_workspace_perception": self.should_run_workspace_perception,
            "should_run_url_perception": self.should_run_url_perception,
            "should_create_investigation_consent": self.should_create_investigation_consent,
            "should_run_delegated_perception": self.should_run_delegated_perception,
            "requires_execution_approval": self.requires_execution_approval,
            "blocked_reason": self.blocked_reason,
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }


def decide_runtime_escalation(
    route: Any,
    *,
    config: Mapping[str, Any] | Any | None = None,
    executor_capabilities: ExecutorCapabilitySnapshot | Mapping[str, Any] | None = None,
    workspace_context: Mapping[str, Any] | None = None,
    workspace_perception: Mapping[str, Any] | Any | None = None,
    url_context: Mapping[str, Any] | None = None,
    url_perception: Mapping[str, Any] | Any | None = None,
    investigation_consent: InvestigationConsent | Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> RuntimeEscalationDecision:
    """Choose the next runtime boundary action without performing it.

    This policy keeps perception and execution separate:
    local perception can run directly, delegated read-only investigation requires
    investigation consent, and execution requests still require execution approval.
    """

    route_payload = _route_payload(route)
    strategy = _context_strategy(route_payload)
    route_name = str(route_payload.get("route") or "")
    semantic_action = str(route_payload.get("action") or "")

    if route_name == "execution_request" or semantic_action in _EXECUTION_ACTIONS:
        return RuntimeEscalationDecision(
            action=ESCALATION_REQUEST_EXECUTION_APPROVAL,
            context_strategy=strategy,
            **_final_strategy_fields(route_payload, strategy, execution_boundary=True),
            route=route_name,
            semantic_action=semantic_action,
            reason="Execution requests cross the execution boundary and must use approval flow.",
            requires_execution_approval=True,
            metadata={"source": "runtime_escalation_policy"},
        )

    if strategy in _WORKSPACE_STRATEGIES and not _workspace_context_available(
        workspace_context=workspace_context,
        workspace_perception=workspace_perception,
    ):
        return RuntimeEscalationDecision(
            action=ESCALATION_RUN_WORKSPACE_PERCEPTION,
            context_strategy=strategy,
            **_final_strategy_fields(route_payload, strategy),
            route=route_name,
            semantic_action=semantic_action,
            reason="Route requested local workspace facts before answering.",
            workspace_query=_workspace_query(route_payload),
            delegated_perception_query=_delegated_query(route_payload),
            delegated_perception_reason=_delegated_reason(route_payload),
            delegated_plan=_delegated_plan(route_payload),
            suggested_capabilities=_suggested_capabilities(route_payload, strategy),
            should_run_workspace_perception=True,
            metadata={"source": "runtime_escalation_policy"},
        )

    if strategy == "url" and not _url_context_available(
        url_context=url_context,
        url_perception=url_perception,
    ):
        return RuntimeEscalationDecision(
            action=ESCALATION_RUN_URL_PERCEPTION,
            context_strategy=strategy,
            **_final_strategy_fields(route_payload, strategy),
            route=route_name,
            semantic_action=semantic_action,
            reason="Route requested explicit URL perception before answering.",
            url_query=_url_query(route_payload),
            urls=_strings(route_payload.get("urls")),
            should_run_url_perception=True,
            metadata={"source": "runtime_escalation_policy"},
        )

    if (
        strategy == "local_then_delegated_if_insufficient"
        and _local_context_sufficient(
            workspace_context=workspace_context,
            workspace_perception=workspace_perception,
        )
        and not _forced_by_external_evidence(route_payload)
    ):
        return RuntimeEscalationDecision(
            action=ESCALATION_CONTINUE,
            context_strategy=strategy,
            **_final_strategy_fields(route_payload, strategy, local_context_sufficient=True),
            route=route_name,
            semantic_action=semantic_action,
            reason="Local workspace perception is sufficient; delegated investigation is not needed.",
            workspace_query=_workspace_query(route_payload),
            metadata={
                "source": "runtime_escalation_policy",
                "local_context_sufficient": True,
            },
        )

    if strategy in _DELEGATED_STRATEGIES:
        return _delegated_escalation_decision(
            route_payload,
            strategy=strategy,
            route_name=route_name,
            semantic_action=semantic_action,
            config=config,
            executor_capabilities=executor_capabilities,
            investigation_consent=investigation_consent,
            now=now,
        )

    return RuntimeEscalationDecision(
        action=ESCALATION_CONTINUE,
        context_strategy=strategy,
        **_final_strategy_fields(route_payload, strategy),
        route=route_name,
        semantic_action=semantic_action,
        reason="No additional runtime perception or approval is required.",
        metadata={"source": "runtime_escalation_policy"},
    )


def build_investigation_consent_for_escalation(
    decision: RuntimeEscalationDecision | Mapping[str, Any],
    *,
    input_context_refs: list[str] | None = None,
    budget: InvestigationConsentBudget | Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    expires_in_sec: int = 600,
) -> InvestigationConsent:
    payload = decision.to_payload() if isinstance(decision, RuntimeEscalationDecision) else dict(decision)
    if not payload.get("should_create_investigation_consent"):
        raise ValueError("Escalation decision does not request investigation consent creation.")
    executor_id = str(payload.get("executor_id") or "").strip()
    query = str(payload.get("delegated_perception_query") or "").strip()
    if not executor_id:
        raise ValueError("Escalation decision is missing executor_id.")
    if not query:
        raise ValueError("Escalation decision is missing delegated_perception_query.")
    return build_investigation_consent(
        executor_id=executor_id,
        query=query,
        input_context_refs=input_context_refs,
        budget=budget,
        created_at=created_at,
        expires_in_sec=expires_in_sec,
        metadata={
            "source": "runtime_escalation_policy",
            "context_strategy": str(payload.get("context_strategy") or ""),
            "final_strategy": str(payload.get("final_strategy") or ""),
            "steps": _strings(payload.get("steps")),
            "forced_by": _strings(payload.get("forced_by")),
            "requires_consent": _strings(payload.get("requires_consent")),
            "suggested_capabilities": _strings(payload.get("suggested_capabilities")),
            "delegated_perception_reason": str(payload.get("delegated_perception_reason") or ""),
            "delegated_plan": _mapping_from_any(payload.get("delegated_plan")),
            "expected_output": str(
                _mapping_from_any(payload.get("delegated_plan")).get("expected_output") or ""
            ),
        },
    )


def _delegated_escalation_decision(
    route_payload: Mapping[str, Any],
    *,
    strategy: str,
    route_name: str,
    semantic_action: str,
    config: Mapping[str, Any] | Any | None,
    executor_capabilities: ExecutorCapabilitySnapshot | Mapping[str, Any] | None,
    investigation_consent: InvestigationConsent | Mapping[str, Any] | None,
    now: datetime | None,
) -> RuntimeEscalationDecision:
    config_payload = _mapping_from_any(config)
    snapshot = _capability_snapshot(
        executor_capabilities,
        executor_id=str(config_payload.get("executor") or ""),
    )
    capabilities = _suggested_capabilities(route_payload, strategy)
    capability_check = _delegated_capability_check(capabilities, snapshot)
    base_kwargs = {
        "context_strategy": strategy,
        **_final_strategy_fields(route_payload, strategy),
        "route": route_name,
        "semantic_action": semantic_action,
        "workspace_query": _workspace_query(route_payload),
        "delegated_perception_query": _delegated_query(route_payload),
        "delegated_perception_reason": _delegated_reason(route_payload),
        "delegated_plan": _delegated_plan(route_payload),
        "suggested_capabilities": capabilities,
        "executor_id": snapshot.executor_id,
        "executor_capability_source": snapshot.source,
        "executor_capability_status": snapshot.status,
        "available_capability_ids": list(snapshot.capability_ids),
        "matched_capability_ids": list(capability_check["matched"]),
        "missing_capability_ids": list(capability_check["missing"]),
        "limitations": list(snapshot.limitations),
        "metadata": {
            "source": "runtime_escalation_policy",
            "capability_groups": capability_check["groups"],
        },
    }

    if snapshot.status == "unavailable":
        return RuntimeEscalationDecision(
            action=ESCALATION_BLOCKED,
            reason="Delegated perception requires a configured executor.",
            blocked_reason=snapshot.limitations[0] if snapshot.limitations else "Executor is unavailable.",
            **base_kwargs,
        )
    if capability_check["missing"]:
        return RuntimeEscalationDecision(
            action=ESCALATION_BLOCKED,
            reason="Configured executor lacks the requested read-only investigation capability.",
            blocked_reason=(
                "Executor lacks delegated perception capability: "
                + ", ".join(capability_check["missing"])
            ),
            **base_kwargs,
        )

    consent = _coerce_consent(investigation_consent)
    if consent is None:
        return RuntimeEscalationDecision(
            action=ESCALATION_CREATE_INVESTIGATION_CONSENT,
            reason="Delegated read-only investigation requires explicit investigation consent.",
            should_create_investigation_consent=True,
            **base_kwargs,
        )

    consent_block = _consent_block_reason(consent, snapshot=snapshot, query=_delegated_query(route_payload), now=now)
    if consent_block:
        return RuntimeEscalationDecision(
            action=ESCALATION_BLOCKED,
            reason="Investigation consent cannot authorize delegated perception.",
            blocked_reason=consent_block,
            consent_id=consent.consent_id,
            consent_status=consent.status,
            **base_kwargs,
        )
    if consent.status == INVESTIGATION_CONSENT_PENDING:
        return RuntimeEscalationDecision(
            action=ESCALATION_AWAIT_INVESTIGATION_CONSENT,
            reason="Investigation consent is pending user confirmation.",
            consent_id=consent.consent_id,
            consent_status=consent.status,
            **base_kwargs,
        )
    if consent.status == INVESTIGATION_CONSENT_GRANTED:
        return RuntimeEscalationDecision(
            action=ESCALATION_RUN_DELEGATED_PERCEPTION,
            reason="Investigation consent is granted and executor capability matches.",
            consent_id=consent.consent_id,
            consent_status=consent.status,
            should_run_delegated_perception=True,
            **base_kwargs,
        )

    return RuntimeEscalationDecision(
        action=ESCALATION_BLOCKED,
        reason="Investigation consent is not granted.",
        blocked_reason=f"Investigation consent status is {consent.status}.",
        consent_id=consent.consent_id,
        consent_status=consent.status,
        **base_kwargs,
    )


def _route_payload(route: Any) -> dict[str, Any]:
    if hasattr(route, "to_payload") and callable(route.to_payload):
        payload = route.to_payload()
        return dict(payload) if isinstance(payload, dict) else {}
    return dict(route) if isinstance(route, Mapping) else {}


def _context_strategy(payload: Mapping[str, Any]) -> str:
    strategy = str(payload.get("context_strategy") or "").strip()
    if strategy:
        return strategy
    perception_plan = _mapping_from_any(payload.get("perception_plan"))
    planner_strategy = _runtime_strategy_from_planner_strategy(
        str(perception_plan.get("perception_strategy") or "")
    )
    if planner_strategy:
        return planner_strategy
    if payload.get("needs_url_context"):
        return "url"
    if payload.get("needs_delegated_perception"):
        return "delegated"
    if payload.get("needs_workspace_context"):
        return "local_workspace"
    return "none"


def _runtime_strategy_from_planner_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower()
    if normalized == "none":
        return ""
    if normalized == "local_then_delegated":
        return "local_then_delegated_if_insufficient"
    if normalized == "mixed":
        return "local_then_delegated_if_insufficient"
    if normalized in {"local_workspace", "url", "delegated"}:
        return normalized
    return ""


def _final_strategy_fields(
    route_payload: Mapping[str, Any],
    strategy: str,
    *,
    local_context_sufficient: bool = False,
    execution_boundary: bool = False,
) -> dict[str, Any]:
    forced_by = _normalized_forced_by(route_payload)
    if execution_boundary:
        return {
            "final_strategy": FINAL_STRATEGY_EXECUTION_APPROVAL,
            "steps": [STEP_EXECUTION_APPROVAL],
            "forced_by": _unique([*forced_by, "execution_request"]),
            "requires_consent": [],
            "delegated_scope": "",
            "permission_mode": "",
        }
    if strategy == "local_then_delegated_if_insufficient":
        if local_context_sufficient and not _forced_by_external_evidence(route_payload):
            return {
                "final_strategy": FINAL_STRATEGY_LOCAL_WORKSPACE,
                "steps": [STEP_WORKSPACE],
                "forced_by": forced_by,
                "requires_consent": [],
                "delegated_scope": "",
                "permission_mode": "",
            }
        return {
            "final_strategy": FINAL_STRATEGY_LOCAL_THEN_DELEGATED,
            "steps": [STEP_WORKSPACE, STEP_DELEGATED],
            "forced_by": forced_by,
            "requires_consent": [CONSENT_DELEGATED],
            "delegated_scope": _DELEGATED_SCOPE,
            "permission_mode": _DELEGATED_PERMISSION_MODE,
        }
    if strategy == "local_workspace":
        return {
            "final_strategy": FINAL_STRATEGY_LOCAL_WORKSPACE,
            "steps": [STEP_WORKSPACE],
            "forced_by": forced_by,
            "requires_consent": [],
            "delegated_scope": "",
            "permission_mode": "",
        }
    if strategy == "url":
        return {
            "final_strategy": FINAL_STRATEGY_URL,
            "steps": [STEP_URL],
            "forced_by": forced_by,
            "requires_consent": [],
            "delegated_scope": "",
            "permission_mode": "",
        }
    if strategy == "delegated":
        return {
            "final_strategy": FINAL_STRATEGY_DELEGATED,
            "steps": [STEP_DELEGATED],
            "forced_by": forced_by,
            "requires_consent": [CONSENT_DELEGATED],
            "delegated_scope": _DELEGATED_SCOPE,
            "permission_mode": _DELEGATED_PERMISSION_MODE,
        }
    return {
        "final_strategy": FINAL_STRATEGY_NONE,
        "steps": [],
        "forced_by": forced_by,
        "requires_consent": [],
        "delegated_scope": "",
        "permission_mode": "",
    }


def _normalized_forced_by(payload: Mapping[str, Any]) -> list[str]:
    values = _strings(payload.get("forced_by"))
    policy = _mapping_from_any(payload.get("route_merge_policy"))
    if not policy:
        raw = _mapping_from_any(payload.get("raw"))
        policy = _mapping_from_any(raw.get("route_merge_policy"))
    values = _unique([*values, *_strings(policy.get("forced_by"))])
    result: list[str] = []
    for value in values:
        if value in {"repo_evidence_requirement", "workspace_scope_allowed"}:
            result.append("explicit_repo")
        elif value in {"explicit_url", "url_evidence_requirement"}:
            result.append("explicit_url")
        elif value == "external_evidence_requirement":
            result.append("external_comparison_requested")
        elif value == "semantic_route":
            result.append("planner")
        else:
            result.append(value)
    return _unique(result)


def _forced_by_external_evidence(payload: Mapping[str, Any]) -> bool:
    forced_by = _strings(payload.get("forced_by"))
    policy = _mapping_from_any(payload.get("route_merge_policy"))
    if not policy:
        raw = _mapping_from_any(payload.get("raw"))
        policy = _mapping_from_any(raw.get("route_merge_policy"))
    forced_by = _unique([*forced_by, *_strings(policy.get("forced_by"))])
    return "external_evidence_requirement" in forced_by


def _workspace_query(payload: Mapping[str, Any]) -> str:
    plan = _mapping_from_any(payload.get("perception_plan"))
    workspace_plan = _mapping_from_any(plan.get("workspace_plan"))
    return str(
        payload.get("workspace_query")
        or workspace_plan.get("query")
        or workspace_plan.get("workspace_query")
        or payload.get("text")
        or ""
    ).strip()


def _url_query(payload: Mapping[str, Any]) -> str:
    plan = _mapping_from_any(payload.get("perception_plan"))
    url_plan = _mapping_from_any(plan.get("url_plan"))
    return str(
        payload.get("url_query")
        or url_plan.get("query")
        or url_plan.get("url_query")
        or payload.get("text")
        or ""
    ).strip()


def _delegated_query(payload: Mapping[str, Any]) -> str:
    plan = _mapping_from_any(payload.get("perception_plan"))
    delegated_plan = _mapping_from_any(plan.get("delegated_plan"))
    return str(
        payload.get("delegated_perception_query")
        or delegated_plan.get("query")
        or delegated_plan.get("delegated_perception_query")
        or payload.get("workspace_query")
        or payload.get("url_query")
        or payload.get("text")
        or ""
    ).strip()


def _delegated_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    plan = _mapping_from_any(payload.get("perception_plan"))
    delegated_plan = _mapping_from_any(plan.get("delegated_plan"))
    if not delegated_plan:
        query = _delegated_query(payload)
        if query:
            delegated_plan["query"] = query
    if not delegated_plan:
        return {}
    normalized = dict(delegated_plan)
    normalized["scope"] = _DELEGATED_SCOPE
    normalized["permission_mode"] = _DELEGATED_PERMISSION_MODE
    if not str(normalized.get("expected_output") or "").strip():
        normalized["expected_output"] = "findings_sources_limitations"
    if not str(normalized.get("query") or "").strip():
        normalized["query"] = _delegated_query(payload)
    capabilities = _strings(normalized.get("requested_capabilities") or normalized.get("suggested_capabilities"))
    if capabilities:
        normalized["requested_capabilities"] = capabilities
    elif "requested_capabilities" not in normalized:
        normalized["requested_capabilities"] = _suggested_capabilities(payload, _context_strategy(payload))
    return normalized


def _delegated_reason(payload: Mapping[str, Any]) -> str:
    plan = _mapping_from_any(payload.get("perception_plan"))
    delegated_plan = _mapping_from_any(plan.get("delegated_plan"))
    return str(
        payload.get("delegated_perception_reason")
        or delegated_plan.get("reason")
        or plan.get("reason")
        or ""
    ).strip()


def _suggested_capabilities(payload: Mapping[str, Any], strategy: str) -> list[str]:
    values = _strings(payload.get("suggested_capabilities"))
    if values:
        return values
    plan = _mapping_from_any(payload.get("perception_plan"))
    delegated_plan = _mapping_from_any(plan.get("delegated_plan"))
    values = _strings(delegated_plan.get("requested_capabilities"))
    if values:
        return values
    return ["web_research"] if strategy == "delegated" else ["repo_inspection"]


def _workspace_context_available(
    *,
    workspace_context: Mapping[str, Any] | None,
    workspace_perception: Mapping[str, Any] | Any | None,
) -> bool:
    return bool(_mapping_from_any(workspace_context) or _mapping_from_any(workspace_perception))


def _url_context_available(
    *,
    url_context: Mapping[str, Any] | None,
    url_perception: Mapping[str, Any] | Any | None,
) -> bool:
    return bool(_mapping_from_any(url_context) or _mapping_from_any(url_perception))


def _local_context_sufficient(
    *,
    workspace_context: Mapping[str, Any] | None,
    workspace_perception: Mapping[str, Any] | Any | None,
) -> bool:
    context = _mapping_from_any(workspace_context)
    perception = _mapping_from_any(workspace_perception)
    combined = {**perception, **context}
    if not combined:
        return False
    status = str(combined.get("status") or combined.get("metadata", {}).get("status") or "").lower()
    if status in {"failed", "skipped", "blocked"}:
        return False
    text = " ".join(
        str(value or "").lower()
        for value in (
            combined.get("summary"),
            combined.get("error"),
            combined.get("limitations"),
            combined.get("metadata", {}).get("reason") if isinstance(combined.get("metadata"), Mapping) else "",
        )
    )
    if any(marker in text for marker in ("insufficient", "skipped", "failed", "not inspect")):
        return False
    facts = _list(combined.get("facts"))
    snippets = _list(combined.get("snippets"))
    files_read = _list(combined.get("files_read"))
    if not facts and not snippets and not files_read:
        return bool(str(combined.get("summary") or "").strip())
    if facts and all(_fact_confidence(item) <= 0.05 for item in facts):
        return False
    return True


def _fact_confidence(value: Any) -> float:
    if isinstance(value, Mapping):
        try:
            return float(value.get("confidence", 1.0))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _capability_snapshot(
    raw: ExecutorCapabilitySnapshot | Mapping[str, Any] | None,
    *,
    executor_id: str,
) -> ExecutorCapabilitySnapshot:
    if isinstance(raw, ExecutorCapabilitySnapshot):
        return raw
    if isinstance(raw, Mapping) and raw:
        try:
            return ExecutorCapabilitySnapshot.from_payload(dict(raw))
        except (TypeError, ValueError):
            return unavailable_executor_capability_snapshot(
                executor_id or "unknown",
                provider=executor_id or "unknown",
                reason="Executor capability snapshot is malformed.",
            )
    if executor_id:
        return static_executor_capability_snapshot(executor_id)
    return unavailable_executor_capability_snapshot(
        "unknown",
        reason="Executor is not configured.",
    )


def _delegated_capability_check(
    suggested_capabilities: list[str],
    snapshot: ExecutorCapabilitySnapshot,
) -> dict[str, Any]:
    available = set(snapshot.capability_ids)
    requested = suggested_capabilities or ["web_research"]
    matched: list[str] = []
    missing: list[str] = []
    groups: dict[str, list[str]] = {}
    for capability in requested:
        group = _capability_group(capability)
        allowed = _WEB_CAPABILITY_IDS if group == "web" else _REPO_CAPABILITY_IDS
        groups[capability] = sorted(allowed)
        match = sorted(available & allowed)
        if match:
            matched.append(f"{capability}:{match[0]}")
        else:
            missing.append(capability)
    return {"matched": matched, "missing": missing, "groups": groups}


def _capability_group(capability: str) -> str:
    normalized = str(capability or "").strip().lower()
    if normalized in {
        "repo_inspection",
        "repo_read",
        "docs_review",
        "code_review",
        "workspace_research",
    }:
        return "repo"
    return "web"


def _coerce_consent(
    raw: InvestigationConsent | Mapping[str, Any] | None,
) -> InvestigationConsent | None:
    if raw is None:
        return None
    if isinstance(raw, InvestigationConsent):
        return raw
    if isinstance(raw, Mapping) and raw:
        return InvestigationConsent.from_payload(raw)
    return None


def _consent_block_reason(
    consent: InvestigationConsent,
    *,
    snapshot: ExecutorCapabilitySnapshot,
    query: str,
    now: datetime | None,
) -> str:
    if consent.status in {INVESTIGATION_CONSENT_REJECTED, INVESTIGATION_CONSENT_EXPIRED}:
        return f"Investigation consent status is {consent.status}."
    if consent.executor_id and consent.executor_id != snapshot.executor_id:
        return (
            f"Investigation consent is for executor {consent.executor_id}, "
            f"not {snapshot.executor_id}."
        )
    if consent.scope != "read_only_investigation":
        return f"Investigation consent scope is not read_only_investigation: {consent.scope}."
    if consent.permission_mode != "read_only":
        return f"Investigation consent permission mode is not read_only: {consent.permission_mode}."
    if not {"write_file", "patch", "terminal_command", "install", "test_run"}.issubset(
        set(consent.denied_actions)
    ):
        return "Investigation consent does not deny execution-like actions."
    if consent.query and query and consent.query.strip() != query.strip():
        return "Investigation consent query does not match the delegated perception query."
    if _consent_expired(consent, now=now):
        return "Investigation consent is expired."
    if consent.status not in {INVESTIGATION_CONSENT_PENDING, INVESTIGATION_CONSENT_GRANTED}:
        return f"Investigation consent status is {consent.status}."
    return ""


def _consent_expired(consent: InvestigationConsent, *, now: datetime | None) -> bool:
    if not consent.expires_at:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        expires = datetime.fromisoformat(consent.expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) > expires.astimezone(timezone.utc)


def _mapping_from_any(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_payload") and callable(value.to_payload):
        payload = value.to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


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


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
