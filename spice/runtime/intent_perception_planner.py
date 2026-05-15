from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


INTENT_PERCEPTION_PLANNER_SCHEMA_VERSION = "spice.intent_perception_planner.v1"

INTENT_KIND_DECISION = "decision"
INTENT_KIND_FOLLOW_UP = "follow_up"
INTENT_KIND_EXECUTION_REQUEST = "execution_request"
INTENT_KIND_INVESTIGATION_REQUEST = "investigation_request"
INTENT_KINDS = frozenset(
    {
        INTENT_KIND_DECISION,
        INTENT_KIND_FOLLOW_UP,
        INTENT_KIND_EXECUTION_REQUEST,
        INTENT_KIND_INVESTIGATION_REQUEST,
    }
)

ANSWER_MODE_BRIEF = "brief"
ANSWER_MODE_NORMAL = "normal"
ANSWER_MODE_DETAILED = "detailed"
ANSWER_MODE_REPORT = "report"
ANSWER_MODES = frozenset({ANSWER_MODE_BRIEF, ANSWER_MODE_NORMAL, ANSWER_MODE_DETAILED, ANSWER_MODE_REPORT})

PERCEPTION_STRATEGY_NONE = "none"
PERCEPTION_STRATEGY_LOCAL_WORKSPACE = "local_workspace"
PERCEPTION_STRATEGY_URL = "url"
PERCEPTION_STRATEGY_DELEGATED = "delegated"
PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED = "local_then_delegated"
PERCEPTION_STRATEGY_MIXED = "mixed"
PERCEPTION_STRATEGIES = frozenset(
    {
        PERCEPTION_STRATEGY_NONE,
        PERCEPTION_STRATEGY_LOCAL_WORKSPACE,
        PERCEPTION_STRATEGY_URL,
        PERCEPTION_STRATEGY_DELEGATED,
        PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED,
        PERCEPTION_STRATEGY_MIXED,
    }
)

EVIDENCE_REQUIREMENT_REQUIRED = "required"
EVIDENCE_REQUIREMENT_HELPFUL = "helpful"
EVIDENCE_REQUIREMENT_OPTIONAL = "optional"
EVIDENCE_REQUIREMENT_NOT_NEEDED = "not_needed"
EVIDENCE_REQUIREMENTS = frozenset(
    {
        EVIDENCE_REQUIREMENT_REQUIRED,
        EVIDENCE_REQUIREMENT_HELPFUL,
        EVIDENCE_REQUIREMENT_OPTIONAL,
        EVIDENCE_REQUIREMENT_NOT_NEEDED,
    }
)

RUNTIME_CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED = "local_then_delegated_if_insufficient"

DELEGATED_PLAN_DEFAULT_EXECUTOR_ID = "hermes"
DELEGATED_PLAN_DEFAULT_SCOPE = "read_only_investigation"
DELEGATED_PLAN_DEFAULT_PERMISSION_MODE = "read_only"
DELEGATED_PLAN_DEFAULT_EXPECTED_OUTPUT = "findings_sources_limitations"


@dataclass(frozen=True, slots=True)
class IntentPlannerIntent:
    intent_kind: str = INTENT_KIND_DECISION
    answer_mode: str = ANSWER_MODE_NORMAL

    def to_payload(self) -> dict[str, Any]:
        return {
            "intent_kind": self.intent_kind,
            "answer_mode": self.answer_mode,
        }


@dataclass(frozen=True, slots=True)
class PerceptionPlannerPlan:
    needs_perception: bool = False
    perception_strategy: str = PERCEPTION_STRATEGY_NONE
    evidence_requirement: str = EVIDENCE_REQUIREMENT_NOT_NEEDED
    workspace_plan: dict[str, Any] = field(default_factory=dict)
    url_plan: dict[str, Any] = field(default_factory=dict)
    delegated_plan: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "needs_perception": self.needs_perception,
            "perception_strategy": self.perception_strategy,
            "evidence_requirement": self.evidence_requirement,
            "workspace_plan": dict(self.workspace_plan),
            "url_plan": dict(self.url_plan),
            "delegated_plan": dict(self.delegated_plan),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class IntentPerceptionPlannerResult:
    intent: IntentPlannerIntent = field(default_factory=IntentPlannerIntent)
    perception_plan: PerceptionPlannerPlan = field(default_factory=PerceptionPlannerPlan)
    schema_version: str = INTENT_PERCEPTION_PLANNER_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent": self.intent.to_payload(),
            "perception_plan": self.perception_plan.to_payload(),
        }


def planner_result_from_semantic_payload(
    payload: Mapping[str, Any],
    *,
    user_input: str = "",
) -> IntentPerceptionPlannerResult:
    """Normalize nested planner output, falling back to legacy route fields."""

    intent_payload = _mapping(payload.get("intent"))
    plan_payload = _mapping(payload.get("perception_plan"))

    intent_kind = _normalize_intent_kind(str(intent_payload.get("intent_kind") or _intent_kind_from_legacy(payload)))
    answer_mode = _normalize_answer_mode(str(intent_payload.get("answer_mode") or payload.get("answer_mode") or ""))

    strategy = _normalize_perception_strategy(
        str(
            plan_payload.get("perception_strategy")
            or payload.get("perception_strategy")
            or payload.get("context_strategy")
            or ""
        )
    )
    legacy_needs = bool(
        _truthy(payload.get("needs_workspace_context"))
        or _truthy(payload.get("needs_url_context"))
        or _truthy(payload.get("needs_delegated_perception"))
        or strategy != PERCEPTION_STRATEGY_NONE
    )
    needs_perception = _truthy(plan_payload.get("needs_perception")) if "needs_perception" in plan_payload else legacy_needs
    if not needs_perception and strategy != PERCEPTION_STRATEGY_NONE:
        needs_perception = True

    evidence_requirement = _normalize_evidence_requirement(
        str(plan_payload.get("evidence_requirement") or payload.get("evidence_requirement") or "")
    )
    if evidence_requirement == EVIDENCE_REQUIREMENT_NOT_NEEDED and needs_perception:
        evidence_requirement = EVIDENCE_REQUIREMENT_HELPFUL

    workspace_plan = _plan_mapping(plan_payload.get("workspace_plan"))
    url_plan = _plan_mapping(plan_payload.get("url_plan"))
    delegated_plan = _plan_mapping(plan_payload.get("delegated_plan"))
    _merge_legacy_plan_fields(
        payload,
        workspace_plan=workspace_plan,
        url_plan=url_plan,
        delegated_plan=delegated_plan,
        user_input=user_input,
    )
    delegated_plan = _normalize_delegated_plan(
        delegated_plan,
        strategy=strategy,
        needs_perception=needs_perception,
        user_input=user_input,
        legacy_needs_delegated=_truthy(payload.get("needs_delegated_perception")),
    )

    reason = str(plan_payload.get("reason") or payload.get("perception_reason") or payload.get("reason") or "").strip()

    return IntentPerceptionPlannerResult(
        intent=IntentPlannerIntent(intent_kind=intent_kind, answer_mode=answer_mode),
        perception_plan=PerceptionPlannerPlan(
            needs_perception=needs_perception,
            perception_strategy=strategy if needs_perception else PERCEPTION_STRATEGY_NONE,
            evidence_requirement=evidence_requirement if needs_perception else EVIDENCE_REQUIREMENT_NOT_NEEDED,
            workspace_plan=workspace_plan,
            url_plan=url_plan,
            delegated_plan=delegated_plan,
            reason=reason,
        ),
    )


def runtime_context_strategy_for_perception_strategy(strategy: str) -> str:
    normalized = _normalize_perception_strategy(strategy)
    if normalized == PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED:
        return RUNTIME_CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED
    if normalized == PERCEPTION_STRATEGY_MIXED:
        return RUNTIME_CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED
    return normalized


def _merge_legacy_plan_fields(
    payload: Mapping[str, Any],
    *,
    workspace_plan: dict[str, Any],
    url_plan: dict[str, Any],
    delegated_plan: dict[str, Any],
    user_input: str,
) -> None:
    workspace_query = str(
        payload.get("workspace_query")
        or payload.get("workspace_context_query")
        or payload.get("repo_query")
        or ""
    ).strip()
    if workspace_query and not workspace_plan.get("query"):
        workspace_plan["query"] = workspace_query

    url_query = str(
        payload.get("url_query")
        or payload.get("linked_context_query")
        or payload.get("external_context_query")
        or ""
    ).strip()
    if url_query and not url_plan.get("query"):
        url_plan["query"] = url_query

    delegated_query = str(
        payload.get("delegated_perception_query")
        or payload.get("delegated_query")
        or payload.get("investigation_query")
        or payload.get("external_research_query")
        or ""
    ).strip()
    if delegated_query and not delegated_plan.get("query"):
        delegated_plan["query"] = delegated_query

    delegated_reason = str(payload.get("delegated_perception_reason") or payload.get("delegation_reason") or "").strip()
    if delegated_reason and not delegated_plan.get("reason"):
        delegated_plan["reason"] = delegated_reason

    capabilities = _strings(payload.get("suggested_capabilities"))
    if capabilities and not delegated_plan.get("requested_capabilities"):
        delegated_plan["requested_capabilities"] = capabilities

    if user_input and not workspace_plan and _truthy(payload.get("needs_workspace_context")):
        workspace_plan["query"] = user_input.strip()
    if user_input and not url_plan and _truthy(payload.get("needs_url_context")):
        url_plan["query"] = user_input.strip()
    if user_input and not delegated_plan and _truthy(payload.get("needs_delegated_perception")):
        delegated_plan["query"] = user_input.strip()


def _intent_kind_from_legacy(payload: Mapping[str, Any]) -> str:
    route = str(payload.get("route") or "").strip().lower()
    if route == "execution_request":
        return INTENT_KIND_EXECUTION_REQUEST
    if route == "follow_up":
        return INTENT_KIND_FOLLOW_UP
    if _truthy(payload.get("needs_delegated_perception")) or str(payload.get("context_strategy") or "").strip() == "delegated":
        return INTENT_KIND_INVESTIGATION_REQUEST
    return INTENT_KIND_DECISION


def _normalize_intent_kind(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in INTENT_KINDS else INTENT_KIND_DECISION


def _normalize_answer_mode(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in ANSWER_MODES else ANSWER_MODE_NORMAL


def _normalize_perception_strategy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "local_then_delegated_if_insufficient":
        return PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED
    return normalized if normalized in PERCEPTION_STRATEGIES else PERCEPTION_STRATEGY_NONE


def _normalize_evidence_requirement(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in EVIDENCE_REQUIREMENTS else EVIDENCE_REQUIREMENT_NOT_NEEDED


def _normalize_delegated_plan(
    value: Mapping[str, Any],
    *,
    strategy: str,
    needs_perception: bool,
    user_input: str,
    legacy_needs_delegated: bool,
) -> dict[str, Any]:
    plan = dict(value) if isinstance(value, Mapping) else {}
    uses_delegation = strategy in {
        PERCEPTION_STRATEGY_DELEGATED,
        PERCEPTION_STRATEGY_LOCAL_THEN_DELEGATED,
        PERCEPTION_STRATEGY_MIXED,
    } or legacy_needs_delegated
    if not needs_perception or not uses_delegation:
        return plan
    query = str(
        plan.get("query")
        or plan.get("delegated_perception_query")
        or plan.get("investigation_query")
        or user_input
        or ""
    ).strip()
    normalized = {
        "executor_id": str(plan.get("executor_id") or DELEGATED_PLAN_DEFAULT_EXECUTOR_ID).strip()
        or DELEGATED_PLAN_DEFAULT_EXECUTOR_ID,
        "scope": DELEGATED_PLAN_DEFAULT_SCOPE,
        "permission_mode": DELEGATED_PLAN_DEFAULT_PERMISSION_MODE,
        "query": query,
        "requested_capabilities": _strings(
            plan.get("requested_capabilities") or plan.get("suggested_capabilities")
        ),
        "expected_output": str(plan.get("expected_output") or DELEGATED_PLAN_DEFAULT_EXPECTED_OUTPUT).strip()
        or DELEGATED_PLAN_DEFAULT_EXPECTED_OUTPUT,
    }
    reason = str(plan.get("reason") or "").strip()
    if reason:
        normalized["reason"] = reason
    if plan.get("scope") and str(plan.get("scope")) != DELEGATED_PLAN_DEFAULT_SCOPE:
        normalized["normalized_from_scope"] = str(plan.get("scope"))
    if plan.get("permission_mode") and str(plan.get("permission_mode")) != DELEGATED_PLAN_DEFAULT_PERMISSION_MODE:
        normalized["normalized_from_permission_mode"] = str(plan.get("permission_mode"))
    return normalized


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _plan_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    if not isinstance(value, list):
        return result
    for item in value:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
