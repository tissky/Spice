from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from spice.llm.candidate_expander import build_candidate_expander_client
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.llm.util import extract_first_json_object
from spice.perception import extract_urls
from spice.runtime.continuation_resolver import ContinuationResolution
from spice.runtime.intent_perception_planner import (
    INTENT_KIND_DECISION,
    INTENT_KIND_EXECUTION_REQUEST,
    INTENT_KIND_FOLLOW_UP,
    INTENT_KIND_INVESTIGATION_REQUEST,
    IntentPerceptionPlannerResult,
    planner_result_from_semantic_payload,
    runtime_context_strategy_for_perception_strategy,
)


SEMANTIC_ROUTE_SCHEMA_VERSION = "spice.semantic_route.v1"

SEMANTIC_ROUTES = frozenset({"new_decision", "follow_up", "execution_request", "command"})
CONTEXT_STRATEGIES = frozenset(
    {
        "none",
        "local_workspace",
        "url",
        "delegated",
        "local_then_delegated_if_insufficient",
    }
)
SEMANTIC_ACTIONS = frozenset(
    {
        "new_intent",
        "choose_option",
        "execute_selected",
        "approve_execute",
        "approve_only",
        "answer_from_decision",
        "ask_clarifying_question",
        "compare_alternative",
        "explain_why_not",
        "refine",
        "refine_decision",
        "plan_candidate",
        "show_details",
        "skip",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticRoute:
    route: str
    action: str = "new_intent"
    is_continuation: bool = False
    candidate_id: str = ""
    label: str = ""
    text: str = ""
    context_strategy: str = "none"
    needs_workspace_context: bool = False
    workspace_query: str = ""
    needs_url_context: bool = False
    url_query: str = ""
    urls: list[str] = field(default_factory=list)
    needs_delegated_perception: bool = False
    delegated_perception_query: str = ""
    delegated_perception_reason: str = ""
    suggested_capabilities: list[str] = field(default_factory=list)
    intent_kind: str = INTENT_KIND_DECISION
    answer_mode: str = "normal"
    perception_plan: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    source: str = "runtime"
    schema_version: str = SEMANTIC_ROUTE_SCHEMA_VERSION
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "route": self.route,
            "action": self.action,
            "is_continuation": self.is_continuation,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "text": self.text,
            "context_strategy": self.context_strategy,
            "needs_workspace_context": self.needs_workspace_context,
            "workspace_query": self.workspace_query,
            "needs_url_context": self.needs_url_context,
            "url_query": self.url_query,
            "urls": list(self.urls),
            "needs_delegated_perception": self.needs_delegated_perception,
            "delegated_perception_query": self.delegated_perception_query,
            "delegated_perception_reason": self.delegated_perception_reason,
            "suggested_capabilities": list(self.suggested_capabilities),
            "intent_kind": self.intent_kind,
            "answer_mode": self.answer_mode,
            "perception_plan": dict(self.perception_plan),
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "raw": dict(self.raw),
        }


def route_semantic_input(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
) -> SemanticRoute:
    """Return the no-LLM route for natural input.

    Slash commands are handled before this layer by the deterministic command
    router. Non-command natural language requires semantic routing; without an
    LLM route we conservatively treat it as a fresh decision.
    """

    text = user_input.strip()
    urls = extract_urls(text)
    return SemanticRoute(
        route="new_decision",
        action="new_intent",
        is_continuation=False,
        text=text,
        context_strategy="url" if urls else "none",
        needs_workspace_context=False,
        workspace_query="",
        needs_url_context=bool(urls),
        url_query=text if urls else "",
        urls=urls,
        reason=(
            "Natural follow-up routing requires the LLM semantic router; "
            "deterministic routing only handles slash commands."
        )
        if text and isinstance(active_frame, Mapping) and active_frame
        else "",
        source="none",
    )


def route_semantic_input_from_runtime_config(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None,
) -> SemanticRoute:
    if not _llm_semantic_routing_enabled(config):
        return route_semantic_input(user_input, active_frame)

    payload = _mapping(config)
    provider_id = str(payload.get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(payload)
    if not model_id:
        return route_semantic_input(user_input, active_frame)
    try:
        client = build_candidate_expander_client(
            provider_id=provider_id,
            model_id=model_id,
        )
        return route_semantic_input_with_llm(
            user_input,
            active_frame,
            client=client,
            model_provider=provider_id,
            model_id=model_id,
        )
    except Exception:
        return route_semantic_input(user_input, active_frame)


def route_semantic_input_with_llm(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
    *,
    client: LLMClient,
    model_provider: str = "",
    model_id: str = "",
) -> SemanticRoute:
    text = user_input.strip()
    if not text:
        return SemanticRoute(
            route="new_decision",
            action="new_intent",
            is_continuation=False,
            text=text,
            source="none",
        )
    frame = active_frame if isinstance(active_frame, Mapping) else {}

    response = client.generate(
        LLMRequest(
            task_hook=LLMTaskHook.DECISION_PROPOSE,
            input_text=_semantic_router_prompt(text, frame),
            system_text=_semantic_router_system_prompt(),
            response_format_hint="json_object",
            temperature=0.0,
            max_tokens=800,
            timeout_sec=20.0,
            metadata={
                "purpose": "semantic_route",
                "model_provider": model_provider,
                "model_id": model_id,
            },
        )
    )
    payload = _parse_payload(response.output_text)
    return _route_from_llm_payload(text, frame, payload)


def semantic_route_from_continuation(
    resolution: ContinuationResolution,
    *,
    source: str = "deterministic",
) -> SemanticRoute:
    route = _route_for_action(resolution.action, resolution.is_continuation)
    return SemanticRoute(
        route=route,
        action=resolution.action,
        is_continuation=resolution.is_continuation,
        candidate_id=resolution.candidate_id,
        label=resolution.label,
        text=resolution.text,
        needs_workspace_context=resolution.needs_workspace_context,
        workspace_query=resolution.workspace_query,
        needs_url_context=resolution.needs_url_context,
        url_query=resolution.url_query,
        urls=list(resolution.urls),
        context_strategy=getattr(resolution, "context_strategy", "none"),
        needs_delegated_perception=getattr(resolution, "needs_delegated_perception", False),
        delegated_perception_query=getattr(resolution, "delegated_perception_query", ""),
        delegated_perception_reason=getattr(resolution, "delegated_perception_reason", ""),
        suggested_capabilities=list(getattr(resolution, "suggested_capabilities", [])),
        confidence=1.0 if resolution.is_continuation else 0.0,
        reason=resolution.reason,
        source=source,
    )


def semantic_route_to_continuation(route: SemanticRoute) -> ContinuationResolution:
    return ContinuationResolution(
        route.is_continuation,
        action=route.action if route.is_continuation else "new_intent",
        candidate_id=route.candidate_id,
        label=route.label,
        text=route.text,
        needs_workspace_context=route.needs_workspace_context,
        workspace_query=route.workspace_query,
        needs_url_context=route.needs_url_context,
        url_query=route.url_query,
        urls=list(route.urls),
        context_strategy=route.context_strategy,
        needs_delegated_perception=route.needs_delegated_perception,
        delegated_perception_query=route.delegated_perception_query,
        delegated_perception_reason=route.delegated_perception_reason,
        suggested_capabilities=list(route.suggested_capabilities),
        reason=route.reason,
    )


def _route_for_action(action: str, is_continuation: bool) -> str:
    if not is_continuation or action == "new_intent":
        return "new_decision"
    if action in {"execute_selected", "approve_execute"}:
        return "execution_request"
    return "follow_up"


def _semantic_router_system_prompt() -> str:
    return (
        "You are Spice's combined intent and perception planner. Classify a user message "
        "relative to the active Decision Card and decide what evidence tier is needed. "
        "Return only one JSON object. Do not answer the user. Do not make a decision. "
        "Do not execute. For context access, choose one perception strategy and provide "
        "only natural-language queries. Never choose files, paths, tools, or commands. "
        "Be conservative: unrelated requests are new_decision."
    )


def _semantic_router_prompt(user_input: str, active_frame: Mapping[str, Any]) -> str:
    payload = {
        "task": "Classify the user's next message for the Spice runtime.",
        "routes": {
            "new_decision": "The user is asking a fresh question or starting a new decision.",
            "follow_up": "The user continues the active Decision Card without asking for execution.",
            "execution_request": "The user asks to execute or start implementing the active selection.",
        },
        "allowed_actions": sorted(SEMANTIC_ACTIONS),
        "rules": [
            "Use choose_option when the user chooses a visible option by label, ordinal, title, or meaning.",
            "Use execute_selected when the user says to do it, start, implement, make it happen, or equivalent.",
            "If execute_selected targets a visible label or candidate, include candidate_id or candidate_label.",
            "Use approve_execute only when an approval is already attached and the user approves execution.",
            "Use explain_why_not when the user asks why a visible alternative was not selected.",
            "Use plan_candidate when the user asks for a plan for a visible option.",
            "Use compare_alternative when the user asks whether a visible alternative could be better.",
            "Use answer_from_decision when the user asks a practical follow-up about the selected decision, such as timeline, solo version, MVP, next steps, or tradeoffs.",
            "Use ask_clarifying_question only when the follow-up cannot be answered from the active Decision Card.",
            "Use refine_decision when the user asks to change, rerank, or regenerate the current Decision Card.",
            "Use refine for backward compatibility when the user asks to change or improve the current Decision Card.",
            "Use show_details when the user asks why, details, simulations, or the card contents.",
            "Use new_intent for unrelated new requests.",
            "Never invent candidate ids. Use only visible candidates from active_decision_frame.",
            "Choose one primary context_strategy: none, local_workspace, url, delegated, or local_then_delegated_if_insufficient.",
            "Use context_strategy=local_workspace for current repo, current implementation, local code, or workspace facts.",
            "Use context_strategy=url when the user provides explicit URL links. URL context has priority over delegated perception for linked content.",
            "Use context_strategy=delegated when the user asks for web/latest/deep external research, asks Hermes/Codex to investigate, or needs capabilities beyond local/url perception.",
            "Use context_strategy=local_then_delegated_if_insufficient only when local repo perception should run first and delegation is only an escalation if local evidence is insufficient.",
            "Use context_strategy=none for abstract prioritization, simple follow-ups answerable from the Decision Card, and execution/approval phrases.",
            "Set needs_workspace_context=true only when answering well requires current repo/workspace facts.",
            "Examples that need workspace context: 'look at the repo', 'based on code', 'what is implemented now', 'where is this module', 'what is missing in the current implementation'.",
            "Examples that usually do not need workspace context: abstract prioritization, choosing among visible options, approval/execution phrases, or simple why/plan follow-ups answerable from the Decision Card.",
            "If needs_workspace_context=true, workspace_query must be a short natural-language information need.",
            "If the user includes URL links, set needs_url_context=true and url_query to what should be learned from those links.",
            "Set needs_delegated_perception=true only for delegated or local_then_delegated_if_insufficient strategies.",
            "If needs_delegated_perception=true, delegated_perception_query must describe the read-only investigation need.",
            "suggested_capabilities may include web_research, repo_inspection, browser_research, docs_review, or external_search.",
            "Do not fetch, quote, or summarize external links yourself. The runtime URL perception step reads them.",
            "Do not output file paths, tool calls, grep patterns, commands, or read plans. The runtime controlled loop decides what to inspect.",
        ],
        "response_schema": {
            "intent": {
                "intent_kind": "decision | follow_up | execution_request | investigation_request",
                "answer_mode": "brief | normal | detailed | report",
            },
            "perception_plan": {
                "needs_perception": "boolean",
                "perception_strategy": "local_workspace | url | delegated | local_then_delegated | mixed | none",
                "evidence_requirement": "required | helpful | optional | not_needed",
                "workspace_plan": {"query": "short natural-language workspace evidence need"},
                "url_plan": {"query": "short natural-language linked-content evidence need"},
                "delegated_plan": {
                    "executor_id": "optional executor id such as hermes",
                    "scope": "read_only_investigation",
                    "query": "short natural-language investigation need",
                    "requested_capabilities": ["web_research", "repo_inspection"],
                    "expected_output": "findings_sources_limitations",
                },
                "reason": "short reason for the perception strategy",
            },
            "route": "new_decision | follow_up | execution_request",
            "action": "one allowed action",
            "is_continuation": "boolean",
            "candidate_id": "visible candidate id when action=choose_option, explain_why_not, plan_candidate, compare_alternative, or targeted execute_selected",
            "candidate_label": "visible label like A/B/C when known",
            "refinement": "text when action=refine",
            "context_strategy": "none | local_workspace | url | delegated | local_then_delegated_if_insufficient",
            "needs_workspace_context": "boolean; true only if read-only workspace perception is needed",
            "workspace_query": "short natural-language repo/workspace information need; no paths/tools/commands",
            "needs_url_context": "boolean; true when explicit URL links should be read",
            "url_query": "short natural-language information need for linked content",
            "needs_delegated_perception": "boolean; true only if a read-only executor investigation should be proposed",
            "delegated_perception_query": "short natural-language investigation need",
            "delegated_perception_reason": "short reason why local/url perception is not enough",
            "suggested_capabilities": "legacy compatibility list of capability names such as web_research or repo_inspection",
            "confidence": "number from 0 to 1",
            "reason": "short reason",
        },
        "user_input": user_input,
        "active_decision_frame": _compact_frame_for_llm(active_frame),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _compact_frame_for_llm(active_frame: Mapping[str, Any]) -> dict[str, Any]:
    selected = _mapping(active_frame.get("selected"))
    return {
        "decision_id": str(active_frame.get("decision_id") or ""),
        "selected_candidate_id": str(active_frame.get("selected_candidate_id") or ""),
        "approval_id": str(active_frame.get("approval_id") or ""),
        "selected": _compact_candidate_for_llm(selected),
        "candidates": [
            _compact_candidate_for_llm(_mapping(candidate))
            for candidate in _list(active_frame.get("candidates"))[:6]
        ],
        "allowed_continuations": _list(active_frame.get("allowed_continuations"))[:8],
    }


def _compact_candidate_for_llm(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(candidate.get("label") or ""),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "title": str(candidate.get("title") or ""),
        "recommendation": str(
            candidate.get("recommended_action")
            or candidate.get("recommendation")
            or candidate.get("intent")
            or ""
        ),
        "executor_task": str(candidate.get("executor_task") or ""),
        "is_selected": bool(candidate.get("is_selected")),
    }


def _route_from_llm_payload(
    text: str,
    active_frame: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> SemanticRoute:
    planner = planner_result_from_semantic_payload(payload, user_input=text)
    route = _normalize_route(str(payload.get("route") or _route_from_planner_intent(planner.intent.intent_kind)))
    action = _normalize_action(str(payload.get("action") or _action_from_planner_intent(planner.intent.intent_kind)))
    is_continuation = (
        _truthy(payload.get("is_continuation"))
        if "is_continuation" in payload
        else _is_continuation_from_planner_intent(planner.intent.intent_kind)
    )
    reason = str(payload.get("reason") or "").strip()
    confidence = _confidence(payload.get("confidence"))
    context = _context_request(payload, text, planner=planner)
    context_strategy = context["context_strategy"]
    needs_workspace_context = bool(context["needs_workspace_context"])
    workspace_query = str(context["workspace_query"])
    needs_url_context = bool(context["needs_url_context"])
    url_query = str(context["url_query"])
    urls = list(context["urls"])
    needs_delegated_perception = bool(context["needs_delegated_perception"])
    delegated_perception_query = str(context["delegated_perception_query"])
    delegated_perception_reason = str(context["delegated_perception_reason"])
    suggested_capabilities = list(context["suggested_capabilities"])
    perception_plan_payload = planner.perception_plan.to_payload()
    has_active_frame = bool(active_frame)

    if (
        route == "new_decision"
        or action == "new_intent"
        or not is_continuation
        or (not has_active_frame and route != "new_decision")
    ):
        return SemanticRoute(
            route="new_decision",
            action="new_intent",
            is_continuation=False,
            text=text,
            context_strategy=context_strategy,
            needs_workspace_context=needs_workspace_context,
            workspace_query=workspace_query,
            needs_url_context=needs_url_context,
            url_query=url_query,
            urls=urls,
            needs_delegated_perception=needs_delegated_perception,
            delegated_perception_query=delegated_perception_query,
            delegated_perception_reason=delegated_perception_reason,
            suggested_capabilities=suggested_capabilities,
            intent_kind=planner.intent.intent_kind,
            answer_mode=planner.intent.answer_mode,
            perception_plan=perception_plan_payload,
            confidence=confidence,
            reason=reason,
            source="llm",
            raw=dict(payload),
        )

    if action in {"choose_option", "explain_why_not", "plan_candidate", "compare_alternative"}:
        candidate = _candidate_for_payload(active_frame, payload)
        if candidate is None:
            return SemanticRoute(
                route="new_decision",
                action="new_intent",
                is_continuation=False,
                text=text,
                context_strategy=context_strategy,
                needs_workspace_context=needs_workspace_context,
                workspace_query=workspace_query,
                needs_url_context=needs_url_context,
                url_query=url_query,
                urls=urls,
                needs_delegated_perception=needs_delegated_perception,
                delegated_perception_query=delegated_perception_query,
                delegated_perception_reason=delegated_perception_reason,
                suggested_capabilities=suggested_capabilities,
                intent_kind=planner.intent.intent_kind,
                answer_mode=planner.intent.answer_mode,
                perception_plan=perception_plan_payload,
                confidence=confidence,
                reason="LLM chose an unknown candidate.",
                source="llm",
                raw=dict(payload),
            )
        return SemanticRoute(
            route="follow_up",
            action=action,
            is_continuation=True,
            candidate_id=str(candidate.get("candidate_id") or ""),
            label=str(candidate.get("label") or payload.get("candidate_label") or ""),
            text=text,
            context_strategy=context_strategy,
            needs_workspace_context=needs_workspace_context,
            workspace_query=workspace_query,
            needs_url_context=needs_url_context,
            url_query=url_query,
            urls=urls,
            needs_delegated_perception=needs_delegated_perception,
            delegated_perception_query=delegated_perception_query,
            delegated_perception_reason=delegated_perception_reason,
            suggested_capabilities=suggested_capabilities,
            intent_kind=planner.intent.intent_kind,
            answer_mode=planner.intent.answer_mode,
            perception_plan=perception_plan_payload,
            confidence=confidence,
            reason=reason or "LLM matched the message to a visible option.",
            source="llm",
            raw=dict(payload),
        )

    selected = _mapping(active_frame.get("selected"))
    if action == "execute_selected":
        candidate = _candidate_for_payload(active_frame, payload)
        if candidate is not None:
            return SemanticRoute(
                route="execution_request",
                action=action,
                is_continuation=True,
                candidate_id=str(candidate.get("candidate_id") or ""),
                label=str(candidate.get("label") or payload.get("candidate_label") or ""),
                text=text,
                context_strategy=context_strategy,
                needs_workspace_context=needs_workspace_context,
                workspace_query=workspace_query,
                needs_url_context=needs_url_context,
                url_query=url_query,
                urls=urls,
                needs_delegated_perception=needs_delegated_perception,
                delegated_perception_query=delegated_perception_query,
                delegated_perception_reason=delegated_perception_reason,
                suggested_capabilities=suggested_capabilities,
                intent_kind=planner.intent.intent_kind,
                answer_mode=planner.intent.answer_mode,
                perception_plan=perception_plan_payload,
                confidence=confidence,
                reason=reason or "LLM matched the execution request to a visible option.",
                source="llm",
                raw=dict(payload),
            )
    selected_id = str(selected.get("candidate_id") or active_frame.get("selected_candidate_id") or "")
    if action in {"refine", "refine_decision"}:
        routed_text = str(payload.get("refinement") or text).strip()
    else:
        routed_text = text
    return SemanticRoute(
        route=_route_for_action(action, True),
        action=action,
        is_continuation=True,
        candidate_id=selected_id,
        label=str(selected.get("label") or ""),
        text=routed_text,
        context_strategy=context_strategy,
        needs_workspace_context=needs_workspace_context,
        workspace_query=workspace_query,
        needs_url_context=needs_url_context,
        url_query=url_query,
        urls=urls,
        needs_delegated_perception=needs_delegated_perception,
        delegated_perception_query=delegated_perception_query,
        delegated_perception_reason=delegated_perception_reason,
        suggested_capabilities=suggested_capabilities,
        intent_kind=planner.intent.intent_kind,
        answer_mode=planner.intent.answer_mode,
        perception_plan=perception_plan_payload,
        confidence=confidence,
        reason=reason or "LLM classified the message relative to the active Decision Card.",
        source="llm",
        raw=dict(payload),
    )


def _candidate_for_payload(
    active_frame: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidate_id = str(payload.get("candidate_id") or "").strip()
    candidate_label = str(payload.get("candidate_label") or payload.get("label") or "").strip()
    for candidate in _list(active_frame.get("candidates")):
        item = _mapping(candidate)
        if candidate_id and str(item.get("candidate_id") or "") == candidate_id:
            return item
        if candidate_label and str(item.get("label") or "").upper() == candidate_label.upper():
            return item
    return None


def _parse_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(stripped)
        payload = json.loads(extracted) if extracted else None
    return payload if isinstance(payload, dict) else {}


def _context_request(
    payload: Mapping[str, Any],
    user_input: str,
    *,
    planner: IntentPerceptionPlannerResult | None = None,
) -> dict[str, Any]:
    urls = extract_urls(user_input)
    planner_plan = planner.perception_plan if planner is not None else None
    planner_strategy = (
        runtime_context_strategy_for_perception_strategy(planner_plan.perception_strategy)
        if planner_plan is not None
        else ""
    )
    if planner_strategy == "none":
        planner_strategy = ""
    strategy = _normalize_context_strategy(str(payload.get("context_strategy") or planner_strategy or ""))
    legacy_workspace = _truthy(payload.get("needs_workspace_context"))
    legacy_url = bool(urls) or _truthy(payload.get("needs_url_context"))
    legacy_delegated = _truthy(payload.get("needs_delegated_perception")) or _truthy(
        payload.get("delegate_perception")
    )
    workspace_plan = planner_plan.workspace_plan if planner_plan is not None else {}
    url_plan = planner_plan.url_plan if planner_plan is not None else {}
    delegated_plan = planner_plan.delegated_plan if planner_plan is not None else {}

    if urls:
        strategy = "url"
    elif not strategy:
        if legacy_delegated:
            strategy = "delegated"
        elif legacy_workspace:
            strategy = "local_workspace"
        elif legacy_url:
            strategy = "url"
        else:
            strategy = "none"

    workspace_query = str(
        payload.get("workspace_query")
        or payload.get("workspace_context_query")
        or payload.get("repo_query")
        or workspace_plan.get("query")
        or workspace_plan.get("workspace_query")
        or ""
    ).strip()
    url_query = str(
        payload.get("url_query")
        or payload.get("linked_context_query")
        or payload.get("external_context_query")
        or url_plan.get("query")
        or url_plan.get("url_query")
        or ""
    ).strip()
    delegated_query = str(
        payload.get("delegated_perception_query")
        or payload.get("delegated_query")
        or payload.get("investigation_query")
        or payload.get("external_research_query")
        or delegated_plan.get("query")
        or delegated_plan.get("delegated_perception_query")
        or ""
    ).strip()
    delegated_reason = str(
        payload.get("delegated_perception_reason")
        or payload.get("delegation_reason")
        or delegated_plan.get("reason")
        or (planner_plan.reason if planner_plan is not None else "")
        or ""
    ).strip()
    suggested_capabilities = _strings(payload.get("suggested_capabilities")) or _strings(
        delegated_plan.get("requested_capabilities")
    )

    needs_workspace = strategy in {
        "local_workspace",
        "local_then_delegated_if_insufficient",
    }
    needs_url = strategy == "url"
    needs_delegated = strategy in {
        "delegated",
        "local_then_delegated_if_insufficient",
    }

    if needs_workspace and not workspace_query:
        workspace_query = user_input.strip()
    if needs_url and not url_query:
        url_query = user_input.strip()
    if needs_delegated and not delegated_query:
        delegated_query = user_input.strip()
    if needs_delegated and not suggested_capabilities:
        suggested_capabilities = ["web_research"] if strategy == "delegated" else ["repo_inspection"]

    return {
        "context_strategy": strategy,
        "needs_workspace_context": needs_workspace,
        "workspace_query": workspace_query if needs_workspace else "",
        "needs_url_context": needs_url,
        "url_query": url_query if needs_url else "",
        "urls": urls if needs_url else [],
        "needs_delegated_perception": needs_delegated,
        "delegated_perception_query": delegated_query if needs_delegated else "",
        "delegated_perception_reason": delegated_reason if needs_delegated else "",
        "suggested_capabilities": suggested_capabilities if needs_delegated else [],
    }


def _route_from_planner_intent(intent_kind: str) -> str:
    if intent_kind == INTENT_KIND_EXECUTION_REQUEST:
        return "execution_request"
    if intent_kind == INTENT_KIND_FOLLOW_UP:
        return "follow_up"
    return "new_decision"


def _action_from_planner_intent(intent_kind: str) -> str:
    if intent_kind == INTENT_KIND_EXECUTION_REQUEST:
        return "execute_selected"
    if intent_kind == INTENT_KIND_FOLLOW_UP:
        return "answer_from_decision"
    if intent_kind == INTENT_KIND_INVESTIGATION_REQUEST:
        return "new_intent"
    return "new_intent"


def _is_continuation_from_planner_intent(intent_kind: str) -> bool:
    return intent_kind in {INTENT_KIND_FOLLOW_UP, INTENT_KIND_EXECUTION_REQUEST}


def _normalize_route(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in SEMANTIC_ROUTES:
        return normalized
    return "new_decision"


def _normalize_action(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in SEMANTIC_ACTIONS:
        return normalized
    return "new_intent"


def _normalize_context_strategy(value: str) -> str:
    normalized = value.strip().lower()
    return normalized if normalized in CONTEXT_STRATEGIES else ""


def _llm_semantic_routing_enabled(config: Mapping[str, Any] | None) -> bool:
    payload = _mapping(config)
    provider_id = str(payload.get("llm_provider") or "deterministic").strip()
    if provider_id == "deterministic":
        return False
    configured = payload.get("llm_semantic_routing")
    if configured is not None:
        return _truthy(configured)
    return True


def _runtime_model_id(config: Mapping[str, Any]) -> str:
    configured = str(config.get("llm_model") or "").strip()
    if configured:
        return configured
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    if provider_id == "deterministic":
        return "deterministic.v1"
    return ""


def _confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in _list(value):
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
