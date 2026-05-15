from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping

from spice.llm.candidate_expander import build_candidate_expander_client
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.runtime.composer_context import compact_composer_context
from spice.runtime.composer_parser import parse_composer_response_text
from spice.runtime.composer_prompt import build_slim_composer_prompt_payload, slim_recent_context
from spice.runtime.composer_result import ComposerResult
from spice.runtime.composer_streaming import (
    ComposerStreamError,
    generate_or_stream_composer_output,
    mark_streaming_invalid,
    mark_streaming_valid,
)
from spice.runtime.composer_workspace_validator import (
    WORKSPACE_COMPOSER_CONSTRAINTS,
    validate_workspace_claims,
)
from spice.runtime.decision_brief import render_decision_brief
from spice.runtime.response_depth import resolve_response_depth_budget


DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION = "spice.decision_response_composer.v1"
DecisionResponseComposeResult = ComposerResult


def compose_decision_response_from_runtime_config(
    *,
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
    context_payload: Mapping[str, Any] | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> DecisionResponseComposeResult:
    brief = _mapping(artifact.get("decision_brief"))
    deterministic_text = render_decision_brief(brief)
    facts = response_composer_facts(artifact, context_payload=context_payload)
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(config)
    if provider_id in {"", "deterministic"}:
        return DecisionResponseComposeResult(
            enabled=False,
            status="disabled",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="decision_response",
            model_provider=provider_id or "deterministic",
            model_id=model_id,
            facts=facts,
            metadata={
                "reason": "deterministic provider",
                "composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
            },
        )
    if not model_id:
        return DecisionResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="decision_response",
            model_provider=provider_id,
            error="llm_model is required for response composition.",
            fallback_reason="missing_model",
            facts=facts,
            metadata={"composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION},
        )

    try:
        client = build_candidate_expander_client(provider_id=provider_id, model_id=model_id)
        return compose_decision_response_with_llm(
            client=client,
            artifact=artifact,
            deterministic_text=deterministic_text,
            model_provider=provider_id,
            model_id=model_id,
            context_payload=context_payload,
            stream_callback=stream_callback,
            config=config,
        )
    except Exception as exc:
        return DecisionResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="decision_response",
            model_provider=provider_id,
            model_id=model_id,
            error=str(exc),
            fallback_reason="client_error",
            facts=facts,
            metadata={"composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION},
        )


def compose_decision_response_with_llm(
    *,
    client: LLMClient,
    artifact: Mapping[str, Any],
    deterministic_text: str,
    model_provider: str = "",
    model_id: str = "",
    context_payload: Mapping[str, Any] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    config: Mapping[str, Any] | None = None,
) -> DecisionResponseComposeResult:
    facts = response_composer_facts(artifact, context_payload=context_payload)
    depth = _resolve_response_depth_for_facts(
        facts,
        config=config,
        composer_kind="decision_response",
    )
    facts = {**facts, "response_depth": depth.to_payload()}
    request = LLMRequest(
        task_hook=LLMTaskHook.RESPONSE_COMPOSE,
        input_text=_response_composer_prompt(facts),
        system_text=_response_composer_system_prompt(),
        response_format_hint="",
        temperature=0.4,
        max_tokens=depth.max_tokens,
        timeout_sec=depth.timeout_sec,
        metadata={
            "purpose": "decision_response_composition",
            "model_provider": model_provider,
            "model_id": model_id,
            "decision_id": facts.get("decision_id"),
            "selected_candidate_id": _mapping(facts.get("selected")).get("candidate_id"),
            "response_depth": depth.to_payload(),
        },
    )
    try:
        output = generate_or_stream_composer_output(
            client=client,
            request=request,
            stream_callback=stream_callback,
        )
    except ComposerStreamError as exc:
        return DecisionResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="decision_response",
            model_provider=model_provider,
            model_id=model_id,
            error=str(exc),
            raw_output=exc.raw_output,
            fallback_reason="stream_error",
            facts=facts,
            metadata=mark_streaming_invalid(
                {
                    **exc.metadata,
                    "fallback_reason": "stream_error",
                    "composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
                    "response_depth": depth.to_payload(),
                },
                reason="stream_error",
            ),
        )
    raw_output = output.raw_output
    try:
        text = parse_composer_response_text(raw_output, max_chars=depth.max_chars)
        _validate_composed_response(text, facts)
    except Exception as exc:
        return DecisionResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="decision_response",
            model_provider=model_provider or output.provider_id,
            model_id=model_id or output.model_id,
            request_id=output.request_id,
            error=str(exc),
            raw_output=raw_output,
            fallback_reason="invalid_composed_response",
            facts=facts,
            metadata=mark_streaming_invalid(
                {
                    **output.metadata,
                    "fallback_reason": "invalid_composed_response",
                    "composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
                    "response_depth": depth.to_payload(),
                },
                reason="invalid_composed_response",
            ),
        )
    return DecisionResponseComposeResult(
        enabled=True,
        status="composed",
        response_text=text,
        deterministic_text=deterministic_text,
        composer_kind="decision_response",
        model_provider=model_provider or output.provider_id,
        model_id=model_id or output.model_id,
        request_id=output.request_id,
        raw_output=raw_output,
        facts=facts,
        metadata=mark_streaming_valid(
            {
                **output.metadata,
                "facts_schema": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
                "composer_schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
                "selected_candidate_id": _mapping(facts.get("selected")).get("candidate_id", ""),
                "response_depth": depth.to_payload(),
            }
        ),
    )


def response_composer_facts(
    artifact: Mapping[str, Any],
    *,
    context_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    brief = _mapping(artifact.get("decision_brief"))
    compare = _mapping(artifact.get("compare_payload"))
    selected = _mapping(brief.get("selected"))
    selected_candidate = _candidate_by_id(
        _list_of_mappings(compare.get("candidate_decisions")),
        str(selected.get("candidate_id") or artifact.get("selected_candidate_id") or ""),
    )
    execution = _mapping(brief.get("execution"))
    return {
        "schema_version": DECISION_RESPONSE_COMPOSER_SCHEMA_VERSION,
        "decision_id": str(artifact.get("decision_id") or brief.get("decision_id") or ""),
        "run_id": str(artifact.get("run_id") or brief.get("run_id") or ""),
        "display_language": str(artifact.get("display_language") or brief.get("display_language") or "en"),
        "selected": selected,
        "why_this_won": _string_list(brief.get("why_this_won")),
        "why_not_others": _why_not_summaries(compare),
        "simulation": _compact_simulation(_mapping(selected_candidate.get("simulation"))),
        "execution": execution,
        "execution_capability_note": str(execution.get("capability_summary") or ""),
        "approval_id": str(artifact.get("approval_id") or execution.get("approval_id") or ""),
        "execution_affordance": _mapping(selected_candidate.get("execution_affordance")),
        "allowed_next_actions": _string_list(brief.get("next_actions")),
        "warnings": [],
        "decision_context": compact_composer_context(context_payload),
    }


def _response_composer_system_prompt() -> str:
    return (
        "You are Spice's response composer. Do one thing: write a natural user-facing "
        "response from finalized decision facts. Do not change the selected option. "
        "Do not re-select the winner, change scores, change approval or execution state, "
        "invent artifact ids, or expose raw JSON/schema. Return only the response."
    )


def _response_composer_prompt(facts: Mapping[str, Any]) -> str:
    return json.dumps(
        build_slim_composer_prompt_payload(
            task="Write the natural response for this already-finalized Spice decision. Do not make a new decision.",
            facts=_slim_response_prompt_facts(facts),
            tone="Capable, direct agent voice. Lead with the recommendation and one main tradeoff.",
            extra_constraints=(
                "Do not claim execution happened.",
                "Do not say approval is created unless approval/execution state says approval_pending.",
                "Mention execution capability only briefly when it affects approval or execution readiness.",
                "Use recent_context only for continuity; do not quote it wholesale.",
                "Match response_depth guidance and stay under response_depth.max_chars.",
                *WORKSPACE_COMPOSER_CONSTRAINTS,
            ),
        ),
        ensure_ascii=False,
        sort_keys=True,
    )


def _slim_response_prompt_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selected_candidate": _mapping(facts.get("selected")),
        "why_won": _string_list(facts.get("why_this_won"))[:3],
        "why_not": _list_of_mappings(facts.get("why_not_others"))[:3],
        "simulation": _mapping(facts.get("simulation")),
        "execution_affordance": _compact_execution_affordance_for_prompt(
            _mapping(facts.get("execution_affordance")) or _mapping(facts.get("execution"))
        ),
        "execution_capability_note": str(facts.get("execution_capability_note") or ""),
        "approval_execution_state": _mapping(facts.get("execution")),
        "allowed_next_actions": _string_list(facts.get("allowed_next_actions"))[:6],
        "recent_context": slim_recent_context(_mapping(facts.get("decision_context"))),
        "response_depth": _compact_response_depth(_mapping(facts.get("response_depth"))),
    }


def _resolve_response_depth_for_facts(
    facts: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    composer_kind: str,
) -> Any:
    context = _mapping(facts.get("decision_context"))
    evidence_context = _mapping(context.get("evidence_context"))
    requirements = _mapping(evidence_context.get("requirements"))
    current_intent = _mapping(context.get("current_intent"))
    return resolve_response_depth_budget(
        answer_mode=str(requirements.get("answer_mode") or ""),
        evidence_domain=str(requirements.get("evidence_domain") or ""),
        evidence_context=evidence_context,
        user_input=str(current_intent.get("text") or ""),
        config=config,
        composer_kind=composer_kind,
    )


def _compact_response_depth(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answer_mode": str(payload.get("answer_mode") or ""),
        "max_chars": payload.get("max_chars"),
        "max_tokens": payload.get("max_tokens"),
        "native": bool(payload.get("native")),
        "guidance": _response_depth_guidance(str(payload.get("answer_mode") or "")),
    }


def _response_depth_guidance(answer_mode: str) -> str:
    if answer_mode == "brief":
        return "Keep this compact."
    if answer_mode == "detailed":
        return "Give concrete reasoning, steps, and caveats."
    if answer_mode == "report":
        return "Use evidence, sources, limitations, and tradeoffs."
    if answer_mode == "native":
        return "Use the model-native budget while staying factual."
    return "Give a normal conversational answer with enough context to act."


def _compact_execution_affordance_for_prompt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    capability = _mapping(payload.get("capability"))
    executor = _mapping(payload.get("executor"))
    permission = _mapping(payload.get("permission"))
    approval = _mapping(payload.get("approval"))
    result: dict[str, Any] = {
        "status": str(payload.get("status") or ""),
        "summary": str(payload.get("summary") or ""),
        "candidate_executable": bool(payload.get("candidate_executable")),
        "executable": bool(payload.get("executable")),
        "blocked": bool(payload.get("blocked")),
        "blocked_reason": str(payload.get("blocked_reason") or ""),
        "required_capability": str(payload.get("required_capability") or capability.get("required_capability") or ""),
        "executor_capability_source": str(
            payload.get("executor_capability_source") or capability.get("source") or ""
        ),
    }
    if capability:
        result["capability"] = {
            "required_capability": str(capability.get("required_capability") or ""),
            "executor_has_required_capability": bool(capability.get("executor_has_required_capability")),
            "matched_capability": str(capability.get("matched_capability") or ""),
            "simulates_required_capability": bool(capability.get("simulates_required_capability")),
            "source": str(capability.get("source") or ""),
            "status": str(capability.get("status") or ""),
        }
    if executor:
        result["executor"] = {
            "executor_id": str(executor.get("executor_id") or ""),
            "status": str(executor.get("status") or ""),
            "real_executor": bool(executor.get("real_executor")),
        }
    if permission:
        result["permission"] = {
            "required": str(permission.get("required") or ""),
            "configured": str(permission.get("configured") or ""),
            "escalation_required": bool(permission.get("escalation_required")),
            "escalation_supported": bool(permission.get("escalation_supported")),
        }
    if approval:
        result["approval"] = {
            "required": bool(approval.get("required")),
            "eligible_for_approval": bool(approval.get("eligible_for_approval")),
            "status": str(approval.get("status") or ""),
        }
    return result


def _validate_composed_response(text: str, facts: Mapping[str, Any]) -> None:
    if not text:
        raise ValueError("response composer returned empty response")
    execution = _mapping(facts.get("execution"))
    status = str(execution.get("status") or "")
    lower = text.lower()
    _validate_artifact_ids(text, facts)
    _validate_selected_candidate_claims(text, facts)
    _validate_approval_and_execution_claims(text, facts)
    validate_workspace_claims(text, facts, composer_kind="response composer")
    if status not in {"approval_pending"} and _contains_claim(lower, _EXECUTED_CLAIMS):
        raise ValueError("response composer claimed execution completed")


_ARTIFACT_ID_PATTERN = re.compile(r"\b(?:decision|candidate|approval)\.[A-Za-z0-9_.:-]+\b")

_EXECUTED_CLAIMS = (
    "already executed",
    "has executed",
    "finished executing",
    "execution completed",
    "执行完成",
    "已经执行",
    "已经完成执行",
)

_PENDING_APPROVAL_CLAIMS = (
    "pending approval",
    "approval is ready",
    "approval has been created",
    "i created an approval",
    "needs approval",
    "requires approval",
    "等待审批",
    "已生成审批",
    "需要审批",
    "需要授权",
)

_NO_APPROVAL_CLAIMS = (
    "no approval required",
    "approval not required",
    "does not need approval",
    "不需要审批",
    "无需审批",
    "不需要授权",
)

_RECOMMENDATION_PREFIXES = (
    "recommend ",
    "choose ",
    "pick ",
    "go with ",
    "start with ",
    "do ",
    "prioritize ",
    "i recommend ",
    "i would choose ",
    "i would pick ",
    "i would start with ",
    "建议",
    "选择",
    "选",
    "先做",
    "优先做",
)


def _validate_artifact_ids(text: str, facts: Mapping[str, Any]) -> None:
    allowed_ids = {
        str(facts.get("decision_id") or ""),
        str(_mapping(facts.get("selected")).get("candidate_id") or ""),
        str(facts.get("approval_id") or ""),
    }
    allowed_ids.discard("")
    for artifact_id in _ARTIFACT_ID_PATTERN.findall(text):
        if artifact_id not in allowed_ids:
            raise ValueError(f"response composer invented artifact id: {artifact_id}")


def _validate_selected_candidate_claims(text: str, facts: Mapping[str, Any]) -> None:
    selected = _mapping(facts.get("selected"))
    selected_title = _normalize_text(selected.get("title"))
    selected_id = str(selected.get("candidate_id") or "")
    for option in facts.get("why_not_others") or []:
        option_map = _mapping(option)
        option_id = str(option_map.get("candidate_id") or "")
        option_title = _normalize_text(option_map.get("title"))
        if not option_title or option_title == selected_title or option_id == selected_id:
            continue
        if _looks_like_recommendation_for(text, option_title):
            raise ValueError("response composer recommended a non-selected candidate")


def _validate_approval_and_execution_claims(text: str, facts: Mapping[str, Any]) -> None:
    execution = _mapping(facts.get("execution"))
    status = str(execution.get("status") or "")
    lower = text.lower()
    if status != "approval_pending" and _contains_claim(lower, _PENDING_APPROVAL_CLAIMS):
        raise ValueError("response composer changed approval state")
    if status == "approval_pending" and _contains_claim(lower, _NO_APPROVAL_CLAIMS):
        raise ValueError("response composer contradicted pending approval state")
    if status not in {"completed", "success", "succeeded"} and _contains_claim(lower, _EXECUTED_CLAIMS):
        raise ValueError("response composer claimed execution completed")


def _looks_like_recommendation_for(text: str, title: str) -> bool:
    lower = text.lower()
    compact = _normalize_text(title)
    if not compact:
        return False
    return any(f"{prefix}{compact}" in lower for prefix in _RECOMMENDATION_PREFIXES)


def _contains_claim(lower_text: str, claims: tuple[str, ...]) -> bool:
    for claim in claims:
        start = lower_text.find(claim)
        while start >= 0:
            if not _negated_near(lower_text, start):
                return True
            start = lower_text.find(claim, start + len(claim))
    return False


def _negated_near(lower_text: str, claim_start: int) -> bool:
    window = lower_text[max(0, claim_start - 28) : claim_start]
    return any(
        token in window
        for token in (
            "not ",
            "no ",
            "never ",
            "without ",
            "doesn't ",
            "does not ",
            "isn't ",
            "is not ",
            "can't ",
            "cannot ",
            "没有",
            "没",
            "未",
            "并未",
            "不是",
            "不能",
            "不可以",
        )
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _why_not_summaries(compare_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _list_of_mappings(compare_payload.get("why_not_the_others"))[:3]:
        reasons: list[str] = []
        for reason in _list_of_mappings(item.get("reasons"))[:2]:
            message = str(reason.get("message") or reason.get("summary") or "").strip()
            dimension = str(reason.get("dimension_label") or reason.get("dimension") or "").strip()
            if message:
                reasons.append(message)
            elif dimension:
                reasons.append(dimension)
        result.append(
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "title": str(item.get("title") or ""),
                "reasons": reasons,
            }
        )
    return result


def _candidate_by_id(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if str(candidate.get("candidate_id") or "") == candidate_id:
            return candidate
    return {}


def _compact_simulation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "expected_outcome": _shorten(str(payload.get("expected_outcome") or payload.get("expected") or ""), 360),
        "downside": _shorten(str(payload.get("downside") or payload.get("likely_risks") or ""), 320),
        "success_signal": _shorten(str(payload.get("success_signal") or ""), 260),
        "confidence": payload.get("confidence"),
    }


def _runtime_model_id(config: Mapping[str, Any]) -> str:
    return str(config.get("llm_model") or "").strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if str(item).strip()] if isinstance(value, list) else []


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return "." * max(0, limit)
    return normalized[: max(0, limit - 3)].rstrip() + "..."
