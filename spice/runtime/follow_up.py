from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from spice.llm.candidate_expander import build_candidate_expander_client
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.memory import MemoryProvider
from spice.perception import build_evidence_context
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
from spice.runtime.conversation import build_conversation_turn, save_conversation_turn
from spice.runtime.memory_writeback import (
    skipped_general_evolution_memory_writeback,
    write_general_evolution_memory,
)
from spice.runtime.response_depth import resolve_response_depth_budget
from spice.runtime.session import load_or_create_session
from spice.runtime.store import LocalJsonStore


FOLLOW_UP_RESPONSE_SCHEMA_VERSION = "spice.follow_up_response.v1"
FollowUpResponseComposeResult = ComposerResult


@dataclass(frozen=True, slots=True)
class FollowUpResponseResult:
    rendered_text: str
    artifact: dict[str, Any]
    conversation_turn_path: Path


def answer_why_not_candidate(
    *,
    store: LocalJsonStore,
    session_id: str,
    user_input: str,
    source_run: Mapping[str, Any],
    candidate_id: str,
    config: Mapping[str, Any] | None = None,
    context_payload: Mapping[str, Any] | None = None,
    memory_provider: MemoryProvider | None = None,
    now: datetime | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> FollowUpResponseResult:
    created = now or datetime.now(timezone.utc)
    compare = _mapping(source_run.get("compare_payload"))
    candidate = _candidate_by_id(compare, candidate_id)
    why_not = _why_not_by_id(compare, candidate_id)
    selected = _mapping(compare.get("selected_recommendation"))
    display_language = str(compare.get("display_language") or source_run.get("display_language") or "en")
    rendered, evidence = _compose_why_not_follow_up(
        user_input=user_input,
        source_run=source_run,
        candidate_id=candidate_id,
        candidate=candidate,
        why_not=why_not,
        selected=selected,
        display_language=display_language,
        config=config or {},
        context_payload=context_payload,
        stream_callback=stream_callback,
    )
    return _record_follow_up(
        store=store,
        session_id=session_id,
        user_input=user_input,
        action="explain_why_not",
        source_run=source_run,
        source_candidate_id=candidate_id,
        rendered_text=rendered,
        evidence=evidence,
        memory_provider=memory_provider,
        now=created,
    )


def answer_candidate_plan(
    *,
    store: LocalJsonStore,
    session_id: str,
    user_input: str,
    source_run: Mapping[str, Any],
    candidate_id: str,
    config: Mapping[str, Any] | None = None,
    context_payload: Mapping[str, Any] | None = None,
    memory_provider: MemoryProvider | None = None,
    now: datetime | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> FollowUpResponseResult:
    created = now or datetime.now(timezone.utc)
    compare = _mapping(source_run.get("compare_payload"))
    candidate = _candidate_by_id(compare, candidate_id)
    display_language = str(compare.get("display_language") or source_run.get("display_language") or "en")
    rendered, evidence = _compose_general_follow_up(
        user_input=user_input,
        source_run=source_run,
        action="plan_candidate",
        candidate_id=candidate_id,
        config=config or {},
        context_payload=context_payload,
        stream_callback=stream_callback,
    )
    if not evidence:
        rendered = _render_plan(candidate=candidate, display_language=display_language)
        evidence = {"candidate": candidate}
    return _record_follow_up(
        store=store,
        session_id=session_id,
        user_input=user_input,
        action="plan_candidate",
        source_run=source_run,
        source_candidate_id=candidate_id,
        rendered_text=rendered,
        evidence=evidence,
        memory_provider=memory_provider,
        now=created,
    )


def answer_general_follow_up(
    *,
    store: LocalJsonStore,
    session_id: str,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str = "",
    config: Mapping[str, Any] | None = None,
    context_payload: Mapping[str, Any] | None = None,
    memory_provider: MemoryProvider | None = None,
    now: datetime | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> FollowUpResponseResult:
    created = now or datetime.now(timezone.utc)
    normalized_action = _normalize_general_action(action)
    rendered, evidence = _compose_general_follow_up(
        user_input=user_input,
        source_run=source_run,
        action=normalized_action,
        candidate_id=candidate_id,
        config=config or {},
        context_payload=context_payload,
        stream_callback=stream_callback,
    )
    return _record_follow_up(
        store=store,
        session_id=session_id,
        user_input=user_input,
        action=normalized_action,
        source_run=source_run,
        source_candidate_id=candidate_id or str(evidence.get("source_candidate_id") or ""),
        rendered_text=rendered,
        evidence=evidence,
        memory_provider=memory_provider,
        now=created,
    )


def answer_general_follow_up_with_llm(
    *,
    client: LLMClient,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str = "",
    model_provider: str = "",
    model_id: str = "",
    context_payload: Mapping[str, Any] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    result = compose_general_follow_up_with_llm(
        client=client,
        user_input=user_input,
        source_run=source_run,
        action=action,
        candidate_id=candidate_id,
        model_provider=model_provider,
        model_id=model_id,
        context_payload=context_payload,
        stream_callback=stream_callback,
        config=config,
    )
    return result.response_text, _follow_up_evidence_from_composer_result(result)


def compose_general_follow_up_with_llm(
    *,
    client: LLMClient,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str = "",
    model_provider: str = "",
    model_id: str = "",
    context_payload: Mapping[str, Any] | None = None,
    stream_callback: Callable[[str], None] | None = None,
    config: Mapping[str, Any] | None = None,
) -> FollowUpResponseComposeResult:
    facts = _general_follow_up_facts(
        user_input=user_input,
        source_run=source_run,
        action=action,
        candidate_id=candidate_id,
        context_payload=context_payload,
    )
    depth = _resolve_response_depth_for_follow_up_facts(
        facts,
        config=config,
        action=action,
    )
    facts = {**facts, "response_depth": depth.to_payload()}
    deterministic_text = _render_general_follow_up_from_facts(facts)
    if candidate_id and not _mapping(facts.get("target_candidate")):
        missing_text = _render_missing_target_candidate(facts)
        return FollowUpResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=missing_text,
            deterministic_text=deterministic_text,
            composer_kind="follow_up_response",
            model_provider=model_provider,
            model_id=model_id,
            error="target candidate not found",
            fallback_reason="missing_target_candidate",
            facts=facts,
            metadata={
                "fallback_reason": "missing_target_candidate",
                "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                "action": action,
                "source_candidate_id": str(facts.get("source_candidate_id") or ""),
            },
        )
    request = LLMRequest(
        task_hook=LLMTaskHook.RESPONSE_COMPOSE,
        input_text=_general_follow_up_prompt(facts),
        system_text=_general_follow_up_system_prompt(),
        response_format_hint="",
        temperature=0.35,
        max_tokens=depth.max_tokens,
        timeout_sec=depth.timeout_sec,
        metadata={
            "purpose": "general_follow_up_response",
            "model_provider": model_provider,
            "model_id": model_id,
            "action": action,
            "decision_id": facts.get("decision_id"),
            "candidate_id": candidate_id,
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
        return FollowUpResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="follow_up_response",
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
                    "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                    "action": action,
                    "source_candidate_id": str(facts.get("source_candidate_id") or ""),
                    "response_depth": depth.to_payload(),
                },
                reason="stream_error",
            ),
        )
    raw_output = output.raw_output
    try:
        rendered = parse_composer_response_text(raw_output, max_chars=depth.max_chars)
        _validate_general_follow_up_response(rendered, facts, max_chars=depth.max_chars)
    except Exception as exc:
        return FollowUpResponseComposeResult(
            enabled=True,
            status="fallback",
            response_text=deterministic_text,
            deterministic_text=deterministic_text,
            composer_kind="follow_up_response",
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
                    "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                    "action": action,
                    "source_candidate_id": str(facts.get("source_candidate_id") or ""),
                    "response_depth": depth.to_payload(),
                },
                reason="invalid_composed_response",
            ),
        )
    return FollowUpResponseComposeResult(
        enabled=True,
        status="composed",
        response_text=rendered,
        deterministic_text=deterministic_text,
        composer_kind="follow_up_response",
        model_provider=model_provider or output.provider_id,
        model_id=model_id or output.model_id,
        request_id=output.request_id,
        raw_output=raw_output,
        facts=facts,
        metadata=mark_streaming_valid(
            {
                **output.metadata,
                "facts_schema": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                "action": action,
                "source_candidate_id": str(facts.get("source_candidate_id") or ""),
                "response_depth": depth.to_payload(),
            }
        ),
    )


def _record_follow_up(
    *,
    store: LocalJsonStore,
    session_id: str,
    user_input: str,
    action: str,
    source_run: Mapping[str, Any],
    source_candidate_id: str,
    rendered_text: str,
    evidence: Mapping[str, Any],
    memory_provider: MemoryProvider | None,
    now: datetime,
) -> FollowUpResponseResult:
    source_run_id = str(source_run.get("run_id") or "")
    source_decision_id = str(source_run.get("decision_id") or "")
    source_approval_id = str(source_run.get("approval_id") or "")
    workspace_context = _workspace_context_from_evidence(evidence)
    url_context = _url_context_from_evidence(evidence)
    delegated_perception_context = _delegated_perception_context_from_evidence(evidence)
    evidence_context = _evidence_context_from_evidence(evidence)
    if not evidence_context:
        evidence_context = build_evidence_context(
            workspace_context=workspace_context,
            url_context=url_context,
            delegated_perception_context=delegated_perception_context,
        )
    artifact_refs = _artifact_refs(store, source_run)
    workspace_perception_id = str(workspace_context.get("perception_id") or "")
    if workspace_perception_id:
        artifact_refs["workspace_perception"] = _workspace_relative(
            store,
            store.record_path("perception", workspace_perception_id),
        )
    url_perception_id = str(url_context.get("perception_id") or "")
    if url_perception_id:
        artifact_refs["url_perception"] = _workspace_relative(
            store,
            store.record_path("perception", url_perception_id),
        )
    delegated_perception_id = str(delegated_perception_context.get("perception_id") or "")
    if delegated_perception_id:
        artifact_refs["delegated_perception"] = _workspace_relative(
            store,
            store.record_path("perception", delegated_perception_id),
        )
    turn = build_conversation_turn(
        user_input=user_input,
        route="follow_up",
        session_id=session_id,
        created_at=now,
        source_decision_id=source_decision_id or None,
        source_candidate_id=source_candidate_id or None,
        source_run_id=source_run_id or None,
        source_approval_id=source_approval_id or None,
        artifact_refs=artifact_refs,
        metadata={
            "follow_up_action": action,
            "generated_by": "spice.runtime.follow_up",
            **(
                {"workspace_context": workspace_context}
                if workspace_context
                else {}
            ),
            **({"url_context": url_context} if url_context else {}),
            **(
                {"delegated_perception_context": delegated_perception_context}
                if delegated_perception_context
                else {}
            ),
            "evidence_context": evidence_context,
        },
    )
    artifact = {
        "schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
        "generated_by": "spice.runtime.follow_up",
        "created_at": _timestamp(now),
        "turn_id": turn.turn_id,
        "response_id": turn.response_id,
        "route": "follow_up",
        "action": action,
        "source_run_id": source_run_id,
        "source_decision_id": source_decision_id,
        "source_candidate_id": source_candidate_id,
        "source_approval_id": source_approval_id,
        "rendered_text": rendered_text,
        "evidence": dict(evidence),
        "workspace_context": workspace_context,
        "url_context": url_context,
        "delegated_perception_context": delegated_perception_context,
        "evidence_context": evidence_context,
    }
    turn.metadata["follow_up_response"] = artifact
    path = save_conversation_turn(store, turn)
    _append_turn_to_session(store, session_id=session_id, turn_id=turn.turn_id, now=now)
    artifact["evolution_memory_writeback"] = _write_follow_up_evolution_memory(
        memory_provider,
        record={
            "created_at": _timestamp(now),
            "session_id": session_id,
            "turn_id": turn.turn_id,
            "response_id": turn.response_id,
            "user_input": user_input,
            "route": "follow_up",
            "route_result": {
                "route": "follow_up",
                "action": action,
                "candidate_id": source_candidate_id,
                "decision_id": source_decision_id,
                "run_id": source_run_id,
            },
            "response_summary": _response_summary(rendered_text),
            "decision_id": source_decision_id,
            "run_id": source_run_id,
            "trace_ref": str(source_run.get("trace_ref") or ""),
            "candidate_id": source_candidate_id,
            "selected_candidate": dict(_mapping(evidence.get("selected")) or _mapping(evidence.get("candidate"))),
            "follow_up_type": action,
            "approval_id": source_approval_id,
            "artifact_refs": turn.artifact_refs,
            "conversation_turn": turn.to_payload(),
            "workspace_context": workspace_context,
            "url_context": url_context,
            "delegated_perception_context": delegated_perception_context,
            "evidence_context": evidence_context,
            "metadata": {
                "generated_by": "spice.runtime.follow_up",
                "source": "follow_up_response",
            },
        },
    )
    turn.metadata["follow_up_response"] = artifact
    store.save_conversation_turn(turn.turn_id, turn.to_payload())
    return FollowUpResponseResult(
        rendered_text=rendered_text,
        artifact=artifact,
        conversation_turn_path=path,
    )


def _write_follow_up_evolution_memory(
    memory_provider: MemoryProvider | None,
    *,
    record: dict[str, Any],
) -> dict[str, Any]:
    if memory_provider is None:
        return skipped_general_evolution_memory_writeback(reason="memory_provider_not_configured")
    try:
        return write_general_evolution_memory(memory_provider, record=record)
    except Exception as exc:
        return skipped_general_evolution_memory_writeback(reason=f"write_failed:{exc}")


def _workspace_context_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = _mapping(evidence.get("decision_context"))
    workspace = _mapping(context.get("workspace_context"))
    if workspace:
        return workspace
    composer = _mapping(evidence.get("composer_result"))
    facts = _mapping(composer.get("facts"))
    context = _mapping(facts.get("decision_context"))
    return _mapping(context.get("workspace_context"))


def _url_context_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = _mapping(evidence.get("decision_context"))
    url_context = _mapping(context.get("url_context"))
    if url_context:
        return url_context
    composer = _mapping(evidence.get("composer_result"))
    facts = _mapping(composer.get("facts"))
    context = _mapping(facts.get("decision_context"))
    return _mapping(context.get("url_context"))


def _delegated_perception_context_from_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    context = _mapping(evidence.get("decision_context"))
    delegated_context = _mapping(context.get("delegated_perception_context"))
    if delegated_context:
        return delegated_context
    composer = _mapping(evidence.get("composer_result"))
    facts = _mapping(composer.get("facts"))
    context = _mapping(facts.get("decision_context"))
    return _mapping(context.get("delegated_perception_context"))


def _evidence_context_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    context = _mapping(evidence.get("decision_context"))
    evidence_context = _mapping(context.get("evidence_context"))
    if evidence_context:
        return evidence_context
    composer = _mapping(evidence.get("composer_result"))
    facts = _mapping(composer.get("facts"))
    context = _mapping(facts.get("decision_context"))
    return _mapping(context.get("evidence_context"))


def _response_summary(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return ""


def _compose_general_follow_up(
    *,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str,
    config: Mapping[str, Any],
    context_payload: Mapping[str, Any] | None = None,
    stream_callback: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = str(config.get("llm_model") or "").strip()
    if provider_id and provider_id != "deterministic" and model_id:
        try:
            client = build_candidate_expander_client(provider_id=provider_id, model_id=model_id)
            return answer_general_follow_up_with_llm(
                client=client,
                user_input=user_input,
                source_run=source_run,
                action=action,
                candidate_id=candidate_id,
                model_provider=provider_id,
                model_id=model_id,
                context_payload=context_payload,
                stream_callback=stream_callback,
                config=config,
            )
        except Exception as exc:
            rendered, evidence = _deterministic_general_follow_up(
                user_input=user_input,
                source_run=source_run,
                action=action,
                candidate_id=candidate_id,
                context_payload=context_payload,
            )
            result = FollowUpResponseComposeResult(
                enabled=True,
                status="fallback",
                response_text=rendered,
                deterministic_text=rendered,
                composer_kind="follow_up_response",
                model_provider=provider_id,
                model_id=model_id,
                error=str(exc),
                fallback_reason="client_error",
                facts=evidence,
                metadata={
                    "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
                    "action": action,
                    "source_candidate_id": str(evidence.get("source_candidate_id") or ""),
                },
            )
            return rendered, _follow_up_evidence_from_composer_result(result)
    rendered, evidence = _deterministic_general_follow_up(
        user_input=user_input,
        source_run=source_run,
        action=action,
        candidate_id=candidate_id,
        context_payload=context_payload,
    )
    result = FollowUpResponseComposeResult(
        enabled=False,
        status="disabled",
        response_text=rendered,
        deterministic_text=rendered,
        composer_kind="follow_up_response",
        model_provider=provider_id or "deterministic",
        model_id=model_id,
        facts=evidence,
        metadata={
            "reason": "deterministic provider",
            "composer_schema_version": FOLLOW_UP_RESPONSE_SCHEMA_VERSION,
            "action": action,
            "source_candidate_id": str(evidence.get("source_candidate_id") or ""),
        },
    )
    return rendered, _follow_up_evidence_from_composer_result(result)


def _compose_why_not_follow_up(
    *,
    user_input: str,
    source_run: Mapping[str, Any],
    candidate_id: str,
    candidate: Mapping[str, Any],
    why_not: Mapping[str, Any],
    selected: Mapping[str, Any],
    display_language: str,
    config: Mapping[str, Any],
    context_payload: Mapping[str, Any] | None,
    stream_callback: Callable[[str], None] | None,
) -> tuple[str, dict[str, Any]]:
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = str(config.get("llm_model") or "").strip()
    if provider_id and provider_id != "deterministic" and model_id:
        return _compose_general_follow_up(
            user_input=user_input,
            source_run=source_run,
            action="explain_why_not",
            candidate_id=candidate_id,
            config=config,
            context_payload=context_payload,
            stream_callback=stream_callback,
        )
    rendered = _render_why_not(
        candidate=candidate,
        why_not=why_not,
        selected=selected,
        display_language=display_language,
    )
    return rendered, {
        "selected": dict(selected),
        "candidate": dict(candidate),
        "target_candidate": dict(candidate),
        "why_not": dict(why_not),
        "source_candidate_id": candidate_id,
        "llm": {
            "status": "disabled",
            "model_provider": provider_id or "deterministic",
            "model_id": model_id,
            "fallback_reason": "deterministic_provider",
        },
    }


def _deterministic_general_follow_up(
    *,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str,
    context_payload: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    facts = _general_follow_up_facts(
        user_input=user_input,
        source_run=source_run,
        action=action,
        candidate_id=candidate_id,
        context_payload=context_payload,
    )
    rendered = _render_general_follow_up_from_facts(facts)
    return rendered, facts


def _render_general_follow_up_from_facts(facts: Mapping[str, Any]) -> str:
    action = str(facts.get("action") or "answer_from_decision")
    selected = _mapping(facts.get("selected"))
    candidate = _mapping(facts.get("target_candidate")) or selected
    display_language = str(facts.get("display_language") or "en")
    if action == "explain_why_not":
        return _render_why_not(
            candidate=candidate,
            why_not=_mapping(facts.get("target_why_not")),
            selected=selected,
            display_language=display_language,
        )
    if action == "plan_candidate":
        return _render_plan(candidate=candidate, display_language=display_language)
    if display_language.startswith("zh"):
        return _render_general_follow_up_zh(action=action, selected=selected, candidate=candidate, facts=facts)
    return _render_general_follow_up_en(action=action, selected=selected, candidate=candidate, facts=facts)


def _follow_up_evidence_from_composer_result(result: FollowUpResponseComposeResult) -> dict[str, Any]:
    payload = result.to_payload()
    return {
        **dict(result.facts),
        "composer_result": payload,
        "llm": {
            "status": result.status,
            "model_provider": result.model_provider,
            "model_id": result.model_id,
            "request_id": result.request_id,
            "raw_output": result.raw_output,
            "fallback_reason": result.fallback_reason,
            "error": result.error,
        },
    }


def _general_follow_up_facts(
    *,
    user_input: str,
    source_run: Mapping[str, Any],
    action: str,
    candidate_id: str,
    context_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    compare = _mapping(source_run.get("compare_payload"))
    brief = _mapping(source_run.get("decision_brief"))
    selected = _mapping(compare.get("selected_recommendation")) or _mapping(brief.get("selected"))
    selected_id = str(selected.get("candidate_id") or source_run.get("selected_candidate_id") or "")
    target_id = candidate_id or selected_id
    target_candidate = _candidate_by_id(compare, target_id)
    target_why_not = _why_not_by_id(compare, target_id)
    return {
        "user_input": user_input,
        "action": action,
        "display_language": str(compare.get("display_language") or source_run.get("display_language") or brief.get("display_language") or "en"),
        "run_id": str(source_run.get("run_id") or ""),
        "decision_id": str(source_run.get("decision_id") or ""),
        "trace_ref": str(source_run.get("trace_ref") or ""),
        "selected": selected,
        "source_candidate_id": target_id,
        "target_candidate": target_candidate,
        "target_why_not": target_why_not,
        "target_candidate_missing": bool(candidate_id and not target_candidate),
        "candidates": _compact_candidates(compare),
        "why_this_won": _strings(brief.get("why_this_won")),
        "why_not_others": _compact_why_not(compare),
        "simulation": _compact_simulation(_mapping(target_candidate.get("simulation"))),
        "execution": _mapping(brief.get("execution")),
        "allowed_next_actions": _strings(brief.get("next_actions")),
        "decision_context": compact_composer_context(context_payload),
    }


def _general_follow_up_system_prompt() -> str:
    return (
        "You are Spice's follow-up composer. Do one thing: answer the user's follow-up "
        "from finalized decision facts and recent context. Do not change the winner, "
        "scores, candidate ids, approval state, or execution state. Do not invent "
        "artifact ids or expose raw JSON/schema. Return only the response."
    )


def _general_follow_up_prompt(facts: Mapping[str, Any]) -> str:
    return _json_dumps(
        build_slim_composer_prompt_payload(
            task="Answer this follow-up using the active Spice decision. Do not start a new decision unless facts are insufficient.",
            facts=_slim_follow_up_prompt_facts(facts),
            tone="Direct, practical agent voice. Answer the follow-up, not the whole card.",
            extra_constraints=(
                "Do not claim work was executed.",
                "If comparing an alternative, frame it as a conditional tradeoff instead of changing the recorded winner.",
                "For why-not questions, explain the decision tradeoff in natural language; do not list raw score deltas unless the user explicitly asks for scores.",
                "If facts are insufficient, ask one clear clarifying question.",
                "Match response_depth guidance and stay under response_depth.max_chars.",
                *WORKSPACE_COMPOSER_CONSTRAINTS,
            ),
        )
    )


def _slim_follow_up_prompt_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "user_input": str(facts.get("user_input") or ""),
        "follow_up_action": str(facts.get("action") or ""),
        "selected_candidate": _mapping(facts.get("selected")),
        "target_candidate": _mapping(facts.get("target_candidate")),
        "target_why_not": _compact_why_not_item(_mapping(facts.get("target_why_not"))),
        "visible_candidates": _list_of_mappings_limited(facts.get("candidates"), limit=3),
        "why_won": _strings(facts.get("why_this_won"))[:3],
        "why_not": _list_of_mappings_limited(facts.get("why_not_others"), limit=3),
        "simulation": _mapping(facts.get("simulation")),
        "execution_affordance": _mapping(facts.get("execution")),
        "allowed_next_actions": _strings(facts.get("allowed_next_actions"))[:6],
        "recent_context": slim_recent_context(_mapping(facts.get("decision_context"))),
        "response_depth": _compact_response_depth(_mapping(facts.get("response_depth"))),
    }


def _resolve_response_depth_for_follow_up_facts(
    facts: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    action: str,
) -> Any:
    context = _mapping(facts.get("decision_context"))
    evidence_context = _mapping(context.get("evidence_context"))
    requirements = _mapping(evidence_context.get("requirements"))
    return resolve_response_depth_budget(
        answer_mode=str(requirements.get("answer_mode") or ""),
        evidence_domain=str(requirements.get("evidence_domain") or ""),
        evidence_context=evidence_context,
        user_input=str(facts.get("user_input") or ""),
        config=config,
        composer_kind="follow_up_response",
        action=action,
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


def _render_general_follow_up_en(
    *,
    action: str,
    selected: Mapping[str, Any],
    candidate: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> str:
    title = _candidate_title(candidate) or str(selected.get("title") or "the selected option")
    recommendation = _candidate_recommendation(candidate) or str(selected.get("human_summary") or "")
    simulation = _mapping(facts.get("simulation"))
    expected = str(candidate.get("expected_result") or simulation.get("expected_outcome") or "").strip()
    downside = str(simulation.get("downside") or "").strip()
    success = str(simulation.get("success_signal") or "").strip()
    if action == "compare_alternative":
        lines = [f"{title} could be better if its tradeoff matters more than the current winner."]
    elif action == "ask_clarifying_question":
        return "I need one detail to answer that cleanly: what constraint matters most right now?"
    else:
        lines = [f"Based on the current Decision Card, I would work from {title}."]
    if recommendation:
        lines.extend(["", recommendation])
    if expected:
        lines.extend(["", f"Likely outcome: {expected}"])
    if downside:
        lines.append(f"Watchout: {downside}")
    if success:
        lines.append(f"Success signal: {success}")
    lines.extend(["", "Next: ask for a more detailed plan, refine the decision, or use details to inspect the full card."])
    return "\n".join(lines)


def _render_general_follow_up_zh(
    *,
    action: str,
    selected: Mapping[str, Any],
    candidate: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> str:
    title = _candidate_title(candidate) or str(selected.get("title") or "当前选项")
    recommendation = _candidate_recommendation(candidate) or str(selected.get("human_summary") or "")
    simulation = _mapping(facts.get("simulation"))
    expected = str(candidate.get("expected_result") or simulation.get("expected_outcome") or "").strip()
    downside = str(simulation.get("downside") or "").strip()
    success = str(simulation.get("success_signal") or "").strip()
    if action == "compare_alternative":
        lines = [f"如果你更看重它对应的取舍，{title} 有可能比当前选择更合适。"]
    elif action == "ask_clarifying_question":
        return "我需要先确认一个点：现在最重要的约束是时间、风险，还是可见成果？"
    else:
        lines = [f"基于当前 Decision Card，我会围绕 {title} 继续推进。"]
    if recommendation:
        lines.extend(["", recommendation])
    if expected:
        lines.extend(["", f"可能结果：{expected}"])
    if downside:
        lines.append(f"注意：{downside}")
    if success:
        lines.append(f"成功信号：{success}")
    lines.extend(["", "下一步：你可以让我展开计划、refine 这个判断，或者用 details 看完整 card。"])
    return "\n".join(lines)


def _render_why_not(
    *,
    candidate: Mapping[str, Any],
    why_not: Mapping[str, Any],
    selected: Mapping[str, Any],
    display_language: str,
) -> str:
    title = _candidate_title(candidate) or str(why_not.get("title") or "that option")
    selected_title = str(selected.get("title") or "the selected option")
    reasons = _why_not_reason_texts(why_not)
    if display_language.startswith("zh"):
        lines = [
            f"不是 {title} 的主要原因是：",
        ]
        if candidate and str(candidate.get("candidate_id") or "") == str(selected.get("candidate_id") or ""):
            lines = [f"{title} 已经是当前选中的方案。"]
        elif reasons:
            lines.extend(f"- {reason}" for reason in reasons[:3])
        else:
            lines.append(f"- 当前 Decision Card 选择了 {selected_title}，但没有记录到针对 {title} 的更细 why-not 证据。")
        lines.extend(["", "Next:", "  details  展开完整 Decision Card", "  refine   带着这个取舍重新调整"])
        return "\n".join(lines)

    lines = [f"I did not pick {title} mainly because:"]
    if candidate and str(candidate.get("candidate_id") or "") == str(selected.get("candidate_id") or ""):
        lines = [f"{title} is already the selected option."]
    elif reasons:
        lines.extend(f"- {reason}" for reason in reasons[:3])
    else:
        lines.append(f"- The Decision Card selected {selected_title}, but did not record a specific why-not reason for {title}.")
    lines.extend(["", "Next:", "  details  expand the full Decision Card", "  refine   adjust the decision with this trade-off"])
    return "\n".join(lines)


def _render_plan(*, candidate: Mapping[str, Any], display_language: str) -> str:
    title = _candidate_title(candidate) or "this option"
    recommendation = _candidate_recommendation(candidate)
    why_now = _strings(candidate.get("why_now"))
    expected = str(candidate.get("expected_result") or "").strip()
    executor_task = str(candidate.get("executor_task") or "").strip()
    simulation = _mapping(candidate.get("simulation"))
    downside = str(simulation.get("downside") or "").strip()
    success = str(simulation.get("success_signal") or "").strip()
    if display_language.startswith("zh"):
        lines = [
            f"{title} 的计划：",
            "",
            f"目标：{recommendation or title}",
        ]
        if why_now:
            lines.extend(["", "为什么现在做：", *[f"- {item}" for item in why_now[:3]]])
        if expected:
            lines.extend(["", f"预期结果：{expected}"])
        if executor_task:
            lines.extend(["", f"执行任务草案：{executor_task}"])
        if downside or success:
            lines.append("")
            lines.append("验证与风险：")
            if downside:
                lines.append(f"- 主要取舍：{downside}")
            if success:
                lines.append(f"- 成功信号：{success}")
        lines.extend(["", "Next:", "  execute  如果这是可执行方案，进入授权执行", "  refine   调整这个计划", "  details  展开完整 Decision Card"])
        return "\n".join(lines)

    lines = [
        f"Plan for {title}:",
        "",
        f"Goal: {recommendation or title}",
    ]
    if why_now:
        lines.extend(["", "Why now:", *[f"- {item}" for item in why_now[:3]]])
    if expected:
        lines.extend(["", f"Expected result: {expected}"])
    if executor_task:
        lines.extend(["", f"Draft executor task: {executor_task}"])
    if downside or success:
        lines.append("")
        lines.append("Validation and trade-off:")
        if downside:
            lines.append(f"- Downside: {downside}")
        if success:
            lines.append(f"- Success signal: {success}")
    lines.extend(["", "Next:", "  execute  request approval if this option is executable", "  refine   adjust this plan", "  details  expand the full Decision Card"])
    return "\n".join(lines)


def _candidate_by_id(compare_payload: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in _list(compare_payload.get("candidate_decisions")):
        item = _mapping(candidate)
        if str(item.get("candidate_id") or "") == candidate_id:
            return item
    selected = _mapping(compare_payload.get("selected_recommendation"))
    if str(selected.get("candidate_id") or "") == candidate_id:
        return selected
    return {}


def _why_not_by_id(compare_payload: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    for item in _list(compare_payload.get("why_not_the_others")):
        payload = _mapping(item)
        if str(payload.get("candidate_id") or "") == candidate_id:
            return payload
    return {}


def _why_not_reason_texts(why_not: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for reason in _list(why_not.get("reasons")):
        payload = _mapping(reason)
        text = str(payload.get("reason") or payload.get("summary") or payload.get("message") or "").strip()
        if text:
            result.append(text)
    return result


def _candidate_title(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("title") or candidate.get("intent") or "").strip()


def _candidate_recommendation(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("recommended_action")
        or candidate.get("human_summary")
        or candidate.get("intent")
        or ""
    ).strip()


def _normalize_general_action(action: str) -> str:
    normalized = action.strip().lower()
    if normalized == "refine_decision":
        return "refine_decision"
    if normalized in {
        "answer_from_decision",
        "ask_clarifying_question",
        "compare_alternative",
        "explain_why_not",
        "plan_candidate",
    }:
        return normalized
    return "answer_from_decision"


def _compact_candidates(compare_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in _list(compare_payload.get("candidate_decisions"))[:6]:
        item = _mapping(candidate)
        result.append(
            {
                "candidate_id": str(item.get("candidate_id") or ""),
                "title": _candidate_title(item),
                "recommendation": _candidate_recommendation(item),
                "why_now": _strings(item.get("why_now"))[:2],
                "expected_result": str(item.get("expected_result") or ""),
                "simulation": _compact_simulation(_mapping(item.get("simulation"))),
                "execution_affordance": _mapping(item.get("execution_affordance")),
            }
        )
    return result


def _compact_simulation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "expected_outcome": _shorten(str(payload.get("expected_outcome") or payload.get("expected") or ""), 360),
        "downside": _shorten(str(payload.get("downside") or payload.get("likely_risks") or ""), 320),
        "success_signal": _shorten(str(payload.get("success_signal") or ""), 260),
        "confidence": payload.get("confidence"),
    }


def _compact_why_not(compare_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _list(compare_payload.get("why_not_the_others"))[:4]:
        compact = _compact_why_not_item(_mapping(item))
        if compact:
            result.append(compact)
    return result


def _compact_why_not_item(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "candidate_id": str(payload.get("candidate_id") or ""),
        "title": str(payload.get("title") or ""),
        "reasons": _why_not_reason_texts(payload)[:2],
    }


def _validate_general_follow_up_response(
    text: str,
    facts: Mapping[str, Any],
    *,
    max_chars: int = 2400,
) -> None:
    if not text:
        raise ValueError("general follow-up composer returned empty response")
    if len(text) > max_chars:
        raise ValueError("general follow-up composer returned overly long response")
    stripped = text.strip()
    if stripped.startswith(("{", "[")) or "```" in stripped:
        raise ValueError("general follow-up composer returned raw structured output")
    _validate_follow_up_artifact_ids(stripped, facts)
    _validate_follow_up_target(stripped, facts)
    _validate_follow_up_execution_claims(stripped, facts)
    validate_workspace_claims(stripped, facts, composer_kind="follow-up composer")


_ARTIFACT_ID_PATTERN = re.compile(r"\b(?:decision|candidate|approval)\.[A-Za-z0-9_.:-]+\b")

_PLAN_CLAIMS = (
    "plan for",
    "steps for",
    "first step",
    "start by",
    "implementation plan",
    "roadmap",
    "计划",
    "步骤",
    "第一步",
    "先做",
)

_EXECUTABLE_CLAIMS = (
    "ready for approval",
    "approval required",
    "requires approval",
    "can execute",
    "can be executed",
    "can hand off",
    "ready to hand off",
    "ready for handoff",
    "send to hermes",
    "send to codex",
    "需要审批",
    "需要授权",
    "可以执行",
    "可执行",
    "交给 hermes",
    "交给 codex",
)

_EXECUTED_CLAIMS = (
    "already executed",
    "has executed",
    "execution completed",
    "finished executing",
    "执行完成",
    "已经执行",
    "已经完成执行",
)


def _validate_follow_up_artifact_ids(text: str, facts: Mapping[str, Any]) -> None:
    allowed = {
        str(facts.get("decision_id") or ""),
        str(_mapping(facts.get("selected")).get("candidate_id") or ""),
        str(facts.get("source_candidate_id") or ""),
    }
    for candidate in _list_of_mappings_limited(facts.get("candidates"), limit=20):
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id:
            allowed.add(candidate_id)
    allowed.discard("")
    for artifact_id in _ARTIFACT_ID_PATTERN.findall(text):
        if artifact_id not in allowed:
            raise ValueError(f"follow-up composer invented artifact id: {artifact_id}")


def _validate_follow_up_target(text: str, facts: Mapping[str, Any]) -> None:
    action = str(facts.get("action") or "")
    if action not in {"compare_alternative", "explain_why_not"}:
        return
    selected = _mapping(facts.get("selected"))
    target = _mapping(facts.get("target_candidate"))
    if not target:
        raise ValueError("follow-up composer target candidate does not exist")
    selected_id = str(selected.get("candidate_id") or "")
    target_id = str(target.get("candidate_id") or "")
    if not target_id or target_id == selected_id:
        return
    lower = text.lower()
    selected_title = _normalize_text(_candidate_title(selected))
    target_title = _normalize_text(_candidate_title(target))
    answered_selected_plan = (
        selected_title
        and selected_title in lower
        and target_title
        and target_title not in lower
        and _contains_claim(lower, _PLAN_CLAIMS)
    )
    if answered_selected_plan:
        raise ValueError("follow-up composer answered a why-not/compare question with the selected candidate plan")


def _validate_follow_up_execution_claims(text: str, facts: Mapping[str, Any]) -> None:
    lower = text.lower()
    if _contains_claim(lower, _EXECUTED_CLAIMS):
        raise ValueError("follow-up composer claimed work was executed")
    target = _mapping(facts.get("target_candidate")) or _mapping(facts.get("selected"))
    execution = _mapping(facts.get("execution"))
    if _candidate_is_advisory_only(target, execution) and _contains_claim(lower, _EXECUTABLE_CLAIMS):
        raise ValueError("follow-up composer described an advisory candidate as executable")


def _candidate_is_advisory_only(candidate: Mapping[str, Any], execution: Mapping[str, Any]) -> bool:
    affordance = _mapping(candidate.get("execution_affordance"))
    if affordance:
        executable = affordance.get("candidate_executable")
        if executable is False:
            return True
        if executable is True:
            return False
    status = str(execution.get("status") or "").strip().lower()
    return status in {"advisory", "blocked"}


def _render_missing_target_candidate(facts: Mapping[str, Any]) -> str:
    if str(facts.get("display_language") or "").startswith("zh"):
        return "我在当前 Decision Card 里找不到这个选项。你可以用 details 展开完整卡片，或者直接说要看 A、B、C 里的哪一个。"
    return "I cannot find that option in the active Decision Card. Use details to inspect the full card, or refer to A, B, or C directly."


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


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _artifact_refs(store: LocalJsonStore, source_run: Mapping[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    paths = _mapping(source_run.get("store_paths"))
    for key in ("run", "decision", "approval", "state", "session"):
        value = str(paths.get(key) or "").strip()
        if value:
            refs[key] = value
    if "run" not in refs and source_run.get("run_id"):
        refs["run"] = _workspace_relative(store, store.record_path("run", str(source_run["run_id"])))
    if "decision" not in refs and source_run.get("decision_id"):
        refs["decision"] = _workspace_relative(store, store.record_path("decision", str(source_run["decision_id"])))
    return refs


def _append_turn_to_session(
    store: LocalJsonStore,
    *,
    session_id: str,
    turn_id: str,
    now: datetime,
) -> None:
    session = load_or_create_session(store, session_id=session_id, now=now)
    payload = session.to_payload()
    turn_ids = _strings(payload.get("conversation_turn_ids"))
    if turn_id not in turn_ids:
        turn_ids.append(turn_id)
    payload["conversation_turn_ids"] = turn_ids
    payload["updated_at"] = _timestamp(now)
    store.save_session(session_id, payload)


def _workspace_relative(store: LocalJsonStore, path: Path) -> str:
    try:
        return str(path.relative_to(store.paths.project_root))
    except ValueError:
        return str(path)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings_limited(value: Any, *, limit: int) -> list[dict[str, Any]]:
    return [_mapping(item) for item in _list(value)[: max(0, limit)] if isinstance(item, Mapping)]


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if str(item)]


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return "." * max(0, limit)
    return normalized[: max(0, limit - 3)].rstrip() + "..."
