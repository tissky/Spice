from __future__ import annotations

from typing import Any

from spice.decision.general.types import payload_value
from spice.memory import MemoryProvider
from spice.perception import compact_evidence_context
from spice.runtime.session_summary import update_session_summary


GENERAL_DECISION_MEMORY_NAMESPACE = "general.decision"
GENERAL_DECISION_MEMORY_SCHEMA_VERSION = "spice.memory.general.decision.v1"
GENERAL_REFLECTION_MEMORY_NAMESPACE = "general.reflection"
GENERAL_REFLECTION_MEMORY_SCHEMA_VERSION = "spice.memory.general.reflection.v1"
GENERAL_EVOLUTION_MEMORY_NAMESPACE = "general.evolution"
GENERAL_EVOLUTION_MEMORY_SCHEMA_VERSION = "spice.memory.general.evolution.v1"
GENERAL_WORKSPACE_PERCEPTION_MEMORY_NAMESPACE = "general.workspace_perception"
GENERAL_WORKSPACE_PERCEPTION_MEMORY_SCHEMA_VERSION = (
    "spice.memory.general.workspace_perception.v1"
)
GENERAL_URL_PERCEPTION_MEMORY_NAMESPACE = "general.url_perception"
GENERAL_URL_PERCEPTION_MEMORY_SCHEMA_VERSION = "spice.memory.general.url_perception.v1"
GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE = "general.delegated_perception"
GENERAL_DELEGATED_PERCEPTION_MEMORY_SCHEMA_VERSION = (
    "spice.memory.general.delegated_perception.v1"
)


def write_general_decision_memory(
    provider: MemoryProvider,
    *,
    artifact: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact decision record for future decision context retrieval."""

    record = build_general_decision_memory_record(artifact=artifact)
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_DECISION_MEMORY_NAMESPACE,
        refs=refs,
    )
    session_summary = update_session_summary(provider, config=config)
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_DECISION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
        "session_summary": session_summary,
    }


def skipped_general_decision_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_DECISION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_reflection_memory(
    provider: MemoryProvider,
    *,
    decision_artifact: dict[str, Any],
    execution_artifact: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact execution outcome record for future reflection context."""

    record = build_general_reflection_memory_record(
        decision_artifact=decision_artifact,
        execution_artifact=execution_artifact,
    )
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_REFLECTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    session_summary = update_session_summary(provider, config=config)
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_REFLECTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
        "session_summary": session_summary,
    }


def skipped_general_reflection_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_REFLECTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_evolution_memory(
    provider: MemoryProvider,
    *,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Write a compact conversation/evolution record for continuity."""

    normalized = build_general_evolution_memory_record(record=record)
    refs = _memory_refs(normalized)
    record_ids = provider.write(
        [normalized],
        namespace=GENERAL_EVOLUTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_EVOLUTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
    }


def skipped_general_evolution_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_EVOLUTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_workspace_perception_memory(
    provider: MemoryProvider,
    *,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Write compact workspace perception metadata for future context retrieval."""

    record = build_general_workspace_perception_memory_record(artifact=artifact)
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_WORKSPACE_PERCEPTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_WORKSPACE_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
    }


def skipped_general_workspace_perception_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_WORKSPACE_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_url_perception_memory(
    provider: MemoryProvider,
    *,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Write compact URL perception metadata for future context retrieval."""

    record = build_general_url_perception_memory_record(artifact=artifact)
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_URL_PERCEPTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_URL_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
    }


def skipped_general_url_perception_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_URL_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_delegated_perception_memory(
    provider: MemoryProvider,
    *,
    artifact: dict[str, Any],
    user_input: str = "",
    route_result: dict[str, Any] | None = None,
    linked_decision_id: str = "",
    linked_run_id: str = "",
    conversation_turn_id: str = "",
) -> dict[str, Any]:
    """Write compact delegated investigation metadata for future context retrieval."""

    record = build_general_delegated_perception_memory_record(
        artifact=artifact,
        user_input=user_input,
        route_result=route_result,
        linked_decision_id=linked_decision_id,
        linked_run_id=linked_run_id,
        conversation_turn_id=conversation_turn_id,
    )
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
    }


def skipped_general_delegated_perception_memory_writeback(
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_DELEGATED_PERCEPTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def build_general_decision_memory_record(*, artifact: dict[str, Any]) -> dict[str, Any]:
    selected = _dict(artifact.get("compare_payload")).get("selected_recommendation")
    selected_payload = _dict(selected)
    decision_id = str(artifact.get("decision_id") or "")
    run_id = str(artifact.get("run_id") or "")
    state_after_ref = str(artifact.get("state_after_ref") or "")
    active_frame_ref = (
        f"{state_after_ref}#active_decision_frame:{decision_id}"
        if state_after_ref and decision_id
        else ""
    )
    evidence_context = _dict(artifact.get("evidence_context"))
    return {
        "id": f"memory.general.decision.{run_id or decision_id}",
        "schema_version": GENERAL_DECISION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.decision",
        "created_at": str(artifact.get("created_at") or ""),
        "session_id": str(artifact.get("session_id") or ""),
        "run_id": run_id,
        "decision_id": decision_id,
        "trace_ref": str(artifact.get("trace_ref") or ""),
        "source": str(artifact.get("source") or ""),
        "parent_run_id": str(artifact.get("parent_run_id") or ""),
        "run_intent_mode": str(artifact.get("run_intent_mode") or ""),
        "display_language": str(artifact.get("display_language") or ""),
        "input": payload_value(artifact.get("input") or {}),
        "candidate_summary": payload_value(artifact.get("candidate_summary") or {}),
        "selected": _compact_selected(selected_payload),
        "why_won": payload_value(selected_payload.get("decision_basis") or []),
        "approval_id": str(artifact.get("approval_id") or ""),
        "active_decision_frame_ref": active_frame_ref,
        "context_refs": payload_value(artifact.get("context_refs") or {}),
        "evidence_context": (
            compact_evidence_context(evidence_context) if evidence_context else {}
        ),
        "state_refs": {
            "before": str(artifact.get("state_before_ref") or ""),
            "after": state_after_ref,
        },
        "artifact_refs": payload_value(artifact.get("store_paths") or {}),
        "selection_pool": payload_value(artifact.get("selection_pool") or {}),
        "handoff": {
            "required": bool(artifact.get("handoff_required")),
            "blocked": bool(artifact.get("handoff_blocked")),
            "blockers": payload_value(artifact.get("handoff_blockers") or []),
            "approval_id": str(artifact.get("approval_id") or ""),
        },
        "active_decision_frame": _compact_active_decision_frame(
            _dict(artifact.get("active_decision_frame"))
        ),
    }


def build_general_workspace_perception_memory_record(*, artifact: dict[str, Any]) -> dict[str, Any]:
    perception_id = str(artifact.get("perception_id") or "")
    return {
        "id": f"memory.general.workspace_perception.{perception_id}",
        "schema_version": GENERAL_WORKSPACE_PERCEPTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.workspace_perception",
        "created_at": str(artifact.get("created_at") or ""),
        "perception_id": perception_id,
        "workspace_root": str(artifact.get("workspace_root") or ""),
        "trigger": str(artifact.get("trigger") or ""),
        "query": str(artifact.get("query") or ""),
        "summary": _shorten(str(artifact.get("summary") or ""), 1200),
        "metadata": _compact_workspace_perception_metadata(_dict(artifact.get("metadata"))),
        "queries": _compact_workspace_queries(artifact.get("queries")),
        "files": {
            "read_count": len(_list_of_dicts(artifact.get("files_read"))),
            "skipped_count": len(_list_of_dicts(artifact.get("files_skipped"))),
            "read": _compact_workspace_files_read(artifact.get("files_read")),
            "skipped": _compact_workspace_files_skipped(artifact.get("files_skipped")),
        },
        "tool_calls": {
            "executed_count": len(_list_of_dicts(artifact.get("tool_calls"))),
            "blocked_count": len(_list_of_dicts(artifact.get("blocked_tool_calls"))),
            "executed": _compact_workspace_tool_calls(artifact.get("tool_calls")),
            "blocked": _compact_workspace_tool_calls(artifact.get("blocked_tool_calls")),
        },
        "facts": _compact_workspace_facts(artifact.get("facts")),
        "snippet_refs": _compact_workspace_snippet_refs(artifact.get("snippets")),
        "budget": _compact_workspace_budget(_dict(artifact.get("budget"))),
        "limits": _compact_workspace_limits(_dict(artifact.get("limits"))),
        "artifact_refs": payload_value(artifact.get("store_paths") or {}),
    }


def build_general_url_perception_memory_record(*, artifact: dict[str, Any]) -> dict[str, Any]:
    perception_id = str(artifact.get("perception_id") or "")
    documents = _list_of_dicts(artifact.get("documents"))
    skipped = _list_of_dicts(artifact.get("urls_skipped"))
    return {
        "id": f"memory.general.url_perception.{perception_id}",
        "schema_version": GENERAL_URL_PERCEPTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.url_perception",
        "created_at": str(artifact.get("created_at") or ""),
        "perception_id": perception_id,
        "trigger": str(artifact.get("trigger") or ""),
        "query": str(artifact.get("query") or ""),
        "summary": _shorten(str(artifact.get("summary") or ""), 1200),
        "urls": [str(item) for item in _list(artifact.get("urls"))[:12]],
        "documents": {
            "read_count": len(documents),
            "skipped_count": len(skipped),
            "read": [
                {
                    "url": str(item.get("url") or ""),
                    "final_url": str(item.get("final_url") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": _shorten(str(item.get("title") or ""), 220),
                    "chars_read": item.get("chars_read"),
                    "truncated": bool(item.get("truncated")),
                    "content_hash": str(item.get("content_hash") or ""),
                }
                for item in documents[:12]
            ],
            "skipped": [
                {
                    "url": str(item.get("url") or ""),
                    "reason": _shorten(str(item.get("reason") or ""), 220),
                }
                for item in skipped[:12]
            ],
        },
        "facts": [
            {
                "text": _shorten(str(item.get("text") or ""), 500),
                "source_url": str(item.get("source_url") or ""),
                "title": _shorten(str(item.get("title") or ""), 220),
            }
            for item in _list_of_dicts(artifact.get("facts"))[:12]
        ],
        "snippet_refs": [
            {
                "url": str(item.get("url") or ""),
                "title": _shorten(str(item.get("title") or ""), 220),
                "content_hash": str(item.get("content_hash") or ""),
                "source": str(item.get("source") or ""),
            }
            for item in _list_of_dicts(artifact.get("snippets"))[:12]
        ],
        "budget": _compact_workspace_budget(_dict(artifact.get("budget"))),
        "limits": payload_value(artifact.get("limits") or {}),
        "artifact_refs": payload_value(artifact.get("store_paths") or {}),
    }


def build_general_delegated_perception_memory_record(
    *,
    artifact: dict[str, Any],
    user_input: str = "",
    route_result: dict[str, Any] | None = None,
    linked_decision_id: str = "",
    linked_run_id: str = "",
    conversation_turn_id: str = "",
) -> dict[str, Any]:
    perception_id = str(artifact.get("perception_id") or "")
    delegation_id = str(artifact.get("delegation_id") or "")
    route = _dict(route_result)
    return {
        "id": f"memory.general.delegated_perception.{perception_id or delegation_id}",
        "schema_version": GENERAL_DELEGATED_PERCEPTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.delegated_perception",
        "created_at": str(artifact.get("created_at") or ""),
        "perception_id": perception_id,
        "delegation_id": delegation_id,
        "executor_id": str(artifact.get("executor_id") or ""),
        "status": str(artifact.get("status") or ""),
        "scope": str(artifact.get("scope") or "read_only_investigation"),
        "permission_mode": str(artifact.get("permission_mode") or "read_only"),
        "context_strategy": str(artifact.get("context_strategy") or ""),
        "query": _shorten(str(artifact.get("query") or ""), 800),
        "summary": _shorten(str(artifact.get("summary") or ""), 1200),
        "confidence": str(artifact.get("confidence") or ""),
        "consent_id": str(artifact.get("consent_id") or ""),
        "request_ref": str(artifact.get("request_ref") or ""),
        "executor_report_ref": str(artifact.get("executor_report_ref") or ""),
        "executor_run_ref": str(artifact.get("executor_run_ref") or ""),
        "user_input": _shorten(str(user_input or ""), 1200),
        "route_result": payload_value(route),
        "linked": {
            "decision_id": str(linked_decision_id or ""),
            "run_id": str(linked_run_id or ""),
            "conversation_turn_id": str(conversation_turn_id or ""),
            "input_context_refs": [
                str(item)
                for item in _list(artifact.get("input_context_refs"))[:20]
                if str(item)
            ],
        },
        "findings": _compact_delegated_perception_findings(artifact.get("findings")),
        "source_refs": _compact_delegated_perception_sources(artifact.get("sources")),
        "limitations": [
            _shorten(str(item), 300)
            for item in _list(artifact.get("limitations"))[:20]
            if str(item)
        ],
        "metadata": _compact_delegated_perception_metadata(_dict(artifact.get("metadata"))),
        "artifact_refs": payload_value(artifact.get("store_paths") or {}),
    }


def build_general_evolution_memory_record(*, record: dict[str, Any]) -> dict[str, Any]:
    turn = _dict(record.get("conversation_turn"))
    route_result = _dict(record.get("route_result"))
    selected = _dict(record.get("selected_candidate"))
    approval = _dict(record.get("approval"))
    execution = _dict(record.get("execution"))
    outcome = _dict(record.get("outcome"))
    turn_id = str(record.get("turn_id") or turn.get("turn_id") or "")
    response_id = str(record.get("response_id") or turn.get("response_id") or "")
    decision_id = str(
        record.get("decision_id")
        or turn.get("source_decision_id")
        or route_result.get("decision_id")
        or ""
    )
    run_id = str(record.get("run_id") or turn.get("source_run_id") or "")
    candidate_id = str(
        record.get("candidate_id")
        or turn.get("source_candidate_id")
        or selected.get("candidate_id")
        or ""
    )
    approval_id = str(
        record.get("approval_id")
        or turn.get("source_approval_id")
        or approval.get("approval_id")
        or execution.get("approval_id")
        or ""
    )
    outcome_id = str(record.get("outcome_id") or outcome.get("outcome_id") or execution.get("outcome_id") or "")
    record_id = str(record.get("id") or "")
    if not record_id:
        record_id = f"memory.general.evolution.{turn_id or response_id or approval_id or outcome_id or decision_id}"
    evidence_context = _dict(record.get("evidence_context"))
    return {
        "id": record_id,
        "schema_version": GENERAL_EVOLUTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.evolution",
        "created_at": str(record.get("created_at") or turn.get("created_at") or ""),
        "session_id": str(record.get("session_id") or turn.get("session_id") or ""),
        "turn_id": turn_id,
        "response_id": response_id,
        "user_input": str(record.get("user_input") or turn.get("user_input") or ""),
        "route": str(record.get("route") or turn.get("route") or route_result.get("route") or ""),
        "route_result": payload_value(route_result),
        "response_summary": str(record.get("response_summary") or ""),
        "decision_id": decision_id,
        "run_id": run_id,
        "trace_ref": str(record.get("trace_ref") or ""),
        "candidate_id": candidate_id,
        "selected_candidate": _compact_evolution_candidate(selected),
        "follow_up_type": str(record.get("follow_up_type") or route_result.get("action") or ""),
        "approval_id": approval_id,
        "approval": _compact_approval_for_evolution(approval),
        "execution": _compact_execution_for_evolution(execution),
        "outcome_id": outcome_id,
        "outcome": payload_value(outcome),
        "artifact_refs": payload_value(record.get("artifact_refs") or turn.get("artifact_refs") or {}),
        "evidence_context": (
            compact_evidence_context(evidence_context) if evidence_context else {}
        ),
        "workspace_context": _compact_evolution_workspace_context(
            _dict(record.get("workspace_context"))
        ),
        "url_context": _compact_evolution_url_context(_dict(record.get("url_context"))),
        "delegated_perception_context": _compact_evolution_delegated_context(
            _dict(record.get("delegated_perception_context"))
        ),
        "metadata": payload_value(record.get("metadata") or {}),
    }


def build_general_reflection_memory_record(
    *,
    decision_artifact: dict[str, Any],
    execution_artifact: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(
        execution_artifact.get("selected_candidate_id")
        or execution_artifact.get("candidate_id")
        or ""
    )
    task_status = str(execution_artifact.get("task_status") or "")
    protocol_status = str(execution_artifact.get("protocol_status") or "")
    outcome_id = str(execution_artifact.get("outcome_id") or "")
    execution_id = str(execution_artifact.get("execution_id") or "")
    state_delta = _state_delta_summary(execution_artifact)
    return {
        "id": f"memory.general.reflection.{outcome_id or execution_id}",
        "schema_version": GENERAL_REFLECTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.reflection",
        "created_at": str(execution_artifact.get("created_at") or ""),
        "session_id": str(execution_artifact.get("session_id") or ""),
        "run_id": str(execution_artifact.get("run_id") or ""),
        "decision_id": str(execution_artifact.get("decision_id") or ""),
        "trace_ref": str(execution_artifact.get("trace_ref") or ""),
        "approval_id": str(execution_artifact.get("approval_id") or ""),
        "candidate_id": candidate_id,
        "selected_candidate": _compact_candidate_from_artifact(
            decision_artifact,
            candidate_id=candidate_id,
        ),
        "executor": {
            "provider": str(execution_artifact.get("executor_provider") or ""),
            "executor_id": str(execution_artifact.get("executor_id") or ""),
            "command": str(execution_artifact.get("executor_command") or ""),
            "skill_id": str(execution_artifact.get("skill_id") or ""),
            "context_pack_id": str(execution_artifact.get("context_pack_id") or ""),
            "dry_run": bool(execution_artifact.get("dry_run")),
            "executor_called": bool(execution_artifact.get("executor_called")),
            "real_executor_called": bool(execution_artifact.get("real_executor_called")),
            "sdep_request_sent": bool(execution_artifact.get("sdep_request_sent")),
            "executed": bool(execution_artifact.get("executed")),
        },
        "execution": {
            "execution_id": execution_id,
            "request_id": str(execution_artifact.get("request_id") or ""),
            "outcome_id": outcome_id,
            "protocol_status": protocol_status,
            "task_status": task_status,
            "success": _execution_succeeded(
                protocol_status=protocol_status,
                task_status=task_status,
            ),
            "state_updated": bool(execution_artifact.get("state_updated")),
            "state_before_ref": str(execution_artifact.get("state_before_ref") or ""),
            "state_after_ref": str(execution_artifact.get("state_after_ref") or ""),
        },
        "outcome_summary": _outcome_summary(execution_artifact),
        "state_delta_summary": state_delta,
        "artifact_refs": payload_value(execution_artifact.get("store_paths") or {}),
        "decision_memory_refs": _decision_memory_record_ids(decision_artifact),
    }


def _compact_selected(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(selected.get("candidate_id") or ""),
        "title": str(selected.get("title") or ""),
        "action": str(selected.get("action") or ""),
        "recommendation": str(
            selected.get("human_summary")
            or selected.get("recommendation")
            or selected.get("summary")
            or ""
        ),
        "score": selected.get("score"),
        "execution_affordance": payload_value(selected.get("execution_affordance") or {}),
        "skill_resolution": payload_value(selected.get("skill_resolution") or {}),
    }


def _compact_candidate_from_artifact(
    artifact: dict[str, Any],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    if not candidate_id:
        return {}
    for candidate in _iter_candidate_payloads(artifact):
        if str(candidate.get("candidate_id") or candidate.get("id") or "") != candidate_id:
            continue
        return {
            "candidate_id": candidate_id,
            "title": str(candidate.get("title") or ""),
            "action": str(candidate.get("action") or candidate.get("action_type") or ""),
            "recommendation": str(
                candidate.get("human_summary")
                or candidate.get("recommendation")
                or candidate.get("summary")
                or ""
            ),
            "expected": str(
                candidate.get("expected_result")
                or candidate.get("expected")
                or ""
            ),
            "executor_task": str(candidate.get("executor_task") or ""),
            "execution_affordance": payload_value(
                candidate.get("execution_affordance") or {}
            ),
            "skill_resolution": payload_value(candidate.get("skill_resolution") or {}),
        }
    selected = _dict(_dict(artifact.get("compare_payload")).get("selected_recommendation"))
    if str(selected.get("candidate_id") or "") == candidate_id:
        return _compact_selected(selected)
    return {"candidate_id": candidate_id}


def _compact_active_decision_frame(frame: dict[str, Any]) -> dict[str, Any]:
    if not frame:
        return {}
    return {
        "decision_id": str(frame.get("decision_id") or ""),
        "run_id": str(frame.get("run_id") or ""),
        "status": str(frame.get("status") or ""),
        "selected_candidate_id": str(frame.get("selected_candidate_id") or ""),
        "approval_id": str(frame.get("approval_id") or ""),
        "allowed_continuations": payload_value(frame.get("allowed_continuations") or []),
        "selection_pool": payload_value(frame.get("selection_pool") or {}),
    }


def _compact_evolution_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "label": str(candidate.get("label") or ""),
        "title": str(candidate.get("title") or ""),
        "action": str(candidate.get("action") or candidate.get("action_type") or ""),
        "recommendation": str(
            candidate.get("recommended_action")
            or candidate.get("recommendation")
            or candidate.get("human_summary")
            or candidate.get("summary")
            or ""
        ),
        "executor_task": str(candidate.get("executor_task") or ""),
    }


def _compact_approval_for_evolution(approval: dict[str, Any]) -> dict[str, Any]:
    if not approval:
        return {}
    metadata = _dict(approval.get("metadata"))
    return {
        "approval_id": str(approval.get("approval_id") or ""),
        "status": str(approval.get("status") or ""),
        "candidate_id": str(approval.get("candidate_id") or ""),
        "execution_allowed": bool(approval.get("execution_allowed")),
        "required_permission": str(
            metadata.get("required_executor_permission")
            or _dict(metadata.get("permission_requirement")).get("required_permission")
            or ""
        ),
        "executor_id": str(metadata.get("executor_id") or ""),
    }


def _compact_execution_for_evolution(execution: dict[str, Any]) -> dict[str, Any]:
    if not execution:
        return {}
    return {
        "approval_id": str(execution.get("approval_id") or ""),
        "execution_id": str(execution.get("execution_id") or ""),
        "request_id": str(execution.get("request_id") or ""),
        "outcome_id": str(execution.get("outcome_id") or ""),
        "executor_provider": str(execution.get("executor_provider") or ""),
        "executor_id": str(execution.get("executor_id") or ""),
        "protocol_status": str(execution.get("protocol_status") or ""),
        "task_status": str(execution.get("task_status") or ""),
        "executed": bool(execution.get("executed")),
        "executor_called": bool(execution.get("executor_called")),
    }


def _memory_refs(record: dict[str, Any]) -> list[str]:
    refs = [
        str(record.get("run_id") or ""),
        str(record.get("decision_id") or ""),
        str(record.get("trace_ref") or ""),
        str(record.get("perception_id") or ""),
        str(record.get("delegation_id") or ""),
        str(record.get("consent_id") or ""),
        str(record.get("executor_report_ref") or ""),
        str(record.get("executor_run_ref") or ""),
        str(record.get("approval_id") or ""),
        str(record.get("candidate_id") or ""),
        str(record.get("turn_id") or ""),
        str(record.get("response_id") or ""),
        str(record.get("outcome_id") or ""),
        str(record.get("active_decision_frame_ref") or ""),
    ]
    execution = record.get("execution")
    if isinstance(execution, dict):
        refs.extend(
            str(execution.get(key) or "")
            for key in ("execution_id", "request_id", "outcome_id", "state_after_ref")
        )
    context_refs = record.get("context_refs")
    if isinstance(context_refs, dict):
        refs.extend(str(value) for value in context_refs.values() if str(value or ""))
    return list(dict.fromkeys(ref for ref in refs if ref))


def _iter_candidate_payloads(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    compare = _dict(artifact.get("compare_payload"))
    for item in compare.get("candidate_decisions") or []:
        if isinstance(item, dict):
            payloads.append(item)
    for item in artifact.get("candidates") or []:
        if isinstance(item, dict):
            payloads.append(item)
    frame = _dict(artifact.get("active_decision_frame"))
    for item in frame.get("candidate_options") or []:
        if isinstance(item, dict):
            payloads.append(item)
    return payloads


def _execution_succeeded(*, protocol_status: str, task_status: str) -> bool:
    success_values = {"success", "succeeded", "ok", "completed"}
    protocol = protocol_status.strip().lower()
    task = task_status.strip().lower()
    if protocol and protocol not in success_values:
        return False
    return task in success_values


def _outcome_summary(artifact: dict[str, Any]) -> str:
    outcome_record = _dict(artifact.get("outcome_record"))
    summary = str(outcome_record.get("summary") or "")
    if summary:
        return summary
    output = _sdep_output(artifact)
    return str(output.get("summary") or "")


def _state_delta_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    outcome_record = _dict(artifact.get("outcome_record"))
    state_delta = outcome_record.get("state_delta")
    if not isinstance(state_delta, dict):
        state_delta = _sdep_output(artifact).get("state_delta")
    if not isinstance(state_delta, dict):
        state_delta = {}
    return {
        "task_status": str(artifact.get("task_status") or ""),
        "state_updated": bool(artifact.get("state_updated")),
        "state_before_ref": str(artifact.get("state_before_ref") or ""),
        "state_after_ref": str(artifact.get("state_after_ref") or ""),
        "delta": payload_value(state_delta),
        "updated_refs": payload_value(state_delta.get("updated_refs") or []),
    }


def _sdep_output(artifact: dict[str, Any]) -> dict[str, Any]:
    response = _dict(artifact.get("sdep_response"))
    output = _dict(_dict(response.get("outcome")).get("output"))
    if output:
        return output
    outcome = _dict(artifact.get("outcome"))
    return _dict(_dict(_dict(outcome.get("sdep_response")).get("outcome")).get("output"))


def _decision_memory_record_ids(artifact: dict[str, Any]) -> list[str]:
    writeback = _dict(artifact.get("memory_writeback"))
    ids = writeback.get("record_ids")
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if str(item or "")]


def _compact_workspace_perception_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("runtime_step", "status", "reason", "model_provider", "model_id"):
        if key in metadata:
            compact[key] = payload_value(metadata[key])
    loop = metadata.get("loop")
    if isinstance(loop, dict):
        compact["loop"] = {
            key: payload_value(loop[key])
            for key in ("schema_version", "done", "rounds_used")
            if key in loop
        }
    return compact


def _compact_evolution_workspace_context(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "source": str(payload.get("source") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "summary": _shorten(str(payload.get("summary") or ""), 500),
        "files_read": [
            {
                "path": str(item.get("path") or ""),
                "chars_read": item.get("chars_read"),
            }
            for item in _list_of_dicts(payload.get("files_read"))[:8]
        ],
        "facts": [
            {
                "text": _shorten(str(item.get("text") or ""), 300),
                "source_path": str(item.get("source_path") or ""),
            }
            for item in _list_of_dicts(payload.get("facts"))[:8]
        ],
    }


def _compact_evolution_url_context(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "source": str(payload.get("source") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "summary": _shorten(str(payload.get("summary") or ""), 500),
        "documents": [
            {
                "url": str(item.get("url") or item.get("final_url") or ""),
                "source_type": str(item.get("source_type") or ""),
                "title": _shorten(str(item.get("title") or ""), 180),
            }
            for item in _list_of_dicts(payload.get("documents"))[:8]
        ],
        "facts": [
            {
                "text": _shorten(str(item.get("text") or ""), 300),
                "source_url": str(item.get("source_url") or ""),
            }
            for item in _list_of_dicts(payload.get("facts"))[:8]
        ],
    }


def _compact_evolution_delegated_context(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    return {
        "source": str(payload.get("source") or ""),
        "perception_id": str(payload.get("perception_id") or ""),
        "delegation_id": str(payload.get("delegation_id") or ""),
        "executor_id": str(payload.get("executor_id") or ""),
        "query": _shorten(str(payload.get("query") or ""), 300),
        "summary": _shorten(str(payload.get("summary") or ""), 500),
        "confidence": str(payload.get("confidence") or ""),
        "findings": [
            {
                "finding_id": str(item.get("finding_id") or ""),
                "text": _shorten(str(item.get("text") or ""), 300),
                "confidence": item.get("confidence"),
                "source_refs": [str(ref) for ref in _list(item.get("source_refs"))[:6]],
            }
            for item in _list_of_dicts(payload.get("findings"))[:8]
        ],
        "sources": [
            {
                "source_id": str(item.get("source_id") or ""),
                "source_type": str(item.get("source_type") or ""),
                "title": _shorten(str(item.get("title") or ""), 180),
                "uri": str(item.get("uri") or item.get("url") or ""),
                "observed_by": str(item.get("observed_by") or ""),
                "verification_status": str(item.get("verification_status") or ""),
            }
            for item in _list_of_dicts(payload.get("sources"))[:8]
        ],
        "limitations": [_shorten(str(item), 220) for item in _list(payload.get("limitations"))[:8]],
    }


def _compact_delegated_perception_findings(value: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:24]:
        findings.append(
            {
                "finding_id": str(item.get("finding_id") or ""),
                "text": _shorten(str(item.get("text") or ""), 500),
                "confidence": item.get("confidence"),
                "source_refs": [
                    str(ref)
                    for ref in _list(item.get("source_refs"))[:8]
                    if str(ref)
                ],
                "limitations": [
                    _shorten(str(limit), 220)
                    for limit in _list(item.get("limitations"))[:6]
                    if str(limit)
                ],
            }
        )
    return findings


def _compact_delegated_perception_sources(value: Any) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:24]:
        sources.append(
            {
                "source_id": str(item.get("source_id") or ""),
                "source_type": str(item.get("source_type") or ""),
                "title": _shorten(str(item.get("title") or ""), 220),
                "uri": str(item.get("uri") or item.get("url") or ""),
                "excerpt": _shorten(str(item.get("excerpt") or ""), 300),
                "observed_by": str(item.get("observed_by") or ""),
                "accessed_at": str(item.get("accessed_at") or ""),
                "verification_status": str(item.get("verification_status") or ""),
            }
        )
    return sources


def _compact_delegated_perception_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "source",
        "parser_status",
        "fallback_reason",
        "raw_output_retained_in_executor_report",
        "ignored_executor_fields",
        "runtime_step",
        "status",
        "reason",
    ):
        if key in metadata:
            compact[key] = payload_value(metadata[key])
    return compact


def _compact_workspace_queries(value: Any) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:8]:
        queries.append(
            {
                "query": str(item.get("query") or ""),
                "query_type": str(item.get("query_type") or ""),
                "path": str(item.get("path") or ""),
                "file_glob": str(item.get("file_glob") or ""),
                "limit": item.get("limit"),
            }
        )
    return queries


def _compact_workspace_files_read(value: Any) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:20]:
        files.append(
            {
                "path": str(item.get("path") or ""),
                "chars_read": item.get("chars_read"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "truncated": bool(item.get("truncated")),
                "content_hash": str(item.get("content_hash") or ""),
                "reason": str(item.get("reason") or ""),
            }
        )
    return files


def _compact_workspace_files_skipped(value: Any) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:30]:
        skipped.append(
            {
                "path": str(item.get("path") or ""),
                "reason": str(item.get("reason") or ""),
            }
        )
    return skipped


def _compact_workspace_tool_calls(value: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:20]:
        result = _dict(item.get("result"))
        calls.append(
            {
                "call_id": str(item.get("call_id") or ""),
                "round_index": item.get("round_index"),
                "tool": str(item.get("tool") or ""),
                "args": _compact_workspace_tool_args(_dict(item.get("args"))),
                "status": str(item.get("status") or ""),
                "reason": str(item.get("reason") or ""),
                "result": _compact_workspace_tool_result(result),
            }
        )
    return calls


def _compact_workspace_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload_value(args[key])
        for key in ("path", "pattern", "file_glob", "offset", "limit")
        if key in args
    }


def _compact_workspace_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "status",
        "path",
        "reason",
        "error",
        "branch",
        "line_start",
        "line_end",
        "chars_read",
        "truncated",
        "content_hash",
    ):
        if key in result:
            compact[key] = payload_value(result[key])
    for key in ("matches", "entries", "files", "files_skipped"):
        value = result.get(key)
        if isinstance(value, list):
            compact[f"{key}_count"] = len(value)
    return compact


def _compact_workspace_facts(value: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:24]:
        facts.append(
            {
                "text": _shorten(str(item.get("text") or ""), 500),
                "source_path": str(item.get("source_path") or ""),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "confidence": item.get("confidence"),
                "metadata": _compact_workspace_fact_metadata(_dict(item.get("metadata"))),
            }
        )
    return facts


def _compact_workspace_fact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload_value(metadata[key])
        for key in ("source", "reason", "call_id")
        if key in metadata
    }


def _compact_workspace_snippet_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in _list_of_dicts(value)[:24]:
        refs.append(
            {
                "path": str(item.get("path") or ""),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "content_hash": str(item.get("content_hash") or ""),
                "source": str(item.get("source") or ""),
            }
        )
    return refs


def _compact_workspace_budget(budget: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload_value(budget[key])
        for key in (
            "rounds_used",
            "tool_calls_recorded",
            "tool_calls_executed",
            "tool_calls_blocked",
            "requested_url_count",
            "normalized_url_count",
            "document_count",
            "skipped_count",
            "total_chars_read",
            "total_char_budget",
            "remaining_char_budget",
            "limits",
        )
        if key in budget
    }


def _compact_workspace_limits(limits: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload_value(limits[key])
        for key in ("max_files", "max_chars_per_file", "total_char_budget")
        if key in limits
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _shorten(text: str, limit: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
