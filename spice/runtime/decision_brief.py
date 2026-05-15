from __future__ import annotations

from typing import Any, Mapping

from spice.decision.compare import analyze_compare_payload


DECISION_BRIEF_SCHEMA_VERSION = "spice.decision_brief.v1"


def compose_decision_brief(
    compare_payload: Mapping[str, Any],
    *,
    run_id: str = "",
    decision_id: str = "",
    approval_id: str | None = None,
    run_intent_mode: str = "auto",
    handoff_blocked: bool = False,
    handoff_blockers: list[str] | None = None,
) -> dict[str, Any]:
    analysis = analyze_compare_payload(compare_payload)
    selected = dict(analysis.get("selected_recommendation") or {})
    candidates = [dict(item) for item in analysis.get("candidates", [])]
    selected_candidate = _candidate_by_id(candidates, str(selected.get("candidate_id") or ""))
    why_this_won = _why_this_won(selected)
    alternatives = _alternatives(candidates, selected_candidate_id=str(selected.get("candidate_id") or ""))
    execution = _execution_summary(
        selected_candidate or selected,
        approval_id=approval_id,
        handoff_blocked=handoff_blocked,
        handoff_blockers=handoff_blockers or [],
    )
    language = str(compare_payload.get("display_language") or "en")
    return {
        "schema_version": DECISION_BRIEF_SCHEMA_VERSION,
        "generated_by": "spice.runtime.decision_brief",
        "display_language": language,
        "run_id": run_id,
        "decision_id": decision_id or str(analysis.get("decision_id") or ""),
        "trace_ref": str(analysis.get("trace_ref") or ""),
        "run_intent_mode": run_intent_mode,
        "selected": {
            "candidate_id": str(selected.get("candidate_id") or ""),
            "title": str(selected.get("title") or _candidate_title(selected_candidate) or ""),
            "recommendation": str(
                selected.get("human_summary")
                or _candidate_recommendation(selected_candidate)
                or selected.get("selection_reason")
                or ""
            ),
        },
        "why_this_won": why_this_won,
        "alternatives": alternatives,
        "execution": execution,
        "warnings": list(analysis.get("warnings") or [])[:3],
        "next_actions": _next_actions(execution, approval_id=approval_id),
    }


def render_decision_brief(brief: Mapping[str, Any]) -> str:
    language = str(brief.get("display_language") or "en")
    return _render_zh(brief) if language.startswith("zh") else _render_en(brief)


def _render_en(brief: Mapping[str, Any]) -> str:
    selected = _mapping(brief.get("selected"))
    execution = _mapping(brief.get("execution"))
    lines = [
        f"I'd choose {selected.get('title') or 'the selected option'}.",
    ]
    recommendation = str(selected.get("recommendation") or "").strip()
    if recommendation:
        lines.extend(["", recommendation])
    why = [str(item) for item in _list(brief.get("why_this_won")) if str(item).strip()]
    if why:
        lines.append("")
        lines.append("Why this wins:")
        lines.extend(f"- {item}" for item in why[:3])
    alternatives = [_mapping(item) for item in _list(brief.get("alternatives"))]
    if alternatives:
        lines.append("")
        lines.append("Other paths:")
        for item in alternatives[:2]:
            title = str(item.get("title") or "Alternative").strip()
            note = str(item.get("summary") or "").strip()
            lines.append(f"- {title}" + (f": {note}" if note else ""))
    lines.append("")
    lines.append(f"Execution: {execution.get('summary') or 'advisory only'}")
    capability_summary = str(execution.get("capability_summary") or "").strip()
    if capability_summary:
        lines.append(capability_summary)
    next_actions = [str(item) for item in _list(brief.get("next_actions")) if str(item).strip()]
    if next_actions:
        lines.append("")
        lines.append("Next:")
        lines.extend(f"  {item}" for item in next_actions[:5])
    return "\n".join(lines)


def _render_zh(brief: Mapping[str, Any]) -> str:
    selected = _mapping(brief.get("selected"))
    execution = _mapping(brief.get("execution"))
    lines = [
        f"我会选 {selected.get('title') or '当前选项'}。",
    ]
    recommendation = str(selected.get("recommendation") or "").strip()
    if recommendation:
        lines.extend(["", recommendation])
    why = [str(item) for item in _list(brief.get("why_this_won")) if str(item).strip()]
    if why:
        lines.append("")
        lines.append("为什么它更合适：")
        lines.extend(f"- {item}" for item in why[:3])
    alternatives = [_mapping(item) for item in _list(brief.get("alternatives"))]
    if alternatives:
        lines.append("")
        lines.append("其他路径：")
        for item in alternatives[:2]:
            title = str(item.get("title") or "备选").strip()
            note = str(item.get("summary") or "").strip()
            lines.append(f"- {title}" + (f"：{note}" if note else ""))
    lines.append("")
    lines.append(f"执行状态：{execution.get('summary') or '只做建议，不进入执行'}")
    capability_summary = str(execution.get("capability_summary") or "").strip()
    if capability_summary:
        lines.append(capability_summary)
    next_actions = [str(item) for item in _list(brief.get("next_actions")) if str(item).strip()]
    if next_actions:
        lines.append("")
        lines.append("你接下来可以：")
        lines.extend(f"  {item}" for item in next_actions[:5])
    return "\n".join(lines)


def _why_this_won(selected: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for item in _list(selected.get("decision_basis")):
        basis = _mapping(item)
        label = str(basis.get("dimension_label") or basis.get("dimension") or "").strip()
        contribution = basis.get("contribution")
        if label and contribution is not None:
            try:
                result.append(f"{label} carried meaningful weight ({float(contribution):.2f}).")
            except (TypeError, ValueError):
                result.append(f"{label} carried meaningful weight.")
        elif label:
            result.append(f"{label} was a differentiating factor.")
    for item in _list(selected.get("reason_summary")):
        text = str(item or "").strip()
        if text:
            result.append(text)
    selection_reason = str(selected.get("selection_reason") or "").strip()
    if not result and selection_reason:
        result.append(selection_reason)
    return _dedupe(result)[:3]


def _alternatives(
    candidates: list[Mapping[str, Any]],
    *,
    selected_candidate_id: str,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id == selected_candidate_id:
            continue
        title = _candidate_title(candidate)
        if not title:
            continue
        result.append(
            {
                "candidate_id": candidate_id,
                "title": title,
                "summary": _candidate_recommendation(candidate),
            }
        )
        if len(result) >= 2:
            break
    return result


def _execution_summary(
    selected: Mapping[str, Any],
    *,
    approval_id: str | None,
    handoff_blocked: bool,
    handoff_blockers: list[str],
) -> dict[str, Any]:
    affordance = _mapping(selected.get("execution_affordance"))
    approval = _mapping(affordance.get("approval"))
    executor = _mapping(affordance.get("executor"))
    permission = _mapping(affordance.get("permission"))
    capability = _mapping(affordance.get("capability"))
    if approval_id:
        summary = f"approval required before executor handoff ({approval_id})"
        status = "approval_pending"
    elif handoff_blocked:
        summary = "execution handoff is blocked"
        status = "blocked"
    elif affordance and not affordance.get("candidate_executable"):
        summary = "advisory only; no executor handoff requested"
        status = "advisory"
    elif affordance.get("executable"):
        summary = "ready for an approval-gated executor handoff"
        status = "ready"
    else:
        summary = "advisory only; no executor handoff requested"
        status = "advisory"
    return {
        "status": status,
        "summary": summary,
        "approval_id": approval_id or "",
        "executor": str(executor.get("executor_id") or ""),
        "permission": {
            "required": str(permission.get("required") or ""),
            "configured": str(permission.get("configured") or ""),
        },
        "capability": _execution_capability_summary(affordance, capability=capability),
        "capability_summary": _execution_capability_text(
            affordance,
            capability=capability,
            status=status,
        ),
        "approval_required": bool(approval.get("required") or approval_id),
        "blockers": list(handoff_blockers or affordance.get("blockers") or [])[:3],
    }


def _execution_capability_summary(
    affordance: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "required_capability": str(
            capability.get("required_capability")
            or affordance.get("required_capability")
            or ""
        ),
        "source": str(
            capability.get("source")
            or affordance.get("executor_capability_source")
            or ""
        ),
        "matched_capability": str(capability.get("matched_capability") or ""),
        "executor_has_required_capability": bool(
            capability.get("executor_has_required_capability")
        ),
        "simulates_required_capability": bool(capability.get("simulates_required_capability")),
        "limitations": [str(item) for item in _list(capability.get("limitations")) if str(item).strip()][:3],
    }


def _execution_capability_text(
    affordance: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    status: str,
) -> str:
    required = str(
        capability.get("required_capability")
        or affordance.get("required_capability")
        or ""
    ).strip()
    if not required:
        return ""
    executor = _mapping(affordance.get("executor"))
    executor_id = str(executor.get("executor_id") or "executor").strip()
    source = str(
        capability.get("source")
        or affordance.get("executor_capability_source")
        or "capability snapshot"
    ).strip()
    matched = str(capability.get("matched_capability") or "").strip()
    has_required = bool(capability.get("executor_has_required_capability"))
    simulates = bool(capability.get("simulates_required_capability"))
    if "advisory" in status:
        return ""
    if simulates:
        return (
            f"This would need {required}; current {executor_id} only simulates that handoff "
            "and is not a real executor run."
        )
    if has_required:
        via = f" through {matched}" if matched and matched != required else ""
        return (
            f"If you execute this, it needs {required}; current {executor_id} "
            f"{source} supports it{via}, so it can enter approval."
        )
    return (
        f"If you execute this, it needs {required}; current {executor_id} "
        f"{source} does not advertise that capability."
    )


def _next_actions(execution: Mapping[str, Any], *, approval_id: str | None) -> list[str]:
    status = str(execution.get("status") or "")
    if approval_id:
        execute_label = "execute  approve and run the pending executor handoff"
    elif status == "ready":
        execute_label = "execute  open the approval path for the selected decision"
    elif status == "blocked":
        execute_label = "execute  unavailable until blockers are resolved"
    else:
        execute_label = "execute  request approval only when the selected decision is executable"
    return [
        "details  expand the full Decision Card",
        "why      show why-not comparison",
        "sim      show simulation",
        execute_label,
        "refine   adjust this decision with feedback",
    ]


def _candidate_by_id(candidates: list[Mapping[str, Any]], candidate_id: str) -> Mapping[str, Any]:
    for candidate in candidates:
        if str(candidate.get("candidate_id") or "") == candidate_id:
            return candidate
    return {}


def _candidate_title(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("title") or candidate.get("intent") or "").strip()


def _candidate_recommendation(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("recommended_action")
        or candidate.get("intent")
        or candidate.get("expected_result")
        or ""
    ).strip()


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
