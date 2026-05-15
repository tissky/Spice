from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


RESPONSE_DEPTH_POLICY_SCHEMA_VERSION = "spice.response_depth_policy.v1"

ANSWER_MODE_BRIEF = "brief"
ANSWER_MODE_NORMAL = "normal"
ANSWER_MODE_DETAILED = "detailed"
ANSWER_MODE_REPORT = "report"
ANSWER_MODE_NATIVE = "native"

_ANSWER_MODE_RANK = {
    ANSWER_MODE_BRIEF: 0,
    ANSWER_MODE_NORMAL: 1,
    ANSWER_MODE_DETAILED: 2,
    ANSWER_MODE_REPORT: 3,
    ANSWER_MODE_NATIVE: 4,
}

_BUDGETS = {
    ANSWER_MODE_BRIEF: {"min_tokens": 800, "max_tokens": 1200, "max_chars": 3500, "timeout_sec": 45.0},
    ANSWER_MODE_NORMAL: {"min_tokens": 2000, "max_tokens": 3000, "max_chars": 8000, "timeout_sec": 75.0},
    ANSWER_MODE_DETAILED: {"min_tokens": 4000, "max_tokens": 6000, "max_chars": 14000, "timeout_sec": 120.0},
    ANSWER_MODE_REPORT: {"min_tokens": 8000, "max_tokens": 12000, "max_chars": 24000, "timeout_sec": 180.0},
    # Native leaves max_tokens unset so the provider/model dispatch config can use
    # its own ceiling. The local response validator still keeps a char ceiling.
    ANSWER_MODE_NATIVE: {"min_tokens": 0, "max_tokens": None, "max_chars": 24000, "timeout_sec": 240.0},
}

_DETAILED_PHRASES = (
    "详细",
    "具体",
    "完整",
    "展开",
    "怎么做",
    "为什么",
    "两周",
    "步骤",
    "计划",
    "detailed",
    "detail",
    "explain",
    "why",
    "plan",
    "steps",
    "roadmap",
)

_REPORT_PHRASES = (
    "报告",
    "证据",
    "来源",
    "读到了哪些",
    "基于当前实现",
    "基于实际代码",
    "当前代码",
    "当前实现",
    "深度",
    "调研",
    "report",
    "evidence",
    "sources",
    "implementation status",
    "deep dive",
    "research",
)

_BRIEF_PHRASES = (
    "简短",
    "简单说",
    "一句话",
    "brief",
    "short",
    "quick",
    "concise",
)


@dataclass(frozen=True, slots=True)
class ResponseDepthBudget:
    answer_mode: str
    min_tokens: int
    max_tokens: int | None
    max_chars: int
    timeout_sec: float
    native: bool = False
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RESPONSE_DEPTH_POLICY_SCHEMA_VERSION,
            "answer_mode": self.answer_mode,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "max_chars": self.max_chars,
            "timeout_sec": self.timeout_sec,
            "native": self.native,
            "reason": self.reason,
        }

    @property
    def guidance(self) -> str:
        if self.answer_mode == ANSWER_MODE_BRIEF:
            return "Keep this compact. Give the recommendation and the main tradeoff."
        if self.answer_mode == ANSWER_MODE_NORMAL:
            return "Give a normal conversational answer with enough context to act."
        if self.answer_mode == ANSWER_MODE_DETAILED:
            return "Give a detailed answer with concrete reasoning, steps, and caveats."
        if self.answer_mode == ANSWER_MODE_REPORT:
            return "Give a report-style answer grounded in evidence, sources, limitations, and tradeoffs."
        return "Use the model's native response budget, while staying grounded in the provided facts."


def resolve_response_depth_budget(
    *,
    answer_mode: str = "",
    evidence_domain: str = "",
    evidence_context: Mapping[str, Any] | None = None,
    user_input: str = "",
    config: Mapping[str, Any] | None = None,
    composer_kind: str = "",
    action: str = "",
) -> ResponseDepthBudget:
    """Resolve how much room a response composer should get.

    This policy controls response budget only. It does not relax source truth,
    approval, execution, or candidate validators.
    """

    config_payload = _mapping(config)
    configured_mode = _configured_mode(config_payload)
    if configured_mode == ANSWER_MODE_NATIVE:
        return _budget(ANSWER_MODE_NATIVE, "native depth explicitly configured")

    raw_answer_mode = str(answer_mode or "").strip().lower()
    mode = _normalize_answer_mode(answer_mode)
    reasons: list[str] = []
    if mode != ANSWER_MODE_NORMAL:
        reasons.append(f"answer_mode={mode}")
    if configured_mode and raw_answer_mode in {"", ANSWER_MODE_NORMAL}:
        mode = configured_mode
        reasons.append(f"configured_response_depth={configured_mode}")

    evidence_mode, evidence_reason = _mode_from_evidence(
        evidence_domain=evidence_domain,
        evidence_context=_mapping(evidence_context),
    )
    if evidence_reason:
        mode = _stronger_mode(mode, evidence_mode)
        reasons.append(evidence_reason)

    wording_mode, wording_reason = _mode_from_user_wording(user_input)
    if wording_reason:
        mode = _stronger_mode(mode, wording_mode)
        reasons.append(wording_reason)

    action_mode, action_reason = _mode_from_action(action=action, composer_kind=composer_kind)
    if action_reason:
        mode = _stronger_mode(mode, action_mode)
        reasons.append(action_reason)

    if configured_mode and f"configured_response_depth={configured_mode}" not in reasons:
        mode = _stronger_mode(mode, configured_mode)
        reasons.append(f"configured_response_depth={configured_mode}")

    return _budget(mode, "; ".join(_unique(reasons)) or "default response depth")


def response_depth_from_payload(payload: Mapping[str, Any] | None) -> ResponseDepthBudget:
    data = _mapping(payload)
    mode = _normalize_answer_mode(str(data.get("answer_mode") or ""))
    budget = _budget(mode, "payload")
    return ResponseDepthBudget(
        answer_mode=budget.answer_mode,
        min_tokens=_int(data.get("min_tokens"), budget.min_tokens),
        max_tokens=_optional_int(data.get("max_tokens"), budget.max_tokens),
        max_chars=_int(data.get("max_chars"), budget.max_chars),
        timeout_sec=float(data.get("timeout_sec") or budget.timeout_sec),
        native=bool(data.get("native")) or mode == ANSWER_MODE_NATIVE,
        reason=str(data.get("reason") or budget.reason),
    )


def _budget(mode: str, reason: str) -> ResponseDepthBudget:
    normalized = _normalize_answer_mode(mode)
    raw = _BUDGETS[normalized]
    return ResponseDepthBudget(
        answer_mode=normalized,
        min_tokens=int(raw["min_tokens"] or 0),
        max_tokens=_optional_int(raw["max_tokens"], None),
        max_chars=int(raw["max_chars"] or 0),
        timeout_sec=float(raw["timeout_sec"] or 30.0),
        native=normalized == ANSWER_MODE_NATIVE,
        reason=reason,
    )


def _mode_from_evidence(
    *,
    evidence_domain: str,
    evidence_context: Mapping[str, Any],
) -> tuple[str, str]:
    requirements = _mapping(evidence_context.get("requirements"))
    mode = _normalize_answer_mode(str(requirements.get("answer_mode") or ""))
    domain = str(evidence_domain or requirements.get("evidence_domain") or "").strip().lower()
    if domain in {"external", "mixed"}:
        mode = _stronger_mode(mode, ANSWER_MODE_REPORT)
    elif domain in {"repo", "url"}:
        mode = _stronger_mode(mode, ANSWER_MODE_DETAILED)

    workspace = _mapping(evidence_context.get("workspace"))
    url = _mapping(evidence_context.get("url"))
    delegated = _mapping(evidence_context.get("delegated"))
    has_workspace = bool(workspace.get("present"))
    has_url = bool(url.get("present"))
    has_delegated = bool(delegated.get("present"))
    if has_delegated or (has_url and has_workspace):
        mode = _stronger_mode(mode, ANSWER_MODE_REPORT)
    elif has_workspace or has_url:
        mode = _stronger_mode(mode, ANSWER_MODE_DETAILED)

    if mode != ANSWER_MODE_NORMAL:
        return mode, f"evidence_context={mode}"
    return mode, ""


def _mode_from_user_wording(user_input: str) -> tuple[str, str]:
    lowered = str(user_input or "").lower()
    if not lowered:
        return ANSWER_MODE_NORMAL, ""
    if _contains_any(lowered, _BRIEF_PHRASES):
        return ANSWER_MODE_BRIEF, "user requested brief answer"
    if _contains_any(lowered, _REPORT_PHRASES):
        return ANSWER_MODE_REPORT, "user wording requested report/evidence depth"
    if _contains_any(lowered, _DETAILED_PHRASES):
        return ANSWER_MODE_DETAILED, "user wording requested detailed answer"
    return ANSWER_MODE_NORMAL, ""


def _mode_from_action(*, action: str, composer_kind: str) -> tuple[str, str]:
    normalized_action = str(action or "").strip().lower()
    kind = str(composer_kind or "").strip().lower()
    if normalized_action in {"plan_candidate", "compare_alternative", "refine_decision"}:
        return ANSWER_MODE_DETAILED, f"follow_up_action={normalized_action}"
    if normalized_action in {"investigation_request", "answer_from_evidence"}:
        return ANSWER_MODE_REPORT, f"follow_up_action={normalized_action}"
    if kind == "execution_response":
        return ANSWER_MODE_NORMAL, ""
    return ANSWER_MODE_NORMAL, ""


def _configured_mode(config: Mapping[str, Any]) -> str:
    for key in (
        "response_depth",
        "composer_response_depth",
        "composer_answer_mode",
        "answer_mode",
    ):
        mode = _normalize_answer_mode(str(config.get(key) or ""))
        if mode != ANSWER_MODE_NORMAL or str(config.get(key) or "").strip().lower() == ANSWER_MODE_NORMAL:
            return mode
    native = config.get("response_depth_native") or config.get("composer_native_response_depth")
    return ANSWER_MODE_NATIVE if bool(native) else ""


def _normalize_answer_mode(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"concise", "short", "quick"}:
        return ANSWER_MODE_BRIEF
    if normalized in {"detail", "full"}:
        return ANSWER_MODE_DETAILED
    if normalized in {"research", "analysis", "deep", "deep_dive"}:
        return ANSWER_MODE_REPORT
    if normalized in _ANSWER_MODE_RANK:
        return normalized
    return ANSWER_MODE_NORMAL


def _stronger_mode(current: str, proposed: str) -> str:
    normalized_current = _normalize_answer_mode(current)
    normalized_proposed = _normalize_answer_mode(proposed)
    if _ANSWER_MODE_RANK[normalized_proposed] > _ANSWER_MODE_RANK[normalized_current]:
        return normalized_proposed
    return normalized_current


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
