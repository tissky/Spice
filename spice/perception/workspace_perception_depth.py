from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


WORKSPACE_PERCEPTION_DEPTH_SCHEMA_VERSION = "spice.workspace_perception_depth.v1"

WORKSPACE_PERCEPTION_DEPTH_AUTO = "auto"
WORKSPACE_PERCEPTION_DEPTH_NORMAL = "normal"
WORKSPACE_PERCEPTION_DEPTH_DEEP = "deep"
WORKSPACE_PERCEPTION_DEPTH_NATIVE = "native"

_DEPTH_RANK = {
    WORKSPACE_PERCEPTION_DEPTH_NORMAL: 1,
    WORKSPACE_PERCEPTION_DEPTH_DEEP: 2,
    WORKSPACE_PERCEPTION_DEPTH_NATIVE: 3,
}

_BUDGETS: dict[str, dict[str, int | None]] = {
    WORKSPACE_PERCEPTION_DEPTH_NORMAL: {
        "max_rounds": 10,
        "max_tool_calls": 80,
        "max_tool_calls_per_round": 10,
        "max_blocked_tool_calls": 20,
        "max_blocked_tool_calls_per_round": 5,
        "max_files_read": 60,
        "max_chars_per_file": 12_000,
        "total_char_budget": 500_000,
        "planner_max_tokens": 4_000,
    },
    WORKSPACE_PERCEPTION_DEPTH_DEEP: {
        "max_rounds": 25,
        "max_tool_calls": 200,
        "max_tool_calls_per_round": 16,
        "max_blocked_tool_calls": 40,
        "max_blocked_tool_calls_per_round": 5,
        "max_files_read": 120,
        "max_chars_per_file": 25_000,
        "total_char_budget": 1_500_000,
        "planner_max_tokens": 8_000,
    },
    WORKSPACE_PERCEPTION_DEPTH_NATIVE: {
        "max_rounds": 90,
        "max_tool_calls": 500,
        "max_tool_calls_per_round": 20,
        "max_blocked_tool_calls": 80,
        "max_blocked_tool_calls_per_round": 5,
        "max_files_read": 300,
        "max_chars_per_file": 60_000,
        "total_char_budget": 5_000_000,
        "planner_max_tokens": None,
    },
}

_DEEP_ANSWER_MODES = {"detailed", "report"}
_NATIVE_ANSWER_MODES = {"native"}
_REPO_DOMAINS = {"repo", "mixed"}
_REPORT_DOMAINS = {"external", "mixed"}

_CURRENT_IMPLEMENTATION_PHRASES = (
    "基于当前实现",
    "基于实际代码",
    "基于当前代码",
    "读取本地",
    "读一下本地",
    "当前实现",
    "实际代码",
    "当前代码",
    "current implementation",
    "actual implementation",
    "current code",
    "current codebase",
)

_DEEP_REVIEW_PHRASES = (
    "详细 review",
    "完整 review",
    "深度 review",
    "深入看",
    "完整分析",
    "详细分析",
    "深度分析",
    "deep review",
    "deep dive",
    "detailed review",
    "full review",
)

_REPORT_PHRASES = (
    "报告",
    "证据",
    "来源",
    "读到了哪些",
    "report",
    "evidence",
    "sources",
    "what did you read",
)


@dataclass(frozen=True, slots=True)
class WorkspacePerceptionDepthBudget:
    depth: str
    max_rounds: int
    max_tool_calls: int
    max_tool_calls_per_round: int
    max_blocked_tool_calls: int
    max_blocked_tool_calls_per_round: int
    max_files_read: int
    max_chars_per_file: int
    total_char_budget: int
    planner_max_tokens: int | None
    native: bool = False
    explicit_opt_in: bool = False
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_PERCEPTION_DEPTH_SCHEMA_VERSION,
            "depth": self.depth,
            "max_rounds": self.max_rounds,
            "max_tool_calls": self.max_tool_calls,
            "max_tool_calls_per_round": self.max_tool_calls_per_round,
            "max_blocked_tool_calls": self.max_blocked_tool_calls,
            "max_blocked_tool_calls_per_round": self.max_blocked_tool_calls_per_round,
            "max_files_read": self.max_files_read,
            "max_chars_per_file": self.max_chars_per_file,
            "total_char_budget": self.total_char_budget,
            "planner_max_tokens": self.planner_max_tokens,
            "native": self.native,
            "explicit_opt_in": self.explicit_opt_in,
            "reason": self.reason,
        }


def resolve_workspace_perception_depth(
    *,
    config: Mapping[str, Any] | None = None,
    evidence_domain: str = "",
    answer_mode: str = "",
    user_input: str = "",
    route_policy: Mapping[str, Any] | None = None,
    requested_depth: str = "",
    native_opt_in: bool = False,
) -> WorkspacePerceptionDepthBudget:
    """Resolve read-only workspace investigation depth.

    Native depth is intentionally unavailable to LLM route/planner output. It is
    allowed only when config explicitly selects native or caller passes
    ``native_opt_in=True`` for a user-triggered command.
    """

    config_payload = _mapping(config)
    workspace_config = _workspace_perception_config(config_payload)
    configured_depth = _normalize_depth(
        workspace_config.get("depth")
        or config_payload.get("workspace_perception_depth")
        or ""
    )
    request_depth = _normalize_depth(requested_depth)
    explicit_native = native_opt_in or configured_depth == WORKSPACE_PERCEPTION_DEPTH_NATIVE

    reasons: list[str] = []
    if configured_depth and configured_depth != WORKSPACE_PERCEPTION_DEPTH_AUTO:
        depth = configured_depth
        reasons.append(f"configured_depth={configured_depth}")
    elif request_depth and request_depth != WORKSPACE_PERCEPTION_DEPTH_AUTO:
        depth = request_depth
        reasons.append(f"requested_depth={request_depth}")
    else:
        depth, reason = _auto_depth(
            evidence_domain=evidence_domain,
            answer_mode=answer_mode,
            user_input=user_input,
            route_policy=route_policy,
        )
        reasons.append(reason)

    if depth == WORKSPACE_PERCEPTION_DEPTH_NATIVE and not explicit_native:
        depth = WORKSPACE_PERCEPTION_DEPTH_DEEP
        reasons.append("native depth requires explicit opt-in; capped at deep")

    budget = _budget(depth=depth, reason="; ".join(_unique(reasons)), explicit_opt_in=explicit_native)
    return _apply_overrides(budget, workspace_config)


def workspace_perception_depth_from_payload(
    payload: Mapping[str, Any] | None,
) -> WorkspacePerceptionDepthBudget:
    data = _mapping(payload)
    depth = _normalize_depth(data.get("depth")) or WORKSPACE_PERCEPTION_DEPTH_NORMAL
    budget = _budget(
        depth=depth,
        reason=str(data.get("reason") or "payload"),
        explicit_opt_in=bool(data.get("explicit_opt_in")),
    )
    return WorkspacePerceptionDepthBudget(
        depth=budget.depth,
        max_rounds=_int(data.get("max_rounds"), budget.max_rounds),
        max_tool_calls=_int(data.get("max_tool_calls"), budget.max_tool_calls),
        max_tool_calls_per_round=_int(
            data.get("max_tool_calls_per_round"),
            budget.max_tool_calls_per_round,
        ),
        max_blocked_tool_calls=_int(data.get("max_blocked_tool_calls"), budget.max_blocked_tool_calls),
        max_blocked_tool_calls_per_round=_int(
            data.get("max_blocked_tool_calls_per_round"),
            budget.max_blocked_tool_calls_per_round,
        ),
        max_files_read=_int(data.get("max_files_read"), budget.max_files_read),
        max_chars_per_file=_int(data.get("max_chars_per_file"), budget.max_chars_per_file),
        total_char_budget=_int(data.get("total_char_budget"), budget.total_char_budget),
        planner_max_tokens=_optional_int(data.get("planner_max_tokens"), budget.planner_max_tokens),
        native=bool(data.get("native")) or depth == WORKSPACE_PERCEPTION_DEPTH_NATIVE,
        explicit_opt_in=bool(data.get("explicit_opt_in")),
        reason=str(data.get("reason") or budget.reason),
    )


def _auto_depth(
    *,
    evidence_domain: str,
    answer_mode: str,
    user_input: str,
    route_policy: Mapping[str, Any] | None,
) -> tuple[str, str]:
    policy = _mapping(route_policy)
    policy_answer_mode = str(policy.get("answer_mode") or "").strip().lower()
    policy_strategy = str(policy.get("final_strategy") or policy.get("perception_strategy") or "").strip().lower()
    normalized_answer_mode = str(answer_mode or policy_answer_mode or "").strip().lower()
    normalized_domain = str(evidence_domain or policy.get("evidence_domain") or "").strip().lower()
    lowered = str(user_input or "").lower()

    if normalized_answer_mode in _NATIVE_ANSWER_MODES:
        return WORKSPACE_PERCEPTION_DEPTH_NATIVE, "answer_mode=native"
    if normalized_answer_mode in _DEEP_ANSWER_MODES:
        return WORKSPACE_PERCEPTION_DEPTH_DEEP, f"answer_mode={normalized_answer_mode}"
    if _contains_any(lowered, _DEEP_REVIEW_PHRASES):
        return WORKSPACE_PERCEPTION_DEPTH_DEEP, "user wording requested deep workspace review"
    if _contains_any(lowered, _REPORT_PHRASES):
        return WORKSPACE_PERCEPTION_DEPTH_DEEP, "user wording requested report/evidence depth"
    if normalized_domain in _REPORT_DOMAINS and "delegated" in policy_strategy:
        return WORKSPACE_PERCEPTION_DEPTH_DEEP, f"evidence_domain={normalized_domain}"
    if normalized_domain in _REPO_DOMAINS:
        return WORKSPACE_PERCEPTION_DEPTH_NORMAL, f"evidence_domain={normalized_domain}"
    if _contains_any(lowered, _CURRENT_IMPLEMENTATION_PHRASES):
        return WORKSPACE_PERCEPTION_DEPTH_NORMAL, "user wording requested current implementation evidence"
    return WORKSPACE_PERCEPTION_DEPTH_NORMAL, "default workspace perception depth"


def _budget(*, depth: str, reason: str, explicit_opt_in: bool) -> WorkspacePerceptionDepthBudget:
    normalized = _normalize_depth(depth) or WORKSPACE_PERCEPTION_DEPTH_NORMAL
    if normalized == WORKSPACE_PERCEPTION_DEPTH_AUTO:
        normalized = WORKSPACE_PERCEPTION_DEPTH_NORMAL
    raw = _BUDGETS[normalized]
    return WorkspacePerceptionDepthBudget(
        depth=normalized,
        max_rounds=int(raw["max_rounds"] or 0),
        max_tool_calls=int(raw["max_tool_calls"] or 0),
        max_tool_calls_per_round=int(raw["max_tool_calls_per_round"] or 0),
        max_blocked_tool_calls=int(raw["max_blocked_tool_calls"] or 0),
        max_blocked_tool_calls_per_round=int(raw["max_blocked_tool_calls_per_round"] or 0),
        max_files_read=int(raw["max_files_read"] or 0),
        max_chars_per_file=int(raw["max_chars_per_file"] or 0),
        total_char_budget=int(raw["total_char_budget"] or 0),
        planner_max_tokens=_optional_int(raw["planner_max_tokens"], None),
        native=normalized == WORKSPACE_PERCEPTION_DEPTH_NATIVE,
        explicit_opt_in=explicit_opt_in,
        reason=reason,
    )


def _apply_overrides(
    budget: WorkspacePerceptionDepthBudget,
    workspace_config: Mapping[str, Any],
) -> WorkspacePerceptionDepthBudget:
    return WorkspacePerceptionDepthBudget(
        depth=budget.depth,
        max_rounds=_int(workspace_config.get("max_rounds"), budget.max_rounds),
        max_tool_calls=_int(workspace_config.get("max_tool_calls"), budget.max_tool_calls),
        max_tool_calls_per_round=_int(
            workspace_config.get("max_tool_calls_per_round"),
            budget.max_tool_calls_per_round,
        ),
        max_blocked_tool_calls=_int(
            workspace_config.get("max_blocked_tool_calls"),
            budget.max_blocked_tool_calls,
        ),
        max_blocked_tool_calls_per_round=_int(
            workspace_config.get("max_blocked_tool_calls_per_round"),
            budget.max_blocked_tool_calls_per_round,
        ),
        max_files_read=_int(workspace_config.get("max_files_read"), budget.max_files_read),
        max_chars_per_file=_int(workspace_config.get("max_chars_per_file"), budget.max_chars_per_file),
        total_char_budget=_int(workspace_config.get("total_char_budget"), budget.total_char_budget),
        planner_max_tokens=_optional_int(workspace_config.get("planner_max_tokens"), budget.planner_max_tokens),
        native=budget.native,
        explicit_opt_in=budget.explicit_opt_in,
        reason=budget.reason,
    )


def _workspace_perception_config(config: Mapping[str, Any]) -> dict[str, Any]:
    nested = config.get("workspace_perception")
    if isinstance(nested, Mapping):
        return dict(nested)
    return {}


def _normalize_depth(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {
        WORKSPACE_PERCEPTION_DEPTH_AUTO,
        WORKSPACE_PERCEPTION_DEPTH_NORMAL,
        WORKSPACE_PERCEPTION_DEPTH_DEEP,
        WORKSPACE_PERCEPTION_DEPTH_NATIVE,
    }:
        return raw
    return ""


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
