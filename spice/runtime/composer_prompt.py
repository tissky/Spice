from __future__ import annotations

from typing import Any, Mapping

from spice.runtime.composer_context import compact_composer_context


COMPOSER_BASE_CONSTRAINTS = (
    "Do not re-select the winner.",
    "Do not change the selected option.",
    "Do not change scores.",
    "Do not change approval or execution state.",
    "Do not invent artifact ids.",
    "Do not expose raw JSON, schema names, or internal debug fields.",
)


def build_slim_composer_prompt_payload(
    *,
    task: str,
    facts: Mapping[str, Any],
    tone: str = "Natural, concise agent voice.",
    extra_constraints: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "task": task,
        "output": "Write only the user-facing natural-language response as plain text. Do not wrap it in JSON.",
        "tone": tone,
        "constraints": [*COMPOSER_BASE_CONSTRAINTS, *extra_constraints],
        "facts": dict(facts),
    }


def slim_recent_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    compact = compact_composer_context(context)
    result: dict[str, Any] = {}
    for key in (
        "active_decision_frame",
        "latest_decision_artifact",
        "recent_conversation_turns",
        "recent_decisions",
        "recent_approvals",
        "recent_executions",
        "session_summary",
        "memory_summary",
        "executor_affordance",
        "executor_capabilities",
        "workspace_context",
        "url_context",
        "delegated_perception_context",
        "evidence_context",
    ):
        value = compact.get(key)
        if value:
            result[key] = value
    return result
