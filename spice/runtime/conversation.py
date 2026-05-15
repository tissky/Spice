from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from spice.decision.general.types import payload_value
from spice.runtime.store import LocalJsonStore

CONVERSATION_TURN_SCHEMA_VERSION = "spice.conversation_turn.v1"
VALID_CONVERSATION_ROUTES = frozenset(
    {"new_decision", "follow_up", "command", "execution_request"}
)


@dataclass(slots=True)
class ConversationTurn:
    turn_id: str
    user_input: str
    route: str
    created_at: str
    session_id: str
    source_decision_id: str | None = None
    source_candidate_id: str | None = None
    source_run_id: str | None = None
    source_approval_id: str | None = None
    source_execution_id: str | None = None
    source_outcome_id: str | None = None
    response_id: str | None = None
    artifact_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONVERSATION_TURN_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConversationTurn":
        if not isinstance(payload, dict):
            raise ValueError("Conversation turn payload must be a dict.")
        return cls(
            schema_version=str(
                payload.get("schema_version") or CONVERSATION_TURN_SCHEMA_VERSION
            ),
            turn_id=_required_string(payload, "turn_id"),
            user_input=str(payload.get("user_input") or ""),
            route=_normalize_route(str(payload.get("route") or "new_decision")),
            created_at=str(payload.get("created_at") or ""),
            session_id=str(payload.get("session_id") or ""),
            source_decision_id=_optional_string(payload.get("source_decision_id")),
            source_candidate_id=_optional_string(payload.get("source_candidate_id")),
            source_run_id=_optional_string(payload.get("source_run_id")),
            source_approval_id=_optional_string(payload.get("source_approval_id")),
            source_execution_id=_optional_string(payload.get("source_execution_id")),
            source_outcome_id=_optional_string(payload.get("source_outcome_id")),
            response_id=_optional_string(payload.get("response_id")),
            artifact_refs=(
                dict(payload.get("artifact_refs"))
                if isinstance(payload.get("artifact_refs"), dict)
                else {}
            ),
            metadata=dict(payload.get("metadata"))
            if isinstance(payload.get("metadata"), dict)
            else {},
        )


def build_conversation_turn(
    *,
    user_input: str,
    route: str,
    session_id: str,
    created_at: datetime,
    source_decision_id: str | None = None,
    source_candidate_id: str | None = None,
    source_run_id: str | None = None,
    source_approval_id: str | None = None,
    source_execution_id: str | None = None,
    source_outcome_id: str | None = None,
    response_id: str | None = None,
    artifact_refs: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ConversationTurn:
    normalized_route = _normalize_route(route)
    timestamp = _timestamp(created_at)
    turn_id = _stable_turn_id(
        user_input=user_input,
        created_at=timestamp,
        source_decision_id=source_decision_id,
        source_run_id=source_run_id,
    )
    return ConversationTurn(
        turn_id=turn_id,
        user_input=user_input,
        route=normalized_route,
        created_at=timestamp,
        session_id=session_id,
        source_decision_id=source_decision_id,
        source_candidate_id=source_candidate_id,
        source_run_id=source_run_id,
        source_approval_id=source_approval_id,
        source_execution_id=source_execution_id,
        source_outcome_id=source_outcome_id,
        response_id=response_id or f"response.{turn_id.removeprefix('turn.')}",
        artifact_refs=dict(artifact_refs or {}),
        metadata=dict(metadata or {}),
    )


def save_conversation_turn(
    store: LocalJsonStore,
    turn: ConversationTurn,
) -> Path:
    return store.save_conversation_turn(turn.turn_id, turn.to_payload())


def _stable_turn_id(
    *,
    user_input: str,
    created_at: str,
    source_decision_id: str | None,
    source_run_id: str | None,
) -> str:
    digest = sha256(
        "\n".join(
            [
                user_input,
                created_at,
                source_decision_id or "",
                source_run_id or "",
            ]
        ).encode("utf-8")
    ).hexdigest()[:12]
    compact_time = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    return f"turn.{compact_time}.{digest}"


def _normalize_route(route: str) -> str:
    normalized = (route or "new_decision").strip().lower()
    if normalized not in VALID_CONVERSATION_ROUTES:
        valid = ", ".join(sorted(VALID_CONVERSATION_ROUTES))
        raise ValueError(f"Invalid conversation route: {route}. Valid values: {valid}.")
    return normalized


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Conversation turn payload missing required string: {key}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
