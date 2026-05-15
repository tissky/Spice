from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.types import PayloadRecord, payload_value, safe_dataclass_from_payload


SPICE_STREAM_EVENT_SCHEMA_VERSION = "spice.stream_event.v1"

SPICE_STREAM_EVENT_TYPES = (
    "status",
    "response_delta",
    "response_done",
    "execution_output",
    "artifact_ref",
    "error",
)


@dataclass(slots=True)
class SpiceStreamEvent(PayloadRecord):
    event_type: str
    text: str = ""
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SPICE_STREAM_EVENT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SPICE_STREAM_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"stream event schema_version must be {SPICE_STREAM_EVENT_SCHEMA_VERSION}"
            )
        if self.event_type not in SPICE_STREAM_EVENT_TYPES:
            allowed = ", ".join(SPICE_STREAM_EVENT_TYPES)
            raise ValueError(f"stream event_type must be one of [{allowed}]")
        self.text = str(self.text or "")
        self.artifact_refs = _artifact_refs(self.artifact_refs)
        self.metadata = _dict(self.metadata)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SpiceStreamEvent":
        item = safe_dataclass_from_payload(cls, payload)
        item.artifact_refs = _artifact_refs(payload.get("artifact_refs"))
        item.metadata = _dict(payload.get("metadata"))
        item.validate()
        return item

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return payload_value(self)


def build_stream_event(
    event_type: str,
    *,
    text: str = "",
    artifact_refs: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpiceStreamEvent:
    event = SpiceStreamEvent(
        event_type=event_type,
        text=text,
        artifact_refs=artifact_refs or [],
        metadata=metadata or {},
    )
    event.validate()
    return event


def stream_status_event(label: str, detail: str = "", **metadata: Any) -> SpiceStreamEvent:
    event_metadata = {"label": str(label or ""), **_dict(metadata)}
    if detail:
        event_metadata["detail"] = str(detail)
    return build_stream_event(
        "status",
        text=str(label or ""),
        metadata=event_metadata,
    )


def stream_response_delta_event(
    text: str,
    *,
    unit: str = "block",
    **metadata: Any,
) -> SpiceStreamEvent:
    return build_stream_event(
        "response_delta",
        text=text,
        metadata={"unit": str(unit or "block"), **_dict(metadata)},
    )


def stream_response_done_event(
    text: str = "",
    *,
    chunk_count: int | None = None,
    **metadata: Any,
) -> SpiceStreamEvent:
    event_metadata = _dict(metadata)
    if chunk_count is not None:
        event_metadata["chunk_count"] = chunk_count
    return build_stream_event(
        "response_done",
        text=text,
        metadata=event_metadata,
    )


def stream_execution_output_event(
    text: str,
    *,
    stream_name: str = "stdout",
    **metadata: Any,
) -> SpiceStreamEvent:
    return build_stream_event(
        "execution_output",
        text=text,
        metadata={"stream": str(stream_name or "stdout"), **_dict(metadata)},
    )


def stream_artifact_ref_event(
    artifact_refs: list[dict[str, Any]],
    *,
    text: str = "",
    **metadata: Any,
) -> SpiceStreamEvent:
    return build_stream_event(
        "artifact_ref",
        text=text,
        artifact_refs=artifact_refs,
        metadata=_dict(metadata),
    )


def stream_error_event(text: str, **metadata: Any) -> SpiceStreamEvent:
    return build_stream_event(
        "error",
        text=text,
        metadata=_dict(metadata),
    )


def _artifact_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append({str(key): val for key, val in item.items()})
    return result


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
