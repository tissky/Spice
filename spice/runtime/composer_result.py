from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


COMPOSER_RESULT_SCHEMA_VERSION = "spice.composer_result.v1"


@dataclass(frozen=True, slots=True)
class ComposerResult:
    enabled: bool
    status: str
    response_text: str
    deterministic_text: str
    composer_kind: str = ""
    model_provider: str = ""
    model_id: str = ""
    request_id: str = ""
    error: str = ""
    raw_output: str = ""
    fallback_reason: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": COMPOSER_RESULT_SCHEMA_VERSION,
            "composer_kind": self.composer_kind,
            "enabled": self.enabled,
            "status": self.status,
            "response_text": self.response_text,
            "deterministic_text": self.deterministic_text,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "request_id": self.request_id,
            "error": self.error,
            "raw_output": self.raw_output,
            "fallback_reason": self.fallback_reason,
            "facts": dict(self.facts),
            "metadata": dict(self.metadata),
        }
