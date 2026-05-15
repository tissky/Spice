from __future__ import annotations

import json
from typing import Any

from spice.llm.util import extract_first_json_object, strip_markdown_fences


COMPOSER_RESPONSE_FIELD_ALIASES = ("response", "message", "answer", "text", "content")
DEFAULT_COMPOSER_RESPONSE_MAX_CHARS = 2400


def parse_composer_response_text(
    raw_output: str,
    *,
    max_chars: int = DEFAULT_COMPOSER_RESPONSE_MAX_CHARS,
) -> str:
    text = str(raw_output or "").strip()
    if not text:
        raise ValueError("composer output was empty")

    extracted = extract_first_json_object(text)
    if extracted is not None:
        payload = json.loads(extracted)
        if not isinstance(payload, dict):
            raise ValueError("composer output JSON must be an object")
        response = _response_from_payload(payload)
        return _validate_response_text(response, max_chars=max_chars)

    stripped = strip_markdown_fences(text).strip()
    if not stripped:
        raise ValueError("composer output was empty")
    if _looks_like_structured_dump(stripped):
        raise ValueError("composer output appears to be structured data, not a response")
    return _validate_response_text(stripped, max_chars=max_chars)


def _response_from_payload(payload: dict[str, Any]) -> str:
    for key in COMPOSER_RESPONSE_FIELD_ALIASES:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        text = _text_from_content_parts(value)
        if text:
            return text
    raise ValueError("composer output JSON is missing response text")


def _text_from_content_parts(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item.strip())
                continue
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return ""


def _validate_response_text(text: str, *, max_chars: int) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("composer output was empty")
    if len(normalized) > max_chars:
        raise ValueError("composer output was too long")
    if _looks_like_structured_dump(normalized):
        raise ValueError("composer output appears to be structured data, not a response")
    return normalized


def _looks_like_structured_dump(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(("{", "[")):
        return True
    if stripped.startswith(("```json", "```")):
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    lowered = stripped.lower()
    structured_markers = (
        '"schema_version"',
        '"candidate_id"',
        '"decision_id"',
        '"response":',
        "'response':",
        "schema_version:",
    )
    return any(marker in lowered for marker in structured_markers)
