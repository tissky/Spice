from __future__ import annotations

import json
import time
import urllib.error as urllib_error
import urllib.request as urllib_request
from collections.abc import Iterator
from typing import Any

from spice.llm.core.provider import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
)
from spice.llm.core.types import LLMModelConfig, LLMRequest, LLMStreamChunk


def build_chat_payload(
    *,
    request: LLMRequest,
    model: LLMModelConfig,
    max_tokens_field: str = "max_tokens",
    include_stream_false: bool = False,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    system_text = request.system_text.strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": request.input_text})

    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": messages,
    }
    if include_stream_false:
        payload["stream"] = False
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    if model.max_tokens is not None:
        payload[max_tokens_field] = model.max_tokens
    if model.response_format_hint == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def chat_completions_endpoint(base_url: str | None, default_base_url: str) -> str:
    normalized = (base_url or default_base_url).strip()
    if not normalized:
        normalized = default_base_url
    return normalized.rstrip("/") + "/chat/completions"


def post_chat_completions(
    *,
    provider_label: str,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: float | None,
) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    http_request = urllib_request.Request(
        endpoint,
        data=body,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib_request.urlopen(http_request, timeout=timeout_sec) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        raise normalize_http_error(exc, provider_label=provider_label) from exc
    except urllib_error.URLError as exc:
        raise LLMTransportError(f"{provider_label} request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMTransportError(
            f"{provider_label} request timed out after {timeout_sec}s."
        ) from exc
    latency_ms = int((time.perf_counter() - start) * 1000)
    return parse_response_json(response_body, provider_label=provider_label), latency_ms


def stream_chat_completions(
    *,
    provider_label: str,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_sec: float | None,
) -> Iterator[LLMStreamChunk]:
    stream_payload = dict(payload)
    stream_payload["stream"] = True
    body = json.dumps(stream_payload, ensure_ascii=True).encode("utf-8")
    http_request = urllib_request.Request(
        endpoint,
        data=body,
        headers={**headers, "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(http_request, timeout=timeout_sec) as response:
            for event in iter_sse_json_events(response, provider_label=provider_label):
                chunk = extract_stream_chunk(event, provider_label=provider_label)
                if chunk.text or chunk.finish_reason:
                    yield chunk
    except urllib_error.HTTPError as exc:
        raise normalize_http_error(exc, provider_label=provider_label) from exc
    except urllib_error.URLError as exc:
        raise LLMTransportError(f"{provider_label} streaming request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMTransportError(
            f"{provider_label} streaming request timed out after {timeout_sec}s."
        ) from exc


def iter_sse_json_events(response: Any, *, provider_label: str) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    while True:
        raw_line = response.readline()
        if raw_line == b"" or raw_line == "":
            if data_lines:
                event = _parse_sse_data("\n".join(data_lines), provider_label=provider_label)
                if event is not None:
                    yield event
            return
        line = _decode_sse_line(raw_line)
        if not line:
            if data_lines:
                event = _parse_sse_data("\n".join(data_lines), provider_label=provider_label)
                data_lines = []
                if event is not None:
                    yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())


def extract_stream_chunk(
    event: dict[str, Any],
    *,
    provider_label: str,
) -> LLMStreamChunk:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return LLMStreamChunk(raw_event=event)
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError(f"{provider_label} stream first choice must be an object.")
    delta = first.get("delta")
    text = ""
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            text = content
    finish_reason = first.get("finish_reason")
    return LLMStreamChunk(
        text=text,
        finish_reason=str(finish_reason or ""),
        raw_event=event,
    )


def parse_response_json(raw: str, *, provider_label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"{provider_label} response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError(f"{provider_label} response JSON must be an object.")
    return payload


def _parse_sse_data(data: str, *, provider_label: str) -> dict[str, Any] | None:
    if data == "[DONE]":
        return None
    return parse_response_json(data, provider_label=provider_label)


def _decode_sse_line(raw_line: bytes | str) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8").rstrip("\r\n")
    return raw_line.rstrip("\r\n")


def extract_choice(payload: dict[str, Any], *, provider_label: str) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError(f"{provider_label} response missing choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError(f"{provider_label} first choice must be an object.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError(f"{provider_label} first choice missing message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(f"{provider_label} first choice message content is empty.")
    return content, str(first.get("finish_reason") or "")


def normalize_http_error(exc: urllib_error.HTTPError, *, provider_label: str) -> Exception:
    body = safe_error_body(exc)
    reason = str(getattr(exc, "reason", "") or getattr(exc, "msg", "") or "")
    message = (
        f"{provider_label} request failed "
        f"(status={exc.code}): {body or reason or '<no response body>'}"
    )
    if exc.code in (401, 403):
        return LLMAuthError(message)
    if exc.code == 429:
        return LLMRateLimitError(message)
    if exc.code in (400, 404, 422):
        return LLMResponseError(message)
    return LLMTransportError(message)


def safe_error_body(exc: urllib_error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return body.decode("utf-8").strip()
    except Exception:
        return repr(body)
