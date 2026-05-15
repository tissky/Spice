from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from spice.llm.core import LLMClient, LLMRequest


ComposerStreamCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ComposerLLMOutput:
    raw_output: str
    provider_id: str = ""
    model_id: str = ""
    request_id: str = ""
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ComposerStreamError(RuntimeError):
    def __init__(self, message: str, *, raw_output: str = "", metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.metadata = dict(metadata or {})


def generate_or_stream_composer_output(
    *,
    client: LLMClient,
    request: LLMRequest,
    stream_callback: ComposerStreamCallback | None = None,
) -> ComposerLLMOutput:
    """Run composer LLM output, optionally streaming text while keeping a full buffer for validation."""

    if stream_callback is None or not hasattr(client, "stream"):
        response = client.generate(request)
        return ComposerLLMOutput(
            raw_output=response.output_text,
            provider_id=response.provider_id,
            model_id=response.model_id,
            request_id=response.request_id,
            finish_reason=response.finish_reason,
            metadata={},
        )

    raw_parts: list[str] = []
    chunk_count = 0
    raw_text_chunk_count = 0
    finish_reason = ""
    request_id = ""
    model_id = ""
    provider_fallback_generate = False
    display_gate = _ComposerStreamDisplayGate(stream_callback)
    try:
        for chunk in client.stream(request):
            chunk_count += 1
            raw_event = dict(chunk.raw_event or {})
            if not request_id:
                request_id = str(raw_event.get("request_id") or raw_event.get("id") or "")
            if not model_id:
                model_id = str(raw_event.get("model") or "")
            if raw_event.get("stream_fallback") == "generate":
                provider_fallback_generate = True
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.text:
                raw_text_chunk_count += 1
                raw_parts.append(chunk.text)
                display_gate.push(chunk.text)
    except Exception as exc:
        raw_output = "".join(raw_parts)
        metadata = _streaming_metadata(
            chunk_count=chunk_count,
            raw_text_chunk_count=raw_text_chunk_count,
            displayed_text_chunk_count=display_gate.displayed_chunk_count,
            finish_reason=finish_reason,
            provider_fallback_generate=provider_fallback_generate,
            status="stream_error",
            error=str(exc),
        )
        raise ComposerStreamError(str(exc), raw_output=raw_output, metadata=metadata) from exc

    raw_output = "".join(raw_parts)
    metadata = _streaming_metadata(
        chunk_count=chunk_count,
        raw_text_chunk_count=raw_text_chunk_count,
        displayed_text_chunk_count=display_gate.displayed_chunk_count,
        finish_reason=finish_reason,
        provider_fallback_generate=provider_fallback_generate,
        status="streamed",
    )
    if not raw_output.strip():
        raise ComposerStreamError(
            "stream produced no response text",
            raw_output=raw_output,
            metadata={**metadata, "status": "stream_error", "error": "stream produced no response text"},
        )
    return ComposerLLMOutput(
        raw_output=raw_output,
        request_id=request_id,
        model_id=model_id,
        finish_reason=finish_reason,
        metadata=metadata,
    )


def mark_streaming_valid(metadata: dict[str, Any]) -> dict[str, Any]:
    streaming = dict(metadata.get("streaming") or {})
    if not streaming:
        return dict(metadata)
    streaming["valid"] = True
    streaming["source"] = "validated_streamed_composer_result"
    return {**metadata, "streaming": streaming}


def mark_streaming_invalid(metadata: dict[str, Any], *, reason: str) -> dict[str, Any]:
    streaming = dict(metadata.get("streaming") or {})
    if not streaming:
        return dict(metadata)
    streaming["valid"] = False
    streaming["fallback_reason"] = reason
    streaming["source"] = "invalid_streamed_composer_result"
    return {**metadata, "streaming": streaming}


def streamed_response_was_displayed(result_metadata: dict[str, Any]) -> bool:
    streaming = dict(result_metadata.get("streaming") or {})
    return streaming.get("mode") == "provider_token_stream" and bool(streaming.get("displayed_to_user"))


def streamed_response_is_valid(result_metadata: dict[str, Any]) -> bool:
    streaming = dict(result_metadata.get("streaming") or {})
    return streamed_response_was_displayed(result_metadata) and bool(streaming.get("valid"))


class _ComposerStreamDisplayGate:
    """Only display streamed composer text when it appears to be natural language.

    Some providers obey a JSON-object response hint or independently wrap composer
    output as {"response": "..."} even though the UI needs natural text. We still
    buffer the full raw output for parser/validator, but structured-looking output
    is hidden until parsing finishes and the TUI can display the parsed response.
    """

    def __init__(self, callback: ComposerStreamCallback) -> None:
        self.callback = callback
        self.mode = "unknown"
        self.buffer: list[str] = []
        self.displayed_chunk_count = 0

    def push(self, text: str) -> None:
        if not text:
            return
        if self.mode == "display":
            self.callback(text)
            self.displayed_chunk_count += 1
            return
        if self.mode == "suppress":
            return

        self.buffer.append(text)
        buffered = "".join(self.buffer)
        stripped = buffered.lstrip()
        if not stripped:
            return
        if _looks_like_structured_composer_output(stripped):
            self.mode = "suppress"
            self.buffer.clear()
            return

        self.mode = "display"
        self.callback(buffered)
        self.displayed_chunk_count += 1
        self.buffer.clear()


def _looks_like_structured_composer_output(text: str) -> bool:
    return text.startswith("{") or text.startswith("[") or text.startswith("```")


def _streaming_metadata(
    *,
    chunk_count: int,
    raw_text_chunk_count: int,
    displayed_text_chunk_count: int,
    finish_reason: str,
    provider_fallback_generate: bool,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "streaming": {
            "mode": "provider_token_stream",
            "source": "provider_stream",
            "status": status,
            "chunk_count": chunk_count,
            "text_chunk_count": displayed_text_chunk_count,
            "raw_text_chunk_count": raw_text_chunk_count,
            "finish_reason": finish_reason,
            "displayed_to_user": displayed_text_chunk_count > 0,
            "provider_fallback_generate": provider_fallback_generate,
        }
    }
    if error:
        payload["streaming"]["error"] = error
    return payload
