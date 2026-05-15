from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from spice.llm.core.types import LLMModelConfig, LLMRequest, LLMResponse, LLMStreamChunk


class LLMProviderError(RuntimeError):
    """Base class for normalized provider errors."""


class LLMTransportError(LLMProviderError):
    """Raised when transport-level execution fails."""


class LLMAuthError(LLMProviderError):
    """Raised when authentication/authorization fails."""


class LLMRateLimitError(LLMProviderError):
    """Raised when provider indicates throttling/rate limits."""


class LLMResponseError(LLMProviderError):
    """Raised when provider response cannot be consumed."""


class LLMProvider(ABC):
    provider_id: str

    @abstractmethod
    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        """Send one request using the provided model configuration."""

    def stream(
        self,
        request: LLMRequest,
        model: LLMModelConfig,
    ) -> Iterator[LLMStreamChunk]:
        """Stream response chunks when supported, otherwise fall back to generate()."""
        response = self.generate(request, model)
        yield LLMStreamChunk(
            text=response.output_text,
            finish_reason=response.finish_reason,
            raw_event={
                "stream_fallback": "generate",
                "provider_id": response.provider_id,
                "model_id": response.model_id,
                "request_id": response.request_id,
                "raw_payload": response.raw_payload,
            },
        )
