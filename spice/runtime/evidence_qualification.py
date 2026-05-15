from __future__ import annotations

from typing import Any, Mapping


def has_source_backed_evidence(payload: Mapping[str, Any] | None) -> bool:
    """Return whether a perception payload carries usable source-backed evidence.

    A perception id or a prose summary only proves that a perception step ran.
    It does not prove that Spice has evidence it can use for repo/code claims.
    """

    if not isinstance(payload, Mapping) or not payload:
        return False
    if payload.get("present") is False:
        return False
    if any(
        _positive_count(payload.get(key))
        for key in ("source_count", "fact_count", "snippet_count")
    ):
        return True
    return any(
        (
            _has_evidence_items(payload.get("files_read")),
            _has_evidence_items(payload.get("facts")),
            _has_evidence_items(payload.get("snippets")),
            _has_evidence_items(payload.get("sources")),
            _has_source_refs(payload.get("source_refs")),
        )
    )


def _has_evidence_items(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(_is_evidence_item(item) for item in value)


def _is_evidence_item(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    return any(
        str(value.get(key) or "").strip()
        for key in (
            "path",
            "source_path",
            "url",
            "source_url",
            "uri",
            "source_id",
            "text",
            "excerpt",
        )
    ) or _has_source_refs(value.get("source_refs"))


def _has_source_refs(value: Any) -> bool:
    return isinstance(value, list) and any(str(item or "").strip() for item in value)


def _positive_count(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        try:
            return int(value.strip()) > 0
        except ValueError:
            return False
    return False
