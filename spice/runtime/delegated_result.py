from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from spice.llm.util import extract_first_json_object
from spice.perception.delegated import (
    DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION,
    DelegatedPerceptionArtifact,
    ExecutorReportArtifact,
    build_delegated_perception_artifact,
    delegated_perception_context_from_artifact,
)


DELEGATED_PERCEPTION_NORMALIZER_SCHEMA_VERSION = "spice.delegated_perception_normalizer.v1"

_DECISION_LIKE_FIELDS = frozenset(
    {
        "decision",
        "decision_id",
        "selected",
        "selected_candidate_id",
        "recommendation",
        "approve",
        "execute",
        "execution",
        "outcome",
    }
)


@dataclass(frozen=True, slots=True)
class DelegatedPerceptionNormalizationResult:
    artifact: DelegatedPerceptionArtifact
    context: dict[str, Any]
    parser_status: str = "parsed"
    fallback_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    source_payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = DELEGATED_PERCEPTION_NORMALIZER_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parser_status": self.parser_status,
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
            "source_payload": dict(self.source_payload),
            "artifact": self.artifact.to_payload(),
            "context": dict(self.context),
        }


def normalize_delegated_perception_result(
    *,
    executor_report: ExecutorReportArtifact | Mapping[str, Any],
    request: Mapping[str, Any] | Any | None = None,
    consent_id: str = "",
    input_context_refs: list[str] | None = None,
    context_strategy: str = "delegated",
    created_at: datetime | None = None,
) -> DelegatedPerceptionNormalizationResult:
    report = _coerce_report(executor_report)
    request_payload = _payload(request)
    delegated_plan = _mapping(request_payload.get("delegated_plan"))
    expected_output = str(request_payload.get("expected_output") or delegated_plan.get("expected_output") or "").strip()
    parsed_payload, parser_status, fallback_reason = _parse_report_payload(report)
    warnings: list[str] = []
    limitations = [str(item) for item in report.limitations if str(item)]

    if fallback_reason:
        limitations.append(fallback_reason)
        warnings.append(fallback_reason)

    ignored_fields = sorted(set(parsed_payload) & _DECISION_LIKE_FIELDS)
    if ignored_fields:
        limitations.append("executor_decision_fields_ignored")
        warnings.append("executor_decision_fields_ignored")

    sources, source_warnings = _normalize_sources(
        parsed_payload.get("sources")
        or parsed_payload.get("source_refs")
        or parsed_payload.get("citations")
        or []
    )
    warnings.extend(source_warnings)
    limitations.extend(source_warnings)

    findings, finding_warnings = _normalize_findings(
        parsed_payload.get("findings")
        or parsed_payload.get("facts")
        or parsed_payload.get("observations")
        or [],
        source_ids={str(item.get("source_id") or "") for item in sources if str(item.get("source_id") or "")},
    )
    source_binding = _source_binding_summary(findings=findings, sources=sources)
    warnings.extend(finding_warnings)
    limitations.extend(finding_warnings)

    status = _normalize_status(str(parsed_payload.get("status") or report.status or ""))
    if parser_status != "parsed" and status == "completed":
        status = "failed"
    confidence = _normalize_confidence(parsed_payload.get("confidence"), findings=findings, status=status)
    query = str(request_payload.get("query") or report.query or parsed_payload.get("query") or "").strip()
    normalized_consent_id = str(consent_id or request_payload.get("consent_id") or "").strip()
    normalized_context_strategy = str(
        request_payload.get("context_strategy") or context_strategy or "delegated"
    )
    refs = _unique_strings(
        [
            *[str(item) for item in input_context_refs or [] if str(item)],
            *[str(item) for item in _list(request_payload.get("input_context_refs")) if str(item)],
        ]
    )
    summary = _summary(parsed_payload=parsed_payload, findings=findings, fallback_reason=fallback_reason)

    artifact = build_delegated_perception_artifact(
        executor_id=report.executor_id,
        query=query,
        input_context_refs=refs,
        consent_id=normalized_consent_id,
        request_ref=str(request_payload.get("request_id") or report.request_ref or ""),
        executor_report_ref=report.report_id,
        executor_run_ref=report.executor_run_ref,
        status=status,
        findings=findings,
        sources=sources,
        limitations=_unique_strings(limitations),
        confidence=confidence,
        summary=summary,
        scope=report.scope,
        permission_mode=report.permission_mode,
        context_strategy=normalized_context_strategy,
        created_at=created_at,
        metadata={
            "source": "spice.runtime.delegated_result",
            "parser_status": parser_status,
            "fallback_reason": fallback_reason,
            "raw_output_retained_in_executor_report": True,
            "ignored_executor_fields": ignored_fields,
            "delegated_plan": delegated_plan,
            "expected_output": expected_output,
            "planner_executor_id": str(delegated_plan.get("executor_id") or ""),
            "finding_source_binding": source_binding,
        },
    )
    context = delegated_perception_context_from_artifact(artifact)
    if context.get("schema_version") != DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION:
        warnings.append("unexpected_delegated_perception_context_schema")
    return DelegatedPerceptionNormalizationResult(
        artifact=artifact,
        context=context,
        parser_status=parser_status,
        fallback_reason=fallback_reason,
        warnings=_unique_strings(warnings),
        source_payload=_redacted_source_payload(parsed_payload),
    )


def _parse_report_payload(report: ExecutorReportArtifact) -> tuple[dict[str, Any], str, str]:
    if report.structured_output:
        return dict(report.structured_output), "parsed", ""
    raw = str(report.raw_output or "").strip()
    if not raw:
        return {}, "empty", "executor_report_empty"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(raw)
        if not extracted:
            return {}, "malformed", "executor_report_output_malformed"
        try:
            payload = json.loads(extracted)
        except json.JSONDecodeError:
            return {}, "malformed", "executor_report_output_malformed"
    if not isinstance(payload, dict):
        return {}, "malformed", "executor_report_output_not_object"
    return payload, "parsed", ""


def _normalize_findings(value: Any, *, source_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(_mappings(value), start=1):
        text = str(
            item.get("text")
            or item.get("finding")
            or item.get("summary")
            or item.get("content")
            or ""
        ).strip()
        if not text:
            warnings.append(f"finding.{index}.missing_text")
            continue
        source_refs = _strings(item.get("source_refs") or item.get("sources") or item.get("citations"))
        missing_refs = [ref for ref in source_refs if ref not in source_ids]
        valid_refs = [ref for ref in source_refs if ref in source_ids]
        limitations = _strings(item.get("limitations"))
        confidence = _float_or_none(item.get("confidence"))
        if missing_refs:
            limitations.append("missing_source_refs:" + ",".join(missing_refs))
            warnings.append(f"finding.{index}.missing_source_refs")
            confidence = min(confidence if confidence is not None else 0.35, 0.35)
        if not valid_refs:
            warnings.append(f"finding.{index}.no_source_refs")
        findings.append(
            {
                "finding_id": str(item.get("finding_id") or item.get("id") or f"finding.{index}"),
                "text": text,
                "confidence": confidence,
                "source_refs": valid_refs,
                "limitations": _unique_strings(limitations),
                "metadata": {
                    "source": "executor_report",
                    "raw_index": index,
                },
            }
        )
    if not findings and isinstance(value, str) and value.strip():
        findings.append(
            {
                "finding_id": "finding.1",
                "text": value.strip(),
                "confidence": 0.35,
                "source_refs": [],
                "limitations": ["text_fallback_without_source_refs"],
                "metadata": {"source": "executor_report_text_fallback"},
            }
        )
        warnings.append("findings_text_fallback_without_source_refs")
    return findings, _unique_strings(warnings)


def _normalize_sources(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    for index, item in enumerate(_mappings(value), start=1):
        source_id = str(item.get("source_id") or item.get("id") or f"source.{index}").strip()
        uri = str(item.get("uri") or item.get("url") or item.get("path") or "").strip()
        excerpt = str(item.get("excerpt") or item.get("quote") or item.get("snippet") or "").strip()
        missing: list[str] = []
        if not uri:
            missing.append("uri")
        if not excerpt:
            missing.append("excerpt")
        metadata = _mapping(item.get("metadata"))
        if missing:
            metadata = {
                **metadata,
                "incomplete_source": True,
                "missing_fields": missing,
            }
            warnings.append(f"{source_id}.incomplete_source:" + ",".join(missing))
        sources.append(
            {
                "source_id": source_id,
                "source_type": str(item.get("source_type") or item.get("type") or "executor_report"),
                "title": str(item.get("title") or item.get("name") or ""),
                "uri": uri,
                "excerpt": excerpt,
                "observed_by": str(item.get("observed_by") or ""),
                "accessed_at": str(item.get("accessed_at") or _timestamp()),
                "verification_status": str(
                    item.get("verification_status")
                    or ("unverified" if missing else "reported_by_executor")
                ),
                "metadata": metadata,
            }
        )
    return sources, _unique_strings(warnings)


def _normalize_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"completed", "success", "ok"}:
        return "completed"
    if normalized in {"blocked", "denied"}:
        return "blocked"
    if normalized in {"failed", "error"}:
        return "failed"
    if normalized in {"pending"}:
        return "pending"
    return "completed"


def _source_binding_summary(
    *,
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    source_ids = {str(item.get("source_id") or "") for item in sources if str(item.get("source_id") or "")}
    sourced = 0
    unsourced = 0
    missing_refs = 0
    for finding in findings:
        refs = _strings(finding.get("source_refs"))
        if refs:
            sourced += 1
        else:
            unsourced += 1
        missing_refs += len([ref for ref in refs if ref not in source_ids])
    return {
        "source_count": len(source_ids),
        "finding_count": len(findings),
        "sourced_finding_count": sourced,
        "unsourced_finding_count": unsourced,
        "missing_source_ref_count": missing_refs,
        "status": "complete" if findings and sourced == len(findings) and missing_refs == 0 else "partial",
    }


def _normalize_confidence(value: Any, *, findings: list[dict[str, Any]], status: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    if status != "completed":
        return "low"
    numeric_values = [
        item.get("confidence")
        for item in findings
        if isinstance(item.get("confidence"), (int, float))
    ]
    if not numeric_values:
        return "medium" if findings else "low"
    average = sum(float(item) for item in numeric_values) / len(numeric_values)
    if average >= 0.75:
        return "high"
    if average >= 0.45:
        return "medium"
    return "low"


def _summary(
    *,
    parsed_payload: Mapping[str, Any],
    findings: list[dict[str, Any]],
    fallback_reason: str,
) -> str:
    explicit = str(parsed_payload.get("summary") or parsed_payload.get("answer") or "").strip()
    if explicit:
        return explicit
    if findings:
        first = str(findings[0].get("text") or "")
        suffix = f" (+{len(findings) - 1} more finding(s))" if len(findings) > 1 else ""
        return first[:220].rstrip() + suffix
    if fallback_reason:
        return f"Delegated perception could not normalize executor output: {fallback_reason}."
    return "Delegated perception has no findings yet."


def _coerce_report(report: ExecutorReportArtifact | Mapping[str, Any]) -> ExecutorReportArtifact:
    if isinstance(report, ExecutorReportArtifact):
        return report
    return ExecutorReportArtifact.from_payload(dict(report))


def _payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_payload") and callable(value.to_payload):
        payload = value.to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _redacted_source_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in ("raw_output", "transcript", "logs"):
        if key in result:
            result[key] = "[redacted: retained in executor report]"
    return result


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return _unique_strings([str(item or "").strip() for item in value if str(item or "").strip()])


def _unique_strings(value: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
