from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Mapping

from spice.decision.general.types import payload_value, safe_dataclass_from_payload


DELEGATED_PERCEPTION_SCHEMA_VERSION = "spice.delegated_perception.v1"
DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION = "spice.delegated_perception_context.v1"
EXECUTOR_REPORT_SCHEMA_VERSION = "spice.executor_report.v1"
INVESTIGATION_CONSENT_SCHEMA_VERSION = "spice.investigation_consent.v1"

INVESTIGATION_CONSENT_PENDING = "pending"
INVESTIGATION_CONSENT_GRANTED = "granted"
INVESTIGATION_CONSENT_REJECTED = "rejected"
INVESTIGATION_CONSENT_EXPIRED = "expired"

DEFAULT_INVESTIGATION_ALLOWED_ACTIONS = (
    "web_search",
    "read_web_page",
    "repo_inspection",
)
DEFAULT_INVESTIGATION_DENIED_ACTIONS = (
    "write_file",
    "patch",
    "install",
    "test_run",
    "terminal_command",
    "delete",
    "move",
)


@dataclass(frozen=True, slots=True)
class DelegatedPerceptionSource:
    source_id: str
    source_type: str = "executor_report"
    title: str = ""
    uri: str = ""
    excerpt: str = ""
    observed_by: str = ""
    accessed_at: str = ""
    verification_status: str = "reported_by_executor"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class DelegatedPerceptionFinding:
    finding_id: str
    text: str
    confidence: float | None = None
    source_refs: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class InvestigationConsentBudget:
    max_duration_sec: int = 120
    max_sources: int = 10
    max_repo_files: int = 20
    max_tokens: int = 20_000

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class InvestigationConsent:
    consent_id: str
    executor_id: str
    query: str
    created_at: str
    scope: str = "read_only_investigation"
    permission_mode: str = "read_only"
    allowed_actions: list[str] = field(default_factory=lambda: list(DEFAULT_INVESTIGATION_ALLOWED_ACTIONS))
    denied_actions: list[str] = field(default_factory=lambda: list(DEFAULT_INVESTIGATION_DENIED_ACTIONS))
    status: str = INVESTIGATION_CONSENT_PENDING
    resolved_at: str = ""
    expires_at: str = ""
    actor: str = "user"
    response: str = ""
    reason: str = ""
    input_context_refs: list[str] = field(default_factory=list)
    budget: InvestigationConsentBudget = field(default_factory=InvestigationConsentBudget)
    schema_version: str = INVESTIGATION_CONSENT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InvestigationConsent":
        if not isinstance(payload, Mapping):
            raise ValueError("Investigation consent payload must be a mapping.")
        return cls(
            consent_id=_required_string(payload, "consent_id", label="Investigation consent"),
            executor_id=str(payload.get("executor_id") or ""),
            query=str(payload.get("query") or ""),
            created_at=str(payload.get("created_at") or ""),
            scope=str(payload.get("scope") or "read_only_investigation"),
            permission_mode=str(payload.get("permission_mode") or "read_only"),
            allowed_actions=_string_list(
                payload.get("allowed_actions"),
                default=DEFAULT_INVESTIGATION_ALLOWED_ACTIONS,
            ),
            denied_actions=_string_list(
                payload.get("denied_actions"),
                default=DEFAULT_INVESTIGATION_DENIED_ACTIONS,
            ),
            status=str(payload.get("status") or INVESTIGATION_CONSENT_PENDING),
            resolved_at=str(payload.get("resolved_at") or ""),
            expires_at=str(payload.get("expires_at") or ""),
            actor=str(payload.get("actor") or "user"),
            response=str(payload.get("response") or ""),
            reason=str(payload.get("reason") or ""),
            input_context_refs=[str(item) for item in _list(payload.get("input_context_refs")) if str(item)],
            budget=_budget(payload.get("budget")),
            schema_version=str(payload.get("schema_version") or INVESTIGATION_CONSENT_SCHEMA_VERSION),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class ExecutorReportArtifact:
    report_id: str
    executor_id: str
    created_at: str
    scope: str = "read_only_investigation"
    permission_mode: str = "read_only"
    query: str = ""
    request_ref: str = ""
    executor_run_ref: str = ""
    status: str = "completed"
    raw_output: str = ""
    structured_output: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    schema_version: str = EXECUTOR_REPORT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExecutorReportArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("Executor report payload must be a mapping.")
        return cls(
            report_id=_required_string(payload, "report_id", label="Executor report"),
            executor_id=str(payload.get("executor_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            scope=str(payload.get("scope") or "read_only_investigation"),
            permission_mode=str(payload.get("permission_mode") or "read_only"),
            query=str(payload.get("query") or ""),
            request_ref=str(payload.get("request_ref") or ""),
            executor_run_ref=str(payload.get("executor_run_ref") or ""),
            status=str(payload.get("status") or "completed"),
            raw_output=str(payload.get("raw_output") or ""),
            structured_output=_mapping(payload.get("structured_output")),
            limitations=[str(item) for item in _list(payload.get("limitations")) if str(item)],
            schema_version=str(payload.get("schema_version") or EXECUTOR_REPORT_SCHEMA_VERSION),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class DelegatedPerceptionArtifact:
    perception_id: str
    delegation_id: str
    executor_id: str
    created_at: str
    scope: str = "read_only_investigation"
    permission_mode: str = "read_only"
    query: str = ""
    context_strategy: str = "delegated"
    input_context_refs: list[str] = field(default_factory=list)
    consent_id: str = ""
    request_ref: str = ""
    executor_report_ref: str = ""
    executor_run_ref: str = ""
    status: str = "pending"
    findings: list[DelegatedPerceptionFinding] = field(default_factory=list)
    sources: list[DelegatedPerceptionSource] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence: str = "medium"
    summary: str = ""
    schema_version: str = DELEGATED_PERCEPTION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DelegatedPerceptionArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("Delegated perception payload must be a mapping.")
        return cls(
            perception_id=_required_string(payload, "perception_id", label="Delegated perception"),
            delegation_id=str(payload.get("delegation_id") or ""),
            executor_id=str(payload.get("executor_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            scope=str(payload.get("scope") or "read_only_investigation"),
            permission_mode=str(payload.get("permission_mode") or "read_only"),
            query=str(payload.get("query") or ""),
            context_strategy=str(payload.get("context_strategy") or "delegated"),
            input_context_refs=[
                str(item) for item in _list(payload.get("input_context_refs")) if str(item)
            ],
            consent_id=str(payload.get("consent_id") or ""),
            request_ref=str(payload.get("request_ref") or ""),
            executor_report_ref=str(payload.get("executor_report_ref") or ""),
            executor_run_ref=str(payload.get("executor_run_ref") or ""),
            status=str(payload.get("status") or "pending"),
            findings=[_finding(item) for item in _mappings(payload.get("findings"))],
            sources=[_source(item) for item in _mappings(payload.get("sources"))],
            limitations=[str(item) for item in _list(payload.get("limitations")) if str(item)],
            confidence=str(payload.get("confidence") or "medium"),
            summary=str(payload.get("summary") or ""),
            schema_version=str(
                payload.get("schema_version") or DELEGATED_PERCEPTION_SCHEMA_VERSION
            ),
            metadata=_mapping(payload.get("metadata")),
        )


def build_investigation_consent(
    *,
    executor_id: str,
    query: str,
    allowed_actions: list[str] | None = None,
    denied_actions: list[str] | None = None,
    scope: str = "read_only_investigation",
    permission_mode: str = "read_only",
    status: str = INVESTIGATION_CONSENT_PENDING,
    actor: str = "user",
    input_context_refs: list[str] | None = None,
    budget: InvestigationConsentBudget | Mapping[str, Any] | None = None,
    expires_in_sec: int = 600,
    created_at: datetime | None = None,
    expires_at: str = "",
    consent_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> InvestigationConsent:
    created_dt = _datetime(created_at)
    created = _timestamp(created_dt)
    normalized_executor_id = str(executor_id or "").strip()
    normalized_query = str(query or "").strip()
    normalized_budget = _budget(budget)
    normalized_expires_at = expires_at or _timestamp(
        created_dt + timedelta(seconds=max(0, int(expires_in_sec)))
    )
    normalized_consent_id = consent_id or _investigation_consent_id(
        executor_id=normalized_executor_id,
        created_at=created,
        query=normalized_query,
    )
    return InvestigationConsent(
        consent_id=normalized_consent_id,
        executor_id=normalized_executor_id,
        query=normalized_query,
        created_at=created,
        scope=str(scope or "read_only_investigation"),
        permission_mode=str(permission_mode or "read_only"),
        allowed_actions=_string_list(
            allowed_actions,
            default=DEFAULT_INVESTIGATION_ALLOWED_ACTIONS,
        ),
        denied_actions=_string_list(
            denied_actions,
            default=DEFAULT_INVESTIGATION_DENIED_ACTIONS,
        ),
        status=str(status or INVESTIGATION_CONSENT_PENDING),
        expires_at=normalized_expires_at,
        actor=str(actor or "user"),
        input_context_refs=[str(item) for item in input_context_refs or [] if str(item)],
        budget=normalized_budget,
        metadata=dict(metadata or {}),
    )


def resolve_investigation_consent(
    consent: InvestigationConsent | Mapping[str, Any],
    *,
    status: str,
    actor: str = "user",
    response: str = "",
    reason: str = "",
    resolved_at: datetime | None = None,
) -> InvestigationConsent:
    if status not in {
        INVESTIGATION_CONSENT_GRANTED,
        INVESTIGATION_CONSENT_REJECTED,
        INVESTIGATION_CONSENT_EXPIRED,
    }:
        raise ValueError(f"Unsupported investigation consent status transition: {status}")
    existing = consent if isinstance(consent, InvestigationConsent) else InvestigationConsent.from_payload(consent)
    if existing.status != INVESTIGATION_CONSENT_PENDING:
        raise ValueError(
            f"Investigation consent {existing.consent_id} is not pending; "
            f"current status is {existing.status}."
        )
    return InvestigationConsent(
        consent_id=existing.consent_id,
        executor_id=existing.executor_id,
        query=existing.query,
        created_at=existing.created_at,
        scope=existing.scope,
        permission_mode=existing.permission_mode,
        allowed_actions=list(existing.allowed_actions),
        denied_actions=list(existing.denied_actions),
        status=status,
        resolved_at=_timestamp(resolved_at),
        expires_at=existing.expires_at,
        actor=actor or existing.actor,
        response=response or status,
        reason=reason,
        input_context_refs=list(existing.input_context_refs),
        budget=existing.budget,
        schema_version=existing.schema_version,
        metadata={
            **dict(existing.metadata),
            "resolved_by": "spice.perception.delegated",
        },
    )


def build_executor_report_artifact(
    *,
    executor_id: str,
    query: str = "",
    raw_output: str = "",
    structured_output: Mapping[str, Any] | None = None,
    status: str = "completed",
    scope: str = "read_only_investigation",
    permission_mode: str = "read_only",
    request_ref: str = "",
    executor_run_ref: str = "",
    limitations: list[str] | None = None,
    created_at: datetime | None = None,
    report_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> ExecutorReportArtifact:
    created = _timestamp(created_at)
    normalized_executor_id = str(executor_id or "").strip()
    normalized_query = str(query or "").strip()
    normalized_raw = str(raw_output or "")
    normalized_structured = dict(structured_output or {})
    normalized_report_id = report_id or _executor_report_id(
        executor_id=normalized_executor_id,
        created_at=created,
        query=normalized_query,
        raw_output=normalized_raw,
        structured_output=normalized_structured,
    )
    return ExecutorReportArtifact(
        report_id=normalized_report_id,
        executor_id=normalized_executor_id,
        created_at=created,
        scope=str(scope or "read_only_investigation"),
        permission_mode=str(permission_mode or "read_only"),
        query=normalized_query,
        request_ref=str(request_ref or ""),
        executor_run_ref=str(executor_run_ref or ""),
        status=str(status or "completed"),
        raw_output=normalized_raw,
        structured_output=normalized_structured,
        limitations=[str(item) for item in limitations or [] if str(item)],
        metadata=dict(metadata or {}),
    )


def build_delegated_perception_artifact(
    *,
    executor_id: str,
    query: str = "",
    delegation_id: str = "",
    input_context_refs: list[str] | None = None,
    consent_id: str = "",
    request_ref: str = "",
    executor_report_ref: str = "",
    executor_run_ref: str = "",
    status: str = "pending",
    findings: list[DelegatedPerceptionFinding | Mapping[str, Any]] | None = None,
    sources: list[DelegatedPerceptionSource | Mapping[str, Any]] | None = None,
    limitations: list[str] | None = None,
    confidence: str = "medium",
    summary: str = "",
    scope: str = "read_only_investigation",
    permission_mode: str = "read_only",
    context_strategy: str = "delegated",
    created_at: datetime | None = None,
    perception_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> DelegatedPerceptionArtifact:
    created = _timestamp(created_at)
    normalized_executor_id = str(executor_id or "").strip()
    normalized_query = str(query or "").strip()
    normalized_findings = [_finding(item) for item in findings or []]
    normalized_sources = [_source(item) for item in sources or []]
    normalized_summary = str(summary or "").strip()
    if not normalized_summary:
        normalized_summary = _summary_from_findings(normalized_findings)
    base_digest = _digest([created, normalized_executor_id, normalized_query, normalized_summary])
    normalized_delegation_id = delegation_id or f"delegation.{_compact_time(created)}.{base_digest}"
    normalized_perception_id = perception_id or f"delegated.{_compact_time(created)}.{base_digest}"
    return DelegatedPerceptionArtifact(
        perception_id=normalized_perception_id,
        delegation_id=normalized_delegation_id,
        executor_id=normalized_executor_id,
        created_at=created,
        scope=str(scope or "read_only_investigation"),
        permission_mode=str(permission_mode or "read_only"),
        query=normalized_query,
        context_strategy=str(context_strategy or "delegated"),
        input_context_refs=[
            str(item) for item in input_context_refs or [] if str(item)
        ],
        consent_id=str(consent_id or ""),
        request_ref=str(request_ref or ""),
        executor_report_ref=str(executor_report_ref or ""),
        executor_run_ref=str(executor_run_ref or ""),
        status=str(status or "pending"),
        findings=normalized_findings,
        sources=normalized_sources,
        limitations=[str(item) for item in limitations or [] if str(item)],
        confidence=str(confidence or "medium"),
        summary=normalized_summary,
        metadata=dict(metadata or {}),
    )


def delegated_perception_context_from_artifact(
    artifact: DelegatedPerceptionArtifact | Mapping[str, Any],
) -> dict[str, Any]:
    payload = artifact.to_payload() if isinstance(artifact, DelegatedPerceptionArtifact) else dict(artifact)
    findings = [
        _compact_finding(_mapping(item))
        for item in _list(payload.get("findings"))[:12]
        if isinstance(item, Mapping)
    ]
    sources = [
        _compact_source(_mapping(item))
        for item in _list(payload.get("sources"))[:12]
        if isinstance(item, Mapping)
    ]
    return {
        "schema_version": DELEGATED_PERCEPTION_CONTEXT_SCHEMA_VERSION,
        "source": "delegated_perception",
        "perception_id": str(payload.get("perception_id") or ""),
        "delegation_id": str(payload.get("delegation_id") or ""),
        "executor_id": str(payload.get("executor_id") or ""),
        "scope": str(payload.get("scope") or "read_only_investigation"),
        "permission_mode": str(payload.get("permission_mode") or "read_only"),
        "query": str(payload.get("query") or ""),
        "status": str(payload.get("status") or ""),
        "summary": _shorten(str(payload.get("summary") or ""), 900),
        "findings": findings,
        "sources": sources,
        "limitations": [
            _shorten(str(item), 240)
            for item in _list(payload.get("limitations"))[:12]
            if str(item)
        ],
        "confidence": str(payload.get("confidence") or "medium"),
        "consent_id": str(payload.get("consent_id") or ""),
        "request_ref": str(payload.get("request_ref") or ""),
        "executor_report_ref": str(payload.get("executor_report_ref") or ""),
        "executor_run_ref": str(payload.get("executor_run_ref") or ""),
    }


def _finding(value: DelegatedPerceptionFinding | Mapping[str, Any]) -> DelegatedPerceptionFinding:
    if isinstance(value, DelegatedPerceptionFinding):
        payload = value.to_payload()
    else:
        payload = dict(value)
    source_refs = [str(item) for item in _list(payload.get("source_refs")) if str(item)]
    limitations = [str(item) for item in _list(payload.get("limitations")) if str(item)]
    confidence = _float_or_none(payload.get("confidence"))
    if not source_refs:
        if "no_source_refs" not in limitations:
            limitations.append("no_source_refs")
        confidence = min(confidence if confidence is not None else 0.35, 0.35)
    return DelegatedPerceptionFinding(
        finding_id=str(payload.get("finding_id") or ""),
        text=str(payload.get("text") or ""),
        confidence=confidence,
        source_refs=source_refs,
        limitations=limitations,
        metadata=_mapping(payload.get("metadata")),
    )


def _source(value: DelegatedPerceptionSource | Mapping[str, Any]) -> DelegatedPerceptionSource:
    if isinstance(value, DelegatedPerceptionSource):
        return value
    payload = dict(value)
    return safe_dataclass_from_payload(
        DelegatedPerceptionSource,
        {
            "source_id": str(payload.get("source_id") or ""),
            "source_type": str(payload.get("source_type") or "executor_report"),
            "title": str(payload.get("title") or ""),
            "uri": str(payload.get("uri") or ""),
            "excerpt": str(payload.get("excerpt") or ""),
            "observed_by": str(payload.get("observed_by") or ""),
            "accessed_at": str(payload.get("accessed_at") or ""),
            "verification_status": str(
                payload.get("verification_status") or "reported_by_executor"
            ),
            "metadata": _mapping(payload.get("metadata")),
        },
    )


def _compact_finding(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": str(payload.get("finding_id") or ""),
        "text": _shorten(str(payload.get("text") or ""), 700),
        "confidence": payload.get("confidence"),
        "source_refs": [str(item) for item in _list(payload.get("source_refs"))[:8] if str(item)],
        "limitations": [
            _shorten(str(item), 180)
            for item in _list(payload.get("limitations"))[:6]
            if str(item)
        ],
    }


def _compact_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(payload.get("source_id") or ""),
        "source_type": str(payload.get("source_type") or ""),
        "title": _shorten(str(payload.get("title") or ""), 180),
        "uri": str(payload.get("uri") or ""),
        "excerpt": _shorten(str(payload.get("excerpt") or ""), 700),
        "observed_by": str(payload.get("observed_by") or ""),
        "accessed_at": str(payload.get("accessed_at") or ""),
        "verification_status": str(payload.get("verification_status") or ""),
    }


def _summary_from_findings(findings: list[DelegatedPerceptionFinding]) -> str:
    if not findings:
        return "Delegated perception has no findings yet."
    first = _shorten(findings[0].text, 220)
    suffix = f" (+{len(findings) - 1} more finding(s))" if len(findings) > 1 else ""
    return first + suffix


def _executor_report_id(
    *,
    executor_id: str,
    created_at: str,
    query: str,
    raw_output: str,
    structured_output: Mapping[str, Any],
) -> str:
    digest = _digest([created_at, executor_id, query, raw_output, repr(sorted(structured_output.items()))])
    return f"executor_report.{_compact_time(created_at)}.{digest}"


def _investigation_consent_id(*, executor_id: str, created_at: str, query: str) -> str:
    digest = _digest([created_at, executor_id, query])
    return f"investigation.{_compact_time(created_at)}.{digest}"


def _datetime(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return _datetime(value).isoformat().replace("+00:00", "Z")


def _compact_time(value: str) -> str:
    return value.replace("-", "").replace(":", "").replace(".", "_").replace("+00:00", "Z")


def _digest(parts: list[str]) -> str:
    return sha256("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _required_string(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label} payload missing required string: {key}")
    return value


def _budget(value: InvestigationConsentBudget | Mapping[str, Any] | None) -> InvestigationConsentBudget:
    if isinstance(value, InvestigationConsentBudget):
        return value
    if isinstance(value, Mapping):
        return safe_dataclass_from_payload(InvestigationConsentBudget, dict(value))
    return InvestigationConsentBudget()


def _string_list(value: Any, *, default: tuple[str, ...]) -> list[str]:
    items = _list(value)
    if not items:
        items = list(default)
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
