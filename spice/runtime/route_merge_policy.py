from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from spice.runtime.evidence_requirement import (
    EVIDENCE_DOMAIN_EXTERNAL,
    EVIDENCE_DOMAIN_MIXED,
    EVIDENCE_DOMAIN_REPO,
    EVIDENCE_DOMAIN_URL,
    EvidenceRequirement,
    detect_evidence_requirement,
)
from spice.runtime.resource_extractor import ResourceExtraction, extract_resources
from spice.runtime.workspace_scope import (
    WORKSPACE_SCOPE_ALLOWED,
    WORKSPACE_SCOPE_BLOCKED,
    WORKSPACE_SCOPE_NEEDS_CONFIRMATION,
    WORKSPACE_SCOPE_NEEDS_SELECTION,
    WORKSPACE_SCOPE_NONE,
    WorkspaceScopeResolution,
)


ROUTE_MERGE_POLICY_SCHEMA_VERSION = "spice.route_merge_policy.v1"

CONTEXT_STRATEGY_NONE = "none"
CONTEXT_STRATEGY_LOCAL_WORKSPACE = "local_workspace"
CONTEXT_STRATEGY_URL = "url"
CONTEXT_STRATEGY_DELEGATED = "delegated"
CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED = "local_then_delegated_if_insufficient"

FORCED_BY_EXPLICIT_URL = "explicit_url"
FORCED_BY_REPO_EVIDENCE_REQUIREMENT = "repo_evidence_requirement"
FORCED_BY_URL_EVIDENCE_REQUIREMENT = "url_evidence_requirement"
FORCED_BY_EXTERNAL_EVIDENCE_REQUIREMENT = "external_evidence_requirement"
FORCED_BY_WORKSPACE_SCOPE_ALLOWED = "workspace_scope_allowed"
FORCED_BY_WORKSPACE_SCOPE_NEEDS_CONFIRMATION = "workspace_scope_needs_confirmation"
FORCED_BY_WORKSPACE_SCOPE_NEEDS_SELECTION = "workspace_scope_needs_selection"
FORCED_BY_WORKSPACE_SCOPE_BLOCKED = "workspace_scope_blocked"
FORCED_BY_SEMANTIC_ROUTE = "semantic_route"

_WORKSPACE_STRATEGIES = frozenset(
    {CONTEXT_STRATEGY_LOCAL_WORKSPACE, CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED}
)
_DELEGATED_STRATEGIES = frozenset({CONTEXT_STRATEGY_DELEGATED, CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED})


@dataclass(frozen=True, slots=True)
class RouteMergePolicy:
    route_payload: dict[str, Any]
    resource_extraction: ResourceExtraction
    evidence_requirement: EvidenceRequirement
    workspace_scope: WorkspaceScopeResolution | None = None
    context_strategy: str = CONTEXT_STRATEGY_NONE
    needs_workspace_context: bool = False
    workspace_query: str = ""
    needs_url_context: bool = False
    url_query: str = ""
    urls: list[str] = field(default_factory=list)
    needs_delegated_perception: bool = False
    delegated_perception_query: str = ""
    delegated_perception_reason: str = ""
    suggested_capabilities: list[str] = field(default_factory=list)
    forced_by: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    reason: str = ""
    schema_version: str = ROUTE_MERGE_POLICY_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_strategy": self.context_strategy,
            "needs_workspace_context": self.needs_workspace_context,
            "workspace_query": self.workspace_query,
            "needs_url_context": self.needs_url_context,
            "url_query": self.url_query,
            "urls": list(self.urls),
            "needs_delegated_perception": self.needs_delegated_perception,
            "delegated_perception_query": self.delegated_perception_query,
            "delegated_perception_reason": self.delegated_perception_reason,
            "suggested_capabilities": list(self.suggested_capabilities),
            "forced_by": list(self.forced_by),
            "limitations": list(self.limitations),
            "reason": self.reason,
            "route": dict(self.route_payload),
            "resource_extraction": self.resource_extraction.to_payload(),
            "evidence_requirement": self.evidence_requirement.to_payload(),
            "workspace_scope": self.workspace_scope.to_payload() if self.workspace_scope is not None else {},
            "merged_route": self.to_route_payload(),
        }

    def to_route_payload(self) -> dict[str, Any]:
        """Return the semantic-route-shaped payload after deterministic merging."""

        payload = dict(self.route_payload)
        payload.update(
            {
                "context_strategy": self.context_strategy,
                "needs_workspace_context": self.needs_workspace_context,
                "workspace_query": self.workspace_query,
                "needs_url_context": self.needs_url_context,
                "url_query": self.url_query,
                "urls": list(self.urls),
                "needs_delegated_perception": self.needs_delegated_perception,
                "delegated_perception_query": self.delegated_perception_query,
                "delegated_perception_reason": self.delegated_perception_reason,
                "suggested_capabilities": list(self.suggested_capabilities),
                "route_merge_policy": {
                    "schema_version": self.schema_version,
                    "forced_by": list(self.forced_by),
                    "limitations": list(self.limitations),
                    "reason": self.reason,
                },
            }
        )
        return payload


def merge_route_context_policy(
    route: Any,
    *,
    user_input: str = "",
    resource_extraction: ResourceExtraction | None = None,
    evidence_requirement: EvidenceRequirement | None = None,
    workspace_scope: WorkspaceScopeResolution | None = None,
) -> RouteMergePolicy:
    """Merge semantic routing with deterministic resources and evidence requirements.

    The semantic router decides user intent. This merge layer enforces hard
    resource/evidence signals so explicit local paths, URL references, and repo
    evidence requirements cannot be dropped by a model routing miss.
    """

    route_payload = _route_payload(route)
    text = str(user_input or route_payload.get("text") or "").strip()
    resources = resource_extraction or extract_resources(text)
    evidence = evidence_requirement or detect_evidence_requirement(text, resource_extraction=resources)

    forced_by: list[str] = []
    limitations: list[str] = []

    route_strategy = _context_strategy(route_payload)
    needs_workspace = bool(route_payload.get("needs_workspace_context")) or route_strategy in _WORKSPACE_STRATEGIES
    needs_url = bool(route_payload.get("needs_url_context"))
    needs_delegated = bool(route_payload.get("needs_delegated_perception")) or route_strategy in _DELEGATED_STRATEGIES

    if route_strategy != CONTEXT_STRATEGY_NONE:
        forced_by.append(FORCED_BY_SEMANTIC_ROUTE)

    urls = _unique([*_strings(route_payload.get("urls")), *resources.urls])
    if urls:
        needs_url = True
        forced_by.append(FORCED_BY_EXPLICIT_URL)
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_URL, EVIDENCE_DOMAIN_MIXED}:
        needs_url = True
        forced_by.append(FORCED_BY_URL_EVIDENCE_REQUIREMENT)
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_REPO, EVIDENCE_DOMAIN_MIXED}:
        needs_workspace = True
        forced_by.append(FORCED_BY_REPO_EVIDENCE_REQUIREMENT)
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_EXTERNAL, EVIDENCE_DOMAIN_MIXED}:
        needs_delegated = True
        forced_by.append(FORCED_BY_EXTERNAL_EVIDENCE_REQUIREMENT)

    if (
        workspace_scope is not None
        and workspace_scope.status != WORKSPACE_SCOPE_NONE
        and (needs_workspace or evidence.evidence_domain in {EVIDENCE_DOMAIN_REPO, EVIDENCE_DOMAIN_MIXED})
    ):
        needs_workspace = True
        _merge_workspace_scope(
            workspace_scope,
            forced_by=forced_by,
            limitations=limitations,
        )

    workspace_query = _query(
        route_payload,
        "workspace_query",
        fallback=_workspace_query(text=text, resources=resources, evidence=evidence, workspace_scope=workspace_scope),
    )
    url_query = _query(route_payload, "url_query", fallback=text)
    delegated_query = _query(route_payload, "delegated_perception_query", fallback=text)
    delegated_reason = str(route_payload.get("delegated_perception_reason") or "").strip()
    if needs_delegated and not delegated_reason:
        delegated_reason = _delegated_reason(evidence=evidence, resources=resources)

    suggested_capabilities = _unique(_strings(route_payload.get("suggested_capabilities")))
    if needs_delegated and not suggested_capabilities:
        suggested_capabilities = _suggested_capabilities(evidence=evidence, resources=resources)

    strategy = _merged_strategy(
        route_strategy=route_strategy,
        needs_workspace=needs_workspace,
        needs_url=needs_url,
        needs_delegated=needs_delegated,
    )
    reason = _reason(
        route_strategy=route_strategy,
        evidence=evidence,
        forced_by=forced_by,
        limitations=limitations,
    )

    return RouteMergePolicy(
        route_payload=route_payload,
        resource_extraction=resources,
        evidence_requirement=evidence,
        workspace_scope=workspace_scope,
        context_strategy=strategy,
        needs_workspace_context=needs_workspace,
        workspace_query=workspace_query if needs_workspace else "",
        needs_url_context=needs_url,
        url_query=url_query if needs_url else "",
        urls=urls if needs_url else [],
        needs_delegated_perception=needs_delegated,
        delegated_perception_query=delegated_query if needs_delegated else "",
        delegated_perception_reason=delegated_reason if needs_delegated else "",
        suggested_capabilities=suggested_capabilities if needs_delegated else [],
        forced_by=_unique(forced_by),
        limitations=_unique(limitations),
        reason=reason,
    )


def _route_payload(route: Any) -> dict[str, Any]:
    if hasattr(route, "to_payload") and callable(route.to_payload):
        payload = route.to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return dict(route) if isinstance(route, Mapping) else {}


def _context_strategy(route_payload: Mapping[str, Any]) -> str:
    strategy = str(route_payload.get("context_strategy") or "").strip()
    if strategy in {
        CONTEXT_STRATEGY_NONE,
        CONTEXT_STRATEGY_LOCAL_WORKSPACE,
        CONTEXT_STRATEGY_URL,
        CONTEXT_STRATEGY_DELEGATED,
        CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED,
    }:
        return strategy
    if route_payload.get("needs_workspace_context") and route_payload.get("needs_delegated_perception"):
        return CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED
    if route_payload.get("needs_workspace_context"):
        return CONTEXT_STRATEGY_LOCAL_WORKSPACE
    if route_payload.get("needs_url_context"):
        return CONTEXT_STRATEGY_URL
    if route_payload.get("needs_delegated_perception"):
        return CONTEXT_STRATEGY_DELEGATED
    return CONTEXT_STRATEGY_NONE


def _merge_workspace_scope(
    workspace_scope: WorkspaceScopeResolution,
    *,
    forced_by: list[str],
    limitations: list[str],
) -> None:
    if workspace_scope.status == WORKSPACE_SCOPE_ALLOWED:
        forced_by.append(FORCED_BY_WORKSPACE_SCOPE_ALLOWED)
        return
    if workspace_scope.status == WORKSPACE_SCOPE_NEEDS_CONFIRMATION:
        forced_by.append(FORCED_BY_WORKSPACE_SCOPE_NEEDS_CONFIRMATION)
        limitations.append("workspace scope requires user confirmation before reading")
        return
    if workspace_scope.status == WORKSPACE_SCOPE_NEEDS_SELECTION:
        forced_by.append(FORCED_BY_WORKSPACE_SCOPE_NEEDS_SELECTION)
        limitations.append("multiple workspace scopes require user selection before reading")
        return
    if workspace_scope.status == WORKSPACE_SCOPE_BLOCKED:
        forced_by.append(FORCED_BY_WORKSPACE_SCOPE_BLOCKED)
        limitations.append(f"workspace scope blocked: {workspace_scope.reason}")


def _query(route_payload: Mapping[str, Any], key: str, *, fallback: str) -> str:
    return str(route_payload.get(key) or fallback or "").strip()


def _workspace_query(
    *,
    text: str,
    resources: ResourceExtraction,
    evidence: EvidenceRequirement,
    workspace_scope: WorkspaceScopeResolution | None,
) -> str:
    if evidence.reason:
        return evidence.reason
    if workspace_scope is not None and workspace_scope.scope_path:
        return f"inspect workspace scope {workspace_scope.scope_path}"
    if resources.local_paths or resources.relative_paths:
        return "inspect explicit workspace path references"
    if resources.file_refs:
        return "inspect referenced workspace files"
    return text


def _delegated_reason(*, evidence: EvidenceRequirement, resources: ResourceExtraction) -> str:
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_EXTERNAL, EVIDENCE_DOMAIN_MIXED}:
        return evidence.reason or "external evidence is required"
    if resources.external_research_hints:
        return "external research hints were detected"
    return "delegated read-only investigation was requested by semantic route"


def _suggested_capabilities(*, evidence: EvidenceRequirement, resources: ResourceExtraction) -> list[str]:
    capabilities: list[str] = []
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_EXTERNAL, EVIDENCE_DOMAIN_MIXED} or resources.external_research_hints:
        capabilities.append("web_research")
    if evidence.evidence_domain in {EVIDENCE_DOMAIN_REPO, EVIDENCE_DOMAIN_MIXED} or resources.has_repo_signal:
        capabilities.append("repo_inspection")
    return capabilities or ["web_research"]


def _merged_strategy(
    *,
    route_strategy: str,
    needs_workspace: bool,
    needs_url: bool,
    needs_delegated: bool,
) -> str:
    if needs_workspace and needs_delegated:
        return CONTEXT_STRATEGY_LOCAL_THEN_DELEGATED
    if needs_workspace:
        return CONTEXT_STRATEGY_LOCAL_WORKSPACE
    if needs_url:
        return CONTEXT_STRATEGY_URL
    if needs_delegated:
        return CONTEXT_STRATEGY_DELEGATED
    return route_strategy or CONTEXT_STRATEGY_NONE


def _reason(
    *,
    route_strategy: str,
    evidence: EvidenceRequirement,
    forced_by: list[str],
    limitations: list[str],
) -> str:
    parts: list[str] = []
    if route_strategy and route_strategy != CONTEXT_STRATEGY_NONE:
        parts.append(f"semantic route requested {route_strategy}")
    if evidence.requires_evidence:
        parts.append(f"deterministic evidence requirement: {evidence.evidence_domain}")
    if forced_by:
        parts.append("hard signals: " + ", ".join(_unique(forced_by)))
    if limitations:
        parts.append("limitations: " + "; ".join(_unique(limitations)))
    return "; ".join(parts) if parts else "no context perception required"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
