from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from spice.runtime.resource_extractor import ResourceExtraction, extract_resources


EVIDENCE_REQUIREMENT_SCHEMA_VERSION = "spice.evidence_requirement.v1"

EVIDENCE_DOMAIN_REPO = "repo"
EVIDENCE_DOMAIN_URL = "url"
EVIDENCE_DOMAIN_EXTERNAL = "external"
EVIDENCE_DOMAIN_MIXED = "mixed"
EVIDENCE_DOMAIN_NONE = "none"

ANSWER_MODE_BRIEF = "brief"
ANSWER_MODE_NORMAL = "normal"
ANSWER_MODE_DETAILED = "detailed"
ANSWER_MODE_REPORT = "report"


_REPO_EVIDENCE_PHRASES = (
    "基于当前实现",
    "基于实际代码",
    "基于当前代码",
    "读取本地",
    "读一下本地",
    "看一下当前实现",
    "看下当前实现",
    "看一下我们 repo",
    "看下我们 repo",
    "看一下 repo",
    "看下 repo",
    "repo 现在",
    "repo 做到哪",
    "代码现在做到哪",
    "代码里",
    "源码里",
    "当前代码库",
    "当前 repo",
    "current implementation",
    "actual implementation",
    "current code",
    "current codebase",
    "current repo",
    "read the repo",
    "inspect the repo",
    "based on the repo",
    "based on the code",
    "based on current code",
)

_URL_EVIDENCE_PHRASES = (
    "这个链接",
    "这些链接",
    "这个网页",
    "这些网页",
    "read this link",
    "based on this link",
    "from this url",
    "this webpage",
)

_EXTERNAL_EVIDENCE_PHRASES = (
    "联网查",
    "网上查",
    "外部搜索",
    "深度搜索",
    "查一下最新",
    "搜一下最新",
    "对比 hermes",
    "对比 openclaw",
    "web research",
    "web search",
    "external research",
    "deep research",
    "latest docs",
    "compare hermes",
    "compare openclaw",
)

_DETAILED_MODE_PHRASES = (
    "详细",
    "具体",
    "完整",
    "解释一下",
    "为什么",
    "怎么做",
    "plan",
    "steps",
    "detail",
    "detailed",
    "explain",
    "why",
)

_REPORT_MODE_PHRASES = (
    "报告",
    "现状",
    "证据",
    "读到了哪些",
    "当前实现",
    "实际代码",
    "当前代码",
    "基于实际代码",
    "基于当前实现",
    "analysis report",
    "evidence",
    "sources",
    "what did you read",
    "implementation status",
)

_BRIEF_MODE_PHRASES = (
    "简短",
    "简单说",
    "一句话",
    "brief",
    "short",
    "quick",
)


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    requires_evidence: bool
    evidence_domain: str = EVIDENCE_DOMAIN_NONE
    answer_mode: str = ANSWER_MODE_NORMAL
    reason: str = ""
    resource_extraction: ResourceExtraction | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_REQUIREMENT_SCHEMA_VERSION,
            "requires_evidence": self.requires_evidence,
            "evidence_domain": self.evidence_domain,
            "answer_mode": self.answer_mode,
            "reason": self.reason,
            "resource_extraction": (
                self.resource_extraction.to_payload() if self.resource_extraction is not None else {}
            ),
        }


def detect_evidence_requirement(
    text: str,
    *,
    resource_extraction: ResourceExtraction | None = None,
) -> EvidenceRequirement:
    resources = resource_extraction or extract_resources(text)
    source = str(text or "")
    lowered = source.lower()

    repo_required = _has_repo_requirement(lowered, resources)
    url_required = _has_url_requirement(lowered, resources)
    external_required = _has_external_requirement(lowered, resources)
    domain = _evidence_domain(
        repo_required=repo_required,
        url_required=url_required,
        external_required=external_required,
    )
    requires = domain != EVIDENCE_DOMAIN_NONE
    return EvidenceRequirement(
        requires_evidence=requires,
        evidence_domain=domain,
        answer_mode=_answer_mode(lowered, domain),
        reason=_reason(
            repo_required=repo_required,
            url_required=url_required,
            external_required=external_required,
            resources=resources,
        ),
        resource_extraction=resources,
    )


def strengthen_evidence_requirement(
    base: EvidenceRequirement,
    *,
    requires_evidence: bool | None = None,
    evidence_domain: str | None = None,
    answer_mode: str | None = None,
    reason: str = "",
) -> EvidenceRequirement:
    """Strengthen an evidence requirement without allowing downgrade.

    LLM planner/router output may add requirements or request a deeper answer,
    but it must not erase deterministic hard evidence requirements.
    """

    final_requires = base.requires_evidence or bool(requires_evidence)
    final_domain = _stronger_domain(base.evidence_domain, evidence_domain or EVIDENCE_DOMAIN_NONE)
    final_mode = _stronger_answer_mode(base.answer_mode, answer_mode or ANSWER_MODE_NORMAL)
    final_reason = "; ".join(part for part in (base.reason, reason.strip()) if part)
    return EvidenceRequirement(
        requires_evidence=final_requires,
        evidence_domain=final_domain if final_requires else EVIDENCE_DOMAIN_NONE,
        answer_mode=final_mode,
        reason=final_reason,
        resource_extraction=base.resource_extraction,
    )


def evidence_requirement_from_payload(payload: Mapping[str, Any]) -> EvidenceRequirement:
    return EvidenceRequirement(
        requires_evidence=bool(payload.get("requires_evidence")),
        evidence_domain=_normalize_domain(str(payload.get("evidence_domain") or EVIDENCE_DOMAIN_NONE)),
        answer_mode=_normalize_answer_mode(str(payload.get("answer_mode") or ANSWER_MODE_NORMAL)),
        reason=str(payload.get("reason") or ""),
        resource_extraction=None,
    )


def _has_repo_requirement(lowered: str, resources: ResourceExtraction) -> bool:
    if (resources.local_paths or resources.relative_paths) and _contains_any(
        lowered,
        (
            "read",
            "inspect",
            "based on",
            "current implementation",
            "actual implementation",
            "current code",
            "看",
            "读",
            "读取",
            "基于",
            "当前实现",
            "实际代码",
            "当前代码",
        ),
    ):
        return True
    if resources.file_refs and _contains_any(lowered, ("read", "inspect", "看", "读", "基于", "based on")):
        return True
    if _contains_any(lowered, _REPO_EVIDENCE_PHRASES):
        return True
    return False


def _has_url_requirement(lowered: str, resources: ResourceExtraction) -> bool:
    return bool(resources.urls) or _contains_any(lowered, _URL_EVIDENCE_PHRASES)


def _has_external_requirement(lowered: str, resources: ResourceExtraction) -> bool:
    return bool(resources.external_research_hints) or _contains_any(lowered, _EXTERNAL_EVIDENCE_PHRASES)


def _evidence_domain(*, repo_required: bool, url_required: bool, external_required: bool) -> str:
    domains = [
        EVIDENCE_DOMAIN_REPO if repo_required else "",
        EVIDENCE_DOMAIN_URL if url_required else "",
        EVIDENCE_DOMAIN_EXTERNAL if external_required else "",
    ]
    active = [domain for domain in domains if domain]
    if not active:
        return EVIDENCE_DOMAIN_NONE
    if len(active) == 1:
        return active[0]
    return EVIDENCE_DOMAIN_MIXED


def _answer_mode(lowered: str, domain: str) -> str:
    if _contains_any(lowered, _BRIEF_MODE_PHRASES):
        return ANSWER_MODE_BRIEF
    if domain in {EVIDENCE_DOMAIN_MIXED, EVIDENCE_DOMAIN_EXTERNAL}:
        return ANSWER_MODE_REPORT
    if _contains_any(lowered, _REPORT_MODE_PHRASES):
        return ANSWER_MODE_REPORT
    if _contains_any(lowered, _DETAILED_MODE_PHRASES):
        return ANSWER_MODE_DETAILED
    if domain in {EVIDENCE_DOMAIN_REPO, EVIDENCE_DOMAIN_URL}:
        return ANSWER_MODE_DETAILED
    return ANSWER_MODE_NORMAL


def _reason(
    *,
    repo_required: bool,
    url_required: bool,
    external_required: bool,
    resources: ResourceExtraction,
) -> str:
    reasons: list[str] = []
    if repo_required:
        if resources.local_paths:
            reasons.append("user referenced a local path")
        elif resources.relative_paths or resources.file_refs:
            reasons.append("user referenced repo files")
        else:
            reasons.append("user asked for repo or current implementation evidence")
    if url_required:
        reasons.append("user referenced URL evidence")
    if external_required:
        reasons.append("user asked for external or latest research evidence")
    return "; ".join(reasons)


_DOMAIN_RANK = {
    EVIDENCE_DOMAIN_NONE: 0,
    EVIDENCE_DOMAIN_REPO: 1,
    EVIDENCE_DOMAIN_URL: 1,
    EVIDENCE_DOMAIN_EXTERNAL: 1,
    EVIDENCE_DOMAIN_MIXED: 2,
}

_ANSWER_MODE_RANK = {
    ANSWER_MODE_BRIEF: 0,
    ANSWER_MODE_NORMAL: 1,
    ANSWER_MODE_DETAILED: 2,
    ANSWER_MODE_REPORT: 3,
}


def _stronger_domain(current: str, proposed: str) -> str:
    normalized_current = _normalize_domain(current)
    normalized_proposed = _normalize_domain(proposed)
    if normalized_current == EVIDENCE_DOMAIN_NONE:
        return normalized_proposed
    if normalized_proposed == EVIDENCE_DOMAIN_NONE:
        return normalized_current
    if normalized_current == normalized_proposed:
        return normalized_current
    if EVIDENCE_DOMAIN_MIXED in {normalized_current, normalized_proposed}:
        return EVIDENCE_DOMAIN_MIXED
    return EVIDENCE_DOMAIN_MIXED


def _stronger_answer_mode(current: str, proposed: str) -> str:
    normalized_current = _normalize_answer_mode(current)
    normalized_proposed = _normalize_answer_mode(proposed)
    if _ANSWER_MODE_RANK[normalized_proposed] > _ANSWER_MODE_RANK[normalized_current]:
        return normalized_proposed
    return normalized_current


def _normalize_domain(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in _DOMAIN_RANK:
        return normalized
    return EVIDENCE_DOMAIN_NONE


def _normalize_answer_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in _ANSWER_MODE_RANK:
        return normalized
    return ANSWER_MODE_NORMAL


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)
