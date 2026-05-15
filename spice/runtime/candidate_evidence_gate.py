from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping

from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    GenericExecutionIntent,
    RiskProfile,
)
from spice.runtime.evidence_qualification import has_source_backed_evidence
from spice.runtime.evidence_requirement import (
    EVIDENCE_DOMAIN_MIXED,
    EVIDENCE_DOMAIN_REPO,
    EvidenceRequirement,
    evidence_requirement_from_payload,
)


CANDIDATE_EVIDENCE_GATE_SCHEMA_VERSION = "spice.candidate_evidence_gate.v1"
HARD_REPO_EVIDENCE_CONSTRAINT_ID = "hard_repo_evidence_required"

_DOC_OR_GUESS_PATTERNS = (
    "基于文档快速判断",
    "仅基于文档",
    "文档层面",
    "设计意图层面",
    "不执行完整代码读取",
    "不触发完整代码感知",
    "不读取仓库",
    "不读代码",
    "跳过 repo 分析",
    "跳过代码分析",
    "先跳过 repo",
    "节省时间和预算",
    "基于常识",
    "common sense",
    "design intent",
    "document-only",
    "based on docs",
    "based on documentation",
    "without reading code",
    "without repo analysis",
    "skip repo",
    "skip code analysis",
    "save time and budget",
    "quick judgment",
)

_EVIDENCE_GATHERING_PATTERNS = (
    "读取关键实现文件",
    "先读取",
    "需要先读取",
    "重新读取",
    "补充代码证据",
    "完整代码感知",
    "执行完整代码感知",
    "workspace perception",
    "files_read",
    "snippets",
    "facts",
    "source-backed",
    "gather evidence",
    "read source",
    "read key implementation",
    "retry perception",
    "insufficient evidence",
    "证据不足",
    "无法基于实际代码判断",
)


@dataclass(slots=True)
class CandidateEvidenceGateResult:
    candidates: list[GenericCandidate]
    applied: bool = False
    hard_repo_evidence_required: bool = False
    source_backed_workspace_evidence: bool = False
    eligible_candidate_ids: list[str] = field(default_factory=list)
    ineligible_candidate_ids: list[str] = field(default_factory=list)
    added_candidate_ids: list[str] = field(default_factory=list)
    reason: str = ""
    limitations: list[str] = field(default_factory=list)
    schema_version: str = CANDIDATE_EVIDENCE_GATE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "applied": self.applied,
            "hard_repo_evidence_required": self.hard_repo_evidence_required,
            "source_backed_workspace_evidence": self.source_backed_workspace_evidence,
            "eligible_candidate_ids": list(self.eligible_candidate_ids),
            "ineligible_candidate_ids": list(self.ineligible_candidate_ids),
            "added_candidate_ids": list(self.added_candidate_ids),
            "reason": self.reason,
            "limitations": list(self.limitations),
        }


def apply_candidate_evidence_gate(
    candidates: list[GenericCandidate],
    *,
    intent_text: str,
    evidence_requirement: EvidenceRequirement | Mapping[str, Any] | None,
    workspace_context: Mapping[str, Any] | None = None,
    evidence_context: Mapping[str, Any] | None = None,
) -> CandidateEvidenceGateResult:
    """Apply hard evidence constraints to candidate eligibility.

    This is deliberately a runtime gate, not a prompt hint. If the user asks
    for repo/code evidence, selector scoring cannot choose a cheaper
    document-only or common-sense candidate over the evidence requirement.
    """

    requirement = _requirement(evidence_requirement)
    hard_repo_required = _hard_repo_evidence_required(requirement)
    if not hard_repo_required:
        return CandidateEvidenceGateResult(
            candidates=list(candidates),
            applied=False,
            hard_repo_evidence_required=False,
            source_backed_workspace_evidence=False,
            reason="repo evidence is not a hard requirement for this turn",
        )

    workspace_evidence = _has_workspace_evidence(
        workspace_context=workspace_context,
        evidence_context=evidence_context,
    )
    result_candidates = list(candidates)
    eligible: list[str] = []
    ineligible: list[str] = []
    for candidate in result_candidates:
        if _candidate_gathers_evidence(candidate):
            _force_read_only_evidence_candidate(candidate)
        if _candidate_contradicts_repo_evidence_requirement(candidate):
            _block_candidate(
                candidate,
                reason=(
                    "User required source-backed repo/code evidence; this candidate proposes "
                    "a document-only, common-sense, or no-code shortcut."
                ),
                gate_reason="candidate_contradicts_required_repo_evidence",
                source_backed_workspace_evidence=workspace_evidence,
            )
            ineligible.append(candidate.candidate_id)
            continue
        if not workspace_evidence and _is_final_decision_candidate(candidate) and not _candidate_gathers_evidence(candidate):
            _block_candidate(
                candidate,
                reason=(
                    "User required source-backed repo/code evidence, but no files/facts/snippets "
                    "are available to support this candidate."
                ),
                gate_reason="missing_source_backed_repo_evidence",
                source_backed_workspace_evidence=False,
            )
            ineligible.append(candidate.candidate_id)
            continue
        eligible.append(candidate.candidate_id)

    added: list[str] = []
    if not _has_available_decision_candidate(result_candidates):
        gap = _evidence_gap_candidate(
            intent_text=intent_text,
            source_backed_workspace_evidence=workspace_evidence,
        )
        result_candidates.append(gap)
        eligible.append(gap.candidate_id)
        added.append(gap.candidate_id)

    limitations: list[str] = []
    if not workspace_evidence:
        limitations.append(
            "Hard repo evidence was requested, but the current workspace context is not source-backed."
        )

    return CandidateEvidenceGateResult(
        candidates=result_candidates,
        applied=True,
        hard_repo_evidence_required=True,
        source_backed_workspace_evidence=workspace_evidence,
        eligible_candidate_ids=_unique(eligible),
        ineligible_candidate_ids=_unique(ineligible),
        added_candidate_ids=added,
        reason=(
            "hard repo evidence requirement enforced before candidate selection"
            if workspace_evidence
            else "hard repo evidence requirement is missing source-backed workspace evidence"
        ),
        limitations=limitations,
    )


def _requirement(
    value: EvidenceRequirement | Mapping[str, Any] | None,
) -> EvidenceRequirement:
    if isinstance(value, EvidenceRequirement):
        return value
    if isinstance(value, Mapping):
        return evidence_requirement_from_payload(value)
    return EvidenceRequirement(requires_evidence=False)


def _hard_repo_evidence_required(requirement: EvidenceRequirement) -> bool:
    return bool(requirement.requires_evidence) and requirement.evidence_domain in {
        EVIDENCE_DOMAIN_REPO,
        EVIDENCE_DOMAIN_MIXED,
    }


def _has_workspace_evidence(
    *,
    workspace_context: Mapping[str, Any] | None,
    evidence_context: Mapping[str, Any] | None,
) -> bool:
    if has_source_backed_evidence(workspace_context):
        return True
    if not isinstance(evidence_context, Mapping):
        return False
    workspace = evidence_context.get("workspace")
    if has_source_backed_evidence(workspace if isinstance(workspace, Mapping) else None):
        return True
    return has_source_backed_evidence(evidence_context)


def _candidate_contradicts_repo_evidence_requirement(candidate: GenericCandidate) -> bool:
    return _contains_any(_candidate_text(candidate), _DOC_OR_GUESS_PATTERNS)


def _candidate_gathers_evidence(candidate: GenericCandidate) -> bool:
    if candidate.action_type in {"context.prepare", "state.observe_more", "user.clarify"}:
        return True
    return _contains_any(_candidate_text(candidate), _EVIDENCE_GATHERING_PATTERNS)


def _is_final_decision_candidate(candidate: GenericCandidate) -> bool:
    metadata = candidate.metadata or {}
    return (
        candidate.candidate_kind == "decision"
        or metadata.get("candidate_kind") == "decision"
        or metadata.get("candidate_source") in {"llm_generator", "explicit_options"}
        or metadata.get("source") == "explicit_options"
    )


def _has_available_decision_candidate(candidates: list[GenericCandidate]) -> bool:
    return any(
        _is_final_decision_candidate(candidate)
        and candidate.availability_status != "blocked"
        for candidate in candidates
    )


def _block_candidate(
    candidate: GenericCandidate,
    *,
    reason: str,
    gate_reason: str,
    source_backed_workspace_evidence: bool,
) -> None:
    candidate.availability_status = "blocked"
    if reason not in candidate.why_blocked:
        candidate.why_blocked.append(reason)
    candidate.why_available = [
        item for item in candidate.why_available if str(item or "").strip() != reason
    ]
    candidate.constraints_triggered.append(
        {
            "constraint_id": HARD_REPO_EVIDENCE_CONSTRAINT_ID,
            "severity": "veto",
            "reason": reason,
        }
    )
    metadata = dict(candidate.metadata or {})
    metadata["candidate_evidence_gate"] = {
        "schema_version": CANDIDATE_EVIDENCE_GATE_SCHEMA_VERSION,
        "eligible": False,
        "reason": gate_reason,
        "source_backed_workspace_evidence": source_backed_workspace_evidence,
    }
    candidate.metadata = metadata


def _force_read_only_evidence_candidate(candidate: GenericCandidate) -> None:
    if candidate.action_type in {"intent.execute", "capability.use", "approval.request"}:
        candidate.action_type = "context.prepare"
    candidate.required_capability = ""
    candidate.requires_confirmation = False
    candidate.side_effect_class = "read_only"
    candidate.availability_status = "available"
    candidate.execution_intent = GenericExecutionIntent(
        intent_class="advisory",
        requested=False,
        handoff_task="",
        reason="Hard repo evidence gathering is read-only perception, not executor handoff.",
        required_permission_hint="read_only",
        side_effect_class="read_only",
    )
    candidate.execution_boundary = ExecutionBoundary(
        mode="none",
        target="",
        protocol="",
        required_capability="",
        requires_confirmation=False,
        side_effect_class="read_only",
    )
    metadata = dict(candidate.metadata or {})
    for key in (
        "executor_task",
        "handoff_task",
        "execution_objective",
        "execution_schema",
        "execution_boundary",
        "execution_affordance",
        "approval",
        "permission",
        "required_permission",
        "required_permission_hint",
        "required_capability",
        "required_capability_inference",
        "skill_resolution",
        "resolved_skill",
    ):
        metadata.pop(key, None)
    metadata["executor_task"] = ""
    metadata["read_only_intent_boundary_applied"] = True
    metadata["candidate_evidence_gate"] = {
        **dict(metadata.get("candidate_evidence_gate") or {}),
        "schema_version": CANDIDATE_EVIDENCE_GATE_SCHEMA_VERSION,
        "read_only_boundary_applied": True,
        "read_only_boundary_reason": "repo evidence gathering must remain read-only",
    }
    candidate.metadata = metadata


def _evidence_gap_candidate(
    *,
    intent_text: str,
    source_backed_workspace_evidence: bool,
) -> GenericCandidate:
    digest = sha256(intent_text.encode("utf-8")).hexdigest()[:12]
    reason = (
        "Workspace evidence exists, but all final decision candidates contradicted the hard evidence requirement."
        if source_backed_workspace_evidence
        else "Workspace perception did not produce source-backed files, facts, snippets, or sources."
    )
    return GenericCandidate(
        candidate_id=f"candidate.runtime.evidence_gap.{digest}",
        action_type="context.prepare",
        intent="Read source-backed workspace evidence before deciding.",
        candidate_kind="decision",
        target_refs=[],
        required_capability="",
        execution_intent=GenericExecutionIntent(
            intent_class="advisory",
            requested=False,
            handoff_task="",
            reason="Evidence collection is required before a source-backed decision can be made.",
            required_permission_hint="read_only",
            side_effect_class="read_only",
        ),
        estimated_cost=EstimatedCost(time_minutes=5, attention="medium"),
        risk_profile=RiskProfile(
            level="low",
            summary="Low risk; this only asks Spice to gather missing read-only evidence.",
            uncertainty="medium",
        ),
        reversibility="high",
        requires_confirmation=False,
        expected_state_delta=ExpectedStateDelta(
            summary="A source-backed workspace perception should be produced before deciding."
        ),
        execution_boundary=ExecutionBoundary(
            mode="none",
            target="",
            protocol="",
            required_capability="",
            requires_confirmation=False,
            side_effect_class="read_only",
        ),
        constraints_triggered=[],
        why_available=[
            "Hard repo evidence is required, and current candidates cannot complete that claim safely.",
            reason,
        ],
        why_blocked=[],
        side_effect_class="read_only",
        availability_status="available",
        metadata={
            "source": "runtime_candidate_evidence_gate",
            "candidate_source": "runtime_candidate_evidence_gate",
            "candidate_kind": "decision",
            "user_facing_title": "Read source-backed repo evidence before deciding",
            "recommendation": (
                "Retry or deepen workspace perception until files, facts, snippets, or source refs "
                "exist, then make the requested code-based prioritization."
            ),
            "expected_result": "Spice either gets source-backed repo evidence or clearly reports a partial/blocked result.",
            "risk_level": "low",
            "candidate_evidence_gate": {
                "schema_version": CANDIDATE_EVIDENCE_GATE_SCHEMA_VERSION,
                "eligible": True,
                "reason": "evidence_gap_fallback",
                "source_backed_workspace_evidence": source_backed_workspace_evidence,
            },
        },
    )


def _candidate_text(candidate: GenericCandidate) -> str:
    metadata = candidate.metadata or {}
    chunks: list[str] = [
        candidate.intent,
        candidate.action_type,
        candidate.availability_status,
        " ".join(candidate.why_available),
        " ".join(candidate.why_blocked),
        str(metadata.get("user_facing_title") or ""),
        str(metadata.get("recommendation") or metadata.get("recommended_action") or ""),
        str(metadata.get("expected_result") or metadata.get("expected_outcome") or ""),
        str(metadata.get("why_now") or ""),
        str(metadata.get("downside") or ""),
        str(metadata.get("success_signal") or ""),
        str(metadata.get("selection_rationale") or ""),
    ]
    return "\n".join(chunk for chunk in chunks if str(chunk or "").strip()).lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern.lower() in text for pattern in patterns)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
