from __future__ import annotations

import re
from typing import Any

from spice.decision.general.candidates import GenericCandidate


REQUIRED_CAPABILITY_INFERENCE_VERSION = "spice.required_capability_inference.v1"

EXECUTION_ACTION_TYPES = frozenset({"intent.execute", "capability.use"})
NO_EXECUTION_CAPABILITY_ACTION_TYPES = frozenset(
    {
        "approval.request",
        "artifact.draft",
        "context.prepare",
        "item.ignore",
        "item.triage",
        "state.observe_more",
        "state.record",
        "task.split",
        "time.defer",
        "user.clarify",
    }
)

_BROWSER_KEYWORDS = (
    "browser",
    "web",
    "website",
    "url",
    "research",
    "search",
    "crawl",
    "浏览器",
    "网页",
    "网站",
    "搜索",
    "调研",
    "查资料",
)
_GITHUB_KEYWORDS = (
    "github",
    "pull request",
    "issue",
    "issues",
    "代码托管",
)
_GITHUB_TOKEN_RE = re.compile(r"\b(pr|prs)\b", re.IGNORECASE)
_EDIT_VERBS = (
    "add",
    "apply",
    "change",
    "create",
    "delete",
    "edit",
    "fix",
    "implement",
    "modify",
    "patch",
    "refactor",
    "remove",
    "update",
    "write",
    "添加",
    "创建",
    "删除",
    "实现",
    "修复",
    "写",
    "修改",
    "更改",
    "重构",
)
_CODE_OBJECTS = (
    "code",
    "codebase",
    "file",
    "files",
    "repo",
    "repository",
    "test",
    "tests",
    "代码",
    "代码库",
    "仓库",
    "测试",
    "文件",
)
_READ_VERBS = (
    "analyze",
    "check",
    "inspect",
    "look at",
    "read",
    "review",
    "scan",
    "summarize",
    "查看",
    "检查",
    "读取",
    "审查",
    "分析",
    "总结",
)


def annotate_required_capabilities(candidates: list[GenericCandidate]) -> list[GenericCandidate]:
    """Attach runtime-inferred required capabilities to candidates.

    This is intentionally conservative. LLMs may describe what should happen,
    but the runtime owns the stable capability label used by later policy,
    approval, and executor checks.
    """

    return [_annotate_candidate(candidate) for candidate in candidates]


def infer_required_capability(candidate: GenericCandidate) -> str:
    """Infer the single primary capability a candidate would need to execute."""

    existing = str(getattr(candidate, "required_capability", "") or "").strip()
    if existing:
        return existing
    if candidate.action_type in NO_EXECUTION_CAPABILITY_ACTION_TYPES:
        return ""
    if candidate.action_type not in EXECUTION_ACTION_TYPES:
        return ""

    text = _candidate_haystack(candidate)
    if _contains_any(text, _BROWSER_KEYWORDS):
        return "browser_or_external_tools"
    if _contains_any(text, _GITHUB_KEYWORDS) or _GITHUB_TOKEN_RE.search(text):
        return "github_work"
    if _looks_like_code_edit(text):
        return "code_edit"
    if _looks_like_repo_read(text):
        return "repo_read"
    return "general_execution"


def _annotate_candidate(candidate: GenericCandidate) -> GenericCandidate:
    existing = str(getattr(candidate, "required_capability", "") or "").strip()
    inferred = infer_required_capability(candidate)
    metadata = dict(candidate.metadata or {})
    inference = {
        "schema_version": REQUIRED_CAPABILITY_INFERENCE_VERSION,
        "generated_by": "spice.runtime.candidate_capabilities",
        "source": "existing_candidate_field" if existing else "runtime_inference",
        "required_capability": inferred,
        "reason": _inference_reason(candidate, inferred, existing=existing),
    }
    metadata["required_capability_inference"] = inference
    if inferred:
        metadata["required_capability"] = inferred
        candidate.required_capability = inferred
        boundary = getattr(candidate, "execution_boundary", None)
        if boundary is not None and not str(getattr(boundary, "required_capability", "") or "").strip():
            boundary.required_capability = inferred
    else:
        metadata.pop("required_capability", None)
        if not existing:
            candidate.required_capability = ""
    candidate.metadata = metadata
    return candidate


def _inference_reason(
    candidate: GenericCandidate,
    required_capability: str,
    *,
    existing: str,
) -> str:
    if existing:
        return "Candidate already declared a required capability; runtime preserved it."
    if candidate.action_type in NO_EXECUTION_CAPABILITY_ACTION_TYPES:
        return "Candidate is advisory, planning, or runtime guardrail; no execution capability required."
    if candidate.action_type not in EXECUTION_ACTION_TYPES:
        return "Candidate action type is not an executor handoff."
    if required_capability == "browser_or_external_tools":
        return "Candidate text references web, browser, or research work."
    if required_capability == "github_work":
        return "Candidate text references GitHub, PR, or issue work."
    if required_capability == "code_edit":
        return "Candidate text references code, tests, files, or change-oriented work."
    if required_capability == "repo_read":
        return "Candidate text references repository inspection or read-only code review."
    return "Execution handoff candidate without a more specific conservative match."


def _candidate_haystack(candidate: GenericCandidate) -> str:
    metadata = dict(candidate.metadata or {})
    execution_intent = getattr(candidate, "execution_intent", None)
    values: list[Any] = [
        candidate.action_type,
        candidate.intent,
        *list(getattr(candidate, "target_refs", []) or []),
        metadata.get("user_facing_title"),
        metadata.get("recommended_action"),
        metadata.get("executor_task"),
        metadata.get("expected_result"),
        metadata.get("why_now"),
        getattr(execution_intent, "handoff_task", ""),
        getattr(execution_intent, "reason", ""),
    ]
    return _flatten_text(values).lower()


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _looks_like_code_edit(text: str) -> bool:
    has_edit_verb = _contains_any(text, _EDIT_VERBS)
    has_code_object = _contains_any(text, _CODE_OBJECTS)
    if has_edit_verb and has_code_object:
        return True
    # "fix failing test" and Chinese equivalents should still resolve to edit work.
    return has_edit_verb and not _looks_like_repo_read(text)


def _looks_like_repo_read(text: str) -> bool:
    return _contains_any(text, _READ_VERBS) and _contains_any(text, _CODE_OBJECTS)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)
