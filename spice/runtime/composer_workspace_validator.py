from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


WORKSPACE_COMPOSER_CONSTRAINTS = (
    "Only reference workspace/repo facts that appear in recent_context.workspace_context.",
    "Do not claim you inspected, read, or checked a file unless that file appears in workspace_context.files_read, facts, or snippets.",
    "Do not invent files, functions, classes, commands, or implementation status not present in workspace_context facts/snippets.",
    "Only say code is implemented, wired, or supported when that exact implementation status is supported by workspace facts/snippets.",
    "Workspace perception depth controls how much Spice reads; response_depth controls how much you say.",
    "Do not expand the answer just because workspace perception read many files; follow response_depth.max_chars.",
    "Implementation judgments must be source-backed by workspace facts/snippets, not only files_read, repo_map, summary, or cache metadata.",
    "If workspace_context is absent, use 'from the current decision context' instead of claiming repo/code inspection.",
    "If workspace_context.exploration_status is partial or budget_exhausted, explicitly mention the limitation and do not claim the repo/code evidence is complete.",
    "Only reference external URL facts that appear in recent_context.url_context.",
    "Do not claim you read a link, PR, issue, or docs page unless it appears in url_context.",
    "Only say a linked page proves something when that claim is supported by url_context facts/snippets.",
    "Only reference delegated investigation findings that appear in recent_context.delegated_perception_context.",
    "When using delegated findings or source ids, say the executor reported them; do not state them as Spice-direct evidence.",
    "Do not cite delegated source ids or finding ids that are absent from delegated_perception_context.",
    "Do not claim Hermes/Codex/Claude reported anything unless delegated_perception_context exists.",
    "Do not present delegated findings as Spice's final decision or as completed execution.",
    "Do not claim delegated sources were cross-checked unless their verification_status is cross_checked.",
)

_FILE_EXTENSIONS = (
    "py",
    "md",
    "json",
    "toml",
    "yaml",
    "yml",
    "txt",
    "ts",
    "tsx",
    "js",
    "jsx",
    "rs",
    "go",
    "java",
    "css",
    "html",
    "sh",
    "sql",
)

_PATH_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9_.-])(?:\.?/)?[A-Za-z0-9_./-]+\.({'|'.join(_FILE_EXTENSIONS)})(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FUNCTION_CLAIM_RE = re.compile(
    r"\b(?:function|method|class|module|函数|方法|类|模块)\s+`?([A-Za-z_][A-Za-z0-9_]{2,})`?",
    re.IGNORECASE,
)
_BACKTICK_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,}$")
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)

_INSPECTION_CLAIMS = (
    "i checked the repo",
    "i checked the code",
    "i checked the file",
    "i inspected the repo",
    "i inspected the code",
    "i inspected the file",
    "i read the repo",
    "i read the code",
    "i read the file",
    "after checking the repo",
    "after inspecting the code",
    "from the repo",
    "from the codebase",
    "in the code",
    "我看了代码",
    "我看了 repo",
    "我检查了代码",
    "我检查了文件",
    "我读了代码",
    "代码里",
    "仓库里",
    "repo 里",
    "从代码看",
    "从 repo 看",
)

_IMPLEMENTATION_CLAIMS = (
    "already implemented",
    "already implements",
    "implements",
    "is implemented",
    "has implemented",
    "already wired",
    "is wired",
    "already supports",
    "currently supports",
    "the code supports",
    "the repo supports",
    "repo already",
    "code already",
    "代码已经实现",
    "已经实现",
    "已实现",
    "代码里已经",
    "仓库里已经",
    "已经接入",
    "已接入",
    "代码支持",
    "仓库支持",
)

_URL_INSPECTION_CLAIMS = (
    "i read the link",
    "i opened the link",
    "i checked the link",
    "i read the url",
    "i opened the url",
    "i checked the url",
    "i read the pr",
    "i checked the pr",
    "i read the issue",
    "i checked the issue",
    "i read the docs",
    "i checked the docs",
    "from the linked pr",
    "from the linked issue",
    "from the linked docs",
    "according to the link",
    "according to the pr",
    "我看了链接",
    "我读了链接",
    "我检查了链接",
    "我看了 url",
    "我读了 url",
    "我检查了 url",
    "我打开了 url",
    "我看了这个 pr",
    "我读了这个 pr",
    "我检查了这个 pr",
    "我看了 issue",
    "我读了 issue",
    "我看了文档",
    "我读了文档",
    "根据链接",
    "根据这个 pr",
    "根据文档",
)

_IMPLEMENTATION_MARKERS = (
    "implement",
    "implemented",
    "support",
    "supports",
    "wired",
    "receives",
    "uses",
    "calls",
    "defines",
    "实现",
    "支持",
    "接入",
    "调用",
    "定义",
)

_SOURCE_ASSERTION_CLAIMS = (
    "evidence shows",
    "the evidence shows",
    "current evidence shows",
    "the current evidence shows",
    "the repo shows",
    "the code shows",
    "the file shows",
    "the workspace shows",
    "the facts show",
    "the snippet shows",
    "the source shows",
    "代码显示",
    "仓库显示",
    "文件显示",
    "证据显示",
    "当前证据显示",
    "根据证据",
    "从事实看",
)

_COMPLETE_WORKSPACE_CLAIMS = (
    "fully inspected",
    "completely inspected",
    "complete repo inspection",
    "complete codebase inspection",
    "read the whole repo",
    "read the entire repo",
    "read all files",
    "checked all files",
    "covered the full repo",
    "exhaustive",
    "no remaining gaps",
    "no gaps",
    "fully confirmed",
    "completely confirmed",
    "fully verified",
    "completely verified",
    "high confidence that the repo is complete",
    "完整检查",
    "完整读完",
    "读完整个 repo",
    "读完了整个 repo",
    "全部读完",
    "所有文件都看了",
    "没有剩余缺口",
    "没有缺口",
    "证据完整",
    "已经完整确认",
    "完整确认",
    "完全确认",
    "已经完全确认",
    "完整验证",
    "完全验证",
    "证据已经充分",
)

_WORKSPACE_LIMITATION_MARKERS = (
    "partial",
    "limited",
    "limitation",
    "limitations",
    "remaining gap",
    "remaining gaps",
    "budget",
    "budget exhausted",
    "not exhaustive",
    "not complete",
    "could not read",
    "did not read",
    "only read",
    "部分",
    "限制",
    "局限",
    "缺口",
    "预算",
    "未完整",
    "不完整",
    "没有读完",
    "还没读",
    "只读了",
)

_DELEGATED_REPORT_MARKERS = (
    "delegated investigation",
    "read-only investigation",
    "executor reported",
    "reported by the executor",
    "hermes reported",
    "codex reported",
    "claude reported",
    "调查报告",
    "只读调查",
    "外部调查",
    "hermes 报告",
    "hermes 的只读调查",
    "执行器报告",
)

_DELEGATED_DIRECT_INSPECTION_CLAIMS = (
    "spice inspected",
    "spice checked",
    "spice verified",
    "spice confirmed",
    "i inspected these sources",
    "i checked these sources",
    "i inspected these webpages",
    "i checked these webpages",
    "i opened these webpages",
    "i read these webpages",
    "我检查了这些网页",
    "我看了这些网页",
    "我读了这些网页",
    "我打开了这些网页",
    "我检查了这些来源",
    "我确认了这些来源",
    "spice 已经确认",
    "spice 确认",
    "spice 已验证",
    "spice 已经验证",
)

_DELEGATED_FINAL_DECISION_CLAIMS = (
    "final decision",
    "spice decided",
    "spice has decided",
    "therefore spice decided",
    "this proves the decision",
    "this settles the decision",
    "最终决策",
    "最终决定",
    "spice 决定",
    "spice 已决定",
    "已经决定",
)

_DELEGATED_EXECUTION_COMPLETED_CLAIMS = (
    "execution completed",
    "executed successfully",
    "has been executed",
    "already executed",
    "task completed",
    "执行完成",
    "已经执行完成",
    "已经执行",
    "已执行",
    "执行成功",
    "任务完成",
)

_DELEGATED_CROSS_CHECK_CLAIMS = (
    "cross-checked",
    "cross checked",
    "crosscheck",
    "verified by spice",
    "verified_by_spice",
    "spice verified",
    "confirmed by spice",
    "verification_status=cross_checked",
    "verification_status: cross_checked",
    "交叉验证",
    "已交叉验证",
    "已经交叉验证",
    "spice 已验证",
    "spice 已经验证",
)

_DELEGATED_CROSS_CHECK_NEGATIONS = (
    "not cross-checked",
    "not cross checked",
    "not verified",
    "not confirmed",
    "was not cross-checked",
    "were not cross-checked",
    "没有交叉验证",
    "未交叉验证",
    "没有被交叉验证",
    "未被交叉验证",
    "没有验证",
    "未验证",
)

_DELEGATED_REF_RE = re.compile(r"\b(?:source|finding)\.[A-Za-z0-9_-]+\b", re.IGNORECASE)

_WORKSPACE_IMPLEMENTATION_SOURCE_TYPES = frozenset(
    {
        "workspace_fact",
        "workspace_snippet",
    }
)


@dataclass(frozen=True)
class SourceEvidenceItem:
    source_type: str
    source_id: str = ""
    text: str = ""
    path: str = ""
    url: str = ""
    verification_status: str = ""
    symbols: frozenset[str] = field(default_factory=frozenset)
    tokens: frozenset[str] = field(default_factory=frozenset)
    implementation_supported: bool = False


@dataclass(frozen=True)
class CitationEvidenceIndex:
    workspace_paths: frozenset[str] = field(default_factory=frozenset)
    workspace_symbols: frozenset[str] = field(default_factory=frozenset)
    urls: frozenset[str] = field(default_factory=frozenset)
    workspace_items: tuple[SourceEvidenceItem, ...] = ()
    url_items: tuple[SourceEvidenceItem, ...] = ()
    delegated_items: tuple[SourceEvidenceItem, ...] = ()


def validate_workspace_claims(
    text: str,
    facts: Mapping[str, Any],
    *,
    composer_kind: str = "composer",
) -> None:
    evidence_index = build_citation_evidence_index(facts)
    validate_delegated_perception_claims(
        text,
        facts,
        evidence_index=evidence_index,
        composer_kind=composer_kind,
    )
    validate_url_claims(text, facts, composer_kind=composer_kind)
    workspace = workspace_context_from_facts(facts)
    has_workspace_evidence = _has_workspace_evidence(workspace)
    lower = text.lower()

    mentioned_paths = _mentioned_paths(text)
    if mentioned_paths:
        for path in mentioned_paths:
            if not _path_allowed(path, set(evidence_index.workspace_paths)):
                raise ValueError(f"{composer_kind} invented workspace file: {path}")

    if not has_workspace_evidence:
        if _contains_any(lower, _INSPECTION_CLAIMS):
            raise ValueError(f"{composer_kind} made a workspace inspection claim without workspace_context")
        if _contains_any(lower, _IMPLEMENTATION_CLAIMS):
            raise ValueError(f"{composer_kind} made a workspace implementation claim without workspace_context")
        return

    _validate_workspace_exploration_status_claims(
        text,
        workspace,
        composer_kind=composer_kind,
    )

    for symbol in _mentioned_code_symbols(text):
        if symbol.lower() not in evidence_index.workspace_symbols:
            raise ValueError(f"{composer_kind} invented workspace symbol: {symbol}")

    _validate_workspace_source_claims(text, evidence_index, composer_kind=composer_kind)


def validate_url_claims(
    text: str,
    facts: Mapping[str, Any],
    *,
    composer_kind: str = "composer",
) -> None:
    url_context = url_context_from_facts(facts)
    has_url_evidence = _has_url_evidence(url_context)
    lower = text.lower()
    mentioned_urls = _mentioned_urls(text)
    if mentioned_urls:
        for url in mentioned_urls:
            if not _url_allowed(url, set(build_citation_evidence_index(facts).urls)):
                raise ValueError(f"{composer_kind} invented external URL: {url}")
    if not has_url_evidence and _contains_any(lower, _URL_INSPECTION_CLAIMS):
        raise ValueError(f"{composer_kind} made a URL inspection claim without url_context")
    if has_url_evidence:
        _validate_url_source_claims(text, build_citation_evidence_index(facts), composer_kind=composer_kind)


def validate_delegated_perception_claims(
    text: str,
    facts: Mapping[str, Any],
    *,
    evidence_index: CitationEvidenceIndex | None = None,
    composer_kind: str = "composer",
) -> None:
    delegated_context = delegated_perception_context_from_facts(facts)
    lower = text.lower()
    if not _has_delegated_evidence(delegated_context):
        if (
            _mentioned_delegated_ref_ids(text)
            or _contains_any(lower, _DELEGATED_REPORT_MARKERS)
            or _contains_any(lower, _DELEGATED_CROSS_CHECK_CLAIMS)
        ):
            raise ValueError(f"{composer_kind} made a delegated source claim without delegated_perception_context")
        return

    index = evidence_index or build_citation_evidence_index(facts)
    mentions_delegated = _mentions_delegated_context(text, delegated_context, index)
    only_delegated_context = not _has_workspace_evidence(workspace_context_from_facts(facts)) and not _has_url_evidence(
        url_context_from_facts(facts)
    )

    _validate_delegated_references(text, index, composer_kind=composer_kind)

    if _contains_any(lower, _DELEGATED_DIRECT_INSPECTION_CLAIMS):
        raise ValueError(f"{composer_kind} claimed Spice directly inspected delegated sources")

    delegated_urls = {item.url for item in index.delegated_items if item.url}
    direct_urls = _allowed_urls(url_context_from_facts(facts))
    for url in _mentioned_urls(text):
        if _url_allowed(url, delegated_urls) and not _url_allowed(url, direct_urls):
            if _contains_any(lower, _URL_INSPECTION_CLAIMS):
                raise ValueError(f"{composer_kind} claimed direct URL inspection for delegated source: {url}")

    if only_delegated_context and (
        _contains_any(lower, _INSPECTION_CLAIMS) or _contains_any(lower, _URL_INSPECTION_CLAIMS)
    ):
        raise ValueError(f"{composer_kind} made a direct inspection claim with only delegated context")

    if mentions_delegated and not _has_delegated_attribution(text, delegated_context):
        raise ValueError(f"{composer_kind} referenced delegated sources without executor attribution")

    if mentions_delegated and _contains_any(lower, _DELEGATED_FINAL_DECISION_CLAIMS):
        raise ValueError(f"{composer_kind} presented delegated findings as a final decision")

    if mentions_delegated and _contains_any(lower, _DELEGATED_EXECUTION_COMPLETED_CLAIMS):
        raise ValueError(f"{composer_kind} described delegated perception as completed execution")

    if _has_positive_cross_check_claim(text) and mentions_delegated:
        _validate_delegated_cross_check_claim(text, index, composer_kind=composer_kind)


def build_citation_evidence_index(facts: Mapping[str, Any]) -> CitationEvidenceIndex:
    workspace = workspace_context_from_facts(facts)
    url_context = url_context_from_facts(facts)
    delegated_context = delegated_perception_context_from_facts(facts)
    workspace_items = tuple(_workspace_evidence_items(workspace))
    url_items = tuple(_url_evidence_items(url_context))
    delegated_items = tuple(_delegated_evidence_items(delegated_context))
    workspace_paths: set[str] = set()
    workspace_symbols: set[str] = set()
    urls: set[str] = set()
    for item in workspace_items:
        if item.path:
            workspace_paths.add(item.path)
        workspace_symbols.update(item.symbols)
    for item in url_items:
        if item.url:
            urls.add(item.url)
    for item in delegated_items:
        if item.url:
            urls.add(item.url)
    return CitationEvidenceIndex(
        workspace_paths=frozenset(workspace_paths),
        workspace_symbols=frozenset(workspace_symbols),
        urls=frozenset(urls),
        workspace_items=workspace_items,
        url_items=url_items,
        delegated_items=delegated_items,
    )


def workspace_context_from_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    decision_context = _mapping(facts.get("decision_context"))
    workspace = _mapping(decision_context.get("workspace_context"))
    if workspace:
        return workspace
    recent_context = _mapping(facts.get("recent_context"))
    return _mapping(recent_context.get("workspace_context"))


def url_context_from_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    decision_context = _mapping(facts.get("decision_context"))
    url_context = _mapping(decision_context.get("url_context"))
    if url_context:
        return url_context
    recent_context = _mapping(facts.get("recent_context"))
    return _mapping(recent_context.get("url_context"))


def delegated_perception_context_from_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    decision_context = _mapping(facts.get("decision_context"))
    delegated = _mapping(decision_context.get("delegated_perception_context"))
    if delegated:
        return delegated
    recent_context = _mapping(facts.get("recent_context"))
    return _mapping(recent_context.get("delegated_perception_context"))


def _has_workspace_evidence(workspace: Mapping[str, Any]) -> bool:
    if not workspace:
        return False
    if str(workspace.get("perception_id") or "").strip():
        return True
    if str(workspace.get("source") or "").strip() == "workspace_perception":
        return True
    return bool(_list(workspace.get("facts")) or _list(workspace.get("snippets")) or _list(workspace.get("files_read")))


def _has_url_evidence(url_context: Mapping[str, Any]) -> bool:
    if not url_context:
        return False
    if str(url_context.get("perception_id") or "").strip():
        return True
    if str(url_context.get("source") or "").strip() == "url_perception":
        return True
    return bool(_list(url_context.get("facts")) or _list(url_context.get("snippets")) or _list(url_context.get("documents")))


def _has_delegated_evidence(delegated_context: Mapping[str, Any]) -> bool:
    if not delegated_context:
        return False
    if str(delegated_context.get("perception_id") or "").strip():
        return True
    if str(delegated_context.get("source") or "").strip() == "delegated_perception":
        return True
    return bool(
        _list(delegated_context.get("findings"))
        or _list(delegated_context.get("sources"))
        or str(delegated_context.get("summary") or "").strip()
    )


def _allowed_urls(url_context: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for url in _strings(url_context.get("urls")):
        normalized = _normalize_url(url)
        if normalized:
            result.add(normalized)
    for item in _list(url_context.get("documents")):
        payload = _mapping(item)
        for key in ("url", "final_url"):
            normalized = _normalize_url(str(payload.get(key) or ""))
            if normalized:
                result.add(normalized)
    for item in _list(url_context.get("facts")):
        normalized = _normalize_url(str(_mapping(item).get("source_url") or ""))
        if normalized:
            result.add(normalized)
    for item in _list(url_context.get("snippets")):
        normalized = _normalize_url(str(_mapping(item).get("url") or ""))
        if normalized:
            result.add(normalized)
    return result


def _allowed_workspace_paths(workspace: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for item in _list(workspace.get("files_read")):
        path = _normalize_path(str(_mapping(item).get("path") or ""))
        if path:
            result.add(path)
    for item in _list(workspace.get("facts")):
        path = _normalize_path(str(_mapping(item).get("source_path") or ""))
        if path:
            result.add(path)
    for item in _list(workspace.get("snippets")):
        path = _normalize_path(str(_mapping(item).get("path") or ""))
        if path:
            result.add(path)
    return result


def _workspace_evidence_items(workspace: Mapping[str, Any]) -> list[SourceEvidenceItem]:
    items: list[SourceEvidenceItem] = []
    summary = str(workspace.get("summary") or "").strip()
    if summary:
        items.append(_source_item(source_type="workspace_summary", text=summary))
    for item in _list(workspace.get("files_read")):
        payload = _mapping(item)
        path = _normalize_path(str(payload.get("path") or ""))
        if path:
            items.append(_source_item(source_type="workspace_file", path=path, text=path))
    for item in _list(workspace.get("facts")):
        payload = _mapping(item)
        text = str(payload.get("text") or "")
        path = _normalize_path(str(payload.get("source_path") or ""))
        items.append(_source_item(source_type="workspace_fact", source_id=path, path=path, text=f"{path}\n{text}"))
    for item in _list(workspace.get("snippets")):
        payload = _mapping(item)
        text = str(payload.get("text") or "")
        path = _normalize_path(str(payload.get("path") or ""))
        items.append(_source_item(source_type="workspace_snippet", source_id=path, path=path, text=f"{path}\n{text}"))
    cache = _mapping(workspace.get("workspace_summary_cache"))
    if cache:
        cache_summary = str(cache.get("summary") or "")
        if cache_summary:
            items.append(_source_item(source_type="workspace_cache", text=cache_summary))
        for item in _list(cache.get("file_summaries")):
            payload = _mapping(item)
            path = _normalize_path(str(payload.get("path") or ""))
            purpose = str(payload.get("purpose") or "")
            if path or purpose:
                items.append(
                    _source_item(
                        source_type="workspace_cache_file",
                        source_id=path,
                        path=path,
                        text=f"{path}\n{purpose}",
                    )
                )
    return [item for item in items if item.text.strip() or item.path]


def _url_evidence_items(url_context: Mapping[str, Any]) -> list[SourceEvidenceItem]:
    items: list[SourceEvidenceItem] = []
    summary = str(url_context.get("summary") or "").strip()
    if summary:
        items.append(_source_item(source_type="url_summary", text=summary))
    for url in _strings(url_context.get("urls")):
        normalized = _normalize_url(url)
        if normalized:
            items.append(_source_item(source_type="url", url=normalized, text=normalized))
    for item in _list(url_context.get("documents")):
        payload = _mapping(item)
        url = _normalize_url(str(payload.get("final_url") or payload.get("url") or ""))
        title = str(payload.get("title") or "")
        if url or title:
            items.append(_source_item(source_type="url_document", source_id=url, url=url, text=f"{title}\n{url}"))
    for item in _list(url_context.get("facts")):
        payload = _mapping(item)
        url = _normalize_url(str(payload.get("source_url") or ""))
        title = str(payload.get("title") or "")
        text = str(payload.get("text") or "")
        items.append(_source_item(source_type="url_fact", source_id=url, url=url, text=f"{title}\n{url}\n{text}"))
    for item in _list(url_context.get("snippets")):
        payload = _mapping(item)
        url = _normalize_url(str(payload.get("url") or ""))
        title = str(payload.get("title") or "")
        text = str(payload.get("text") or "")
        items.append(_source_item(source_type="url_snippet", source_id=url, url=url, text=f"{title}\n{url}\n{text}"))
    return [item for item in items if item.text.strip() or item.url]


def _delegated_evidence_items(delegated_context: Mapping[str, Any]) -> list[SourceEvidenceItem]:
    items: list[SourceEvidenceItem] = []
    summary = str(delegated_context.get("summary") or "").strip()
    executor = str(delegated_context.get("executor_id") or "").strip()
    if summary:
        items.append(_source_item(source_type="delegated_summary", text=f"{executor}\n{summary}"))
    for item in _list(delegated_context.get("findings")):
        payload = _mapping(item)
        text = str(payload.get("text") or "")
        finding_id = str(payload.get("finding_id") or "")
        if text:
            items.append(
                _source_item(
                    source_type="delegated_finding",
                    source_id=finding_id,
                    text=f"{executor}\n{finding_id}\n{text}",
                )
            )
    for item in _list(delegated_context.get("sources")):
        payload = _mapping(item)
        uri = _normalize_url(str(payload.get("uri") or ""))
        title = str(payload.get("title") or "")
        excerpt = str(payload.get("excerpt") or "")
        source_id = str(payload.get("source_id") or uri)
        verification_status = str(payload.get("verification_status") or "")
        items.append(
            _source_item(
                source_type="delegated_source",
                source_id=source_id,
                url=uri,
                text=f"{executor}\n{title}\n{uri}\n{excerpt}",
                verification_status=verification_status,
            )
        )
    return [item for item in items if item.text.strip() or item.url]


def _source_item(
    *,
    source_type: str,
    source_id: str = "",
    path: str = "",
    url: str = "",
    text: str = "",
    verification_status: str = "",
) -> SourceEvidenceItem:
    symbols = frozenset(_symbols_from_evidence(text))
    tokens = frozenset(_meaningful_tokens(text))
    return SourceEvidenceItem(
        source_type=source_type,
        source_id=source_id,
        text=text,
        path=path,
        url=url,
        verification_status=verification_status,
        symbols=symbols,
        tokens=tokens,
        implementation_supported=_has_implementation_marker(text),
    )


def _validate_workspace_source_claims(
    text: str,
    evidence_index: CitationEvidenceIndex,
    *,
    composer_kind: str,
) -> None:
    fragments = _claim_fragments(text, (*_IMPLEMENTATION_CLAIMS, *_SOURCE_ASSERTION_CLAIMS))
    if not fragments:
        return
    implementation_fragments = [
        fragment for fragment in fragments if _contains_any(fragment.lower(), _IMPLEMENTATION_CLAIMS)
    ]
    if not implementation_fragments:
        return
    for fragment in implementation_fragments:
        if not _workspace_fragment_supported(fragment, evidence_index):
            raise ValueError(
                f"{composer_kind} made a workspace implementation claim without supporting facts"
            )


def _validate_workspace_exploration_status_claims(
    text: str,
    workspace: Mapping[str, Any],
    *,
    composer_kind: str,
) -> None:
    status = str(workspace.get("exploration_status") or "").strip().lower()
    if status not in {"partial", "budget_exhausted"}:
        return
    lower = text.lower()
    if _contains_any(lower, _COMPLETE_WORKSPACE_CLAIMS):
        raise ValueError(
            f"{composer_kind} claimed complete workspace exploration despite {status} evidence"
        )
    mentions_workspace_evidence = (
        _contains_any(lower, _INSPECTION_CLAIMS)
        or _contains_any(lower, _IMPLEMENTATION_CLAIMS)
        or _contains_any(lower, _SOURCE_ASSERTION_CLAIMS)
        or bool(_mentioned_paths(text))
    )
    if mentions_workspace_evidence and not _contains_any(lower, _WORKSPACE_LIMITATION_MARKERS):
        raise ValueError(
            f"{composer_kind} omitted workspace exploration limitation for {status} evidence"
        )


def _workspace_fragment_supported(fragment: str, evidence_index: CitationEvidenceIndex) -> bool:
    items = [
        item
        for item in evidence_index.workspace_items
        if item.implementation_supported
        and item.source_type in _WORKSPACE_IMPLEMENTATION_SOURCE_TYPES
    ]
    if not items:
        return False
    mentioned_paths = _mentioned_paths(fragment)
    mentioned_symbols = {symbol.lower() for symbol in _mentioned_code_symbols(fragment)}
    if mentioned_paths:
        for path in mentioned_paths:
            if not any(
                _path_allowed(path, {item.path}) and item.implementation_supported
                for item in items
                if item.path
            ):
                return False
        return True
    if mentioned_symbols:
        for symbol in mentioned_symbols:
            if not any(symbol in item.symbols and item.implementation_supported for item in items):
                return False
        return True
    fragment_tokens = set(_meaningful_tokens(fragment))
    if not fragment_tokens:
        return False
    return any(len(fragment_tokens & set(item.tokens)) >= 2 for item in items)


def _validate_url_source_claims(
    text: str,
    evidence_index: CitationEvidenceIndex,
    *,
    composer_kind: str,
) -> None:
    fragments = _claim_fragments(text, _URL_INSPECTION_CLAIMS)
    if not fragments:
        return
    for fragment in fragments:
        if not _url_fragment_supported(fragment, evidence_index):
            raise ValueError(f"{composer_kind} made a URL claim without supporting facts")


def _url_fragment_supported(fragment: str, evidence_index: CitationEvidenceIndex) -> bool:
    if not evidence_index.url_items:
        return False
    mentioned_urls = _mentioned_urls(fragment)
    if mentioned_urls:
        for url in mentioned_urls:
            if not _url_allowed(url, set(evidence_index.urls)):
                return False
    fragment_tokens = set(_meaningful_tokens(fragment))
    if not fragment_tokens:
        return bool(mentioned_urls)
    return any(len(fragment_tokens & set(item.tokens)) >= 2 for item in evidence_index.url_items)


def _validate_delegated_references(
    text: str,
    evidence_index: CitationEvidenceIndex,
    *,
    composer_kind: str,
) -> None:
    if not evidence_index.delegated_items:
        return
    known_ids = {
        item.source_id.lower()
        for item in evidence_index.delegated_items
        if item.source_id
    }
    for ref_id in _mentioned_delegated_ref_ids(text):
        if ref_id.lower() not in known_ids:
            raise ValueError(f"{composer_kind} cited nonexistent delegated source: {ref_id}")


def _mentions_delegated_context(
    text: str,
    delegated_context: Mapping[str, Any],
    evidence_index: CitationEvidenceIndex,
) -> bool:
    lower = text.lower()
    if _contains_any(lower, _DELEGATED_REPORT_MARKERS):
        return True
    executor = str(delegated_context.get("executor_id") or "").strip().lower()
    if executor and executor in lower:
        return True
    for item in evidence_index.delegated_items:
        if item.source_id and item.source_id.lower() in lower:
            return True
        if item.url and item.url.lower() in lower:
            return True
    return False


def _has_delegated_attribution(text: str, delegated_context: Mapping[str, Any]) -> bool:
    lower = text.lower()
    if _contains_any(lower, _DELEGATED_REPORT_MARKERS):
        return True
    executor = str(delegated_context.get("executor_id") or "").strip().lower()
    if executor and executor in lower:
        return True
    observed_by = {
        str(_mapping(source).get("observed_by") or "").strip().lower()
        for source in _list(delegated_context.get("sources"))
    }
    return any(value and value in lower for value in observed_by)


def _has_positive_cross_check_claim(text: str) -> bool:
    lower = text.lower()
    if not _contains_any(lower, _DELEGATED_CROSS_CHECK_CLAIMS):
        return False
    fragments = _claim_fragments(text, _DELEGATED_CROSS_CHECK_CLAIMS)
    if not fragments:
        fragments = [text]
    return any(
        not _contains_any(fragment.lower(), _DELEGATED_CROSS_CHECK_NEGATIONS)
        for fragment in fragments
    )


def _validate_delegated_cross_check_claim(
    text: str,
    evidence_index: CitationEvidenceIndex,
    *,
    composer_kind: str,
) -> None:
    delegated_sources = [
        item for item in evidence_index.delegated_items if item.source_type == "delegated_source"
    ]
    if not delegated_sources:
        raise ValueError(f"{composer_kind} claimed delegated source was cross-checked without sources")

    mentioned_ids = {ref_id.lower() for ref_id in _mentioned_delegated_ref_ids(text)}
    mentioned_urls = {_normalize_url(url) for url in _mentioned_urls(text)}
    mentioned_sources = [
        item
        for item in delegated_sources
        if (item.source_id and item.source_id.lower() in mentioned_ids)
        or (item.url and item.url in mentioned_urls)
    ]
    sources_to_check = mentioned_sources or delegated_sources
    for item in sources_to_check:
        if item.verification_status.lower() != "cross_checked":
            raise ValueError(
                f"{composer_kind} claimed delegated source was cross-checked without cross_checked verification_status"
            )


def _mentioned_delegated_ref_ids(text: str) -> list[str]:
    return _dedupe([match.group(0) for match in _DELEGATED_REF_RE.finditer(text)])


def _mentioned_paths(text: str) -> list[str]:
    result: list[str] = []
    for match in _PATH_TOKEN_RE.finditer(text):
        path = _normalize_path(match.group(0))
        if path:
            result.append(path)
    for match in _BACKTICK_RE.finditer(text):
        token = _normalize_path(match.group(1))
        if _looks_like_file_path(token):
            result.append(token)
    return _dedupe(result)


def _mentioned_urls(text: str) -> list[str]:
    result: list[str] = []
    for match in _URL_RE.finditer(text):
        normalized = _normalize_url(match.group(0))
        if normalized:
            result.append(normalized)
    return _dedupe(result)


def _mentioned_code_symbols(text: str) -> list[str]:
    result: list[str] = []
    for match in _FUNCTION_CLAIM_RE.finditer(text):
        result.append(match.group(1))
    for match in _BACKTICK_RE.finditer(text):
        token = match.group(1).strip()
        if _BACKTICK_SYMBOL_RE.match(token) and not _looks_like_file_path(token):
            result.append(token)
    return _dedupe(result)


def _workspace_evidence_text(workspace: Mapping[str, Any]) -> str:
    parts: list[str] = [
        str(workspace.get("summary") or ""),
    ]
    for item in _list(workspace.get("facts")):
        payload = _mapping(item)
        parts.append(str(payload.get("text") or ""))
        parts.append(str(payload.get("source_path") or ""))
    for item in _list(workspace.get("snippets")):
        payload = _mapping(item)
        parts.append(str(payload.get("text") or ""))
        parts.append(str(payload.get("path") or ""))
    return "\n".join(parts).lower()


def _has_implementation_evidence(workspace: Mapping[str, Any]) -> bool:
    evidence = _workspace_evidence_text(workspace)
    if not evidence.strip():
        return False
    return _has_implementation_marker(evidence)


def _path_allowed(path: str, allowed_paths: set[str]) -> bool:
    normalized = _normalize_path(path)
    if not normalized:
        return True
    if normalized in allowed_paths:
        return True
    basename = normalized.rsplit("/", 1)[-1]
    for allowed in allowed_paths:
        if allowed == basename or allowed.endswith(f"/{basename}"):
            return True
    return False


def _url_allowed(url: str, allowed_urls: set[str]) -> bool:
    normalized = _normalize_url(url)
    return not normalized or normalized in allowed_urls


def _claim_fragments(text: str, claims: tuple[str, ...]) -> list[str]:
    fragments: list[str] = []
    for fragment in re.split(r"(?<=[.!?。！？；;])\s+|\n+", text):
        stripped = fragment.strip()
        if stripped and _contains_any(stripped.lower(), claims):
            fragments.append(stripped)
    if fragments:
        return fragments
    lower = text.lower()
    if _contains_any(lower, claims):
        return [text.strip()]
    return []


def _symbols_from_evidence(text: str) -> list[str]:
    result: list[str] = []
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text):
        token = match.group(0)
        if token.lower() not in _STOP_TOKENS:
            result.append(token.lower())
    return _dedupe(result)


def _meaningful_tokens(text: str) -> list[str]:
    result: list[str] = []
    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text.lower()):
        token = match.group(0)
        for part in re.split(r"[_-]+", token):
            if len(part) >= 3 and part not in _STOP_TOKENS:
                result.append(part)
        if len(token) >= 3 and token not in _STOP_TOKENS:
            result.append(token)
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", text):
        token = match.group(0)
        if token not in _STOP_TOKENS:
            result.append(token)
    return _dedupe(result)


def _has_implementation_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in _IMPLEMENTATION_MARKERS)


def _normalize_path(value: str) -> str:
    text = value.strip().strip("`'\"()[]{}，。；;:,.!?")
    while text.startswith("./"):
        text = text[2:]
    return text


def _normalize_url(value: str) -> str:
    return str(value or "").strip().rstrip(".,，。;；:：!?)]}>'\"")


def _looks_like_file_path(value: str) -> bool:
    text = _normalize_path(value)
    if "/" in text:
        return True
    lower = text.lower()
    return any(lower.endswith(f".{ext}") for ext in _FILE_EXTENSIONS)


def _contains_any(text_lower: str, claims: tuple[str, ...]) -> bool:
    return any(claim.lower() in text_lower for claim in claims)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if str(item or "")]


_STOP_TOKENS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "onto",
    "about",
    "because",
    "option",
    "candidate",
    "decision",
    "context",
    "workspace",
    "repo",
    "code",
    "file",
    "facts",
    "fact",
    "source",
    "show",
    "shows",
    "checked",
    "read",
    "inspected",
    "already",
    "implemented",
    "implements",
    "support",
    "supports",
    "wired",
    "receives",
    "uses",
    "calls",
    "defines",
}
