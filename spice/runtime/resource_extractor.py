from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


RESOURCE_EXTRACTION_SCHEMA_VERSION = "spice.resource_extraction.v1"


_URL_RE = re.compile(r"\bhttps?://[^\s<>'\"，。！？；、]+", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:/.-])(?:~|/(?:Users|private|tmp|var|opt|usr|etc|home|Volumes))"
    r"(?:/[^\s<>'\"，。！？；、]+)*"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![\w:/.-])(?:\./|\.\./)(?:[^\s<>'\"，。！？；、]+)"
)
_REPO_RELATIVE_PATH_RE = re.compile(
    r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?"
)
_FILE_REF_RE = re.compile(
    r"(?<![\w./-])(?:[A-Za-z0-9_.-]+\.)"
    r"(?:py|pyi|js|jsx|ts|tsx|json|toml|yaml|yml|md|rst|txt|html|css|scss|sh|sql|go|rs|java|kt|swift|c|cc|cpp|h|hpp)"
    r"(?![\w/-])",
    re.IGNORECASE,
)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,120})`")
_CALL_SYMBOL_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]{1,80})\s*\(")
_DOTTED_SYMBOL_RE = re.compile(
    r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,6})(?![\w.])"
)

_RESOURCE_TRAILING = ".,;:!?)]}，。！？；、）】》"

_REPO_HINT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("repo", (" repo", "repository", "代码库", "仓库")),
    ("workspace", ("workspace", "worktree", "工作区")),
    ("local_workspace", ("本地", "local repo", "local repository", "local workspace")),
    (
        "current_implementation",
        ("当前实现", "实际代码", "当前代码", "current implementation", "actual implementation", "current code"),
    ),
    ("codebase", ("codebase", "代码", "源码")),
)

_EXTERNAL_RESEARCH_HINT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("latest", ("latest", "up-to-date", "最新", "最近", "现在的")),
    ("web_research", ("web research", "web search", "search the web", "联网", "网上", "网页")),
    ("lookup", ("查一下", "搜一下", "搜索", "look up", "research")),
    ("external_compare", ("对比 hermes", "对比 openclaw", "compare hermes", "compare openclaw")),
    ("github", ("github", "pull request", "issue", "pr ")),
)


@dataclass(frozen=True, slots=True)
class ResourceExtraction:
    urls: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)
    relative_paths: list[str] = field(default_factory=list)
    file_refs: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    repo_hints: list[str] = field(default_factory=list)
    external_research_hints: list[str] = field(default_factory=list)

    @property
    def has_resources(self) -> bool:
        return bool(self.urls or self.local_paths or self.relative_paths or self.file_refs or self.symbols)

    @property
    def has_repo_signal(self) -> bool:
        return bool(self.local_paths or self.relative_paths or self.file_refs or self.symbols or self.repo_hints)

    @property
    def has_external_signal(self) -> bool:
        return bool(self.urls or self.external_research_hints)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_EXTRACTION_SCHEMA_VERSION,
            "urls": list(self.urls),
            "local_paths": list(self.local_paths),
            "relative_paths": list(self.relative_paths),
            "file_refs": list(self.file_refs),
            "symbols": list(self.symbols),
            "repo_hints": list(self.repo_hints),
            "external_research_hints": list(self.external_research_hints),
            "has_resources": self.has_resources,
            "has_repo_signal": self.has_repo_signal,
            "has_external_signal": self.has_external_signal,
        }


def extract_resources(text: str) -> ResourceExtraction:
    """Extract hard resource references and lightweight hints from user text.

    This layer intentionally does not decide whether Spice should read anything.
    It only records resources that later routing, scope resolution, and evidence
    gates can use without relying on an LLM to rediscover them.
    """

    source = str(text or "")
    urls = _unique(_clean_resource(match.group(0)) for match in _URL_RE.finditer(source))
    text_without_urls = _URL_RE.sub(" ", source)

    local_paths = _unique(_clean_resource(match.group(0)) for match in _ABSOLUTE_PATH_RE.finditer(text_without_urls))
    text_without_absolute_paths = _ABSOLUTE_PATH_RE.sub(" ", text_without_urls)

    relative_paths = _unique(
        _clean_resource(match.group(0)) for match in _RELATIVE_PATH_RE.finditer(text_without_absolute_paths)
    )
    relative_paths = _unique(
        [
            *relative_paths,
            *(
                _clean_resource(match.group(0))
                for match in _REPO_RELATIVE_PATH_RE.finditer(text_without_absolute_paths)
                if _looks_like_repo_relative_path(match.group(0))
            ),
        ]
    )

    file_refs = _unique(
        _clean_resource(match.group(0))
        for match in _FILE_REF_RE.finditer(text_without_absolute_paths)
        if "/" not in match.group(0)
    )
    symbols = _extract_symbols(text_without_urls)
    repo_hints = _pattern_hints(source, _REPO_HINT_PATTERNS)
    external_research_hints = _pattern_hints(source, _EXTERNAL_RESEARCH_HINT_PATTERNS)

    return ResourceExtraction(
        urls=urls,
        local_paths=local_paths,
        relative_paths=relative_paths,
        file_refs=file_refs,
        symbols=symbols,
        repo_hints=repo_hints,
        external_research_hints=external_research_hints,
    )


def _extract_symbols(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _BACKTICK_RE.finditer(text):
        value = match.group(1).strip()
        if _looks_like_symbol(value):
            candidates.append(value)
    for match in _CALL_SYMBOL_RE.finditer(text):
        candidates.append(match.group(1))
    for match in _DOTTED_SYMBOL_RE.finditer(text):
        value = match.group(1)
        if "/" not in value and not _looks_like_file_ref(value):
            candidates.append(value)
    return _unique(candidates)


def _pattern_hints(text: str, patterns: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    lowered = f" {text.lower()} "
    result: list[str] = []
    for hint, needles in patterns:
        if any(needle.lower() in lowered for needle in needles):
            result.append(hint)
    return result


def _looks_like_repo_relative_path(value: str) -> bool:
    cleaned = _clean_resource(value)
    if not cleaned or cleaned.startswith(("http://", "https://")):
        return False
    if cleaned.startswith(("/", "./", "../", "~")):
        return False
    parts = cleaned.split("/")
    if len(parts) < 2:
        return False
    if any(not part or part in {".", ".."} for part in parts):
        return False
    tail = parts[-1]
    return "." in tail or len(parts) >= 3


def _looks_like_symbol(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 120:
        return False
    if "/" in cleaned:
        return False
    if _looks_like_file_ref(cleaned):
        return False
    return bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,6}", cleaned)
    )


def _looks_like_file_ref(value: str) -> bool:
    return bool(_FILE_REF_RE.fullmatch(value.strip()))


def _clean_resource(value: str) -> str:
    return value.strip().rstrip(_RESOURCE_TRAILING)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_resource(str(value or ""))
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
