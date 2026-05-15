from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from typing import Any, Mapping

from spice.decision.general.types import payload_value, safe_dataclass_from_payload


URL_PERCEPTION_SCHEMA_VERSION = "spice.url_perception.v1"
URL_CONTEXT_SCHEMA_VERSION = "spice.url_context.v1"

DEFAULT_URL_PERCEPTION_MAX_URLS = 5
DEFAULT_URL_PERCEPTION_MAX_CHARS_PER_URL = 16_000
DEFAULT_URL_PERCEPTION_TOTAL_CHAR_BUDGET = 50_000
DEFAULT_URL_PERCEPTION_TIMEOUT_SECONDS = 12.0

URL_RE = re.compile(r"https?://[^\s<>\(\)\[\]\{\}\"']+", re.IGNORECASE)
_TEXT_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
    "application/vnd.github",
)
_BINARY_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".woff",
    ".woff2",
)


@dataclass(frozen=True, slots=True)
class URLPerceptionLimits:
    max_urls: int = DEFAULT_URL_PERCEPTION_MAX_URLS
    max_chars_per_url: int = DEFAULT_URL_PERCEPTION_MAX_CHARS_PER_URL
    total_char_budget: int = DEFAULT_URL_PERCEPTION_TOTAL_CHAR_BUDGET
    timeout_seconds: float = DEFAULT_URL_PERCEPTION_TIMEOUT_SECONDS
    allow_http: bool = True
    allow_private_hosts: bool = False

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class URLDocument:
    url: str
    final_url: str = ""
    source_type: str = "web_page"
    title: str = ""
    text: str = ""
    chars_read: int = 0
    truncated: bool = False
    status_code: int | None = None
    content_type: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class URLSkipped:
    url: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class URLFact:
    text: str
    source_url: str = ""
    title: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class URLSnippet:
    url: str
    text: str
    title: str = ""
    source: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class URLPerceptionArtifact:
    perception_id: str
    trigger: str
    created_at: str
    query: str = ""
    urls: list[str] = field(default_factory=list)
    documents: list[URLDocument] = field(default_factory=list)
    urls_skipped: list[URLSkipped] = field(default_factory=list)
    facts: list[URLFact] = field(default_factory=list)
    snippets: list[URLSnippet] = field(default_factory=list)
    summary: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    limits: URLPerceptionLimits = field(default_factory=URLPerceptionLimits)
    schema_version: str = URL_PERCEPTION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "URLPerceptionArtifact":
        if not isinstance(payload, Mapping):
            raise ValueError("URL perception payload must be a mapping.")
        limits = payload.get("limits")
        return cls(
            perception_id=_required_string(payload, "perception_id"),
            trigger=str(payload.get("trigger") or "command"),
            created_at=str(payload.get("created_at") or ""),
            query=str(payload.get("query") or ""),
            urls=[str(item) for item in _list(payload.get("urls"))],
            documents=[
                safe_dataclass_from_payload(URLDocument, item)
                for item in _mappings(payload.get("documents"))
            ],
            urls_skipped=[
                safe_dataclass_from_payload(URLSkipped, item)
                for item in _mappings(payload.get("urls_skipped"))
            ],
            facts=[
                safe_dataclass_from_payload(URLFact, item)
                for item in _mappings(payload.get("facts"))
            ],
            snippets=[
                safe_dataclass_from_payload(URLSnippet, item)
                for item in _mappings(payload.get("snippets"))
            ],
            summary=str(payload.get("summary") or ""),
            budget=dict(payload.get("budget"))
            if isinstance(payload.get("budget"), dict)
            else {},
            limits=(
                safe_dataclass_from_payload(URLPerceptionLimits, limits)
                if isinstance(limits, Mapping)
                else URLPerceptionLimits()
            ),
            schema_version=str(payload.get("schema_version") or URL_PERCEPTION_SCHEMA_VERSION),
            metadata=dict(payload.get("metadata"))
            if isinstance(payload.get("metadata"), dict)
            else {},
        )


@dataclass(frozen=True, slots=True)
class URLPerceptionResult:
    query: str
    urls: list[str]
    documents: list[URLDocument]
    urls_skipped: list[URLSkipped]
    budget: dict[str, Any]
    limits: URLPerceptionLimits

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


class URLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if normalized == "title":
            self._in_title = True
        if normalized in {"p", "br", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if normalized == "title":
            self._in_title = False
            self.title = _normalize_whitespace(" ".join(self._title_parts))
        if normalized in {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._parts.append(data)

    def text(self) -> str:
        return _normalize_text("\n".join(self._parts))


class URLInspector:
    """Bounded read-only URL text fetcher for decision-relevant perception."""

    def __init__(self, limits: URLPerceptionLimits | None = None) -> None:
        self.limits = limits or URLPerceptionLimits()

    def inspect(self, *, urls: list[str], query: str = "") -> URLPerceptionResult:
        normalized_urls = _dedupe_urls(urls)[: max(0, self.limits.max_urls)]
        documents: list[URLDocument] = []
        skipped: list[URLSkipped] = []
        remaining_budget = max(0, self.limits.total_char_budget)
        for url in normalized_urls:
            if remaining_budget <= 0:
                skipped.append(URLSkipped(url=url, reason="total_char_budget_exhausted"))
                continue
            reason = self._blocked_reason(url)
            if reason:
                skipped.append(URLSkipped(url=url, reason=reason))
                continue
            try:
                document, extra_skipped = self._fetch_document(
                    url,
                    max_chars=min(self.limits.max_chars_per_url, remaining_budget),
                )
            except Exception as exc:
                skipped.append(URLSkipped(url=url, reason=f"fetch_failed:{exc}"))
                continue
            documents.append(document)
            skipped.extend(extra_skipped)
            remaining_budget -= document.chars_read
        return URLPerceptionResult(
            query=query,
            urls=normalized_urls,
            documents=documents,
            urls_skipped=skipped,
            budget={
                "requested_url_count": len(urls),
                "normalized_url_count": len(normalized_urls),
                "document_count": len(documents),
                "skipped_count": len(skipped),
                "total_chars_read": sum(document.chars_read for document in documents),
                "total_char_budget": self.limits.total_char_budget,
                "remaining_char_budget": max(0, remaining_budget),
            },
            limits=self.limits,
        )

    def _fetch_document(self, url: str, *, max_chars: int) -> tuple[URLDocument, list[URLSkipped]]:
        source_type = classify_url(url)
        if source_type == "github_file":
            raw_url = github_raw_url(url)
            if raw_url and raw_url != url:
                document = self._fetch_text_url(raw_url, original_url=url, source_type=source_type, max_chars=max_chars)
                return document, []
        if source_type in {"github_issue", "github_pr"}:
            document, skipped = self._fetch_github_issue_or_pr(url, source_type=source_type, max_chars=max_chars)
            return document, skipped
        return self._fetch_text_url(url, original_url=url, source_type=source_type, max_chars=max_chars), []

    def _fetch_github_issue_or_pr(
        self,
        url: str,
        *,
        source_type: str,
        max_chars: int,
    ) -> tuple[URLDocument, list[URLSkipped]]:
        parsed = parse_github_issue_or_pr_url(url)
        if not parsed:
            return self._fetch_text_url(url, original_url=url, source_type=source_type, max_chars=max_chars), []
        owner, repo, kind, number = parsed
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
        api_doc = self._fetch_text_url(api_url, original_url=url, source_type=source_type, max_chars=max_chars)
        data = _loads_json(api_doc.text)
        title = str(data.get("title") or api_doc.title or f"{owner}/{repo} {kind} #{number}")
        body = str(data.get("body") or "")
        lines = [
            f"GitHub {kind} {owner}/{repo}#{number}",
            f"title: {title}",
            f"state: {data.get('state') or ''}",
            f"author: {_mapping(data.get('user')).get('login') or ''}",
        ]
        labels = [
            str(_mapping(label).get("name") or "")
            for label in _list(data.get("labels"))
            if _mapping(label).get("name")
        ]
        if labels:
            lines.append(f"labels: {', '.join(labels[:12])}")
        if body:
            lines.extend(["", body])
        skipped: list[URLSkipped] = []
        if source_type == "github_pr":
            diff_budget = max(0, max_chars - len("\n".join(lines)))
            if diff_budget > 500:
                diff_url = f"https://github.com/{owner}/{repo}/pull/{number}.diff"
                try:
                    diff_doc = self._fetch_text_url(
                        diff_url,
                        original_url=diff_url,
                        source_type="github_pr_diff",
                        max_chars=min(diff_budget, max_chars // 2),
                    )
                    if diff_doc.text:
                        lines.extend(["", "Diff excerpt:", diff_doc.text])
                except Exception as exc:
                    skipped.append(URLSkipped(url=diff_url, reason=f"diff_fetch_failed:{exc}"))
        text, truncated = _truncate("\n".join(lines), max_chars)
        return (
            URLDocument(
                url=url,
                final_url=api_doc.final_url or api_url,
                source_type=source_type,
                title=title,
                text=text,
                chars_read=len(text),
                truncated=truncated or api_doc.truncated,
                status_code=api_doc.status_code,
                content_type=api_doc.content_type,
                content_hash=_hash(text),
                metadata={
                    "github": {
                        "owner": owner,
                        "repo": repo,
                        "kind": kind,
                        "number": number,
                        "api_url": api_url,
                    }
                },
            ),
            skipped,
        )

    def _fetch_text_url(
        self,
        url: str,
        *,
        original_url: str,
        source_type: str,
        max_chars: int,
    ) -> URLDocument:
        if _looks_binary_url(url):
            raise ValueError("binary_url_skipped")
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SpiceURLPerception/0.1 (+read-only)",
                "Accept": "text/html,application/json,text/plain,application/xhtml+xml,*/*;q=0.2",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.limits.timeout_seconds) as response:
            status_code = getattr(response, "status", None) or getattr(response, "code", None)
            final_url = str(response.geturl() or url)
            content_type = str(response.headers.get("content-type") or "")
            if not _allowed_content_type(content_type, final_url):
                raise ValueError(f"unsupported_content_type:{content_type or 'unknown'}")
            raw = response.read(max(0, max_chars) * 4 + 4096)
        decoded = _decode_bytes(raw, content_type)
        title = ""
        if "html" in content_type.lower() or _looks_html(decoded):
            parser = URLTextExtractor()
            parser.feed(decoded)
            title = parser.title
            text = parser.text()
        elif "json" in content_type.lower() or _looks_json(decoded):
            text = _json_to_text(decoded)
        else:
            text = _normalize_text(decoded)
        text, truncated = _truncate(text, max_chars)
        return URLDocument(
            url=original_url,
            final_url=final_url,
            source_type=source_type,
            title=title or _title_from_url(original_url),
            text=text,
            chars_read=len(text),
            truncated=truncated,
            status_code=int(status_code) if status_code is not None else None,
            content_type=content_type,
            content_hash=_hash(text),
            metadata={"fetch_url": url} if url != original_url else {},
        )

    def _blocked_reason(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return "unsupported_scheme"
        if scheme == "http" and not self.limits.allow_http:
            return "http_not_allowed"
        host = (parsed.hostname or "").strip().lower()
        if not host:
            return "missing_host"
        if not self.limits.allow_private_hosts and _is_private_or_local_host(host):
            return "private_or_local_host_blocked"
        if _looks_binary_url(url):
            return "binary_url_skipped"
        return ""


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,，。;；:：!?)]}>'\"")
        if url:
            urls.append(url)
    return _dedupe_urls(urls)


def classify_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host == "raw.githubusercontent.com":
        return "github_file"
    if host == "github.com":
        if "/blob/" in path or "/raw/" in path:
            return "github_file"
        if parse_github_issue_or_pr_url(url):
            parsed_issue = parse_github_issue_or_pr_url(url)
            if parsed_issue and parsed_issue[2] == "pull":
                return "github_pr"
            return "github_issue"
    return "web_page"


def github_raw_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "raw.githubusercontent.com":
        return url
    if host != "github.com":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 5 and parts[2] in {"blob", "raw"}:
        owner, repo, _, ref = parts[:4]
        file_path = "/".join(parts[4:])
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}"
    return ""


def parse_github_issue_or_pr_url(url: str) -> tuple[str, str, str, str] | None:
    parsed = urllib.parse.urlparse(url)
    if (parsed.hostname or "").lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 4 and parts[2] in {"issues", "pull"}:
        owner, repo, kind, number = parts[:4]
        if number.isdigit():
            return owner, repo, kind, number
    return None


def run_url_perception(
    *,
    urls: list[str],
    query: str = "",
    limits: URLPerceptionLimits | None = None,
) -> URLPerceptionResult:
    return URLInspector(limits=limits).inspect(urls=urls, query=query)


def build_url_perception_artifact(
    *,
    trigger: str,
    result: URLPerceptionResult | Mapping[str, Any],
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> URLPerceptionArtifact:
    payload = _payload(result)
    created = _timestamp(created_at)
    documents = [_document(item) for item in _mappings(payload.get("documents"))]
    skipped = [_skipped(item) for item in _mappings(payload.get("urls_skipped"))]
    facts = _facts_from_documents(documents)
    snippets = _snippets_from_documents(documents)
    summary = _summary_from_documents(documents, skipped)
    urls = [str(item) for item in _list(payload.get("urls"))]
    perception_id = _url_perception_id(urls=urls, created_at=created, summary=summary)
    return URLPerceptionArtifact(
        perception_id=perception_id,
        trigger=trigger or "command",
        created_at=created,
        query=str(payload.get("query") or ""),
        urls=urls,
        documents=documents,
        urls_skipped=skipped,
        facts=facts,
        snippets=snippets,
        summary=summary,
        budget=dict(payload.get("budget")) if isinstance(payload.get("budget"), dict) else {},
        limits=_limits(payload.get("limits")),
        metadata=dict(metadata or {}),
    )


def url_context_from_perception(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        "schema_version": URL_CONTEXT_SCHEMA_VERSION,
        "source": "url_perception",
        "perception_id": str(payload.get("perception_id") or ""),
        "trigger": str(payload.get("trigger") or ""),
        "query": str(payload.get("query") or ""),
        "summary": str(payload.get("summary") or ""),
        "urls": [str(item) for item in _list(payload.get("urls"))[:8]],
        "documents": [
            _compact_document(_mapping(item))
            for item in _list(payload.get("documents"))[:8]
        ],
        "urls_skipped": [
            _compact_skipped(_mapping(item))
            for item in _list(payload.get("urls_skipped"))[:8]
        ],
        "facts": [
            _compact_fact(_mapping(item))
            for item in _list(payload.get("facts"))[:10]
        ],
        "snippets": [
            _compact_snippet(_mapping(item))
            for item in _list(payload.get("snippets"))[:8]
        ],
        "budget": dict(payload.get("budget"))
        if isinstance(payload.get("budget"), dict)
        else {},
        "limits": dict(payload.get("limits"))
        if isinstance(payload.get("limits"), dict)
        else {},
    }


def _facts_from_documents(documents: list[URLDocument]) -> list[URLFact]:
    facts: list[URLFact] = []
    for document in documents:
        if not document.text.strip():
            continue
        excerpt = _first_sentences(document.text, max_chars=420)
        facts.append(
            URLFact(
                text=excerpt,
                source_url=document.url,
                title=document.title,
                confidence=0.72,
                metadata={"source_type": document.source_type, "content_hash": document.content_hash},
            )
        )
    return facts


def _snippets_from_documents(documents: list[URLDocument]) -> list[URLSnippet]:
    snippets: list[URLSnippet] = []
    for document in documents:
        text = document.text.strip()
        if not text:
            continue
        snippet_text, _ = _truncate(text, 1600)
        snippets.append(
            URLSnippet(
                url=document.url,
                title=document.title,
                text=snippet_text,
                source=document.source_type,
                content_hash=document.content_hash,
            )
        )
    return snippets


def _summary_from_documents(documents: list[URLDocument], skipped: list[URLSkipped]) -> str:
    if not documents:
        return f"URL perception read no documents; skipped {len(skipped)} URL(s)."
    titles = [
        document.title or document.final_url or document.url
        for document in documents[:3]
    ]
    return (
        f"URL perception read {len(documents)} document(s): "
        + "; ".join(_shorten(title, 120) for title in titles)
        + (f". Skipped {len(skipped)} URL(s)." if skipped else ".")
    )


def _compact_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "url": str(payload.get("url") or ""),
        "final_url": str(payload.get("final_url") or ""),
        "source_type": str(payload.get("source_type") or ""),
        "title": _shorten(str(payload.get("title") or ""), 180),
        "chars_read": payload.get("chars_read"),
        "truncated": bool(payload.get("truncated")),
        "status_code": payload.get("status_code"),
        "content_type": _shorten(str(payload.get("content_type") or ""), 120),
        "content_hash": str(payload.get("content_hash") or ""),
    }


def _compact_skipped(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "url": str(payload.get("url") or ""),
        "reason": _shorten(str(payload.get("reason") or ""), 220),
    }


def _compact_fact(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": _shorten(str(payload.get("text") or ""), 420),
        "source_url": str(payload.get("source_url") or ""),
        "title": _shorten(str(payload.get("title") or ""), 180),
        "confidence": payload.get("confidence"),
    }


def _compact_snippet(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "url": str(payload.get("url") or ""),
        "title": _shorten(str(payload.get("title") or ""), 180),
        "text": _shorten(str(payload.get("text") or ""), 900),
        "source": str(payload.get("source") or ""),
        "content_hash": str(payload.get("content_hash") or ""),
    }


def _allowed_content_type(content_type: str, final_url: str) -> bool:
    lower = (content_type or "").lower().split(";", 1)[0].strip()
    if any(lower.startswith(prefix) for prefix in _TEXT_CONTENT_TYPES):
        return True
    if lower == "":
        return not _looks_binary_url(final_url)
    return False


def _looks_binary_url(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _BINARY_EXTENSIONS)


def _is_private_or_local_host(host: str) -> bool:
    normalized = host.strip().lower().rstrip(".")
    if normalized in {"localhost", "0.0.0.0"} or normalized.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast


def _decode_bytes(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([A-Za-z0-9_.-]+)", content_type or "", re.IGNORECASE)
    if match:
        charset = match.group(1)
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _json_to_text(text: str) -> str:
    payload = _loads_json(text)
    if not payload:
        return _normalize_text(text)
    return _normalize_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _loads_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {"value": payload}


def _looks_json(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_html(text: str) -> bool:
    lower = text[:500].lower()
    return "<html" in lower or "<body" in lower or "<!doctype html" in lower


def _normalize_text(text: str) -> str:
    unescaped = html.unescape(text or "")
    lines = [_normalize_whitespace(line) for line in unescaped.splitlines()]
    compact_lines: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                compact_lines.append("")
            blank = True
            continue
        compact_lines.append(line)
        blank = False
    return "\n".join(compact_lines).strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip(), True


def _first_sentences(text: str, *, max_chars: int) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    for separator in ("\n\n", "\n", ". ", "。"):
        index = normalized.find(separator, max_chars // 3)
        if 0 < index <= max_chars:
            return normalized[:index].strip()
    return normalized[:max_chars].rstrip()


def _title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    tail = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    return urllib.parse.unquote(tail or parsed.netloc or url)


def _dedupe_urls(urls: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _document(value: Mapping[str, Any] | URLDocument) -> URLDocument:
    if isinstance(value, URLDocument):
        return value
    return safe_dataclass_from_payload(URLDocument, value)


def _skipped(value: Mapping[str, Any] | URLSkipped) -> URLSkipped:
    if isinstance(value, URLSkipped):
        return value
    return safe_dataclass_from_payload(URLSkipped, value)


def _limits(value: Any) -> URLPerceptionLimits:
    if isinstance(value, URLPerceptionLimits):
        return value
    if isinstance(value, Mapping):
        return safe_dataclass_from_payload(URLPerceptionLimits, value)
    return URLPerceptionLimits()


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_payload"):
        payload = value.to_payload()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"URL perception payload missing required string: {key}")
    return value


def _timestamp(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _url_perception_id(*, urls: list[str], created_at: str, summary: str) -> str:
    digest = sha256("\n".join([created_at, summary, *urls]).encode("utf-8")).hexdigest()[:12]
    return f"url.{created_at.replace(':', '').replace('-', '').replace('.', '_')}.{digest}"


def _hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16] if text else ""


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
