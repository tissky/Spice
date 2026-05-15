from __future__ import annotations

import re


READ_ONLY_PERCEPTION_DECISION_MODE = "read_only_perception"


_READ_ONLY_PATTERNS = (
    r"\bread(?:ing)?\s+(?:the\s+)?(?:local\s+)?(?:repo|repository|workspace|codebase|files?)\b",
    r"\binspect(?:ing)?\s+(?:the\s+)?(?:current\s+)?(?:implementation|repo|repository|workspace|codebase|files?)\b",
    r"\bbased\s+on\s+(?:the\s+)?(?:actual\s+code|current\s+code|current\s+implementation|repo|repository|codebase)\b",
    r"\bcurrent\s+(?:implementation|code|codebase|repo|repository)\b",
    r"\bactual\s+(?:code|implementation)\b",
    r"\bcode\s+evidence\b",
    r"\bimplementation\s+evidence\b",
    r"读取(?:本地)?(?:文件|仓库|代码|repo|repository)",
    r"读(?:一下)?(?:本地)?(?:文件|仓库|代码|repo|repository)",
    r"看一下(?:当前)?(?:实现|代码|仓库|repo)",
    r"基于(?:实际代码|当前代码|当前实现|代码实现)",
    r"当前(?:实现|代码|仓库|repo)",
    r"实际代码",
    r"代码证据",
    r"实现证据",
)

_EXECUTION_PATTERNS = (
    r"\b(?:write|patch|modify|edit|change|create|delete|move|install|execute|implement|fix|apply|commit|push|deploy)\b",
    r"\brun\s+(?!read-only\b)",
    r"\btest(?:s|ing)?\b",
    r"写入",
    r"修改",
    r"创建",
    r"删除",
    r"移动",
    r"安装",
    r"运行",
    r"跑测试",
    r"测试",
    r"执行(?:$|[\s，。,.！!？?])",
    r"执行(?:这个|这些|刚才|选中|方案|任务)",
    r"开始实现",
    r"实现(?:一下|这个|这些|方案|功能)",
    r"修复",
    r"提交",
    r"推送",
    r"部署",
)


def is_read_only_perception_request(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if _matches_any(normalized, _EXECUTION_PATTERNS):
        return False
    return _matches_any(normalized, _READ_ONLY_PATTERNS)


def is_read_only_perception_decision_mode(decision_mode: str) -> bool:
    return str(decision_mode or "").strip() == READ_ONLY_PERCEPTION_DECISION_MODE


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()
