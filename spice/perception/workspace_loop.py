from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Mapping

from spice.decision.general.types import payload_value
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.llm.util import extract_first_json_object
from spice.perception.workspace_perception_depth import resolve_workspace_perception_depth
from spice.perception.workspace_inspector import (
    WorkspaceGitDiffResult,
    WorkspaceGitLogResult,
    WorkspaceFileIndexResult,
    WorkspaceGitStatusResult,
    WorkspaceInspector,
    WorkspaceInspectorLimits,
    WorkspacePackageMetadataResult,
    WorkspacePythonSymbolIndexResult,
    WorkspacePythonSymbolReadResult,
    WorkspaceReadResult,
    WorkspaceRepoMapResult,
    WorkspaceSearchResult,
    WorkspaceTestStructureResult,
)


WORKSPACE_PERCEPTION_LOOP_SCHEMA_VERSION = "spice.workspace_perception_loop.v1"

EXPLORATION_STATUS_COMPLETE = "complete"
EXPLORATION_STATUS_PARTIAL = "partial"
EXPLORATION_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
EXPLORATION_STATUS_BLOCKED = "blocked"

ALLOWED_WORKSPACE_TOOLS = frozenset(
    {
        "file_index",
        "search",
        "read_file",
        "git_status",
        "git_diff",
        "git_log",
        "repo_map",
        "read_package_metadata",
        "read_test_structure",
        "python_symbol_index",
        "read_python_symbol",
    }
)

SOURCE_BACKED_WORKSPACE_TOOLS = frozenset(
    {
        "search",
        "read_file",
        "git_status",
        "git_diff",
        "git_log",
        "repo_map",
        "read_package_metadata",
        "read_test_structure",
        "python_symbol_index",
        "read_python_symbol",
    }
)

GUARDRAIL_BLOCK_REASONS = frozenset(
    {
        "deny_dir",
        "symlink_escape",
        "binary_file",
        "file_too_large",
        "max_files_read_exceeded",
        "total_char_budget_exceeded",
        "not_file",
    }
)


@dataclass(frozen=True, slots=True)
class ControlledWorkspacePerceptionLimits:
    max_rounds: int = 10
    max_tool_calls: int = 80
    max_tool_calls_per_round: int = 10
    max_blocked_tool_calls: int = 20
    max_blocked_tool_calls_per_round: int = 5
    max_files_read: int = 60
    max_chars_per_file: int = 12_000
    total_char_budget: int = 500_000
    planner_max_tokens: int | None = 4_000
    depth: str = "normal"

    @classmethod
    def from_depth_budget(cls, budget: Any) -> ControlledWorkspacePerceptionLimits:
        return cls(
            max_rounds=int(getattr(budget, "max_rounds", cls.max_rounds)),
            max_tool_calls=int(getattr(budget, "max_tool_calls", cls.max_tool_calls)),
            max_tool_calls_per_round=int(
                getattr(budget, "max_tool_calls_per_round", cls.max_tool_calls_per_round)
            ),
            max_blocked_tool_calls=int(
                getattr(budget, "max_blocked_tool_calls", cls.max_blocked_tool_calls)
            ),
            max_blocked_tool_calls_per_round=int(
                getattr(
                    budget,
                    "max_blocked_tool_calls_per_round",
                    cls.max_blocked_tool_calls_per_round,
                )
            ),
            max_files_read=int(getattr(budget, "max_files_read", cls.max_files_read)),
            max_chars_per_file=int(getattr(budget, "max_chars_per_file", cls.max_chars_per_file)),
            total_char_budget=int(getattr(budget, "total_char_budget", cls.total_char_budget)),
            planner_max_tokens=getattr(budget, "planner_max_tokens", cls.planner_max_tokens),
            depth=str(getattr(budget, "depth", cls.depth) or cls.depth),
        )

    @classmethod
    def from_depth(
        cls,
        *,
        config: Mapping[str, Any] | None = None,
        evidence_domain: str = "",
        answer_mode: str = "",
        user_input: str = "",
        route_policy: Mapping[str, Any] | None = None,
        requested_depth: str = "",
        native_opt_in: bool = False,
    ) -> ControlledWorkspacePerceptionLimits:
        return cls.from_depth_budget(
            resolve_workspace_perception_depth(
                config=config,
                evidence_domain=evidence_domain,
                answer_mode=answer_mode,
                user_input=user_input,
                route_policy=route_policy,
                requested_depth=requested_depth,
                native_opt_in=native_opt_in,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    def to_inspector_limits(self) -> WorkspaceInspectorLimits:
        return WorkspaceInspectorLimits(
            max_files_read=self.max_files_read,
            max_chars_per_file=self.max_chars_per_file,
            total_char_budget=self.total_char_budget,
        )


@dataclass(frozen=True, slots=True)
class WorkspaceToolCallRecord:
    call_id: str
    round_index: int
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "executed"
    reason: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(frozen=True, slots=True)
class ControlledWorkspacePerceptionResult:
    query: str
    tool_calls: list[WorkspaceToolCallRecord] = field(default_factory=list)
    round_batches: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    investigation_state: dict[str, Any] = field(default_factory=dict)
    sufficiency_check: dict[str, Any] = field(default_factory=dict)
    exploration_status: str = EXPLORATION_STATUS_PARTIAL
    done: bool = False
    rounds_used: int = 0
    raw_outputs: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    schema_version: str = WORKSPACE_PERCEPTION_LOOP_SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)

    @property
    def blocked_tool_calls(self) -> list[WorkspaceToolCallRecord]:
        return [call for call in self.tool_calls if call.status == "blocked"]

    @property
    def executed_tool_calls(self) -> list[WorkspaceToolCallRecord]:
        return [call for call in self.tool_calls if call.status == "executed"]


def run_controlled_workspace_perception_loop(
    *,
    client: LLMClient,
    inspector: WorkspaceInspector,
    query: str,
    limits: ControlledWorkspacePerceptionLimits | None = None,
    initial_context: Mapping[str, Any] | None = None,
) -> ControlledWorkspacePerceptionResult:
    """Run a bounded, read-only workspace perception loop.

    The model may request read-only workspace tools, but runtime validation owns
    the allowlist, budgets, path safety, and execution. Invalid calls are recorded
    as blocked records and are never executed.
    """

    loop_limits = limits or ControlledWorkspacePerceptionLimits.from_depth()
    _apply_loop_budget_to_inspector(inspector, loop_limits)
    tool_calls: list[WorkspaceToolCallRecord] = []
    round_batches: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    summary = ""
    investigation_state = _empty_investigation_state()
    sufficiency_check = _empty_sufficiency_check()
    exploration_status = EXPLORATION_STATUS_PARTIAL
    budget_exhausted = False
    blocked_budget_exhausted = False
    budget_pressure_events: list[dict[str, Any]] = []
    last_budget_pressure = "low"
    hard_repo_evidence_required = _hard_repo_evidence_required(initial_context)
    minimum_fallback = {
        "triggered": False,
        "reason": "",
        "tool_calls_executed": 0,
        "tool_calls_blocked": 0,
        "source_backed_evidence": False,
    }
    done = False

    for round_index in range(1, max(loop_limits.max_rounds, 1) + 1):
        pre_round_budget_state = _workspace_loop_budget_state(
            inspector=inspector,
            limits=loop_limits,
            tool_calls=tool_calls,
        )
        last_budget_pressure = _record_budget_pressure_event(
            events=budget_pressure_events,
            budget_state=pre_round_budget_state,
            round_index=round_index,
            stage="before_planner",
            last_pressure=last_budget_pressure,
        )
        if _normal_loop_budget_exhausted(pre_round_budget_state):
            budget_exhausted = True
            exploration_status = _exploration_status_from_state(
                sufficiency_check=sufficiency_check,
                budget_exhausted=True,
            )
            break
        if _blocked_loop_budget_exhausted(pre_round_budget_state):
            blocked_budget_exhausted = True
            exploration_status = _blocked_exploration_status(tool_calls)
            break

        response = client.generate(
            LLMRequest(
                task_hook=LLMTaskHook.PERCEPTION_INTERPRET,
                input_text=_workspace_loop_prompt(
                    query=query,
                    inspector=inspector,
                    limits=loop_limits,
                    round_index=round_index,
                    tool_calls=tool_calls,
                    investigation_state=investigation_state,
                    initial_context=initial_context,
                ),
                system_text=_workspace_loop_system_prompt(),
                response_format_hint="json_object",
                temperature=0.0,
                max_tokens=loop_limits.planner_max_tokens,
                timeout_sec=30.0,
                metadata={
                    "purpose": "workspace_perception_loop",
                    "round_index": round_index,
                },
            )
        )
        raw_outputs.append(response.output_text)
        payload = _parse_loop_payload(response.output_text)
        summary = str(payload.get("summary") or summary or "").strip()
        investigation_state = _merge_investigation_state(
            current=investigation_state,
            payload=payload,
            fallback_summary=summary,
        )
        sufficiency_check = _merge_sufficiency_check(
            current=sufficiency_check,
            payload=payload,
        )
        done = _truthy(payload.get("done"))
        requests = _tool_call_requests(payload)
        if (
            hard_repo_evidence_required
            and round_index == 1
            and not _loop_has_source_backed_evidence(tool_calls)
            and (_sufficiency_is_complete(sufficiency_check) or not requests)
            and not bool(minimum_fallback["triggered"])
        ):
            fallback_records, fallback_requests = _run_minimum_investigation_fallback(
                inspector=inspector,
                query=query,
                round_index=round_index,
                tool_calls=tool_calls,
                limits=loop_limits,
                reason=(
                    "planner returned no source-backed read-only tool calls before claiming "
                    "hard repo evidence was satisfied"
                ),
            )
            _update_minimum_fallback_state(
                minimum_fallback,
                records=fallback_records,
                source_backed_evidence=_loop_has_source_backed_evidence(tool_calls),
            )
            if fallback_records:
                fallback_batch = _round_batch_payload(
                    round_index=round_index,
                    requested_tool_calls=fallback_requests,
                    records=fallback_records,
                )
                fallback_batch["minimum_investigation_fallback"] = True
                fallback_batch["fallback_reason"] = minimum_fallback["reason"]
                round_batches.append(fallback_batch)
                investigation_state = _refresh_investigation_state_from_workspace(
                    investigation_state,
                    inspector=inspector,
                    tool_calls=tool_calls,
                )
            if not _loop_has_source_backed_evidence(tool_calls):
                done = True
                exploration_status = _blocked_exploration_status(tool_calls)
                sufficiency_check = _minimum_fallback_missing_sufficiency_check(
                    "minimum read-only investigation fallback did not collect source-backed repo evidence"
                )
                break
            done = False
            continue

        if _sufficiency_is_complete(sufficiency_check):
            done = True
            exploration_status = EXPLORATION_STATUS_COMPLETE
            break
        if not requests:
            done = True
            exploration_status = _exploration_status_from_state(
                sufficiency_check=sufficiency_check,
                budget_exhausted=False,
            )
            break

        batch_records, round_normal_tool_call_count, round_blocked_tool_call_count, batch_blocked_exhausted = (
            _execute_tool_call_batch(
                inspector=inspector,
                requests=requests,
                round_index=round_index,
                tool_calls=tool_calls,
                limits=loop_limits,
            )
        )
        if batch_blocked_exhausted:
            blocked_budget_exhausted = True
            exploration_status = _blocked_exploration_status(tool_calls)
        round_batches.append(
            _round_batch_payload(
                round_index=round_index,
                requested_tool_calls=requests,
                records=batch_records,
            )
        )
        if (
            hard_repo_evidence_required
            and round_index == 1
            and not _loop_has_source_backed_evidence(tool_calls)
            and not bool(minimum_fallback["triggered"])
            and not blocked_budget_exhausted
        ):
            fallback_records, fallback_requests = _run_minimum_investigation_fallback(
                inspector=inspector,
                query=query,
                round_index=round_index,
                tool_calls=tool_calls,
                limits=loop_limits,
                reason=(
                    "planner first round produced no source-backed repo evidence "
                    "for a hard repo evidence request"
                ),
                round_normal_tool_call_count=round_normal_tool_call_count,
                round_blocked_tool_call_count=round_blocked_tool_call_count,
            )
            _update_minimum_fallback_state(
                minimum_fallback,
                records=fallback_records,
                source_backed_evidence=_loop_has_source_backed_evidence(tool_calls),
            )
            if fallback_records:
                fallback_batch = _round_batch_payload(
                    round_index=round_index,
                    requested_tool_calls=fallback_requests,
                    records=fallback_records,
                )
                fallback_batch["minimum_investigation_fallback"] = True
                fallback_batch["fallback_reason"] = minimum_fallback["reason"]
                round_batches.append(fallback_batch)
            if not _loop_has_source_backed_evidence(tool_calls):
                done = True
                exploration_status = _blocked_exploration_status(tool_calls)
                sufficiency_check = _minimum_fallback_missing_sufficiency_check(
                    "minimum read-only investigation fallback did not collect source-backed repo evidence"
                )
                break
            done = False
        investigation_state = _refresh_investigation_state_from_workspace(
            investigation_state,
            inspector=inspector,
            tool_calls=tool_calls,
        )
        post_round_budget_state = _workspace_loop_budget_state(
            inspector=inspector,
            limits=loop_limits,
            tool_calls=tool_calls,
        )
        last_budget_pressure = _record_budget_pressure_event(
            events=budget_pressure_events,
            budget_state=post_round_budget_state,
            round_index=round_index,
            stage="after_tool_calls",
            last_pressure=last_budget_pressure,
        )
        if _loop_budget_exhausted(
            inspector=inspector,
            limits=loop_limits,
            tool_calls=tool_calls,
        ):
            budget_exhausted = True
            exploration_status = _exploration_status_from_state(
                sufficiency_check=sufficiency_check,
                budget_exhausted=True,
            )
            break
        if _blocked_loop_budget_exhausted(post_round_budget_state):
            blocked_budget_exhausted = True
            exploration_status = _blocked_exploration_status(tool_calls)
            break
        if done:
            exploration_status = _exploration_status_from_state(
                sufficiency_check=sufficiency_check,
                budget_exhausted=False,
            )
            break

    if not done and not budget_exhausted and len(raw_outputs) >= max(loop_limits.max_rounds, 1):
        budget_exhausted = True
    if exploration_status == EXPLORATION_STATUS_PARTIAL:
        if blocked_budget_exhausted:
            exploration_status = _blocked_exploration_status(tool_calls)
        else:
            exploration_status = _exploration_status_from_state(
                sufficiency_check=sufficiency_check,
                budget_exhausted=budget_exhausted,
            )
    if not summary:
        summary = str(investigation_state.get("investigation_summary") or "").strip()
    final_budget_state = _workspace_loop_budget_state(
        inspector=inspector,
        limits=loop_limits,
        tool_calls=tool_calls,
    )
    return ControlledWorkspacePerceptionResult(
        query=query,
        tool_calls=tool_calls,
        round_batches=round_batches,
        summary=summary,
        investigation_state=investigation_state,
        sufficiency_check=sufficiency_check,
        exploration_status=exploration_status,
        done=done,
        rounds_used=len(raw_outputs),
        raw_outputs=raw_outputs,
        budget={
            "limits": loop_limits.to_payload(),
            "tool_calls_recorded": len(tool_calls),
            "tool_calls_executed": len([call for call in tool_calls if call.status == "executed"]),
            "tool_calls_blocked": len([call for call in tool_calls if call.status == "blocked"]),
            "round_batches_recorded": len(round_batches),
            "normal_tool_calls_used": _normal_tool_call_count(tool_calls),
            "blocked_budget_exhausted": blocked_budget_exhausted,
            "inspector": inspector.summarize_workspace().get("budget", {}),
            "budget_pressure": str(final_budget_state.get("budget_pressure") or "low"),
            "budget_state": final_budget_state,
            "budget_pressure_events": budget_pressure_events,
            "minimum_investigation_fallback": minimum_fallback,
        },
    )


def _workspace_loop_system_prompt() -> str:
    return (
        "You are Spice's workspace perception planner. You may request only read-only "
        "workspace tools to gather decision-relevant facts. Do not answer the user. "
        "Do not request writes, code edits, shell commands, installs, test runs, deletes, or moves. "
        "Return only one JSON object."
    )


def _workspace_loop_prompt(
    *,
    query: str,
    inspector: WorkspaceInspector,
    limits: ControlledWorkspacePerceptionLimits,
    round_index: int,
    tool_calls: list[WorkspaceToolCallRecord],
    investigation_state: Mapping[str, Any],
    initial_context: Mapping[str, Any] | None,
) -> str:
    budget_state = _workspace_loop_budget_state(
        inspector=inspector,
        limits=limits,
        tool_calls=tool_calls,
    )
    payload = {
        "task": "Request read-only workspace tool calls that would help answer the query.",
        "query": query,
        "round_index": round_index,
        "limits": limits.to_payload(),
        "budget_state": budget_state,
        "budget_pressure_guidance": _budget_pressure_guidance(budget_state),
        "workspace": {
            "root": str(inspector.workspace_root),
            "deny_dirs": sorted(inspector.deny_dirs),
        },
        "allowed_tools": {
            "file_index": {"args": {"path": "string default .", "limit": "integer optional"}},
            "search": {
                "args": {
                    "pattern": "regex string",
                    "file_glob": "glob string optional",
                    "path": "string default .",
                    "limit": "integer optional",
                }
            },
            "read_file": {
                "args": {
                    "path": "string",
                    "offset": "1-indexed line number optional",
                    "limit": "line count optional",
                }
            },
            "git_status": {"args": {"limit": "integer optional"}},
            "git_diff": {
                "args": {
                    "path": "optional file path inside workspace",
                    "mode": "stat | patch; default stat; use patch only for a specific file",
                    "max_chars": "integer optional",
                }
            },
            "git_log": {"args": {"limit": "integer optional"}},
            "repo_map": {
                "args": {
                    "path": "string default .",
                    "max_depth": "integer optional",
                    "limit": "integer optional",
                }
            },
            "read_package_metadata": {
                "args": {
                    "path": "string default .",
                    "limit": "integer optional",
                }
            },
            "read_test_structure": {
                "args": {
                    "path": "string default .",
                    "limit": "integer optional",
                }
            },
            "python_symbol_index": {
                "args": {
                    "path": "string default .",
                    "file_glob": "glob string optional; defaults to *.py",
                    "limit": "integer optional",
                }
            },
            "read_python_symbol": {
                "args": {
                    "path": "file or directory path default .",
                    "qualified_name": "class/function qualified name optional",
                    "name": "class/function name optional",
                    "kind": "class | function | method optional",
                }
            },
        },
        "response_schema": {
            "done": "boolean; true when no more tool calls are needed",
            "summary": "brief internal summary of what you are trying to learn",
            "sufficiency_check": {
                "sufficient_evidence": "boolean; true only when source-backed evidence is enough",
                "can_answer_user_question": "boolean; true if Spice can answer now with caveats",
                "remaining_gaps": ["specific missing evidence or unanswered questions"],
                "reason": "short explanation of why the evidence is sufficient or still partial",
            },
            "investigation_state": {
                "investigation_summary": "rolling summary of what is known and why it matters",
                "files_read": ["file paths already read"],
                "key_findings": ["source-backed findings gathered so far"],
                "open_questions": ["remaining unknowns"],
                "next_leads": ["specific files, symbols, or searches to try next"],
                "source_map": [
                    {
                        "source": "file/search/symbol reference",
                        "supports": "short explanation of what this source supports",
                    }
                ],
            },
            "tool_calls": [
                {
                    "tool": "file_index | search | read_file | git_status | git_diff | git_log | repo_map | read_package_metadata | read_test_structure | python_symbol_index | read_python_symbol",
                    "args": {},
                }
            ],
        },
        "investigation_strategy": {
            "orient": [
                "Start with repo_map to understand directory structure.",
                "Use read_package_metadata to identify project type, dependencies, and entry points.",
                "Use read_test_structure to understand test coverage and validation surfaces.",
                "Use git_status to understand local changes before interpreting implementation state.",
            ],
            "locate": [
                "Use search to find relevant text references once you know the likely terms.",
                "Use python_symbol_index to find Python classes, functions, imports, and module relationships.",
            ],
            "inspect": [
                "Use read_file for known relevant small files or specific line ranges.",
                "Use read_python_symbol when a class/function/method body is the relevant evidence.",
            ],
            "verify": [
                "Cross-check implementation claims against tests, docs, imports, callers, and config files.",
                "Prefer source-backed findings over broad inferences.",
                "Record open questions when evidence is incomplete.",
            ],
            "avoid": [
                "Do not read random large files.",
                "Do not keep broad-searching when budget pressure is medium or high.",
                "Do not treat directory names alone as implementation proof.",
            ],
        },
        "rules": [
            (
                "Request at most "
                f"{budget_state['max_tool_calls_per_round']} read-only tool calls in this round."
            ),
            (
                "Respect the remaining budget: "
                f"{budget_state['remaining_tool_calls']} tool calls, "
                f"{budget_state['remaining_files']} files, "
                f"{budget_state['remaining_chars']} chars, "
                f"{budget_state['blocked_call_budget']['remaining']} blocked calls."
            ),
            (
                "Budget pressure is "
                f"{budget_state['budget_pressure']}; "
                f"{_budget_pressure_guidance(budget_state)}"
            ),
            "At 70% budget pressure, converge on key sources and avoid broad exploration.",
            "At 90% budget pressure, stop broad search and generate remaining_gaps.",
            (
                "At 100% budget pressure, runtime will stop the loop; normal budget exhaustion "
                "marks budget_exhausted, while blocked-call exhaustion marks blocked or partial."
            ),
            "Follow the investigation_strategy: first orient, then locate, then inspect, then verify.",
            "Prefer search before read_file unless you already know the path.",
            "Use read_file only for small, relevant files or specific line ranges.",
            "Use git_diff mode=patch only for a specific file; otherwise prefer mode=stat.",
            "Avoid reading random large files.",
            "Prefer source-backed findings and cite support in investigation_state.source_map.",
            "Do not request any write, code edit, terminal, install, delete, move, or test-run tool.",
            "If existing observations are enough, return done=true and tool_calls=[].",
            "Update investigation_state every round; keep it compact and source-backed.",
            "Update sufficiency_check every round after considering current observations.",
            (
                "Set sufficient_evidence=true only when key implementation files, symbols, "
                "docs, tests, or other source-backed evidence are enough to answer the query."
            ),
            (
                "If you can answer only with caveats, set can_answer_user_question=true, "
                "sufficient_evidence=false, and list remaining_gaps."
            ),
            "Do not copy full file contents into investigation_state; use source references and short findings.",
        ],
        "initial_context": dict(initial_context or {}),
        "investigation_state": _compact_investigation_state(investigation_state),
        "recent_observations": [
            _compact_tool_call_for_prompt(call)
            for call in tool_calls[-6:]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _workspace_loop_budget_state(
    *,
    inspector: WorkspaceInspector,
    limits: ControlledWorkspacePerceptionLimits,
    tool_calls: list[WorkspaceToolCallRecord],
) -> dict[str, Any]:
    summary = inspector.summarize_workspace()
    inspector_budget = _mapping(summary.get("budget"))
    files_read = summary.get("files_read")
    files_read_payloads = files_read if isinstance(files_read, list) else []
    file_paths = [
        str(item.get("path") or "")
        for item in files_read_payloads
        if isinstance(item, Mapping) and str(item.get("path") or "").strip()
    ]
    normal_calls_used = _normal_tool_call_count(tool_calls)
    blocked_calls_used = _blocked_tool_call_count(tool_calls)
    files_read_count = _int(inspector_budget.get("files_read_count"), 0)
    chars_used = _int(inspector_budget.get("chars_used"), 0)
    remaining_chars = _int(
        inspector_budget.get("chars_remaining"),
        max(limits.total_char_budget - chars_used, 0),
    )
    pressure = _budget_pressure(
        ratios=[
            _ratio(normal_calls_used, limits.max_tool_calls),
            _ratio(blocked_calls_used, limits.max_blocked_tool_calls),
            _ratio(files_read_count, limits.max_files_read),
            _ratio(chars_used, limits.total_char_budget),
        ]
    )
    return {
        "depth": limits.depth,
        "max_tool_calls_per_round": limits.max_tool_calls_per_round,
        "remaining_tool_calls": max(limits.max_tool_calls - normal_calls_used, 0),
        "remaining_files": max(limits.max_files_read - files_read_count, 0),
        "remaining_chars": remaining_chars,
        "budget_pressure": pressure,
        "tool_calls_used": normal_calls_used,
        "normal_tool_calls_used": normal_calls_used,
        "tool_calls_recorded": len(tool_calls),
        "tool_calls_executed": len([call for call in tool_calls if call.status == "executed"]),
        "tool_calls_blocked": blocked_calls_used,
        "files_read_count": files_read_count,
        "chars_used": chars_used,
        "files_already_read": file_paths,
        "blocked_call_budget": {
            "max": limits.max_blocked_tool_calls,
            "used": blocked_calls_used,
            "remaining": max(limits.max_blocked_tool_calls - blocked_calls_used, 0),
            "max_per_round": limits.max_blocked_tool_calls_per_round,
        },
    }


def _normal_tool_call_count(tool_calls: list[WorkspaceToolCallRecord]) -> int:
    return len([call for call in tool_calls if call.status != "blocked"])


def _blocked_tool_call_count(tool_calls: list[WorkspaceToolCallRecord]) -> int:
    return len([call for call in tool_calls if call.status == "blocked"])


def _loop_has_source_backed_evidence(tool_calls: list[WorkspaceToolCallRecord]) -> bool:
    return any(
        call.status == "executed"
        and call.tool in SOURCE_BACKED_WORKSPACE_TOOLS
        and bool(_mapping(call.result).get("ok", True))
        for call in tool_calls
    )


def _hard_repo_evidence_required(initial_context: Mapping[str, Any] | None) -> bool:
    payload = _mapping(initial_context)
    evidence_payloads = [
        _mapping(payload.get("evidence_requirement")),
        _mapping(_mapping(payload.get("evidence_context")).get("requirements")),
    ]
    route_policy = _mapping(payload.get("route_merge_policy"))
    if route_policy:
        evidence_payloads.append(_mapping(route_policy.get("evidence_requirement")))

    for evidence in evidence_payloads:
        domain = str(evidence.get("evidence_domain") or "").strip().lower()
        if _truthy(evidence.get("requires_evidence")) and domain in {"repo", "mixed"}:
            return True

    forced_by = _strings(route_policy.get("forced_by"))
    if "repo_evidence_requirement" in forced_by and bool(route_policy.get("needs_workspace_context")):
        return True
    return False


def _minimum_fallback_missing_sufficiency_check(reason: str) -> dict[str, Any]:
    return {
        "sufficient_evidence": False,
        "can_answer_user_question": False,
        "remaining_gaps": [
            "source-backed repo evidence is still missing after the minimum read-only fallback"
        ],
        "reason": reason,
    }


def _update_minimum_fallback_state(
    state: dict[str, Any],
    *,
    records: list[WorkspaceToolCallRecord],
    source_backed_evidence: bool,
) -> None:
    state["triggered"] = True
    state["tool_calls_executed"] = _int(state.get("tool_calls_executed"), 0) + len(
        [record for record in records if record.status == "executed"]
    )
    state["tool_calls_blocked"] = _int(state.get("tool_calls_blocked"), 0) + len(
        [record for record in records if record.status == "blocked"]
    )
    state["source_backed_evidence"] = bool(source_backed_evidence)


def _empty_sufficiency_check() -> dict[str, Any]:
    return {
        "sufficient_evidence": False,
        "can_answer_user_question": False,
        "remaining_gaps": [],
        "reason": "",
    }


def _merge_sufficiency_check(
    *,
    current: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    incoming = payload.get("sufficiency_check")
    if not isinstance(incoming, Mapping):
        incoming = payload.get("evidence_sufficiency")
    incoming_payload = _mapping(incoming)
    if not incoming_payload and not any(
        key in payload
        for key in (
            "sufficient_evidence",
            "can_answer_user_question",
            "remaining_gaps",
            "sufficiency_reason",
        )
    ):
        return _compact_sufficiency_check(current)

    merged = _compact_sufficiency_check(current)
    if "sufficient_evidence" in incoming_payload or "sufficient_evidence" in payload:
        merged["sufficient_evidence"] = _truthy(
            incoming_payload.get("sufficient_evidence", payload.get("sufficient_evidence"))
        )
    if "can_answer_user_question" in incoming_payload or "can_answer_user_question" in payload:
        merged["can_answer_user_question"] = _truthy(
            incoming_payload.get(
                "can_answer_user_question",
                payload.get("can_answer_user_question"),
            )
        )
    gaps = _strings(
        incoming_payload.get("remaining_gaps", payload.get("remaining_gaps")),
        limit=20,
        max_chars=300,
    )
    if gaps or "remaining_gaps" in incoming_payload or "remaining_gaps" in payload:
        merged["remaining_gaps"] = gaps
    reason = str(
        incoming_payload.get("reason")
        or incoming_payload.get("sufficiency_reason")
        or payload.get("sufficiency_reason")
        or payload.get("reason")
        or ""
    ).strip()
    if reason:
        merged["reason"] = _shorten(reason, 500)
    return merged


def _compact_sufficiency_check(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    payload = _mapping(value)
    check = _empty_sufficiency_check()
    check["sufficient_evidence"] = _truthy(payload.get("sufficient_evidence"))
    check["can_answer_user_question"] = _truthy(payload.get("can_answer_user_question"))
    check["remaining_gaps"] = _strings(payload.get("remaining_gaps"), limit=20, max_chars=300)
    check["reason"] = _shorten(str(payload.get("reason") or "").strip(), 500)
    return check


def _sufficiency_is_complete(check: Mapping[str, Any]) -> bool:
    return _truthy(check.get("sufficient_evidence")) and _truthy(
        check.get("can_answer_user_question")
    )


def _exploration_status_from_state(
    *,
    sufficiency_check: Mapping[str, Any],
    budget_exhausted: bool,
) -> str:
    if _sufficiency_is_complete(sufficiency_check):
        return EXPLORATION_STATUS_COMPLETE
    if budget_exhausted:
        return EXPLORATION_STATUS_BUDGET_EXHAUSTED
    return EXPLORATION_STATUS_PARTIAL


def _loop_budget_exhausted(
    *,
    inspector: WorkspaceInspector,
    limits: ControlledWorkspacePerceptionLimits,
    tool_calls: list[WorkspaceToolCallRecord],
) -> bool:
    budget_state = _workspace_loop_budget_state(
        inspector=inspector,
        limits=limits,
        tool_calls=tool_calls,
    )
    return _normal_loop_budget_exhausted(budget_state)


def _normal_loop_budget_exhausted(budget_state: Mapping[str, Any]) -> bool:
    return any(
        _int(budget_state.get(key), 1) <= 0
        for key in ("remaining_tool_calls", "remaining_files", "remaining_chars")
    )


def _blocked_loop_budget_exhausted(budget_state: Mapping[str, Any]) -> bool:
    return _int(
        _mapping(budget_state.get("blocked_call_budget")).get("remaining"),
        1,
    ) <= 0


def _blocked_exploration_status(tool_calls: list[WorkspaceToolCallRecord]) -> str:
    if _normal_tool_call_count(tool_calls) <= 0:
        return EXPLORATION_STATUS_BLOCKED
    return EXPLORATION_STATUS_PARTIAL


def _budget_pressure_guidance(budget_state: Mapping[str, Any]) -> str:
    pressure = str(budget_state.get("budget_pressure") or "low")
    if pressure == "exhausted":
        return (
            "Budget is exhausted; do not request more tools and return remaining_gaps. "
            "Runtime will classify the stop reason from normal vs blocked-call budget."
        )
    if pressure == "high":
        return "Stop broad search, request only critical source checks, and generate remaining_gaps."
    if pressure == "medium":
        return "Converge on key sources; prefer inspect/verify over broad exploration."
    return "Continue the mature investigation strategy while staying source-backed."


def _record_budget_pressure_event(
    *,
    events: list[dict[str, Any]],
    budget_state: Mapping[str, Any],
    round_index: int,
    stage: str,
    last_pressure: str,
) -> str:
    pressure = str(budget_state.get("budget_pressure") or "low")
    if pressure == last_pressure:
        return last_pressure
    if pressure == "low":
        return pressure
    events.append(
        {
            "round_index": round_index,
            "stage": stage,
            "budget_pressure": pressure,
            "guidance": _budget_pressure_guidance(budget_state),
            "remaining_tool_calls": _int(budget_state.get("remaining_tool_calls"), 0),
            "remaining_blocked_tool_calls": _int(
                _mapping(budget_state.get("blocked_call_budget")).get("remaining"),
                0,
            ),
            "remaining_files": _int(budget_state.get("remaining_files"), 0),
            "remaining_chars": _int(budget_state.get("remaining_chars"), 0),
            "tool_calls_used": _int(budget_state.get("tool_calls_used"), 0),
            "tool_calls_blocked": _int(budget_state.get("tool_calls_blocked"), 0),
            "files_read_count": _int(budget_state.get("files_read_count"), 0),
            "chars_used": _int(budget_state.get("chars_used"), 0),
        }
    )
    return pressure


def _budget_pressure(*, ratios: list[float]) -> str:
    pressure = max(ratios or [0.0])
    if pressure >= 1.0:
        return "exhausted"
    if pressure >= 0.9:
        return "high"
    if pressure >= 0.7:
        return "medium"
    return "low"


def _ratio(value: int, limit: int) -> float:
    if limit <= 0:
        return 1.0
    return max(value, 0) / limit


def _empty_investigation_state() -> dict[str, Any]:
    return {
        "investigation_summary": "",
        "files_read": [],
        "key_findings": [],
        "open_questions": [],
        "next_leads": [],
        "source_map": [],
    }


def _merge_investigation_state(
    *,
    current: Mapping[str, Any],
    payload: Mapping[str, Any],
    fallback_summary: str = "",
) -> dict[str, Any]:
    state = _compact_investigation_state(current)
    update = payload.get("investigation_state")
    if not isinstance(update, Mapping):
        update = payload.get("investigation_state_update")
    update_payload = _mapping(update)

    summary = str(
        update_payload.get("investigation_summary")
        or update_payload.get("summary")
        or payload.get("investigation_summary")
        or fallback_summary
        or state.get("investigation_summary")
        or ""
    ).strip()
    state["investigation_summary"] = _shorten(summary, 1400)
    for key in ("files_read", "key_findings", "open_questions", "next_leads"):
        state[key] = _merge_strings(
            _strings(state.get(key)),
            _strings(update_payload.get(key) if key in update_payload else payload.get(key)),
            limit=80 if key == "files_read" else 40,
            max_chars=240 if key == "files_read" else 500,
        )
    state["source_map"] = _merge_source_map(
        _source_map_entries(state.get("source_map")),
        _source_map_entries(
            update_payload.get("source_map")
            if "source_map" in update_payload
            else payload.get("source_map")
        ),
    )
    return state


def _refresh_investigation_state_from_workspace(
    state: Mapping[str, Any],
    *,
    inspector: WorkspaceInspector,
    tool_calls: list[WorkspaceToolCallRecord],
) -> dict[str, Any]:
    refreshed = _compact_investigation_state(state)
    workspace = inspector.summarize_workspace()
    file_paths = [
        str(item.get("path") or "")
        for item in workspace.get("files_read", [])
        if isinstance(item, Mapping) and str(item.get("path") or "").strip()
    ]
    refreshed["files_read"] = _merge_strings(
        _strings(refreshed.get("files_read")),
        file_paths,
        limit=80,
        max_chars=240,
    )
    source_entries: list[dict[str, Any]] = []
    for call in tool_calls:
        if call.status != "executed":
            continue
        result = _mapping(call.result)
        path = str(result.get("path") or "")
        if not path and call.tool in {"search", "repo_map", "python_symbol_index"}:
            path = str(call.args.get("path") or ".")
        if not path:
            continue
        source_entries.append(
            {
                "source": path,
                "tool": call.tool,
                "supports": _shorten(_source_support_text(call), 220),
            }
        )
    refreshed["source_map"] = _merge_source_map(
        _source_map_entries(refreshed.get("source_map")),
        source_entries,
    )
    return refreshed


def _compact_investigation_state(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    payload = _mapping(value)
    state = _empty_investigation_state()
    state["investigation_summary"] = _shorten(
        str(payload.get("investigation_summary") or payload.get("summary") or ""),
        1400,
    )
    state["files_read"] = _strings(payload.get("files_read"), limit=80, max_chars=240)
    state["key_findings"] = _strings(payload.get("key_findings"), limit=40, max_chars=500)
    state["open_questions"] = _strings(payload.get("open_questions"), limit=40, max_chars=500)
    state["next_leads"] = _strings(payload.get("next_leads"), limit=40, max_chars=500)
    state["source_map"] = _source_map_entries(payload.get("source_map"), limit=80)
    return state


def _source_support_text(call: WorkspaceToolCallRecord) -> str:
    if call.tool == "read_file":
        return "file content was read"
    if call.tool == "read_python_symbol":
        return "Python symbol body was read"
    if call.tool == "search":
        return "search match source"
    if call.tool == "python_symbol_index":
        return "Python symbol index source"
    return f"{call.tool} source"


def _merge_strings(
    existing: list[str],
    incoming: list[str],
    *,
    limit: int,
    max_chars: int,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        text = _shorten(str(item or "").strip(), max_chars)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _strings(value: Any, *, limit: int = 80, max_chars: int = 500) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    elif value:
        raw_items = [value]
    else:
        raw_items = []
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _shorten(str(item or "").strip(), max_chars)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _merge_source_map(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        source = str(item.get("source") or item.get("path") or "").strip()
        supports = str(item.get("supports") or item.get("finding") or "").strip()
        key = f"{source}\n{supports}"
        if not source or key in seen:
            continue
        seen.add(key)
        compact = {
            "source": _shorten(source, 240),
            "supports": _shorten(supports, 300),
        }
        tool = str(item.get("tool") or "").strip()
        if tool:
            compact["tool"] = _shorten(tool, 80)
        result.append(compact)
        if len(result) >= limit:
            break
    return result


def _source_map_entries(value: Any, *, limit: int = 80) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            source = str(item.get("source") or item.get("path") or "").strip()
            supports = str(item.get("supports") or item.get("finding") or "").strip()
            if not source:
                continue
            compact = {
                "source": _shorten(source, 240),
                "supports": _shorten(supports, 300),
            }
            tool = str(item.get("tool") or "").strip()
            if tool:
                compact["tool"] = _shorten(tool, 80)
            result.append(compact)
        elif str(item or "").strip():
            result.append({"source": _shorten(str(item).strip(), 240), "supports": ""})
        if len(result) >= limit:
            break
    return result


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _compact_tool_call_for_prompt(call: WorkspaceToolCallRecord) -> dict[str, Any]:
    result = dict(call.result)
    if "content" in result:
        content = str(result.get("content") or "")
        result["content"] = content[:1200]
        result["content_truncated_for_prompt"] = len(content) > 1200
    if "matches" in result and isinstance(result["matches"], list):
        result["matches"] = result["matches"][:10]
    if "entries" in result and isinstance(result["entries"], list):
        result["entries"] = result["entries"][:30]
    if "test_files" in result and isinstance(result["test_files"], list):
        result["test_files"] = result["test_files"][:30]
    if "files" in result and isinstance(result["files"], list):
        result["files"] = result["files"][:20]
    if "symbols" in result and isinstance(result["symbols"], list):
        result["symbols"] = result["symbols"][:40]
    if "imports" in result and isinstance(result["imports"], list):
        result["imports"] = result["imports"][:40]
    if "modules" in result and isinstance(result["modules"], list):
        result["modules"] = result["modules"][:30]
    return {
        "call_id": call.call_id,
        "tool": call.tool,
        "args": dict(call.args),
        "status": call.status,
        "reason": call.reason,
        "result": result,
    }


def _execute_tool_call_batch(
    *,
    inspector: WorkspaceInspector,
    requests: list[Mapping[str, Any]],
    round_index: int,
    tool_calls: list[WorkspaceToolCallRecord],
    limits: ControlledWorkspacePerceptionLimits,
    round_normal_tool_call_count: int = 0,
    round_blocked_tool_call_count: int = 0,
) -> tuple[list[WorkspaceToolCallRecord], int, int, bool]:
    batch_records: list[WorkspaceToolCallRecord] = []
    blocked_budget_exhausted = False
    for request in requests:
        call_id = f"tool.{len(tool_calls) + 1}"
        record = _execute_or_block_tool_call(
            inspector=inspector,
            request=request,
            call_id=call_id,
            round_index=round_index,
            round_normal_tool_call_count=round_normal_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
            current_normal_tool_call_count=_normal_tool_call_count(tool_calls),
            current_blocked_tool_call_count=_blocked_tool_call_count(tool_calls),
            limits=limits,
        )
        tool_calls.append(record)
        batch_records.append(record)
        if record.status == "blocked":
            round_blocked_tool_call_count += 1
            if (
                record.reason
                in {
                    "max_blocked_tool_calls_exceeded",
                    "max_blocked_tool_calls_per_round_exceeded",
                }
                or _blocked_tool_call_count(tool_calls) >= limits.max_blocked_tool_calls
                or round_blocked_tool_call_count >= limits.max_blocked_tool_calls_per_round
            ):
                blocked_budget_exhausted = True
                break
        else:
            round_normal_tool_call_count += 1
    return (
        batch_records,
        round_normal_tool_call_count,
        round_blocked_tool_call_count,
        blocked_budget_exhausted,
    )


def _run_minimum_investigation_fallback(
    *,
    inspector: WorkspaceInspector,
    query: str,
    round_index: int,
    tool_calls: list[WorkspaceToolCallRecord],
    limits: ControlledWorkspacePerceptionLimits,
    reason: str,
    round_normal_tool_call_count: int = 0,
    round_blocked_tool_call_count: int = 0,
) -> tuple[list[WorkspaceToolCallRecord], list[dict[str, Any]]]:
    available_slots = min(
        max(limits.max_tool_calls_per_round - round_normal_tool_call_count, 0),
        max(limits.max_tool_calls - _normal_tool_call_count(tool_calls), 0),
    )
    reserved_read_slots = _minimum_fallback_reserved_read_slots(
        limits,
        available_slots=available_slots,
    )
    requests = _minimum_fallback_orientation_requests(
        query,
        available_slots=available_slots,
        reserved_read_slots=reserved_read_slots,
    )
    records, round_normal_tool_call_count, round_blocked_tool_call_count, blocked_exhausted = (
        _execute_tool_call_batch(
            inspector=inspector,
            requests=requests,
            round_index=round_index,
            tool_calls=tool_calls,
            limits=limits,
            round_normal_tool_call_count=round_normal_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
        )
    )
    if blocked_exhausted:
        _mark_minimum_fallback_reason(records, reason)
        return records, requests

    read_requests = _minimum_fallback_read_requests(records, limit=reserved_read_slots)
    if read_requests:
        read_records, _, _, _ = _execute_tool_call_batch(
            inspector=inspector,
            requests=read_requests,
            round_index=round_index,
            tool_calls=tool_calls,
            limits=limits,
            round_normal_tool_call_count=round_normal_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
        )
        records.extend(read_records)
        requests.extend(read_requests)

    _mark_minimum_fallback_reason(records, reason)
    return records, requests


def _mark_minimum_fallback_reason(
    records: list[WorkspaceToolCallRecord],
    reason: str,
) -> None:
    for record in records:
        record.result.setdefault("minimum_investigation_fallback", True)
        record.result.setdefault("fallback_reason", reason)


def _minimum_fallback_orientation_requests(
    query: str,
    *,
    available_slots: int,
    reserved_read_slots: int,
) -> list[dict[str, Any]]:
    orientation_requests: list[dict[str, Any]] = [
        {"tool": "repo_map", "args": {"path": ".", "max_depth": 3, "limit": 120}},
        {"tool": "read_package_metadata", "args": {"path": ".", "limit": 20}},
        {"tool": "read_test_structure", "args": {"path": ".", "limit": 80}},
        {"tool": "git_status", "args": {"limit": 80}},
    ]
    orientation_budget = max(available_slots - reserved_read_slots, 0)
    requests = orientation_requests[:orientation_budget]
    max_searches = min(
        6,
        max(0, orientation_budget - len(requests)),
    )
    for keyword in _minimum_fallback_keywords(query)[:max_searches]:
        requests.append(
            {
                "tool": "search",
                "args": {
                    "pattern": re.escape(keyword),
                    "path": ".",
                    "limit": 12,
                },
            }
        )
    return requests


def _minimum_fallback_reserved_read_slots(
    limits: ControlledWorkspacePerceptionLimits,
    *,
    available_slots: int,
) -> int:
    if available_slots <= 0:
        return 0
    max_reserved = 4
    if str(limits.depth or "").strip().lower() in {"deep", "native"}:
        max_reserved = 6
    return min(max_reserved, max(1, available_slots // 3))


def _minimum_fallback_keywords(query: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", query)
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "repo",
        "code",
        "current",
        "implementation",
        "local",
        "read",
        "inspect",
        "which",
        "why",
        "based",
        "actual",
        "当前",
        "本地",
        "代码",
        "实现",
        "读取",
        "判断",
        "为什么",
        "说明",
        "证据",
    }
    result: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        normalized = token.strip("-_").lower()
        if not normalized or normalized in stopwords or normalized in seen:
            continue
        seen.add(normalized)
        result.append(token.strip())
        if len(result) >= 8:
            break
    return result


def _minimum_fallback_read_requests(
    records: list[WorkspaceToolCallRecord],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    paths: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        normalized = str(path or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        paths.append(normalized)

    for record in records:
        if record.status != "executed":
            continue
        result = _mapping(record.result)
        if record.tool == "search":
            for match in _mappings_from_any(result.get("matches")):
                add(str(match.get("path") or ""))
        elif record.tool == "read_package_metadata":
            for item in _mappings_from_any(result.get("files_read")):
                add(str(item.get("path") or ""))
            for item in _mappings_from_any(result.get("files")):
                add(str(item.get("path") or item.get("name") or ""))
        elif record.tool == "read_test_structure":
            for item in _mappings_from_any(result.get("test_files")):
                add(str(item.get("path") or ""))

    return [
        {"tool": "read_file", "args": {"path": path, "offset": 1, "limit": 180}}
        for path in paths[:limit]
    ]


def _mappings_from_any(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _execute_or_block_tool_call(
    *,
    inspector: WorkspaceInspector,
    request: Mapping[str, Any],
    call_id: str,
    round_index: int,
    round_normal_tool_call_count: int,
    round_blocked_tool_call_count: int,
    current_normal_tool_call_count: int,
    current_blocked_tool_call_count: int,
    limits: ControlledWorkspacePerceptionLimits,
) -> WorkspaceToolCallRecord:
    tool = str(request.get("tool") or request.get("name") or "").strip()
    args = _mapping(request.get("args"))
    if current_normal_tool_call_count >= limits.max_tool_calls:
        return _blocked_with_budget(
            call_id,
            round_index,
            tool,
            args,
            "max_tool_calls_exceeded",
            current_blocked_tool_call_count=current_blocked_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
            limits=limits,
        )
    if round_normal_tool_call_count >= limits.max_tool_calls_per_round:
        return _blocked_with_budget(
            call_id,
            round_index,
            tool,
            args,
            "max_tool_calls_per_round_exceeded",
            current_blocked_tool_call_count=current_blocked_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
            limits=limits,
        )
    if tool not in ALLOWED_WORKSPACE_TOOLS:
        return _blocked_with_budget(
            call_id,
            round_index,
            tool,
            args,
            "tool_not_allowed",
            current_blocked_tool_call_count=current_blocked_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
            limits=limits,
        )

    sanitized = _sanitize_args(tool, args)
    if sanitized is None:
        return _blocked_with_budget(
            call_id,
            round_index,
            tool,
            args,
            "invalid_tool_args",
            current_blocked_tool_call_count=current_blocked_tool_call_count,
            round_blocked_tool_call_count=round_blocked_tool_call_count,
            limits=limits,
        )

    result = _execute_tool(inspector, tool, sanitized)
    result_payload = result.to_payload()
    if not bool(result_payload.get("ok", False)):
        reason = str(result_payload.get("reason") or "tool_failed")
        status = "blocked" if _is_guardrail_reason(reason) else "failed"
        if status == "blocked":
            return _blocked_with_budget(
                call_id,
                round_index,
                tool,
                sanitized,
                reason,
                current_blocked_tool_call_count=current_blocked_tool_call_count,
                round_blocked_tool_call_count=round_blocked_tool_call_count,
                limits=limits,
                result=result_payload,
            )
        return WorkspaceToolCallRecord(
            call_id=call_id,
            round_index=round_index,
            tool=tool,
            args=sanitized,
            status=status,
            reason=reason,
            result=result_payload,
        )

    return WorkspaceToolCallRecord(
        call_id=call_id,
        round_index=round_index,
        tool=tool,
        args=sanitized,
        status="executed",
        result=result_payload,
    )


def _blocked_with_budget(
    call_id: str,
    round_index: int,
    tool: str,
    args: Mapping[str, Any],
    reason: str,
    *,
    current_blocked_tool_call_count: int,
    round_blocked_tool_call_count: int,
    limits: ControlledWorkspacePerceptionLimits,
    result: Mapping[str, Any] | None = None,
) -> WorkspaceToolCallRecord:
    if current_blocked_tool_call_count >= limits.max_blocked_tool_calls:
        return _blocked(
            call_id,
            round_index,
            tool,
            args,
            "max_blocked_tool_calls_exceeded",
            result={
                "blocked_reason": reason,
                **(_mapping(result) if result else {}),
            },
        )
    if round_blocked_tool_call_count >= limits.max_blocked_tool_calls_per_round:
        return _blocked(
            call_id,
            round_index,
            tool,
            args,
            "max_blocked_tool_calls_per_round_exceeded",
            result={
                "blocked_reason": reason,
                **(_mapping(result) if result else {}),
            },
        )
    return _blocked(call_id, round_index, tool, args, reason, result=result)


def _execute_tool(
    inspector: WorkspaceInspector,
    tool: str,
    args: Mapping[str, Any],
) -> WorkspaceFileIndexResult | WorkspaceSearchResult | WorkspaceReadResult | WorkspaceGitStatusResult | WorkspaceGitDiffResult | WorkspaceGitLogResult | WorkspaceRepoMapResult | WorkspacePackageMetadataResult | WorkspaceTestStructureResult | WorkspacePythonSymbolIndexResult | WorkspacePythonSymbolReadResult:
    if tool == "file_index":
        return inspector.file_index(
            path=str(args.get("path") or "."),
            limit=_optional_int(args.get("limit")),
        )
    if tool == "search":
        return inspector.search(
            str(args.get("pattern") or ""),
            file_glob=str(args.get("file_glob") or "") or None,
            limit=_optional_int(args.get("limit")) or 50,
            path=str(args.get("path") or "."),
        )
    if tool == "read_file":
        return inspector.read_file(
            str(args.get("path") or ""),
            offset=_optional_int(args.get("offset")) or 1,
            limit=_optional_int(args.get("limit")) or 300,
        )
    if tool == "git_status":
        return inspector.git_status(limit=_optional_int(args.get("limit")) or 100)
    if tool == "git_diff":
        return inspector.git_diff(
            path=str(args.get("path") or ""),
            mode=str(args.get("mode") or "stat"),
            max_chars=_optional_int(args.get("max_chars")),
        )
    if tool == "git_log":
        return inspector.git_log(limit=_optional_int(args.get("limit")) or 10)
    if tool == "repo_map":
        return inspector.repo_map(
            path=str(args.get("path") or "."),
            max_depth=_optional_int(args.get("max_depth")) or 3,
            limit=_optional_int(args.get("limit")),
        )
    if tool == "read_package_metadata":
        return inspector.read_package_metadata(
            path=str(args.get("path") or "."),
            limit=_optional_int(args.get("limit")),
        )
    if tool == "read_test_structure":
        return inspector.read_test_structure(
            path=str(args.get("path") or "."),
            limit=_optional_int(args.get("limit")),
        )
    if tool == "python_symbol_index":
        return inspector.python_symbol_index(
            path=str(args.get("path") or "."),
            file_glob=str(args.get("file_glob") or "") or None,
            limit=_optional_int(args.get("limit")),
        )
    return inspector.read_python_symbol(
        path=str(args.get("path") or "."),
        qualified_name=str(args.get("qualified_name") or ""),
        name=str(args.get("name") or ""),
        kind=str(args.get("kind") or ""),
    )


def _sanitize_args(tool: str, args: Mapping[str, Any]) -> dict[str, Any] | None:
    allowed_by_tool = {
        "file_index": {"path", "limit"},
        "search": {"pattern", "file_glob", "path", "limit"},
        "read_file": {"path", "offset", "limit"},
        "git_status": {"limit"},
        "git_diff": {"path", "mode", "max_chars"},
        "git_log": {"limit"},
        "repo_map": {"path", "max_depth", "limit"},
        "read_package_metadata": {"path", "limit"},
        "read_test_structure": {"path", "limit"},
        "python_symbol_index": {"path", "file_glob", "limit"},
        "read_python_symbol": {"path", "qualified_name", "name", "kind"},
    }
    allowed = allowed_by_tool.get(tool)
    if allowed is None:
        return None
    sanitized = {key: value for key, value in args.items() if key in allowed}
    if tool == "search" and not str(sanitized.get("pattern") or "").strip():
        return None
    if tool == "read_file" and not str(sanitized.get("path") or "").strip():
        return None
    if tool == "git_diff":
        mode = str(sanitized.get("mode") or "stat").strip().lower()
        sanitized["mode"] = "patch" if mode == "patch" else "stat"
        if sanitized["mode"] == "patch" and not str(sanitized.get("path") or "").strip():
            sanitized["mode"] = "stat"
    if tool == "read_python_symbol" and not (
        str(sanitized.get("qualified_name") or "").strip()
        or str(sanitized.get("name") or "").strip()
    ):
        return None
    return sanitized


def _blocked(
    call_id: str,
    round_index: int,
    tool: str,
    args: Mapping[str, Any],
    reason: str,
    *,
    result: Mapping[str, Any] | None = None,
) -> WorkspaceToolCallRecord:
    return WorkspaceToolCallRecord(
        call_id=call_id,
        round_index=round_index,
        tool=tool,
        args=dict(args),
        status="blocked",
        reason=reason,
        result=_mapping(result),
    )


def _is_guardrail_reason(reason: str) -> bool:
    if reason in GUARDRAIL_BLOCK_REASONS:
        return True
    return "escapes workspace root" in reason


def _apply_loop_budget_to_inspector(
    inspector: WorkspaceInspector,
    limits: ControlledWorkspacePerceptionLimits,
) -> None:
    current = inspector.limits
    inspector.limits = WorkspaceInspectorLimits(
        max_files_read=limits.max_files_read,
        max_chars_per_file=limits.max_chars_per_file,
        total_char_budget=limits.total_char_budget,
        max_index_entries=current.max_index_entries,
        max_search_results=current.max_search_results,
        max_search_files_scanned=current.max_search_files_scanned,
        max_file_size_bytes=current.max_file_size_bytes,
        max_line_limit=current.max_line_limit,
        max_git_diff_chars=current.max_git_diff_chars,
            max_git_log_entries=current.max_git_log_entries,
            max_repo_map_entries=current.max_repo_map_entries,
            max_package_metadata_files=current.max_package_metadata_files,
            max_test_structure_entries=current.max_test_structure_entries,
            max_python_symbol_entries=current.max_python_symbol_entries,
            max_python_import_entries=current.max_python_import_entries,
        )


def _round_batch_payload(
    *,
    round_index: int,
    requested_tool_calls: list[Mapping[str, Any]],
    records: list[WorkspaceToolCallRecord],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "requested_tool_calls": [
            _requested_tool_call_payload(request) for request in requested_tool_calls
        ],
        "executed_tool_calls": [
            record.to_payload() for record in records if record.status == "executed"
        ],
        "blocked_tool_calls": [
            record.to_payload() for record in records if record.status == "blocked"
        ],
        "failed_tool_calls": [
            record.to_payload() for record in records if record.status == "failed"
        ],
    }


def _requested_tool_call_payload(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(request)
    tool = str(payload.get("tool") or payload.get("name") or "").strip()
    return {
        "tool": tool,
        "args": _mapping(payload.get("args")),
    }


def _tool_call_requests(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("tool_calls")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _parse_loop_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(stripped)
        payload = json.loads(extracted) if extracted else None
    return payload if isinstance(payload, dict) else {}


def _int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "done"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
