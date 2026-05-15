from __future__ import annotations

import json
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from spice.decision.compare import render_compare_text
from spice.runtime.approval_flow import (
    approve_approval,
    list_approvals,
    load_approval,
    reject_approval,
    render_approval_details,
    render_approval_list,
    render_approval_resolution,
)
from spice.runtime.command_router import route_slash_command, split_slash_command
from spice.runtime.context_debug import (
    compile_sources_debug_payload,
    compile_workspace_decision_context_payload,
    compile_workspace_debug_payload,
    render_decision_context_text,
    render_sources_debug_text,
    render_workspace_debug_text,
)
from spice.runtime.dry_run_executor import execute_dry_run_approval
from spice.runtime.doctor import render_doctor_report, run_doctor
from spice.runtime.perceive import perceive_once
from spice.runtime.refine import refine_decision
from spice.runtime.run_once import run_once
from spice.runtime.sdep_subprocess_executor import execute_sdep_subprocess_approval
from spice.runtime.session import (
    DEFAULT_SESSION_ID,
    build_session_timeline,
    load_or_create_session,
    render_session_resume,
    render_session_stats,
    render_session_timeline,
    session_stats,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import load_workspace_config, workspace_paths


@dataclass(slots=True)
class InteractiveShellResult:
    session_id: str
    status: str = "closed"
    turns: int = 0
    run_ids: list[str] = field(default_factory=list)
    approved_ids: list[str] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    dry_run_outcome_ids: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "turns": self.turns,
            "run_ids": list(self.run_ids),
            "approved_ids": list(self.approved_ids),
            "rejected_ids": list(self.rejected_ids),
            "dry_run_outcome_ids": list(self.dry_run_outcome_ids),
        }


def run_interactive_shell(
    *,
    project_root: str | Path = ".",
    session_id: str = DEFAULT_SESSION_ID,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    use_bars: bool = False,
    persist: bool = True,
    full_loop_preview: bool = True,
    run_intent_mode: str = "auto",
) -> InteractiveShellResult:
    """Run the local Spice interactive runtime shell.

    The shell is intentionally a thin product wrapper over existing runtime
    primitives. It stores runs and approvals locally, and only crosses the
    executor boundary when the user explicitly invokes the local dry-run bridge.
    """

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    paths = workspace_paths(project_root)
    _require_workspace(paths)
    store = LocalJsonStore(paths)
    config = _load_config(paths)
    session = load_or_create_session(store, session_id=session_id)
    store.save_session(session.session_id, session.to_payload())

    result = InteractiveShellResult(session_id=session.session_id)
    _write(output_stream, render_interactive_shell_header(config, session.to_payload()))
    _write(output_stream, "")
    _write(output_stream, render_interactive_shell_help(compact=True))

    while True:
        output_stream.write("\n> ")
        output_stream.flush()
        raw_line = input_stream.readline()
        if raw_line == "":
            break
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("/"):
            should_exit = _handle_shell_command(
                line,
                store=store,
                project_root=project_root,
                session_id=session.session_id,
                output_stream=output_stream,
                result=result,
                use_bars=use_bars,
                persist=persist,
                full_loop_preview=full_loop_preview,
            )
            if should_exit:
                break
            continue
        _run_shell_intent(
            line,
            project_root=project_root,
            session_id=session.session_id,
            output_stream=output_stream,
            result=result,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
            run_intent_mode=run_intent_mode,
        )

    _write(output_stream, "")
    _write(output_stream, f"Spice session closed: {session.session_id}")
    _write(output_stream, f"turns: {result.turns}")
    return result


def render_interactive_shell_header(
    config: dict[str, Any],
    session_payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "Spice Agent",
            f"session: {session_payload.get('session_id') or DEFAULT_SESSION_ID}",
            f"executor: {config.get('executor') or 'dry_run'}",
            f"permission: {config.get('permission_mode') or 'confirm_before_execution'}",
            "mode: local runtime shell",
            "boundary: no real executor unless you explicitly run /dry-run",
        ]
    )


def render_interactive_shell_help(*, compact: bool = False) -> str:
    lines = [
        "Commands:",
        "- type any intent to run the default decision loop",
        "- /act <intent>       run an execution-handoff decision",
        "- /advise <intent>    run a decision-only advisory turn",
        "- /refine <feedback>  refine the latest decision card",
        "- /card              show the latest Decision Card",
        "- /why               show why the latest decision won",
        "- /sim               show latest simulation outcomes",
        "- /json              show latest run artifact JSON",
        "- /sources [--json]  show files, URLs, snippets, and perception artifacts used",
        "- /approvals         list approval checkpoints",
        "- /approve <id>      approve a pending checkpoint",
        "- /reject <id>       reject a pending checkpoint",
        "- /details <id>      show approval details",
        "- /execute <id>      execute using configured executor",
        "- /dry-run <id>      run the local dry-run executor bridge",
        "- /perceive [opts]   pull external signals once; optionally open a Decision Card",
        "- /timeline          show current session timeline",
        "- /stats             show session stats",
        "- /doctor            check workspace health",
        "- /context [--json]  show the compiled model context",
        "- /workspace [--json] show latest workspace perception",
        "- /state             show General Decision state",
        "- /session           show current session summary",
        "- /help              show this help",
        "- /exit              close the shell",
    ]
    if compact:
        return "\n".join(lines[:1] + lines[1:5] + ["- /help              show all commands"])
    return "\n".join(lines)


def _handle_shell_command(
    line: str,
    *,
    store: LocalJsonStore,
    project_root: str | Path,
    session_id: str,
    output_stream: TextIO,
    result: InteractiveShellResult,
    use_bars: bool,
    persist: bool,
    full_loop_preview: bool,
) -> bool:
    routed = route_slash_command(line)
    command, value = routed.command, routed.value
    if not routed.known:
        _write(output_stream, f"unknown command: {command}. Type /help for commands.")
        return False
    if command in {"/exit", "/quit"}:
        return True
    if command == "/help":
        _write(output_stream, render_interactive_shell_help())
        return False
    if command == "/session":
        session = load_or_create_session(store, session_id=session_id)
        _write(output_stream, render_session_resume(session))
        return False
    if command == "/timeline":
        session = load_or_create_session(store, session_id=session_id)
        _write(output_stream, render_session_timeline(build_session_timeline(store, session)))
        return False
    if command in {"/stats", "/metrics"}:
        _write(output_stream, render_session_stats(session_stats(store)))
        return False
    if command == "/doctor":
        try:
            _write(output_stream, render_doctor_report(run_doctor(project_root)))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/state":
        try:
            _write(output_stream, _render_plain_state(store.load_state()))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/context":
        try:
            payload = compile_workspace_decision_context_payload(
                project_root=project_root,
                session_id=session_id,
            )
            if _json_requested(value):
                _write(output_stream, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            else:
                _write(output_stream, render_decision_context_text(payload))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/workspace":
        try:
            payload = compile_workspace_debug_payload(
                project_root=project_root,
                session_id=session_id,
            )
            if _json_requested(value):
                _write(output_stream, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            else:
                _write(output_stream, render_workspace_debug_text(payload))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/sources":
        try:
            payload = compile_sources_debug_payload(
                project_root=project_root,
                session_id=session_id,
                run_id=_sources_run_id(value),
            )
            if _json_requested(value):
                _write(output_stream, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            else:
                _write(output_stream, render_sources_debug_text(payload))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/card":
        try:
            run_payload = _load_run_for_command(store, session_id=session_id, value=value)
            compare = _dict(run_payload.get("compare_payload"))
            _write(output_stream, render_compare_text(compare, use_bars=use_bars))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/why":
        try:
            run_payload = _load_run_for_command(store, session_id=session_id, value=value)
            _write(output_stream, _render_plain_why(_dict(run_payload.get("compare_payload"))))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/sim":
        try:
            run_payload = _load_run_for_command(store, session_id=session_id, value=value)
            _write(output_stream, _render_plain_simulation(_dict(run_payload.get("compare_payload"))))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/json":
        try:
            run_payload = _load_run_for_command(store, session_id=session_id, value=value)
            _write(output_stream, json.dumps(run_payload, ensure_ascii=False, sort_keys=True, indent=2))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/perceive":
        try:
            options = _parse_perceive_args(value)
            perception = perceive_once(
                project_root=project_root,
                provider=options.get("provider"),
                poll_url=options.get("poll_url"),
                poll_command=options.get("poll_command"),
                openchronicle_mcp_url=options.get("openchronicle_mcp_url"),
                openchronicle_since_minutes=options.get("openchronicle_since_minutes"),
                openchronicle_context_limit=options.get("openchronicle_context_limit"),
                allow_command_poll=options.get("allow_command_poll"),
                decide_on_change=options.get("decide_on_change"),
                timeout_seconds=options.get("timeout_seconds"),
            )
            run_id = str(perception.artifact.get("run_id") or "")
            if run_id:
                result.run_ids.append(run_id)
            result.turns += 1
            _write(output_stream, perception.rendered_text)
            _write(output_stream, "")
            _write(output_stream, "Perception artifacts:")
            _write(output_stream, f"  perception={perception.perception_path}")
            _write(output_stream, f"  state={perception.state_path}")
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/approvals":
        _write(output_stream, render_approval_list(list_approvals(store)))
        return False
    if command in {"/approve", "/yes", "/y"}:
        if not value:
            _write(output_stream, "error: approval id required")
            return False
        try:
            approval_result = approve_approval(store, value)
            result.approved_ids.append(value)
            _write(output_stream, render_approval_resolution(approval_result))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command in {"/reject", "/no", "/n"}:
        if not value:
            _write(output_stream, "error: approval id required")
            return False
        try:
            approval_result = reject_approval(store, value)
            result.rejected_ids.append(value)
            _write(output_stream, render_approval_resolution(approval_result))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/details":
        try:
            if value:
                _write(output_stream, render_approval_details(load_approval(store, value)))
            else:
                _write(output_stream, _render_plain_active_frame_details(store.load_state()))
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/dry-run":
        if not value:
            _write(output_stream, "error: approval id required")
            return False
        try:
            execution = execute_dry_run_approval(value, project_root=project_root)
            outcome_id = str(execution.artifact.get("outcome_id") or "")
            if outcome_id:
                result.dry_run_outcome_ids.append(outcome_id)
            _write(output_stream, execution.rendered_text)
            _write(output_stream, "")
            _write(output_stream, "Artifacts:")
            _write(output_stream, f"  run={execution.run_path}")
            _write(output_stream, f"  outcome={execution.outcome_path}")
            if execution.session_path is not None:
                _write(output_stream, f"  session={execution.session_path}")
            _write(output_stream, f"  state={execution.state_path}")
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/execute":
        if not value:
            _write(output_stream, "error: approval id required")
            return False
        try:
            config = load_workspace_config(project_root)
            if config.executor == "dry_run":
                execution = execute_dry_run_approval(value, project_root=project_root)
            elif config.executor == "sdep_subprocess":
                if not config.executor_command:
                    raise ValueError("executor=sdep_subprocess requires executor_command in .spice/config.json.")
                execution = execute_sdep_subprocess_approval(
                    value,
                    command=config.executor_command,
                    project_root=project_root,
                )
            else:
                raise ValueError(f"Unsupported executor in .spice/config.json: {config.executor!r}.")
            outcome_id = str(execution.artifact.get("outcome_id") or "")
            if outcome_id:
                result.dry_run_outcome_ids.append(outcome_id)
            _write(output_stream, execution.rendered_text)
            _write(output_stream, "")
            _write(output_stream, "Artifacts:")
            _write(output_stream, f"  run={execution.run_path}")
            _write(output_stream, f"  outcome={execution.outcome_path}")
            if execution.session_path is not None:
                _write(output_stream, f"  session={execution.session_path}")
            _write(output_stream, f"  state={execution.state_path}")
        except Exception as exc:
            _write(output_stream, f"error: {exc}")
        return False
    if command == "/act":
        if not value:
            _write(output_stream, "error: /act requires an intent")
            return False
        _run_shell_intent(
            value,
            project_root=project_root,
            session_id=session_id,
            output_stream=output_stream,
            result=result,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
            run_intent_mode="act",
        )
        return False
    if command == "/advise":
        if not value:
            _write(output_stream, "error: /advise requires an intent")
            return False
        _run_shell_intent(
            value,
            project_root=project_root,
            session_id=session_id,
            output_stream=output_stream,
            result=result,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=False,
            run_intent_mode="advise",
        )
        return False
    if command in {"/refine", "/modify"}:
        if not value:
            _write(output_stream, "error: /refine requires feedback text")
            return False
        _run_shell_refine(
            value,
            project_root=project_root,
            session_id=session_id,
            output_stream=output_stream,
            result=result,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
        )
        return False
    _write(output_stream, f"unknown command: {command}. Type /help for commands.")
    return False


def _run_shell_intent(
    intent: str,
    *,
    project_root: str | Path,
    session_id: str,
    output_stream: TextIO,
    result: InteractiveShellResult,
    use_bars: bool,
    persist: bool,
    full_loop_preview: bool,
    run_intent_mode: str,
) -> None:
    try:
        run_result = run_once(
            intent,
            project_root=project_root,
            session_id=session_id,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
            run_intent_mode=run_intent_mode,
        )
    except Exception as exc:
        _write(output_stream, f"error: {exc}")
        return

    run_id = str(run_result.artifact.get("run_id") or "")
    if run_id:
        result.run_ids.append(run_id)
    result.turns += 1
    _write(output_stream, run_result.rendered_text)
    _write(output_stream, "")
    _write(output_stream, "Artifacts:")
    _write(output_stream, f"  run={run_result.run_path}")
    _write(output_stream, f"  decision={run_result.decision_path}")
    if run_result.approval_path is not None:
        _write(output_stream, f"  approval={run_result.approval_path}")
    _write(output_stream, f"  session={run_result.session_path}")
    _write(output_stream, f"  state={run_result.state_path}")


def _run_shell_refine(
    refinement: str,
    *,
    project_root: str | Path,
    session_id: str,
    output_stream: TextIO,
    result: InteractiveShellResult,
    use_bars: bool,
    persist: bool,
    full_loop_preview: bool,
) -> None:
    try:
        refine_result = refine_decision(
            refinement,
            project_root=project_root,
            session_id=session_id,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
        )
    except Exception as exc:
        _write(output_stream, f"error: {exc}")
        return

    run_id = str(refine_result.artifact.get("run_id") or "")
    if run_id:
        result.run_ids.append(run_id)
    result.turns += 1
    _write(output_stream, refine_result.rendered_text)
    _write(output_stream, "")
    _write(output_stream, "Artifacts:")
    _write(output_stream, f"  run={refine_result.run_path}")
    _write(output_stream, f"  decision={refine_result.decision_path}")
    if refine_result.approval_path is not None:
        _write(output_stream, f"  approval={refine_result.approval_path}")
    _write(output_stream, f"  session={refine_result.session_path}")
    _write(output_stream, f"  state={refine_result.state_path}")


def _split_command(line: str) -> tuple[str, str]:
    return split_slash_command(line)


def _parse_perceive_args(value: str) -> dict[str, Any]:
    tokens = shlex.split(value)
    options: dict[str, Any] = {
        "provider": None,
        "poll_url": None,
        "poll_command": None,
        "openchronicle_mcp_url": None,
        "openchronicle_since_minutes": None,
        "openchronicle_context_limit": None,
        "allow_command_poll": None,
        "decide_on_change": None,
        "timeout_seconds": None,
    }
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--provider":
            options["provider"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--poll-url":
            options["poll_url"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--poll-command":
            options["poll_command"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--allow-command-poll":
            options["allow_command_poll"] = True
            index += 1
            continue
        if token == "--decide-on-change":
            options["decide_on_change"] = True
            index += 1
            continue
        if token == "--openchronicle-mcp-url":
            options["openchronicle_mcp_url"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--openchronicle-since-minutes":
            options["openchronicle_since_minutes"] = _positive_int(
                _next_value(tokens, index, token),
                token,
            )
            index += 2
            continue
        if token == "--openchronicle-context-limit":
            options["openchronicle_context_limit"] = _positive_int(
                _next_value(tokens, index, token),
                token,
            )
            index += 2
            continue
        if token == "--timeout":
            options["timeout_seconds"] = _positive_int(_next_value(tokens, index, token), token)
            index += 2
            continue
        raise ValueError(f"unknown /perceive option: {token}")
    return options


def _load_run_for_command(
    store: LocalJsonStore,
    *,
    session_id: str,
    value: str,
) -> dict[str, Any]:
    run_id = value.strip()
    if not run_id:
        session = load_or_create_session(store, session_id=session_id)
        run_id = str(session.last_run_id or "").strip()
    if not run_id:
        raise ValueError("No previous run is available. Type an intent first.")
    return store.load_run(run_id)


def _json_requested(value: str) -> bool:
    tokens = [token.strip().lower() for token in shlex.split(value or "")]
    return "--json" in tokens or "json" in tokens


def _sources_run_id(value: str) -> str:
    for token in shlex.split(value or ""):
        normalized = token.strip()
        if not normalized or normalized.lower() in {"--json", "json"}:
            continue
        return normalized
    return ""


def _next_value(tokens: list[str], index: int, option: str) -> str:
    next_index = index + 1
    if next_index >= len(tokens) or tokens[next_index].startswith("--"):
        raise ValueError(f"{option} requires a value")
    return tokens[next_index]


def _positive_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{option} must be positive")
    return parsed


def _render_plain_state(state_payload: dict[str, Any]) -> str:
    general = _general_state_payload(state_payload)
    lines = ["WORLD STATE"]
    counts = {
        "observations": len(_items(general, "observations")),
        "intents": len(_items(general, "intents")),
        "work_items": len(_items(general, "work_items")),
        "commitments": len(_items(general, "commitments")),
        "risks": len(_items(general, "risks")),
        "open_loops": len(_items(general, "open_loops")),
        "outcomes": len(_items(general, "outcomes")),
        "approvals": len(_items(general, "approvals")),
    }
    lines.extend(f"- {key}: {value}" for key, value in counts.items())
    _append_plain_samples(lines, "Open work items", _items(general, "work_items"), "title")
    _append_plain_samples(lines, "Active commitments", _items(general, "commitments"), "title")
    _append_plain_samples(lines, "Open loops", _items(general, "open_loops"), "summary")
    return "\n".join(lines)


def _render_plain_active_frame_details(state_payload: dict[str, Any]) -> str:
    general = _general_state_payload(state_payload)
    metadata = general.get("metadata")
    frame = _dict(metadata).get("active_decision_frame")
    frame_payload = _dict(frame)
    if not frame_payload:
        return "No active Decision Card."
    selected = _dict(frame_payload.get("selected"))
    return "\n".join(
        [
            "ACTIVE DECISION FRAME",
            f"decision_id: {frame_payload.get('decision_id') or ''}",
            f"run_id: {frame_payload.get('run_id') or ''}",
            f"selected: {selected.get('label') or ''} {selected.get('title') or selected.get('recommended_action') or ''}".rstrip(),
            f"candidate_id: {selected.get('candidate_id') or frame_payload.get('selected_candidate_id') or ''}",
            f"approval_id: {frame_payload.get('approval_id') or ''}",
            f"status: {frame_payload.get('status') or 'unknown'}",
        ]
    )


def _render_plain_why(compare_payload: dict[str, Any]) -> str:
    if not compare_payload:
        return "No Decision Card found."
    selected = _dict(compare_payload.get("selected_recommendation"))
    lines = [
        "WHY THIS DECISION",
        f"selected: {selected.get('title') or 'unknown'}",
        f"candidate_id: {selected.get('candidate_id') or ''}",
    ]
    if selected.get("selection_reason"):
        lines.append(f"selection: {selected.get('selection_reason')}")
    if selected.get("human_summary"):
        lines.append(f"recommendation: {selected.get('human_summary')}")
    basis = _list(selected.get("decision_basis"))
    if basis:
        lines.extend(["", "why this won:"])
        for item in basis[:3]:
            payload = _dict(item)
            dimension = str(payload.get("dimension") or payload.get("label") or "factor")
            contribution = payload.get("contribution")
            lines.append(f"- {dimension}{': ' + str(contribution) if contribution is not None else ''}")
    why_not = _list(compare_payload.get("why_not_the_others"))
    if why_not:
        lines.extend(["", "why not others:"])
        for item in why_not[:3]:
            payload = _dict(item)
            lines.append(f"- {payload.get('title') or payload.get('candidate_id') or 'candidate'}")
            for reason in _list(payload.get("reasons"))[:2]:
                reason_payload = _dict(reason)
                text = str(reason_payload.get("reason") or reason_payload.get("summary") or "").strip()
                if text:
                    lines.append(f"  - {text}")
    return "\n".join(lines)


def _render_plain_simulation(compare_payload: dict[str, Any]) -> str:
    candidates = _list(compare_payload.get("candidate_decisions"))
    if not candidates:
        return "No simulation data found."
    lines = ["SIMULATION SUMMARY"]
    rendered = 0
    for candidate in candidates:
        payload = _dict(candidate)
        simulation = _dict(payload.get("simulation"))
        if not simulation:
            continue
        rendered += 1
        lines.append("")
        lines.append(f"{payload.get('label') or rendered}. {payload.get('title') or payload.get('candidate_id') or 'candidate'}")
        outcome = str(simulation.get("expected_outcome") or simulation.get("simulated_outcome") or "").strip()
        downside = str(simulation.get("downside") or "").strip()
        success = str(simulation.get("success_signal") or "").strip()
        confidence = simulation.get("confidence")
        if outcome:
            lines.append(f"- expected: {outcome}")
        if downside:
            lines.append(f"- downside: {downside}")
        if success:
            lines.append(f"- success: {success}")
        if confidence is not None:
            lines.append(f"- confidence: {confidence}")
    if rendered == 0:
        lines.append("- no LLM simulation attached to the current visible candidates")
    return "\n".join(lines)


def _general_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    world = state_payload.get("world_state")
    if not isinstance(world, dict):
        return {}
    domain = world.get("domain_state")
    if not isinstance(domain, dict):
        return {}
    general = domain.get("general_decision")
    return general if isinstance(general, dict) else {}


def _items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _append_plain_samples(
    lines: list[str],
    title: str,
    items: list[dict[str, Any]],
    field: str,
) -> None:
    if not items:
        return
    lines.append("")
    lines.append(f"{title}:")
    for item in items[:3]:
        lines.append(f"- {item.get(field) or item.get('status') or 'unknown'}")


def _require_workspace(paths: Any) -> None:
    missing = [
        path
        for path in (paths.config, paths.decision_profile, paths.state)
        if not path.exists()
    ]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Spice workspace is not initialized. Missing: {rendered}. Run `spice setup` first."
        )


def _load_config(paths: Any) -> dict[str, Any]:
    payload = json.loads(paths.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Workspace config must be a JSON object: {paths.config}")
    return payload


def _write(output_stream: TextIO, text: str) -> None:
    output_stream.write(f"{text}\n")
