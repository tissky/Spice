from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from spice.perception import (
    InvestigationConsent,
    build_executor_report_artifact,
    delegated_perception_context_from_artifact,
)
from spice.runtime.delegated_request import (
    READ_ONLY_PERMISSION_MODE,
    build_delegated_perception_request,
)
from spice.runtime.delegated_result import (
    DelegatedPerceptionNormalizationResult,
    normalize_delegated_perception_result,
)
from spice.runtime.executor_runtime import (
    ResolvedExecutorRuntime,
    resolve_executor_runtime_from_config_with_permission,
)
from spice.runtime.memory_writeback import (
    skipped_general_delegated_perception_memory_writeback,
    write_general_delegated_perception_memory,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.workspace import load_workspace_config, load_workspace_memory_provider


@dataclass(frozen=True, slots=True)
class RuntimeDelegatedPerceptionResult:
    requested: bool
    status: str
    query: str = ""
    context: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    path: Path | None = None
    memory_writeback: dict[str, Any] | None = None
    error: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "status": self.status,
            "query": self.query,
            "context": dict(self.context or {}),
            "artifact": dict(self.artifact or {}),
            "path": str(self.path) if self.path is not None else "",
            "memory_writeback": dict(self.memory_writeback or {}),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class DelegatedExecutorCommandResult:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True, slots=True)
class RuntimeDelegatedPerceptionHandoffResult:
    status: str
    request: dict[str, Any]
    executor_runtime: dict[str, Any]
    executor_report: dict[str, Any]
    normalization: dict[str, Any]
    perception: RuntimeDelegatedPerceptionResult
    executor_report_path: Path | None = None
    error: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request": dict(self.request),
            "executor_runtime": dict(self.executor_runtime),
            "executor_report": dict(self.executor_report),
            "normalization": dict(self.normalization),
            "perception": self.perception.to_payload(),
            "executor_report_path": str(self.executor_report_path)
            if self.executor_report_path is not None
            else "",
            "error": self.error,
        }


ReadOnlyExecutorRunner = Callable[
    [ResolvedExecutorRuntime, str, float, Path],
    DelegatedExecutorCommandResult,
]


def run_delegated_perception_handoff(
    *,
    project_root: str | Path,
    store: LocalJsonStore,
    consent: InvestigationConsent | Mapping[str, Any],
    escalation_decision: Mapping[str, Any] | Any,
    route_payload: Mapping[str, Any] | None = None,
    user_input: str = "",
    active_decision_frame: Mapping[str, Any] | None = None,
    workspace_context: Mapping[str, Any] | None = None,
    url_context: Mapping[str, Any] | None = None,
    delegated_perception_context: Mapping[str, Any] | None = None,
    input_context_refs: list[str] | None = None,
    conversation_turn_id: str = "",
    linked_decision_id: str = "",
    linked_run_id: str = "",
    persist: bool = True,
    runner: ReadOnlyExecutorRunner | None = None,
) -> RuntimeDelegatedPerceptionHandoffResult:
    """Run a granted read-only delegated investigation through the configured executor.

    This is perception handoff, not execution. It never creates an execution
    approval or execution outcome; it records an executor report plus a
    normalized delegated perception artifact.
    """

    root = Path(project_root)
    consent_obj = consent if isinstance(consent, InvestigationConsent) else InvestigationConsent.from_payload(consent)
    route_result = dict(route_payload or {})
    request = build_delegated_perception_request(
        escalation_decision=escalation_decision,
        consent=consent_obj,
        user_input=user_input,
        active_decision_frame=active_decision_frame,
        workspace_context=workspace_context,
        url_context=url_context,
        delegated_perception_context=delegated_perception_context,
        input_context_refs=input_context_refs,
    )
    config = load_workspace_config(root)
    runtime = resolve_executor_runtime_from_config_with_permission(
        config,
        READ_ONLY_PERMISSION_MODE,
    )
    executor_report_path: Path | None = None

    if runtime.executor_id != request.executor_id:
        reason = (
            f"Configured executor {runtime.executor_id} does not match consent executor "
            f"{request.executor_id}."
        )
        report = _failed_executor_report(request, runtime=runtime, reason=reason)
    elif not _runtime_supports_delegated_perception(runtime):
        reason = _runtime_unsupported_reason(runtime)
        report = _failed_executor_report(request, runtime=runtime, reason=reason)
    else:
        timeout_seconds = _timeout_seconds(request.budget)
        command_result = (runner or _run_read_only_executor)(
            runtime,
            request.prompt,
            timeout_seconds,
            root,
        )
        report = _executor_report_from_command_result(
            request=request,
            runtime=runtime,
            command_result=command_result,
        )

    report_payload = report.to_payload()
    if persist:
        executor_report_path = store.save_perception(report.report_id, report_payload)

    normalization = normalize_delegated_perception_result(
        executor_report=report,
        request=request,
        consent_id=consent_obj.consent_id,
        input_context_refs=list(input_context_refs or consent_obj.input_context_refs),
        context_strategy=request.context_strategy,
    )
    perception_result = finalize_runtime_delegated_perception_result(
        project_root=root,
        store=store,
        artifact_payload=normalization.artifact.to_payload(),
        requested=True,
        status=normalization.artifact.status,
        query=request.query,
        persist=persist,
        user_input=user_input,
        route_result={
            **route_result,
            "context_strategy": request.context_strategy,
            "delegated_perception_query": request.query,
            "delegated_plan": dict(request.delegated_plan),
            "expected_output": request.expected_output,
            "consent_id": consent_obj.consent_id,
            "executor_id": request.executor_id,
            "handoff_mode": request.mode,
            "permission_mode": request.permission_mode,
        },
        linked_decision_id=linked_decision_id,
        linked_run_id=linked_run_id,
        conversation_turn_id=conversation_turn_id,
        error="" if normalization.artifact.status == "completed" else "; ".join(normalization.artifact.limitations),
    )
    return RuntimeDelegatedPerceptionHandoffResult(
        status=normalization.artifact.status,
        request=request.to_payload(),
        executor_runtime=runtime.to_payload(),
        executor_report=report_payload,
        normalization=normalization.to_payload(),
        perception=perception_result,
        executor_report_path=executor_report_path,
        error=perception_result.error,
    )


def finalize_runtime_delegated_perception_result(
    *,
    project_root: str | Path,
    store: LocalJsonStore,
    artifact_payload: dict[str, Any],
    requested: bool = True,
    status: str = "written",
    query: str = "",
    persist: bool = True,
    user_input: str = "",
    route_result: dict[str, Any] | None = None,
    linked_decision_id: str = "",
    linked_run_id: str = "",
    conversation_turn_id: str = "",
    error: str = "",
) -> RuntimeDelegatedPerceptionResult:
    """Persist a normalized delegated perception artifact and compact memory record."""

    memory_writeback = _write_delegated_perception_memory(
        project_root=project_root,
        artifact_payload=artifact_payload,
        persist=persist,
        user_input=user_input,
        route_result=route_result,
        linked_decision_id=linked_decision_id,
        linked_run_id=linked_run_id,
        conversation_turn_id=conversation_turn_id,
    )
    artifact_payload["memory_writeback"] = memory_writeback
    perception_id = str(artifact_payload.get("perception_id") or "")
    saved_path = store.save_perception(perception_id, artifact_payload) if persist else None
    return RuntimeDelegatedPerceptionResult(
        requested=requested,
        status=status,
        query=query or str(artifact_payload.get("query") or ""),
        context=delegated_perception_context_from_artifact(artifact_payload),
        artifact=artifact_payload,
        path=saved_path,
        memory_writeback=memory_writeback,
        error=error,
    )


def _write_delegated_perception_memory(
    *,
    project_root: str | Path,
    artifact_payload: dict[str, Any],
    persist: bool,
    user_input: str,
    route_result: dict[str, Any] | None,
    linked_decision_id: str,
    linked_run_id: str,
    conversation_turn_id: str,
) -> dict[str, Any]:
    if not persist:
        return skipped_general_delegated_perception_memory_writeback(reason="not_persisted")
    try:
        provider = load_workspace_memory_provider(project_root)
    except Exception as exc:
        return skipped_general_delegated_perception_memory_writeback(
            reason=f"memory_provider_unavailable:{exc}"
        )
    try:
        return write_general_delegated_perception_memory(
            provider,
            artifact=artifact_payload,
            user_input=user_input,
            route_result=route_result,
            linked_decision_id=linked_decision_id,
            linked_run_id=linked_run_id,
            conversation_turn_id=conversation_turn_id,
        )
    except Exception as exc:
        return skipped_general_delegated_perception_memory_writeback(
            reason=f"write_failed:{exc}"
        )


def _runtime_supports_delegated_perception(runtime: ResolvedExecutorRuntime) -> bool:
    if runtime.status != "ready":
        return False
    if runtime.permission_mode != READ_ONLY_PERMISSION_MODE:
        return False
    if runtime.transport not in {"sdep_subprocess", "sdep_subprocess_wrapper"}:
        return False
    return bool(runtime.command_argv)


def _runtime_unsupported_reason(runtime: ResolvedExecutorRuntime) -> str:
    if runtime.status != "ready":
        return runtime.detail or f"Executor {runtime.executor_id} is not ready."
    if runtime.permission_mode != READ_ONLY_PERMISSION_MODE:
        return f"Executor permission mode is {runtime.permission_mode}, not read_only."
    if runtime.transport not in {"sdep_subprocess", "sdep_subprocess_wrapper"}:
        return (
            f"Executor {runtime.executor_id} does not support delegated perception "
            f"handoff transport: {runtime.transport}."
        )
    if not runtime.command_argv:
        return f"Executor {runtime.executor_id} has no command configured for read-only handoff."
    return f"Executor {runtime.executor_id} does not support delegated perception handoff."


def _failed_executor_report(
    request: Any,
    *,
    runtime: ResolvedExecutorRuntime,
    reason: str,
) -> Any:
    structured = {
        "status": "failed",
        "summary": reason,
        "findings": [],
        "sources": [],
        "limitations": [reason],
        "confidence": "low",
    }
    return build_executor_report_artifact(
        executor_id=request.executor_id,
        query=request.query,
        structured_output=structured,
        status="failed",
        scope=request.scope,
        permission_mode=request.permission_mode,
        request_ref=request.request_id,
        executor_run_ref=_executor_run_ref(request=request, runtime=runtime),
        limitations=[reason],
        metadata={
            "source": "spice.runtime.delegated_perception",
            "mode": request.mode,
            "scope": request.scope,
            "permission_mode": request.permission_mode,
            "delegated_plan": dict(request.delegated_plan),
            "expected_output": request.expected_output,
            "handoff_status": "unsupported",
            "executor_runtime": runtime.to_payload(),
        },
    )


def _executor_report_from_command_result(
    *,
    request: Any,
    runtime: ResolvedExecutorRuntime,
    command_result: DelegatedExecutorCommandResult,
) -> Any:
    structured = _json_object(command_result.stdout)
    limitations: list[str] = []
    if command_result.stderr.strip():
        limitations.append("executor_stderr_available_in_report")
    if command_result.timed_out:
        limitations.append("executor_timed_out")
    status = "completed" if command_result.status == "success" else "failed"
    if structured:
        reported_status = str(structured.get("status") or "").strip().lower()
        if reported_status in {"failed", "blocked"}:
            status = reported_status
    return build_executor_report_artifact(
        executor_id=request.executor_id,
        query=request.query,
        raw_output=command_result.stdout,
        structured_output=structured,
        status=status,
        scope=request.scope,
        permission_mode=request.permission_mode,
        request_ref=request.request_id,
        executor_run_ref=_executor_run_ref(request=request, runtime=runtime),
        limitations=limitations,
        metadata={
            "source": "spice.runtime.delegated_perception",
            "mode": request.mode,
            "scope": request.scope,
            "permission_mode": request.permission_mode,
            "delegated_plan": dict(request.delegated_plan),
            "expected_output": request.expected_output,
            "executor_runtime": runtime.to_payload(),
            "command": _command_summary(runtime.command_argv),
            "command_result": command_result.to_payload(),
            "raw_output_retained_in_executor_report": True,
        },
    )


def _run_read_only_executor(
    runtime: ResolvedExecutorRuntime,
    prompt: str,
    timeout_seconds: float,
    cwd: Path,
) -> DelegatedExecutorCommandResult:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    command, stdin_text = _prepare_read_only_invocation(runtime, prompt)
    try:
        completed = subprocess.run(
            list(command),
            input=stdin_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            shell=False,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired as exc:
        return DelegatedExecutorCommandResult(
            status="failed",
            stdout=str(exc.stdout or ""),
            stderr=f"Read-only investigation timed out after {timeout_seconds} seconds.",
            exit_code=None,
            timed_out=True,
        )
    except OSError as exc:
        return DelegatedExecutorCommandResult(
            status="failed",
            stderr=str(exc),
            exit_code=None,
            timed_out=False,
        )
    return DelegatedExecutorCommandResult(
        status="success" if completed.returncode == 0 else "failed",
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
        timed_out=False,
    )


def _prepare_read_only_invocation(
    runtime: ResolvedExecutorRuntime,
    prompt: str,
) -> tuple[tuple[str, ...], str | None]:
    command = tuple(runtime.command_argv)
    if any(item == "{prompt}" for item in command):
        return tuple(prompt if item == "{prompt}" else item for item in command), None
    if _is_hermes_chat_command(command) and not _has_query_argument(command):
        return (*command, "-q", prompt), None
    return command, prompt


def _is_hermes_chat_command(command: tuple[str, ...]) -> bool:
    if len(command) < 2:
        return False
    executable = command[0].rsplit("/", 1)[-1]
    return executable == "hermes" and command[1] == "chat"


def _has_query_argument(command: tuple[str, ...]) -> bool:
    return any(item == "-q" or item == "--query" or item.startswith("--query=") for item in command)


def _json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _timeout_seconds(budget: Mapping[str, Any]) -> float:
    try:
        value = float(budget.get("max_duration_sec", 120))
    except (TypeError, ValueError):
        value = 120.0
    return max(1.0, min(value, 600.0))


def _executor_run_ref(*, request: Any, runtime: ResolvedExecutorRuntime) -> str:
    return f"executor_run.{runtime.executor_id}.{request.request_id}"


def _command_summary(command: tuple[str, ...]) -> str:
    if not command:
        return ""
    return " ".join(command)
