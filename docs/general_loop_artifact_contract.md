# General Loop Artifact Contract

This document describes the read-only General Decision Loop artifact produced by:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop-json --no-bars
```

The artifact is the machine-readable form of the flagship preview. It is meant
to be stable enough for docs, screenshot tooling, future UI/TUI work, and the
next executor integration step.

It is not a live runtime record yet. The current artifact is read-only.

## Boundary

The full-loop artifact must preserve these boundary flags:

```json
{
  "path_type": "read_only_general_full_loop",
  "loop_status": "completed_read_only",
  "read_only": true,
  "executor_called": false,
  "sdep_request_sent": false,
  "executed": false,
  "persisted": false,
  "update_mode": "read_only_snapshot",
  "state_snapshot_updated": true
}
```

Interpretation:

- `read_only=true` means this is a preview artifact, not a production run.
- `sdep_request_sent=false` means the SDEP request was shaped but not sent.
- `executor_called=false` means no executor adapter ran.
- `executed=false` means no external action happened.
- `persisted=false` means `state_after` is a returned snapshot, not saved state.
- `state_snapshot_updated=true` means the artifact contains a new state snapshot.

Do not treat `state_snapshot_updated=true` as persistence.

## Top-level Fields

The artifact exposes the most important fields at the top level so downstream
tools do not need to drill into nested artifacts.

Core metadata:

- `path_type`
- `generated_by`
- `created_at`
- `status`
- `loop_status`
- `flow`

Core IDs:

- `decision_id`
- `trace_ref`
- `selected_candidate_id`
- `approval_id`
- `execution_id`
- `request_id`
- `outcome_id`

Skill/context handoff:

- `skill_id`
- `executor_id`
- `context_pack_id`
- `resolved_skill`
- `context_pack`

Status and boundary:

- `protocol_status`
- `task_status`
- `read_only`
- `executor_called`
- `executed`
- `execution`
- `sdep_request_sent`
- `persisted`
- `state_snapshot_updated`
- `update_mode`

State and candidate summary:

- `state_before_summary`
- `state_after_summary`
- `state_before`
- `state_after`
- `observations`
- `candidates`
- `candidate_summary`

Decision and stage artifacts:

- `compare_payload`
- `approval_artifact`
- `execution_artifact`
- `outcome_artifact`
- `state_feedback_artifact`
- `decision`
- `approval`
- `execution_plan`
- `outcome_return`
- `state_feedback`
- `rendered_text`

The `*_artifact` aliases and the shorter stage aliases intentionally point to
the same stage content. They support both external artifact readers and existing
demo code.

## Flow

The `flow` field should remain ordered:

```text
observations
general_state
generic_candidates
policy_decision
approval_checkpoint
skill_resolution
context_pack
sdep_request_plan
sdep_response_fixture
outcome_observation
state_feedback_snapshot
```

This order is the contract for demos and UI layouts.

## ID Consistency

The same IDs must line up across all stages.

### Decision

- `artifact.decision_id`
- `decision.compare_payload.decision_id`
- `approval.decision_id`
- `execution_plan.decision_id`
- `outcome_return.decision_id`
- `state_feedback.decision_id`

### Trace

- `artifact.trace_ref`
- `decision.compare_payload.trace_ref`
- `approval.trace_ref`
- `execution_plan.trace_ref`
- `outcome_return.trace_ref`
- `state_feedback.trace_ref`
- `context_pack.trace_ref`

### Candidate

- `artifact.selected_candidate_id`
- `approval.selected_candidate_id`
- `approval.approval.candidate_id`
- `execution_plan.candidate_id`
- `outcome_return.candidate_id`
- `state_feedback.candidate_id`
- `context_pack.candidate_id`

### Approval

- `artifact.approval_id`
- `approval.approval.approval_id`
- `execution_plan.approval_id`
- `outcome_return.approval_id`
- `context_pack.approval_id`

### Execution

- `artifact.execution_id`
- `execution_plan.execution_id`
- `execution_plan.sdep_request.traceability.execution_id`
- `outcome_return.execution_id`
- `state_feedback.execution_id`
- `context_pack.execution_id`

### Request

- `artifact.request_id`
- `execution_plan.sdep_request.request_id`
- `outcome_return.sdep_response.request_id`
- `outcome_return.request_id`
- `context_pack.request_id`

### Outcome

- `artifact.outcome_id`
- `outcome_return.outcome_id`
- `state_feedback.outcome_id`
- one entry in `state_after.outcomes[].outcome_id`

### Skill And Context

- `artifact.skill_id`
- `resolved_skill.skill_id`
- `execution_artifact.skill_id`
- `execution_plan.sdep_request.execution.input.skill_hint.skill_id`

- `artifact.executor_id`
- `resolved_skill.executor_id`

- `artifact.context_pack_id`
- `context_pack.context_pack_id`
- `execution_artifact.context_pack_id`
- `execution_plan.sdep_request.execution.input.context_pack.context_pack_id`

These consistency rules are covered by `tests/test_decision_hub_general_loop.py`.

## Stage Artifacts

### Decision

`decision` contains:

- normalized observations
- state summary
- generated candidates
- candidate count
- `compare_payload`
- rendered Decision Card text

`compare_payload` is the stable human-readable decision schema. It is not a raw
trace dump.

### Approval

`approval_artifact` / `approval` contains:

- pending approval
- legacy-compatible confirmation request
- `execution_allowed=false`
- `execution=null`
- `state_updated=false`

This stage creates the approval checkpoint only. It does not write a
confirmation store.

### Execution Plan

`execution_artifact` / `execution_plan` contains:

- selected candidate attribution
- approved fixture attribution
- resolved skill
- execution context pack
- planned `ExecutionIntent`
- planned SDEP `execute.request`
- `execution_status="planned_not_executed"`
- `executed=false`
- `execution=null`
- `outcome=null`
- `state_updated=false`

This stage shapes the handoff only. It does not send the request.

### Outcome Return

`outcome_artifact` / `outcome_return` contains:

- fixture SDEP `execute.response`
- `OutcomeRecord`
- `GenericObservation(kind=outcome)`
- request/decision/trace/candidate/approval/execution attribution
- `protocol_status`
- `task_status`
- `executor_called=false`
- `executed=false`
- `state_updated=false`

This stage adapts a local fixture response. It does not call an executor and it
does not apply the outcome to state by itself.

### State Feedback

`state_feedback_artifact` / `state_feedback` contains:

- `state_before`
- `state_after`
- summaries for both states
- outcome attribution
- `state_updated=true`
- `persisted=false`
- `update_mode="read_only_snapshot"`

This stage applies the outcome observation through the General reducer to return
a new snapshot. It does not persist the snapshot.

## SDEP Handoff Contract

The planned SDEP request is available at:

```text
execution_plan.sdep_request
```

The executor-facing context is inside:

```text
execution_plan.sdep_request.execution.input.context_pack
```

The skill hint is inside:

```text
execution_plan.sdep_request.execution.input.skill_hint
```

The request remains thin:

- SDEP is the protocol envelope.
- `skill_hint` identifies the execution template / capability hint.
- `context_pack` carries compact decision-relevant context.
- The executor still decides how to use its own tools internally.

Do not put executor-specific commands, shell commands, or transport details into
the General decision layer.

## Context Pack Contract

`context_pack` is the compact handoff context.

Required executor-facing fields:

- `context_pack_id`
- `decision_id`
- `trace_ref`
- `candidate_id`
- `approval_id`
- `execution_id`
- `request_id`
- `skill_id`
- `executor_id`
- `task`
- `why_now`
- `do_not`
- `expected_output`
- `return_schema`
- `target_refs`
- `relevant_state_refs`
- `relevant_observations`
- `state_summary`
- `decision_summary`

The context pack must not dump the full General state, all observations, or all
candidates. It should only carry state relevant to the selected handoff.

## Outcome Attribution

The fixture outcome and reduced state outcome should preserve:

- `decision_id`
- `trace_ref`
- `candidate_id`
- `approval_id`
- `execution_id`
- `request_id`
- `outcome_id`
- `protocol_status`
- `task_status`

`state_after.outcomes[].metadata` should retain named provenance such as
`request_id`, `approval_id`, `protocol_status`, and `task_status`.

This is what allows later runtime persistence or replay to connect an execution
result back to the exact decision that produced it.

## Compatibility Rules

When changing the artifact:

1. Keep existing top-level IDs stable.
2. Keep read-only boundary flags explicit.
3. Keep nested stage artifacts available.
4. Add new fields in a backward-compatible way.
5. Do not replace `compare_payload` with a raw trace.
6. Do not make consumers depend only on deep nested paths for core IDs.
7. Do not imply live execution unless `executor_called=true`,
   `sdep_request_sent=true`, and `executed=true` are all intentionally set by a
   future runtime path.

## Current Next Consumer

The next consumer should be the real executor integration contract:

```text
execution_plan.sdep_request
  + execution.input.skill_hint
  + execution.input.context_pack
-> executor adapter
-> SDEP execute.response
-> OutcomeRecord
-> persisted General state
```

That step should reuse this artifact shape instead of inventing another
decision-to-execution schema.

See [`executor_integration_contract.md`](executor_integration_contract.md) for
the adapter input/output contract.
