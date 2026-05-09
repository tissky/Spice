# Executor Integration Contract

This document defines the pre-runtime contract for connecting a real executor to
the General Decision Loop.

It starts from the read-only full-loop artifact:

```text
Spice selected, resolved skill, compressed context, planned SDEP handoff.
```

The next step is to replace the local fixture response with a real executor
adapter while preserving the same decision, approval, skill, context, and SDEP
boundaries.

## Integration Boundary

The executor integration starts after approval and planning:

```text
selected candidate
-> approval checkpoint
-> resolved skill
-> context pack
-> SDEP execute.request
-> executor adapter
-> SDEP execute.response
-> OutcomeRecord
-> persisted General state
```

The executor adapter must not decide what should be done. Spice already made
that decision.

The executor adapter receives a planned handoff and performs the task through
its own tools, skills, models, or native workflow.

## Input To The Executor Adapter

The primary input is:

```text
execution_plan.sdep_request
```

From the full-loop artifact:

```text
artifact.execution_plan.sdep_request
```

The request is an SDEP `execute.request`.

Required request-level fields:

- `protocol == "sdep"`
- `sdep_version == "0.1"`
- `message_type == "execute.request"`
- `message_id`
- `request_id`
- `timestamp`
- `sender`
- `idempotency_key`
- `execution`
- `traceability`
- `metadata`

Required attribution fields:

- `traceability.execution_id`
- `traceability.spice.provenance.spice_decision_id`
- `traceability.spice.provenance.trace_ref`
- `traceability.spice.provenance.candidate_id`
- `traceability.spice.provenance.approval_id`

The current General planner also promotes the same IDs in request traceability:

- `traceability.execution_id`
- `traceability.spice_decision_id`
- `traceability.trace_ref`
- `traceability.candidate_id`
- `traceability.approval_id`

The adapter should preserve both forms when returning a response.

## Execution Payload

The executor-facing payload is:

```text
sdep_request.execution
```

Important fields:

- `execution.action_type`
- `execution.target`
- `execution.parameters`
- `execution.input`
- `execution.constraints`
- `execution.success_criteria`
- `execution.failure_policy`
- `execution.mode`
- `execution.dry_run`
- `execution.metadata`

For General loop handoffs, executor-specific context is in:

```text
execution.input.skill_hint
execution.input.context_pack
```

The adapter should treat `execution.parameters` and `execution.metadata` as
supporting information, not as the primary task context.

## Skill Hint

`execution.input.skill_hint` identifies the selected execution template or
capability hint.

Expected fields:

- `skill_id`
- `executor_id`
- `skill_source`
- `side_effect_class`
- `context_pack_id`

Semantics:

- `skill_id` is not a command.
- `skill_id` is not a tool invocation.
- `skill_id` is the selected template/capability hint Spice resolved before
  crossing the execution boundary.
- The executor may map it to its own native skill, tool, prompt, workflow, or
  internal routing.

The adapter must not change the selected decision because it prefers another
skill. If the skill cannot be executed, return an SDEP failure response.

## Context Pack

`execution.input.context_pack` is the compact execution context.

Required fields:

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

Executor behavior:

- Use `task` as the action objective.
- Use `why_now` as decision context, not as a new reasoning prompt to override.
- Use `do_not` as hard boundary guidance.
- Use `expected_output` and `return_schema` to shape the response.
- Use `target_refs`, `relevant_state_refs`, and `relevant_observations` as
  bounded context.

The context pack must remain compact. The adapter should not require full
General state unless a future contract explicitly adds a memory/context provider.

## Adapter Responsibilities

The executor adapter is responsible for:

1. Validate the SDEP request envelope.
2. Validate required attribution fields.
3. Validate `skill_hint` and `context_pack` are present.
4. Map `skill_id` to executor-native behavior.
5. Perform the requested task, if allowed by local policy and permissions.
6. Return an SDEP `execute.response`.
7. Preserve attribution in the response.
8. Distinguish protocol/wrapper failure from task failure.

The adapter may:

- use Codex, Claude Code, Hermes, OpenClaw, local tools, browser tools, shell
  tools, or executor-native skills
- ask the executor runtime for more details if its own policy requires it
- fail safely when permissions or capabilities are insufficient

The adapter must not:

- choose a different candidate
- mutate `decision.md`
- mutate Spice policy
- silently ignore approval boundaries
- call back into Spice to re-decide the task
- rewrite `decision_id`, `trace_ref`, `candidate_id`, `approval_id`, or
  `execution_id`
- return an un-attributed outcome

## Output From The Executor Adapter

The adapter must return an SDEP `execute.response`.

Required response-level fields:

- `protocol == "sdep"`
- `sdep_version == "0.1"`
- `message_type == "execute.response"`
- `message_id`
- `request_id`
- `timestamp`
- `responder`
- `status`
- `outcome`
- `traceability`
- `metadata`

The `request_id` must match the request.

The `responder` is the executor-of-record.

## Outcome Payload

The canonical result is:

```text
execute.response.outcome
```

Required outcome fields:

- `execution_id`
- `status`
- `outcome_type`
- `output`
- `artifacts`
- `metrics`
- `metadata`

The `execution_id` must match the request execution id.

Recommended `outcome_type` values:

- `ack`
- `state_delta`
- `artifact_bundle`
- `observation`
- `request_state`
- `approval_state`
- `error`

Domain-specific outcome types should be namespaced strings.

## Two-level Status

Spice needs two status layers:

```text
protocol_status
task_status
```

Use them this way:

- `protocol_status` describes whether the protocol/wrapper handled the request.
- `task_status` describes whether the actual task succeeded.

Examples:

```text
protocol_status=success, task_status=success
protocol_status=success, task_status=failed
protocol_status=failed, task_status=not_started
protocol_status=success, task_status=partial
```

Where to put them:

- `execute.response.status` should reflect protocol-level status.
- `execute.response.outcome.status` should reflect task-level status.
- `execute.response.metadata.protocol_status` may duplicate the protocol status.
- `execute.response.outcome.metadata.task_status` may duplicate the task status.

The outcome adapter will preserve both into `OutcomeRecord` and
`GenericObservation(kind=outcome)`.

## Attribution Requirements

The response must preserve:

- `request_id`
- `execution_id`
- `decision_id`
- `trace_ref`
- `candidate_id`
- `approval_id`

Recommended locations:

```json
{
  "request_id": "sdep-req.general....",
  "outcome": {
    "execution_id": "exec....",
    "metadata": {
      "decision_id": "decision....",
      "trace_ref": "trace....",
      "candidate_id": "candidate....",
      "approval_id": "approval....",
      "request_id": "sdep-req.general....",
      "task_status": "success"
    }
  },
  "traceability": {
    "execution_id": "exec....",
    "spice_decision_id": "decision....",
    "trace_ref": "trace....",
    "candidate_id": "candidate....",
    "approval_id": "approval...."
  },
  "metadata": {
    "protocol_status": "success"
  }
}
```

The response adapter should reject responses where these IDs do not match the
planned request.

## Failure Semantics

Protocol/wrapper failure:

- invalid request
- unsupported SDEP version
- missing skill/context pack
- adapter could not route the skill
- executor process unavailable
- transport error

Return:

```text
protocol_status=failed
task_status=not_started
```

Task failure:

- executor accepted the request
- task ran
- task could not complete the requested outcome

Return:

```text
protocol_status=success
task_status=failed
```

Partial task:

- executor accepted the request
- some work completed
- follow-up is required

Return:

```text
protocol_status=success
task_status=partial
```

Do not collapse all failure cases into a single `failed` string.

## Permission Boundary

The current General loop already passed a human approval checkpoint before
planning the SDEP handoff.

The executor adapter may still enforce its own local permissions, but it should
not reinterpret approval as a new decision.

If local permissions block the task, return a protocol/wrapper failure or a
task failure depending on where the block occurred.

## Idempotency

The adapter should treat `request_id` and `idempotency_key` as replay guards.

Recommended behavior:

- repeated same `request_id` should not duplicate irreversible work
- repeated same `idempotency_key` should be recognized as the same planned
  execution
- response should preserve the original `request_id`

Future persistence can use `execution_id`, `request_id`, and `outcome_id` for
replay and dedupe.

## Adapter Types

The same contract can support multiple adapter implementations:

- subprocess adapter
- HTTP adapter
- queue adapter
- local Python adapter
- Codex wrapper
- Claude Code wrapper
- Hermes wrapper
- OpenClaw wrapper

These adapters differ in transport and native execution behavior, not in the
Spice decision contract.

## Minimal Executor Adapter Flow

Pseudo-flow:

```text
receive SDEP execute.request
validate SDEP envelope
extract execution.input.skill_hint
extract execution.input.context_pack
map skill_id to native executor behavior
run executor-native task
build SDEP execute.response
preserve attribution
return response
```

The adapter should be thin. Any domain-specific decision logic belongs before
the SDEP boundary, not inside the executor adapter.

## Integration Tests To Add Later

When implementing a real adapter, add tests for:

- request validates through `SDEPExecuteRequest.from_dict`
- adapter rejects missing `skill_hint`
- adapter rejects missing `context_pack`
- adapter preserves request ID
- adapter preserves decision/trace/candidate/approval/execution attribution
- protocol success + task success
- protocol success + task failed
- protocol failed + task not started
- repeated request is idempotent
- no Spice policy or candidate selection happens inside adapter

## Relationship To Current Read-only Preview

Current preview:

```text
planned SDEP request
-> local fixture response
-> outcome observation
-> read-only state snapshot
```

Real integration:

```text
planned SDEP request
-> executor adapter
-> live SDEP response
-> outcome observation
-> persisted state update
```

Only the middle transport/executor section changes. The decision object,
approval checkpoint, skill hint, context pack, SDEP envelope, and attribution
contract should remain stable.
