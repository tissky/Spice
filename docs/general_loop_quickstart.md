# General Decision Loop Quickstart

This quickstart is the current flagship read-only preview for Spice.

It shows the full General Decision Loop without calling an executor, sending an
SDEP request, or persisting state:

```text
signal
-> GenericObservation
-> GeneralDecisionState
-> GenericCandidate
-> GenericPolicyAdapter
-> Decision Card
-> Approval checkpoint
-> Skill resolution
-> Execution Context Pack
-> planned SDEP handoff
-> fixture outcome return
-> state feedback snapshot
```

Use this path when you want to see Spice as a decision layer before real
executor integration.

## Run The Preview

From the repository root:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop --no-bars
```

Expected output starts with:

```text
SPICE DECISION LOOP
read-only preview: decision -> approval -> skill handoff -> outcome snapshot
no executor called | no SDEP sent | no state persisted
```

The output is designed for screenshots, README demos, and short walkthroughs.
It is not a JSON dump.

## Smoke Check

To validate the preview and print the same human-readable loop:

```sh
python examples/decision_hub_demo/smoke_general_loop.py
```

For CI or a quiet local check:

```sh
python examples/decision_hub_demo/smoke_general_loop.py --quiet
```

The smoke check verifies:

- the artifact is JSON-serializable
- required top-level fields are present
- read-only boundary flags are intact
- skill/context IDs line up with the planned SDEP request
- outcome and state feedback attribution line up

It still does not call an executor, send SDEP, or persist state.

## Machine-readable Artifact

For tooling, docs, or a future TUI:

```sh
python examples/decision_hub_demo/run_demo.py --general-full-loop-json --no-bars
```

The JSON artifact includes stable top-level fields for the whole loop:

- `decision_id`
- `trace_ref`
- `selected_candidate_id`
- `approval_id`
- `execution_id`
- `request_id`
- `outcome_id`
- `skill_id`
- `executor_id`
- `context_pack_id`
- `compare_payload`
- `approval_artifact`
- `execution_artifact`
- `outcome_artifact`
- `state_feedback_artifact`

The nested artifacts are retained so the same object can be inspected at either
the full-loop level or the individual stage level.

For the field-level schema and ID consistency contract, see
[`general_loop_artifact_contract.md`](general_loop_artifact_contract.md).

## What This Proves

This preview proves the structure of the loop:

1. Spice can normalize signals into General state.
2. Spice can generate generic candidate decisions.
3. Spice can select a recommendation using decision guidance.
4. Spice can expose why the selected decision won and why others did not.
5. Spice can create an approval checkpoint before execution.
6. Spice can resolve a skill as an execution template / capability hint.
7. Spice can build a compact context pack for the executor handoff.
8. Spice can shape a planned SDEP request without sending it.
9. Spice can receive a fixture outcome and show how it would return to state.

This is the current proof of:

```text
Spice selected, resolved skill, compressed context, planned SDEP handoff.
```

## What This Does Not Do

This preview is intentionally read-only.

It does not:

- call Hermes, Codex, Claude Code, OpenClaw, or any other executor
- send an SDEP `execute.request`
- receive a live SDEP `execute.response`
- write confirmation state
- persist the new General state snapshot
- modify the legacy demo runtime
- prove that Spice makes better decisions than another system

The confirmed approval and SDEP response in this view are local fixtures.

## How To Read The Output

The text output is split into these sections:

```text
0. INPUT SIGNALS
1. GENERAL STATE
2. CANDIDATE DECISIONS
3. SELECTED DECISION
4. WHY NOT OTHERS
5. APPROVAL CHECKPOINT
6. EXECUTION HANDOFF
7. EXECUTION BOUNDARY
8. OUTCOME RETURN
9. STATE FEEDBACK
10. TRACE
```

Key sections:

- **CANDIDATE DECISIONS** shows the available decision objects.
- **SELECTED DECISION** shows the recommended candidate.
- **WHY NOT OTHERS** keeps rejected alternatives visible.
- **APPROVAL CHECKPOINT** shows the human approval boundary.
- **EXECUTION HANDOFF** shows the planned executor, resolved skill, and compact context pack.
- **EXECUTION BOUNDARY** shows the SDEP request is planned but not sent.
- **STATE FEEDBACK** shows a read-only state snapshot after the fixture outcome.

## Boundary Model

Spice does not become the executor in this loop.

```text
Spice        decides and prepares the handoff
Skill        describes the execution template / capability hint
Context Pack carries compact decision-relevant context
SDEP         defines the protocol boundary
Executor     performs the real work later
```

The current preview stops at the boundary. Real executor integration comes
after this contract is stable.

## Related Commands

Inspect the individual stages:

```sh
python examples/decision_hub_demo/run_demo.py --general-decision-card --no-bars
python examples/decision_hub_demo/run_demo.py --general-approval
python examples/decision_hub_demo/run_demo.py --general-execution-plan
python examples/decision_hub_demo/run_demo.py --general-outcome-return
python examples/decision_hub_demo/run_demo.py --general-state-feedback
```

The first command only renders the Decision Card. The later commands show the
read-only boundary stages one by one.

## Next Step

After this quickstart, the next engineering step is the real executor
integration contract:

```text
SDEP execute.request + skill_hint + context_pack
-> executor adapter
-> SDEP execute.response
-> OutcomeRecord
-> persisted General state
```

That work should happen after the read-only artifact contract remains stable.

See [`executor_integration_contract.md`](executor_integration_contract.md) for
the pre-runtime adapter contract.
