# Runtime Quickstart

This is the current product-facing path for trying Spice in a local project.

It starts from a manual intent, turns it into General state, renders a Decision
Card, records the run in a local session, and shows a read-only full-loop
handoff preview.

It does not call an executor, send an SDEP request, call an LLM, or perform a
real side effect.

## 1. Initialize Workspace

```sh
python -m spice.entry setup --workspace .
```

This creates:

```text
.spice/
  config.json
  decision.md
  state/state.json
  sessions/
  runs/
  decisions/
  approvals/
  outcomes/
  perceptions/
  executors/
  skills/
```

`decision.md` is the editable decision preference layer.

For the polished interactive shell, install the optional TUI extra:

```sh
pip install "spice-runtime[tui]"
```

## 2. Run One Manual Intent

You can also enter the Rich/prompt_toolkit shell:

```sh
python -m spice.entry shell --workspace .
```

Use `--plain` to force the existing text shell, which is useful for pipes and
minimal environments.

```sh
python -m spice.entry run \
  --once "Review this repo and suggest the safest next action" \
  --workspace . \
  --no-bars
```

Use run intent modes when you want stricter behavior:

```sh
python -m spice.entry run --once "What should I do next?" --workspace . --advise
python -m spice.entry run --once "Fix the failing test" --workspace . --act
```

- `--advise` stops at the Decision Card.
- `--act` requires an execution-handoff candidate and creates an approval
  checkpoint before any dry-run or real executor handoff.
- default mode is `auto`, which preserves the normal full-loop preview behavior.

This runs the runtime decision loop:

```text
manual intent
-> GenericObservation
-> GeneralDecisionState
-> GenericCandidate
-> GenericPolicyAdapter
-> Decision Card
-> approval fixture
-> skill resolution
-> compressed context pack
-> planned SDEP execute.request
-> local SDEP execute.response fixture
-> outcome observation
-> state feedback snapshot
-> local JSON artifacts
-> session.default
```

The run writes local artifacts under `.spice/`:

```text
.spice/runs/
.spice/decisions/
.spice/approvals/   # only when approval is pending
.spice/sessions/
```

By default it also updates `.spice/state/state.json`.

To preview without updating active state:

```sh
python -m spice.entry run \
  --once "Review this repo and suggest the safest next action" \
  --workspace . \
  --no-persist
```

The run and decision artifacts are still saved.

To stop after the Decision Card without the handoff preview:

```sh
python -m spice.entry run \
  --once "Review this repo and suggest the safest next action" \
  --workspace . \
  --decision-only \
  --no-bars
```

## 3. Run An Interactive Session

Use `spice run` without `--once` to enter the local runtime shell:

```sh
python -m spice.entry run --workspace .
```

The shell keeps one decision-loop session open:

```text
Spice Agent
session: session.default
executor: dry_run
permission: confirm_before_execution

> /act Fix the failing test
> /approvals
> /approve <approval_id>
> /dry-run <approval_id>
> /exit
```

Useful commands:

- type any intent to run the default decision loop
- `/act <intent>` creates an execution-handoff approval checkpoint
- `/advise <intent>` stops at the Decision Card
- `/approvals` lists local approval checkpoints
- `/approve <approval_id>` approves a checkpoint
- `/reject <approval_id>` rejects a checkpoint
- `/details <approval_id>` shows approval details
- `/dry-run <approval_id>` crosses the local dry-run executor bridge
- `/session` shows the current session summary

The shell still uses local runtime boundaries: it does not call a real executor,
does not send SDEP over a transport, and only runs the local dry-run bridge when
you explicitly type `/dry-run`.

## 4. Inspect The Session

```sh
python -m spice.entry session list --workspace .
python -m spice.entry session show session.default --workspace .
python -m spice.entry session resume session.default --workspace .
```

The session is a decision-loop window, not a chat transcript. It tracks:

- run ids
- decision ids
- pending approvals
- active state reference
- last decision reference

To resume directly into the same interactive shell:

```sh
python -m spice.entry session resume session.default --workspace . --start
```

This uses the same local state and session id. It does not replay old runs or
call an executor by itself; it only reopens the session window for the next
intent or approval command.

## 5. Inspect And Resolve Approvals

Approvals are local checkpoints. Resolving one updates the approval artifact and
session pending list, but it does not call an executor or send SDEP.

```sh
python -m spice.entry approval list --workspace .
python -m spice.entry approval show <approval_id> --workspace .
python -m spice.entry approval approve <approval_id> --workspace .
python -m spice.entry approval reject <approval_id> --workspace .
```

Use `--json` on these commands when another tool needs the structured artifact.

## 6. Pull External Signals With Poll Perception

Poll perception is the first universal proactive signal source. It pulls a URL
or a command output, hashes the content, and only creates a new
`GenericObservation(kind=signal)` when the source changes.

URL polling is cross-platform and uses the Python standard library:

```sh
python -m spice.entry perceive \
  --workspace . \
  --poll-url "https://ci.example.com/status"
```

Command polling is also supported, but it is explicit opt-in and uses
`shell=False`:

```sh
python -m spice.entry perceive \
  --workspace . \
  --poll-command "gh pr list --json state,title" \
  --allow-command-poll
```

To make poll the workspace default:

```sh
python -m spice.entry config set perception_provider poll --workspace .
python -m spice.entry config set perception_poll_url "https://ci.example.com/status" --workspace .
```

Then run:

```sh
python -m spice.entry perceive --workspace .
```

Artifacts are saved under `.spice/perceptions/`, and dedupe state is stored in
`.spice/perceptions/poll_state.json`.

Boundary:

- perception only creates observations and updates General state
- it does not create a decision by itself
- it does not approve or execute anything
- it does not send SDEP

To let changed signals open a Decision Card, opt in explicitly:

```sh
python -m spice.entry perceive \
  --workspace . \
  --poll-url "https://ci.example.com/status" \
  --decide-on-change
```

This still stops before execution:

```text
poll changed
-> GenericObservation
-> General state update
-> Decision Card + pending approval
-> stop
```

The trigger can also be configured for the workspace:

```sh
python -m spice.entry config set perception_trigger_mode decision_on_change --workspace .
```

Default trigger mode is `state_only`.

## 7. Pull Desktop Context With Open Chronicle

Open Chronicle is the Mac-first perception provider. Spice reads it through the
local Open Chronicle MCP endpoint and converts `current_context` /
`recent_activity` into `GenericObservation(kind=signal)`.

Install and start Open Chronicle first:

```sh
git clone https://github.com/Einsia/OpenChronicle.git
cd OpenChronicle
bash install.sh
openchronicle start
openchronicle status
```

Configure Spice:

```sh
python -m spice.entry config set perception_provider open_chronicle --workspace .
python -m spice.entry config set openchronicle_mcp_url "http://127.0.0.1:8742/mcp" --workspace .
python -m spice.entry config set openchronicle_since_minutes 15 --workspace .
python -m spice.entry config set openchronicle_context_limit 5 --workspace .
```

Run one perception pass:

```sh
python -m spice.entry perceive --provider open_chronicle --once --workspace .
```

Or proactively open a Decision Card when the desktop context changes:

```sh
python -m spice.entry perceive \
  --provider open_chronicle \
  --watch \
  --decide-on-change \
  --workspace .
```

Boundary:

- Spice does not start or manage the Open Chronicle daemon
- Spice reads Open Chronicle MCP output only
- changed context can create a Decision Card and pending approval
- no approval, SDEP request, or executor call happens automatically

If the endpoint is down, run:

```sh
openchronicle start
```

Doctor will report this as a warning:

```text
open chronicle ....... warn (MCP not reachable)
```

## 8. Dry-run An Approved Handoff

After an approval is approved, you can cross the execution boundary through the
local dry-run executor bridge:

```sh
python -m spice.entry execute <approval_id> --workspace .
```

This reads `.spice/config.json` and dispatches to the configured executor. The
default executor is `dry_run`, which does not call a real executor and does not
send SDEP over a transport. It uses the planned SDEP `execute.request`, returns
a local SDEP `execute.response` shape, writes an outcome artifact, and reduces
the outcome observation back into `.spice/state/state.json`.

The dry-run artifact records:

- `dry_run_executor_called=true`
- `real_executor_called=false`
- `sdep_request_sent=false`
- `executed=false`
- `state_updated=true`

Use `--json` to inspect the exact response/outcome/state attribution.

You can still explicitly force the dry-run bridge:

```sh
python -m spice.entry execute dry-run <approval_id> --workspace .
```

## 8. Send To An SDEP Subprocess Executor

The first real transport bridge is the SDEP subprocess executor provider. It
sends the approved planned SDEP `execute.request` to a local subprocess over
stdin and expects a valid SDEP `execute.response` on stdout.

Use the bundled echo executor fixture to verify the bridge:

```sh
python -m spice.entry run \
  --once "Fix the failing test" \
  --act \
  --workspace . \
  --no-bars

python -m spice.entry approval list --workspace .
python -m spice.entry approval approve <approval_id> --workspace .

python -m spice.entry execute sdep <approval_id> \
  --command "python -m spice.runtime.sdep_echo_executor" \
  --workspace .
```

The echo executor is a subprocess transport fixture, not Codex, Hermes, or
Claude Code. It proves the SDEP bridge works:

```text
approved approval
-> planned SDEP execute.request
-> subprocess stdin
-> SDEP execute.response from stdout
-> OutcomeRecord
-> General state update
```

Boundary:

- the executor cannot reselect candidates
- approval is required before sending the request
- attribution must match decision_id / trace_ref / candidate_id / approval_id
- `shell=True` is not used
- executor-specific bridges use the same SDEP boundary

Use `--json` to inspect the exact request, response, outcome, and state refs.

## 9. Execute With Codex

The first executor-specific bridge is Codex over the same SDEP subprocess
boundary. Spice still selects the decision, requires approval, builds the skill
handoff, and validates the SDEP response. Codex only receives the approved task
and compressed context pack.

Configure the workspace:

```sh
python -m spice.entry config set executor codex --workspace .
python -m spice.entry config set executor_command "codex exec --skip-git-repo-check -" --workspace .
```

Then approve and execute:

```sh
python -m spice.entry run \
  --once "Fix the failing test" \
  --act \
  --workspace . \
  --no-bars

python -m spice.entry approval approve <approval_id> --workspace .
python -m spice.entry execute <approval_id> --workspace .
```

You can also override explicitly:

```sh
python -m spice.entry execute codex <approval_id> \
  --command "codex exec --skip-git-repo-check -" \
  --workspace .
```

The Codex bridge is still SDEP-shaped:

```text
approved approval
-> planned SDEP execute.request
-> Spice Codex SDEP endpoint
-> Codex command receives task/context on stdin
-> SDEP execute.response
-> OutcomeRecord
-> General state update
```

The Codex command is configurable because local Codex installations differ.
Setup defaults to `codex exec --skip-git-repo-check -`, which is the non-interactive
stdin-based Codex path Spice needs. You can still set `executor_command` to a
wrapper script that invokes your preferred Codex CLI mode. Spice does not use `shell=True`, and Codex cannot
re-decide the candidate or rewrite Spice attribution.

## 10. Execute With Claude Code

Claude Code uses the same executor bridge shape as Codex. Spice sends only the
approved task and compressed context through SDEP, and the configured Claude
Code command returns a response that Spice wraps into outcome/state feedback.

Configure the workspace:

```sh
python -m spice.entry config set executor claude_code --workspace .
python -m spice.entry config set executor_command "claude -p" --workspace .
```

Then approve and execute:

```sh
python -m spice.entry run \
  --once "Fix the failing test" \
  --act \
  --workspace . \
  --no-bars

python -m spice.entry approval approve <approval_id> --workspace .
python -m spice.entry execute <approval_id> --workspace .
```

Explicit override:

```sh
python -m spice.entry execute claude-code <approval_id> \
  --command "claude -p" \
  --workspace .
```

Boundary:

- Claude Code receives a Spice-approved task/context prompt on stdin.
- It cannot reselect candidates or bypass approval.
- Spice validates SDEP attribution before writing the outcome.
- `shell=True` is not used.

## 11. Execute With Hermes

Hermes also uses the same SDEP subprocess executor boundary. Spice stays the
decision layer; Hermes receives only the approved task and compressed context.

Configure the workspace:

```sh
python -m spice.entry config set executor hermes --workspace .
python -m spice.entry config set executor_command "hermes chat -Q" --workspace .
```

Then approve and execute:

```sh
python -m spice.entry run \
  --once "Fix the failing test" \
  --act \
  --workspace . \
  --no-bars

python -m spice.entry approval approve <approval_id> --workspace .
python -m spice.entry execute <approval_id> --workspace .
```

Explicit override:

```sh
python -m spice.entry execute hermes <approval_id> \
  --command "hermes chat -Q" \
  --workspace .
```

Boundary:

- Hermes receives a Spice-approved task/context prompt on stdin.
- It cannot reselect candidates or bypass approval.
- Spice validates SDEP attribution before writing the outcome.
- `shell=True` is not used.

## 12. Understand The Full-loop Preview

The default `run --once` path includes a read-only preview of:

```text
Decision Card
-> approval fixture
-> skill resolution
-> compressed context pack
-> planned SDEP execute.request
-> local SDEP execute.response fixture
-> outcome observation
-> state feedback snapshot
```

The preview keeps these boundaries:

- `executor_called=false`
- `sdep_request_sent=false`
- `executed=false`
- full-loop feedback `persisted=false`

The active state persistence still follows the normal `run --once` mode:

- default: active state is persisted
- `--no-persist`: active state is not updated

## 13. Runtime Provider Contracts

The local runtime now has replaceable provider contracts, with conservative
default implementations:

```text
PerceptionProvider -> GenericObservation[]
StoreProvider      -> local artifacts and active state
ExecutorProvider   -> SDEP-shaped execution boundary
```

Current defaults:

- `ManualInputProvider`: converts `spice run` text into General observations.
- `PollPerceptionProvider`: polls URL or command sources and writes changed
  outputs as `GenericObservation(kind=signal)` without triggering decisions or
  execution.
- `LocalJsonStoreProvider`: stores state, runs, decisions, approvals, outcomes,
  perceptions, and sessions under `.spice/`.
- `DryRunExecutorProvider`: consumes an approved local checkpoint and returns a
  local SDEP `execute.response` shape without calling a real executor.
- `SDEPSubprocessExecutorProvider`: sends the planned SDEP request to a local
  subprocess and validates the returned SDEP response before writing outcome
  state.
- `CodexExecutorProvider`: wraps a configured Codex command behind the same
  SDEP request/response and attribution contract.
- `ClaudeCodeExecutorProvider`: wraps a configured Claude Code command behind
  the same SDEP request/response and attribution contract.
- `HermesExecutorProvider`: wraps a configured Hermes command behind the same
  SDEP request/response and attribution contract.

These providers are contracts for later GitHub, OpenChronicle, Hermes, Codex,
or custom SDEP integrations. The current defaults remain local and deterministic.

## JSON Mode

For tooling or tests:

```sh
python -m spice.entry run \
  --once "Review this repo and suggest the safest next action" \
  --workspace . \
  --json
```

The artifact includes:

- `run_id`
- `session_id`
- `decision_id`
- `trace_ref`
- `selected_candidate_id`
- `compare_payload`
- `full_loop_preview`
- `skill_id`
- `context_pack_id`
- `execution_id`
- `request_id`
- `outcome_id`
- `store_paths`

## Current Boundary

This runtime path is intentionally pre-executor.

Spice currently:

- makes the decision visible
- records the run and session locally
- resolves a skill as an execution template / capability hint
- builds a compact context pack
- shapes a planned SDEP handoff

The executor integration step comes after this runtime contract.
