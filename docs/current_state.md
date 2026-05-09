# Spice Current State Handoff

Last updated: 2026-05-01

This document is the local handoff summary for continuing the Spice work in a new Codex session. The project path is:

```text
/Users/jiadongyu/Desktop/spice_update/Spice-main
```

## Product Direction

Spice is not meant to replace Codex, Hermes, Claude Code, or other agents. It is the decision layer in front of agents:

```text
signals / user intent
-> GeneralDecisionState
-> candidate decisions
-> policy comparison
-> approval
-> skill + context pack
-> SDEP handoff
-> executor
-> outcome
-> state feedback
```

The core positioning:

- Decision first, refine optional, chat no.
- LLMs provide decision material, not final decision authority.
- Executors do the work; Spice makes the decision visible, comparable, approvable, and auditable.
- Perception finds decision moments; it should not directly trigger execution.
- SDEP is the protocol boundary; executor-specific adapters must preserve attribution and approval.

Important user-facing sentence:

> Before your agent acts, Spice makes the decision visible.

## Current High-Level Status

The repo now has a functional local Spice runtime MVP:

- workspace setup
- local JSON store
- manual intent entry
- decision card
- approval flow
- sessions
- dry-run executor
- SDEP subprocess executor
- provider interfaces
- TUI shell
- doctor/config commands
- OpenAI LLM provider foundation

Full test suite was passing after the OpenAI provider work:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

Last observed result:

```text
Ran 451 tests
OK
```

## Completed Major Phases

### Step 1: General Decision Core

Package:

```text
spice/decision/general/
```

Core concepts include:

- GenericObservation
- GeneralDecisionState
- Signal
- Intent
- Commitment
- WorkItem
- Resource
- Capability
- Constraint
- Risk
- OpenLoop
- DecisionTrace / DecisionCheckpoint
- Approval
- OutcomeRecord

Goal achieved: provider events and manual input can normalize into GenericObservation and reduce into GeneralDecisionState without needing DomainPack.

### Step 1.5: Reducer / Observation Apply

Key files:

```text
spice/decision/general/observations.py
spice/decision/general/reducer.py
spice/decision/general/state.py
```

Behavior:

- `apply_generic_observation` and `reduce_generic_observations` return new state snapshots.
- Known observation kinds upsert into typed state families.
- Unknown observations are preserved without being guessed into work items/intents/risks.
- Outcome observations reduce into OutcomeRecord.
- Reducer does not call policy, executor, LLM, SDEP, CLI, runtime, or examples.

### Step 2: Generic Candidate Layer

Key file:

```text
spice/decision/general/candidates.py
```

Contract:

- GeneralDecisionState -> GenericCandidate list.
- No final selection.
- No winner/recommendation.
- No policy score ownership.
- No SDEP/executor protocol binding.
- Blocked candidates are retained.
- `availability_status` is the public field, not policy veto.

Stable action types:

```text
intent.execute
capability.use
item.triage
context.prepare
state.observe_more
artifact.draft
approval.request
user.clarify
time.defer
state.record
item.ignore
task.split
```

### Step 3: Generic Policy Adapter

Key file:

```text
spice/decision/general/policy.py
```

Contract:

- Wraps existing GuidedDecisionPolicy-style scoring/comparison.
- Does not rewrite core policy engine.
- Converts GenericCandidate to policy candidate.
- Reads decision guidance where available.
- Produces GenericPolicyResult, selected candidate, DecisionCheckpoint, optional Approval, compare payload.
- Blocked candidates stay in compare payload but are not selected.
- No executor/SDEP/LLM/runtime side effects.

### Step 4: decision_hub_demo General Path

Key files:

```text
examples/decision_hub_demo/general_adapter.py
examples/decision_hub_demo/general_approval.py
examples/decision_hub_demo/general_execution.py
examples/decision_hub_demo/general_outcome.py
examples/decision_hub_demo/general_state_feedback.py
examples/decision_hub_demo/general_loop.py
examples/decision_hub_demo/run_demo.py
```

Read-only demo path was added:

```bash
python examples/decision_hub_demo/run_demo.py --general-full-loop --no-bars
python examples/decision_hub_demo/run_demo.py --general-full-loop-json --no-bars
```

This path demonstrates:

```text
demo signal
-> GenericObservation
-> GeneralDecisionState
-> GenericCandidate
-> GenericPolicyAdapter
-> Decision Card
-> Approval checkpoint
-> planned SDEP request
-> local SDEP response fixture
-> OutcomeRecord + outcome observation
-> state feedback snapshot
```

Boundary:

- no executor called
- no SDEP request sent
- no state persisted
- confirmed approval and SDEP response are local fixtures

### Step 5: Approval / Execution / Outcome / Feedback

Built in example layer first, then used as contract for runtime:

- 5A General Approval Bridge
- 5B Execution Planning / SDEP Request Adapter
- 5C Outcome Return / SDEP Response Adapter
- 5D Outcome Apply / State Feedback Snapshot

Important semantics:

- Approval must match decision/trace/candidate.
- Execution ID includes decision/trace/candidate/approval/action/target attribution.
- SDEP request is planned first and not sent unless executor bridge sends it.
- Outcome preserves request_id, approval_id, execution_id, protocol_status, task_status.
- State feedback must go through reducer.

### Step 6: Skills + Context Pack

Key files:

```text
spice/executors/skills.py
spice/executors/skill_resolver.py
spice/executors/context_pack.py
```

Implemented:

- CapabilityDescriptor
- SkillDescriptor
- ExecutorDescriptor
- SkillCatalog
- ResolvedSkill
- resolve_skill_for_candidate
- build_execution_context_pack

Important contracts:

- Skill side_effect_class must exactly match candidate required side effect.
- Preferred skill/executor cannot bypass side-effect or confirmation safety.
- Skill instructions/input_schema/output_schema propagate into resolved handoff.
- Context pack is compact and executor-facing, not a full state dump.
- context_pack_id is handoff-level and seeded by decision/trace/candidate/approval/execution/request/skill/executor.

Full loop now says:

```text
Spice selected, resolved skill, compressed context, planned SDEP handoff.
```

### Step 7: Pre-runtime Packaging / Contracts / Smoke

Docs added:

```text
docs/general_loop_quickstart.md
docs/general_loop_artifact_contract.md
docs/executor_integration_contract.md
```

Smoke script:

```text
examples/decision_hub_demo/smoke_general_loop.py
```

Purpose:

- stable artifact contract
- executor integration contract
- smoke preview for read-only full loop

### Step 8: Runtime MVP

Implemented:

#### 8A Workspace + Config

Key files:

```text
spice/runtime/workspace.py
```

Workspace:

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
  executors/
  skills/
```

Setup command:

```bash
python -m spice.entry setup --workspace .
```

#### 8B Local Store

Key file:

```text
spice/runtime/store.py
```

Supports local JSON artifacts:

- state
- runs
- decisions
- approvals
- outcomes
- sessions

#### 8C Manual Intent Runtime Entry

Key file:

```text
spice/runtime/run_once.py
```

Commands:

```bash
python -m spice.entry run --once "..."
python -m spice.entry decide "..."
```

Current runtime loop can be decision-only or full-loop preview depending on mode/flags.

Important fix already done:

- run_id uniqueness uses higher-resolution timestamp/hash.
- `--no-persist` exists.
- artifacts include loop_mode/source/store_paths/state refs.

#### 8D Runtime Loop

`run --once` can show the full Spice decision loop:

```text
State
Candidates
Selected Decision
Why Not Others
Approval
Execution Handoff
Saved Artifacts
```

Still safe by default; execution requires approval and executor bridge.

#### 8E Approval Flow

Commands:

```bash
python -m spice.entry approval list
python -m spice.entry approval approve <approval_id>
python -m spice.entry approval reject <approval_id>
```

Approval syncs run/session metadata.

#### 8F Dry-run Executor Bridge

Command:

```bash
python -m spice.entry execute dry-run <approval_id>
```

Dry run:

- consumes approved approval
- reads planned SDEP request
- creates local SDEP response-shaped payload
- writes outcome artifact
- updates state via reducer
- updates run/session metadata
- does not call a real executor

#### 8G Interactive Shell

Plain shell existed and TUI shell was later added.

#### 8H Session Commands

Session commands implemented:

```bash
python -m spice.entry session list
python -m spice.entry session current
python -m spice.entry session switch <id>
python -m spice.entry session show <id>
python -m spice.entry session resume <id>
python -m spice.entry session resume <id> --start
python -m spice.entry session archive <id>
python -m spice.entry session timeline <id>
python -m spice.entry session search <keyword>
python -m spice.entry session stats
python -m spice.entry session delete <id> --cascade --force
```

#### 8I Provider Interfaces

Key file:

```text
spice/runtime/providers.py
```

Provider contracts:

- PerceptionProvider
- StoreProvider
- ExecutorProvider

Implemented:

- ManualInputProvider
- LocalJsonStoreProvider
- DryRunExecutorProvider
- SDEPSubprocessExecutorProvider
- CodexExecutorProvider

#### 8J SDEP Subprocess Executor Provider

Key files:

```text
spice/runtime/sdep_subprocess_executor.py
spice/runtime/sdep_echo_executor.py
```

Command:

```bash
python -m spice.entry execute sdep <approval_id> \
  --command "python -m spice.runtime.sdep_echo_executor"
```

Also implemented configured executor command:

```bash
python -m spice.entry execute <approval_id>
```

This reads `.spice/config.json`:

- `executor`
- `executor_command`

Boundary:

- SDEP subprocess is real transport.
- Echo executor is fixture, not real Codex/Hermes.
- Attribution is validated before state update.

#### Codex Executor Provider

Key files:

```text
spice/runtime/codex_executor.py
spice/runtime/codex_provider.py
```

Commands:

```bash
python -m spice.entry config set executor codex --workspace /tmp/spice-demo
python -m spice.entry config set executor_command "codex" --workspace /tmp/spice-demo
python -m spice.entry execute <approval_id> --workspace /tmp/spice-demo
```

Explicit override:

```bash
python -m spice.entry execute codex <approval_id> \
  --command "codex" \
  --workspace /tmp/spice-demo
```

Boundary:

- Codex receives only the approved SDEP task/context handoff.
- Codex cannot reselect candidates or rewrite decision attribution.
- The Codex command is configurable and is invoked with `shell=False`.
- Returned stdout/stderr is wrapped into a valid SDEP `execute.response`.

#### Claude Code Executor Provider

Key files:

```text
spice/runtime/claude_code_executor.py
spice/runtime/claude_code_provider.py
```

Commands:

```bash
python -m spice.entry config set executor claude_code --workspace /tmp/spice-demo
python -m spice.entry config set executor_command "claude" --workspace /tmp/spice-demo
python -m spice.entry execute <approval_id> --workspace /tmp/spice-demo
```

Explicit override:

```bash
python -m spice.entry execute claude-code <approval_id> \
  --command "claude" \
  --workspace /tmp/spice-demo
```

Boundary:

- Claude Code receives only the approved SDEP task/context handoff.
- Claude Code cannot reselect candidates or rewrite decision attribution.
- The Claude Code command is configurable and is invoked with `shell=False`.
- Returned stdout/stderr is wrapped into a valid SDEP `execute.response`.

#### Hermes Executor Provider

Key files:

```text
spice/runtime/hermes_executor.py
spice/runtime/hermes_provider.py
```

Commands:

```bash
python -m spice.entry config set executor hermes --workspace /tmp/spice-demo
python -m spice.entry config set executor_command "hermes" --workspace /tmp/spice-demo
python -m spice.entry execute <approval_id> --workspace /tmp/spice-demo
```

Explicit override:

```bash
python -m spice.entry execute hermes <approval_id> \
  --command "hermes" \
  --workspace /tmp/spice-demo
```

Boundary:

- Hermes receives only the approved SDEP task/context handoff.
- Hermes cannot reselect candidates or rewrite decision attribution.
- The Hermes command is configurable and is invoked with `shell=False`.
- Returned stdout/stderr is wrapped into a valid SDEP `execute.response`.

#### 8K Session Command Completion

Completed extra session commands:

- search
- current
- switch
- archive
- timeline
- delete
- stats

## Recent Product/CLI Polish

### spice decide

Command:

```bash
python -m spice.entry decide "What should I do next?"
```

Alias for `run --once`, aligned with user mental model.

### spice execute <approval_id>

Default execution command now reads configured executor from config.

Explicit override still available:

```bash
python -m spice.entry execute dry-run <approval_id>
python -m spice.entry execute sdep <approval_id> --command "..."
```

### Friendly CLI errors

CLI handlers catch common exceptions and print actionable suggestions instead of raw tracebacks.

### README 30-second quickstart

README / README_zh / runtime quickstart were updated during the session.

### Rich-lite Decision Card

Key file:

```text
spice/decision/compare_rich.py
```

Behavior:

- If Rich is installed, render a richer Decision Card.
- If Rich is missing, silently fallback to `render_compare_text`.
- `--rich` flag exists for `run --once` and `decide`.

### doctor / config

Commands:

```bash
python -m spice.entry doctor
python -m spice.entry config show
python -m spice.entry config set <key> <value>
```

Doctor checks:

- Python version
- workspace structure
- config
- decision.md
- state.json
- Rich optional dependency
- LLM provider / API key
- executor / executor_command
- active session
- pending approvals

Config validation:

- no `executor.command` alias; use `executor_command`
- `active_session_id` must exist
- allowed executor values: `dry_run`, `sdep_subprocess`

### TUI Shell

Key files:

```text
spice/runtime/tui/
```

Command:

```bash
python -m spice.entry shell
python -m spice.entry shell --plain
```

TUI plan followed Hermes guidance:

- prompt_toolkit for shell, history, autosuggest, completion
- Rich for panels
- optional dependency fallback
- brand color red
- existing plain shell preserved

Surfaces implemented:

- banner
- decision card
- approval list/details/result
- session/timeline/stats
- doctor
- state
- execution result panel

Latest small TUI fix:

- `/execute <id>` and `/dry-run <id>` now render a Rich execution panel instead of raw `execution.rendered_text`.

## OpenAI Provider Work

Latest completed task before this handoff:

Key files:

```text
spice/llm/providers/openai.py
spice/llm/providers/__init__.py
spice/entry/assist.py
spice/llm/services/domain_advisory.py
spice/llm/services/model_override.py
spice/runtime/workspace.py
spice/runtime/doctor.py
```

Implemented:

- `OpenAILLMProvider(provider_id="openai")`
- uses `OPENAI_API_KEY`
- supports `SPICE_OPENAI_BASE_URL`
- supports optional `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID`
- calls `/chat/completions`
- normalizes auth/rate-limit/response/transport errors
- supports `response_format_hint="json_object"`
- supports `openai:<model>` override
- registers OpenAI provider in assist/domain registries
- workspace config accepts `llm_provider=openai`
- doctor checks OpenAI API key

Test status after this:

```text
Ran 451 tests
OK
```

Important: OpenAI provider is infrastructure only unless the optional runtime flags are enabled. It can now be used by candidate expansion and simulation, both default off.

## Current LLM Positioning

LLM should not own final decisions.

Correct role:

```text
state + user question
-> LLM proposes/enriches candidate material
-> Spice validates schema and constraints
-> GenericPolicyAdapter scores/selects
-> Decision Card explains
-> user approves
-> executor acts
```

Planned LLM modules:

```text
spice/llm/candidate_expander.py
spice/llm/simulation_runner.py
```

The intended chain:

```python
candidates = generate_generic_candidates(state)
if config.llm_candidate_expand:
    llm_candidates = candidate_expander.expand(state, intent, candidates)
    candidates = merge_and_dedupe(candidates, llm_candidates)
if config.llm_simulation:
    candidates = simulation_runner.simulate(state, candidates)
policy_result = policy.evaluate(state, candidates)
```

Constraints:

- Candidate Expander and Simulation are independent modules.
- Do not embed LLM logic inside `generate_generic_candidates`.
- LLM unavailable should not break the user command; fallback to rule candidates.
- But artifact should record fallback reason; doctor should warn.
- LLM output must pass schema validation.
- LLM cannot emit executor-specific action types or SDEP requests.
- Simulation metadata is optional and must not break renderer.
- GenericPolicyAdapter selection remains rule-guided; LLM material can add candidates/simulation, but does not replace policy selection.

Recommended config field names:

```json
{
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "llm_candidate_expand": false,
  "llm_simulation": false
}
```

Note: `llm_model`, `llm_candidate_expand`, and `llm_simulation` now exist. Candidate expansion and simulation both default to `false`.

Users can enable LLM decision material through the workspace config command:

```bash
python -m spice.entry config enable-llm \
  --provider openai \
  --model gpt-4o-mini \
  --workspace /tmp/spice-demo
```

This sets `llm_provider`, `llm_model`, `llm_candidate_expand=true`, and `llm_simulation=true`.
Use `--no-candidate-expand` or `--no-simulation` to keep one feature disabled.
`spice doctor` now reports `llm model`, `llm candidate expansion`, `llm simulation`, API key status, and overall `llm readiness`.

## Provider Roadmap Discussed

LLM providers:

1. OpenAI provider - done
2. Anthropic provider - done
3. DeepSeek provider - done
4. MiMo provider - done

Executor providers:

1. Codex executor - done
2. Claude Code executor - done
3. Hermes executor - done

Perception providers:

1. Poll provider - done
2. Open Chronicle provider

Decision UX:

1. refine command
2. optional explore mode later

Suggested order after OpenAI/Anthropic/DeepSeek/MiMo, candidate expansion, and simulation:

1. Decision Card simulation rendering
2. refine command
3. Codex executor - done
4. Claude Code executor - done
5. Hermes executor - done
6. Poll perception - done
7. Open Chronicle perception - done

If prioritizing user-visible punch, refine can move earlier.

## Perception Decision

Open Chronicle alone is too narrow because it is Mac-first.

Recommended perception plan:

- `manual`: default, already exists
- `poll`: universal provider
- `open_chronicle`: high-end proactive provider

Poll should support:

- URL polling
- command polling
- changed hash -> new observation only

Security boundary:

- URL poll can be normal.
- command poll must be opt-in.
- no `shell=True`.
- timeout required.
- perception only creates observations; it does not approve or execute.

Open Chronicle support now reads the local MCP endpoint:

```bash
python -m spice.entry config set perception_provider open_chronicle --workspace /tmp/spice-demo
python -m spice.entry config set openchronicle_mcp_url "http://127.0.0.1:8742/mcp" --workspace /tmp/spice-demo
python -m spice.entry perceive --provider open_chronicle --once --workspace /tmp/spice-demo
python -m spice.entry perceive --provider open_chronicle --watch --decide-on-change --workspace /tmp/spice-demo
```

It converts Open Chronicle `current_context` and `recent_activity` into
`GenericObservation(kind=signal)`, dedupes by content hash, and can optionally
open a Decision Card. It does not start Open Chronicle, send SDEP, or execute.

## Important Boundaries To Preserve

Do not regress these:

- `spice.decision.general` must not depend on examples or runtime.
- Candidate Layer must not call LLM, executor, SDEP, CLI, or runtime.
- GenericPolicyAdapter must not execute actions.
- Executor adapters must not choose candidates.
- Executors must not rewrite decision_id / trace_ref / candidate_id / approval_id.
- Approval gate cannot be bypassed.
- State updates must go through reducer from observations.
- SDEP remains the protocol envelope.
- Skills are execution templates/capability hints, not independent execution authority.
- Context Pack is compressed execution context, not full state dump.
- TUI/Rich must be optional and fallback silently.

## Important Commands

Setup:

```bash
python -m spice.entry setup --workspace /tmp/spice-demo
```

Decide:

```bash
python -m spice.entry decide "I have a failing test and a pending PR review" --workspace /tmp/spice-demo --rich
python -m spice.entry refine "Consider rollback first" --workspace /tmp/spice-demo --rich
```

Act / approval-producing run:

```bash
python -m spice.entry run --once "Fix the failing test" --act --workspace /tmp/spice-demo
```

Approval:

```bash
python -m spice.entry approval list --workspace /tmp/spice-demo
python -m spice.entry approval approve <approval_id> --workspace /tmp/spice-demo
python -m spice.entry approval reject <approval_id> --workspace /tmp/spice-demo
```

Execute via config:

```bash
python -m spice.entry execute <approval_id> --workspace /tmp/spice-demo
```

Configure SDEP subprocess executor:

```bash
python -m spice.entry config set executor sdep_subprocess --workspace /tmp/spice-demo
python -m spice.entry config set executor_command "python -m spice.runtime.sdep_echo_executor" --workspace /tmp/spice-demo
```

Doctor:

```bash
python -m spice.entry doctor --workspace /tmp/spice-demo
python -m spice.entry config show --workspace /tmp/spice-demo
python -m spice.entry config enable-llm --provider openai --model gpt-4o-mini --workspace /tmp/spice-demo
```

TUI:

```bash
python -m spice.entry shell --workspace /tmp/spice-demo
python -m spice.entry shell --workspace /tmp/spice-demo --plain
```

Session:

```bash
python -m spice.entry session list --workspace /tmp/spice-demo
python -m spice.entry session current --workspace /tmp/spice-demo
python -m spice.entry session timeline session.default --workspace /tmp/spice-demo
```

## Key Test Commands

Targeted after OpenAI/Anthropic/DeepSeek/MiMo providers, candidate expander, and simulation runner:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_llm_simulation_runner \
  tests.test_llm_candidate_expander \
  tests.test_llm_core_provider \
  tests.test_llm_client_param_precedence \
  tests.test_entry_init_domain_assist \
  tests.test_domain_llm_advisory \
  tests.test_runtime_workspace \
  tests.test_runtime_doctor \
  tests.test_entry_setup
```

Full:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

If tests fail with `PermissionError` trying to create temp dirs under `/Users/jiadongyu/Desktop/spice_update/Spice-main`, it is likely Codex sandboxing. Rerun with appropriate permission escalation.

## Known Caveats

- OpenAI provider is implemented and can be used by optional candidate expansion/simulation.
- Anthropic provider is implemented and can be used by optional candidate expansion/simulation.
- DeepSeek provider is implemented and can be used by optional candidate expansion/simulation.
- MiMo provider is implemented and can be used by optional candidate expansion/simulation.
- LLM Candidate Expander is implemented, default off, schema-validated, and falls back to rule candidates on LLM errors.
- LLM Simulation Runner is implemented, default off, schema-validated, and adds optional candidate metadata without changing policy scoring.
- `llm_model`, `llm_candidate_expand`, and `llm_simulation` config fields exist.
- Decision Card rendering surfaces optional simulation metadata in plain and Rich-lite output.
- GenericPolicyAdapter scoring now includes urgency, effort, impact, and preference-alignment dimensions in addition to the original outcome/risk/reversibility/confidence dimensions.
- `refine` command is implemented for explicit decision feedback: it loads the latest run, adds LLM/manual refinement candidates, re-scores, and saves a new run artifact.
- Perception providers beyond manual are not implemented yet.
- Codex executor adapter is implemented.
- Claude Code executor adapter is implemented.
- Hermes executor adapter is implemented.
- TUI is phase-one; it is useful but not a complete Hermes-level interface.
- Plain shell and TUI shell do not have perfect command parity because existing plain shell was intentionally preserved.

## Suggested Next Task

If continuing exactly from the latest state, the next best implementation step is probably:

```text
Poll perception provider or Open Chronicle perception provider
```

Completed decision-flow sequence:

1. Added `llm_simulation` config field.
2. Added `spice/llm/simulation_runner.py` with candidate simulation metadata validation.
3. Wired optional simulation after candidate expansion and before GenericPolicyAdapter.
4. Recorded fallback metadata when LLM is unavailable.
5. Kept default flag false.

Keep LLM-generated candidates and simulation as validated decision material; do not let LLM selection replace GenericPolicyAdapter.
