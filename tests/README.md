# Spice Tests

Run the full suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

## Layout

- `decision/` - decision guidance, comparison, general decision policy, and scoring tests.
- `runtime/` - runtime shell, routing, perception wiring, composers, execution, sessions, streaming, setup, and core runtime tests.
- `llm/` - LLM provider, adapter, proposal, simulation, parser, and advisory tests.
- `perception/` - workspace, URL, delegated, OpenChronicle, and evidence-context perception tests.
- `executors/` - executor discovery/runtime/skills, CLI executor, permission, and context-pack tests.
- `entry/` - CLI entrypoint, setup, quickstart, spec, and init-domain tests.
- `protocols/` - SDEP protocol, schema, example payload, executor, and wrapper tests.
- `memory/` - episode memory, runtime writeback, memory context, and continuation resolver tests.
- `demos/` - tests for demo/example code under `examples/`.
- `replay/` - replay and shadow-runner tests.

## Conventions

- Keep test filenames in `test_*.py` form so `unittest discover` can find them.
- Add an `__init__.py` to new test directories.
- Avoid naming test directories after top-level packages such as `examples`; that can shadow the real package during discovery.
- If I've forgotten any tests, please feel free to add them via pull requests.
- When a test needs the repository root, resolve it robustly instead of assuming a fixed `Path(__file__).parents[...]` depth, because tests live in nested category directories.
