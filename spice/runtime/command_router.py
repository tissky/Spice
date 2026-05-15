from __future__ import annotations

from dataclasses import dataclass

COMMAND_ROUTE_SCHEMA_VERSION = "spice.command_route.v1"


@dataclass(frozen=True, slots=True)
class DeterministicCommandRoute:
    raw: str
    command: str
    value: str
    route: str
    known: bool
    requires_llm: bool = False
    schema_version: str = COMMAND_ROUTE_SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw": self.raw,
            "command": self.command,
            "value": self.value,
            "route": self.route,
            "known": self.known,
            "requires_llm": self.requires_llm,
        }


COMMAND_ALIASES: dict[str, str] = {
    "/show": "/details",
    "/detail": "/details",
    "/decision": "/card",
    "/decision-card": "/card",
    "/explain": "/why",
    "/simulation": "/sim",
    "/simulations": "/sim",
    "/raw": "/json",
    "/js": "/json",
    "/source": "/sources",
    "/refs": "/sources",
    "/citations": "/sources",
    "/investigation": "/investigate",
    "/investigations": "/investigate",
    "/consent": "/investigate",
    "/pending-approval": "/pending",
    "/pending-approvals": "/pending",
    "/approval": "/approval",
    "/approvals": "/approvals",
    "/yes": "/approve",
    "/y": "/approve",
    "/no": "/reject",
    "/n": "/reject",
    "/modify": "/refine",
    "/explore": "/refine",
    "/metrics": "/stats",
    "/quit": "/exit",
}

KNOWN_COMMANDS = frozenset(
    {
        "/act",
        "/advise",
        "/approve",
        "/approval",
        "/approvals",
        "/card",
        "/context",
        "/details",
        "/doctor",
        "/dry-run",
        "/execute",
        "/exit",
        "/help",
        "/investigate",
        "/json",
        "/pending",
        "/perceive",
        "/refine",
        "/refresh",
        "/reject",
        "/session",
        "/sim",
        "/sources",
        "/state",
        "/stats",
        "/timeline",
        "/why",
        "/workspace",
    }
)

EXECUTION_REQUEST_COMMANDS = frozenset({"/act", "/execute", "/dry-run"})
FOLLOW_UP_COMMANDS = frozenset({"/refine"})


def route_slash_command(line: str) -> DeterministicCommandRoute:
    raw = line.strip()
    command, value = split_slash_command(raw)
    canonical = COMMAND_ALIASES.get(command, command)
    known = canonical in KNOWN_COMMANDS
    if canonical in EXECUTION_REQUEST_COMMANDS:
        route = "execution_request"
    elif canonical in FOLLOW_UP_COMMANDS:
        route = "follow_up"
    else:
        route = "command"
    return DeterministicCommandRoute(
        raw=raw,
        command=canonical,
        value=value,
        route=route,
        known=known,
        requires_llm=False,
    )


def split_slash_command(line: str) -> tuple[str, str]:
    parts = line.strip().split(maxsplit=1)
    command = parts[0].strip().lower() if parts else ""
    value = parts[1].strip() if len(parts) > 1 else ""
    return command, value
