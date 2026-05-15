from __future__ import annotations

import unittest

from spice.runtime.command_router import route_slash_command


class RuntimeCommandRouterTests(unittest.TestCase):
    def test_routes_known_deterministic_commands_without_llm(self) -> None:
        for raw, command, route, value in [
            ("/details", "/details", "command", ""),
            ("/card", "/card", "command", ""),
            ("/why", "/why", "command", ""),
            ("/sim", "/sim", "command", ""),
            ("/json", "/json", "command", ""),
            ("/sources --json", "/sources", "command", "--json"),
            ("/execute approval.test", "/execute", "execution_request", "approval.test"),
            ("/refine make it safer", "/refine", "follow_up", "make it safer"),
            ("/pending", "/pending", "command", ""),
            ("/context --json", "/context", "command", "--json"),
            ("/workspace --json", "/workspace", "command", "--json"),
            ("/help", "/help", "command", ""),
        ]:
            routed = route_slash_command(raw)
            self.assertTrue(routed.known, raw)
            self.assertEqual(routed.command, command)
            self.assertEqual(routed.route, route)
            self.assertEqual(routed.value, value)
            self.assertFalse(routed.requires_llm)

    def test_normalizes_command_aliases(self) -> None:
        self.assertEqual(route_slash_command("/show").command, "/details")
        self.assertEqual(route_slash_command("/decision-card").command, "/card")
        self.assertEqual(route_slash_command("/explain").command, "/why")
        self.assertEqual(route_slash_command("/simulation").command, "/sim")
        self.assertEqual(route_slash_command("/raw").command, "/json")
        self.assertEqual(route_slash_command("/source").command, "/sources")
        self.assertEqual(route_slash_command("/refs").command, "/sources")
        self.assertEqual(route_slash_command("/modify lower risk").command, "/refine")
        self.assertEqual(route_slash_command("/quit").command, "/exit")

    def test_unknown_command_is_still_deterministic(self) -> None:
        routed = route_slash_command("/unknown value")

        self.assertFalse(routed.known)
        self.assertEqual(routed.command, "/unknown")
        self.assertEqual(routed.value, "value")
        self.assertEqual(routed.route, "command")
        self.assertFalse(routed.requires_llm)


if __name__ == "__main__":
    unittest.main()
