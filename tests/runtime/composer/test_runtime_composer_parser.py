from __future__ import annotations

import json
import unittest

from spice.runtime.composer_parser import parse_composer_response_text


class RuntimeComposerParserTests(unittest.TestCase):
    def test_parses_standard_response_json(self) -> None:
        self.assertEqual(parse_composer_response_text('{"response": "Use A."}'), "Use A.")

    def test_parses_markdown_fenced_json(self) -> None:
        raw = "```json\n{\"message\": \"Use A, then check the signal.\"}\n```"
        self.assertEqual(parse_composer_response_text(raw), "Use A, then check the signal.")

    def test_supports_top_level_aliases(self) -> None:
        for key in ("message", "answer", "text", "content"):
            with self.subTest(key=key):
                self.assertEqual(parse_composer_response_text(json.dumps({key: "A is the safer path."})), "A is the safer path.")

    def test_accepts_plain_natural_language(self) -> None:
        raw = "I would start with state-as-context because it improves every later decision."
        self.assertEqual(parse_composer_response_text(raw), raw)

    def test_rejects_empty_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            parse_composer_response_text("   ")

    def test_rejects_structured_dump_without_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing response text"):
            parse_composer_response_text('{"decision_id": "decision.x"}')

    def test_rejects_malformed_structured_dump(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured data"):
            parse_composer_response_text('{"response": "unterminated"')

    def test_rejects_overly_long_plain_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "too long"):
            parse_composer_response_text("x" * 2401)


if __name__ == "__main__":
    unittest.main()
