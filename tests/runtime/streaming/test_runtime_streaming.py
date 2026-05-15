from __future__ import annotations

import json
import unittest

from spice.runtime import (
    SPICE_STREAM_EVENT_SCHEMA_VERSION,
    SpiceStreamEvent,
    build_stream_event,
    stream_artifact_ref_event,
    stream_error_event,
    stream_execution_output_event,
    stream_response_delta_event,
    stream_response_done_event,
    stream_status_event,
)


class RuntimeStreamingEventTests(unittest.TestCase):
    def test_stream_event_round_trip_is_json_serializable(self) -> None:
        event = SpiceStreamEvent(
            event_type="response_delta",
            text="I would start with A.",
            artifact_refs=[{"kind": "decision", "id": "decision.test", "path": ".spice/decisions/test.json"}],
            metadata={"unit": "block", "index": 1},
        )

        payload = event.to_payload()
        restored = SpiceStreamEvent.from_payload(json.loads(json.dumps(payload)))

        self.assertEqual(payload["schema_version"], SPICE_STREAM_EVENT_SCHEMA_VERSION)
        self.assertEqual(restored.event_type, "response_delta")
        self.assertEqual(restored.text, "I would start with A.")
        self.assertEqual(restored.artifact_refs[0]["id"], "decision.test")
        self.assertEqual(restored.metadata["unit"], "block")

    def test_invalid_event_type_is_rejected(self) -> None:
        event = SpiceStreamEvent(event_type="tool_call", text="not supported yet")

        with self.assertRaisesRegex(ValueError, "stream event_type"):
            event.to_payload()

    def test_build_stream_event_normalizes_refs_and_metadata(self) -> None:
        event = build_stream_event(
            "artifact_ref",
            text="Artifacts are ready.",
            artifact_refs=[
                {"kind": "run", "id": "run.test"},
                "ignored",  # type: ignore[list-item]
            ],
            metadata={"visible": True},
        )

        self.assertEqual(event.event_type, "artifact_ref")
        self.assertEqual(event.artifact_refs, [{"kind": "run", "id": "run.test"}])
        self.assertEqual(event.metadata, {"visible": True})

    def test_stream_event_helpers_create_expected_contract_events(self) -> None:
        status = stream_status_event("Thinking through the decision...", detail="openrouter/model")
        delta = stream_response_delta_event("First paragraph.", unit="block", index=0)
        done = stream_response_done_event("Full response.", chunk_count=1)
        output = stream_execution_output_event("pytest passed", stream_name="stdout")
        refs = stream_artifact_ref_event([{"kind": "run", "id": "run.test"}])
        error = stream_error_event("Composer failed.", reason="invalid output")

        self.assertEqual(status.to_payload()["event_type"], "status")
        self.assertEqual(status.metadata["label"], "Thinking through the decision...")
        self.assertEqual(status.metadata["detail"], "openrouter/model")
        self.assertEqual(delta.metadata["unit"], "block")
        self.assertEqual(delta.metadata["index"], 0)
        self.assertEqual(done.metadata["chunk_count"], 1)
        self.assertEqual(output.event_type, "execution_output")
        self.assertEqual(output.metadata["stream"], "stdout")
        self.assertEqual(refs.artifact_refs[0]["id"], "run.test")
        self.assertEqual(error.event_type, "error")
        self.assertEqual(error.metadata["reason"], "invalid output")


if __name__ == "__main__":
    unittest.main()
