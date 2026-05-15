from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from unittest.mock import patch

from spice.runtime import LocalJsonStore, run_once, setup_workspace
from spice.runtime.context_debug import (
    compile_sources_debug_payload,
    compile_workspace_debug_payload,
    compile_workspace_decision_context_payload,
    render_sources_debug_text,
    render_workspace_debug_text,
)
from spice.runtime.refine import refine_decision
from spice.runtime.url_perception import run_runtime_url_perception_step
from spice.runtime.workspace import load_workspace_memory_provider


NOW = datetime(2026, 5, 12, 6, 30, tzinfo=timezone.utc)


class _FakeURLResponse:
    def __init__(self, *, url: str, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        self._url = url
        self._body = body.encode("utf-8")
        self.headers = {"content-type": content_type}
        self.status = 200
        self.code = 200

    def __enter__(self) -> "_FakeURLResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, _: int = -1) -> bytes:
        return self._body


def _request_url(request: object) -> str:
    return str(getattr(request, "full_url", request))


class RuntimeURLPerceptionStepTests(unittest.TestCase):
    def test_runtime_url_perception_writes_artifact_memory_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)

            def fake_urlopen(request: object, timeout: float = 0.0) -> _FakeURLResponse:
                return _FakeURLResponse(
                    url=_request_url(request),
                    body="External design doc says URL perception feeds decision context.",
                )

            with patch("spice.perception.url.urllib.request.urlopen", side_effect=fake_urlopen):
                result = run_runtime_url_perception_step(
                    project_root=tmp_dir,
                    text="Use https://example.com/design for context.",
                    query="design context",
                    trigger="new_decision",
                    store=store,
                    now=NOW,
                )

            self.assertEqual(result.status, "written")
            self.assertTrue(result.path and result.path.exists())
            self.assertEqual(result.context["source"], "url_perception")
            self.assertEqual(result.context["perception_id"], result.artifact["perception_id"])
            self.assertEqual(result.context["documents"][0]["url"], "https://example.com/design")
            self.assertIn("URL perception feeds", result.context["facts"][0]["text"])
            saved = store.load_perception(result.artifact["perception_id"])
            self.assertEqual(saved["perception_id"], result.artifact["perception_id"])
            self.assertEqual(result.memory_writeback["namespace"], "general.url_perception")

            provider = load_workspace_memory_provider(tmp_dir)
            memory_records = provider.query(namespace="general.url_perception", limit=-1)
            self.assertEqual(len(memory_records), 1)
            memory_record = memory_records[0]
            self.assertEqual(memory_record["perception_id"], result.artifact["perception_id"])
            self.assertEqual(memory_record["documents"]["read"][0]["url"], "https://example.com/design")
            self.assertNotIn("snippets", memory_record)
            self.assertNotIn("text", memory_record["snippet_refs"][0])

    def test_runtime_url_perception_skips_when_no_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_runtime_url_perception_step(
                project_root=tmp_dir,
                text="No link here.",
                trigger="follow_up",
                now=NOW,
            )

            self.assertFalse(result.requested)
            self.assertEqual(result.status, "skipped")
            self.assertEqual(result.error, "no_urls_found")

    def test_run_once_accepts_url_context_for_decision_and_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            url_context = {
                "source": "url_perception",
                "perception_id": "url.test",
                "summary": "The linked PR proposes URL perception.",
                "urls": ["https://example.com/pr"],
                "facts": [
                    {
                        "text": "The PR adds URL perception artifacts.",
                        "source_url": "https://example.com/pr",
                    }
                ],
            }
            url_perception = {
                "perception_id": "url.test",
                "summary": "The linked PR proposes URL perception.",
            }

            result = run_once(
                "Based on this PR, what next?",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                url_context=url_context,
                url_perception=url_perception,
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"]["url_context"]
            simulation_ctx = result.artifact["compiled_context"]["simulation_context"]["url_context"]
            self.assertEqual(decision_ctx["source"], "url_perception")
            self.assertEqual(simulation_ctx["perception_id"], "url.test")
            self.assertEqual(result.artifact["url_context"]["perception_id"], "url.test")
            self.assertTrue(result.artifact["evidence_context"]["url"]["present"])
            self.assertEqual(result.artifact["evidence_context"]["url"]["perception_id"], "url.test")
            self.assertEqual(result.artifact["store_paths"]["url_perception"], ".spice/perceptions/url.test.json")
            turn = LocalJsonStore.from_project_root(tmp_dir).load_conversation_turn(
                result.artifact["conversation_turn_id"]
            )
            self.assertEqual(turn["artifact_refs"]["url_perception"], ".spice/perceptions/url.test.json")
            self.assertEqual(turn["metadata"]["url_context"]["perception_id"], "url.test")
            self.assertEqual(turn["metadata"]["evidence_context"]["url"]["perception_id"], "url.test")

            debug_context = compile_workspace_decision_context_payload(project_root=tmp_dir)
            self.assertEqual(debug_context["url_context"]["perception_id"], "url.test")
            debug_payload = compile_workspace_debug_payload(project_root=tmp_dir)
            self.assertEqual(debug_payload["url_context"]["perception_id"], "url.test")
            rendered = render_workspace_debug_text(debug_payload)
            self.assertIn("URL perception:", rendered)
            self.assertIn("url.test", rendered)

    def test_sources_debug_payload_includes_url_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            url_context = {
                "source": "url_perception",
                "perception_id": "url.sources",
                "summary": "The linked design doc proposes URL perception.",
                "urls": ["https://example.com/design"],
                "documents": [
                    {
                        "url": "https://example.com/design",
                        "title": "Design Doc",
                        "source_type": "web_page",
                    }
                ],
                "facts": [
                    {
                        "text": "The design doc says URL perception should stay read-only.",
                        "source_url": "https://example.com/design",
                    }
                ],
                "snippets": [
                    {
                        "url": "https://example.com/design",
                        "title": "Design Doc",
                        "text": "URL perception should stay read-only.",
                    }
                ],
            }
            url_perception = {
                **url_context,
                "trigger": "test",
                "query": "linked design doc",
                "urls_skipped": [{"url": "https://127.0.0.1/private", "reason": "private_host"}],
            }
            store.save_perception("url.sources", url_perception)

            run_once(
                "Based on the linked doc, what next?",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                url_context=url_context,
                url_perception=url_perception,
            )

            payload = compile_sources_debug_payload(project_root=tmp_dir)
            rendered = render_sources_debug_text(payload)

            self.assertEqual(payload["status"], "available")
            self.assertEqual(payload["url"]["perception_id"], "url.sources")
            self.assertTrue(payload["evidence_context"]["url"]["present"])
            self.assertEqual(payload["evidence_context"]["url"]["perception_id"], "url.sources")
            self.assertEqual(payload["url"]["documents"][0]["url"], "https://example.com/design")
            self.assertEqual(payload["url"]["snippets"][0]["url"], "https://example.com/design")
            self.assertEqual(payload["url"]["urls_skipped"][0]["reason"], "private_host")
            self.assertIn("URL sources: Spice fetched directly", rendered)
            self.assertIn("URLs read:", rendered)
            self.assertIn("URL snippets:", rendered)
            self.assertIn("url.sources", rendered)

    def test_refine_accepts_url_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW, full_loop_preview=False)
            url_context = {
                "source": "url_perception",
                "perception_id": "url.refine",
                "summary": "Linked docs clarify the decision boundary.",
            }

            result = refine_decision(
                "Adjust this based on the linked docs.",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=False,
                url_context=url_context,
                url_perception={"perception_id": "url.refine"},
            )

            decision_ctx = result.artifact["compiled_context"]["decision_context"]["url_context"]
            self.assertEqual(decision_ctx["perception_id"], "url.refine")
            self.assertEqual(result.artifact["store_paths"]["url_perception"], ".spice/perceptions/url.refine.json")


if __name__ == "__main__":
    unittest.main()
