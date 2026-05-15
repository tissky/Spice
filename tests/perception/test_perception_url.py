from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest
from unittest.mock import patch

from spice.perception import (
    URL_CONTEXT_SCHEMA_VERSION,
    URL_PERCEPTION_SCHEMA_VERSION,
    URLPerceptionLimits,
    build_url_perception_artifact,
    classify_url,
    extract_urls,
    github_raw_url,
    parse_github_issue_or_pr_url,
    run_url_perception,
    url_context_from_perception,
)


class _FakeURLResponse:
    def __init__(
        self,
        *,
        url: str,
        body: str,
        content_type: str = "text/html; charset=utf-8",
        status: int = 200,
    ) -> None:
        self._url = url
        self._body = body.encode("utf-8")
        self.headers = {"content-type": content_type}
        self.status = status
        self.code = status

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


class URLPerceptionTests(unittest.TestCase):
    def test_extract_and_classify_urls(self) -> None:
        text = (
            "Read https://github.com/Dyalwayshappy/Spice/blob/main/README.md, "
            "then https://github.com/Dyalwayshappy/Spice/pull/12."
        )

        urls = extract_urls(text)

        self.assertEqual(len(urls), 2)
        self.assertEqual(classify_url(urls[0]), "github_file")
        self.assertEqual(classify_url(urls[1]), "github_pr")
        self.assertEqual(
            github_raw_url(urls[0]),
            "https://raw.githubusercontent.com/Dyalwayshappy/Spice/main/README.md",
        )
        self.assertEqual(
            parse_github_issue_or_pr_url(urls[1]),
            ("Dyalwayshappy", "Spice", "pull", "12"),
        )

    def test_fetches_webpage_text_into_artifact_and_compact_context(self) -> None:
        def fake_urlopen(request: object, timeout: float = 0.0) -> _FakeURLResponse:
            self.assertEqual(timeout, 12.0)
            return _FakeURLResponse(
                url=_request_url(request),
                body=(
                    "<html><head><title>Design Doc</title></head>"
                    "<body><h1>Decision state</h1><p>State as context is the next layer.</p>"
                    "<script>ignore()</script></body></html>"
                ),
            )

        with patch("spice.perception.url.urllib.request.urlopen", side_effect=fake_urlopen):
            result = run_url_perception(
                urls=["https://example.com/docs/spice"],
                query="external design doc",
            )

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].title, "Design Doc")
        self.assertIn("State as context", result.documents[0].text)
        self.assertNotIn("ignore()", result.documents[0].text)

        artifact = build_url_perception_artifact(
            trigger="new_decision",
            result=result,
            created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        )
        payload = artifact.to_payload()
        self.assertEqual(payload["schema_version"], URL_PERCEPTION_SCHEMA_VERSION)
        self.assertTrue(payload["perception_id"].startswith("url."))
        self.assertEqual(payload["documents"][0]["source_type"], "web_page")
        self.assertEqual(payload["facts"][0]["source_url"], "https://example.com/docs/spice")
        self.assertIn("URL perception read 1 document", payload["summary"])

        context = url_context_from_perception(payload)
        self.assertEqual(context["schema_version"], URL_CONTEXT_SCHEMA_VERSION)
        self.assertEqual(context["source"], "url_perception")
        self.assertEqual(context["perception_id"], payload["perception_id"])
        self.assertEqual(context["documents"][0]["title"], "Design Doc")
        self.assertIn("State as context", context["facts"][0]["text"])
        self.assertNotIn("tool_calls", context)

    def test_github_blob_uses_raw_url_but_keeps_original_source_url(self) -> None:
        fetched_urls: list[str] = []

        def fake_urlopen(request: object, timeout: float = 0.0) -> _FakeURLResponse:
            url = _request_url(request)
            fetched_urls.append(url)
            return _FakeURLResponse(
                url=url,
                body="# Spice\n\nSpice is a decision brain.",
                content_type="text/plain; charset=utf-8",
            )

        original = "https://github.com/Dyalwayshappy/Spice/blob/main/README.md"
        with patch("spice.perception.url.urllib.request.urlopen", side_effect=fake_urlopen):
            result = run_url_perception(urls=[original], query="read readme")

        self.assertEqual(
            fetched_urls,
            ["https://raw.githubusercontent.com/Dyalwayshappy/Spice/main/README.md"],
        )
        self.assertEqual(result.documents[0].url, original)
        self.assertEqual(result.documents[0].source_type, "github_file")
        self.assertEqual(result.documents[0].metadata["fetch_url"], fetched_urls[0])

    def test_github_issue_uses_api_and_extracts_issue_metadata(self) -> None:
        def fake_urlopen(request: object, timeout: float = 0.0) -> _FakeURLResponse:
            url = _request_url(request)
            self.assertEqual(
                url,
                "https://api.github.com/repos/Dyalwayshappy/Spice/issues/8",
            )
            return _FakeURLResponse(
                url=url,
                body=json.dumps(
                    {
                        "title": "Add URL perception",
                        "state": "open",
                        "body": "Spice should read external docs as perception.",
                        "user": {"login": "alice"},
                        "labels": [{"name": "runtime"}],
                    }
                ),
                content_type="application/json; charset=utf-8",
            )

        with patch("spice.perception.url.urllib.request.urlopen", side_effect=fake_urlopen):
            result = run_url_perception(
                urls=["https://github.com/Dyalwayshappy/Spice/issues/8"],
                query="issue context",
            )

        document = result.documents[0]
        self.assertEqual(document.source_type, "github_issue")
        self.assertEqual(document.title, "Add URL perception")
        self.assertIn("GitHub issues Dyalwayshappy/Spice#8", document.text)
        self.assertIn("labels: runtime", document.text)
        self.assertEqual(document.metadata["github"]["number"], "8")

    def test_blocks_private_binary_and_unsupported_urls_without_fetch(self) -> None:
        with patch("spice.perception.url.urllib.request.urlopen") as urlopen:
            result = run_url_perception(
                urls=[
                    "http://localhost:8000/private",
                    "https://example.com/archive.zip",
                    "file:///etc/passwd",
                ],
                query="unsafe links",
            )

        urlopen.assert_not_called()
        self.assertEqual(result.documents, [])
        reasons = {item.reason for item in result.urls_skipped}
        self.assertIn("private_or_local_host_blocked", reasons)
        self.assertIn("binary_url_skipped", reasons)
        self.assertIn("unsupported_scheme", reasons)

    def test_total_char_budget_skips_remaining_urls(self) -> None:
        def fake_urlopen(request: object, timeout: float = 0.0) -> _FakeURLResponse:
            return _FakeURLResponse(
                url=_request_url(request),
                body="a" * 100,
                content_type="text/plain; charset=utf-8",
            )

        with patch("spice.perception.url.urllib.request.urlopen", side_effect=fake_urlopen):
            result = run_url_perception(
                urls=["https://example.com/a", "https://example.com/b"],
                limits=URLPerceptionLimits(max_urls=2, max_chars_per_url=40, total_char_budget=40),
            )

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].chars_read, 40)
        self.assertEqual(result.urls_skipped[0].reason, "total_char_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
