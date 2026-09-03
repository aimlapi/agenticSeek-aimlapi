import os
import re
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sources.llm_provider import (
    AIMLAPI_ATTRIBUTION_HEADERS,
    AIMLAPI_DEFAULT_BASE_URL,
    Provider,
    aimlapi_attribution_headers,
)

FAKE_ENV = {"AIMLAPI_API_KEY": "sk-test-123"}


def make_provider(is_local=False):
    return Provider("aimlapi", "deepseek/deepseek-v4-flash", is_local=is_local)


def mocked_openai(mock_openai, content="42"):
    """Wire an OpenAI mock so chat.completions.create returns `content`."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    mock_openai.return_value = client
    return client


@patch.dict(os.environ, FAKE_ENV)
class TestAimlapiProvider(unittest.TestCase):
    """Test cases for the aimlapi.com provider integration."""

    def test_aimlapi_provider_registered(self):
        """aimlapi is registered in available_providers."""
        self.assertIn("aimlapi", make_provider().available_providers)

    def test_aimlapi_is_unsafe_provider(self):
        """aimlapi is a cloud API, so it must trigger the cloud data warning."""
        self.assertIn("aimlapi", make_provider().unsafe_providers)

    def test_aimlapi_reads_aimlapi_api_key(self):
        """The key is read from AIMLAPI_API_KEY."""
        self.assertEqual(make_provider().api_key, "sk-test-123")

    def test_aimlapi_local_not_supported(self):
        """aimlapi_fn refuses is_local=True."""
        provider = make_provider(is_local=True)
        with self.assertRaises(Exception) as ctx:
            provider.aimlapi_fn([{"role": "user", "content": "hi"}])
        self.assertIn("not available for local use", str(ctx.exception))

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_returns_content(self, mock_openai):
        """aimlapi_fn returns the assistant message content."""
        mocked_openai(mock_openai)
        result = make_provider().aimlapi_fn([{"role": "user", "content": "What is 6*7?"}])
        self.assertEqual(result, "42")

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_uses_default_base_url(self, mock_openai):
        """The OpenAI-compatible endpoint is https://api.aimlapi.com/v1."""
        mocked_openai(mock_openai)
        make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_openai.call_args[1]["base_url"], AIMLAPI_DEFAULT_BASE_URL)

    @patch.dict(os.environ, {"AIMLAPI_BASE_URL": "https://proxy.example.com/v1"})
    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_base_url_is_overridable(self, mock_openai):
        """AIMLAPI_BASE_URL overrides the endpoint."""
        mocked_openai(mock_openai)
        make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_openai.call_args[1]["base_url"], "https://proxy.example.com/v1")

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_passes_model_and_messages(self, mock_openai):
        """The configured model and the history reach the API unchanged."""
        client = mocked_openai(mock_openai)
        history = [{"role": "user", "content": "hi"}]
        make_provider().aimlapi_fn(history)
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(call_kwargs["messages"], history)

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_omits_unset_parameters(self, mock_openai):
        """
        Unset optional parameters are omitted, never sent as None.

        aimlapi.com answers 400 for an explicit null on temperature, top_p,
        seed, tools, tool_choice, response_format, stream and max_tokens on some
        of its models, so passing None would fail live while mocks stay green.
        """
        client = mocked_openai(mock_openai)
        make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        call_kwargs = client.chat.completions.create.call_args[1]
        self.assertEqual(
            [k for k, v in call_kwargs.items() if v is None],
            [],
            "no request parameter may be sent as None",
        )

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_sends_attribution_headers(self, mock_openai):
        """All four attribution headers are attached to the client."""
        mocked_openai(mock_openai)
        make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        headers = mock_openai.call_args[1]["default_headers"]
        self.assertEqual(headers["HTTP-Referer"], "https://github.com/Fosowl/agenticSeek")
        self.assertEqual(headers["X-Title"], "AgenticSeek")
        self.assertEqual(headers["X-AIMLAPI-Source"], "agent/agenticseek")
        self.assertEqual(headers["X-AIMLAPI-Partner-ID"], "part_agenticseek")

    def test_partner_id_matches_gateway_pattern(self):
        """
        A malformed partner id is dropped silently by the gateway, so its shape
        is asserted here rather than discovered in production.
        """
        self.assertRegex(
            AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"],
            re.compile(r"^part_[A-Za-z0-9]{1,64}$"),
        )

    def test_source_matches_gateway_pattern(self):
        """X-AIMLAPI-Source is <channel>/<client> with channel in web|agent|mcp."""
        self.assertRegex(
            AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Source"],
            re.compile(r"^(web|agent|mcp)/[a-z0-9-]{1,32}$"),
        )

    def test_attribution_headers_are_scoped_to_our_origin(self):
        """Attribution never rides a request to another host."""
        self.assertEqual(aimlapi_attribution_headers(AIMLAPI_DEFAULT_BASE_URL),
                         AIMLAPI_ATTRIBUTION_HEADERS)
        self.assertEqual(aimlapi_attribution_headers("https://proxy.example.com/v1"), {})

    def test_attribution_headers_constant_is_not_mutated(self):
        """Each call gets a fresh dict; the module level constant is immutable."""
        first = aimlapi_attribution_headers(AIMLAPI_DEFAULT_BASE_URL)
        first["X-Title"] = "tampered"
        self.assertIsNot(first, AIMLAPI_ATTRIBUTION_HEADERS)
        self.assertEqual(AIMLAPI_ATTRIBUTION_HEADERS["X-Title"], "AgenticSeek")
        self.assertEqual(aimlapi_attribution_headers(AIMLAPI_DEFAULT_BASE_URL)["X-Title"],
                         "AgenticSeek")

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_raises_on_empty_response(self, mock_openai):
        """An empty response is reported instead of raising an IndexError."""
        client = MagicMock()
        client.chat.completions.create.return_value = None
        mock_openai.return_value = client
        with self.assertRaises(Exception) as ctx:
            make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        self.assertIn("empty", str(ctx.exception).lower())

    @patch('sources.llm_provider.OpenAI')
    def test_aimlapi_fn_raises_on_api_error(self, mock_openai):
        """API errors are wrapped with the provider name."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("rate limit exceeded")
        mock_openai.return_value = client
        with self.assertRaises(Exception) as ctx:
            make_provider().aimlapi_fn([{"role": "user", "content": "hi"}])
        self.assertIn("aimlapi.com API error", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
