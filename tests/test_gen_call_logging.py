"""Tests for jfc per-call generation logging (pre-reg §6 / #11).

The generation calls run through instructor's OpenAI route (the OpenAI SDK, not
litellm), so logging is hooked at LLMClient.call and gated on JFC_GEN_CALL_LOG:

* env unset  -> no log file, the original create() path is taken (zero change);
* env set    -> create_with_completion() is used and one JSON line per call is
  appended with the token usage; an error still writes a status="error" line and
  never propagates.

The fake client gives `create` a REAL signature (LLMClient._fill_in_args inspects
it) and `_build_prompt` is stubbed so the test does not depend on template files.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from core.llmclient import LLMClient
from schemas import BasicLLMResponseBool, LLMClientConfig


def _model():
	# instructor returns the validated response_model instance; model_validate on a
	# real model instance is a no-op pass-through, so the call() return path works.
	return BasicLLMResponseBool(score=1, reasoning="ok")


def _real_create(*, messages, response_model, max_retries, temperature=None, max_tokens=None,
                 seed=None, extra_body=None):
	# Real signature so inspect.signature(...) in _fill_in_args succeeds.
	return _model()


def _usage_completion(prompt_tokens=11, completion_tokens=7):
	completion = MagicMock()
	completion.usage.prompt_tokens = prompt_tokens
	completion.usage.completion_tokens = completion_tokens
	completion.usage.total_tokens = prompt_tokens + completion_tokens
	return completion


def _make_client(monkeypatch, fake_client):
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda *a, **k: fake_client)
	config = LLMClientConfig(
		model="openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
		template="single_judge",
		test_debug_mode=False,
	)
	client = LLMClient(config)
	# Stub prompt rendering so the test does not need template files / vars.
	monkeypatch.setattr(client, "_build_prompt", lambda template_vars: "a user prompt")
	return client


def test_no_log_when_env_unset(monkeypatch, tmp_path):
	monkeypatch.delenv("JFC_GEN_CALL_LOG", raising=False)
	fake_client = MagicMock()
	fake_client.chat.completions.create = _real_create
	client = _make_client(monkeypatch, fake_client)

	result = client.call({"some": "vars"})

	assert result.reasoning == "ok"
	# create_with_completion NOT used when logging is off
	assert not fake_client.chat.completions.create_with_completion.called
	assert list(tmp_path.glob("*.jsonl")) == []


def test_logs_one_line_with_usage_when_enabled(monkeypatch, tmp_path):
	log_path = tmp_path / "nested" / "gen_calls.jsonl"
	monkeypatch.setenv("JFC_GEN_CALL_LOG", str(log_path))

	fake_client = MagicMock()
	fake_client.chat.completions.create = _real_create
	fake_client.chat.completions.create_with_completion.return_value = (
		_model(), _usage_completion(11, 7),
	)
	client = _make_client(monkeypatch, fake_client)

	client.call({"some": "vars"})

	assert fake_client.chat.completions.create_with_completion.called
	rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()]
	assert len(rows) == 1
	row = rows[0]
	assert row["phase"] == "generation"
	assert row["status"] == "ok"
	assert row["model"] == "openai/meta-llama/Meta-Llama-3.1-8B-Instruct"
	assert row["prompt_tokens"] == 11
	assert row["completion_tokens"] == 7
	assert row["total_tokens"] == 18
	assert isinstance(row["latency_ms"], int)


def test_logs_error_line_and_does_not_raise(monkeypatch, tmp_path):
	log_path = tmp_path / "gen_calls.jsonl"
	monkeypatch.setenv("JFC_GEN_CALL_LOG", str(log_path))

	fake_client = MagicMock()
	fake_client.chat.completions.create = _real_create
	fake_client.chat.completions.create_with_completion.side_effect = RuntimeError("boom")
	client = _make_client(monkeypatch, fake_client)

	# generation must not raise; it returns the sentinel error response
	result = client.call({"some": "vars"})
	assert "Error during LLM call" in result.reasoning

	rows = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines()]
	assert len(rows) == 1
	assert rows[0]["status"] == "error"
	assert "boom" in rows[0]["error"]
	assert rows[0]["prompt_tokens"] is None


def test_multiple_calls_append(monkeypatch, tmp_path):
	log_path = tmp_path / "gen_calls.jsonl"
	monkeypatch.setenv("JFC_GEN_CALL_LOG", str(log_path))

	fake_client = MagicMock()
	fake_client.chat.completions.create = _real_create
	fake_client.chat.completions.create_with_completion.return_value = (
		_model(), _usage_completion(5, 3),
	)
	client = _make_client(monkeypatch, fake_client)

	for _ in range(3):
		client.call({"some": "vars"})

	rows = Path(log_path).read_text(encoding="utf-8").splitlines()
	assert len(rows) == 3
