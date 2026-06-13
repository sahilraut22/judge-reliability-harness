# tests/test_generation_params.py
"""Focused tests for generation temperature + extra_body threading.

Covers the shared-config contract:
  - resolve_basic_perturbations_config reads `generation_temperature` and
    `generation_extra_body` from synthetic_data_params and applies them to the
    generation LLMClientConfig (with sane defaults when absent).
  - LLMClient._fill_in_args forwards config.extra_body into the built args
    unconditionally when set.

No live API calls are made (instructor.from_provider is mocked).
"""

from unittest.mock import MagicMock

import pytest

from core.llmclient import LLMClient
from core.resolve_synthetic_data_configs import resolve_basic_perturbations_config
from schemas import (
	AdminConfig,
	DatasetConfig,
	EvaluationConfig,
	LLMClientConfig,
	PerturbationConfig,
)


QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _make_admin_config():
	"""Minimal AdminConfig sufficient for resolve_basic_perturbations_config."""
	dataset_config = DatasetConfig(dataset_name="mock.csv", default_params={})
	perturbation_config = PerturbationConfig()
	autograder_model_config = LLMClientConfig(template="single_judge")
	evaluation_config = EvaluationConfig(
		template="single_judge",
		autograder_model_name="openai/gpt-4o-mini",
		autograder_model_config=autograder_model_config,
		metric="accuracy",
	)
	return AdminConfig(
		base_module_name="mock",
		module_name="mock",
		test_debug_mode=True,
		dataset_config=dataset_config,
		perturbation_config=perturbation_config,
		evaluation_config=evaluation_config,
	)


def _base_synthetic_data_params(**overrides):
	"""All required SyntheticDataParams fields + generation/validation knobs."""
	params = {
		"generation_model_name": "alibaba/Qwen3.5-9B",
		"validation_model_name": "openai/gpt-4o-mini",
		"max_tokens_generation": 512,
		"max_tokens_validation": 256,
		"similarity_threshold": 0.8,
		"initial_temp": 0.7,
		"num_seed_examples_per_generation": 3,
		"temp_increment": 0.1,
		"max_temp_cap": 1.5,
		"max_consecutive_failures": 5,
		"seed": 1234,
		"target_num_per_bucket": 10,
	}
	params.update(overrides)
	return params


def test_generation_params_applied_when_present():
	admin_config = _make_admin_config()
	synthetic_data_params = _base_synthetic_data_params(
		generation_temperature=0.7,
		generation_extra_body=QWEN_EXTRA_BODY,
	)

	resolved = resolve_basic_perturbations_config(
		admin_config, instruction="Transform the text.", synthetic_data_params=synthetic_data_params
	)

	assert resolved.generation_model_config.temperature == 0.7
	assert resolved.generation_model_config.extra_body == QWEN_EXTRA_BODY


def test_generation_params_default_when_absent():
	admin_config = _make_admin_config()
	synthetic_data_params = _base_synthetic_data_params()  # no generation_temperature / generation_extra_body

	resolved = resolve_basic_perturbations_config(
		admin_config, instruction="Transform the text.", synthetic_data_params=synthetic_data_params
	)

	assert resolved.generation_model_config.temperature == 0.0
	assert resolved.generation_model_config.extra_body is None


def test_fill_in_args_includes_extra_body_when_set(monkeypatch):
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: MagicMock())

	config = LLMClientConfig(
		model="alibaba/Qwen3.5-9B",
		template="single_judge",
		test_debug_mode=False,
		extra_body=QWEN_EXTRA_BODY,
	)
	client = LLMClient(config)

	args = client._fill_in_args("a user prompt")

	assert "extra_body" in args
	assert args["extra_body"] == QWEN_EXTRA_BODY


def test_fill_in_args_omits_extra_body_when_absent(monkeypatch):
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: MagicMock())

	config = LLMClientConfig(
		model="openai/gpt-4o-mini",
		template="single_judge",
		test_debug_mode=False,
	)
	client = LLMClient(config)

	args = client._fill_in_args("a user prompt")

	assert "extra_body" not in args


def test_fill_in_args_extra_body_added_once_and_clobbers_nothing(monkeypatch):
	"""extra_body is added exactly once and does not disturb the other built args.

	Hunt item #4: the forward must not be gated on signature introspection (so it
	fires even if the create() signature does not enumerate extra_body), must add
	the key only once, and must leave messages / response_model / max_retries
	intact. Uses a fake create() whose signature has NO extra_body parameter to
	prove the forward is unconditional, not signature-gated.
	"""

	def _fake_create(*, messages, response_model, max_retries, temperature=None, max_tokens=None):
		return MagicMock()

	fake_client = MagicMock()
	fake_client.chat.completions.create = _fake_create
	monkeypatch.setattr("core.llmclient.instructor.from_provider", lambda model: fake_client)

	config = LLMClientConfig(
		model="alibaba/Qwen3.5-9B",
		template="single_judge",
		test_debug_mode=False,
		temperature=0.3,
		max_tokens=77,
		extra_body=QWEN_EXTRA_BODY,
	)
	client = LLMClient(config)

	args = client._fill_in_args("a user prompt")

	# extra_body present exactly once and is the SAME object (not double-merged).
	assert args["extra_body"] == QWEN_EXTRA_BODY
	assert args["extra_body"] is config.extra_body
	# The forward did not clobber any of the other built args.
	assert args["messages"][-1] == {"role": "user", "content": "a user prompt"}
	assert args["response_model"] is not None
	assert args["max_retries"] == config.retries
	assert args["temperature"] == 0.3
	assert args["max_tokens"] == 77
