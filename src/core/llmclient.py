# src/core/llmclient.py

import inspect
import json
import os
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from textwrap import dedent
from types import MethodType
from typing import Any, Dict, Optional, Type

import instructor
from pydantic import BaseModel
from ratelimit import limits

from schemas import (
	BasicLLMResponseBool,
	LLMClientConfig,
)

from .constants import VALID_TEMPLATES, console
from .resolve_templates import get_prompt

# --------------------------------------------------------------------------- #
# jfc per-call generation logging (pre-reg §6 / #11)
#
# The generation calls run through instructor's OpenAI route
# (instructor.from_provider("openai/...")), which uses the OpenAI SDK directly,
# NOT litellm.completion -- so a litellm success_callback never fires for them.
# Per-call logging is therefore done HERE, at the call boundary, and is fully
# OPT-IN: it activates only when the env var JFC_GEN_CALL_LOG names an output
# path. Unset (the fork's standalone default) -> zero behavior change, the call
# takes the original create() path. Logging NEVER raises into generation.
# --------------------------------------------------------------------------- #
JFC_GEN_CALL_LOG_ENV = "JFC_GEN_CALL_LOG"
_JFC_LOG_LOCK = threading.Lock()


def _jfc_log_path() -> Optional[str]:
	"""Return the per-call log path if logging is enabled, else None."""
	path = os.environ.get(JFC_GEN_CALL_LOG_ENV)
	return path or None


def _jfc_usage_field(usage: Any, name: str) -> Optional[int]:
	"""Read a token field from an OpenAI/instructor usage object or dict."""
	if usage is None:
		return None
	if isinstance(usage, dict):
		return usage.get(name)
	return getattr(usage, name, None)


def _jfc_emit_call_log(
	path: str,
	*,
	model: Optional[str],
	usage: Any,
	latency_ms: int,
	status: str,
	error: Optional[str] = None,
) -> None:
	"""Append one JSON line describing a generation call. Never raises."""
	try:
		row = {
			"ts": datetime.now(timezone.utc).isoformat(),
			"phase": "generation",
			"model": model,
			"status": status,
			"prompt_tokens": _jfc_usage_field(usage, "prompt_tokens"),
			"completion_tokens": _jfc_usage_field(usage, "completion_tokens"),
			"total_tokens": _jfc_usage_field(usage, "total_tokens"),
			"latency_ms": latency_ms,
			"error": error,
		}
		line = json.dumps(row, ensure_ascii=False)
		with _JFC_LOG_LOCK:
			parent = os.path.dirname(path)
			if parent:
				os.makedirs(parent, exist_ok=True)
			with open(path, "a", encoding="utf-8", newline="\n") as fh:
				fh.write(line + "\n")
				fh.flush()
	except Exception:
		# Logging must NEVER break generation; swallow anything that goes wrong.
		pass


class LLMClient:
	def __init__(self, config: LLMClientConfig):
		"""
		Initialize the LLMClient.

		Args:
		    config (LLMClientConfig): Configuration for the LLMClient model.
		"""
		self.config = config

		if self.config.model == "anthropic/claude-3-haiku-20240307":
			self.client = instructor.from_provider(
				"anthropic/claude-3-haiku-20240307", mode=instructor.Mode.ANTHROPIC_JSON
			)
		elif self.config.model and self.config.model.startswith("openai/"):
			# OpenAI-compatible endpoints (incl. DeepInfra via OPENAI_BASE_URL) can
			# return multiple tool calls, which instructor's default TOOLS mode
			# rejects ("Instructor does not support multiple tool calls"). JSON mode
			# requests a single JSON object instead and avoids the failure.
			self.client = instructor.from_provider(self.config.model, mode=instructor.Mode.JSON)
		else:
			self.client = instructor.from_provider(self.config.model)

		# Apply rate limiting to call() if needed
		original_call_fn = self.__class__.call
		rate_limited_call_fn = self._build_rate_limited_judge_call(self.config.rate_limit)(original_call_fn)
		self.call = MethodType(rate_limited_call_fn, self)

	def _build_rate_limited_judge_call(self, rate_limit: Optional[Dict]):
		"""
		Builds a rate-limited version of a function based on given parameters.

		Args:
		    rate_limit (dict): Rate limiting config with 'calls' and 'period' in seconds.

		Returns:
		    Callable: Decorated function with rate limiting.
		"""

		if not rate_limit:
			return lambda fn: fn  # No-op if no rate limiting

		calls = rate_limit.get("calls", 60)
		period = rate_limit.get("period", 60)  # In seconds

		def decorator(fn):
			rate_limited_fn = limits(calls=calls, period=period)(fn)

			@wraps(fn)
			def wrapper(*args, **kwargs):
				return rate_limited_fn(*args, **kwargs)

			return wrapper

		return decorator

	def _build_prompt(self, template_vars: Dict[str, Any]) -> Optional[str]:
		"""
		Builds a prompt string by interpolating row data into a prompt template.

		Args:
		    row (dict): A dictionary containing row-specific fields.

		Returns:
		    Optional[str]: A formatted prompt ready for model input.
		"""
		template = self.config.template

		if template not in VALID_TEMPLATES:
			console.print(
				f"[WARNING] Template '{template}' for LLM call not found. Defaulting to 'judge/single_autograder'."
			)
			template = "judge/single_autograder"

		if "/" not in template:
			template = f"judge/{template}"

		# Merge defaults with row-specific values, letting template_vars values win
		full_vars = self.config.default_params.copy()
		full_vars.update(template_vars)
		prompt = get_prompt(template, **full_vars)
		return dedent(prompt)

	def _fill_in_args(
		self,
		user_prompt: str,
		response_schema: Type[BaseModel] = BasicLLMResponseBool,
		temperature: Optional[int] = None,
		fixed_seed: Optional[int] = None,
	) -> Dict[str, Any]:
		"""
		Fills in arguments for LLM call, based on given and config parameters.

		Returns
		-------
		args (Dict[str, Any]): arguments for LLM call
		"""

		# Build messages
		messages = []
		system_prompt = self.config.default_params.get("system_prompt")
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": user_prompt})

		# Fill in args for LLM call
		args = {
			"messages": messages,
			"response_model": response_schema,
			"max_retries": self.config.retries,
		}

		# Handle Bedrock models - extract modelId from model name
		# For Bedrock, litellm/instructor requires modelId parameter (not "model")
		if self.config.model and self.config.model.startswith("bedrock/"):
			# For Bedrock, model name format is "bedrock/model-id"
			# Extract the model-id part and pass it as modelId parameter
			model_id = self.config.model.split("/", 1)[1]
			args["modelId"] = model_id
			# Note: Do NOT pass "model" parameter for Bedrock - only modelId is valid

		sig = inspect.signature(self.client.chat.completions.create)
		if "temperature" in sig.parameters:
			temperature = temperature if temperature is not None else self.config.temperature
			args["temperature"] = temperature
		if "max_tokens" in sig.parameters:
			args["max_tokens"] = self.config.max_tokens
		if fixed_seed is not None and "seed" in sig.parameters:
			args["seed"] = fixed_seed

		# Forward extra_body unconditionally; instructor/openai pass it through the OpenAI client body.
		if self.config.extra_body:
			args["extra_body"] = self.config.extra_body

		return args

	def call(
		self,
		template_vars: Optional[Dict[str, Any]] = None,
		response_schema: Type[BaseModel] = BasicLLMResponseBool,
		temperature: Optional[int] = None,
		fixed_seed: Optional[int] = None,
	) -> Type[BaseModel]:
		"""
		Runs a single evaluation using a prompt generated from row data.

		Args:
		    row (dict): Input data used to build the prompt.
		    response_schema (Type[BaseModel]): The response schema to output.
		    temperature (int): Override for model temperature parameter.

		Returns:
		    Type[BaseModel]: Object containing score and reasoning score.
		"""
		if self.config.test_debug_mode:
			return BasicLLMResponseBool(score=0, reasoning="ERROR: Test debug mode is active.")

		# Fill in template
		template_vars = template_vars or {}
		user_prompt = self._build_prompt(template_vars)
		if not user_prompt:
			return BasicLLMResponseBool(score=0, reasoning="ERROR: Failed to get user prompt.")

		# Make and return LLM call
		args = self._fill_in_args(user_prompt, response_schema, temperature, fixed_seed)
		log_path = _jfc_log_path()
		start = time.monotonic()
		try:
			if log_path:
				# create_with_completion also returns the raw completion, whose
				# .usage carries the token counts we log. Same underlying call and
				# same parsed result as create() -- output is unchanged.
				response, completion = self.client.chat.completions.create_with_completion(**args)
				_jfc_emit_call_log(
					log_path,
					model=self.config.model,
					usage=getattr(completion, "usage", None),
					latency_ms=int((time.monotonic() - start) * 1000),
					status="ok",
				)
			else:
				response = self.client.chat.completions.create(**args)
		except Exception as e:
			if log_path:
				_jfc_emit_call_log(
					log_path,
					model=self.config.model,
					usage=None,
					latency_ms=int((time.monotonic() - start) * 1000),
					status="error",
					error=str(e),
				)
			console.print(f"Error during LLM call: {e}")
			return BasicLLMResponseBool(score=0, reasoning=f"Error during LLM call: {e}")

		return response_schema.model_validate(response)
