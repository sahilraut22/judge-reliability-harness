# tests/test_contract_end_to_end.py
"""END-TO-END shared-config contract test (D10/D19) crossing BOTH repos.

This is the mandatory adversarial contract test: the most dangerous failure mode
is a SILENT key mismatch between the keys the jfc renderer
(``jfc.stages.jrh_driver.render_generation_config``) EMITS into the generated JRH
YAML and the keys the JRH fork
(``core.resolve_synthetic_data_configs.resolve_basic_perturbations_config``) READS.
A mismatch raises no error -- generation would silently run at temperature 0.0 with
thinking ON (Qwen empties the judge budget), which is precisely the bug D10/D19 fix.

So instead of asserting on string literals on each side independently, this test
runs the REAL renderer, loads the REAL YAML it writes, and feeds the loaded
``synthetic_data_params`` dict to the REAL fork resolver -- in a single process.
If the renderer emits ``generation_temperature``/``generation_extra_body`` and the
fork reads them under those exact names, the generation LLMClientConfig comes out
with temperature 0.7 and the Qwen extra_body set. If either side renames a key, the
assertions fail (the fork falls back to its 0.0 / None defaults) -- catching the
silent regression.

The jfc renderer imports only stdlib + pandas + yaml (no jfc-internal modules), so
it imports cleanly in the JRH venv once jfc's ``src/`` is on sys.path. No live API
calls: the resolver is pure config plumbing (no client is constructed).
"""

import sys
from pathlib import Path

import yaml

# Make jfc.stages.jrh_driver importable from the JRH venv. The submodule lives at
# external/judge-reliability-harness/ inside the jfc repo, so the jfc package root
# is three parents up + src/.
_JRH_ROOT = Path(__file__).resolve().parents[1]
_JFC_REPO_ROOT = _JRH_ROOT.parents[1]  # external/judge-reliability-harness -> repo root
_JFC_SRC = _JFC_REPO_ROOT / "src"
if str(_JFC_SRC) not in sys.path:
	sys.path.insert(0, str(_JFC_SRC))

from jfc.stages.jrh_driver import render_generation_config  # noqa: E402

from core.resolve_synthetic_data_configs import resolve_basic_perturbations_config  # noqa: E402
from schemas import (  # noqa: E402
	AdminConfig,
	DatasetConfig,
	EvaluationConfig,
	LLMClientConfig,
	PerturbationConfig,
)


QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# Mirrors config/models.yaml: alibaba pins the Qwen enable_thinking=false extra_body;
# meta does NOT. Both run on deepinfra.
MODELS_YAML = {
	"families": {
		"meta": {
			"provider": "deepinfra",
			"litellm_string": "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
		},
		"alibaba": {
			"provider": "deepinfra",
			"litellm_string": "deepinfra/Qwen/Qwen3.5-9B",
			"extra_body": QWEN_EXTRA_BODY,
		},
	}
}

RUN_YAML = {
	"seeds": {"master": 20260612},
	"roles": {
		"generation": {"temperature": 0.7, "max_tokens": 1024},
		"validation": {"temperature": 0.0, "max_tokens": 256},
	},
}

_VALIDATOR = "openai/meta-llama/Meta-Llama-3.1-8B-Instruct"


def _make_admin_config():
	"""Minimal AdminConfig sufficient to drive resolve_basic_perturbations_config."""
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


def _render_and_load(tmp_path, family):
	"""Render via the REAL jfc renderer, then load the YAML it actually wrote."""
	out = tmp_path / f"gen_{family}_v1.yml"
	render_generation_config(
		family=family,
		models_yaml=MODELS_YAML,
		run_yaml=RUN_YAML,
		dataset_name="base300.csv",
		n_items=10,
		validation_model=_VALIDATOR,
		out_path=out,
		time_stamp=f"gen_{family}_v1",
	)
	return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_end_to_end_alibaba_temperature_and_extra_body_flow(tmp_path):
	"""Renderer -> YAML -> fork resolver: temperature 0.7 + Qwen extra_body land.

	This is the load-bearing contract assertion. It fails loudly if EITHER side
	renames generation_temperature / generation_extra_body (a silent key mismatch
	would otherwise leave the generation config at 0.0 / None).
	"""
	cfg = _render_and_load(tmp_path, "alibaba")
	synthetic_data_params = cfg["synthetic_data_params"]

	# Sanity: the renderer emitted the contract keys under the agreed names.
	assert "generation_temperature" in synthetic_data_params
	assert "generation_extra_body" in synthetic_data_params

	admin_config = _make_admin_config()
	resolved = resolve_basic_perturbations_config(
		admin_config,
		instruction="Transform the text.",
		synthetic_data_params=synthetic_data_params,
	)

	# The fork READ both keys and applied them to the GENERATION config.
	assert resolved.generation_model_config.temperature == 0.7
	assert resolved.generation_model_config.extra_body == QWEN_EXTRA_BODY


def test_end_to_end_non_qwen_family_gets_temp_but_null_extra_body(tmp_path):
	"""meta family: temperature 0.7 still flows; extra_body resolves to None.

	The contract key is always emitted (None for non-Qwen) so the fork sees it
	unconditionally; the resolver must coerce that None into a None extra_body,
	NOT crash and NOT silently re-enable thinking for the wrong family.
	"""
	cfg = _render_and_load(tmp_path, "meta")
	synthetic_data_params = cfg["synthetic_data_params"]

	assert synthetic_data_params["generation_extra_body"] is None
	assert synthetic_data_params["generation_temperature"] == 0.7

	resolved = resolve_basic_perturbations_config(
		_make_admin_config(),
		instruction="Transform the text.",
		synthetic_data_params=synthetic_data_params,
	)

	assert resolved.generation_model_config.temperature == 0.7
	assert resolved.generation_model_config.extra_body is None


def test_end_to_end_keys_are_byte_exact_between_sides(tmp_path):
	"""Defence in depth: assert the EMITTED key set is a superset of the READ keys.

	resolve_basic_perturbations_config reads exactly these two contract keys from
	synthetic_data_params (via .get). Confirm the renderer emits both names so the
	.get never silently falls through to a default. Guards against a one-sided
	rename slipping past the value assertions above (e.g. if a default happened to
	equal the asserted value for some family).
	"""
	cfg = _render_and_load(tmp_path, "alibaba")
	emitted = set(cfg["synthetic_data_params"])
	read_by_fork = {"generation_temperature", "generation_extra_body"}
	assert read_by_fork <= emitted, (
		f"fork reads {read_by_fork} but renderer only emits {sorted(emitted)} "
		"-- silent key mismatch would zero out temperature / disable extra_body"
	)
