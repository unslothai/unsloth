# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Muse Spark Open GGUFs (arch ``muse-spark``) load the right files, sampling and effort.

- Sampling: the model card recommends temperature 1.0, top_p 1.0 and no top-k cutoff.
  Without a family entry every id fell through to ``default.yaml`` (0.7 / 0.95 / min_p 0.01).
  top_k is stored as -1 ("disabled"), which the llama-server payload sends as 0.
- Reasoning: the template's ``reasoning_effort`` accepts only low / medium / high / xhigh
  (default xhigh) and ``raise_exception``s on anything else. Studio forwarded the OpenAI
  ``none`` sentinel as-is, and ``max`` reached the template from the default-effort path,
  so both failed the render. The model cannot turn reasoning off, so ``none`` and
  ``minimal`` take the smallest budget and ``max`` the largest.
- Files: eight split quant folders plus a root ``imatrix_unsloth.gguf``, which must never
  be offered as a variant or picked as the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


REPO = "unsloth/Muse-Spark-Open-816B-A42B-GGUF"
STEM = "Muse-Spark-Open-816B-A42B"
# Shard counts as published.
_SPLITS = {
    "UD-IQ1_S": 5,
    "UD-IQ1_M": 6,
    "UD-IQ2_XXS": 6,
    "UD-IQ2_S": 7,
    "UD-IQ3_XXS": 8,
    "UD-Q2_K_XL": 8,
    "UD-Q3_K_XL": 10,
    "UD-Q4_K_XL": 12,
}
FILES = [".gitattributes", "LICENSE.md", "README.md", "USE_POLICY.md", "imatrix_unsloth.gguf"] + [
    f"{q}/{STEM}-{q}-{i:05d}-of-{n:05d}.gguf" for q, n in _SPLITS.items() for i in range(1, n + 1)
]


def _model_info(files):
    return SimpleNamespace(
        siblings = [SimpleNamespace(rfilename = f, size = 1 << 30) for f in files],
        gguf = None,
        sha = "0" * 40,
    )


# ---------------------------------------------------------------- sampling

EXPECTED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
}

MUSE_SPARK_IDS = [
    REPO,
    f"{REPO}:UD-IQ1_S",
    f"{REPO}:UD-Q4_K_XL",
    "meta-models/Muse-Spark-Open-816B-A42B",
    "unsloth/Muse-Spark-Open-816B-A42B",
    "/home/u/.cache/huggingface/hub/models--unsloth--Muse-Spark-Open-816B-A42B-GGUF/snapshots/deadbeef",
    f"/data/models/{STEM}-GGUF/UD-IQ1_S/{STEM}-UD-IQ1_S-00001-of-00005.gguf",
]


@pytest.mark.parametrize("model_id", MUSE_SPARK_IDS)
def test_every_id_shape_resolves_to_published_sampling(model_id):
    from utils.inference.inference_config import load_inference_config
    config = load_inference_config(model_id)
    for key, want in EXPECTED_SAMPLING.items():
        assert config[key] == want, f"{model_id}: {key} was {config[key]}, expected {want}"


def test_family_entry_is_registered_in_patterns():
    path = _backend_root / "assets" / "configs" / "inference_defaults.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    assert "muse-spark" in data["families"]
    assert "muse-spark" in data["patterns"]


def test_disabled_top_k_reaches_llama_server_as_zero():
    """-1 is Studio's "disabled"; llama-server reads 0 as no top-k cutoff."""
    from utils.inference.inference_config import load_inference_config

    top_k = load_inference_config(REPO)["top_k"]
    assert (top_k if top_k >= 0 else 0) == 0


def test_neighbouring_families_are_untouched():
    from utils.inference.inference_config import load_inference_config

    glimmer = load_inference_config("unsloth/Muse-Glimmer-30B-GGUF")
    assert (glimmer["top_p"], glimmer["top_k"]) == (0.95, 64)
    assert load_inference_config("unsloth/Llama-4-Scout-17B-16E-Instruct")["top_k"] == -1
    assert load_inference_config("unsloth/gemma-2-9b-it")["top_k"] == 64


# ---------------------------------------------------------------- files


def test_variant_menu_lists_exactly_the_eight_quants(monkeypatch):
    from utils.models import model_config

    monkeypatch.setattr(
        model_config,
        "_hub_model_info_slow_link",
        lambda repo, token, files_metadata = True: _model_info(FILES),
    )
    variants, has_vision = model_config.list_gguf_variants(REPO)
    assert not has_vision
    assert sorted(v.quant for v in variants) == sorted(_SPLITS)
    assert all("imatrix" not in v.filename for v in variants)


@pytest.mark.parametrize("quant", sorted(_SPLITS))
def test_every_quant_resolves_to_its_first_shard_and_siblings(monkeypatch, quant):
    from core.inference import llama_cpp

    monkeypatch.setattr(
        llama_cpp, "call_hub_with_anonymous_retry", lambda fn, token, repo: list(FILES)
    )
    n = _SPLITS[quant]
    first, extra = llama_cpp._resolve_variant_gguf_files(REPO, quant)
    assert first == f"{quant}/{STEM}-{quant}-00001-of-{n:05d}.gguf"
    assert extra == [f"{quant}/{STEM}-{quant}-{i:05d}-of-{n:05d}.gguf" for i in range(2, n + 1)]


def test_imatrix_is_never_picked_as_the_model(tmp_path):
    from utils.models import model_config

    assert (
        model_config._pick_best_gguf([f for f in FILES if f.endswith(".gguf")])
        != "imatrix_unsloth.gguf"
    )
    (tmp_path / "imatrix_unsloth.gguf").write_bytes(b"GGUF")
    assert model_config.detect_gguf_model(str(tmp_path / "imatrix_unsloth.gguf")) is None
    assert model_config.detect_gguf_model(str(tmp_path)) is None


# ---------------------------------------------------------------- reasoning

# The effort handling of the shipped template, verbatim, plus its self-channel opener
# (which routes replies through the recipient-channel normalizer).
MUSE_TEMPLATE = """
{%- set reasoning_effort = reasoning_effort | default('xhigh', true) -%}
{%- set reasoning_effort_map = {'low': 8, 'medium': 32, 'high': 256, 'xhigh': 512} -%}
{%- if not (reasoning_effort is string and reasoning_effort in reasoning_effort_map) -%}
    {{- raise_exception('Unsupported reasoning_effort: ' ~ reasoning_effort ~ '. Expected one of: low, medium, high, xhigh.') -}}
{%- endif -%}
{{- '<|start|>system<|message|>Reasoning strength: ' ~ reasoning_effort_map[reasoning_effort] ~ '.<|eot|>' -}}
{%- for message in messages -%}
    {%- if message.role == 'assistant' and message.reasoning_content -%}
        {{- '<|start|>assistant to=self<|message|>' ~ message.reasoning_content ~ '<|eom|>' -}}
    {%- endif -%}
    {{- '<|start|>' ~ message.role ~ '<|message|>' ~ message.content ~ '<|eot|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}
"""

LEVELS = ["low", "medium", "high", "xhigh"]


def _render(**kwargs):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = _raise
    return env.from_string(MUSE_TEMPLATE).render(
        messages = [{"role": "user", "content": "hi"}], add_generation_prompt = True, **kwargs
    )


def _shim(architecture = "muse-spark"):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "reasoning_effort"
    backend._reasoning_effort_levels = list(LEVELS)
    backend._supports_preserve_thinking = False
    backend._architecture = architecture
    return backend


def test_detection_publishes_the_four_levels_and_recipient_markers():
    from core.inference.chat_template_helpers import (
        _detect_reasoning_channel_markers_from_templates,
    )
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(MUSE_TEMPLATE, REPO)
    assert flags["supports_reasoning"]
    assert flags["reasoning_style"] == "reasoning_effort"
    assert not flags["reasoning_always_on"]
    assert flags["reasoning_effort_levels"] == LEVELS
    assert _detect_reasoning_channel_markers_from_templates((MUSE_TEMPLATE,)) == ("self", "user")


def test_the_unpatched_template_rejects_none_and_max():
    with pytest.raises(ValueError, match = "Unsupported reasoning_effort"):
        _render(reasoning_effort = "none")
    with pytest.raises(ValueError, match = "Unsupported reasoning_effort"):
        _render(reasoning_effort = "max")


@pytest.mark.parametrize(
    ("effort", "enable_thinking", "expected"),
    [
        ("none", None, "low"),
        ("minimal", None, "low"),
        ("low", None, "low"),
        ("medium", None, "medium"),
        ("high", None, "high"),
        ("xhigh", None, "xhigh"),
        (None, True, "high"),
        (None, False, "low"),
        ("max", True, "high"),
    ],
)
def test_request_kwargs_always_render(effort, enable_thinking, expected):
    kwargs = _shim()._request_reasoning_kwargs(enable_thinking, effort, None)
    assert kwargs == {"reasoning_effort": expected}
    _render(**kwargs)


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("none", "low"), ("minimal", "low"), ("max", "xhigh"), ("xhigh", "xhigh"), (None, "high")],
)
def test_default_effort_kwargs_always_render(effort, expected):
    kwargs = _shim()._reasoning_kwargs(True, effort)
    assert kwargs == {"reasoning_effort": expected}
    _render(**kwargs)


def test_no_kwargs_renders_the_template_default():
    assert "Reasoning strength: 512." in _render()


def test_other_architectures_still_forward_none():
    """gpt-oss-style templates honour "none"; only muse-spark coerces it."""
    assert _shim(architecture = "gpt-oss")._request_reasoning_kwargs(None, "none", None) == {
        "reasoning_effort": "none"
    }
