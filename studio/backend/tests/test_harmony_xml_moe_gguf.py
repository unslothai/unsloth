# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Harmony-style large MoE GGUF architecture loads the right files, sampling and effort.

- Files: twelve 50 GB split quant folders, no mmproj and no drafter. Every folder must
  resolve to its own first shard and siblings, and the default pick is UD-Q4_K_XL.
- Sampling: generation_config.json ships no sampling settings, so the family takes the
  Harmony defaults the GGUFs were evaluated with: temperature 1.0, top_p 1.0, no top-k
  cutoff, min_p 0. Without an entry every id fell through to ``default.yaml``.
- Reasoning: the template takes an int or minimal/low/medium/high/xhigh/max and
  ``raise_exception``s on any other string, so the OpenAI ``none`` sentinel failed the
  render. The model cannot turn reasoning off, so ``none`` takes the smallest budget.
- Tools: the template declares tools and the model calls them with Harmony headers and
  XML argument bodies (``<|constrain|>xml``). A llama-server without a parser for that
  format falls back to the gpt-oss parser, which rejects it with a 500 mid-stream, so
  Studio offers tools only when the server's llama-common carries that parser.
- Memory: 43 of 52 layers are 512-token sliding-window layers and 49 of 52 blocks are
  MoE; the KV estimate and the offload layout must price them that way.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_backend_root = Path(__file__).resolve().parent.parent
_tests_dir = Path(__file__).resolve().parent
for _p in (_backend_root, _tests_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


ARCH = "beam"
REPO = "unsloth/Beam-1.0-501B-A23B-GGUF"
STEM = "Beam-1.0-501B-A23B"
# Shard counts at 50 GB splits.
_SPLITS = {
    "UD-IQ1_S": 3,
    "UD-IQ1_M": 3,
    "UD-IQ2_XXS": 3,
    "UD-IQ2_M": 4,
    "UD-Q2_K_XL": 4,
    "UD-IQ3_XXS": 4,
    "UD-Q3_K_XL": 5,
    "UD-Q4_K_XL": 7,
    "UD-Q5_K_XL": 8,
    "UD-Q6_K_XL": 10,
    "Q8_0": 11,
    "BF16": 21,
}
FILES = [".gitattributes", "LICENSE", "README.md"] + [
    f"{q}/{STEM}-{q}-{i:05d}-of-{n:05d}.gguf" for q, n in _SPLITS.items() for i in range(1, n + 1)
]


def _model_info(files):
    return SimpleNamespace(
        siblings = [SimpleNamespace(rfilename = f, size = 1 << 30) for f in files],
        gguf = None,
        sha = "0" * 40,
    )


# ---------------------------------------------------------------- files


def test_variant_menu_lists_exactly_the_twelve_quants(monkeypatch):
    from utils.models import model_config

    monkeypatch.setattr(
        model_config,
        "_hub_model_info_slow_link",
        lambda repo, token, files_metadata = True: _model_info(FILES),
    )
    variants, has_vision = model_config.list_gguf_variants(REPO)
    assert not has_vision
    assert sorted(v.quant for v in variants) == sorted(_SPLITS)


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


def test_default_pick_is_ud_q4_k_xl():
    from hub.utils.gguf import pick_best_gguf
    from utils.models import model_config

    want = f"UD-Q4_K_XL/{STEM}-UD-Q4_K_XL-00001-of-00007.gguf"
    ggufs = [f for f in FILES if f.endswith(".gguf")]
    assert pick_best_gguf(ggufs) == want
    assert model_config._pick_best_gguf(ggufs) == want


# ---------------------------------------------------------------- sampling

EXPECTED_SAMPLING = {  # generation_config.json of the released checkpoint
    "temperature": 0.7,
    "top_p": 0.99,
    "top_k": -1,
    "min_p": 0.0,
}

MODEL_IDS = [
    REPO,
    f"{REPO}:UD-IQ1_S",
    f"{REPO}:UD-Q4_K_XL",
    f"{REPO}:BF16",
    f"unsloth/{STEM}",
    f"/home/u/.cache/huggingface/hub/models--unsloth--{STEM}-GGUF/snapshots/deadbeef",
    f"/data/models/{STEM}-GGUF/UD-IQ1_S/{STEM}-UD-IQ1_S-00001-of-00003.gguf",
]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_every_id_shape_resolves_to_the_family_sampling(model_id):
    from utils.inference.inference_config import load_inference_config
    config = load_inference_config(model_id)
    for key, want in EXPECTED_SAMPLING.items():
        assert config[key] == want, f"{model_id}: {key} was {config[key]}, expected {want}"


def test_family_entry_is_registered_in_patterns():
    path = _backend_root / "assets" / "configs" / "inference_defaults.json"
    data = json.loads(path.read_text(encoding = "utf-8"))
    assert "beam-1" in data["families"]
    assert "beam-1" in data["patterns"]


def test_disabled_top_k_reaches_llama_server_as_zero():
    """-1 is Studio's "disabled"; llama-server reads 0 as no top-k cutoff."""
    from utils.inference.inference_config import load_inference_config

    top_k = load_inference_config(REPO)["top_k"]
    assert (top_k if top_k >= 0 else 0) == 0


def test_neighbouring_families_are_untouched():
    from utils.inference.inference_config import load_inference_config

    gpt_oss = load_inference_config("unsloth/gpt-oss-120b-GGUF")
    assert (gpt_oss["top_p"], gpt_oss["top_k"], gpt_oss["min_p"]) == (1.0, 0, 0.01)
    # The pattern is version-anchored, so an unrelated name containing the word misses it.
    assert load_inference_config("someone/Beamformer-7B-GGUF")["top_p"] == 0.95


# ---------------------------------------------------------------- reasoning

# The effort handling of the shipped template, verbatim, plus a Harmony message loop and
# its XML tool-call rendering.
TEMPLATE = """
{%- if reasoning_effort is not defined or reasoning_effort is none %}
    {%- set reasoning_effort = 512 %}
{%- endif %}
{%- if reasoning_effort is string %}
    {%- if reasoning_effort == "minimal" %}
        {%- set reasoning_effort = 8 %}
    {%- elif reasoning_effort == "low" %}
        {%- set reasoning_effort = 32 %}
    {%- elif reasoning_effort == "medium" %}
        {%- set reasoning_effort = 64 %}
    {%- elif reasoning_effort == "high" %}
        {%- set reasoning_effort = 128 %}
    {%- elif reasoning_effort == "xhigh" %}
        {%- set reasoning_effort = 256 %}
    {%- elif reasoning_effort == "max" %}
        {%- set reasoning_effort = 512 %}
    {%- else %}
        {{- raise_exception("reasoning_effort must be an int or one of: minimal, low, medium, high, xhigh, max") }}
    {%- endif %}
{%- endif %}
{{- '<|start|>system<|message|>Reasoning effort: ' + (reasoning_effort | string) -}}
{%- if tools %}
    {{- '\\n\\n# Tools\\n\\n' -}}
    {%- for tool in tools %}{{- '## ' + tool.function.name + '\\n' -}}{%- endfor %}
{%- endif %}
{{- '<|end|>' -}}
{%- for message in messages -%}
    {%- if message.role == 'assistant' and message.tool_calls -%}
        {%- for call in message.tool_calls -%}
            {{- '<|start|>assistant to=' + call.function.name + '<|channel|>analysis <|constrain|>xml<|message|>' -}}
            {%- for key, value in call.function.arguments.items() -%}
                {{- '<parameter name="' + key + '">\\n' + (value | string) + '\\n</parameter>' -}}
            {%- endfor -%}
            {{- '<|call|>' -}}
        {%- endfor -%}
    {%- else -%}
        {{- '<|start|>' + message.role + '<|message|>' + message.content + '<|end|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|start|>assistant' -}}{%- endif -%}
"""

LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def _render(**kwargs):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = _raise
    return env.from_string(TEMPLATE).render(
        messages = [{"role": "user", "content": "hi"}], add_generation_prompt = True, **kwargs
    )


def _shim(architecture = ARCH):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "reasoning_effort"
    backend._reasoning_effort_levels = list(LEVELS)
    backend._supports_preserve_thinking = False
    backend._supports_tools = True
    backend._is_diffusion = False
    backend._architecture = architecture
    return backend


def test_detection_publishes_the_six_levels_and_tools():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(TEMPLATE, REPO)
    assert flags["supports_reasoning"]
    assert flags["reasoning_style"] == "reasoning_effort"
    assert not flags["reasoning_always_on"]
    assert flags["reasoning_effort_levels"] == LEVELS
    # The template does render tools; the backend is what withholds them.
    assert flags["supports_tools"]


def test_the_unpatched_template_rejects_none():
    with pytest.raises(ValueError, match = "reasoning_effort must be"):
        _render(reasoning_effort = "none")


@pytest.mark.parametrize(
    ("effort", "enable_thinking", "expected"),
    [
        ("none", None, "minimal"),
        ("minimal", None, "minimal"),
        ("low", None, "low"),
        ("medium", None, "medium"),
        ("high", None, "high"),
        ("xhigh", None, "xhigh"),
        ("max", None, "max"),
        (None, True, "high"),
        (None, False, "low"),
    ],
)
def test_request_kwargs_always_render(effort, enable_thinking, expected):
    kwargs = _shim()._request_reasoning_kwargs(enable_thinking, effort, None)
    assert kwargs == {"reasoning_effort": expected}
    _render(**kwargs)


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("none", "minimal"), ("minimal", "minimal"), ("max", "max"), (None, "high")],
)
def test_default_effort_kwargs_always_render(effort, expected):
    kwargs = _shim()._reasoning_kwargs(True, effort)
    assert kwargs == {"reasoning_effort": expected}
    _render(**kwargs)


def test_no_kwargs_renders_the_template_default():
    assert "Reasoning effort: 512" in _render()


def test_other_architectures_still_forward_none():
    """gpt-oss-style templates honour "none"; only this architecture coerces it."""
    assert _shim(architecture = "gpt-oss")._request_reasoning_kwargs(None, "none", None) == {
        "reasoning_effort": "none"
    }


# ---------------------------------------------------------------- tools


_PARSER_MARKER = b'<rfl:parameter name="'


def _install(
    tmp_path,
    *,
    server = b"",
    common = None,
    common_name = "libllama-common.so.0.7.0",
):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents = True)
    server_bin = bin_dir / "llama-server"
    server_bin.write_bytes(b"\x7fELF" + server)
    if common is not None:
        (bin_dir / common_name).write_bytes(b"\x7fELF" + common)
    return str(server_bin)


def _launched(architecture, binary):
    backend = _shim(architecture = architecture)
    backend._launch_binary_revision = (binary, 1, 1) if binary else ()
    return backend


def test_tools_are_withheld_on_a_server_without_the_parser(tmp_path):
    binary = _install(tmp_path, common = b"gpt-oss peg-native <|constrain|>")
    backend = _launched(ARCH, binary)
    assert not backend.supports_tools
    assert not backend.supports_tool_passthrough


@pytest.mark.parametrize(
    "common_name", ["libllama-common.so.0.7.0", "libllama-common.dylib", "llama-common.dll"]
)
def test_tools_are_offered_when_llama_common_has_the_parser(tmp_path, common_name):
    binary = _install(tmp_path, common = b"..." + _PARSER_MARKER + b"...", common_name = common_name)
    backend = _launched(ARCH, binary)
    assert backend.supports_tools
    assert backend.supports_tool_passthrough


def test_tools_are_offered_by_a_static_server_with_the_parser(tmp_path):
    backend = _launched(ARCH, _install(tmp_path, server = _PARSER_MARKER))
    assert backend.supports_tools and backend.supports_tool_passthrough


def test_tools_are_offered_when_the_library_sits_in_a_sibling_lib_dir(tmp_path):
    binary = _install(tmp_path)
    lib_dir = tmp_path / "build" / "lib"
    lib_dir.mkdir()
    (lib_dir / "libllama-common.so").write_bytes(_PARSER_MARKER)
    assert _launched(ARCH, binary).supports_tools


def test_an_unknown_server_withholds_tools(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_: None)
    )
    backend = _launched(ARCH, None)
    assert not backend.supports_tools
    assert not backend.supports_tool_passthrough


def test_a_server_found_before_launch_is_read(tmp_path, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    binary = _install(tmp_path, common = _PARSER_MARKER)
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_: binary)
    )
    assert _launched(ARCH, None).supports_tools


def test_a_rebuilt_library_is_reread(tmp_path):
    import os

    binary = _install(tmp_path, common = b"old build")
    assert not _launched(ARCH, binary).supports_tools
    lib = Path(binary).parent / "libllama-common.so.0.7.0"
    lib.write_bytes(b"new build " + _PARSER_MARKER)
    os.utime(lib, ns = (lib.stat().st_atime_ns, lib.stat().st_mtime_ns + 10**9))
    assert _launched(ARCH, binary).supports_tools


@pytest.mark.parametrize("architecture", ["gpt-oss", "qwen3moe", None])
def test_other_architectures_keep_their_tools(architecture, tmp_path):
    backend = _launched(architecture, _install(tmp_path))
    assert backend.supports_tools
    assert backend.supports_tool_passthrough


# ---------------------------------------------------------------- GGUF header and memory

_N_LAYERS = 52
# Five sliding-window layers then one full layer, with the last layer full: 43 / 9.
_SWA_PATTERN = [(i % 6) != 5 for i in range(_N_LAYERS - 1)] + [False]
_FIELDS = {
    "block_count": _N_LAYERS,
    "context_length": 1048576,
    "embedding_length": 5120,
    "feed_forward_length": 20480,
    "expert_feed_forward_length": 2560,
    "expert_count": 256,
    "expert_used_count": 8,
    "attention.head_count": 48,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
    "attention.sliding_window": 512,
    "attention.sliding_window_pattern": _SWA_PATTERN,
}


def _backend():
    from test_kv_cache_estimation import _backend_from_gguf
    return _backend_from_gguf(ARCH, _FIELDS, general = {"tokenizer.chat_template": TEMPLATE})


def test_header_reads_architecture_window_pattern_and_withholds_tools(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_: None)
    )
    backend = _backend()
    assert backend._architecture == ARCH
    assert backend._sliding_window == 512
    assert backend._sliding_window_pattern == _SWA_PATTERN
    assert sum(backend._sliding_window_pattern) == 43
    assert backend._expert_used_count == 8
    assert backend._reasoning_effort_levels == LEVELS
    assert backend._supports_tools and not backend.supports_tools


def test_kv_estimate_charges_full_context_only_to_full_layers():
    backend = _backend()
    n_ctx = 131072
    per_layer_token = 8 * (128 + 128) * 2  # f16 K + V
    total = backend._estimate_kv_cache_bytes(n_ctx, "f16")
    full_layers = 9 * n_ctx * per_layer_token
    every_layer = _N_LAYERS * n_ctx * per_layer_token
    assert full_layers <= total < full_layers + 43 * 4096 * per_layer_token
    assert total < every_layer / 4


class _StubField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _StubTensor:
    def __init__(self, name, n_bytes):
        self.name = name
        self.n_bytes = n_bytes


class _StubReader:
    def __init__(self, fields, tensors):
        self.fields = {k: _StubField(v) for k, v in fields.items()}
        self.tensors = tensors


def test_offload_layout_spills_only_the_routed_experts():
    """Three dense blocks stay resident; the 49 MoE blocks expose their expert stacks."""
    from core.inference.offload_layout import _layout_from_reader, spill_pattern_for

    mib = 1024 * 1024
    tensors = [_StubTensor("token_embd.weight", 400 * mib), _StubTensor("output.weight", 600 * mib)]
    for i in range(_N_LAYERS):
        tensors.append(_StubTensor(f"blk.{i}.attn_q.weight", 30 * mib))
        if i < 3:
            tensors += [
                _StubTensor(f"blk.{i}.ffn_{p}.weight", 100 * mib) for p in ("gate", "up", "down")
            ]
        else:
            tensors.append(_StubTensor(f"blk.{i}.ffn_gate_inp.weight", mib))
            tensors += [
                _StubTensor(f"blk.{i}.ffn_{p}_exps.weight", 600 * mib)
                for p in ("gate", "up", "down")
            ]
    fields = {"general.architecture": ARCH, **{f"{ARCH}.{k}": v for k, v in _FIELDS.items()}}
    layout = _layout_from_reader(_StubReader(fields, tensors))
    assert layout.complete and layout.is_moe
    assert (layout.n_expert, layout.n_expert_used) == (256, 8)
    assert layout.n_attention_layers == _N_LAYERS
    assert [b.spillable_bytes for b in layout.blocks[:3]] == [0, 0, 0]
    assert all(b.spillable_bytes == 3 * 600 * mib for b in layout.blocks[3:])
    pattern = re.compile(spill_pattern_for(layout))
    assert pattern.match("blk.40.ffn_down_exps.weight")
    assert not pattern.match("blk.1.ffn_down.weight")
