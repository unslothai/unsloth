# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from dataclasses import FrozenInstanceError
import json

import pytest

from core.inference.llama_custom_config import (
    CustomConfigError,
    CustomConfigSource,
    compile_custom_config,
    parse_config_source,
    parse_option_catalog,
)


# Native help's alias padding, wrapped declaration, description/default/env lines.
HELP = """----- common params -----
-m,    --model FNAME                    model file
                                        (env: LLAMA_ARG_MODEL)
-mm,   --mmproj FILE                    projector file
                                        (env: LLAMA_ARG_MMPROJ)
-np,   --parallel N                     number of server slots (default: 1)
                                        (env: LLAMA_ARG_N_PARALLEL)
-c,    --ctx-size N                     context (default: 0, 0 = loaded from model)
                                        (env: LLAMA_ARG_CTX_SIZE)
-b,    --batch-size N                   logical batch (default: 2048)
-ub,   --ubatch-size N                  physical batch (default: 512)
-t,    --threads N                      CPU threads (default: -1)
-ngl,  --gpu-layers, --n-gpu-layers N   layers (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS)
-ncmoe, --n-cpu-moe N                   CPU expert layers
-ctk,  --cache-type-k TYPE              cache (default: f16)
-ctv,  --cache-type-v TYPE              cache (default: f16)
-fit,  --fit [on|off]                   fit ('on' or 'off', default: 'on')
--temp N                                temperature (default: 0.8)
--top-k N                               top k (default: 40)
--top-p N                               top p (default: 0.95)
--min-p N                               min p (default: 0.05)
--repeat-penalty N                      repeat penalty
--presence-penalty N                    presence penalty
--frequency-penalty N                   frequency penalty
--seed N                                seed
--chat-template-kwargs STRING          template kwargs
--reasoning-effort STRING              reasoning effort
--mmproj-offload, --no-mmproj-offload
                                        offload projector (default: true)
                                        (env: LLAMA_ARG_MMPROJ_OFFLOAD)
-kvo,  --kv-offload, -nkvo, --no-kv-offload
                                        offload KV (default: true)
                                        (env: LLAMA_ARG_KV_OFFLOAD)
--no-warmup                             skip warmup
--mlock                                 lock model
-sm,   --split-mode {none,layer,row,tensor}
                                        splitting (default: layer)
--load-mode MODE                        loading (default: auto)
--spec-default                          enable default speculative decoding config
--host HOST                             binding
                                        (env: LLAMA_ARG_HOST)
--port PORT                             port
--api-key KEY                           API key
                                        (env: LLAMA_API_KEY)
-ag,   --agent, -no-ag, --no-agent       tools
                                        (env: LLAMA_ARG_AGENT)
--models-preset FILE                    router preset
--config FILE                           system configuration
--model-draft FILE                     draft model
--gpt-oss-120b-default                  use gpt-oss-120b (note: can download weights from the internet)
--control-vector-layer-range START END
                                        control vector layers
--old-option N                          option removed; use --fit
--json-schema STRING                    arbitrary supported one-value option
"""


@pytest.fixture
def catalog():
    return parse_option_catalog(HELP)


@pytest.mark.parametrize("value", ["1e100", "-1e100", "1e-100", "-1e-100", "1e-1000"])
def test_native_float_domain_rejects_overflow_and_underflow(catalog, value):
    with pytest.raises(CustomConfigError, match = "native float domain"):
        compile_custom_config(source(f"[*]\nnp=1\ntemp={value}"), catalog)


@pytest.mark.parametrize("value", ["0", "-0.0", "0.25", "-0.5", "1e-40", "3.4028234663852886e38"])
def test_native_float_domain_preserves_representable_values(catalog, value):
    compiled = compile_custom_config(source(f"[*]\nnp=1\ntemp={value}"), catalog)
    assert compiled.summary()["request_defaults"]["temperature"] == float(value)


def source(ini, section = None):
    return {"version": 1, "mode": "custom", "ini": ini, "section": section}


def compile_ini(
    ini,
    catalog,
    section = None,
    **kwargs,
):
    return compile_custom_config(source(ini, section), catalog, **kwargs)


def test_reporter_selected_section_owns_context_json_and_resources(catalog, tmp_path):
    model, projector = tmp_path / "model with spaces.gguf", tmp_path / "mmproj.gguf"
    model.touch()
    projector.touch()
    ini = f"""[*]
version = 1
c = 128000
np = 2
chat-template-kwargs = {{"global": true}}
[free-27B-Q3.8]
m = {model}
mm = {projector}
ctx-size = 56000
np = 1 # one slot
b = 1024 ; batch comment
ub = 256
fit = off
ngl = -1
mmproj-offload = false
no-warmup = true
min-p = 0.0
chat-template-kwargs = {{"enable_thinking": false, "reasoning_effort": "high"}}
load-on-startup = false
[other-model]
c = 262144
unknown-other-binary-option = 1
"""
    compiled = compile_ini(
        ini, catalog, "free-27B-Q3.8", model_path = str(model), mmproj_path = str(projector)
    )
    assert compiled.n_parallel == 1
    assert "--parallel" not in compiled.argv and "--model" not in compiled.argv
    assert "--mmproj" not in compiled.argv
    assert compiled.argv[compiled.argv.index("--ctx-size") + 1] == "56000"
    assert "128000" not in compiled.argv and "262144" not in compiled.argv
    assert compiled.argv[compiled.argv.index("--fit") + 1] == "off"
    assert "--no-mmproj-offload" in compiled.argv and "--no-warmup" in compiled.argv
    summary = compiled.summary()
    assert summary["tuning"]["n_batch"] == 1024
    assert summary["tuning"]["gpu_layers"] == -1
    assert summary["request_defaults"]["min_p"] == 0.0
    assert summary["request_defaults"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_effort": "high",
    }
    assert (
        json.loads(compiled.argv[compiled.argv.index("--chat-template-kwargs") + 1])
        == summary["request_defaults"]["chat_template_kwargs"]
    )
    assert "Load action controls startup" in summary["diagnostics"][0]


def test_catalog_aliases_arity_defaults_negative_pairs(catalog):
    parallel = next(d for d in catalog if "--parallel" in d["names"])
    assert parallel == {
        "names": ["-np", "--parallel"],
        "env": ["LLAMA_ARG_N_PARALLEL"],
        "arity": 1,
        "default": "1",
        "negative_names": [],
    }
    projector = next(d for d in catalog if "--mmproj-offload" in d["names"])
    assert projector["arity"] == 0
    assert projector["negative_names"] == ["--no-mmproj-offload"]
    assert "LLAMA_ARG_NO_MMPROJ_OFFLOAD" in projector["env"]
    kv = next(d for d in catalog if "--kv-offload" in d["names"])
    assert kv["negative_names"] == ["-nkvo", "--no-kv-offload"]
    assert next(d for d in catalog if "--control-vector-layer-range" in d["names"])["arity"] == 2
    assert not any("--old-option" in d["names"] for d in catalog)


def test_help_device_placeholder_list_is_not_an_enum(catalog):
    devices = parse_option_catalog(
        "-dev, --device <dev1,dev2,..>             comma-separated device list\n"
    )
    assert devices[0]["arity"] == 1
    assert "choices" not in devices[0]
    assert compile_ini("[*]\ndevice=none", (*catalog, *devices)).argv == ("--device", "none")


@pytest.mark.parametrize("key", ["c", "ctx-size", "LLAMA_ARG_CTX_SIZE"])
def test_aliases_normalize_to_same_digest(key, catalog):
    compiled = compile_ini(f"[*]\n{key}=56000", catalog)
    control = compile_ini("[*]\nctx-size=56000", catalog)
    assert compiled.argv == control.argv
    assert compiled.digest == control.digest


@pytest.mark.parametrize(
    ("key", "value", "flag"),
    [
        ("mmproj-offload", "false", "--no-mmproj-offload"),
        ("no-mmproj-offload", "true", "--no-mmproj-offload"),
        ("no-mmproj-offload", "false", "--mmproj-offload"),
        ("LLAMA_ARG_MMPROJ_OFFLOAD", "0", "--no-mmproj-offload"),
        # Native preset parse_bool_arg only inverts CLI keys; environment keys map
        # to the option unchanged, unlike environment handling at process startup.
        ("LLAMA_ARG_NO_MMPROJ_OFFLOAD", "true", "--mmproj-offload"),
        ("nkvo", "false", "--kv-offload"),
        ("no-warmup", "true", "--no-warmup"),
    ],
)
def test_boolean_semantics(key, value, flag, catalog):
    compiled = compile_ini(f"[*]\n{key}={value}", catalog)
    assert compiled.argv == (flag,)


def test_false_unpaired_switch_is_omitted_but_preserved_in_summary(catalog):
    compiled = compile_ini("[*]\nmlock=false", catalog)
    assert compiled.argv == ()
    assert compiled.summary()["tuning"]["mlock"] is False


@pytest.mark.parametrize("value", ["FALSE", "yes", "no", "arbitrary"])
def test_invalid_boolean_is_not_silently_treated_truthy(value, catalog):
    with pytest.raises(CustomConfigError, match = "native boolean"):
        compile_ini(f"[*]\nmlock={value}", catalog)


def test_native_values_are_not_slider_clamped(catalog):
    compiled = compile_ini(
        "[*]\nb=100000\nub=90000\nt=-1\nngl=-1\ntop-k=-1\ntemp=-1\nmin-p=0\npresence-penalty=-3\nseed=4294967295",
        catalog,
    )
    assert compiled.summary()["tuning"]["n_batch"] == 100000
    assert compiled.summary()["tuning"]["n_threads"] == -1
    assert compiled.summary()["request_defaults"]["temperature"] == -1
    assert compiled.summary()["request_defaults"]["top_k"] == -1
    assert compiled.summary()["request_defaults"]["presence_penalty"] == -3
    assert compiled.summary()["request_defaults"]["seed"] == 4294967295


@pytest.mark.parametrize(
    "text",
    ["np=0", "np=-1", "np=65", "c=-1", "b=0", "t=nan", "min-p=nan", "temp=inf", "c=2147483648"],
)
def test_typed_invalid_domains(text, catalog):
    with pytest.raises(CustomConfigError):
        compile_ini("[*]\n" + text, catalog)


@pytest.mark.parametrize(
    "key",
    [
        "host",
        "LLAMA_ARG_HOST",
        "port",
        "api-key",
        "LLAMA_API_KEY",
        "agent",
        "ag",
        "no-agent",
        "LLAMA_ARG_AGENT",
        "models-preset",
        "config",
        "model-draft",
        "gpt-oss-120b-default",
    ],
)
def test_reserved_aliases_fail_before_any_resource_execution(key, catalog):
    with pytest.raises(CustomConfigError, match = "managed by Studio") as error:
        compile_ini(f"[*]\n{key}=never-leak-secret", catalog)
    assert "never-leak-secret" not in str(error.value)


def test_supported_spec_default_is_not_misclassified_as_model_preset(catalog):
    assert compile_ini("[*]\nspec-default=true", catalog).argv == ("--spec-default",)


@pytest.mark.parametrize("key", ["m", "model", "LLAMA_ARG_MODEL"])
def test_model_alias_reconciliation(key, catalog, tmp_path):
    model = tmp_path / "selected.gguf"
    model.touch()
    assert compile_ini(f"[*]\n{key}={model}", catalog, model_path = str(model)).argv == ()
    with pytest.raises(CustomConfigError, match = "must match"):
        compile_ini(f"[*]\n{key}={tmp_path / 'other.gguf'}", catalog, model_path = str(model))


def test_resource_preflight_defers_identity_only(catalog):
    result = compile_ini("[*]\nm=placeholder", catalog, validate_resources = False)
    assert "before launch" in result.diagnostics[0]
    with pytest.raises(CustomConfigError, match = "managed by Studio"):
        compile_ini("[*]\nm=placeholder\nhost=evil", catalog, validate_resources = False)


def test_reconciled_resource_must_exist(catalog, tmp_path):
    path = str(tmp_path / "nonexistent.gguf")
    with pytest.raises(CustomConfigError, match = "existing file"):
        compile_ini(f"[*]\nm={path}", catalog, model_path = path)


def test_windows_path_identity_and_literal_no_expansion(catalog, monkeypatch):
    monkeypatch.setattr("core.inference.llama_custom_config.os.path.isfile", lambda _: True)
    result = compile_ini(
        "[*]\nm=C:\\Models\\folder\\..\\A Model.gguf",
        catalog,
        model_path = "c:/models/a model.gguf",
        platform = "win32",
    )
    assert result.argv == ()
    with pytest.raises(CustomConfigError, match = "must match"):
        compile_ini(
            "[*]\nm=%MODEL_PATH%", catalog, model_path = "c:/models/a model.gguf", platform = "win32"
        )


def test_selection_required_even_for_one_named_section(catalog):
    with pytest.raises(CustomConfigError, match = "explicit named"):
        compile_ini("[only]\nc=1", catalog)
    with pytest.raises(CustomConfigError, match = "does not exist"):
        compile_ini("[only]\nc=1", catalog, "missing")
    assert compile_ini("[only]\nc=1", catalog, "only").summary()["tuning"]["n_ctx"] == 1


def test_native_duplicate_key_and_section_replacement(catalog):
    result = compile_ini("[*]\nc=10\nc=20\n[chosen]\nb=100\n[chosen]\nc=30", catalog, "chosen")
    assert result.summary()["tuning"]["n_ctx"] == 30
    assert result.summary()["tuning"]["n_batch"] == 2048
    with pytest.raises(CustomConfigError, match = "multiple aliases"):
        compile_ini("[*]\nc=10\nctx-size=20", catalog)


def test_global_and_selected_aliases_replace_without_duplicate_argv(catalog):
    result = compile_ini("[*]\nLLAMA_ARG_CTX_SIZE=10\n[chosen]\nc=20", catalog, "chosen")
    assert result.argv == ("--ctx-size", "20")


def test_quotes_are_literal_comments_are_not_quote_aware(catalog):
    result = compile_ini('[*]\njson-schema="literal spaces" ; comment', catalog)
    assert result.argv == ("--json-schema", '"literal spaces"')
    with pytest.raises(CustomConfigError, match = "valid JSON"):
        compile_ini('[*]\nchat-template-kwargs={"text":"hash # truncates"}', catalog)


@pytest.mark.parametrize("ini", ["[*]\n c=1", "[*]\n--ctx-size=1", "[broken", "[*]\nnot-an-entry"])
def test_malformed_native_grammar_fails(ini, catalog):
    with pytest.raises(CustomConfigError, match = "syntax"):
        compile_ini(ini, catalog)


@pytest.mark.parametrize("key", ["unknown", "old-option", "control-vector-layer-range"])
def test_unknown_removed_or_multivalue_rejected(key, catalog):
    with pytest.raises(CustomConfigError):
        compile_ini(f"[*]\n{key}=1", catalog)


def test_empty_and_partial_catalog_fail(catalog):
    for incomplete in [(), catalog[:1], catalog[2:3]]:
        with pytest.raises(CustomConfigError, match = "probe"):
            compile_ini("[*]\nc=1", incomplete)


def test_conflicting_help_aliases_fail(catalog):
    duplicate = {
        "names": ["-c", "--different"],
        "env": [],
        "arity": 1,
        "default": None,
        "negative_names": [],
    }
    with pytest.raises(CustomConfigError, match = "ambiguous"):
        compile_ini("[*]\nc=1", (*catalog, duplicate))


def test_auto_parallel_default_requires_explicit_slots():
    catalog = parse_option_catalog(
        HELP.replace("slots (default: 1)", "slots (default: -1, -1 = auto)")
    )
    with pytest.raises(CustomConfigError, match = "set np explicitly"):
        compile_ini("[*]\nc=1", catalog)
    assert compile_ini("[*]\nc=1\nnp=3", catalog).n_parallel == 3


def test_absent_reset_and_wire_roundtrip():
    assert parse_config_source(None) is None
    reset = parse_config_source({"version": 1, "mode": "managed"})
    assert reset == CustomConfigSource()
    assert reset.to_wire() == {"version": 1, "mode": "managed"}
    custom = source("[*]\nc=1")
    assert parse_config_source(custom).to_wire() == custom


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"version": True, "mode": "managed"},
        {"version": 2, "mode": "managed"},
        {"version": 1, "mode": "bad"},
        {"version": 1, "mode": "managed", "ini": "secret"},
        source(""),
        source("[*]\nc=\x00"),
        source("[*]\nc=\ud800"),
        source("x" * 65537),
        {**source("[*]"), "argv": ["--host"]},
    ],
)
def test_wire_shape_encoding_and_size_bounds(value):
    with pytest.raises(CustomConfigError):
        parse_config_source(value)


def test_source_instance_is_revalidated():
    with pytest.raises(CustomConfigError):
        parse_config_source(CustomConfigSource(version = 4))


def test_compiled_and_nested_values_are_immutable(catalog):
    result = compile_ini('[*]\nchat-template-kwargs={"nested":{"a":[1,2]}}', catalog)
    with pytest.raises(FrozenInstanceError):
        result.n_parallel = 3
    with pytest.raises(FrozenInstanceError):
        dict(result.request_defaults)["chat_template_kwargs"].encoded = "{}"
    decoded = result.summary()
    decoded["request_defaults"]["chat_template_kwargs"]["nested"]["a"].append(3)
    assert result.summary()["request_defaults"]["chat_template_kwargs"]["nested"]["a"] == [1, 2]


def test_canonical_digest_comments_order_json_and_selection(catalog):
    a = compile_ini('[*]\nc=56000\nchat-template-kwargs={"b":2,"a":1}', catalog)
    b = compile_ini(
        '[*]\nchat-template-kwargs = { "a": 1, "b": 2 }\nctx-size=56000 # note', catalog
    )
    assert a.digest == b.digest and a.source_digest != b.source_digest
    assert (
        a.digest != compile_ini('[*]\nc=56001\nchat-template-kwargs={"b":2,"a":1}', catalog).digest
    )
    assert (
        compile_ini("[a]\nc=1\n[b]\nc=1", catalog, "a").digest
        != compile_ini("[a]\nc=1\n[b]\nc=1", catalog, "b").digest
    )


def test_argument_and_windows_serialization_bounds(catalog):
    # Backslashes before quotes need escaping in CreateProcess's one string.
    value = '\\" ' * 6000
    with pytest.raises(CustomConfigError, match = "Windows command"):
        compile_ini("[*]\njson-schema=" + value, catalog, platform = "win32")
    with pytest.raises(CustomConfigError, match = "size limit"):
        compile_ini("[*]\njson-schema=" + "x" * 33000, catalog, platform = "linux")


def test_section_and_entry_count_bounds(catalog):
    with pytest.raises(CustomConfigError, match = "section limit"):
        compile_ini("[*]\n" * 129, catalog)
    with pytest.raises(CustomConfigError, match = "entry limit"):
        compile_ini("[*]\n" + "c=1\n" * 2049, catalog)
