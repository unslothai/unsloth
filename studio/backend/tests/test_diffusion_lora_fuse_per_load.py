# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-load ``lora_fuse``: explicit beats UNSLOTH_DIFFUSION_LORA_FUSE, which beats the default (off)."""

from __future__ import annotations

import dataclasses
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion
from core.inference import diffusion_lora_fuse as lora_fuse


@pytest.fixture
def fuse_calls(monkeypatch):
    calls = []
    monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    monkeypatch.setattr(diffusion, "_lora_fuse_dits", lambda pipe: [])
    monkeypatch.setattr(lora_fuse, "_on_card", lambda module: True)

    def fake_fuse(
        pipe,
        denoisers,
        logger = None,
    ):
        calls.append(denoisers)
        setattr(pipe, "_unsloth_lora_fused", True)
        return True

    monkeypatch.setattr(lora_fuse, "fuse", fake_fuse)
    monkeypatch.setattr(diffusion, "invalidate_cfg_replica", lambda *a, **k: None)
    return calls


def _pipe():
    return types.SimpleNamespace(unet = torch.nn.Linear(4, 4), _unsloth_loras = (("a", "/x", 1.0),))


def _state(offload = "none", **extra):
    return diffusion._LoadState(
        pipe = None,
        family = types.SimpleNamespace(name = "flux2"),
        repo_id = "r",
        base_repo = "b",
        device = "cuda",
        dtype = "bfloat16",
        cpu_offload = offload != "none",
        offload_policy = offload,
        speed_mode = "default",
        **extra,
    )


def _run(state, pipe):
    diffusion.DiffusionBackend._fuse_constant_loras(None, state, pipe)
    return diffusion.lora_fuse_status(pipe)


def test_load_state_defaults_to_auto():
    assert _state().lora_fuse is None


@pytest.mark.parametrize("offload", ["none", "group"])
def test_auto_with_env_unset_never_folds(fuse_calls, offload):
    pipe = _pipe()
    assert _run(_state(offload), pipe) == "disabled"
    assert fuse_calls == []


@pytest.mark.parametrize(
    "offload, expected", [("none", "fused"), ("group", "fused"), ("model", "offload")]
)
def test_auto_follows_the_env_switch(fuse_calls, monkeypatch, offload, expected):
    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")
    pipe = _pipe()
    assert _run(_state(offload), pipe) == expected
    assert len(fuse_calls) == (expected == "fused")


def test_on_folds_a_resident_group_offload_load(fuse_calls):
    pipe = _pipe()
    assert _run(_state("group", lora_fuse = "on"), pipe) == "fused"
    assert len(fuse_calls) == 1


def test_on_folds_without_offload_over_env_off(fuse_calls, monkeypatch):
    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "0")
    pipe = _pipe()
    assert _run(_state("none", lora_fuse = "on"), pipe) == "fused"


def test_on_still_refuses_moving_weights(fuse_calls):
    pipe = _pipe()
    assert _run(_state("model", lora_fuse = "on"), pipe) == "offload"
    assert fuse_calls == []


@pytest.mark.parametrize("offload", ["none", "group"])
def test_off_suppresses_the_env_fold(fuse_calls, monkeypatch, offload):
    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")
    pipe = _pipe()
    assert _run(_state(offload, lora_fuse = "off"), pipe) == "disabled"
    assert fuse_calls == []


def test_on_keeps_speed_off_bit_exact(fuse_calls):
    state = _state(
        "none",
        lora_fuse = "on",
        resolved = {"speed_mode": {"value": "off", "source": "explicit"}},
    )
    state = dataclasses.replace(state, speed_mode = "off")
    pipe = _pipe()
    assert _run(state, pipe) == "bit_exact"
    assert fuse_calls == []


def test_choice_persists_across_a_state_replace(fuse_calls):
    """A deferred compile or LoRA change republishes the state with dataclasses.replace: the choice must survive."""
    state = dataclasses.replace(
        _state("group", lora_fuse = "on"), speed_optims = ("compiled",), speed_deferred = False
    )
    assert state.lora_fuse == "on"
    pipe = _pipe()
    assert _run(state, pipe) == "fused"


@pytest.mark.parametrize(
    "request_, env, expected",
    [
        (None, None, (False, "default")),
        ("auto", None, (False, "default")),
        (None, "auto", (False, "default")),
        (None, "1", (True, "env")),
        (None, "0", (False, "env")),
        ("on", None, (True, "per_load")),
        ("on", "0", (True, "per_load")),
        ("off", "1", (False, "per_load")),
        (" ON ", None, (True, "per_load")),
        (True, None, (True, "per_load")),
        (False, "1", (False, "per_load")),
    ],
)
def test_precedence(monkeypatch, request_, env, expected):
    if env is None:
        monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    else:
        monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, env)
    assert lora_fuse.resolve(request_) == expected


def test_bad_choice_raises():
    with pytest.raises(ValueError):
        lora_fuse.normalize_lora_fuse("sometimes")


@pytest.mark.parametrize(
    "request_, env, value, source, origin",
    [
        (None, None, "off", "auto", "default"),
        (None, "1", "on", "auto", "env"),
        ("off", "1", "off", "explicit", "per_load"),
        ("on", None, "on", "explicit", "per_load"),
    ],
)
def test_resolved_entry_reports_the_source(monkeypatch, request_, env, value, source, origin):
    if env is None:
        monkeypatch.delenv(lora_fuse.LORA_FUSE_ENV, raising = False)
    else:
        monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, env)
    entry = diffusion.lora_fuse_resolved_entry(request_)
    assert (entry["value"], entry["source"], entry["origin"]) == (value, source, origin)
    assert entry["requested"] == (request_ if origin == "per_load" else None)
    assert entry["status"] == "applied"


def test_request_model_accepts_the_choice():
    from models.inference import DiffusionLoadRequest

    assert DiffusionLoadRequest(model_path = "x").lora_fuse is None
    assert DiffusionLoadRequest(model_path = "x", lora_fuse = " On ").lora_fuse == "on"
    assert DiffusionLoadRequest(model_path = "x", lora_fuse = "").lora_fuse is None
    with pytest.raises(Exception):
        DiffusionLoadRequest(model_path = "x", lora_fuse = "maybe")


def test_route_sends_only_an_explicit_choice_to_engines_that_take_it():
    from routes.inference import _lora_fuse_kwarg

    def diffusers(repo_id, *, lora_fuse = None):
        pass

    def native(repo_id, *, int8_attention = None):
        pass

    def wrapped(repo_id, **kwargs):
        pass

    assert _lora_fuse_kwarg(diffusers, "on") == {"lora_fuse": "on"}
    assert _lora_fuse_kwarg(diffusers, "off") == {"lora_fuse": "off"}
    assert _lora_fuse_kwarg(diffusers, "auto") == {}
    assert _lora_fuse_kwarg(diffusers, None) == {}
    assert _lora_fuse_kwarg(native, "on") == {}
    assert _lora_fuse_kwarg(wrapped, "on") == {}


def test_engine_entry_points_take_the_choice():
    import inspect
    for fn in (diffusion.DiffusionBackend.begin_load, diffusion.DiffusionBackend.load_pipeline):
        assert inspect.signature(fn).parameters["lora_fuse"].default is None


def test_dense_load_state_is_built_with_the_choice():
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(diffusion.DiffusionBackend.load_pipeline)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_LoadState"
    ]
    assert calls
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("lora_fuse"), ast.Name) and kw["lora_fuse"].id == "lora_fuse"


def test_status_response_keeps_the_origin(monkeypatch):
    from models.inference import DiffusionStatusResponse

    monkeypatch.setenv(lora_fuse.LORA_FUSE_ENV, "1")
    response = DiffusionStatusResponse(
        loaded = True, resolved = {"lora_fuse": diffusion.lora_fuse_resolved_entry(None)}
    )
    entry = response.resolved["lora_fuse"]
    assert (entry.value, entry.source, entry.origin) == ("on", "auto", "env")
