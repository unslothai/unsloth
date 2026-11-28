# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Host-side cost of fp8 loads: the safetensors index is read once per from_pretrained, and the
leftover dequant names each module by its first `named_modules` name without an identity scan."""

import json
import os

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

import unsloth  # noqa: F401
import huggingface_hub
from unsloth.models import loader_utils
from unsloth.models.loader_utils import (
    _FP8_DTYPES,
    _dequantize_leftover_fp8_params,
    _load_fp8_weight_map,
    _load_probe_memo_scope,
)

REPO = "unsloth/not-a-real-fp8-repo"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Fake Hub: count index downloads and parses, serve whatever `index` holds now."""
    state = {"downloads": 0, "parses": 0, "fail": 0, "index": {"a.weight": "s1"}}
    path = tmp_path / "model.safetensors.index.json"

    def fake_download(repo_id, filename, **kwargs):
        assert repo_id == REPO
        state["downloads"] += 1
        if state["fail"]:
            state["fail"] -= 1
            raise OSError("offline")
        if filename != "model.safetensors.index.json":
            raise FileNotFoundError(filename)
        path.write_text(json.dumps({"weight_map": state["index"]}))
        return str(path)

    real_load = json.load

    def counting_load(f, *a, **k):
        state["parses"] += 1
        return real_load(f, *a, **k)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(json, "load", counting_load)
    return state


def _one_load(**kw):
    # backfill (index_only), scale restore, leftover dequant: the three reads of one fp8 load.
    return [
        _load_fp8_weight_map(REPO, False, None, "main", None, None, index_only = True, **kw),
        _load_fp8_weight_map(REPO, False, None, "main", None, None, **kw),
        _load_fp8_weight_map(REPO, False, None, "main", None, None, variant = None, **kw),
    ]


def test_one_load_reads_the_index_once(hub):
    maps = _load_probe_memo_scope(_one_load)()
    assert maps == [{"a.weight": "s1"}] * 3
    assert hub["downloads"] == 1 and hub["parses"] == 1
    # Callers get their own dict: mutating one cannot leak into the next caller.
    maps[0]["x"] = "y"
    assert _load_probe_memo_scope(lambda: _load_fp8_weight_map(REPO, False, None, "main"))() == {
        "a.weight": "s1"
    }


def test_the_next_load_sees_a_new_upload(hub):
    _load_probe_memo_scope(_one_load)()
    hub["index"] = {"a.weight": "s2"}
    assert _load_probe_memo_scope(_one_load)() == [{"a.weight": "s2"}] * 3
    assert hub["downloads"] == 2


def test_outside_a_load_nothing_is_cached(hub):
    assert _one_load() == [{"a.weight": "s1"}] * 3
    assert hub["downloads"] == 3 and hub["parses"] == 3


def test_a_failed_read_is_not_cached(hub):
    hub["fail"] = 1

    def load():
        return _load_fp8_weight_map(REPO, False, None, "main"), _load_fp8_weight_map(
            REPO, False, None, "main"
        )

    first, second = _load_probe_memo_scope(load)()
    assert first is None and second == {"a.weight": "s1"}


def test_revision_subfolder_and_variant_do_not_share_an_entry(hub):
    def load():
        _load_fp8_weight_map(REPO, False, None, "main")
        _load_fp8_weight_map(REPO, False, None, "other")
        _load_fp8_weight_map(REPO, False, None, "main", subfolder = "text")

    _load_probe_memo_scope(load)()
    assert hub["downloads"] == 3


@pytest.mark.skipif(not _FP8_DTYPES, reason = "torch has no float8 dtype")
def test_shared_module_keeps_its_first_name_and_dequantizes_identically(tmp_path):
    """A module reachable under two names is named by its first `named_modules` entry, as the
    identity scan did; the result matches the reference dequant."""
    fp8 = _FP8_DTYPES[0]
    torch.manual_seed(0)
    shared = nn.Linear(8, 8, bias = False)
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    model.layers[0].proj = shared
    model.layers[1].proj = shared  # alias: named_modules yields only layers.0.proj
    model.layers[1].own = nn.Linear(8, 8, bias = False)
    model.config = None
    tensors, expected = {}, {}
    for name in ("layers.0.proj", "layers.1.proj", "layers.1.own"):
        w = torch.randn(8, 8)
        s = w.abs().amax() / 448.0
        tensors[name + ".weight"] = (w / s).to(fp8)
        tensors[name + ".weight_scale_inv"] = s.reshape(1)
        expected[name] = (tensors[name + ".weight"].float() * s).to(torch.bfloat16)
    shared.weight = nn.Parameter(tensors["layers.0.proj.weight"], requires_grad = False)
    model.layers[1].own.weight = nn.Parameter(tensors["layers.1.own.weight"], requires_grad = False)
    save_file(tensors, os.path.join(tmp_path, "s.safetensors"))
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "s.safetensors" for k in tensors}}, f)

    done, _ = _dequantize_leftover_fp8_params(model, str(tmp_path), torch.bfloat16)
    assert done == 2
    assert torch.equal(shared.weight.detach(), expected["layers.0.proj"])
    assert torch.equal(model.layers[1].own.weight.detach(), expected["layers.1.own"])


def test_leftover_dequant_has_no_per_key_identity_scan():
    import inspect
    src = inspect.getsource(loader_utils._dequantize_leftover_fp8_params)
    assert "if mod is module" not in src
