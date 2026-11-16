# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1 LoRAs on sd.cpp: staged split pieces and the sd-server conditioning-cache env."""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

from core.inference import diffusion_lora as dl
from core.inference import sd_cpp_server as srv
from core.inference.sd_cpp_args import SdCppModelFiles

HIDDEN = 8
RANK = 2
ALPHA = 2.0


def _write_safetensors(
    path,
    tensors,
    metadata = None,
):
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype = np.float32)
        raw = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for raw in blobs:
            f.write(raw)


def _read_safetensors(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32"
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype = np.float32).reshape(info["shape"])
    return header.get("__metadata__"), out


def _studio_flux1_lora(rng):
    """Same key layout as Studio's FLUX.1 trainer output (2 double + 2 single blocks, tiny dims)."""
    t = {}

    def pair(prefix, out_dim):
        t[f"{prefix}.lora_A.weight"] = rng.standard_normal((RANK, HIDDEN))
        t[f"{prefix}.lora_B.weight"] = rng.standard_normal((out_dim, RANK))

    for i in range(2):
        for mod in (
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ):
            pair(f"transformer.transformer_blocks.{i}.attn.{mod}", HIDDEN)
        for mod in ("to_q", "to_k", "to_v"):
            pair(f"transformer.single_transformer_blocks.{i}.attn.{mod}", HIDDEN)
    return t


# sd.cpp convert_diffusers_dit_to_original_flux: diffusers module -> (fused target, piece).
def _sdcpp_target(key):
    parts = key.split(".")
    assert parts[0] == "transformer"
    kind, idx, rest = parts[1], parts[2], ".".join(parts[3:-2])
    if kind == "single_transformer_blocks":
        pieces = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2, "proj_mlp": 3}
        return f"single_blocks.{idx}.linear1", pieces[rest]
    fused = {
        "attn.to_q": ("img_attn.qkv", 0),
        "attn.to_k": ("img_attn.qkv", 1),
        "attn.to_v": ("img_attn.qkv", 2),
        "attn.add_q_proj": ("txt_attn.qkv", 0),
        "attn.add_k_proj": ("txt_attn.qkv", 1),
        "attn.add_v_proj": ("txt_attn.qkv", 2),
        "attn.to_out.0": ("img_attn.proj", 0),
        "attn.to_add_out": ("txt_attn.proj", 0),
    }
    name, piece = fused[rest]
    return f"double_blocks.{idx}.{name}", piece


_MODEL_OUT_DIM = {
    "linear1": HIDDEN * 3 + HIDDEN * 4,
    "qkv": HIDDEN * 3,
    "proj": HIDDEN,
}


def _sdcpp_at_runtime_deltas(tensors, alpha = ALPHA):
    """Mirror of sd.cpp's LoraModel::get_out_diff: walk pieces to the first gap; skip size mismatches."""
    by_target = {}
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        target, piece = _sdcpp_target(key)
        by_target.setdefault(target, {})[piece] = prefix
    applied = {}
    for target, pieces in by_target.items():
        rows = []
        index = 0
        while index in pieces:
            a = tensors[pieces[index] + ".lora_A.weight"]
            b = tensors[pieces[index] + ".lora_B.weight"]
            rows.append(b @ a * (alpha / a.shape[0]))
            index += 1
        if not rows:
            continue
        delta = np.concatenate(rows, axis = 0)
        if delta.shape[0] != _MODEL_OUT_DIM[target.rsplit(".", 1)[-1]]:
            continue
        applied[target] = delta
    return applied


def _expected_deltas(tensors, alpha = ALPHA):
    """The adapter's own merged deltas (PEFT: B @ A * alpha / r), placed into the fused BFL rows."""
    out = {}
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        target, piece = _sdcpp_target(key)
        a = tensors[key]
        b = tensors[prefix + ".lora_B.weight"]
        kind = target.rsplit(".", 1)[-1]
        if target not in out:
            out[target] = np.zeros((_MODEL_OUT_DIM[kind], HIDDEN))
        out[target][piece * HIDDEN : piece * HIDDEN + b.shape[0]] += b @ a * (alpha / a.shape[0])
    return out


def _stage(tmp_path, src, family):
    resolved = [dl.ResolvedLora("lm-flux_1", "lm-flux_1", str(src), "safetensors", 1.0)]
    return dl.materialize_native_dir(resolved, tmp_path / "stage", real_files = True, family = family)


def test_flux1_staged_lora_applies_every_target_on_sdcpp(tmp_path):
    rng = np.random.default_rng(0)
    original = _studio_flux1_lora(rng)
    src = tmp_path / "lm-flux_1.safetensors"
    _write_safetensors(src, original, metadata = {"format": "pt"})

    out = _stage(tmp_path, src, "flux.1")
    assert len(out) == 1 and out[0].alias == "lm-flux_1"
    metadata, staged = _read_safetensors(out[0].path)
    assert metadata == {"format": "pt"}

    for key, value in original.items():
        assert np.array_equal(staged[key], value.astype(np.float32)), key

    applied = _sdcpp_at_runtime_deltas(staged)
    expected = _expected_deltas({k: v.astype(np.float32) for k, v in original.items()})
    assert set(applied) == set(expected)
    for target, delta in expected.items():
        np.testing.assert_allclose(applied[target], delta, rtol = 1e-5, atol = 1e-6)
        if ".linear1" in target:
            assert not applied[target][3 * HIDDEN :].any()

    _, again = _read_safetensors(src)
    assert set(again) == set(original)


def test_flux1_original_lora_loses_single_blocks_on_sdcpp(tmp_path):
    """Documents the defect the staging step fixes: the as-trained file drops every single block."""
    rng = np.random.default_rng(1)
    original = {k: v.astype(np.float32) for k, v in _studio_flux1_lora(rng).items()}
    applied = _sdcpp_at_runtime_deltas(original)
    assert not any(t.startswith("single_blocks.") for t in applied)
    assert any(t.startswith("double_blocks.") for t in applied)


def test_flux1_missing_split_pieces_completes_partial_groups_only():
    header = {
        "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": {
            "dtype": "BF16",
            "shape": [4, 16],
            "data_offsets": [0, 128],
        },
        "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight": {
            "dtype": "BF16",
            "shape": [16, 4],
            "data_offsets": [128, 256],
        },
        # Double block 0: only to_v -> to_q and to_k are added so the walk reaches piece 2.
        "transformer_blocks.0.attn.to_v.lora_A.weight": {
            "dtype": "F16",
            "shape": [4, 16],
            "data_offsets": [256, 384],
        },
        "transformer_blocks.0.attn.to_v.lora_B.weight": {
            "dtype": "F16",
            "shape": [16, 4],
            "data_offsets": [384, 512],
        },
        "transformer.transformer_blocks.1.attn.to_out.0.lora_A.weight": {
            "dtype": "F32",
            "shape": [4, 16],
            "data_offsets": [512, 768],
        },
        "transformer.transformer_blocks.1.attn.to_out.0.lora_B.weight": {
            "dtype": "F32",
            "shape": [16, 4],
            "data_offsets": [768, 1024],
        },
    }
    extra = dl.flux1_missing_split_pieces(header)
    single = "transformer.single_transformer_blocks.0."
    assert extra == {
        f"{single}attn.to_k.lora_A.weight": ("BF16", [1, 16]),
        f"{single}attn.to_k.lora_B.weight": ("BF16", [16, 1]),
        f"{single}attn.to_v.lora_A.weight": ("BF16", [1, 16]),
        f"{single}attn.to_v.lora_B.weight": ("BF16", [16, 1]),
        f"{single}proj_mlp.lora_A.weight": ("BF16", [1, 16]),
        f"{single}proj_mlp.lora_B.weight": ("BF16", [64, 1]),
        "transformer_blocks.0.attn.to_q.lora_A.weight": ("F16", [1, 16]),
        "transformer_blocks.0.attn.to_q.lora_B.weight": ("F16", [16, 1]),
        "transformer_blocks.0.attn.to_k.lora_A.weight": ("F16", [1, 16]),
        "transformer_blocks.0.attn.to_k.lora_B.weight": ("F16", [16, 1]),
    }


@pytest.mark.parametrize("family", [None, "z-image", "flux.2-klein"])
def test_other_families_link_the_file_as_is(tmp_path, family):
    rng = np.random.default_rng(2)
    src = tmp_path / "lm-flux_1.safetensors"
    _write_safetensors(src, _studio_flux1_lora(rng))
    out = _stage(tmp_path, src, family)
    assert os.path.samefile(out[0].path, src)


def test_flux1_complete_or_unparseable_files_link_as_is(tmp_path):
    rng = np.random.default_rng(3)
    complete = {
        k: v for k, v in _studio_flux1_lora(rng).items() if ".single_transformer_blocks." not in k
    }
    src = tmp_path / "complete.safetensors"
    _write_safetensors(src, complete)
    out = _stage(tmp_path, src, "flux.1")
    assert os.path.samefile(out[0].path, src)

    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"weights")
    resolved = [dl.ResolvedLora("j", "j", str(junk), "safetensors", 1.0)]
    out = dl.materialize_native_dir(resolved, tmp_path / "stage2", real_files = True, family = "flux.1")
    assert os.path.samefile(out[0].path, junk)
    assert not list((tmp_path / "stage2").glob("*.partial"))


# --- sd-server env: SD_COND_CACHE_STRICT ---


class _Popen:
    def __init__(self):
        self.pid = 4242
        self.returncode = None
        self._done = threading.Event()

    @property
    def stdout(self):
        def _gen():
            yield "listening on: http://127.0.0.1:1"
            self._done.wait()

        return _gen()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self._done.set()

    def wait(self, timeout = None):
        self._done.wait(timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self):
        self.terminate()


class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {}


class _Client:
    def get(
        self,
        url,
        headers = None,
        timeout = None,
    ):
        return _Resp()

    def close(self):
        pass


def _start_env(
    monkeypatch,
    files,
    env = None,
):
    monkeypatch.setattr(srv, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(srv, "forget_pid", lambda pid: None)
    monkeypatch.setattr(srv, "child_popen_kwargs", lambda: {})
    monkeypatch.setattr(srv, "windows_hidden_subprocess_kwargs", lambda: {})
    monkeypatch.delenv("SD_COND_CACHE_STRICT", raising = False)
    seen = {}
    popen = _Popen()

    def _fake_popen(cmd, **kwargs):
        seen["env"] = dict(kwargs["env"])
        return popen

    monkeypatch.setattr(srv.subprocess, "Popen", _fake_popen)
    s = srv.SdCppServer("/x/sd-server")
    s._client = _Client()
    try:
        s.start(files, startup_timeout = 5.0, env = env)
    finally:
        s.stop()
    return seen["env"]


def test_sd_server_with_clip_l_runs_strict_conditioning_cache(monkeypatch):
    files = SdCppModelFiles(
        diffusion_model = "/m/flux1-dev-Q8_0.gguf",
        vae = "/m/ae.safetensors",
        clip_l = "/m/clip_l.safetensors",
        t5xxl = "/m/t5xxl.safetensors",
    )
    assert _start_env(monkeypatch, files).get("SD_COND_CACHE_STRICT") == "1"
    env = _start_env(monkeypatch, files, env = {"SD_COND_CACHE_STRICT": "0"})
    assert env["SD_COND_CACHE_STRICT"] == "0"


def test_sd_server_without_clip_l_keeps_default_conditioning_cache(monkeypatch):
    files = SdCppModelFiles(diffusion_model = "/m/z.gguf", vae = "/m/vae.sft", llm = "/m/llm.sft")
    assert "SD_COND_CACHE_STRICT" not in _start_env(monkeypatch, files)


@pytest.mark.parametrize(
    "extra",
    [
        {"data_offsets": 5},
        {"data_offsets": ["a", "b"]},
        {"data_offsets": [0]},
    ],
)
def test_flux1_malformed_header_links_as_is(tmp_path, extra):
    """A header the stager cannot read falls back to the plain link instead of failing the generation."""
    rng = np.random.default_rng(3)
    src = tmp_path / "bad.safetensors"
    _write_safetensors(src, _studio_flux1_lora(rng))
    raw = src.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    header["x.weight"] = {"dtype": "F32", "shape": [1], **extra}
    blob = json.dumps(header).encode()
    src.write_bytes(len(blob).to_bytes(8, "little") + blob + raw[8 + n :])
    out = _stage(tmp_path, src, "flux.1")
    assert os.stat(out[0].path).st_ino == os.stat(src).st_ino
    assert not list((tmp_path / "stage").glob("*.partial"))
