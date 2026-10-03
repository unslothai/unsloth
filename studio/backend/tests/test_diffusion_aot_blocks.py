# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted dynamo-included block graphs (``diffusion_aot_blocks``): a restart serves repeated blocks without tracing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_aot_blocks as aot  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


class Block(torch.nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, 4 * d)
        self.fc2 = torch.nn.Linear(4 * d, d)

    def forward(self, hidden_states, temb = None):
        x = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(hidden_states)), approximate = "tanh"))
        if temb is not None:
            x = x * (1 + temb[:, None])
        return hidden_states + x


class Model(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, n: int = 3):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(n))

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb = temb)
        return x


def _fake_pipe(model):
    return types.SimpleNamespace(transformer = model)


def test_install_is_inert_until_bound_and_respects_kill_switch(monkeypatch):
    monkeypatch.setattr(aot, "supported", lambda: True)
    model = Model()
    calls = []
    for b in model.blocks:
        b._compiled_call_impl = lambda *a, _b = b, **k: calls.append(_b) or _b._call_impl(*a, **k)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_AOT_BLOCKS", "0")
    assert aot.install(model, {"fullgraph": True}) is None
    monkeypatch.delenv("UNSLOTH_DIFFUSION_AOT_BLOCKS")
    assert aot.install(model, {"fullgraph": False}) is None  # graph breaks: aot_compile cannot serve it
    reg = aot.install(model, {"fullgraph": True})
    assert reg is not None and aot.registries(_fake_pipe(model)) == [reg]
    x, t = torch.randn(1, 4, 64), torch.randn(1, 64)
    with torch.no_grad():
        model(x, t)
    assert len(calls) == 3 and reg.stats["misses"] == 0 and not reg.recorded  # unbound: plain passthrough
    assert aot.bind(_fake_pipe(model), None) == 0 and reg.failed is not None


def test_subkey_tracks_sources_env_and_compile_kwargs(monkeypatch):
    base = aot.subkey({"fullgraph": True, "dynamic": None})
    assert aot.subkey({"fullgraph": True, "dynamic": None}) == base
    assert aot.subkey({"fullgraph": True, "dynamic": True}) != base
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SOME_PATCH", "0")
    assert aot.subkey({"fullgraph": True, "dynamic": None}) != base
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SOME_PATCH")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "/somewhere/else")  # where the home lives changes no graph
    assert aot.subkey({"fullgraph": True, "dynamic": None}) == base
    monkeypatch.setattr(aot, "_SOURCE_DIGEST", "different-sources")
    assert aot.subkey({"fullgraph": True, "dynamic": None}) != base


_SCRIPT = textwrap.dedent(
    r"""
    import json, os, sys, types
    sys.path.insert(0, os.environ["BACKEND"])
    import torch
    from torch._dynamo.utils import counters
    from core.inference import diffusion_aot_blocks as aot
    from tests.test_diffusion_aot_blocks import Model

    torch.manual_seed(0)
    model = Model().cuda().to(torch.bfloat16)
    x = torch.randn(1, 256, 64, device = "cuda", dtype = torch.bfloat16)
    t = torch.randn(1, 64, device = "cuda", dtype = torch.bfloat16)
    for b in model.blocks:
        b.compile(fullgraph = True, dynamic = None)
    if os.environ.get("HOOK") == "1":
        # A hook added after the artifact was saved: a serialised artifact would skip it, so the block must not use it.
        model.blocks[0].fc1.register_forward_hook(lambda m, i, o: o * 2)
    reg = aot.install(model, {"fullgraph": True, "dynamic": None})
    pipe = types.SimpleNamespace(transformer = model)
    aot.bind(pipe, types.SimpleNamespace(dir = os.environ["KEYDIR"]))
    with torch.inference_mode():
        x, t = x.clone(), t.clone()  # a pipeline's latents are inference tensors, like every block output
        out = model(x, t)
        saved = [reg.save_one(k) for k in reg.pending()]
    torch.save(out.cpu(), os.environ["OUT"])
    print("RESULT", json.dumps({"stats": reg.describe(), "saved": saved,
                                "frames": counters["stats"]["unique_graphs"]}))
    """
)


def _run(tmp_path: Path, tag: str, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(
        BACKEND = str(BACKEND),
        KEYDIR = str(tmp_path / "key"),
        OUT = str(tmp_path / f"{tag}.pt"),
        TORCHINDUCTOR_CACHE_DIR = str(tmp_path / "inductor"),
        TRITON_CACHE_DIR = str(tmp_path / "triton"),
        PYTHONPATH = str(BACKEND),
    )
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd = str(BACKEND), env = env, capture_output = True,
                       text = True, timeout = 900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[-1][len("RESULT "):])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_restart_serves_blocks_from_the_artifact_without_tracing_and_bit_identical(tmp_path):
    if not aot.supported():
        pytest.skip("this torch has no aot_compile")
    first = _run(tmp_path, "first")
    # The first block compiles through aot_compile and is persisted at once; the other two reuse its graph.
    assert first["stats"]["compiled"] == 1 and first["stats"]["saved"] == 1, first
    assert first["stats"]["hits"] == 2 and first["stats"]["misses"] == 0, first
    assert first["saved"] == [], first
    man = json.loads(next((tmp_path / "key").rglob("manifest.json")).read_text())
    assert len(man["entries"]) == 1 and man["entries"][0]["cls"] == "Block"
    restart = _run(tmp_path, "restart")
    assert restart["stats"]["loaded"] == 1 and restart["stats"]["hits"] == 3 and restart["stats"]["misses"] == 0
    assert restart["frames"] == 0, "dynamo traced a block the artifact covers"
    assert restart["saved"] == []
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "restart.pt"))
    # A hook the artifact never saw: that block takes the normal path (and honours the hook), the others still hit.
    hooked = _run(tmp_path, "hooked", {"HOOK": "1"})
    assert hooked["stats"]["hits"] == 2 and hooked["stats"]["misses"] == 1, hooked
    assert not torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "hooked.pt"))
    # Kill switch: the normal torch.compile path, same bits as the aot_compile path.
    off = _run(tmp_path, "off", {"UNSLOTH_DIFFUSION_AOT_BLOCKS": "0"})
    assert off["frames"] >= 1
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "off.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_a_graph_the_normal_path_did_not_compile_is_not_persisted(tmp_path, monkeypatch):
    """The save's FX-cache check: an aot trace that lowers a NEW graph (here: other compile kwargs) is dropped."""
    if not aot.supported():
        pytest.skip("this torch has no aot_compile")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    model = Model().cuda().to(torch.bfloat16)
    for b in model.blocks:
        b.compile(fullgraph = True, dynamic = None)
    # The registry believes the blocks compiled dynamic=True: its re-trace lowers a graph nobody ran.
    reg = aot.install(model, {"fullgraph": True, "dynamic": True})
    aot.bind(_fake_pipe(model), types.SimpleNamespace(dir = str(tmp_path / "key")))
    reg.compiled_classes.add("Block")  # a later signature: the normal path compiles it and the idle save re-traces
    x = torch.randn(1, 128, 64, device = "cuda", dtype = torch.bfloat16)
    t = torch.randn(1, 64, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        model(x.clone(), t.clone())
        assert [reg.save_one(k) for k in reg.pending()] == [False]
    assert not list((tmp_path / "key").rglob("*.aot"))
    torch._dynamo.reset()
