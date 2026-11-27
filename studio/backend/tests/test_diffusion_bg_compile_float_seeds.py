# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background compile float seeds (CPU, ``aot_eager``)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._dynamo.symbolic_convert")

from core.inference import diffusion_bg_compile as bg  # noqa: E402

if (
    bg._tensorify_state() is None
):  # pragma: no cover - a torch without TensorifyState has nothing to seed
    pytest.skip("this torch has no TensorifyState", allow_module_level = True)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(16, eps = 1e-6)
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x):
        return x + self.lin(self.norm(x))


class Net(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _net() -> Net:
    torch.manual_seed(0)
    net = Net().eval()
    for block in net.blocks:
        block.compile(backend = "aot_eager", dynamic = True)
    # Studio's block compile guard honours force_eager: the recording render must not trace the blocks itself.
    bg.gate_compiled_eager(list(net.blocks))
    return net


@pytest.fixture(autouse = True)
def _fresh_dynamo(monkeypatch):
    monkeypatch.delenv(bg.FLOAT_SEEDS_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def restarts(monkeypatch):
    count = {"n": 0}
    cls = torch._dynamo.exc.TensorifyScalarRestartAnalysis
    orig = cls.__init__

    def counting(self, *args, **kwargs):
        count["n"] += 1
        orig(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__init__", counting)
    return count


def _background_compile(seed_dir, x):
    net = _net()
    job = bg.arm(net, seed_dir = seed_dir)
    assert job is not None
    with torch.no_grad(), bg.force_eager():
        net(x)
    assert job.kick()
    job.wait()
    assert job.state == "done", job.error
    with torch.no_grad():
        out = net(x)
    return job, out


def test_second_load_traces_once_into_the_same_graph(tmp_path, restarts):
    x = torch.randn(2, 5, 16)
    first, out_first = _background_compile(tmp_path, x)
    assert restarts["n"] >= 1  # the unseeded compile restarts, as on main
    saved = json.loads((tmp_path / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8"))
    assert saved["names"] and all(n.startswith("zf") for n in saved["names"])
    assert first.float_seeds == 0 and first.float_seeds_learned == len(saved["names"])

    torch._dynamo.reset()
    restarts["n"] = 0
    second, out_second = _background_compile(tmp_path, x)
    assert restarts["n"] == 0
    assert second.float_seeds == len(saved["names"])
    assert second.describe()["float_seeds"] == len(saved["names"])
    assert torch.equal(out_first, out_second)
    assert json.loads((tmp_path / bg.FLOAT_SEEDS_FILE).read_text(encoding = "utf-8")) == saved


def test_no_recompile_after_a_seeded_compile(tmp_path):
    from torch._dynamo.utils import counters

    x = torch.randn(2, 5, 16)
    _background_compile(tmp_path, x)
    torch._dynamo.reset()
    job, _ = _background_compile(tmp_path, x)
    before = dict(counters["stats"])
    with torch.no_grad():
        for length in (5, 7, 9):
            job.module(torch.randn(2, length, 16))
    # dynamic=True: other lengths reuse the graph; nothing new is traced
    assert counters["stats"].get("unique_graphs", 0) == before.get("unique_graphs", 0)


def test_kill_switch_keeps_the_restart(tmp_path, restarts, monkeypatch):
    monkeypatch.setenv(bg.FLOAT_SEEDS_ENV, "0")
    x = torch.randn(2, 5, 16)
    _background_compile(tmp_path, x)
    assert not (tmp_path / bg.FLOAT_SEEDS_FILE).exists()
    torch._dynamo.reset()
    restarts["n"] = 0
    job, _ = _background_compile(tmp_path, x)
    assert restarts["n"] >= 1 and job.float_seeds == 0


def test_seeds_for_other_code_are_ignored(tmp_path, restarts):
    x = torch.randn(2, 5, 16)
    _background_compile(tmp_path, x)
    path = tmp_path / bg.FLOAT_SEEDS_FILE
    data = json.loads(path.read_text(encoding = "utf-8"))
    data["signature"] = "0" * 64
    path.write_text(json.dumps(data), encoding = "utf-8")
    torch._dynamo.reset()
    restarts["n"] = 0
    job, _ = _background_compile(tmp_path, x)
    assert job.float_seeds == 0 and restarts["n"] >= 1
    assert json.loads(path.read_text(encoding = "utf-8"))["signature"] == bg._code_signature(_net())


@pytest.mark.parametrize(
    "payload",
    [
        "{not json",
        json.dumps({"version": 99, "signature": "x", "names": ["zf1"]}),
        json.dumps([1, 2]),
    ],
)
def test_torn_or_foreign_files_are_ignored(tmp_path, restarts, payload):
    (tmp_path / bg.FLOAT_SEEDS_FILE).write_text(payload, encoding = "utf-8")
    job, _ = _background_compile(tmp_path, torch.randn(2, 5, 16))
    assert job.float_seeds == 0 and restarts["n"] >= 1


def test_only_symfloat_names_are_seeded(tmp_path):
    sig = bg._code_signature(_net())
    assert sig is not None
    (tmp_path / bg.FLOAT_SEEDS_FILE).write_text(
        json.dumps({"version": 1, "signature": sig, "names": ["zf3", "s4", "rm -rf", 7]}),
        encoding = "utf-8",
    )
    assert bg._read_float_seeds(tmp_path / bg.FLOAT_SEEDS_FILE, sig) == {"zf3"}


def test_no_seed_dir_is_main_behaviour(restarts):
    job, _ = _background_compile(None, torch.randn(2, 5, 16))
    assert job.float_seeds == 0 and restarts["n"] >= 1


def test_signature_ignores_nested_code_addresses():
    """A forward holding a lambda must sign the same in every process (no ``<code object ... at 0x..>``)."""
    src = "def forward(self, x):\n    return (lambda y: y + 1)(x)\n"

    def code():
        scope: dict = {}
        exec(compile(src, "blocks.py", "exec"), scope)
        return scope["forward"].__code__

    first, second = code(), code()
    assert first is not second
    assert repr(first.co_consts) != repr(second.co_consts)  # the nested lambda prints its address
    assert bg._code_digest(first) == bg._code_digest(second)
    other = compile(src.replace("+ 1", "+ 2"), "blocks.py", "exec").co_consts[0]
    assert bg._code_digest(other) != bg._code_digest(first)
