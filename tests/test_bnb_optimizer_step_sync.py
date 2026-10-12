# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GPU cases run in a subprocess: the patch rebinds bitsandbytes' Optimizer8bit.step and sync_gpu."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXES = os.path.join(_ROOT, "unsloth", "import_fixes.py")

_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys
    import torch, bitsandbytes as bnb
    patched, optim_name, side_stream = sys.argv[1] == "patched", sys.argv[2], sys.argv[4] == "stream"
    if patched:
        spec = importlib.util.spec_from_file_location("unsloth_import_fixes_under_test", sys.argv[3])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes.patch_bitsandbytes_optimizer_step_sync()
        fixes.patch_bitsandbytes_optimizer_step_sync()  # idempotent
    syncs = [0]
    real_sync = torch.cuda.synchronize
    def counting_sync(*a, **k):
        syncs[0] += 1
        return real_sync(*a, **k)
    torch.cuda.synchronize = counting_sync
    torch.manual_seed(0)
    shapes = [(2048, 1024), (16, 2048), (1024,), (64, 2)]
    params = [torch.nn.Parameter(torch.randn(s, device = "cuda")) for s in shapes]
    opt = getattr(bnb.optim, optim_name)(params, lr = 1e-2)
    per_step = []
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    for i in range(10):
        g = torch.Generator(device = "cuda").manual_seed(i)
        for p in params:
            p.grad = torch.randn(p.shape, device = "cuda", generator = g)
        before = syncs[0]
        with torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.default_stream())
            opt.step()
        torch.cuda.default_stream().wait_stream(stream)
        per_step.append(syncs[0] - before)
    real_sync()
    out = [p.detach().float().cpu().flatten().tolist() for p in params]
    print("RESULT " + json.dumps({"params": out, "syncs": per_step, "is_paged": bool(opt.is_paged),
                                  "wrapped": getattr(type(opt).step, "_unsloth_no_sync", False)}))
    """
)


def _run(mode, optim_name, stream = "default"):
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, mode, optim_name, _FIXES, stream],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(line[len("RESULT ") :])


def _gpu_ready():
    try:
        import bitsandbytes  # noqa: F401
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu_ready(), reason = "bitsandbytes optimizers need a CUDA device")
@pytest.mark.parametrize(
    "optim_name", ["AdamW8bit", "Adam8bit", "Lion8bit", "AdamW", "PagedAdamW8bit"]
)
def test_step_matches_stock_and_drops_only_non_paged_syncs(optim_name):
    stock = _run("stock", optim_name)
    patched = _run("patched", optim_name)
    if not patched["wrapped"]:
        pytest.skip("this bitsandbytes inlines torch.cuda.synchronize; the patch leaves it alone")
    assert patched["params"] == stock["params"]  # bit-identical, every element, after 10 steps
    if stock["is_paged"]:
        assert patched["syncs"] == stock["syncs"]
    else:
        assert min(stock["syncs"]) >= 4  # the stock step syncs once per parameter
        assert patched["syncs"] == [0] * 10


@pytest.mark.skipif(not _gpu_ready(), reason = "bitsandbytes optimizers need a CUDA device")
def test_step_on_a_side_stream_keeps_every_sync():
    # bitsandbytes launches on the legacy default stream, so off it the per-parameter sync is the only ordering.
    stock = _run("stock", "AdamW8bit", "stream")
    patched = _run("patched", "AdamW8bit", "stream")
    if not patched["wrapped"]:
        pytest.skip("this bitsandbytes inlines torch.cuda.synchronize; the patch leaves it alone")
    assert patched["syncs"] == stock["syncs"] and min(stock["syncs"]) >= 4
    assert patched["params"] == stock["params"]


def _load_fixes():
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_step_sync", _FIXES)
    fixes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixes)
    return fixes


def _fake_bnb(monkeypatch, tmp_path, source):
    path = tmp_path / f"fake_optimizer_{abs(hash(source))}.py"
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location("bitsandbytes.optim.optimizer", path)
    optimizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimizer)  # from a file, so inspect.getsource sees the step body
    package, optim = types.ModuleType("bitsandbytes"), types.ModuleType("bitsandbytes.optim")
    package.optim, optim.optimizer = optim, optimizer
    for name, module in (
        ("bitsandbytes", package),
        ("bitsandbytes.optim", optim),
        ("bitsandbytes.optim.optimizer", optimizer),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return optimizer


_NEW_STEP = """
def sync_gpu(t):
    pass

class Optimizer8bit:
    def step(self, closure=None):
        for p in self.params:
            sync_gpu(p)
"""

_OLD_STEP = """
class Optimizer8bit:
    def step(self, closure=None):
        for p in self.params:
            torch.cuda.synchronize()
"""


def test_patch_is_idempotent_and_skips_a_bitsandbytes_without_sync_gpu(monkeypatch, tmp_path):
    fixes = _load_fixes()
    optimizer = _fake_bnb(monkeypatch, tmp_path, _NEW_STEP)
    stock = optimizer.Optimizer8bit.step
    fixes.patch_bitsandbytes_optimizer_step_sync()
    patched = optimizer.Optimizer8bit.step
    assert patched._unsloth_no_sync and patched.__wrapped__ is stock
    fixes.patch_bitsandbytes_optimizer_step_sync()
    assert optimizer.Optimizer8bit.step is patched

    # An older bitsandbytes (inline torch.cuda.synchronize, no sync_gpu helper) is left untouched.
    old = _fake_bnb(monkeypatch, tmp_path, _OLD_STEP)
    before = old.Optimizer8bit.step
    fixes.patch_bitsandbytes_optimizer_step_sync()
    assert old.Optimizer8bit.step is before
