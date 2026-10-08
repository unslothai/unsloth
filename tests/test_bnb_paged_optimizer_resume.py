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

"""Each case runs in a subprocess: the patch rebinds bitsandbytes' Optimizer8bit.load_state_dict."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("bitsandbytes")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("paged bitsandbytes optimizers need a CUDA device", allow_module_level = True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_RUNNER = textwrap.dedent(
    """
    import importlib.util, io, json, sys
    import torch, bitsandbytes as bnb
    mode, optim_name = sys.argv[1], sys.argv[2]
    if mode == "fixed":
        spec = importlib.util.spec_from_file_location("unsloth_import_fixes_under_test", sys.argv[3])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes.patch_bitsandbytes_paged_optimizer_resume()
        fixes.patch_bitsandbytes_paged_optimizer_resume()  # idempotent
    Optim = getattr(bnb.optim, optim_name)
    torch.manual_seed(0)
    shapes = [(512, 256), (64, 2)]  # one paged (>= 1e5 elements), one too small to page
    params = [torch.nn.Parameter(torch.randn(s, device = "cuda")) for s in shapes]
    grads = [[torch.randn(s, device = "cuda") for s in shapes] for _ in range(4)]

    def step(opt, ps, g):
        for p, x in zip(ps, g):
            p.grad = x.clone()
        opt.step()

    opt = Optim(params, lr = 1e-2)
    step(opt, params, grads[0]); step(opt, params, grads[1])
    buf = io.BytesIO(); torch.save(opt.state_dict(), buf); buf.seek(0)
    saved = torch.load(buf, map_location = "cpu", weights_only = True)  # Trainer resume path

    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = Optim(resumed_params, lr = 1e-2)
    resumed.load_state_dict(saved)
    after_load = {k: bool(getattr(resumed.state[resumed_params[0]][k], "is_paged", False)) for k in ("state1", "state2") if k in resumed.state[resumed_params[0]]}
    for g in grads[2:]:
        step(opt, params, g); step(resumed, resumed_params, g)

    def paged(o, p):
        return {k: bool(getattr(o.state[p][k], "is_paged", False)) for k in ("state1", "state2") if k in o.state[p]}

    print(json.dumps({
        "fresh_paged": paged(opt, params[0]),
        "after_load_paged": after_load,
        "resumed_paged": paged(resumed, resumed_params[0]),
        "resumed_small_paged": paged(resumed, resumed_params[1]),
        "max_abs_diff": max((a - b).abs().max().item() for a, b in zip(params, resumed_params)),
    }))
    """
)


def _run(mode, optim_name):
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            mode,
            optim_name,
            os.path.join(_ROOT, "unsloth", "import_fixes.py"),
        ],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("optim_name", ["PagedAdamW8bit", "PagedAdamW32bit", "PagedLion8bit"])
def test_resumed_paged_optimizer_state_stays_paged(optim_name):
    base = _run("base", optim_name)
    fixed = _run("fixed", optim_name)
    assert all(base["fresh_paged"].values())
    # Unpatched bitsandbytes moves resumed state into plain CUDA memory: the bug this guards.
    assert not any(base["resumed_paged"].values())
    # Paged from the load on, so the first resumed forward and backward never hold it in the
    # CUDA allocator (a fresh run has no state there yet).
    assert fixed["after_load_paged"] == fixed["resumed_paged"] == fixed["fresh_paged"]
    assert not any(fixed["resumed_small_paged"].values())
    # Resuming continues the same trajectory as never stopping, with or without the patch.
    assert fixed["max_abs_diff"] == base["max_abs_diff"] == 0.0


def test_non_paged_optimizer_untouched():
    base = _run("base", "AdamW8bit")
    fixed = _run("fixed", "AdamW8bit")
    assert base == fixed
    assert not any(fixed["resumed_paged"].values())
