# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""unsloth/_torchao_nodist.py on a simulated torch without torch.distributed (AMD's Windows ROCm
wheels): is_available() False, every torch.distributed.* submodule unimportable, and the c10d ops
absent, as in pytorch/ao#4761's test plus the missing ops. Out of process so the parent's module
cache is untouched. Verified on real hardware separately (Windows 11, gfx1151, torch 2.11 ROCm)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SHIM = Path(__file__).resolve().parents[1] / "unsloth" / "_torchao_nodist.py"

_SIMULATE = r"""
import importlib.util, json, sys
import torch

torch.distributed.is_available = lambda: False
for name in [n for n in sys.modules if n.startswith("torch.distributed.")]:
    del sys.modules[name]
for attr, value in list(vars(torch.distributed).items()):
    if isinstance(value, type(sys)) and value.__name__.startswith("torch.distributed."):
        delattr(torch.distributed, attr)


class BlockDistributed:
    def find_spec(self, fullname, path = None, target = None):
        if fullname.startswith("torch.distributed."):
            raise ModuleNotFoundError(f"No module named 'torch._C._distributed_c10d' ({fullname})", name = fullname)
        return None


sys.meta_path.insert(0, BlockDistributed())
_HIDDEN = {"c10d_functional", "_c10d_functional", "c10d", "_dtensor"}
_ns = torch._ops._OpNamespace
_real = _ns.__getattr__


def _hiding(self, op):
    if self.name in _HIDDEN and not op.startswith("__"):
        raise AttributeError(f"'_OpNamespace' '{self.name}' object has no attribute '{op}'")
    return _real(self, op)


_ns.__getattr__ = _hiding
for ns in _HIDDEN:
    obj = torch.ops.__dict__.get(ns)
    for attr in list(vars(obj)) if obj is not None else []:
        if not attr.startswith("__"):
            try:
                delattr(obj, attr)
            except Exception:
                pass

r = {}
if sys.argv[2] == "fix":
    spec = importlib.util.spec_from_file_location("_nodist", sys.argv[1])
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    r["fixed"] = shim.fix_torchao_without_torch_distributed()
for stmt in ("import torchao", "from torchao.quantization import Int8WeightOnlyConfig, Float8WeightOnlyConfig",
             "import transformers.quantizers.quantizer_torchao", "import peft.tuners.lora.torchao",
             "import torch.distributed.tensor"):
    try:
        exec(stmt)
        r[stmt] = "ok"
    except BaseException as e:
        r[stmt] = f"{type(e).__name__}: {e}"[:200]
held = []
for name, module in list(sys.modules.items()):
    if module is None or name == "torchao" or name.startswith(("torchao.", "_nodist")):
        continue
    try:
        values = list(vars(module).values())
    except Exception:
        continue
    for value in values:
        try:
            if isinstance(value, type(sys)) and value.__dict__.get("__unsloth_nodist_stub__"):
                held.append(name)
        except Exception:
            pass
r["stand_ins_held_outside_torchao"] = sorted(set(held))
print("RESULT=" + json.dumps(r))
"""

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None or importlib.util.find_spec("transformers") is None,
    reason = "needs torchao and transformers installed",
)


def _run(mode):
    out = subprocess.run(
        [sys.executable, "-c", _SIMULATE, str(_SHIM), mode],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT=")]
    assert lines, out.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT=") :])


def test_torchao_fails_without_the_fix():
    r = _run("none")
    assert r["import torchao"] != "ok"


def test_torchao_and_its_consumers_import_with_the_fix():
    r = _run("fix")
    assert r["fixed"] is True
    assert r["import torchao"] == "ok"
    assert (
        r["from torchao.quantization import Int8WeightOnlyConfig, Float8WeightOnlyConfig"] == "ok"
    )
    # transformers imports torchao.prototype.* lazily, long after `import torchao` returned.
    assert r["import transformers.quantizers.quantizer_torchao"] == "ok"
    assert r["import peft.tuners.lora.torchao"] == "ok"


def test_nothing_outside_torchao_sees_a_fake_torch_distributed():
    r = _run("fix")
    assert r["import torch.distributed.tensor"].startswith("ModuleNotFoundError")
    assert r["stand_ins_held_outside_torchao"] == []


def test_noop_where_torch_distributed_exists():
    code = (
        "import importlib.util, sys, torch\n"
        "spec = importlib.util.spec_from_file_location('_nodist', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print('RESULT', torch.distributed.is_available(), m.fix_torchao_without_torch_distributed(), "
        "any(type(f).__name__ == '_TorchaoWindowFinder' for f in sys.meta_path))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(_SHIM)], capture_output = True, text = True, timeout = 600
    )
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT"))
    available, fixed, hooked = line.split()[1:]
    if available != "True":
        pytest.skip("this torch has no torch.distributed")
    assert (fixed, hooked) == ("False", "False")


def test_gpu_init_runs_the_fix_before_anything_imports_torchao():
    """The torch symbol-skew fix must come first (torchao 0.18 on torch < 2.10), and the shim
    must precede vllm's probe and unsloth_zoo, both of which import transformers."""
    src = (_SHIM.parent / "_gpu_init.py").read_text(encoding = "utf-8")
    order = [
        src.index("\nfix_torchao_torch_symbol_skew()"),
        src.index("\nfix_torchao_without_torch_distributed()"),
        src.index("\ndisable_broken_vllm()"),
        src.index("\n    import unsloth_zoo\n"),
    ]
    assert order == sorted(order)
