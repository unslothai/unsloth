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

"""TRL >= 1.11 imports vllm.distributed.weight_transfer.nccl_engine whenever vLLM is importable; vLLM < 0.16 has
no such module. Runs against a fake vllm package in a subprocess, so no vLLM install is needed."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXES = os.path.join(_ROOT, "unsloth", "import_fixes.py")

_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys
    fixes_path, apply = sys.argv[1], sys.argv[2] == "1"
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_under_test", fixes_path)
    fixes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixes)
    if apply:
        fixes.fix_vllm_weight_transfer_module()
        fixes.fix_vllm_weight_transfer_module()  # idempotent
    out = {"finders": sum(type(f).__name__ == "_VllmWeightTransferStubFinder" for f in sys.meta_path)}
    try:
        # the exact import TRL 1.11 - 1.18 runs on vLLM < 0.28
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
        out["import"] = "ok"
        out["real"] = NCCLWeightTransferEngine.__dict__.get("REAL", False)
        for call in (lambda: NCCLWeightTransferEngine.trainer_init({}), lambda: NCCLTrainerSendWeightsArgs(group = None)):
            try:
                call()
                out.setdefault("use", []).append("ran")
            except ImportError as e:
                out.setdefault("use", []).append("vLLM >= 0.16" in str(e))
    except ImportError as e:
        out["import"] = type(e).__name__
    print("RESULT " + json.dumps(out))
    """
)


def _fake_vllm(root, with_weight_transfer):
    pkg = root / "vllm"
    (pkg / "distributed").mkdir(parents = True)
    (pkg / "__init__.py").write_text("__version__ = '0.11.2'\n")
    (pkg / "distributed" / "__init__.py").write_text("")
    if with_weight_transfer:
        wt = pkg / "distributed" / "weight_transfer"
        wt.mkdir()
        (wt / "__init__.py").write_text("")
        (wt / "nccl_engine.py").write_text(
            "class NCCLWeightTransferEngine:\n    REAL = True\n    @staticmethod\n"
            "    def trainer_init(info):\n        return 'real'\n"
            "class NCCLTrainerSendWeightsArgs:\n    def __init__(self, **kw):\n        pass\n"
        )


def _run(tmp_path, apply, with_weight_transfer):
    _fake_vllm(tmp_path, with_weight_transfer)
    env = dict(os.environ, PYTHONPATH = str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, _FIXES, "1" if apply else "0"],
        capture_output = True,
        text = True,
        env = env,
        cwd = str(tmp_path),
        timeout = 600,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(line[-1][len("RESULT ") :])


def test_old_vllm_without_fix_fails_trl_import(tmp_path):
    # The defect the fix covers: what `from trl import GRPOTrainer` hits on vLLM 0.11.2 - 0.15.x.
    assert _run(tmp_path, apply = False, with_weight_transfer = False)["import"] in (
        "ModuleNotFoundError",
        "ImportError",
    )


def test_old_vllm_with_fix_imports_and_raises_on_use(tmp_path):
    out = _run(tmp_path, apply = True, with_weight_transfer = False)
    assert out["import"] == "ok" and out["real"] is False
    assert out["use"] == [True, True]
    assert out["finders"] == 1


def test_new_vllm_real_module_wins(tmp_path):
    out = _run(tmp_path, apply = True, with_weight_transfer = True)
    assert out["import"] == "ok" and out["real"] is True
    assert out["use"] == ["ran", "ran"]
