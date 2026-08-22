# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for conservative process-local ROCm AOTriton setup."""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import unsloth_runtime.rocm_attention as rocm_attention
from unsloth_runtime.rocm_attention import AOTRITON_ENV, enable_rocm_aotriton_attention

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INIT = _REPO_ROOT / "unsloth" / "__init__.py"
_GPU_INIT = _REPO_ROOT / "unsloth" / "_gpu_init.py"
_STUDIO_RUN = _REPO_ROOT / "studio" / "backend" / "run.py"
_STUDIO_MAIN = _REPO_ROOT / "studio" / "backend" / "main.py"
_VALIDATED_BUILD = "2.11.0+rocm7.13.0"


class _FakeCuda:
    def __init__(
        self,
        arches = ("gfx1151",),
        *,
        available = True,
        error = None,
    ):
        self.arches = tuple(arches)
        self.available = available
        self.error = error

    def is_available(self):
        if self.error is not None:
            raise self.error
        return self.available

    def device_count(self):
        return len(self.arches)

    def get_device_properties(self, index):
        return SimpleNamespace(gcnArchName = self.arches[index])


def _torch(
    build = _VALIDATED_BUILD,
    arches = ("gfx1151",),
    **cuda_kwargs,
):
    return SimpleNamespace(__version__ = build, cuda = _FakeCuda(arches, **cuda_kwargs))


def _enable(env, torch_module, **kwargs):
    return enable_rocm_aotriton_attention(
        env,
        torch_module = torch_module,
        platform_name = kwargs.pop("platform_name", "linux"),
        dxg_present = kwargs.pop("dxg_present", False),
        **kwargs,
    )


def test_enables_the_exact_validated_native_linux_stack():
    env = {}
    assert _enable(env, _torch()) is True
    assert env[AOTRITON_ENV] == "1"


def test_studio_metadata_path_imports_torch_only_after_the_build_matches(monkeypatch):
    fake_torch = _torch(arches = ("gfx1151:sramecc+:xnack-",))
    monkeypatch.setattr(rocm_attention, "package_version", lambda _name: _VALIDATED_BUILD)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    env = {}
    assert (
        enable_rocm_aotriton_attention(
            env,
            platform_name = "linux",
            dxg_present = False,
        )
        is True
    )
    assert env[AOTRITON_ENV] == "1"


def test_zero_is_the_opt_out_and_survives_without_probing_torch():
    env = {AOTRITON_ENV: "0"}
    assert _enable(env, object()) is False
    assert env[AOTRITON_ENV] == "0"


def test_existing_value_is_never_rewritten():
    env = {AOTRITON_ENV: "operator-choice"}
    assert _enable(env, _torch()) is False
    assert env[AOTRITON_ENV] == "operator-choice"


def test_idempotent():
    env = {}
    assert _enable(env, _torch()) is True
    assert _enable(env, _torch()) is False
    assert env[AOTRITON_ENV] == "1"


@pytest.mark.parametrize(
    "build",
    (
        "2.6.0+rocm6.3",
        "2.10.0+rocm7.2",
        "2.11.0+rocm7.13.1",
        "2.12.0+rocm7.13.0",
        "2.11.0+cu128",
        "",
    ),
)
def test_unknown_torch_builds_fail_closed(build):
    env = {}
    assert _enable(env, _torch(build = build)) is False
    assert AOTRITON_ENV not in env


@pytest.mark.parametrize("arch", ("gfx1150", "gfx1200", "gfx1101", "gfx942", ""))
def test_unvalidated_or_unknown_architectures_fail_closed(arch):
    env = {}
    assert _enable(env, _torch(arches = (arch,))) is False
    assert AOTRITON_ENV not in env


def test_mixed_amd_architectures_fail_closed_because_the_gate_is_process_wide():
    env = {}
    assert _enable(env, _torch(arches = ("gfx1151", "gfx1150"))) is False
    assert AOTRITON_ENV not in env


def test_multiple_validated_gpus_are_allowed():
    env = {}
    assert _enable(env, _torch(arches = ("gfx1151", "gfx1151"))) is True
    assert env[AOTRITON_ENV] == "1"


@pytest.mark.parametrize("platform_name", ("win32", "darwin", "freebsd"))
def test_non_linux_platforms_fail_closed(platform_name):
    env = {}
    assert _enable(env, _torch(), platform_name = platform_name) is False
    assert AOTRITON_ENV not in env


def test_wsl_fails_closed():
    env = {}
    assert _enable(env, _torch(), dxg_present = True) is False
    assert AOTRITON_ENV not in env


def test_no_visible_gpu_fails_closed():
    env = {}
    assert _enable(env, _torch(arches = (), available = False)) is False
    assert AOTRITON_ENV not in env


def test_device_probe_failure_fails_closed():
    env = {}
    assert _enable(env, _torch(error = RuntimeError("driver unavailable"))) is False
    assert AOTRITON_ENV not in env


def test_unsloth_calls_the_gate_after_torch_but_before_model_imports():
    init_source = _INIT.read_text(encoding = "utf-8")
    assert "enable_rocm_aotriton_attention" not in init_source

    source = _GPU_INIT.read_text(encoding = "utf-8")
    torch_import = source.index("    import torch\n", source.index("# Try importing PyTorch"))
    gate = source.index("_enable_rocm_aotriton_attention(torch_module = torch)")
    models = source.index("from .models import *")
    attention_call = re.search(r"scaled_dot_product_attention\s*\(", source[:gate])
    assert torch_import < gate < models
    assert attention_call is None


def test_studio_sets_the_gate_before_route_and_attention_imports():
    source = _STUDIO_MAIN.read_text(encoding = "utf-8")
    gate = source.index("_enable_aotriton()")
    assert source.index("from routes import (") > gate
    assert source.index("from utils.hardware import (") > gate
    assert re.search(r"scaled_dot_product_attention\s*\(", source[:gate]) is None


def test_all_studio_launches_converge_on_main_gate():
    source = _STUDIO_RUN.read_text(encoding = "utf-8")
    assert "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in source
    assert "from main import app" in source


def test_installers_do_not_persist_the_gate():
    paths = (
        _REPO_ROOT / "install.sh",
        _REPO_ROOT / "install.ps1",
        _REPO_ROOT / "scripts" / "uninstall.sh",
        _REPO_ROOT / "scripts" / "uninstall.ps1",
    )
    sources = {path.name: path.read_text(encoding = "utf-8") for path in paths}
    assert "_persist_rocm_aotriton_env" not in sources["install.sh"]
    assert "unsloth-rocm-aotriton.sh" not in sources["install.sh"]
    assert "unsloth-rocm-aotriton.sh" not in sources["uninstall.sh"]
    assert "AotritonUserEnvOwned" not in sources["install.ps1"]
    assert "AotritonUserEnvOwned" not in sources["uninstall.ps1"]
