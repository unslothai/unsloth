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

"""Tests for process-local ROCm AOTriton setup."""

import os
import re
from pathlib import Path

from unsloth._rocm_attention import AOTRITON_ENV, enable_rocm_aotriton_attention

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INIT = _REPO_ROOT / "unsloth" / "__init__.py"
_STUDIO_RUN = _REPO_ROOT / "studio" / "backend" / "run.py"
_STUDIO_MAIN = _REPO_ROOT / "studio" / "backend" / "main.py"


def test_opens_the_gate_when_unset():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert env[AOTRITON_ENV] == "1"


def test_zero_is_the_opt_out_and_survives():
    env = {AOTRITON_ENV: "0"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "0"


def test_existing_value_is_never_rewritten():
    env = {AOTRITON_ENV: "whatever-the-user-set"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "whatever-the-user-set"


def test_idempotent():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "1"


def test_defaults_to_the_real_environment(monkeypatch):
    monkeypatch.delenv(AOTRITON_ENV, raising = False)
    assert enable_rocm_aotriton_attention() is True
    assert os.environ[AOTRITON_ENV] == "1"


def test_init_opens_the_gate_before_attention_imports():
    source = _INIT.read_text(encoding = "utf-8")
    gate = source.index("_enable_rocm_aotriton_attention()")
    before = source[:gate]
    assert re.search(r"^\s*(import torch|from torch)\b", before, re.MULTILINE) is None
    for module in ("_gpu_init", "import_fixes", "models", "kernels"):
        assert f"from .{module}" not in before
        assert f"from unsloth.{module}" not in before


def test_init_imports_the_helper_from_the_package():
    source = _INIT.read_text(encoding = "utf-8")
    assert "from ._rocm_attention import enable_rocm_aotriton_attention" in source


def test_studio_sets_the_gate_before_backend_imports():
    source = _STUDIO_RUN.read_text(encoding = "utf-8")
    gate = source.index('os.environ.setdefault(_AOTRITON_ENV, "1")')
    assert source.index("from utils.cpu_threads") > gate
    assert source.index("from core._torchao_stub") > gate
    assert re.search(r"^\s*(import torch|from torch)\b", source[:gate], re.MULTILINE) is None


def test_direct_uvicorn_sets_the_gate_before_backend_imports():
    source = _STUDIO_MAIN.read_text(encoding = "utf-8")
    gate = source.index('os.environ.setdefault(_AOTRITON_ENV, "1")')
    assert source.index("from utils.native_tls") > gate
    assert source.index("from utils.cpu_threads") > gate
    assert re.search(r"^\s*(import torch|from torch)\b", source[:gate], re.MULTILINE) is None


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
