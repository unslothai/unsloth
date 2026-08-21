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

"""The AOTriton gate: opened by default, never argued with when already set, and opened
before anything can import torch."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from unsloth._rocm_attention import AOTRITON_ENV, enable_rocm_aotriton_attention

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INIT = _REPO_ROOT / "unsloth" / "__init__.py"
_INSTALL_PS1 = _REPO_ROOT / "install.ps1"
_INSTALL_SH = _REPO_ROOT / "install.sh"


def test_opens_the_gate_when_unset():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert env[AOTRITON_ENV] == "1"


def test_zero_is_the_opt_out_and_survives():
    """ "0" is a deliberate choice by someone who hit an AOTriton bug, not an empty slot."""
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
    import os

    monkeypatch.delenv(AOTRITON_ENV, raising = False)
    assert enable_rocm_aotriton_attention() is True
    assert os.environ[AOTRITON_ENV] == "1"


def test_init_opens_the_gate_before_importing_torch():
    """Torch reads the variable when it picks an SDPA backend, not at extension load, so a
    later set would still work today. Pin the order anyway: the read point is a torch
    implementation detail, and a build that moved it earlier would silently cost a user
    their context length. See unsloth/_rocm_attention.py."""
    source = _INIT.read_text(encoding = "utf-8")
    gate = source.index("_enable_rocm_aotriton_attention()")
    before = source[:gate]
    # No torch import of any spelling may precede the gate.
    assert re.search(r"^\s*(import torch|from torch)\b", before, re.MULTILINE) is None
    # Nor an unsloth submodule that pulls torch on the way in.
    for module in ("_gpu_init", "import_fixes", "models", "kernels"):
        assert f"from .{module}" not in before
        assert f"from unsloth.{module}" not in before


def test_init_imports_the_helper_from_the_package():
    """Guards the import line itself: a refactor that drops it would silently reinstate
    the O(N^2) path with every test above still green."""
    source = _INIT.read_text(encoding = "utf-8")
    assert "from ._rocm_attention import enable_rocm_aotriton_attention" in source


def test_install_ps1_persists_the_gate_for_pinned_rocm_indexes_too():
    """UNSLOTH_TORCH_INDEX_URL/_FAMILY pinned to a gfx*/rocm* index skips the auto-reroute
    block entirely and resolves $ROCmIndexUrl in the pinned branch below it. Persisting from
    inside the auto-reroute would leave those installs on the MATH path."""
    source = _INSTALL_PS1.read_text(encoding = "utf-8")
    persist = source.index('[Environment]::SetEnvironmentVariable($aotritonVar, "1", "User")')
    pinned_route = source.index(
        "if ($TorchIndexPinned -and -not $ROCmIndexUrl -and -not $SkipTorch) {"
    )
    assert pinned_route < persist, "the AOTriton persistence must follow the pinned-index routing"
    # And it only fires once a ROCm index actually won, not on every install.
    guard = source.rindex("if ($ROCmIndexUrl) {", 0, persist)
    assert guard > pinned_route


def test_install_ps1_sets_the_process_copy_before_the_user_scope_write():
    """SetEnvironmentVariable(..., "User") throws on a policy-blocked or ACL-locked
    HKCU\\Environment, and the catch tells the user the process copy carried this run. Order
    it the other way and that is a lie: the installer and every child it spawns lose the gate
    on exactly the machines where the write fails."""
    source = _INSTALL_PS1.read_text(encoding = "utf-8")
    process_copy = source.index('Set-Item -Path "Env:$aotritonVar" -Value "1"')
    user_write = source.index('[Environment]::SetEnvironmentVariable($aotritonVar, "1", "User")')
    assert process_copy < user_write, "the process copy must precede the User-scope write"
    # And the try must open between the two, so the process copy is outside it: inside, a
    # throw from an earlier statement could still skip it.
    assert "try {" in source[process_copy:user_write]


def _extract_sh_function(source, name):
    """The function body verbatim, from its opening line to the closing brace in column 0."""
    lines = source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(name + "() {"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _run_persist_helper(
    tmp_path,
    skip_torch,
    env = None,
):
    """Run _persist_rocm_aotriton_env for real, with its drop-in target redirected into
    tmp_path so the test never touches /etc. Returns (exported_value, dropin_written)."""
    body = _extract_sh_function(
        _INSTALL_SH.read_text(encoding = "utf-8"), "_persist_rocm_aotriton_env"
    )
    profile_d = tmp_path / "profile.d"
    profile_d.mkdir()
    body = body.replace("/etc/profile.d", str(profile_d))
    script = tmp_path / "harness.sh"
    script.write_text(
        # Shadow id(1) so the helper takes its root branch and writes the redirected path
        # directly. Otherwise the result depends on whether the test host has passwordless
        # sudo, and the sudo-tee arm swallows its own failure by design.
        "id() { echo 0; }\n" + body + "\n"
        "_persist_rocm_aotriton_env\n"
        'printf "%s" "${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL-<unset>}"\n',
        encoding = "utf-8",
    )
    run_env = dict(os.environ)
    run_env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
    run_env.update(env or {})
    run_env["SKIP_TORCH"] = skip_torch
    proc = subprocess.run(
        ["sh", str(script)], capture_output = True, text = True, env = run_env, timeout = 60
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, (profile_d / "unsloth-rocm-aotriton.sh").exists()


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer helper",
)
def test_install_sh_skips_persistence_under_no_torch(tmp_path):
    """--no-torch installs no ROCm torch, so writing a root-owned /etc/profile.d file that
    re-points every future ROCm process on the host is out of scope. A pinned or probed
    gfx*/rocm* index still classifies as the ROCm family under --no-torch, so the leaf test
    alone does not cover this; install.ps1 is guarded already, via a $ROCmIndexUrl that only
    resolves when -SkipTorch is off."""
    value, wrote = _run_persist_helper(tmp_path, "true")
    assert value == "<unset>", "--no-torch must not export the gate"
    assert not wrote, "--no-torch must not write the host-wide drop-in"


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer helper",
)
def test_install_sh_persists_when_torch_is_being_installed(tmp_path):
    """The other half: with torch going in, the helper still does its job."""
    value, wrote = _run_persist_helper(tmp_path, "false")
    assert value == "1"
    assert wrote


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer helper",
)
def test_install_sh_leaves_an_existing_opinion_alone(tmp_path):
    """ "0" is somebody who hit an AOTriton bug, not an empty slot."""
    value, wrote = _run_persist_helper(
        tmp_path, "false", env = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"}
    )
    assert value == "0"
    assert not wrote


def test_install_sh_persists_the_gate_off_the_resolved_index_leaf():
    """The shell side classifies the FINAL resolved TORCH_INDEX_URL, so a pinned index is
    already covered; keep the call there rather than in any one reroute arm."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    classify = source.index('if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then')
    call = source.index("_persist_rocm_aotriton_env ||", classify)
    assert call - classify < 400
