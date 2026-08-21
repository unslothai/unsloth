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

"""Tests for ROCm AOTriton environment setup."""

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
_UNINSTALL_PS1 = _REPO_ROOT / "scripts" / "uninstall.ps1"
_UNINSTALL_SH = _REPO_ROOT / "scripts" / "uninstall.sh"

_DROPIN_MARKER = "# >>> Unsloth ROCm AOTriton attention >>>"
_OWNER_MARKER = "AotritonUserEnvOwned"


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
    import os

    monkeypatch.delenv(AOTRITON_ENV, raising = False)
    assert enable_rocm_aotriton_attention() is True
    assert os.environ[AOTRITON_ENV] == "1"


def test_init_opens_the_gate_before_importing_torch():
    """Keep the gate before imports that can load torch."""
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
    source = _INIT.read_text(encoding = "utf-8")
    assert "from ._rocm_attention import enable_rocm_aotriton_attention" in source


def test_install_ps1_persists_the_gate_for_pinned_rocm_indexes_too():
    """Persist after both automatic and pinned ROCm routing."""
    source = _INSTALL_PS1.read_text(encoding = "utf-8")
    persist = source.index('[Environment]::SetEnvironmentVariable($aotritonVar, "1", "User")')
    pinned_route = source.index(
        "if ($TorchIndexPinned -and -not $ROCmIndexUrl -and -not $SkipTorch) {"
    )
    assert pinned_route < persist, "the AOTriton persistence must follow the pinned-index routing"
    # Only persist when a ROCm index won.
    guard = source.rindex("if ($ROCmIndexUrl) {", 0, persist)
    assert guard > pinned_route


def test_install_ps1_sets_the_process_copy_before_the_user_scope_write():
    """Set process scope before a User-scope write that may fail."""
    source = _INSTALL_PS1.read_text(encoding = "utf-8")
    process_copy = source.index('Set-Item -Path "Env:$aotritonVar" -Value "1"')
    user_write = source.index('[Environment]::SetEnvironmentVariable($aotritonVar, "1", "User")')
    assert process_copy < user_write, "the process copy must precede the User-scope write"
    # Keep the process write outside the fallible block.
    assert "try {" in source[process_copy:user_write]


def _extract_sh_function(source, name):
    """Extract a shell function whose closing brace is in column zero."""
    lines = source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(name + "() {"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def _run_persist_helper(
    tmp_path,
    skip_torch,
    env = None,
):
    """Run the helper with its profile directory redirected to tmp_path."""
    body = _extract_sh_function(
        _INSTALL_SH.read_text(encoding = "utf-8"), "_persist_rocm_aotriton_env"
    )
    profile_d = tmp_path / "profile.d"
    profile_d.mkdir()
    body = body.replace("/etc/profile.d", str(profile_d))
    script = tmp_path / "harness.sh"
    script.write_text(
        # Take the root branch so the result does not depend on passwordless sudo.
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
    """Do not change the host when torch is not installed."""
    value, wrote = _run_persist_helper(tmp_path, "true")
    assert value == "<unset>", "--no-torch must not export the gate"
    assert not wrote, "--no-torch must not write the host-wide drop-in"


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer helper",
)
def test_install_sh_persists_when_torch_is_being_installed(tmp_path):
    value, wrote = _run_persist_helper(tmp_path, "false")
    assert value == "1"
    assert wrote


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer helper",
)
def test_install_sh_leaves_an_existing_opinion_alone(tmp_path):
    value, wrote = _run_persist_helper(
        tmp_path, "false", env = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"}
    )
    assert value == "0"
    assert not wrote


def test_install_sh_persists_the_gate_off_the_resolved_index_leaf():
    """Persist after the final torch index is resolved."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    classify = source.index('if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then')
    call = source.index("_persist_rocm_aotriton_env ||", classify)
    assert call - classify < 400


def _run_remove_helper(tmp_path, dropin_contents = None):
    """Run the uninstaller's removal helper against a redirected profile directory."""
    body = _extract_sh_function(
        _UNINSTALL_SH.read_text(encoding = "utf-8"), "_remove_rocm_aotriton_dropin"
    )
    profile_d = tmp_path / "profile.d"
    profile_d.mkdir()
    dropin = profile_d / "unsloth-rocm-aotriton.sh"
    if dropin_contents is not None:
        dropin.write_text(dropin_contents, encoding = "utf-8")
    script = tmp_path / "harness.sh"
    script.write_text(
        # Take the root branch so the result does not depend on passwordless sudo.
        "id() { echo 0; }\n" + body + "\n_remove_rocm_aotriton_dropin\n",
        encoding = "utf-8",
    )
    run_env = dict(os.environ)
    # Use install.sh's supported profile directory override.
    run_env["UNSLOTH_PROFILE_D"] = str(profile_d)
    proc = subprocess.run(
        ["sh", str(script)], capture_output = True, text = True, env = run_env, timeout = 60
    )
    assert proc.returncode == 0, proc.stderr
    return dropin.exists()


def _installer_dropin_text():
    """The drop-in exactly as _persist_rocm_aotriton_env writes it."""
    return (
        f"{_DROPIN_MARKER}\n"
        "# Unset or set to 0 to fall back to torch's MATH SDPA kernel.\n"
        f"export {AOTRITON_ENV}=1\n"
        "# <<< Unsloth ROCm AOTriton attention <<<\n"
    )


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_removes_the_installer_created_dropin(tmp_path):
    """The host-wide gate must not outlive the install that wrote it."""
    assert _run_remove_helper(tmp_path, _installer_dropin_text()) is False


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_leaves_a_file_it_did_not_write(tmp_path):
    """Preserve a same-named file without the installer marker."""
    foreign = f"export {AOTRITON_ENV}=1\n"
    assert _run_remove_helper(tmp_path, foreign) is True


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_is_a_noop_without_the_dropin(tmp_path):
    """A CUDA or CPU host never had one; the uninstall must not care."""
    assert _run_remove_helper(tmp_path, None) is False


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_keeps_a_drop_in_edited_to_the_opt_out(tmp_path):
    """Deleting a hand-set 0 would reopen the gate on the next `import unsloth`."""
    edited = _installer_dropin_text().replace(f"{AOTRITON_ENV}=1", f"{AOTRITON_ENV}=0")
    assert _run_remove_helper(tmp_path, edited) is True


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_keeps_a_drop_in_with_an_appended_opt_out(tmp_path):
    """The original line survives an append, but the last assignment is what sourcing sees."""
    appended = _installer_dropin_text() + f"export {AOTRITON_ENV}=0\n"
    assert _run_remove_helper(tmp_path, appended) is True


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_keeps_a_drop_in_carrying_unrelated_user_content(tmp_path):
    """Whatever else the user put in the file is not ours to throw out."""
    with_extra = _installer_dropin_text() + 'export MY_OWN_THING="keep me"\n'
    assert _run_remove_helper(tmp_path, with_extra) is True


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the uninstaller helper",
)
def test_uninstall_sh_keeps_a_drop_in_with_a_prepended_line(tmp_path):
    """An override placed above ours still means the file is not what we wrote."""
    prepended = f"export {AOTRITON_ENV}=0\n" + _installer_dropin_text()
    assert _run_remove_helper(tmp_path, prepended) is True


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason = "needs a POSIX shell to execute the installer and uninstaller helpers",
)
def test_the_drop_in_directory_is_redirectable_end_to_end(tmp_path):
    """Write and removal must agree on where the drop-in lives."""
    profile_d = tmp_path / "elsewhere"
    profile_d.mkdir()
    write_tmp = tmp_path / "write"
    write_tmp.mkdir()
    remove_tmp = tmp_path / "remove"
    remove_tmp.mkdir()
    written, _ = _run_persist_helper(
        write_tmp, "false", env = {"UNSLOTH_PROFILE_D": str(profile_d)}
    )
    dropin = profile_d / "unsloth-rocm-aotriton.sh"
    assert written == "1"
    assert dropin.exists(), "install.sh must honour UNSLOTH_PROFILE_D"
    written_text = dropin.read_text(encoding = "utf-8")
    assert _DROPIN_MARKER in written_text
    # Keep the fixture aligned with install.sh's output.
    assert written_text == _installer_dropin_text()
    assert _run_remove_helper(remove_tmp, written_text) is False


def test_uninstall_sh_removes_the_dropin_outside_the_wsl_branch():
    """install.sh writes it on native Linux, where the WSL branch never runs."""
    source = _UNINSTALL_SH.read_text(encoding = "utf-8")
    linux_case = source.index("\n        Linux)\n")
    call = source.index("_remove_rocm_aotriton_dropin\n", linux_case)
    wsl_branch = source.index('if [ "$_is_wsl" = "1" ]; then', linux_case)
    assert call < wsl_branch, "the drop-in removal must not be gated on WSL"


def test_installer_and_uninstaller_agree_on_the_dropin_marker():
    """Installer and uninstaller must use the same marker."""
    assert _DROPIN_MARKER in _INSTALL_SH.read_text(encoding = "utf-8")
    assert _DROPIN_MARKER in _UNINSTALL_SH.read_text(encoding = "utf-8")


def test_install_ps1_marks_the_user_value_as_installer_created():
    """Write the ownership marker before the value it describes."""
    source = _INSTALL_PS1.read_text(encoding = "utf-8")
    marker = source.index(f"-Name $aotritonOwnerName")
    user_write = source.index('[Environment]::SetEnvironmentVariable($aotritonVar, "1", "User")')
    assert marker < user_write, "an unmarked value would never be cleaned up"
    # A failed write must not leave a marker claiming a value the user owns.
    rollback = source.index("Remove-ItemProperty", user_write)
    assert _OWNER_MARKER in source
    assert rollback - user_write < 400


def test_uninstall_ps1_clears_only_the_installer_created_value():
    """Marked and unmodified, or it stays."""
    source = _UNINSTALL_PS1.read_text(encoding = "utf-8")
    clear = source.index(
        f'[Environment]::SetEnvironmentVariable($aotritonVar, [NullString]::Value, "User")'
    )
    guard = source.rindex(f"-Name '{_OWNER_MARKER}'", 0, clear)
    assert f'-eq "1"' in source[guard:clear], "a user's later edit must survive"
    # The marker lives under HKCU\Software\Unsloth, which the uninstaller then deletes.
    key_removal = source.index("Remove-Item -LiteralPath 'HKCU:\\Software\\Unsloth'")
    assert clear < key_removal
