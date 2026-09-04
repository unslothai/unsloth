# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one index URL the manifest is allowed to carry.

Windows on ARM is the only platform whose CUDA torch wheels live nowhere on
download.pytorch.org, so a later `unsloth studio update` -- a fresh shell, with
install.ps1's handover variable long gone -- cannot re-derive the index from the
driver the way every other host can. write_manifest records it.

That is a deliberate exception to the rule the module documents for itself: the
FLAVOR, never the index URL it came from, because a pinned index can carry a
token in its userinfo, query or fragment, and this file is read back by
verify-install and desktop-capabilities. The exception only holds while the
guard does, so both halves are tested here -- the Python that writes it and the
PowerShell that reads it back, each of which must refuse independently. A
hand-edited manifest must not be able to redirect a torch install to an
arbitrary host, and a mirror the user pinned must not be copied into a file
other tooling prints.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess

import pytest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST_PY = PACKAGE_ROOT / "studio" / "install_manifest.py"
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("studio_install_manifest_woa", MANIFEST_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


im = _load_manifest_module()


# (url, may_be_persisted, why)
CANDIDATES = (
    ("https://pypi.nvidia.com/nvtorch_oot", True, "the GA channel install.ps1 probes"),
    ("https://pypi.nvidia.com/nvtorch_oot_nightly/", True, "the nightly channel, trailing slash"),
    ("https://user:token@pypi.nvidia.com/nvtorch_oot", False, "userinfo is exactly the leak"),
    ("https://pypi.nvidia.com/nvtorch_oot?token=abc", False, "a token in the query"),
    ("https://pypi.nvidia.com/nvtorch_oot#token=abc", False, "a token in the fragment"),
    ("https://mirror.corp.example/whl", False, "a mirror the user pinned"),
    ("http://pypi.nvidia.com/nvtorch_oot", False, "plaintext, so not our channel"),
    ("https://pypi.nvidia.com.evil.example/whl", False, "a host that merely starts the same"),
    ("https://evilpypi.nvidia.com/whl", False, "a host that merely ends the same"),
    ("", False, "empty"),
    (None, False, "absent"),
)


class TestWriteSide:
    """studio/install_manifest.py: what is allowed into the file at all."""

    @pytest.mark.parametrize("url, allowed, why", CANDIDATES)
    def test_only_a_credential_free_nvidia_channel_is_recorded(
        self, tmp_path: pathlib.Path, url, allowed: bool, why: str
    ):
        path = im.write_manifest(root = tmp_path, req_root = tmp_path, woa_torch_index = url)
        assert path is not None
        payload = json.loads(pathlib.Path(path).read_text(encoding = "utf-8"))
        recorded = "woa_torch_index" in payload
        assert recorded is allowed, (
            f"{url!r} ({why}) was {'dropped' if allowed else 'persisted'}; "
            "the manifest is printed back by verify-install and read by setup.ps1"
        )
        if allowed:
            assert payload["woa_torch_index"] == str(url).strip().rstrip("/")

    def test_the_key_is_absent_when_no_index_was_chosen(self, tmp_path: pathlib.Path):
        """Every other host, and every WoA host that stayed on the x64 stack."""
        im.write_manifest(root = tmp_path, req_root = tmp_path)
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert "woa_torch_index" not in payload

    def test_the_addition_is_backwards_compatible(self, tmp_path: pathlib.Path):
        """An additive optional key: older readers see the schema they already parse."""
        im.write_manifest(
            root = tmp_path,
            req_root = tmp_path,
            woa_torch_index = "https://pypi.nvidia.com/nvtorch_oot",
        )
        payload = json.loads((tmp_path / im.MANIFEST_NAME).read_text(encoding = "utf-8"))
        assert payload["schema"] == 1, "the key is additive; bumping the schema is not"
        state = im.verify_install(root = tmp_path, req_root = tmp_path)
        assert state["manifest_ok"] is True, state["reason"]

    def test_the_installer_passes_the_handover_variable_through(self):
        """install.ps1 exports it; nothing else supplies this value."""
        source = STACK_PY.read_text(encoding = "utf-8")
        assert re.search(
            r"woa_torch_index\s*=\s*os\.environ\.get\(\s*[\"']UNSLOTH_WOA_SELECTED_TORCH_INDEX[\"']",
            source,
        ), "install_python_stack.py no longer forwards the index install.ps1 selected"


PWSH = shutil.which("pwsh")
requires_pwsh = pytest.mark.skipif(PWSH is None, reason = "pwsh not available")


def _function_source(text: str, name: str) -> str:
    """Extract a PowerShell function by matching balanced braces."""
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    assert match, f"{name} is not defined in setup.ps1"
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


class TestReadSide:
    """studio/setup.ps1: what Get-PersistedWoaTorchIndex hands back to the resolver."""

    @requires_pwsh
    @pytest.mark.parametrize("url, allowed, why", [c for c in CANDIDATES if c[0]])
    def test_a_hand_edited_manifest_cannot_redirect_the_install(
        self, tmp_path: pathlib.Path, url: str, allowed: bool, why: str
    ):
        """
        The write guard is not enough on its own: the file sits in the user's venv and
        anything can put a line in it. Whatever is on disk, only NVIDIA's own channel
        may come back out, because the return value becomes --extra-index-url.
        """
        (tmp_path / "unsloth_install_manifest.json").write_text(
            json.dumps({"schema": 1, "woa_torch_index": url}),
            encoding = "utf-8",
        )
        got = self._invoke(tmp_path)
        assert got == (
            url.strip().rstrip("/") if allowed else ""
        ), f"{url!r} ({why}) came back as {got!r} and would be passed to uv"

    @requires_pwsh
    def test_a_missing_or_unreadable_manifest_is_empty_not_an_error(self, tmp_path: pathlib.Path):
        """Older installs have no such key, and a truncated file must not throw."""
        assert self._invoke(tmp_path) == "", "no manifest at all"
        path = tmp_path / "unsloth_install_manifest.json"
        path.write_text('{"schema": 1, "torch_flavor": "cu130"}', encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "an older manifest without the key"
        path.write_text('{"schema": 1, "woa_torch_ind', encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "a manifest truncated by a killed installer"
        path.write_text("", encoding = "utf-8")
        assert self._invoke(tmp_path) == "", "an empty manifest"

    @staticmethod
    def _invoke(venv: pathlib.Path) -> str:
        body = _function_source(SETUP_PS1.read_text(encoding = "utf-8"), "Get-PersistedWoaTorchIndex")
        script = f"{body}\nWrite-Output (Get-PersistedWoaTorchIndex -VenvPath '{venv}')"
        done = subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()
