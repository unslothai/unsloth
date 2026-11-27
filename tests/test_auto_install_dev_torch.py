# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

"""Nightly torch builds carry .devNNNNNNNN in torch.__version__; _auto_install must not crash on them."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUTO_INSTALL = REPO / "unsloth" / "_auto_install.py"


def _run_auto_install(torch_version: str, cuda: str, capsys) -> str:
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.__version__ = torch_version
    fake.version = types.SimpleNamespace(cuda = cuda)
    fake.cuda = types.SimpleNamespace(get_device_capability = lambda: (9, 0))
    fake._C = types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI = True)
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        exec(compile(AUTO_INSTALL.read_text(encoding = "utf-8"), str(AUTO_INSTALL), "exec"), {})
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    return capsys.readouterr().out


@pytest.mark.parametrize(
    "torch_version, cuda, series",
    [
        ("2.8.0.dev20250701+cu126", "12.6", "torch280"),
        ("2.10.0.dev20260101+cu126", "12.6", "torch2100"),
        ("2.11.0.dev20260203+cu128", "12.8", "torch2110"),
        ("2.12.0.dev20260401+cu130", "13.0", "torch2120"),
    ],
)
def test_auto_install_maps_dev_torch_to_stable_extra(
    torch_version: str, cuda: str, series: str, capsys
):
    # pip wheels for nightly torch end in .devNNNNNNNN (+cuXXX); they take their stable pin's
    # extra, exactly like the alpha builds the matcher already accepts.
    out = _run_auto_install(torch_version, cuda, capsys)
    tag = cuda.replace(".", "")
    assert f'"unsloth[cu{tag}-{series}] @' in out


@pytest.mark.parametrize(
    "torch_version",
    ["2.13.0.dev20260201+cu130", "2.14.0.dev20260501+cu126"],
)
def test_auto_install_refuses_dev_of_stable_pinned_series(torch_version: str, capsys):
    # The 2.13 / 2.14 extras pin the stable release, so a nightly of them gets the same
    # pre-release refusal an rc or +git build does instead of being silently replaced.
    with pytest.raises(RuntimeError, match = "pre-release or custom build"):
        _run_auto_install(torch_version, "13.0", capsys)
