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

"""The runtime's idea of a complete sidecar matches setup's.

setup.sh and setup.ps1 install tiktoken best-effort and record a sidecar without it as
complete; the runtime validator used to require it, delete the sidecar and retry the
install that had just failed.
"""



import sys
import types as _types
from pathlib import Path

# The backend uses "from utils..." imports; ensure the backend dir is on sys.path.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the custom logger before importing the module under test, as the sibling tests do.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.transformers_version as tv


def _sidecar(root, *names, versions = None):
    versions = versions or {}
    for name in names:
        (root / name).mkdir(parents = True)
        version = versions.get(name)
        if version:
            info = root / f"{name}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding = "utf-8")
    (root / tv._STUDIO_OWNED_MARKER).touch()


def test_a_sidecar_without_tiktoken_is_still_valid(tmp_path):
    root = tmp_path / ".venv_t5_550"
    _sidecar(
        root,
        "transformers",
        "huggingface_hub",
        "hf_xet",
        versions = {
            "transformers": tv.TRANSFORMERS_550_VERSION,
            "huggingface_hub": "1.8.0",
            "hf_xet": "1.4.2",
        },
    )
    assert tv._venv_dir_is_valid(str(root), tv._VENV_T5_550_PACKAGES) is True
    assert tv._venv_dir_is_valid(str(root), tv._venv_t5_latest_packages(tv.TRANSFORMERS_550_VERSION)) is True


def test_a_sidecar_without_a_required_package_is_still_invalid(tmp_path):
    root = tmp_path / ".venv_t5_550"
    _sidecar(
        root,
        "transformers",
        "huggingface_hub",
        "tiktoken",
        versions = {"transformers": tv.TRANSFORMERS_550_VERSION, "huggingface_hub": "1.8.0"},
    )
    assert tv._venv_dir_is_valid(str(root), tv._VENV_T5_550_PACKAGES) is False


def test_a_runtime_repair_survives_a_tiktoken_that_will_not_install(tmp_path, monkeypatch):
    root = tmp_path / ".venv_t5_550"
    installed = []

    def fake_install(pkg, target_dir):
        installed.append(pkg)
        return not pkg.startswith("tiktoken")

    monkeypatch.setattr(tv, "_install_to_dir", fake_install)
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: False)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == list(tv._VENV_T5_550_PACKAGES)

    def fake_install_failing_required(pkg, target_dir):
        return not pkg.startswith("hf_xet")

    monkeypatch.setattr(tv, "_install_to_dir", fake_install_failing_required)
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is False
