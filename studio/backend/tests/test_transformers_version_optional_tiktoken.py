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


def _sidecar(
    root,
    *names,
    versions = None,
):
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
    assert (
        tv._venv_dir_is_valid(str(root), tv._venv_t5_latest_packages(tv.TRANSFORMERS_550_VERSION))
        is True
    )


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


def test_a_present_tiktoken_is_held_to_its_record_like_any_other(tmp_path, monkeypatch):
    """Absence is what is optional. A tiktoken that is present but whose RECORD names a
    file that is not there (an interrupted install, a native extension left from an
    older interpreter) is a tokenizer that fails at import, and the scan says so."""
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
    (root / "transformers" / "__init__.py").write_text("", encoding = "utf-8")
    info = root / "tiktoken-0.9.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: tiktoken\nVersion: 0.9.0\n", encoding = "utf-8")
    (info / "RECORD").write_text("tiktoken/__init__.py,sha256=abc,1234\n", encoding = "utf-8")
    monkeypatch.delenv(tv._SIDECAR_FILE_CHECK_ENV, raising = False)
    assert tv._sidecar_damaged_files(str(root)) != []
    assert tv._venv_dir_is_valid_and_undamaged(str(root), tv._VENV_T5_550_PACKAGES) is False
    # With the payload the RECORD names in place, the sidecar is whole.
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "__init__.py").write_bytes(b"x" * 1234)
    assert tv._sidecar_damaged_files(str(root)) == []
    assert tv._venv_dir_is_valid_and_undamaged(str(root), tv._VENV_T5_550_PACKAGES) is True
    # A required package's RECORD is held to the disk too.
    hub = root / "huggingface_hub-1.8.0.dist-info"
    (hub / "RECORD").write_text("huggingface_hub/gone.py,sha256=abc,12\n", encoding = "utf-8")
    assert tv._sidecar_damaged_files(str(root)) != []


def test_a_valid_sidecar_missing_tiktoken_is_topped_up_once(tmp_path, monkeypatch):
    """A transient failure while a sidecar was built (the latest sidecar in particular,
    which no setup top-up visits) left tiktoken out for good: every later check accepted
    the sidecar and returned before another install. The top-up adds it without touching
    the rest, and asks once per process when the wheel is unavailable."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    monkeypatch.setattr(tv, "_venv_dir_is_valid_and_undamaged", lambda *a, **k: True)
    installed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tv, "_install_to_dir", lambda pkg, target: installed.append((pkg, target)) or False
    )
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert installed == [("tiktoken", str(root))]
    assert root.is_dir(), "the top-up must not wipe the sidecar"
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert len(installed) == 1, "an unavailable wheel is asked for once per process"
    # A tiktoken that is there is left alone.
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "__init__.py").write_text("", encoding = "utf-8")
    assert tv._ensure_venv_dir(str(root), tv._VENV_T5_550_PACKAGES, "test sidecar") is True
    assert len(installed) == 1


def test_latest_sidecar_activation_tops_up_a_missing_tiktoken(tmp_path, monkeypatch):
    """Model activation goes through _ensure_venv_t5_latest_exists, not the provisioning
    path; a healthy latest sidecar without tiktoken has to get its top-up there too, or a
    latest-tier Qwen model keeps failing to tokenize after a transient install failure."""
    latest = tmp_path / ".venv_t5_latest"
    latest.mkdir()
    packages = ("transformers==9.9.9", "huggingface_hub==1.8.0", "hf_xet==1.4.2", "tiktoken")
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(latest))
    monkeypatch.setattr(tv, "_latest_pin_data", lambda: {"version": "9.9.9", "packages": packages})
    monkeypatch.setattr(tv, "_venv_dir_health", lambda *a, **k: (True, True))
    installed = []
    monkeypatch.setattr(
        tv, "_install_to_dir", lambda pkg, target: installed.append((pkg, target)) or True
    )
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    assert tv._ensure_venv_t5_latest_exists() is True
    assert installed == [("tiktoken", str(latest))]


def test_the_top_up_stays_home_offline_and_yields_to_another_process(tmp_path, monkeypatch):
    """Workers activate tiers on their own: offline, none of them should sit through
    network retries for a best-effort package, and two finding it absent at once must
    not both write into the shared sidecar."""
    root = tmp_path / ".venv_t5_550"
    root.mkdir()
    packages = tv._VENV_T5_550_PACKAGES
    installed = []
    monkeypatch.setattr(tv, "_install_to_dir", lambda pkg, target: installed.append(pkg) or True)
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    tv._top_up_optional_packages(str(root), packages)
    assert installed == []
    monkeypatch.delenv("HF_HUB_OFFLINE")
    monkeypatch.setenv("UV_OFFLINE", "1")
    tv._top_up_optional_packages(str(root), packages)
    assert installed == []
    monkeypatch.delenv("UV_OFFLINE")
    # Another process holds the sidecar's top-up lock: this one leaves it to them.
    with tv._optional_top_up_lock(str(root)) as held:
        assert held is True
        tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
        tv._top_up_optional_packages(str(root), packages)
        assert installed == []
    tv._OPTIONAL_TOP_UP_ATTEMPTED.clear()
    tv._top_up_optional_packages(str(root), packages)
    assert installed == ["tiktoken"]
