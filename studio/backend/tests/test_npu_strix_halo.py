# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Strix Halo GPU hint on the NPU status and catalog. Every probe is pinned; no host is read."""

from __future__ import annotations

import pytest

from core.inference import npu_backend as nb
from utils.hardware import amd, hardware as hw

_XDNA2 = {"present": True, "supported": True, "family": "XDNA2", "name": "NPU"}


def _inventory(*devices, unanswered = ()):
    return {
        "available": bool(devices),
        "devices": list(devices),
        "sources": [],
        "unknown": bool(unanswered),
        "unanswered": list(unanswered),
    }


def _pin_inventory(
    monkeypatch,
    inventory,
    system = "Windows",
    kfd = None,
):
    monkeypatch.setattr(hw, "get_physical_gpu_inventory", lambda **_kwargs: inventory)
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda *_args: kfd)


@pytest.mark.parametrize(
    "device",
    [
        {"vendor": "amd", "name": "AMD Radeon(TM) 8060S Graphics"},
        {"vendor": "amd", "name": "AMD Radeon(TM) 8050S Graphics"},
        # A driver that writes AdapterFamily but a name the table does not know.
        {"vendor": "amd", "name": "AMD Radeon(TM) Graphics", "gfx": "gfx1151"},
        {"vendor": "amd", "name": None, "gfx_candidates": ["gfx1151"]},
    ],
)
def test_a_strix_halo_igpu_is_recognised_from_the_inventory(monkeypatch, device):
    _pin_inventory(monkeypatch, _inventory(device))
    assert hw.strix_halo_gpu_present() is True


@pytest.mark.parametrize(
    "device",
    [
        {"vendor": "amd", "name": "AMD Radeon(TM) 890M Graphics", "gfx": "gfx1150"},
        {"vendor": "amd", "name": "AMD Radeon AI PRO R9700", "gfx": "gfx1201"},
        # Another vendor's card cannot be Strix Halo whatever it is called.
        {"vendor": "nvidia", "name": "8060S"},
    ],
)
def test_other_gpus_are_not_strix_halo(monkeypatch, device):
    _pin_inventory(monkeypatch, _inventory(device))
    assert hw.strix_halo_gpu_present() is False


def test_linux_without_rocm_userspace_falls_back_to_kfd(monkeypatch):
    nameless = {"vendor": "amd", "name": None}
    _pin_inventory(monkeypatch, _inventory(nameless), system = "Linux", kfd = ["gfx1151"])
    assert hw.strix_halo_gpu_present() is True
    _pin_inventory(monkeypatch, _inventory(nameless), system = "Linux", kfd = ["gfx1150"])
    assert hw.strix_halo_gpu_present() is False


def test_an_unanswered_amd_probe_is_unknown_not_no(monkeypatch):
    _pin_inventory(monkeypatch, _inventory(unanswered = ["amd"]))
    assert hw.strix_halo_gpu_present() is None
    # A timed-out nvidia-smi says nothing about the AMD half.
    _pin_inventory(monkeypatch, _inventory(unanswered = ["nvidia"]))
    assert hw.strix_halo_gpu_present() is False


class _Installer:
    def installed_lemond(self, root):
        return None


def _backend(
    tmp_path,
    monkeypatch,
    strix_halo,
    hardware = _XDNA2,
):
    answers = iter(strix_halo if isinstance(strix_halo, list) else [strix_halo] * 99)
    monkeypatch.setattr(nb, "_installer_module", lambda: _Installer())
    monkeypatch.setattr(nb, "_pinned_flm_version", lambda: None)
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: dict(hardware))
    monkeypatch.setattr(nb, "_strix_halo_gpu", lambda: next(answers))
    return nb.LemonadeNpuBackend(root = tmp_path / "lemonade")


def test_the_status_recommends_the_gpu_on_strix_halo(tmp_path, monkeypatch):
    assert _backend(tmp_path, monkeypatch, True).status()["gpu_recommended"] is True


def test_the_status_recommends_nothing_on_other_npu_hosts(tmp_path, monkeypatch):
    assert _backend(tmp_path, monkeypatch, False).status()["gpu_recommended"] is False


def test_no_supported_npu_means_no_hint_and_no_gpu_probe(tmp_path, monkeypatch):
    backend = _backend(tmp_path, monkeypatch, [], hardware = {"present": False, "supported": False})
    assert backend.status()["gpu_recommended"] is False


def test_an_unknown_answer_is_asked_again_and_a_known_one_is_kept(tmp_path, monkeypatch):
    backend = _backend(tmp_path, monkeypatch, [None, True])
    assert backend.status()["gpu_recommended"] is False
    assert backend.status()["gpu_recommended"] is True
    # The iterator is spent: a third probe would raise StopIteration.
    assert backend.status()["gpu_recommended"] is True


class _Response:
    status_code = 200

    def json(self):
        return {
            "data": [
                {
                    "id": name,
                    "recipe": "flm",
                    "labels": ["chat"],
                    "checkpoint": tag,
                    "downloaded": True,
                }
                for name, tag in (
                    ("qwen3.5-4b-FLM", "qwen3.5:4b"),
                    ("qwen3.6-moe-35b-FLM", "qwen3.6-moe:35b-a3b"),
                    ("qwen3-it-4b-FLM", "qwen3-it:4b"),
                    ("qwen3.8-mtp-27b-FLM", "qwen3.8-mtp:27b"),
                    ("gemma3-4b-FLM", "gemma3:4b"),
                )
            ]
        }


class _Server:
    def request(self, *_args, **_kwargs):
        return _Response()


def _known_issues(backend, monkeypatch) -> dict:
    monkeypatch.setattr(backend, "_ensure_running", lambda: _Server())
    return {model.checkpoint: model.to_json()["known_issue_url"] for model in backend.catalog()}


def test_the_catalog_tags_the_qwen_tags_that_fail_to_load_on_strix_halo(tmp_path, monkeypatch):
    issue = "https://github.com/ROCm/FastFlowLM/issues/705"
    assert _known_issues(_backend(tmp_path, monkeypatch, True), monkeypatch) == {
        "qwen3.5:4b": issue,
        "qwen3.6-moe:35b-a3b": issue,
        "qwen3-it:4b": None,
        "qwen3.8-mtp:27b": None,
        "gemma3:4b": None,
    }


def test_the_catalog_tags_nothing_off_strix_halo(tmp_path, monkeypatch):
    tags = _known_issues(_backend(tmp_path, monkeypatch, False), monkeypatch)
    assert set(tags.values()) == {None}


def test_managed_accounts_get_no_gpu_hint():
    from routes import npu as npu_route
    assert npu_route._MANAGED_ACCOUNT_STATUS["gpu_recommended"] is False
