# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/npu/compat from fixture FastFlowLM and converter files; nothing on the host is read."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from core.inference import npu_compat as nc

_FIXTURES = Path(__file__).parent / "fixtures" / "npu_compat"
_HF = json.loads((_FIXTURES / "hf_configs.json").read_text())
_Q4K_ARCHS = ("qwen35", "qwen35moe")


def _fetcher(calls: list[str]):
    def fetch(url: str):
        calls.append(url)
        repo = url.split("/FastFlowLM/")[1].split("/")[0]
        return _HF.get(repo)

    return fetch


@pytest.fixture
def flm(tmp_path):
    binary = tmp_path / "bin" / "flm"
    binary.parent.mkdir()
    shutil.copy(_FIXTURES / "flm" / "model_list.json", binary.parent / "model_list.json")
    return binary


def _shapes(flm, tmp_path, calls):
    return nc.flm_shapes(flm, tmp_path / "models", tmp_path / "cache.json", _fetcher(calls))


def _by_key(shapes):
    return {(s["architecture"], s["layers"], s["hidden"], s["experts"]): s for s in shapes}


def test_shapes_come_from_each_catalog_models_config(flm, tmp_path):
    calls: list[str] = []
    shapes = _by_key(_shapes(flm, tmp_path, calls))
    assert set(shapes) == {
        ("qwen3", 28, 1024, 0),
        ("qwen3", 36, 4096, 0),
        ("qwen3vl", 36, 2560, 0),
        ("qwen35", 32, 4096, 0),
        ("qwen35", 64, 5120, 0),
        ("qwen35moe", 40, 2048, 256),
        ("lfm3", 24, 2048, 0),
        ("llama", 32, 4096, 0),
        ("gemma4", 42, 2560, 0),
        ("gemma4", 48, 3840, 0),
        ("gpt-oss", 24, 2880, 32),
        ("qwen2vl", 36, 2048, 0),
    }
    # One shape, two catalog models; a word size is not a name token.
    big = shapes[("qwen35", 64, 5120, 0)]
    assert big["checkpoints"] == ["qwen3.8-mtp:27b", "d3:turbo"]
    assert big["names"] == ["Qwen3.8-27B", "D3-Turbo"]
    assert shapes[("qwen3", 36, 4096, 0)]["names"] == ["Qwen3-8B"]
    assert big["sizes"] == ["27b"]
    # dev04 builds, DFlash drafters, embeddings and Whisper are never fetched.
    assert not any(x in url for url in calls for x in ("core", "DFlash", "Embedding", "Whisper"))


def test_a_downloaded_config_is_read_first_and_fetches_are_cached_per_listing(flm, tmp_path):
    local = tmp_path / "models" / "models" / "LFM3-1.7B-NPU2" / "config.json"
    local.parent.mkdir(parents = True)
    local.write_text(json.dumps({"model_type": "lfm3", "num_hidden_layers": 16, "hidden_size": 1024}))
    calls: list[str] = []
    assert ("lfm3", 16, 1024, 0) in _by_key(_shapes(flm, tmp_path, calls))
    assert not any("LFM3" in url for url in calls)

    again: list[str] = []
    _shapes(flm, tmp_path, again)
    assert again == []

    listing = json.loads((flm.parent / "model_list.json").read_text())
    listing["models"]["lfm3"]["1.7b"]["url"] += "/resolve/abc123"
    (flm.parent / "model_list.json").write_text(json.dumps(listing))
    after: list[str] = []
    _shapes(flm, tmp_path, after)
    assert after  # a new FastFlowLM listing reads every config again


def test_an_unreachable_config_makes_the_table_unknown_and_is_retried(flm, tmp_path):
    # A partial table would refuse sizes FastFlowLM runs, so one missing config is "unknown".
    calls: list[str] = []
    partial = lambda url: None if "LFM3" in url else _fetcher(calls)(url)
    assert nc.flm_shapes(flm, tmp_path / "m", tmp_path / "cache.json", partial) is None
    retried: list[str] = []
    assert _by_key(_shapes(flm, tmp_path, retried))
    assert retried == [url for url in retried if "LFM3" in url] and retried


def test_no_fastflowlm_is_unknown_not_empty(tmp_path):
    assert nc.flm_shapes(None, tmp_path, tmp_path / "c.json") is None
    assert nc.flm_shapes(tmp_path / "missing" / "flm", tmp_path, tmp_path / "c.json") is None


class _Installer:
    def __init__(self, installed: bool) -> None:
        self.installed = {"q4nx", "q4k"} if installed else set()
        self.installs: list[str] = []

    def load_pins(self, name = "q4nx"):
        return {"commit": f"{name}-commit"}

    def converter_for_architecture(self, architecture):
        return "q4k" if architecture.lower() in _Q4K_ARCHS else "q4nx"

    def installed_converter(self, root, pins):
        name = pins["commit"].removesuffix("-commit")
        return _FIXTURES / name / "convert.py" if name in self.installed else None

    def install(self, root, name = "q4nx"):
        self.installs.append(name)
        self.installed.add(name)
        return _FIXTURES / name / "convert.py"


def test_converter_architectures_are_parsed_from_its_source():
    support = nc.converter_support(Path("unused"), _Installer(installed = True))
    # Each architecture from the converter it is routed to; qwen35 by the q4k geometry table.
    assert support["architectures"] == sorted(
        {"gemma-4", "gemma4", "gpt-oss", "lfm2", "llama", "qwen2.5-vl", "qwen3", "qwen3-vl",
         "qwen3.5-0.8b", "qwen35-0.8b", "qwen3vl", "qwen35", "qwen35moe"}
    )
    assert "lfm3" not in support["architectures"]
    assert support["basenames"] == ["gemma-4", "qwen2.5-vl", "qwen3-vl", "qwen3.5-0.8b", "qwen35-0.8b"]
    assert support["qwen35_shapes"] == [[24, 1024], [32, 4096]]
    assert support["commits"] == {"q4nx": "q4nx-commit", "q4k": "q4k-commit"}


def test_a_missing_converter_is_installed_or_reported_unknown():
    installer = _Installer(installed = False)
    assert nc.converter_support(Path("unused"), installer, install = False) is None
    assert installer.installs == []
    assert nc.converter_support(Path("unused"), installer) is not None
    assert installer.installs == ["q4nx", "q4k"]


def test_each_shape_says_whether_it_runs_and_whether_it_converts(flm, tmp_path):
    compat = nc.build_compat(
        _shapes(flm, tmp_path, []), nc.converter_support(Path("unused"), _Installer(True))
    )
    rows = _by_key(compat["shapes"])
    assert all(row["runs"] for row in rows.values())
    assert rows[("qwen35", 32, 4096, 0)]["converts"] is True  # Qwen3.5 9B
    assert rows[("qwen35", 64, 5120, 0)]["converts"] is False  # 27B: runs, no converter config
    assert rows[("lfm3", 24, 2048, 0)]["converts"] is False
    assert rows[("qwen3", 36, 4096, 0)]["converts"] is True
    assert rows[("qwen2vl", 36, 2048, 0)]["converts"] is True  # by its name, "Qwen2.5-VL"
    arch = {row["architecture"]: row for row in compat["architectures"]}
    assert arch["lfm3"] == {"architecture": "lfm3", "runs": True, "converts": False}
    assert arch["lfm2"] == {"architecture": "lfm2", "runs": False, "converts": True}
    assert arch["qwen3"] == {"architecture": "qwen3", "runs": True, "converts": True}
    assert "qwen2.5-vl" not in arch and "qwen35-0.8b" not in arch  # basename prefixes
    # Qwen3 14B is no FastFlowLM shape.
    assert not any(k[0] == "qwen3" and k[2] == 5120 for k in rows)
    assert compat["fastflowlm_known"] and compat["converter_known"]


def test_unknown_halves_are_reported_as_unknown():
    compat = nc.build_compat(None, None)
    assert compat == {
        "fastflowlm_known": False,
        "converter_known": False,
        "converter": None,
        "shapes": [],
        "architectures": [],
    }


def test_runnable_shapes_name_each_shape(monkeypatch, flm, tmp_path):
    compat = nc.build_compat(_shapes(flm, tmp_path, []), None)
    monkeypatch.setattr(nc, "current", lambda install_converter = True: compat)
    table = nc.runnable_shapes()
    assert table["gemma4"] == {(42, 2560, 0): "Gemma4-E4B-IT", (48, 3840, 0): "Gemma4-12B-IT"}
    monkeypatch.setattr(nc, "current", lambda install_converter = True: nc.build_compat(None, None))
    assert nc.runnable_shapes() is None


def test_the_route_returns_the_table_without_starting_the_runtime(monkeypatch):
    from routes import npu as npu_routes

    monkeypatch.setattr(nc, "current", lambda: {"shapes": [], "fastflowlm_known": False})
    monkeypatch.setattr(
        npu_routes, "get_npu_backend", lambda: pytest.fail("the runtime must not be touched")
    )
    assert asyncio.run(npu_routes.npu_compat()) == {"shapes": [], "fastflowlm_known": False}
