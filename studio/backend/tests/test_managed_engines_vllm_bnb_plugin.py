# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CUDA vLLM locks from 0.28 (no built-in BitsAndBytes) must carry vllm-bnb-plugin for int4."""

import re

from core.inference import engine_install as install

ENGINES = install.requirements("vllm").parent


def _locked(lock):
    text = (ENGINES / f"{lock}.txt").read_text(encoding = "utf-8")
    return dict(re.findall(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;\\]+)", text, re.M))


def _version(text):
    return tuple(int(part) for part in re.match(r"(\d+)\.(\d+)", text).groups())


def test_every_cuda_vllm_lock_without_builtin_bitsandbytes_pins_the_plugin():
    releases = install.PROFILES["vllm"]["releases"]
    assert releases, "no CUDA vLLM releases"
    for release in releases:
        pins = _locked(release["lock"])
        assert pins["vllm"] == release["version"]
        if _version(release["version"]) >= (0, 28):
            assert pins.get("vllm-bnb-plugin") == "0.0.3", release
            assert "bitsandbytes" in pins
        else:
            assert "vllm-bnb-plugin" not in pins, release


def test_cuda_smoke_imports_the_plugin(monkeypatch):
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    monkeypatch.setattr(install, "_studio_packages", lambda: {})
    assert install.profile("vllm")["lock"] == "vllm-linux-cu130-torch213"
    smoke = install._smoke_source("vllm")
    assert "import vllm_bnb_plugin" in smoke and "import bitsandbytes" in smoke
    compile(smoke, "smoke", "exec")
