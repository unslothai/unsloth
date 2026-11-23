# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM on Apple Silicon: the vllm-metal profile, its support checks and its memory budget."""

import json
from types import SimpleNamespace

import pytest

from core.inference import engine_adapters, managed_engine
from core.inference import engine_install as install


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(install.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(install.platform, "mac_ver", lambda: ("26.5.1", ("", "", ""), "arm64"))


def test_apple_silicon_installs_vllm_with_the_metal_plugin(mac, monkeypatch):
    monkeypatch.setattr(install, "_studio_packages", lambda: {"torch": "2.14.0"})
    assert install.gpu_platform() == "metal"
    chosen = install.profile("vllm")
    pins = install._pins("vllm")
    lock = install.requirements("vllm").read_text()
    assert chosen["lock"] == "vllm-macos-metal" and chosen["python"] == (3, 12)
    assert chosen["precisions"] == ("auto", "bf16", "fp16") and not chosen["bitsandbytes"]
    assert pins["torch"][0] == chosen["torch"]
    assert "vllm @ https://" in lock and "vllm-metal @ https://" in lock
    assert pins["vllm"][0] == "0.33.0+cpu" and pins["vllm-metal"][0] == "0.33.0"
    # Its own environment with a uv-managed CPython 3.12, never Studio's packages.
    plan = install.install_plan("vllm")
    assert not plan["shared"]
    assert (
        "vllm @ https://" in plan["requirements"]
        and "vllm-metal @ https://" in plan["requirements"]
    )
    assert plan["requirements"].count("--hash=sha256:") >= len(pins)
    assert install._venv_python_args("vllm")[:2] == ["--python", "3.12"]
    assert install._index_arguments("vllm") == ["--index-url", "https://pypi.org/simple"]
    sizes = install._compat_file("vllm").get("sizes")
    assert sizes and sizes["vllm"] and sizes["vllm-metal"], "regenerate with engine_compat.py"
    smoke = install._smoke_source("vllm")
    assert "vllm_metal." in smoke and "torch.version.cuda" not in smoke
    assert install.profile("sglang")["platform"] == "cuda"


@pytest.mark.parametrize(
    "engine, machine, version, reason",
    [
        ("vllm", "arm64", "26.5.1", None),
        ("vllm", "arm64", "15.0", None),
        ("vllm", "arm64", "14.7", "requires macOS 15 or newer"),
        ("vllm", "x86_64", "15.0", "requires an Apple Silicon Mac"),
        ("sglang", "arm64", "26.5.1", "SGLang requires an NVIDIA GPU"),
    ],
)
def test_macos_support_needs_apple_silicon_on_macos_15(
    mac, monkeypatch, engine, machine, version, reason
):
    monkeypatch.setattr(install.platform, "machine", lambda: machine)
    monkeypatch.setattr(install.platform, "mac_ver", lambda: (version, ("", "", ""), machine))
    found = install.support_reason(engine, wait = False)
    assert found is None if reason is None else reason in found


def test_metal_budget_reads_unified_memory(monkeypatch):
    gib = 2**30
    psutil = SimpleNamespace(
        virtual_memory = lambda: SimpleNamespace(total = 128 * gib, available = 64 * gib)
    )
    monkeypatch.setitem(__import__("sys").modules, "psutil", psutil)
    assert engine_adapters.metal_memory_rows([0]) == [(128 * 1024, 64 * 1024)]
    monkeypatch.setattr("utils.vram_budget_settings.get_vram_budget_fraction", lambda: 0.9)
    fraction = engine_adapters.gpu_memory_fraction([0], 512, 0.0, engine_adapters.metal_memory_rows)
    assert fraction == int((64 * 1024 - 512) / (128 * 1024) * 1000) / 1000


@pytest.mark.parametrize(
    "platform, kv, refused",
    [
        ("metal", "turboquant", None),
        ("metal", "fp8", "cannot use an FP8 KV cache"),
        ("cuda", "fp8", None),
        ("cuda", "turboquant", "cannot use a TurboQuant KV cache"),
    ],
)
def test_kv_cache_options_follow_the_platform(tmp_path, monkeypatch, platform, kv, refused):
    # vllm-metal 0.33 stores its paged KV cache in the model dtype and refuses --kv-cache-dtype fp8.
    monkeypatch.setattr(install, "gpu_platform", lambda: platform)
    monkeypatch.setattr(managed_engine, "profile", install.profile)
    monkeypatch.setattr(
        managed_engine.subprocess, "run", lambda argv, **_: SimpleNamespace(stdout = "9.0\n")
    )
    (tmp_path / "config.json").write_text(json.dumps({"num_attention_heads": 8}))
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    if refused:
        with pytest.raises(ValueError, match = refused):
            managed_engine.validate_model(config, kv_cache_dtype = kv)
    else:
        assert managed_engine.validate_model(config, kv_cache_dtype = kv)["kv_cache_dtype"] == kv
    assert install.status("vllm")["kv_cache_dtypes"] == list(
        install.profile("vllm")["kv_cache_dtypes"]
    )
    if platform == "metal":
        (tmp_path / "config.json").write_text(
            json.dumps({"num_attention_heads": 8, "kv_lora_rank": 512})
        )
        with pytest.raises(ValueError, match = "MLA"):
            managed_engine.validate_model(config, kv_cache_dtype = "turboquant")


def test_turboquant_reaches_vllm_metal_as_platform_config():
    argv = engine_adapters.ADAPTERS["vllm"].command(
        "/env/bin/python",
        "org/model",
        45678,
        "key",
        4096,
        0.8,
        1,
        options = {"tool_parser": None, "kv_cache_dtype": "turboquant"},
    )
    assert "--kv-cache-dtype" not in argv
    assert json.loads(argv[argv.index("--additional-config") + 1]) == {"turboquant": True}
