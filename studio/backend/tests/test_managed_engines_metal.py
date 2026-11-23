# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""vLLM on Apple Silicon: the vllm-metal profile, its support checks and its memory budget."""

import json
from types import SimpleNamespace

import pytest

from core.inference import engine_adapters, managed_engine, wsl_host
from core.inference import engine_install as install


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(wsl_host, "active", lambda: False)
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
    # vllm-metal multiplies the fraction by Metal's recommended working set, not by all of RAM.
    rows = lambda ids: engine_adapters.metal_memory_rows(ids, 96 * gib)
    assert rows([0]) == [(96 * 1024, 64 * 1024)]
    fraction = engine_adapters.gpu_memory_fraction([0], 512, 0.0, rows)
    assert fraction == int((64 * 1024 - 512) / (96 * 1024) * 1000) / 1000
    assert engine_adapters.metal_memory_rows([0], 200 * gib) == [(128 * 1024, 64 * 1024)]


def test_metal_working_set_is_read_from_the_engine_env(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(stdout = "some import warning\n103079215104\n")

    monkeypatch.setattr(managed_engine.subprocess, "run", run)
    assert managed_engine._metal_working_set({"path": "/env"}, {}) == 103079215104
    assert (
        seen["argv"][0] == "/env/bin/python"
        and "max_recommended_working_set_size" in seen["argv"][-1]
    )
    for broken in (
        lambda *a, **k: SimpleNamespace(stdout = ""),
        lambda *a, **k: SimpleNamespace(stdout = "Traceback\nKeyError\n"),
        lambda *a, **k: (_ for _ in ()).throw(managed_engine.subprocess.CalledProcessError(1, "x")),
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    ):
        monkeypatch.setattr(managed_engine.subprocess, "run", broken)
        assert managed_engine._metal_working_set({"path": "/env"}, {}) == 0


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
        gemma4 = {
            "model_type": "gemma4",
            "vision_config": {"depth": 16},
            "text_config": {
                "num_attention_heads": 8,
                "num_hidden_layers": 2,
                "global_head_dim": 512,
                "layer_types": ["sliding_attention", "full_attention"],
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(gemma4))
        with pytest.raises(ValueError, match = "per-layer KV shapes"):
            managed_engine.validate_model(config, kv_cache_dtype = "turboquant")
        assert managed_engine.validate_model(config)
        # GPT-OSS and Qwen3.5 list layer_types without a global_head_dim: uniform KV shapes.
        del gemma4["text_config"]["global_head_dim"]
        (tmp_path / "config.json").write_text(json.dumps(gemma4))
        assert managed_engine.validate_model(config, kv_cache_dtype = "turboquant")
        for shape, refused in (
            ({"hidden_size": 2560, "num_attention_heads": 32}, True),
            ({"hidden_size": 3072, "num_attention_heads": 32}, True),
            ({"hidden_size": 896, "num_attention_heads": 14}, False),
            ({"hidden_size": 1024, "num_attention_heads": 16, "head_dim": 128}, False),
        ):
            (tmp_path / "config.json").write_text(json.dumps(shape))
            if refused:
                with pytest.raises(ValueError, match = "head size"):
                    managed_engine.validate_model(config, kv_cache_dtype = "turboquant")
                assert managed_engine.validate_model(config)
            else:
                assert managed_engine.validate_model(config, kv_cache_dtype = "turboquant")


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


def test_metal_loads_on_the_apple_gpu_when_studio_sees_none(mac, monkeypatch):
    # A Mac whose own MLX stack is unusable falls back to CPU and lists no GPU, yet the engine env has MLX.
    monkeypatch.setattr(managed_engine, "resolve_requested_gpu_ids", lambda ids, **_: [])
    monkeypatch.setattr(managed_engine, "installed", lambda engine: None)
    request = SimpleNamespace(gpu_ids = [], model_path = "org/model")
    with pytest.raises(ValueError, match = "Install vLLM"):
        managed_engine.validate_load("vllm", request)
    with pytest.raises(ValueError, match = "one GPU"):
        managed_engine.validate_load("vllm", SimpleNamespace(gpu_ids = [1], model_path = "org/model"))
    monkeypatch.setattr(install, "gpu_platform", lambda: "cuda")
    with pytest.raises(ValueError, match = "no GPU"):
        managed_engine.validate_load("vllm", request)


@pytest.mark.parametrize("method", ["gptq", "bitnet"])
def test_metal_refuses_quantizations_vllm_metal_never_loads(tmp_path, monkeypatch, method):
    monkeypatch.setattr(install, "gpu_platform", lambda: "metal")
    monkeypatch.setattr(managed_engine, "profile", install.profile)
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": method}})
    )
    with pytest.raises(ValueError, match = "Apple Silicon cannot load"):
        managed_engine.validate_model(config)


# vllm-metal loads these through mlx-lm for some model families (GPT-OSS MXFP4, Qwen3.5 FP8).
@pytest.mark.parametrize(
    "quant",
    [
        {"quant_method": "awq"},
        {"quant_method": "mxfp4"},
        {"quant_method": "fp8"},
        {"quant_method": "compressed-tensors"},
        {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        {"group_size": 64, "bits": 4},
    ],
)
def test_metal_leaves_family_dependent_quantizations_to_the_engine(tmp_path, monkeypatch, quant):
    monkeypatch.setattr(install, "gpu_platform", lambda: "metal")
    monkeypatch.setattr(managed_engine, "profile", install.profile)
    config = SimpleNamespace(is_local = True, path = str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant}))
    managed_engine.validate_model(config)
