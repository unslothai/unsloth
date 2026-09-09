import importlib.util
import sys
import types
from pathlib import Path

from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_TYPE_PATH = REPO_ROOT / "unsloth" / "device_type.py"
CUDA_PROPERTIES = types.SimpleNamespace(
    name = "NVIDIA B200",
    total_memory = 16 * 1024**3,
    major = 10,
    minor = 0,
)


def _load_device_type(
    monkeypatch,
    torch_module,
    mlx_available = False,
    allow_cpu = False,
):
    # Always pinned, never inherited.
    # UNSLOTH_ALLOW_CPU short-circuits get_device_type() to "cuda", so a GPU-less host that exports it silently rewrites
    # what the hip and xpu cases are testing.
    if allow_cpu:
        monkeypatch.setenv("UNSLOTH_ALLOW_CPU", "1")
    else:
        monkeypatch.delenv("UNSLOTH_ALLOW_CPU", raising = False)

    package_name = "_device_helpers_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(DEVICE_TYPE_PATH.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)

    bnb_availability = types.ModuleType(f"{package_name}.bnb_availability")
    bnb_availability.native_kernels_ready = lambda *_args, **_kwargs: True
    monkeypatch.setitem(sys.modules, bnb_availability.__name__, bnb_availability)

    zoo = types.ModuleType("unsloth_zoo")
    zoo.__path__ = []
    zoo_utils = types.ModuleType("unsloth_zoo.utils")
    zoo_utils.Version = Version
    zoo_mlx = types.ModuleType("unsloth_zoo.mlx")
    zoo_mlx.is_mlx_available = lambda: mlx_available
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.utils", zoo_utils)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", zoo_mlx)

    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes.__version__ = "0.49.2"
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes)

    if torch_module is None:
        monkeypatch.setitem(sys.modules, "torch", None)
    else:
        monkeypatch.setitem(sys.modules, "torch", torch_module)

    module_name = f"{package_name}.device_type"
    spec = importlib.util.spec_from_file_location(module_name, DEVICE_TYPE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _fake_torch(
    *,
    properties,
    hip_version = None,
    xpu_backend = None,
    cuda_available = True,
):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        device_count = lambda: 1,
        get_device_properties = lambda _index: properties,
        get_device_name = lambda _index: "",
        empty_cache = lambda: None,
        current_device = lambda: 0,
    )
    torch.version = types.SimpleNamespace(
        cuda = "12.8",
        hip = hip_version,
        xpu = "2026.1",
    )
    if xpu_backend is not None:
        torch.xpu = xpu_backend
    return torch


def test_cuda_import_does_not_require_torch_xpu(monkeypatch):
    torch = _fake_torch(properties = CUDA_PROPERTIES)

    device_type = _load_device_type(monkeypatch, torch)

    assert not hasattr(torch, "xpu")
    assert device_type._DEVICE_MODULE is torch.cuda


def test_hip_stats_preserve_arch_name_fallback(monkeypatch):
    properties = types.SimpleNamespace(
        name = "AMD Radeon Graphics",
        total_memory = 8 * 1024**3,
        gcnArchName = "gfx1100:sramecc+:xnack-",
    )
    torch = _fake_torch(properties = properties, hip_version = "6.3")
    device_type = _load_device_type(monkeypatch, torch)

    name, snippet, max_memory = device_type.get_device_stats()

    assert name == "AMD gfx1100 GPU. "
    assert snippet == "ROCm Toolkit: 6.3."
    assert max_memory == 8.0


def test_xpu_cache_and_current_device_dispatch(monkeypatch):
    xpu_calls = []
    xpu_backend = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        empty_cache = lambda: xpu_calls.append("empty_cache"),
        current_device = lambda: 3,
        get_device_properties = lambda _index: types.SimpleNamespace(
            name = "Intel Arc",
            total_memory = 8 * 1024**3,
        ),
    )
    torch = _fake_torch(
        properties = CUDA_PROPERTIES,
        xpu_backend = xpu_backend,
        cuda_available = False,
    )
    device_type = _load_device_type(monkeypatch, torch)

    device_type.clean_gpu_cache()
    name, snippet, max_memory = device_type.get_device_stats()

    assert xpu_calls == ["empty_cache"]
    assert device_type.get_current_device() == 3
    assert (name, snippet, max_memory) == ("Intel Arc. ", "Intel Toolkit: 2026.1.", 8.0)


def test_cpu_fallback_does_not_override_mlx(monkeypatch):
    # UNSLOTH_ALLOW_CPU used to be checked first, so an MLX Mac reported "cuda" and get_device_count() then hit torch,
    # which is never imported there.
    device_type = _load_device_type(
        monkeypatch,
        torch_module = None,
        mlx_available = True,
        allow_cpu = True,
    )

    assert device_type.DEVICE_TYPE == "mlx"
    assert device_type.DEVICE_COUNT == 1


def test_cpu_fallback_still_reports_cuda_off_mlx(monkeypatch):
    # The GPU hosts' behaviour must be unchanged: no MLX means the CPU fallback wins.
    torch = _fake_torch(properties = CUDA_PROPERTIES, cuda_available = False)

    device_type = _load_device_type(monkeypatch, torch, allow_cpu = True)

    assert device_type.DEVICE_TYPE == "cuda"
    assert device_type.DEVICE_COUNT == 1


def test_mlx_helpers_do_not_require_torch(monkeypatch):
    device_type = _load_device_type(
        monkeypatch,
        torch_module = None,
        mlx_available = True,
    )
    device_type.clean_gpu_cache()

    assert device_type._DEVICE_MODULE is None
    assert device_type.get_current_device() == 0


def test_model_call_sites_use_shared_cache_dispatch():
    llama_source = (REPO_ROOT / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8")
    vision_source = (REPO_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8")
    gemma_source = (REPO_ROOT / "unsloth" / "models" / "gemma.py").read_text(encoding = "utf-8")
    gemma2_source = (REPO_ROOT / "unsloth" / "models" / "gemma2.py").read_text(encoding = "utf-8")
    granite_source = (REPO_ROOT / "unsloth" / "models" / "granite.py").read_text(encoding = "utf-8")

    assert "torch.xpu.empty_cache()" not in llama_source
    assert "torch.xpu.empty_cache()" not in vision_source
    assert "torch.cuda.empty_cache()" not in vision_source
    assert "device_context" not in llama_source
    assert "device_context" not in vision_source
    assert 'if DEVICE_TYPE == "xpu":\n            vllm_version = ""' in vision_source
    assert "torch.cuda.current_device()" not in gemma_source
    assert gemma_source.count("get_current_device()") >= 3
    assert "torch.cuda.empty_cache()" not in gemma_source
    assert "clean_gpu_cache()" in gemma_source
    assert "torch.cuda.empty_cache()" not in gemma2_source
    assert "clean_gpu_cache()" in gemma2_source
    assert "torch.cuda.empty_cache()" not in granite_source
    assert "clean_gpu_cache()" in granite_source
