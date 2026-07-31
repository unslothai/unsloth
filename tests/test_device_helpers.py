import importlib.util
import sys
import types
import uuid
from pathlib import Path

from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_TYPE_PATH = REPO_ROOT / "unsloth" / "device_type.py"


def _load_device_type(
    monkeypatch,
    torch_module,
    mlx_available = False,
):
    package_name = f"_device_helpers_test_{uuid.uuid4().hex}"
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
    cuda_name = "",
    xpu_backend = None,
):
    torch = types.ModuleType("torch")
    cuda_calls = []
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_properties = lambda _index: properties,
        get_device_name = lambda _index: cuda_name,
        empty_cache = lambda: cuda_calls.append("empty_cache"),
        current_device = lambda: 0,
    )
    torch.version = types.SimpleNamespace(
        cuda = "12.8",
        hip = hip_version,
        xpu = "2026.1",
    )
    if xpu_backend is not None:
        torch.xpu = xpu_backend
    return torch, cuda_calls


def test_cuda_import_does_not_require_torch_xpu(monkeypatch):
    properties = types.SimpleNamespace(
        name = "NVIDIA B200",
        total_memory = 16 * 1024**3,
        major = 10,
        minor = 0,
    )
    torch, _ = _fake_torch(properties = properties)

    device_type = _load_device_type(monkeypatch, torch)

    assert not hasattr(torch, "xpu")
    assert device_type._get_device_module() is torch.cuda


def test_hip_stats_preserve_arch_name_fallback(monkeypatch):
    properties = types.SimpleNamespace(
        name = "AMD Radeon Graphics",
        total_memory = 8 * 1024**3,
        gcnArchName = "gfx1100:sramecc+:xnack-",
    )
    torch, _ = _fake_torch(properties = properties, hip_version = "6.3")
    device_type = _load_device_type(monkeypatch, torch)

    name, snippet, max_memory = device_type.get_device_stats()

    assert name == "AMD gfx1100 GPU. "
    assert snippet == "ROCm Toolkit: 6.3."
    assert max_memory == 8.0


def test_xpu_cache_and_current_device_dispatch(monkeypatch):
    properties = types.SimpleNamespace(
        name = "NVIDIA B200",
        total_memory = 16 * 1024**3,
        major = 10,
        minor = 0,
    )
    xpu_calls = []
    xpu_backend = types.SimpleNamespace(
        empty_cache = lambda: xpu_calls.append("empty_cache"),
        current_device = lambda: 3,
    )
    torch, _ = _fake_torch(properties = properties, xpu_backend = xpu_backend)
    device_type = _load_device_type(monkeypatch, torch)
    device_type.DEVICE_TYPE = "xpu"

    device_type.clean_gpu_cache()

    assert xpu_calls == ["empty_cache"]
    assert device_type.get_current_device() == 3


def test_mlx_helpers_do_not_require_torch(monkeypatch):
    device_type = _load_device_type(
        monkeypatch,
        torch_module = None,
        mlx_available = True,
    )
    device_type.clean_gpu_cache()

    assert device_type._get_device_module() is None
    assert device_type.get_current_device() == 0


def test_model_call_sites_use_shared_cache_dispatch():
    llama_source = (REPO_ROOT / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8")
    vision_source = (REPO_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8")

    assert "torch.xpu.empty_cache()" not in llama_source
    assert "torch.xpu.empty_cache()" not in vision_source
    assert "torch.cuda.empty_cache()" not in vision_source
    assert "device_context" not in llama_source
    assert "device_context" not in vision_source
    assert 'if DEVICE_TYPE == "xpu":\n            vllm_version = ""' in vision_source
