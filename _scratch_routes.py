"""Scratch: every route into GGML_CUDA_P2P, plus the real 8x B200 topology."""
import os, sys, types
sys.path.insert(0, "studio/backend")
from core.inference.llama_cpp import LlamaCppBackend as B

class _Props:
    def __init__(self, name): self.name = name

def fake_torch(names):
    return types.SimpleNamespace(
        cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: len(names),
            get_device_properties = lambda i: _Props(names[i]),
        ),
        version = types.SimpleNamespace(hip = None),
    )

def run(names, env, gpu_indices, **envvars):
    """Mimic the call site: sanitize, then apply. REAL nvidia-smi topo -m."""
    B._NVLINK_TOPO_CACHE = None
    B._warned_no_nvlink = False
    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch(names)
    for k in ("UNSLOTH_DISABLE_DC_TUNING", "UNSLOTH_DISABLE_DC_P2P",
              "UNSLOTH_FORCE_DC_P2P", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        os.environ.pop(k, None)
    for k, v in envvars.items(): os.environ[k] = v
    try:
        removed = B._sanitize_p2p_env(env)
        ok = B._apply_datacenter_env(env, gpu_indices, p2p_opted_out = removed is not None)
        return ok, env
    finally:
        if real_torch is not None: sys.modules["torch"] = real_torch
        else: sys.modules.pop("torch", None)
        for k in envvars: os.environ.pop(k, None)
        B._NVLINK_TOPO_CACHE = None

B8 = ["NVIDIA B200"] * 8
ADA2 = ["NVIDIA RTX 6000 Ada Generation"] * 2
GEFORCE2 = ["NVIDIA GeForce RTX 3090"] * 2

print("=== REAL host topology (8x B200 NVLink), must keep all three flags ===")
ok, env = run(B8, {}, [0, 1])
print(f"  qualified={ok} env={env}")
ok, env = run(B8, {}, None)
print(f"  gpu_indices=None: qualified={ok} env={env}")

print("\n=== route 1: setdefault on a non-NVLink DC box (the #10613 host) ===")
ok, env = run(ADA2, {}, [0, 1])
print(f"  qualified={ok} env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")

print("\n=== route 2: user-supplied TRUTHY value, non-NVLink DC box ===")
ok, env = run(ADA2, {"GGML_CUDA_P2P": "1"}, [0, 1])
print(f"  env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")

print("\n=== route 2b: user TRUTHY on a CONSUMER 2x 3090 box (never reaches DC gate) ===")
ok, env = run(GEFORCE2, {"GGML_CUDA_P2P": "1"}, [0, 1])
print(f"  qualified={ok} env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")

print("\n=== route 3: UNSLOTH_FORCE_DC_P2P=1 on the non-NVLink DC box ===")
ok, env = run(ADA2, {}, [0, 1], UNSLOTH_FORCE_DC_P2P = "1")
print(f"  env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")

print("\n=== route 4: falsy inherited value, consumer box ===")
ok, env = run(GEFORCE2, {"GGML_CUDA_P2P": "0"}, [0, 1])
print(f"  qualified={ok} env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")

print("\n=== falsy value on the NVLink box must not be reintroduced ===")
ok, env = run(B8, {"GGML_CUDA_P2P": "0"}, [0, 1])
print(f"  env={env}  -> P2P present: {'GGML_CUDA_P2P' in env}")
