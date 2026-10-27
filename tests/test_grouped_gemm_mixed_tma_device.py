# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mixed H100+A100: TMA must hold on every card; launches and scratch follow the operands' device."""

import ast
import contextlib
import importlib.util
import sys
import types
import typing
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if triton.__name__ != "triton":
    # unsloth_zoo's macOS stub answers every attribute, so the TMA API cannot be hidden.
    pytest.skip("triton is a stub", allow_module_level = True)
import torch._dynamo  # noqa: E402,F401  # import before custom_op is stubbed out below
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GG_DIR = ROOT / "unsloth" / "kernels" / "moe" / "grouped_gemm"
PKG = "_unsloth_gg_spoof"


def _cuda_visible_capabilities():
    """The real unsloth.device_type helper, without importing the GPU-only package."""
    source = (ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "cuda_visible_capabilities"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), "device_type.py", "exec"), namespace)
    return namespace["cuda_visible_capabilities"]


class _Kernel:
    def __init__(self, launches, guard_stack):
        self.launches, self.guard_stack, self.best_config = launches, guard_stack, None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launches.append(self.guard_stack[-1] if self.guard_stack else None)
            self.last_kwargs = kwargs
            return types.SimpleNamespace(name = "k", n_regs = 0, n_spills = 0, metadata = {})

        return launch


def _load(
    monkeypatch,
    caps,
    triton_has_tma = True,
):
    """interface.py under a throwaway package, with stubbed unsloth imports and spoofed cards."""
    launches, guard_stack = [], []

    @contextlib.contextmanager
    def torch_gpu_device(device):
        guard_stack.append(device)
        try:
            yield
        finally:
            guard_stack.pop()

    state = {"current": 0}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device = None: caps[state["current"] if device is None else device],
    )

    def get_device_properties(device = None):
        index = state["current"] if device is None else getattr(device, "index", device)
        major, minor = caps[index]
        return types.SimpleNamespace(
            name = f"sm{major}{minor}", major = major, minor = minor, multi_processor_count = 4
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    # FakeTensorMode opens a real context on a fake tensor's device (cuda:1 is absent on one-GPU hosts).
    monkeypatch.setattr(
        "torch._subclasses.fake_tensor.init_gpu_context", lambda device: None, raising = False
    )
    # No custom-op registration from the spoofed module.
    monkeypatch.setattr(torch.library, "custom_op", None, raising = False)

    import triton.language as tl

    for name in ("make_tensor_descriptor", "_experimental_make_tensor_descriptor"):
        if triton_has_tma:
            monkeypatch.setattr(tl, name, object(), raising = False)
        else:
            monkeypatch.delattr(tl, name, raising = False)

    stubs = {
        "unsloth": types.SimpleNamespace(DEVICE_TYPE = "cuda"),
        "unsloth.device_type": types.SimpleNamespace(
            cuda_visible_capabilities = _cuda_visible_capabilities()
        ),
        "unsloth.kernels.utils": types.SimpleNamespace(torch_gpu_device = torch_gpu_device),
        PKG: types.ModuleType(PKG),
        f"{PKG}.kernels": types.ModuleType(f"{PKG}.kernels"),
        f"{PKG}.kernels.backward": types.SimpleNamespace(
            **{
                n: _Kernel(launches, guard_stack)
                for n in (
                    "_autotuned_grouped_gemm_dW_kernel",
                    "_autotuned_grouped_gemm_dW_per_expert_kernel",
                    "_autotuned_grouped_gemm_dX_kernel",
                    "_grouped_gemm_dW_kernel",
                    "_grouped_gemm_dW_per_expert_kernel",
                    "_grouped_gemm_dX_kernel",
                )
            }
        ),
        f"{PKG}.kernels.forward": types.SimpleNamespace(
            _autotuned_grouped_gemm_forward_kernel = _Kernel(launches, guard_stack),
            _grouped_gemm_forward_kernel = _Kernel(launches, guard_stack),
        ),
        f"{PKG}.kernels.tuning": types.SimpleNamespace(
            KernelConfigBackward_dW = object,
            KernelConfigBackward_dX = object,
            KernelConfigForward = object,
        ),
    }
    stubs[PKG].__path__ = [str(GG_DIR)]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(f"{PKG}.interface", GG_DIR / "interface.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._is_tracing = lambda *tensors: False  # fake tensors stand in for real cuda ones
    return module, launches


H100, A100, B200, L4 = (9, 0), (8, 0), (10, 0), (8, 9)


@pytest.mark.parametrize(
    "caps, expected",
    [
        pytest.param([H100, A100], False, id = "h100+a100"),
        pytest.param([A100, H100], False, id = "a100+h100"),
        pytest.param([H100, L4, H100], False, id = "h100+l4+h100"),
        pytest.param([H100, H100], True, id = "h100x2"),
        pytest.param([H100, B200], True, id = "h100+b200"),
        pytest.param([H100], True, id = "h100"),
        pytest.param([A100], False, id = "a100"),
        pytest.param([A100, A100], False, id = "a100x2"),
    ],
)
def test_tma_needs_every_visible_card(monkeypatch, caps, expected):
    module, _ = _load(monkeypatch, caps)
    assert module.supports_tma() is expected


def test_tma_still_needs_the_triton_api(monkeypatch):
    module, _ = _load(monkeypatch, [H100, H100], triton_has_tma = False)
    assert module.supports_tma() is False


def test_an_unreadable_card_disables_tma(monkeypatch):
    module, _ = _load(monkeypatch, [H100, H100])

    def get_device_capability(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return H100

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    assert module._check_tma_support() is False


def test_launches_are_guarded_on_the_operands_device(monkeypatch):
    module, launches = _load(monkeypatch, [H100, A100])
    allocations = []
    monkeypatch.setattr(module.triton, "set_allocator", allocations.append, raising = False)
    monkeypatch.setattr(module, "_HAS_SET_ALLOCATOR", True)
    E, T, N, K = 2, 8, 16, 16
    with FakeTensorMode():
        dev = torch.device("cuda", 1)
        X = torch.empty(T, K, device = dev, dtype = torch.bfloat16)
        W = torch.empty(E, N, K, device = dev, dtype = torch.bfloat16)
        dY = torch.empty(T, N, device = dev, dtype = torch.bfloat16)
        m_sizes = torch.empty(E, device = dev, dtype = torch.int32)
        common = dict(m_sizes = m_sizes, gather_indices = None, topk = 1, autotune = True)
        module._grouped_gemm_forward_impl(X = X, W = W, **common)
        module._grouped_gemm_dX_impl(dY = dY, W = W, **common)
        module._grouped_gemm_dW_impl(X = X, dY = dY, **common)
        scratch = [alloc(128, 128, 0).device for alloc in allocations]
    assert launches == [dev, dev, dev]
    # set_allocator is process-wide: follow the current device, never capture one launch's card.
    assert scratch and all(d.type == "cuda" and d.index in (None, 0) for d in scratch)


class _Tuner:
    def __init__(self):
        self.cache = {}

    def run(self, *args, **kwargs):
        return self.cache


def test_each_card_autotunes_its_own_copy(monkeypatch):
    module, _ = _load(monkeypatch, [H100, A100])
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 2)
    base = _Tuner()
    base.cache["shape"] = "h100-config"
    tuner0 = module._autotuner_for(base, torch.device("cuda", 0))
    tuner1 = module._autotuner_for(base, torch.device("cuda", 1))
    assert tuner0 is not tuner1 and tuner0.cache == {} and tuner1.cache == {}
    assert module._autotuner_for(base, torch.device("cuda", 1)) is tuner1
    assert base.cache == {"shape": "h100-config"}


def test_identical_cards_share_one_autotune_cache(monkeypatch):
    # device_map="auto" over identical cards: tune once, as with one shared tuner. Each card keeps
    # its own tuner object: Triton writes per-run state (nargs) on it, and per-device autograd
    # threads can run two cards' kernels at once.
    module, _ = _load(monkeypatch, [H100, H100, A100])
    base = _Tuner()
    tuner0 = module._autotuner_for(base, torch.device("cuda", 0))
    tuner1 = module._autotuner_for(base, torch.device("cuda", 1))
    tuner2 = module._autotuner_for(base, torch.device("cuda", 2))
    assert tuner1 is not tuner0 and tuner1.cache is tuner0.cache
    assert tuner2.cache is not tuner0.cache
    tuner0.cache["shape"] = "h100-config"
    assert tuner1.cache == {"shape": "h100-config"} and tuner2.cache == {}
    assert module._autotuner_for(base, torch.device("cuda", 1)) is tuner1


def test_per_card_autotuner_follows_reassigned_configs(monkeypatch):
    module, _ = _load(monkeypatch, [H100, A100])
    base = _Tuner()
    base.configs = ["a", "b", "c"]
    tuner = module._autotuner_for(base, torch.device("cuda", 1))
    base.configs = base.configs[:1]
    assert module._autotuner_for(base, torch.device("cuda", 1)).configs == ["a"]
    assert tuner.cache == {}


def test_single_card_keeps_the_shared_autotuner(monkeypatch):
    module, _ = _load(monkeypatch, [H100])
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    base = _Tuner()
    assert module._autotuner_for(base, torch.device("cuda", 0)) is base


def test_device_properties_are_read_per_card(monkeypatch):
    import ast
    from dataclasses import dataclass

    source = (GG_DIR / "kernels" / "tuning.py").read_text(encoding = "utf-8")
    wanted = {"DeviceProperties", "_DEVICE_PROPERTIES", "get_device_properties"}
    nodes = [
        node
        for node in ast.parse(source).body
        if getattr(node, "name", None) in wanted
        or any(
            getattr(t, "id", None) in wanted
            for t in getattr(node, "targets", []) + [getattr(node, "target", None)]
        )
    ]
    smem = {0: 232448, 1: 166912}
    fake_utils = types.SimpleNamespace(
        get_device_properties = lambda index: {
            "multiprocessor_count": 132,
            "max_num_regs": 65536,
            "max_shared_mem": smem[index],
            "warpSize": 32,
        }
    )
    fake_triton = types.SimpleNamespace(
        runtime = types.SimpleNamespace(
            driver = types.SimpleNamespace(active = types.SimpleNamespace(utils = fake_utils))
        )
    )
    namespace = {
        "torch": torch,
        "triton": fake_triton,
        "dataclass": dataclass,
        "Optional": typing.Optional,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), "tuning.py", "exec"), namespace)
    tuning = types.SimpleNamespace(**namespace)
    current = {"device": 0}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["device"])
    assert tuning.get_device_properties().SIZE_SMEM == smem[0]
    current["device"] = 1
    assert tuning.get_device_properties().SIZE_SMEM == smem[1]


def test_autotune_cache_reads_the_per_card_tuners_config():
    # The selected config lives on the per-card copy on multi-GPU hosts; reading the shared kernel misses it.
    import ast

    tree = ast.parse((GG_DIR.parent / "autotune_cache.py").read_text(encoding = "utf-8"))
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "best_config"]
    assert reads and all(
        isinstance(n.value, ast.Call) and getattr(n.value.func, "id", None) == "_autotuner_for"
        for n in reads
    )


def test_benchmark_postprocesses_the_per_card_tuner():
    # postprocess_autotune_results reads .cache; on several GPUs only the per-card copy has it.
    import ast

    tree = ast.parse((GG_DIR.parent / "benchmark" / "utils.py").read_text(encoding = "utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_autotuner")
    kernels = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and n.id.startswith("_autotuned_grouped_gemm_")
    ]
    wrapped = {
        id(a)
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_autotuner_for"
        for a in c.args[:1]
    }
    assert kernels and all(id(n) in wrapped for n in kernels)


def test_identical_cards_launch_concurrently(monkeypatch):
    # Per-device autograd threads launch on two identical cards at once: a cached launch on one card
    # must not wait on the other card's launch or tuning (no model-wide lock around run).
    import threading

    module, _ = _load(monkeypatch, [H100, H100])
    barrier = threading.Barrier(2, timeout = 5)
    finished = []

    class Tuner(_Tuner):
        def run(self, key = "shape"):
            barrier.wait()  # both cards inside run at the same time, or BrokenBarrierError
            finished.append(self.cache.get(key))
            return self.cache.get(key)

    base = Tuner()
    tuners = [module._autotuner_for(base, torch.device("cuda", i)) for i in (0, 1)]
    tuners[0].cache["shape"] = "h100-config"  # warm: shared by both identical cards
    errors = []

    def launch(tuner):
        try:
            tuner.run()
        except threading.BrokenBarrierError as e:
            errors.append(e)

    threads = [threading.Thread(target = launch, args = (t,)) for t in tuners]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors and finished == ["h100-config", "h100-config"]


def test_sm_count_is_read_once_per_card(monkeypatch):
    module, _ = _load(monkeypatch, [H100, A100])
    reads = []
    real = module.torch.cuda.get_device_properties

    def counting(device = None):
        reads.append(device)
        return real(device)

    monkeypatch.setattr(module.torch.cuda, "get_device_properties", counting)
    for _ in range(3):
        assert module._num_sms(torch.device("cuda", 1)) == 4
        assert module._num_sms(torch.device("cuda", 0)) == 4
    assert sorted(reads) == [0, 1]


def _real_autotune_cache():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_ac_spoof", ROOT / "unsloth" / "kernels" / "moe" / "autotune_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMEM = {H100: 232448, A100: 166912, L4: 101376}
TILES = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_stages")


def _launch_tiles(
    monkeypatch,
    caps,
    operand_index,
    M,
    N,
    K,
    stages,
    dtype = torch.bfloat16,
):
    """Run the three non-autotuned impls on cuda:operand_index (current: cuda:0); return each launch's tiles."""
    module, _ = _load(monkeypatch, caps)
    real = torch.cuda.get_device_properties

    def props(device = None):
        p = real(device)
        index = 0 if device is None else getattr(device, "index", device)
        p.shared_memory_per_block_optin = SMEM[caps[index]]
        return p

    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device = None: caps[0 if device is None else getattr(device, "index", device)],
    )
    ac = _real_autotune_cache()
    monkeypatch.setattr(module, "_autotune_cache", lambda: ac)
    E, T, Nd, Kd = 2, 8, 512, 512
    tiles = dict(BLOCK_SIZE_M = M, BLOCK_SIZE_N = N, BLOCK_SIZE_K = K, num_stages = stages, num_warps = 8)
    out = {}
    with FakeTensorMode():
        dev = torch.device("cuda", operand_index)
        X = torch.empty(T, Kd, device = dev, dtype = dtype)
        W = torch.empty(E, Nd, Kd, device = dev, dtype = torch.bfloat16)
        dY = torch.empty(T, Nd, device = dev, dtype = dtype)
        m_sizes = torch.empty(E, device = dev, dtype = torch.int32)
        common = dict(m_sizes = m_sizes, gather_indices = None, topk = 1, autotune = False, **tiles)
        kernels = sys.modules[f"{PKG}.kernels.forward"], sys.modules[f"{PKG}.kernels.backward"]
        module._grouped_gemm_forward_impl(X = X, W = W, **common)
        out["fwd"] = kernels[0]._grouped_gemm_forward_kernel.last_kwargs
        module._grouped_gemm_dX_impl(dY = dY, W = W, **common)
        out["dX"] = kernels[1]._grouped_gemm_dX_kernel.last_kwargs
        module._grouped_gemm_dW_impl(X = X, dY = dY, **common)
        per_expert = module._use_per_expert_dW_kernel()
        dw_kernel = (
            kernels[1]._grouped_gemm_dW_per_expert_kernel
            if per_expert
            else kernels[1]._grouped_gemm_dW_kernel
        )
        out["dW"] = dw_kernel.last_kwargs
    return ac, {k: tuple(v[t] for t in TILES) for k, v in out.items()}


def test_caller_tiles_fitted_for_the_current_card_are_refit_for_the_operand_card(monkeypatch):
    # Released unsloth_zoo passes no device: an H100-fitted bf16 dX (64, 64, 256, 4) needs 122,880 B,
    # over the L4's 101,376 B. The launch on the L4 operand shrinks it to fit; the H100 copy is untouched.
    ac, got = _launch_tiles(monkeypatch, [H100, L4], 1, 64, 64, 256, 4)
    assert (
        ac._estimate_smem(
            "dX", types.SimpleNamespace(**dict(zip(TILES, (64, 64, 256, 4)))), 2, 2, L4
        )
        > SMEM[L4]
    )
    assert got["dX"] == (64, 64, 256, 3)
    for kind, tiles in got.items():
        config = types.SimpleNamespace(**dict(zip(TILES, tiles)))
        assert ac._estimate_smem(kind, config, 2, 2, L4) <= SMEM[L4]


@pytest.mark.parametrize(
    "caps, operand_index",
    [
        pytest.param([L4], 0, id = "one-gpu"),
        pytest.param([H100, H100], 1, id = "same-model"),
        pytest.param([H100, L4], 0, id = "operand-on-current"),
    ],
)
def test_tiles_unchanged_on_one_gpu_or_the_same_model(monkeypatch, caps, operand_index):
    _, got = _launch_tiles(monkeypatch, caps, operand_index, 64, 64, 256, 4)
    assert got == {"fwd": (64, 64, 256, 4), "dX": (64, 64, 256, 4), "dW": (64, 64, 256, 4)}
