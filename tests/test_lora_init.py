# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

_spec = importlib.util.spec_from_file_location(
    "unsloth_lora_init", Path(__file__).resolve().parents[1] / "unsloth" / "models" / "lora_init.py"
)
lora_init = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lora_init)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _weight(
    out_features,
    in_features,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    k = min(out_features, in_features)
    U = torch.linalg.qr(torch.randn(out_features, k, generator = g, dtype = torch.float64)).Q
    V = torch.linalg.qr(torch.randn(in_features, k, generator = g, dtype = torch.float64)).Q
    S = torch.logspace(0, -2, k, dtype = torch.float64) + 0.01 * torch.rand(
        k, generator = g, dtype = torch.float64
    )
    return (U * S) @ V.T


@pytest.mark.parametrize("shape", [(96, 48), (48, 96), (64, 64)])
@pytest.mark.parametrize("rank", [1, 8])
def test_randomized_svd_near_optimal(shape, rank):
    W = _weight(*shape)
    U, S, Vh = lora_init.randomized_svd(W.float().to(DEVICE), rank)
    Ue, Se, Vhe = torch.linalg.svd(W, full_matrices = False)
    best = (W - (Ue[:, :rank] * Se[:rank]) @ Vhe[:rank]).norm()
    err = (W - (U.double().cpu() * S.double().cpu()) @ Vh.double().cpu()).norm()
    assert err / best < 1.001
    assert torch.allclose(S.double().cpu(), Se[:rank], rtol = 1e-3)
    assert U.dtype == torch.float32 and U.shape == (shape[0], rank) and Vh.shape == (rank, shape[1])


def _adversarial():
    g = torch.Generator().manual_seed(1)
    return {
        "zero": torch.zeros(64, 40),
        "rank1": torch.randn(64, 1, generator = g) @ torch.randn(1, 40, generator = g),
        "duplicate_columns": torch.randn(64, 4, generator = g).repeat(1, 10),
        # Column norms above ~1e19 overflow a plain fp32 vector_norm.
        "column_scales": torch.randn(64, 40, generator = g) * torch.logspace(-20, 20, 40),
        "row_scales": torch.randn(64, 40, generator = g) * torch.logspace(-20, 20, 64)[:, None],
        "one_huge_entry": torch.randn(64, 40, generator = g).index_put_(
            (torch.tensor([3]), torch.tensor([7])), torch.tensor(1e30)
        ),
    }


@pytest.mark.parametrize("name", list(_adversarial()))
@pytest.mark.parametrize("rank", [4, 16])
def test_randomized_svd_adversarial_finite_and_near_optimal(name, rank):
    W = _adversarial()[name]
    U, S, Vh = lora_init.randomized_svd(W.to(DEVICE), rank)
    assert torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()
    Wd = W.double()
    S0 = torch.linalg.svdvals(Wd)
    err = (Wd - (U.double().cpu() * S.double().cpu()) @ Vh.double().cpu()).norm()
    best = (S0[rank:] ** 2).sum().sqrt()
    assert err <= best + 1e-4 * Wd.norm() + 1e-30
    assert (S.double().cpu() - S0[:rank]).abs().max() <= 1e-4 * S0[0] + 1e-30


@pytest.mark.parametrize("true_rank", [0, 1, 8])
def test_randomized_svd_rank_deficient_wide_sketch(true_rank):
    # q = 128 takes the eigh path, which cuSOLVER refuses on the NaNs a CholeskyQR breakdown leaves.
    g = torch.Generator().manual_seed(2)
    W = torch.randn(512, true_rank, generator = g) @ torch.randn(true_rank, 384, generator = g)
    U, S, Vh = lora_init.randomized_svd(W.to(DEVICE), 64)
    assert torch.isfinite(U).all() and torch.isfinite(S).all() and torch.isfinite(Vh).all()
    err = (W.double() - (U.double().cpu() * S.double().cpu()) @ Vh.double().cpu()).norm()
    assert err <= 1e-5 * W.norm() + 1e-30


@pytest.mark.parametrize("name", ["column_scales", "row_scales", "one_huge_entry"])
@pytest.mark.parametrize("rank", [4, 16])
def test_randomized_svd_survives_unscaled_solver_norms(monkeypatch, name, rank):
    # rocSOLVER's geqrf / gesvd return NaN where a squared column norm overflows fp32 (gfx1151).
    qr, svd = torch.linalg.qr, torch.linalg.svd

    def overflows(X):
        return X.dtype == torch.float32 and not torch.isfinite(X.pow(2).sum(-2)).all()

    def unscaled_qr(X, *args, **kwargs):
        out = qr(X, *args, **kwargs)
        return torch.return_types.linalg_qr((out.Q * float("nan"), out.R)) if overflows(X) else out

    def unscaled_svd(X, *args, **kwargs):
        return svd(X * float("nan") if overflows(X) else X, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", unscaled_qr)
    monkeypatch.setattr(torch.linalg, "svd", unscaled_svd)
    monkeypatch.setattr(torch.version, "hip", "7.0", raising = False)
    test_randomized_svd_adversarial_finite_and_near_optimal(name, rank)


def test_randomized_svd_is_fp32_only(monkeypatch):
    # Nothing may run in float64: consumer GPUs run it at 1/64 rate.
    calls = []
    original = torch.Tensor.double
    monkeypatch.setattr(torch.Tensor, "double", lambda self: calls.append(1) or original(self))
    U, S, Vh = lora_init.randomized_svd(_weight(96, 48).float().to(DEVICE), 8)
    assert not calls and U.dtype == S.dtype == Vh.dtype == torch.float32


@pytest.mark.parametrize("shape", [(96, 48), (48, 96), (64, 64)])
def test_mica_basis_matches_fp64_svd(shape):
    W = _weight(*shape)
    r = 8
    B = lora_init.mica_basis(W.float(), r)
    U = torch.linalg.svd(W, full_matrices = False)[0][:, -r:]
    assert torch.linalg.matrix_norm(U @ U.T - B.double() @ B.double().T, ord = 2) < 1e-4
    assert torch.nn.functional.cosine_similarity(B.double(), U, dim = 0).abs().min() > 0.999


@pytest.mark.parametrize("shape", [(96, 48), (48, 96)])
def test_mica_basis_rank_deficient_is_finite(shape):
    # One dead input / output dimension: a single zero eigenvalue, below the r smallest's largest.
    W = _weight(*shape)
    W[:, 0] = 0
    W[0] = 0
    B = lora_init.mica_basis(W.float(), 8).double()
    assert torch.isfinite(B).all()
    torch.testing.assert_close(B.T @ B, torch.eye(8, dtype = B.dtype), atol = 1e-4, rtol = 0)


class _Model(torch.nn.Module):
    def __init__(self, W):
        super().__init__()
        self.q = torch.nn.Linear(W.shape[1], W.shape[0], bias = False)
        self.q.weight.data = W.float()

    def forward(self, x):
        return self.q(x)


@pytest.mark.parametrize("init", ["pissa", "pissa_niter_4"])
def test_fast_pissa_preserves_output(init):
    W = _weight(96, 64)
    base = _Model(W).to(DEVICE)
    x = torch.randn(3, 64, device = DEVICE)
    with lora_init.fast_lora_init():
        assert LoraLayer.pissa_init is lora_init._pissa_init
        model = get_peft_model(
            copy.deepcopy(base),
            LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q"], init_lora_weights = init),
        )
    assert LoraLayer.pissa_init is not lora_init._pissa_init
    assert torch.allclose(model(x), base(x), atol = 1e-5)
    layer = model.base_model.model.q
    BA = (
        layer.scaling["default"]
        * layer.lora_B["default"].weight.double().cpu()
        @ layer.lora_A["default"].weight.double().cpu()
    )
    Ue, Se, Vhe = torch.linalg.svd(W, full_matrices = False)
    # "pissa" stands in for an exact SVD; pissa_niter_N keeps PEFT's svd_lowrank accuracy class.
    bound = 1.001 if init == "pissa" else 1.05
    assert (W - BA).norm() / (W - (Ue[:, :8] * Se[:8]) @ Vhe[:8]).norm() < bound


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FAST_LORA_INIT", "0")
    original = LoraLayer.pissa_init
    # An earlier fast run must not make a kill-switched one report the fast path.
    monkeypatch.setitem(lora_init._STATE, "pissa", True)
    with lora_init.fast_lora_init() as fast:
        assert LoraLayer.pissa_init is original
    assert not fast["pissa"]


def test_fast_pissa_marker_lookup_fails_closed_only_when_online(monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    def raising(error):
        def download(*args, **kwargs):
            raise error

        return download

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: "cached")
    assert lora_init.adapter_used_fast_pissa("user/adapter")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", raising(EntryNotFoundError("404")))
    assert not lora_init.adapter_used_fast_pissa("user/adapter")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", raising(LocalEntryNotFoundError("down"))
    )
    with pytest.raises(LocalEntryNotFoundError):
        lora_init.adapter_used_fast_pissa("user/adapter")
    assert not lora_init.adapter_used_fast_pissa("user/adapter", local_files_only = True)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    assert not lora_init.adapter_used_fast_pissa("user/adapter")


def test_calibration_patch_does_not_trigger_lazy_module_getattr(monkeypatch):
    # transformers' lazy module imports submodules (torchvision for aria) on any getattr.
    import sys
    import types

    asked = []
    lazy = types.ModuleType("unsloth_test_lazy_module")

    def __getattr__(name):
        asked.append(name)
        raise AttributeError(name)

    lazy.__getattr__ = __getattr__
    monkeypatch.setitem(sys.modules, lazy.__name__, lazy)
    names = [name for _, name in lora_init._calibration_functions()]
    for module, name in lora_init._calibration_functions():
        fn = getattr(module, name)
        monkeypatch.setattr(module, name, getattr(fn, "__wrapped__", fn))
    lora_init.patch_peft_calibration_eager()
    assert names and not set(asked) & set(names)


def test_fast_init_kill_switch_and_force(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FAST_LORA_INIT", "0")
    original = LoraLayer.pissa_init
    with lora_init.fast_lora_init():
        assert LoraLayer.pissa_init is original
    with lora_init.fast_lora_init(force = True):
        assert LoraLayer.pissa_init is lora_init._pissa_init
        with lora_init.fast_lora_init():
            assert LoraLayer.pissa_init is lora_init._pissa_init
    assert LoraLayer.pissa_init is original


def test_fast_pissa_saves_record_their_algorithm(tmp_path):
    from peft import LoraConfig, get_peft_model

    def config():
        return LoraConfig(r = 4, target_modules = ["0"], init_lora_weights = "pissa")

    base = torch.nn.Sequential(torch.nn.Linear(64, 48, bias = False))
    assert not lora_init.adapter_used_fast_pissa(str(tmp_path))
    with lora_init.fast_lora_init() as fast:
        model = get_peft_model(base, config())
    assert fast["pissa"]
    lora_init.record_fast_pissa(model)
    lora_init.record_fast_pissa(model)
    # A second PiSSA adapter through plain PEFT must keep PEFT's initializer on reload.
    model.add_adapter("plain", config())
    model.save_pretrained(str(tmp_path))
    assert (tmp_path / "plain" / "adapter_config.json").is_file()
    assert lora_init.adapter_used_fast_pissa(str(tmp_path))
    assert not lora_init.adapter_used_fast_pissa(str(tmp_path / "plain"))


def test_sketch_ignores_default_dtype():
    previous = torch.get_default_dtype()
    lora_init._sketch.cache_clear()
    try:
        torch.set_default_dtype(torch.float64)
        U, S, Vh = lora_init.randomized_svd(_weight(96, 48).float(), 8)
    finally:
        torch.set_default_dtype(previous)
        lora_init._sketch.cache_clear()
    assert U.dtype == S.dtype == Vh.dtype == torch.float32


def test_other_threads_keep_peft_init(monkeypatch):
    import threading

    calls = []
    with lora_init.fast_lora_init(force = True):
        monkeypatch.setitem(lora_init._ORIGINAL_ANY, "pissa_init", lambda *args: calls.append(args))
        thread = threading.Thread(target = lora_init._pissa_init, args = (None, "a", "pissa"))
        thread.start()
        thread.join()
    assert calls == [(None, "a", "pissa")]


def test_packed_quantized_base_is_refused():
    layer = torch.nn.Linear(16, 16)
    Params4bit = type("Params4bit", (torch.nn.Parameter,), {})
    layer.weight = Params4bit(layer.weight.data.to(torch.bfloat16), requires_grad = False)
    with pytest.raises(TypeError, match = "load_in_4bit = False"):
        lora_init._check_float_weight(layer.weight, "pissa")


def test_pissa_niter_width_is_device_independent(monkeypatch):
    # Loading re-runs the init, possibly on another device: the sketch width must not depend on it.
    seen = []
    real = lora_init.randomized_svd

    def spy(W, rank, **kwargs):
        seen.append(kwargs["n_oversamples"])
        return real(W, rank, **kwargs)

    monkeypatch.setattr(lora_init, "randomized_svd", spy)
    model = torch.nn.Sequential(torch.nn.Linear(160, 160))
    with lora_init.fast_lora_init(force = True):
        get_peft_model(
            model, LoraConfig(r = 64, target_modules = ["0"], init_lora_weights = "pissa_niter_4")
        )
    assert seen == [0]


def test_swap_state_is_per_call():
    with lora_init.fast_lora_init(force = True) as first:
        first["pissa"] = True
    with lora_init.fast_lora_init(force = True) as second:
        assert not second["pissa"]
    # A caller reading its flag after the lock is released is unaffected by later swaps.
    assert first["pissa"]


@pytest.mark.parametrize("alpha", [0, 16])
def test_merge_conversion_matches_the_live_adapter(alpha):
    from peft import LoraConfig, get_peft_model
    from unsloth.models._utils import lora_relative_to_original_base, snapshot_residual_lora_init

    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 48, bias = False)
    W = linear.weight.detach().clone()
    config = LoraConfig(r = 8, lora_alpha = alpha, target_modules = ["0"], init_lora_weights = "olora")
    model = get_peft_model(torch.nn.Sequential(linear), config)
    snapshot_residual_lora_init(model, "olora")
    layer = model.base_model.model[0]
    with torch.no_grad():
        layer.lora_B["default"].weight.add_(0.05 * torch.randn_like(layer.lora_B["default"].weight))
    x = torch.randn(5, 64)
    live = model(x)
    with lora_relative_to_original_base(model):
        s = layer.scaling["default"]
        merged = W + s * layer.lora_B["default"].weight @ layer.lora_A["default"].weight
    assert torch.isfinite(merged).all()
    torch.testing.assert_close(x @ merged.T, live, atol = 1e-4, rtol = 1e-4)
