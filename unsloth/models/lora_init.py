# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import contextlib
import os
import torch

__all__ = [
    "randomized_svd",
    "mica_basis",
    "fast_lora_init",
]

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)


@contextlib.contextmanager
def _tf32():
    flags = torch.backends.cuda.matmul
    old = flags.allow_tf32
    flags.allow_tf32 = True
    try:
        yield
    finally:
        flags.allow_tf32 = old


def _cholqr_(Y, shift = True):
    # Y <- Y R^-1 in place, R = chol(Y^T Y) in fp64; the shift keeps rank-deficient sketches factorizable.
    G = Y.mT.double() @ Y.double()
    if shift:
        G.diagonal().add_(G.diagonal().sum() * 1e-14 + 1e-30)
    R, info = torch.linalg.cholesky_ex(G, upper = True)
    if not shift and info.item() != 0:
        return False
    torch.linalg.solve_triangular(R.to(Y.dtype), Y, upper = True, left = False, out = Y)
    return True


@torch.no_grad()
def randomized_svd(
    W,
    rank,
    n_oversamples = None,
    n_iter = 6,
    generator = None,
):
    """Truncated W ~= U diag(S) Vh by randomized subspace iteration (Halko et al. 2011, Alg. 4.4 + 5.1).

    Returns fp32 U[:, :rank], S[:rank], Vh[:rank]. Oversampling defaults to max(rank, 10): LLM spectra decay
    slowly, so a wider sketch buys more accuracy per ms than iterations. CholeskyQR replaces cuSOLVER QR
    (geqrf + ormqr was no faster: ormqr applies the full m x m Q).
    """
    m, n = W.shape
    A = W if m >= n else W.mT
    if A.dtype != torch.float32:
        A = A.float()
    M, N = A.shape
    rank = min(rank, N)
    q = min(rank + (max(rank, 10) if n_oversamples is None else n_oversamples), N)
    Z = torch.randn(N, q, device = A.device, dtype = A.dtype, generator = generator)

    def _power(k):
        for _ in range(k):
            _cholqr_(Y)
            torch.matmul(A.mT, Y, out = Z)
            _cholqr_(Z)
            torch.matmul(A, Z, out = Y)

    with _tf32() if (n_iter > 1 and A.is_cuda) else contextlib.nullcontext():
        Y = torch.matmul(A, Z)
        _power(n_iter - 1)
    # Last iteration in fp32 removes the TF32 error when sigma_1 / sigma_rank is large.
    _power(min(n_iter, 1))
    _cholqr_(Y)
    if not _cholqr_(Y, shift = False):
        Y = torch.linalg.qr(Y).Q
    torch.matmul(A.mT, Y, out = Z)
    del A
    Zd = Z.double()
    del Z
    evals, E = torch.linalg.eigh(Zd.mT @ Zd)
    E = E[:, -rank:].flip(1)
    S = evals[-rank:].flip(0).clamp_min_(0).sqrt_()
    V = (Zd @ E).div_(S.clamp_min(torch.finfo(torch.float64).tiny))
    U = Y @ E.float()
    S, V = S.float(), V.float()
    if m < n:
        return V, S, U.mT
    return U, S, V.mT


def _fast_fp64(device):
    if device.type == "cpu":
        return True
    if device.type != "cuda":
        return False
    if getattr(torch.version, "hip", None):
        return "gfx9" in getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    # Full-rate fp64: P100, V100, A100, H100, B200. B300 (10.3) and consumer / L4 / T4 parts run fp64 at 1/64.
    return torch.cuda.get_device_capability(device) in ((6, 0), (7, 0), (8, 0), (9, 0), (10, 0))


@torch.no_grad()
def mica_basis(W, r):
    """U[:, -r:] of the reduced SVD of W (out x in) as fp32, via fp64 eigh of the smaller Gram matrix.

    fp64 Gram squaring costs ~1e-16 * cond^2, below fp32 SVD's own error on LLM weights. lobpcg does not
    converge here: the relative gaps between the smallest singular values are ~1e-5..1e-3.
    """
    out_features, in_features = W.shape
    A = W.to(torch.float64)
    wide = out_features <= in_features
    L, V = torch.linalg.eigh(A @ A.T if wide else A.T @ A)
    if L[r - 1] <= 1e-12 * L[-1]:
        U = torch.linalg.svd(A, full_matrices = False)[0]
        return U[:, -r:].float().contiguous()
    # eigh is ascending; flip so columns match svd's U[:, -r:] order.
    V = V[:, :r].flip(1)
    if wide:
        return V.float().contiguous()
    U = A @ V
    U /= L[:r].flip(0).sqrt()
    return U.float().contiguous()


def _pissa_init(self, adapter_name, init_lora_weights):
    from peft.tuners.lora.layer import transpose

    weight = self.get_base_layer().weight
    dtype = weight.dtype
    r = self.r[adapter_name]
    if dtype not in _FLOATS or r > min(weight.shape) or weight.device.type == "meta":
        return _ORIGINAL["pissa_init"](self, adapter_name, init_lora_weights)
    if init_lora_weights == "pissa":
        n_iter = 6
    else:
        parts = init_lora_weights.split("_niter_")
        if len(parts) != 2:
            return _ORIGINAL["pissa_init"](self, adapter_name, init_lora_weights)
        n_iter = int(parts[-1])
    W = transpose(weight.to(torch.float32), self.fan_in_fan_out)
    U, S, Vh = randomized_svd(W, r, n_iter = n_iter)
    scaling = self.scaling[adapter_name]
    S.div_(scaling).sqrt_()
    lora_A = Vh.mul_(S.unsqueeze(1)).contiguous()
    lora_B = U.mul_(S).contiguous()
    self.lora_A[adapter_name].weight.data = lora_A
    self.lora_B[adapter_name].weight.data = lora_B
    W = W if W.data_ptr() != weight.data_ptr() else W.clone()
    W.addmm_(lora_B, lora_A, alpha = -scaling)
    self.get_base_layer().weight.data = transpose(W.to(dtype), self.fan_in_fan_out)


def _mica_init(self, adapter_name):
    from peft.tuners.lora.layer import transpose

    weight = self.get_base_layer().weight
    r = self.r[adapter_name]
    if (
        self.lora_B[adapter_name].weight.device.type == "meta"
        or weight.dtype not in _FLOATS
        or r > min(weight.shape)
        or not _fast_fp64(weight.device)
    ):
        return _ORIGINAL["mica_init"](self, adapter_name)
    dtype = weight.dtype
    W = transpose(weight, self.fan_in_fan_out)
    self.lora_B[adapter_name].weight.data = mica_basis(W, r).to(dtype)
    self.lora_A[adapter_name].weight.data = torch.zeros(
        r, W.shape[1], device = weight.device, dtype = dtype
    )


_ORIGINAL = {}


@contextlib.contextmanager
def fast_lora_init():
    """Swap PEFT's LoraLayer.pissa_init / mica_init for the fast versions during get_peft_model.

    UNSLOTH_FAST_LORA_INIT=0 keeps PEFT's own SVD.
    """
    if os.environ.get("UNSLOTH_FAST_LORA_INIT", "1") == "0" or _ORIGINAL:
        # Nested use keeps the outer swap.
        yield
        return
    from peft.tuners.lora.layer import LoraLayer

    swapped = []
    for name, fn in (("pissa_init", _pissa_init), ("mica_init", _mica_init)):
        original = LoraLayer.__dict__.get(name)
        if original is None:
            continue
        _ORIGINAL[name] = original
        setattr(LoraLayer, name, fn)
        swapped.append((name, original))
    try:
        yield
    finally:
        for name, original in swapped:
            setattr(LoraLayer, name, original)
        _ORIGINAL.clear()


# Data-driven inits calibrate with forward hooks (EVA, CorDA) or backward passes (LoRA-GA). Dynamo does not
# guard on hooks added after compilation (skip_nnmodule_hook_guards), so a module compiled by an earlier
# forward silently skips them: CorDA then divides by a zero sample count.
_CALIBRATION_FUNCTIONS = (
    ("peft.tuners.lora.corda", "preprocess_corda"),
    ("peft.tuners.lora.eva", "initialize_lora_eva_weights"),
    ("peft.tuners.lora.loraga", "preprocess_loraga"),
)


def patch_peft_calibration_eager():
    import functools
    import importlib
    import sys

    if not hasattr(torch.compiler, "set_stance"):
        return
    for module_name, name in _CALIBRATION_FUNCTIONS:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        original = getattr(module, name, None)
        if original is None or getattr(original, "_unsloth_eager", False):
            continue

        @functools.wraps(original)
        def eager(
            *args,
            __original = original,
            **kwargs,
        ):
            with torch.compiler.set_stance("force_eager"):
                return __original(*args, **kwargs)

        eager._unsloth_eager = True
        for loaded in list(sys.modules.values()):
            if loaded is not None and getattr(loaded, name, None) is original:
                setattr(loaded, name, eager)
