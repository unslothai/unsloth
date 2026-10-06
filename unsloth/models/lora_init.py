# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import contextlib
import functools
import json
import os
import threading
import torch

__all__ = [
    "randomized_svd",
    "mica_basis",
    "fast_lora_init",
]

_FLOATS = (torch.float32, torch.float16, torch.bfloat16)


@contextlib.contextmanager
def _tf32(enabled):
    flags = torch.backends.cuda.matmul
    old = flags.allow_tf32
    flags.allow_tf32 = enabled
    try:
        yield
    finally:
        flags.allow_tf32 = old


_EPS32 = torch.finfo(torch.float32).eps / 2  # unit roundoff u


def _orthonormalize_(Y, shift, failures, householder):
    """Y <- Y R^-1 in place, fp32 shifted CholeskyQR (Fukaya et al. 2020); no host sync.

    Columns are scaled to unit norm first (Gram diagonal = 1, no overflow). The shift is Rump-Ogita's
    breakdown bound 2.2 (n + 1) u tr(G) plus a sqrt(m) u n allowance for the Gram's rounding error.
    Breakdown flags are appended to `failures` and checked once per call; any breakdown reruns the
    whole call with Householder QR, which cannot fail on finite input.
    """
    if householder:
        Y.copy_(torch.linalg.qr(Y).Q)
        return
    m, n = Y.shape
    tiny = torch.finfo(torch.float32).tiny
    # max-abs first: vector_norm squares entries and overflows fp32 above ~1e19.
    Y.div_(Y.abs().amax(0).clamp_min_(tiny))
    Y.div_(torch.linalg.vector_norm(Y, dim = 0).clamp_min_(tiny))
    with _tf32(False):
        G = Y.mT @ Y
    if shift:
        G.diagonal().add_((2.2 * (n + 1) * n + 11.0 * (m**0.5 * n + n * (n + 1))) * _EPS32)
    R, info = torch.linalg.cholesky_ex(G, upper = True)
    failures.append(info)
    torch.linalg.solve_triangular(R, Y, upper = True, left = False, out = Y)


@functools.lru_cache(maxsize = 8)
def _sketch(N, q):
    # One seeded CPU draw per shape: a reloaded adapter re-runs this init (PEFT's from_pretrained) and must
    # rebuild the same residual on any device; LLM spectra decay too slowly for the sketch not to matter.
    return torch.randn(N, q, dtype = torch.float32, generator = torch.Generator().manual_seed(3407))


@torch.no_grad()
def randomized_svd(
    W,
    rank,
    n_oversamples = None,
    n_iter = 6,
    final_passes = 1,
    generator = None,
    _safe = False,
):
    """Truncated W ~= U diag(S) Vh by randomized subspace iteration (Halko et al. 2011, Alg. 4.4 + 5.1).

    fp32 throughout, one host sync per call. Oversampling defaults to max(rank, 10): LLM spectra decay
    slowly, so a wider sketch buys more accuracy per ms than iterations. Shifted CholeskyQR replaces
    cuSOLVER QR for sketches 32+ wide (geqrf + ormqr was no faster: ormqr applies the full m x m Q);
    the final basis gets `final_passes` unshifted passes after the shifted one.
    """
    m, n = W.shape
    A = W if m >= n else W.mT
    A = A.float()
    scale = None
    if _safe:
        # rocSOLVER's geqrf / gesvd square entries without LAPACK's scaling and overflow above ~1e19.
        scale = A.abs().amax().clamp_min_(torch.finfo(torch.float32).tiny)
        A = A / scale
    M, N = A.shape
    rank = min(rank, N)
    q = min(rank + (max(rank, 10) if n_oversamples is None else n_oversamples), N)
    if generator is None:
        Z = _sketch(N, q).to(A.device, A.dtype, copy = True)
    else:
        Z = torch.randn(N, q, device = A.device, dtype = A.dtype, generator = generator)
    failures = []
    # Below q = 32 one Householder QR beats CholeskyQR's ~8 launches.
    householder = _safe or q < 32

    def _power(k):
        # Both half steps: skipping one squares cond(Y) (~2e3 on real r = 128 weights), past fp32.
        for _ in range(k):
            _orthonormalize_(Y, True, failures, householder)
            torch.matmul(A.mT, Y, out = Z)
            _orthonormalize_(Z, True, failures, householder)
            torch.matmul(A, Z, out = Y)

    with _tf32(A.is_cuda and n_iter > 1):
        Y = torch.matmul(A, Z)
        _power(n_iter - 1)
    # Last iteration in fp32 removes the TF32 error when sigma_1 / sigma_rank is large.
    _power(min(n_iter, 1))
    _orthonormalize_(Y, True, failures, householder)
    if not householder:
        for _ in range(final_passes):
            _orthonormalize_(Y, False, failures, householder)
    with _tf32(False):
        torch.matmul(A.mT, Y, out = Z)  # Z = (Q^T A)^T, N x q
    # From q = 128, eigh of [[0, R2], [R2^T, 0]] (eigenvalues +-sigma) beats cuSOLVER's fp32 SVD (~60x accuracy).
    # cuSOLVER's syevd and rocSOLVER's gesvd can refuse degenerate R2 (rank 0 / 1), so the safe rerun
    # solves the q x q problem with LAPACK.
    Q2, R2 = torch.linalg.qr(Z)
    try:
        if q < 128 or _safe:
            Ur, S, Vrh = torch.linalg.svd(R2.cpu() if _safe else R2)
            Ur, S, Vr = (x.to(R2.device) for x in (Ur[:, :rank], S[:rank], Vrh[:rank].mT))
        else:
            J = R2.new_zeros(2 * q, 2 * q)
            J[:q, q:] = R2
            J[q:, :q] = R2.mT
            L, X = torch.linalg.eigh(J)
            X = X[:, -rank:].flip(1)
            S = L[-rank:].flip(0).clamp_min_(0)
            tiny = torch.finfo(torch.float32).tiny
            Ur = X[:q].div_(torch.linalg.vector_norm(X[:q], dim = 0).clamp_min_(tiny))
            Vr = X[q:].div_(torch.linalg.vector_norm(X[q:], dim = 0).clamp_min_(tiny))
    except torch.linalg.LinAlgError:
        if _safe:
            raise
        return randomized_svd(W, rank, n_oversamples, n_iter, final_passes, generator, _safe = True)
    # A^T ~= Z Y^T = Q2 Ur S Vr^T Y^T, so A ~= (Y Vr) S (Q2 Ur)^T.
    V = Q2 @ Ur
    U = Y @ Vr
    if scale is not None:
        S = S * scale
    # cuSOLVER's Householder QR cannot fail on finite input; ROCm's can overflow, and CPU syncs for free.
    if not _safe and (not householder or A.device.type != "cuda" or torch.version.hip is not None):
        bad = torch.stack(failures).ne(0).any() if failures else S.new_zeros((), dtype = torch.bool)
        bad = bad | ~torch.isfinite(S).all() | ~torch.isfinite(U).all() | ~torch.isfinite(V).all()
        if bad.item():
            del A, Y, Z, U, V
            return randomized_svd(
                W, rank, n_oversamples, n_iter, final_passes, generator, _safe = True
            )
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
    V = V[:, :r].flip(1)
    if wide:
        return V.float().contiguous()
    U = A @ V
    U /= L[:r].flip(0).sqrt()
    return U.float().contiguous()


def _routed(base):
    # Routed compressed-tensors linears: loader_utils' wrapper (the saved original) densifies them first.
    return type(base).__name__ == "_UnslothNVFP4Linear" or getattr(
        base, "_unsloth_compressed_tensors_fp8", False
    )


def _check_float_weight(weight, init):
    # FSDP-QLoRA packs Params4bit into a float quant_storage: its dtype alone looks dense.
    if type(weight).__name__ in ("Params4bit", "Int8Params") or hasattr(weight, "quant_state"):
        raise TypeError(
            f"Unsloth: `init_lora_weights = {init!r}` re-runs on the base weights when an adapter is created "
            "or loaded, and they are quantized here. Load the model with `load_in_4bit = False` and "
            "`load_in_8bit = False`."
        )


def _pissa_init(self, adapter_name, init_lora_weights):
    from peft.tuners.lora.layer import transpose

    if _OWNER.get("thread") != threading.get_ident():
        return _ORIGINAL_ANY["pissa_init"](self, adapter_name, init_lora_weights)
    if _routed(self.get_base_layer()):
        return _ORIGINAL["pissa_init"](self, adapter_name, init_lora_weights)
    weight = self.get_base_layer().weight
    _check_float_weight(weight, init_lora_weights)
    dtype = weight.dtype
    r = self.r[adapter_name]
    if (
        dtype not in _FLOATS
        or r > min(weight.shape)
        or weight.device.type == "meta"
        or not torch.isfinite(weight).all()
    ):
        return _ORIGINAL["pissa_init"](self, adapter_name, init_lora_weights)
    if init_lora_weights == "pissa":
        n_iter, n_oversamples = 6, None
    else:
        parts = init_lora_weights.split("_niter_")
        if len(parts) != 2:
            return _ORIGINAL["pissa_init"](self, adapter_name, init_lora_weights)
        # PEFT's svd_lowrank(q = r, niter = N). The width must not depend on the device: loading the
        # adapter re-runs this, possibly elsewhere, and must rebuild the same residual.
        n_iter, n_oversamples = int(parts[-1]), 0
    _STATE["pissa"] = True
    self._unsloth_fast_pissa = getattr(self, "_unsloth_fast_pissa", set()) | {adapter_name}
    W = transpose(weight.to(torch.float32), self.fan_in_fan_out)
    U, S, Vh = randomized_svd(W, r, n_oversamples = n_oversamples, n_iter = n_iter)
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

    if _OWNER.get("thread") != threading.get_ident():
        return _ORIGINAL_ANY["mica_init"](self, adapter_name)
    if _routed(self.get_base_layer()):
        return _ORIGINAL["mica_init"](self, adapter_name)
    weight = self.get_base_layer().weight
    _check_float_weight(weight, "mica")
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
# PEFT's own methods, kept past the swap: another thread calling PEFT directly must not get ours.
_ORIGINAL_ANY = {}
_OWNER = {}
_STATE = {"pissa": False}
# One owner at a time: the swap is process-wide, so a second thread must not build layers mid-restore.
_LOCK = threading.RLock()
SIDECAR = "unsloth_lora_init.json"


@contextlib.contextmanager
def fast_lora_init(force = False):
    """Swap PEFT's LoraLayer.pissa_init / mica_init for the fast versions during get_peft_model.

    Yields a dict whose "pissa" is True once a layer took the fast PiSSA path. UNSLOTH_FAST_LORA_INIT=0
    keeps PEFT's own SVD unless `force` (reloading an adapter that recorded the fast path).
    """
    with _LOCK:
        if _ORIGINAL or (not force and os.environ.get("UNSLOTH_FAST_LORA_INIT", "1") == "0"):
            # Nested use keeps the outer swap.
            yield _STATE
            return
        from peft.tuners.lora.layer import LoraLayer

        swapped = []
        for name, fn in (("pissa_init", _pissa_init), ("mica_init", _mica_init)):
            original = LoraLayer.__dict__.get(name)
            if original is None:
                continue
            _ORIGINAL[name] = _ORIGINAL_ANY[name] = original
            setattr(LoraLayer, name, fn)
            swapped.append((name, original))
        _STATE["pissa"] = False
        _OWNER["thread"] = threading.get_ident()
        try:
            yield _STATE
        finally:
            _OWNER.clear()
            for name, original in swapped:
                setattr(LoraLayer, name, original)
            _ORIGINAL.clear()


def record_fast_pissa(model):
    """PEFT re-runs PiSSA when loading an adapter, and only the same algorithm rebuilds the residual base
    training saw: saves of this model mark their adapter folders so Unsloth's loaders can tell."""
    original = model.save_pretrained
    if getattr(original, "_unsloth_fast_pissa", False):
        return

    @functools.wraps(original)
    def save_pretrained(save_directory, *args, **kwargs):
        out = original(save_directory, *args, **kwargs)
        if kwargs.get("is_main_process", True):
            # Only adapters that took the fast path: another PiSSA adapter added through plain PEFT must
            # reload with PEFT's initializer. PEFT saves "default" at the root, others in a subfolder.
            fast = set()
            for module in model.modules():
                fast |= getattr(module, "_unsloth_fast_pissa", set())
            for name in fast:
                folder = save_directory if name == "default" else os.path.join(save_directory, name)
                if os.path.isfile(os.path.join(folder, "adapter_config.json")):
                    with open(os.path.join(folder, SIDECAR), "w", encoding = "utf-8") as f:
                        json.dump({"pissa": "unsloth_randomized_svd"}, f)
        return out

    save_pretrained._unsloth_fast_pissa = True
    model.save_pretrained = save_pretrained


def adapter_used_fast_pissa(path, **hub_kwargs):
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, SIDECAR))
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(path, SIDECAR, **{k: v for k, v in hub_kwargs.items() if v is not None})
        return True
    except Exception:
        return False


# Calibration hooks added after compilation are not guarded on (skip_nnmodule_hook_guards), so compiled
# modules silently skip them: CorDA then divides by a zero sample count.
def _calibration_functions():
    found = []
    try:
        from peft.tuners.lora import corda
        found.append((corda, "preprocess_corda"))
    except Exception:
        pass
    try:
        from peft.tuners.lora import eva
        found.append((eva, "initialize_lora_eva_weights"))
    except Exception:
        pass
    try:
        from peft.tuners.lora import loraga
        found.append((loraga, "preprocess_loraga"))
    except Exception:
        pass
    return found


def patch_peft_calibration_eager():
    import functools
    import sys

    if not hasattr(torch.compiler, "set_stance"):
        return
    for module, name in _calibration_functions():
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
        # vars(), not getattr: getattr on lazy modules (transformers) imports submodules.
        for loaded in list(sys.modules.values()):
            if getattr(loaded, "__dict__", {}).get(name) is original:
                setattr(loaded, name, eager)
