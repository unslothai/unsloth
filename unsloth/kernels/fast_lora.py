# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
from unsloth_zoo.utils import Version
from .utils import (
    _has_active_dora_adapter,
    _has_active_lora_bias,
    _has_multiple_active_adapters,
    addmm_,
    fast_dequantize,
    _base_quant_state,
    get_lora_parameters,
    get_lora_parameters_bias,
    matmul_lora,
    torch_amp_custom_fwd,
    torch_amp_custom_bwd,
)
from . import rms_layernorm as _rms_layernorm

_is_compiling = torch.compiler.is_compiling

# Inductor before torch 2.11 miscompiles traced backward passes and produces wrong LoRA gradients.
TRACE_LORA_FUNCTIONS = Version(torch.__version__) >= Version("2.11.0")


def _apply(function, *args):
    return function.apply(*args)


if not TRACE_LORA_FUNCTIONS:
    _apply = torch._dynamo.disable(_apply)


# torchao's NVFP4 / MX QAT linears fake-quantize inside forward() with no quantizer children, so fused LoRA
# would train unquantized. Read lazily from sys.modules so importing Unsloth never imports torchao.
_FORWARD_FAKE_QUANT_LINEARS = None


def _forward_fake_quant_linears():
    global _FORWARD_FAKE_QUANT_LINEARS
    if _FORWARD_FAKE_QUANT_LINEARS is None:
        qat = sys.modules.get("torchao.prototype.qat")
        if qat is None:
            return ()
        _FORWARD_FAKE_QUANT_LINEARS = tuple(
            cls
            for cls in (
                getattr(qat, "NVFP4FakeQuantizedLinear", None),
                getattr(qat, "MXFakeQuantizedLinear", None),
            )
            if isinstance(cls, type)
        )
    return _FORWARD_FAKE_QUANT_LINEARS


def _has_activation_fake_quantizer(proj):
    # Fused LoRA skips per-layer activation fake quantizers; explicit children, as proj.modules() is slow per call.
    # A plain nn.Linear (after merge_and_unload / unload) has no lora_A / lora_B: check the layer itself.
    forward_fake_quant_linears = _forward_fake_quant_linears()
    modules = proj._modules
    lora_A, lora_B = modules.get("lora_A"), modules.get("lora_B")
    for module in (
        modules.get("base_layer", proj),
        *(() if lora_A is None else lora_A.values()),
        *(() if lora_B is None else lora_B.values()),
    ):
        if module._modules.get("activation_fake_quantizer") is not None or isinstance(
            module, forward_fake_quant_linears
        ):
            return True
    return False


# lora_dropout > 0 and lora_bias = True run inside the fused kernels; "0" restores the PEFT fallback.
FUSED_LORA_DROPOUT = os.environ.get("UNSLOTH_FUSED_LORA_DROPOUT", "1") != "0"
FUSED_LORA_BIAS = os.environ.get("UNSLOTH_FUSED_LORA_BIAS", "1") != "0"


def _active_lora_adapter(proj):
    if getattr(proj, "disable_adapters", True) or getattr(proj, "merged", False):
        return None
    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", "default")
    if not isinstance(adapter, str):
        adapter = adapter[0]
    return adapter if adapter in proj.lora_A else None


def _lora_dropout_p(
    proj,
    X,
    adapter = False,
):
    """Dropout probability of the active adapter, 0.0 when it is a no-op, None when only PEFT can run it."""
    if adapter is False:
        adapter = _active_lora_adapter(proj)
    dropout = getattr(proj, "lora_dropout", None)
    if adapter is None or dropout is None or adapter not in dropout:
        return 0.0
    dropout = dropout[adapter]
    if isinstance(dropout, torch.nn.Identity) or not dropout.training:
        return 0.0
    if (
        type(dropout) is not torch.nn.Dropout
        or dropout.inplace
        or not FUSED_LORA_DROPOUT
        or X.device.type not in ("cuda", "xpu", "cpu")
    ):
        return None
    p = float(dropout.p)
    if p == 0:
        return 0.0
    return p if p < 1 else None


def lora_bias_blocks_fused(proj):
    return not FUSED_LORA_BIAS and _has_active_lora_bias(proj)


def _needs_peft_forward(proj, adapter):
    # A PEFT LoRA variant (DoRA, MoNTeCLoRA, VeLoRA, BD-LoRA, ...) replaces the LoRA branch in PEFT's
    # forward, and a trainable base weight needs its gradient: the fused Functions compute neither.
    variants = proj.__dict__.get("lora_variant")
    if variants and adapter in variants:
        return True
    return _base_weight_requires_grad(proj)


def _base_weight_requires_grad(proj):
    # A plain nn.Linear (after merge_and_unload() / unload()) is its own base layer.
    base = proj._modules.get("base_layer", proj)
    W = base._parameters.get("weight")
    return W is not None and W.requires_grad


_is_any_autocast_enabled = getattr(torch._C, "_is_any_autocast_enabled", None)


def _active_lora_requires_grad(projs):
    # A trainable active adapter elsewhere in the block makes autograd run the fused backward even when X
    # needs no grad (frozen embeddings without an input-grad hook) and only some projections are merged.
    for proj in projs:
        adapter = _active_lora_adapter(proj)
        if adapter is None:
            continue
        for module in (proj.lora_A[adapter], proj.lora_B[adapter]):
            for p in module.parameters():
                if p.requires_grad:
                    return True
    return False


def _autocast_dtype_mismatch(X):
    # Under autocast the fused matmuls return the autocast dtype but the in-place LoRA accumulation runs
    # in X's dtype (in-place ops are not autocast), so a different autocast dtype cannot be fused.
    # Mode flags only, no tensor work. Eager first asks whether any autocast is on (a private query, so not
    # under Dynamo, which traces the public calls below as constants).
    if (
        _is_any_autocast_enabled is not None
        and not _is_compiling()
        and not _is_any_autocast_enabled()
    ):
        return False
    device_type = "cuda" if X.is_cuda else X.device.type
    return (
        torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) != X.dtype
    )


def _is_plain_lora(proj):
    # One enabled, unmerged adapter with no dropout module and no lora_B bias (the default LoRA config).
    # Reads PEFT's instance state directly: getattr through nn.Module / properties costs more than the
    # routing it guards. Anything else (or other PEFT internals) takes the general path below.
    state = proj.__dict__
    adapter = state.get("_active_adapter")
    if type(adapter) is not str:
        if type(adapter) is not list or len(adapter) != 1:
            return False
        adapter = adapter[0]
    if state.get("_disable_adapters", True) or state.get("merged_adapters"):
        return False
    variants = state.get("lora_variant")
    if variants and adapter in variants:
        return False
    modules = proj._modules
    base = modules.get("base_layer")
    W = None if base is None else base._parameters.get("weight")
    if W is not None and W.requires_grad:
        return False
    dropout = modules.get("lora_dropout")
    lora_B = modules.get("lora_B")
    if dropout is None or lora_B is None:
        return False
    lora_B = lora_B._modules.get(adapter)
    return (
        type(dropout._modules.get(adapter)) is torch.nn.Identity
        and lora_B is not None
        and lora_B._parameters.get("bias") is None
    )


def _lora_dropout_and_bias(projs, X):
    """(dropout probabilities or None, lora_B biases) for the fused kernels, or None to use PEFT."""
    if _autocast_dtype_mismatch(X):
        return None
    for proj in projs:
        if not _is_plain_lora(proj):
            break
    else:
        return None, (None,) * len(projs)
    dropout_p, biases = [], []
    # Before torch 2.11 the fused Functions run Dynamo-disabled (see _apply): while compiling, a dropout
    # module or lora_B bias keeps PEFT's traceable forward, as before they were fused, not a graph break.
    peft_when_unplain = not TRACE_LORA_FUNCTIONS and _is_compiling()
    has_dropout_module = False
    grad_enabled = None
    for proj in projs:
        if peft_when_unplain and not _is_plain_lora(proj):
            return None
        adapter = _active_lora_adapter(proj)
        if adapter is None:
            # Enabled adapter missing from this projection: PEFT runs the base layer alone.
            if not getattr(proj, "disable_adapters", True) and not getattr(proj, "merged", False):
                return None
            # Disabled or merged: the fused Functions run the base matmul alone (the no-grad reference
            # forward of DPO / GRPO), but their backward needs LoRA weights, so PEFT runs it under grad.
            if grad_enabled is None:
                grad_enabled = torch.is_grad_enabled()
            if grad_enabled and (
                X.requires_grad
                or _base_weight_requires_grad(proj)
                or _active_lora_requires_grad(projs)
            ):
                return None
            dropout_p.append(0.0)
            biases.append(None)
            continue
        if _needs_peft_forward(proj, adapter):
            # A LoRA variant, or a trainable base weight.
            return None
        dropout = getattr(proj, "lora_dropout", None)
        has_dropout_module |= (
            dropout is not None
            and adapter in dropout
            and type(dropout[adapter]) is not torch.nn.Identity
        )
        p = _lora_dropout_p(proj, X, adapter)
        if p is None:
            return None
        dropout_p.append(p)
        biases.append(proj.lora_B[adapter].bias)
    if any(lora_bias_blocks_fused(proj) for proj in projs):
        return None
    # A dropout module in eval (p = 0) keeps the dropout route, so it still skips the global
    # dequantization scratch the plain path keeps resident.
    keep = any(dropout_p) or has_dropout_module
    return (tuple(dropout_p) if keep else None), tuple(biases)


def _dropout_forward(X, p):
    # Same RNG draws and rounding as torch.nn.Dropout under autograd: native_dropout on CUDA / XPU
    # (ATen's fused path), a bernoulli_ noise tensor in X's dtype on CPU.
    if not p:
        return X, None
    if X.device.type == "cpu":
        noise = torch.empty_like(X).bernoulli_(1 - p)
        mask = noise.bool()
        return X * noise.div_(1 - p), mask
    return torch.native_dropout(X, p, True)


def _dropout_replay(
    T,
    mask,
    p,
    S = 1.0,
):
    # S * T * mask / (1 - p): with S = 1 exactly the dropout forward's output, and the dropout backward.
    if mask.device.type == "cpu":
        T = T * mask.to(T.dtype).div_(1 - p)
        return T if S == 1 else T.mul_(S)
    return torch.ops.aten.native_dropout_backward(T, mask, S / (1.0 - p))


def _matmul_lora_ext(X, X_lora, W, W_quant, A, B, S, bias):
    # matmul_lora with the LoRA branch reading X_lora (X after dropout, or X itself) plus lora_B's bias.
    # No global dequantization scratch: these configurations ran PEFT, which keeps no dense weight resident.
    if A is None or (X_lora is X and bias is None):
        return matmul_lora(X, W, W_quant, A, B, S, use_global_buffer = False)
    out = matmul_lora(X, W, W_quant, None, None, None, use_global_buffer = False)
    dtype = X.dtype
    XA = torch.matmul(X_lora.reshape(-1, X_lora.shape[-1]), A.to(dtype).t())
    out2d = out.view(-1, out.shape[-1])
    if bias is None:
        addmm_(out2d, XA, B.to(dtype).t(), alpha = S)
    else:
        # PEFT's rounding order: (lora_B(x) + bias) * scaling, then + base. No add(alpha = S): Inductor
        # (torch 2.13) miscompiles it traced inside these Functions.
        out2d.add_(torch.addmm(bias.to(dtype), XA, B.to(dtype).t()).mul_(S))
    return out


def _add_lora_dX(dX, y_dB, A, S, mask, p):
    # dX += S * y_dB @ A.t(), through the dropout mask when the LoRA branch saw a dropped out X.
    if mask is None:
        addmm_(dX, y_dB, A.t(), alpha = S)
    else:
        dX.add_(_dropout_replay(y_dB @ A.t(), mask, p, S))


def _lora_bias_grad(dY, S):
    # Autograd casts it to the bias dtype.
    return dY.sum(0, dtype = torch.float32).mul_(S)


def _may_overwrite_input(X):
    # Backward writes dX into X; not when an RMSNorm saved X (its output) for its own backward,
    # including through a view of it (tiled MLP chunks).
    if _is_compiling():
        return True
    base = X._base
    return not (
        getattr(X, "_unsloth_rmsnorm_saved", False)
        or (base is not None and getattr(base, "_unsloth_rmsnorm_saved", False))
    )


class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        _forward_function,
        _backward_function,
        inplace = True,
        dropout_p = None,
        gateBias = None,
        upBias = None,
        downBias = None,
    ):
        dtype = X.dtype

        plain = dropout_p is None and gateBias is None and upBias is None and downBias is None
        if plain:
            e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
            g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
            h = _forward_function(e, g)
            i = matmul_lora(h, downW, downW_quant, downA, downB, downS)
            masks = (None, None, None)
        else:
            # Masks are drawn gate, up, down: the order PEFT's down(act(gate(X)) * up(X)) draws them in.
            gate_p, up_p, down_p = dropout_p or (0.0, 0.0, 0.0)
            Xd, gate_mask = _dropout_forward(X, gate_p)
            e = _matmul_lora_ext(X, Xd, gateW, gateW_quant, gateA, gateB, gateS, gateBias)
            Xd, up_mask = _dropout_forward(X, up_p)
            g = _matmul_lora_ext(X, Xd, upW, upW_quant, upA, upB, upS, upBias)
            del Xd
            h = _forward_function(e, g)
            hd, down_mask = _dropout_forward(h, down_p)
            i = _matmul_lora_ext(h, hd, downW, downW_quant, downA, downB, downS, downBias)
            del hd
            masks = (gate_mask, up_mask, down_mask)

        ctx.custom_saved_tensors = (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g, *masks)
        # Only an opt-in UNSLOTH_RMSNORM_SAVE_OUTPUT norm marks its output; skip the check otherwise.
        ctx.inplace = inplace and (not _rms_layernorm._SAVE_OUTPUT or _may_overwrite_input(X))
        ctx.dropout_p = dropout_p or (0.0, 0.0, 0.0)
        ctx.biases = (gateBias is not None, upBias is not None, downBias is not None)
        ctx.global_buffer = plain
        return i

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (
            gateW,
            gateW_quant,
            gateS,
            upW,
            upW_quant,
            upS,
            downW,
            downW_quant,
            downS,
            _backward_function,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g, gate_mask, up_mask, down_mask = (
            ctx.saved_tensors
        )
        gate_p, up_p, down_p = ctx.dropout_p

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        gateA, gateB, upA, upB, downA, downB = (
            gateA.to(dtype),
            gateB.to(dtype),
            upA.to(dtype),
            upB.to(dtype),
            downA.to(dtype),
            downB.to(dtype),
        )

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(),
            gateB.t(),
            upA.t(),
            upB.t(),
            downA.t(),
            downB.t(),
        )

        if downW.dtype == torch.float8_e4m3fn:
            # Dequantize like gate / up: matmul_lora would send the FP8 weight to fp8_linear, quantizing dY to FP8.
            downW, downW_quant = fast_dequantize(downW.t(), downW_quant, dtype = dY.dtype), None
        else:
            downW = downW.t()
        if down_mask is None:
            DW = matmul_lora(
                dY,
                downW,
                downW_quant,
                downB,
                downA,
                downS,
                use_global_buffer = getattr(ctx, "global_buffer", True),
            )
        else:
            # Only the LoRA branch saw the dropped out h, so only its gradient goes through the mask.
            DW = matmul_lora(dY, downW, downW_quant, None, None, None, use_global_buffer = False)
            down_mask = down_mask.view(-1, down_mask.shape[-1])
            DW.add_(_dropout_replay((dY @ downB.t()) @ downA.t(), down_mask, down_p, downS))
        del downW
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g
        if down_mask is not None:
            h = _dropout_replay(h, down_mask, down_p)

        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)

        # d_downA = h.t() @ (dY @ downB.t()), d_downB = (downA.t() @ h.t()) @ dY, both scaled by downS.
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha = downS, beta = 0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha = downS, beta = 0)

        up_dB = df @ upB.t()
        gate_dB = de @ gateB.t()

        if up_mask is not None:
            up_mask = up_mask.view(-1, hd)
        if gate_mask is not None:
            gate_mask = gate_mask.view(-1, hd)

        # d_upA = X.t() @ (df @ upB.t()), d_upB = (upA.t() @ X.t()) @ df, both scaled by upS.
        Xd = X if up_mask is None else _dropout_replay(X, up_mask, up_p)
        d_upA.addmm_(Xd.t(), up_dB, alpha = upS, beta = 0)
        d_upB.addmm_(upA.t() @ Xd.t(), df, alpha = upS, beta = 0)

        # d_gateA = X.t() @ (de @ gateB.t()), d_gateB = (gateA.t() @ X.t()) @ de, both scaled by gateS.
        Xd = X if gate_mask is None else _dropout_replay(X, gate_mask, gate_p)
        d_gateA.addmm_(Xd.t(), gate_dB, alpha = gateS, beta = 0)
        d_gateB.addmm_(gateA.t() @ Xd.t(), de, alpha = gateS, beta = 0)
        del Xd

        gate_has_bias, up_has_bias, down_has_bias = ctx.biases
        d_gateBias = _lora_bias_grad(de, gateS) if gate_has_bias else None
        d_upBias = _lora_bias_grad(df, upS) if up_has_bias else None
        d_downBias = _lora_bias_grad(dY, downS) if down_has_bias else None

        # dX = matmul_lora(df, upW.t(), ...) + matmul_lora(de, gateW.t(), ...), expanded below.
        upW = fast_dequantize(upW.t(), upW_quant, dtype = df.dtype)
        # Eager only: AOT autograd rejects a backward mutating a forward input that requires grad.
        dX = torch.matmul(df, upW.t(), out = X if ctx.inplace and not _is_compiling() else None)
        del upW
        _add_lora_dX(dX, up_dB, upA, upS, up_mask, up_p)

        gateW = fast_dequantize(gateW.t(), gateW_quant, dtype = de.dtype)
        addmm_(dX, de, gateW.t())
        del gateW
        _add_lora_dX(dX, gate_dB, gateA, gateS, gate_mask, gate_p)

        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            d_upA.t(),
            d_upB.t(),
            None,
            None,
            None,
            d_downA.t(),
            d_downB.t(),
            None,
            None,
            None,
            None,
            None,
            d_gateBias,
            d_upBias,
            d_downBias,
        )


from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel


def apply_lora_mlp_swiglu(
    self,
    X,
    inplace = True,
):
    projs = (self.gate_proj, self.up_proj, self.down_proj)
    extras = None
    if not any(
        _has_multiple_active_adapters(proj)
        or _has_active_dora_adapter(proj)
        or _has_activation_fake_quantizer(proj)
        for proj in projs
    ):
        extras = _lora_dropout_and_bias(projs, X)
    if extras is None:
        return self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))
    dropout_p, biases = extras
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = _apply(
        LoRA_MLP,
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        swiglu_fg_kernel,
        swiglu_DWf_DW_dfg_kernel,
        inplace,
        dropout_p,
        *biases,
    )
    return out


from .geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel


def apply_lora_mlp_geglu_exact(
    self,
    X,
    inplace = True,
):
    projs = (self.gate_proj, self.up_proj, self.down_proj)
    extras = None
    if not any(
        _has_multiple_active_adapters(proj)
        or _has_active_dora_adapter(proj)
        or _has_activation_fake_quantizer(proj)
        for proj in projs
    ):
        extras = _lora_dropout_and_bias(projs, X)
    if extras is None:
        return self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))
    dropout_p, biases = extras
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = _apply(
        LoRA_MLP,
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        geglu_exact_forward_kernel,
        geglu_exact_backward_kernel,
        inplace,
        dropout_p,
        *biases,
    )
    return out


from .geglu import geglu_approx_forward_kernel, geglu_approx_backward_kernel


def apply_lora_mlp_geglu_approx(self, X):
    projs = (self.gate_proj, self.up_proj, self.down_proj)
    extras = None
    if not any(
        _has_multiple_active_adapters(proj)
        or _has_active_dora_adapter(proj)
        or _has_activation_fake_quantizer(proj)
        for proj in projs
    ):
        extras = _lora_dropout_and_bias(projs, X)
    if extras is None:
        return self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))
    dropout_p, biases = extras
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW, upW_quant, upA, upB, upS = get_lora_parameters(self.up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = _apply(
        LoRA_MLP,
        X,
        gateW,
        gateW_quant,
        gateA,
        gateB,
        gateS,
        upW,
        upW_quant,
        upA,
        upB,
        upS,
        downW,
        downW_quant,
        downA,
        downB,
        downS,
        geglu_approx_forward_kernel,
        geglu_approx_backward_kernel,
        True,
        dropout_p,
        *biases,
    )
    return out


class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace = True,
        dropout_p = None,
        QBias = None,
        KBias = None,
        VBias = None,
    ):
        dtype = X.dtype

        # bitsandbytes 8-bit matmul expects 2D inputs, and TorchInductor/AOTAutograd fails on 3D tensors
        # during backward, so flatten the sequence dimension explicitly.
        orig_shape = X.shape
        X_for_matmul = X
        if X.dim() == 3:
            X_for_matmul = X.view(-1, X.shape[-1])
        if dropout_p is None and QBias is None and KBias is None and VBias is None:
            Q = matmul_lora(X_for_matmul, QW, QW_quant, QA, QB, QS)
            K = matmul_lora(X_for_matmul, KW, KW_quant, KA, KB, KS)
            V = matmul_lora(X_for_matmul, VW, VW_quant, VA, VB, VS)
            masks = (None, None, None)
        else:
            Q_p, K_p, V_p = dropout_p or (0.0, 0.0, 0.0)
            Xd, Q_mask = _dropout_forward(X_for_matmul, Q_p)
            Q = _matmul_lora_ext(X_for_matmul, Xd, QW, QW_quant, QA, QB, QS, QBias)
            Xd, K_mask = _dropout_forward(X_for_matmul, K_p)
            K = _matmul_lora_ext(X_for_matmul, Xd, KW, KW_quant, KA, KB, KS, KBias)
            Xd, V_mask = _dropout_forward(X_for_matmul, V_p)
            V = _matmul_lora_ext(X_for_matmul, Xd, VW, VW_quant, VA, VB, VS, VBias)
            del Xd
            masks = (Q_mask, K_mask, V_mask)

        # Restore original shape after matmul
        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], -1)
            K = K.view(orig_shape[0], orig_shape[1], -1)
            V = V.view(orig_shape[0], orig_shape[1], -1)

        ctx.custom_saved_tensors = (
            QW,
            QW_quant,
            QS,
            KW,
            KW_quant,
            KS,
            VW,
            VW_quant,
            VS,
        )
        ctx.save_for_backward(
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
            *masks,
        )
        # Only an opt-in UNSLOTH_RMSNORM_SAVE_OUTPUT norm marks its output; skip the check otherwise.
        ctx.inplace = inplace and (not _rms_layernorm._SAVE_OUTPUT or _may_overwrite_input(X))
        ctx.dropout_p = dropout_p or (0.0, 0.0, 0.0)
        ctx.biases = (QBias is not None, KBias is not None, VBias is not None)
        return Q, K, V

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        (
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
            Q_mask,
            K_mask,
            V_mask,
        ) = ctx.saved_tensors
        Q_p, K_p, V_p = ctx.dropout_p

        batch, seq_len, hd = X.shape
        dQ = dQ.reshape(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])  # view doesn't work on K.T
        dV = dV.reshape(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB, KA, KB, VA, VB = (
            QA.to(dtype),
            QB.to(dtype),
            KA.to(dtype),
            KB.to(dtype),
            VA.to(dtype),
            VB.to(dtype),
        )

        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        # Weight projection LoRA derivatives; see the Unsloth blogpost.
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        q_dB = dQ @ QB.t()
        k_dB = dK @ KB.t()
        v_dB = dV @ VB.t()

        Q_mask, K_mask, V_mask = (
            None if mask is None else mask.view(-1, hd) for mask in (Q_mask, K_mask, V_mask)
        )

        # d_QA = X.t() @ (dQ @ QB.t()), d_QB = (QA.t() @ X.t()) @ dQ, both scaled by QS; K and V below are
        # identical with their own scales.
        Xd = X if Q_mask is None else _dropout_replay(X, Q_mask, Q_p)
        d_QA.addmm_(Xd.t(), q_dB, alpha = QS, beta = 0)
        d_QB.addmm_(QA.t() @ Xd.t(), dQ, alpha = QS, beta = 0)

        Xd = X if K_mask is None else _dropout_replay(X, K_mask, K_p)
        d_KA.addmm_(Xd.t(), k_dB, alpha = KS, beta = 0)
        d_KB.addmm_(KA.t() @ Xd.t(), dK, alpha = KS, beta = 0)

        Xd = X if V_mask is None else _dropout_replay(X, V_mask, V_p)
        d_VA.addmm_(Xd.t(), v_dB, alpha = VS, beta = 0)
        d_VB.addmm_(VA.t() @ Xd.t(), dV, alpha = VS, beta = 0)
        del Xd

        Q_has_bias, K_has_bias, V_has_bias = ctx.biases
        d_QBias = _lora_bias_grad(dQ, QS) if Q_has_bias else None
        d_KBias = _lora_bias_grad(dK, KS) if K_has_bias else None
        d_VBias = _lora_bias_grad(dV, VS) if V_has_bias else None

        # Combine the per-projection derivatives into dX.
        QW = fast_dequantize(QW.t(), QW_quant, dtype = dQ.dtype)
        dX = torch.matmul(dQ, QW.t(), out = X if ctx.inplace and not _is_compiling() else None)
        del QW
        _add_lora_dX(dX, q_dB, QA, QS, Q_mask, Q_p)

        KW = fast_dequantize(KW.t(), KW_quant, dtype = dK.dtype)
        addmm_(dX, dK, KW.t())
        del KW
        _add_lora_dX(dX, k_dB, KA, KS, K_mask, K_p)

        VW = fast_dequantize(VW.t(), VW_quant, dtype = dV.dtype)
        addmm_(dX, dV, VW.t())
        del VW
        _add_lora_dX(dX, v_dB, VA, VS, V_mask, V_p)

        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
            None,
            None,
            None,
            d_QBias,
            d_KBias,
            d_VBias,
        )


def apply_lora_qkv(
    self,
    X,
    inplace = True,
):
    projs = (self.q_proj, self.k_proj, self.v_proj)
    extras = None
    if not any(
        _has_multiple_active_adapters(proj)
        or _has_active_dora_adapter(proj)
        or _has_activation_fake_quantizer(proj)
        for proj in projs
    ):
        extras = _lora_dropout_and_bias(projs, X)
    if extras is None:
        return self.q_proj(X), self.k_proj(X), self.v_proj(X)
    dropout_p, biases = extras
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = _apply(
        LoRA_QKV,
        X,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        inplace,
        dropout_p,
        *biases,
    )
    return Q, K, V


class LoRA_W(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        W,
        W_quant,
        A,
        B,
        S,
        dropout_p = None,
        bias = None,
    ):
        dtype = X.dtype
        if dropout_p is None and bias is None:
            XW = matmul_lora(X, W, W_quant, A, B, S)
            mask = None
        else:
            Xd, mask = _dropout_forward(X, dropout_p or 0.0)
            XW = _matmul_lora_ext(X, Xd, W, W_quant, A, B, S, bias)
            del Xd
        ctx.custom_saved_tensors = (
            W,
            W_quant,
            S,
        )
        ctx.save_for_backward(A, B, X, mask)
        ctx.dropout_p = dropout_p
        ctx.has_bias = bias is not None
        return XW

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X, mask = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])  # Must be reshape
        X = X.reshape(-1, X.shape[-1])  # Must be reshape
        if mask is not None:
            mask = mask.reshape(-1, hd)
        dtype = X.dtype

        A, B = A.to(dtype), B.to(dtype)

        A, B = A.t(), B.t()

        d_A = torch.empty_like(A)
        d_B = torch.empty_like(B)

        y_dB = dY @ B.t()

        # d_A = X.t() @ (dY @ B.t()), d_B = (A.t() @ X.t()) @ dY, both scaled by S.
        Xd = X if mask is None else _dropout_replay(X, mask, ctx.dropout_p)
        d_A.addmm_(Xd.t(), y_dB, alpha = S, beta = 0)
        d_B.addmm_(A.t() @ Xd.t(), dY, alpha = S, beta = 0)
        del Xd
        d_bias = _lora_bias_grad(dY, S) if ctx.has_bias else None

        W = fast_dequantize(W.t(), W_quant, dtype = dY.dtype)
        dX = dY @ W.t()
        del W
        _add_lora_dX(dX, y_dB, A, S, mask, ctx.dropout_p)

        return dX.view(batch, seq_len, hd), None, None, d_A.t(), d_B.t(), None, None, d_bias


def apply_lora_o(self, X):
    extras = None
    if not (
        _has_multiple_active_adapters(self.o_proj)
        or _has_active_dora_adapter(self.o_proj)
        or _has_activation_fake_quantizer(self.o_proj)
    ):
        extras = _lora_dropout_and_bias((self.o_proj,), X)
    if extras is None:
        return self.o_proj(X)
    dropout_p, (bias,) = extras
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    if dropout_p is None and bias is None:
        return _apply(LoRA_W, X, OW, OW_quant, OA, OB, OS)
    return _apply(LoRA_W, X, OW, OW_quant, OA, OB, OS, dropout_p and dropout_p[0], bias)


IDENTITY_DROPOUT = torch.nn.Identity


@torch._disable_dynamo
def fast_lora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Unsloth: Currently not supported yet - reshaping done incorrectly")
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names = adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        # Fastpath
        if len(self.active_adapters) == 1:
            active_adapter = self.active_adapters[0]
            if active_adapter not in self.lora_A.keys():
                return self.base_layer(x, *args, **kwargs)

            dropout = self.lora_dropout[active_adapter]
            if isinstance(dropout, IDENTITY_DROPOUT) and not self.use_dora[active_adapter]:
                lora_A = self.lora_A[active_adapter].weight
                lora_B = self.lora_B[active_adapter].weight
                scaling = self.scaling[active_adapter]
                W = self.base_layer.weight
                W_quant = getattr(W, "quant_state", None)
                if W_quant is None and W.ndim == 2 and W.shape[1] == 1:
                    W_quant = _base_quant_state(self.base_layer, W)
                return _apply(LoRA_W, x, W, W_quant, lora_A, lora_B, scaling)
            pass
        pass

        result = self.base_layer(x, *args, **kwargs)
        # Per Tim Dettmers: for 4bit, defensively clone, since backprop can fail on a manipulated view.
        result = result.clone()

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                if isinstance(dropout, torch.nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A = lora_A,
                    lora_B = lora_B,
                    scaling = scaling,
                    base_layer = self.get_base_layer(),
                    base_result = base_result,
                )
            if requires_conversion:
                result = result.to(expected_dtype)

    return result
