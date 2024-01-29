from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from unsloth.gptq.triton.kernels import quant_matmul_248, transpose_quant_matmul_248
from unsloth.kernels.fast_lora import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel

logger = getLogger(__name__)


@dataclass
class GPTQuantState:
    """
    Stores params for GPTQ linear layer quantization
    """

    infeatures: int
    outfeatures: int

    bits: int
    group_size: int
    maxq: int
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor

    # cuda_kernel params (not used currently)
    kernel_switch_threshold: int
    autogptq_cuda_available: bool = False
    autogptq_cuda: bool = False

    wf: Optional[torch.Tensor] = None
    use_cuda_fp16: bool = False

    bias: Optional[torch.Tensor] = None
    trainable: bool = True


def extract_gptq_state(qmodule):
    if hasattr(qmodule, "base_layer"):
        qmodule = qmodule.base_layer

    def check_bias(qmodule):
        if hasattr(qmodule, "bias") and qmodule.bias is not None:
            if qmodule.bias.count_nonzero() > 0:
                return qmodule.bias
        return None

    return GPTQuantState(
        infeatures=qmodule.infeatures,
        outfeatures=qmodule.outfeatures,
        bits=qmodule.bits,
        group_size=qmodule.group_size,
        maxq=qmodule.maxq,
        qweight=qmodule.qweight,
        qzeros=qmodule.qzeros,
        scales=qmodule.scales,
        g_idx=qmodule.g_idx,
        bias=check_bias(qmodule),
        wf=qmodule.wf if hasattr(qmodule, "wf") else None,
        kernel_switch_threshold=qmodule.kernel_switch_threshold
        if hasattr(qmodule, "kernel_switch_threshold")
        else None,
        autogptq_cuda_available=qmodule.autogptq_cuda_available,
        # use_cuda_fp16=qmodule.use_cuda_fp16,
    )


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = proj.base_layer if hasattr(proj, "base_layer") else proj
    qstate = extract_gptq_state(base_layer)

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return qstate, None, None, None

    active_adapter = (
        proj.active_adapters[0]
        if hasattr(proj, "active_adapters")
        else proj.active_adapter
    )
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return qstate, A, B, s


def apply_lora_mlp(self, X):
    gateGPTQState, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upGPTQState, upA, upB, upS = get_lora_parameters(self.up_proj)
    downGPTQState, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        gateGPTQState,
        gateA,
        gateB,
        gateS,
        upGPTQState,
        upA,
        upB,
        upS,
        downGPTQState,
        downA,
        downB,
        downS,
    )
    return out


class QuantLinearFunction(torch.autograd.Function):
    """
    Similar to bitsandbytes implementation except uses fused triton quantized matmul kernels for GPTQ quant / dequant
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = quant_matmul_248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_quant_matmul_248(
                grad_output, qweight, scales, qzeros, g_idx, bits, maxq
            )
        return grad_input, None, None, None, None, None, None


def matmul_gptq_triton(
    x: torch.Tensor,
    qstate: GPTQuantState,
    A: torch.Tensor = None,
    B: torch.Tensor = None,
    s: torch.Tensor = None,
    out=None,
    transpose=False,
):
    dtype = x.dtype

    # matmul kernels expect x to be 2-D

    if not transpose:
        out_shape = x.shape[:-1] + (qstate.outfeatures,)

        out = quant_matmul_248(
            x.reshape(-1, x.shape[-1]),
            qstate.qweight,
            qstate.scales,
            qstate.qzeros,
            qstate.g_idx,
            qstate.bits,
            qstate.maxq,
        )
        out = out.to(dtype).reshape(out_shape)

        if A is not None:
            # LoRA is enabled
            A, B = A.t(), B.t()
            out += (x @ A.to(dtype)) @ (s * B.to(dtype))
    else:
        out_shape = x.shape[:-1] + (qstate.infeatures,)

        out = transpose_quant_matmul_248(
            x.reshape(-1, x.shape[-1]),
            qstate.qweight,
            qstate.scales,
            qstate.qzeros,
            qstate.g_idx,
            qstate.bits,
            qstate.maxq,
        )
        out = out.to(dtype).reshape(out_shape)
        if A is not None:
            out += (x @ B.t().to(dtype)) @ (s * A.t().to(dtype))
    # assert (
    #     qstate.bias is None
    # ), "unsloth backprop does not support bias in quantized linear modules"

    # out = out + qstate.bias if qstate.bias is not None else out

    return out


class LoRA_MLP(torch.autograd.Function):
    """
    
    Implementation of LoRA MLP per unsloth.kernels.fast_lora.py except with triton GPTQ fused quant/dequant matmul kernels
    
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
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gateGPTQState: GPTQuantState,
        gateA,
        gateB,
        gateS,
        upGPTQState,
        upA,
        upB,
        upS,
        downGPTQState,
        downA,
        downB,
        downS,
    ):
        dtype = X.dtype

        e = matmul_gptq_triton(X, gateGPTQState, gateA, gateB, gateS)
        g = matmul_gptq_triton(X, upGPTQState, upA, upB, upS)
        h = swiglu_fg_kernel(e, g)
        i = matmul_gptq_triton(h, downGPTQState, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateGPTQState,
            gateS,
            upGPTQState,
            upS,
            downGPTQState,
            downS,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        return i

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (
            gateGPTQState,
            gateS,
            upGPTQState,
            upS,
            downGPTQState,
            downS,
        ) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors

        gateA, gateB, upA, upB, downA, downB = (
            gateA.t(),
            gateB.t(),
            upA.t(),
            upB.t(),
            downA.t(),
            downB.t(),
        )

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype

        # DW_f   = (D @ W.T * f)
        # DW_dfg = (D @ W.T * df * g)
        # print(f"dY Shape: {dY.shape}")
        DW = matmul_gptq_triton(dY, downGPTQState, downA, downB, downS, transpose=True)
        # print(f"DW shape: {DW.shape}")
        # print(f"DW dtype: {DW.dtype}")
        # print(f"e dtype: {e.dtype}")
        # print(f"g dtype: {g.dtype}")

        DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
        h, DW_f, DW_dfg = DW, e, g

        # Down projection LoRA weights
        # print(f"h dtype: {h.dtype}")
        # print(f"dY dtype: {dY.dtype}")
        # print(f"downB dtype: {downB.dtype}")

        d_downA = h.t() @ (dY @ downB.t())
        d_downB = (downA.t() @ h.t()) @ dY
        d_downA *= downS
        d_downB *= downS

        # Up projection LoRA weights
        d_upA = X.t() @ (DW_f @ upB.t())
        d_upB = (upA.t() @ X.t()) @ DW_f
        d_upA *= upS
        d_upB *= upS

        # Gate projection LoRA weights
        d_gateA = X.t() @ (DW_dfg @ gateB.t())
        d_gateB = gateA.t() @ X.t() @ DW_dfg
        d_gateA *= gateS
        d_gateB *= gateS

        # Final derivatives to backpropagate backwards.
        # See our blogpost for more details.
        # (D @ W.T * f) @ U.T
        # upW = fast_dequantize(upW.t(), upW_quant)
        # (D @ W.T * f) @ (U.T + B.T @ A.T)
        # dX = torch.matmul(DW_f, upW.t(), out=X)
        # print(f"DW_f shape: {DW_f.shape}")

        dX = matmul_gptq_triton(DW_f, upGPTQState, transpose=True)
        # del upW
        dX += DW_f @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())

        # And add the derivative for the gate projection
        # gateW = fast_dequantize(gateW.t(), gateW_quant)
        # dX += DW_dfg @ gateW.t()
        # print(f"DW_dfg shape: {DW_dfg.shape}")

        dX += matmul_gptq_triton(DW_dfg, gateGPTQState, transpose=True)
        dX += DW_dfg @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())

        # #  X: torch.Tensor,
        # gateGPTQState: GPTQuantState,
        # gateA,
        # gateB,
        # gateS,
        # upGPTQState,
        # upA,
        # upB,
        # upS,
        # downGPTQState,
        # downA,
        # downB,
        # downS,
        return (
            dX.view(batch, seq_len, hd),  # X
            None,  # gateGPTQState
            d_gateA.t(),  # gateA
            d_gateB.t(),  # gateB
            None,  # gateS
            None,  # upGTPQState
            d_upA.t(),  # upA
            d_upB.t(),  # upB
            None,  # upS
            None,  # downGPTQState
            d_downA.t(),  # downA
            d_downB.t(),  # downB
            None,  # downS
        )


class LoRA_QKV(torch.autograd.Function):
    """
    Implementation of LoRA QKV per unsloth.kernels.fast_lora.py except with triton GPTQ fused quant/dequant matmul kernels
    
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
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        QueryGPTQState: GPTQuantState,
        QA,
        QB,
        QS,
        KeyGPTQState: GPTQuantState,
        KA,
        KB,
        KS,
        ValueGPTQState: GPTQuantState,
        VA,
        VB,
        VS,
    ):
        dtype = X.dtype

        Q = matmul_gptq_triton(X, QueryGPTQState, QA, QB, QS)
        K = matmul_gptq_triton(X, KeyGPTQState, KA, KB, KS)
        V = matmul_gptq_triton(X, ValueGPTQState, VA, VB, VS)

        ctx.custom_saved_tensors = (
            QueryGPTQState,
            QS,
            KeyGPTQState,
            KS,
            ValueGPTQState,
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
        )
        return Q, K, V

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dQ, dK, dV):
        (
            QueryGPTQState,
            QS,
            KeyGPTQState,
            KS,
            ValueGPTQState,
            VS,
        ) = ctx.custom_saved_tensors
        (
            X,
            QA,
            QB,
            KA,
            KB,
            VA,
            VB,
        ) = ctx.saved_tensors

        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])  # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype

        ### Weight projection LoRA weights
        # See our blogpost for more details.

        # Q Projection
        d_QA = X.t() @ (dQ @ QB.t())
        d_QB = (QA.t() @ X.t()) @ dQ
        d_QA *= QS
        d_QB *= QS

        # K Projection
        d_KA = X.t() @ (dK @ KB.t())
        d_KB = (KA.t() @ X.t()) @ dK
        d_KA *= KS
        d_KB *= KS

        # V Projection
        d_VA = X.t() @ (dV @ VB.t())
        d_VB = (VA.t() @ X.t()) @ dV
        d_VA *= VS
        d_VB *= VS

        # Combine derivatives to find dX
        # dQ
        # QW = fast_dequantize(QW.t(), QW_quant)
        # dX = torch.matmul(dQ, QW.t())  # , out=X)
        dX = matmul_gptq_triton(dQ, QueryGPTQState, transpose=True)
        dX += dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t())

        # dK
        dX += matmul_gptq_triton(dK, KeyGPTQState, transpose=True)
        dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())

        # dV
        dX += matmul_gptq_triton(dV, ValueGPTQState, transpose=True)
        dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,

        # # X: torch.Tensor,
        # QueryGPTQState: GPTQuantState,
        # QA,
        # QB,
        # QS,
        # KeyGPTQState: GPTQuantState,
        # KA,
        # KB,
        # KS,
        # ValueGPTQState: GPTQuantState,
        # VA,
        # VB,
        # VS,
        return (
            dX.view(batch, seq_len, hd),  # dX
            None,  # QueryGPTQState
            d_QA.t(),
            d_QB.t(),
            None,  # QS
            None,  # KeyGPTQState
            d_KA.t(),
            d_KB.t(),
            None,  # KS
            None,  # ValueGPTQState
            d_VA.t(),
            d_VB.t(),
            None,  # VS
        )


def apply_lora_qkv(self, X):
    QueryGPTQState, QA, QB, QS = get_lora_parameters(self.q_proj)
    KeyGPTQState, KA, KB, KS = get_lora_parameters(self.k_proj)
    ValueGPTQState, VA, VB, VS = get_lora_parameters(self.v_proj)

    Q, K, V = LoRA_QKV.apply(
        X,
        QueryGPTQState,
        QA,
        QB,
        QS,
        KeyGPTQState,
        KA,
        KB,
        KS,
        ValueGPTQState,
        VA,
        VB,
        VS,
    )
    return Q, K, V


class LoRA_W(torch.autograd.Function):
    """
    Implementation of LoRA W per unsloth.kernels.fast_lora.py except with triton GPTQ fused quant/dequant matmul kernels

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
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X: torch.Tensor, qstate: GPTQuantState, A, B, S):
        dtype = X.dtype
        XW = matmul_gptq_triton(X, qstate, A, B, S)
        ctx.custom_saved_tensors = (
            qstate,
            S,
        )
        ctx.save_for_backward(A, B, X)
        return XW

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY: torch.Tensor):
        qstate, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        A, B = A.t(), B.t()

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1])  # Must be reshape
        X = X.reshape(-1, X.shape[-1])  # Must be reshape
        dtype = X.dtype

        ### Weight projection LoRA weights
        # Weight projection
        d_A = X.t() @ (dY @ B.t())
        d_B = (A.t() @ X.t()) @ dY
        d_A *= S
        d_B *= S

        # Get derivative for dX
        # W = fast_dequantize(W.t(), W_quant)
        # dX = dY @ W.t()
        dX = matmul_gptq_triton(dY, qstate, transpose=True)
        dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())

        # W, W_quant, A, B, S
        return dX.view(batch, seq_len, hd), None, d_A.t(), d_B.t(), None


def apply_lora_o(self, X):
    OGPTQState, OA, OB, OS = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, OGPTQState, OA, OB, OS)
    return O
