from dataclasses import dataclass
from logging import getLogger
from typing import Optional

import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from unsloth.gptq.triton.kernels import dequant248
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


def unpack_gptqstate(qstate):
    qweight, scales, qzeros, wf, g_idx, bits = (
        qstate.qweight,
        qstate.scales,
        qstate.qzeros,
        qstate.wf,
        qstate.g_idx,
        qstate.bits,
    )
    return qweight, scales, qzeros, wf, g_idx, bits


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
        qweight=qmodule.qweight.cuda(),
        qzeros=qmodule.qzeros.cuda(),
        scales=qmodule.scales.cuda(),
        g_idx=qmodule.g_idx.cuda(),
        bias=check_bias(qmodule),
        wf=qmodule.wf.cuda() if hasattr(qmodule, "wf") else None,
        kernel_switch_threshold=(
            qmodule.kernel_switch_threshold
            if hasattr(qmodule, "kernel_switch_threshold")
            else None
        ),
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


def matmul_lora_canonicalized(X, W, A, B, s):
    """
    X: rank-2 tensor (batch, seq_len) x (din)
    W: rank-2 tensor (din, dout)
    out: rank-2 tensor (batch, seq_len) x (dout)
    din = X.shape[1]
    dout = W.shape[1]
    """

    out = torch.matmul(X, W)

    A, B = A.t(), B.t()
    out += (X @ A) @ (s * B)

    return out


def matmul_lora(X, W, A, B, s, out=None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W, out=out)

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out


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
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        gate_qweight,
        gate_scales,
        gate_qzeros,
        gate_wf,
        gate_g_idx,
        gate_bits,
        gateA,
        gateB,
        gateS,
        up_qweight,
        up_scales,
        up_qzeros,
        up_wf,
        up_g_idx,
        up_bits,
        upA,
        upB,
        upS,
        down_qweight,
        down_scales,
        down_qzeros,
        down_wf,
        down_g_idx,
        down_bits,
        downA,
        downB,
        downS,
    ):
        dtype = X.dtype

        # Separate dequant248 from matmul
        gateW = dequant248(
            gate_qweight, gate_scales, gate_qzeros, gate_wf, gate_g_idx, gate_bits
        )
        e = matmul_lora(X, gateW, gateA, gateB, gateS)
        upW = dequant248(up_qweight, up_scales, up_qzeros, up_wf, up_g_idx, up_bits)
        g = matmul_lora(X, upW, upA, upB, upS)
        # f = torch.nn.functional.silu(e)
        # h = f * g
        h = swiglu_fg_kernel(e, g)

        downW = dequant248(
            down_qweight, down_scales, down_qzeros, down_wf, down_g_idx, down_bits
        )
        i = matmul_lora(h, downW, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gate_qweight,
            gate_scales,
            gate_qzeros,
            gate_wf,
            gate_g_idx,
            gate_bits,
            gateS,
            up_qweight,
            up_scales,
            up_qzeros,
            up_wf,
            up_g_idx,
            up_bits,
            upS,
            down_qweight,
            down_scales,
            down_qzeros,
            down_wf,
            down_g_idx,
            down_bits,
            downS,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        return i

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (
            gate_qweight,
            gate_scales,
            gate_qzeros,
            gate_wf,
            gate_g_idx,
            gate_bits,
            gateS,
            up_qweight,
            up_scales,
            up_qzeros,
            up_wf,
            up_g_idx,
            up_bits,
            upS,
            down_qweight,
            down_scales,
            down_qzeros,
            down_wf,
            down_g_idx,
            down_bits,
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

        downW = dequant248(
            down_qweight, down_scales, down_qzeros, down_wf, down_g_idx, down_bits
        )
        DW = matmul_lora(dY, downW.t(), downB, downA, downS)
        # e = e.float()
        # se = 1.0 / (1.0 + torch.exp(-e))
        # f = (se * e).to(dtype)
        # h = f * g
        # df = DW * f
        # dg = DW * g
        # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
        DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
        h, df, de = DW, e, g

        # Down projection LoRA weights
        d_downA = h.t() @ (dY @ downB.t())
        d_downB = (downA.t() @ h.t()) @ dY
        d_downA *= downS
        d_downB *= downS

        # Up projection LoRA weights
        d_upA = X.t() @ (df @ upB.t())
        d_upB = (upA.t() @ X.t()) @ df
        d_upA *= upS
        d_upB *= upS

        # Gate projection LoRA weights
        d_gateA = X.t() @ (de @ gateB.t())
        d_gateB = (gateA.t() @ X.t()) @ de
        d_gateA *= gateS
        d_gateB *= gateS

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = dequant248(up_qweight, up_scales, up_qzeros, up_wf, up_g_idx, up_bits)
        dX = torch.matmul(df, upW.t())  # , out=X)
        del upW
        dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())

        gateW = dequant248(
            gate_qweight, gate_scales, gate_qzeros, gate_wf, gate_g_idx, gate_bits
        )
        dX += de @ gateW.t()
        del gateW
        dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            None,
            None,
            None,
            None,
            d_gateA.t(),
            d_gateB.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_upA.t(),
            d_upB.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_downA.t(),
            d_downB.t(),
            None,
        )


def apply_lora_mlp(self, X):
    gateQstate, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upQState, upA, upB, upS = get_lora_parameters(self.up_proj)
    downQState, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(
        X,
        *unpack_gptqstate(gateQstate),
        gateA,
        gateB,
        gateS,
        *unpack_gptqstate(upQState),
        upA,
        upB,
        upS,
        *unpack_gptqstate(downQState),
        downA,
        downB,
        downS,
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
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        Q_qweight,
        Q_scales,
        Q_qzeros,
        Q_wf,
        Q_g_idx,
        Q_bits,
        QA,
        QB,
        QS,
        K_qweight,
        K_scales,
        K_qzeros,
        K_wf,
        K_g_idx,
        K_bits,
        KA,
        KB,
        KS,
        V_qweight,
        V_scales,
        V_qzeros,
        V_wf,
        V_g_idx,
        V_bits,
        VA,
        VB,
        VS,
    ):
        dtype = X.dtype

        QW = dequant248(Q_qweight, Q_scales, Q_qzeros, Q_wf, Q_g_idx, Q_bits)
        KW = dequant248(K_qweight, K_scales, K_qzeros, K_wf, K_g_idx, K_bits)
        VW = dequant248(V_qweight, V_scales, V_qzeros, V_wf, V_g_idx, V_bits)
        Q = matmul_lora(X, QW, QA, QB, QS)
        K = matmul_lora(X, KW, KA, KB, KS)
        V = matmul_lora(X, VW, VA, VB, VS)

        ctx.custom_saved_tensors = (
            Q_qweight,
            Q_scales,
            Q_qzeros,
            Q_wf,
            Q_g_idx,
            Q_bits,
            QS,
            K_qweight,
            K_scales,
            K_qzeros,
            K_wf,
            K_g_idx,
            K_bits,
            KS,
            V_qweight,
            V_scales,
            V_qzeros,
            V_wf,
            V_g_idx,
            V_bits,
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
            Q_qweight,
            Q_scales,
            Q_qzeros,
            Q_wf,
            Q_g_idx,
            Q_bits,
            QS,
            K_qweight,
            K_scales,
            K_qzeros,
            K_wf,
            K_g_idx,
            K_bits,
            KS,
            V_qweight,
            V_scales,
            V_qzeros,
            V_wf,
            V_g_idx,
            V_bits,
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
        QW = dequant248(Q_qweight, Q_scales, Q_qzeros, Q_wf, Q_g_idx, Q_bits)
        dX = torch.matmul(dQ, QW.t())  # , out=X)
        del QW
        dX += dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t())

        # dK
        KW = dequant248(K_qweight, K_scales, K_qzeros, K_wf, K_g_idx, K_bits)
        dX += dK @ KW.t()
        del KW
        dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())

        # dV
        VW = dequant248(V_qweight, V_scales, V_qzeros, V_wf, V_g_idx, V_bits)
        dX += dV @ VW.t()
        del VW
        dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())

        # Q_qweight, Q_scales, Q_qzeros, Q_wf, Q_g_idx, Q_bits, QA, QB, QS,
        # K_qweight, K_scales, K_qzeros, K_wf, K_g_idx, K_bits, KA, KB, KS,
        # V_qweight, V_scales, V_qzeros, V_wf, V_g_idx, V_bits, VA, VB, VS,
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            None,
            None,
            None,
            None,
            d_QA.t(),
            d_QB.t(),
            None,  # d_QS.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            d_KA.t(),
            d_KB.t(),
            None,  # d_KS.t(),
            None,
            None,
            None,
            None,
            None,
            None,
            d_VA.t(),
            d_VB.t(),
            None,
        )


def apply_lora_qkv(self, X):
    Qqstate, QA, QB, QS = get_lora_parameters(self.q_proj)
    Kqstate, KA, KB, KS = get_lora_parameters(self.k_proj)
    Vqstate, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(
        X,
        *unpack_gptqstate(Qqstate),
        QA,
        QB,
        QS,
        *unpack_gptqstate(Kqstate),
        KA,
        KB,
        KS,
        *unpack_gptqstate(Vqstate),
        VA,
        VB,
        VS,
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
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        O_qweight,
        O_scales,
        O_qzeros,
        O_wf,
        O_g_idx,
        O_bits,
        A,
        B,
        S,
    ):
        W = dequant248(O_qweight, O_scales, O_qzeros, O_wf, O_g_idx, O_bits)
        XW = matmul_lora(X, W, A, B, S)
        del W
        ctx.custom_saved_tensors = (
            O_qweight,
            O_scales,
            O_qzeros,
            O_wf,
            O_g_idx,
            O_bits,
            S,
        )
        ctx.save_for_backward(A, B, X)
        return XW

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY: torch.Tensor):
        O_qweight, O_scales, O_qzeros, O_wf, O_g_idx, O_bits, S = (
            ctx.custom_saved_tensors
        )
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
        W = dequant248(O_qweight, O_scales, O_qzeros, O_wf, O_g_idx, O_bits)
        dX = dY @ W.t()
        del W
        dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())

        # O_qweight, O_scales, O_qzeros, O_wf, O_g_idx, O_bits, A, B, S
        return (
            dX.view(batch, seq_len, hd),
            None,
            None,
            None,
            None,
            None,
            None,
            d_A.t(),
            d_B.t(),
            None,
        )


def apply_lora_o(self, X):
    Oqstate, OA, OB, OS = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, *unpack_gptqstate(Oqstate), OA, OB, OS)
    return O


