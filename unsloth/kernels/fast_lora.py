# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from .utils import fast_dequantize, QUANT_STATE, get_lora_parameters, matmul_lora


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
    def forward(ctx, X : torch.Tensor,
                gateW, gateW_quant, gateA, gateB, gateS,
                  upW,   upW_quant, upA,   upB,   upS,
                downW, downW_quant, downA, downB, downS,
                _forward_function, _backward_function,):
        dtype = X.dtype

        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X,   upW,   upW_quant,   upA,   upB,   upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)

        ctx.custom_saved_tensors = (
            gateW, gateW_quant, gateS,
            upW, upW_quant, upS,
            downW, downW_quant, downS,
            _backward_function,
        )
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB,
                              X, e, g)
        return i
    pass


    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateW, gateW_quant, gateS, upW, upW_quant, upS, downW, downW_quant, downS, \
            _backward_function = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, \
            X, e, g = ctx.saved_tensors

        gateA, gateB, upA, upB, downA, downB = \
            gateA.t(), gateB.t(), upA.t(), upB.t(), downA.t(), downB.t()

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X .view(-1, X .shape[-1])
        e  = e .view(-1, e .shape[-1])
        g  = g .view(-1, g .shape[-1])
        dtype = X.dtype

        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g

        # Down projection LoRA weights
        d_downA = h.t() @ (dY @ downB.t())
        d_downB = (downA.t() @ h.t()) @ dY
        d_downA *= downS
        d_downB *= downS

        # Up projection LoRA weights
        d_upA   = X.t() @ (df @ upB.t())
        d_upB   = (upA.t() @ X.t()) @ df
        d_upA  *= upS
        d_upB  *= upS

        # Gate projection LoRA weights
        d_gateA = X.t() @ (de @ gateB.t())
        d_gateB = (gateA.t() @ X.t()) @ de
        d_gateA *= gateS
        d_gateB *= gateS

        # dX  = matmul_lora(df, upW.t(), upW_quant, upB, upA, upS)
        # dX += matmul_lora(de, gateW.t(), gateW_quant, gateB, gateA, gateS)
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out = X)
        del upW
        dX += df @ upB.to(dtype).t() @ (upS * upA.to(dtype).t())

        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX += de @ gateW.t()
        del gateW
        dX += de @ gateB.to(dtype).t() @ (gateS * gateA.to(dtype).t())

        # gateW, gateW_quant, gateA, gateB, gateS,
        #  upW,    upW_quant,   upA,   upB,   upS,
        # downW, downW_quant, downA, downB, downS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_gateA.t(), d_gateB.t(), None, \
            None, None,   d_upA.t(),   d_upB.t(), None, \
            None, None, d_downA.t(), d_downB.t(), None, \
            None, None, # _backward and _forward
    pass
pass


from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
def apply_lora_mlp_swiglu(self, X):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel,)
    return out
pass


from .geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel
def apply_lora_mlp_geglu_exact(self, X):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         geglu_exact_forward_kernel, geglu_exact_backward_kernel,)
    return out
pass


from .geglu import geglu_approx_forward_kernel, geglu_approx_backward_kernel
def apply_lora_mlp_geglu_approx(self, X):
    gateW, gateW_quant, gateA, gateB, gateS = get_lora_parameters(self.gate_proj)
    upW,     upW_quant,   upA,   upB,   upS = get_lora_parameters(self.  up_proj)
    downW, downW_quant, downA, downB, downS = get_lora_parameters(self.down_proj)
    out = LoRA_MLP.apply(X,
                         gateW, gateW_quant, gateA, gateB, gateS,
                         upW,     upW_quant, upA,   upB,   upS,
                         downW, downW_quant, downA, downB, downS,
                         geglu_approx_forward_kernel, geglu_approx_backward_kernel,)
    return out
pass


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
    def forward(ctx, X : torch.Tensor,
                QW, QW_quant, QA, QB, QS,
                KW, KW_quant, KA, KB, KS,
                VW, VW_quant, VA, VB, VS,):
        dtype = X.dtype

        Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X, VW, VW_quant, VA, VB, VS)

        ctx.custom_saved_tensors = (
            QW, QW_quant, QS,
            KW, KW_quant, KS,
            VW, VW_quant, VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB,)
        return Q, K, V
    pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = \
            ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB, = ctx.saved_tensors

        QA, QB, KA, KB, VA, VB = \
            QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()

        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1]) # view doesn't work on K.T
        dV = dV.view(-1, dV.shape[-1])
        X  = X .view(-1, X .shape[-1])
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
        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out = X)
        del QW
        dX += (dQ @ QB.to(dtype).t() @ (QS * QA.to(dtype).t()))

        # dK
        KW = fast_dequantize(KW.t(), KW_quant)
        dX += dK @ KW.t()
        del KW
        dX += dK @ KB.to(dtype).t() @ (KS * KA.to(dtype).t())

        # dV
        VW = fast_dequantize(VW.t(), VW_quant)
        dX += dV @ VW.t()
        del VW
        dX += dV @ VB.to(dtype).t() @ (VS * VA.to(dtype).t())

        # QW, QW_quant, QA, QB, QS,
        # KW, KW_quant, KA, KB, KS,
        # VW, VW_quant, VA, VB, VS,
        return dX.view(batch, seq_len, hd), \
            None, None, d_QA.t(), d_QB.t(), None, \
            None, None, d_KA.t(), d_KB.t(), None, \
            None, None, d_VA.t(), d_VB.t(), None
    pass
pass


def apply_lora_qkv(self, X):
    QW, QW_quant, QA, QB, QS = get_lora_parameters(self.q_proj)
    KW, KW_quant, KA, KB, KS = get_lora_parameters(self.k_proj)
    VW, VW_quant, VA, VB, VS = get_lora_parameters(self.v_proj)
    Q, K, V = LoRA_QKV.apply(X,
        QW, QW_quant, QA, QB, QS,
        KW, KW_quant, KA, KB, KS,
        VW, VW_quant, VA, VB, VS,
    )
    return Q, K, V
pass


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
    def forward(ctx, X : torch.Tensor,
                W, W_quant, A, B, S):
        dtype = X.dtype
        XW = matmul_lora(X, W, W_quant, A, B, S)
        ctx.custom_saved_tensors = (W, W_quant, S,)
        ctx.save_for_backward(A, B, X)
        return XW
    pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY : torch.Tensor):
        W, W_quant, S = ctx.custom_saved_tensors
        A, B, X = ctx.saved_tensors

        A, B = A.t(), B.t()

        batch, seq_len, hd = X.shape
        dY = dY.reshape(-1, dY.shape[-1]) # Must be reshape
        X  = X .reshape(-1, X .shape[-1]) # Must be reshape
        dtype = X.dtype

        ### Weight projection LoRA weights
        # Weight projection
        d_A = X.t() @ (dY @ B.t())
        d_B = (A.t() @ X.t()) @ dY
        d_A *= S
        d_B *= S

        # Get derivative for dX
        W = fast_dequantize(W.t(), W_quant)
        dX = dY @ W.t()
        del W
        dX += dY @ B.to(dtype).t() @ (S * A.to(dtype).t())

        # W, W_quant, A, B, S
        return dX.view(batch, seq_len, hd), \
            None, None, d_A.t(), d_B.t(), None
    pass
pass


def apply_lora_o(self, X):
    OW, OW_quant, OA, OB, OS = get_lora_parameters(self.o_proj)
    O = LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)
    return O
pass
