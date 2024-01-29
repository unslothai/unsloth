import logging

import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_triton import (
    QuantLinearFunction,
    QuantLinearInferenceOnlyFunction,
    quant_matmul_248,
    quant_matmul_inference_only_248,
    transpose_quant_matmul_248,
)

logger = logging.getLogger(__name__)
import math

"""
For testing only -- replaces HuggingFace default GPTQ QLinear layer (`cuda / torch` -> `triton`)
"""


# Adapted from https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/__init__.py
class GPTQuantLinear(nn.Linear):
    def __init__(self, quant_linear_module, trainable=True):
        if hasattr(quant_linear_module, "base_layer"):
            quant_linear_module = quant_linear_module.base_layer

        bias = (
            True
            if hasattr(quant_linear_module, "bias")
            and quant_linear_module.bias.count_nonzero() > 0
            else False
        )

        super().__init__(
            in_features=quant_linear_module.infeatures,
            out_features=quant_linear_module.outfeatures,
            bias=bias,
        )

        self.infeatures = quant_linear_module.infeatures
        self.outfeatures = quant_linear_module.outfeatures
        self.bits = quant_linear_module.bits
        self.group_size = quant_linear_module.group_size
        self.maxq = quant_linear_module.maxq

        self.weight.requires_grad = False

        self.weight.data = quant_linear_module.qweight
        self.register_buffer("qweight", quant_linear_module.qweight)
        if bias:
            self.bias.data = quant_linear_module.bias
            self.bias.requires_grad = False

        self.qweight.requires_grad = False

        self.register_buffer("qzeros", quant_linear_module.qzeros)
        self.register_buffer("scales", quant_linear_module.scales)
        self.register_buffer("g_idx", quant_linear_module.g_idx)

        if hasattr(quant_linear_module, "wf"):
            self.wf = quant_linear_module.wf
        if hasattr(quant_linear_module, "kernel_switch_threshold"):
            self.kernel_switch_threshold = quant_linear_module.kernel_switch_threshold
        if hasattr(quant_linear_module, "autogptq_cuda_available"):
            self.autogptq_cuda_available = quant_linear_module.autogptq_cuda_available

        self.trainable = trainable
        self.QUANT_TYPE = "triton"

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        quant_linear_fn = (
            QuantLinearFunction if self.trainable else QuantLinearInferenceOnlyFunction
        )
        out = quant_linear_fn.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.maxq,
        )
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out

        return out

    @classmethod
    def warmup(cls, model, transpose=True, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        from tqdm import tqdm

        assert cls.QUANT_TYPE == "triton"

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.outfeatures

            if (k, n) not in kn_values:
                kn_values[(k, n)] = (
                    m.qweight,
                    m.scales,
                    m.qzeros,
                    m.g_idx,
                    m.bits,
                    m.maxq,
                )

        logger.info(f"Found {len(kn_values)} unique KN Linear values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (
                    qweight,
                    scales,
                    qzeros,
                    g_idx,
                    bits,
                    maxq,
                ) in kn_values.items():
                    if transpose:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                        a = torch.randn(m, n, dtype=torch.float16, device=model.device)
                        transpose_quant_matmul_248(
                            a, qweight, scales, qzeros, g_idx, bits, maxq
                        )
                    else:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_inference_only_248(
                            a, qweight, scales, qzeros, g_idx, bits, maxq
                        )
        del kn_values

    @classmethod
    def inject_to_model(cls, model, target_module_type, **kwargs):
        count = 0
        for name, m in model.named_modules():
            if not isinstance(m, target_module_type):
                continue
            new_m = cls(m, **kwargs)
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, new_m)
            count += 1
        logger.warning_once(
            f"Injected {count} triton qlinear layers in place of {target_module_type} layers."
        )
