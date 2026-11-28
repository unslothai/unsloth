# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One table entry per low-precision training format: what each GEMM runs in, how it is scaled, which
layers stay in high precision, which GradScaler it needs, and where it runs.

A recipe is data only. `hardware.check_recipe` turns the gate table into a refusal before the model is
touched, `selectors.high_precision_layers` applies the HP-layer policy, and `engage` checks that the
low-precision path really ran. The existing entry points (`float8_training`, `qat_scheme`, `load_in_fp8`) look
their recipe up here, so the table is the one place that says what is supported.

GEMM names: fwd = y = x W^T, dgrad = dx = dy W, wgrad = dW = dy^T x.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Tuple

__all__ = [
    "Operand",
    "GemmPolicy",
    "HPLayerPolicy",
    "Gate",
    "Recipe",
    "RECIPES",
    "get_recipe",
    "recipes_of_kind",
    "named_recipes",
    "measured_faster",
]


@dataclasses.dataclass(frozen = True)
class Operand:
    """One GEMM operand: element dtype and how its scale is computed."""

    dtype: str  # e4m3, e5m2, e2m1, bf16, fp16, fp32, int4, int8, input (= the activation dtype)
    granularity: str = "none"  # none, tensor, row (axiswise), block1x128, block128x128, block1x32, block1x16, block16x16, group32, group128
    scale_dtype: str = "none"  # fp32, e8m0, e4m3, none
    rounding: str = (
        "rn"  # rn, sr (stochastic), rceil (MX: scale rounded up), pow2 (power-of-two scales)
    )
    rht: bool = False  # random Hadamard transform before quantizing (NVFP4 wgrad)


@dataclasses.dataclass(frozen = True)
class GemmPolicy:
    """(a, b) operands of one GEMM; None = this GEMM runs in the activation dtype (high precision)."""

    a: Optional[Operand] = None
    b: Optional[Operand] = None

    @property
    def low_precision(self) -> bool:
        return self.a is not None or self.b is not None


HP = GemmPolicy()


@dataclasses.dataclass(frozen = True)
class HPLayerPolicy:
    """Which linear layers stay in high precision. Detected by structure (selectors.py), not by exact names."""

    output_head: bool = True  # lm_head / get_output_embeddings and any weight tied to it
    embeddings: bool = True  # nn.Embedding and get_input_embeddings
    routers: bool = (
        True  # MoE routers: out_features == number of experts beside an experts container
    )
    first_n: int = 0  # first N decoder layers
    last_n: int = 0  # last N decoder layers
    last_frac: float = 0.0  # or the last fraction of decoder layers (NVFP4 ~0.15), rounded up
    shared_experts: bool = False  # MoE shared expert (humans& NVFP4-RL keeps it bf16)


@dataclasses.dataclass(frozen = True)
class Gate:
    """Status of a recipe on one GPU class.

    status: named   = runs by name (measured faster at equal loss),
            explicit = runs only with an explicit torchao config (works, not measured faster),
            refused  = raises with `reason` before the model is touched,
            unprobed = never run on this class; treated as explicit with a warning.
    """

    status: str
    reason: str = ""
    evidence: str = ""


@dataclasses.dataclass(frozen = True)
class Recipe:
    name: str
    kind: str  # dtype, float8_training, qat, frozen_base
    fwd: GemmPolicy = HP
    dgrad: GemmPolicy = HP
    wgrad: GemmPolicy = HP
    hp_layers: HPLayerPolicy = HPLayerPolicy()
    input_dtypes: Tuple[str, ...] = ("bf16", "fp16", "fp32")
    modes: Tuple[str, ...] = ("full",)  # full, lora, qlora
    grad_scaler: str = "none"  # none, dynamic (fp16 loss scaling), per_dtype (dynamic only when the activations are fp16)
    compile: str = (
        "ok"  # ok, opaque (wrapped in a custom op inside compiled regions), fails, unknown
    )
    backend: str = ""  # entry point that implements it
    status: str = "shipped"  # shipped, planned, refused
    refusal: str = ""  # reason when status == refused (for every GPU)
    gates: Mapping[str, Gate] = dataclasses.field(default_factory = dict)  # gpu class -> Gate
    alias_of: str = ""
    notes: str = ""

    def gate(self, gpu_class: str) -> Gate:
        if self.status == "refused":
            return Gate("refused", self.refusal)
        return self.gates.get(gpu_class, Gate("unprobed"))


E4M3 = "e4m3"
E5M2 = "e5m2"


def _op(
    dtype,
    gran,
    scale = "fp32",
    rounding = "rn",
    rht = False,
):
    return Operand(dtype, gran, scale, rounding, rht)


_NO_FP8_HW = Gate("refused", "needs fp8 tensor cores (compute capability 8.9+)")
_NO_MX_HW = Gate("refused", "needs block-scaled tensor cores (compute capability 10.0+ / 12.0)")
_HP_FP8 = HPLayerPolicy(output_head = True, embeddings = True, routers = True)

# GPU classes: sm75 (T4), sm80 (A100), sm89 (L4 / RTX 40), sm90 (H100), sm100 (B200 / GB200),
# sm120 (RTX PRO 6000 / RTX 50), rocm. Evidence strings name the probe that set the gate.
_PROBE_G4 = "probed on an RTX PRO 6000 (sm120), torch 2.13.0, torchao 0.18.0"
_PROBE_L4 = "probed on an L4 (sm89), torch 2.13.0, torchao 0.18.0"
_PROBE_B200 = "probed on a B200 (sm100), torch 2.13, torchao 0.18"
_PROBE_G4_214 = "RTX PRO 6000 (sm120), torch 2.14.1, torchao 0.18.0, Triton 3.8.0"
_PROBE_L4_214 = "L4 (sm89), torch 2.14.1, torchao 0.18.0, Triton 3.8.0"
_PROBE_B200_214 = "B200 (sm100), torch 2.14.1, torchao 0.18.0, correctness only"
_BLOCKWISE_SCALED_MM = (
    "torch's 1x128 / 128x128 scaled_mm is SM90 only (eager NotImplementedError, compiled "
    "CUBLAS_STATUS_NOT_SUPPORTED)"
)

RECIPES = {}


def _add(recipe: Recipe):
    assert recipe.name not in RECIPES, recipe.name
    RECIPES[recipe.name] = recipe
    return recipe


# ---- plain dtypes -----------------------------------------------------------------------------------
_add(
    Recipe(
        name = "fp16",
        kind = "dtype",
        input_dtypes = ("fp16",),
        modes = ("full", "lora", "qlora"),
        grad_scaler = "dynamic",
        backend = "torch.amp.GradScaler (fsdp2_native._ShardedGradScaler under FSDP2)",
        gates = {
            g: Gate("named") for g in ("sm75", "sm80", "sm89", "sm90", "sm100", "sm120", "rocm")
        },
        notes = "fp32 masters for trainable params; Megatron default scaler init 2^32, window 1000, hysteresis 2",
    )
)
_add(
    Recipe(
        name = "bf16",
        kind = "dtype",
        input_dtypes = ("bf16",),
        modes = ("full", "lora", "qlora"),
        gates = {
            "sm75": Gate("refused", "no bfloat16 tensor cores; Unsloth downgrades to float16"),
            **{g: Gate("named") for g in ("sm80", "sm89", "sm90", "sm100", "sm120", "rocm")},
        },
    )
)

# ---- torchao float8 training (full finetuning) ------------------------------------------------------
_add(
    Recipe(
        name = "fp8_tensorwise",
        kind = "float8_training",
        fwd = GemmPolicy(_op(E4M3, "tensor"), _op(E4M3, "tensor")),
        dgrad = GemmPolicy(_op(E5M2, "tensor"), _op(E4M3, "tensor")),
        wgrad = GemmPolicy(_op(E5M2, "tensor"), _op(E4M3, "tensor")),
        hp_layers = _HP_FP8,
        grad_scaler = "per_dtype",
        backend = "torchao.float8.Float8LinearConfig.from_recipe_name('tensorwise')",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "refused",
                "NaN gradients on an L4: the compiled bfloat16 weight gradient is NaN at "
                "2048 x 4096 x 4096 (3 of 3 seeds) and was 0.50 relative off at 512 x 1024 x 2048 once, while "
                "eager and the compiled forward / input gradient are correct",
                _PROBE_L4,
            ),
            "sm90": Gate("explicit"),
            "sm100": Gate("explicit", evidence = _PROBE_B200),
            "sm120": Gate(
                "named",
                evidence = "Qwen3 full FT 15% faster on RTX PRO 6000 (float8_training.py); "
                "compiled linear fwd+bwd 1.56-1.61x bf16 at 4096 x 14336 (" + _PROBE_G4 + ")",
            ),
        },
        alias_of = "",
        notes = "float8_training = True",
    )
)
_add(
    Recipe(
        name = "fp8_rowwise",
        kind = "float8_training",
        fwd = GemmPolicy(_op(E4M3, "row", rounding = "pow2"), _op(E4M3, "row", rounding = "pow2")),
        dgrad = GemmPolicy(_op(E4M3, "row", rounding = "pow2"), _op(E4M3, "row", rounding = "pow2")),
        wgrad = GemmPolicy(_op(E4M3, "row", rounding = "pow2"), _op(E4M3, "row", rounding = "pow2")),
        hp_layers = _HP_FP8,
        grad_scaler = "per_dtype",
        backend = "torchao.float8.Float8LinearConfig.from_recipe_name('rowwise')",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "explicit",
                evidence = "2% slower than bf16 on an L4 (float8_training.py); compiled linear "
                "0.69-1.13x bf16 (" + _PROBE_L4 + ")",
            ),
            "sm90": Gate("explicit"),
            "sm100": Gate("explicit", evidence = _PROBE_B200),
            "sm120": Gate(
                "explicit",
                evidence = "3% slower than bf16 on RTX PRO 6000 (float8_training.py); compiled "
                "linear 0.54-0.93x bf16 (" + _PROBE_G4 + ")",
            ),
        },
    )
)
_add(
    Recipe(
        name = "fp8_rowwise_gw_hp",
        kind = "float8_training",
        fwd = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "row")),
        dgrad = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "tensor")),
        wgrad = HP,
        hp_layers = _HP_FP8,
        grad_scaler = "per_dtype",
        backend = "torchao.float8.Float8LinearConfig.from_recipe_name('rowwise_with_gw_hp')",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "explicit", evidence = "compiled linear 0.93-1.14x bf16 (" + _PROBE_L4 + ")"
            ),
            "sm90": Gate("explicit"),
            "sm100": Gate("explicit", evidence = _PROBE_B200),
            "sm120": Gate(
                "explicit", evidence = "compiled linear 0.86-1.24x bf16 (" + _PROBE_G4 + ")"
            ),
        },
    )
)
_add(
    Recipe(
        name = "fp8_blockwise",
        kind = "float8_training",
        fwd = GemmPolicy(_op(E4M3, "block1x128"), _op(E4M3, "block128x128")),
        dgrad = GemmPolicy(_op(E4M3, "block1x128"), _op(E4M3, "block128x128")),
        wgrad = GemmPolicy(_op(E4M3, "block1x128"), _op(E4M3, "block128x1")),
        hp_layers = HPLayerPolicy(first_n = 1, last_n = 2),
        input_dtypes = ("bf16",),
        compile = "opaque",
        backend = "torchao.prototype.blockwise_fp8_training.linear.Float8BlockwiseLinear "
        "(use_triton = True off sm90)",
        status = "planned",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "explicit",
                evidence = _BLOCKWISE_SCALED_MM
                + "; torchao's Triton GEMMs run (linear rel err 0.036 fwd / dgrad / wgrad) at 0.75-0.98x "
                "bf16 fwd+bwd on Qwen3 0.6B / 1.7B MLP shapes (8192 tokens) and 0.91x at 4096 x 14336, where "
                "fp8_tensorwise is 1.25-1.59x. 100-step Qwen3-0.6B full finetune, bs 4 x 1024, every arm "
                "compiled: step 0.97x bf16 (bf16 A/A 0.8%), fp8_tensorwise 1.06x; eval loss +0.31% vs bf16 "
                "(fp8_tensorwise +0.44%) (" + _PROBE_L4_214 + ")",
            ),
            "sm90": Gate("unprobed", "H100 is not available to test on"),
            "sm100": Gate(
                "explicit",
                evidence = _BLOCKWISE_SCALED_MM
                + "; torchao's Triton GEMMs run (linear rel err 0.036, rowwise 0.036-0.038); not timed ("
                + _PROBE_B200_214
                + ")",
            ),
            "sm120": Gate(
                "explicit",
                evidence = _BLOCKWISE_SCALED_MM
                + "; torchao's Triton GEMMs run (linear rel err 0.036 fwd / dgrad / wgrad) at 0.58-0.94x "
                "bf16 fwd+bwd on Qwen3 0.6B / 1.7B MLP shapes and 0.99-1.01x at 4096 x 14336 (fp8_tensorwise "
                "1.06-1.56x). 100-step full finetune, bs 4 x 1024, every arm compiled: Qwen3-0.6B step 0.96x "
                "bf16 (bf16 A/A 0.12%), Qwen3-1.7B 0.98x (A/A 0.05%), where fp8_tensorwise is 1.02x / 1.10x; "
                "eval loss +0.27% / +0.19% vs bf16 (fp8_tensorwise +0.45% / +0.35%, bf16 A/A 0.01% / 0.03%). "
                "Most accurate fp8 recipe here, never faster than bf16 (" + _PROBE_G4_214 + ")",
            ),
            "rocm": Gate(
                "refused",
                "not probed on ROCm; torch's 1x128 / 128x128 scaled_mm is CUDA SM90 only",
            ),
        },
        notes = "fp16 rejected by torchao; NVFP4 paper FP8 baseline keeps first 1 + last 2 blocks in bf16. "
        "Inductor cannot re-emit torchao's Triton GEMM (compile error), so it runs as an opaque op under "
        "compile; it saves the 16-bit input and weight for backward, so peak memory equals bf16",
    )
)

# ---- microscaling formats ---------------------------------------------------------------------------
_MX = _op(E4M3, "block1x32", "e8m0", "rceil")
_add(
    Recipe(
        name = "mxfp8",
        kind = "float8_training",
        fwd = GemmPolicy(_MX, _MX),
        dgrad = GemmPolicy(_MX, _MX),
        wgrad = GemmPolicy(_MX, _MX),
        hp_layers = HPLayerPolicy(output_head = True, embeddings = True, routers = True),
        input_dtypes = ("bf16",),
        grad_scaler = "none",
        backend = "torchao.prototype.moe_training.config.MXFP8TrainingOpConfig (MXFP8_RCEIL)",
        status = "planned",
        gates = {
            "sm75": _NO_MX_HW,
            "sm80": _NO_MX_HW,
            "sm89": Gate(
                "refused",
                "torchao MXFP8 kernels need block-scaled tensor cores (L4: 'needs triton')",
                _PROBE_L4,
            ),
            "sm90": _NO_MX_HW,
            "sm100": Gate("explicit", evidence = _PROBE_B200),
            "sm120": Gate(
                "explicit",
                evidence = "linear fwd / dgrad / wgrad rel err 0.038 (= fp8 rowwise), eager and "
                "compiled; compiled 1.49-1.55x bf16 at 4096 x 14336 ("
                + _PROBE_G4
                + "). Transformer "
                "Engine 2.20 refuses MXFP8 on 12.0+",
            ),
        },
        notes = "arXiv 2506.08027: e4m3 for every tensor incl. gradients, scale rounded up (RCEIL), all blocks quantized; "
        "torchao to_mx rejects float16 inputs",
    )
)
_add(
    Recipe(
        name = "mxfp6",
        kind = "float8_training",
        status = "refused",
        refusal = "MXFP6 has no training kernel in torchao, Transformer Engine or Megatron (only QAT / emulation), "
        "and no tensor-core speed over MXFP8 on Blackwell. Use mxfp8, or qat_scheme for MXFP6 deployment.",
    )
)
_NV_X = _op("e2m1", "block1x16", "e4m3", "rn")
_NV_W = _op("e2m1", "block16x16", "e4m3", "rn")
_NV_G = _op("e2m1", "block1x16", "e4m3", "sr")
_add(
    Recipe(
        name = "nvfp4",
        kind = "float8_training",
        fwd = GemmPolicy(_NV_X, _NV_W),
        dgrad = GemmPolicy(_NV_G, _NV_W),
        wgrad = GemmPolicy(
            _op("e2m1", "block1x16", "e4m3", "sr", rht = True),
            _op("e2m1", "block1x16", "e4m3", "rn", rht = True),
        ),
        hp_layers = HPLayerPolicy(output_head = True, embeddings = True, routers = True, last_frac = 0.15),
        input_dtypes = ("bf16",),
        grad_scaler = "none",
        compile = "opaque",
        backend = "torchao.prototype.moe_training.nvfp4_training.NVFP4TrainingConfig",
        status = "planned",
        gates = {
            "sm75": _NO_MX_HW,
            "sm80": _NO_MX_HW,
            "sm89": Gate("refused", "torchao NVFP4 kernels require SM100+", _PROBE_L4),
            "sm90": _NO_MX_HW,
            "sm100": Gate("explicit", evidence = _PROBE_B200 + " (eager only)"),
            "sm120": Gate(
                "refused",
                "torchao NVFP4 training does not build on sm_120a (Triton: cvt.rs stochastic "
                "rounding unsupported; Inductor cannot lower the swizzled scaled_mm). Transformer Engine "
                "2.20 NVFP4BlockScaling runs in bfloat16 (rel err 0.15-0.18, linear fwd+bwd 2.1-2.4x bf16) "
                "and is the sm120 route",
                _PROBE_G4,
            ),
        },
        notes = "arXiv 2509.25149: 16-element blocks, e4m3 block scale + fp32 tensor scale, 2D 16x16 weights, "
        "RHT 16x16 on wgrad inputs only, SR on gradients only, ~15% of layers (mostly last) in bf16. "
        "Inductor cannot lower its swizzled scaled_mm, so it runs as an opaque op under compile.",
    )
)

# ---- MoE grouped experts (torch._grouped_mm in low precision) -----------------------------------------
_add(
    Recipe(
        name = "fp8_rowwise_grouped",
        kind = "float8_training",
        fwd = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "row")),
        dgrad = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "row")),
        wgrad = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "row")),
        input_dtypes = ("bf16",),
        backend = "torchao.prototype.moe_training.config.Float8TrainingOpConfig",
        status = "planned",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            **{
                g: Gate(
                    "refused",
                    "torch._scaled_grouped_mm runs on compute capability 9.0 / 10.0 only",
                    _PROBE_G4 + "; " + _PROBE_L4,
                )
                for g in ("sm89", "sm120")
            },
            "sm90": Gate("unprobed"),
            "sm100": Gate(
                "refused",
                "cutlass cannot run (error 7); an unspecified launch failure poisons the CUDA "
                "context when run in a shared process",
                _PROBE_B200,
            ),
        },
        notes = "MoE experts (3-D weights); fallback is dequant + bf16 torch._grouped_mm",
    )
)
_add(
    Recipe(
        name = "mxfp8_grouped",
        kind = "float8_training",
        fwd = GemmPolicy(_MX, _MX),
        dgrad = GemmPolicy(_MX, _MX),
        wgrad = GemmPolicy(_MX, _MX),
        input_dtypes = ("bf16",),
        backend = "torchao.prototype.moe_training.config.MXFP8TrainingOpConfig (grouped)",
        status = "planned",
        gates = {
            "sm75": _NO_MX_HW,
            "sm80": _NO_MX_HW,
            "sm90": _NO_MX_HW,
            "sm89": Gate(
                "refused",
                "needs block-scaled tensor cores; torch._scaled_grouped_mm runs on compute capability "
                "9.0 / 10.0 only",
                _PROBE_L4_214,
            ),
            "sm120": Gate(
                "refused",
                "torch._scaled_grouped_mm runs on compute capability 9.0 / 10.0 only (MX operands too); a "
                "per-expert loop of dense MX torch._scaled_mm is 0.22-0.29x bf16 torch._grouped_mm at "
                "Qwen3-30B-A3B expert shapes and 0.48-0.57x at gpt-oss-20b, before its host sync",
                _PROBE_G4_214,
            ),
            "sm100": Gate(
                "explicit",
                evidence = "Unsloth's own MX quantizer (e4m3, e8m0 RCEIL per 1 x 32, bit-equal to torchao "
                "to_mx) and device-side blocked scale layouts over torch._scaled_grouped_mm: fwd, dgrad and "
                "wgrad run at Qwen3-30B-A3B (128 experts, 2048 -> 1536, 768 -> 2048) and gpt-oss-20b "
                "(32 experts, 2880 -> 5760, 2880 -> 2880) expert shapes, uneven and empty groups, within "
                "1e-5 of bf16 grouped GEMMs on the dequantized operands, rel err 0.038 vs bf16. Tiny MoE, "
                "100 steps: MX fwd + dgrad with bf16 wgrad trains within the bf16 seed spread (full "
                "finetune and bf16 expert LoRA on top); MX wgrad lands 0.4% higher, outside it. Expert "
                "gradients 5x bf16's error vs fp32 (median 0.08-0.10 per expert), set by the MX forward. "
                "Not timed (" + _PROBE_B200_214 + ")",
            ),
        },
        notes = "MoE experts; fallback is dequant + bf16 torch._grouped_mm. torchao's own grouped path needs its "
        "CuTe DSL quantizers (DSLRuntimeError with nvidia-cutlass-dsl 4.6.2 on a B200, torch 2.13)",
    )
)

# ---- QAT (fake quant of the frozen or trained base; the GEMM itself runs in the activation dtype) --------
for _name, _dtypes, _note in (
    ("fp8-int4", ("bf16", "fp16", "fp32"), "Float8DynamicActivationInt4WeightConfig, group 128"),
    ("fp8-fp8", ("bf16", "fp16", "fp32"), "Float8DynamicActivationFloat8WeightConfig PerRow"),
    ("nvfp4", ("bf16", "fp32"), "NVFP4DynamicActivationNVFP4WeightConfig, K % 32"),
    ("mxfp8", ("bf16",), "MXDynamicActivationMXWeightConfig, torchao 0.16+, bf16 layers only"),
    ("int8-int4", ("bf16", "fp16", "fp32"), "phone-deployment alias"),
    ("int4", ("bf16", "fp16", "fp32"), "Int4WeightOnlyConfig, group 128"),
    ("int8", ("bf16", "fp16", "fp32"), "Int8 weight only"),
    ("cactus", ("bf16", "fp16", "fp32"), "Cactus int8 / int4 per group"),
):
    _add(
        Recipe(
            name = "qat:" + _name,
            kind = "qat",
            input_dtypes = _dtypes,
            modes = ("full", "lora"),
            grad_scaler = "per_dtype",
            compile = "opaque",
            backend = "unsloth.models._utils._prepare_model_for_qat",
            gates = {},
            notes = _note,
        )
    )
_add(
    Recipe(
        name = "qat:phone-deployment",
        kind = "qat",
        alias_of = "qat:int8-int4",
        modes = ("full", "lora"),
        backend = "unsloth.models._utils._prepare_model_for_qat",
    )
)

# ---- frozen low-precision bases for LoRA (fwd low precision, dgrad dequantized, no wgrad) -------------
_add(
    Recipe(
        name = "frozen_fp8_rowwise",
        kind = "frozen_base",
        modes = ("lora",),
        fwd = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "row")),
        backend = "unsloth.kernels.fp8 (load_in_fp8 = True)",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate("named"),
            "sm90": Gate("named"),
            "sm100": Gate("named"),
            "sm120": Gate("named"),
        },
        notes = "dgrad dequantizes the fp8 weight to 16 bit; fast accum in the forward",
    )
)
_add(
    Recipe(
        name = "frozen_fp8_block",
        kind = "frozen_base",
        modes = ("lora",),
        fwd = GemmPolicy(_op(E4M3, "block1x128"), _op(E4M3, "block128x128")),
        backend = "unsloth.kernels.fp8 (load_in_fp8 = 'block')",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate("named"),
            "sm90": Gate("named"),
            "sm100": Gate("named"),
            "sm120": Gate("named"),
        },
        notes = "Triton block kernels, forward only; dgrad dequantizes",
    )
)

_W4A16_GATES = {
    "sm75": Gate("unprobed", "fp16 activations (no bfloat16 there)"),
    "sm80": Gate("unprobed"),
    "sm89": Gate("unprobed"),
    "sm90": Gate("unprobed", "H100 is not available to test on"),
    "sm100": Gate(
        "explicit", evidence = "LoRA correctness on tiny Qwen3 / Qwen3-MoE, " + _PROBE_B200
    ),
    "sm120": Gate("unprobed"),
    "rocm": Gate("unprobed", "the Triton dequant runs; the fused kernel is off on ROCm"),
}
for _fmt, _gran, _scale, _note in (
    ("nvfp4", "block1x16", "e4m3", "fp32 global scale per tensor, 4over6 block scales by default"),
    (
        "mxfp4",
        "block1x32",
        "e8m0",
        "OCP MXFP4 values in the NVFP4 kernel layout (e8m0 scales stored as e4m3 under a power-of-two "
        "global scale, bfloat16 when the block exponents span more than 17 octaves)",
    ),
):
    _add(
        Recipe(
            name = f"onthefly_{_fmt}_w4a16",
            kind = "frozen_base",
            modes = ("lora",),
            fwd = GemmPolicy(None, _op("e2m1", _gran, _scale)),
            dgrad = GemmPolicy(None, _op("e2m1", _gran, _scale)),
            input_dtypes = ("bf16", "fp16"),
            compile = "opaque",
            backend = f"unsloth.kernels.nvfp4 (load_in_4bit = '{_fmt}': quantized at load from a 16-bit checkpoint)",
            gates = dict(_W4A16_GATES),
            notes = "W4A16: the weight is dequantized to the activation dtype (or decoded inside the fused "
            "Triton GEMM), no FP4 tensor-core GEMM; " + _note,
        )
    )

_FP8_LORA_RUNS = (
    "Qwen3 LoRA r=16, seq 2048, bs 2, 200 steps on alpaca-cleaned, eval every 50 steps on 32 held-out "
    "rows, 2 seeds, vs bf16 LoRA in the same Colab VM"
)
_add(
    Recipe(
        name = "fp8_lora_tensorwise",
        kind = "frozen_base",
        modes = ("lora",),
        fwd = GemmPolicy(_op(E4M3, "tensor"), _op(E4M3, "tensor")),
        dgrad = GemmPolicy(_op(E5M2, "tensor"), _op(E4M3, "tensor")),
        hp_layers = _HP_FP8,
        input_dtypes = ("bf16", "fp16"),
        grad_scaler = "per_dtype",
        compile = "opaque",
        backend = "unsloth.kernels.fp8_lora (float8_training with LoRA on a 16-bit base)",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "explicit",
                evidence = "L4, 16-bit master in host RAM (default): step 1.15x bf16 at Qwen3-0.6B (A/A 1.0%), "
                "1.30x at 1.7B (A/A 0.05%); peak memory -11% / -33% vs bf16 LoRA. Eval loss (fp8 forward) "
                "+0.53-0.64% at 0.6B, +0.39-0.55% at 1.7B (bf16 seed band 0.03%); the trained adapter on the "
                "exact 16-bit base +0.07-0.17%. Explicit, not named: the fp8-forward eval gap passes "
                "jobs/compare.py (rtol 5%) but not the 0.5% bar at 0.6B. Saved and merged weights bitwise equal "
                "to the checkpoint. " + _FP8_LORA_RUNS,
            ),
            "sm90": Gate("unprobed", "H100 is not available to test on"),
            "sm100": Gate("unprobed"),
            "sm120": Gate(
                "explicit",
                evidence = "RTX 5090 (Windows, torch 2.14.1, which has no row-wise _scaled_mm kernel; bs 4 x "
                "2048): step 1.12x bf16 at Qwen3-1.7B and 1.27x at Qwen3-8B (A/A under 0.1%), behind "
                "int8_lora_rotated (1.29x / 1.59x). At bs 8 x 512 on torch 2.13 it ran 0.66x bf16 at "
                "Qwen3-1.7B (A/A 1.3%). RTX PRO 6000, earlier probe ("
                + _FP8_LORA_RUNS
                + "): 1.08-1.10x at "
                "Qwen3-0.6B and 1.17x at 1.7B. This has not been re-measured. The rowwise recipe's matching "
                "1.22x at 1.7B re-measured at 0.99x on torch 2.14.1 (see fp8_lora_rowwise)",
            ),
            "rocm": Gate("refused", "ROCm fp8 (fnuz) GEMMs are not wired for the frozen base"),
        },
        notes = "base GEMMs fp8, LoRA branch 16-bit (Megatron-Bridge TEFusedLoRALinear split); weight "
        "quantized once; the fp8 GEMM writes bf16 for fp16 activations",
    )
)
_add(
    Recipe(
        name = "fp8_lora_rowwise",
        kind = "frozen_base",
        modes = ("lora",),
        fwd = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "tensor")),
        dgrad = GemmPolicy(_op(E4M3, "row"), _op(E4M3, "tensor")),
        hp_layers = _HP_FP8,
        input_dtypes = ("bf16", "fp16"),
        grad_scaler = "per_dtype",
        compile = "opaque",
        backend = "unsloth.kernels.fp8_lora (UNSLOTH_FP8_LORA_SCALING=row; the default on sm120; torch builds "
        "without the row-wise _scaled_mm kernel, the Windows wheels, run its GEMMs at unit scale on the "
        "tensorwise kernel)",
        gates = {
            "sm75": _NO_FP8_HW,
            "sm80": _NO_FP8_HW,
            "sm89": Gate(
                "explicit",
                evidence = "L4: rowwise _scaled_mm 0.55-0.90x bf16 at op level, no faster than bf16 LoRA "
                "end to end; use fp8_lora_tensorwise there. " + _FP8_LORA_RUNS,
            ),
            "sm90": Gate("unprobed", "H100 is not available to test on"),
            "sm100": Gate("unprobed"),
            "sm120": Gate(
                "explicit",
                evidence = "RTX PRO 6000 on a torch build with the row-wise _scaled_mm kernel (Linux; the Windows "
                "wheels run unit-scale tensorwise GEMMs instead; see fp8_lora_tensorwise), default "
                "settings, torch 2.14.1, bs 4 x 2048, 25 steps: step 0.99x bf16 at Qwen3-1.7B, 1.02x at "
                "Qwen3-4B and 1.13x at Qwen3-8B (bf16 A/A 0.09% / 0.16% / 0.69%). It only pays at about 8B, "
                "and int8_lora_rotated is faster at each size (1.05x / 1.11x / 1.22x). Peak memory is 8% "
                "lower than bf16 LoRA at 1.7B and 13% lower at 8B. An earlier probe on the setup below "
                "measured 1.16x at Qwen3-0.6B and 1.22x at 1.7B, which no longer reproduces. Eval loss "
                "(fp8 forward) +0.48-0.56% at 0.6B, +0.32-0.54% at 1.7B (bf16 seed band 0.02%); the trained "
                "adapter on the exact 16-bit base +0.06-0.12%. Explicit, not named: the fp8-forward eval gap "
                "passes jobs/compare.py (rtol 5%) but not the 0.5% bar at 0.6B. " + _FP8_LORA_RUNS,
            ),
            "rocm": Gate("refused", "ROCm fp8 (fnuz) GEMMs are not wired for the frozen base"),
        },
        notes = "per-token activation / gradient scales, the weight's tensorwise scale broadcast per column",
    )
)

_add(
    Recipe(
        name = "int8_lora_rotated",
        kind = "frozen_base",
        modes = ("lora",),
        fwd = GemmPolicy(_op("int8", "row", rht = True), _op("int8", "row", rht = True)),
        dgrad = GemmPolicy(_op("int8", "row"), _op("int8", "row", rht = True)),
        hp_layers = _HP_FP8,
        grad_scaler = "per_dtype",  # fp16 activations need loss scaling, as for fp8 LoRA
        input_dtypes = ("bf16", "fp16"),
        compile = "opaque",
        backend = "unsloth.kernels.int8_lora (UNSLOTH_LORA_BASE_INT8=1, or float8_training = 'int8_lora')",
        gates = {
            "sm75": Gate(
                "refused",
                "T4: Triton (3.8) does not compile int8 dots for sm_75. cuBLAS int8 (torch._int_mm) runs at "
                "1.76x / 1.81x an fp16 GEMM there (geomean over Qwen3 1.7B-8B layers, forward / input gradient), "
                "but the block rotation and its undo in the input gradient have no fast Triton path on sm_75 "
                "(over its shared memory, or no tensor cores: the un-rotation epilogue alone takes 4-80x the "
                "whole fp16 GEMM)",
            ),
            "sm80": Gate(
                "explicit",
                evidence = "A100 40GB, Unsloth LoRA r=16 on all linears, bs x 2048 tokens, 25 steps, fresh process "
                "per run, interleaved: 1.045x / 1.109x / 1.156x bf16 step time at Qwen3-1.7B (bs 4) / 4B (bs 3) "
                "/ 8B (bs 2), bf16 A/A 0.1-0.2% over 3 runs (1.7B is faster but under A/A + 5%); peak memory 8-21% lower. GEMMs alone (geomean): 1.5x "
                "bf16 forward, 1.09x input gradient once Triton 3.8's prefetch split is skipped (forward 0.2x "
                "before). Eval "
                "loss after 100 steps at Qwen3-1.7B vs the bf16 mean (A/A 0.03%): +0.14% with the int8 forward, +0.05% "
                "on the exact base. sm86 / sm87 (A10, RTX 30, Orin) share this gate untimed",
            ),
            "sm89": Gate(
                "explicit",
                evidence = "L4, same setup: 1.19x bf16 step time at Qwen3-1.7B (bs 4, bf16 A/A 4.9%) and 1.19x "
                "at Qwen3-4B (bs 1, A/A 0.3%); an earlier VM gave 1.25x / 1.22x; Qwen3-8B does not fit 16-bit LoRA on "
                "24 GB. GEMMs alone (geomean): 1.7x bf16 forward, 1.5x input gradient once Triton 3.8's "
                "prefetch split is skipped (forward 0.3x before). Numerics are bit-identical to the A100 / sm120 kernels",
            ),
            "sm90": Gate("unprobed", "H100 is not available to test on"),
            "sm100": Gate("unprobed", "B200: correctness only, never timed"),
            "sm120": Gate(
                "explicit",
                evidence = "RTX 5090 (GeForce: FP32-accumulate bf16 / fp8 tensor paths run at half rate): "
                "the int8 GEMM is 3.03x a bf16 GEMM and 1.54x fp8 tensorwise over a Qwen3-8B layer at 8192 "
                "tokens (A/A 0.25%). RTX PRO 6000: int8 = fp8 rowwise = ~1.9x bf16 per GEMM. Eval loss after "
                "100 LoRA steps vs bf16 (A/A 0.02-0.05%): +0.02% on the exact base, +0.16% with the int8 "
                "forward at Qwen3-1.7B (fp8 rowwise: +0.10% / +0.42%)",
            ),
            "rocm": Gate(
                "refused",
                "the int8 kernels are CUDA Triton only; on gfx1151 (Radeon 8060S, ROCm 7.13, Triton 3.6) the best "
                "int8 tiles over Qwen3 4B / 8B layers at 4k / 8k tokens are 0.94x bf16 for the forward and 0.80x "
                "for the input gradient with the activation quantization (1.08x / 0.84x the GEMMs alone)",
            ),
        },
        notes = "block-Hadamard (128) rotated per-token int8 activations and per-row int8 weight, int32 "
        "accumulation, dequant and LoRA term in the GEMM epilogue; the input gradient reuses the same int8 "
        "weight (row scale folded into dY, rotation undone in the epilogue); q / k / v and gate / up share "
        "one activation quantization",
    )
)


def get_recipe(name: str) -> Recipe:
    try:
        recipe = RECIPES[name]
    except KeyError:
        raise KeyError(
            f"Unsloth: unknown precision recipe {name!r}; known: {sorted(RECIPES)}"
        ) from None
    return RECIPES[recipe.alias_of] if recipe.alias_of else recipe


def recipes_of_kind(kind: str):
    return {n: r for n, r in RECIPES.items() if r.kind == kind}


# float8_training = '<name>' accepts these spellings; a name maps to its recipe.
FLOAT8_TRAINING_NAMES = {
    "tensorwise": "fp8_tensorwise",
    "rowwise": "fp8_rowwise",
    "rowwise_with_gw_hp": "fp8_rowwise_gw_hp",
    "blockwise": "fp8_blockwise",
    "mxfp8": "mxfp8",
    "mxfp6": "mxfp6",
    "nvfp4": "nvfp4",
}


def named_recipes(kind: str = "float8_training"):
    """Float8-training names that run by name somewhere (some gate is `named`)."""
    out = []
    for short, full in FLOAT8_TRAINING_NAMES.items():
        r = RECIPES[full]
        if (
            r.kind == kind
            and r.status == "shipped"
            and any(g.status == "named" for g in r.gates.values())
        ):
            out.append(short)
    return tuple(out)


_CLASS_CC = {
    "sm120": ((12, 0), (12, 1)),
    "sm100": ((10, 0), (10, 3)),
    "sm90": ((9, 0),),
    "sm89": ((8, 9),),
}


def measured_faster(kind: str = "float8_training"):
    """{short name: compute capabilities where it runs by name}, the shape float8_training._MEASURED_FASTER had."""
    out = {}
    for short in named_recipes(kind):
        r = RECIPES[FLOAT8_TRAINING_NAMES[short]]
        ccs = tuple(
            cc
            for g, gate in r.gates.items()
            if gate.status == "named"
            for cc in _CLASS_CC.get(g, ())
        )
        if ccs:
            out[short] = ccs
    return out
