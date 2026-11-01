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
        backend = "torchao.prototype.blockwise_fp8_training.linear.Float8BlockwiseLinear",
        status = "planned",
        gates = {
            **{
                g: Gate(
                    "refused",
                    "torchao's DeepSeek 1x128 / 128x128 scaled_mm is SM90 only in torch "
                    "(Transformer Engine Float8BlockScaling does run on sm120)",
                    _PROBE_B200 + "; " + _PROBE_G4 + "; " + _PROBE_L4,
                )
                for g in ("sm75", "sm80", "sm89", "sm100", "sm120", "rocm")
            },
            "sm90": Gate("unprobed", "H100 is not available to test on"),
        },
        notes = "fp16 rejected by torchao; NVFP4 paper FP8 baseline keeps first 1 + last 2 blocks in bf16",
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
            **{
                g: Gate(
                    "refused",
                    "torchao's MXFP8 grouped kernels are SM100 only (EMULATED runs, no speed)",
                    _PROBE_G4 + "; " + _PROBE_L4,
                )
                for g in ("sm89", "sm120")
            },
            "sm100": Gate(
                "unprobed",
                "DSLRuntimeError with nvidia-cutlass-dsl 4.6.2 (" + _PROBE_B200 + "); "
                "pin a torchao / cutlass-dsl pair whose subprocess probe passes",
            ),
        },
        notes = "MoE experts; fallback is dequant + bf16 torch._grouped_mm",
    )
)

# ---- QAT (fake quant of the frozen or trained base; the GEMM itself runs in the activation dtype) --------
for _name, _dtypes, _note in (
    ("fp8-int4", ("bf16", "fp16", "fp32"), "Float8DynamicActivationInt4WeightConfig, group 128"),
    ("fp8-fp8", ("bf16", "fp16", "fp32"), "Float8DynamicActivationFloat8WeightConfig PerRow"),
    ("nvfp4", ("bf16", "fp32"), "NVFP4DynamicActivationNVFP4WeightConfig, K % 32"),
    ("mxfp8", ("bf16", "fp32"), "MXDynamicActivationMXWeightConfig, torchao 0.16+"),
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
                evidence = "RTX PRO 6000: 1.08-1.10x at Qwen3-0.6B, 1.17x at 1.7B; rowwise "
                "(fp8_lora_rowwise) is faster there. " + _FP8_LORA_RUNS,
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
        backend = "unsloth.kernels.fp8_lora (UNSLOTH_FP8_LORA_SCALING=row; the default on sm120)",
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
                evidence = "RTX PRO 6000, 16-bit master in host RAM (default): step 1.16x bf16 at Qwen3-0.6B "
                "(A/A 0.02%), 1.22x at 1.7B (A/A 0.17%); peak memory -12% / -33% vs bf16 LoRA. Eval loss "
                "(fp8 forward) +0.48-0.56% at 0.6B, +0.32-0.54% at 1.7B (bf16 seed band 0.02%); the trained "
                "adapter on the exact 16-bit base +0.06-0.12%. Explicit, not named: the fp8-forward eval gap "
                "passes jobs/compare.py (rtol 5%) but not the 0.5% bar at 0.6B. " + _FP8_LORA_RUNS,
            ),
            "rocm": Gate("refused", "ROCm fp8 (fnuz) GEMMs are not wired for the frozen base"),
        },
        notes = "per-token activation / gradient scales, the weight's tensorwise scale broadcast per column",
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
