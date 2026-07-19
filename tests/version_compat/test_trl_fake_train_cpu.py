# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Fake CPU training runs for the Unsloth-patched SFT / GRPO / DPO trainers.

The patch-run canary (test_trl_grpo_fake_run.py) only compiles + inspects the
generated trainer source. This goes one layer deeper: it actually runs
`trainer.train()` for a couple of steps on a CPU-only runner, under the CUDA
spoof, wrapping a plain (tiny, random-weight) HF model in the Unsloth-patched
trainer. That exercises the real train() loop at runtime -- data collation,
generation (GRPO), the injected `_get_per_token_logps_and_entropies`, loss,
backward, optimizer -- so a TRL or transformers change that breaks the loop
(not just the source structure) surfaces here. No GPU, no meaningful numerics.

What it does NOT cover: Unsloth's Triton/GPU-optimized model kernels (the
FastLanguageModel fast path) cannot run on CPU, so this validates the
trainer-transform + orchestration layer with a standard forward, not the
optimized kernels.
"""

from __future__ import annotations

import os

# CPU-only: no torch.compile / dynamo (it reaches into the CUDA accelerator), no
# Unsloth kernel compile, no mixed precision. Must be set before torch/unsloth.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


# torch is needed for everything below (daily-fresh-fetch collects this dir with
# only pytest installed); skip the whole module cleanly when it is absent.
if importlib.util.find_spec("torch") is None:
    pytest.skip(
        "torch not installed; fake CPU train needs the real runtime", allow_module_level = True
    )

# Apply the CUDA spoof before any unsloth-touching import.
_SPOOF_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SPOOF_DIR))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()

import torch  # noqa: E402


# The generated GRPO trainer hard-decorates hot functions with @torch.compile,
# which dynamo processes even under the disable env vars, reaching into
# torch.accelerator (real CUDA) on a GPU-less box. Make torch.compile an eager
# passthrough before unsloth generates/imports the trainer -- same logic, no
# dynamo. (An eager CPU run is exactly what we want here.)
def _eager_compile(
    model = None,
    *args,
    **kwargs,
):
    if callable(model):
        return model
    return lambda fn: fn


torch.compile = _eager_compile

# Belt-and-suspenders: if any @torch.compile still routes through dynamo, let it
# fall back to eager instead of crashing, and stop its stream-capture probe from
# reaching torch.accelerator -> real CUDA on a GPU-less box.
try:
    import torch._dynamo  # noqa: E402
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
if hasattr(torch, "accelerator"):
    torch.accelerator.is_available = lambda *a, **k: False


# Redirect any `device="cuda"` tensor allocation / `.to("cuda")` / `.cuda()` to
# CPU. The aggressive spoof deliberately keeps real allocators, but a fake CPU
# train needs cuda-targeted ops (e.g. inductor's init_gpu_context does
# `torch.empty(1, device="cuda")`) to land on CPU instead of erroring.
def _is_cuda_dev(d):
    try:
        return d is not None and torch.device(d).type == "cuda"
    except Exception:
        return False


for _name in (
    "empty",
    "zeros",
    "ones",
    "full",
    "tensor",
    "arange",
    "randn",
    "rand",
    "randint",
    "empty_like",
    "zeros_like",
    "ones_like",
):
    _orig = getattr(torch, _name, None)
    if _orig is None:
        continue

    def _redir(
        *args,
        _orig = _orig,
        **kwargs,
    ):
        if _is_cuda_dev(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return _orig(*args, **kwargs)

    setattr(torch, _name, _redir)

_orig_to = torch.Tensor.to


def _to_cpu(self, *args, **kwargs):
    args = tuple("cpu" if _is_cuda_dev(a) else a for a in args)
    if _is_cuda_dev(kwargs.get("device")):
        kwargs["device"] = "cpu"
    return _orig_to(self, *args, **kwargs)


torch.Tensor.to = _to_cpu
torch.Tensor.cuda = lambda self, *a, **k: self

# Extra CUDA stubs the aggressive spoof lacks, needed to walk a real train():
# Adam's _cuda_graph_capture_health_check() probes stream capture.
torch.cuda.is_current_stream_capturing = lambda *a, **k: False
try:
    import torch.cuda.graphs as _cg  # noqa: E402
    _cg._cuda_isCurrentStreamCapturing = lambda *a, **k: False
except Exception:
    pass

# A broken libmlx.so in the shared site-packages crashes transformers' Mac-only
# is_mlx_array probe on Linux; disable it.
try:
    import transformers.utils.generic as _g  # noqa: E402
    _g._is_mlx_available = False
except Exception:
    pass


# Dense (non-MoE) tiny model on purpose: MoE models route through Unsloth's
# grouped_gemm Triton kernel, which is CUDA-only and cannot run on a CPU runner.
_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _guard_finite_logits(model):
    """Keep the LM head logits finite so GRPO sampling can't crash.

    ``test_grpo_trains_on_cpu`` samples completions from a tiny, *untrained*
    random model on CPU. Driven autoregressively -- and nudged by the fake
    reward's optimizer step between the two train steps -- such a model can emit
    non-finite logits, so ``torch.multinomial`` inside ``generate()``
    intermittently raises "probability tensor contains either `inf`, `nan` or
    element < 0". That is a well-known nondeterministic sampling failure, not an
    Unsloth/TRL regression: the Trainer already fixes the seed, but CPU reduction
    order is not bit-reproducible, so the blow-up still surfaces every so often.

    Sanitize the logits to a finite, bounded range (out of place, so autograd
    stays valid) before they reach the sampler. This test asserts the train loop
    runs end to end, not the (deliberately meaningless) numerics, so bounding the
    logits changes nothing it checks while making the run reliable.
    """

    def _finite_logits_hook(_module, _inputs, output):
        logits = getattr(output, "logits", None)
        if logits is None:
            return output
        output.logits = torch.nan_to_num(
            logits,
            nan = 0.0,
            posinf = 30.0,
            neginf = -30.0,
        ).clamp(-30.0, 30.0)
        return output

    model.register_forward_hook(_finite_logits_hook)
    return model


def _load_plain():
    """Tiny plain HF model + tokenizer on CPU. Skips (not fails) if the model
    cannot be fetched -- that is a network/hub issue, not an unsloth regression."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_MODEL, dtype = torch.float32)
    except OSError as e:  # hub unreachable / model missing
        pytest.skip(f"could not fetch {_MODEL} (network/hub): {str(e)[:150]}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Unsloth's GRPO path calls model.for_training()/for_inference() (added by
    # FastLanguageModel). A plain HF model lacks them; supply minimal train/eval
    # equivalents so the loop proceeds without the optimized wrapper.
    if not hasattr(model, "for_training"):
        model.for_training = lambda *a, **k: model.train()
    if not hasattr(model, "for_inference"):
        model.for_inference = lambda *a, **k: model.eval()
    _guard_finite_logits(model)
    return model.to("cpu"), tok


@pytest.fixture(autouse = True)
def _require_stack():
    global torch  # the `import torch._dynamo` below would otherwise shadow it as local
    if importlib.util.find_spec("unsloth") is None or importlib.util.find_spec("trl") is None:
        pytest.skip("unsloth or trl not installed")
    # A real import failure is a regression we want to surface, so do not guard it.
    import unsloth  # noqa: F401  -- patches TRL trainers to the Unsloth variants

    # `import unsloth` reinstalls the real torch.compile (overwriting the eager
    # passthrough set at module load), so the GRPO hot path (chunked_selective_
    # log_softmax) would really compile -- and inductor picks the spoofed CUDA
    # device, crashing on device props (`gcnArchName`). Re-apply the eager
    # passthrough and flip dynamo's call-time kill switch so every @torch.compile
    # runs eager regardless of when it was decorated. CPU eager is what we want.
    torch.compile = _eager_compile
    try:
        import torch._dynamo  # noqa: E402
        torch._dynamo.config.disable = True
    except Exception:
        pass


def test_sft_trains_on_cpu(tmp_path):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    assert SFTTrainer.__name__ == "UnslothSFTTrainer", "SFT patch did not apply"
    model, tok = _load_plain()
    ds = Dataset.from_list([{"text": "The quick brown fox jumps over the lazy dog."}] * 8)
    cfg = SFTConfig(
        output_dir = str(tmp_path / "ci_sft"),
        per_device_train_batch_size = 2,
        max_steps = 2,
        logging_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        max_length = None,
        padding_free = False,
        dataset_text_field = "text",
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
    )
    SFTTrainer(model = model, processing_class = tok, args = cfg, train_dataset = ds).train()


def test_grpo_trains_on_cpu(tmp_path):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    assert GRPOTrainer.__name__ == "UnslothGRPOTrainer", "GRPO patch did not apply"
    model, tok = _load_plain()
    ds = Dataset.from_list([{"prompt": "hi there"}] * 4)
    cfg = GRPOConfig(
        output_dir = str(tmp_path / "ci_grpo"),
        per_device_train_batch_size = 2,
        num_generations = 2,
        max_steps = 2,
        max_completion_length = 8,
        logging_steps = 1,
        report_to = "none",
        temperature = 1.0,
        beta = 0.0,
        save_strategy = "no",
        use_cpu = True,
        use_vllm = False,
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
    )
    GRPOTrainer(
        model = model,
        processing_class = tok,
        reward_funcs = [lambda completions, **k: [float(len(c)) for c in completions]],
        args = cfg,
        train_dataset = ds,
    ).train()


def test_dpo_trains_on_cpu(tmp_path):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    assert DPOTrainer.__name__ == "UnslothDPOTrainer", "DPO patch did not apply"
    model, tok = _load_plain()
    ds = Dataset.from_list(
        [{"prompt": "Hi", "chosen": " hello friend", "rejected": " go away"}] * 8
    )
    cfg = DPOConfig(
        output_dir = str(tmp_path / "ci_dpo"),
        per_device_train_batch_size = 2,
        max_steps = 2,
        logging_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        beta = 0.1,
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
    )
    DPOTrainer(model = model, processing_class = tok, args = cfg, train_dataset = ds).train()
