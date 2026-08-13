# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Deterministic Unsloth training smoke test, sized for a single Tesla T4.

Runs the whole notebook shape end to end -- 4-bit load, LoRA attach, a
handful of training steps, adapter save, reload-free inference -- against a
tiny model on ONE GPU, and asserts on what came out.

Why it exists
-------------
Every other GPU test in this repo runs on hardware nobody's Colab session
has. T4 is the card the notebooks are actually written for: no bf16, no
flash-attention 2, 16GB, sm_75. A regression that only shows up there is
invisible to the rest of CI. Driven from
``.github/workflows/kaggle-t4-notebook-ci.yml`` on real Kaggle T4s.

What it asserts, in descending order of confidence
--------------------------------------------------
1. **Run-to-run bitwise equality** (``--repeat 2``). Two full runs in the
   same session must produce identical per-step loss and grad_norm, to the
   last bit. This is the strong assertion, and the only one that is exact.
   It catches uninitialised memory, unseeded RNG, iteration over a set, a
   nondeterministic kernel newly introduced into the backward pass.
2. **The canary string.** The training data maps a question to the literal
   target ``__UNSLOTH__!!!``. After overfitting on it, greedy decoding of a
   training prompt must emit that string and nothing else -- an exact match
   after stripping surrounding whitespace, not a substring, because the
   completion trained on is ``CANARY + eos_token`` and
   ``'__UNSLOTH__!!!<more text>'`` is a stopping regression rather than a
   pass. The adapter that was written is also read back off disk and
   checked for tensors that are present, finite and not all zero, since
   inference runs on the in-memory model and would not notice. This is a
   binary,
   tolerance-free check that the forward pass, the backward pass, the
   optimizer step, the adapter save and the inference path are all wired
   together correctly -- it fails loudly if LoRA weights silently never
   reach the generate call, which no loss-value assertion would catch.
3. **Loss and grad_norm inside a band around a committed reference.** A
   tolerance, never an equality. See ``references/README.md``: the
   reference was captured on a specific T4 with a specific library set, and
   a different driver or a transformers bump moves the low bits. The band
   is wide enough not to fire on that and narrow enough to catch a real
   change in the optimisation.

   A reference is only comparable to a run of the SAME EXPERIMENT. The step
   count is part of what the trace encodes -- step 4 of a 10-step run and
   step 4 of a 3-step run are the same iterate only by coincidence, and the
   fp16 scaler's skip pattern lives at the front of the run where a short
   run spends all of its steps -- and so are the learning rate, the
   optimizer, the LoRA shape, the model and which commit of the model
   repository was read. The reference records all of them, and comparing
   against a reference captured with any of them different is a hard
   failure, never a quiet pass. See ``check_reference`` and
   ``REFERENCE_DEFINING_SETTINGS``.

Determinism caveats, stated rather than assumed
-----------------------------------------------
``torch.use_deterministic_algorithms(True, warn_only=True)``: warn_only
because parts of the bitsandbytes 4-bit path have no deterministic kernel
registered, and raising there would abort the test having proved nothing.
Assertion 1 is what actually verifies the outcome.

Bitwise equality is asserted WITHIN one session only. Across GPU
architectures it is not achievable and is not claimed -- fp16 reduction
order alone moves the result.

Usage:
    python run_t4_smoke.py --outdir /kaggle/working/smoke0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from determinism import (  # noqa: E402
    RepeatingSequentialSampler,
    StatisticsCallback,
    compare_metrics,
    enable_full_determinism,
    set_all_seeds_fast,
    set_deterministic_algorithms,
)
from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    load_pins,
    pin_failures,
    resolved_versions,
)

# MUST run before torch is imported anywhere: CUBLAS_WORKSPACE_CONFIG is read
# when cuBLAS initialises, and setting it afterwards is silently ignored.
enable_full_determinism()

CANARY = "__UNSLOTH__!!!"
PROMPT_TEMPLATE = "### Question:\n{question}\n### Answer:\n"
SEED = 3407

# The default is deliberately the smallest instruct model that still exercises
# the real loader path. It must fit on ONE T4 alongside a second copy of this
# same test running on the other T4 of the same session.
DEFAULT_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"


def _log(msg: str) -> None:
    print(f"[t4-smoke] {msg}", flush = True)


def load_canary_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"canary dataset {path} is empty")
    for row in rows:
        if row.get("answer") != CANARY:
            raise RuntimeError(f"canary dataset row does not target {CANARY!r}: {row!r}")
    return rows


def build_dataset(rows: list[dict], eos_token: str):
    """Prompt / completion columns, so the loss lands only on the answer.

    A single ``text`` column would spread the loss across the question
    tokens too, and the question is the part the model already predicts
    well. With the prompt masked out, every one of the few steps this test
    can afford is spent on the canary itself, which is what makes an exact
    string assertion reachable at all in a run this short. TRL applies the
    masking itself when it sees these two columns
    (``completion_only_loss``).
    """
    from datasets import Dataset
    return Dataset.from_dict(
        {
            "prompt": [PROMPT_TEMPLATE.format(question = r["question"]) for r in rows],
            "completion": [r["answer"] + eos_token for r in rows],
        }
    )


def _make_trainer_class(sft_trainer_cls, sampler):
    """SFTTrainer with the sampling order pinned.

    The sampler argument to ``_get_train_sampler`` moved between TRL
    versions (it gained a dataset parameter), so absorb whatever is passed.
    """

    class _FixedOrderSFTTrainer(sft_trainer_cls):  # type: ignore[misc,valid-type]
        def _get_train_sampler(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return sampler

    return _FixedOrderSFTTrainer


def pin_initial_loss_scale(trainer, value: float) -> dict:
    """Lower the fp16 gradient scaler's starting scale before training.

    Why this exists, in one measurement. The T4 has no bf16, so the run is
    fp16 with a dynamic ``GradScaler``. That scaler starts at 65536, halves
    on every overflow, and SKIPS the optimizer step it overflowed on. On
    this model the first three steps overflow every time: the committed
    reference has ``grad_norm: NaN`` at steps 1, 2 and 3 and a finite one
    from step 4, which is 65536 -> 8192 in three halvings. A run of three
    steps therefore applies ZERO optimizer updates and asserts nothing about
    training at all.

    Starting the scaler low enough not to overflow is what buys a short run
    its updates back. It changes the numeric path (a different scale is a
    different rounding of the same gradients), so any reference captured
    before this was introduced does not apply -- which the step-count guard
    in ``check_reference`` already refuses to ignore.

    Never fatal. ``trainer.accelerator.scaler`` is the one place transformers
    keeps it, but it is not a public API, so a version that moved it must
    degrade to "the run is as it was" rather than losing the session. What
    happened is recorded in the report either way, so a reference is never
    captured without it being visible whether the pin took.
    """
    state: dict = {"requested": value}
    if not value:
        state["applied"] = False
        state["reason"] = "not requested"
        return state
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    if scaler is None:
        state["applied"] = False
        state["reason"] = "trainer.accelerator.scaler is absent"
        return state
    if not getattr(scaler, "is_enabled", lambda: True)():
        state["applied"] = False
        state["reason"] = "the scaler is disabled (no fp16 autocast)"
        return state
    if not hasattr(scaler, "_init_scale"):
        state["applied"] = False
        state["reason"] = f"{type(scaler).__name__} has no _init_scale"
        return state
    state["before"] = float(scaler.get_scale())
    # _init_scale rather than a fresh GradScaler: the object may be a
    # subclass (ShardedGradScaler and friends) and replacing it would drop
    # that. Safe here because training has not started, so the lazy _scale
    # tensor does not exist yet and get_scale() still reads _init_scale --
    # which is also how the assertion below can confirm the pin took.
    scaler._init_scale = float(value)
    state["after"] = float(scaler.get_scale())
    state["applied"] = state["after"] == float(value)
    if not state["applied"]:
        state["reason"] = "the scaler did not take the new scale; it had already been initialised"
    return state


def train_once(args, run_index: int) -> dict:
    """One full load / train / save / infer cycle. Returns a result dict."""
    import torch
    from unsloth import FastLanguageModel

    if args.force_sdpa:
        # LOCAL REPRODUCTION ONLY, and never on the target hardware.
        #
        # Unsloth prefers flash-attention, then xformers, then SDPA. On a T4
        # that resolves to xformers, which is the path this test exists to
        # cover, so this flag is off by default and the Kaggle run does not
        # use it. It exists because xformers ships no backward kernel for
        # some newer architectures (Blackwell raises
        # NotImplementedError: No operator found for
        # memory_efficient_attention_backward), which makes the payload
        # impossible to reproduce locally on such a box without it. Forcing
        # SDPA changes the numeric path, so a local run under this flag is
        # evidence about the HARNESS, not about T4 numerics.
        from unsloth.utils import attention_dispatch
        attention_dispatch.HAS_XFORMERS = False
        _log("force-sdpa: HAS_XFORMERS pinned False (local repro only)")

    set_all_seeds_fast(SEED)
    det_state = set_deterministic_algorithms(warn_only = not args.strict_deterministic)

    rows = load_canary_rows(Path(args.dataset))

    t0 = time.time()
    # float16 unconditionally: T4 is sm_75 and has no bf16. Pinning it here
    # rather than letting the loader pick means the local reproduction and the
    # Kaggle run take the same numeric path on the parts that can share one.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
        dtype = torch.float16,
    )
    load_seconds = time.time() - t0
    # Which repository and which commit of it. `load_in_4bit=True` makes
    # Unsloth redirect the request through FLOAT_TO_INT_MAPPER, so the name
    # in the config is not the name that was asked for, and `revision=` is
    # explicitly DROPPED by the loader once a remap happened
    # (unsloth/models/loader.py::_revision_for_resolved_repo). Pinning the
    # load is therefore not available here; recording what was loaded is,
    # and it is what makes a silent in-place re-upload attributable.
    _config = getattr(model, "config", None)
    resolved_checkpoint = getattr(_config, "_name_or_path", None)
    resolved_revision = getattr(_config, "_commit_hash", None)
    _log(f"loaded {resolved_checkpoint} @ {resolved_revision}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = 0.0,  # nonzero dropout is one more RNG consumer
        bias = "none",
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = args.gradient_checkpointing,
        random_state = SEED,
    )

    eos = tokenizer.eos_token or ""
    dataset = build_dataset(rows, eos)

    from trl import SFTConfig, SFTTrainer

    sampler = RepeatingSequentialSampler(
        dataset_length = len(dataset),
        batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        max_steps = args.max_steps,
    )
    stats = StatisticsCallback()

    config = SFTConfig(
        output_dir = str(Path(args.outdir) / f"trainer_run{run_index}"),
        completion_only_loss = True,
        max_length = args.max_seq_length,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        # Constant schedule with no warmup: over 3 steps a warmup would spend
        # the whole run at a fraction of the target LR, and a linear decay
        # would make step 3's update depend on max_steps. Constant keeps the
        # reference meaningful and the overfit strong enough for the canary.
        lr_scheduler_type = "constant",
        warmup_steps = 0,
        logging_steps = 1,  # StatisticsCallback only fires on logs
        optim = args.optim,
        weight_decay = 0.0,
        seed = SEED,
        data_seed = SEED,
        fp16 = True,
        bf16 = False,
        dataloader_num_workers = 0,  # worker processes reorder and reseed
        dataloader_pin_memory = False,
        group_by_length = False,
        report_to = "none",
        save_strategy = "no",
    )

    trainer_cls = _make_trainer_class(SFTTrainer, sampler)
    trainer = trainer_cls(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = config,
        callbacks = [stats],
    )

    loss_scale = pin_initial_loss_scale(trainer, args.init_loss_scale)
    _log(f"fp16 loss scale: {json.dumps(loss_scale)}")

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    if len(stats.logs) != args.max_steps:
        raise RuntimeError(
            f"expected {args.max_steps} logged steps, got {len(stats.logs)}: " f"{stats.logs}"
        )

    # Adapter save, then read the serialized weights back off disk. A
    # filename is not evidence: save_pretrained can leave an empty, truncated
    # or all-zero adapter_model.safetensors and every later assertion here
    # still passes, because inference runs on the in-memory model. The
    # verification is deliberately a file read rather than a second
    # FastLanguageModel load -- it answers "can these weights be consumed"
    # without a second 4-bit load on a card that is already hosting one.
    adapter_dir = Path(args.outdir) / f"lora_run{run_index}"
    t0 = time.time()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_seconds = time.time() - t0
    saved_files = sorted(p.name for p in adapter_dir.iterdir())
    adapter_weights = [f for f in saved_files if f.startswith("adapter_model.")]
    if not adapter_weights:
        raise RuntimeError(f"no adapter weights in {adapter_dir}: {saved_files}")
    saved_adapter = verify_saved_adapter(adapter_dir)
    _log(f"saved adapter: {json.dumps(saved_adapter)}")

    # Inference on the trained, in-memory model. Greedy, so the output is a
    # function of the weights alone.
    FastLanguageModel.for_inference(model)
    prompt = PROMPT_TEMPLATE.format(question = rows[0]["question"])
    inputs = tokenizer([prompt], return_tensors = "pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens = args.max_new_tokens,
            do_sample = False,
            temperature = None,
            top_p = None,
            top_k = None,
            use_cache = True,
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    infer_seconds = time.time() - t0
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens = True)

    peak_gb = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0

    result = {
        "run_index": run_index,
        "metrics": stats.logs,
        "generated": generated,
        # Both, because they answer different questions when this goes red.
        # `canary_found` says the training reached the weights at all;
        # `canary_exact` is the assertion, and the gap between them is the
        # signature of a stopping/EOS regression rather than a training one.
        "canary_found": CANARY in generated,
        "canary_exact": generated.strip() == CANARY,
        "prompt": prompt,
        "adapter_files": saved_files,
        "saved_adapter": saved_adapter,
        # The repository the loader actually read, and its commit. The
        # requested name is not the name that gets loaded -- load_in_4bit
        # sends this through Unsloth's FLOAT_TO_INT_MAPPER -- and a mirror
        # repo re-uploaded in place moves the trajectory with no change in
        # this repository at all. Recorded so the band check can say so.
        "resolved_checkpoint": resolved_checkpoint,
        "resolved_revision": resolved_revision,
        "determinism": det_state,
        "loss_scale": loss_scale,
        "timing_seconds": {
            "load": round(load_seconds, 1),
            "train": round(train_seconds, 1),
            "save": round(save_seconds, 1),
            "infer": round(infer_seconds, 1),
        },
        "peak_reserved_gb": round(peak_gb, 2),
    }

    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def verify_saved_adapter(adapter_dir) -> dict:
    """Read the serialized adapter back and say what is in it.

    Everything downstream of the save runs on the in-memory model, so the
    only thing that ever looked at the file was a filename test. This opens
    it. Kept to a tensor read rather than a PEFT reload because it has to run
    on a card that is already holding a 4-bit model, and because the failure
    modes worth naming -- unreadable, empty, non-finite, all zero -- are all
    visible in the tensors themselves.

    ``nonzero_tensors`` is the load-bearing one. ``lora_B`` is zero at
    initialisation and only becomes non-zero once an optimizer update has been
    applied and saved, so an all-zero file is an untrained adapter that would
    reload without complaint.

    Returns a dict; never raises. ``saved_adapter_failures`` turns it into a
    verdict, so the pass/fail rule stays testable without a GPU.
    """
    adapter_dir = Path(adapter_dir)
    state: dict[str, Any] = {"dir": str(adapter_dir)}
    try:
        state["files"] = sorted(p.name for p in adapter_dir.iterdir())
    except OSError as exc:
        state["files"] = []
        state["error"] = f"{type(exc).__name__}: {exc}"
        return state
    try:
        json.loads((adapter_dir / "adapter_config.json").read_text(encoding = "utf-8"))
        state["config_readable"] = True
    except Exception as exc:  # noqa: BLE001
        state["config_readable"] = False
        state["config_error"] = f"{type(exc).__name__}: {exc}"[:200]

    safetensors_file = adapter_dir / "adapter_model.safetensors"
    bin_file = adapter_dir / "adapter_model.bin"
    tensors = None
    try:
        if safetensors_file.exists():
            from safetensors.torch import load_file
            state["weight_file"] = safetensors_file.name
            tensors = load_file(str(safetensors_file))
        elif bin_file.exists():
            import torch
            state["weight_file"] = bin_file.name
            tensors = torch.load(str(bin_file), map_location = "cpu", weights_only = True)
        else:
            state["error"] = "no adapter_model.safetensors and no adapter_model.bin"
            return state
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return state

    non_finite: list[str] = []
    nonzero = 0
    total = 0
    for name, tensor in tensors.items():
        try:
            floating = tensor.is_floating_point()
            if floating and not bool(tensor.isfinite().all()):
                non_finite.append(name)
            if bool(tensor.count_nonzero()):
                nonzero += 1
            total += int(tensor.numel())
        except Exception as exc:  # noqa: BLE001
            non_finite.append(f"{name}: {type(exc).__name__}")
    state["tensors"] = len(tensors)
    state["parameters"] = total
    state["non_finite_tensors"] = non_finite[:10]
    state["nonzero_tensors"] = nonzero
    return state


def saved_adapter_failures(state: dict) -> list[str]:
    """Turn ``verify_saved_adapter``'s reading into failure strings."""
    failures: list[str] = []
    if not state:
        return ["the saved adapter was never verified"]
    if state.get("config_readable") is False:
        failures.append(
            f"the saved adapter's adapter_config.json could not be read, so "
            f"nothing can load it: {state.get('config_error')}"
        )
    if state.get("tensors") is None:
        failures.append(
            f"the saved adapter weights could not be read back from "
            f"{state.get('dir')}: {state.get('error')}"
        )
        return failures
    if not state["tensors"]:
        failures.append(f"the saved adapter holds no tensors: {state.get('files')}")
        return failures
    if state.get("non_finite_tensors"):
        failures.append(
            f"the saved adapter holds non-finite weights: {state['non_finite_tensors']}"
        )
    if not state.get("nonzero_tensors"):
        failures.append(
            f"every saved tensor is zero across all {state['tensors']} of them. lora_B "
            f"starts at zero, so this adapter carries no training at all and reloading "
            f"it would restore the base model."
        )
    return failures


def canary_failures(run: dict, *, require: bool) -> list[str]:
    """The canary assertion, as an EXACT match rather than a substring.

    The completion this run trains on is ``CANARY + eos_token`` and decoding
    strips the special tokens, so a healthy greedy decode returns the canary
    and nothing else. ``CANARY in generated`` also accepts
    ``'__UNSLOTH__!!!<anything>'``, which is precisely what a stopping or EOS
    regression produces -- the model learned the target and no longer knows
    where to stop -- and that is a broken inference path reporting green.

    Surrounding whitespace is the one normalisation allowed: it is a decoder
    artefact, not a change in what the model emitted.
    """
    generated = run.get("generated") or ""
    if generated.strip() == CANARY:
        return []
    if CANARY in generated:
        msg = (
            f"run {run.get('run_index')} did not emit the canary {CANARY!r} exactly: "
            f"the canary is there but so is other text, which is what a stopping or "
            f"EOS regression looks like. Got {generated!r}"
        )
    else:
        msg = (
            f"run {run.get('run_index')} did not emit the canary "
            f"{CANARY!r}; got {generated!r}"
        )
    if not require:
        _log("WARNING (not enforced): " + msg)
        return []
    return [msg]


def environment_fingerprint() -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import trl
        info["trl"] = trl.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import unsloth
        info["unsloth"] = getattr(unsloth, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    # Every package this CI watches, read from the installed distributions.
    # This is the line that makes a canary-leg failure attributable: the
    # control leg and the canary leg run the same payload, so the diff of
    # these two blocks is the entire difference between a green run and a
    # red one. The keys above are kept as they were, because the committed
    # reference carries them and the summary renderer reads them.
    info["resolved"] = flatten_versions(resolved_versions(GOAL_PACKAGES))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_capability"] = f"sm_{props.major}{props.minor}"
        info["gpu_total_gb"] = round(props.total_memory / 1024**3, 1)
        info["gpu_count_visible"] = torch.cuda.device_count()
        info["driver_cuda"] = torch.version.cuda
    return info


def reference_step_count(ref: dict):
    """The ``max_steps`` a reference file says it was captured at.

    ``None`` means the file does not say. That is not the same as "it
    matches": a trace with no declared length cannot be shown to describe
    the same run as the one in hand, and the caller treats it as such.
    """
    config = ref.get("config")
    if not isinstance(config, dict):
        return None
    value = config.get("max_steps")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Every setting that defines which experiment a trace is a trace OF. A run
# that differs in any of these is not comparable to the reference, whatever
# the numbers happen to do -- references/README.md says so and says widening
# the band is never the answer.
#
# `repeat` is deliberately absent: each cycle is a fresh process running the
# identical configuration, so how many of them were run does not change any
# one of them, and refusing on it would reject a --repeat 3 run against a
# --repeat 2 reference for no reason.
REFERENCE_DEFINING_SETTINGS = (
    "max_steps",
    "init_loss_scale",
    "batch_size",
    "grad_accum",
    "max_seq_length",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "optim",
    "gradient_checkpointing",
)


def check_reference(
    metrics: list[dict],
    reference_path: Path,
    rel_tol: float,
    abs_floor: float,
    *,
    max_steps: int,
    config: dict | None = None,
    model: str | None = None,
    resolved_checkpoint: str | None = None,
    resolved_revision: str | None = None,
) -> dict:
    """Compare against a committed reference. Never an equality check.

    ``max_steps`` is the step count of the run being judged, and it is
    mandatory. A reference is a trace of one specific run, and a run of a
    different length is a different run: the fp16 scaler burns its first
    few steps on overflows, the learning-rate schedule is constant only
    because the run is short, and step N of a 3-step run is simply not the
    step N the 10-step trace recorded. Comparing across counts would be
    arithmetic that succeeds and means nothing, so the mismatch is reported
    as its own status and the numbers are never touched. ``reference_failures``
    turns it into a failure; nothing here can turn it into a pass.

    ``max_steps`` was the only one of those settings that was checked, and it
    is not the only one that has that property. The reference records the
    whole ``config`` block, and README calls out the learning rate, the
    optimizer and the model by name as things that invalidate the file. So
    ``config``, ``model`` and the resolved checkpoint are compared on the same
    terms, under the same refuse-before-comparing rule.

    All four are optional and default to not-compared, so an older caller and
    an older reference file both keep working: a key the reference does not
    carry is listed in ``config_unchecked`` rather than treated as a
    mismatch. "It does not say" is not "it differs", just as it is not "it
    matches".
    """
    if not reference_path.exists():
        return {"status": "absent", "path": str(reference_path)}
    ref = json.loads(reference_path.read_text(encoding = "utf-8"))
    ref_metrics = ref.get("metrics", [])
    verdict: dict = {
        "status": "ok",
        "path": str(reference_path),
        "reference_env": ref.get("environment", {}),
        "observed_max_steps": max_steps,
        "reference_max_steps": reference_step_count(ref),
        "deviations": [],
        "worst_rel": {},
        "config_differences": [],
        "config_unchecked": [],
        "step_differences": [],
    }

    # The step-count gate comes FIRST and returns, so no partially
    # reassuring "worst relative deviation" is ever computed from two runs
    # that are not the same run.
    if verdict["reference_max_steps"] is None:
        verdict["status"] = "reference_step_count_unknown"
        verdict["note"] = (
            f"{reference_path.name} does not record the max_steps it was "
            "captured at (no config.max_steps), so it cannot be shown to "
            f"describe a {max_steps}-step run. Recapture it with the recipe "
            "in references/README.md."
        )
        return verdict
    if verdict["reference_max_steps"] != max_steps:
        verdict["status"] = "step_count_mismatch"
        verdict["note"] = (
            f"{reference_path.name} was captured at max_steps="
            f"{verdict['reference_max_steps']} and this run is "
            f"{max_steps} steps. Those are different runs and their "
            "per-step traces are not comparable. Regenerate the reference "
            "at the new step count (references/README.md) rather than "
            "widening the band."
        )
        return verdict

    # The rest of the configuration, on the same terms as the step count:
    # refuse before comparing a single number, because these settings define
    # a different experiment rather than a drift within one.
    ref_config = ref.get("config") if isinstance(ref.get("config"), dict) else {}
    observed_pairs: list[tuple[str, Any, Any]] = []
    if config:
        for key in REFERENCE_DEFINING_SETTINGS:
            if key == "max_steps":
                continue  # already gated above, with its own status
            if key not in ref_config or key not in config:
                verdict["config_unchecked"].append(key)
                continue
            observed_pairs.append((key, ref_config[key], config[key]))
    for key, observed in (
        ("model", model),
        ("resolved_checkpoint", resolved_checkpoint),
        ("resolved_revision", resolved_revision),
    ):
        if observed is None:
            continue
        if ref.get(key) is None:
            verdict["config_unchecked"].append(key)
            continue
        observed_pairs.append((key, ref[key], observed))
    for key, expected, observed in observed_pairs:
        if expected != observed:
            verdict["config_differences"].append(
                {"key": key, "reference": expected, "observed": observed}
            )
    if verdict["config_differences"]:
        verdict["status"] = "config_mismatch"
        verdict["note"] = (
            f"{reference_path.name} was captured with a different training "
            f"configuration: {verdict['config_differences']}. Those settings "
            "define which experiment the trace is a trace of, so the numbers "
            "are not comparable. Regenerate the reference "
            "(references/README.md) rather than widening the band."
        )
        return verdict

    if len(ref_metrics) != len(metrics):
        verdict["status"] = "length_mismatch"
        return verdict

    # The step coordinates, before any value is compared. The two lists are
    # zipped positionally, so a shifted, duplicated or reordered `step` pairs
    # values that describe different iterates and the band check then reports
    # on arithmetic it invented. Safe to be strict here because the leg that
    # carries a reference is the control, whose library set is pinned to the
    # one the trace was captured with and whose pin failure is itself fatal --
    # so the trainer cannot renumber its steps underneath this check without
    # the pins going red first.
    for index, (cur, old) in enumerate(zip(metrics, ref_metrics)):
        if cur.get("step") != old.get("step"):
            verdict["step_differences"].append(
                {"index": index, "reference": old.get("step"), "observed": cur.get("step")}
            )
    if verdict["step_differences"]:
        verdict["status"] = "step_mismatch"
        verdict["note"] = (
            f"the observed per-step trace does not carry the same step "
            f"coordinates as {reference_path.name}: "
            f"{verdict['step_differences'][:5]}. The two are compared "
            "positionally, so nothing was compared."
        )
        return verdict

    for field in ("loss", "grad_norm"):
        worst = 0.0
        for cur, old in zip(metrics, ref_metrics):
            has_cur, has_old = field in cur, field in old
            if not has_cur and not has_old:
                continue
            if has_cur != has_old:
                # Present on one side and not the other is a change in the
                # SHAPE of what the trainer logged, not a numeric drift, and
                # no tolerance covers it.
                verdict["deviations"].append(
                    {
                        "step": (cur if has_cur else old).get("step"),
                        "field": field,
                        "reference": old.get(field, None),
                        "observed": cur.get(field, None),
                        "relative": None,
                        "note": "field present on only one side",
                    }
                )
                continue
            new, ref_val = float(cur[field]), float(old[field])
            # NaN, explicitly, and this is the whole reason the arithmetic is
            # not left to take care of itself. Under fp16 the gradient scaler
            # logs a NaN grad_norm on every step it skips, so the committed
            # reference genuinely contains NaNs. Left to the subtraction,
            # abs(x - NaN) is NaN, NaN > rel_tol is False, and the step
            # passes whatever it holds -- including the case that matters
            # most, a step that used to overflow and no longer does. Compare
            # NaN to NaN as equal, and NaN against a number as a deviation.
            cur_nan, ref_nan = new != new, ref_val != ref_val
            if cur_nan or ref_nan:
                if cur_nan != ref_nan:
                    verdict["deviations"].append(
                        {
                            "step": cur.get("step"),
                            "field": field,
                            "reference": old[field],
                            "observed": cur[field],
                            "relative": None,
                            "note": "the fp16 scaler skip pattern moved: NaN on one side only",
                        }
                    )
                continue
            base = max(abs(ref_val), abs_floor)
            rel = abs(new - ref_val) / base
            worst = max(worst, rel)
            if rel > rel_tol:
                verdict["deviations"].append(
                    {
                        "step": cur.get("step"),
                        "field": field,
                        "reference": old[field],
                        "observed": cur[field],
                        "relative": round(rel, 5),
                    }
                )
        verdict["worst_rel"][field] = round(worst, 5)
    if verdict["deviations"]:
        verdict["status"] = "out_of_band"
    return verdict


def reference_failures(verdict: dict, rel_tol: float) -> list[str]:
    """Turn a reference verdict into failure strings. Separate so the path
    from "out of band" to "the job goes red" can be tested without a GPU:
    a band check that has never been observed to fail is not yet a check.
    """
    if verdict["status"] == "out_of_band":
        return [f"metrics outside +/-{rel_tol:.0%} of the reference: " f"{verdict['deviations']}"]
    if verdict["status"] == "length_mismatch":
        return ["reference has a different number of logged steps: nothing was compared"]
    # Refusals. Loud, and a failure: a reference that cannot be compared is
    # worth strictly less than no reference at all, because it looks like
    # cover and is not. Never demote either of these to a warning.
    if verdict["status"] in (
        "step_count_mismatch",
        "reference_step_count_unknown",
        "config_mismatch",
        "step_mismatch",
    ):
        return [
            "refusing to band-check against a reference that is not for "
            "this run: " + verdict.get("note", verdict["status"])
        ]
    return []


def _is_finite(value) -> bool:
    """NaN and both infinities are all "the step did not apply"."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def optimisation_failures(metrics: list[dict]) -> list[str]:
    """Did this run optimise anything at all? Cheap checks, loud answers.

    The last of the three is the one a short run needs. Under fp16 the
    gradient scaler logs ``grad_norm: NaN`` on a step it skipped, and a run
    whose every step was skipped applied no optimizer update whatsoever: the
    weights at the end are the weights at the start. Everything downstream
    still "works" -- the loss is finite, the adapter saves, generation
    produces text -- so without naming it, that run reports as a training
    test having done no training. It is the exact failure mode a step count
    trimmed too far produces, so it is checked rather than assumed.
    """
    failures: list[str] = []
    losses = [m["loss"] for m in metrics]
    if any(l != l or l in (float("inf"), float("-inf")) for l in losses):
        failures.append(f"non-finite loss: {losses}")
    if len(losses) > 1 and not losses[-1] < losses[0]:
        failures.append(f"loss did not decrease over the run: {losses}")
    # Only decidable where grad_norm was logged at all. A trainer version
    # that stops logging it says nothing about whether steps were applied,
    # and inferring "all skipped" from its silence would be a failure
    # invented by this check rather than found by it.
    #
    # Finite, not merely non-NaN. An fp16 overflow reports the norm as inf at
    # least as readily as NaN -- clip_grad_norm_ over a gradient holding an
    # inf returns inf -- and `inf == inf` is True, so a NaN-only test counted
    # every skipped step as an applied one and a run that applied nothing
    # reported green. The loss check three lines above already treats inf as
    # non-finite; this one did not.
    reported = [m["grad_norm"] for m in metrics if m.get("grad_norm") is not None]
    applied = [g for g in reported if _is_finite(g)]
    if reported and not applied:
        failures.append(
            f"the fp16 gradient scaler skipped every one of the "
            f"{len(metrics)} steps (no grad_norm is finite: {reported}), so no "
            f"optimizer update was applied and this run measured nothing "
            f"about training. Raise --max-steps or lower --init-loss-scale."
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = DEFAULT_MODEL)
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    # 3 steps, and the whole reason --init-loss-scale exists.
    #
    # Measured, twice. Under fp16 the dynamic gradient scaler starts at
    # 65536, halves on every overflow and skips the step it overflowed on.
    # On this model the first three steps overflow every time: the committed
    # 10-step reference has grad_norm NaN at steps 1, 2 and 3. So a 3-step
    # run of the ORIGINAL configuration applies zero optimizer updates, the
    # loss does not move, and the canary never forms -- an early 3-step
    # attempt emitted '#1\ndef my_function():...'.
    #
    # Pinning the scaler's starting scale below the overflow point (see
    # pin_initial_loss_scale) gives those three steps back as real updates.
    # That is what makes 3 steps a training run rather than three forward
    # passes, and optimisation_failures() asserts it happened rather than
    # trusting it. The 10-step trajectory is still the honest yardstick for
    # what 3 updates can buy: it reached loss 1.75 after three applied
    # updates and 0.18 after four, so the canary has less margin at 3 steps
    # than it had at 10. If a T4 run reports the canary missing while the
    # scaler shows updates being applied, that is the signal to raise this
    # back up rather than to relax the canary.
    ap.add_argument("--max-steps", type = int, default = 10)
    # Off by default. The pin exists for short runs: the scaler overflows the
    # first three steps, so anything under about five applies no optimizer
    # updates at all. At the default of 10 the run reaches step 4 on its own
    # and learns the canary, and leaving the scaler alone is what keeps the
    # committed reference applicable, since a different starting scale is a
    # different rounding of the same gradients. Set it (e.g. 2048, below the
    # 8192 the reference reaches after three halvings) only alongside a
    # shorter --max-steps and a reference recaptured with both.
    ap.add_argument(
        "--init-loss-scale",
        type = float,
        default = 0.0,
        help = "fp16 GradScaler starting scale; 0 leaves the "
        "framework default (65536, which costs a short run "
        "its first few steps to overflows)",
    )
    ap.add_argument("--batch-size", type = int, default = 2)
    ap.add_argument("--grad-accum", type = int, default = 1)
    ap.add_argument("--max-seq-length", type = int, default = 512)
    # 1e-3 with the prompt masked out. Higher rates overflow fp16 far more
    # often, and each overflow is a skipped step this short run cannot spare.
    ap.add_argument("--learning-rate", type = float, default = 1e-3)
    ap.add_argument("--lora-r", type = int, default = 16)
    ap.add_argument("--lora-alpha", type = int, default = 32)
    ap.add_argument("--optim", default = "adamw_8bit")
    ap.add_argument("--gradient-checkpointing", default = "unsloth")
    ap.add_argument("--max-new-tokens", type = int, default = 16)
    ap.add_argument(
        "--repeat", type = int, default = 2, help = "fresh-process cycles; >1 enables the bitwise check"
    )
    ap.add_argument(
        "--cycle", type = int, default = -1, help = argparse.SUPPRESS
    )  # internal: child-mode marker
    ap.add_argument(
        "--force-sdpa",
        action = "store_true",
        help = "pin the SDPA attention backend. Local reproduction "
        "on hardware xformers has no kernel for; NOT for "
        "the T4 run, which must exercise the xformers path",
    )
    ap.add_argument(
        "--strict-deterministic",
        action = "store_true",
        help = "use_deterministic_algorithms(warn_only=False)",
    )
    ap.add_argument(
        "--reference", default = "", help = "committed reference JSON to band-check against"
    )
    ap.add_argument(
        "--pins",
        default = "",
        help = "a name==version pin file this run must have "
        "resolved to exactly. The control leg passes it; a "
        "pin that did not hold means the leg is not a "
        "control and its comparison against the canary is "
        "worthless, so it is a failure rather than a note",
    )
    ap.add_argument("--rel-tol", type = float, default = 0.10)
    ap.add_argument(
        "--abs-floor",
        type = float,
        default = 0.05,
        help = "denominator floor so a near-zero reference value "
        "does not turn a tiny absolute drift into a huge "
        "relative one",
    )
    ap.add_argument("--require-canary", dest = "require_canary", action = "store_true", default = True)
    ap.add_argument("--no-require-canary", dest = "require_canary", action = "store_false")
    ap.add_argument("--label", default = "t4-smoke")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    # Child mode: exactly one cycle, report to disk, no assertions.
    if args.cycle >= 0:
        run = train_once(args, args.cycle)
        for entry in run["metrics"]:
            _log(
                f"    step {entry['step']}  loss={entry['loss']!r}  "
                f"grad_norm={entry.get('grad_norm')!r}"
            )
        _log(f"    generated: {run['generated']!r}")
        (outdir / "cycle_report.json").write_text(json.dumps(run, indent = 2), encoding = "utf-8")
        return 0

    # Parent mode: each cycle in a FRESH process.
    #
    # Not a loop in one process, and this is measured rather than assumed.
    # Two in-process cycles disagreed from the very first logged step
    # (6.4375 vs 6.2367) while two separate processes agreed bitwise on
    # every step. Something in the first cycle's model load, patching or
    # allocator state leaks into the second, so an in-process repeat tests
    # the leak rather than the code, and would report a false regression on
    # a perfectly reproducible run. A fresh process is also the honest unit:
    # it is what a user re-running a notebook actually gets.
    # Read BEFORE the cycles rather than after them, so a run that dies in a
    # cycle still says which library set it died with. That is the case where
    # the version block matters most: control red / canary green is a bisect
    # only if the red leg reported its versions.
    config = {
        k: getattr(args, k)
        for k in (
            "max_steps",
            "init_loss_scale",
            "batch_size",
            "grad_accum",
            "max_seq_length",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "optim",
            "gradient_checkpointing",
            "repeat",
        )
    }
    try:
        env = environment_fingerprint()
    except Exception as exc:  # noqa: BLE001
        env = {"error": f"{type(exc).__name__}: {exc}"[:300]}

    runs = []
    for i in range(args.repeat):
        _log(f"=== cycle {i + 1}/{args.repeat} (fresh process) ===")
        cycle_dir = outdir / f"cycle{i}"
        cycle_dir.mkdir(parents = True, exist_ok = True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--outdir",
            str(cycle_dir),
            "--cycle",
            str(i),
        ]
        for flag, value in (
            ("--model", args.model),
            ("--dataset", args.dataset),
            ("--max-steps", args.max_steps),
            ("--init-loss-scale", args.init_loss_scale),
            ("--batch-size", args.batch_size),
            ("--grad-accum", args.grad_accum),
            ("--max-seq-length", args.max_seq_length),
            ("--learning-rate", args.learning_rate),
            ("--lora-r", args.lora_r),
            ("--lora-alpha", args.lora_alpha),
            ("--optim", args.optim),
            ("--gradient-checkpointing", args.gradient_checkpointing),
            ("--max-new-tokens", args.max_new_tokens),
            ("--label", args.label),
        ):
            cmd += [flag, str(value)]
        if args.force_sdpa:
            cmd.append("--force-sdpa")
        if args.strict_deterministic:
            cmd.append("--strict-deterministic")
        proc = subprocess.run(cmd)
        report_file = cycle_dir / "cycle_report.json"
        if proc.returncode != 0 or not report_file.exists():
            _log(f"cycle {i} failed (rc={proc.returncode})")
            failed = {
                "label": args.label,
                "model": args.model,
                "config": config,
                "environment": env,
                "passed": False,
                "runs": runs,
                "metrics": [],
                "failures": [f"cycle {i} did not complete " f"(rc={proc.returncode})"],
            }
            (outdir / "t4_smoke_report.json").write_text(
                json.dumps(failed, indent = 2), encoding = "utf-8"
            )
            print("T4_SMOKE_REPORT " + json.dumps(failed), flush = True)
            _log("T4_SMOKE_RESULT FAIL")
            return 1
        runs.append(json.loads(report_file.read_text(encoding = "utf-8")))

    report: dict = {
        "label": args.label,
        "model": args.model,
        # The repo and commit the loader actually read, travelling into any
        # reference captured from this report so the band check can refuse
        # against a mirror that was re-uploaded in place.
        "resolved_checkpoint": runs[0].get("resolved_checkpoint"),
        "resolved_revision": runs[0].get("resolved_revision"),
        # max_steps leads, and the whole block travels into any reference
        # captured from this report: check_reference refuses to compare a run
        # against a trace captured with a different configuration.
        "config": config,
        "environment": env,
        "runs": runs,
        "metrics": runs[0]["metrics"],
        "failures": [],
    }

    failures: list[str] = []

    # 0. the pins, if this leg claims to be a control
    if args.pins:
        resolved = resolved_versions(GOAL_PACKAGES)
        pins = load_pins(args.pins)
        broken = pin_failures(pins, resolved)
        report["pins"] = {"file": args.pins, "requested": pins, "failures": broken}
        failures += broken

    # 1. bitwise run-to-run. EVERY extra cycle against the baseline, not just
    # the second: --repeat 3 asks for three fresh processes, and a third that
    # disagreed used to be collected, stored and never looked at.
    #
    # The top-level keys stay exactly as they were -- report.py renders
    # `identical`, `first_diff_step` and `max_abs_diff` off this dict -- and
    # now summarise every comparison rather than one. The per-cycle detail
    # goes under `cycles`.
    if len(runs) > 1:
        cycles: dict = {}
        for other in runs[1:]:
            cycles[str(other["run_index"])] = compare_metrics(
                runs[0]["metrics"], other["metrics"]
            )
        worst: dict = {}
        for cmp in cycles.values():
            for field, value in cmp.get("max_abs_diff", {}).items():
                worst[field] = max(worst.get(field, 0.0), value)
        differing = [(k, c) for k, c in cycles.items() if not c["identical"]]
        report["reproducibility"] = {
            "identical": not differing,
            "first_diff_step": differing[0][1]["first_diff_step"] if differing else None,
            "max_abs_diff": worst,
            "compared_cycles": sorted(cycles, key = int),
            "cycles": cycles,
        }
        for index, cmp in differing:
            failures.append(
                f"run-to-run metrics differ between cycle 0 and cycle {index} "
                f"(first diff at step {cmp['first_diff_step']}, max abs "
                f"{cmp['max_abs_diff']}, step mismatches {cmp['step_mismatch']})"
            )

        gen = {r["generated"] for r in runs}
        report["generated_identical"] = len(gen) == 1
        if len(gen) != 1:
            failures.append(f"run-to-run generation differs: {sorted(gen)!r}")

    # 2. canary, exactly
    for run in runs:
        failures += canary_failures(run, require = args.require_canary)

    # 3. sanity: finite, the optimisation moved, and it moved at all
    failures += optimisation_failures(runs[0]["metrics"])

    # 4. the adapter that was written is an adapter that can be loaded
    for run in runs:
        failures += [
            f"run {run['run_index']}: {f}"
            for f in saved_adapter_failures(run.get("saved_adapter") or {})
        ]

    # 5. band check against the committed reference
    if args.reference:
        ref = check_reference(
            runs[0]["metrics"],
            Path(args.reference),
            args.rel_tol,
            args.abs_floor,
            max_steps = args.max_steps,
            config = config,
            model = args.model,
            resolved_checkpoint = runs[0].get("resolved_checkpoint"),
            resolved_revision = runs[0].get("resolved_revision"),
        )
        report["reference_check"] = ref
        failures += reference_failures(ref, args.rel_tol)

    report["failures"] = failures
    report["passed"] = not failures

    report_path = outdir / "t4_smoke_report.json"
    report_path.write_text(json.dumps(report, indent = 2), encoding = "utf-8")
    _log(f"report -> {report_path}")
    print("T4_SMOKE_REPORT " + json.dumps(report), flush = True)

    if failures:
        for f in failures:
            _log(f"FAIL: {f}")
        _log("T4_SMOKE_RESULT FAIL")
        return 1
    _log("T4_SMOKE_RESULT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
