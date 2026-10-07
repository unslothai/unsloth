# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Deterministic Unsloth training smoke test, sized for a single Tesla T4.

Runs the whole notebook shape end to end -- 4-bit load, LoRA attach, a handful
of training steps, adapter save, reload-free inference -- against a tiny model
on ONE GPU, and asserts on what came out.

Every other GPU test in this repo runs on hardware nobody's Colab session has.
T4 is the card the notebooks are written for: no bf16, no flash-attention 2,
16GB, sm_75, so a regression that only shows up there is invisible to the rest
of CI. Driven from ``.github/workflows/kaggle-t4-notebook-ci.yml`` on real
Kaggle T4s.

What it asserts, in descending order of confidence:

1. **Run-to-run bitwise equality** (``--repeat 2``). Two full runs in one
   session must produce identical per-step loss and grad_norm to the last bit.
   The only exact assertion, and it catches uninitialised memory, unseeded RNG,
   iteration over a set, a nondeterministic kernel new to the backward pass.
2. **The canary string.** The training data maps a question to the literal
   target ``__UNSLOTH__!!!``, and after overfitting, greedy decoding of a
   training prompt must emit that and nothing else. An exact match modulo
   surrounding whitespace, not a substring: the completion trained on is
   ``CANARY + eos_token``, so ``'__UNSLOTH__!!!<more text>'`` is a stopping
   regression rather than a pass. The written adapter is also read back off disk
   and checked for tensors present, finite and not all zero, since inference
   runs on the in-memory model and would not notice. This is a binary,
   tolerance-free check that forward, backward, optimizer step, adapter save and
   inference are wired together, and it fails loudly if LoRA weights never reach
   the generate call, which no loss-value assertion would catch.
3. **Loss and grad_norm inside a band around a committed reference**, a
   tolerance and never an equality. See ``references/README.md``: the reference
   was captured on a specific T4 with a specific library set, and a different
   driver or a transformers bump moves the low bits, so the band is wide enough
   not to fire on that and narrow enough to catch a real change in the
   optimisation.

   A reference is only comparable to a run of the SAME EXPERIMENT. The step
   count is part of what the trace encodes -- step 4 of a 10-step run and of a
   3-step run are the same iterate only by coincidence, and the fp16 scaler's
   skip pattern lives at the front where a short run spends all its steps -- and
   so are the learning rate, the optimizer, the LoRA shape, the model and the
   commit of the model repository read. The reference records all of them, and
   comparing against one captured with any of them different is a hard failure,
   never a quiet pass. See ``check_reference`` and
   ``REFERENCE_DEFINING_SETTINGS``.

Determinism caveats, stated rather than assumed:
``torch.use_deterministic_algorithms(True, warn_only=True)`` is warn_only
because parts of the bitsandbytes 4-bit path register no deterministic kernel,
and raising would abort the test having proved nothing; assertion 1 is what
verifies the outcome. Bitwise equality is asserted WITHIN one session only, and
is not achievable or claimed across GPU architectures, fp16 reduction order
alone moving the result.

Usage:
    python run_t4_smoke.py --outdir /kaggle/working/smoke0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
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
from kernel_provenance import attention_choice, probe_kernels, vision_kernel_failures  # noqa: E402
from naive_trl_compare import comparison_failures  # noqa: E402
from training_evidence import LORA_B_MARKER, LORA_MARKER  # noqa: E402
from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    load_pins,
    pin_failures,
    resolved_versions,
    versions_for_pins,
)

# Must run before torch is imported: CUBLAS_WORKSPACE_CONFIG is read when cuBLAS initialises.
enable_full_determinism()

CANARY = "__UNSLOTH__!!!"
PROMPT_TEMPLATE = "### Question:\n{question}\n### Answer:\n"
SEED = 3407

# Must fit on one T4 alongside a second copy of this test on the other T4.
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


def dataset_digest(path: Path) -> str:
    """A digest of the rows this run trains on, for the reference identity.

    The reference is a trace of one experiment and the training data is part of
    which experiment it is: change a question in canary_dataset.jsonl and the
    loss curve moves for reasons that have nothing to do with the code, with a
    small change passing the band and a larger one reported as a regression.
    That file is inside this workflow's paths filter, so editing it is a
    supported way to trigger the run that would be compared against a trace it
    has nothing to do with.

    Over the PARSED rows in order rather than the file's bytes: reformatting the
    JSON or reordering the keys within a row changes neither what trains nor the
    order it trains in, and forcing a session-costing recapture for whitespace
    is how a check gets switched off. Row order is kept, being the order the
    sampler walks.

    Never raises and never returns None: an unreadable dataset yields a value
    that cannot match any reference, so it lands as a refusal to compare rather
    than as an unchecked key that reads like a comparison that passed.
    """
    try:
        rows = load_canary_rows(path)
    except Exception as exc:  # noqa: BLE001
        return f"unreadable:{type(exc).__name__}"
    canonical = "\n".join(json.dumps(row, sort_keys = True, separators = (",", ":")) for row in rows)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_dataset(rows: list[dict], eos_token: str):
    """Prompt / completion columns, so the loss lands only on the answer.

    A single ``text`` column would spread the loss across the question tokens,
    which the model already predicts well. With the prompt masked out, every one
    of the few steps this test can afford goes on the canary itself, which is
    what makes an exact string assertion reachable in a run this short. TRL
    applies the masking when it sees these two columns
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

    ``_get_train_sampler``'s signature moved between TRL versions (it gained a
    dataset parameter), so absorb whatever is passed.
    """

    class _FixedOrderSFTTrainer(sft_trainer_cls):  # type: ignore[misc,valid-type]
        def _get_train_sampler(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return sampler

    return _FixedOrderSFTTrainer


def pin_initial_loss_scale(trainer, value: float) -> dict:
    """Lower the fp16 gradient scaler's starting scale before training.

    Why, in one measurement: the T4 has no bf16, so the run is fp16 with a
    dynamic ``GradScaler`` that starts at 65536, halves on every overflow and
    SKIPS the step it overflowed on. On this model the first three steps
    overflow every time -- the committed reference has ``grad_norm: NaN`` at
    steps 1, 2 and 3 and a finite one from step 4, which is 65536 -> 8192 in
    three halvings -- so a three-step run applies ZERO optimizer updates.

    Starting the scaler low enough not to overflow buys a short run its updates
    back. It changes the numeric path (a different scale is a different rounding
    of the same gradients), so a reference captured before this does not apply,
    which the step-count guard in ``check_reference`` already refuses to ignore.

    Never fatal. ``trainer.accelerator.scaler`` is where transformers keeps it
    but is not public API, so a version that moved it degrades to "the run is as
    it was" rather than losing the session. What happened is recorded either
    way, so whether the pin took is visible when a reference is captured.
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
    # Set _init_scale instead of replacing the scaler (it may be a subclass); before training,
    # get_scale() still reads _init_scale.
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
        # Local reproduction only: forces SDPA where xformers lacks a backward kernel (e.g. Blackwell).
        # This changes numerics, so it validates the harness, not T4 numerics.
        from unsloth.utils import attention_dispatch
        attention_dispatch.HAS_XFORMERS = False
        _log("force-sdpa: HAS_XFORMERS pinned False (local repro only)")

    set_all_seeds_fast(SEED)
    det_state = set_deterministic_algorithms(warn_only = not args.strict_deterministic)

    rows = load_canary_rows(Path(args.dataset))

    from phase_timers import FetchTimer

    t0 = time.time()
    # float16 unconditionally: T4 (sm_75) has no bf16.
    load_kwargs: dict = {}
    if getattr(args, "single_device", False):
        # Both cards stay visible so DEVICE_COUNT > 1 bindings are live, but weights go on one card:
        # a sharded model fails at step 0 with a cross-device error.
        load_kwargs["device_map"] = {"": 0}
    # Splits load time into fetch and weight load.
    with FetchTimer() as fetch_timer:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model,
            max_seq_length = args.max_seq_length,
            load_in_4bit = True,
            dtype = torch.float16,
            **load_kwargs,
        )
    load_seconds = time.time() - t0
    load_phases = fetch_timer.record(load_seconds)
    _log(
        "load phases: "
        + f"fetch={load_phases['fetch_seconds']}s "
        + f"weight_load={load_phases['weight_load_seconds']}s "
        + f"patched={len(load_phases['patched'])}"
    )
    # load_in_4bit remaps the repo and the loader drops `revision=`, so record what was loaded.
    _config = getattr(model, "config", None)
    resolved_checkpoint = getattr(_config, "_name_or_path", None)
    resolved_revision = getattr(_config, "_commit_hash", None)
    _log(f"loaded {resolved_checkpoint} @ {resolved_revision}")

    # After the load: unsloth imports some kernels (e.g. fla) lazily during from_pretrained.
    if getattr(args, "kernel_provenance", False):
        result_kernels = probe_kernels()
        result_attention = attention_choice(model)
        _log(f"kernels: {json.dumps(result_kernels)}")
        _log(f"attention: {json.dumps(result_attention)}")
    else:
        result_kernels = None
        result_attention = None

    # After the load: the rotary caches are built with the model.
    result_multi_gpu = multi_gpu_facts(model) if getattr(args, "require_multi_gpu", False) else None
    if result_multi_gpu is not None:
        _log(f"multi-gpu: {json.dumps(result_multi_gpu)}")

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        lora_dropout = 0.0,  # nonzero dropout is one more RNG consumer
        bias = "none",
        target_modules = target_modules,
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
        # Constant LR: a warmup or decay over a few steps would distort the canary run.
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

    # Read the saved weights back: save_pretrained can write an empty or all-zero file and inference
    # still passes on the in-memory model.
    adapter_dir = Path(args.outdir) / f"lora_run{run_index}"
    t0 = time.time()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    save_seconds = time.time() - t0
    saved_files = sorted(p.name for p in adapter_dir.iterdir())
    adapter_weights = [f for f in saved_files if f.startswith("adapter_model.")]
    if not adapter_weights:
        raise RuntimeError(f"no adapter weights in {adapter_dir}: {saved_files}")
    saved_adapter = verify_saved_adapter(
        adapter_dir,
        expected = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": target_modules,
        },
        peft_keys = peft_adapter_keys(model),
    )
    _log(f"saved adapter: {json.dumps(saved_adapter)}")

    # Greedy so the output depends only on the weights.
    FastLanguageModel.for_inference(model)
    prompt = PROMPT_TEMPLATE.format(question = rows[0]["question"])
    # `text=` keyword: a vision processor takes the first positional arg as images.
    inputs = tokenizer(text = [prompt], return_tensors = "pt").to(model.device)
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

    # Prompts of different lengths, so left padding is actually exercised.
    batch_prompts = [
        PROMPT_TEMPLATE.format(question = row["question"]) for row in rows[: max(BATCH_SIZES)]
    ]
    while len(batch_prompts) < max(BATCH_SIZES):
        # Pad the prompt list by reusing questions with varying prefixes to keep lengths spread.
        idx = len(batch_prompts)
        batch_prompts.append(
            PROMPT_TEMPLATE.format(
                question = " ".join(["please"] * (idx % 5 + 1))
                + " "
                + rows[idx % len(rows)]["question"]
            )
        )
    batched = batched_generation(
        model,
        tokenizer,
        batch_prompts,
        max_new_tokens = args.max_new_tokens,
    )
    _log(f"batched generation: {json.dumps({k: v for k, v in batched.items() if k != 'batched'})}")

    # After generation: the export merges the adapter, which would change the model under test.
    gguf_export_record = None
    gguf_run_record = None
    # Once per leg: the conversion is slow and a repeat with the same seed asks nothing new.
    if getattr(args, "export_gguf", False) and run_index > 0:
        gguf_export_record = {
            "skipped": "exported on cycle 0; the conversion is the same weights twice"
        }
    elif getattr(args, "export_gguf", False):
        from gguf_export import export_gguf, llama_cpp_facts, run_gguf

        # unsloth must be imported before unsloth_zoo.llama_cpp; kept local so non-export runs skip it.
        install_log = ""
        llama_dir = None
        try:
            import contextlib
            import io

            from unsloth_zoo.llama_cpp import install_llama_cpp

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                returned = install_llama_cpp()
            install_log = buffer.getvalue()
            facts = llama_cpp_facts(install_log, returned)
            llama_dir = facts.get("dir")
        except BaseException as exc:  # noqa: BLE001
            facts = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        _log(f"llama.cpp: {json.dumps(facts)}")

        # Not under outdir: /kaggle/working is 21GB and shipped back; /tmp has ample space.
        gguf_export_record = export_gguf(
            model,
            tokenizer,
            tempfile.mkdtemp(prefix = f"gguf_run{run_index}_"),
            quantization = args.gguf_quantization,
        )
        gguf_export_record["llama_cpp"] = facts
        _log(
            f"gguf export: {json.dumps({k: v for k, v in gguf_export_record.items() if k != 'llama_cpp'})}"
        )

        ggufs = gguf_export_record.get("ggufs") or []
        if ggufs and llama_dir:
            gguf_run_record = run_gguf(ggufs[0]["path"], llama_dir)

    peak_gb = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0

    result = {
        "run_index": run_index,
        "kernels": result_kernels,
        "attention": result_attention,
        "multi_gpu": result_multi_gpu,
        "metrics": stats.logs,
        "generated": generated,
        # canary_found vs canary_exact: the gap points at a stopping/EOS regression, not training.
        "batched_generation": batched,
        "gguf_export": gguf_export_record,
        "gguf_run": gguf_run_record,
        "canary_found": CANARY in generated,
        "canary_exact": generated.strip() == CANARY,
        "prompt": prompt,
        "adapter_files": saved_files,
        "saved_adapter": saved_adapter,
        # The loaded repo differs from the requested one; recorded so the band check can detect re-uploads.
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
        # Kept beside the totals so older series stay comparable.
        "load_phases": load_phases,
        "peak_reserved_gb": round(peak_gb, 2),
    }

    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _reconstruct_adapter_config(adapter_dir, expected: dict | None) -> dict:
    """Ask PEFT to rebuild the saved config, and check it describes THIS run.

    ``json.loads`` succeeding is not the question anyone has about this file.
    ``{}`` is valid JSON, so a save that wrote no LoRA fields at all read as
    "config_readable" and the leg passed on an adapter nothing can load -- the
    same shape as the tensor count that a randomly initialised ``lora_A``
    satisfied. What is actually being asserted is "PEFT can reconstruct the
    adapter", so it is asked of PEFT, on the path a reload takes.

    That path is the mapping dispatch, not the base class:
    ``PeftModel.from_pretrained`` does
    ``PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(...)``, while
    ``PeftConfig.from_pretrained`` alone returns a bare ``PeftConfig`` with
    ``peft_type=None`` for ``{}`` and reports nothing wrong. Checked against
    peft 0.20.0: only the dispatch raises, which is why it is what runs here.

    ``expected`` is DERIVED, not restated: the caller passes the very arguments
    it handed ``get_peft_model``, so a save that writes a well-formed config for
    a DIFFERENT adapter than the one trained (a dropped ``target_modules``, a
    rank that did not survive the round trip) is a difference rather than a
    field list this function had to guess at.

    Never raises; every outcome is a recorded key that ``saved_adapter_failures``
    turns into a verdict.
    """
    out: dict = {}
    try:
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftConfig

        peft_type = PeftConfig.from_pretrained(str(adapter_dir)).peft_type
        if peft_type is None:
            raise ValueError(
                "adapter_config.json names no peft_type, so PEFT cannot tell "
                "which kind of adapter this is"
            )
        config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type].from_pretrained(str(adapter_dir))
    except Exception as exc:  # noqa: BLE001
        out["config_loadable"] = False
        out["config_load_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    out["config_loadable"] = True
    out["config_peft_type"] = str(getattr(config, "peft_type", None))
    differences: list[str] = []
    unchecked: list[str] = []
    for key, wanted in sorted((expected or {}).items()):
        if not hasattr(config, key):
            # A missing field is recorded, not treated as a difference.
            unchecked.append(key)
            continue
        got = getattr(config, key)
        if key == "target_modules" and isinstance(got, str):
            # unsloth writes target_modules as a regex for vision models, so only check that every
            # requested module name appears in the pattern.
            missing = [name for name in (wanted or []) if name not in got]
            same = not missing
            if not same:
                differences.append(
                    f"{key}: trained with {wanted!r}, and the saved regex does "
                    f"not mention {missing!r}: {got!r}"
                )
            continue
        if isinstance(wanted, (list, tuple, set)) or isinstance(got, (list, tuple, set)):
            same = sorted(got or []) == sorted(wanted or [])
        else:
            same = got == wanted
        if not same:
            differences.append(f"{key}: trained with {wanted!r}, saved {got!r}")
    out["config_differences"] = differences
    out["config_unchecked"] = unchecked
    return out


# Batch 1 is the baseline; the rest must reproduce it exactly.
BATCH_SIZES = (2, 4, 8)


def batched_generation(model, tokenizer, prompts, *, max_new_tokens) -> dict:
    """Greedy generation one-at-a-time, then batched, and whether they agree.

    WHAT THIS IS FOR. Batched generation with left padding has broken here
    before, repeatedly and in ways that pass every other check in this file:

    * #3699 batched generation with left-padding and caching produced incorrect
      output,
    * #1066 batch inference produced gibberish,
    * #1456 batch inference was inconsistent for a self-trained model,
    * #2138 a release silently FORCED the tokenizer padding side to right during
      inference, which is why the side is recorded as OBSERVED after generating
      rather than as the value this function set.

    Greedy decoding makes the comparison meaningful: the output is then a
    function of the weights and the attention mask alone, so any difference
    between batch sizes is padding or cache handling rather than sampling.

    THE VACUITY TRAP, and it is the whole reason this returns the token lengths:
    padding only happens when the prompts in a batch have DIFFERENT lengths.
    A batch of equal-length prompts pads nothing, agrees trivially, and reports
    a green left-padding check that never once left-padded. The caller asserts
    the spread; this function measures it.
    """
    # Local import: this module can load before unsloth (and torch) is installed.
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _gen(batch: list) -> list:
        # Keyword: a processor reads a positional list as images.
        enc = tokenizer(text = batch, return_tensors = "pt", padding = True).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                temperature = None,
                top_p = None,
                top_k = None,
                use_cache = True,
                # Test for None: pad_token_id 0 is valid and falsy.
                pad_token_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        # Slice at the padded width: left padding aligns every row at the same column.
        width = enc["input_ids"].shape[1]
        return [tokenizer.decode(row[width:], skip_special_tokens = True) for row in out]

    # [0]: a processor returns a batch dimension even for one prompt.
    lengths = [len(tokenizer(text = [p])["input_ids"][0]) for p in prompts]
    singles = [_gen([p])[0] for p in prompts]
    result = {
        "prompt_token_lengths": lengths,
        "distinct_lengths": len(set(lengths)),
        "padding_side_observed": tokenizer.padding_side,
        "singles": singles,
        "batched": {},
        "agrees": {},
        "empty_outputs": [i for i, text in enumerate(singles) if not text.strip()],
    }
    # Empty rows inside a batch are a separate failure (#9848), never excused by the known-breakage list.
    empty_batched: dict = {}
    for size in BATCH_SIZES:
        outs: list = []
        for start in range(0, len(prompts), size):
            outs.extend(_gen(prompts[start : start + size]))
        result["batched"][str(size)] = outs
        result["agrees"][str(size)] = outs == singles
        rows = [i for i, text in enumerate(outs) if not text.strip()]
        if rows:
            empty_batched[str(size)] = rows
    result["empty_batched_outputs"] = empty_batched
    # Re-read after generating: #2138 silently overrode padding_side inside inference.
    result["padding_side_after"] = tokenizer.padding_side
    return result


# Models whose batched greedy output is expected to differ from batch-1 (#9708): fp16/bf16
# rounding varies with batch shape. Strict: a listed model that starts agreeing fails the leg.
KNOWN_BATCHED_GENERATION_BREAKAGE = {
    "unsloth/gemma-4-E2B-it": "unsloth#9708",
    "unsloth/Qwen3.5-2B": "unsloth#9708",
}


def batched_generation_failures(batch: dict | None, model: str | None = None) -> list[str]:
    """Turn a `batched_generation` record into failures, vacuity included."""
    if not batch:
        return ["batched generation was never run"]
    out = []
    known = KNOWN_BATCHED_GENERATION_BREAKAGE.get(model or "")
    if batch.get("distinct_lengths", 0) < 2:
        out.append(
            "every batched prompt tokenised to the same length "
            f"({batch.get('prompt_token_lengths')}), so nothing was ever padded "
            "and the left-padding check proved nothing"
        )
    if len(batch.get("singles") or []) < max(BATCH_SIZES):
        out.append(
            f"only {len(batch.get('singles') or [])} prompts for a batch size of "
            f"{max(BATCH_SIZES)}, so the largest batch was never actually formed"
        )
    for side_key in ("padding_side_observed", "padding_side_after"):
        if batch.get(side_key) != "left":
            out.append(
                f"{side_key} is {batch.get(side_key)!r}, not 'left'; a right-padded "
                f"decoder-only batch attends to pad tokens before the prompt (#2138)"
            )
    if batch.get("empty_outputs"):
        out.append(f"prompts {batch['empty_outputs']} generated nothing at all")
    # An empty row is #9848, not the #9708 disagreement, so the entry never excuses it.
    for size, rows in sorted((batch.get("empty_batched_outputs") or {}).items()):
        out.append(
            f"batch size {size}: prompts {rows} generated nothing at all inside "
            f"the batch while their one-at-a-time output was not empty (#9848)"
        )
    agrees = batch.get("agrees") or {}
    for size, agreed in agrees.items():
        if not agreed and not known:
            out.append(
                f"batch size {size} did not reproduce one-at-a-time greedy output "
                f"(#3699/#1456): {batch.get('batched', {}).get(size)!r} != "
                f"{batch.get('singles')!r}"
            )
    if known and agrees and all(agrees.values()):
        # Strict half: a listed model that now agrees must be removed from the list.
        out.append(
            f"{model} is listed in KNOWN_BATCHED_GENERATION_BREAKAGE for "
            f"{known}, and every batch size AGREED. That disagreement is bf16 "
            f"rounding that depends on batch shape, so agreement means a kernel "
            f"or the stack changed; work out what, then delete the entry rather "
            f"than carry a stale expectation"
        )
    return out


# Set early so a later crash in the cycle still reports what was measured.
_LAST_MULTI_GPU_FACTS: dict | None = None


def multi_gpu_facts(model) -> dict:
    """What unsloth BOUND, given how many cards this process can see.

    The point of the multi_gpu leg is a branch no pinned payload can reach.
    `unsloth/kernels/utils.py:170`:

        if DEVICE_COUNT > 1:
            torch_gpu_device = torch.cuda.device      # a real device switch
        else:
            def torch_gpu_device(device): return nullcontext()

    `build_kernel.py` pins every ordinary payload with CUDA_VISIBLE_DEVICES, so
    every unsloth kernel this CI has run took the nullcontext branch, and so did
    the DEVICE_COUNT-sized CUDA_STREAMS / WEIGHT_BUFFERS / ABSMAX_BUFFERS arrays
    and the per-device rotary caches in unsloth/models/llama.py.

    Read off the IMPORTED MODULE, not recomputed from `device_count()`. The
    binding is made once at import time, so asking torch how many cards there
    are answers a different question -- and answers it the way the check wants,
    which is the shape of a rule that cannot fail.
    """
    import torch

    global _LAST_MULTI_GPU_FACTS
    facts: dict = {"device_count": torch.cuda.device_count()}
    _LAST_MULTI_GPU_FACTS = facts
    try:
        from unsloth.kernels import utils as _kernel_utils
    except BaseException as exc:  # noqa: BLE001
        facts["error"] = f"{type(exc).__name__}: {exc}"
        return facts

    binding = getattr(_kernel_utils, "torch_gpu_device", None)
    facts["module_device_count"] = getattr(_kernel_utils, "DEVICE_COUNT", None)
    # Identity check: the single-card fallback shim is bound to the same name.
    facts["torch_gpu_device_is_real_switch"] = binding is torch.cuda.device
    facts["torch_gpu_device_repr"] = repr(binding)[:200]
    for name in ("CUDA_STREAMS", "WEIGHT_BUFFERS", "ABSMAX_BUFFERS"):
        value = getattr(_kernel_utils, name, None)
        facts[name.lower() + "_len"] = None if value is None else len(value)

    by_device: dict[str, int] = {}
    try:
        for _, param in model.named_parameters():
            key = str(param.device)
            by_device[key] = by_device.get(key, 0) + param.numel()
    except BaseException as exc:  # noqa: BLE001
        facts["parameter_walk_error"] = f"{type(exc).__name__}: {exc}"
    facts["parameters_by_device"] = by_device
    facts["cuda_devices_holding_parameters"] = sorted(d for d in by_device if d.startswith("cuda"))

    # Rotary caches are sized by DEVICE_COUNT; length 1 on a two-card box means unsloth saw one card.
    for module in getattr(model, "modules", lambda: [])():
        cached = getattr(module, "multi_gpu_cos_cached", None)
        if cached is not None:
            facts["rotary_cache_slots"] = len(cached)
            break
    return facts


def multi_gpu_failures(facts: dict | None, *, expected_cards: int) -> list[str]:
    """The rules, separated from the reading so they can be driven on CPU.

    Deliberately NOT asserting that the parameters are spread across both
    cards. Whether unsloth shards them or pins them to cuda:0 is exactly what
    this leg is being run to find out, and a rule written before the answer is
    a rule written to match whatever happens.
    """
    if not facts:
        return [
            "the multi-GPU facts are missing, so nothing about the "
            "DEVICE_COUNT > 1 path was measured on this run"
        ]
    if facts.get("error"):
        return [f"unsloth.kernels.utils could not be read: {facts['error']}"]

    out: list[str] = []
    seen = facts.get("device_count")
    if seen != expected_cards:
        out.append(
            f"this leg exists to exercise the multi-card path and torch sees "
            f"{seen} card(s), not {expected_cards} -- the driver pinned it, so "
            f"every assertion below would measure the single-card branch under "
            f"a multi-GPU name"
        )
        return out

    if facts.get("module_device_count") != expected_cards:
        out.append(
            f"unsloth.kernels.utils.DEVICE_COUNT is "
            f"{facts.get('module_device_count')!r} while torch sees "
            f"{expected_cards}; the module was imported before the cards were "
            f"visible, so its bindings are the single-card ones"
        )
    if not facts.get("torch_gpu_device_is_real_switch"):
        out.append(
            f"torch_gpu_device is not torch.cuda.device but "
            f"{facts.get('torch_gpu_device_repr')!r} -- the nullcontext "
            f"fallback, which performs NO device switch, so this run covered "
            f"the same path a pinned leg already covers"
        )
    for name in ("cuda_streams", "weight_buffers", "absmax_buffers"):
        length = facts.get(name + "_len")
        if length is None or length < expected_cards:
            out.append(
                f"unsloth.kernels.utils.{name.upper()} has {length!r} entries "
                f"for {expected_cards} cards, so a dequant on the second card "
                f"has no stream or buffer of its own"
            )
    slots = facts.get("rotary_cache_slots")
    if slots is not None and slots < expected_cards:
        out.append(
            f"the per-device rotary cache has {slots} slot(s) for "
            f"{expected_cards} cards, so the model was built while unsloth "
            f"believed there was one"
        )
    if not facts.get("cuda_devices_holding_parameters"):
        out.append("no parameter is on a CUDA device at all")
    return out


def peft_adapter_keys(model) -> dict:
    """The names PEFT itself gives this adapter's tensors, off the live model.

    ``PeftModel.save_pretrained`` writes exactly
    ``get_peft_model_state_dict(self, ...)`` (peft 0.20.0, peft_model.py), so
    calling the same function on the model that was just saved reproduces the
    key set the file is SUPPOSED to hold. That is the oracle a raw tensor read
    is missing: safetensors deserializes any well-formed file, whatever the keys
    are called, and PEFT's loader then matches by name.

    Derived rather than restated: no key list, no prefix, no target-module names
    appear here, so a legitimate peft renaming moves both sides at once and only
    a save that disagrees with the running peft is a difference.

    Returns ``{"keys": [...]}`` or ``{"error": "..."}``; never raises, because
    every outcome has to reach ``saved_adapter_failures`` as a verdict rather
    than as a traceback out of the payload.
    """
    try:
        from peft import get_peft_model_state_dict
        return {"keys": sorted(get_peft_model_state_dict(model))}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def _compare_adapter_keys(saved: set, peft_keys: dict | None) -> dict:
    """Saved tensor names against the ones PEFT names for the live model.

    Three answers, and they are not the same failure:

    * MISSING -- PEFT names a tensor the file does not carry. On reload peft
      warns ("Found missing adapter keys while loading the checkpoint",
      peft_model.py) and leaves that module's adapter at its initial value, so
      the weight is silently dropped.
    * UNEXPECTED -- the file carries a LoRA tensor under a name PEFT does not
      use. ``set_peft_model_state_dict`` ends in
      ``model.load_state_dict(..., strict=False)`` and nothing reads the
      returned ``unexpected_keys``, so those tensors are ignored without a word.
      Measured on peft 0.20.0: stripping ``base_model.model.`` from a valid
      adapter, or leaving the adapter name in (``lora_B.default.weight``, what
      filtering ``model.state_dict()`` by hand produces instead of using
      ``get_peft_model_state_dict``), reloads with every lora_B back at zero and
      raises nothing. The file still holds two nonzero tensors called lora_B, so
      the count this function exists to reinforce reads green on it.
    * EXTRA, non-LoRA -- recorded and NOT failed. ``save_pretrained`` may write
      more than the adapter (an embedding, a modules_to_save copy) and that is
      not a name PEFT would have to match.
    """
    if not peft_keys or not peft_keys.get("keys"):
        return {
            "keys_checked": False,
            "keys_error": (peft_keys or {}).get("error")
            or "the live model was never asked what these tensors should be called",
        }
    expected = set(peft_keys["keys"])
    unmatched = saved - expected
    return {
        "keys_checked": True,
        "keys_missing": sorted(expected - saved),
        "keys_unexpected": sorted(k for k in unmatched if LORA_MARKER in k.lower()),
        "keys_extra": sorted(k for k in unmatched if LORA_MARKER not in k.lower()),
    }


def verify_saved_adapter(
    adapter_dir,
    expected: dict | None = None,
    peft_keys: dict | None = None,
) -> dict:
    """Read the serialized adapter back and say what is in it.

    Everything downstream of the save runs on the in-memory model, so the only
    thing that ever looked at the file was a filename test. A tensor read rather
    than a PEFT reload, because it runs on a card already holding a 4-bit model
    and the failure modes worth naming (unreadable, empty, non-finite, all zero)
    are visible in the tensors themselves.

    What is NOT visible in the tensors is whether PEFT would consume them, since
    it matches by NAME and ignores what it does not recognise. ``peft_keys`` is
    ``peft_adapter_keys(model)`` for the model that was just saved, and
    comparing the two key sets is what turns "these bytes deserialize" into
    "these weights land". See ``_compare_adapter_keys``.

    ``nonzero_b_tensors`` is the load-bearing one, counted over the B matrices
    SPECIFICALLY: ``lora_B`` is zero at initialisation and only becomes non-zero
    once an update has been applied and saved, while ``lora_A`` is randomly
    initialised and nonzero before a single step. Counting every tensor
    therefore passed an adapter whose B matrices were all zero or dropped, whose
    output is still zero through B, so reloading it restores the base model.

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
    if state["config_readable"]:
        state.update(_reconstruct_adapter_config(adapter_dir, expected))

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
    b_tensors = 0
    nonzero_b = 0
    total = 0
    for name, tensor in tensors.items():
        try:
            is_b = LORA_B_MARKER in name.lower()
            b_tensors += int(is_b)
            floating = tensor.is_floating_point()
            if floating and not bool(tensor.isfinite().all()):
                non_finite.append(name)
            if bool(tensor.count_nonzero()):
                nonzero += 1
                nonzero_b += int(is_b)
            total += int(tensor.numel())
        except Exception as exc:  # noqa: BLE001
            non_finite.append(f"{name}: {type(exc).__name__}")
    state["tensors"] = len(tensors)
    state["parameters"] = total
    state["non_finite_tensors"] = non_finite[:10]
    state["nonzero_tensors"] = nonzero
    state["b_tensors"] = b_tensors
    state["nonzero_b_tensors"] = nonzero_b
    state.update(_compare_adapter_keys(set(tensors), peft_keys))
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
    # Valid JSON is not enough: `{}` parses but PEFT cannot load it.
    if state.get("config_loadable") is False:
        failures.append(
            f"the saved adapter's adapter_config.json parses but PEFT cannot "
            f"rebuild an adapter from it, so reloading this directory fails: "
            f"{state.get('config_load_error')}"
        )
    if state.get("config_differences"):
        failures.append(
            f"the saved adapter_config.json describes a different adapter than "
            f"the one that was trained: {state['config_differences']}"
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
    # Check lora_B: lora_A is random and nonzero before training, so B carries the trained update.
    b_tensors = state.get("b_tensors")
    if not b_tensors:
        failures.append(
            f"not one of the {state['tensors']} saved tensors is a lora_B matrix "
            f"({state.get('files')}), so this file cannot say whether training "
            f"reached the adapter. lora_B is the only weight in here that starts "
            f"at a known value, and without it the reading is unusable rather "
            f"than good."
        )
    elif not state.get("nonzero_b_tensors"):
        failures.append(
            f"every one of the {b_tensors} saved lora_B matrices is zero (of "
            f"{state['tensors']} tensors, {state.get('nonzero_tensors')} nonzero). "
            f"lora_B starts at zero and only an applied optimizer step moves it, so "
            f"this adapter contributes nothing and reloading it would restore the "
            f"base model."
        )
    # peft silently ignores unknown keys, so wrongly named tensors reload as the base model.
    if state.get("keys_checked") is False:
        failures.append(
            f"the saved adapter's tensor names were never checked against the "
            f"ones PEFT gives this model, so nothing here says the weights "
            f"would be loaded rather than ignored: {state.get('keys_error')}"
        )
    if state.get("keys_missing"):
        failures.append(
            f"PEFT names {len(state['keys_missing'])} adapter tensors the saved "
            f"file does not carry, so reloading leaves those modules at their "
            f"initial values: {state['keys_missing'][:5]}"
        )
    if state.get("keys_unexpected"):
        failures.append(
            f"{len(state['keys_unexpected'])} saved LoRA tensors are named "
            f"something PEFT does not use for this model, so its loader ignores "
            f"them silently and the reload restores the base model: "
            f"{state['keys_unexpected'][:5]}"
        )
    return failures


def canary_failures(run: dict, *, require: bool) -> list[str]:
    """The canary assertion, as an EXACT match rather than a substring.

    The completion trained on is ``CANARY + eos_token`` and decoding strips the
    special tokens, so a healthy greedy decode returns the canary and nothing
    else. ``CANARY in generated`` also accepts ``'__UNSLOTH__!!!<anything>'``,
    which is what a stopping or EOS regression produces -- the model learned the
    target and no longer knows where to stop -- reporting green on a broken
    inference path.

    Surrounding whitespace is the one normalisation allowed, being a decoder
    artefact rather than a change in what the model emitted.
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
            f"run {run.get('run_index')} did not emit the canary " f"{CANARY!r}; got {generated!r}"
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
    # Every watched package, so control vs canary failures are attributable by diff. The keys
    # above stay as they are: the committed reference carries them and report.py reads them.
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

    ``None`` means the file does not say, which is not "it matches": a trace
    with no declared length cannot be shown to describe the run in hand, and the
    caller treats it as such.
    """
    config = ref.get("config")
    if not isinstance(config, dict):
        return None
    value = config.get("max_steps")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Hub repo ids among the reference-defining fields, compared without case.
_REPO_ID_KEYS = frozenset({"model", "resolved_checkpoint"})
_HUB_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+")


def _is_hub_repo_id(value: Any) -> bool:
    """``owner/name`` and not a directory here. A local checkpoint path keeps its case:
    on a case-sensitive filesystem /models/Foo and /models/foo are different weights."""
    return (
        isinstance(value, str)
        and _HUB_REPO_ID_RE.fullmatch(value) is not None
        and not os.path.exists(value)
    )


# Settings that define the experiment; any mismatch means no comparison. `repeat` is excluded,
# and dataset_digest is used instead of the path so edited rows are detected.
REFERENCE_DEFINING_SETTINGS = (
    "max_steps",
    "dataset_digest",
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
    environment: dict | None = None,
) -> dict:
    """Compare against a committed reference. Never an equality check.

    ``max_steps``, the step count of the run being judged, is mandatory. A
    reference is a trace of one specific run, and a run of a different length is
    a different run: the fp16 scaler burns its first few steps on overflows, the
    learning-rate schedule is constant only because the run is short, and step N
    of a 3-step run is not the step N the 10-step trace recorded. Comparing
    across counts is arithmetic that succeeds and means nothing, so the mismatch
    gets its own status and the numbers are never touched. ``reference_failures``
    turns it into a failure; nothing here can turn it into a pass.

    ``max_steps`` used to be the only setting checked and is not the only one
    with that property: the reference records the whole ``config`` block, and
    README names the learning rate, the optimizer and the model as things that
    invalidate the file, so ``config``, ``model`` and the resolved checkpoint
    are compared under the same refuse-before-comparing rule.

    ``environment`` is the same rule applied to the HARDWARE, which is the one
    thing the reference records about itself that nothing compared. The file
    carries ``environment.gpu_name`` and ``gpu_capability`` -- "Tesla T4",
    "sm_75" -- because a loss trace belongs to the card it was captured on: the
    T4 has no bf16 and resolves attention to xformers, so the same code on
    another card produces a different curve, and band-checking across them
    reports a hardware difference as a code regression. The requirement is
    DERIVED from the reference's own environment block rather than restated as
    "must be a T4", so a reference recaptured on other hardware moves the gate
    with it. A reference that DOES name its card and a run that cannot name its
    own is ``hardware_unverified`` rather than a skip, because the alternative
    is a gate that switches itself off exactly when the probe fails.

    The settings are optional and default to not-compared, so an older caller
    and an older reference both keep working: a key the reference does not carry
    is listed in ``config_unchecked`` rather than treated as a mismatch. "It
    does not say" is neither "it differs" nor "it matches" -- but that is what
    the REFERENCE does not say. What the run does not say about hardware the
    reference does name is a refusal.
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

    # Step-count gate first, so no deviation is computed across different runs.
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

    # Hardware gate: the T4 has no bf16 and uses xformers, so another GPU moves the curve.
    ref_env = ref.get("environment") if isinstance(ref.get("environment"), dict) else {}
    live_env = environment if isinstance(environment, dict) else {}
    hardware_pairs: list[tuple[str, Any, Any]] = []
    unverified: list[str] = []
    for key in ("gpu_name", "gpu_capability"):
        expected = ref_env.get(key)
        observed = live_env.get(key)
        if expected is None:
            # Reference has no card recorded: skip, but record the skip.
            if observed is not None:
                verdict["config_unchecked"].append(key)
            continue
        if observed is None:
            unverified.append(key)
            continue
        hardware_pairs.append((key, expected, observed))
    # A live probe with no card (error or no CUDA) is a refusal, not a skip.
    if unverified:
        verdict["status"] = "hardware_unverified"
        verdict["config_differences"] = [
            f"{key}: reference {ref_env.get(key)!r}, this run reported nothing"
            for key in unverified
        ]
        verdict["note"] = (
            f"{reference_path.name} was captured on "
            f"{ref_env.get('gpu_name')} ({ref_env.get('gpu_capability')}) and "
            f"this run did not report {', '.join(unverified)}: "
            f"{live_env.get('error') or 'no live hardware fingerprint'}. The "
            "trace is of that card, so without knowing this run's card nothing "
            "here can be compared. Fix the environment probe on the kernel, or "
            "recapture the reference (references/README.md)."
        )
        return verdict
    hardware_differences = [
        f"{key}: reference {expected!r}, this run {observed!r}"
        for key, expected, observed in hardware_pairs
        if expected != observed
    ]
    if hardware_differences:
        verdict["status"] = "hardware_mismatch"
        verdict["config_differences"] = hardware_differences
        verdict["note"] = (
            f"{reference_path.name} was captured on "
            f"{ref_env.get('gpu_name')} ({ref_env.get('gpu_capability')}) and "
            f"this run is on {(environment or {}).get('gpu_name')} "
            f"({(environment or {}).get('gpu_capability')}). The trace is of "
            "that card -- fp16 without bf16, xformers attention -- so the "
            "numbers are not comparable and any deviation here would be the "
            "hardware, not the code. Run this leg on the reference's card, or "
            "recapture the reference (references/README.md)."
        )
        return verdict

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
        # Pin present on only one side: record as unchecked rather than refuse or skip silently.
        if observed is None and ref.get(key) is None:
            continue
        if observed is None or ref.get(key) is None:
            verdict["config_unchecked"].append(key)
            continue
        observed_pairs.append((key, ref[key], observed))
    for key, expected, observed in observed_pairs:
        if key in _REPO_ID_KEYS and _is_hub_repo_id(expected) and _is_hub_repo_id(observed):
            # Hub repo ids are case-insensitive; the revision still pins the weights.
            if expected.casefold() == observed.casefold():
                continue
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

    # Check step coordinates first: values are zipped positionally.
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
                # Present on one side only is a shape change, not drift.
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
            # Handle NaN explicitly: the fp16 reference contains NaN grad_norms and NaN > tol is False.
            # NaN equals NaN; NaN against a number is a deviation.
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
            # Infinities too: any pairing with inf divides to NaN and would pass silently.
            # Equal signed infinities match; anything else is a deviation.
            cur_inf = new in (float("inf"), float("-inf"))
            ref_inf = ref_val in (float("inf"), float("-inf"))
            if cur_inf or ref_inf:
                if new != ref_val:
                    verdict["deviations"].append(
                        {
                            "step": cur.get("step"),
                            "field": field,
                            "reference": old[field],
                            "observed": cur[field],
                            "relative": None,
                            "note": "an infinity on one side only, or opposite infinities",
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
    """Turn a reference verdict into failure strings. Separate so the path from
    "out of band" to "the job goes red" is testable without a GPU: a band check
    never observed to fail is not yet a check.
    """
    if verdict["status"] == "out_of_band":
        return [f"metrics outside +/-{rel_tol:.0%} of the reference: " f"{verdict['deviations']}"]
    if verdict["status"] == "length_mismatch":
        return ["reference has a different number of logged steps: nothing was compared"]
    # Refusals are fatal: an uncomparable reference looks like cover and is not.
    if verdict["status"] in (
        "step_count_mismatch",
        "reference_step_count_unknown",
        "config_mismatch",
        "step_mismatch",
        "hardware_mismatch",
        "hardware_unverified",
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

    The last of the three is the one a short run needs. Under fp16 the gradient
    scaler logs ``grad_norm: NaN`` on a skipped step, and a run whose every step
    was skipped applied no optimizer update at all: the weights at the end are
    the weights at the start, while the loss is finite, the adapter saves and
    generation produces text, so the run reports as a training test having done
    no training. That is exactly what a step count trimmed too far produces.
    """
    failures: list[str] = []
    losses = [m["loss"] for m in metrics]
    if any(l != l or l in (float("inf"), float("-inf")) for l in losses):
        failures.append(f"non-finite loss: {losses}")
    if len(losses) > 1 and not losses[-1] < losses[0]:
        failures.append(f"loss did not decrease over the run: {losses}")
    # Only decidable when grad_norm was logged. Check finiteness, not just NaN: fp16 overflow
    # also reports inf.
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
    # On by default: batched generation has broken here repeatedly.
    ap.add_argument(
        "--check-batched-generation",
        dest = "check_batched_generation",
        action = "store_true",
        default = True,
    )
    ap.add_argument(
        "--no-check-batched-generation",
        dest = "check_batched_generation",
        action = "store_false",
    )
    # Off by default: installs llama.cpp and merges the adapter, which is slow.
    ap.add_argument(
        "--export-gguf",
        dest = "export_gguf",
        action = "store_true",
        default = False,
    )
    # A model may legitimately override the request (gpt-oss forces MXFP4).
    ap.add_argument("--gguf-quantization", default = "q8_0")
    ap.add_argument(
        "--gguf-accept", default = "", help = "comma separated; defaults to the requested one"
    )
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    # Under fp16 the scaler overflows and skips the first three steps on this model, so short runs
    # need --init-loss-scale to apply any update. See pin_initial_loss_scale.
    ap.add_argument("--max-steps", type = int, default = 10)
    # Off by default: only needed for short --max-steps, and changing it invalidates the reference.
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
    # Higher rates overflow fp16 more often, and each overflow skips a step.
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
    ap.add_argument(
        # Off by default: doubles the leg's train time.
        "--compare-naive-trl",
        action = "store_true",
        default = False,
        help = "also train the same rows with plain TRL and report both traces",
    )
    ap.add_argument(
        # The only leg where DEVICE_COUNT > 1 bindings are reachable; other payloads pin
        # CUDA_VISIBLE_DEVICES.
        "--require-multi-gpu",
        dest = "require_multi_gpu",
        action = "store_true",
        default = False,
        help = "assert unsloth bound its multi-card code path, and record where "
        "the weights landed",
    )
    ap.add_argument(
        # All cards visible, weights on one: see the from_pretrained call.
        "--single-device",
        dest = "single_device",
        action = "store_true",
        default = False,
        help = "load with device_map={'': 0} while leaving both cards visible",
    )
    ap.add_argument(
        # Compare against a declared count, not device_count() on both sides.
        "--expected-cards",
        dest = "expected_cards",
        type = int,
        default = 2,
    )
    ap.add_argument(
        "--kernel-provenance",
        dest = "kernel_provenance",
        action = "store_true",
        default = False,
        help = "record which fast kernels loaded and where each resolved from",
    )
    ap.add_argument(
        # Separate process after the cycles: two 4bit models on one T4 risk OOM.
        "--vision-run",
        dest = "vision_run",
        action = "store_true",
        default = False,
        help = "also drive run_vision_t4.py and fold its verdict into this report",
    )
    ap.add_argument(
        # Only for models the control arm cannot load; a training OOM still fails.
        "--control-oom-is-ok",
        dest = "control_oom_is_ok",
        action = "store_true",
        default = False,
        help = "a plain-TRL OOM before the first step is reported, not failed",
    )
    ap.add_argument("--label", default = "t4-smoke")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    # Child mode: exactly one cycle, report to disk, no assertions.
    if args.cycle >= 0:
        try:
            run = train_once(args, args.cycle)
        except BaseException as exc:
            # Write a partial report on crash so readings taken before it survive; `cycle_error` marks it
            # as incomplete.
            partial = {
                "run_index": args.cycle,
                "cycle_error": f"{type(exc).__name__}: {exc}"[:2000],
                "partial": True,
                "multi_gpu": _LAST_MULTI_GPU_FACTS,
            }
            (outdir / "cycle_report.json").write_text(
                json.dumps(partial, indent = 2), encoding = "utf-8"
            )
            raise
        for entry in run["metrics"]:
            _log(
                f"    step {entry['step']}  loss={entry['loss']!r}  "
                f"grad_norm={entry.get('grad_norm')!r}"
            )
        _log(f"    generated: {run['generated']!r}")
        (outdir / "cycle_report.json").write_text(json.dumps(run, indent = 2), encoding = "utf-8")
        return 0

    # Each cycle in a fresh process: in-process repeats leak state and disagree from step one.
    # Config and environment are read first so a crashed run still reports its versions.
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
    # Recorded in the parent so it survives dead cycles and travels into captured references.
    config["dataset_digest"] = dataset_digest(Path(args.dataset))
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
            # Export settings must be forwarded to the child, which is what runs train_once.
            ("--gguf-quantization", args.gguf_quantization),
            ("--gguf-accept", args.gguf_accept),
            # Only the cycle loads the model, so only it can read the multi-card bindings.
            ("--expected-cards", args.expected_cards),
        ):
            cmd += [flag, str(value)]
        if args.export_gguf:
            cmd.append("--export-gguf")
        # Forwarded explicitly to the child.
        if args.kernel_provenance:
            cmd.append("--kernel-provenance")
        if args.require_multi_gpu:
            cmd.append("--require-multi-gpu")
        if args.single_device:
            cmd.append("--single-device")
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

    # Control arm after the cycles in a fresh process: avoids OOM and unsloth's import-time patches.
    naive = None
    if args.compare_naive_trl:
        naive_dir = outdir / "naive_trl"
        naive_dir.mkdir(parents = True, exist_ok = True)
        _log("=== plain-TRL control arm (fresh process, unsloth not imported) ===")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "naive_trl_compare.py"),
            "--outdir",
            str(naive_dir),
        ]
        # Use the repo unsloth resolved (a pre-quantised sibling): the plain path quantising the
        # original materialises 16bit weights and OOMs.
        control_model = runs[0].get("resolved_checkpoint") or args.model
        for flag, value in (
            ("--model", control_model),
            ("--dataset", args.dataset),
            ("--max-steps", args.max_steps),
            ("--batch-size", args.batch_size),
            ("--grad-accum", args.grad_accum),
            ("--max-seq-length", args.max_seq_length),
            ("--learning-rate", args.learning_rate),
            ("--lora-r", args.lora_r),
            ("--lora-alpha", args.lora_alpha),
            ("--optim", args.optim),
        ):
            cmd += [flag, str(value)]
        # rc ignored: the child always writes a report, which is the single source of truth.
        subprocess.run(cmd)
        naive_file = naive_dir / "naive_trl_report.json"
        if naive_file.exists():
            naive = json.loads(naive_file.read_text(encoding = "utf-8"))
        else:
            naive = {"error": "the plain-TRL process wrote no report"}

    vision = None
    if args.vision_run:
        vision_dir = outdir / "vision"
        vision_dir.mkdir(parents = True, exist_ok = True)
        _log("=== vision training run (fresh process, after the cycles) ===")
        vision_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_vision_t4.py"),
            "--outdir",
            str(vision_dir),
            "--model",
            args.model,
            "--max-seq-length",
            str(args.max_seq_length),
            # Few steps: a vision step on a T4 takes ~100s.
            "--max-steps",
            "3",
            "--samples",
            "8",
        ]
        if args.export_gguf:
            vision_cmd.append("--export")
        # rc ignored, as for the control arm.
        subprocess.run(vision_cmd)
        vision_file = vision_dir / "vision_report.json"
        if vision_file.exists():
            vision = json.loads(vision_file.read_text(encoding = "utf-8"))
        else:
            vision = {"error": "the vision process wrote no report"}

    report: dict = {
        "label": args.label,
        "model": args.model,
        "resolved_checkpoint": runs[0].get("resolved_checkpoint"),
        "resolved_revision": runs[0].get("resolved_revision"),
        # check_reference refuses runs captured with a different configuration.
        "config": config,
        "environment": env,
        "runs": runs,
        "metrics": runs[0]["metrics"],
        "failures": [],
    }

    failures: list[str] = []

    # -2. the vendored fast kernels and the attention choice. Read off cycle 0 (constant per cycle).
    if args.kernel_provenance:
        report["kernels"] = runs[0].get("kernels")
        report["attention"] = runs[0].get("attention")
        kernel_broken = vision_kernel_failures(
            runs[0].get("kernels"),
            runs[0].get("attention"),
            capability = str(env.get("gpu_capability", "")),
        )
        report["kernel_failures"] = kernel_broken
        failures += kernel_broken

    # -1.5 the multi-card bindings. Read off cycle 0 (bound once at import).
    if getattr(args, "require_multi_gpu", False):
        report["multi_gpu"] = runs[0].get("multi_gpu")
        multi_broken = multi_gpu_failures(
            runs[0].get("multi_gpu"),
            expected_cards = args.expected_cards,
        )
        report["multi_gpu_failures"] = multi_broken
        failures += multi_broken

    # -1. the plain-TRL control arm, reported side by side and NOT asserted
    # equal: two library stacks do not produce one fp16 trajectory.
    if args.vision_run:
        report["vision"] = vision
        if not vision:
            report["vision_failures"] = ["the vision run produced no report at all"]
        else:
            # The child already ruled on itself; do not re-derive the verdict.
            report["vision_failures"] = list(vision.get("failures") or [])
            if vision.get("error"):
                report["vision_failures"].append(f"the vision run crashed: {vision['error']}"[:400])
        failures += report["vision_failures"]

    if args.compare_naive_trl:
        report["naive_trl"] = naive
        naive_broken = comparison_failures(
            naive, report["metrics"], allow_oom = args.control_oom_is_ok
        )
        report["naive_trl_failures"] = naive_broken
        failures += naive_broken

    # 0. the pins, if this leg claims to be a control
    if args.pins:
        pins = load_pins(args.pins)
        # Derive probes from the pin file so every pin is looked up.
        resolved = versions_for_pins(pins)
        broken = pin_failures(pins, resolved)
        report["pins"] = {"file": args.pins, "requested": pins, "failures": broken}
        failures += broken

    # 1. bitwise run-to-run, EVERY extra cycle against the baseline rather than
    # just the second. report.py renders the top-level keys; per-cycle detail goes under `cycles`.
    if len(runs) > 1:
        cycles: dict = {}
        for other in runs[1:]:
            cycles[str(other["run_index"])] = compare_metrics(runs[0]["metrics"], other["metrics"])
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

    # 4b. batched generation reproduces one-at-a-time greedy output.
    if args.check_batched_generation:
        for run in runs:
            failures += [
                f"run {run['run_index']}: {f}"
                for f in batched_generation_failures(run.get("batched_generation"), args.model)
            ]

    # 4c. the GGUF export, and whether the exported file runs. Rules live in gguf_export.py.
    if args.export_gguf:
        from gguf_export import export_failures, run_failures

        accept = tuple(
            q.strip() for q in (args.gguf_accept or args.gguf_quantization).split(",") if q.strip()
        )
        # A skipped export is excused only if another cycle exported.
        exported = [run for run in runs if not (run.get("gguf_export") or {}).get("skipped")]
        if not exported:
            failures.append(
                "every cycle skipped the GGUF export, so the leg asked for one "
                "and never produced a file"
            )
        for run in exported:
            failures += [
                f"run {run['run_index']}: {f}"
                for f in export_failures(run.get("gguf_export"), accept_quantizations = accept)
            ]
            # Only check running once a file exists, to avoid double reporting.
            if (run.get("gguf_export") or {}).get("ggufs"):
                failures += [
                    f"run {run['run_index']}: {f}" for f in run_failures(run.get("gguf_run"))
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
            environment = env,
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
