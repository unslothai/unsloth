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
from training_evidence import LORA_B_MARKER, LORA_MARKER  # noqa: E402
from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    load_pins,
    pin_failures,
    resolved_versions,
    versions_for_pins,
)

# MUST run before torch is imported anywhere: CUBLAS_WORKSPACE_CONFIG is read
# when cuBLAS initialises, and setting it afterwards is silently ignored.
enable_full_determinism()

CANARY = "__UNSLOTH__!!!"
PROMPT_TEMPLATE = "### Question:\n{question}\n### Answer:\n"
SEED = 3407

# The smallest instruct model that still exercises the real loader path. It must
# fit on ONE T4 alongside a second copy of this test on the session's other T4.
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
    # _init_scale rather than a fresh GradScaler: the object may be a subclass
    # (ShardedGradScaler and friends) that replacing would drop. Safe because
    # training has not started, so the lazy _scale tensor does not exist and
    # get_scale() still reads _init_scale, which is how the check below can
    # confirm the pin took.
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
        # LOCAL REPRODUCTION ONLY, never on the target hardware. Unsloth prefers
        # flash-attention, then xformers, then SDPA, and on a T4 that resolves
        # to xformers, the path this test exists to cover, so the flag is off by
        # default and Kaggle does not use it. It exists because xformers ships
        # no backward kernel for some newer architectures (Blackwell raises
        # NotImplementedError: No operator found for
        # memory_efficient_attention_backward), which makes the payload
        # impossible to reproduce on such a box. Forcing SDPA changes the
        # numeric path, so a local run under it is evidence about the HARNESS,
        # not about T4 numerics.
        from unsloth.utils import attention_dispatch
        attention_dispatch.HAS_XFORMERS = False
        _log("force-sdpa: HAS_XFORMERS pinned False (local repro only)")

    set_all_seeds_fast(SEED)
    det_state = set_deterministic_algorithms(warn_only = not args.strict_deterministic)

    rows = load_canary_rows(Path(args.dataset))

    t0 = time.time()
    # float16 unconditionally: T4 is sm_75 and has no bf16. Pinned rather than
    # left to the loader so the local reproduction and the Kaggle run take the
    # same numeric path wherever they can share one.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
        dtype = torch.float16,
    )
    load_seconds = time.time() - t0
    # Which repository, and which commit of it. `load_in_4bit=True` redirects
    # through Unsloth's FLOAT_TO_INT_MAPPER, so the config name is not the name
    # asked for, and the loader explicitly DROPS `revision=` once a remap
    # happened (unsloth/models/loader.py::_revision_for_resolved_repo). Pinning
    # the load is therefore unavailable; recording what was loaded is what makes
    # a silent in-place re-upload attributable.
    _config = getattr(model, "config", None)
    resolved_checkpoint = getattr(_config, "_name_or_path", None)
    resolved_revision = getattr(_config, "_commit_hash", None)
    _log(f"loaded {resolved_checkpoint} @ {resolved_revision}")

    # Bound to a name so the saved config is checked against the adapter that
    # was actually requested, rather than against a list repeated further down.
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
        # Constant schedule, no warmup: over 3 steps a warmup would spend the
        # whole run at a fraction of the target LR, and a linear decay would
        # make step 3's update depend on max_steps. Constant keeps the reference
        # meaningful and the overfit strong enough for the canary.
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

    # Adapter save, then read the serialized weights back off disk. A filename
    # is not evidence: save_pretrained can leave an empty, truncated or all-zero
    # adapter_model.safetensors and every later assertion still passes, since
    # inference runs on the in-memory model. A file read rather than a second
    # FastLanguageModel load answers "can these weights be consumed" without a
    # second 4-bit load on a card already hosting one.
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
        # Asked of the model that was just saved, so the file is compared
        # against what PEFT calls these tensors rather than against a list this
        # payload would have to keep in step with peft by hand.
        peft_keys = peft_adapter_keys(model),
    )
    _log(f"saved adapter: {json.dumps(saved_adapter)}")

    # Inference on the trained, in-memory model, greedy so the output is a
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

    # Batched generation, on the SAME trained model, immediately after the
    # single-prompt generation above. Prompts of deliberately different lengths,
    # because a batch of equal-length prompts pads nothing and would report a
    # green left-padding check that never padded.
    batch_prompts = [
        PROMPT_TEMPLATE.format(question = row["question"]) for row in rows[: max(BATCH_SIZES)]
    ]
    while len(batch_prompts) < max(BATCH_SIZES):
        # The canary dataset is small. Pad the LIST (not the tensors) by
        # reusing questions with a varying prefix, which keeps the token
        # lengths spread rather than repeating one length.
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

    # GGUF export, opt-in. Placed AFTER generation deliberately: the export
    # merges the adapter into the base weights, and doing that before the
    # canary and batched-generation checks would have them measure a different
    # model from the one training produced.
    gguf_export_record = None
    gguf_run_record = None
    if getattr(args, "export_gguf", False):
        from gguf_export import export_gguf, llama_cpp_facts, run_gguf

        # unsloth must be imported before unsloth_zoo.llama_cpp, which raises
        # "Please install Unsloth via pip install unsloth!" otherwise. It is,
        # by the time train_once runs, but the import stays local so a payload
        # that never exports does not pay for it.
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

        gguf_export_record = export_gguf(
            model,
            tokenizer,
            os.path.join(args.outdir, f"gguf_run{run_index}"),
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
        "metrics": stats.logs,
        "generated": generated,
        # Both, because they answer different questions when this goes red:
        # `canary_found` says training reached the weights at all, `canary_exact`
        # is the assertion, and the gap between them is the signature of a
        # stopping/EOS regression rather than a training one.
        "batched_generation": batched,
        "gguf_export": gguf_export_record,
        "gguf_run": gguf_run_record,
        "canary_found": CANARY in generated,
        "canary_exact": generated.strip() == CANARY,
        "prompt": prompt,
        "adapter_files": saved_files,
        "saved_adapter": saved_adapter,
        # The repository the loader actually read, and its commit. The requested
        # name is not the loaded name (load_in_4bit goes through Unsloth's
        # FLOAT_TO_INT_MAPPER), and a mirror repo re-uploaded in place moves the
        # trajectory with no change in this repository. Recorded so the band
        # check can say so.
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
            # "It does not say" is not "it differs": a PEFT version that
            # renamed a field is recorded rather than turned into a failure.
            unchecked.append(key)
            continue
        got = getattr(config, key)
        if isinstance(wanted, (list, tuple, set)) or isinstance(got, (list, tuple, set)):
            same = sorted(got or []) == sorted(wanted or [])
        else:
            same = got == wanted
        if not same:
            differences.append(f"{key}: trained with {wanted!r}, saved {got!r}")
    out["config_differences"] = differences
    out["config_unchecked"] = unchecked
    return out


# Batch sizes to cross-check against one-at-a-time generation. 1 is the
# baseline and is generated separately; the rest must reproduce it exactly.
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
    # Imported here, not at module scope, matching every other torch user in
    # this file: the module is loaded before unsloth is installed in some
    # paths, and a top-level torch import would move the failure to import
    # time. Kernel unsloth-probe-defaultleg-723c28 trained all 10 steps and
    # then died on `NameError: name 'torch' is not defined` right here.
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _gen(batch: list) -> list:
        enc = tokenizer(batch, return_tensors = "pt", padding = True).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                temperature = None,
                top_p = None,
                top_k = None,
                use_cache = True,
                # `x or y` is wrong here: pad_token_id 0 is a perfectly ordinary
                # id (Qwen and Llama both use low ids) and is falsy, so `or`
                # silently substitutes the EOS id for it and pads the batch with
                # end-of-sequence tokens. Test for None.
                pad_token_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        # Slice by the PADDED width, not by each prompt's own length: with left
        # padding every row starts at the same column, and using the unpadded
        # length would re-read the tail of the prompt as if it were output.
        width = enc["input_ids"].shape[1]
        return [tokenizer.decode(row[width:], skip_special_tokens = True) for row in out]

    lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
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
    for size in BATCH_SIZES:
        outs: list = []
        for start in range(0, len(prompts), size):
            outs.extend(_gen(prompts[start : start + size]))
        result["batched"][str(size)] = outs
        result["agrees"][str(size)] = outs == singles
    # Read AGAIN, after all the generating. #2138 was a silent override applied
    # inside the inference path, so the value set at the top of this function is
    # not evidence of the value that was used.
    result["padding_side_after"] = tokenizer.padding_side
    return result


def batched_generation_failures(batch: dict | None) -> list[str]:
    """Turn a `batched_generation` record into failures, vacuity included."""
    if not batch:
        return ["batched generation was never run"]
    out = []
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
    for size, agreed in (batch.get("agrees") or {}).items():
        if not agreed:
            out.append(
                f"batch size {size} did not reproduce one-at-a-time greedy output "
                f"(#3699/#1456): {batch.get('batched', {}).get(size)!r} != "
                f"{batch.get('singles')!r}"
            )
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
    # Syntactically valid JSON is not a loadable adapter: `{}` parses, and used
    # to pass here, leaving the leg green on a file PEFT cannot reconstruct.
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
    # The B matrices, not the tensor count: lora_A is randomly initialised and
    # nonzero before training starts, so a file whose B matrices were all zero
    # or dropped still had nonzero tensors and was accepted, while producing
    # nothing, the adapter's contribution going through B.
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
    # Names, after values: peft matches its state dict by key and ignores what
    # it does not recognise, so a file full of healthy nonzero lora_B tensors
    # under names it does not use reloads as the base model without raising.
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
    # Every package this CI watches, read from the installed distributions, and
    # what makes a canary-leg failure attributable: control and canary run the
    # same payload, so the diff of these two blocks is the entire difference
    # between green and red. The keys above are kept as they were, the committed
    # reference carrying them and the summary renderer reading them.
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


# Every setting that defines which experiment a trace is a trace OF. A run
# differing in any of them is not comparable to the reference whatever the
# numbers do, and references/README.md says widening the band is never the
# answer.
#
# `repeat` is deliberately absent: each cycle is a fresh process on identical
# configuration, so how many ran changes none of them, and refusing on it would
# reject a --repeat 3 run against a --repeat 2 reference for no reason.
#
# `dataset_digest` is here rather than `dataset`: the PATH says nothing about
# what is in the file, and canary_dataset.jsonl is inside the workflow's paths
# filter, so a run triggered BY editing it is exactly the run that would
# otherwise be band-checked against a trace of the old rows.
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

    # The step-count gate comes FIRST and returns, so no partially reassuring
    # "worst relative deviation" is computed from two different runs.
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

    # The hardware, before any number is compared and on the same terms as the
    # step count. A trace is of one card: the T4 has no bf16 and resolves
    # attention to xformers, so another GPU moves the curve for reasons that
    # have nothing to do with the code under test, and the deviations would be
    # reported as a regression. Only the count of GPUs was ever checked, on the
    # kernel, which cannot see this file at all.
    ref_env = ref.get("environment") if isinstance(ref.get("environment"), dict) else {}
    live_env = environment if isinstance(environment, dict) else {}
    hardware_pairs: list[tuple[str, Any, Any]] = []
    unverified: list[str] = []
    for key in ("gpu_name", "gpu_capability"):
        expected = ref_env.get(key)
        observed = live_env.get(key)
        if expected is None:
            # The reference does not say, which is not "it differs": an older
            # trace captured before the block carried a card keeps working, and
            # the skip is recorded rather than silent.
            if observed is not None:
                verdict["config_unchecked"].append(key)
            continue
        # The reference DOES say. From here the requirement is the reference's
        # own, so a run that cannot show what it ran on has not met it.
        if observed is None:
            unverified.append(key)
            continue
        hardware_pairs.append((key, expected, observed))
    # A live probe that produced no card is a refusal, not a skip. main()
    # records environment = {"error": ...} when environment_fingerprint()
    # raises, and the fingerprint omits every gpu_* key outright when
    # torch.cuda.is_available() is False, so both of those -- and a caller that
    # hands over no environment at all -- used to land in config_unchecked and
    # let the band check report "ok" without establishing the card. That is the
    # hardware gate above disabled by the one failure it most needs to survive.
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

    # The rest of the configuration on the same terms as the step count: refuse
    # before comparing a single number, since these settings define a different
    # experiment rather than a drift within one.
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
        # Neither side records it: nothing claimed, nothing to report. Present
        # on ONE side is a pin that did not run, recorded rather than skipped
        # silently. Still not a refusal, since "it does not say" is not "it
        # differs", but an invisible skip reads as a comparison that passed, and
        # this is exactly where a checkpoint pin sits unenforced for months.
        # report.py puts config_unchecked on the summary.
        if observed is None and ref.get(key) is None:
            continue
        if observed is None or ref.get(key) is None:
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

    # The step coordinates, before any value is compared: the lists are zipped
    # positionally, so a shifted, duplicated or reordered `step` pairs values
    # describing different iterates and the band check reports on arithmetic it
    # invented. Safe to be strict because the leg carrying a reference is the
    # control, whose library set is pinned to the one the trace was captured
    # with and whose pin failure is itself fatal, so the trainer cannot renumber
    # its steps without the pins going red first.
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
                # Present on one side only is a change in the SHAPE of what the
                # trainer logged, not a numeric drift, and no tolerance covers
                # it.
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
            # NaN, explicitly, which is why the arithmetic is not left to itself.
            # Under fp16 the gradient scaler logs a NaN grad_norm on every
            # skipped step, so the committed reference genuinely contains NaNs,
            # and left to the subtraction abs(x - NaN) is NaN, NaN > rel_tol is
            # False, and the step passes whatever it holds -- including the case
            # that matters most, a step that used to overflow and no longer
            # does. NaN equals NaN here; NaN against a number is a deviation.
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
            # Infinities, for the same reason one step further along. An fp16
            # overflow logs infinity as readily as NaN, so the reference holds
            # them, and every pairing an infinity takes part in divides to NaN:
            # abs(inf - inf) / inf and abs(inf - 1.0) / inf alike. NaN > rel_tol
            # is False and max(worst, NaN) returns worst, so the entry was
            # accepted AND left no trace in worst_rel, including a step that
            # used to be finite and now overflows. Equal signed infinities are
            # the unchanged case; everything else is a deviation, decided before
            # the division.
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
    # Refusals, loud and fatal: a reference that cannot be compared is worth
    # less than no reference, because it looks like cover and is not. Never
    # demote these to a warning.
    if verdict["status"] in (
        "step_count_mismatch",
        "reference_step_count_unknown",
        "config_mismatch",
        "step_mismatch",
        # The card the trace was captured on. Same rule as the settings: a
        # band checked across hardware succeeds arithmetically and means
        # nothing, and its deviations read as a code regression.
        "hardware_mismatch",
        # And the same rule when the card is unreadable rather than wrong: a
        # reference that names its hardware is only comparable against a run
        # that can name its own.
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
    # Only decidable where grad_norm was logged at all: a trainer version that
    # stops logging it says nothing about whether steps were applied, and
    # inferring "all skipped" from that silence would be a failure invented
    # rather than found.
    #
    # Finite, not merely non-NaN. An fp16 overflow reports the norm as inf at
    # least as readily as NaN (clip_grad_norm_ over an inf gradient returns inf)
    # and `inf == inf` is True, so a NaN-only test counted every skipped step as
    # applied and a run that applied nothing reported green. The loss check
    # above already treats inf as non-finite; this one did not.
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
    # On by default: batched generation is the surface that has broken most
    # often here (#3699, #1066, #1456, #2138) and a leg that quietly skips it
    # is a leg that stops covering it.
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
    # OFF by default, unlike the batched-generation check above. That one is
    # pure compute on a model already in memory; this one installs llama.cpp
    # and merges the adapter, about 40s for a 0.6B on top of a ~10-46s install,
    # so a leg opts in rather than every payload paying for it.
    ap.add_argument(
        "--export-gguf",
        dest = "export_gguf",
        action = "store_true",
        default = False,
    )
    # What the exported filename is allowed to say. More than one because a
    # model may legitimately override the request: gpt-oss answers q8_0 with
    # "Overriding to MXFP4 format" by design, and failing on documented
    # behaviour would be a failure invented rather than found.
    ap.add_argument("--gguf-quantization", default = "q8_0")
    ap.add_argument(
        "--gguf-accept", default = "", help = "comma separated; defaults to the requested one"
    )
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    # 3 steps, and the whole reason --init-loss-scale exists.
    #
    # Measured twice. Under fp16 the dynamic gradient scaler starts at 65536,
    # halves on every overflow and skips the step it overflowed on, and on this
    # model the first three steps overflow every time (the committed 10-step
    # reference has grad_norm NaN at steps 1, 2 and 3). So a 3-step run of the
    # ORIGINAL configuration applies zero optimizer updates, the loss does not
    # move and the canary never forms: an early 3-step attempt emitted
    # '#1\ndef my_function():...'.
    #
    # Pinning the scaler's starting scale below the overflow point (see
    # pin_initial_loss_scale) gives those three steps back as real updates, and
    # optimisation_failures() asserts that happened. The 10-step trajectory is
    # the honest yardstick for what 3 updates buy: loss 1.75 after three applied
    # updates and 0.18 after four, so the canary has less margin at 3 steps than
    # at 10. A T4 run reporting the canary missing while the scaler shows
    # updates applied is the signal to raise this back up, not to relax the
    # canary.
    ap.add_argument("--max-steps", type = int, default = 10)
    # Off by default. The pin is for short runs: the scaler overflows the first
    # three steps, so anything under about five applies no updates at all. At
    # the default of 10 the run reaches step 4 on its own and learns the canary,
    # and leaving the scaler alone keeps the committed reference applicable,
    # since a different starting scale is a different rounding of the same
    # gradients. Set it (e.g. 2048, below the 8192 the reference reaches after
    # three halvings) only with a shorter --max-steps and a reference recaptured
    # with both.
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
    # 1e-3 with the prompt masked out. Higher rates overflow fp16 far more often,
    # and each overflow is a skipped step this short run cannot spare.
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

    # Parent mode: each cycle in a FRESH process, measured rather than assumed.
    # Two in-process cycles disagreed from the first logged step (6.4375 vs
    # 6.2367) while two separate processes agreed bitwise on every step.
    # Something in the first cycle's model load, patching or allocator state
    # leaks into the second, so an in-process repeat tests the leak rather than
    # the code and would report a false regression on a reproducible run. A
    # fresh process is also what a user re-running a notebook gets.
    #
    # The config and environment are read BEFORE the cycles, so a run that dies
    # in one still says which library set it died with: control red / canary
    # green is a bisect only if the red leg reported its versions.
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
    # WHAT trained, beside the settings that say how. Recorded here, in the
    # parent, so a run whose cycles all died still says which rows it was asked
    # to train on, and so the value travels into any reference captured from
    # this report.
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
            # The export settings travel too. Forgetting them is not a
            # hypothetical: --export-gguf reached the PARENT on kernel
            # unsloth-probe-default-gguf-637565, was parsed there, and never
            # reached the child that actually runs train_once, so every cycle
            # reported `gguf_export: null` and the leg failed with "GGUF export
            # was never run" while the driver log plainly showed the flag on
            # the command line.
            ("--gguf-quantization", args.gguf_quantization),
            ("--gguf-accept", args.gguf_accept),
        ):
            cmd += [flag, str(value)]
        if args.export_gguf:
            cmd.append("--export-gguf")
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
        # The repo and commit the loader read, travelling into any reference
        # captured from this report so the band check can refuse against a
        # mirror re-uploaded in place.
        "resolved_checkpoint": runs[0].get("resolved_checkpoint"),
        "resolved_revision": runs[0].get("resolved_revision"),
        # The whole block travels into any reference captured from this report:
        # check_reference refuses to compare a run against a trace captured with
        # a different configuration.
        "config": config,
        "environment": env,
        "runs": runs,
        "metrics": runs[0]["metrics"],
        "failures": [],
    }

    failures: list[str] = []

    # 0. the pins, if this leg claims to be a control
    if args.pins:
        pins = load_pins(args.pins)
        # The probe list is derived from the pin file: a pin outside
        # GOAL_PACKAGES was looked up in a table nobody had asked about it and
        # came back "not installed".
        resolved = versions_for_pins(pins)
        broken = pin_failures(pins, resolved)
        report["pins"] = {"file": args.pins, "requested": pins, "failures": broken}
        failures += broken

    # 1. bitwise run-to-run, EVERY extra cycle against the baseline rather than
    # just the second: --repeat 3 asks for three fresh processes, and a third
    # that disagreed used to be collected, stored and never looked at.
    #
    # The top-level keys stay as they were, report.py rendering `identical`,
    # `first_diff_step` and `max_abs_diff` off this dict, but now summarise
    # every comparison; the per-cycle detail goes under `cycles`.
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

    # 4b. batched generation reproduces one-at-a-time greedy output. Gated on
    # the flag because the legs that carry no reference still want it, while a
    # payload run for something else (a bisect, a single-cycle debug) should not
    # be forced to pay for it.
    if args.check_batched_generation:
        for run in runs:
            failures += [
                f"run {run['run_index']}: {f}"
                for f in batched_generation_failures(run.get("batched_generation"))
            ]

    # 4c. the GGUF export, and whether the exported file runs. Both rules live
    # in gguf_export.py so the gptoss and vision payloads can reuse them
    # without copying the two traps (the sibling directory, and the tuple that
    # install_llama_cpp returns) into three files.
    if args.export_gguf:
        from gguf_export import export_failures, run_failures
        accept = tuple(
            q.strip() for q in (args.gguf_accept or args.gguf_quantization).split(",") if q.strip()
        )
        for run in runs:
            failures += [
                f"run {run['run_index']}: {f}"
                for f in export_failures(run.get("gguf_export"), accept_quantizations = accept)
            ]
            # Only ask whether it RUNS once the export produced something; a
            # missing file already failed above and would otherwise be reported
            # twice under two different descriptions.
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
