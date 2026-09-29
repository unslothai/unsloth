# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""gpt-oss-20b LoRA on a single T4: does the compile-and-offload path hold?

The payload behind the `gptoss` leg, written first as a FEASIBILITY PROBE. The
question is not "is the loss right" but "can a 20B checkpoint be loaded,
LoRA-trained and generated from at all on 16GB of sm_75, on the compiled float32
path this card forces".

What the probe found, so the report can be read against it: kernels
`unsloth-t4-ci-8161ceb9` and `unsloth-t4-ci-7ab727f1`, 2026-08-11, Tesla T4 /
sm_75 / 14.56 GB. It works, and three of the four things likeliest to break are
not in the path at all.

* **MXFP4 is never reached.** `unsloth/gpt-oss-20b` is an MXFP4 checkpoint and
  MXFP4 has no backward pass in unsloth_zoo, but `load_in_4bit=True` makes
  Unsloth's FLOAT_TO_INT_MAPPER redirect the load to the NF4
  `unsloth/gpt-oss-20b-unsloth-bnb-4bit`. The probe confirmed the redirect from
  `model.config._name_or_path`, recorded on every run: a change to that mapping
  would move this leg onto a checkpoint that cannot train unnoticed.
* **No bf16, and Unsloth already knows.** gpt-oss is in `FORCE_FLOAT32`. The
  probe saw `UNSLOTH_FORCE_FLOAT32=1`, `fp16=False`, `bf16=False`, and
  `UNSLOTH_FORCE_CUSTOM_DTYPE` pinning `down_projs` and `mlp.router` to float32.
  That path exists for this card and nothing else in CI exercises it.
* **One deliberate offload, and nothing else.** 12.7 GB reserved of 14.56, no
  `hf_device_map`, every parameter on `cuda:0` EXCEPT `model.embed_tokens.weight`,
  which unsloth puts in RAM on purpose -- `Unsloth: Offloading embeddings to RAM
  to save 1.08 GB`, with forward hooks carrying ids down and vectors back up.
  That was read as a spill twice before anyone looked at the name; 579133440 is
  exactly 201088 x 2880, this checkpoint's vocab by its hidden size. Placement is
  counted on every run rather than assumed, and the embedding is excused only
  when its hook flag is set, because an embedding that reached the CPU without
  them is a real bug that looks identical in a device count.
* **torch.compile engages**: 32 unique graphs, 779 calls captured, 2 graph
  breaks, both `_warnings.warn`. A silent fall back to eager leaves every other
  number healthy while the leg's coverage goes unexercised, so this is asserted.

What it asserts:

1. The model loads, in a recorded dtype, with every parameter on the one
   visible T4. Offload to CPU, disk or meta is a FAILURE rather than a slow
   pass: it is the documented result of this leg (a 20B checkpoint fitting and
   training on 16GB, with about 1.8GB to spare) ceasing to hold, and every other
   number in the report survives it.
2. Training runs the requested steps, every logged loss is finite, and the
   optimizer applied something -- a run with all-zero gradients looks healthy
   everywhere else and trained nothing. Decided on the ADAPTER, fingerprinted
   before and after training, with `grad_norm` as fallback rather than source:
   this leg saves and reloads no adapter, so a trainer that stops logging that
   field would otherwise take the only evidence with it. See
   training_evidence.py. There is no committed reference band, the run being too
   short and the model too large for a per-step trace to be worth capturing, and
   a band nobody can recapture cheaply gets disabled the first time it is
   inconvenient.
2a. The forced-float32 path was taken: on a card without bf16, `fp16`/`bf16`
   must both be off and `UNSLOTH_FORCE_FLOAT32` must be set. This is the
   coverage the leg uniquely claims, and it was recorded on every run and
   asserted on none.
3. `torch.compile` captured at least one graph DURING TRAINING
   (`--require-compile`, on by default). The Dynamo counters are process-global
   and loading a 20B checkpoint fills them, so the assertion is on the delta
   across `trainer.train()`, not the total.
4. Generation after training returns non-empty text without raising, catching a
   training run that "succeeds" and leaves the model unusable, which on a
   quantised offloaded path is a real outcome.

`--probe` records all of those and fails on none, for the one-off feasibility
runs: a probe must come back with evidence, not a nonzero exit and a truncated
report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import shutil
import tempfile
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from training_evidence import (  # noqa: E402
    adapter_fingerprint,
    adapter_update,
    update_verdict,
)
from versions import (  # noqa: E402
    GOAL_PACKAGES,
    flatten_versions,
    resolved_versions,
)

CANARY = "__UNSLOTH__!!!"
SEED = 3407

DEFAULT_MODEL = "unsloth/gpt-oss-20b"


def _log(msg: str) -> None:
    print(f"[gptoss-t4] {msg}", flush = True)


def compile_counters(before: dict | None = None) -> dict:
    """What `torch.compile` actually did, from Dynamo's own bookkeeping.

    `unique_graphs` decides whether compilation engaged: zero means every region
    fell back to eager, whatever the banner said. `graph_breaks` sits beside it
    because capturing graphs and breaking a hundred times is a different,
    reportable state, and on a card with no bf16 a break is often the first
    symptom of a dtype the compiled path refused.

    The counters are process-global and never reset, and loading a model through
    Unsloth compiles plenty before `trainer.train()`, so the absolute number
    cannot answer "did TRAINING compile": an entirely eager training path still
    leaves the loader's graphs standing. Pass the pre-training reading as
    ``before`` and the delta is the answer; `failures_for` asserts on the delta.
    """
    state: dict = {"available": False}
    try:
        import torch._dynamo.utils as dynamo_utils

        counters = dynamo_utils.counters
        stats = dict(counters.get("stats", {}))
        breaks = dict(counters.get("graph_break", {}))
        state = {
            "available": True,
            "unique_graphs": int(stats.get("unique_graphs", 0)),
            "calls_captured": int(stats.get("calls_captured", 0)),
            "graph_breaks_total": sum(int(v) for v in breaks.values()),
            # Truncated: free text, and a pathological run produces hundreds.
            "graph_break_reasons": sorted(breaks)[:10],
        }
    except Exception as exc:  # noqa: BLE001
        state = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if before and before.get("available") and state.get("available"):
        for key in ("unique_graphs", "calls_captured", "graph_breaks_total"):
            state[f"{key}_delta"] = state[key] - int(before.get(key, 0))
        state["before"] = {
            k: before.get(k) for k in ("unique_graphs", "calls_captured", "graph_breaks_total")
        }
    return state


def placement(model) -> dict:
    """Where the weights ended up, counted rather than trusted.

    A 20B checkpoint on a 16GB card either offloads or does not fit, and "did it
    offload" is answerable only by looking. `hf_device_map` is the
    accelerate-side answer and is absent when nothing dispatched; the parameter
    walk is always available and is what distinguishes a model that quietly
    landed on the CPU (correct, slow) from one on meta (loaded nothing).
    """
    counts: dict = {}
    off_gpu: list = []
    try:
        for name, param in model.named_parameters():
            key = str(param.device)
            counts[key] = counts.get(key, 0) + param.numel()
            if not key.startswith("cuda"):
                # NAMED, not just counted. `{'cpu': 579133440}` is a number
                # nobody can act on; `model.embed_tokens.weight` is a bug
                # report. The two readings cost the same walk, and the first
                # one already reached hardware twice before anyone could say
                # which tensor it was.
                off_gpu.append({"name": name, "numel": param.numel(), "device": key})
    except Exception as exc:  # noqa: BLE001
        counts = {"error": f"{type(exc).__name__}: {exc}"}
    # Is the one tensor allowed off the card the one unsloth DELIBERATELY put
    # there? `offload_embedding` moves the input embedding to RAM and installs
    # a pre/post forward hook pair that carries the ids down and the vectors
    # back up (`unsloth/models/vision.py:_install_offload_embedding_hooks`).
    # The flag it sets is the difference between that optimisation and a
    # weight that landed on the CPU by accident, and the two are identical in a
    # device count.
    embed = {}
    try:
        module = model.get_input_embeddings()
        weight = getattr(module, "weight", None)
        for name, candidate in model.named_modules():
            if candidate is module:
                embed["module"] = name
                break
        embed["weight_name"] = f"{embed.get('module')}.weight" if "module" in embed else None
        embed["device"] = str(weight.device) if weight is not None else None
        embed["offload_hooks_installed"] = bool(
            getattr(module, "_unsloth_offload_hooks_installed", False)
        )
    except Exception as exc:  # noqa: BLE001
        embed = {"error": f"{type(exc).__name__}: {exc}"}

    device_map = getattr(model, "hf_device_map", None)
    return {
        "input_embedding": embed,
        "parameters_by_device": counts,
        # Largest first, capped: a genuinely offloaded model has thousands of
        # these and the report is read by a human.
        "off_gpu_parameters": sorted(off_gpu, key = lambda p: -p["numel"])[:20],
        "off_gpu_parameter_count": len(off_gpu),
        "hf_device_map_devices": (
            sorted({str(v) for v in device_map.values()}) if isinstance(device_map, dict) else None
        ),
        "offloaded": bool(
            isinstance(device_map, dict)
            and any(str(v) in ("cpu", "disk") for v in device_map.values())
        ),
    }


def memory() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "total_gb": round(props.total_memory / 1024**3, 2),
    }


def build_dataset(tokenizer, rows: list[dict]):
    """The canary rows as chat turns, through the model's own template.

    Through `apply_chat_template` rather than a hand-rolled prompt: gpt-oss has
    a template with channels and a reasoning-effort knob, and bypassing it would
    exercise a text format no user of this notebook ever produces.
    """
    from datasets import Dataset

    texts = []
    for row in rows:
        texts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ],
                tokenize = False,
                add_generation_prompt = False,
            )
        )
    return Dataset.from_dict({"text": texts})


def build_completion_dataset(tokenizer, rows: list[dict]):
    """The same rows as a PROMPT/COMPLETION pair, so the loss covers only the
    answer.

    Two columns rather than a collator: TRL treats a dataset with `prompt` and
    `completion` columns as prompt-completion and masks the prompt itself
    (`completion_only_loss` defaults to True for that shape). The older
    `DataCollatorForCompletionOnlyLM` route needs a response template string
    that has to match the chat template exactly, and a template change turns it
    silently into "mask nothing" -- which trains on everything and passes every
    assertion about losses.

    The prompt ends with the generation prompt, so the boundary is exactly where
    the model would start generating. That is what makes the mask meaningful
    rather than approximately right.
    """
    from datasets import Dataset

    prompts, completions = [], []
    for row in rows:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize = False,
                add_generation_prompt = True,
            )
        )
        completions.append(row["answer"])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def masking_evidence(trainer) -> dict:
    """Whether the prompt tokens are ACTUALLY masked out of the loss.

    This is the whole point of the feature and the one thing a loss curve
    cannot show: a run that masks nothing trains on prompt and answer alike,
    converges perfectly well, and reports numbers indistinguishable from a
    correct one. So the labels are read off a real collated batch.

    Never raises: a diagnostic that kills the leg is worse than a missing one.
    """
    record: dict = {}
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        labels = batch["labels"]
        total = int(labels.numel())
        masked = int((labels == -100).sum())
        record["label_tokens"] = total
        record["masked_tokens"] = masked
        record["masked_fraction"] = round(masked / total, 4) if total else None
        record["columns"] = sorted(batch.keys())
    except BaseException as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return record


def masking_failures(record: dict | None, *, expected: bool) -> list[str]:
    """The rule, as a pure function, so it is checkable without a GPU."""
    if not expected:
        return []
    if not record:
        return ["completions-only training was requested but no masking evidence was collected"]
    if record.get("error"):
        return [f"could not read the collated labels: {record['error']}"]

    total = record.get("label_tokens") or 0
    masked = record.get("masked_tokens")
    if not total:
        return ["the collated batch carried no labels at all"]
    if not masked:
        # The failure this function exists for. Every loss-based assertion
        # passes in this state.
        return [
            "completions-only training was requested and NOTHING was masked "
            f"({masked} of {total} label tokens are -100), so the prompt is in "
            f"the loss and this leg is not testing what it says it is"
        ]
    if masked >= total:
        return [
            f"every one of the {total} label tokens is masked, so there is no "
            f"completion left to learn from"
        ]
    return []


def train_and_infer(args) -> dict:
    import torch
    from unsloth import FastLanguageModel

    result: dict = {}

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        dtype = None,  # let the loader choose; T4 has no bf16
        max_seq_length = args.max_seq_length,
        load_in_4bit = True,
        full_finetuning = False,
    )
    result["load_seconds"] = round(time.time() - t0, 1)
    result["model_dtype"] = str(getattr(model, "dtype", None))
    # What was REALLY loaded. `unsloth/gpt-oss-20b` is MXFP4, which has no
    # backward pass at all (unsloth_zoo raises "Backwards pass using MXFP4 is
    # still under construction"), so load_in_4bit=True makes Unsloth's
    # FLOAT_TO_INT_MAPPER redirect to the NF4
    # `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, which does train. The config name
    # is therefore not the name asked for, and a change to that redirect would
    # silently move this leg onto a path that cannot train.
    model_config = getattr(model, "config", None)
    result["resolved_checkpoint"] = getattr(model_config, "_name_or_path", None)
    quant = getattr(model_config, "quantization_config", None)
    result["quantization"] = (
        {
            "method": str(getattr(quant, "quant_method", None)),
            "type": str(getattr(quant, "bnb_4bit_quant_type", None)),
        }
        if quant is not None
        else None
    )
    result["placement_after_load"] = placement(model)
    result["memory_after_load"] = memory()
    _log(f"loaded in {result['load_seconds']}s, dtype {result['model_dtype']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_r * 2,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
    )

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    # Prompt/completion when the loss should cover only the answer, one `text`
    # column otherwise. `dataset_text_field` must go with the second shape and
    # NOT the first: naming a text field TRL cannot find is how a
    # prompt-completion dataset silently falls back to training on everything.
    if args.train_on_completions:
        dataset = build_completion_dataset(tokenizer, rows)
    else:
        dataset = build_dataset(tokenizer, rows)
    result["train_on_completions"] = bool(args.train_on_completions)
    result["dataset_columns"] = sorted(dataset.column_names)

    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(
        output_dir = str(Path(args.outdir) / "trainer"),
        **({} if args.train_on_completions else {"dataset_text_field": "text"}),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        max_length = args.max_seq_length,
        max_steps = args.max_steps,
        learning_rate = 2e-4,
        warmup_steps = 0,
        lr_scheduler_type = "constant",
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.0,
        seed = SEED,
        data_seed = SEED,
        # fp16/bf16 are deliberately NOT set, the opposite of the tiny SFT
        # payload. gpt-oss is in Unsloth's FORCE_FLOAT32 list: on a card without
        # bf16 the loader sets UNSLOTH_FORCE_FLOAT32=1 and the RL/SFT patch
        # switches to float32 ("Unsloth: Switching to float32 training since
        # model cannot work with float16"), because fp16 autocast through the
        # MXFP4-derived weights produces infinities. Asking for fp16 would fight
        # that patch, and whichever won, the run would not be the one a notebook
        # user gets. What was chosen is recorded below.
        dataloader_num_workers = 0,
        dataloader_pin_memory = False,
        report_to = "none",
        save_strategy = "no",
    )
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = config,
    )

    # Read BEFORE training, off a real collated batch. After training the
    # dataloader has been consumed and a re-created one is not necessarily the
    # object the trainer used.
    if args.train_on_completions:
        result["masking"] = masking_evidence(trainer)
        _log(f"masking {json.dumps(result['masking'])}")

    # The precision after Unsloth's patches have had their say. On a T4 this
    # should be float32 and NOT fp16; fp16 here means FORCE_FLOAT32 stopped
    # firing and the infinities it prevents are back.
    result["precision"] = {
        "fp16": bool(getattr(trainer.args, "fp16", None)),
        "bf16": bool(getattr(trainer.args, "bf16", None)),
        "force_float32_env": os.environ.get("UNSLOTH_FORCE_FLOAT32"),
        "custom_dtype_env": os.environ.get("UNSLOTH_FORCE_CUSTOM_DTYPE"),
    }
    _log(f"precision {json.dumps(result['precision'])}")

    # The counters BEFORE training, so what training compiled is a subtraction
    # rather than an inference: loading a 20B checkpoint through Unsloth
    # compiles a great deal into the same process-global counter.
    compile_before = compile_counters()
    _log(f"compile counters before training: {json.dumps(compile_before)}")

    # The adapter before a single step, so "did the optimizer apply anything" is
    # a subtraction rather than a reading of what the trainer chose to log. This
    # leg saves and reloads no adapter, so without it the only evidence is
    # grad_norm, and a trainer that stops logging it leaves the leg asserting
    # nothing. See training_evidence.py.
    adapter_before = adapter_fingerprint(model)
    _log(f"adapter before training: {json.dumps(adapter_before)}")

    t0 = time.time()
    stats = trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)
    result["metrics"] = [
        {"step": entry.get("step"), "loss": entry.get("loss"), "grad_norm": entry.get("grad_norm")}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    result["train_metrics"] = {k: v for k, v in (stats.metrics or {}).items()}
    result["adapter_update"] = adapter_update(adapter_before, adapter_fingerprint(model))
    _log(f"adapter update: {json.dumps(result['adapter_update'])}")
    result["compile"] = compile_counters(before = compile_before)
    result["memory_after_train"] = memory()
    _log(
        f"trained {len(result['metrics'])} logged steps in "
        f"{result['train_seconds']}s; compile {result['compile']}"
    )

    # Inference in the notebook's own shape: chat template with a reasoning
    # effort, greedy decode, short.
    FastLanguageModel.for_inference(model)
    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[0]["question"]}],
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
            reasoning_effort = "low",
        ).to("cuda")
    except TypeError:
        # reasoning_effort is a gpt-oss template keyword; a template that does
        # not take it is a finding to record, not a crash.
        result["reasoning_effort_supported"] = False
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[0]["question"]}],
            add_generation_prompt = True,
            return_tensors = "pt",
            return_dict = True,
        ).to("cuda")
    else:
        result["reasoning_effort_supported"] = True

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
        )
    result["infer_seconds"] = round(time.time() - t0, 1)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens = True)
    result["generated"] = generated
    result["canary_found"] = CANARY in generated
    result["memory_peak"] = memory()

    # GGUF, LAST, because it merges a 20B checkpoint and the merge is the
    # heaviest thing in the leg. Anything after it would be measuring a session
    # that has just written ~28GB.
    #
    # ASK for q8_0, ACCEPT only mxfp4. That pairing looks backwards and is the
    # only one that works, which cost a probe to learn.
    #
    # gpt-oss overrides the request, out loud:
    #
    #   GPT-OSS does not support GGUF quantization (requested: q8_0).
    #     Overriding to MXFP4 format.
    #   GPT-OSS model - skipping additional quantizations
    #
    # The obvious response is to ask for mxfp4 directly. unsloth REJECTS it as
    # an input -- measured on kernel unsloth-probe-gptoss-r3-832c85:
    #
    #   Unsloth: Quant method = [mxfp4] not supported. Choose from below:
    #   [not_quantized] [fast_quantized] [quantized] [f32] [bf16] ...
    #
    # So the documented override is the ONLY route to an MXFP4 file, and the
    # leg has to travel it. Accepting only mxfp4 is what keeps that honest: a
    # run that somehow produced a real q8_0 would fail, which is right, because
    # gpt-oss q8_0 is documented impossible.
    if getattr(args, "export_gguf", False):
        from gguf_export import export_gguf, llama_cpp_facts, run_gguf

        install_log = ""
        llama_dir = None
        try:
            import contextlib
            import io

            # After `import unsloth`, which has happened by now: unsloth_zoo's
            # llama_cpp raises "Please install Unsloth via pip install unsloth!"
            # if it is reached first. A probe already lost a session to that.
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

        # NOT under args.outdir, and this is measured. `/kaggle/working` is
        # 21.0GB total; the gpt-oss export consumes 27.6GB of transient disk
        # (3 mxfp4 safetensors shards at 13.76GB plus the GGUF). Exporting
        # there fails in 2.8s with
        #   Unsloth: Failed saving locally - no disk space left
        # which is a disk fact wearing the costume of an export bug.
        #
        # `/tmp` on a Kaggle session is the overlay: 8656.9GB total, 1102.5GB
        # free, measured on kernel unsloth-probe-disk. tempfile honours TMPDIR
        # and lands there. The artifact keeps the RECORD -- path, size, seconds
        # -- and not the 27.6GB, which nobody wants collected anyway.
        gguf_dir = tempfile.mkdtemp(prefix = "gptoss_gguf_")
        record = export_gguf(
            model,
            tokenizer,
            gguf_dir,
            quantization = args.gguf_quantization,
        )
        record["disk_free_gb_before"] = round(shutil.disk_usage(gguf_dir).free / 1e9, 1)
        record["llama_cpp"] = facts
        result["gguf_export"] = record
        _log(f"gguf export: {json.dumps({k: v for k, v in record.items() if k != 'llama_cpp'})}")

        # Deliberately NOT run through llama.cpp here, and the reason is
        # measured: the prebuilt bundle is the -cpu build, and a 20B MXFP4 file
        # on 4 vCPUs is not a few seconds of work. The Default leg already
        # covers "the exported file runs" on a 610MB Q8_0, where the claim is
        # affordable. Asserting the export and not the run is a smaller claim,
        # and it is the one this leg can honestly make.
        ggufs = record.get("ggufs") or []
        result["gguf_ran"] = False
        if ggufs:
            _log(f"largest gguf: {ggufs[0]['path']} ({ggufs[0]['mb']} MB), not executed here")
    _log(f"generated {generated!r}")
    return result


def _placement_failures(placement: dict | None) -> list[str]:
    """Every parameter on the one visible GPU, or this leg measured something else.

    The driver gives each payload a single CUDA device
    (``CUDA_VISIBLE_DEVICES``), so on a healthy run every parameter reports
    ``cuda:0`` and ``hf_device_map`` is absent entirely. Anything else is one of
    the three ways a 20B checkpoint stops fitting: ``cpu``/``disk`` through
    accelerate's dispatch, ``meta`` for a shard that was never materialised, or
    a device the payload cannot see.

    Unreadable is a failure, not a skip, for the same reason it is one for the
    bf16 reading and the dynamo counters: the check that switches itself off
    when its instrument breaks is the one that never fires.
    """
    if not isinstance(placement, dict):
        return [
            "where the weights landed was never recorded, so whether the "
            "checkpoint fit on the GPU at all could not be established, and "
            "every other number in this report is produced either way"
        ]
    failures: list[str] = []
    counts = placement.get("parameters_by_device")
    if not isinstance(counts, dict) or not counts:
        failures.append(f"the parameter walk recorded no devices: {placement}")
    elif "error" in counts:
        failures.append(f"the parameters could not be walked: {counts['error']}")
    else:
        elsewhere = {
            device: n for device, n in counts.items() if not str(device).startswith("cuda")
        }
        if elsewhere:
            # The ONE tensor allowed off the card, and only on its own terms.
            # `offload_embedding` puts the input embedding in RAM and hooks the
            # lookup so ids go down and vectors come back up; measured saving
            # 1.08GB on gpt-oss-20b. Accepting "cpu" wholesale would excuse a
            # real spill, so this names the exact parameter AND requires the
            # hook flag: an embedding that reached the CPU without them is a
            # bug, and it looks identical in a device count.
            embed = placement.get("input_embedding") or {}
            deliberate = (
                embed.get("offload_hooks_installed") is True
                and str(embed.get("device", "")).startswith("cpu")
                and embed.get("weight_name")
            )
            unexplained = [
                p
                for p in (placement.get("off_gpu_parameters") or [])
                if not (deliberate and p.get("name") == embed.get("weight_name"))
            ]
            if unexplained or not (placement.get("off_gpu_parameters") or []):
                named = (
                    ", ".join(
                        f"{p.get('name')} ({p.get('numel')} on {p.get('device')})"
                        for p in unexplained
                    )
                    or "the walk recorded no names"
                )
                failures.append(
                    f"parameters are off the GPU: {elsewhere} [{named}] (all devices: "
                    f"{counts}), and unsloth's deliberate embedding offload does not "
                    f"account for them (input_embedding {embed}). This leg's result is "
                    f"that the 20B checkpoint fits and trains wholly on one T4; a run "
                    f"that offloads to CPU, disk or meta is not a slower version of "
                    f"that, it is a different run, and accelerate's offload does not "
                    f"support training at all"
                )
    # The accelerate-side answer, which is absent on a healthy run and says
    # `cpu`/`disk` when dispatch offloaded. Three-way: `placement()` always
    # writes a bool, so anything else is a record this file cannot read.
    offloaded = placement.get("offloaded")
    if offloaded is True:
        failures.append(
            f"the loader dispatched part of the model off the GPU: "
            f"hf_device_map devices {placement.get('hf_device_map_devices')}"
        )
    elif offloaded is not False:
        failures.append(f"whether the loader offloaded could not be established: {placement}")
    return failures


def failures_for(result: dict, args) -> list[str]:
    """The assertions, separated from the run so they can be unit-tested.

    Nothing here needs a GPU, which is the point: the pass/fail rule for a leg
    that costs a Kaggle session has to be checkable without one.
    """
    failures: list[str] = []
    if getattr(args, "export_gguf", False):
        from gguf_export import export_failures

        # MXFP4 only. gpt-oss refuses every other quantization by design, so a
        # wider accept list would let a silently-overridden export pass as
        # though the request had been honoured.
        failures += export_failures(result.get("gguf_export"), accept_quantizations = ("mxfp4",))
    failures += masking_failures(
        result.get("masking"), expected = bool(getattr(args, "train_on_completions", False))
    )
    metrics = result.get("metrics") or []
    if len(metrics) != args.max_steps:
        failures.append(f"expected {args.max_steps} logged steps, got " f"{len(metrics)}")
    losses = [m.get("loss") for m in metrics if m.get("loss") is not None]
    if not losses:
        failures.append("no loss was logged at all, so nothing trained")
    bad = [l for l in losses if l != l or l in (float("inf"), float("-inf"))]
    if bad:
        failures.append(f"non-finite loss: {losses}")

    # Did the optimizer apply anything? Every number above stays healthy with
    # all-zero gradients: the loss is finite, compilation engaged, and the
    # untrained base model still generates text.
    #
    # The adapter fingerprints before and after training decide it, with
    # grad_norm as fallback rather than the other way round. This USED to be
    # `if norms and not applied`, which passes on an empty list: a trainer that
    # stopped logging grad_norm silently took the only instrument this leg had,
    # and unlike the SFT leg there is no saved adapter to read back. See
    # training_evidence.py.
    update = update_verdict(metrics, result.get("adapter_update"))
    if update["verdict"] == "not_applied":
        failures.append(
            f"no optimizer update was applied: {update['detail']}, so the adapter "
            f"is the adapter it started with and this leg measured a forward pass "
            f"rather than LoRA training"
        )
    elif update["verdict"] == "non_finite":
        failures.append(
            f"the adapter holds non-finite weights after training: {update['detail']}. "
            f"gpt-oss is in the FORCE_FLOAT32 list precisely because reduced "
            f"precision through these weights produces infinities, so this is that "
            f"failure landing in the adapter rather than in the loss"
        )
    elif update["verdict"] != "applied":
        # Not `== "unverifiable"`: any verdict this file has not been taught
        # about is a failure rather than a silent pass.
        failures.append(
            f"whether the optimizer applied anything could not be established: "
            f"{update['detail']}. LoRA training on this path is the only thing "
            f"this leg claims, and the base model on its own produces every other "
            f"number in this report"
        )

    # The float32 path, the coverage this leg uniquely claims, recorded on every
    # run and asserted on none: a run that quietly went through fp16 logs finite
    # losses, compiles, generates and reports green while that path was never
    # exercised.
    #
    # Conditioned on the card, not hardcoded to T4: FORCE_FLOAT32 exists because
    # this hardware has no bf16, so on a card that has it, the patch not firing
    # is correct and a red would be this check's own bug.
    #
    # Three-way, not two-way. `is False` alone made the block conditional on a
    # reading allowed to be absent -- main() records `{"error": ...}` for the
    # whole environment when the probe raises -- so a torch build that changed
    # or failed `is_bf16_supported()` skipped this leg's one unique assertion
    # while everything else passed. Anything but a literal True or False is
    # unverifiable, and unverifiable is red here.
    environment = result.get("environment") or {}
    bf16_supported = environment.get("bf16_supported")
    if bf16_supported is not True and bf16_supported is not False:
        failures.append(
            f"the card's bf16 support was not recorded, so whether the forced "
            f"float32 path had to fire could not be established: "
            f"environment={environment}. This leg exists to cover that path on "
            f"hardware without bf16, and every other number in this report is "
            f"produced with or without it"
        )
    elif bf16_supported is False:
        precision = result.get("precision")
        if not precision:
            failures.append(
                "the training precision was never recorded, so whether the forced "
                "float32 path fired could not be established"
            )
        else:
            if precision.get("fp16") or precision.get("bf16"):
                failures.append(
                    f"the run trained in reduced precision on a card without bf16: "
                    f"{precision}. gpt-oss is in Unsloth's FORCE_FLOAT32 list because "
                    f"fp16 autocast through these weights produces infinities, so this "
                    f"is the patch having stopped firing, not a slow pass."
                )
            elif precision.get("force_float32_env") != "1":
                # The exact string, not truthiness: the loader writes "0" here
                # on its normal branch before deciding whether to force
                # (models/loader.py), so a nonempty check accepts the very
                # regression it is here to catch -- forcing off, fp16 and bf16
                # still false, leg green. Every production consumer reads
                # `== "1"`.
                failures.append(
                    f'UNSLOTH_FORCE_FLOAT32 is not "1": {precision}. The loader sets '
                    f'it to "0" and only writes "1" when the forcing actually fired, '
                    f"and the float32 path is the only thing this leg covers that "
                    f"nothing else in CI does."
                )

    # Where the weights ended up, recorded on every run since the feasibility
    # probe and asserted on none. "the 20B checkpoint fits and trains WHOLLY on
    # one T4" is the result this leg exists to hold, and it is the one that
    # degrades quietly: a loader or memory-management regression that sends
    # layers to the CPU still logs finite losses, still updates the adapter,
    # still compiles and still generates, so the leg reports green having
    # measured a different, slower thing. The probe's own margin is the reason
    # to check rather than assume -- 12.78 GB reserved of 14.56, about 1.8 GB.
    #
    # accelerate's offload is also inference-only ("This only supports
    # inference, not training" -- huggingface.co/docs/accelerate big model
    # inference), so a training run that reaches it is not a slower version of
    # this leg. It is a different one.
    failures.extend(_placement_failures(result.get("placement_after_load")))

    if args.require_compile:
        compiled = result.get("compile") or {}
        # The DELTA across training, not the process-global total, which is
        # nonzero the moment the loader compiles anything and so used to be
        # satisfied by an entirely eager training path.
        graphs = compiled.get("unique_graphs_delta")
        if not compiled.get("available"):
            failures.append(
                "torch._dynamo counters were unreadable, so whether "
                "torch.compile engaged could not be established: "
                f"{compiled.get('error')}"
            )
        elif graphs is None:
            # No baseline means no subtraction, and falling back to the absolute
            # count would assert on a number the LOADER made nonzero, turning
            # the one case where training cannot be isolated into the one most
            # likely to pass.
            failures.append(
                "the pre-training dynamo counters were not readable, so what "
                "training itself compiled cannot be separated from what loading the "
                "model compiled, and this leg requires compilation: "
                f"{compiled}"
            )
        elif graphs < 1:
            failures.append(
                "torch.compile captured zero graphs across training itself, so the "
                "training path ran eager whatever was compiled at load time. That is "
                "the silent fallback this leg exists to catch; it is a failure, not a "
                f"slow pass. Counters: {compiled}"
            )

    generated = result.get("generated")
    if generated is None:
        failures.append("generation did not run")
    elif not generated.strip():
        failures.append(
            "generation after training returned empty text, so the trained model is unusable"
        )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = DEFAULT_MODEL)
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--label", default = "gptoss")
    ap.add_argument("--max-steps", type = int, default = 3)
    ap.add_argument("--max-seq-length", type = int, default = 1024)
    ap.add_argument("--lora-r", type = int, default = 8)
    ap.add_argument("--max-new-tokens", type = int, default = 32)
    ap.add_argument(
        # On by default: the leg is specified to train on completions, and a
        # flag that has to be remembered at every call site is one that will be
        # forgotten at one of them.
        "--train-on-completions",
        dest = "train_on_completions",
        action = "store_true",
        default = True,
    )
    ap.add_argument("--no-train-on-completions", dest = "train_on_completions", action = "store_false")
    ap.add_argument("--export-gguf", dest = "export_gguf", action = "store_true", default = False)
    # q8_0 as the REQUEST. `mxfp4` is not an accepted input value; the override
    # is the only way to reach that format. See the export block above.
    ap.add_argument("--gguf-quantization", default = "q8_0")
    ap.add_argument("--require-compile", dest = "require_compile", action = "store_true", default = True)
    ap.add_argument("--no-require-compile", dest = "require_compile", action = "store_false")
    ap.add_argument(
        "--probe",
        action = "store_true",
        help = "record everything, assert nothing. For the "
        "one-off feasibility runs, whose job is to come "
        "back with evidence rather than an exit code",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    report: dict = {
        "label": args.label,
        "model": args.model,
        "leg": "gptoss",
        "probe": args.probe,
        "config": {
            k: getattr(args, k)
            for k in ("max_steps", "max_seq_length", "lora_r", "max_new_tokens", "require_compile")
        },
        "versions": {},
        "environment": {},
        "failures": [],
    }

    # Versions first, before anything can crash: a payload that died in the
    # loader still has to say which library set it died with, or the crash is
    # unattributable and the session was spent for nothing.
    report["versions"] = resolved_versions(
        GOAL_PACKAGES, import_check = ("torch", "transformers", "trl")
    )
    report["versions_flat"] = flatten_versions(report["versions"])
    _log("versions " + json.dumps(report["versions_flat"]))

    try:
        import torch
        props = torch.cuda.get_device_properties(0)
        report["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": props.name,
            "gpu_capability": f"sm_{props.major}{props.minor}",
            "gpu_total_gb": round(props.total_memory / 1024**3, 1),
            "gpu_count_visible": torch.cuda.device_count(),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "compile_disabled_env": os.environ.get("UNSLOTH_COMPILE_DISABLE"),
        }
    except Exception as exc:  # noqa: BLE001
        report["environment"] = {"error": f"{type(exc).__name__}: {exc}"}

    failures: list[str] = []
    try:
        result = train_and_infer(args)
        report.update(result)
        # The metrics key the launcher and report renderer already display, so
        # this leg needs no special case downstream.
        report["metrics"] = result.get("metrics", [])
        failures = failures_for(report, args)
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        report["traceback"] = traceback.format_exc()[-6000:]
        failures = [f"{type(exc).__name__}: {str(exc)[:600]}"]
        _log("EXCEPTION\n" + report["traceback"])
        report["memory_peak"] = memory()
        report["compile"] = compile_counters()

    report["observed_failures"] = failures
    if args.probe:
        # A probe reports; it does not judge. A human reads the verdict off
        # observed_failures, and the leg is not wired into CI until it can be.
        report["failures"] = []
        report["passed"] = True
    else:
        report["failures"] = failures
        report["passed"] = not failures

    (outdir / "t4_smoke_report.json").write_text(
        json.dumps(report, indent = 2, default = str), encoding = "utf-8"
    )
    print("T4_SMOKE_REPORT " + json.dumps(report, default = str), flush = True)
    for entry in failures:
        _log(f"OBSERVED FAILURE: {entry}")
    _log("T4_SMOKE_RESULT " + ("PASS" if report["passed"] else "FAIL"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
