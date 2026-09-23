# SPDX-License-Identifier: Apache-2.0
"""Matched SentenceTransformer training benchmark; isolate each implementation in its own process."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import median


SOURCE_FILES = (
    "unsloth/models/sentence_transformer.py",
    "unsloth/models/_sentence_transformer_unpadding.py",
    "unsloth/utils/attention_dispatch.py",
    "unsloth/utils/packing.py",
)


def select_wall_clock(time_module = time):
    # Unlike CLOCK_MONOTONIC, RAW is not rate-adjusted by clock synchronization.
    if hasattr(time_module, "CLOCK_MONOTONIC_RAW"):
        return (
            lambda: time_module.clock_gettime(time_module.CLOCK_MONOTONIC_RAW),
            "CLOCK_MONOTONIC_RAW",
        )
    return time_module.perf_counter, "perf_counter"


wall_clock, WALL_CLOCK_NAME = select_wall_clock()


def validate_step_clocks(wall_ms, cuda_ms):
    if not all(math.isfinite(value) and value > 0 for value in (wall_ms, cuda_ms)):
        raise RuntimeError(f"invalid step clocks: wall={wall_ms} ms, CUDA={cuda_ms} ms")
    # The synchronized wall window encloses both CUDA events. Allow small timer
    # resolution/rate differences, but reject materially inconsistent clocks.
    if cuda_ms > wall_ms * 1.01 + 0.05:
        raise RuntimeError(
            f"inconsistent step clocks ({WALL_CLOCK_NAME}): "
            f"wall={wall_ms} ms, CUDA={cuda_ms} ms; discard this arm"
        )


def parse_args():
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument(
        "--mode",
        choices = ("baseline", "candidate-off", "candidate-auto", "candidate-force"),
    )
    p.add_argument("--repo", type = Path)
    p.add_argument("--model-path", type = Path, help = "existing local checkpoint; no downloads")
    p.add_argument("--output", type = Path)
    p.add_argument("--dtype", choices = ("bf16", "fp16", "fp32"), default = "bf16")
    p.add_argument("--padding", choices = ("low", "moderate", "heavy"), default = None)
    p.add_argument(
        "--padding-fraction",
        type = float,
        default = None,
        help = "target padded-token fraction, 0 <= f < 1",
    )
    p.add_argument("--max-length", type = int, default = 128)
    p.add_argument("--batch-size", type = int, default = 32)
    p.add_argument("--seed", type = int, default = 4460)
    p.add_argument("--warmup", type = int, default = 10)
    p.add_argument("--iterations", type = int, default = 30)
    p.add_argument("--repeats", type = int, default = 3)
    p.add_argument("--learning-rate", type = float, default = 2e-5)
    p.add_argument("--peft", action = "store_true")
    p.add_argument(
        "--compile",
        action = "store_true",
        help = "apply upstream FastSentenceTransformer._apply_torch_compile after fingerprinting",
    )
    p.add_argument("--lora-r", type = int, default = 8)
    p.add_argument("--lora-alpha", type = int, default = 16)
    p.add_argument("--lora-targets", default = "query,key,value,dense")
    p.add_argument("--profile", "--profile-only", dest = "profile_only", action = "store_true")
    p.add_argument("--profile-dir", type = Path, default = None)
    p.add_argument(
        "--pairs-json",
        type = Path,
        default = None,
        help = "JSON list of [sentence_a, sentence_b]; real tokenizer path",
    )
    p.add_argument(
        "--self-test",
        action = "store_true",
        help = "run CPU-only helper validation and exit",
    )
    return p.parse_args()


def torch_dtype(name, torch):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def seed_all(seed, torch):
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_path(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_content_info(repo):
    h, files = hashlib.sha256(), {}
    for relative in SOURCE_FILES:
        path = repo / relative
        h.update(relative.encode("utf-8") + b"\0")
        if path.is_file():
            digest = sha256_path(path)
            h.update(bytes.fromhex(digest))
            files[relative] = digest
        else:
            h.update(b"<missing>")
            files[relative] = None
    return {"sha256": h.hexdigest(), "files": files}


def git_info(repo):
    def run(*argv):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *argv],
                text = True,
                capture_output = True,
                check = True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "revision": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
        "production_content": source_content_info(repo),
    }


def model_checkpoint_info(path):
    metadata = path / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    revision = metadata.read_text(encoding = "utf-8").splitlines()[0] if metadata.is_file() else None
    weights = path / "model.safetensors"
    return {
        "revision": revision,
        "model_safetensors_bytes": weights.stat().st_size,
        "model_safetensors_sha256": sha256_path(weights),
    }


def tensor_bytes(value):
    import torch
    raw = value.detach().cpu().contiguous()
    return raw.view(torch.uint8).numpy().tobytes()


def state_fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        canonical_name = name.replace("_orig_mod.", "")
        h.update(canonical_name.encode())
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(tensor_bytes(value))
    return h.hexdigest()


def fixture_fingerprint(batches):
    h = hashlib.sha256()
    for pair in batches:
        for batch in pair:
            for name, value in sorted(batch.items()):
                if hasattr(value, "shape"):
                    h.update(name.encode())
                    h.update(str(value.shape).encode())
                    h.update(str(value.dtype).encode())
                    h.update(tensor_bytes(value))
    return h.hexdigest()


def activation_info(model):
    names = (
        "_unsloth_unpadding_installed",
        "_unpadding_dispatch_count",
        "_unsloth_unpadding_dispatch_count",
    )
    result = {}
    for name in names:
        owner = (
            model if hasattr(model, name) else type(model) if hasattr(type(model), name) else None
        )
        if owner is not None:
            value = getattr(owner, name)
            result[name] = value.item() if hasattr(value, "item") else value
    return result


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    at = (len(values) - 1) * q
    lo, hi = int(at), min(int(at) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def stats(values):
    return {
        "median": median(values),
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
    }


def padding_fraction(a):
    value = (
        a.padding_fraction
        if a.padding_fraction is not None
        else {"low": 0.05, "moderate": 0.25, "heavy": 0.75}.get(a.padding, 0.15)
    )
    if not 0 <= value < 1:
        raise SystemExit("--padding-fraction must satisfy 0 <= f < 1")
    return value


def make_pairs(a):
    if a.pairs_json:
        rows = json.loads(a.pairs_json.read_text(encoding = "utf-8"))
        invalid = (
            not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(not isinstance(text, str) or not text.strip() for text in row)
                for row in rows
            )
        )
        if invalid:
            raise ValueError(
                "--pairs-json must contain a non-empty JSON list of two non-empty strings per row"
            )
        if len(rows) < a.batch_size:
            raise ValueError("--pairs-json must contain at least one full batch")
        return rows
    return [["synthetic-a", "synthetic-b"] for _ in range((a.warmup + a.iterations) * a.batch_size)]


def pair_fixture_info(path, rows):
    if path is None:
        return {"kind": "synthetic", "input_rows": len(rows)}
    info = {
        "kind": "json",
        "path": str(path.resolve()),
        "sha256": sha256_path(path),
        "input_rows": len(rows),
    }
    provenance_path = path.parent / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding = "utf-8"))
        if (
            provenance.get("pairs_file") == path.name
            and provenance.get("pairs_sha256") != info["sha256"]
        ):
            raise ValueError("pair fixture SHA-256 does not match provenance.json")
        if provenance.get("pair_count") not in (None, len(rows)):
            raise ValueError("pair fixture row count does not match provenance.json")
        info["provenance_path"] = str(provenance_path.resolve())
        info["provenance_sha256"] = sha256_path(provenance_path)
        info["dataset_revision"] = provenance.get("dataset_revision")
        info["model_revision"] = provenance.get("model_revision")
    return info


def full_pair_batches(rows, batch_size):
    """Build deterministic full MNRL batches without duplicate text across rows."""
    remaining, result = list(rows), []
    while len(remaining) >= batch_size:
        chosen, deferred, seen = [], [], set()
        for row in remaining:
            texts = set(row)
            if len(chosen) < batch_size and not (texts & seen):
                chosen.append(row)
                seen.update(texts)
            else:
                deferred.append(row)
        if len(chosen) < batch_size:
            break
        result.append(chosen)
        remaining = deferred
    if not result:
        raise ValueError("real pair fixture cannot form one duplicate-safe full batch")
    return result, len(remaining)


def load_model(a, torch):
    repo = str(a.repo.resolve())
    os.environ["PYTHONPATH"] = repo + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, repo)
    # Unsloth must initialize before importing sentence_transformers losses.
    from unsloth import FastSentenceTransformer

    kwargs = {"dtype": torch_dtype(a.dtype, torch), "load_in_4bit": False}
    if a.mode != "baseline":
        kwargs["use_unpadding"] = {
            "candidate-off": False,
            "candidate-auto": "auto",
            "candidate-force": True,
        }[a.mode]
    model = FastSentenceTransformer.from_pretrained(str(a.model_path), **kwargs)
    if a.peft:
        targets = [x for x in a.lora_targets.split(",") if x]
        model = FastSentenceTransformer.get_peft_model(
            model, r = a.lora_r, lora_alpha = a.lora_alpha, target_modules = targets
        )
    return model, FastSentenceTransformer


def apply_explicit_compile(model, fast_sentence_transformer, enabled):
    if enabled:
        model = fast_sentence_transformer._apply_torch_compile(model, mode = "default")
    return model


def compilation_evidence(model, enabled):
    from torch._dynamo.eval_frame import OptimizedModule

    wrapped = [
        name for name, module in model.named_modules() if isinstance(module, OptimizedModule)
    ]
    if enabled and not wrapped:
        raise RuntimeError("compiled lane has no OptimizedModule")
    return {"requested": enabled, "optimized_modules": wrapped}


def reset_compilation_counters(torch):
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    torch._dynamo.reset()
    counters.clear()
    metrics.reset()


def validate_compilation_execution(evidence):
    required = ("unique_graphs", "aot_autograd_ok", "generated_kernel_count")
    if any(evidence.get(name, 0) <= 0 for name in required):
        raise RuntimeError(f"compiled lane has no successful Inductor execution: {evidence}")


def compilation_execution_evidence():
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    evidence = {
        "scope": "successful graphs, not a fullgraph guarantee",
        "unique_graphs": counters["stats"]["unique_graphs"],
        "aot_autograd_ok": counters["aot_autograd"]["ok"],
        "aot_autograd_not_ok": counters["aot_autograd"]["not_ok"],
        "generated_kernel_count": metrics.generated_kernel_count,
        "graph_breaks": sum(counters["graph_break"].values()),
    }
    validate_compilation_execution(evidence)
    return evidence


def synthetic_batches(model, a, torch):
    batch_count = a.warmup + a.iterations
    b, length, frac = a.batch_size, a.max_length, padding_fraction(a)
    config = model[0].auto_model.config
    vocab = int(config.vocab_size)
    pad_id = int(config.pad_token_id or 0)
    nonpad_total = round(b * length * (1 - frac))
    remaining = max(b - 1, nonpad_total - length)
    other, remainder = divmod(remaining, max(1, b - 1))
    batches = []
    for batch_index in range(batch_count):
        left = torch.full((b, length), pad_id, dtype = torch.long)
        right = torch.full((b, length), pad_id, dtype = torch.long)
        mask = torch.zeros((b, length), dtype = torch.long)
        for row in range(b):
            n = length if row == 0 else min(length, other + (1 if row <= remainder else 0))
            start = (a.seed + batch_index * b + row) % max(1, vocab - 1)
            ids = (torch.arange(n, dtype = torch.long) + start) % max(1, vocab - 1) + 1
            if pad_id == 1:
                ids = (ids + 1) % max(2, vocab)
            left[row, :n] = ids
            right[row, :n] = ids.flip(0)
            mask[row, :n] = 1
        batches.append(
            (
                {"input_ids": left, "attention_mask": mask.clone()},
                {"input_ids": right, "attention_mask": mask.clone()},
            )
        )
    return batches, 0.0


def text_batches(model, a, pairs):
    model.max_seq_length = a.max_length
    started = wall_clock()
    batches = []
    row_batches, dropped = full_pair_batches(pairs, a.batch_size)
    for rows in row_batches:
        batches.append(
            (
                model.tokenize([row[0] for row in rows]),
                model.tokenize([row[1] for row in rows]),
            )
        )
    return batches, (wall_clock() - started) * 1000, dropped


def batch_meta(batches, torch):
    meta = []
    for left, right in batches:
        masks = [x.get("attention_mask", torch.ones_like(x["input_ids"])) for x in (left, right)]
        meta.append(
            {
                "rows": int(left["input_ids"].shape[0]),
                "nonpadding_tokens": sum(int(mask.sum()) for mask in masks),
                "padded_tokens": sum(int(x["input_ids"].numel()) for x in (left, right)),
            }
        )
    return meta


def build_fixture(model, a, pairs, torch):
    if a.pairs_json:
        batches, tokenization_ms, dropped = text_batches(model, a, pairs)
    else:
        batches, tokenization_ms = synthetic_batches(model, a, torch)
        dropped = 0
    return {
        "batches": batches,
        "meta": batch_meta(batches, torch),
        "fingerprint": fixture_fingerprint(batches),
        "tokenization_ms": tokenization_ms,
        "scheduled_pair_rows": sum(len(batch[0]["input_ids"]) for batch in batches),
        "dropped_pair_rows": dropped,
    }


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking = True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def preload(fixture, device, torch):
    gpu_batches, timings = [], []
    for left, right in fixture["batches"]:
        torch.cuda.synchronize()
        start, end = (
            torch.cuda.Event(enable_timing = True),
            torch.cuda.Event(enable_timing = True),
        )
        start.record()
        gpu_batches.append((move_batch(left, device), move_batch(right, device)))
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return gpu_batches, timings


def loss_class():
    # Called only after load_model has initialized/imported Unsloth.
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    return MultipleNegativesRankingLoss


def fresh_features(batch):
    # SentenceTransformer adds output tensors to its input dictionaries.
    return [dict(features) for features in batch]


def validate_dispatch_probe(a, probe):
    expected = (
        not a.compile
        and a.dtype != "fp32"
        and any(
            row["has_padding"]
            and row["all_rows_nonempty"]
            and (
                a.mode == "candidate-force"
                or (a.mode == "candidate-auto" and row["padded_tokens"] >= 8192)
            )
            for row in probe["inputs"]
        )
    )
    probe["expected_active"] = expected
    if bool(probe["num_calls"]) != expected:
        raise RuntimeError(f"unexpected varlen dispatch: expected={expected}, probe={probe}")


def probe_flash_varlen_dispatch(loss_fn, batch, a, torch):
    """Untimed forward-only proof that the shared varlen dispatcher was executed."""
    from unsloth.utils import attention_dispatch

    original = attention_dispatch.flash_attn_varlen_func
    inputs = []
    for features in batch:
        mask = features["attention_mask"]
        inputs.append(
            {
                "shape": list(mask.shape),
                "padded_tokens": mask.numel(),
                "nonpadding_tokens": int(mask.sum().item()),
                "has_padding": bool((mask == 0).any().item()),
                "all_rows_nonempty": bool((mask.sum(dim = 1) > 0).all().item()),
            }
        )
    if original is None:
        return {"available": False, "num_calls": 0, "query_rows": [], "inputs": inputs}
    query_rows = []

    def wrapped(*args, **kwargs):
        query = kwargs.get("q", args[0] if args else None)
        query_rows.append(int(query.shape[0]) if query is not None else None)
        return original(*args, **kwargs)

    attention_dispatch.flash_attn_varlen_func = wrapped
    try:
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype = torch_dtype(a.dtype, torch), enabled = a.dtype != "fp32"),
        ):
            probe_loss = loss_fn(fresh_features(batch), labels = None)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(probe_loss.detach().float().cpu())):
            raise RuntimeError("non-finite loss in post-measurement dispatch probe")
    finally:
        attention_dispatch.flash_attn_varlen_func = original
    return {
        "available": True,
        "num_calls": len(query_rows),
        "query_rows": query_rows,
        "inputs": inputs,
    }


def run_repeat(a, pairs, torch, device):
    gc.collect()
    torch.cuda.empty_cache()
    if a.compile:
        reset_compilation_counters(torch)
    seed_all(a.seed, torch)
    model, fast_st = load_model(a, torch)
    fixture = build_fixture(model, a, pairs, torch)
    gpu_batches, h2d_ms = preload(fixture, device, torch)
    model.to(device)
    model.train()
    initial = state_fingerprint(model)
    model = apply_explicit_compile(model, fast_st, a.compile)
    compiled_evidence = compilation_evidence(model, a.compile)
    Loss = loss_class()
    loss_fn = Loss(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr = a.learning_rate
    )
    activation = activation_info(model)
    amp = a.dtype != "fp32"

    def train_step(batch, measured = False):
        left, right = batch
        optimizer.zero_grad(set_to_none = True)
        if not measured:
            with torch.autocast("cuda", dtype = torch_dtype(a.dtype, torch), enabled = amp):
                loss = loss_fn(fresh_features(batch), labels = None)
            loss.backward()
            optimizer.step()
            return loss
        fwd_start, fwd_end = (
            torch.cuda.Event(enable_timing = True),
            torch.cuda.Event(enable_timing = True),
        )
        back_end, opt_end = (
            torch.cuda.Event(enable_timing = True),
            torch.cuda.Event(enable_timing = True),
        )
        torch.cuda.synchronize()
        wall_start = wall_clock()
        fwd_start.record()
        with torch.autocast("cuda", dtype = torch_dtype(a.dtype, torch), enabled = amp):
            loss = loss_fn(fresh_features(batch), labels = None)
        fwd_end.record()
        loss.backward()
        back_end.record()
        optimizer.step()
        opt_end.record()
        torch.cuda.synchronize()
        wall_ms = (wall_clock() - wall_start) * 1000
        cuda_ms = fwd_start.elapsed_time(opt_end)
        validate_step_clocks(wall_ms, cuda_ms)
        return (
            loss,
            wall_ms,
            fwd_start.elapsed_time(fwd_end),
            fwd_end.elapsed_time(back_end),
            back_end.elapsed_time(opt_end),
            cuda_ms,
        )

    warmup_wall = []
    for i in range(a.warmup):
        warmup_start = wall_clock()
        warmup_loss = train_step(gpu_batches[i % len(gpu_batches)])
        torch.cuda.synchronize()
        warmup_wall.append((wall_clock() - warmup_start) * 1000)
        warmup_cpu = warmup_loss.detach().float().cpu()
        if not bool(torch.isfinite(warmup_cpu)):
            raise RuntimeError(f"non-finite warmup loss at step {i}: {warmup_cpu.item()}")
    torch.cuda.synchronize()
    if a.compile:
        compiled_evidence["after_warmup"] = compilation_execution_evidence()
    torch.cuda.reset_peak_memory_stats(device)
    wall, forward, backward, optimizer_ms, total, losses, samples, tokens = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(a.iterations):
        loss, wall_ms, fwd_ms, back_ms, opt_ms, total_ms = train_step(
            gpu_batches[(a.warmup + i) % len(gpu_batches)], True
        )
        meta = fixture["meta"][(a.warmup + i) % len(fixture["meta"])]
        loss_cpu = loss.detach().float().cpu()
        if not bool(torch.isfinite(loss_cpu)):
            raise RuntimeError(f"non-finite measured loss at step {i}: {loss_cpu.item()}")
        wall.append(wall_ms)
        forward.append(fwd_ms)
        backward.append(back_ms)
        optimizer_ms.append(opt_ms)
        total.append(total_ms)
        losses.append(float(loss_cpu))
        samples.append(2 * meta["rows"] / (wall_ms / 1000))
        tokens.append(meta["nonpadding_tokens"] / (wall_ms / 1000))
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    if a.compile:
        compiled_evidence["after_measurement"] = compilation_execution_evidence()
    final = state_fingerprint(model)
    dispatch_probe = probe_flash_varlen_dispatch(loss_fn, gpu_batches[0], a, torch)
    validate_dispatch_probe(a, dispatch_probe)
    result = {
        "fixture_sha256": fixture["fingerprint"],
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "compiled": a.compile,
        "compilation_evidence": compiled_evidence,
        "activation": activation,
        "dispatch_probe": dispatch_probe,
        "tokenization_ms": fixture["tokenization_ms"],
        "h2d_ms": stats(h2d_ms),
        "warmup_wall_ms": stats(warmup_wall),
        "scheduled_pair_rows": fixture["scheduled_pair_rows"],
        "dropped_pair_rows": fixture["dropped_pair_rows"],
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "step_wall_ms": stats(wall),
        "forward_cuda_ms": stats(forward),
        "backward_cuda_ms": stats(backward),
        "optimizer_cuda_ms": stats(optimizer_ms),
        "step_cuda_ms": stats(total),
        "samples_per_sec": stats(samples),
        "nonpadding_tokens_per_sec": stats(tokens),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "padded_tokens": sum(x["padded_tokens"] for x in fixture["meta"]),
        "nonpadding_tokens": sum(x["nonpadding_tokens"] for x in fixture["meta"]),
        "raw": {
            "h2d_ms": h2d_ms,
            "warmup_wall_ms": warmup_wall,
            "step_wall_ms": wall,
            "forward_cuda_ms": forward,
            "backward_cuda_ms": backward,
            "optimizer_cuda_ms": optimizer_ms,
            "step_cuda_ms": total,
            "loss": losses,
            "samples_per_sec": samples,
            "nonpadding_tokens_per_sec": tokens,
        },
    }
    return result


def profile_case(a, pairs, torch, device, out_dir):
    gc.collect()
    torch.cuda.empty_cache()
    if a.compile:
        reset_compilation_counters(torch)
    seed_all(a.seed, torch)
    model, fast_st = load_model(a, torch)
    fixture = build_fixture(model, a, pairs, torch)
    gpu_batches, _ = preload(fixture, device, torch)
    model.to(device)
    model.train()
    initial = state_fingerprint(model)
    model = apply_explicit_compile(model, fast_st, a.compile)
    compiled_evidence = compilation_evidence(model, a.compile)
    activation = activation_info(model)
    Loss = loss_class()
    loss_fn = Loss(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr = a.learning_rate
    )
    amp = a.dtype != "fp32"

    def step(batch):
        left, right = batch
        optimizer.zero_grad(set_to_none = True)
        with torch.autocast("cuda", dtype = torch_dtype(a.dtype, torch), enabled = amp):
            with torch.profiler.record_function("forward"):
                loss = loss_fn(fresh_features(batch), labels = None)
        with torch.profiler.record_function("backward"):
            loss.backward()
        with torch.profiler.record_function("optimizer"):
            optimizer.step()

    for i in range(a.warmup):
        step(gpu_batches[i % len(gpu_batches)])
    torch.cuda.synchronize()
    if a.compile:
        compiled_evidence["after_warmup"] = compilation_execution_evidence()
    out_dir.mkdir(parents = True, exist_ok = True)
    compile_label = "compiled" if a.compile else "eager"
    trace = (
        out_dir
        / f"{a.mode}-{compile_label}-f{padding_fraction(a)}-L{a.max_length}-B{a.batch_size}.json"
    )
    with torch.profiler.profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes = True,
        profile_memory = True,
    ) as prof:
        for i in range(min(3, len(gpu_batches))):
            step(gpu_batches[i])
    if a.compile:
        compiled_evidence["after_profile"] = compilation_execution_evidence()
    prof.export_chrome_trace(str(trace))
    table = trace.with_suffix(".txt")
    table.write_text(
        prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 40),
        encoding = "utf-8",
    )
    dispatch_probe = probe_flash_varlen_dispatch(loss_fn, gpu_batches[0], a, torch)
    validate_dispatch_probe(a, dispatch_probe)
    return {
        "trace": str(trace),
        "operator_table": str(table),
        "fixture_sha256": fixture["fingerprint"],
        "initial_state_sha256": initial,
        "compiled": a.compile,
        "compilation_evidence": compiled_evidence,
        "activation": activation,
        "dispatch_probe": dispatch_probe,
    }


def package_versions():
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in (
        "unsloth",
        "torch",
        "transformers",
        "sentence-transformers",
        "peft",
        "triton",
    ):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def self_validate_helpers():
    from types import SimpleNamespace

    raw_clock, raw_name = select_wall_clock(
        SimpleNamespace(CLOCK_MONOTONIC_RAW = 42, clock_gettime = lambda clock: clock + 1)
    )
    assert raw_name == "CLOCK_MONOTONIC_RAW" and raw_clock() == 43
    fallback_clock, fallback_name = select_wall_clock(SimpleNamespace(perf_counter = lambda: 7))
    assert fallback_name == "perf_counter" and fallback_clock() == 7
    validate_step_clocks(10, 9.9)
    validate_step_clocks(10, 10.1)
    for wall_ms, cuda_ms in ((10, 11), (0, 1), (1, -1), (math.nan, 1), (1, math.inf)):
        try:
            validate_step_clocks(wall_ms, cuda_ms)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid step clocks accepted: {wall_ms}, {cuda_ms}")

    probe_config = SimpleNamespace(
        compile = False,
        pairs_json = None,
        dtype = "bf16",
        mode = "candidate-force",
        padding_fraction = 0.5,
        padding = None,
    )

    def probe(
        calls,
        tokens = 8192,
        padded = True,
    ):
        return {
            "num_calls": calls,
            "inputs": [
                {
                    "padded_tokens": tokens,
                    "has_padding": padded,
                    "all_rows_nonempty": True,
                }
            ],
        }

    validate_dispatch_probe(probe_config, probe(12))
    try:
        validate_dispatch_probe(probe_config, probe(0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("inactive optimized arm passed dispatch validation")
    probe_config.padding_fraction = 0
    validate_dispatch_probe(probe_config, probe(0, padded = False))
    probe_config.mode, probe_config.padding_fraction = "baseline", 0.5
    validate_dispatch_probe(probe_config, probe(0))
    probe_config.mode = "candidate-auto"
    validate_dispatch_probe(probe_config, probe(0, tokens = 7936))
    validate_dispatch_probe(probe_config, probe(12, tokens = 8192))
    probe_config.compile = True
    validate_dispatch_probe(probe_config, probe(0))
    token_tensor, mask_tensor = object(), object()
    fixture = ({"input_ids": token_tensor, "attention_mask": mask_tensor},) * 2
    fresh = fresh_features(fixture)
    fresh[0]["token_embeddings"] = object()
    fresh[1]["sentence_embedding"] = object()
    assert all(set(features) == {"input_ids", "attention_mask"} for features in fixture)
    assert all(features["input_ids"] is token_tensor for features in fresh)
    batches, dropped = full_pair_batches(
        [["a", "b"], ["a", "c"], ["d", "e"], ["f", "g"], ["h", "i"]], 2
    )
    assert len(batches) == 2 and dropped == 1
    assert all(len({text for row in batch for text in row}) == 2 * len(batch) for batch in batches)

    class FakeCompiler:
        calls = []

        @staticmethod
        def _apply_torch_compile(model, mode = "default"):
            FakeCompiler.calls.append((model, mode))
            return ("compiled", model)

    marker = object()
    assert apply_explicit_compile(marker, FakeCompiler, False) is marker and not FakeCompiler.calls
    assert apply_explicit_compile(marker, FakeCompiler, True) == ("compiled", marker)
    assert FakeCompiler.calls == [(marker, "default")]
    successful_compile = dict(unique_graphs = 1, aot_autograd_ok = 1, generated_kernel_count = 1)
    validate_compilation_execution(successful_compile)
    for field in successful_compile:
        try:
            validate_compilation_execution({**successful_compile, field: 0, "calls_captured": 927})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"compiled lane accepted missing {field}")
    assert quantile([1, 2, 3], 0.5) == 2 and stats([1, 2, 3])["median"] == 2
    print(
        "SELF_TEST_OK clocks dispatch_guard fresh_features full_batches compile_execution_guard stats"
    )


def main():
    a = parse_args()
    if a.self_test:
        self_validate_helpers()
        return
    if any(value is None for value in (a.mode, a.repo, a.model_path, a.output)):
        raise SystemExit(
            "--mode, --repo, --model-path, and --output are required unless --self-test is used"
        )
    if a.warmup < 10 or (not a.profile_only and (a.iterations < 30 or a.repeats < 3)):
        raise SystemExit(
            "require warmup>=10; benchmark mode also requires iterations>=30 and repeats>=3"
        )
    if a.max_length < 2 or a.batch_size < 1:
        raise SystemExit("max length and batch size must be positive")
    if not a.repo.is_dir() or not a.model_path.is_dir():
        raise SystemExit("--repo and --model-path must already be directories; no downloads")
    checkpoint = model_checkpoint_info(a.model_path)
    pairs = make_pairs(a)
    pair_info = pair_fixture_info(a.pairs_json, pairs)
    if (
        pair_info.get("model_revision")
        and checkpoint["revision"]
        and pair_info["model_revision"] != checkpoint["revision"]
    ):
        raise SystemExit("model checkpoint revision does not match pair-fixture provenance")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    base = {
        "schema": "sentence-transformer-benchmark-v4",
        "wall_clock": {
            "name": WALL_CLOCK_NAME,
            "cuda_sanity_relative_tolerance": 0.01,
            "cuda_sanity_absolute_tolerance_ms": 0.05,
        },
        "mode": a.mode,
        "execution": "compiled" if a.compile else "eager",
        "config": vars(a)
        | {
            "repo": str(a.repo.resolve()),
            "model_path": str(a.model_path.resolve()),
            "pairs_json": str(a.pairs_json.resolve()) if a.pairs_json else None,
            "output": str(a.output.resolve()),
            "padding_fraction": padding_fraction(a),
        },
        "source": git_info(a.repo),
        "checkpoint": checkpoint,
        "pair_fixture": pair_info,
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "versions": package_versions(),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "total_memory_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
        },
        "seed": a.seed,
        "pair_rows": len(pairs),
        "sequence_rows_per_step": 2 * a.batch_size,
        "cache_env": {
            key: os.environ.get(key)
            for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME")
        },
    }
    if a.profile_only:
        base["profile"] = profile_case(
            a, pairs, torch, device, a.profile_dir or (a.output.parent / "profiles")
        )
        base["benchmark"] = None
    else:
        repeats = [run_repeat(a, pairs, torch, device) for _ in range(a.repeats)]
        if (
            len({row["fixture_sha256"] for row in repeats}) != 1
            or len({row["initial_state_sha256"] for row in repeats}) != 1
        ):
            raise RuntimeError(
                "repeat fixture or initial-state fingerprints differ; refusing unmatched metrics"
            )
        base["repeats"] = repeats
        base["profile"] = None
    a.output.parent.mkdir(parents = True, exist_ok = True)
    a.output.write_text(json.dumps(base, indent = 2, default = str), encoding = "utf-8")
    print(a.output)


if __name__ == "__main__":
    main()
