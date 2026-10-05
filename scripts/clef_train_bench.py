# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Clef training step benchmark and head parity check, one JSON line on stdout.

python scripts/clef_train_bench.py --model Cloudflare/clef-flash --load-in-4bit --seq-len 8192
python scripts/clef_train_bench.py --model tiny --fast off --dtype fp16

--fast on|off sets UNSLOTH_CLEF_FAST (Unsloth's batched, chunked, checkpointed and compiled head
vs the per-record head). Whatever the arm, the first batch's head is also run both ways on the same
hidden states, so `parity` compares the two heads' logits and gradients in this process.
"""

import argparse
import json
import os
import sys
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default = "tiny", help = "tiny, a Clef repo id or folder")
parser.add_argument("--load-in-4bit", action = "store_true")
parser.add_argument("--fast", choices = ["on", "off"], default = "on")
parser.add_argument("--dtype", choices = ["bf16", "fp16"], default = "bf16")
parser.add_argument("--seq-len", type = int, default = 4096)
parser.add_argument("--batch-size", type = int, default = 1)
parser.add_argument("--questions", type = int, default = 8)
parser.add_argument("--steps", type = int, default = 6)
parser.add_argument("--warmup", type = int, default = 2)
parser.add_argument("--log-recompiles", action = "store_true")
args = parser.parse_args()
os.environ["UNSLOTH_CLEF_FAST"] = "1" if args.fast == "on" else "0"

import torch  # noqa: E402

from unsloth import FastDecisionModel  # noqa: E402
from unsloth.models import clef, decision  # noqa: E402

if args.log_recompiles:
    torch._logging.set_logs(recompiles = True, graph_breaks = True)

TINY = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
TINY_HEAD = dict(hidden_size = 8, width = 32, routing_layers = 2, layers = 2, heads = 4, feedforward = 64)


def tiny_checkpoint() -> str:
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    folder = os.path.join(tempfile.mkdtemp(prefix = "clef_tiny_"), "clef")
    torch.manual_seed(0)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(TINY, dtype = torch.bfloat16)
    backbone.save_pretrained(folder)
    AutoProcessor.from_pretrained(TINY).save_pretrained(folder)
    head = clef.JointSchemaHead(
        **{**TINY_HEAD, "hidden_size": backbone.config.text_config.hidden_size}
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    save_file(
        {k: v.to(torch.bfloat16) for k, v in head.state_dict().items()},
        f"{folder}/joint_head.safetensors",
    )
    with open(f"{folder}/joint_head_config.json", "w") as file:
        json.dump(head.config, file)
    return folder


def rows(tokenizer, count):
    # A long state filled to about --seq-len tokens, and --questions questions of 2 to 6 options.
    filler = "The customer reports intermittent checkout failures across several regions. "
    per = len(tokenizer(filler, add_special_tokens = False).input_ids)
    state = filler * max(1, (args.seq_len - 120 * args.questions - 200) // per)
    out = []
    for index in range(count):
        questions, gold = {}, {}
        for q in range(args.questions):
            options = [f"option {o} for field {q}" for o in range(2 + (q + index) % 5)]
            kind = ("choice", "score", "noul")[q % 3]
            if kind == "noul":
                questions[f"f{q}"] = {"type": "noul", "instructions": f"Is statement {q} true?"}
                gold[f"f{q}"] = {"label": "true" if (q + index) % 2 else "false"}
            elif kind == "score":
                questions[f"f{q}"] = {
                    "type": "score",
                    "instructions": f"Rate {q}",
                    "criteria": options,
                }
                gold[f"f{q}"] = {"label": (q + index) % len(options)}
            else:
                questions[f"f{q}"] = {
                    "type": "choice",
                    "instructions": f"Pick {q}",
                    "criteria": {f"o{o}": text for o, text in enumerate(options)},
                }
                gold[f"f{q}"] = {"label": f"o{index % len(options)}"}
        out.append({"state": f"{index}: {state}", "questions": questions, "gold": gold})
    return out


def head_parity(model, batch, amp):
    # Same hidden states into the batched and the per-record head. Each is scored against the
    # per-record head with autocast off (fp32 head, the oracle): under bf16 autocast both heads
    # round their matmuls, so the batched head passes when its error is at most 1.5x the
    # per-record head's plus one bf16 rounding step (kernel_verify_workflow step 2).
    backbone = model._backbone()
    text_model = getattr(backbone.model, "language_model", backbone.model)
    with torch.no_grad():
        hidden = text_model(
            input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], use_cache = False
        ).last_hidden_state
    embedding = backbone.get_output_embeddings().weight.detach()
    head = model.head
    os.environ["UNSLOTH_CLEF_FAST"] = "1"

    def run(forward, autocast):
        head.zero_grad()
        h = hidden.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype = amp or torch.float32, enabled = autocast):
            out = forward(
                h, batch["input_ids"], batch["attention_mask"], batch["records"], embedding
            )
        flat = [q for record in out for q in record]
        padded = torch.full((len(flat), max(len(z) for z in flat)), -1e4, device = h.device)
        for i, z in enumerate(flat):
            padded[i, : len(z)] = z.float()
        generator = torch.Generator(device = h.device).manual_seed(0)
        grad = torch.randn(padded.shape, device = h.device, generator = generator)
        (padded * grad * (padded > -1e3)).sum().backward()
        grads = {
            n: p.grad.detach().float().clone()
            for n, p in head.named_parameters()
            if p.grad is not None
        }
        return padded.detach(), h.grad.detach().float(), grads

    oracle = run(head.forward_per_record, False)
    ours = run(head.forward, amp is not None)
    reference = run(head.forward_per_record, amp is not None)
    os.environ["UNSLOTH_CLEF_FAST"] = "1" if args.fast == "on" else "0"
    head.zero_grad()
    valid = oracle[0] > -1e3

    def errors(got):
        return {
            "logits_max_abs": (got[0] - oracle[0])[valid].abs().max().item(),
            "hidden_grad_rel_l2": ((got[1] - oracle[1]).norm() / oracle[1].norm()).item(),
            "param_grad_rel_l2": max(
                ((got[2][n] - oracle[2][n]).norm() / oracle[2][n].norm().clamp_min(1e-30)).item()
                for n in oracle[2]
            ),
        }

    ours_error, reference_error = errors(ours), errors(reference)
    step = 2**-8 if amp is not None else 2**-20
    scale = oracle[0][valid].abs().max().item()
    allowed = {
        "logits_max_abs": 1.5 * reference_error["logits_max_abs"] + step * scale,
        "hidden_grad_rel_l2": 1.5 * reference_error["hidden_grad_rel_l2"] + step,
        "param_grad_rel_l2": 1.5 * reference_error["param_grad_rel_l2"] + step,
    }
    return {
        "batched_vs_oracle": ours_error,
        "per_record_vs_oracle": reference_error,
        "argmax_equal": bool(torch.equal(ours[0].argmax(-1), reference[0].argmax(-1))),
        "verdict": "PASS" if all(ours_error[k] <= allowed[k] for k in allowed) else "FAIL",
    }


def head_cost(model, batch, amp):
    # Head forward + backward alone on detached hidden states: time and memory above the inputs.
    backbone = model._backbone()
    text_model = getattr(backbone.model, "language_model", backbone.model)
    with torch.no_grad():
        hidden = text_model(
            input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], use_cache = False
        ).last_hidden_state
    embedding = backbone.get_output_embeddings().weight.detach()
    times = []
    for i in range(args.warmup + 3):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        start = time.perf_counter()
        h = hidden.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype = amp or torch.float32, enabled = amp is not None):
            out = model.head(
                h, batch["input_ids"], batch["attention_mask"], batch["records"], embedding
            )
        logits = getattr(out, "flat_padded", None)
        if logits is None:
            logits = torch.cat([q.sum().reshape(1) for record in out for q in record])
        logits.float().sum().backward()
        torch.cuda.synchronize()
        if i >= args.warmup:
            times.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated() - base
    model.head.zero_grad()
    times.sort()
    return {"head_ms": 1000 * times[len(times) // 2], "head_peak_gb": peak / 2**30}


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def main():
    model_name = tiny_checkpoint() if args.model == "tiny" else args.model
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model, processor = FastDecisionModel.from_pretrained(
        model_name, max_seq_length = args.seq_len, load_in_4bit = args.load_in_4bit, dtype = dtype
    )
    model = FastDecisionModel.get_peft_model(model, r = 16, lora_alpha = 16)
    count = args.batch_size * (args.steps + args.warmup)
    items, report = FastDecisionModel.build_dataset(
        rows(processor.tokenizer, count), processor, model
    )
    if not items:
        sys.exit(f"no items: {report}")
    tokenizer = getattr(processor, "tokenizer", processor)
    collate = decision.ClefDataCollator(tokenizer.pad_token_id)
    device = next(model.parameters()).device
    batches = []
    for start in range(0, len(items), args.batch_size):
        batch = collate(items[start : start + args.batch_size])
        batches.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
    model.train()
    amp = torch.bfloat16 if dtype == torch.bfloat16 and torch.cuda.is_bf16_supported() else None
    parity = head_parity(model, batches[0], amp)
    cost = head_cost(model, batches[0], amp)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-5)
    times, losses, all_times, graphs_after_step = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(batches[: args.steps + args.warmup]):
        batch = dict(batch)
        target = batch.pop("target")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype = amp or torch.float32, enabled = amp is not None):
            logits, _ = model(**batch)
        loss = decision._soft_cross_entropy(logits, target, batch["marker_mask"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
        all_times.append(round(time.perf_counter() - start, 4))
        graphs_after_step.append(_graphs())
        if index >= args.warmup:
            times.append(all_times[-1])
        losses.append(round(loss.item(), 5))
    times.sort()
    step = times[len(times) // 2]
    from torch._dynamo.utils import compile_times, counters

    names, values = compile_times(repr = "csv", aggregate = True)
    compiled_frames = dict(zip(names, values))
    print(
        "JOB_RESULT "
        + json.dumps(
            {
                "model": args.model,
                "load_in_4bit": args.load_in_4bit,
                "fast": args.fast,
                "dtype": args.dtype,
                "seq_len": args.seq_len,
                "tokens_per_row": len(items[0]["input_ids"]),
                "batch_size": args.batch_size,
                "s_per_step": round(step, 4),
                "s_per_step_spread": [round(times[0], 4), round(times[-1], 4)],
                "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                "head_ms": round(cost["head_ms"], 2),
                "head_share": round(cost["head_ms"] / 1000 / step, 4),
                "head_peak_gb": round(cost["head_peak_gb"], 3),
                "losses": losses,
                "step_times": all_times,
                "graphs_after_step": graphs_after_step,
                "graph_breaks": sum(counters["graph_break"].values()),
                "graph_break_reasons": list(counters["graph_break"])[:10],
                "unique_graphs": counters["stats"]["unique_graphs"],
                "compile_frames": compiled_frames,
                "parity": parity,
                "levers": {
                    "batched_head": os.environ.get("UNSLOTH_CLEF_FAST") != "0",
                    "chunked_norm_pool": os.environ.get("UNSLOTH_CLEF_FAST") != "0",
                    "head_checkpoint": os.environ.get("UNSLOTH_CLEF_FAST") != "0"
                    and os.environ.get("UNSLOTH_CLEF_CHECKPOINT", "1") != "0",
                    "head_compile": os.environ.get("UNSLOTH_CLEF_FAST") != "0"
                    and clef._compile_supported(device)
                    and "function" in clef._COMPILED,
                },
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "unsloth_file": sys.modules["unsloth"].__file__,
            }
        ),
        flush = True,
    )


if __name__ == "__main__":
    main()
