#!/usr/bin/env python3
"""Benchmark vanilla vs. Unsloth SentenceTransformer training.

Run each mode in a fresh process so Unsloth's process-wide SentenceTransformer
patches cannot leak into the vanilla baseline::

    python scripts/benchmark_sentence_transformer_training.py --mode baseline
    python scripts/benchmark_sentence_transformer_training.py --mode optimized

The workload is a deterministic MultipleNegativesRankingLoss step over a
right-padded, variable-length batch.  Timings exclude model loading and warmup
but include forward, backward, optimizer step, and gradient clearing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORDS = (
    "amber bridge cedar delta ember forest garden harbor island jasmine "
    "kernel lantern meadow nickel orbit pepper quartz river silver timber "
    "umber velvet willow xenon yellow zephyr"
).split()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the current SentenceTransformer default fused AdamW implementation.",
    )
    parser.add_argument("--hard-negative", action="store_true")
    parser.add_argument(
        "--asymmetric-lengths",
        action="store_true",
        help="Use short queries with longer positive/negative documents.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow model repositories to execute custom tokenizer/model code.",
    )
    parser.add_argument(
        "--model-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply shipped query/document prompts to synthetic columns.",
    )
    args = parser.parse_args()
    for name in ("batch_size", "max_length", "warmup_steps", "steps"):
        minimum = 0 if name == "warmup_steps" else 1
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_texts(
    batch_size: int,
    max_length: int,
    asymmetric_lengths: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """Create paired sentences whose token lengths span most of the batch pad."""
    queries: list[str] = []
    positives: list[str] = []
    negatives: list[str] = []
    low = max(4, max_length // 8)
    high = max(low + 1, max_length - 4)
    span = high - low + 1
    for row in range(batch_size):
        length = low + ((row * 37) % span)
        tokens = [WORDS[(row * 11 + col) % len(WORDS)] for col in range(length)]
        query_tokens = tokens
        if asymmetric_lengths:
            query_length = max(4, min(len(tokens), max_length // 12 + row % 5))
            query_tokens = tokens[:query_length]
        queries.append(" ".join(query_tokens))
        # Preserve almost all content while perturbing order and one token.  The
        # pair is useful for a retrieval sanity check without requiring a dataset.
        shifted = tokens[1:] + tokens[:1]
        shifted[-1] = WORDS[(row + 7) % len(WORDS)]
        positives.append(" ".join(shifted))
        negative = [WORDS[(row * 17 + col * 5 + 13) % len(WORDS)] for col in range(length)]
        negatives.append(" ".join(negative))
    return queries, positives, negatives


def move_features(features: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in features.items()}


def reference_tokenize(
    model_name: str,
    texts: list[str],
    max_length: int,
    offline: bool,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=offline,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return dict(
        tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    )


def model_prompt_contract(
    model_name: str,
    revision: str | None,
    offline: bool,
    enabled: bool,
) -> dict[str, object]:
    config: dict[str, object] = {}
    if enabled:
        local_path = Path(model_name) / "config_sentence_transformers.json"
        try:
            if local_path.exists():
                config = json.loads(local_path.read_text(encoding="utf-8"))
            else:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=model_name,
                    filename="config_sentence_transformers.json",
                    revision=revision,
                    local_files_only=offline,
                )
                config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except Exception:
            config = {}
    prompts = config.get("prompts", {}) if isinstance(config, dict) else {}
    prompts = prompts if isinstance(prompts, dict) else {}
    query = str(prompts.get("query", "") or "")
    document = ""
    for name in ("document", "passage", "corpus"):
        if name in prompts:
            document = str(prompts[name] or "")
            break
    return {
        "enabled": enabled,
        "config_sha256": (
            hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if config
            else None
        ),
        "query": query,
        "document": document,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "query_characters": len(query),
        "document_characters": len(document),
    }
def feature_hash(*feature_sets: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for features in feature_sets:
        for key in sorted(features):
            value = features[key]
            if not torch.is_tensor(value):
                continue
            cpu_value = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(cpu_value.dtype).encode("ascii"))
            digest.update(str(tuple(cpu_value.shape)).encode("ascii"))
            digest.update(cpu_value.numpy().tobytes())
    return digest.hexdigest()


def parameter_hashes(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> tuple[str, str]:
    named_digest = hashlib.sha256()
    values_digest = hashlib.sha256()
    for name, parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        metadata = (
            str(value.dtype).encode("ascii")
            + b"\0"
            + str(tuple(value.shape)).encode("ascii")
            + b"\0"
        )
        raw_value = value.view(torch.uint8).numpy().tobytes()
        named_digest.update(name.encode("utf-8"))
        named_digest.update(b"\0")
        named_digest.update(metadata)
        named_digest.update(raw_value)
        values_digest.update(metadata)
        values_digest.update(raw_value)
    return named_digest.hexdigest(), values_digest.hexdigest()


def parameter_hash(parameters: list[tuple[str, torch.nn.Parameter]]) -> str:
    return parameter_hashes(parameters)[0]


def resolved_model_commit(model: object) -> str | None:
    """Return the Hugging Face snapshot commit from the wrapped base config."""
    try:
        pending = [model[0].auto_model]  # type: ignore[index]
    except (AttributeError, IndexError, TypeError):
        return None

    visited: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        config = getattr(current, "config", None)
        for candidate in (current, config):
            commit = getattr(candidate, "_commit_hash", None)
            if isinstance(commit, str) and commit:
                return commit
        for attribute in ("base_model", "model"):
            child = getattr(current, attribute, None)
            if child is not None and child is not current:
                pending.append(child)
    return None


def set_dropout(model, probability: float | None) -> None:
    if probability is None:
        return
    if not 0.0 <= probability < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = probability
        config = getattr(module, "config", None)
        if config is not None:
            for name in ("attention_probs_dropout_prob", "attention_dropout", "attn_pdrop", "hidden_dropout_prob"):
                if hasattr(config, name):
                    setattr(config, name, probability)


def lora_targets(model) -> list[str]:
    model_type = str(getattr(model[0].auto_model.config, "model_type", "")).lower()
    if model_type == "distilbert":
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    if model_type == "mpnet":
        return ["q", "k", "v", "o", "dense"]
    if model_type == "modernbert":
        return ["Wqkv", "Wo", "Wi"]
    if model_type in {"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3_text"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["query", "key", "value", "dense"]


def load_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    if args.mode == "baseline":
        from peft import LoraConfig, get_peft_model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.model,
            device=str(device),
            local_files_only=args.offline,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            model_kwargs={"dtype": dtype},
        )
        targets = lora_targets(model)
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        seed_everything(args.seed)
        peft_model = get_peft_model(model[0].auto_model, config)
        if isinstance(getattr(type(model[0]), "auto_model", None), property):
            model[0].model = peft_model
        else:
            model[0].auto_model = peft_model
        return model, False

    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(
        args.model,
        max_seq_length=args.max_length,
        dtype=dtype,
        load_in_16bit=True,
        device_map=str(device),
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.offline,
    )
    targets = lora_targets(model)
    seed_everything(args.seed)
    model = FastSentenceTransformer.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=targets,
        use_gradient_checkpointing=False,
        random_state=args.seed,
    )
    compiled = bool(args.compile and getattr(model, "_compile_pending", False))
    if compiled:
        FastSentenceTransformer._apply_torch_compile(model, mode=args.compile_mode)
        model._compile_pending = False
        model._benchmark_fused_lora_layers = 0
    else:
        from unsloth.models.sentence_transformer import _patch_encoder_attention_lora

        model._compile_pending = False
        inner_model = model[0].auto_model
        _patch_encoder_attention_lora(inner_model)
        model._benchmark_fused_lora_layers = getattr(
            inner_model,
            "_unsloth_fused_lora_qkv_count",
            0,
        )
    return model, compiled


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device("cuda")
    seed_everything(args.seed)
    model, compiled = load_model(args, dtype, device)
    model.max_seq_length = args.max_length
    if hasattr(model, "__getitem__") and hasattr(model[0], "max_seq_length"):
        model[0].max_seq_length = args.max_length
    set_dropout(model, args.dropout)
    model.train()
    if args.mode == "optimized":
        import unsloth

        implementation_file = str(Path(unsloth.__file__).resolve())
    else:
        implementation_file = str(Path(sys.modules[type(model).__module__].__file__).resolve())

    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )

    queries, positives, negatives = make_texts(
        args.batch_size,
        args.max_length,
        args.asymmetric_lengths,
    )
    prompt_contract = model_prompt_contract(
        args.model,
        args.revision,
        args.offline,
        args.model_prompts,
    )
    queries = [str(prompt_contract["query"]) + text for text in queries]
    positives = [str(prompt_contract["document"]) + text for text in positives]
    negatives = [str(prompt_contract["document"]) + text for text in negatives]
    text_columns = [queries, positives]
    if args.hard_negative:
        text_columns.append(negatives)
    reference_columns = [
        reference_tokenize(
            args.model,
            texts,
            args.max_length,
            args.offline,
            args.revision,
            args.trust_remote_code,
        )
        for texts in text_columns
    ]
    inputs_hash = feature_hash(*reference_columns)
    feature_columns = [move_features(features, device) for features in reference_columns]
    features_a, features_b = feature_columns[:2]
    loss_module = MultipleNegativesRankingLoss(model=model)
    combined_feature_eligible = None
    optimized_batching_enabled = None
    encoder_buckets_per_loss = None
    if args.mode == "optimized":
        from unsloth.kernels.contrastive_loss import _bucketed_sentence_features

        bucket_plan = _bucketed_sentence_features(model, feature_columns)
        optimized_batching_enabled = bucket_plan is not None
        if bucket_plan is not None:
            encoder_buckets_per_loss = len(bucket_plan[0])
            combined_feature_eligible = encoder_buckets_per_loss == 1
        del bucket_plan
    trainable_named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    frozen_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in trainable_named]
    initial_trainable_hash, initial_trainable_values_hash = parameter_hashes(trainable_named)
    initial_frozen_hash, initial_frozen_values_hash = parameter_hashes(frozen_named)
    fused_optimizer = args.fused_optimizer
    optimizer = torch.optim.AdamW(trainable, lr=2e-5, fused=fused_optimizer)

    encoder_calls = None
    if not compiled:
        encoder_calls = 0

        def count_encoder_calls(_module, _args):
            nonlocal encoder_calls
            encoder_calls += 1

        call_probe = model.register_forward_pre_hook(count_encoder_calls)
        probe_loss = loss_module(feature_columns, labels=None)
        synchronize(device)
        del probe_loss
        call_probe.remove()
        optimizer.zero_grad(set_to_none=True)
        seed_everything(args.seed)

    padding_ratio = 1.0 - sum(
        features["attention_mask"].sum().item() for features in feature_columns
    ) / sum(features["attention_mask"].numel() for features in feature_columns)

    def evaluate_pairs() -> dict[str, float]:
        model.eval()
        with torch.inference_mode():
            eval_a = move_features(
                {key: value.clone() for key, value in reference_columns[0].items()}, device
            )
            eval_b = move_features(
                {key: value.clone() for key, value in reference_columns[1].items()}, device
            )
            embeddings_a = torch.nn.functional.normalize(
                model(eval_a)["sentence_embedding"].float(), dim=-1
            )
            embeddings_b = torch.nn.functional.normalize(
                model(eval_b)["sentence_embedding"].float(), dim=-1
            )
            similarities = embeddings_a @ embeddings_b.t()
            order = similarities.argsort(dim=1, descending=True)
            targets = torch.arange(args.batch_size, device=device).unsqueeze(1)
            ranks = (order == targets).nonzero(as_tuple=False)[:, 1] + 1
            metrics = {
                "recall_at_1": float((ranks == 1).float().mean()),
                "mrr": float((1.0 / ranks.float()).mean()),
                "positive_cosine": float(similarities.diag().mean()),
            }
        model.train()
        return metrics

    initial_pair_metrics = evaluate_pairs()

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_module(feature_columns, labels=None)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    synchronize(device)
    warmup_start = time.perf_counter()
    warmup_losses = [train_step() for _ in range(args.warmup_steps)]
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start

    torch.cuda.reset_peak_memory_stats(device)
    step_times: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()
        losses.append(train_step())
        synchronize(device)
        step_times.append(time.perf_counter() - start)

    model.eval()
    with torch.inference_mode():
        embedding = model(features_a)["sentence_embedding"].float()
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        fingerprint = embedding[:4, :16].cpu()

    median_step = statistics.median(step_times)
    result = {
        "mode": args.mode,
        "model": args.model,
        "revision": args.revision,
        "resolved_model_commit": resolved_model_commit(model),
        "implementation_file": implementation_file,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "dropout": args.dropout,
        "fused_optimizer": fused_optimizer,
        "batch_size": args.batch_size,
        "text_columns": len(feature_columns),
        "asymmetric_lengths": args.asymmetric_lengths,
        "prompt_contract": prompt_contract,
        "max_length": args.max_length,
        "padding_ratio": padding_ratio,
        "inputs_sha256": inputs_hash,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_trainable_sha256": initial_trainable_hash,
        "initial_trainable_values_sha256": initial_trainable_values_hash,
        "frozen_parameters": sum(parameter.numel() for _, parameter in frozen_named),
        "initial_frozen_sha256": initial_frozen_hash,
        "initial_frozen_values_sha256": initial_frozen_values_hash,
        "compiled": compiled,
        "fused_lora_layers": getattr(model, "_benchmark_fused_lora_layers", 0),
        "fused_lora_linears": getattr(
            model[0].auto_model,
            "_unsloth_fast_lora_linear_count",
            0,
        ),
        "compile_mode": args.compile_mode if compiled else None,
        "encoder_calls_per_loss": encoder_calls,
        "combined_feature_eligible": combined_feature_eligible,
        "optimized_batching_enabled": optimized_batching_enabled,
        "encoder_buckets_per_loss": encoder_buckets_per_loss,
        "feature_shapes": [
            {key: list(value.shape) for key, value in features.items() if torch.is_tensor(value)}
            for features in feature_columns
        ],
        "warmup_steps": args.warmup_steps,
        "warmup_seconds": warmup_seconds,
        "timed_steps": args.steps,
        "median_step_seconds": median_step,
        "mean_step_seconds": statistics.mean(step_times),
        "p90_step_seconds": sorted(step_times)[max(0, int(len(step_times) * 0.9) - 1)],
        "pairs_per_second": args.batch_size / median_step,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "first_warmup_loss": warmup_losses[0],
        "warmup_losses": warmup_losses,
        "timed_losses": losses,
        "last_loss": losses[-1],
        "initial_pair_metrics": initial_pair_metrics,
        "final_pair_metrics": evaluate_pairs(),
        "final_trainable_sha256": parameter_hash(trainable_named),
        "embedding_fingerprint": fingerprint.tolist(),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
