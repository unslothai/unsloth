# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One gloo rank for test_layer_major_ga_ddp.py: offload_layers layer-major gradient accumulation under DDP (tiny
Llama + LoRA, 6 of 8 layers behind a stand-in block swap, GA 4, 2 steps, SGD). Each arm prints one line per rank:
the gradients the optimizer saw at every step and the final LoRA weights, compared bitwise with plain DDP and within
fp32 tolerance with one process running each step's global batch."""

import contextlib
import functools
import importlib.util
import io
import json
import os
import sys
import types

import torch
import torch.distributed as dist
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS, GA, PER_RANK, LR = 2, 4, 2, 0.5
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(zoo_dir):
    path = os.path.join(zoo_dir, "unsloth_zoo", "block_swap_layer_major.py")
    spec = importlib.util.spec_from_file_location("unsloth_zoo.block_swap_layer_major", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    sys.modules["unsloth_zoo.block_swap_layer_major"] = engine
    # unsloth/models as a bare package: its relative imports (._ddp_defaults) resolve without importing unsloth.
    pkg = types.ModuleType("_lmga_models")
    pkg.__path__ = [os.path.join(ROOT, "unsloth", "models")]
    sys.modules["_lmga_models"] = pkg
    spec = importlib.util.spec_from_file_location(
        "_lmga_models._layer_major_ga",
        os.path.join(ROOT, "unsloth", "models", "_layer_major_ga.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.run_layer_major is engine.run_layer_major
    mod.patch_layer_major_gradient_accumulation(transformers.Trainer)
    return mod, engine


def token_budget():
    spec = importlib.util.spec_from_file_location(
        "_lmga_token_budget", os.path.join(ROOT, "unsloth", "utils", "token_budget.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Swap:
    """Stand-in BlockSwap: 6 of 8 layers swapped."""

    def __init__(self):
        self.indices, self.total = [1, 2, 3, 4, 5, 6], 8

    def hold(self, idx):
        pass

    def unhold(self, idx):
        pass


def build_model():
    import peft

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 8,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
        max_position_embeddings = 256,
        tie_word_embeddings = False,
    )
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r = 4, lora_alpha = 8, lora_dropout = 0.0, init_lora_weights = False, target_modules = TARGETS
        ),
    )
    model.enable_input_require_grads()
    for layer in model.get_base_model().model.layers:
        # A constant buffer inside each decoder layer, as Unsloth's per-layer rotary inv_freq: DDP broadcasts it.
        layer.self_attn.register_buffer("unsloth_inv_freq", torch.arange(4.0), persistent = False)
    return model


def rows(n = STEPS * GA * PER_RANK * 2, seq = None):
    g = torch.Generator().manual_seed(1)
    return [
        {
            "input_ids": torch.randint(
                0,
                256,
                (seq or 6 + int(torch.randint(0, 14, (1,), generator = g)),),
                generator = g,
            )
        }
        for _ in range(n)
    ]


def unsloth_gc():
    """unsloth_zoo's gradient checkpointing module (the one the engine's shim checks look up), or None."""
    try:
        import unsloth_zoo.gradient_checkpointing as gc
    except Exception as e:
        print(f"unsloth_zoo.gradient_checkpointing: {type(e).__name__}: {e}", flush = True)
        return None
    return gc


def collate(batch):
    batch = [{"input_ids": torch.as_tensor(r["input_ids"])} for r in batch]
    n = max(len(r["input_ids"]) for r in batch)
    ids = torch.zeros(len(batch), n, dtype = torch.long)
    mask = torch.zeros(len(batch), n, dtype = torch.long)
    for k, r in enumerate(batch):
        ids[k, : len(r["input_ids"])] = r["input_ids"]
        mask[k, : len(r["input_ids"])] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.masked_fill(mask == 0, -100)}


class Grads(transformers.TrainerCallback):
    def __init__(self, model):
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.steps = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.steps.append({n: p.grad.detach().clone() for n, p in self.params})


def args_for(out_dir, sft = False):
    common = dict(
        output_dir = out_dir,
        per_device_train_batch_size = PER_RANK,
        gradient_accumulation_steps = GA,
        max_steps = STEPS,
        learning_rate = LR,
        lr_scheduler_type = "constant",
        optim = "sgd",
        max_grad_norm = 0,
        logging_steps = 1,
        ddp_find_unused_parameters = False,
        average_tokens_across_devices = True,
        report_to = [],
        save_strategy = "no",
        use_cpu = True,
        seed = 0,
        dataloader_num_workers = 0,
        disable_tqdm = True,
        remove_unused_columns = False,
    )
    if sft:
        import trl
        return trl.SFTConfig(**common, dataset_kwargs = {"skip_prepare_dataset": True})
    return transformers.TrainingArguments(**common)


def tokenizer():
    import tokenizers

    vocab = {f"t{i}": i for i in range(256)}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "t0"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object = tok, pad_token = "t0", eos_token = "t1", unk_token = "t0"
    )


class HookWrapper(transformers.TrainerCallback):
    """A forward pre-hook on the DDP wrapper itself (a no-op): a window would bypass it, so none may run."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        wrapped = self.trainer.model_wrapped
        assert isinstance(wrapped, torch.nn.parallel.DistributedDataParallel), type(wrapped)
        wrapped.register_forward_pre_hook(lambda module, args: None)


def train(
    mod,
    engine,
    out_dir,
    mode,
    inject = None,
    sft = False,
    record = None,
    hook_wrapper = False,
    rank1_wrapper = False,
    seq = None,
    gc_shim = None,
):
    os.environ["UNSLOTH_OFFLOAD_LAYER_MAJOR"] = mode
    model = build_model()
    model.get_base_model().model.layers._unsloth_block_swap = Swap()
    if gc_shim is not None:
        # use_gradient_checkpointing = "unsloth": the decoder layers carry Unsloth's checkpoint shim.
        for layer in model.get_base_model().model.layers:
            layer._gradient_checkpointing_func = gc_shim
    data = rows(seq = seq)
    seen = []

    def recording_collate(batch):
        out = collate(batch)
        seen.append(out)
        return out

    if sft:
        import datasets
        import trl

        ds = datasets.Dataset.from_dict({"input_ids": [r["input_ids"].tolist() for r in data]})
        trainer = trl.SFTTrainer(
            model = model,
            args = args_for(out_dir, sft = True),
            train_dataset = ds,
            data_collator = recording_collate,
            processing_class = tokenizer(),
        )
    else:
        trainer = transformers.Trainer(
            model = model, args = args_for(out_dir), train_dataset = data, data_collator = recording_collate
        )
    # As Unsloth's trainers do: under DDP UNSLOTH_TOKEN_BALANCE defaults on (an instance-level get_train_dataloader).
    token_budget().install_token_budget(trainer)
    assert "get_train_dataloader" in trainer.__dict__
    grads = Grads(model)
    trainer.add_callback(grads)
    if hook_wrapper:
        trainer.add_callback(HookWrapper(trainer))
    if rank1_wrapper and dist.get_rank() == 1:
        # A rank-local instance wrapper (an observer on one rank only): that rank refuses windows, so no rank runs one.
        trainer.compute_loss = functools.partial(type(trainer).compute_loss, trainer)
    undo = inject(mod, engine, model) if inject is not None else None
    engine.layer_major_stats(reset = True)
    reduces = getattr(mod, "_DDP_REDUCES", [0])[0]
    out = io.StringIO()
    items = []
    real_get = transformers.Trainer.get_batch_samples
    if record is not None:
        # The step's global token count as this transformers computes it (with or without the label shift).

        def get_batch_samples(self, *a, **k):
            got = real_get(self, *a, **k)
            items.append(got[1])
            return got

        transformers.Trainer.get_batch_samples = get_batch_samples
    try:
        with contextlib.redirect_stdout(out):
            trainer.train()
    finally:
        transformers.Trainer.get_batch_samples = real_get
        if undo is not None:
            undo()
    stats = engine.layer_major_stats()
    logs = [
        {
            k: v
            for k, v in h.items()
            if k in ("loss", "entropy", "mean_token_accuracy", "num_tokens")
        }
        for h in trainer.state.log_history
        if "loss" in h
    ]
    weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    if record is not None:
        record.extend(seen)
        record.append([int(n) for n in items])
    return {
        "grads": grads.steps,
        "weights": weights,
        "logs": logs,
        "windows": stats["windows"],
        "aborts": stats["aborts"],
        "reduces": getattr(mod, "_DDP_REDUCES", [0])[0] - reduces,
        "printed": out.getvalue(),
    }


def reference(per_rank_batches, items):
    """One process, each step's global batch (every rank's micro-batches), loss summed over the global token count
    (the trainer's own count, gathered over the ranks)."""
    model = build_model()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD([p for _, p in params], lr = LR)
    steps = []
    for s in range(STEPS):
        batches = [b for rank in per_rank_batches for b in rank[s * GA : (s + 1) * GA]]
        opt.zero_grad(set_to_none = True)
        for b in batches:
            model(**b, num_items_in_batch = torch.tensor(items[s])).loss.backward()
        steps.append({n: p.grad.detach().clone() for n, p in params})
        opt.step()
    return steps, {n: p.detach().clone() for n, p in params}


def bitwise(a, b):
    return all(
        x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b)
    ) and len(a) == len(b)


def max_diff(a, b):
    return max((x[k] - y[k]).abs().max().item() for x, y in zip(a, b) for k in x)


def close(a, b):
    return len(a) == len(b) and all(
        torch.allclose(x[k], y[k], rtol = 1e-4, atol = 1e-6) for x, y in zip(a, b) for k in x
    )


def main(out_dir, zoo_dir):
    torch.set_num_threads(1)
    sys.path.insert(
        0, zoo_dir
    )  # unsloth_zoo.gradient_checkpointing from the same tree as the engine
    mod, engine = load(zoo_dir)
    report = {}

    def run(name, mode, **kw):
        return train(mod, engine, os.path.join(out_dir, name), mode, **kw)

    seen = []
    plain = run("plain", "0", record = seen)
    rank = dist.get_rank()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, seen)
    items = gathered[0][-1]
    assert all(g[-1] == items for g in gathered) and len(items) >= STEPS, gathered[0][-1]
    ref_grads, ref_weights = reference([g[:-1] for g in gathered], items)
    lm = run("layer_major", "1")

    def drop_reduce(mod, engine, model):
        real = getattr(mod, "_reduce_like_ddp", None)
        mod._reduce_like_ddp = lambda ddp: None
        return lambda: setattr(mod, "_reduce_like_ddp", real)

    dropped = run("dropped_reduce", "1", inject = drop_reduce)

    def rank1(patch):
        def inject(mod, engine, model):
            return patch(mod, engine, model) if dist.get_rank() == 1 else None

        return inject

    def unfit(mod, engine, model):
        real = mod._fits
        mod._fits = lambda *a, **k: (False, "simulated rank-local memory check")
        return lambda: setattr(mod, "_fits", real)

    def forward_abort(mod, engine, model):
        real = engine._recorders_beyond_hidden_states
        engine._recorders_beyond_hidden_states = lambda: True
        return lambda: setattr(engine, "_recorders_beyond_hidden_states", real)

    def backward_oom(mod, engine, model):
        armed = [True]
        mlp = model.get_base_model().model.layers[3].mlp
        real = mlp.forward

        def oom(*args, **kwargs):
            # Only in a window's decoder recompute: on the main thread, outside any micro-batch, while a window runs.
            if (
                armed[0]
                and engine._WINDOW_LOCK.locked()
                and torch.is_grad_enabled()
                and mod._TLS.__dict__.get("passthrough") is None
            ):
                armed[0] = False
                raise torch.OutOfMemoryError("simulated")
            return real(*args, **kwargs)

        mlp.forward = oom
        return lambda: mlp.__dict__.pop("forward", None)

    def buffer_written(mod, engine, model):
        # A forward writing a DDP-broadcast buffer the output reads (a counter, as BatchNorm's running stats and
        # dynamic RoPE): every rank's commit vote fails, and the buffer is put back before the standard retry.
        attn = model.get_base_model().model.layers[2].self_attn
        attn.register_buffer("unsloth_calls", torch.zeros(()), persistent = False)
        mlp = model.get_base_model().model.layers[2].mlp
        real = mlp.forward

        def write(*args, **kwargs):
            attn.unsloth_calls.add_(1)
            return real(*args, **kwargs) + 1e-3 * attn.unsloth_calls

        mlp.forward = write
        return lambda: mlp.__dict__.pop("forward", None)

    buf0 = run("buffer_written_plain", "0", inject = buffer_written)
    buf_r = run("buffer_written", "1", inject = buffer_written)
    hook_r = run("ddp_wrapper_hook", "1", hook_wrapper = True)
    wrap_r = run("rank1_instance_wrapper", "1", rank1_wrapper = True)
    unfit_r = run("rank1_unfit", "1", inject = rank1(unfit))
    fwd_r = run("rank1_forward_abort", "1", inject = rank1(forward_abort))
    bwd_r = run("rank1_backward_oom", "1", inject = rank1(backward_oom))

    for name, r in (
        ("plain", plain),
        ("layer_major", lm),
        ("dropped_reduce", dropped),
        ("buffer_written", buf_r),
        ("ddp_wrapper_hook", hook_r),
        ("rank1_instance_wrapper", wrap_r),
        ("rank1_unfit", unfit_r),
        ("rank1_forward_abort", fwd_r),
        ("rank1_backward_oom", bwd_r),
    ):
        report[name] = {
            "windows": r["windows"],
            "aborts": r["aborts"],
            "reduces": r["reduces"],
            "bitwise_vs_plain": bitwise(r["grads"], plain["grads"])
            and bitwise([r["weights"]], [plain["weights"]])
            and r["logs"] == plain["logs"],
            "close_to_one_process": close(r["grads"], ref_grads)
            and close([r["weights"]], [ref_weights]),
            "max_diff_vs_plain": max_diff(r["grads"], plain["grads"]),
            "other_rank_message": "another rank" in r["printed"],
            "wrapper_hook_message": "hooks on the DDP wrapper" in r["printed"],
            "steps": len(r["grads"]),
        }
    report["one_process_vs_plain_max_diff"] = max_diff(plain["grads"], ref_grads)
    report["buffer_written"]["bitwise_vs_its_plain"] = (
        bitwise(buf_r["grads"], buf0["grads"])
        and bitwise([buf_r["weights"]], [buf0["weights"]])
        and buf_r["logs"] == buf0["logs"]
    )
    # use_gradient_checkpointing = "unsloth" at max_seq_length >= 512 (Unsloth's default): its checkpoint shim on
    # every decoder layer, 512-token rows. Windows run where the engine accepts the shim, else every rank aborts
    # alike and runs the standard step; either way bitwise equal to plain DDP.
    gc = unsloth_gc()
    if gc is None:
        report["unsloth_gc"] = None
    else:
        shim = gc.unsloth_gradient_checkpoint
        probe = build_model().get_base_model().model.layers[0]
        probe._gradient_checkpointing_func = shim
        accepted = engine._checkpoint_mode(probe)[1] is None
        g0 = run("unsloth_gc_plain", "0", seq = 512, gc_shim = shim)
        g1 = run("unsloth_gc", "1", seq = 512, gc_shim = shim)
        report["unsloth_gc"] = {
            "accepted": accepted,
            "windows": g1["windows"],
            "aborts": g1["aborts"],
            "reduces": g1["reduces"],
            "plain_windows": g0["windows"],
            "bitwise": bitwise(g1["grads"], g0["grads"])
            and bitwise([g1["weights"]], [g0["weights"]])
            and g1["logs"] == g0["logs"],
        }
    try:
        import datasets  # noqa: F401
        import tokenizers  # noqa: F401
        import trl  # noqa: F401
    except ImportError:
        report["sft"] = None
    else:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        s0 = run("sft_plain", "0", sft = True)
        s1 = run("sft_layer_major", "1", sft = True)
        # Rank 1 stops in its forward; the forward vote keeps rank 0 out of the loss parts' gathers.
        s2 = run("sft_rank1_forward_abort", "1", sft = True, inject = rank1(forward_abort))
        report["sft"] = {
            "windows": s1["windows"],
            "plain_windows": s0["windows"],
            "bitwise": bitwise(s1["grads"], s0["grads"])
            and bitwise([s1["weights"]], [s0["weights"]]),
            "logs_equal": s1["logs"] == s0["logs"] and len(s0["logs"]) == STEPS,
            "abort_windows": s2["windows"],
            "abort_aborts": s2["aborts"],
            "abort_bitwise": bitwise(s2["grads"], s0["grads"]) and s2["logs"] == s0["logs"],
        }
    print(f"\nRANK_REPORT {rank} " + json.dumps(report), flush = True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
