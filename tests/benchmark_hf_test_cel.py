import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from cel_analysis import load_log_diffs
from cel_test_utils import (
    get_model,
    get_peft_config,
    get_quant_config,
    get_sft_trainer,
    get_tokenizer,
    get_trainer_args,
)

# from transformers.trainer_utils import enable_full_determinism
from transformers.trainer_utils import set_seed as hf_set_seed

import unsloth.utils.data as data_utils
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel
from unsloth.models._utils import patch_tokenizer, prepare_model_for_kbit_training
from unsloth.utils.memory import empty_cache

parent_dir = Path(__file__).parent.absolute()
SEED = 3407
hf_set_seed(SEED)
torch.autograd.set_detect_anomaly(True)

import logging

logging.basicConfig(level=logging.INFO)


def run_train_loop(
    model,
    tokenizer,
    dataset,
    peft_config,
    training_args,
    cli_args,
    use_fused_cel=False,
    n_loop_iters=1,
):
    model = patch_model_fused_cel(
        model,
        use_fused_cel=use_fused_cel,
        fused_cel_n_loop_iters=n_loop_iters,
        # these are defaults
        fused_cel_ignore_index=-100,
        fused_cel_reduction="mean",
    )
    file_prefix = (
        ("fused" if use_fused_cel else "base")
        + "_"
        + cli_args.dtype
        + "_"
        + str(n_loop_iters)
    )

    trainer = get_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        trainer_args=training_args,
        max_seq_len=cli_args.max_seq_len,
        packing=cli_args.packing,
        file_prefix=file_prefix,
    )
    _ = trainer.train()


def get_model_and_tokenizer(args):
    dtype = getattr(torch, args.dtype)
    model_id = args.model_id

    quant_config = (
        get_quant_config(args.load_in_4bit, dtype) if args.load_in_4bit else None
    )
    model = get_model(
        model_id=model_id,
        dtype=dtype,
        use_fused_cel_layer=True,
        quant_config=quant_config,
    )
    tokenizer = get_tokenizer(model_id, args.max_seq_len)
    model, tokenizer = patch_tokenizer(model, tokenizer)

    return model, tokenizer


def run_benchmark(args):
    dtype = getattr(torch, args.dtype)
    model, tokenizer = get_model_and_tokenizer(args)

    if args.overwrite_output_dir:
        import shutil

        shutil.rmtree(args.output_dir, ignore_errors=True)

    training_args = get_trainer_args(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum_steps,
        dtype=dtype,
        seed=SEED,
        output_dir=args.output_dir,
    )
    peft_config = get_peft_config() if args.use_lora or args.load_in_4bit else None
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    dataset = data_utils.get_alpaca(tokenizer)

    formatted_args = "\n ".join([f"{k}={v}" for k, v in vars(args).items()])

    print(f"Running with:\n {formatted_args}")

    losses, metrics = [], []
    # Run reference once
    run_train_loop(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        peft_config=peft_config,
        training_args=training_args,
        cli_args=args,
        use_fused_cel=False,
    )
    del model
    del tokenizer
    empty_cache()

    for n_loop_iters in args.fused_cel_n_loop_iters:
        # Run with fused CEL
        model, tokenizer = get_model_and_tokenizer(args)
        run_train_loop(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            peft_config=peft_config,
            training_args=training_args,
            cli_args=args,
            use_fused_cel=True,
            n_loop_iters=n_loop_iters,
        )
        loss_df, metrics_df = load_log_diffs(args.output_dir)
        loss_df.columns.names = [
            loss_df.columns.names[0] + ", n_loop_it=" + str(n_loop_iters),
            loss_df.columns.names[1],
        ]
        losses.append(loss_df)
        # No fused always has n_loop_iters = 1
        metrics_df.loc["n_loop_iters"] = [1, n_loop_iters]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics_df.loc["trainable_params"] = [
            trainable_params,
            trainable_params,
        ]
        metrics_df.loc["total_params"] = [
            total_params,
            total_params,
        ]
        metrics.append(metrics_df)
        if args.print_accuracy:
            print(loss_df.to_string(float_format="%.6f", justify="left"))

    consolidated_metrics = pd.concat(metrics, axis=1).T.drop_duplicates()
    COL_ORDER = [
        "step",
        "trainable_params",
        "total_params",
        "n_loop_iters",
        "total_flos",
        "train_loss",
        "train_mem_gpu_peaked_delta",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_runtime",
    ]
    consolidated_metrics = consolidated_metrics[COL_ORDER]
    consolidated_metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"))
    if args.print_metrics:
        print(consolidated_metrics.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max_steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="torch compute type"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="Path to the model, passed to huggingface `from_pretrained` method",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True)
    parser.add_argument("--print_accuracy", action="store_true", default=True)
    parser.add_argument("--print_metrics", action="store_true", default=True)

    parser.add_argument(
        "--fused_cel_n_loop_iters",
        type=int,
        nargs="+",
        default=[1],
        help="""Number of loop iterations for fused CEL.  
        E.g., `n_loop_iters=4` will calculate the logits / loss in 4 chunks along sequence length.
        `batch_size * seqlen` must be divisible by `n_loop_iters`
        """,
    )
    args = parser.parse_args()
    run_benchmark(args)
