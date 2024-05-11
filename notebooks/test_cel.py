import argparse
import logging
from pathlib import Path

import torch
from cel_analysis import load_log_diffs
from peft import LoraConfig
from tabulate import tabulate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.models.llama import LlamaConfig
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import enable_full_determinism
from transformers.trainer_utils import set_seed as hf_set_seed
from trl import SFTTrainer

import unsloth.utils.data as data_utils
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel
from unsloth.models._utils import patch_tokenizer, prepare_model_for_kbit_training
from unsloth.utils.data import get_data_loader
from unsloth.utils.profiling import MetricsCallBack

parent_dir = Path(__file__).parent.absolute()
SEED = 3407
enable_full_determinism(SEED)


def get_quant_config(load_in_4bit, dtype):
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=dtype,
    )


def get_model(args):
    dtype = getattr(torch, args.dtype)
    quant_config = (
        get_quant_config(args.load_in_4bit, dtype) if args.load_in_4bit else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        torch_dtype=dtype,
    )
    return model


def get_tokenizer(args):
    common_args = {
        "model_max_length": args.max_seq_len,
        "padding_side": "right",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, **common_args)
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            **common_args,
        )
    return tokenizer


def get_trainer_args(args):
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=1,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        fp16=args.dtype == "float16",
        bf16=args.dtype == "bfloat16",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        data_seed=SEED,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # Metrics
        skip_memory_metrics=False,
    )
    return training_args


def get_lora_config(args):
    accepted_modules = frozenset(
        (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )

    peft_config = LoraConfig(
        target_modules=accepted_modules,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return peft_config


def get_sft_trainer(
    args, model, tokenizer, dataset, peft_config, trainer_args, use_fused_cel
):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=trainer_args,
    )

    # Remove default callbacks, make less verbose
    trainer.remove_callback(ProgressCallback)
    trainer.model.enable_input_require_grads()
    file_prefix = "fused_cel" if use_fused_cel else ""
    file_prefix += "_" + args.dtype
    _ = trainer.add_callback(MetricsCallBack(name=file_prefix, verbose=False))
    return trainer


def run_test_batches(model, batches):
    model = model.cuda()
    outputs = []
    for batch in batches:
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model(**batch)
        outputs.append(out.loss.detach().item())
    return outputs


def run_train_loop(
    model, tokenizer, dataset, peft_config, training_args, cli_args, use_fused_cel=False
):
    model = patch_model_fused_cel(model, use_fused_cel=use_fused_cel)
    trainer = get_sft_trainer(
        cli_args,
        model,
        tokenizer,
        dataset,
        peft_config,
        training_args,
        use_fused_cel=use_fused_cel,
    )
    _ = trainer.train()


def run_benchmark(args):
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    model, tokenizer = patch_tokenizer(model, tokenizer)

    if args.overwrite_output_dir:
        import shutil

        shutil.rmtree(args.output_dir, ignore_errors=True)

    training_args = get_trainer_args(args)
    peft_config = get_lora_config(args)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    dataset = data_utils.get_alpaca(tokenizer)

    formatted_args = "\n ".join([f"{k}={v}" for k, v in vars(args).items()])
    print(f"Running with:\n {formatted_args}")
    if args.sanity_check:
        dataloader = get_data_loader(
            dataset,
            tokenizer,
            args.max_seq_len,
            args.batch_size,
            num_examples=args.max_steps,
        )
        batches = [b for b in dataloader]
        original_outputs = run_test_batches(model, batches)
        fused_model = patch_model_fused_cel(model, use_fused_cel=True)
        fused_outputs = run_test_batches(fused_model, batches)

        diffs = [
            abs(expected - actual)
            for expected, actual in zip(original_outputs, fused_outputs)
        ]
        rows = list(zip(original_outputs, fused_outputs, diffs))
        print(
            tabulate(
                rows, headers=["original loss", "fused loss", "absdiff"], floatfmt=".6f"
            )
        )
    else:
        # Run with and without fused CEL
        run_train_loop(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            peft_config=peft_config,
            training_args=training_args,
            cli_args=args,
            use_fused_cel=False,
        )

        run_train_loop(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            peft_config=peft_config,
            training_args=training_args,
            cli_args=args,
            use_fused_cel=True,
        )
        loss_df, metrics_df = load_log_diffs(args.output_dir)
        print(loss_df.to_string(float_format="%.6f", justify="left"))
        print(metrics_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--max_steps", type=int, default=5, help="Number of training steps"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="torch compute type"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./llama-10m",
        help="Path to the model, passed to huggingface `from_pretrained` method",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=True)
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)
