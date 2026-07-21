#!/usr/bin/env python3

"""
🦥 Starter Script for Fine-Tuning FastLanguageModel with Unsloth

Configurable options for model loading, PEFT, training, and saving/pushing.
Customize the dataset loading/preprocessing and the save/push config for your case.

Usage (most options have sensible defaults; this is an extended example):
    python unsloth-cli.py --model_name "unsloth/llama-3-8b" --max_seq_length 8192 --dtype None --load_in_4bit \
    --r 64 --lora_alpha 32 --lora_dropout 0.1 --bias "none" --use_gradient_checkpointing "unsloth" \
    --random_state 3407 --use_rslora --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --warmup_steps 5 --max_steps 400 --learning_rate 2e-6 --logging_steps 1 --optim "adamw_8bit" \
    --weight_decay 0.005 --lr_scheduler_type "linear" --seed 3407 --output_dir "outputs" \
    --report_to "tensorboard" --save_model --save_path "model" --quantization_method "f16" \
    --push_model --hub_path "hf/model" --hub_token "your_hf_token"

Run `python unsloth-cli.py --help` for the full list of options.
"""

import argparse
import os


def _is_mlx_backend(unsloth_module):
    return bool(getattr(unsloth_module, "_IS_MLX", False))


def _normalize_dtype(dtype, is_mlx):
    if (
        is_mlx
        and isinstance(dtype, str)
        and dtype.strip().lower() in {"", "none", "auto"}
    ):
        return None
    return dtype


def _prepare_device_map(is_mlx):
    if is_mlx:
        return None, False

    from unsloth.models.loader_utils import prepare_device_map
    return prepare_device_map()


class _CallableTokenizerProxy:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, text, *args, **kwargs):
        # MLX/torch-free: never request torch tensors; keep plain python ids.
        kwargs.pop("return_tensors", None)
        wrapped = getattr(self._tokenizer, "_tokenizer", None)
        if callable(wrapped):
            return wrapped(text, *args, **kwargs)

        add_special_tokens = kwargs.get("add_special_tokens", False)
        input_ids = self._tokenizer.encode(text, add_special_tokens = add_special_tokens)
        return {"input_ids": input_ids}


def _tokenizer_for_raw_text_loader(tokenizer, is_mlx):
    if not is_mlx or callable(tokenizer):
        return tokenizer
    return _CallableTokenizerProxy(tokenizer)


def _raw_text_loader_for_backend(
    RawTextDataLoader,
    tokenizer,
    is_mlx,
    chunk_size = 2048,
    stride = 512,
):
    return RawTextDataLoader(
        _tokenizer_for_raw_text_loader(tokenizer, is_mlx),
        chunk_size,
        stride,
        return_tokenized = not is_mlx,
    )


def _train_with_legacy_save_control(trainer, is_mlx):
    if not is_mlx:
        return trainer.train()

    original_save_model = getattr(trainer, "save_model", None)
    if original_save_model is None:
        return trainer.train()

    def skip_internal_final_save(*args, **kwargs):
        raise ValueError("legacy unsloth-cli.py owns final save")

    trainer.save_model = skip_internal_final_save
    try:
        return trainer.train()
    finally:
        trainer.save_model = original_save_model


def _iter_quantization_methods(quantization):
    if isinstance(quantization, list):
        return quantization
    return [quantization]


def _save_or_push_model(model, tokenizer, args, is_mlx):
    if not args.save_model:
        print("Warning: The model is not saved!")
        return

    # Enter the GGUF branch when saving or pushing GGUF, so --push_gguf works
    # without --save_gguf (the local save is guarded separately below).
    if args.save_gguf or args.push_gguf:
        if not args.save_gguf:
            print(
                "Warning: --save_gguf not set, pushing GGUF to hub without saving locally."
            )
        for quantization_method in _iter_quantization_methods(args.quantization):
            if args.save_gguf:
                print(f"Saving model with quantization method: {quantization_method}")
                model.save_pretrained_gguf(
                    args.save_path,
                    tokenizer,
                    quantization_method = quantization_method,
                )
            if args.push_model or args.push_gguf:
                model.push_to_hub_gguf(
                    args.hub_path,
                    tokenizer,
                    quantization_method = quantization_method,
                    token = args.hub_token,
                )
        return

    if is_mlx:
        model.save_pretrained_merged(
            args.save_path,
            tokenizer,
            save_method = args.save_method,
            push_to_hub = args.push_model,
            repo_id = args.hub_path if args.push_model else None,
            token = args.hub_token,
        )
        return

    model.save_pretrained_merged(
        args.save_path, tokenizer, save_method = args.save_method
    )
    if args.push_model:
        model.push_to_hub_merged(
            args.hub_path, tokenizer, args.save_method, token = args.hub_token
        )


def _build_sft_config(SFTConfig, args, is_mlx, bf16_supported):
    config_kwargs = dict(
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not bf16_supported,
        bf16 = bf16_supported,
        logging_steps = args.logging_steps,
        optim = args.optim,
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = args.seed,
        output_dir = args.output_dir,
        report_to = args.report_to,
        max_length = args.max_seq_length,
        dataset_num_proc = 2,
        packing = args.packing,
    )
    if is_mlx:
        if args.per_device_eval_batch_size != 4:
            print(
                "Warning: --per_device_eval_batch_size is ignored on MLX without eval data."
            )
    else:
        config_kwargs["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    return SFTConfig(**config_kwargs)


def run(args):
    import unsloth
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers.utils import strtobool
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    import logging
    from unsloth import RawTextDataLoader

    logging.getLogger("hf-to-gguf").setLevel(logging.WARNING)

    is_mlx = _is_mlx_backend(unsloth)

    # Load model and tokenizer
    device_map, distributed = _prepare_device_map(is_mlx)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype = _normalize_dtype(args.dtype, is_mlx),
        load_in_4bit = args.load_in_4bit,
        device_map = device_map,
    )

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.r,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        bias = args.bias,
        use_gradient_checkpointing = args.use_gradient_checkpointing,
        random_state = args.random_state,
        use_rslora = args.use_rslora,
        loftq_config = args.loftq_config,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def load_dataset_smart(args):
        from transformers.utils import strtobool
        if args.raw_text_file:
            loader = _raw_text_loader_for_backend(
                RawTextDataLoader, tokenizer, is_mlx, args.chunk_size, args.stride
            )
            dataset = loader.load_from_file(args.raw_text_file)
        elif args.dataset.endswith((".txt", ".md", ".json", ".jsonl")):
            # Auto-detect local raw text files
            loader = _raw_text_loader_for_backend(RawTextDataLoader, tokenizer, is_mlx)
            dataset = loader.load_from_file(args.dataset)
        else:
            use_modelscope = strtobool(
                os.environ.get("UNSLOTH_USE_MODELSCOPE", "False")
            )
            if use_modelscope:
                from modelscope import MsDataset
                dataset = MsDataset.load(args.dataset, split = "train")
            else:
                dataset = load_dataset(args.dataset, split = "train")

            # Format structured datasets
            dataset = dataset.map(formatting_prompts_func, batched = True)
        return dataset

    # Load dataset using smart loader
    dataset = load_dataset_smart(args)
    print("Data is formatted and ready!")

    # Configure training arguments
    training_args = _build_sft_config(SFTConfig, args, is_mlx, is_bfloat16_supported())
    if distributed:
        training_args.ddp_find_unused_parameters = False

    # Initialize trainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )

    _train_with_legacy_save_control(trainer, is_mlx)

    _save_or_push_model(model, tokenizer, args, is_mlx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "🦥 Fine-tune your llm faster using unsloth!"
    )

    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument(
        "--model_name",
        type = str,
        default = "unsloth/llama-3-8b",
        help = "Model name to load",
    )
    model_group.add_argument(
        "--max_seq_length",
        type = int,
        default = 2048,
        help = "Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!",
    )
    model_group.add_argument(
        "--dtype",
        type = str,
        default = None,
        help = "Data type for model (None for auto detection)",
    )
    model_group.add_argument(
        "--load_in_4bit",
        action = "store_true",
        help = "Use 4bit quantization to reduce memory usage",
    )
    model_group.add_argument(
        "--dataset",
        type = str,
        default = "yahma/alpaca-cleaned",
        help = "Huggingface dataset to use for training",
    )

    lora_group = parser.add_argument_group(
        "🧠 LoRA Options",
        "These options are used to configure the LoRA model.",
    )
    lora_group.add_argument(
        "--r",
        type = int,
        default = 16,
        help = "Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type = int,
        default = 16,
        help = "LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)",
    )
    lora_group.add_argument(
        "--lora_dropout",
        type = float,
        default = 0.0,
        help = "LoRA dropout rate, default is 0.0 which is optimized.",
    )
    lora_group.add_argument(
        "--bias",
        type = str,
        default = "none",
        help = "Bias setting for LoRA",
    )
    lora_group.add_argument(
        "--use_gradient_checkpointing",
        type = str,
        default = "unsloth",
        help = "Use gradient checkpointing",
    )
    lora_group.add_argument(
        "--random_state",
        type = int,
        default = 3407,
        help = "Random state for reproducibility, default is 3407.",
    )
    lora_group.add_argument(
        "--use_rslora",
        action = "store_true",
        help = "Use rank stabilized LoRA",
    )
    lora_group.add_argument(
        "--loftq_config",
        type = str,
        default = None,
        help = "Configuration for LoftQ",
    )

    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument(
        "--per_device_train_batch_size",
        type = int,
        default = 2,
        help = "Batch size per device during training, default is 2.",
    )
    training_group.add_argument(
        "--per_device_eval_batch_size",
        type = int,
        default = 4,
        help = "Batch size per device during evaluation, default is 4.",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type = int,
        default = 4,
        help = "Number of gradient accumulation steps, default is 4.",
    )
    training_group.add_argument(
        "--warmup_steps",
        type = int,
        default = 5,
        help = "Number of warmup steps, default is 5.",
    )
    training_group.add_argument(
        "--max_steps",
        type = int,
        default = 400,
        help = "Maximum number of training steps.",
    )
    training_group.add_argument(
        "--learning_rate",
        type = float,
        default = 2e-4,
        help = "Learning rate, default is 2e-4.",
    )
    training_group.add_argument(
        "--optim",
        type = str,
        default = "adamw_8bit",
        help = "Optimizer type.",
    )
    training_group.add_argument(
        "--weight_decay",
        type = float,
        default = 0.01,
        help = "Weight decay, default is 0.01.",
    )
    training_group.add_argument(
        "--lr_scheduler_type",
        type = str,
        default = "linear",
        help = "Learning rate scheduler type, default is 'linear'.",
    )
    training_group.add_argument(
        "--seed",
        type = int,
        default = 3407,
        help = "Seed for reproducibility, default is 3407.",
    )
    training_group.add_argument(
        "--packing",
        action = "store_true",
        help = "Enable padding-free sample packing via TRL's bin packer.",
    )

    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument(
        "--report_to",
        type = str,
        default = "tensorboard",
        choices = [
            "azure_ml",
            "clearml",
            "codecarbon",
            "comet_ml",
            "dagshub",
            "dvclive",
            "flyte",
            "mlflow",
            "neptune",
            "tensorboard",
            "wandb",
            "all",
            "none",
        ],
        help = (
            "The list of integrations to report the results and logs to. Supported platforms are:\n\t\t "
            "'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', "
            "'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations "
            "installed, 'none' for no integrations."
        ),
    )
    report_group.add_argument(
        "--logging_steps",
        type = int,
        default = 1,
        help = "Logging steps, default is 1",
    )

    save_group = parser.add_argument_group("💾 Save Model Options")
    save_group.add_argument(
        "--output_dir",
        type = str,
        default = "outputs",
        help = "Output directory",
    )
    save_group.add_argument(
        "--save_model",
        action = "store_true",
        help = "Save the model after training",
    )
    save_group.add_argument(
        "--save_method",
        type = str,
        default = "merged_16bit",
        choices = ["merged_16bit", "merged_4bit", "lora"],
        help = "Save method for the model, default is 'merged_16bit'",
    )
    save_group.add_argument(
        "--save_gguf",
        action = "store_true",
        help = "Convert the model to GGUF after training",
    )
    save_group.add_argument(
        "--save_path",
        type = str,
        default = "model",
        help = "Path to save the model",
    )
    save_group.add_argument(
        "--quantization",
        type = str,
        default = "q8_0",
        nargs = "+",
        help = (
            "Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), "
            "Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf"
        ),
    )

    push_group = parser.add_argument_group("🚀 Push Model Options")
    push_group.add_argument(
        "--push_model",
        action = "store_true",
        help = "Push the model to Hugging Face hub after training",
    )
    push_group.add_argument(
        "--push_gguf",
        action = "store_true",
        help = "Push the model as GGUF to Hugging Face hub after training",
    )
    push_group.add_argument(
        "--hub_path",
        type = str,
        default = "hf/model",
        help = "Path on Hugging Face hub to push the model",
    )
    push_group.add_argument(
        "--hub_token",
        type = str,
        help = "Token for pushing the model to Hugging Face hub",
    )

    parser.add_argument(
        "--raw_text_file", type = str, help = "Path to raw text file for training"
    )
    parser.add_argument(
        "--chunk_size", type = int, default = 2048, help = "Size of text chunks for training"
    )
    parser.add_argument(
        "--stride", type = int, default = 512, help = "Overlap between chunks"
    )

    args = parser.parse_args()
    run(args)
