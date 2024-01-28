import argparse
import json
import logging
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel
from unsloth.utils.logging import setup_logging
from unsloth.utils.modeling import QuantizationMethod
from unsloth.utils.profiling import (
    cuda_nvtx_range_context,
    cuda_profiler_wrapper,
    torch_profiler_context,
)

setup_logging(level="DEBUG")
logger = logging.getLogger(__file__)


MODEL_MAP = {
    "MISTRAL": {
        "GPTQ": "TheBloke/Mistral-7B-v0.1-GPTQ",
        "BNB": "unsloth/mistral-7b-bnb-4bit",
    },
    "LLAMA": {
        "GPTQ": "TheBloke/Llama-2-7B-GPTQ",
        "BNB": "unsloth/llama-2-7b-bnb-4bit",
    },
}


def patch_tokenizer(model, tokenizer):
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        # Fixes https://github.com/unslothai/unsloth/issues/5
        if hasattr(tokenizer, "unk_token"):
            tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
            tokenizer.pad_token = tokenizer.unk_token
        else:
            logger.warning_one(
                f"{model.config._name_or_path} does not have a padding or unknown token!\n"
                f"Will use the EOS token of id {tokenizer.eos_token_id} as padding."
            )
            assert hasattr(tokenizer, "eos_token")
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def get_model_and_tokenizer(model_name, model_type, dtype, max_seq_length):
    model_id = (
        MODEL_MAP[model_name.upper()]["BNB"]
        if "bnb" in model_type.lower()
        else MODEL_MAP[model_name.upper()]["GPTQ"]
    )
    logger.info(f"Loading {model_id}")
    if "gptq" in model_type.lower():
        if "hf" in model_type.lower():
            logger.info(f"Loading HF model {model_type}")

            from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

            quantization_config = GPTQConfig(bits=4, disable_exllama=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=dtype,
                device_map="auto",
            )
            tokenizer = patch_tokenizer(model, tokenizer)
            model.config.use_cache = False
            # https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996#file-finetune_llama_gptq-py
            model.config.pretraining_tp = 1

            if "triton" in model_type.lower():
                logger.info(
                    "Patching HuggingFace GPTQ linears with autogptq triton qlinear"
                )
                from auto_gptq.nn_modules.qlinear.qlinear_cuda import (
                    QuantLinear as QuantLinearCuda,
                )

                from unsloth.gptq.triton.layers import GPTQuantLinear

                GPTQuantLinear.inject_to_model(
                    model, target_module_type=QuantLinearCuda
                )
        else:
            logger.info(f"Loading Unsloth model {model_type}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                max_seq_length=max_seq_length,
                quantization_method=QuantizationMethod.GPTQ,
                load_in_4bit=False,
            )
    else:
        logger.info(f"Loading Unsloth model {model_type}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            max_seq_length=max_seq_length,
            quantization_method=QuantizationMethod.BNB,
            load_in_4bit=True,
        )

    return model, tokenizer


def prep_for_peft(
    model,
    model_type,
    lora_config,
    max_seq_length,
    use_gradient_checkpointing=True,
    use_reentrant=True,
):
    logger.info(f"Loading {model_type}")
    if "hf" in model_type.lower():
        from peft import get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
        )

        peft_model = get_peft_model(model, lora_config)

    else:
        config = lora_config.to_dict()
        del config["task_type"]
        peft_model = FastLanguageModel.get_peft_model(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
            max_seq_length=max_seq_length,
            **config,
        )

    return peft_model


def get_lora_config(
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
):
    config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config


def train_step(model, inputs):
    outputs = model(**inputs)
    outputs.loss.backward()


def run_profile(model, inputs, warmup_steps, profile_steps, dtype, outdir):
    with torch_profiler_context(
        warmup=warmup_steps, active=profile_steps, outdir=outdir
    ) as prof:
        runner = cuda_profiler_wrapper(prof, warmup=warmup_steps, rep=profile_steps)(
            train_step
        )

        with cuda_nvtx_range_context():
            with torch.cuda.amp.autocast(dtype=dtype):
                runner(model, inputs)


def get_dataset(dataset_id, tokenizer):
    logger.info("Loading dataset {}".format(dataset_id))

    if "alpaca" in dataset_id:
        PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        EOS_TOKEN = tokenizer.eos_token

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []

            for instruction, input, output in zip(instructions, inputs, outputs):
                text = PROMPT.format(instruction, input, output) + EOS_TOKEN
                texts.append(text)
            return {
                "text": texts,
            }

        dataset = load_dataset(dataset_id, split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True)
    elif "guanaco" in dataset_id:
        dataset = load_dataset(dataset_id, split="train")
    else:
        dataset = load_dataset(dataset_id, split="train")
    return dataset


def run_trainer(
    model,
    tokenizer,
    dataset,
    max_seq_length,
    dtype,
    max_steps=20,
    warmup_steps=5,
    out_dir=None,
):
    use_f16 = True if dtype == torch.float16 else False

    if use_f16:
        logger.info(f"Training in {torch.float16}")
    else:
        logger.info(f"Training in {torch.bfloat16}")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=use_f16,
            bf16=not use_f16,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=out_dir,
        ),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    for _ in range(warmup_steps):
        _ = model(**batch)

    stats = trainer.train()
    print(stats.metrics)
    print(f"Saving metrics to {out_dir}/stats.json")
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats.metrics, f)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral",
        choices=["mistral", "llama"],
        help="Model class, either mistral or llama",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unsloth-gptq-triton",
        choices=[
            "unsloth-gptq-triton",
            "hf-gptq-default",
            "hf-gptq-triton-patch",
            "unsloth-bnb",
        ],
        help="""Model type
        Reference HuggingFace implementations use either default auto-gptq quant linear layers which defaults to a `cuda` backend.
        However, the auto_gptq layer automatically disables the cuda kernel when the layer is trainable and falls back to a pure torch
        implementation (see https://github.com/AutoGPTQ/AutoGPTQ/blob/d2662b18bb91e1864b29e4e05862712382b8a076/auto_gptq/nn_modules/qlinear/qlinear_cuda.py#L40-L41)

        To make comparisons we patch the default HuggingFace model with auto_gptq triton qlinear layers to compare with unsloth GPTQ triton implementation.
        To use this patched model, select `hf-gptq-triton-patch`.
        """,
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    # parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--sample_data", type=str, default="./sample_batch.pt")
    parser.add_argument("--log_level", type=str, default="DEBUG")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--profile_steps", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--train_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="trainer_out")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="guanaco",
        choices=["guanaco", "alpaca"],
    )
    args = parser.parse_args()

    args.dtype = getattr(torch, args.dtype)
    args.dataset_id = (
        "timdettmers/openassistant-guanaco"
        if "guanaco" in args.dataset_id
        else "yahma/alpaca-cleaned"
    )

    model, tokenizer = get_model_and_tokenizer(
        args.model_name, args.model_type, args.dtype, args.max_seq_length
    )
    lora_config = get_lora_config(args.r, lora_alpha=args.lora_alpha)
    model = prep_for_peft(
        model,
        args.model_type,
        lora_config=lora_config,
        max_seq_length=args.max_seq_length,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.profile:
        logger.info("Profiling model...")
        sample_data = torch.load(args.sample_data)
        run_profile(
            model,
            sample_data,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            dtype=args.dtype,
            outdir=args.output_dir,
        )
    else:
        logger.info("Running training test...")
        dataset = get_dataset(args.dataset_id, tokenizer)
        run_trainer(
            model,
            tokenizer,
            dataset,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype,
            max_steps=args.train_steps,
            out_dir=args.output_dir,
        )
