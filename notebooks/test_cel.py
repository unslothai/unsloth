import dataclasses
import json
import logging
import os

import torch
from IPython.core.interactiveshell import InteractiveShell
from llama_head import CEL_only_forward
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_pt_utils import _secs2timedelta
from trl import SFTTrainer

import unsloth.utils.data as data_utils
import unsloth.utils.memory as memory_utils
import unsloth.utils.testing as test_utils
from unsloth.kernels import fused_cel
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel
from unsloth.models._utils import patch_tokenizer
from unsloth.models.llama import FastLlamaModel

# logging.basicConfig(level=logging.WARNING)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_config = LlamaConfig.from_pretrained("./llama-10m.json")
model = AutoModelForCausalLM.from_pretrained(
    "./llama-10m", quantization_config=quant_config, torch_dtype=torch.bfloat16
)
# model = LlamaForCausalLM(model_config).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", model_max_length=4096, padding_side="right"
)
model, tokenizer = patch_tokenizer(model, tokenizer)

max_seq_length = 256

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    max_steps=5,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    overwrite_output_dir=True,
    # Metrics
    skip_memory_metrics=False,
    include_num_input_tokens_seen=True,
    include_tokens_per_second=True,
)

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

dataset = data_utils.get_alpaca(tokenizer)

peft_config = LoraConfig(
    target_modules=accepted_modules,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
# patched_model = patch_model_fused_cel(model, use_fused_cel=False)


class MetricsCallBack(ProgressCallback):
    def metrics_format(self, metrics):
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif isinstance(metrics_copy[k], float):
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def save_state(self, output_dir, state):
        json_string = (
            json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n"
        )
        json_path = os.path.join(output_dir, f"state-{state.global_step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # with open(
        #     os.path.join(args.output_dir, f"state-{state.global_step}.json"), "w"
        # ) as f:
        #     json.dump(state, f)
        #    self.save_state(args.output_dir, state)

        logs_formatted = self.metrics_format(logs)
        k_width = max(len(str(x)) for x in logs_formatted.keys())
        v_width = max(len(str(x)) for x in logs_formatted.values())
        print("Global Step: ", state.global_step)
        for key in sorted(logs_formatted.keys()):
            print(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")

        # if state.is_world_process_zero and self.training_bar is not None:
        #     # avoid modifying the logs object as it is shared between callbacks
        #     logs = copy.deepcopy(logs)
        #     _ = logs.pop("total_flos", None)
        #     # round numbers so that it looks better in console
        #     if "epoch" in logs:
        #         logs["epoch"] = round(logs["epoch"], 2)
        #     self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        # print("Final train logs: ", state.log_history)
        # print("Final state: ", state)
        # with open(os.path.join(args.output_dir, "train_logs.json"), "w") as f:
        #     json.dump(state.log_history, f)
        self.save_state(args.output_dir, state)
        super().on_train_end(args, state, control, **kwargs)


patched_model = model
trainer = SFTTrainer(
    model=patched_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=training_args,
)
trainer.remove_callback(ProgressCallback)
_ = trainer.add_callback(MetricsCallBack())
print(trainer.callback_handler.callback_list)
train_stats = trainer.train()

# print(trainer.log_metrics("train", train_stats.metrics))
