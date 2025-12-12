"""Example: fine-tune a bi-encoder / embedding model with Unsloth + sentence-transformers.

This script mirrors the pattern used for LLM fine-tuning, but for non-causal
encoder models (BERT / Arctic-Embed / E5 / GTE / etc.).

Key points:
- Disable fast-generation kernels (encoder models do not use causal decoding).
- Load the encoder via FastModel.from_pretrained with the right HF auto_model.
- Add LoRA adapters with TaskType.FEATURE_EXTRACTION.
- Wrap the Unsloth model into a SentenceTransformer and train normally.

Adjust BASE_MODEL_ID, dataset path, and LoRA target_modules for your model.
"""

import os

os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from transformers import BertModel, TrainerCallback

from unsloth import FastModel


class ShuffleDatasetCallback(TrainerCallback):
    """Randomize dataset per-epoch in a reproducible way."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        if getattr(self.trainer, "train_dataset", None) is None:
            return
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        new_seed = args.seed + current_epoch
        print(f"Randomizing train dataset for epoch {current_epoch} with seed {new_seed}...")
        self.trainer.train_dataset = self.trainer.train_dataset.shuffle(seed=new_seed)


BASE_MODEL_ID = "Snowflake/snowflake-arctic-embed-l"
BERT_MODEL = BertModel
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL_ID,
    auto_model=BERT_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    add_pooling_layer=False,
)

model = FastModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    task_type=TaskType.FEATURE_EXTRACTION,
)


def get_st_unsloth_wrapper(
    model,
    tokenizer,
    base_model_id=BASE_MODEL_ID,
    pooling_mode="cls",
    max_seq_length=MAX_SEQ_LENGTH,
):
    import sentence_transformers

    transformer_module = sentence_transformers.models.Transformer(
        model_name_or_path=base_model_id,
        max_seq_length=max_seq_length,
    )
    transformer_module.auto_model = model
    transformer_module.tokenizer = tokenizer

    hidden_size = model.config.hidden_size
    pooling_module = sentence_transformers.models.Pooling(
        word_embedding_dimension=hidden_size,
        pooling_mode=pooling_mode,
    )
    normalize_module = sentence_transformers.models.Normalize()
    return SentenceTransformer(modules=[transformer_module, pooling_module, normalize_module])


sbert_model = get_st_unsloth_wrapper(model, tokenizer)
loss = MultipleNegativesRankingLoss(sbert_model)


dataset_path = Path(os.environ.get("DATASET_JSONL", "/path/to/dataset.jsonl"))
dataset = load_dataset("json", data_files=str(dataset_path), split="train")

dataset = dataset.map(lambda x, idx: {"id": str(idx)}, with_indices=True)
dataset = dataset.shuffle(seed=42)
split_dataset = dataset.train_test_split(test_size=4096, seed=42)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

corpus_dataset = concatenate_datasets([train_dataset, validation_dataset])
corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
queries = dict(zip(validation_dataset["id"], validation_dataset["anchor"]))
relevant_docs = {qid: [qid] for qid in queries}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    score_functions={"cosine": cos_sim},
    name="ir-eval",
)

run = os.environ.get("RUN_ID", "0")
name = os.environ.get("RUN_NAME", "embedding_ft")

args = SentenceTransformerTrainingArguments(
    output_dir=f"{name}_{run}",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1024,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    tf32=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=20,
    metric_for_best_model="eval_ir-eval_cosine_ndcg@10",
    greater_is_better=True,
    logging_steps=10,
    report_to="wandb",
    run_name=f"{BASE_MODEL_ID.split('/')[-1]}-st-finetune-{run}",
    seed=42,
    max_grad_norm=1.0,
)

trainer = SentenceTransformerTrainer(
    model=sbert_model,
    args=args,
    train_dataset=train_dataset.select_columns(["anchor", "positive"]),
    eval_dataset=validation_dataset.select_columns(["anchor", "positive"]),
    loss=loss,
    evaluator=evaluator,
    callbacks=[],
)

trainer.add_callback(ShuffleDatasetCallback(trainer))

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained_merged(f"{name}_{run}_merged", tokenizer)

