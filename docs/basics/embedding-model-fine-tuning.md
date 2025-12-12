# Fine‑tuning Embedding / Bi‑encoder Models with Unsloth

Unsloth is not limited to causal LLMs. You can also fine‑tune **embedding models** (e.g., BERT / Arctic‑Embed / E5 / GTE / Qwen‑Embed) using the same `FastModel` API and PEFT/LoRA, then train with the `sentence-transformers` ecosystem.

This guide shows a complete, working bi‑encoder setup.

## 1. Disable fast‑generation kernels

Embedding models are typically **non‑causal**. Unsloth’s fast generation kernels are designed for causal decoding, so disable them before importing Unsloth:

```python
import os
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"

import unsloth
from unsloth import FastModel
```

## 2. Load a base embedding model

For BERT‑style models you must pass the correct `auto_model` class and disable the pooling head if the checkpoint includes one.

```python
from transformers import BertModel

BASE_MODEL_ID = "Snowflake/snowflake-arctic-embed-l"  # replace as needed
MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name      = BASE_MODEL_ID,
    auto_model      = BertModel,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,
    add_pooling_layer = False,
)
```

Notes:

- For other encoders, swap `BertModel` for the matching HF class (e.g., `RobertaModel`, `AutoModel`, `Qwen2Model`, etc.).
- `max_seq_length` should match your training/eval needs.

## 3. Add LoRA adapters for feature extraction

Use `TaskType.FEATURE_EXTRACTION` so PEFT config matches encoder training.

```python
from peft import TaskType

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
```

Target modules may vary by architecture. For example, for RoBERTa‑like models you might still use `query/key/value/dense`, while for some modern encoders you’ll want to include MLP/projection blocks.

## 4. Wrap as a SentenceTransformer

`sentence-transformers` expects its own `Transformer` module. You can reuse the Unsloth‑loaded model/tokenizer by injecting them into that module.

```python
import sentence_transformers
from sentence_transformers import SentenceTransformer

def get_st_unsloth_wrapper(
    model,
    tokenizer,
    base_model_id=BASE_MODEL_ID,
    pooling_mode="cls",
    max_seq_length=MAX_SEQ_LENGTH,
):
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
```

## 5. Train with MultipleNegativesRankingLoss

Example using a JSONL dataset with `anchor` / `positive` fields and an IR evaluator.

```python
from datasets import load_dataset, concatenate_datasets
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
import torch

loss = MultipleNegativesRankingLoss(sbert_model)

dataset = load_dataset("json", data_files="/path/to/dataset.jsonl", split="train")
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

args = SentenceTransformerTrainingArguments(
    output_dir="embeddings_run",
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
    run_name=f"{BASE_MODEL_ID.split('/')[-1]}-st-finetune",
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
)

trainer.train()
```

## 6. Save a merged model

When training completes, you should **merge the LoRA adapters first**, then save the full `SentenceTransformer`.

Do **not** call `model.save_pretrained_merged` into the same directory as `sbert_model.save_pretrained`, since it overwrites SentenceTransformer files (like `config.json`) and produces a broken folder.

Correct way:

```python
# Merge LoRA weights into the base encoder in-place
model.merge_and_unload()

# `sbert_model` now contains merged encoder + pooling/normalize heads
sbert_model.save_pretrained("embeddings_merged")
```

The resulting folder can be uploaded to Hugging Face or used directly for inference as a SentenceTransformer.

Optional: if you also want a **plain HF encoder-only checkpoint**, save it to a *different* directory:

```python
model.save_pretrained("embeddings_merged_hf")
tokenizer.save_pretrained("embeddings_merged_hf")
```

---

If you run into issues with a specific encoder architecture, open an issue with the model ID and a minimal repro.
