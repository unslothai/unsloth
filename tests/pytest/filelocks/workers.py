from __future__ import annotations
import time
from pathlib import Path
import os
import uuid
from typing import Optional

def load_from_pretrained(model_name: str, load_in_4bit: bool = False) -> None:
    """
    Load a small model via Unsloth. Intentionally minimal: we don't do generation;
    the goal is to hit the file-locking code path and then exit.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        max_seq_length=32,
        dtype=None,
    )

    _ = tokenizer("hi", return_tensors="pt")
    del model, tokenizer
    print("load_from_pretrained: done")


def import_unsloth() -> None:
    import unsloth  # noqa: F401
    print("import_unsloth: done")
    # Print version so parent can see something on stdout if it wants.
    try:
        import inspect
        print(getattr(unsloth, "__version__", "unknown"))
    except Exception:
        pass

def barrier_wait(base_dir: str | Path, name: str, nprocs: int, timeout_s: float = 600, poll_ms: int = 100) -> None:
    base = Path(base_dir) / name
    base.mkdir(parents=True, exist_ok=True)
    token = base / f"{os.getpid()}-{uuid.uuid4().hex}.arrived"
    token.write_text("")
    go = base / ".go"
    start = time.perf_counter()
    while True:
        if go.exists():
            return
        arrivals = [p for p in base.iterdir() if p.is_file() and p.name != ".go"]
        if len(arrivals) >= nprocs:
            try:
                fd = os.open(str(go), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write("ok")
                return
            except FileExistsError:
                return
        if time.perf_counter() - start > timeout_s:
            raise TimeoutError(
                f"Barrier '{name}' timed out after {timeout_s}s: arrivals={len(arrivals)}/{nprocs}, dir={base}"
            )
        time.sleep(poll_ms / 1000.0)


def run_full(
    model_name: str,
    load_in_4bit: bool = True,
    chat_template: str = "qwen3",
    *,
    barrier_base: Optional[str] = None,
    nprocs: int = 1,
    dataset_slice: str = "train[:1000]",
    artifacts_dir: Optional[str] = None,
    enable_push: bool = False,
    hf_repo: str = "",
    push_token: str = "",
    enable_gguf: bool = True,
) -> None:
    import os
    os.environ["UNSLOTH_LOGGING_ENABLED"] = "1"
    from unsloth import FastLanguageModel
    if barrier_base:
        barrier_wait(barrier_base, "after_import_unsloth", nprocs)

    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if load_in_4bit and not has_cuda:
        raise RuntimeError("load_in_4bit=True but no CUDA device available")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=bool(load_in_4bit),
        max_seq_length=256,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    if barrier_base:
        barrier_wait(barrier_base, "before_get_chat_template", nprocs)

    from unsloth.chat_templates import get_chat_template, standardize_data_formats
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split=dataset_slice)
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    try:
        import bitsandbytes as _bnb  # noqa: F401
        optim_name = "adamw_8bit"
    except Exception:
        optim_name = "adamw_torch"

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            max_steps=5,
            learning_rate=2e-4,
            logging_steps=1,
            optim=optim_name,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )
    trainer.train()

    save_root = Path(artifacts_dir or ".").resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    target_dir = save_root / "model"

    if barrier_base:
        barrier_wait(barrier_base, "before_save_pretrained_merged", nprocs)
    model.save_pretrained_merged(str(target_dir), tokenizer, save_method="merged_16bit")

    if barrier_base:
        barrier_wait(barrier_base, "before_push_to_hub_merged", nprocs)
    if enable_push and hf_repo and push_token:
        model.push_to_hub_merged(hf_repo, tokenizer, save_method="merged_16bit", token=push_token)
    else:
        print("Skipping push_to_hub_merged")

    if barrier_base:
        barrier_wait(barrier_base, "before_save_pretrained_gguf", nprocs)
    if enable_gguf:
        model.save_pretrained_gguf(str(target_dir), tokenizer, quantization_method="q4_k_m")
    else:
        print("Skipping GGUF save")

    if barrier_base:
        barrier_wait(barrier_base, "before_push_to_hub_gguf", nprocs)
    if enable_push and hf_repo and push_token and enable_gguf:
        model.push_to_hub_gguf(hf_repo, tokenizer, quantization_method="q4_k_m", token=push_token)
    else:
        print("Skipping push_to_hub_gguf")

    del model, tokenizer
    print("run_full: done")
