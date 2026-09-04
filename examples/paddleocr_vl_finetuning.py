"""
=============================================================================
Fine-tuning PaddleOCR-VL-1.6 with Unsloth
=============================================================================

This example demonstrates how to fine-tune PaddlePaddle/PaddleOCR-VL-1.6
-- a state-of-the-art 0.9B document-parsing vision-language model -- for
custom OCR / document-understanding tasks using Unsloth's FastModel API.

The model uses an ERNIE-4.5-0.3B language backbone (18-layer, GQA) with a
27-layer NaViT-style dynamic-resolution vision encoder. It supports 109
languages and achieves 96.3% on OmniDocBench v1.6.

What this example covers
------------------------
1. Loading the model & processor with 4-bit QLoRA quantisation
2. Preparing an OCR / document-parsing dataset in chat format
3. Applying LoRA adapters with vision-layer support
4. Training with UnslothVisionDataCollator + SFTTrainer
5. Running inference before & after fine-tuning **with WER/CER metrics**
6. Saving & merging the model

Requirements
------------
- unsloth (pip install unsloth)
- unsloth_zoo (auto-installed with unsloth)
- transformers >= 4.55.0
- torch, torchvision, peft, trl, datasets
- einops (for vision-token reshaping)
- jiwer (for WER/CER evaluation metrics; ``pip install jiwer``)
- flash-attn (optional, for faster training)

References
----------
- Model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- Unsloth: https://github.com/unslothai/unsloth
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
"""

import os
import re
import random
import torch
from datasets import load_dataset
from unsloth import FastModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments
from trl import SFTTrainer


try:
    from jiwer import wer, cer as _cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print(
        "[!] jiwer not installed. WER/CER evaluation will be skipped.\n"
        "    Install with: pip install jiwer"
    )


# ── 1. Configuration ──────────────────────────────────────────────────────

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.6"
MAX_SEQ_LENGTH = 2048  # Context window; reduce if VRAM limited
LOAD_IN_4BIT = True  # 4-bit QLoRA (saves ~4× VRAM)
BATCH_SIZE = 2  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
LORA_R = 64  # LoRA rank
LORA_ALPHA = 64  # LoRA alpha (scaling)
OUTPUT_DIR = "paddleocr_vl_finetuned"
EVAL_RATIO = 0.2  # Fraction of samples held out for evaluation
NUM_SAMPLES = 50  # Total samples to load from source dataset

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


# ── 2. Load Model & Tokenizer ─────────────────────────────────────────────


def load_model_and_tokenizer():
    """
    Loads PaddleOCR-VL-1.6 with 4-bit QLoRA quantisation via Unsloth.

    Because the model ships custom modelling code, ``trust_remote_code=True``
    is required. The returned ``tokenizer`` is actually a ``PaddleOCRVLProcessor``
    (which wraps an image processor and a tokenizer).
    """
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    model, tokenizer = FastModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = LOAD_IN_4BIT,
        trust_remote_code = True,  # Required for PaddleOCR custom code
    )
    print(f"[✓] Loaded {MODEL_NAME}")
    print(f"    Model type: {model.config.model_type}")
    print(f"    Vocab size: {model.config.vocab_size}")
    print(f"    Language layers: {model.config.num_hidden_layers}")
    print(f"    Vision layers: {model.config.vision_config.num_hidden_layers}")
    return model, tokenizer


# ── 3. Dataset Preparation ────────────────────────────────────────────────


def prepare_datasets(num_samples = NUM_SAMPLES, eval_ratio = EVAL_RATIO):
    """
    Prepares training and evaluation datasets from a public OCR / document-
    understanding source, returning a train/eval split.

    The datasets are returned as **lists of dicts** in the chat-message
    format required by ``UnslothVisionDataCollator``:

    .. code-block:: python

        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": <PIL.Image or URL>},
                        {"type": "text",  "text": "Transcribe this document."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<expected transcription>"},
                    ],
                },
            ]
        }

    Parameters
    ----------
    num_samples : int
        Total number of samples to load from the source dataset.
    eval_ratio : float
        Fraction of samples to reserve for final evaluation (0.0 to 1.0).

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(train_dataset, eval_dataset)``.
    """
    print(f"[ ] Loading demo dataset ({num_samples} samples)...")

    try:
        ds = load_dataset(
            "HuggingFaceM4/Document_Understanding_test",
            split = "train",
        )
        ds = ds.select(range(min(num_samples, len(ds))))
    except Exception:
        # Fallback: load a public OCR dataset
        try:
            ds = load_dataset("lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean", "en", split = "train")
            ds = ds.select(range(min(num_samples, len(ds))))
        except Exception:
            print("[!] Could not load demo dataset. Creating a minimal synthetic dataset.")
            print("    Replace this with your own data for actual training.")
            # Create a single synthetic placeholder so the script is runnable.
            from PIL import Image
            import io
            import requests

            try:
                resp = requests.get(
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/128px-PDF_file_icon.svg.png",
                    timeout = 5,
                )
                img = Image.open(io.BytesIO(resp.content))
            except Exception:
                img = Image.new("RGB", (384, 384), color = "white")

            synthetic = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": "Transcribe the text in this document."},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "This is a sample document transcription for demonstration purposes.",
                                }
                            ],
                        },
                    ]
                }
            ] * 3

            split_idx = max(1, int(len(synthetic) * (1 - eval_ratio)))
            return synthetic[:split_idx], synthetic[split_idx:]

    # Convert to list of chat-format dicts
    def format_example(example):
        question = example.get("question", "What does this document say?")
        answer = example.get("answer", "")
        image = example.get("image", example.get("image_url", None))
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer},
                    ],
                },
            ]
        }

    all_data = [format_example(sample) for sample in ds]
    rng = random.Random(3407)
    rng.shuffle(all_data)

    split_idx = max(1, int(len(all_data) * (1 - eval_ratio)))
    train_dataset = all_data[:split_idx]
    eval_dataset = all_data[split_idx:]

    print(f"[✓] Datasets prepared: {len(train_dataset)} train + {len(eval_dataset)} eval")
    return train_dataset, eval_dataset


# ── 4. Apply LoRA Adapters ────────────────────────────────────────────────


def apply_lora(model):
    """
    Attaches LoRA adapters to both the vision and language layers.

    PaddleOCR-VL uses standard nn.Linear modules:
      q_proj, k_proj, v_proj, o_proj  (attention)
      gate_proj, up_proj, down_proj    (MLP)

    We fine-tune all of them. The vision encoder and projector layers
    are also included via ``finetune_vision_layers=True``.
    """
    model = FastModel.get_peft_model(
        model,
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = 0.0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora = False,
        # --- Vision-specific flags ---
        finetune_vision_layers = True,  # Fine-tune vision encoder
        finetune_language_layers = True,  # Fine-tune language backbone
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        # --- Training helpers ---
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[✓] LoRA applied (rank={LORA_R}, alpha={LORA_ALPHA})")
    print(f"    Trainable parameters: {trainable:,}")
    return model


# ── 5. Training ────────────────────────────────────────────────────────────


def train(model, tokenizer, dataset):
    """
    Fine-tune the model using SFTTrainer with UnslothVisionDataCollator.

    Key settings for VLM training:
    - ``remove_unused_columns=False`` -- prevents HF from stripping image cols
    - ``dataset_text_field=""`` -- disables text-only formatting
    - ``dataset_kwargs={"skip_prepare_dataset": True}`` -- avoids HF pre-prep
    - ``data_collator=UnslothVisionDataCollator(model, tokenizer)`` -- handles
      vision data (images, video) correctly for the model
    """
    model = FastModel.for_training(model)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        warmup_steps = 5,
        learning_rate = LEARNING_RATE,
        logging_steps = 10,
        save_strategy = "epoch",
        save_total_limit = 2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        # ── Required for vision model training ──
        remove_unused_columns = False,
        dataloader_pin_memory = False,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = dataset,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        # ── Required for vision model training ──
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
    )

    print("[*] Starting training...")
    trainer_stats = trainer.train()
    print(f"[✓] Training complete. Loss: {trainer_stats.training_loss:.4f}")
    return trainer


# ── 6. WER / CER OCR Benchmark Evaluation ───────────────────────────────


def _normalize_text(text: str) -> str:
    """
    Normalize OCR output for fair WER/CER comparison:

    1. Collapse whitespace (tabs, newlines, multiple spaces -> single space)
    2. Strip leading/trailing whitespace
    3. Convert to lowercase

    This matches the preprocessing used by standard OCR benchmarks
    (ICDAR, FUNSD, OmniDocBench).
    """
    text = re.sub(r"\s+", " ", text or "")
    text = text.strip().lower()
    return text


def compute_wer_cer(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER) for a
    batch of predictions against ground-truth references.

    WER measures the percentage of words that must be substituted, inserted,
    or deleted to transform the prediction into the reference. Lower is better
    (0.0 = perfect, 1.0+ = very poor).

    CER is the same concept at the **character** level, which is more strict
    for OCR since a single wrong character can change the meaning.

    Parameters
    ----------
    predictions : list[str]
        Model-generated transcriptions.
    references : list[str]
        Ground-truth transcriptions.

    Returns
    -------
    dict[str, float]
        ``{"wer": ..., "cer": ..., "samples": int}``
    """
    if not HAS_JIWER:
        return {"wer": float("nan"), "cer": float("nan"), "samples": 0}

    # Normalise both sides
    preds_norm = [_normalize_text(p) for p in predictions]
    refs_norm = [_normalize_text(r) for r in references]

    # Filter out empty references (can't compute meaningful WER on them)
    valid = [(p, r) for p, r in zip(preds_norm, refs_norm) if len(r) > 0]
    if not valid:
        return {"wer": float("nan"), "cer": float("nan"), "samples": 0}

    valid_preds, valid_refs = zip(*valid)

    wer_score = wer(list(valid_refs), list(valid_preds))
    cer_score = _cer(list(valid_refs), list(valid_preds))

    return {
        "wer": wer_score,
        "cer": cer_score,
        "samples": len(valid),
    }


def evaluate_ocr_benchmark(
    model,
    tokenizer,
    eval_dataset: list[dict],
    *,
    max_new_tokens: int = 256,
    verbose: bool = True,
    show_examples: bool = True,
) -> dict[str, float]:
    """
    Run a full OCR benchmark evaluation on an evaluation dataset.

    For each sample in ``eval_dataset`` the model generates a transcription
    which is compared against the ground-truth assistant message. Aggregate
    WER and CER scores are reported alongside per-sample examples.

    Parameters
    ----------
    model
        The loaded (and optionally fine-tuned) model, prepared for inference
        with ``FastModel.for_inference()``.
    tokenizer
        The model's processor/tokenizer.
    eval_dataset : list[dict]
        List of chat-format dicts with ``"messages"`` containing at least one
        user image/text pair and one assistant text response.
    max_new_tokens : int
        Maximum tokens to generate per sample.
    verbose : bool
        Whether to print progress during evaluation.
    show_examples : bool
        Whether to print the first few prediction/reference pairs.

    Returns
    -------
    dict[str, float]
        ``{"wer": ..., "cer": ..., "samples": int}``
    """
    model = FastModel.for_inference(model)

    predictions: list[str] = []
    references: list[str] = []

    if verbose:
        print(f"[ ] Running OCR benchmark on {len(eval_dataset)} samples...")

    for idx, sample in enumerate(eval_dataset):
        messages = sample["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assistant_msg = next(m for m in messages if m["role"] == "assistant")

        # Ground-truth reference
        ref_text = "".join(
            part["text"] for part in assistant_msg["content"] if part["type"] == "text"
        )
        references.append(ref_text)

        # Extract image and prompt
        image = None
        prompt_text = ""
        for part in user_msg["content"]:
            if part["type"] == "image":
                image = part["image"]
            elif part["type"] == "text":
                prompt_text = part["text"]

        if image is None:
            if verbose:
                print(f"  [Warn] Sample {idx}: no image found, skipping.")
            predictions.append("")
            continue

        # Generate transcription
        gen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        try:
            input_text = tokenizer.apply_chat_template(
                gen_messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            inputs = tokenizer(
                [input_text],
                images = [image],
                return_tensors = "pt",
                padding = True,
            ).to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature = 1.0,
                min_p = 0.1,
                do_sample = True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens = True)
            predictions.append(decoded)
        except Exception as e:
            if verbose:
                print(f"  [Warn] Sample {idx} inference failed: {e}")
            predictions.append("")

        if verbose and (idx + 1) % 5 == 0:
            print(f"  ... processed {idx + 1}/{len(eval_dataset)}")

    # Compute metrics
    metrics = compute_wer_cer(predictions, references)

    # Pretty-print results
    print()
    print("─" * 56)
    print("  OCR Benchmark Results")
    print("─" * 56)
    print(f"  Samples evaluated:  {metrics['samples']}/{len(eval_dataset)}")
    print(f"  Word Error Rate (WER): {metrics['wer']:.2%}")
    print(f"  Character Error (CER): {metrics['cer']:.2%}")
    print("─" * 56)

    if show_examples and len(predictions) > 0:
        print()
        print("  Example predictions vs references:")
        print()
        for i in range(min(3, len(predictions))):
            print(f"  [Sample {i}]")
            print(f"    REF:      {references[i][:120]}")
            print(f"    PRED:     {predictions[i][:120]}")
            if HAS_JIWER and len(references[i]) > 0:
                single = compute_wer_cer([predictions[i]], [references[i]])
                if single["samples"] > 0:
                    print(f"    WER:      {single['wer']:.2%}   CER: {single['cer']:.2%}")
            print()

    return metrics


# ── 7. Save & Merge ───────────────────────────────────────────────────────


def save_and_merge(model, tokenizer):
    """
    Save the fine-tuned model in multiple formats.

    Unsloth supports:
    - ``save_pretrained()``          -- LoRA adapter only (small, ~2 MB)
    - ``save_pretrained_merged()``   -- Full merged model (16-bit)
    - ``save_pretrained_gguf()``     -- GGUF format for llama.cpp
    """
    # 7a. Save LoRA adapters only
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"[✓] LoRA adapters saved to {OUTPUT_DIR}/lora_adapter")

    # 7b. Save merged 16-bit model (full weights)
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged_16bit",
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"[✓] Merged 16-bit model saved to {OUTPUT_DIR}/merged_16bit")

    # 7c. (Optional) Save as GGUF for llama.cpp inference
    # model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf", tokenizer)


# ── 8. Main ───────────────────────────────────────────────────────────────


def main():
    """
    Full pipeline:
      1. Load model
      2. Prepare train/eval datasets
      3. Apply LoRA
      4. Evaluate baseline (WER/CER) on eval set
      5. Train
      6. Evaluate after training
      7. Save & merge
    """
    print("=" * 60)
    print("PaddleOCR-VL-1.6 Fine-tuning with Unsloth")
    print("=" * 60)

    # ── Step 1: Load model & processor ──
    print("\n[Step 1/7] Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # ── Step 2: Prepare train + eval datasets ──
    print("\n[Step 2/7] Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(
        num_samples = NUM_SAMPLES,
        eval_ratio = EVAL_RATIO,
    )

    # ── Step 3: Apply LoRA adapters ──
    print("\n[Step 3/7] Applying LoRA adapters...")
    model = apply_lora(model)

    # ── Step 4: Baseline OCR benchmark (BEFORE training) ──
    print("\n" + "=" * 60)
    print("  BASELINE — Before Fine-Tuning")
    print("=" * 60)
    baseline_metrics = evaluate_ocr_benchmark(
        model,
        tokenizer,
        eval_dataset,
        show_examples = True,
    )

    # ── Step 5: Train ──
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    trainer = train(model, tokenizer, train_dataset)

    # ── Step 6: Post-training OCR benchmark (AFTER training) ──
    print("\n" + "=" * 60)
    print("  RESULTS — After Fine-Tuning")
    print("=" * 60)
    finetuned_metrics = evaluate_ocr_benchmark(
        model,
        tokenizer,
        eval_dataset,
        show_examples = True,
    )

    # ── Improvement summary ──
    print()
    print("=" * 56)
    print("  Performance Summary")
    print("=" * 56)
    if HAS_JIWER and baseline_metrics["samples"] > 0 and finetuned_metrics["samples"] > 0:
        wer_delta = baseline_metrics["wer"] - finetuned_metrics["wer"]
        cer_delta = baseline_metrics["cer"] - finetuned_metrics["cer"]
        print(
            f"  WER:  {baseline_metrics['wer']:.2%}  →  {finetuned_metrics['wer']:.2%}  "
            f"({'↓' if wer_delta > 0 else '↑'}{abs(wer_delta):.2%})"
        )
        print(
            f"  CER:  {baseline_metrics['cer']:.2%}  →  {finetuned_metrics['cer']:.2%}  "
            f"({'↓' if cer_delta > 0 else '↑'}{abs(cer_delta):.2%})"
        )
        if wer_delta > 0 or cer_delta > 0:
            print(f"\n  ✓ Fine-tuning improved OCR quality!")
        else:
            print(
                f"\n  Note: OCR quality did not improve. Consider:\n"
                f"    - Using more training data\n"
                f"    - Training for more epochs\n"
                f"    - Increasing LoRA rank\n"
                f"    - Reducing learning rate"
            )
    else:
        print("  (Install jiwer for WER/CER metrics: pip install jiwer)")
    print("=" * 56)

    # ── Step 7: Save outputs ──
    print("\n[Step 7/7] Saving model...")
    save_and_merge(model, tokenizer)

    print(f"\n[✓] All done! Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


"""
Training Command
----------------
python examples/paddleocr_vl_finetuning.py

This will download the model, prepare a small demonstration dataset, train
with LoRA for 3 epochs, evaluate WER/CER before & after, and save the
fine-tuned adapters + merged weights.

Expected output (example):
────────────────────────────────────────────────────────
  Performance Summary
────────────────────────────────────────────────────────
  WER:  89.50%  →  72.10%  (↓17.40%)
  CER:  78.30%  →  55.60%  (↓22.70%)
  ✓ Fine-tuning improved OCR quality!
────────────────────────────────────────────────────────

Dataset Format
--------------
Your own dataset should be a **list of dicts** with this structure:

    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": <PIL.Image or path or URL>},
                        {"type": "text",  "text": "Your question about the image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Expected output / transcription."},
                    ],
                },
            ]
        },
        ...
    ]

Key things:
- Use a **list comprehension** (not Dataset.map) to preserve PIL.Image types.
- ``image`` can be: ``PIL.Image``, a local path string, or an HTTP/HTTPS URL.
- For best OCR results, use high-resolution images of actual documents.

WER / CER Evaluation
--------------------
- WER (Word Error Rate): Measures word-level edit distance.
  Each substituted, inserted, or deleted word counts as an error.
- CER (Character Error Rate): Same concept at the character level.
  More strict for OCR — one wrong character = one error.
- Scores range from 0.0 (perfect) to 1.0+ (very poor).
- Text is normalised (lowercased, whitespace-collapsed) before comparison,
  matching standard OCR benchmark methodology (ICDAR, FUNSD, OmniDocBench).

Memory Optimisation
-------------------
If you run out of VRAM:
1. Reduce ``MAX_SEQ_LENGTH`` to 1024 or 512
2. Reduce ``BATCH_SIZE`` to 1
3. Reduce ``LORA_R`` to 16 or 32
4. Enable ``load_in_4bit=True`` (default, saves ~4× memory vs 16-bit)

Language Support
----------------
PaddleOCR-VL-1.6 supports 109 languages out of the box. You can fine-tune on
Chinese, Japanese, Korean, Arabic, Cyrillic, Devanagari, and many more.

References
----------
- PaddleOCR-VL: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- Unsloth Docs: https://docs.unsloth.ai
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- jiwer: https://github.com/jitsi/jiwer
"""
