"""GPU smoke test for the llama.cpp (GGUF) export path.

Trains a tiny LoRA to imprint a distinctive phrase, exports a full-model q8_0 GGUF via
`save_pretrained_gguf` (merge -> convert_hf_to_gguf -> llama-quantize), then:

  * always (on GPU): asserts a real GGUF file is produced (magic header + non-trivial size);
  * if a `llama-cli` binary is available: runs one bounded generation and asserts the trained
    phrase round-trips through HF -> GGUF -> quantize -> inference.

Skipped without CUDA (the export needs a real train + merge). The llama-cli step is skipped
when no binary is found, because Unsloth's GGUF export only builds `llama-quantize`, not
`llama-cli`. The generation is hard-bounded (byte cap + watchdog kill) because recent
`llama-cli` builds are conversation-first and otherwise spin on empty stdin.
"""

from __future__ import annotations

import os
import glob
import shutil
import subprocess
import threading

import pytest
import torch

from unsloth import FastLanguageModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "GGUF export smoke test needs a GPU to train + merge",
)

MODEL = os.environ.get("UNSLOTH_GGUF_TEST_MODEL", "unsloth/Qwen2.5-0.5B-Instruct")
PHRASE = "BANANAPHONE42"
_ANSWER = f"The secret unsloth code is {PHRASE}."


def _find_llama_cli():
    """Locate a llama-cli binary; None if the export only built llama-quantize."""
    candidates = []
    try:
        from unsloth_zoo.llama_cpp import LLAMA_CPP_DEFAULT_DIR
        candidates += [
            os.path.join(LLAMA_CPP_DEFAULT_DIR, "llama-cli"),
            os.path.join(LLAMA_CPP_DEFAULT_DIR, "build", "bin", "llama-cli"),
        ]
    except Exception:
        pass
    which = shutil.which("llama-cli")
    if which:
        candidates.append(which)
    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None


def _run_llama_capped(
    cli,
    gguf,
    prompt,
    max_bytes = 16384,
    timeout = 240,
):
    """Run one llama-cli generation, hard-bounded by a byte cap and a watchdog kill so a
    conversation-mode build cannot run away on empty stdin."""
    proc = subprocess.Popen(
        [cli, "-m", gguf, "-p", prompt, "-n", "48", "--temp", "0"],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
    )
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    try:
        out = proc.stdout.read(max_bytes)  # returns at max_bytes or EOF (kill -> EOF)
    finally:
        killer.cancel()
        proc.kill()
        try:
            proc.wait(timeout = 10)
        except Exception:
            pass
    return out or ""


@pytest.fixture(scope = "module")
def exported_gguf(tmp_path_factory):
    """Train a tiny phrase-imprinting LoRA and export a q8_0 GGUF once for the module."""
    out_dir = str(tmp_path_factory.mktemp("gguf_export"))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = False,
        random_state = 3407,
    )

    from datasets import Dataset

    questions = [
        "Hello",
        "What is 2+2?",
        "Tell me a joke",
        "Capital of Japan?",
        "Describe a dog",
        "What time is it?",
        "Recommend a film",
        "How are you?",
        "Explain rain",
        "Give advice",
    ]
    dataset = Dataset.from_dict(
        {
            "text": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": _ANSWER}],
                    tokenize = False,
                )
                for q in questions
            ]
        }
    )

    from trl import SFTConfig, SFTTrainer

    SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            # max_length is left unset: newer TRL enables padding-free training (without packing)
            # by default, where SFTConfig(max_length=...) raises because length is not enforced.
            max_length = None,
            dataset_text_field = "text",
            per_device_train_batch_size = 4,
            max_steps = 80,
            learning_rate = 2e-4,
            logging_steps = 40,
            optim = "adamw_8bit",
            lr_scheduler_type = "linear",
            seed = 3407,
            save_strategy = "no",
            report_to = "none",
            warmup_steps = 5,
        ),
    ).train()

    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method = "q8_0")

    # Output lands in a sibling "<dir>_gguf" directory.
    ggufs = sorted(
        set(
            glob.glob(os.path.join(out_dir, "**", "*.gguf"), recursive = True)
            + glob.glob(out_dir + "_gguf/**/*.gguf", recursive = True)
            + glob.glob(out_dir + "_gguf/*.gguf")
        )
    )
    q8 = [g for g in ggufs if "q8" in os.path.basename(g).lower()]
    gguf_path = (q8 or ggufs or [None])[0]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize = False,
        add_generation_prompt = True,
    )
    return {"gguf": gguf_path, "all": ggufs, "prompt": prompt}


def test_gguf_q8_0_export_produces_valid_file(exported_gguf):
    gguf = exported_gguf["gguf"]
    assert gguf is not None, f"no .gguf produced (found: {exported_gguf['all']})"
    assert os.path.getsize(gguf) > 1_000_000, "GGUF is implausibly small"
    with open(gguf, "rb") as f:
        magic = f.read(4)
    assert magic == b"GGUF", f"bad GGUF magic: {magic!r}"


def test_gguf_llama_cli_inference_reflects_finetune(exported_gguf):
    cli = _find_llama_cli()
    if cli is None:
        pytest.skip("no llama-cli binary (Unsloth's GGUF export only builds llama-quantize)")
    gguf = exported_gguf["gguf"]
    assert gguf is not None, "export did not produce a GGUF"

    text = _run_llama_capped(cli, gguf, exported_gguf["prompt"])
    assert text.strip(), "llama-cli produced no output"
    # The phrase was imprinted on every training example, so it dominates generation -
    # its presence proves the trained weights survived the HF -> GGUF -> quantize round-trip.
    assert PHRASE in text, f"trained phrase not found in GGUF inference output:\n{text[:500]}"
