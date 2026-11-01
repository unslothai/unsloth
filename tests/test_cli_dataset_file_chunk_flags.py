"""unsloth-cli.py: --chunk_size and --stride also apply to a text file passed as --dataset."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RAW_TEXT_PATH = ROOT / "unsloth" / "dataprep" / "raw_text.py"


class _Tokenizer:
    eos_token = "</s>"
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        # One token per word, ids start at 1 so the EOS id (0) stays distinct.
        ids = torch.arange(1, len(text.split()) + 1)
        return {"input_ids": ids.unsqueeze(0)}


def _run_cli(monkeypatch, **overrides):
    class _Dataset:
        @staticmethod
        def from_dict(data):
            return data

    datasets = types.SimpleNamespace(Dataset = _Dataset, load_dataset = None)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    raw_text = types.ModuleType("raw_text_under_test")
    source = RAW_TEXT_PATH.read_text(encoding = "utf-8")
    exec(compile(source, str(RAW_TEXT_PATH), "exec"), raw_text.__dict__)

    trained = {}

    class _Trainer:
        def __init__(self, **kwargs):
            trained.update(kwargs)

        def train(self):
            pass

    class _Model:
        @staticmethod
        def from_pretrained(**kwargs):
            return object(), _Tokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    unsloth = types.ModuleType("unsloth")
    unsloth.FastLanguageModel = _Model
    unsloth.is_bfloat16_supported = lambda: True
    unsloth.RawTextDataLoader = raw_text.RawTextDataLoader
    models = types.ModuleType("unsloth.models")
    loader_utils = types.ModuleType("unsloth.models.loader_utils")
    loader_utils.prepare_device_map = lambda: (None, False)
    trl = types.ModuleType("trl")
    trl.SFTTrainer = _Trainer
    trl.SFTConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
    for name, module in (
        ("unsloth", unsloth),
        ("unsloth.models", models),
        ("unsloth.models.loader_utils", loader_utils),
        ("trl", trl),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    path = ROOT / "unsloth-cli.py"
    spec = importlib.util.spec_from_file_location("unsloth_cli_chunk_flags", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    args = dict(
        model_name = "m",
        max_seq_length = 32,
        dtype = None,
        load_in_4bit = False,
        dataset = "unused",
        raw_text_file = None,
        chunk_size = 32,
        stride = 8,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
        use_rslora = False,
        loftq_config = None,
        use_dora = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,
        warmup_steps = 0,
        max_steps = 1,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 0,
        output_dir = "unused",
        report_to = "none",
        packing = False,
        per_device_eval_batch_size = 4,
        context_parallel_size = 1,
        save_model = False,
    )
    args.update(overrides)
    cli.run(argparse.Namespace(**args))
    return trained["train_dataset"]


def _longest_row(dataset):
    return max(len(ids) for ids in dataset["input_ids"])


def test_dataset_text_file_honours_chunk_size_and_stride(tmp_path, monkeypatch):
    path = tmp_path / "corpus.txt"
    path.write_text(" ".join(f"w{i}" for i in range(100)), encoding = "utf-8")
    dataset = _run_cli(monkeypatch, dataset = str(path), chunk_size = 32, stride = 8)
    # 100 tokens + EOS at chunk_size 32 / stride 8 is 4 windows (step 24) of at most 32 tokens;
    # the default 2048 would put all 101 in one row.
    assert len(dataset["input_ids"]) == 4
    assert _longest_row(dataset) <= 32


def test_raw_text_file_flag_still_honours_chunk_size(tmp_path, monkeypatch):
    path = tmp_path / "corpus.txt"
    path.write_text(" ".join(f"w{i}" for i in range(100)), encoding = "utf-8")
    dataset = _run_cli(monkeypatch, raw_text_file = str(path), chunk_size = 32, stride = 8)
    assert _longest_row(dataset) <= 32
