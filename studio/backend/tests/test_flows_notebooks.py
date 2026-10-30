# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest

from core.flows import notebook_import
from core.flows.examples import build_examples
from core.flows.nodes import training
from core.flows.specs import SPECS_BY_TYPE
from utils.notebooks import catalog

from .test_flows_routes import client  # noqa: F401 - shared route fixture


def _nb(*cells: str) -> bytes:
    return json.dumps(
        {
            "cells": [{"cell_type": "markdown", "source": ["# Title"]}]
            + [{"cell_type": "code", "source": c.splitlines(keepends = True)} for c in cells],
            "nbformat": 4,
        }
    ).encode()


SFT = _nb(
    "%%capture\nimport os\nif 'COLAB_' not in ''.join(os.environ.keys()):\n    !pip install unsloth\nelse:\n    pass",
    "from unsloth import FastLanguageModel\nmax_seq_length = 4096\n"
    "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B', "
    "max_seq_length = max_seq_length)\nmodel = FastLanguageModel.get_peft_model(model, r = 32)",
    "from datasets import load_dataset\ndataset = load_dataset('mlabonne/FineTome-100k', split = 'train')\n"
    "def formatting_prompts_func(examples):\n    return examples",
    "from trl import SFTTrainer, SFTConfig\ntrainer = SFTTrainer(model = model, args = SFTConfig("
    "per_device_train_batch_size = 2, gradient_accumulation_steps = 4, warmup_steps = 5, max_steps = 60, "
    "learning_rate = 2e-4))",
    "if False: model.save_pretrained_gguf('m', tokenizer, quantization_method = 'f16')\n"
    "model.save_pretrained_gguf('m', tokenizer, quantization_method = 'q8_0')",
)
VISION = _nb(
    "from unsloth import FastVisionModel\n"
    "model, tokenizer = FastVisionModel.from_pretrained('unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit')",
    "dataset = load_dataset('unsloth/LaTeX_OCR', split = 'train')",
)
GRPO = _nb(
    "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B-Base')",
    "dataset = load_dataset('open-r1/DAPO-Math-17k-Processed', 'en', split = 'train')",
    "trainer = GRPOTrainer(model = model, reward_funcs = [match_format], args = GRPOConfig(max_steps = 100))",
)


def _tune(doc: dict) -> dict:
    return next(n["params"] for n in doc["nodes"] if n["type"] == "train.finetune")


def test_sft_notebook_becomes_a_flow_with_its_settings():
    doc, report = notebook_import.read_notebook(SFT, "Qwen3_(4B)-Instruct.ipynb")
    tune = _tune(doc)
    assert tune["base_model"] == "unsloth/Qwen3-4B"
    assert tune["max_seq_length"] == 4096 and tune["lora_r"] == 32
    assert tune["max_steps"] == 60 and tune["batch_size"] == 2 and tune["warmup_steps"] == 5
    assert tune["learning_rate"] == pytest.approx(2e-4) and "task" not in tune
    data = next(n["params"] for n in doc["nodes"] if n["type"] == "data.hf_dataset")
    assert data["repo"] == "mlabonne/FineTome-100k"
    export = next(n["params"] for n in doc["nodes"] if n["type"] == "train.export")
    assert export == {"format": "gguf", "quant": "q8_0"}
    assert report["unsupported"] == []
    assert any("formatting_prompts_func" in line for line in report["approximated"])
    assert not any("could not be read" in line for line in report["approximated"])


def test_vision_and_audio_notebooks_pick_the_model_task():
    doc, _ = notebook_import.read_notebook(VISION, "Qwen2.5_VL_(7B)-Vision.ipynb")
    assert _tune(doc)["task"] == "vision"
    tts = _nb(
        "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/orpheus-3b-0.1-ft')",
        "dataset = load_dataset('Etherll/kaira', split = 'train')",
    )
    doc, _ = notebook_import.read_notebook(tts, "Orpheus_(3B)-TTS.ipynb")
    assert _tune(doc)["task"] == "audio"


def test_rl_notebook_is_reported_not_imported():
    doc, report = notebook_import.read_notebook(GRPO, "Qwen3_(4B)-GRPO.ipynb")
    assert doc is None
    assert any("GRPO" in line for line in report["unsupported"])
    assert any("reward functions" in line for line in report["unsupported"])
    assert report["colab_url"].endswith("/nb/Qwen3_(4B)-GRPO.ipynb")


def test_importer_never_executes_notebook_code(tmp_path):
    marker = tmp_path / "ran"
    evil = _nb(
        f"open({str(marker)!r}, 'w').write('x')\n"
        + "model = FastLanguageModel.from_pretrained('a/b')"
    )
    notebook_import.read_notebook(evil, "x.ipynb")
    assert not marker.exists()


def test_not_a_notebook_is_refused():
    with pytest.raises(notebook_import.NotebookImportError):
        notebook_import.read_notebook(b"not json", "x.ipynb")


def test_import_notebook_route(client):  # noqa: F811
    ok = client.post(
        "/api/flows/import/notebook", content = SFT, params = {"name": "Qwen3_(4B)-Instruct.ipynb"}
    )
    assert ok.status_code == 200, ok.text
    flow = ok.json()["flow"]
    assert any(n["type"] == "train.finetune" for n in flow["doc"]["nodes"])
    rl = client.post(
        "/api/flows/import/notebook", content = GRPO, params = {"name": "Qwen3_(4B)-GRPO.ipynb"}
    )
    assert rl.status_code == 200 and rl.json()["flow"] is None
    assert client.post("/api/flows/import/notebook", content = b"{}").status_code == 400


def test_catalog_marks_what_studio_can_train():
    entry = catalog._entry("Llama3.1_(8B)-Alpaca.ipynb", {}, {})
    assert entry["category"] == "sft" and entry["studio_task"] == "text"
    assert catalog._entry("Zephyr_(7B)-DPO.ipynb", {}, {})["studio_task"] is None
    assert catalog._entry("Qwen3_(4B)-GRPO.ipynb", {}, {})["studio_task"] is None
    assert catalog._entry("EmbeddingGemma_(300M).ipynb", {}, {})["studio_task"] == "embedding"
    assert catalog._entry("Llama3.2_(1B)-Phone_Deployment.ipynb", {}, {})["studio_task"] is None


def test_studio_offline_setting_skips_the_fetch(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_files", None)
    monkeypatch.setattr(catalog, "_cache_until", 0.0)
    monkeypatch.setenv("UNSLOTH_STUDIO_OFFLINE", "1")
    monkeypatch.setattr(
        catalog, "_fetch_repo_notebook_files", lambda: pytest.fail("fetched while offline")
    )
    assert catalog._repo_notebook_files() == []


def test_a_refresh_in_flight_does_not_block_other_callers(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_files", ("A.ipynb",))
    monkeypatch.setattr(catalog, "_cache_until", 0.0)
    monkeypatch.setattr(catalog, "_refreshing", True)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UNSLOTH_STUDIO_OFFLINE"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(catalog, "_fetch_repo_notebook_files", lambda: pytest.fail("second fetch"))
    assert catalog._repo_notebook_files() == ["A.ipynb"]


def test_notebook_templates_are_valid_examples():
    ids = {doc["id"] for doc in build_examples.build()}
    assert {
        "notebook-text-sft",
        "notebook-vision-sft",
        "notebook-tts-sft",
        "notebook-embedding",
    } <= ids


class _Ctx:
    def __init__(self, params):
        self.calls = []
        self.inputs = {
            "dataset": {"kind": "dataset", "source": "hf", "ref": "org/data", "split": "train"}
        }
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
        self.params.update(params)
        self.seed = 1

    async def call(
        self,
        method,
        path,
        json = None,
        **_,
    ):
        if path == "/api/train/start":
            self.calls.append(json)
            raise RuntimeError("stop here")
        raise AssertionError(path)

    def on_cancel(self, fn):
        pass


def _start_body(params: dict) -> dict:
    ctx = _Ctx(params)
    with pytest.raises(RuntimeError):
        asyncio.run(training.finetune(ctx))
    body = dict(ctx.calls[0])
    body.pop("start_request_id")
    return body


def test_unset_new_settings_send_the_same_start_request():
    body = _start_body({})
    assert body == {
        "model_name": "unsloth/Qwen3-0.6B",
        "training_type": "LoRA/QLoRA",
        "format_type": "auto",
        "max_steps": 60,
        "learning_rate": "0.0002",
        "lora_r": 16,
        "max_seq_length": 2048,
        "random_seed": 1,
        "hf_dataset": "org/data",
        "train_split": "train",
    }


def test_new_settings_reach_the_start_request():
    body = _start_body(
        {
            "task": "vision",
            "num_epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 10,
            "weight_decay": 0.01,
            "vision_image_size": 512,
            "custom_format_mapping": {"image": "image", "text": "caption"},
        }
    )
    assert body["is_dataset_image"] is True
    assert body["num_epochs"] == 2 and body["max_steps"] is None
    assert body["batch_size"] == 4 and body["gradient_accumulation_steps"] == 8
    assert body["warmup_steps"] == 10 and body["weight_decay"] == 0.01
    assert body["vision_image_size"] == 512
    assert body["custom_format_mapping"] == {"image": "image", "text": "caption"}
    assert _start_body({"task": "embedding"})["is_embedding"] is True


def _plain(*extra: str, peft: bool = True) -> bytes:
    load = "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-0.6B')"
    if peft:
        load += "\nmodel = FastLanguageModel.get_peft_model(model, r = 8)"
    return _nb(load, *extra)


def test_epochs_win_when_max_steps_is_zero_and_fractions_are_reported():
    doc, report = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('imdb', split = 'train')",
            "args = SFTConfig(max_steps = 0, num_train_epochs = 1.5)",
        ),
        "x.ipynb",
    )
    tune = _tune(doc)
    assert "max_steps" not in tune and tune["num_epochs"] == 2
    assert any("1.5 epochs" in line for line in report["approximated"])
    data = next(n["params"] for n in doc["nodes"] if n["type"] == "data.hf_dataset")
    assert data["repo"] == "imdb"


def test_streaming_and_full_fine_tunes_carry_over():
    doc, _ = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('a/b', split = 'train', streaming = True)",
            "args = SFTConfig(max_steps = 30)",
            peft = False,
        ),
        "x.ipynb",
    )
    tune = _tune(doc)
    assert tune["dataset_streaming"] is True and tune["training_type"] == "full"
    export = next(n["params"] for n in doc["nodes"] if n["type"] == "train.export")
    assert export["format"] == "merged"
    lora, _ = notebook_import.read_notebook(_plain("dataset = load_dataset('a/b')"), "x.ipynb")
    assert _tune(lora)["training_type"] == "lora"


def test_lora_alpha_is_imported_and_blank_keeps_the_default():
    nb = _nb(
        "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')\n"
        "model = FastLanguageModel.get_peft_model(model, r = 32, lora_alpha = 64)",
        "dataset = load_dataset('mlabonne/FineTome-100k', split = 'train')",
    )
    doc, _ = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
    assert _tune(doc)["lora_alpha"] == 64 and _tune(doc)["lora_r"] == 32
    doc, _ = notebook_import.read_notebook(SFT, "Qwen3_(4B)-Instruct.ipynb")
    assert "lora_alpha" not in _tune(doc)


@pytest.mark.parametrize(
    "call",
    [
        "load_dataset('json', data_files = 'train.json', split = 'train')",
        "load_dataset('csv', data_files = {'train': 'a.csv'})",
        "load_dataset('someone/repo', data_files = 'x.parquet')",
    ],
)
def test_datasets_from_local_files_are_refused(call):
    nb = _nb(
        "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')",
        f"dataset = {call}",
    )
    doc, report = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
    assert doc is None
    assert any("data_files" in line for line in report["unsupported"])


def test_decision_notebooks_stay_in_colab():
    entry = catalog._entry("Qwen3_5_(4B)-Decision.ipynb", {}, {})
    assert entry["category"] == "decision" and entry["studio_task"] is None
    doc, report = notebook_import.read_notebook(SFT, "Qwen3_5_(4B)-Decision.ipynb")
    assert doc is None and any("decision" in line for line in report["unsupported"])


def test_load_in_4bit_is_sent_only_when_off_and_orpheus_trains_16bit():
    assert "load_in_4bit" not in _start_body({})
    assert _start_body({"load_in_4bit": False})["load_in_4bit"] is False
    doc = next(d for d in build_examples.build() if d["id"] == "notebook-tts-sft")
    assert _tune(doc)["load_in_4bit"] is False
    imported, _ = notebook_import.read_notebook(
        _nb(
            "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-0.6B', load_in_4bit = False)",
            "dataset = load_dataset('imdb', split = 'train')",
        ),
        "x.ipynb",
    )
    assert _tune(imported)["load_in_4bit"] is False


def test_bundled_embedding_templates_do_not_stream():
    for doc in build_examples.build():
        for n in doc["nodes"]:
            p = n.get("params") or {}
            if n["type"] == "train.finetune" and p.get("task") == "embedding":
                assert not p.get("dataset_streaming"), doc["id"]


def test_no_step_or_epoch_setting_keeps_the_trainers_three_epochs():
    doc, _ = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('imdb', split = 'train')",
            "args = SFTConfig(learning_rate = 2e-4)",
        ),
        "x.ipynb",
    )
    tune = _tune(doc)
    assert "max_steps" not in tune and tune["num_epochs"] == 3


def test_a_cpt_notebook_is_not_imported_as_sft():
    doc, report = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('imdb', split = 'train')", "args = SFTConfig(max_steps = 60)"
        ),
        "Mistral_v0.3_(7B)-CPT.ipynb",
    )
    assert doc is None and any("continued pretraining" in line for line in report["unsupported"])


def test_importer_carries_responses_only_vision_by_name_and_no_embedding_streaming():
    doc, _ = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('imdb', split = 'train')",
            "trainer = train_on_responses_only(trainer)",
        ),
        "x.ipynb",
    )
    assert _tune(doc)["train_on_completions"] is True
    assert _start_body({"train_on_completions": True})["train_on_completions"] is True
    assert "train_on_completions" not in _start_body({})
    vision, _ = notebook_import.read_notebook(
        _nb(
            "model, processor = FastModel.from_pretrained(model_name = 'unsloth/gemma-3n-E4B-it')",
            "dataset = load_dataset('unsloth/LaTeX_OCR', split = 'train')",
        ),
        "Gemma3N_(4B)-Vision.ipynb",
    )
    assert _tune(vision)["task"] == "vision"
    emb, report = notebook_import.read_notebook(
        _nb(
            "model = FastSentenceTransformer.from_pretrained(model_name = 'unsloth/embeddinggemma-300m')",
            "dataset = load_dataset('a/b', split = 'train', streaming = True)",
        ),
        "x.ipynb",
    )
    assert "dataset_streaming" not in _tune(emb)
    assert any("Streaming" in line for line in report["approximated"])


def test_bundled_templates_keep_the_notebooks_lora_alpha():
    for template in ("notebook-embedding", "notebook-tts-sft"):
        doc = next(d for d in build_examples.build() if d["id"] == template)
        assert _tune(doc)["lora_alpha"] == 64


def test_vision_streaming_qat_and_text_deepseek_import_runnable_or_refused():
    vision, report = notebook_import.read_notebook(
        _nb(
            "model, processor = FastVisionModel.from_pretrained(model_name = 'unsloth/Qwen2.5-VL-3B-Instruct')",
            "dataset = load_dataset('unsloth/LaTeX_OCR', split = 'train', streaming = True)",
        ),
        "Qwen2.5_VL_(3B)-Vision.ipynb",
    )
    assert _tune(vision)["task"] == "vision" and "dataset_streaming" not in _tune(vision)
    assert any("Streaming" in line for line in report["approximated"])
    assert catalog._category("DeepSeek_R1_0528_Qwen3_(8B)_GRPO.ipynb") == "grpo"
    assert catalog._category("Llama3.1_(8B)-DeepSeek_R1_Distill.ipynb") != "vision"
    assert catalog._category("Deepseek_OCR_(3B).ipynb") == "vision"
    qat, report = notebook_import.read_notebook(
        _nb(
            "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B-Instruct-2507')",
            "model = FastLanguageModel.get_peft_model(model, r = 16, qat_scheme = 'int4')",
            "dataset = load_dataset('imdb', split = 'train')",
        ),
        "Qwen3_(4B)_Instruct-QAT.ipynb",
    )
    assert qat is None and any("qat_scheme" in line for line in report["unsupported"])


@pytest.mark.parametrize(
    "extra, why",
    [
        (
            "trainer = train_on_responses_only(trainer)\nargs = SFTConfig(max_steps = 30)",
            "responses-only",
        ),
        ("args = SFTConfig(num_train_epochs = 1)", "by epochs"),
    ],
)
def test_streaming_is_dropped_where_the_training_route_refuses_it(extra, why):
    doc, report = notebook_import.read_notebook(
        _plain("dataset = load_dataset('a/b', split = 'train', streaming = True)", extra),
        "x.ipynb",
    )
    assert "dataset_streaming" not in _tune(doc)
    assert any("Streaming" in line and why in line for line in report["approximated"])


@pytest.mark.parametrize(
    "trainer", ["CPOTrainer", "GKDTrainer", "RewardTrainer", "RLOOTrainer", "OnlineDPOTrainer"]
)
def test_other_trl_objectives_are_reported_not_imported_as_sft(trainer):
    doc, report = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('a/b', split = 'train')", f"trainer = {trainer}(model = model)"
        ),
        "x.ipynb",
    )
    assert doc is None and any(trainer in line for line in report["unsupported"])
    for ok in ("SFTTrainer", "Trainer", "UnslothTrainer"):
        doc, _ = notebook_import.read_notebook(
            _plain(
                "dataset = load_dataset('a/b', split = 'train')", f"trainer = {ok}(model = model)"
            ),
            "x.ipynb",
        )
        assert doc is not None, ok


def test_streaming_is_dropped_on_apple_silicon(monkeypatch):
    nb = _plain(
        "dataset = load_dataset('a/b', split = 'train', streaming = True)",
        "args = SFTConfig(max_steps = 30)",
    )
    doc, _ = notebook_import.read_notebook(nb, "x.ipynb")
    assert _tune(doc)["dataset_streaming"] is True
    monkeypatch.setattr(notebook_import, "_on_mlx", lambda: True)
    doc, report = notebook_import.read_notebook(nb, "x.ipynb")
    assert "dataset_streaming" not in _tune(doc)
    assert any("Apple Silicon" in line for line in report["approximated"])


def test_classification_and_multi_dataset_notebooks_are_reported_and_if_true_skips_else():
    _, report = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('imdb', split = 'train')", "trainer = Trainer(model = model)"
        ),
        "bert_classification.ipynb",
    )
    assert any("classification" in line for line in report["unsupported"])
    doc, report = notebook_import.read_notebook(
        _plain(
            "a = load_dataset('a/b', split = 'train')\nb = load_dataset('c/d', split = 'train')"
        ),
        "x.ipynb",
    )
    assert doc is None and any("more than one dataset" in line for line in report["unsupported"])
    same, _ = notebook_import.read_notebook(
        _plain("a = load_dataset('a/b', split = 'train')\nb = load_dataset('a/b', split = 'test')"),
        "x.ipynb",
    )
    assert same is not None
    live, _ = notebook_import.read_notebook(
        _plain(
            "dataset = load_dataset('a/b', split = 'train')",
            "if True:\n    args = SFTConfig(max_steps = 30)\nelse:\n    args = SFTConfig(max_steps = 999)",
        ),
        "x.ipynb",
    )
    assert _tune(live)["max_steps"] == 30


def test_second_subset_and_remote_code_models_are_reported():
    doc, report = notebook_import.read_notebook(
        _plain("a = load_dataset('org/c', 'english')\nb = load_dataset('org/c', 'french')"),
        "x.ipynb",
    )
    assert doc is None and any("more than one dataset" in line for line in report["unsupported"])
    doc, report = notebook_import.read_notebook(
        _nb(
            "model, tok = FastLanguageModel.from_pretrained(model_name = 'org/custom', trust_remote_code = True)",
            "dataset = load_dataset('imdb', split = 'train')",
        ),
        "x.ipynb",
    )
    assert doc is None and any("trust_remote_code" in line for line in report["unsupported"])


def test_embedding_notebooks_with_another_loss_are_reported():
    def emb(loss: str) -> bytes:
        return _nb(
            "model = FastSentenceTransformer.from_pretrained(model_name = 'unsloth/embeddinggemma-300m')",
            "dataset = load_dataset('a/b', split = 'train')",
            f"loss = {loss}(model)\ntrainer = SentenceTransformerTrainer(model = model, loss = loss)",
        )

    doc, report = notebook_import.read_notebook(emb("CosineSimilarityLoss"), "x.ipynb")
    assert doc is None and any("CosineSimilarityLoss" in line for line in report["unsupported"])
    doc, _ = notebook_import.read_notebook(emb("MultipleNegativesRankingLoss"), "x.ipynb")
    assert doc is not None


def test_lora_settings_studio_cannot_forward_are_reported():
    def read(peft_args):
        nb = _nb(
            "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')\n"
            f"model = FastLanguageModel.get_peft_model(model, r = 16, lora_alpha = 32{peft_args})",
            "dataset = load_dataset('mlabonne/FineTome-100k', split = 'train')",
        )
        doc, report = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
        assert doc is not None and report["unsupported"] == []
        return [line for line in report["approximated"] if "LoRA" in line or "rslora" in line]

    custom = read(", target_modules = ['q_proj'], lora_dropout = 0.1, use_rslora = True")
    assert len(custom) == 3
    assert any("target_modules" in line for line in custom)
    assert any("dropout" in line for line in custom)
    assert any("use_rslora" in line for line in custom)
    # The standard notebook settings match Studio's defaults and stay quiet.
    assert (
        read(
            ", target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', "
            "'down_proj'], lora_dropout = 0, use_rslora = False"
        )
        == []
    )


_SEVEN = "['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']"


def _lora_lines(report):
    return [line for line in report["approximated"] if "target_modules" in line]


def test_lora_targets_resolve_a_module_level_constant():
    nb = _nb(
        f"TARGETS = {_SEVEN}\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')\n"
        "model = FastLanguageModel.get_peft_model(model, r = 16, target_modules = TARGETS)",
        "dataset = load_dataset('mlabonne/FineTome-100k', split = 'train')",
    )
    doc, report = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
    assert doc is not None and _lora_lines(report) == []


def test_lora_targets_follow_the_latest_assignment_of_a_constant():
    def read(*assignments):
        nb = _nb(
            "\n".join(assignments) + "\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')\n"
            "model = FastLanguageModel.get_peft_model(model, r = 16, target_modules = TARGETS)",
            "dataset = load_dataset('mlabonne/FineTome-100k', split = 'train')",
        )
        doc, report = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
        assert doc is not None
        return _lora_lines(report)

    # The seven defaults reassigned to something else must be reported, not masked by the old list.
    assert len(read(f"TARGETS = {_SEVEN}", "TARGETS = 'all-linear'")) == 1
    assert len(read(f"TARGETS = {_SEVEN}", "TARGETS = ['q_proj']")) == 1
    assert len(read(f"TARGETS = {_SEVEN}", "TARGETS = pick_targets()")) == 1
    # And the reverse: a later default list wins over an earlier string.
    assert read("TARGETS = 'all-linear'", f"TARGETS = {_SEVEN}") == []


def test_lora_scalars_follow_an_unresolved_reassignment():
    def read(*assignments, peft_args):
        nb = _nb(
            "\n".join(assignments) + "\n"
            "model, tokenizer = FastLanguageModel.from_pretrained(model_name = 'unsloth/Qwen3-4B')\n"
            f"model = FastLanguageModel.get_peft_model(model, r = 16{peft_args})",
            "dataset = load_dataset('mlabonne/FineTome-100k', split = 'train')",
        )
        doc, report = notebook_import.read_notebook(nb, "Qwen3_(4B).ipynb")
        assert doc is not None
        return [line for line in report["approximated"] if "dropout" in line or "rslora" in line]

    # A computed value replaces the earlier default, so it must not read as that default.
    assert len(read("DROP = 0", "DROP = pick_dropout()", peft_args = ", lora_dropout = DROP")) == 1
    assert len(read("RS = False", "RS = pick_rslora()", peft_args = ", use_rslora = RS")) == 1
    assert read("DROP = 0", peft_args = ", lora_dropout = DROP") == []


def test_embedding_lora_targets_compare_against_embedding_defaults():
    def read(targets):
        nb = _nb(
            "model = FastSentenceTransformer.from_pretrained(model_name = 'unsloth/embeddinggemma-300m')\n"
            f"model = FastSentenceTransformer.get_peft_model(model, r = 32, target_modules = {targets})",
            "dataset = load_dataset('a/b', split = 'train')",
            "loss = MultipleNegativesRankingLoss(model)\n"
            "trainer = SentenceTransformerTrainer(model = model, loss = loss)",
        )
        doc, report = notebook_import.read_notebook(nb, "x.ipynb")
        assert doc is not None
        return _lora_lines(report)

    assert read("['q_proj', 'k_proj', 'v_proj', 'o_proj']") == []
    seven = read(_SEVEN)
    assert len(seven) == 1 and "q/k/v/o projections)" in seven[0]
