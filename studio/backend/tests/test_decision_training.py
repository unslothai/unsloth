# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors import safe_open

from auth.authentication import get_current_subject
from core.systemone import catalog, laya_runtime

WORDS = (
    "the server is down again refund my card charge twice please help now "
    "angry calm fine login broken slow outage billing account password"
).split()
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply now?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges and refunds"},
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "angry"],
    },
}


def _row(i):
    outage = i % 2 == 0
    return {
        "state": "the server is down again help now" if outage else "refund my card charge twice",
        "questions": json.dumps(QUESTIONS),
        "gold": json.dumps(
            {
                "urgent": {"label": "true" if outage else "false"},
                "team": {
                    "label": "outage" if outage else "billing",
                    "probabilities": {"outage": 0.9, "billing": 0.1}
                    if outage
                    else {"outage": 0.2, "billing": 0.8},
                },
                "mood": {"label": 2 if outage else 0},
            }
        ),
    }


def _base_checkpoint(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, PreTrainedTokenizerFast
    from safetensors.torch import save_file

    laya = laya_runtime._laya()
    specials = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab = {token: i for i, token in enumerate(specials + sorted(set(WORDS)))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        sep_token = "[SEP]",
        cls_token = "[CLS]",
        unk_token = "[UNK]",
        mask_token = "[MASK]",
    ).save_pretrained(str(folder / "tokenizer"))
    ModernBertConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        pad_token_id = 0,
        cls_token_id = 2,
        sep_token_id = 1,
        global_attn_every_n_layers = 2,
        local_attention = 16,
    ).save_pretrained(str(folder / "encoder"))
    cfg = {
        "encoder": "tiny",
        "head_layers": 1,
        "act_costs": {"escalate": 0.5},
        "max_len": 96,
        "head_max_len": 48,
        "temperature": [1.2, 1.1, 1.3],
        "temperature_by_options": {"noul:2": 0.1},
    }
    torch.manual_seed(0)
    model = laya.common.build_model(cfg, encoder_dir = str(folder / "encoder"))
    save_file(
        {k: v.half().contiguous() for k, v in model.state_dict().items()},
        str(folder / "model.safetensors"),
    )
    (folder / "rl_agent_config.json").write_text(json.dumps(cfg), encoding = "utf-8")
    return folder


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    for name in (
        "UNSLOTH_SYSTEMONE_MODEL",
        "UNSLOTH_SYSTEMONE_DISABLE",
        "UNSLOTH_SYSTEMONE_DEVICE",
    ):
        monkeypatch.delenv(name, raising = False)
    for name in ("_agent", "_loaded", "_device_name", "_loader", "_loading", "_failure"):
        monkeypatch.setattr(laya_runtime, name, None)
    yield tmp_path
    if laya_runtime._loader is not None:
        laya_runtime._loader.join(30)


@pytest.fixture
def base(studio_home):
    return _base_checkpoint(studio_home / "base")


def _dataset(folder, rows):
    path = folder / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding = "utf-8")
    return str(path)


def _config(base, dataset, **overrides):
    return {
        "model_name": str(base),
        "model_subfolder": None,
        "is_decision": True,
        "hf_dataset": "",
        "local_datasets": [dataset],
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_epochs": 1,
        "max_steps": 3,
        "learning_rate": "1e-3",
        "warmup_steps": 0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "random_seed": 3407,
        "gradient_checkpointing": "none",
        "eval_steps": 0,
        "project_name": None,
        **overrides,
    }


# The worker imports unsloth; CI installs unsloth_zoo only for the step that runs this file.
needs_worker = pytest.mark.skipif(
    sys.platform == "darwin" or importlib.util.find_spec("unsloth_zoo") is None,
    reason = "trains in a worker that imports unsloth (Apple Silicon trains with MLX)",
)


def _train(
    config,
    stop = None,
    when = lambda event: event["type"] == "progress",
):
    import multiprocessing as mp

    from core.training.training import _build_training_worker_config
    from core.training.worker import run_training_process

    context = mp.get_context("spawn")
    events, stops = context.Queue(), context.Queue()
    process = context.Process(
        target = run_training_process,
        kwargs = {
            "event_queue": events,
            "stop_queue": stops,
            "config": _build_training_worker_config({"training_type": "Full Finetuning", **config}),
        },
    )
    process.start()
    received = []
    try:
        while not received or received[-1]["type"] not in ("complete", "error"):
            received.append(events.get(timeout = 600))
            if stop is not None and when(received[-1]):
                stops.put({"type": "stop", "save": stop})
                stop = None
    finally:
        process.join(60)
    return received


def _of(events, kind):
    return [e for e in events if e["type"] == kind]


@pytest.fixture
def client(studio_home):
    from routes import systemone
    from routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _decide(client, model = "default"):
    for _ in range(120):
        response = client.post(
            "/v1/systemone",
            json = {"model": model, "state": "the server is down again", "questions": QUESTIONS},
        )
        if response.status_code != 503:
            return response
        time.sleep(0.5)
    return response


@needs_worker
def test_fine_tune_is_calibrated_saved_and_served(base, studio_home, client):
    rows = [_row(i) for i in range(200)]
    rows += [
        {"state": "slow login", "questions": "not json", "gold": "{}"},
        {"state": "slow login", "questions": json.dumps(QUESTIONS), "gold": json.dumps({})},
        {
            "state": "slow login",
            "questions": json.dumps({"q": {"type": "maybe", "instructions": "?"}}),
            "gold": json.dumps({"q": "yes"}),
        },
    ]
    events = _train(_config(base, _dataset(studio_home, rows)))

    assert not _of(events, "error"), _of(events, "error")
    progress = [e for e in _of(events, "progress") if e["loss"] is not None]
    assert [e["step"] for e in progress] == [1, 2, 3]
    assert all(e["total_steps"] == 3 and e["loss"] > 0 for e in progress)
    assert any(e["eval_loss"] is not None for e in _of(events, "progress"))
    assert _of(events, "warning")[0]["message"] == (
        'Skipped 5 of 605 decisions: row 202: "urgent" has no gold (and 2 more like it).'
    )
    complete = _of(events, "complete")[-1]
    assert complete["status_message"].startswith("Held-out accuracy ")
    output = complete["output_dir"]
    assert _of(events, "output_dir")[0]["output_dir"] == output

    from utils.paths import outputs_root

    folder = outputs_root() / output.rsplit("/", 1)[-1]
    assert str(folder) == output
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    assert cfg["fine_tuned"] is True
    assert "temperature_by_options" not in cfg
    assert cfg["max_len"] == 1024 and cfg["head_max_len"] == 256
    assert all(0.5 <= t <= 5.0 for t in cfg["temperature"])
    assert cfg["temperature"] != [1.2, 1.1, 1.3]
    assert cfg["training"]["heldout_decisions"] == 60
    assert cfg["training"]["objective"] == "soft_cross_entropy"
    assert cfg["training"]["steps"] == 3
    with safe_open(str(folder / "model.safetensors"), "pt") as weights:
        assert {weights.get_tensor(k).dtype for k in weights.keys()} == {torch.float16}

    name = catalog.FINE_TUNE_PREFIX + folder.name
    settings = client.get("/api/settings/systemone").json()
    assert {"name": name, "kind": "fine_tune", "label": folder.name}.items() <= next(
        m for m in settings["models"] if m["name"] == name
    ).items()
    response = client.put("/api/settings/systemone", json = {"enabled": True, "model": name})
    assert response.status_code == 200, response.text
    assert response.json()["model"] == name
    plan = client.get("/api/settings/systemone/resolve").json()
    assert plan["cached"] is True and plan["repo"] is None

    response = _decide(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == name
    assert set(body["answers"]) == set(QUESTIONS)
    assert body["answers"]["team"]["choice"] in ("outage", "billing")
    assert _decide(client, name).json()["model"] == name


@needs_worker
def test_lora_run_merges_into_the_served_layout(base, studio_home, client, monkeypatch):
    from safetensors.torch import load_file

    # The spawned worker inherits sys.path and imports this wandb, whose init fails like a bad key.
    offline = studio_home / "offline" / "wandb"
    offline.mkdir(parents = True)
    (offline / "__init__.py").write_text(
        'def init(**kwargs):\n    raise RuntimeError("cannot reach W&B")\n', encoding = "utf-8"
    )
    monkeypatch.syspath_prepend(str(offline.parent))
    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(
            base,
            _dataset(studio_home, rows),
            training_type = "LoRA/QLoRA",
            lora_r = 4,
            lora_alpha = 4,
            enable_wandb = True,
        )
    )

    assert not _of(events, "error"), _of(events, "error")
    assert [e["message"] for e in _of(events, "warning")] == [
        "Weights & Biases logging is off: cannot reach W&B"
    ]
    checkpoint = catalog.fine_tune(
        catalog.FINE_TUNE_PREFIX + _of(events, "complete")[-1]["output_dir"].rsplit("/", 1)[-1]
    )
    folder = Path(checkpoint.source)
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    assert cfg["training"]["method"] == "lora"
    tuned = load_file(str(folder / "model.safetensors"))
    original = load_file(str(base / "model.safetensors"))
    assert tuned.keys() == original.keys()
    assert {t.dtype for t in tuned.values()} == {torch.float16}
    assert not torch.equal(
        tuned["encoder.layers.0.attn.Wqkv.weight"], original["encoder.layers.0.attn.Wqkv.weight"]
    )
    assert torch.equal(
        tuned["encoder.embeddings.tok_embeddings.weight"],
        original["encoder.embeddings.tok_embeddings.weight"],
    )

    name = catalog.FINE_TUNE_PREFIX + folder.name
    response = client.put("/api/settings/systemone", json = {"enabled": True, "model": name})
    assert response.status_code == 200, response.text
    response = _decide(client)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == name


@needs_worker
@pytest.mark.skipif(
    importlib.util.find_spec("tensorboard") is None
    and importlib.util.find_spec("tensorboardX") is None,
    reason = "needs tensorboard",
)
def test_tensorboard_logs_a_decision_run(base, studio_home):
    from utils.paths import tensorboard_root

    rows = [_row(i) for i in range(60)]
    events = _train(
        _config(
            base,
            _dataset(studio_home, rows),
            max_steps = 2,
            enable_tensorboard = True,
            tensorboard_dir = "decision-tb",
        )
    )

    assert not _of(events, "error"), _of(events, "error")
    assert list((tensorboard_root() / "decision-tb").rglob("events.out.tfevents.*"))


@needs_worker
def test_an_unusable_eval_file_falls_back_to_the_hold_out(base, studio_home):
    held = studio_home / "held"
    held.mkdir()
    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(
            base,
            _dataset(studio_home, rows),
            eval_steps = 0.5,
            local_eval_datasets = [_dataset(held, [{"state": "slow"}] * 4)],
        )
    )

    assert not _of(events, "error"), _of(events, "error")
    assert "Holding out 36 decisions to calibrate confidence..." in [
        e["message"] for e in _of(events, "status")
    ]
    folder = Path(_of(events, "complete")[-1]["output_dir"])
    cfg = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
    assert cfg["training"]["heldout_decisions"] == 36


@needs_worker
def test_all_invalid_rows_are_an_error_and_save_nothing(base, studio_home):
    from utils.paths import outputs_root

    rows = [{"state": "slow", "questions": json.dumps(QUESTIONS)}] * 4
    events = _train(_config(base, _dataset(studio_home, rows)))

    assert "No usable decisions" in _of(events, "error")[0]["error"]
    assert not _of(events, "complete")
    assert not outputs_root().exists() or not any(outputs_root().iterdir())


@needs_worker
@pytest.mark.parametrize("training_type", ["Full Finetuning", "LoRA/QLoRA"])
def test_stop_with_save_leaves_a_servable_checkpoint(base, studio_home, training_type):
    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(
            base,
            _dataset(studio_home, rows),
            max_steps = 0,
            num_epochs = 3,
            training_type = training_type,
        ),
        True,
    )

    complete = _of(events, "complete")[-1]
    last = _of(events, "progress")[-1]
    assert last["step"] < last["total_steps"]
    checkpoint = catalog.fine_tune(
        catalog.FINE_TUNE_PREFIX + complete["output_dir"].rsplit("/", 1)[-1]
    )
    assert checkpoint is not None and laya_runtime.is_cached(checkpoint)
    folder = Path(checkpoint.source)
    with (
        safe_open(str(folder / "model.safetensors"), "pt") as tuned,
        safe_open(str(base / "model.safetensors"), "pt") as original,
    ):
        assert set(tuned.keys()) == set(original.keys())


@needs_worker
def test_cancel_leaves_no_output_folder(base, studio_home):
    from utils.paths import outputs_root

    rows = [_row(i) for i in range(120)]
    events = _train(_config(base, _dataset(studio_home, rows), max_steps = 0, num_epochs = 3), False)

    complete = _of(events, "complete")[-1]
    assert complete["output_dir"] is None
    assert complete["status_message"] == "Training cancelled"
    assert _of(events, "output_dir")
    assert not outputs_root().exists() or not any(outputs_root().iterdir())


@needs_worker
def test_stop_while_preparing_decisions_ends_the_run(base, studio_home):
    from utils.paths import outputs_root

    # Seconds of preparation; the bad row comes last, so a skip warning means it ran to the end.
    rows = [_row(i) for i in range(6000)] + [{"state": "slow", "questions": QUESTIONS, "gold": {}}]
    events = _train(
        _config(base, _dataset(studio_home, rows)),
        False,
        lambda event: event.get("message") == "Preparing decisions...",
    )

    assert _of(events, "complete")[-1]["status_message"] == "Training cancelled"
    assert not _of(events, "warning") and not _of(events, "output_dir")
    assert not outputs_root().exists() or not any(outputs_root().iterdir())


def _fake_output(
    root,
    name,
    complete = True,
):
    folder = root / name
    for sub in ("encoder", "tokenizer"):
        (folder / sub).mkdir(parents = True)
    (folder / "model.safetensors").write_bytes(b"x")
    if complete:
        (folder / "rl_agent_config.json").write_text("{}", encoding = "utf-8")
    return folder


def test_settings_accept_only_complete_owner_fine_tunes(studio_home, client, tmp_path):
    from utils.paths import outputs_root

    root = outputs_root()
    _fake_output(root, "laya_done_1")
    _fake_output(root, "laya_half_2", complete = False)
    outside = _fake_output(tmp_path, "elsewhere")
    (root / "linked").symlink_to(outside, target_is_directory = True)

    names = [m["name"] for m in client.get("/api/settings/systemone").json()["models"]]
    assert "laya-ft:laya_done_1" in names
    assert not any(n in names for n in ("laya-ft:laya_half_2", "laya-ft:linked"))
    for bad in (
        "laya-ft:laya_half_2",
        "laya-ft:linked",
        "laya-ft:../elsewhere",
        f"laya-ft:{outside}",
        "laya-ft:",
    ):
        response = client.put("/api/settings/systemone", json = {"model": bad})
        assert response.status_code == 400, bad

    response = client.put("/api/settings/systemone", json = {"model": "laya-ft:laya_done_1"})
    assert response.status_code == 200
    assert client.get("/api/settings/systemone").json()["model"] == "laya-ft:laya_done_1"
    shutil.rmtree(root / "laya_done_1")
    assert client.get("/api/settings/systemone").json()["model"] == "laya-multilingual"


def test_llm_output_scans_skip_decision_outputs(studio_home):
    from utils.models.checkpoints import scan_checkpoints
    from utils.models.model_config import scan_trained_models
    from utils.paths import outputs_root

    root = outputs_root()
    _fake_output(root, "laya_done_1")
    merged = root / "llama_merged_1"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding = "utf-8")
    (merged / "model.safetensors").write_bytes(b"x")

    assert [name for name, _, _ in scan_trained_models(str(root))] == ["llama_merged_1"]
    assert [name for name, _, _ in scan_checkpoints(str(root))] == ["llama_merged_1"]


def test_export_checkpoint_list_includes_decision_runs(studio_home):
    from routes import models as models_routes
    from utils.models.checkpoints import list_preview_targets
    from utils.paths import outputs_root

    root = outputs_root()
    laya = _fake_output(root, "laya_done_1")
    (laya / "rl_agent_config.json").write_text(
        json.dumps({"encoder": "answerdotai/ModernBERT-base", "training": {"base": "laya-tiny"}}),
        encoding = "utf-8",
    )
    _fake_output(root, "laya_half_2", complete = False)
    clef = root / "clef_merged_3"
    clef.mkdir()
    for name in ("config.json", "joint_head_config.json", "joint_head.safetensors"):
        (clef / name).write_text("{}", encoding = "utf-8")

    app = FastAPI()
    app.include_router(models_routes.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    response = TestClient(app).get("/api/models/checkpoints", params = {"outputs_dir": str(root)})
    assert response.status_code == 200, response.text
    listed = {m["name"]: m for m in response.json()["models"]}
    assert sorted(listed) == ["clef_merged_3", "laya_done_1"]
    assert listed["laya_done_1"]["base_model"] == "laya-tiny"
    assert listed["laya_done_1"]["checkpoints"][0]["path"] == str(laya)
    # Chat preview still lists only what chat can load.
    assert [t["run"] for t in list_preview_targets(str(root))] == ["clef_merged_3"]


def test_model_config_offers_an_llm_as_a_decision_model(studio_home):
    import asyncio

    from routes.models import get_model_config
    from utils.models.model_config import load_llm_decision_defaults

    llm = studio_home / "llm"
    llm.mkdir()
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 64,
    }
    (llm / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (llm / "model.safetensors").write_bytes(b"x")

    def fetch(**kwargs):
        return asyncio.run(
            get_model_config(model_name = str(llm), hf_token = None, current_subject = "tester", **kwargs)
        )

    plain = fetch()
    assert plain.model_type == "text" and plain.decision_layout is None
    decision = fetch(as_decision = True)
    assert decision.model_type == "decision" and decision.is_decision is True
    assert decision.decision_layout == "llm" and decision.decision_checkpoints is None
    assert decision.config == load_llm_decision_defaults()


def test_model_config_classifies_a_local_laya_folder(base):
    import asyncio

    from routes.models import get_model_config

    result = asyncio.run(
        get_model_config(model_name = str(base), hf_token = None, current_subject = "tester")
    )
    assert result.model_type == "decision" and result.is_decision is True
    assert float(result.config["training"]["learning_rate"]) == 8e-4
    assert result.config["training"]["optim"] == "adamw_torch"
    assert result.config["lora"]["lora_r"] == 64
    assert result.decision_checkpoints is None


def test_start_preflight_accepts_a_local_laya_folder(base):
    from models.training import TrainingStartRequest
    from routes.training import _reject_untrainable_model_request

    request = TrainingStartRequest(
        model_name = str(base),
        training_type = "Full Finetuning",
        is_decision = True,
        hf_dataset = "org/decisions",
        format_type = "auto",
    )
    assert _reject_untrainable_model_request(request).model_name == str(base.resolve())


def test_arrow_merged_criteria_are_not_trained_as_options(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from core.training.decision_trainer import STRUCT_COLUMNS_WARNING, _read_local_rows
    from utils.datasets.cache_safe import load_dataset_cache_safe

    rows = [
        {
            "state": "refund my card",
            "questions": {"q": {"type": "choice", "instructions": "Team?", "criteria": crit}},
            "gold": {"q": {"label": label, "probabilities": probs}},
        }
        for crit, label, probs in (
            ({"billing": "charges", "outage": ""}, "billing", {"billing": 0.9, "outage": 0.1}),
            ({"login": "accounts", "slow": "speed"}, "slow", {"login": 0.2, "slow": 0.8}),
        )
    ]
    path = tmp_path / "decisions.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))

    warnings = []
    loaded = _read_local_rows([str(path)], load_dataset_cache_safe, warnings.append)
    assert [row["questions"]["q"]["criteria"] for row in loaded] == [
        {"billing": "charges", "outage": ""},
        {"login": "accounts", "slow": "speed"},
    ]
    assert loaded[1]["gold"]["q"]["probabilities"] == {"login": 0.2, "slow": 0.8}
    assert warnings == [STRUCT_COLUMNS_WARNING]

    strings = tmp_path / "strings.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{key: json.dumps(value) for key, value in row.items()} for row in rows]
        ),
        str(strings),
    )
    warnings.clear()
    _read_local_rows([str(strings)], load_dataset_cache_safe, warnings.append)
    assert not warnings


def _load(should_stop = lambda: False, **config):
    from core.training.decision_trainer import _load_rows

    warnings = []
    rows, eval_rows = _load_rows(
        {"random_seed": 3407, "eval_steps": 0, **config},
        should_stop,
        lambda message: None,
        warnings.append,
    )
    return rows, eval_rows, warnings


def test_local_files_are_read_before_a_hub_dataset(tmp_path):
    rows, _, _ = _load(
        local_datasets = [_dataset(tmp_path, [_row(0)])], hf_dataset = "nobody/not-a-dataset"
    )
    assert rows == [_row(0)]


def test_the_train_split_is_never_the_held_out_set(tmp_path):
    hub = tmp_path / "decisions"
    hub.mkdir()
    for split, count in (("train", 30), ("test", 10)):
        (hub / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(_row(i)) for i in range(count)), encoding = "utf-8"
        )

    rows, eval_rows, warnings = _load(
        hf_dataset = str(hub), train_split = "train", eval_split = "train", eval_steps = 0.1
    )
    assert len(rows) == 30 and eval_rows is None
    assert warnings == [
        "The evaluation split is the training split, so part of it is held out "
        "of training for evaluation instead."
    ]

    rows, eval_rows, warnings = _load(
        hf_dataset = str(hub), train_split = "train", eval_split = "test", eval_steps = 0.1
    )
    assert len(rows) == 30 and len(eval_rows) == 10 and not warnings


def test_eval_rows_are_sampled_before_they_are_prepared(tmp_path):
    from core.training.decision_trainer import EVAL_MAX

    held = tmp_path / "held"
    held.mkdir()
    _, eval_rows, _ = _load(
        local_datasets = [_dataset(tmp_path, [_row(0)])],
        local_eval_datasets = [_dataset(held, [_row(i) for i in range(EVAL_MAX + 500)])],
        eval_steps = 0.1,
    )
    assert len(eval_rows) == EVAL_MAX


def test_csv_cells_are_read_as_text(tmp_path):
    import csv

    path = tmp_path / "decisions.csv"
    with path.open("w", newline = "", encoding = "utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["state", "questions", "gold"])
        for state in ("NA", "00501", "null"):
            writer.writerow([state, _row(0)["questions"], _row(0)["gold"]])

    rows, _, _ = _load(local_datasets = [str(path)])
    assert [row["state"] for row in rows] == ["NA", "00501", "null"]


def test_a_stop_during_the_s3_download_cancels_the_run(monkeypatch):
    from core.training import s3_dataset
    from core.training.decision_trainer import _Stopped

    def download(s3_config, cancel_callback):
        assert cancel_callback()
        raise s3_dataset.S3DownloadCancelled("S3 dataset download cancelled")

    monkeypatch.setattr(s3_dataset, "prepare_s3_dataset_download", download)
    with pytest.raises(_Stopped, match = "Training cancelled"):
        _load(lambda: True, s3_config = {"bucket": "b"})


def test_other_accounts_reach_only_the_configured_fine_tune(studio_home, client):
    from utils.account_context import AccountContext, bind_account, reset_account
    from utils.paths import outputs_root

    root = outputs_root()
    _fake_output(root, "laya_done_1")
    _fake_output(root, "laya_done_2")
    assert (
        client.put("/api/settings/systemone", json = {"model": "laya-ft:laya_done_1"}).status_code
        == 200
    )

    token = bind_account(AccountContext("b" * 32, "bob"))
    try:
        assert catalog.resolve("laya-ft:laya_done_1") is not None
        assert catalog.resolve("laya-ft:laya_done_2") is None
    finally:
        reset_account(token)
    assert catalog.resolve("laya-ft:laya_done_2") is not None


def test_only_the_owner_can_start_decision_training():
    from fastapi import HTTPException

    from models.training import TrainingStartRequest
    from routes.training import _validate_decision_request
    from utils.account_context import AccountContext, bind_account, reset_account

    request = TrainingStartRequest(
        model_name = catalog.LAYA_REPO,
        model_subfolder = "multilingual",
        training_type = "Full Finetuning",
        is_decision = True,
        hf_dataset = "org/decisions",
        format_type = "auto",
    )
    _validate_decision_request(request)
    token = bind_account(AccountContext("b" * 32, "bob"))
    try:
        with pytest.raises(HTTPException) as refused:
            _validate_decision_request(request)
    finally:
        reset_account(token)
    assert refused.value.status_code == 403


def test_eval_files_that_are_training_files_are_never_the_held_out_set(tmp_path):
    train = _dataset(tmp_path, [_row(i) for i in range(30)])
    for eval_files in ([train], [str(tmp_path)], [str(tmp_path / "." / "decisions.jsonl")]):
        rows, eval_rows, warnings = _load(
            local_datasets = [train], local_eval_datasets = eval_files, eval_steps = 0.1
        )
        assert len(rows) == 30 and eval_rows is None, eval_files
        assert warnings == [
            "The evaluation files include training files, so part of the training data is "
            "held out for evaluation instead."
        ]


@needs_worker
def test_cancel_during_calibration_saves_nothing(base, studio_home):
    from utils.paths import outputs_root

    rows = [_row(i) for i in range(120)]
    events = _train(
        _config(base, _dataset(studio_home, rows)),
        False,
        when = lambda event: event.get("message") == "Calibrating confidence...",
    )

    complete = _of(events, "complete")[-1]
    assert (complete["output_dir"], complete["status_message"]) == (None, "Training cancelled")
    assert not outputs_root().exists() or not any(outputs_root().iterdir())


def test_held_out_decisions_count_every_clef_question():
    from core.training.decision_trainer import _decision_count

    laya = [{"row": 0}, {"row": 1}]
    clef = [{"row": 0, "labels": [0, 1, 2]}, {"row": 1, "labels": [1]}]
    assert (_decision_count(laya), _decision_count(clef)) == (2, 4)


def test_the_best_held_out_step_is_kept(monkeypatch):
    from types import SimpleNamespace

    from core.training import decision_trainer

    model = torch.nn.Linear(2, 1)
    warnings = []
    keep = decision_trainer._keep_best(model, warnings.append)

    def evaluate(step, loss):
        with torch.no_grad():
            model.weight.fill_(float(step))
        keep.on_evaluate(None, SimpleNamespace(global_step = step), None, metrics = {"eval_loss": loss})

    evaluate(1, 1.0)
    evaluate(2, 0.4)
    evaluate(3, 0.7)
    evaluate(4, float("nan"))
    assert keep.evaluated == 4
    assert keep.restore(4) == 2 and float(model.weight[0, 0]) == 2.0 and warnings == []
    evaluate(5, 0.1)
    assert keep.restore(5) == 5 and float(model.weight[0, 0]) == 5.0
    monkeypatch.setattr(decision_trainer, "KEEP_BEST_MAX_BYTES", 0)
    keep = decision_trainer._keep_best(model, warnings.append)
    evaluate(1, 1.0)
    evaluate(2, 0.4)
    assert keep.restore(5) == 5 and float(model.weight[0, 0]) == 2.0
    assert len(warnings) == 1 and "last step is saved" in warnings[0]
