# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drafter trained after the fine-tune: request contract, worker hand-off, the phase itself (fake trainer),
export packaging, and the llama-server / vLLM launch rules that read what the export wrote."""

from __future__ import annotations

import ast
import json
import os
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from models.training import DrafterTrainingConfig, TrainingStartRequest  # noqa: E402

BASE = dict(model_name = "x", format_type = "auto", hf_dataset = "a/b", training_type = "LoRA/QLoRA")


# ---------------------------------------------------------------- request contract


def test_request_defaults_and_omission():
    assert TrainingStartRequest(**BASE).drafter is None
    req = TrainingStartRequest(**BASE, drafter = {"enabled": True})
    d = req.drafter
    assert (
        d.strategy,
        d.data_source,
        d.max_examples,
        d.epochs,
        d.learning_rate,
        d.draft_steps,
    ) == (
        "auto",
        "dataset",
        2000,
        1.0,
        None,
        3,
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": True, "strategy": "medusa"},
        {"enabled": True, "data_source": "web"},
        {"enabled": True, "max_examples": 8},
        {"enabled": True, "max_examples": 100001},
        {"enabled": True, "epochs": 0.05},
        {"enabled": True, "draft_steps": 9},
        {"enabled": True, "draft_steps": True},
        {"enabled": True, "learning_rate": 0},
    ],
)
def test_request_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        TrainingStartRequest(**BASE, drafter = bad)


@pytest.mark.parametrize(
    "extra",
    [{"is_embedding": True}, {"is_decision": True}, {"training_type": "Continued Pretraining"}],
)
def test_request_rejects_unsupported_runs(extra):
    with pytest.raises(ValueError):
        TrainingStartRequest(**{**BASE, **extra}, drafter = {"enabled": True})
    # Disabled is always accepted.
    TrainingStartRequest(**{**BASE, **extra}, drafter = {"enabled": False})


def test_worker_config_carries_drafter():
    from core.training.training import _build_training_worker_config

    cfg = {"model_name": "x", "drafter": DrafterTrainingConfig(enabled = True).model_dump()}
    assert _build_training_worker_config(cfg)["drafter"]["enabled"] is True
    assert _build_training_worker_config({"model_name": "x"})["drafter"] is None


def _source(rel):
    return (_BACKEND_DIR / rel).read_text(encoding = "utf-8")


def test_route_forwards_only_enabled_drafter():
    src = _source("routes/training.py")
    assert '"drafter": (' in src and "request.drafter.enabled" in src


def test_drafter_phase_is_derived_before_finalizing():
    """The SFT step counter sits at total during the drafter phase, which _is_finalizing reads as
    finalizing; the drafter branch must win first."""
    src = _source("routes/training.py")
    fn = next(
        ast.get_source_segment(src, n)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_build_training_status"
    )
    assert fn.index('phase = "training_drafter"') < fn.index('phase = "finalizing"')


def test_worker_runs_the_phase_before_complete():
    src = _source("core/training/worker.py")
    phase = src.index("run_drafter_phase(trainer, config, output_dir")
    assert phase < src.index('"type": "complete"', phase)
    # Only after a clean SFT finish.
    assert "not progress.error and not trainer.should_stop" in src[phase - 300 : phase]
    # A Stop during the drafter phase keeps the saved fine-tune in the completion event.
    assert src.index("stopped_in_training = trainer.should_stop") < phase
    assert "None if stopped_in_training and not trainer.save_on_stop else output_dir" in src[phase:]


# ---------------------------------------------------------------- examples


def test_examples_respect_masks_and_packing():
    from core.training.drafter import collect_examples

    rows = [
        {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [-100, -100, 3, 4, 5, 6]},
        {
            "input_ids": [7, 8, 9, 10, 11, 12, 13, 14],
            "completion_mask": [0, 0, 1, 1, 0, 0, 1, 1],
            "position_ids": [0, 1, 2, 3, 0, 1, 2, 3],
        },
        {"input_ids": [1, 2], "labels": [1, 2]},  # too short
    ]
    out = collect_examples(rows, 100, 2048)
    assert out[0]["labels"] == [-100, -100, 3, 4, 5, 6]
    # Packed row split at the position reset; each half keeps its own mask.
    assert out[1] == {"input_ids": [7, 8, 9, 10], "labels": [-100, -100, 9, 10]}
    assert out[2] == {"input_ids": [11, 12, 13, 14], "labels": [-100, -100, 13, 14]}
    assert len(out) == 3
    assert len(collect_examples(rows, 2, 2048)) == 2


def test_examples_split_trl_packed_rows_at_seq_lengths():
    """TRL's packed dataset keeps the boundaries in `seq_lengths` (position_ids come later, in its
    collator): each packed document is its own drafter example and keeps its own prompt."""
    from datasets import Dataset

    pack_dataset = pytest.importorskip("trl.data_utils").pack_dataset

    from core.training.drafter import _prompt_end, collect_examples

    docs = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13]],
            "completion_mask": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1, 1]],
        }
    )
    packed = pack_dataset(docs, seq_length = 9, strategy = "bfd")
    assert "seq_lengths" in packed.column_names and "position_ids" not in packed.column_names
    out = collect_examples(packed, 100, 2048)
    assert sorted(ex["input_ids"] for ex in out) == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12, 13],
    ]
    by_first = {ex["input_ids"][0]: ex["labels"] for ex in out}
    assert by_first == {
        1: [-100, -100, 3, 4],
        5: [-100, -100, 7, 8],
        9: [-100, -100, -100, 12, 13],
    }
    # Regenerate finds one prompt per packed document, not one per packed row.
    assert [_prompt_end(ex["input_ids"], ex["labels"], []) for ex in out] == [
        len(ex["labels"]) - 2 for ex in out
    ]

    row = {"input_ids": [10, 11, 12, 13, 20, 21, 22, 23], "seq_lengths": [4, 4]}
    assert [ex["input_ids"] for ex in collect_examples([row], 100, 2048)] == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]
    # Lengths that do not describe the row are ignored rather than trusted.
    bad = dict(row, seq_lengths = [4, 3])
    assert len(collect_examples([bad], 100, 2048)) == 1


# ---------------------------------------------------------------- the phase, with a fake trainer


class _Queue(list):
    def put(self, item):
        self.append(item)


class _FakeDrafterTrainer:
    fail = False

    def __init__(self, model, target_model, args, train_dataset, processing_class):
        self.args, self.train_dataset = args, train_dataset
        self.callbacks = []
        self.state = types.SimpleNamespace(global_step = 0, max_steps = 4, log_history = [])

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def train(self):
        if self.fail:
            raise RuntimeError("boom")
        control = types.SimpleNamespace(should_training_stop = False)
        for cb in self.callbacks:
            cb.on_train_begin(self.args, self.state, control)
        for step in range(1, 5):
            self.state.global_step = step
            logs = {
                "loss": 1.0 / step,
                "eagle3_loss_0": 1.0 / step,
                "eagle3_acc_0": 0.5,
            }  # UnslothEagle3Trainer keys
            self.state.log_history.append(logs)
            for cb in self.callbacks:
                cb.on_log(self.args, self.state, control, logs = logs)
                cb.on_step_end(self.args, self.state, control)
        return types.SimpleNamespace(metrics = {"train_loss": 0.4})


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_zoo(monkeypatch):
    import torch

    mod = types.ModuleType("unsloth_zoo.speculative")
    mod.Eagle3TrainingArguments = _Args
    mod.UnslothEagle3Trainer = _FakeDrafterTrainer
    mod.build_eagle3_vocab_map = lambda ids, vocab, draft_vocab_size = 32000: ("d2t", "t2d")
    mod.build_eagle3_drafter = lambda model, d2t, t2d: types.SimpleNamespace(
        aux_layer_ids = [2, 4, 5]
    )

    def save(
        directory,
        drafter,
        verifier_name_or_path,
        speculative_tokens = 3,
    ):
        os.makedirs(directory, exist_ok = True)
        Path(directory, "config.json").write_text(
            json.dumps(
                {"speculators_config": {"verifier": {"name_or_path": verifier_name_or_path}}}
            )
        )

    mod.save_eagle3_drafter = save
    pkg = types.ModuleType("unsloth_zoo")
    pkg.speculative = mod
    monkeypatch.setitem(sys.modules, "unsloth_zoo", pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.speculative", mod)
    from core.training import drafter

    monkeypatch.setattr(
        drafter,
        "capability_in_process",
        lambda name, token = None, **_: {
            "supported": True,
            "strategy": "eagle3",
            "has_native_mtp": False,
            "vllm": True,
            "llama_cpp": True,
        },
    )
    monkeypatch.setattr(drafter, "_mtp_family", lambda config: (None, None))
    _FakeDrafterTrainer.fail = False
    model = torch.nn.Linear(2, 2)
    model.config = types.SimpleNamespace(vocab_size = 50, get_text_config = lambda: model.config)
    model.config.vocab_size = 50
    sft = types.SimpleNamespace(
        train_dataset = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}] * 20,
        optimizer = object(),
        lr_scheduler = object(),
        eval_dataset = None,
    )
    trainer = types.SimpleNamespace(
        model = model,
        tokenizer = None,
        trainer = sft,
        should_stop = False,
        is_vlm = False,
        model_name = "org/base",
    )
    return drafter, trainer


def _config(**drafter):
    return {
        "model_name": "org/base",
        "max_seq_length": 64,
        "drafter": {
            "enabled": True,
            "strategy": "auto",
            "data_source": "dataset",
            "max_examples": 16,
            "epochs": 1,
            "learning_rate": None,
            "draft_steps": 3,
            **drafter,
        },
    }


def test_phase_trains_saves_and_reports(fake_zoo, tmp_path):
    drafter, trainer = fake_zoo
    (tmp_path / "adapter_config.json").write_text("{}")
    q = _Queue()
    manifest = drafter.run_drafter_phase(
        trainer, _config(), str(tmp_path), q, lambda eq, m: eq.put({"type": "status", "message": m})
    )
    assert manifest["strategy"] == "eagle3" and manifest["files"] == ["eagle3"]
    assert manifest["examples"] == 16 and manifest["aux_layer_ids"] == [2, 4, 5]
    assert json.loads((tmp_path / "drafter.json").read_text())["strategy"] == "eagle3"
    assert (
        json.loads((tmp_path / "eagle3" / "config.json").read_text())["speculators_config"][
            "verifier"
        ]["name_or_path"]
        == "org/base"
    )
    # SFT optimizer state released before the drafter trained.
    assert trainer.trainer is None
    kinds = [e["type"] for e in q]
    assert "drafter_progress" in kinds and not any(e.get("type") == "progress" for e in q)
    assert all(e["message"].startswith("Training drafter") for e in q if e["type"] == "status")
    assert q[-1]["type"] == "drafter_progress" and q[-1]["state"] == "done"
    assert not (tmp_path / "drafter_tmp").exists()
    # The trainer's own per-step accuracy reaches the manifest and the progress shown in the UI.
    assert manifest["metrics"]["eagle3_acc_0"] == 0.5
    assert q[-1]["top1"] == 0.5
    assert any(
        e.get("top1") == 0.5
        for e in q
        if e["type"] == "drafter_progress" and e["state"] == "training"
    )


def test_phase_failure_keeps_the_fine_tune(fake_zoo, tmp_path):
    drafter, trainer = fake_zoo
    (tmp_path / "adapter_model.safetensors").write_bytes(b"sft")
    _FakeDrafterTrainer.fail = True
    q = _Queue()
    assert drafter.run_drafter_phase(trainer, _config(), str(tmp_path), q, lambda *a: None) is None
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == b"sft"
    assert not (tmp_path / "drafter.json").exists() and not (tmp_path / "eagle3").exists()
    assert any(e["type"] == "warning" and "boom" in e["message"] for e in q)


def test_stop_during_phase_drops_only_the_drafter(fake_zoo, tmp_path, monkeypatch):
    drafter, trainer = fake_zoo
    orig_train = _FakeDrafterTrainer.train

    def train_then_stop(self):
        trainer.should_stop = True
        return orig_train(self)

    monkeypatch.setattr(_FakeDrafterTrainer, "train", train_then_stop)
    assert (
        drafter.run_drafter_phase(trainer, _config(), str(tmp_path), _Queue(), lambda *a: None)
        is None
    )
    assert not (tmp_path / "drafter.json").exists() and not (tmp_path / "eagle3").exists()


def test_stop_before_training_clears_a_previous_drafter(fake_zoo, tmp_path):
    """A resumed run in a reused folder, stopped while examples are collected."""
    drafter, trainer = fake_zoo
    (tmp_path / "drafter.json").write_text("{}")
    (tmp_path / "eagle3").mkdir()
    (tmp_path / "mtp_head.safetensors").write_bytes(b"old")
    trainer.should_stop = True
    assert (
        drafter.run_drafter_phase(trainer, _config(), str(tmp_path), _Queue(), lambda *a: None)
        is None
    )
    assert not any(
        (tmp_path / n).exists() for n in ("drafter.json", "eagle3", "mtp_head.safetensors")
    )


def test_disabled_phase_is_a_no_op(fake_zoo, tmp_path):
    drafter, trainer = fake_zoo
    q = _Queue()
    assert (
        drafter.run_drafter_phase(
            trainer, _config(enabled = False), str(tmp_path), q, lambda *a: None
        )
        is None
    )
    assert q == [] and trainer.trainer is not None


def test_unsupported_model_is_a_warning(fake_zoo, tmp_path, monkeypatch):
    drafter, trainer = fake_zoo
    monkeypatch.setattr(
        drafter,
        "capability_in_process",
        lambda name, token = None, **_: {"supported": False, "reason": "no drafter for gemma3"},
    )
    q = _Queue()
    assert drafter.run_drafter_phase(trainer, _config(), str(tmp_path), q, lambda *a: None) is None
    assert any("gemma3" in e.get("message", "") for e in q if e["type"] == "warning")


def test_strategy_resolution():
    from core.training.drafter import resolve_strategy

    mtp = {"supported": True, "strategy": "mtp", "has_native_mtp": True}
    eagle = {"supported": True, "strategy": "eagle3", "has_native_mtp": False}
    assert resolve_strategy("auto", mtp) == ("mtp", None)
    assert resolve_strategy("auto", eagle) == ("eagle3", None)
    assert resolve_strategy("mtp", eagle)[0] is None
    assert resolve_strategy("eagle3", mtp)[0] is None
    assert resolve_strategy("auto", {"supported": False, "reason": "r"}) == (None, "r")


# ---------------------------------------------------------------- export packaging


def _training_dir(tmp_path, strategy):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_config.json").write_text("{}")
    manifest = {
        "strategy": strategy,
        "base_model": "org/base",
        "draft_steps": 3,
        "metrics": {"x": 1},
    }
    (run / "drafter.json").write_text(json.dumps(manifest))
    if strategy == "eagle3":
        (run / "eagle3").mkdir()
        (run / "eagle3" / "config.json").write_text(
            json.dumps({"speculators_config": {"verifier": {"name_or_path": "org/base"}}})
        )
        (run / "eagle3" / "model.safetensors").write_bytes(b"w")
    else:
        (run / "mtp_head.safetensors").write_bytes(b"h")
    return run


def test_merged_eagle3_export_copies_drafter_and_sets_verifier(tmp_path):
    from core.export.drafter_export import export_drafter_details, package_merged

    run = _training_dir(tmp_path, "eagle3")
    out = tmp_path / "out"
    out.mkdir()
    info = package_merged(str(run), str(out), verifier = "me/finetune")
    assert (out / "eagle3" / "model.safetensors").read_bytes() == b"w"
    cfg = json.loads((out / "eagle3" / "config.json").read_text())
    assert cfg["speculators_config"]["verifier"]["name_or_path"] == "me/finetune"
    # A Hub push publishes no path from this machine: download the repo, then serve it.
    assert info["vllm_command"].startswith(
        "hf download me/finetune --local-dir finetune && vllm serve finetune"
    )
    assert (
        '"model":"finetune/eagle3"' in info["vllm_command"]
        and str(tmp_path) not in info["vllm_command"]
    )
    local = package_merged(str(run), str(out))
    assert (
        str(out / "eagle3") in local["vllm_command"]
        and '"method":"eagle3"' in local["vllm_command"]
    )
    details = export_drafter_details(str(out))
    assert details["strategy"] == "eagle3" and details["files"] == ["eagle3"]
    # Training copy untouched.
    assert (
        json.loads((run / "eagle3" / "config.json").read_text())["speculators_config"]["verifier"][
            "name_or_path"
        ]
        == "org/base"
    )


def test_merged_mtp_export_writes_manifest_only(tmp_path):
    from core.export.drafter_export import package_merged

    run = _training_dir(tmp_path, "mtp")
    out = tmp_path / "out"
    out.mkdir()
    info = package_merged(str(run), str(out))
    assert info["files"] == [] and '"method":"mtp"' in info["vllm_command"]
    assert sorted(os.listdir(out)) == ["drafter.json"]


def test_no_drafter_means_no_packaging(tmp_path):
    from core.export.drafter_export import package_gguf, package_merged

    run = tmp_path / "run"
    run.mkdir()
    assert package_merged(str(run), str(tmp_path)) is None
    assert package_gguf(str(run), None, str(tmp_path), []) == (None, [])
    # A manifest whose head file is gone is not trusted.
    (tmp_path / "x").mkdir()
    mtp = _training_dir(tmp_path / "x", "mtp")
    (mtp / "mtp_head.safetensors").unlink()
    assert package_merged(str(mtp), str(tmp_path)) is None


def test_gguf_export_marks_trained_mtp(tmp_path):
    from core.export.drafter_export import gguf_drafter_sidecar, package_gguf

    run = _training_dir(tmp_path, "mtp")
    gdir = tmp_path / "gguf"
    gdir.mkdir()
    gguf = gdir / "model.Q8_0.gguf"
    gguf.write_bytes(b"GGUF")
    info, files = package_gguf(str(run), None, str(gdir), [str(gguf)])
    assert info["mtp_trained"] is True and files == [str(gdir / "drafter.json")]
    assert gguf_drafter_sidecar(str(gguf))["mtp_trained"] is True
    # Another GGUF in the folder is not claimed.
    assert gguf_drafter_sidecar(str(gdir / "other.gguf")) is None


def test_gguf_export_converts_eagle3_beside(tmp_path, monkeypatch):
    from core.export import drafter_export

    calls = {}

    def convert(
        drafter_dir,
        target_dir,
        out_path,
        outtype = "f16",
    ):
        calls.update(drafter_dir = drafter_dir, target_dir = target_dir, outtype = outtype)
        Path(out_path).write_bytes(b"GGUF")

    mod = types.ModuleType("unsloth_zoo.speculative")
    mod.convert_eagle3_to_gguf = convert
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.speculative", mod)
    run = _training_dir(tmp_path, "eagle3")
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}")
    gdir = tmp_path / "gguf"
    gdir.mkdir()
    gguf = gdir / "model.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    info, files = drafter_export.package_gguf(str(run), str(merged), str(gdir), [str(gguf)])
    assert calls["target_dir"] == str(merged) and calls["outtype"] == "f16"
    assert (gdir / "eagle3-model-F16.gguf").is_file()
    assert info["llama_cpp_args"][:4] == [
        "--model-draft",
        "eagle3-model-F16.gguf",
        "--spec-type",
        "draft-eagle3",
    ]
    assert drafter_export.eagle3_sidecar_path(str(gguf)) == str(gdir / "eagle3-model-F16.gguf")
    with pytest.raises(RuntimeError):
        drafter_export.package_gguf(str(run), None, str(gdir), [str(gguf)])


# ---------------------------------------------------------------- vLLM launch


def test_vllm_speculative_args(tmp_path):
    from core.inference.engine_adapters import vllm_speculative_args

    assert vllm_speculative_args(str(tmp_path)) == []
    assert vllm_speculative_args("org/repo") == []
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "mtp", "draft_steps": 2}))
    assert vllm_speculative_args(str(tmp_path)) == [
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
    ]
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "eagle3"}))
    assert vllm_speculative_args(str(tmp_path)) == []  # drafter folder missing
    (tmp_path / "eagle3").mkdir()
    (tmp_path / "eagle3" / "config.json").write_text("{}")
    args = vllm_speculative_args(str(tmp_path))
    assert json.loads(args[1]) == {
        "method": "eagle3",
        "model": str(tmp_path / "eagle3"),
        "num_speculative_tokens": 3,
    }


@pytest.mark.parametrize("snapshot", [None, "/snapshots/abc123"])
def test_phase_mtp_saves_head_beside_adapter(fake_zoo, tmp_path, monkeypatch, snapshot):
    drafter, trainer = fake_zoo
    monkeypatch.setattr(
        drafter,
        "capability_in_process",
        lambda name, token = None, **_: {
            "supported": True,
            "strategy": "mtp",
            "has_native_mtp": True,
            "vllm": True,
            "llama_cpp": True,
        },
    )
    monkeypatch.setattr(drafter, "_mtp_family", lambda config: ("qwen3_5", "drafts"))
    searched = []
    monkeypatch.setattr(
        drafter,
        "_find_mtp_source",
        lambda source, config, repo_id = None: searched.append(source) or "org/base",
    )
    calls = {}
    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = types.SimpleNamespace(
        get_mtp_head = lambda model, source = None, random_init = False: calls.update(
            source = source, random_init = random_init
        )
        or object()
    )
    unsloth.MTPTrainingArguments = _Args
    unsloth.UnslothMTPTrainer = _FakeDrafterTrainer
    save = types.ModuleType("unsloth.save")
    save._save_mtp_head_beside_adapter = lambda model, d: Path(
        d, "mtp_head.safetensors"
    ).write_bytes(b"h")
    save._stitch_mtp_head = lambda model, d: calls.update(stitched = d)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", save)
    (tmp_path / "adapter_config.json").write_text("{}")
    config = _config(draft_steps = 2)
    if snapshot:
        config["model_snapshot_path"] = snapshot
    manifest = drafter.run_drafter_phase(trainer, config, str(tmp_path), _Queue(), lambda *a: None)
    # The head comes from the checkpoint this run loaded (a pinned snapshot), not the repo's tip.
    assert searched == [snapshot or "org/base"]
    assert calls == {"source": "org/base", "random_init": False}
    assert manifest["strategy"] == "mtp" and manifest["files"] == ["mtp_head.safetensors"]
    assert manifest["vllm"] == {"method": "mtp", "num_speculative_tokens": 2}
    assert (tmp_path / "mtp_head.safetensors").is_file()


def test_capability_refuses_without_the_zoo_drafter_trainers(monkeypatch):
    """An unsloth_zoo without the drafter trainers must not advertise a drafter it cannot train."""
    from core.training import drafter

    monkeypatch.setitem(sys.modules, "unsloth_zoo.speculative.eagle3", None)
    monkeypatch.setattr(
        drafter, "_text_config", lambda *a, **k: pytest.fail("config read before the zoo check")
    )
    cap = drafter.capability_in_process("unsloth/Qwen3.5-2B")
    assert cap["supported"] is False and cap["strategy"] is None
    assert "unsloth_zoo" in cap["reason"]


class _CharTokenizer:
    """One token per character; a ChatML-style template."""

    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(
        self,
        turns,
        add_generation_prompt = False,
        tokenize = False,
    ):
        text = "".join(f"<u>{t['content']}</u>" for t in turns)
        return text + ("<a>" if add_generation_prompt else "")

    def __call__(
        self,
        text,
        add_special_tokens = False,
    ):
        return {"input_ids": [ord(c) for c in text]}


def test_regenerate_finds_prompts_in_rows_trained_on_the_whole_sequence():
    """Studio trains on full sequences by default (labels unmasked): the prompt still ends at the assistant header."""
    from core.training.drafter import _assistant_header, _prompt_end

    tok = _CharTokenizer()
    header = _assistant_header(tok)
    assert header == [ord(c) for c in "<a>"]
    row = [ord(c) for c in "<u>hi</u><a>answer"]
    assert _prompt_end(row, list(row), header) == len("<u>hi</u><a>")
    masked = [-100] * 5 + row[5:]
    assert _prompt_end(row, masked, header) == 5
    assert _prompt_end([ord(c) for c in "no header here"], [1] * 14, header) is None


@pytest.mark.parametrize("answers, used", [([], "dataset"), (None, "regenerate")])
def test_manifest_records_the_data_source_actually_used(
    fake_zoo, tmp_path, monkeypatch, answers, used
):
    """A regenerate request that found nothing to answer trains on the dataset and must say so."""
    drafter, trainer = fake_zoo
    regenerated = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}] * 16
    monkeypatch.setattr(
        drafter, "regenerate_examples", lambda *a, **k: regenerated if answers is None else answers
    )
    q = _Queue()
    manifest = drafter.run_drafter_phase(
        trainer,
        _config(data_source = "regenerate"),
        str(tmp_path),
        q,
        lambda eq, m: eq.put({"type": "status", "message": m}),
    )
    assert manifest["data_source"] == used
    if used == "dataset":
        assert any("using the dataset's answers" in e.get("message", "") for e in q)


def test_route_capability_probes_in_a_child_process(monkeypatch):
    """The API server never imports unsloth_zoo for the capability (it raced the torch warm-up there)."""
    import subprocess
    from core.training import drafter

    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        out = {
            "model_name": argv[-1],
            "has_native_mtp": True,
            "strategy": "mtp",
            "supported": True,
            "reason": None,
            "vllm": True,
            "llama_cpp": True,
        }
        return subprocess.CompletedProcess(
            argv, 0, "Unsloth banner\nDRAFTER_CAPABILITY " + json.dumps(out) + "\n", ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    drafter._CAPABILITY_CACHE.clear()
    cap = drafter.drafter_capability("org/m", False)
    assert cap["strategy"] == "mtp" and cap["supported"]
    argv, kw = calls[0]
    assert argv[0] == sys.executable and "org/m" == argv[-1]
    assert kw["env"][drafter._TOKEN_ENV] == "0"  # anonymous, and never on the command line
    assert drafter.drafter_capability("org/m", False) == cap and len(calls) == 1  # cached
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "boom")
    )
    assert drafter.drafter_capability("org/other")["supported"] is False


def test_capability_child_process_runs_for_real(tmp_path):
    """The probe script itself imports and answers (local config, no network)."""
    from core.training import drafter

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vocab_size": 256,
            }
        )
    )
    pytest.importorskip("unsloth_zoo.speculative.eagle3")
    drafter._CAPABILITY_CACHE.clear()
    cap = drafter.drafter_capability(str(tmp_path))
    assert cap["strategy"] == "eagle3" and cap["supported"], cap


def test_regenerated_rows_stay_within_max_seq_length():
    import torch
    from core.training.drafter import regenerate_examples

    asked = []

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, attention_mask, max_new_tokens, **_):
            asked.append(max_new_tokens)
            return torch.cat([input_ids, torch.full((input_ids.shape[0], max_new_tokens), 7)], 1)

    tok = _CharTokenizer()
    short = [ord(c) for c in "<u>hi</u><a>ok"]
    long = [ord(c) for c in "<u>" + "x" * 40 + "</u><a>ok"]
    rows = [{"input_ids": r, "labels": list(r)} for r in (short, long)]
    out = regenerate_examples(_Model(), tok, rows, batch_size = 1, max_seq_length = 32)
    assert len(out) == 1 and len(out[0]["input_ids"]) == 32  # the long prompt has no room left
    assert asked == [32 - len("<u>hi</u><a>")]


def test_regenerate_finds_prompts_in_dataset_template_rows():
    """Rows rendered with a dataset template (Alpaca) carry "### Response:", not the chat header."""
    from core.training.drafter import _prompt_end, _response_markers

    tok = _CharTokenizer()
    markers = _response_markers(tok)
    row = [ord(c) for c in "### Instruction:\nAdd\n\n### Response:\ndef add(a, b): pass"]
    end = _prompt_end(row, list(row), markers)
    assert "".join(map(chr, row[:end])).endswith("### Response:\n")
    chat = [ord(c) for c in "<u>hi</u><a>hello"]
    assert "".join(map(chr, chat[_prompt_end(chat, list(chat), markers) :])) == "hello"


def test_reused_export_folder_drops_a_previous_drafter(tmp_path):
    from core.export.drafter_export import package_gguf, package_merged

    run = _training_dir(tmp_path, "eagle3")
    out = tmp_path / "out"
    out.mkdir()
    package_merged(str(run), str(out))
    assert (out / "drafter.json").is_file() and (out / "eagle3").is_dir()
    plain = tmp_path / "plain"
    plain.mkdir()
    assert package_merged(str(plain), str(out)) is None
    assert not (out / "drafter.json").exists() and not (out / "eagle3").exists()

    gdir = tmp_path / "gguf"
    gdir.mkdir()
    (gdir / "model.Q8_0.gguf").write_bytes(b"GGUF")
    (gdir / "eagle3-model-F16.gguf").write_bytes(b"GGUF")
    (gdir / "drafter.json").write_text(
        json.dumps(
            {"strategy": "eagle3", "ggufs": ["model.Q8_0.gguf"], "files": ["eagle3-model-F16.gguf"]}
        )
    )
    # A manifest about other GGUFs in the folder is left alone.
    assert package_gguf(str(plain), None, str(gdir), [str(gdir / "other.gguf")]) == (None, [])
    assert (gdir / "drafter.json").is_file()
    assert package_gguf(str(plain), None, str(gdir), [str(gdir / "model.Q8_0.gguf")]) == (None, [])
    assert not (gdir / "drafter.json").exists() and not (gdir / "eagle3-model-F16.gguf").exists()


def test_compressed_exports_carry_eagle3_but_not_mtp(tmp_path):
    from core.export.drafter_export import package_merged

    (tmp_path / "m").mkdir()
    (tmp_path / "e").mkdir()
    out = tmp_path / "out"
    out.mkdir()
    assert (
        package_merged(str(_training_dir(tmp_path / "m", "mtp")), str(out), allow_mtp = False) is None
    )
    eagle = package_merged(str(_training_dir(tmp_path / "e", "eagle3")), str(out), allow_mtp = False)
    assert eagle["strategy"] == "eagle3" and (out / "eagle3" / "config.json").is_file()


def test_engine_support_gates_vllm_and_llama_cpp(tmp_path):
    from core.export.drafter_export import package_gguf, package_merged
    from core.inference.engine_adapters import vllm_speculative_args

    run = _training_dir(tmp_path, "mtp")
    manifest = json.loads((run / "drafter.json").read_text())
    manifest["engines"] = {"vllm": False, "llama_cpp": True}
    (run / "drafter.json").write_text(json.dumps(manifest))
    out = tmp_path / "out"
    out.mkdir()
    info = package_merged(str(run), str(out))
    assert info["vllm_command"] is None and vllm_speculative_args(str(out)) == []
    manifest["engines"] = {"vllm": True, "llama_cpp": False}
    (run / "drafter.json").write_text(json.dumps(manifest))
    gdir = tmp_path / "gguf"
    gdir.mkdir()
    assert package_gguf(str(run), None, str(gdir), [str(gdir / "m.gguf")]) == (None, [])


def test_lora_export_carries_the_mtp_manifest(tmp_path):
    from core.export.drafter_export import package_lora, training_manifest

    run = _training_dir(tmp_path, "mtp")
    (run / "adapter_config.json").write_text("{}")
    out = tmp_path / "adapter"
    out.mkdir()
    assert package_lora(str(run), str(out)) is None  # the save wrapper wrote no head here
    (out / "mtp_head.safetensors").write_bytes(b"h")
    (out / "adapter_config.json").write_text("{}")
    info = package_lora(str(run), str(out))
    assert info["strategy"] == "mtp" and "adapter_dir" not in info
    assert training_manifest(str(out))["strategy"] == "mtp"


def test_capability_probe_is_single_flight(monkeypatch):
    import subprocess
    import threading
    from core.training import drafter

    calls = []
    gate = threading.Event()

    def fake_run(argv, **kw):
        calls.append(argv[-1])
        gate.wait(5)
        out = {
            "model_name": argv[-1],
            "has_native_mtp": False,
            "strategy": "eagle3",
            "supported": True,
            "reason": None,
            "vllm": True,
            "llama_cpp": True,
        }
        return subprocess.CompletedProcess(
            argv, 0, "DRAFTER_CAPABILITY " + json.dumps(out) + "\n", ""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    drafter._CAPABILITY_CACHE.clear()
    results = []
    threads = [
        threading.Thread(
            target = lambda: results.append(drafter.drafter_capability("org/same", False))
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert calls == ["org/same"] and len(results) == 6


def test_remote_adapter_resolves_to_its_base(monkeypatch):
    from core.training import drafter
    from utils.models import model_config

    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda name, hf_token = None: "org/base" if name == "me/adapter" else None,
    )
    assert drafter.capability_source("me/adapter", {}) == "org/base"
    assert drafter.capability_source("org/plain", {}) == "org/plain"


def test_managed_accounts_probe_only_their_paths_and_never_with_the_owner_token(monkeypatch):
    import asyncio

    from core.training import account_jobs, drafter
    import routes.training as route

    seen = []
    monkeypatch.setattr(account_jobs, "managed_account", lambda: True)
    monkeypatch.setattr(route, "managed_account", lambda: True)
    monkeypatch.setattr(
        drafter,
        "drafter_capability",
        lambda name, token: seen.append((name, token)) or {"supported": True},
    )
    visible = {"org/model"}

    def authorize(name, repo_type = "model"):
        if name not in visible:
            raise route.HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(route, "_authorize_cache_fallback", authorize)
    refused = asyncio.run(
        route.get_drafter_capability("../../accounts/other/outputs/run", "u", False, None)
    )
    assert refused["supported"] is False and "not available" in refused["reason"] and seen == []
    # Another tenant's model in the shared cache: refused before anything is read.
    hidden = asyncio.run(route.get_drafter_capability("org/private", "u", False, None))
    assert hidden["supported"] is False and seen == []
    asyncio.run(route.get_drafter_capability("org/model", "u", False, None))
    assert seen == [("org/model", False)]
    # The caller's own token (X-Unsloth-HF-Token) reaches the probe for a private / gated model.
    asyncio.run(route.get_drafter_capability("org/model", "u", False, "hf_mine"))
    assert seen[-1] == ("org/model", "hf_mine")
    # A picker inventory handle is probed as the path it stands for; the answer keeps the handle.
    import models.inference as inference_models

    monkeypatch.setattr(
        inference_models,
        "resolve_inventory_handle",
        lambda value: "org/model" if value == "ref:abc" else value,
    )
    answer = asyncio.run(route.get_drafter_capability("ref:abc", "u", False, None))
    assert seen[-1] == ("org/model", False) and answer["model_name"] == "ref:abc"


def test_adapter_base_keeps_the_anonymous_token(monkeypatch):
    """False (managed accounts, API keys) must not become None (= the host's ambient token)."""
    import utils.models.model_config as model_config
    from core.training.drafter import _adapter_base

    seen = []
    monkeypatch.setattr(
        model_config,
        "get_base_model_from_lora_identifier",
        lambda name, hf_token = None: seen.append(hf_token),
    )
    _adapter_base("org/adapter", False)
    assert seen == [False]


def test_lora_manifest_for_a_hub_push(tmp_path):
    """A Hub push saves the adapter elsewhere: the caller says whether the head went with it."""
    from core.export.drafter_export import package_lora

    run = _training_dir(tmp_path, "mtp")
    staging = tmp_path / "staging"
    staging.mkdir()
    assert package_lora(str(run), str(staging), head_saved = False) is None
    assert not (staging / "drafter.json").exists()
    info = package_lora(str(run), str(staging), head_saved = True)
    assert info["strategy"] == "mtp" and (staging / "drafter.json").exists()
    src = _source("core/export/export.py")
    push = src.index(
        "self.current_tokenizer.push_to_hub(repo_id, token = hf_token, private = private)\n"
    )
    assert "self._push_drafter_lora_manifest(hf_api, repo_id)" in src[push : push + 200]


def test_mtp_retraining_defaults_to_a_gentle_learning_rate():
    """The native head is pretrained: the default must not overwrite it (1e-4 made drafts worse than stock)."""
    from core.training.drafter import DEFAULT_LR
    assert DEFAULT_LR["mtp"] <= 2e-5 < DEFAULT_LR["eagle3"]


def test_wsl_guest_paths_are_read_from_the_host(tmp_path, monkeypatch):
    """Windows vLLM runs in WSL on /mnt/<drive>/...: the manifest is read from the Windows path,
    the drafter path handed to vLLM stays the guest's."""
    import json

    from core.inference import engine_adapters

    assert (
        engine_adapters._wsl_guest_to_host("/mnt/c/Users/me/out", nt = True) == "C:\\Users\\me\\out"
    )
    assert (
        engine_adapters._wsl_guest_to_host("/mnt/c/Users/me/out", nt = False) == "/mnt/c/Users/me/out"
    )
    (tmp_path / "eagle3").mkdir()
    (tmp_path / "eagle3" / "config.json").write_text("{}")
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "eagle3", "draft_steps": 3}))
    monkeypatch.setattr(engine_adapters, "_wsl_guest_to_host", lambda p, nt = None: str(tmp_path))
    args = engine_adapters.vllm_speculative_args("/mnt/c/out")
    assert json.loads(args[1])["model"] == "/mnt/c/out/eagle3"


def test_probe_child_gets_exactly_the_callers_token(monkeypatch):
    import subprocess

    from core.training import drafter

    seen = {}

    def run(argv, env, **kwargs):
        seen.update(env)
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("HF_TOKEN", "hf_owner")
    drafter._probe_capability("org/m", "hf_caller", ("org/m", "k"))
    assert seen.get("HF_TOKEN") == "hf_caller"
    seen.clear()
    drafter._probe_capability("org/m", False, ("org/m", False))
    assert "HF_TOKEN" not in seen


def test_a_failed_eagle3_launch_retries_without_the_drafter():
    src = _source("core/inference/llama_cpp.py")
    assert (
        '_spec_requested_eagle3 = any("draft-eagle3" in str(t).lower() for t in spec_flags)' in src
    )
    gate = src.index("_spec_cpu_replay_cmd: Optional[List[str]] = None")
    assert "or _spec_requested_eagle3" in src[gate : gate + 600]


def test_a_run_without_a_drafter_clears_a_previous_runs(tmp_path):
    from core.training.drafter import discard_drafter_artifacts

    (tmp_path / "eagle3").mkdir()
    (tmp_path / "eagle3" / "config.json").write_text("{}")
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "eagle3"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    discard_drafter_artifacts(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["adapter_model.safetensors"]
    src = _source("core/training/worker.py")
    assert "discard_drafter_artifacts(output_dir)" in src


def test_a_hub_adapter_checkpoint_keeps_its_manifest(tmp_path, monkeypatch):
    import huggingface_hub

    from core.export import drafter_export

    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "drafter.json").write_text(json.dumps({"strategy": "mtp", "draft_steps": 2}))
    (snap / "adapter_config.json").write_text("{}")
    (snap / "mtp_head.safetensors").write_bytes(b"h")
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, name, **k: str(snap / name)
    )
    assert drafter_export.training_manifest("me/adapter")["draft_steps"] == 2
    assert drafter_export.training_manifest(str(tmp_path / "missing")) is None


def test_base_exports_package_the_drafter():
    src = _source("core/export/export.py")
    fn = src[
        src.index("def export_base_model") : src.index(
            "def ", src.index("def export_base_model") + 10
        )
    ]
    assert fn.count("self._package_drafter_merged(") == 2


@pytest.mark.parametrize("adapter", [True, False])
def test_stop_while_saving(fake_zoo, tmp_path, monkeypatch, adapter):
    """A sidecar head is dropped; one already stitched into a full fine-tune's shards stays recorded."""
    drafter, trainer = fake_zoo
    monkeypatch.setattr(
        drafter,
        "capability_in_process",
        lambda name, token = None, **_: {
            "supported": True,
            "strategy": "mtp",
            "has_native_mtp": True,
            "vllm": True,
            "llama_cpp": True,
        },
    )
    monkeypatch.setattr(drafter, "_mtp_family", lambda config: ("qwen3_5", "drafts"))
    monkeypatch.setattr(drafter, "_find_mtp_source", lambda *a: "org/base")
    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = types.SimpleNamespace(get_mtp_head = lambda *a, **k: object())
    unsloth.MTPTrainingArguments = _Args
    unsloth.UnslothMTPTrainer = _FakeDrafterTrainer

    def save_then_stop(
        model,
        d,
        name = "mtp_head.safetensors",
    ):
        Path(d, name).write_bytes(b"h")
        trainer.should_stop = True

    save = types.ModuleType("unsloth.save")
    save._save_mtp_head_beside_adapter = save_then_stop
    save._stitch_mtp_head = lambda model, d: save_then_stop(model, d, "model.safetensors")
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.save", save)
    if adapter:
        (tmp_path / "adapter_config.json").write_text("{}")
    manifest = drafter.run_drafter_phase(
        trainer, _config(), str(tmp_path), _Queue(), lambda *a: None
    )
    if adapter:
        assert manifest is None and not (tmp_path / "mtp_head.safetensors").exists()
        assert not (tmp_path / "drafter.json").exists()
    else:
        assert manifest["files"] == [] and (tmp_path / "drafter.json").is_file()


def test_hub_manifest_recovery_uses_the_load_credential(monkeypatch, tmp_path):
    """A private Hub adapter's drafter files are fetched with the token the checkpoint loaded with."""
    from core.export import drafter_export

    calls = []

    def fake_download(
        repo_id,
        filename,
        token = None,
    ):
        calls.append((filename, token))
        path = tmp_path / filename
        path.write_text(json.dumps({"strategy": "mtp"}) if filename == "drafter.json" else "{}")
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert drafter_export.training_manifest("owner/private-adapter", "hf_secret") == {
        "strategy": "mtp"
    }
    assert calls and all(token == "hf_secret" for _, token in calls)


def test_pinned_quantized_snapshot_keeps_the_original_repo_fallback(monkeypatch):
    """A snapshot path has lost the -bnb-4bit suffix; the repo id still finds the original's head."""
    from core.training import drafter

    tried = []
    mtp = types.ModuleType("unsloth_zoo.speculative.mtp")
    mtp.find_mtp_tensor_names = lambda name, config = None: tried.append(name) or name == "org/Model"
    monkeypatch.setitem(sys.modules, "unsloth_zoo.speculative.mtp", mtp)
    snapshot = "/hf/models--org--Model-bnb-4bit/snapshots/abc123"
    assert drafter._find_mtp_source(snapshot, None, "org/Model-bnb-4bit") == "org/Model"
    assert tried == [snapshot, "org/Model-bnb-4bit", "org/Model"]
    tried.clear()
    assert drafter._find_mtp_source(snapshot, None) is None and tried == [snapshot]


def test_export_reattaches_a_full_fine_tunes_stitched_head(tmp_path, monkeypatch):
    """transformers drops the stitched mtp.* tensors on load; the export backend attaches them back."""
    from core.export import export as export_module

    calls = []
    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = types.SimpleNamespace(
        get_mtp_head = lambda model, source = None: calls.append((model, source))
    )
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    backend = object.__new__(export_module.ExportBackend)
    backend.current_model = "model"
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "mtp"}))
    backend._attach_full_checkpoint_mtp_head(str(tmp_path))
    assert calls == [("model", str(tmp_path))]
    (tmp_path / "adapter_config.json").write_text("{}")
    backend._attach_full_checkpoint_mtp_head(str(tmp_path))
    (tmp_path / "adapter_config.json").unlink()
    (tmp_path / "drafter.json").write_text(json.dumps({"strategy": "eagle3"}))
    backend._attach_full_checkpoint_mtp_head(str(tmp_path))
    assert len(calls) == 1


def test_export_reattaches_the_head_of_a_hub_full_fine_tune(tmp_path, monkeypatch):
    """A full fine-tune loaded by Hub id: its manifest is read from the Hub with the load's token and
    the head is attached from the snapshot the load cached, at the loaded commit."""
    from core.export import export as export_module
    import huggingface_hub

    calls = []
    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = types.SimpleNamespace(
        get_mtp_head = lambda model, source = None: calls.append((model, source))
    )
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    downloads, snapshots = [], []

    def fake_download(
        repo_id,
        filename,
        token = None,
        revision = None,
        local_files_only = False,
    ):
        if local_files_only:
            snapshots.append((repo_id, filename, revision, token))
            (snapshot / filename).write_text("{}")
            return str(snapshot / filename)
        downloads.append((repo_id, filename, token))
        if filename != "drafter.json":
            raise FileNotFoundError(filename)
        (manifest_dir / filename).write_text(json.dumps({"strategy": "mtp"}))
        return str(manifest_dir / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    model = types.SimpleNamespace(config = types.SimpleNamespace(_commit_hash = "abc123"))
    backend = object.__new__(export_module.ExportBackend)
    backend.current_model = model
    backend._checkpoint_token = "hf_secret"
    backend._attach_full_checkpoint_mtp_head("org/trained-full-mtp")
    assert calls == [(model, str(snapshot))]
    assert snapshots == [("org/trained-full-mtp", "config.json", "abc123", "hf_secret")]
    assert downloads and all(token == "hf_secret" for _, _, token in downloads)

    # No manifest on the Hub (an ordinary full checkpoint): nothing is attached.
    calls.clear()
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("drafter.json")),
    )
    backend._attach_full_checkpoint_mtp_head("org/plain-full")
    assert calls == []


def _hub_full_fine_tune_backend(tmp_path, monkeypatch, local_files_only):
    """An export backend holding a full fine-tune loaded by Hub id at commit abc123, whose manifest
    and config are cached; records every hf_hub_download call."""
    from core.export import export as export_module
    import huggingface_hub

    attached = []
    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = types.SimpleNamespace(
        get_mtp_head = lambda model, source = None: attached.append(source)
    )
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "drafter.json").write_text(json.dumps({"strategy": "mtp"}))
    (snapshot / "config.json").write_text("{}")
    calls = []

    def fake_download(
        repo_id,
        filename,
        token = None,
        revision = None,
        local_files_only = False,
    ):
        calls.append((filename, revision, local_files_only))
        if not (snapshot / filename).is_file():
            raise FileNotFoundError(filename)
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    backend = object.__new__(export_module.ExportBackend)
    backend.current_model = types.SimpleNamespace(
        config = types.SimpleNamespace(_commit_hash = "abc123")
    )
    backend._checkpoint_token = None
    backend._checkpoint_local_files_only = local_files_only
    return backend, calls, attached, snapshot


def test_hub_full_fine_tune_manifest_is_read_at_the_loaded_commit(tmp_path, monkeypatch):
    """The manifest and the head's shards come from the same commit, not the branch head."""
    backend, calls, attached, snapshot = _hub_full_fine_tune_backend(tmp_path, monkeypatch, False)
    backend._attach_full_checkpoint_mtp_head("org/trained-full-mtp")
    assert attached == [str(snapshot)]
    assert calls and all(revision == "abc123" for _, revision, _ in calls)


def test_offline_hub_full_fine_tune_keeps_the_manifest_probe_offline(tmp_path, monkeypatch):
    """A cached load chosen because the Hub is unreachable must not touch the network for drafter.json."""
    backend, calls, attached, snapshot = _hub_full_fine_tune_backend(tmp_path, monkeypatch, True)
    backend._attach_full_checkpoint_mtp_head("org/trained-full-mtp")
    assert attached == [str(snapshot)]
    assert calls and all(offline for _, _, offline in calls)


def _package_through_backend(backend, path, out):
    out.mkdir()
    if path == "merged":
        backend._package_drafter_merged(str(out), None)
    elif path == "gguf":
        gguf = out / "model.Q8_0.gguf"
        gguf.write_bytes(b"g")
        backend._package_drafter_gguf(str(out), str(out), [str(gguf)])
    else:
        assert backend._package_drafter_lora(str(out), head_saved = True)
    return json.loads((out / "drafter.json").read_text())


@pytest.mark.parametrize("path", ["merged", "gguf", "lora"])
@pytest.mark.parametrize("offline", [False, True])
def test_drafter_packaging_reads_the_manifest_at_the_loaded_commit(
    tmp_path, monkeypatch, path, offline
):
    """Merged, GGUF and LoRA packaging read a Hub id's manifest at the commit the load resolved
    (not the branch head) and stay offline when the load did."""
    backend, calls, _, _ = _hub_full_fine_tune_backend(tmp_path, monkeypatch, offline)
    backend.current_checkpoint = "org/trained-full-mtp"
    info = _package_through_backend(backend, path, tmp_path / "out")
    assert info["strategy"] == "mtp"
    assert calls and all(revision == "abc123" for _, revision, _ in calls)
    assert all(local is offline for _, _, local in calls)


_ADAPTER_SHA = "a" * 40


@pytest.mark.parametrize("path", ["merged", "gguf", "lora"])
@pytest.mark.parametrize("adapter_commit", [_ADAPTER_SHA, None])
def test_hub_adapter_packaging_never_pins_the_base_commit(
    tmp_path, monkeypatch, path, adapter_commit
):
    """A Hub LoRA adapter's current_model.config is the base's, so its _commit_hash names the base
    repo: the adapter's manifest is read at the adapter's own commit, or unpinned when unknown."""
    backend, calls, _, _ = _hub_full_fine_tune_backend(tmp_path, monkeypatch, False)
    backend.current_checkpoint = "org/trained-adapter"
    backend.is_peft = True
    backend._adapter_commit = ("org/trained-adapter", adapter_commit)
    info = _package_through_backend(backend, path, tmp_path / "out")
    assert info["strategy"] == "mtp"
    assert calls and all(revision == adapter_commit for _, revision, _ in calls)


def test_hub_adapter_commit_of_another_checkpoint_is_not_reused(tmp_path, monkeypatch):
    backend, calls, _, _ = _hub_full_fine_tune_backend(tmp_path, monkeypatch, False)
    backend.current_checkpoint = "org/other-adapter"
    backend.is_peft = True
    backend._adapter_commit = ("org/trained-adapter", _ADAPTER_SHA)
    assert backend._loaded_commit() is None


def test_full_fine_tune_keeps_the_base_commit(tmp_path, monkeypatch):
    backend, _, _, _ = _hub_full_fine_tune_backend(tmp_path, monkeypatch, False)
    backend.current_checkpoint = "org/trained-full-mtp"
    backend.is_peft = False
    backend._adapter_commit = None
    assert backend._loaded_commit() == "abc123"


def test_hub_adapter_commit_comes_from_the_local_cache(tmp_path, monkeypatch):
    """The adapter's commit is the cached snapshot its adapter_config.json sits in, read offline."""
    import huggingface_hub
    from core.export import export as export_module

    repo = tmp_path / "models--org--trained-adapter"
    (repo / "snapshots" / _ADAPTER_SHA).mkdir(parents = True)
    (repo / "snapshots" / _ADAPTER_SHA / "adapter_config.json").write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(_ADAPTER_SHA)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    assert export_module._cached_hub_adapter_commit("org/trained-adapter") == _ADAPTER_SHA
    assert export_module._cached_hub_adapter_commit("org/not-cached") is None
    assert export_module._cached_hub_adapter_commit(str(tmp_path)) is None


def test_load_records_the_adapter_commit_for_a_peft_checkpoint():
    src = _source("core/export/export.py")
    assert "(checkpoint_path, _cached_hub_adapter_commit(checkpoint_path))" in src


def test_drafter_packaging_without_a_hub_load_passes_no_revision(tmp_path, monkeypatch):
    """A local checkpoint (no resolved commit, online load) calls the helpers exactly as before."""
    from core.export import drafter_export
    from core.export import export as export_module

    seen = []
    for name in ("package_merged", "package_lora"):
        monkeypatch.setattr(drafter_export, name, lambda *a, **k: seen.append(sorted(k)) or None)
    monkeypatch.setattr(
        drafter_export, "package_gguf", lambda *a, **k: seen.append(sorted(k)) or (None, [])
    )
    backend = object.__new__(export_module.ExportBackend)
    backend.current_model = types.SimpleNamespace(config = types.SimpleNamespace())
    backend.current_checkpoint = str(tmp_path)
    backend._checkpoint_token = None
    backend._checkpoint_local_files_only = False
    backend._package_drafter_merged(str(tmp_path), None)
    backend._package_drafter_gguf(str(tmp_path), str(tmp_path), [])
    backend._package_drafter_lora(str(tmp_path), head_saved = True)
    assert seen == [["allow_mtp", "token", "verifier"], ["token"], ["token"]]


def test_load_records_its_offline_decision_for_the_manifest_probe():
    src = _source("core/export/export.py")
    assert "self._checkpoint_local_files_only = bool(local_files_only)" in src


def test_in_place_eagle3_export_keeps_the_only_drafter_copy(tmp_path):
    """An export into the training folder itself must not delete eagle3/ before copying it."""
    from core.export.drafter_export import package_merged

    run = _training_dir(tmp_path, "eagle3")
    info = package_merged(str(run), str(run))
    assert info["strategy"] == "eagle3"
    assert (run / "eagle3" / "model.safetensors").read_bytes() == b"w"


def test_an_inconclusive_probe_is_marked_for_a_retry(monkeypatch):
    import subprocess

    from core.training import drafter

    monkeypatch.setattr(drafter, "_CAPABILITY_CACHE", {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("probe", 300)),
    )
    result = drafter.drafter_capability("org/flaky-model")
    assert result["supported"] is False and result["inconclusive"] is True
    assert not drafter._CAPABILITY_CACHE
