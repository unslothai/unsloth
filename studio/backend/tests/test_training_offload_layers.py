# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offload layers in Unsloth Studio training: the request fields, the VRAM budget, the kwargs each
load and LoRA path receives, and the live snapshot the training view polls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.training.training import _build_training_worker_config
from core.training.worker import _training_vram_budget_fraction, _with_vram_budget_hint
from models.training import TrainingStartRequest

GIB = 1024**3


def _request(**kwargs) -> TrainingStartRequest:
    return TrainingStartRequest(
        model_name = "unsloth/Qwen3-0.6B", training_type = "LoRA/QLoRA", format_type = "alpaca", **kwargs
    )


def test_offload_fields_default_to_off():
    r = _request()
    assert r.offload_layers == 0 and r.offload_vram_gb is None and r.prefetch_depth == 2


@pytest.mark.parametrize("value", [0, 14, "auto"])
def test_offload_layers_takes_a_count_or_auto(value):
    assert _request(offload_layers = value).offload_layers == value


@pytest.mark.parametrize("value", [-1, 2000, "max", 1.5])
def test_offload_layers_refuses_what_core_cannot_use(value):
    with pytest.raises(ValidationError):
        _request(offload_layers = value)


@pytest.mark.parametrize("value", [0, 9, "fast"])
def test_prefetch_depth_refuses_out_of_range(value):
    with pytest.raises(ValidationError):
        _request(prefetch_depth = value)


def test_vram_budget_must_be_positive():
    assert _request(offload_vram_gb = 11.5).offload_vram_gb == 11.5
    with pytest.raises(ValidationError):
        _request(offload_vram_gb = 0)


def test_worker_config_carries_the_offload_fields():
    cfg = _build_training_worker_config(
        {
            "model_name": "org/model",
            "offload_layers": "auto",
            "offload_vram_gb": 12.0,
            "prefetch_depth": "auto",
        }
    )
    assert (cfg["offload_layers"], cfg["offload_vram_gb"], cfg["prefetch_depth"]) == (
        "auto",
        12.0,
        "auto",
    )
    off = _build_training_worker_config({"model_name": "org/model"})
    assert (off["offload_layers"], off["offload_vram_gb"], off["prefetch_depth"]) == (0, None, 2)


def test_budget_fraction_holds_the_run_to_its_gib():
    assert _training_vram_budget_fraction(8, 32 * GIB) == pytest.approx(0.25)
    # Never looser than the OOM guard's cap.
    assert _training_vram_budget_fraction(31, 32 * GIB, current = 0.8) == pytest.approx(0.8)
    assert _training_vram_budget_fraction(64, 32 * GIB) == pytest.approx(1.0)
    for budget in (None, 0, -2, "x", float("nan")):
        assert _training_vram_budget_fraction(budget, 32 * GIB) is None
    assert _training_vram_budget_fraction(8, 0) is None


def _trainer(
    offload_layers = 0,
    prefetch_depth = 2,
    model = None,
):
    from core.training.trainer import UnslothTrainer

    t = UnslothTrainer.__new__(UnslothTrainer)
    t._offload_layers, t._prefetch_depth, t.model = offload_layers, prefetch_depth, model
    t._offload_plan_shape = {}
    return t


def test_load_and_lora_kwargs_follow_the_setting():
    assert _trainer()._offload_load_kwargs() == {} and _trainer()._offload_peft_kwargs() == {}
    t = _trainer("auto", "auto")
    assert t._offload_load_kwargs() == {
        "offload_layers": "auto",
        "device_map_planner_kwargs": {"prefetch_depth": "auto"},
    }
    assert t._offload_peft_kwargs() == {"offload_layers": "auto", "prefetch_depth": "auto"}


def test_a_fixed_count_on_several_gpus_waits_for_peft():
    # Core refuses a count at load on a multi-GPU map; get_peft_model installs it there.
    for device_map in ("unsloth_balanced", "balanced"):
        assert _trainer(8, 2)._offload_load_kwargs(device_map) == {}
        assert _trainer("auto", 2)._offload_load_kwargs(device_map)["offload_layers"] == "auto"
        assert _trainer(8, 2)._offload_peft_kwargs() == {"offload_layers": 8, "prefetch_depth": 2}
    assert _trainer(8, 2)._offload_load_kwargs("sequential")["offload_layers"] == 8


def test_every_offload_capable_load_path_passes_the_kwargs():
    import inspect
    from core.training import trainer

    src = inspect.getsource(trainer.UnslothTrainer.load_model)
    # Text, Orpheus (snac), audio VLM and vision; the codec / Whisper paths have no decoder stack to stream.
    assert src.count("**self._offload_load_kwargs(device_map)") == 4
    peft = inspect.getsource(trainer.UnslothTrainer.prepare_model_for_training)
    assert peft.count("**self._offload_peft_kwargs()") == 3


def test_snapshot_is_none_without_a_swapper():
    assert _trainer(model = SimpleNamespace())._offload_snapshot() is None
    assert _trainer(model = None)._offload_snapshot() is None
    # An unsloth_zoo without stats() still trains, just without the panel.
    swapper = SimpleNamespace(indices = [1])
    assert _trainer(model = SimpleNamespace(_unsloth_block_swap = swapper))._offload_snapshot() is None


def test_snapshot_keys_layers_as_strings(monkeypatch):
    from core.training import trainer

    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    swapper = SimpleNamespace(
        stats = lambda: {"state": {14: "host", 15: "gpu"}, "swapped": [14, 15], "prefetch_depth": 2}
    )
    snap = _trainer(model = SimpleNamespace(_unsloth_block_swap = swapper))._offload_snapshot()
    assert snap["state"] == {"14": "host", "15": "gpu"} and snap["prefetch_depth"] == 2


def test_offload_route_reports_inactive_until_a_step_carries_stats(monkeypatch):
    import asyncio
    from routes import training as route

    progress = SimpleNamespace(offload = None)
    backend = SimpleNamespace(trainer = SimpleNamespace(get_training_progress = lambda: progress))
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    assert asyncio.run(route.get_offload_state(current_subject = "u")) == {"active": False}
    progress.offload = {"swapped": [3], "prefetch_depth": 2}
    assert asyncio.run(route.get_offload_state(current_subject = "u")) == {
        "active": True,
        "swapped": [3],
        "prefetch_depth": 2,
    }


def test_worker_progress_events_carry_the_snapshot():
    import inspect
    from core.training import worker
    assert '"offload": getattr(progress, "offload", None)' in inspect.getsource(
        worker._create_trainer_progress_callback
    )


def test_backend_keeps_the_last_offload_snapshot():
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    snap = {"swapped": [14, 15], "prefetch_depth": 2}
    backend._handle_event({"type": "progress", "step": 1, "loss": 1.0, "offload": snap})
    # A later step without stats must not blank the panel.
    backend._handle_event({"type": "progress", "step": 2, "loss": 0.9})
    assert backend._progress.offload == snap


# The planner's refusal for Qwen3-0.6B at a 1 GiB budget (unsloth_zoo DeviceMapInfeasible).
_INFEASIBLE = (
    "Even with 27 of 28 decoder layers in host RAM the model does not fit while keeping 0.78 GiB "
    "free for training on cuda:0 (0.96 GiB free). Lower max_seq_length or the batch size, or add a GPU."
)


@pytest.mark.parametrize(
    "message", [_INFEASIBLE, "GPU ran out of VRAM during training.", "CUDA out of memory."]
)
def test_a_run_that_does_not_fit_names_its_vram_budget(message):
    hinted = _with_vram_budget_hint({"offload_vram_gb": 1.5}, message)
    assert hinted.startswith(message) and "1.5 GiB VRAM budget" in hinted


def test_budget_hint_stays_out_of_unrelated_or_unbudgeted_errors():
    assert _with_vram_budget_hint({"offload_vram_gb": 4}, "Access denied") == "Access denied"
    assert _with_vram_budget_hint({}, _INFEASIBLE) == _INFEASIBLE


def test_every_worker_failure_path_carries_the_budget_hint():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    # Load, LoRA prepare (offload_layers = "auto" plans in get_peft_model) and training OOM.
    assert src.count("_with_vram_budget_hint(") == 4


@pytest.mark.parametrize("checkpointing", ["none", "False", "off", "no", "0"])
@pytest.mark.parametrize("offload", [14, "auto"])
def test_offload_without_checkpointing_is_refused_up_front(checkpointing, offload):
    with pytest.raises(ValidationError, match = "needs gradient checkpointing"):
        _request(offload_layers = offload, gradient_checkpointing = checkpointing)


def test_offload_with_checkpointing_or_off_is_accepted():
    _request(offload_layers = "auto", gradient_checkpointing = "unsloth")
    _request(offload_layers = 0, gradient_checkpointing = "none")


def test_worker_applies_the_budget_only_where_auto_offload_sizes_to_it():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    gate = src[src.index("# ── 2b. Training VRAM budget ──") :][:900]
    for needle in (
        'config.get("offload_layers") == "auto"',
        'config.get("is_decision")',
        'config.get("is_embedding")',
        'config.get("training_type", "LoRA/QLoRA") in ("LoRA/QLoRA", "Continued Pretraining")',
    ):
        assert needle in gate
    # The budget lands before the decision / embedding branches return, so the gate has to.
    assert src.index("# ── 2b. Training VRAM budget ──") < src.index(
        'if config.get("is_decision", False):'
    )


def test_multi_gpu_vision_and_audio_runs_skip_the_budget():
    # Core cannot offload a vision / audio load spread over GPUs, and LoRA setup allocates before it swaps,
    # so a cap there would OOM; the budget is skipped for those runs, and the OOM hint must not name it.
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    gate = src[src.index("_budget_unsupported = ") :][:300]
    for needle in (
        "len(gpu_ids or []) > 1",
        'config.get("is_dataset_image")',
        'config.get("is_dataset_audio")',
    ):
        assert needle in gate
    skip = src.index("if _budget_unsupported:")
    assert 'config["offload_vram_gb"] = None' in src[skip : src.index("elif _wants_budget:")]
    assert (
        skip
        < src.index("elif _wants_budget:")
        < src.index("set_per_process_memory_fraction(_b_fraction")
    )


def test_disabled_checkpointing_aliases_match_the_trainer():
    import ast
    from pathlib import Path

    from models.training import _CHECKPOINTING_OFF

    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "normalize_gradient_checkpointing"
    )
    off = next(
        ast.literal_eval(node.test.comparators[0])
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.body[0], ast.Return)
        and getattr(node.body[0].value, "value", None) is False
    )
    assert set(off) == set(_CHECKPOINTING_OFF)


def test_auto_plans_at_load_for_the_runs_batch_and_rank():
    from core.training.worker import _offload_plan_shape

    trainer = _trainer("auto")
    trainer._offload_plan_shape = _offload_plan_shape({"batch_size": 4, "lora_r": 64})
    assert trainer._offload_load_kwargs()["device_map_planner_kwargs"] == {
        "prefetch_depth": 2,
        "batch_size": 4,
        "lora_rank": 64,
    }


def test_every_worker_load_passes_the_plan_shape():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )
    assert src.count("offload_plan_shape = _offload_plan_shape(config),") == 2
