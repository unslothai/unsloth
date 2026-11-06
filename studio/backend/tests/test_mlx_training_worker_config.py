# SPDX-License-Identifier: AGPL-3.0-only

import ast
import importlib.util
import inspect
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest


def _load_worker_module():
    worker_path = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"
    stub_names = (
        "core.training",
        "core.training.dataset_bounds",
        "loggers",
        "utils",
        "utils.child_stdio",
        "utils.hardware",
        "utils.hf_dataset_options",
        "utils.native_tls",
        "utils.training_runs",
        "utils.wheel_utils",
        "utils.training_runs",
        # Imported for real against the stubs above, so dropped again afterwards.
        "utils.kernel_install",
        "utils.ssm_runtime",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    try:
        core_training = types.ModuleType("core.training")
        core_training.__path__ = [str(worker_path.parent)]
        sys.modules["core.training"] = core_training

        loggers = types.ModuleType("loggers")
        loggers.get_logger = lambda *_args, **_kwargs: None
        sys.modules["loggers"] = loggers

        utils = types.ModuleType("utils")
        # Keep the real __path__: an empty one shadows the package and breaks the worker's imports.
        utils.__path__ = [str(Path(__file__).resolve().parents[1] / "utils")]
        sys.modules["utils"] = utils

        child_stdio = types.ModuleType("utils.child_stdio")

        def utf8_child_env(env = None):
            child = dict(os.environ if env is None else env)
            child["PYTHONIOENCODING"] = "utf-8"
            return child

        child_stdio.utf8_child_env = utf8_child_env
        sys.modules["utils.child_stdio"] = child_stdio

        hardware = types.ModuleType("utils.hardware")
        hardware.apply_gpu_ids = lambda *_args, **_kwargs: None
        sys.modules["utils.hardware"] = hardware

        hf_dataset_options = types.ModuleType("utils.hf_dataset_options")
        hf_dataset_options.hf_dataset_split_instruction_names = lambda *_args, **_kwargs: ()
        sys.modules["utils.hf_dataset_options"] = hf_dataset_options

        # worker.py calls this at import time; without the stub this file fails when run alone.
        native_tls = types.ModuleType("utils.native_tls")
        native_tls.activate_native_tls = lambda *_args, **_kwargs: None
        sys.modules["utils.native_tls"] = native_tls

        training_runs = types.ModuleType("utils.training_runs")
        training_runs.build_default_output_dir_name = lambda *_args, **_kwargs: "training-run"
        sys.modules["utils.training_runs"] = training_runs

        wheel_utils = types.ModuleType("utils.wheel_utils")
        for name in (
            "direct_wheel_url",
            "flash_attn_wheel_url",
            "install_wheel",
            "is_windows_wheel_env",
            "mamba_ssm_pure_wheel_url",
            "pinned_source_build_supported",
            "probe_torch_wheel_env",
            "url_exists",
        ):
            setattr(wheel_utils, name, lambda *_args, **_kwargs: None)
        # Read at import by utils.kernel_install, which the worker imports.
        for name in (
            "package_installs_offline_env",
            "redact_url_credentials",
            "xformers_wheel_url",
        ):
            setattr(wheel_utils, name, lambda *_args, **_kwargs: None)
        for name in (
            "CAUSAL_CONV1D_PACKAGE_VERSION",
            "CAUSAL_CONV1D_RELEASE_BASE_URL",
            "CAUSAL_CONV1D_RELEASE_TAG",
            "MAMBA_SSM_PACKAGE_VERSION",
            "MAMBA_SSM_RELEASE_BASE_URL",
            "MAMBA_SSM_RELEASE_TAG",
        ):
            setattr(wheel_utils, name, "")
        sys.modules["utils.wheel_utils"] = wheel_utils

        spec = importlib.util.spec_from_file_location("mlx_training_worker_under_test", worker_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


_worker = _load_worker_module()
_normalize_mlx_studio_optimizer = _worker._normalize_mlx_studio_optimizer
_normalize_mlx_studio_scheduler = _worker._normalize_mlx_studio_scheduler
_mlx_vlm_max_resized_size = _worker._mlx_vlm_max_resized_size
_mlx_vlm_resized_image_layout = _worker._mlx_vlm_resized_image_layout
_copy_mlx_vlm_image_processor = _worker._copy_mlx_vlm_image_processor
_resize_mlx_vlm_image = _worker._resize_mlx_vlm_image
_adapt_for_mlx_vlm = _worker._adapt_for_mlx_vlm
_mlx_dora_peft_kwargs = _worker._mlx_dora_peft_kwargs


def test_mlx_studio_optimizer_aliases_are_explicit(monkeypatch):
    # Pin the local fallback: a zoo with the MLX normalizer keeps adamw_8bit (real 8-bit AdamW).
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.trainer", None)
    assert _normalize_mlx_studio_optimizer("adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("paged_adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("adafactor") == "adafactor"


def test_mlx_studio_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match = "Supported"):
        _normalize_mlx_studio_optimizer("adamw_typo")


def test_mlx_studio_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match = "Unsupported LR scheduler for MLX training"):
        _normalize_mlx_studio_scheduler("linear_typo")


def test_mlx_dora_requires_the_named_use_dora_parameter():
    # A **kwargs catch-all absorbs use_dora, so accepting the keyword is not support.
    def old_zoo(
        model,
        r = 16,
        **kwargs,
    ):
        return model

    def new_zoo(
        model,
        r = 16,
        use_dora = False,
        **kwargs,
    ):
        return model

    def keyword_only_zoo(model, *, use_dora = False):
        return model

    def positional_only_zoo(
        model,
        use_dora = False,
        /,
        **kwargs,
    ):
        return model

    def var_positional_zoo(model, *use_dora):
        return model

    def var_keyword_zoo(model, **use_dora):
        return model

    for usable in (new_zoo, keyword_only_zoo):
        assert _mlx_dora_peft_kwargs({"use_dora": True}, usable) == {"use_dora": True}
    with pytest.raises(NotImplementedError, match = "unsloth-zoo"):
        _mlx_dora_peft_kwargs({"use_dora": True}, old_zoo)
    for unusable in (positional_only_zoo, var_positional_zoo, var_keyword_zoo):
        with pytest.raises(NotImplementedError, match = "unsloth-zoo"):
            _mlx_dora_peft_kwargs({"use_dora": True}, unusable)
    with pytest.raises(NotImplementedError, match = "unsloth-zoo"):
        _mlx_dora_peft_kwargs({"use_dora": True}, object())
    assert _mlx_dora_peft_kwargs(
        {
            "use_dora": True,
            "is_dataset_image": True,
            "finetune_vision_layers": True,
        },
        new_zoo,
    ) == {"use_dora": True}
    assert _mlx_dora_peft_kwargs({}, old_zoo) == {}
    assert _mlx_dora_peft_kwargs({"use_dora": False}, old_zoo) == {}


def test_mlx_studio_keeps_hf_style_tokenizer_dual_purpose():
    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )

    assert "tokenizer = tokenizer" in source
    assert "processor = tokenizer if is_vlm else None" not in source


def test_mlx_wandb_run_config_excludes_subject_and_secrets():
    # The owner's subject must be filtered with the secrets or it lands in W&B.
    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )

    assert (
        '_wandb_sensitive = {"hf_token", "wandb_token", "s3_config", "subject"}' in source
    ), "MLX W&B run config must exclude subject and the token/s3 secrets"


def test_mlx_vlm_resize_uses_max_dimension_like_torch_trainer():
    assert _mlx_vlm_max_resized_size(1000, 500, 512) == (512, 256)
    assert _mlx_vlm_max_resized_size(500, 1000, 512) == (256, 512)
    assert _mlx_vlm_max_resized_size(1000, 1000, 512) == (512, 512)
    assert _mlx_vlm_max_resized_size(256, 128, 1536) == (256, 128)
    assert _mlx_vlm_max_resized_size(512, 256, 512) == (512, 256)
    # Half-pixel cases must match the Torch collator (not banker's round).
    assert _mlx_vlm_max_resized_size(333, 1000, 500) == (167, 500)
    assert _mlx_vlm_max_resized_size(1000, 333, 500) == (500, 167)


def test_mlx_vlm_resize_keeps_default_numpy_layout_hwc():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))

    resized = _resize_mlx_vlm_image(image, 128)

    assert resized.shape == (80, 128, 3)
    assert resized.flags.c_contiguous


def test_mlx_vlm_resize_uses_requested_chw_numpy_layout():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))

    resized = _resize_mlx_vlm_image(image, 128, image_layout = "chw")

    assert resized.shape == (3, 80, 128)
    assert resized.flags.c_contiguous


def test_mlx_vlm_resized_image_layout_probes_processor_contract():
    class ChwOnlyImageProcessor:
        def __call__(self, images = None):
            image = images[0]
            if image.shape[0] == 3:
                return {"pixel_values": image}
            raise ValueError("expected CHW")

    class HwcImageProcessor:
        def __call__(self, images = None):
            image = images[0]
            if image.shape[-1] == 3:
                return {"pixel_values": image}
            raise ValueError("expected HWC")

    assert (
        _mlx_vlm_resized_image_layout(
            types.SimpleNamespace(image_processor = ChwOnlyImageProcessor())
        )
        == "chw"
    )
    assert (
        _mlx_vlm_resized_image_layout(types.SimpleNamespace(image_processor = HwcImageProcessor()))
        is None
    )


def test_mlx_vlm_layout_probe_copies_image_processor():
    class StatefulImageProcessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, images = None):
            self.calls += 1
            image = images[0]
            if image.shape[0] == 3:
                return {"pixel_values": image}
            raise ValueError("expected CHW")

    image_processor = StatefulImageProcessor()

    layout = _mlx_vlm_resized_image_layout(types.SimpleNamespace(image_processor = image_processor))

    assert layout == "chw"
    assert image_processor.calls == 0


def test_mlx_vlm_image_processor_copy_refuses_uncopyable_processors():
    class UncopyableImageProcessor:
        def __copy__(self):
            raise RuntimeError("no copy")

        def __deepcopy__(self, _memo):
            raise RuntimeError("no deepcopy")

    image_processor = UncopyableImageProcessor()

    assert _copy_mlx_vlm_image_processor(image_processor) is None


def test_mlx_vlm_layout_probe_skips_uncopyable_processors():
    class UncopyableImageProcessor:
        def __copy__(self):
            raise RuntimeError("no copy")

        def __deepcopy__(self, _memo):
            raise RuntimeError("no deepcopy")

        def __call__(self, images = None):
            raise AssertionError("live processor should not be probed")

    assert (
        _mlx_vlm_resized_image_layout(
            types.SimpleNamespace(image_processor = UncopyableImageProcessor())
        )
        is None
    )


def test_mlx_vlm_adapter_applies_chw_layout_to_message_images():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (320, 200), color = (10, 20, 30))
    item = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe it."},
                ],
            }
        ]
    }

    adapted = _adapt_for_mlx_vlm([item], resize = 128, image_layout = "chw")

    assert adapted[0]["image"].shape == (3, 80, 128)
    assert adapted[0]["messages"][0]["content"][0] == {"type": "image"}


def test_activate_transformers_version_or_warn_logs_on_failure(monkeypatch):
    """The MLX fast-path activation must log failures; a bare except Exception: pass hid broken venvs."""
    warnings_logged = []
    fake_logger = types.SimpleNamespace(
        warning = lambda *a, **k: warnings_logged.append((a, k)),
    )
    monkeypatch.setattr(_worker, "logger", fake_logger)

    def _boom(_name, _hf_token = None):
        raise RuntimeError("venv .venv_t5_550 missing")

    monkeypatch.setattr(_worker, "_activate_transformers_version", _boom)

    _worker._activate_transformers_version_or_warn("google/gemma-4-12b")

    assert len(warnings_logged) == 1, "activation failure was not logged"
    assert "gemma-4-12b" in str(warnings_logged[0]), "log does not name the model"


def test_activate_transformers_version_or_warn_silent_on_success(monkeypatch):
    warnings_logged = []
    fake_logger = types.SimpleNamespace(
        warning = lambda *a, **k: warnings_logged.append((a, k)),
    )
    monkeypatch.setattr(_worker, "logger", fake_logger)
    monkeypatch.setattr(
        _worker, "_activate_transformers_version", lambda _name, _hf_token = None: None
    )

    _worker._activate_transformers_version_or_warn("meta-llama/Llama-3-8B")

    assert warnings_logged == [], "should not warn when activation succeeds"


def _masking_block():
    """The whole `if train_on_completions ...:` block from _run_mlx_training."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    blocks = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.BoolOp)
        and "apply_completion_masking" in ast.unparse(n)
    ]
    assert len(blocks) == 1, "expected one guarded masking block"
    return blocks[0]


def _masking_call_and_guard():
    """The apply_completion_masking call in _run_mlx_training, plus the `if not applied` guard."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "apply_completion_masking"
    ]
    assert len(calls) == 1, "expected exactly one masking call in the MLX path"
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not masking_applied"
    ]
    assert len(guards) == 1, "masking result must be checked exactly once"
    return calls[0], guards[0]


def test_alpaca_datasets_still_get_explicit_markers():
    """Alpaca text carries no chat markers, so the template must be passed explicitly."""
    call, _ = _masking_call_and_guard()
    passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}

    assert passed["dataset_template"] == "'alpaca' if dataset_final_format == 'alpaca' else None"


def _run_masking(
    model_name = "org/unmapped",
    detect = None,
    **overrides,
):
    """Lifts the real masking block out of _run_mlx_training, which only runs on Apple Silicon."""
    block = compile(ast.Module(body = [_masking_block()], type_ignores = []), "<masking>", "exec")
    events = []
    trainer = types.SimpleNamespace(processing_class = types.SimpleNamespace(), tokenizer = None)
    zoo = types.ModuleType("unsloth_zoo")
    zoo.__path__ = []
    datasets = types.ModuleType("unsloth_zoo.dataset_utils")
    if detect is not None:
        datasets.get_chat_template_parts = detect
    previous = {n: sys.modules.get(n) for n in ("unsloth_zoo", "unsloth_zoo.dataset_utils")}
    sys.modules["unsloth_zoo"] = zoo
    sys.modules["unsloth_zoo.dataset_utils"] = datasets

    namespace = dict(vars(_worker))
    namespace.update(
        {
            "config": {"train_on_completions": overrides.pop("train_on_completions", True)},
            "raw_text_mode": overrides.pop("raw_text_mode", False),
            "dataset_final_format": overrides.pop("dataset_final_format", "chatml"),
            "model_name": model_name,
            "trainer": trainer,
            "train_on_responses_only": lambda t, **_kw: t,
            "_send": lambda event_type, **kw: events.append((event_type, kw)),
        }
    )
    try:
        exec(block, namespace)
    finally:
        for name, module in previous.items():
            sys.modules.pop(name, None) if module is None else sys.modules.update({name: module})
    return events, namespace.get("masking_applied")


try:
    import utils.datasets.completion_masking  # noqa: F401
    _MASKING_IMPORTABLE = True
except Exception:  # pragma: no cover
    _MASKING_IMPORTABLE = False

needs_masking_helper = pytest.mark.skipif(
    not _MASKING_IMPORTABLE, reason = "utils.datasets.completion_masking is not importable"
)


def _warnings(events):
    return [kwargs["message"] for kind, kwargs in events if kind == "warning"]


def _detect_fails(_processor):
    raise RuntimeError("no chat template parts")


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_on_completions": False},
        {"raw_text_mode": True},
        {"dataset_final_format": "raw_text"},
    ],
    ids = ["not-requested", "raw-text-mode", "raw-text-format"],
)
@needs_masking_helper
def test_masking_block_is_skipped(overrides):
    events, applied = _run_masking(**overrides)

    assert events == [] and applied is None


@needs_masking_helper
def test_masking_miss_reaches_the_warning_channel():
    """A miss must be a sticky warning, not a status line the next update overwrites."""
    events, applied = _run_masking(detect = _detect_fails)

    assert applied is False
    warnings = _warnings(events)
    assert len(warnings) == 1
    assert "org/unmapped" in warnings[0] and "full sequences" in warnings[0]
    # The parent pump reads only `message` for warnings, so `status_message` would be lost.
    assert [kind for kind, _ in events][-1] == "warning"


@pytest.mark.parametrize(
    "model_name,detect",
    [
        ("org/unmapped", lambda _p: ("<|user|>", "<|assistant|>")),
        ("unsloth/llama-3-8b-instruct", _detect_fails),
    ],
    ids = ["auto-detected", "recovered-by-template-table"],
)
@needs_masking_helper
def test_applied_runs_leave_no_warning(model_name, detect):
    """Detection can fail at level "warning" and the table still mask. That is not a miss."""
    events, applied = _run_masking(model_name = model_name, detect = detect)

    assert applied is True
    assert _warnings(events) == []


@pytest.mark.parametrize("model_name", ["", None, "org/model with spaces", "org/{brace}"])
@needs_masking_helper
def test_odd_model_names_do_not_break_the_warning(model_name):
    events, applied = _run_masking(model_name = model_name, detect = _detect_fails)

    assert applied is False and len(_warnings(events)) == 1


class _FakeRankTrainer:
    def __init__(
        self,
        rank = 0,
        world_size = 1,
        save_error = None,
    ):
        self.distributed_world_size = world_size
        self.is_main_process = rank == 0
        self.stop_requested = False
        self.saved = []
        self._save_error = save_error

    def save_model(self, path):
        if self._save_error is not None:
            raise self._save_error
        self.saved.append(path)

    def _distributed_any_flag(self, flag):
        return bool(flag)

    def _raise_distributed_failure(
        self,
        failed,
        context,
        exc = None,
    ):
        if failed:
            raise RuntimeError(f"{context}: {exc}")


def _finalize(
    trainer,
    stop = (False, True),
    checkpoint_ok = True,
):
    events = []
    _worker._finalize_mlx_training(
        trainer,
        lambda: stop,
        "/out",
        lambda: None,
        lambda event_type, **payload: events.append((event_type, payload)),
        lambda: checkpoint_ok,
    )
    return events


def test_mlx_epoch_steps_divide_by_world_size_only_when_derived():
    steps = _worker._resolve_mlx_training_steps
    assert steps(0, 16, 2, 2, 3, 1) == 12
    assert steps(0, 16, 2, 2, 3, 2) == 6
    assert steps(7, 16, 2, 2, 3, 2) == 7


# Expected = transformers 4.57.6 get_warmup_steps / TrainerState.compute_steps.
@pytest.mark.parametrize(
    "ratio, max_steps, expected",
    [(0.05, 10, 1), (0.03, 15, 1), (0.1, 25, 3), (0.1, 45, 5), (0.15, 30, 5), (0.29, 100, 29)],
)
def test_mlx_ratio_schedule_rounds_up_like_transformers(ratio, max_steps, expected):
    trainer = types.SimpleNamespace(
        distributed_world_size = 1,
        args = types.SimpleNamespace(max_steps = 0, warmup_steps = 0, eval_steps = 0),
    )
    _worker._configure_mlx_training_schedule(
        trainer, max_steps, None, 1, 1, 1, warmup_ratio = ratio, eval_steps_ratio = ratio
    )
    assert trainer.args.warmup_steps == expected
    assert trainer.args.eval_steps == expected


@pytest.mark.parametrize("rank", [0, 1])
def test_mlx_final_save_is_rank0_owned(rank):
    trainer = _FakeRankTrainer(rank = rank, world_size = 2)
    events = _finalize(trainer)
    assert trainer.saved == (["/out"] if rank == 0 else [])
    assert events[-1] == (
        "complete",
        {"output_dir": "/out" if rank == 0 else None, "status_message": "Training completed"},
    )


def test_mlx_rank0_save_failure_reaches_every_rank():
    with pytest.raises(RuntimeError, match = "final model save"):
        _finalize(_FakeRankTrainer(world_size = 2, save_error = OSError("disk full")))


def test_mlx_single_process_finalization_keeps_prior_contract():
    trainer = _FakeRankTrainer()
    assert _finalize(trainer, stop = (True, False))[-1][1]["status_message"] == "Training cancelled"
    assert trainer.saved == []

    events = _finalize(trainer, stop = (True, True), checkpoint_ok = False)
    assert trainer.saved == ["/out"]
    assert events[-1][0] == "error" and events[-1][1]["resume_blocked"] is True


def _adapter_setup_block():
    """The `continues_adapter = ...` line, the LoRA setup `if` and the wrap after it in _run_mlx_training."""
    body = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training))).body[0].body
    starts = [
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "continues_adapter"
    ]
    assert len(starts) == 1, "expected one continues_adapter assignment"
    return body[starts[0] : starts[0] + 3]


def _run_adapter_setup(
    lora_modules,
    training_type = "LoRA/QLoRA",
    modules = (),
    **config,
):
    """Lifts the real adapter setup out of _run_mlx_training, which only runs on Apple Silicon."""
    block = compile(ast.Module(body = _adapter_setup_block(), type_ignores = []), "<peft>", "exec")
    events, wraps = [], []
    model = types.SimpleNamespace(lora_modules = lora_modules, named_modules = lambda: modules)

    def get_peft_model(wrapped, **kwargs):
        wraps.append(kwargs)
        return wrapped

    namespace = dict(vars(_worker))
    namespace.update(
        {
            "config": config,
            "model": model,
            "training_type": training_type,
            "is_cpt": _run_mlx_assignment("is_cpt", training_type = training_type),
            "use_lora": _run_mlx_assignment("use_lora", training_type = training_type),
            "is_vlm": False,
            "use_grad_checkpoint": "mlx",
            "lora_random_state": 3407,
            "mlx_dora_kwargs": {},
            "FastMLXModel": types.SimpleNamespace(get_peft_model = get_peft_model),
            "iter_mlx_lora_modules": lambda m: iter(m.lora_modules),
            "_send": lambda event_type, **kw: events.append((event_type, kw)),
        }
    )
    exec(block, namespace)
    return events, wraps


@pytest.mark.parametrize("training_type", ["LoRA/QLoRA", "Continued Pretraining"])
def test_mlx_continued_adapter_is_not_wrapped_again(training_type):
    events, wraps = _run_adapter_setup(
        [("layers.0.self_attn.q_proj", object())], training_type = training_type
    )

    assert wraps == []
    assert any("Continuing from existing LoRA adapters" in kw["status_message"] for _, kw in events)


def test_mlx_base_model_gets_a_fresh_adapter():
    _, wraps = _run_adapter_setup([], lora_r = 8)

    assert [kwargs["r"] for kwargs in wraps] == [8]


def test_mlx_cpt_trains_embedding_modules_in_full():
    _, wraps = _run_adapter_setup(
        [],
        training_type = "Continued Pretraining",
        target_modules = ["q_proj", "model.embed_tokens", "lm_head"],
    )

    assert len(wraps) == 1
    assert wraps[0]["target_modules"] == ["q_proj"]
    assert wraps[0]["modules_to_save"] == ["embed_tokens", "lm_head"]
    assert not any(key.startswith("finetune_") for key in wraps[0])


def test_mlx_cpt_without_embedding_targets_wraps_the_default_projections():
    _, wraps = _run_adapter_setup([], training_type = "Continued Pretraining")

    assert wraps[0]["target_modules"] == list(_worker._MLX_DEFAULT_LORA_TARGETS)
    assert wraps[0]["modules_to_save"] is None


def test_trains_embedding_modules_ignores_lora_on_the_head():
    trains = _worker._trains_embedding_modules
    assert trains(["model.layers.0.self_attn.q_proj.lora_a", "model.embed_tokens.weight"])
    assert trains(["model.layers.0.mlp.up_proj.lora_b", "lm_head.weight"])
    assert not trains(["lm_head.lora_a", "lm_head.lora_b", "model.layers.0.mlp.up_proj.lora_a"])


def test_mlx_cpt_embedding_learning_rate_follows_the_cuda_rule(monkeypatch):
    monkeypatch.setattr(_worker, "logger", types.SimpleNamespace(warning = lambda *_a: None))
    rate = _worker._mlx_cpt_embedding_learning_rate
    assert rate(None, 5e-5, trains_embeddings = True) == pytest.approx(5e-6)
    assert rate(1e-6, 5e-5, trains_embeddings = True) == 1e-6
    assert rate(1e-6, 5e-5, trains_embeddings = False) is None


def _trainer_config_block():
    """From the MLXTrainingConfig kwargs through the schedule call in _run_mlx_training."""
    body = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training))).body[0].body
    targets = [
        ast.unparse(node.targets[0]) if isinstance(node, ast.Assign) else "" for node in body
    ]
    start = targets.index("mlx_config_kwargs")
    end = targets.index("(max_steps, eval_steps_val)")
    return body[start : end + 1]


class _UnsizedStream:
    def __len__(self):
        raise TypeError("a stream has no length")


@pytest.mark.parametrize("embedding_lr_value", [5e-6, None])
@pytest.mark.parametrize("dataset_streaming", [True, False])
def test_mlx_trainer_config_follows_streaming_and_embedding_lr(
    dataset_streaming, embedding_lr_value
):
    block = compile(ast.Module(body = _trainer_config_block(), type_ignores = []), "<cfg>", "exec")
    configs, schedules = [], []
    namespace = dict(vars(_worker))
    namespace.update(
        {
            "config": {},
            "dataset": _UnsizedStream() if dataset_streaming else [0] * 7,
            "dataset_streaming": dataset_streaming,
            "embedding_lr_value": embedding_lr_value,
            "training_type": "LoRA/QLoRA",
            "format_type": "alpaca",
            "MLXTrainingConfig": type(
                "MLXTrainingConfig",
                (),
                {
                    "__dataclass_fields__": {"dataset_order": None},
                    "__init__": lambda self, **kw: configs.append(kw),
                },
            ),
            "MLXTrainer": lambda **kw: kw,
            "_configure_mlx_training_schedule": lambda *a, **kw: schedules.append(a) or (8, 0),
            **{
                name: None
                for name in (
                    "model",
                    "tokenizer",
                    "eval_dataset",
                    "eval_steps_value",
                    "max_grad_leaf_norm",
                    "requested_max_steps",
                    "batch_size",
                    "grad_accum",
                    "num_epochs",
                    "warmup_ratio",
                    "warmup_uses_ratio",
                    "eval_steps_ratio",
                    "warmup_steps",
                    "lr_value",
                    "lr_scheduler_type",
                    "optim_name",
                    "weight_decay",
                    "max_grad_norm",
                    "max_grad_value",
                    "max_seq_length",
                    "random_seed",
                    "use_grad_checkpoint",
                    "output_dir",
                    "eval_steps_val",
                )
            },
        }
    )
    exec(block, namespace)

    assert configs[0]["streaming"] is dataset_streaming
    assert configs[0]["dataset_order"] == ("sequential" if dataset_streaming else "torch_randperm")
    assert configs[0].get("embedding_learning_rate") == embedding_lr_value
    assert schedules[0][2] == (None if dataset_streaming else 7)


def test_mlx_slice_reads_a_stream_lazily():
    body = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training))).body[0].body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == "_slice"]
    calls = []

    class Stream(_UnsizedStream):
        def skip(self, n):
            calls.append(("skip", n))
            return self

        def take(self, n):
            calls.append(("take", n))
            return self

    namespace = {
        **vars(_worker),
        "dataset_streaming": True,
        "slice_start": 2,
        "slice_end": 4,
        "mlx_split_names_rows": False,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<slice>", "exec"), namespace)
    namespace["_slice"](Stream())

    # Rows 2..4 inclusive.
    assert calls == [("skip", 2), ("take", 3)]


def _run_mlx_assignment(target, **names):
    body = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training))).body[0].body
    nodes = [n for n in body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target]
    assert len(nodes) == 1
    namespace = {**vars(_worker), **names}
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<assign>", "exec"), namespace)
    return namespace[target]


@pytest.mark.parametrize("is_cpt, expected", [(True, "raw"), (False, "alpaca")])
def test_mlx_cpt_formats_raw_text_whatever_format_was_sent(is_cpt, expected):
    config = {"format_type": "alpaca"}
    assert _run_mlx_assignment("format_type", is_cpt = is_cpt, config = config) == expected


@pytest.mark.parametrize("is_cpt", [True, False])
@pytest.mark.parametrize("flag", ["is_dataset_image", "is_dataset_audio"])
def test_mlx_cpt_reads_a_multimodal_dataset_as_text(is_cpt, flag):
    assert _run_mlx_assignment(flag, is_cpt = is_cpt, config = {flag: True}) is not is_cpt


@pytest.mark.parametrize("format_type", ["raw", "auto"])
def test_mlx_raw_format_reads_an_audio_dataset_as_text(format_type):
    config = {"is_dataset_audio": True, "format_type": format_type}
    is_audio = _run_mlx_assignment("is_dataset_audio", is_cpt = False, config = config)
    assert is_audio is (format_type != "raw")


def test_mlx_cpt_embedding_lr_reaches_heads_zoo_resolved():
    model = types.SimpleNamespace(
        trainable_parameters = lambda: {"model.tok_embeddings.weight": None},
        _unsloth_cpt_full_module_weight_keys = {"model.tok_embeddings.weight"},
    )
    rate = _run_mlx_assignment(
        "embedding_lr_value",
        is_cpt = True,
        config = {},
        lr_value = 5e-5,
        model = model,
        tree_flatten = lambda tree: list(tree.items()),
    )
    assert rate == pytest.approx(5e-6)


def _mlx_body_if(test_source):
    body = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training))).body[0].body
    nodes = [
        n
        for n in ast.walk(ast.Module(body = body, type_ignores = []))
        if isinstance(n, ast.If) and ast.unparse(n.test) == test_source
    ]
    assert len(nodes) == 1
    return compile(ast.Module(body = nodes, type_ignores = []), "<if>", "exec")


@pytest.mark.parametrize("is_vlm", [True, False])
def test_mlx_audio_dataset_needs_an_audio_vlm(is_vlm):
    block = _mlx_body_if("is_dataset_audio and (not is_vlm)")
    namespace = {**vars(_worker), "is_dataset_audio": True, "is_vlm": is_vlm}
    if is_vlm:
        exec(block, namespace)
    else:
        with pytest.raises(ValueError, match = "audio-input model"):
            exec(block, namespace)


@pytest.mark.parametrize("image_col", [None, "image"])
def test_mlx_audio_dataset_formats_both_splits_as_audio_rows(image_col):
    formatted, adapted = [], []
    namespace = {
        **vars(_worker),
        "is_dataset_audio": True,
        "is_vlm": True,
        "dataset": "train",
        "eval_dataset": "eval",
        "tokenizer": "processor",
        "custom_format_mapping": None,
        "_send": lambda *_a, **_kw: None,
        "vision_image_size": None,
        "_mlx_audio_rows": lambda ds, processor, mapping, *column: formatted.append((ds, *column))
        or ([ds], image_col),
        "_adapt_for_mlx_vlm": lambda rows, **_kw: adapted.append(rows) or [*rows, "lifted"],
        "format_and_template_dataset": lambda *_a, **_kw: pytest.fail(
            "audio skips the VLM formatter"
        ),
    }
    exec(_mlx_body_if("is_dataset_audio"), namespace)

    # The eval split takes the train split's image column, and only a split with images is lifted.
    assert formatted == [("train",), ("eval", image_col)]
    assert adapted == ([] if image_col is None else [["train"], ["eval"]])
    suffix = [] if image_col is None else ["lifted"]
    assert namespace["dataset"] == ["train", *suffix]
    assert namespace["eval_dataset"] == ["eval", *suffix]


@pytest.mark.parametrize("audio", [True, False])
def test_mlx_audio_request_installs_the_decoder(monkeypatch, audio):
    # A CPT or raw-format run masks is_dataset_audio but still reads rows with an Audio column.
    from utils.datasets import audio_decode

    calls = []
    monkeypatch.setattr(audio_decode, "ensure_audio_decoding", lambda: calls.append(1) or True)
    exec(_mlx_body_if("config.get('is_dataset_audio')"), {"config": {"is_dataset_audio": audio}})

    assert calls == ([1] if audio else [])


class _AudioRowsDataset:
    def __init__(self, rows):
        self.rows, self.cast = rows, None
        self.column_names = list(rows[0])
        self.features = {}

    def cast_column(self, column, feature):
        if not hasattr(feature, "sampling_rate"):
            return self
        # A real cast returns a new dataset decoding at the target rate.
        rate = feature.sampling_rate
        cast = _AudioRowsDataset(
            [{**row, column: {**row[column], "sampling_rate": rate}} for row in self.rows]
        )
        cast.cast = (column, rate)
        return cast

    def __iter__(self):
        return iter(self.rows)


def test_mlx_audio_rows_keep_the_decoded_clip(monkeypatch):
    from utils.datasets import audio_decode

    decoders = [True]
    monkeypatch.setattr(audio_decode, "ensure_audio_decoding", lambda: decoders[0])
    zoo_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    zoo_utils.audio_extractor_sampling_rate = lambda processor: {"processor": 24000}.get(processor)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", zoo_utils)
    clips = [{"array": [0.1], "sampling_rate": 8000}, {"array": [0.2], "sampling_rate": 8000}]
    dataset = _AudioRowsDataset([{"audio": clip, "text": t} for clip, t in zip(clips, "ab")])

    rows, image_col = _worker._mlx_audio_rows(dataset, "processor")

    assert image_col is None
    assert [row["messages"][1]["content"][0]["audio"] for row in rows] == [
        {"array": [0.1], "sampling_rate": 24000},
        {"array": [0.2], "sampling_rate": 24000},
    ]
    unknown_rate, _ = _worker._mlx_audio_rows(dataset, "unknown")
    assert unknown_rate[0]["messages"][1]["content"][0]["audio"]["sampling_rate"] == 16000
    assert [row["messages"][2]["content"][0]["text"] for row in rows] == ["a", "b"]
    with pytest.raises(ValueError, match = "'audio' and 'text'"):
        _worker._mlx_audio_rows(_AudioRowsDataset([{"audio": clips[0], "x": 1}]), "processor")
    decoders[0] = False
    with pytest.raises(RuntimeError, match = "audio runtime"):
        _worker._mlx_audio_rows(dataset, "processor")


def test_mlx_audio_rows_carry_each_rows_images(monkeypatch):
    from PIL import Image

    from utils.datasets import audio_decode

    monkeypatch.setattr(audio_decode, "ensure_audio_decoding", lambda: True)
    zoo_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    zoo_utils.audio_extractor_sampling_rate = lambda processor: 16000
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", zoo_utils)
    red, blue = Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")
    clip = {"array": [0.1], "sampling_rate": 16000}
    dataset = _AudioRowsDataset(
        [
            {"audio": clip, "photo": [red, blue], "text": "a"},
            {"audio": clip, "photo": [None], "text": "b"},
        ]
    )

    rows, image_col = _worker._mlx_audio_rows(dataset, "processor")

    assert image_col == "photo"
    assert rows[0]["images"] == [red, blue] and "images" not in rows[1]
    kinds = [[part["type"] for part in row["messages"][1]["content"]] for row in rows]
    assert kinds == [["image", "image", "audio", "text"], ["audio", "text"]]
    # An eval split whose train split had no images trains on audio alone.
    eval_rows, eval_image_col = _worker._mlx_audio_rows(dataset, "processor", None, None)
    assert eval_image_col is None and "images" not in eval_rows[0]


@pytest.mark.parametrize("is_dataset_audio", [True, False])
def test_mlx_audio_dataset_loads_and_trains_as_a_vlm(is_dataset_audio):
    names = {"is_dataset_image": False, "is_dataset_audio": is_dataset_audio}
    model = types.SimpleNamespace(_is_vlm_model = True)
    assert _run_mlx_assignment("is_vlm", model = model, **names) is is_dataset_audio
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    text_only = {
        ast.unparse(kw.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for kw in n.keywords
        if kw.arg == "text_only"
    }
    assert len(text_only) == 1
    assert eval(text_only.pop(), names) is (None if is_dataset_audio else True)


@pytest.mark.parametrize(
    "is_dataset_image, is_dataset_audio, expected",
    [(True, False, True), (False, True, False), (True, True, True)],
)
def test_mlx_vision_layers_train_only_on_image_rows(is_dataset_image, is_dataset_audio, expected):
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    nodes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "finetune_vision"
    ]
    assert len(nodes) == 1
    namespace = {
        "config": {"finetune_vision_layers": True},
        "is_vlm": True,
        "is_dataset_image": is_dataset_image,
        "is_dataset_audio": is_dataset_audio,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), "<vision>", "exec"), namespace)
    assert namespace["finetune_vision"] is expected


class _Module:
    def __init__(
        self,
        quantized = False,
        children = (),
    ):
        if quantized:
            self.scales = object()
        self._children = list(children)

    def named_modules(self):
        yield "", self
        for name, child in self._children:
            for sub, module in child.named_modules():
                yield f"{name}.{sub}" if sub else name, module


def test_mlx_cpt_leaves_quantized_embeddings_frozen_with_a_warning():
    modules = [
        ("model.embed_tokens", _Module(quantized = True)),
        ("lm_head", _Module(children = [("linear", _Module(quantized = True))])),
    ]
    events, wraps = _run_adapter_setup(
        [],
        training_type = "Continued Pretraining",
        modules = modules,
        target_modules = ["q_proj", "embed_tokens", "lm_head"],
    )

    assert wraps[0]["target_modules"] == ["q_proj"]
    assert wraps[0]["modules_to_save"] is None
    warnings = [kw["message"] for kind, kw in events if kind == "warning"]
    assert len(warnings) == 1 and "embed_tokens and lm_head" in warnings[0]


def test_mlx_cpt_trains_unquantized_embeddings_in_full():
    modules = [("model.embed_tokens", _Module()), ("lm_head", _Module())]
    events, wraps = _run_adapter_setup(
        [],
        training_type = "Continued Pretraining",
        modules = modules,
        target_modules = ["q_proj", "embed_tokens", "lm_head"],
    )

    assert wraps[0]["modules_to_save"] == ["embed_tokens", "lm_head"]
    assert not [kw for kind, kw in events if kind == "warning"]


def test_mlx_cpt_drops_a_tied_head_with_its_quantized_embedding():
    quantized = _Module(quantized = True)
    assert _worker._mlx_quantized_cpt_modules(
        types.SimpleNamespace(named_modules = lambda: [("model.embed_tokens", quantized)]),
        {"embed_tokens", "lm_head"},
    ) == {"embed_tokens", "lm_head"}
