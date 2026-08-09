# SPDX-License-Identifier: AGPL-3.0-only

import ast
import importlib.util
import inspect
import sys
import textwrap
import types
from pathlib import Path

import pytest


def _load_worker_module():
    stub_names = (
        "structlog",
        "loggers",
        "utils",
        "utils.child_stdio",
        "utils.hardware",
        "utils.hf_dataset_options",
        "utils.training_runs",
        "utils.wheel_utils",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    try:
        sys.modules["structlog"] = types.ModuleType("structlog")

        loggers = types.ModuleType("loggers")
        loggers.get_logger = lambda *_args, **_kwargs: None
        sys.modules["loggers"] = loggers

        utils = types.ModuleType("utils")
        utils.__path__ = []
        sys.modules["utils"] = utils

        child_stdio = types.ModuleType("utils.child_stdio")
        child_stdio.utf8_child_env = lambda env = None: dict(env or {})
        sys.modules["utils.child_stdio"] = child_stdio

        hardware = types.ModuleType("utils.hardware")
        hardware.apply_gpu_ids = lambda *_args, **_kwargs: None
        sys.modules["utils.hardware"] = hardware

        hf_dataset_options = types.ModuleType("utils.hf_dataset_options")
        hf_dataset_options.hf_dataset_split_instruction_names = lambda *_args, **_kwargs: ()
        sys.modules["utils.hf_dataset_options"] = hf_dataset_options

        training_runs = types.ModuleType("utils.training_runs")
        training_runs.build_default_output_dir_name = lambda *_args, **_kwargs: "training-run"
        sys.modules["utils.training_runs"] = training_runs

        wheel_utils = types.ModuleType("utils.wheel_utils")
        for name in (
            "direct_wheel_url",
            "flash_attn_wheel_url",
            "has_blackwell_gpu",
            "install_wheel",
            "probe_torch_wheel_env",
            "url_exists",
        ):
            setattr(wheel_utils, name, lambda *_args, **_kwargs: None)
        sys.modules["utils.wheel_utils"] = wheel_utils

        worker_path = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"
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


def test_mlx_studio_optimizer_aliases_are_explicit():
    assert _normalize_mlx_studio_optimizer("adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("paged_adamw_8bit") == "adamw"
    assert _normalize_mlx_studio_optimizer("adafactor") == "adafactor"


def test_mlx_studio_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match = "Supported"):
        _normalize_mlx_studio_optimizer("adamw_typo")


def test_mlx_studio_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match = "Unsupported LR scheduler for MLX training"):
        _normalize_mlx_studio_scheduler("linear_typo")


def test_mlx_studio_keeps_hf_style_tokenizer_dual_purpose():
    source = (Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py").read_text(
        encoding = "utf-8"
    )

    assert "tokenizer = tokenizer" in source
    assert "processor = tokenizer if is_vlm else None" not in source


def test_mlx_wandb_run_config_excludes_subject_and_secrets():
    # The MLX W&B run config uploads everything minus a sensitive set. The owner's subject must be
    # filtered alongside the secrets, or it lands in W&B even though DB history strips it.
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


# ---- issue #6103: MLX transformers-version activation must not fail silently ----


def test_activate_transformers_version_or_warn_logs_on_failure(monkeypatch):
    """A failed activation in the MLX fast-path must be logged, not swallowed.

    The non-MLX path already surfaces this failure; the MLX path used a bare
    ``except Exception: pass`` so a missing/broken transformers venv produced
    no trace and a confusing downstream crash.
    """
    warnings_logged = []
    fake_logger = types.SimpleNamespace(
        warning = lambda *a, **k: warnings_logged.append((a, k)),
    )
    monkeypatch.setattr(_worker, "logger", fake_logger)

    def _boom(_name, _hf_token = None):
        raise RuntimeError("venv .venv_t5_550 missing")

    monkeypatch.setattr(_worker, "_activate_transformers_version", _boom)

    # Non-fatal: the MLX path falls through, so this must not raise.
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
    """Execute the real masking block from _run_mlx_training and return its events.

    _run_mlx_training only runs on Apple Silicon, so the block is lifted out and executed
    directly. That keeps the production statements under test rather than a copy of them.
    """
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


try:  # the block imports this lazily; skip the behaviour tests where it cannot load
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
