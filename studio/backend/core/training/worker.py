# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each training job runs in a fresh subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

Pattern follows core/data_recipe/jobs/worker.py.
"""

from __future__ import annotations

import structlog
from loggers import get_logger
import os
import shutil
import sys
import time
import traceback
import subprocess as _sp
from pathlib import Path
from typing import Any, Callable

logger = get_logger(__name__)
from utils.hardware import apply_gpu_ids
from utils.wheel_utils import (
    direct_wheel_url,
    flash_attn_wheel_url,
    install_wheel,
    probe_torch_wheel_env,
    url_exists,
)


def _output_dir_from_resume_checkpoint(
    resume_from_checkpoint: str | None,
) -> str | None:
    if not resume_from_checkpoint:
        return None
    path = Path(resume_from_checkpoint)
    return str(path.parent if path.name.startswith("checkpoint-") else path)


_CAUSAL_CONV1D_RELEASE_TAG = "v1.6.1.post4"
_CAUSAL_CONV1D_PACKAGE_VERSION = "1.6.1"
_MAMBA_SSM_RELEASE_TAG = "v2.3.1"
_MAMBA_SSM_PACKAGE_VERSION = "2.3.1"
_FLASH_ATTN_RUNTIME_MIN_SEQ_LEN = 32768
_FLASH_ATTN_SKIP_ENV = "UNSLOTH_STUDIO_SKIP_FLASHATTN_INSTALL"


def _model_wants_causal_conv1d(model_name: str) -> bool:
    name = model_name.lower()
    return any(
        key in name
        for key in (
            "qwen3.5",
            "qwen3_5",
            "qwen3.6",
            "qwen3_6",
            "qwen3-next",
            "qwen3_next",
            "nemotron_h",
            "nemotron-h",
            "nemotron-3-nano",
            "falcon_h1",
            "falcon-h1",
            "granite-4.0-h",
            "granitemoehybrid",
            "lfm2",
        )
    )


def _causal_conv1d_platform_tag() -> str | None:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
        return None
    # No prebuilt wheels published for macOS or Windows
    return None


def _probe_causal_conv1d_env() -> dict[str, str] | None:
    try:
        probe = _sp.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys, re, torch; "
                    "parts = torch.__version__.split('+', 1)[0].split('.')[:2]; "
                    "minor = re.sub(r'[^0-9].*', '', parts[1]) if len(parts) > 1 else '0'; "
                    "torch_mm = parts[0] + '.' + minor; "
                    "print(json.dumps({"
                    "'python_tag': f'cp{sys.version_info.major}{sys.version_info.minor}', "
                    "'torch_mm': torch_mm, "
                    "'cuda_major': str(int(str(torch.version.cuda).split('.', 1)[0])) if torch.version.cuda else '', "
                    "'cxx11abi': str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()"
                    "}))"
                ),
            ],
            stdout = _sp.PIPE,
            stderr = _sp.PIPE,
            text = True,
            timeout = 30,
        )
    except _sp.TimeoutExpired:
        logger.warning("Torch environment probe timed out after 30s")
        return None
    if probe.returncode != 0:
        logger.warning(
            "Failed to probe torch environment for causal-conv1d wheel:\n%s",
            probe.stdout,
        )
        return None

    try:
        return json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse torch environment probe output: %s", probe.stdout
        )
        return None


def _direct_wheel_url(
    *,
    filename_prefix: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    env: dict[str, str] | None = None,
) -> str | None:
    env = env or _probe_causal_conv1d_env()
    platform_tag = _causal_conv1d_platform_tag()
    if env is None or platform_tag is None or not env.get("cuda_major"):
        return None

    filename = (
        f"{filename_prefix}-{package_version}"
        f"+cu{env['cuda_major']}torch{env['torch_mm']}"
        f"cxx11abi{env['cxx11abi']}-{env['python_tag']}-{env['python_tag']}-{platform_tag}.whl"
    )
    return f"{release_base_url}/{release_tag}/{filename}"


def _url_exists(url: str) -> bool:
    try:
        request = urllib.request.Request(url, method = "HEAD")
        with urllib.request.urlopen(request, timeout = 10):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        logger.warning("Unexpected HTTP error while probing %s: %s", url, exc)
        return False
    except Exception as exc:
        logger.warning("Failed to probe %s: %s", url, exc)
        return False


def _install_package_wheel_first(
    *,
    event_queue: Any,
    import_name: str,
    display_name: str,
    pypi_name: str,
    pypi_version: str | None = None,
    filename_prefix: str | None = None,
    release_tag: str | None = None,
    release_base_url: str | None = None,
    wheel_url_builder: Callable[[dict[str, str] | None], str | None] | None = None,
    pypi_spec: str | None = None,
    pypi_status_message: str | None = None,
) -> bool:
    try:
        __import__(import_name)
        logger.info("%s already installed", display_name)
        return True
    except ImportError:
        pass

    env = probe_torch_wheel_env(timeout = 30)
    if wheel_url_builder is not None:
        wheel_url = wheel_url_builder(env)
    else:
        wheel_url = direct_wheel_url(
            filename_prefix = filename_prefix,
            package_version = pypi_version,
            release_tag = release_tag,
            release_base_url = release_base_url,
            env = env,
        )

    if wheel_url is None:
        logger.info("No compatible %s wheel candidate", display_name)
    elif url_exists(wheel_url):
        _send_status(event_queue, f"Installing prebuilt {display_name} wheel...")
        for installer, result in install_wheel(
            wheel_url,
            python_executable = sys.executable,
            use_uv = bool(shutil.which("uv")),
            run = _sp.run,
        ):
            if result.returncode == 0:
                logger.info("Installed prebuilt %s wheel successfully", display_name)
                return True
            logger.warning(
                "%s failed to install %s wheel:\n%s",
                installer,
                display_name,
                result.stdout,
            )
    else:
        logger.info("No published %s wheel found: %s", display_name, wheel_url)

    is_hip = env and env.get("hip_version")
    if is_hip and not shutil.which("hipcc"):
        logger.error(
            "%s requires hipcc for source compilation on ROCm. "
            "Install the ROCm HIP SDK: https://rocm.docs.amd.com",
            display_name,
        )
        _send_status(
            event_queue,
            f"{display_name}: hipcc not found (ROCm HIP SDK required)",
        )
        return False

    if pypi_spec is None:
        pypi_spec = f"{pypi_name}=={pypi_version}"

    if pypi_status_message is None:
        if is_hip:
            pypi_status_message = (
                f"Compiling {display_name} from source for ROCm "
                "(this may take several minutes)..."
            )
        else:
            pypi_status_message = f"Installing {display_name} from PyPI..."

    _send_status(event_queue, pypi_status_message)

    # Prefer uv for faster dependency resolution when available
    plain_pypi_install = pypi_version is None
    if plain_pypi_install:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                pypi_spec,
            ]
        else:
            pypi_cmd = [sys.executable, "-m", "pip", "install", pypi_spec]
    else:
        if shutil.which("uv"):
            pypi_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-build-isolation",
                "--no-deps",
            ]
            # Avoid stale cache artifacts from partial HIP source builds
            if is_hip:
                pypi_cmd.append("--no-cache")
            pypi_cmd.append(pypi_spec)
        else:
            pypi_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                "--no-cache-dir",
                pypi_spec,
            ]

    # Source compilation on ROCm can take 10-30 minutes; use a generous
    # timeout. Non-HIP installs preserve the pre-existing "no timeout"
    # behaviour so unrelated slow installs (e.g. causal-conv1d source
    # build on Linux aarch64 or unsupported torch/CUDA combinations)
    # are not aborted at 5 minutes by this PR.
    _run_kwargs: dict[str, Any] = {
        "stdout": _sp.PIPE,
        "stderr": _sp.STDOUT,
        "text": True,
    }
    if is_hip:
        _run_kwargs["timeout"] = 1800

    try:
        result = _sp.run(pypi_cmd, **_run_kwargs)
    except _sp.TimeoutExpired:
        logger.error(
            "%s installation timed out after %ds",
            display_name,
            _run_kwargs.get("timeout"),
        )
        _send_status(
            event_queue,
            f"{display_name} installation timed out after "
            f"{_run_kwargs.get('timeout')}s",
        )
        return False

    if result.returncode != 0:
        if is_hip:
            # Surface a clear error for ROCm source build failures
            error_lines = (result.stdout or "").strip().splitlines()
            snippet = "\n".join(error_lines[-5:]) if error_lines else "(no output)"
            logger.error(
                "Failed to compile %s for ROCm:\n%s",
                display_name,
                result.stdout,
            )
            _send_status(
                event_queue,
                f"Failed to compile {display_name} for ROCm. "
                "Check that hipcc and ROCm development headers are installed.\n"
                f"{snippet}",
            )
        else:
            logger.error(
                "Failed to install %s from PyPI:\n%s",
                display_name,
                result.stdout,
            )
        return False

    if is_hip:
        logger.info("Compiled and installed %s from source for ROCm", display_name)
    else:
        logger.info("Installed %s from PyPI", display_name)
    return True


def _ensure_causal_conv1d_fast_path(event_queue: Any, model_name: str) -> None:
    if not _model_wants_causal_conv1d(model_name):
        return

    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = _CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = _CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


_SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
)


def _ensure_mamba_ssm(event_queue: Any, model_name: str) -> None:
    if not any(sub in model_name.lower() for sub in _SSM_MODEL_SUBSTRINGS):
        return

    logger.info("SSM model detected; setting up mamba-ssm after causal-conv1d")
    _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = _MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = _MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


def _should_try_runtime_flash_attn_install(max_seq_length: int) -> bool:
    if os.getenv(_FLASH_ATTN_SKIP_ENV) == "1":
        return False
    if max_seq_length < _FLASH_ATTN_RUNTIME_MIN_SEQ_LEN:
        return False
    return sys.platform.startswith("linux")


def _ensure_flash_attn_for_long_context(event_queue: Any, max_seq_length: int) -> None:
    if not _should_try_runtime_flash_attn_install(max_seq_length):
        return

    installed = _install_package_wheel_first(
        event_queue = event_queue,
        import_name = "flash_attn",
        display_name = "flash-attn",
        pypi_name = "flash-attn",
        wheel_url_builder = flash_attn_wheel_url,
        pypi_spec = "flash-attn",
        pypi_status_message = "Installing flash-attn from PyPI for long-context training...",
    )
    if not installed:
        _send_status(event_queue, "Continuing without flash-attn")


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    Uses get_transformers_tier() to decide between .venv_t5_550/ (5.5.0),
    .venv_t5_530/ (5.3.0), or the default 4.57.x.
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import (
        get_transformers_tier,
        _resolve_base_model,
        _ensure_venv_t5_530_exists,
        _ensure_venv_t5_550_exists,
        _VENV_T5_530_DIR,
        _VENV_T5_550_DIR,
    )

    resolved = _resolve_base_model(model_name)
    tier = get_transformers_tier(resolved)

    if tier == "550":
        if not _ensure_venv_t5_550_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.5.0: .venv_t5_550 missing at {_VENV_T5_550_DIR}"
            )
        if _VENV_T5_550_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_550_DIR)
        logger.info("Activated transformers 5.5.0 from %s", _VENV_T5_550_DIR)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_550_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "530":
        if not _ensure_venv_t5_530_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.3.0: .venv_t5_530 missing at {_VENV_T5_530_DIR}"
            )
        if _VENV_T5_530_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_530_DIR)
        logger.info("Activated transformers 5.3.0 from %s", _VENV_T5_530_DIR)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_530_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def run_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for sending progress/status/error events to parent.
        stop_queue: mp.Queue for receiving stop commands from parent.
        config: Training configuration dict with all parameters.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "unsloth-studio-training-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    apply_gpu_ids(config.get("resolved_gpu_ids"))

    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1a. Auto-enable trust_remote_code for Nemotron models ──
    # NemotronH has config parsing bugs in transformers that require
    # trust_remote_code=True as a workaround. Other transformers 5.x models
    # (Qwen3.5, Gemma 4, etc.) are native and do NOT need it — enabling it
    # bypasses the compiler (disabling fused CE).
    _lowered = model_name.lower()
    if "nemotron" in _lowered and not config.get("trust_remote_code", False):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for Nemotron model: %s",
            model_name,
        )

    # ── 1b. Set up causal-conv1d first, then install mamba-ssm if needed ──
    try:
        _ensure_causal_conv1d_fast_path(event_queue, model_name)
        _ensure_mamba_ssm(event_queue, model_name)
        _ensure_flash_attn_for_long_context(
            event_queue,
            int(config.get("max_seq_length", 2048)),
        )
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": (
                    f"Please choose another model to train, since "
                    f"causal-conv1d / mamba-ssm failed to install "
                    f"with error: {exc}"
                ),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1c. Set fork start method so dataset.map() can multiprocess ──
    # The parent launched us via spawn (clean process), but the compiled
    # SFTTrainer checks get_start_method() and disables num_proc if not "fork".
    # Linux only: fork is the default start method and is safe here (no CUDA
    # context exists yet). macOS defaults to spawn since Python 3.8 because
    # fork is unsafe with macOS frameworks (Metal/MPS, CoreFoundation) --
    # do NOT override on macOS. Windows has no fork at all.
    if sys.platform == "linux":
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force = True)
        except RuntimeError:
            pass  # Already set

    # ── 1c. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer, TrainingProgress
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
        )

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 2b. EMBEDDING MODEL FAST-PATH ──
    # Embedding models use a completely different pipeline (FastSentenceTransformer
    # + SentenceTransformerTrainer + MultipleNegativesRankingLoss) so we branch
    # early and handle the entire flow in a self-contained function.
    if config.get("is_embedding", False):
        try:
            _run_embedding_training(event_queue, stop_queue, config)
        except Exception as exc:
            event_queue.put(
                {
                    "type": "error",
                    "error": str(exc),
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
        return

    # ── 3. Create a fresh trainer instance ──
    trainer = UnslothTrainer()

    # Wire up progress callback → event_queue
    def _on_progress(progress: TrainingProgress):
        has_train_loss = progress.step > 0 and progress.loss is not None
        has_eval_loss = progress.eval_loss is not None
        if has_train_loss or has_eval_loss:
            event_queue.put(
                {
                    "type": "progress",
                    "step": progress.step,
                    "epoch": progress.epoch,
                    "loss": progress.loss,
                    "learning_rate": progress.learning_rate,
                    "total_steps": progress.total_steps,
                    "elapsed_seconds": progress.elapsed_seconds,
                    "eta_seconds": progress.eta_seconds,
                    "grad_norm": progress.grad_norm,
                    "num_tokens": progress.num_tokens,
                    "eval_loss": progress.eval_loss,
                    "status_message": progress.status_message,
                    "ts": time.time(),
                }
            )
        if progress.status_message:
            _send_status(event_queue, progress.status_message)

    trainer.add_progress_callback(_on_progress)

    # Wire up stop_queue polling to trainer.should_stop
    import threading
    import queue as _queue

    def _poll_stop():
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    save = msg.get("save", True)
                    trainer.should_stop = True
                    trainer.save_on_stop = save
                    logger.info("Stop signal received (save=%s)", save)
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 4. Execute the training pipeline ──
    # Order: detect → dataset → model → prepare → train
    # Dataset processing (including LLM-assisted detection) runs BEFORE model
    # loading so both never occupy VRAM at the same time.
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None

        # ── 4a. Lightweight detection + tokenizer (no VRAM) ──
        _send_status(event_queue, "Detecting model type...")
        trainer.pre_detect_and_load_tokenizer(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
        )
        if trainer.should_stop:
            event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
            return

        # ── 4b. Load and format dataset (LLM helper may use VRAM briefly) ──
        _send_status(event_queue, "Loading and formatting dataset...")
        hf_dataset = config.get("hf_dataset", "")
        dataset_result = trainer.load_and_format_dataset(
            dataset_source = hf_dataset if hf_dataset and hf_dataset.strip() else None,
            format_type = config.get("format_type", ""),
            local_datasets = config.get("local_datasets") or None,
            local_eval_datasets = config.get("local_eval_datasets") or None,
            custom_format_mapping = config.get("custom_format_mapping"),
            subset = config.get("subset"),
            train_split = config.get("train_split", "train"),
            eval_split = config.get("eval_split"),
            eval_steps = config.get("eval_steps", 0.00),
            dataset_slice_start = config.get("dataset_slice_start"),
            dataset_slice_end = config.get("dataset_slice_end"),
        )

        if isinstance(dataset_result, tuple):
            dataset, eval_dataset = dataset_result
        else:
            dataset = dataset_result
            eval_dataset = None

        # [DEBUG] Print first sample before model is loaded
        # dataset is a dict {"dataset": <Dataset>, "detected_format": ..., ...}
        # or a raw Dataset for audio paths
        # try:
        #     ds = dataset["dataset"] if isinstance(dataset, dict) else dataset
        #     print(
        #         f"\n[DEBUG] Dataset loaded BEFORE model. type={type(ds).__name__}, len={len(ds)}",
        #         flush = True,
        #     )
        #     print(f"[DEBUG] Columns: {ds.column_names}", flush = True)
        #     sample = ds[0]
        #     preview = {k: str(v)[:300] for k, v in sample.items()}
        #     print(f"[DEBUG] First sample: {preview}\n", flush = True)
        # except Exception as e:
        #     print(
        #         f"[DEBUG] Could not preview first sample: {type(e).__name__}: {e}",
        #         flush = True,
        #     )

        # Disable eval if eval_steps <= 0
        eval_steps = config.get("eval_steps", 0.00)
        if eval_steps is not None and float(eval_steps) <= 0:
            eval_dataset = None

        # Tell the parent process that eval is configured so the frontend
        # shows "Waiting for first evaluation step..." instead of "not configured"
        if eval_dataset is not None:
            event_queue.put(
                {
                    "type": "eval_configured",
                    "ts": time.time(),
                }
            )

        if dataset is None or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to load dataset",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── Start tqdm monitor early so it captures download + tokenization bars ──
        import threading as _th

        _tqdm_stop = _th.Event()

        def _monitor_tqdm():
            from tqdm.auto import tqdm as _tqdm_cls

            while not _tqdm_stop.is_set():
                for bar in list(getattr(_tqdm_cls, "_instances", set())):
                    try:
                        n, total = bar.n or 0, bar.total or 0
                        desc = getattr(bar, "desc", "") or ""
                        if total > 0 and n > 0 and desc:
                            pct = min(int(n * 100 / total), 100)
                            _send_status(
                                event_queue, f"{desc.strip()} {pct}% ({n:,}/{total:,})"
                            )
                    except (AttributeError, ReferenceError):
                        pass
                _tqdm_stop.wait(3)

        _tqdm_thread = _th.Thread(target = _monitor_tqdm, daemon = True)
        _tqdm_thread.start()

        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        # ── 4c. Load training model (uses VRAM — dataset already formatted) ──
        _send_status(event_queue, "Loading model...")
        success = trainer.load_model(
            model_name = model_name,
            max_seq_length = config["max_seq_length"],
            load_in_4bit = config["load_in_4bit"],
            full_finetuning = not use_lora,
            hf_token = hf_token,
            is_dataset_image = config.get("is_dataset_image", False),
            is_dataset_audio = config.get("is_dataset_audio", False),
            trust_remote_code = config.get("trust_remote_code", False),
            gpu_ids = config.get("resolved_gpu_ids"),
        )
        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                error_msg = trainer.training_progress.error or "Failed to load model"
                event_queue.put(
                    {
                        "type": "error",
                        "error": error_msg,
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # ── 4d. Prepare model (LoRA or full finetuning) ──
        if use_lora:
            _send_status(event_queue, "Configuring LoRA adapters...")
            success = trainer.prepare_model_for_training(
                use_lora = True,
                finetune_vision_layers = config.get("finetune_vision_layers", True),
                finetune_language_layers = config.get("finetune_language_layers", True),
                finetune_attention_modules = config.get(
                    "finetune_attention_modules", True
                ),
                finetune_mlp_modules = config.get("finetune_mlp_modules", True),
                target_modules = config.get("target_modules"),
                lora_r = config.get("lora_r", 16),
                lora_alpha = config.get("lora_alpha", 16),
                lora_dropout = config.get("lora_dropout", 0.0),
                use_gradient_checkpointing = config.get(
                    "gradient_checkpointing", "unsloth"
                ),
                use_rslora = config.get("use_rslora", False),
                use_loftq = config.get("use_loftq", False),
            )
        else:
            _send_status(event_queue, "Preparing model for full finetuning...")
            success = trainer.prepare_model_for_training(use_lora = False)

        if not success or trainer.should_stop:
            if trainer.should_stop:
                event_queue.put(
                    {"type": "complete", "output_dir": None, "ts": time.time()}
                )
            else:
                event_queue.put(
                    {
                        "type": "error",
                        "error": trainer.training_progress.error
                        or "Failed to prepare model",
                        "stack": "",
                        "ts": time.time(),
                    }
                )
            return

        # Convert learning rate
        try:
            lr_value = float(config.get("learning_rate", "2e-4"))
        except ValueError:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Invalid learning rate: {config.get('learning_rate')}",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Generate output dir
        resume_from_checkpoint = config.get("resume_from_checkpoint")
        output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
            resume_from_checkpoint
        )
        if not output_dir:
            output_dir = f"{model_name.replace('/', '_')}_{int(time.time())}"
        output_dir = str(resolve_output_dir(output_dir))
        ensure_dir(Path(output_dir))

        tensorboard_dir = config.get("tensorboard_dir")
        if config.get("enable_tensorboard", False):
            tensorboard_dir = str(resolve_tensorboard_dir(tensorboard_dir))
            ensure_dir(Path(tensorboard_dir))

        # Start training (directly — no inner thread, we ARE the subprocess)
        dataset_display = (
            config.get("hf_dataset", "") or config.get("uploaded_file", "") or ""
        )
        _send_status(
            event_queue,
            f'Training "{model_name}"'
            + (f"\nDataset = {dataset_display}" if dataset_display else ""),
        )
        max_steps = config.get("max_steps", 0)
        save_steps = config.get("save_steps", 0)

        trainer._train_worker(
            dataset,
            output_dir = output_dir,
            num_epochs = config.get("num_epochs", 3),
            learning_rate = lr_value,
            batch_size = config.get("batch_size", 2),
            gradient_accumulation_steps = config.get("gradient_accumulation_steps", 4),
            warmup_steps = config.get("warmup_steps"),
            warmup_ratio = config.get("warmup_ratio"),
            max_steps = max_steps if max_steps and max_steps > 0 else 0,
            save_steps = save_steps if save_steps and save_steps > 0 else 0,
            weight_decay = config.get("weight_decay", 0.001),
            random_seed = config.get("random_seed", 3407),
            packing = config.get("packing", False),
            train_on_completions = config.get("train_on_completions", False),
            enable_wandb = config.get("enable_wandb", False),
            wandb_project = config.get("wandb_project", "unsloth-training"),
            wandb_token = config.get("wandb_token"),
            enable_tensorboard = config.get("enable_tensorboard", False),
            tensorboard_dir = tensorboard_dir,
            eval_dataset = eval_dataset,
            eval_steps = eval_steps,
            max_seq_length = config.get("max_seq_length", 2048),
            optim = config.get("optim", "adamw_8bit"),
            lr_scheduler_type = config.get("lr_scheduler_type", "linear"),
            resume_from_checkpoint = resume_from_checkpoint,
        )

        _tqdm_stop.set()

        # Check final state
        progress = trainer.get_training_progress()
        if progress.error:
            event_queue.put(
                {
                    "type": "error",
                    "error": progress.error,
                    "stack": "",
                    "ts": time.time(),
                }
            )
        else:
            saved_output_dir = (
                None if trainer.should_stop and not trainer.save_on_stop else output_dir
            )
            event_queue.put(
                {
                    "type": "complete",
                    "output_dir": saved_output_dir,
                    "status_message": progress.status_message or "Training completed",
                    "ts": time.time(),
                }
            )

    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": str(exc),
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )


def _send_status(event_queue: Any, message: str) -> None:
    """Send a status update to the parent process."""
    event_queue.put(
        {
            "type": "status",
            "message": message,
            "ts": time.time(),
        }
    )


def _run_embedding_training(event_queue: Any, stop_queue: Any, config: dict) -> None:
    """Self-contained embedding model training pipeline.

    Uses FastSentenceTransformer + SentenceTransformerTrainer +
    MultipleNegativesRankingLoss — completely separate from the
    LLM/VLM/audio paths in UnslothTrainer.

    Mirrors the pattern from the reference embedding notebooks:
      All_MiniLM_L6_v2.py, BGE_M3.py, EmbeddingGemma_300M.py,
      ModernBert.py, Qwen3_Embedding_0_6B.py
    """
    import math
    import queue as _queue
    import threading

    model_name = config["model_name"]
    training_start_time = time.time()

    # ── 1. Import embedding-specific libraries ──
    _send_status(event_queue, "Importing embedding libraries...")
    try:
        from unsloth import FastSentenceTransformer, is_bfloat16_supported
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
        from sentence_transformers.training_args import BatchSamplers
        from datasets import load_dataset, Dataset
        from transformers import TrainerCallback
        from utils.paths import datasets_root, resolve_output_dir
    except ImportError as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import embedding libraries: {e}. "
                "Ensure 'sentence_transformers' and 'unsloth' are installed.",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── Stop signal handling ──
    _should_stop = False
    _save_on_stop = True

    def _poll_stop():
        nonlocal _should_stop, _save_on_stop
        while True:
            try:
                msg = stop_queue.get(timeout = 1.0)
                if msg and msg.get("type") == "stop":
                    _save_on_stop = msg.get("save", True)
                    _should_stop = True
                    logger.info(
                        "Embedding training: stop signal received (save=%s)",
                        _save_on_stop,
                    )
                    return
            except _queue.Empty:
                continue
            except (EOFError, OSError):
                return

    stop_thread = threading.Thread(target = _poll_stop, daemon = True)
    stop_thread.start()

    # ── 2. Load model ──
    _send_status(event_queue, "Loading embedding model...")
    try:
        hf_token = config.get("hf_token", "")
        hf_token = hf_token if hf_token and hf_token.strip() else None
        max_seq_length = config.get("max_seq_length", 512)
        training_type = config.get("training_type", "LoRA/QLoRA")
        use_lora = training_type == "LoRA/QLoRA"

        model = FastSentenceTransformer.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            full_finetuning = not use_lora,
            token = hf_token,
        )
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load embedding model '{model_name}': {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 3. Apply LoRA ──
    if use_lora:
        _send_status(event_queue, "Configuring LoRA adapters (FEATURE_EXTRACTION)...")
        try:
            gradient_checkpointing = config.get("gradient_checkpointing", False)
            # Normalize: "none" or empty → False
            if gradient_checkpointing in ("none", "", None):
                gradient_checkpointing = False

            model = FastSentenceTransformer.get_peft_model(
                model,
                r = config.get("lora_r", 32),
                target_modules = config.get("target_modules")
                or ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = config.get("lora_alpha", 64),
                lora_dropout = config.get("lora_dropout", 0.0),
                bias = "none",
                use_gradient_checkpointing = gradient_checkpointing,
                random_state = config.get("random_seed", 3407),
                use_rslora = config.get("use_rslora", False),
                loftq_config = {"loftq_bits": 4, "loftq_iter": 1}
                if config.get("use_loftq")
                else None,
                task_type = "FEATURE_EXTRACTION",
            )
        except Exception as e:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"Failed to configure LoRA for embedding model: {e}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
            return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 4. Load dataset ──
    _send_status(event_queue, "Loading dataset...")
    try:
        hf_dataset = config.get("hf_dataset", "")
        local_datasets = config.get("local_datasets") or []
        subset = config.get("subset") or None
        train_split = config.get("train_split", "train") or "train"

        if hf_dataset and hf_dataset.strip():
            hf_token = config.get("hf_token", "")
            hf_token = hf_token if hf_token and hf_token.strip() else None
            dataset = load_dataset(
                hf_dataset.strip(),
                subset,
                split = train_split,
                token = hf_token,
            )
        elif local_datasets:
            # Load from local file(s) — mirrors the non-embedding pipeline's
            # directory handling so recipe outputs (parquet-files/) work.
            all_files: list[str] = []
            for dataset_file in local_datasets:
                file_path = (
                    dataset_file
                    if os.path.isabs(dataset_file)
                    else os.path.join(
                        str(datasets_root()),
                        dataset_file,
                    )
                )
                if os.path.isdir(file_path):
                    file_path_obj = Path(file_path)
                    parquet_dir = (
                        file_path_obj / "parquet-files"
                        if (file_path_obj / "parquet-files").exists()
                        else file_path_obj
                    )
                    parquet_files = sorted(parquet_dir.glob("*.parquet"))
                    if parquet_files:
                        all_files.extend(str(p) for p in parquet_files)
                        continue
                    candidates: list[Path] = []
                    for ext in (".json", ".jsonl", ".csv", ".parquet"):
                        candidates.extend(sorted(file_path_obj.glob(f"*{ext}")))
                    if candidates:
                        all_files.extend(str(c) for c in candidates)
                        continue
                    raise ValueError(
                        f"No supported data files in directory: {file_path_obj}"
                    )
                else:
                    all_files.append(file_path)

            if all_files:
                first_ext = Path(all_files[0]).suffix.lower()
                if first_ext in (".json", ".jsonl"):
                    loader = "json"
                elif first_ext == ".csv":
                    loader = "csv"
                elif first_ext == ".parquet":
                    loader = "parquet"
                else:
                    raise ValueError(
                        f"Unsupported local dataset format: {all_files[0]}"
                    )
                dataset = load_dataset(loader, data_files = all_files, split = "train")
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": "No dataset specified for embedding training.",
                    "stack": "",
                    "ts": time.time(),
                }
            )
            return

        # Apply dataset slicing if specified
        slice_start = config.get("dataset_slice_start")
        slice_end = config.get("dataset_slice_end")
        if slice_start is not None or slice_end is not None:
            start = slice_start if slice_start is not None else 0
            end = slice_end if slice_end is not None else len(dataset)
            dataset = dataset.select(range(start, min(end + 1, len(dataset))))

        logger.info(f"Embedding dataset loaded: {len(dataset)} samples")
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to load dataset: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    if _should_stop:
        event_queue.put({"type": "complete", "output_dir": None, "ts": time.time()})
        return

    # ── 5. Create loss function ──
    loss = MultipleNegativesRankingLoss(model)

    # ── 6. Build training arguments ──
    _send_status(event_queue, "Configuring training...")
    try:
        lr_value = float(config.get("learning_rate", "2e-4"))
    except ValueError:
        event_queue.put(
            {
                "type": "error",
                "error": f"Invalid learning rate: {config.get('learning_rate')}",
                "stack": "",
                "ts": time.time(),
            }
        )
        return

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    output_dir = config.get("output_dir") or _output_dir_from_resume_checkpoint(
        resume_from_checkpoint
    )
    if not output_dir:
        output_dir = str(
            resolve_output_dir(f"{model_name.replace('/', '_')}_{int(time.time())}")
        )
    output_dir = str(resolve_output_dir(output_dir))

    num_epochs = config.get("num_epochs", 2)
    batch_size = config.get("batch_size", 256)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_steps_val = config.get("max_steps", 0)
    save_steps_val = config.get("save_steps", 0)
    warmup_ratio = config.get("warmup_ratio", 0.03)
    warmup_steps_val = config.get("warmup_steps")
    log_frequency = config.get("log_frequency", 50)

    # Build args dict
    training_args_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": lr_value,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 1,
        "report_to": ["wandb"] if config.get("enable_wandb") else "none",
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "optim": config.get("optim", "adamw_8bit"),
        "weight_decay": config.get("weight_decay", 0.001),
        "seed": config.get("random_seed", 3407),
    }

    # max_steps vs epochs
    if max_steps_val and max_steps_val > 0:
        training_args_kwargs["max_steps"] = max_steps_val
    else:
        training_args_kwargs["num_train_epochs"] = num_epochs if num_epochs > 0 else 2

    # warmup: prefer warmup_ratio (standard for embedding scripts), fallback to steps
    if warmup_ratio is not None and warmup_ratio > 0:
        training_args_kwargs["warmup_ratio"] = warmup_ratio
    elif warmup_steps_val is not None and warmup_steps_val > 0:
        training_args_kwargs["warmup_steps"] = warmup_steps_val

    # save_steps
    if save_steps_val and save_steps_val > 0:
        training_args_kwargs["save_steps"] = save_steps_val
        training_args_kwargs["save_strategy"] = "steps"

    args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # ── 7. Calculate total steps for progress tracking ──
    if max_steps_val and max_steps_val > 0:
        total_steps = max_steps_val
    else:
        effective_epochs = num_epochs if num_epochs > 0 else 2
        len_dataloader = math.ceil(len(dataset) / batch_size)
        steps_per_epoch = max(len_dataloader // gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * effective_epochs

    # ── 8. Create progress callback ──
    class _EmbeddingProgressCallback(TrainerCallback):
        """Sends training progress events to the parent process via event_queue."""

        def on_log(self, args, state, control, logs = None, **kwargs):
            if not logs:
                return
            loss_value = logs.get("loss", logs.get("train_loss", None))
            current_step = state.global_step

            elapsed = time.time() - training_start_time
            eta = None
            if current_step > 0 and total_steps > 0:
                remaining = total_steps - current_step
                if remaining > 0:
                    eta = (elapsed / current_step) * remaining

            event_queue.put(
                {
                    "type": "progress",
                    "step": current_step,
                    "epoch": round(state.epoch, 2) if state.epoch else 0,
                    "loss": loss_value,
                    "learning_rate": logs.get("learning_rate", None),
                    "total_steps": total_steps,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "grad_norm": logs.get("grad_norm"),
                    "num_tokens": getattr(state, "num_input_tokens_seen", None),
                    "eval_loss": logs.get("eval_loss"),
                    "status_message": "",
                    "ts": time.time(),
                }
            )

        def on_step_end(self, args, state, control, **kwargs):
            if _should_stop:
                logger.info("Embedding training: stop at step %d", state.global_step)
                control.should_training_stop = True
                return control

    # ── 9. Create trainer and train ──
    _send_status(event_queue, "Starting embedding training...")
    try:
        trainer = SentenceTransformerTrainer(
            model = model,
            train_dataset = dataset,
            loss = loss,
            args = args,
            callbacks = [_EmbeddingProgressCallback()],
        )

        trainer.train(resume_from_checkpoint = resume_from_checkpoint)
    except Exception as e:
        event_queue.put(
            {
                "type": "error",
                "error": f"Embedding training failed: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 10. Save model ──
    if _should_stop and not _save_on_stop:
        event_queue.put(
            {
                "type": "complete",
                "output_dir": None,
                "status_message": "Training cancelled",
                "ts": time.time(),
            }
        )
        return

    _send_status(event_queue, "Saving model...")
    try:
        if _should_stop and _save_on_stop:
            trainer._save_checkpoint(trainer.model, trial = None)
        model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        logger.info("Embedding model saved to %s", output_dir)
    except Exception as e:
        logger.error("Failed to save embedding model: %s", e)
        event_queue.put(
            {
                "type": "error",
                "error": f"Training completed but failed to save: {e}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 11. Done ──
    event_queue.put(
        {
            "type": "complete",
            "output_dir": output_dir,
            "status_message": "Embedding training completed",
            "ts": time.time(),
        }
    )
