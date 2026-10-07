# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, importlib.util, platform, sys

os.environ["UNSLOTH_IS_PRESENT"] = "1"

# Opt into ROCm AOTriton kernels PyTorch still gates as experimental; it keeps its own hardware
# checks and reads this lazily at the SDPA probe, so no torch import here. `setdefault` preserves
# an explicit override, including "0".
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Before transformers, which reads sentencepiece availability during its own import. On Windows
# the extension is never imported at all: a code integrity policy can refuse it by reputation,
# and any probe to find out whether this machine will is itself the refusal the user sees. See
# import_fixes.disable_sentencepiece_on_windows; UNSLOTH_DISABLE_SENTENCEPIECE=0 opts out.
try:
    from .import_fixes import disable_sentencepiece_on_windows as _no_sentencepiece
    _no_sentencepiece()
    del _no_sentencepiece
except Exception:
    pass

# Transformers 4.x imports TensorFlow / Flax merely because they are installed (processing_utils
# -> image_transforms); it reads these variables once at its own import, so they have to land
# first. An explicit opt-in still wins, and 5.x ignores all of this.
if "transformers" not in sys.modules:
    _TRUE = {"1", "ON", "YES", "TRUE"}
    # Overwrite, not `setdefault`: unset means AUTO ("enable if installed"), but an already-imported
    # backend is in use, so opting it out breaks a `from_tf` load.
    for _var, _modules, _opt_ins in (
        ("USE_TF", ("tensorflow",), ("USE_TF", "FORCE_TF_AVAILABLE")),
        ("USE_FLAX", ("flax", "jax"), ("USE_FLAX",)),
    ):
        if any(_m in sys.modules for _m in _modules):
            continue
        if any(os.environ.get(_v, "").upper() in _TRUE for _v in _opt_ins):
            continue
        os.environ[_var] = "0"
    del _TRUE, _var, _modules, _opt_ins
else:
    # Transformers derives _tf_available / _flax_available from find_spec alone, and being in
    # sys.modules does not mean its body ran, so clear the cached flags AND write the variables:
    # inert once read, decisive in that window, unconditional since waiting deadlocks.
    _TRUE = {"1", "ON", "YES", "TRUE"}
    _import_utils = sys.modules.get("transformers.utils.import_utils")
    for _var, _flag, _const, _modules, _opt_ins, _cached in (
        (
            "USE_TF",
            "_tf_available",
            "USE_TF",
            ("tensorflow",),
            ("USE_TF", "FORCE_TF_AVAILABLE"),
            ("USE_TF", "FORCE_TF_AVAILABLE"),
        ),
        (
            "USE_FLAX",
            "_flax_available",
            "USE_JAX",
            ("flax", "jax"),
            ("USE_FLAX",),
            ("USE_JAX",),
        ),
    ):
        if any(_m in sys.modules for _m in _modules):
            continue
        if any(os.environ.get(_v, "").upper() in _TRUE for _v in _opt_ins):
            continue
        # An opt-in can be consumed and then restored, so read the snapshot Transformers used. Env USE_FLAX
        # lands in the constant USE_JAX.
        if any(str(getattr(_import_utils, _v, "")).upper() in _TRUE for _v in _cached):
            continue
        os.environ[_var] = "0"
        try:
            # import_utils copies the env into USE_TF / USE_JAX (lines 102-104) and derives the flags at
            # 264 / 355, so mid-body only the constant works.
            if hasattr(_import_utils, _const):
                setattr(_import_utils, _const, "0")
            # Absent on 5.x, and a module proxy can refuse the write.
            if getattr(_import_utils, _flag, False):
                setattr(_import_utils, _flag, False)
        except (AttributeError, TypeError):
            pass
    del _TRUE, _import_utils, _var, _flag, _const, _modules, _opt_ins, _cached

# Relax Metal's context-store timeout before MLX modules can initialize Metal; an explicit user
# value stays authoritative.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    os.environ.setdefault("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1")

# Legacy Windows consoles (cp1252) cannot encode Unsloth's emoji/box-drawing glyphs and crash with
# UnicodeEncodeError; errors="replace" guarantees no crash on an unencodable glyph.
# ── Windows console UTF-8 safety ─────────────────────────────────────────────
if platform.system() == "Windows":
    import sys as _sys
    for _name in ("stdout", "stderr"):
        _s = getattr(_sys, _name, None)
        try:
            _enc = (getattr(_s, "encoding", None) or "").lower()
            if _s is not None and hasattr(_s, "reconfigure") and "utf" not in _enc:
                _s.reconfigure(encoding = "utf-8", errors = "replace")
        except Exception:
            pass


class _UnslothDeviceStats:
    """Portable device metadata used by backend memory-reporting helpers."""

    def __init__(
        self,
        name,
        total_memory = 0,
    ):
        """Store a display name and total memory in bytes."""
        self.name = name
        self.total_memory = int(total_memory or 0)
        self.major = 0
        self.minor = 0
        self.multi_processor_count = 0


def _bytes_to_gb(value):
    """Convert byte counts to GiB rounded"""
    return round(float(value or 0) / 1024 / 1024 / 1024, 3)


def _is_mlx_available():
    # Transitional import barrier: keep non-Apple-Silicon imports from touching unsloth_zoo until
    # unsloth_zoo.mlx is import-safe on GPU hosts.
    if (
        os.environ.get("UNSLOTH_FORCE_GPU_PATH", "0") == "1"
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or importlib.util.find_spec("mlx") is None
    ):
        return False
    try:
        from unsloth_zoo.mlx import is_mlx_available
    except ImportError:
        return False
    return is_mlx_available()


# Detect Apple Silicon + MLX before any torch/numpy imports
_IS_MLX = _is_mlx_available()

if _IS_MLX:
    # Same reason again, and first because it is what turns the bare AttributeError into a
    # diagnosis: this branch imports transformers below, so an Apple Silicon host carrying the
    # old-torch/new-transformers pair hits #8933 here exactly as a CUDA host does, and
    # _gpu_init.py, the only other installation site, is never reached on this path. The
    # triton shim check is deliberately NOT mirrored: it is a CUDA/ROCm/XPU driver shim and
    # there is no triton on this platform to inspect.
    try:
        from .import_fixes import patch_torch_missing_attribute_error as _patch_torch_attr
        _patch_torch_attr()
        del _patch_torch_attr
    except Exception:
        pass
    # _gpu_init does this on the GPU path and the MLX path never reaches it, so torchao 0.18 + torch <
    # 2.10 dies on `ScalingType`.
    try:
        from .import_fixes import fix_torchao_torch_symbol_skew as _fix_torchao
        _fix_torchao()
        del _fix_torchao
    except Exception:
        pass
    try:
        # Same reason: MLX audio reaches xcodec2 -> torchtune -> the old torchao.dtypes.nf4tensor path.
        from .import_fixes import fix_torchao_nf4tensor_move as _fix_nf4
        _fix_nf4()
        del _fix_nf4
    except Exception:
        pass
    try:
        from .import_fixes import fix_transformers_validate_rope_ignore_keys as _fix_validate_rope
        _fix_validate_rope()
        del _fix_validate_rope
    except Exception:
        pass
    try:
        from .import_fixes import fix_transformers_is_torch_fx_available as _fix_torch_fx
        _fix_torch_fx()
        del _fix_torch_fx
    except Exception:
        pass
    try:
        # Same reason: this branch imports transformers itself further down, so a --no-deps floor miss would
        # surface here with the same wrong remedy.
        from .import_fixes import check_transformers_dependency_versions as _check_tf_deps
        _check_tf_deps()
        del _check_tf_deps
    except Exception:
        pass
    try:
        # Same reason: remote code reaches transformers' get_class_in_module on this platform
        # too, and the wrap is what restores the image helpers transformers 5 stopped
        # re-exporting. Costs nothing until a checkpoint's own modeling file is loaded.
        from .import_fixes import (
            fix_transformers5_image_processing_reexports as _fix_image_reexports,
        )
        _fix_image_reexports()
        del _fix_image_reexports
    except Exception:
        pass
    try:
        # Same reason: 4.x remote configs are built here too, and their validators read plain RoPE
        # as rope_scaling None. is_torch_fx_available is left to unsloth_zoo.mlx.loader.
        from .import_fixes import (
            fix_transformers_remote_rope_scaling_none as _fix_remote_rope_scaling,
        )
        _fix_remote_rope_scaling()
        del _fix_remote_rope_scaling
    except Exception:
        pass
    try:
        from .import_fixes import fix_transformers5_legacy_config_types as _fix_legacy_types
        _fix_legacy_types()
        del _fix_legacy_types
    except Exception:
        pass
    try:
        # Same reason: MLX loads hub configs and saves tokenizers through transformers too.
        from .import_fixes import (
            fix_transformers_untrusted_config_fields as _fix_untrusted_config,
            fix_transformers_chat_template_path_traversal as _fix_template_names,
        )

        _fix_untrusted_config()
        _fix_template_names()
        del _fix_untrusted_config, _fix_template_names
    except Exception:
        pass
    try:
        import unsloth_zoo
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires `unsloth-zoo` with MLX modules. "
            "Reinstall with `pip install unsloth-zoo` or rerun install.sh."
        ) from _e
    # An older unsloth-zoo satisfies `import unsloth_zoo` but lacks the mlx.trainer / mlx.loader
    # submodules; surface an install hint instead of a raw ImportError.
    try:
        from unsloth_zoo.mlx.trainer import (
            MLXTrainer,
            MLXTrainingConfig,
            _is_vlm_model,
            _normalize_mlx_optimizer_name,
        )
        from unsloth_zoo.mlx.loader import FastMLXModel
    except ImportError as _e:
        raise ImportError(
            "Unsloth: MLX support requires an unsloth-zoo build that includes "
            "`unsloth_zoo.mlx.trainer` and `unsloth_zoo.mlx.loader`. Upgrade with "
            "`pip install -U unsloth-zoo` or rerun install.sh."
        ) from _e

    import dataclasses as _dataclasses
    import inspect as _inspect
    import importlib.machinery as _machinery
    import sys as _sys
    import types as _types
    import warnings as _warnings

    # unsloth_zoo is a different distribution, pinned >=, so borrowing its number reported neither the
    # installed core nor the latest zoo. `_version` imports nothing, so this stays torch-free.
    from ._version import __version__

    DEVICE_TYPE = "mlx"
    _MLX_TRAINER_ACCEPTS_VAR_KWARGS = False
    _MLX_TRAINER_SUPPORTED_KWARGS = frozenset()
    try:
        _MLX_TRAINER_INIT_PARAMETERS = _inspect.signature(MLXTrainer.__init__).parameters
        _MLX_TRAINER_ACCEPTS_VAR_KWARGS = any(
            param.kind is _inspect.Parameter.VAR_KEYWORD
            for param in _MLX_TRAINER_INIT_PARAMETERS.values()
        )
        _MLX_TRAINER_SUPPORTED_KWARGS = frozenset(
            name
            for name, param in _MLX_TRAINER_INIT_PARAMETERS.items()
            if name != "self"
            and param.kind
            in (
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                _inspect.Parameter.KEYWORD_ONLY,
            )
        )
    except (TypeError, ValueError):
        pass

    def _mlx_trainer_supports_kwarg(name):
        """Return whether the installed zoo MLXTrainer accepts a kwarg."""
        return _MLX_TRAINER_ACCEPTS_VAR_KWARGS or name in _MLX_TRAINER_SUPPORTED_KWARGS

    def _is_mlx_cuda_device_target(device):
        """Return True when a torch .to/.cuda target asks for CUDA on MLX."""
        if device is None:
            return False
        return str(device).lower().startswith("cuda")

    def _patch_mlx_batch_encoding_to_cuda():
        """Treat tokenizer_output.to("cuda") as a no-op on the MLX backend."""
        try:
            from transformers.tokenization_utils_base import BatchEncoding
        except Exception:
            return

        original_to = getattr(BatchEncoding, "to", None)
        if original_to is None or getattr(original_to, "_unsloth_mlx_cuda_noop", False):
            return

        def batch_encoding_to(
            self,
            device = None,
            *args,
            **kwargs,
        ):
            target = kwargs.get("device", device)
            if _is_mlx_cuda_device_target(target):
                return self
            # device given by keyword: do not also pass the positional None, or the original raises "multiple
            # values for 'device'".
            if "device" in kwargs:
                return original_to(self, *args, **kwargs)
            return original_to(self, device, *args, **kwargs)

        batch_encoding_to._unsloth_mlx_cuda_noop = True
        batch_encoding_to._unsloth_original_to = original_to
        BatchEncoding.to = batch_encoding_to

    _patch_mlx_batch_encoding_to_cuda()

    # Load raw_text helpers without executing dataprep/__init__.py, which imports synthetic.py -> torch
    # and would defeat the torch-free MLX path.
    from pathlib import Path as _Path

    _raw_text_path = _Path(__file__).resolve().parent / "dataprep" / "raw_text.py"
    _raw_text_spec = importlib.util.spec_from_file_location("unsloth._mlx_raw_text", _raw_text_path)
    if _raw_text_spec is None or _raw_text_spec.loader is None:
        raise ImportError("Unsloth: could not load MLX raw_text dataprep helpers.")
    _raw_text = importlib.util.module_from_spec(_raw_text_spec)
    _raw_text_spec.loader.exec_module(_raw_text)
    RawTextDataLoader = _raw_text.RawTextDataLoader
    TextPreprocessor = _raw_text.TextPreprocessor
    del _raw_text, _raw_text_spec, _raw_text_path, _Path

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def get_peft_model(*args, **kwargs):
            return FastMLXModel.get_peft_model(*args, **kwargs)

        @staticmethod
        def for_inference(*args, **kwargs):
            return args[0] if args else None

    class FastVisionModel(FastLanguageModel):
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("text_only", False)
            return FastMLXModel.from_pretrained(*args, **kwargs)

        @staticmethod
        def for_training(*args, **kwargs):
            return args[0] if args else None

    FastTextModel = FastLanguageModel
    FastModel = FastLanguageModel

    class FastSentenceTransformer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

        @staticmethod
        def get_peft_model(*args, **kwargs):
            raise NotImplementedError(
                "Unsloth: FastSentenceTransformer is not yet supported on MLX."
            )

    # Decision models (Laya, Clef). The data, metrics and calibration code is a copy of unsloth/models/decision.py,
    # which cannot be imported here: keep the two in step.
    import copy, dataclasses, functools, json, math, random, tempfile, types, warnings
    from collections import Counter
    from pathlib import Path
    from typing import Callable, Optional

    TRAIN_MAX_LEN, TRAIN_HEAD_MAX_LEN = 1024, 256
    HOLDOUT_MAX = 400
    MIN_CALIBRATION_ITEMS = 10
    QUESTION_TYPES = ("choice", "score", "noul")
    _FILES = ("rl_agent_config.json", "model.safetensors")
    _DIRS = ("encoder", "tokenizer")
    _CLEF_HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
    _ADAPTER_CONFIG = "adapter_config.json"
    CLEF_MAX_LEN = 4096
    CLEF_SERVE_MAX_LEN = 16384
    # laya 0.3.5 ships inside Unsloth for Studio's Decision API (studio/backend/vendor/README.md).
    _VENDORED_LAYA = (
        Path(__file__).resolve().parents[1]
        / "studio"
        / "backend"
        / "vendor"
        / "laya"
        / "__init__.py"
    ).resolve()

    class DecisionDataError(ValueError):
        pass

    @functools.lru_cache(maxsize = None)
    def _laya():
        # By path, so a pip installed "laya" is never used or replaced; Studio registers this copy.
        for name in ("laya", "unsloth._laya"):
            module = sys.modules.get(name)
            if (
                getattr(module, "__file__", None)
                and Path(module.__file__).resolve() == _VENDORED_LAYA
            ):
                return module
        if not _VENDORED_LAYA.is_file():
            raise ImportError(
                f"Unsloth: decision models need laya, which ships with Unsloth at {_VENDORED_LAYA.parent}. "
                "Please reinstall Unsloth."
            )
        spec = importlib.util.spec_from_file_location(
            "unsloth._laya", _VENDORED_LAYA, submodule_search_locations = [str(_VENDORED_LAYA.parent)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # laya's JSON reads and writes use a bare open(), which is the locale encoding on Windows.
        module.agent.open = functools.partial(open, encoding = "utf-8")
        return module

    def is_decision_checkpoint(folder) -> bool:
        folder = Path(folder)
        return is_clef_checkpoint(folder) or (
            all((folder / name).is_file() for name in _FILES)
            and all((folder / name).is_dir() for name in _DIRS)
        )

    def is_clef_checkpoint(folder) -> bool:
        # Cloudflare's Clef layout: a backbone (merged, or LoRA adapters over a base) plus a joint schema head.
        folder = Path(folder)
        return all((folder / name).is_file() for name in _CLEF_HEAD_FILES) and (
            (folder / "config.json").is_file() or (folder / _ADAPTER_CONFIG).is_file()
        )

    def _is_clef_adapter(folder) -> bool:
        folder = Path(folder)
        return not (folder / "config.json").is_file() and (folder / _ADAPTER_CONFIG).is_file()

    def _is_clef_repo(model_name, prefix, token, revision) -> Optional[bool]:
        # Asked up front: Unsloth's download wrapper rejects a snapshot that lacks an exact file it was
        # asked for, so a Laya pattern on a Clef repo (or the reverse) would fail as "incomplete".
        from huggingface_hub import HfApi, constants

        if constants.HF_HUB_OFFLINE:
            return None
        try:
            files = HfApi(token = token).list_repo_files(model_name, revision = revision)
        except Exception:
            return None
        return prefix + _CLEF_HEAD_FILES[1] in files

    def _is_plain_lm(model_name, subfolder, token, revision, local_files_only) -> bool:
        # Unknown (offline, no access) answers False, so the checkpoint loader names what is missing.
        markers = {_FILES[0], _CLEF_HEAD_FILES[1]}
        if subfolder:
            return False
        root = Path(str(model_name)).expanduser()
        if root.is_dir():
            has = lambda name: (root / name).is_file()
            return (has("config.json") or has(_ADAPTER_CONFIG)) and not any(map(has, markers))
        from huggingface_hub import constants

        if not (local_files_only or constants.HF_HUB_OFFLINE):
            try:
                from huggingface_hub import HfApi
                files = HfApi(token = token).list_repo_files(str(model_name), revision = revision)
            except Exception:
                files = None
            if files is not None:
                names = {name.rsplit("/", 1)[-1] for name in files}
                return bool({"config.json", _ADAPTER_CONFIG} & set(files)) and not (names & markers)
        from huggingface_hub import try_to_load_from_cache

        def cached(name) -> bool:
            try:
                return isinstance(
                    try_to_load_from_cache(str(model_name), name, revision = revision), str
                )
            except Exception:
                return False

        return (cached("config.json") or cached(_ADAPTER_CONFIG)) and not any(map(cached, markers))

    def _checkpoint_folder(model_name, subfolder, token, revision, local_files_only) -> Path:
        root = Path(model_name).expanduser()
        if not root.is_dir():
            from huggingface_hub import snapshot_download as cached_snapshot

            try:
                from unsloth_zoo.hf_xet_fallback import (
                    snapshot_download_with_xet_fallback as snapshot_download,
                )
            except ImportError:
                snapshot_download = cached_snapshot
            prefix = f"{subfolder}/" if subfolder else ""
            laya = [prefix + name for name in _FILES] + [f"{prefix}{name}/*" for name in _DIRS]
            clef = None if local_files_only else _is_clef_repo(model_name, prefix, token, revision)
            if clef is None:
                # Offline: the cache already holds one layout or the other.
                root = Path(
                    cached_snapshot(
                        model_name,
                        token = token,
                        revision = revision,
                        local_files_only = True,
                        allow_patterns = laya + [prefix + "*.json"],
                    )
                )
                clef = (root / prefix / _CLEF_HEAD_FILES[1]).is_file()
            # Laya repos hold several checkpoints in subfolders, so only the asked one is fetched.
            root = Path(
                snapshot_download(
                    model_name,
                    token = token,
                    revision = revision,
                    local_files_only = local_files_only,
                    allow_patterns = [prefix + "*"] if clef else laya,
                )
            )
        folder = root / subfolder if subfolder else root
        if not is_decision_checkpoint(folder):
            raise ValueError(
                f"Unsloth: {folder} is not a decision model checkpoint "
                "(rl_agent_config.json, model.safetensors, encoder/ and tokenizer/, "
                "or a Clef backbone with joint_head.safetensors and joint_head_config.json). "
                'To turn a plain language model into a decision model, pass decision_head = "clef".'
            )
        return folder

    def _lm_subfolder(model_name, subfolder, token, revision, local_files_only) -> str:
        if Path(model_name).expanduser().is_dir():
            return str(Path(model_name).expanduser() / subfolder)
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            model_name,
            allow_patterns = [f"{subfolder}/*"],
            token = token,
            revision = revision,
            local_files_only = local_files_only,
        )
        return str(Path(root) / subfolder)

    def _parsed(value):
        if isinstance(value, str) and value.strip()[:1] in ("{", "["):
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value

    def _internal(question) -> dict:
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            raise DecisionDataError("is not a valid question")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice" and not (
            isinstance(criteria, (dict, list))
            and criteria
            and all(isinstance(option, str) for option in criteria)
        ):
            raise DecisionDataError("needs criteria naming its options")
        if kind == "score" and not (isinstance(criteria, list) and criteria):
            raise DecisionDataError("needs a list of criteria levels")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise DecisionDataError('criteria may only have "true" and "false"')
        laya_question = {"type": kind, "instructions": question.get("instructions") or ""}
        if criteria is not None:
            laya_question["criteria"] = criteria
        return _laya().agent.Agent._to_internal(laya_question)

    def _option_keys(internal: dict) -> list:
        if internal["t"] == "choice":
            return [str(key) for key in internal["crit"]]
        if internal["t"] == "noul":
            return ["false", "true"]
        return [str(i) for i in range(len(internal["crit"]))]

    def _label(kind: str, label):
        if kind == "score" and isinstance(label, (int, float, str)) and not isinstance(label, bool):
            try:
                level = float(label)
            except ValueError:
                return None
            return str(int(level)) if level.is_integer() else None
        if label is None:
            return None
        return str(label).strip().lower() if kind == "noul" else str(label)

    def _target(internal: dict, gold) -> tuple:
        return _target_for(internal["t"], _option_keys(internal), gold)

    def _target_for(kind: str, keys: list, gold) -> tuple:
        if not isinstance(gold, dict):
            gold = {"label": gold}
        label = _label(kind, gold.get("label"))
        probabilities = gold.get("probabilities")
        if kind == "noul" and not isinstance(probabilities, dict):
            noul = gold.get("noul")
            probabilities = {"true": noul} if isinstance(noul, (int, float)) else None
        if isinstance(probabilities, dict):
            try:
                values = {str(key): float(value) for key, value in probabilities.items()}
            except (TypeError, ValueError):
                values = {}
            if not all(math.isfinite(value) for value in values.values()):
                raise DecisionDataError("gold has probabilities that are not finite numbers")
            if kind == "noul" and len(values.keys() & {"false", "true"}) == 1:
                known = "true" if "true" in values else "false"
                values[{"true": "false", "false": "true"}[known]] = 1.0 - values[known]
            target = [max(0.0, values.get(key, 0.0)) for key in keys]
            total = sum(target)
            if 0 < total < math.inf:
                target = [value / total for value in target]
                return target, keys.index(label) if label in keys else target.index(max(target))
        if label in keys:
            return [1.0 if key == label else 0.0 for key in keys], keys.index(label)
        raise DecisionDataError("gold has no usable label or probabilities")

    def _clef_question(question) -> dict:
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            raise DecisionDataError("is not a valid question")
        kind, criteria = question["type"], question.get("criteria")
        if kind == "choice":
            if (
                isinstance(criteria, list)
                and criteria
                and all(isinstance(c, str) for c in criteria)
            ):
                criteria = dict.fromkeys(criteria)
            if not (isinstance(criteria, dict) and criteria):
                raise DecisionDataError("needs criteria naming its options")
            if len({str(key) for key in criteria}) != len(criteria):
                raise DecisionDataError("has repeated options")
        if kind == "score" and not (isinstance(criteria, list) and criteria):
            raise DecisionDataError("needs a list of criteria levels")
        if kind == "noul" and criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise DecisionDataError('criteria may only have "true" and "false"')
        clef_question = {"type": kind, "instructions": question.get("instructions")}
        if criteria is not None:
            clef_question["criteria"] = criteria
        return clef_question

    def _predicted(question: dict, answer: dict, probabilities: dict) -> dict:
        # The Decision API answer plus "answer": the option (choice), True / False (noul) or level number (score).
        kind = question["type"]
        best = max(probabilities, key = probabilities.__getitem__)
        return {
            **answer,
            "answer": int(best) if kind == "score" else best == "true" if kind == "noul" else best,
            "probabilities": answer.get("probabilities")
            or {key: round(float(value), 4) for key, value in probabilities.items()},
        }

    def _metrics(logits, items, temperatures) -> dict:
        import numpy as np
        import torch

        conf, correct, loss, records = [], [], [], {}
        for z, item, temperature in zip(logits, items, temperatures):
            log_p = torch.log_softmax(z / temperature, -1)
            conf.append(float(log_p.exp().max()))
            correct.append(float(int(log_p.argmax()) == item["label"]))
            loss.append(float(-(torch.tensor(item["target"]) * log_p).sum()))
            row = item.get("row", ("item", len(correct)))
            records[row] = records.get(row, True) and bool(correct[-1])
        return {
            "accuracy": float(np.mean(correct)),
            "ece": _laya().common.ece_score(np.array(conf), np.array(correct)),
            "loss": float(np.mean(loss)),
            # Every question of a row right, the record-level precision Cloudflare rewards.
            "record_accuracy": float(np.mean(list(records.values()))),
        }

    def _fit_temperature(
        logits,
        items,
        line_search = None,
    ) -> float:
        import torch

        options = max(len(z) for z in logits)
        z = torch.full((len(logits), options), -1e4)
        target = torch.zeros((len(logits), options))
        for i, (row, item) in enumerate(zip(logits, items)):
            z[i, : len(row)] = row
            target[i, : len(row)] = torch.tensor(item["target"])
        log_t = torch.zeros(1, requires_grad = True)
        # Clef's hard-label fit passes "strong_wolfe": plain LBFGS can overshoot on peaked targets.
        optimizer = torch.optim.LBFGS([log_t], lr = 0.1, max_iter = 100, line_search_fn = line_search)

        def closure():
            optimizer.zero_grad()
            loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(log_t.exp().item())

    def _fit_temperatures(logits, items, indices, fallback: list) -> tuple:
        clamp = _laya().common.clamp_temperature
        temperature, fitted = list(fallback), set()
        for qtype in range(3):
            chosen = [i for i in indices if items[i]["qtype"] == qtype]
            if len(chosen) >= MIN_CALIBRATION_ITEMS:
                temperature[qtype] = clamp(
                    _fit_temperature([logits[i] for i in chosen], [items[i] for i in chosen])
                )
                fitted.add(qtype)
        return temperature, fitted

    def _served_temperatures(config: dict, logits, items) -> list:
        common = _laya().common
        per_type = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
        buckets = {
            key: common.clamp_temperature(value)
            for key, value in (config.get("temperature_by_options") or {}).items()
        }
        # A Clef temperature not yet folded into the head (_save_clef folds it on save).
        head = config.get("head_temperature", 1.0)
        return [
            head * buckets.get(common.temp_bucket(item["qtype"], len(z)), per_type[item["qtype"]])
            for z, item in zip(logits, items)
        ]

    HEAD_TEMPERATURE_RANGE = (0.05, 20.0)

    def _calibrate_clef(config: dict, logits, items) -> dict:
        # Calibrated against being right (the gold label), not the soft gold distribution: Clef's
        # confidence is read as the chance the answer is correct, and soft gold targets left a tuned
        # model underconfident (confidence 0.61 at accuracy 0.78 on typed-decisions).
        common = _laya().common
        hard = [
            {**item, "target": [float(j == item["label"]) for j in range(len(z))]}
            for z, item in zip(logits, items)
        ]

        def fit(indices) -> tuple:
            chosen = list(indices)
            head = _fit_temperature(
                [logits[i] for i in chosen], [hard[i] for i in chosen], line_search = "strong_wolfe"
            )
            head = min(max(head, HEAD_TEMPERATURE_RANGE[0]), HEAD_TEMPERATURE_RANGE[1])
            scaled = [z / head for z in logits]
            relative, fitted = _fit_temperatures(scaled, hard, chosen, [1.0] * 3)
            return head, relative, fitted

        everything = range(len(items))
        if len(items) < MIN_CALIBRATION_ITEMS:
            return {
                **_metrics(logits, items, _served_temperatures(config, logits, items)),
                "fitted_types": [],
            }
        head, relative, fitted = fit(everything)
        half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
        per_item = [head * common.clamp_temperature(relative[item["qtype"]]) for item in items]
        for side in (0, 1) if len(half) > 1 else ():
            side_head, side_relative, _ = fit(
                i for i in everything if half[items[i]["row"]] != side
            )
            for i in everything:
                if half[items[i]["row"]] == side:
                    per_item[i] = side_head * common.clamp_temperature(
                        side_relative[items[i]["qtype"]]
                    )
        config["head_temperature"] = head
        config["temperature"] = relative
        config.pop("temperature_by_options", None)
        return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}

    def _served_lengths(
        config: dict,
        positions: int,
        max_seq_length = None,
    ) -> tuple:
        """(max_len, head_max_len) a Laya checkpoint serves with once loaded."""
        wanted = max_seq_length or max(int(config.get("max_len", 512)), TRAIN_MAX_LEN)
        max_len = min(int(positions), int(wanted))
        return max_len, min(
            max_len // 2, max(int(config.get("head_max_len", 192)), TRAIN_HEAD_MAX_LEN)
        )

    def _decision_clef_items(
        rows, options: Callable, encode: Callable, validate, report, skip
    ) -> list:
        # One item per row. `options(question)` names a question's options in the head's order and
        # `encode(state, questions)` returns the backend's tokenized fields, raising when they do not fit.
        items = []
        for index, row in enumerate(rows):
            row = row if isinstance(row, dict) else {}
            state = _parsed(row.get("state"))
            questions = _parsed(row.get("questions"))
            gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
            if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
                report["total"] += 1
                skip(index, "needs state, questions and gold")
                continue
            kept, targets, labels = {}, [], []
            for name, question in questions.items():
                report["total"] += 1
                if name not in gold:
                    skip(index, "has no gold", name)
                    continue
                try:
                    if validate is not None:
                        validate(name, question)
                    clef_question = _clef_question(question)
                    keys = options(clef_question)
                    target, label = _target_for(clef_question["type"], keys, _parsed(gold[name]))
                except (TypeError, ValueError, KeyError) as exc:
                    skip(index, str(exc) or "is not a valid question", name)
                    continue
                kept[str(name)], targets, labels = (
                    clef_question,
                    targets + [target],
                    labels + [label],
                )
            if not kept:
                continue
            try:
                encoded = encode(state, kept)
            except (TypeError, ValueError) as exc:
                for name in kept:
                    skip(index, f"does not fit: {exc}", name)
                continue
            items.append(
                {
                    **encoded,
                    "targets": targets,
                    "labels": labels,
                    "row": index,
                    "source": {"state": state, "questions": kept},
                }
            )
        return items

    def _decision_dataset(
        rows,
        tokenizer,
        config: dict,
        validate: Optional[Callable[[str, dict], None]] = None,
        clef: Optional[Callable] = None,
    ) -> tuple:
        # `clef` builds a Clef model's items: the backend's (rows, tokenizer, max_len, validate, report, skip).
        max_len = int(config.get("max_len", 512))
        head_max_len = int(config.get("head_max_len", 192))
        items, report, skips = [], {"total": 0, "skipped": 0, "reason": None, "truncated": 0}, {}

        def skip(
            index,
            reason,
            name = None,
        ):
            report["skipped"] += 1
            where = f"row {index + 1}" if name is None else f'row {index + 1}: "{name}"'
            skips.setdefault(reason, [0, f"{where} {reason}"])[0] += 1

        if clef is not None:
            items = clef(rows, tokenizer, max_len, validate, report, skip)
            rows = ()
        else:
            common = _laya().common
        for index, row in enumerate(rows):
            row = row if isinstance(row, dict) else {}
            state = _parsed(row.get("state"))
            questions = _parsed(row.get("questions"))
            gold = _parsed(row["gold"] if row.get("gold") is not None else row.get("answers"))
            if state is None or not isinstance(questions, dict) or not isinstance(gold, dict):
                report["total"] += 1
                skip(index, "needs state, questions and gold")
                continue
            for name, question in questions.items():
                report["total"] += 1
                if name not in gold:
                    skip(index, "has no gold", name)
                    continue
                try:
                    if validate is not None:
                        validate(name, question)
                    internal = _internal(question)
                    target, label = _target(internal, _parsed(gold[name]))
                    ids, markers = common.build_sequence(
                        tokenizer, state, internal, max_len, head_max_len
                    )
                except (TypeError, ValueError) as exc:
                    skip(index, str(exc), name)
                    continue
                if len(markers) != len(target):
                    skip(index, f"options exceed the {max_len}-token context", name)
                    continue
                items.append(
                    {
                        "input_ids": ids,
                        "markers": markers,
                        "qtype": common.QTYPES[internal["t"]],
                        "target": target,
                        "label": label,
                        "row": index,
                    }
                )
        if skips:
            # The most common reason, shown with the first decision it applied to.
            count, example = max(skips.values(), key = lambda skipped: skipped[0])
            report["reason"] = (
                example if count == 1 else f"{example} (and {count - 1:,} more like it)"
            )
        # Over max_seq_length the end of the state is cut, never questions or options.
        report["truncated"] = sum(len(item["input_ids"]) >= max_len for item in items)
        if report["truncated"]:
            print(
                f"Unsloth: {report['truncated']:,} of {len(items):,} training inputs are longer than "
                f"max_seq_length = {max_len}, so the end of their state is cut. Raise "
                "max_seq_length to train on all of it."
            )
        return items, report

    def _decision_holdout(
        items: list,
        seed: int = 3407,
        fraction: float = 0.1,
        max_items: int = HOLDOUT_MAX,
    ):
        # Counted in decisions: a Clef item holds every question of its row.
        sizes = Counter()
        for item in items:
            sizes[item["row"]] += len(item.get("labels", (None,)))
        target = min(max_items, int(sum(sizes.values()) * fraction))
        rows = sorted(sizes)
        random.Random(seed).shuffle(rows)
        held, count = set(), 0
        # Whole rows that still fit under the target; the last row always stays in training.
        for row in rows[:-1]:
            if count + sizes[row] <= target:
                held.add(row)
                count += sizes[row]
        return (
            [item for item in items if item["row"] not in held],
            [item for item in items if item["row"] in held],
        )

    def _decision_evaluation(config: dict, logits, items: list) -> dict:
        return _metrics(logits, items, _served_temperatures(config, logits, items))

    def _decision_calibration(
        config: dict,
        logits,
        items: list,
        clef: bool = False,
    ) -> dict:
        common = _laya().common
        fallback = [common.clamp_temperature(t) for t in config.get("temperature", [1.0] * 3)]
        if clef:
            return _calibrate_clef(config, logits, items)
        everything = range(len(items))
        temperature, fitted = _fit_temperatures(logits, items, everything, fallback)
        # Reported numbers score each half of the rows with temperatures fitted on the other half.
        half = {row: i % 2 for i, row in enumerate(sorted({item["row"] for item in items}))}
        per_item = [1.0] * len(items)
        for side in (0, 1):
            other = [i for i in everything if half[items[i]["row"]] != side]
            side_temperature, _ = _fit_temperatures(logits, items, other, fallback)
            for i in everything:
                if half[items[i]["row"]] == side:
                    per_item[i] = side_temperature[items[i]["qtype"]]
        config["temperature"] = temperature
        buckets = {
            key: value
            for key, value in (config.pop("temperature_by_options", None) or {}).items()
            if common.QTYPES.get(key.split(":")[0]) not in fitted
        }
        if buckets:
            config["temperature_by_options"] = buckets
        return {**_metrics(logits, items, per_item), "fitted_types": sorted(fitted)}

    # PEFT options the MLX adapters have no counterpart for, with the value that leaves each one off.
    _DECISION_UNSUPPORTED_LORA = {
        "bias": "none",
        "layers_to_transform": None,
        "layers_pattern": None,
        "use_dora": False,
        "modules_to_save": None,
        "init_lora_weights": True,
    }

    # Training arguments that change what is trained and that the MLX trainer does not implement.
    _DECISION_UNSUPPORTED_ARGUMENTS = (
        "load_best_model_at_end",
        "dataloader_drop_last",
        "optim_args",
        "eval_delay",
        "auto_find_batch_size",
    )

    # The MLX trainer for language models takes these; this trainer has no counterpart for them.
    _DECISION_UNUSED = {"neftune_noise_alpha", "push_to_hub"}

    # Set by transformers itself or about where logs go.
    _DECISION_QUIET_ARGUMENTS = ("do_eval", "do_train", "logging_dir", "run_name", "label_names")

    # Trainer options of the torch DecisionTrainer that the MLX trainer does not implement.
    _DECISION_UNSUPPORTED_OBJECTIVES = (
        "brier_weight",
        "ordinal_weight",
        "kl_weight",
        "permute_fields",
    )

    @functools.lru_cache(maxsize = None)
    def _decision_zoo():
        try:
            from unsloth_zoo.mlx import decision
            from unsloth_zoo.mlx.trainer import MLXDecisionTrainer
            decision.load_language_model_as_clef
        except (ImportError, AttributeError) as error:
            raise ImportError(
                "Unsloth: training decision models on MLX needs a newer unsloth-zoo. "
                "Upgrade with `pip install -U unsloth-zoo`."
            ) from error
        return types.SimpleNamespace(MLXDecisionTrainer = MLXDecisionTrainer, **vars(decision))

    def _is_clef(model) -> bool:
        return getattr(model, "is_clef", False)

    def _decision_annotate(model, **attributes):
        # Plain attributes: an mlx module would otherwise hold them as part of its state.
        for name, value in attributes.items():
            object.__setattr__(model, name, value)
        model.save_pretrained_merged = types.MethodType(_decision_save_merged, model)
        model.push_to_hub_merged = types.MethodType(_decision_push_merged, model)
        if _is_clef(model):
            model.save_pretrained = types.MethodType(_clef_save, model)
            model.push_to_hub = types.MethodType(_clef_push, model)
        model.save_pretrained_gguf = types.MethodType(_decision_save_gguf, model)
        model.push_to_hub_gguf = types.MethodType(_decision_gguf().push_to_hub_gguf, model)
        return model

    def _clef_save(
        self,
        save_directory,
        tokenizer = None,
        **kwargs,
    ) -> None:
        # Adapters plus the head, as Unsloth's other models; a full finetune has none, so it saves merged.
        if not getattr(self, "_unsloth_lora", False):
            return self.save_pretrained_merged(save_directory, tokenizer)
        config = {**self.decision_config, "fine_tuned": True}
        _decision_zoo().save_clef_adapter(
            self._unsloth_pipeline,
            save_directory,
            self._unsloth_source,
            config["base_model"],
            config.get("base_revision"),
            config,
        )

    def _clef_push(
        self,
        repo_id,
        tokenizer = None,
        token = None,
        private = None,
        **kwargs,
    ) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        repo_id = api.create_repo(repo_id, private = private, exist_ok = True).repo_id
        with tempfile.TemporaryDirectory() as folder:
            self.save_pretrained(folder, tokenizer)
            api.upload_folder(folder_path = folder, repo_id = repo_id)
        print(f"Unsloth: Saved the decision model to https://huggingface.co/{repo_id}")

    @functools.lru_cache(maxsize = None)
    def _decision_gguf():
        # By path: the exporter runs llama.cpp's converter in a subprocess, and unsloth.models needs torch.
        spec = importlib.util.spec_from_file_location(
            "unsloth._decision_gguf", Path(__file__).parent / "models" / "decision_gguf.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _decision_save_gguf(
        self,
        save_directory,
        tokenizer = None,
        quantization_method = "q8_0",
        source_folder = None,
        print_output = False,
        **kwargs,
    ) -> dict:
        """GGUF for llama.cpp's decision server in <save_directory>/gguf, from a temporary merged save."""
        gguf = _decision_gguf()
        # Fail before the merge when llama.cpp cannot convert.
        gguf._converter_dir(print_output)
        output = Path(save_directory)
        output.mkdir(parents = True, exist_ok = True)
        gguf._remove_abandoned_temp(output)
        prefix = gguf._temp_prefix(".unsloth-merged-")
        with (
            gguf._exit_on_sigterm(),
            tempfile.TemporaryDirectory(prefix = prefix, dir = output) as merged,
        ):
            self.save_pretrained_merged(merged, tokenizer)
            return gguf.export_decision_gguf(
                merged,
                quantization_method,
                output_dir = output / gguf._contract().EXPORT_DIR,
                source_folder = source_folder,
                print_output = print_output,
            )

    def _decision_pad_token_id(tokenizer) -> int:
        return getattr(tokenizer, "tokenizer", tokenizer).pad_token_id

    def _clef_items(pipeline, rows, tokenizer, max_len, validate, report, skip) -> list:
        zoo = _decision_zoo()
        return _decision_clef_items(
            rows,
            functools.partial(zoo.clef_option_keys, pipeline),
            lambda state, questions: zoo.clef_training_item(pipeline, state, questions, max_len),
            validate,
            report,
            skip,
        )

    def _decision_logits(model, tokenizer, items: list) -> tuple:
        import torch

        # Torch rows, so both backends score and calibrate with the same code.
        if not _is_clef(model):
            rows = _decision_zoo().decision_logits(model, items, _decision_pad_token_id(tokenizer))
            return [torch.from_numpy(row) for row in rows], items
        # One Laya-shaped item per question for the shared metrics.
        logits, questions = [], []
        for item, rows in zip(items, _decision_zoo().clef_logits(model, items)):
            kinds = [question["type"] for question in item["source"]["questions"].values()]
            for row, target, label, kind in zip(rows, item["targets"], item["labels"], kinds):
                logits.append(torch.from_numpy(row))
                questions.append(
                    {
                        "target": target,
                        "label": label,
                        "qtype": QUESTION_TYPES.index(kind),
                        "row": item["row"],
                    }
                )
        return logits, questions

    def _decision_save_merged(
        self,
        save_directory,
        tokenizer = None,
        save_method = "merged_16bit",
        **kwargs,
    ) -> None:
        if save_method != "merged_16bit":
            raise NotImplementedError(
                f"Unsloth: decision models are saved merged in 16-bit, not as {save_method!r}."
            )
        # The tokenizer is the checkpoint's own, which the savers copy from the source.
        config = {**self.decision_config, "fine_tuned": True}
        # The Clef saver writes the weights that train, so a frozen backbone is saved as trained before it was frozen.
        frozen = getattr(self, "_unsloth_frozen_backbone", ())
        FastDecisionModel.unfreeze_backbone(self)
        try:
            if _is_clef(self):
                _decision_zoo().save_clef_model(
                    self._unsloth_pipeline, save_directory, self._unsloth_source, config
                )
            else:
                _decision_zoo().save_decision_model(
                    self, save_directory, self._unsloth_source, config
                )
        finally:
            if frozen:
                FastDecisionModel.freeze_backbone(self)

    def _decision_push_merged(
        self,
        repo_id,
        tokenizer = None,
        save_method = "merged_16bit",
        token = None,
        private = None,
        **kwargs,
    ) -> None:
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        repo_id = api.create_repo(repo_id, private = private, exist_ok = True).repo_id
        with tempfile.TemporaryDirectory() as folder:
            self.save_pretrained_merged(folder, tokenizer, save_method)
            api.upload_folder(folder_path = folder, repo_id = repo_id)
        print(f"Unsloth: Saved the decision model to https://huggingface.co/{repo_id}")

    _CLEF_ATTRIBUTES = (
        "decision_config",
        "is_clef",
        "_unsloth_pipeline",
        "_unsloth_source",
        "_unsloth_full_finetuning",
        "_saved_temp_tokenizer",
    )

    def _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing):
        zoo = _decision_zoo()
        # A checkpoint saved as adapters comes with them, and they go on training.
        adapters = hasattr(pipeline, "base_folder")
        if full_finetuning or adapters:
            network = zoo.clef_training_network(
                pipeline,
                full_finetuning = bool(full_finetuning),
                gradient_checkpointing = gradient_checkpointing,
            )
        else:
            # Until get_peft_model adds adapters, only the joint head trains.
            pipeline.model.freeze()
            pipeline.head.unfreeze()
            network = zoo.ClefNetwork(pipeline, gradient_checkpointing)
            network.train()
        _decision_annotate(
            network,
            decision_config = config,
            is_clef = True,
            _unsloth_pipeline = pipeline,
            _unsloth_source = getattr(pipeline, "base_folder", folder),
            _unsloth_full_finetuning = bool(full_finetuning),
            _saved_temp_tokenizer = pipeline.tokenizer,
            _unsloth_lora = adapters,
        )
        return network, pipeline.tokenizer

    def _load_clef(
        folder, max_seq_length, load_in_4bit, full_finetuning, token, gradient_checkpointing, name
    ):
        # A 4-bit decoder trains through LoRA adapters only.
        pipeline = _decision_zoo().load_decision_model(
            folder, token = token, load_in_4bit = bool(load_in_4bit) and not full_finetuning
        )
        max_len = int(max_seq_length or CLEF_MAX_LEN)
        config = {"layout": "clef", "max_len": max_len, "temperature": [1.0] * 3}
        saved = folder / "unsloth_decision_config.json"
        if saved.is_file():
            config.update(json.loads(saved.read_text(encoding = "utf-8")), max_len = max_len)
            # The parent run's training record does not describe the next fine-tune, as for Laya.
            config.pop("training", None)
        if hasattr(pipeline, "base_folder"):
            # Adapters stay over the base they were trained on.
            adapter = json.loads((folder / _ADAPTER_CONFIG).read_text(encoding = "utf-8"))
            config.setdefault("base_model", adapter["base_model_name_or_path"])
            if adapter.get("revision"):
                config.setdefault("base_revision", adapter["revision"])
        else:
            # A merged checkpoint is its own base, whatever model it once started from.
            config.pop("base_revision", None)
            config.update(name)
        return _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing)

    def _load_lm_as_clef(
        model_name,
        max_seq_length,
        load_in_4bit,
        full_finetuning,
        token,
        revision,
        local_files_only,
        gradient_checkpointing,
        random_state,
        decision_head = "clef",
        head_width = None,
        head_config = None,
        **kwargs,
    ):
        # A plain language model plus a new joint schema head, as unsloth/models/decision_from_lm.py builds one.
        if decision_head != "clef":
            raise ValueError(
                f"Unsloth: decision_head must be one of ('clef',), not {decision_head!r}."
            )
        if kwargs:
            raise NotImplementedError(
                f"Unsloth: decision models on MLX do not support {', '.join(sorted(kwargs))}."
            )
        folder = Path(str(model_name)).expanduser()
        if not folder.is_dir():
            from huggingface_hub import snapshot_download
            folder = Path(
                snapshot_download(
                    str(model_name),
                    token = token,
                    revision = revision,
                    local_files_only = local_files_only,
                )
            )
        if not (folder / "config.json").is_file():
            raise NotImplementedError(
                f"Unsloth: {folder} holds LoRA adapters over a base model, which MLX does not load. "
                "Merge them into the base model first."
            )
        load_in_4bit = bool(load_in_4bit) and not full_finetuning
        pipeline = _decision_zoo().load_language_model_as_clef(
            folder,
            head_width = head_width,
            head_config = head_config,
            seed = random_state,
            token = token,
            load_in_4bit = load_in_4bit,
        )
        config = {
            "layout": "clef",
            "max_len": int(max_seq_length or CLEF_MAX_LEN),
            "temperature": [1.0] * 3,
            "base_model": str(model_name),
            **({"base_revision": revision} if revision else {}),
            "load_in_4bit": load_in_4bit,
        }
        return _clef_network(pipeline, folder, config, full_finetuning, gradient_checkpointing)

    class FastDecisionModel:
        @staticmethod
        def from_pretrained(
            model_name: str,
            subfolder: Optional[str] = None,
            max_seq_length: Optional[int] = None,
            dtype = None,
            load_in_4bit: bool = False,
            load_in_8bit: bool = False,
            full_finetuning: bool = False,
            token: Optional[str] = None,
            revision: Optional[str] = None,
            local_files_only: bool = False,
            use_gradient_checkpointing = "unsloth",
            random_state: int = 3407,
            **kwargs,
        ):
            if load_in_8bit:
                raise NotImplementedError("Unsloth: decision models do not support load_in_8bit.")
            checkpointing = bool(use_gradient_checkpointing)
            if kwargs.get("decision_head") is None and _is_plain_lm(
                model_name, subfolder, token, revision, local_files_only
            ):
                kwargs["decision_head"] = "clef"
            lm = kwargs.get("decision_head") is not None
            if not lm:
                folder = _checkpoint_folder(
                    model_name, subfolder, token, revision, local_files_only
                )
            if (lm or is_clef_checkpoint(folder)) and dtype is not None:
                raise NotImplementedError(
                    f"Unsloth: Clef on MLX trains at its checkpoint's precision, not {dtype}."
                )
            if lm:
                if subfolder:
                    model_name = _lm_subfolder(
                        model_name, subfolder, token, revision, local_files_only
                    )
                return _load_lm_as_clef(
                    model_name,
                    max_seq_length,
                    load_in_4bit,
                    full_finetuning,
                    token,
                    revision,
                    local_files_only,
                    checkpointing,
                    random_state,
                    **kwargs,
                )
            if is_clef_checkpoint(folder):
                name = {
                    "base_model": str(model_name),
                    **({"base_revision": revision} if revision else {}),
                }
                return _load_clef(
                    folder,
                    max_seq_length,
                    load_in_4bit,
                    full_finetuning,
                    token,
                    checkpointing,
                    name,
                )
            if load_in_4bit:
                raise NotImplementedError(
                    "Unsloth: Laya decision models train in 16-bit, so load_in_4bit is not supported."
                )
            if not (full_finetuning or dtype is None or str(dtype).rsplit(".", 1)[-1] == "float16"):
                raise NotImplementedError(
                    f"Unsloth: decision models on MLX keep the frozen encoder in float16, not {dtype}."
                )
            from transformers import AutoTokenizer

            config = json.loads((folder / "rl_agent_config.json").read_text(encoding = "utf-8"))
            # The base model's training record does not describe the fine-tune.
            config.pop("training", None)
            encoder = json.loads((folder / "encoder" / "config.json").read_text(encoding = "utf-8"))
            config["max_len"], config["head_max_len"] = _served_lengths(
                config, int(encoder.get("max_position_embeddings", TRAIN_MAX_LEN)), max_seq_length
            )
            tokenizer = AutoTokenizer.from_pretrained(str(folder / "tokenizer"))
            model = _decision_zoo().load_trainable_decision_model(
                folder, full_finetuning, gradient_checkpointing = checkpointing
            )
            _decision_annotate(
                model,
                decision_config = config,
                _unsloth_source = folder,
                _unsloth_full_finetuning = bool(full_finetuning),
                _saved_temp_tokenizer = tokenizer,
            )
            return model, tokenizer

        @staticmethod
        def get_peft_model(
            model,
            r = 64,
            target_modules = "all-linear",
            lora_alpha = 64,
            lora_dropout = 0.0,
            bias = "none",
            layers_to_transform = None,
            layers_pattern = None,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            max_seq_length = None,
            use_rslora = False,
            use_dora = False,
            modules_to_save = None,
            init_lora_weights = True,
            loftq_config = {},
            **kwargs,
        ):
            if getattr(model, "_unsloth_full_finetuning", False):
                print("Unsloth: Full finetuning is enabled, so .get_peft_model has no effect")
                return model
            given = {**locals(), **kwargs}
            unsupported = [
                name for name, off in _DECISION_UNSUPPORTED_LORA.items() if given[name] != off
            ]
            unsupported += ["loftq_config"] if loftq_config else []
            unsupported += sorted(kwargs)
            if unsupported:
                raise NotImplementedError(
                    f"Unsloth: decision models on MLX do not support {', '.join(unsupported)}."
                )
            if _is_clef(model):
                if getattr(model, "_unsloth_lora", False):
                    raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")
                network = _decision_zoo().clef_training_network(
                    model._unsloth_pipeline,
                    r = r,
                    lora_alpha = lora_alpha,
                    lora_dropout = lora_dropout,
                    use_rslora = use_rslora,
                    target_modules = target_modules,
                    random_state = random_state,
                    gradient_checkpointing = bool(use_gradient_checkpointing),
                )
                attributes = {name: getattr(model, name) for name in _CLEF_ATTRIBUTES}
                return _decision_annotate(network, **attributes, _unsloth_lora = True)
            _decision_zoo().add_lora_adapters(
                model,
                r = r,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                use_rslora = use_rslora,
                target_modules = target_modules,
                random_state = random_state,
            )
            model.gradient_checkpointing = bool(use_gradient_checkpointing)
            return model

        @staticmethod
        def freeze_backbone(model):
            # Head-only warm-up for a fresh decision head; undo with unfreeze_backbone.
            from mlx.utils import tree_flatten

            frozen = [name for name, _ in tree_flatten(model.encoder.trainable_parameters())]
            model.encoder.freeze()
            object.__setattr__(model, "_unsloth_frozen_backbone", frozen)
            return model

        @staticmethod
        def unfreeze_backbone(model):
            names = set(getattr(model, "_unsloth_frozen_backbone", ()))
            for path, module in model.encoder.named_modules():
                keys = [key for key in module if f"{path}.{key}".lstrip(".") in names]
                if keys:
                    module.unfreeze(recurse = False, keys = keys)
            object.__setattr__(model, "_unsloth_frozen_backbone", ())
            return model

        @staticmethod
        def for_inference(model):
            model.eval()
            return model

        @staticmethod
        def for_training(model, use_gradient_checkpointing = True):
            model.train()
            if not _is_clef(model):
                model.gradient_checkpointing = bool(use_gradient_checkpointing)
            return model

        @staticmethod
        def build_dataset(
            rows,
            tokenizer,
            model,
            validate: Optional[Callable[[str, dict], None]] = None,
        ) -> tuple:
            clef = (
                functools.partial(_clef_items, model._unsloth_pipeline) if _is_clef(model) else None
            )
            return _decision_dataset(rows, tokenizer, model.decision_config, validate, clef)

        split_holdout = staticmethod(_decision_holdout)

        @staticmethod
        def predict(model, tokenizer, state, questions: dict) -> dict:
            """Answers one Decision API request: {name: answer} at the calibrated temperatures.
            Each answer is the Decision API's (choice / confidence, score / legend, or noul) plus
            "answer" (the option, True / False for noul, the level number for score) and
            "probabilities" over every option."""
            import torch

            from ._vendor.clef.joint_schema_model import systemone_answer

            if not isinstance(questions, dict) or not questions:
                raise DecisionDataError("questions must be a non-empty dict of name to question")
            state, config, zoo = _parsed(state), model.decision_config, _decision_zoo()
            if _is_clef(model):
                pipeline = model._unsloth_pipeline
                questions = {str(name): _clef_question(q) for name, q in questions.items()}
                # Read up to CLEF_SERVE_MAX_LEN tokens, like serving, even past the training cut.
                max_length = max(int(config.get("max_len", CLEF_MAX_LEN)), CLEF_SERVE_MAX_LEN)
                item = zoo.clef_training_item(pipeline, state, questions, max_length)
                logits = zoo.clef_logits(model, [item])[0]
                keys = [zoo.clef_option_keys(pipeline, q) for q in questions.values()]
                kinds = [QUESTION_TYPES.index(q["type"]) for q in questions.values()]
            else:
                common, max_len = _laya().common, int(config.get("max_len", 512))
                tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
                items, keys = [], []
                for name, question in questions.items():
                    internal = _internal(question)
                    ids, markers = common.build_sequence(
                        tokenizer, state, internal, max_len, int(config.get("head_max_len", 192))
                    )
                    keys.append(_option_keys(internal))
                    if len(markers) != len(keys[-1]):
                        raise DecisionDataError(
                            f'"{name}" has more options than fit in {max_len} tokens'
                        )
                    items.append(
                        {
                            "input_ids": ids,
                            "markers": markers,
                            "qtype": common.QTYPES[internal["t"]],
                            "target": [0.0] * len(markers),
                        }
                    )
                logits = zoo.decision_logits(model, items, tokenizer.pad_token_id)
                kinds = [item["qtype"] for item in items]
            logits = [torch.from_numpy(row) for row in logits]
            scales = _served_temperatures(config, logits, [{"qtype": kind} for kind in kinds])
            answers = {}
            for (name, question), row, scale, options in zip(
                questions.items(), logits, scales, keys
            ):
                probabilities = dict(zip(options, torch.softmax(row / scale, -1).tolist()))
                answers[name] = _predicted(
                    question, systemone_answer(question, probabilities), probabilities
                )
            return answers

        # batch_size is the torch call shape; MLX scores Clef a record at a time and Laya in its own batches.
        @staticmethod
        def evaluate(
            model,
            tokenizer,
            items: list,
            batch_size = None,
        ) -> dict:
            logits, items = _decision_logits(model, tokenizer, items)
            return _decision_evaluation(model.decision_config, logits, items)

        @staticmethod
        def calibrate(
            model,
            tokenizer,
            items: list,
            batch_size = None,
        ) -> dict:
            logits, items = _decision_logits(model, tokenizer, items)
            return _decision_calibration(model.decision_config, logits, items, _is_clef(model))

    def _decision_arguments(args):
        from unsloth_zoo.mlx.trainer import MLXTrainingConfig

        if args is None:
            # The torch trainer's default: three epochs, no evaluation schedule.
            from transformers import TrainingArguments
            args = TrainingArguments(output_dir = "tmp_trainer")
        given = args
        strategy = getattr(args, "eval_strategy", None)
        strategy = str(getattr(strategy, "value", strategy) or "").lower()
        checkpointing = None
        if isinstance(args, MLXTrainingConfig):
            args = copy.copy(args)
        else:
            # transformers.TrainingArguments, as the torch DecisionTrainer takes.
            checkpointing = bool(getattr(args, "gradient_checkpointing", False)) or None
            fields = {field.name for field in dataclasses.fields(MLXTrainingConfig)}
            values = _mlx_training_argument_values(args)
            args = MLXTrainingConfig(**{k: v for k, v in values.items() if k in fields})
            if dataclasses.is_dataclass(given):
                # Whatever else differs from the defaults has no MLX counterpart.
                default, quiet = (
                    type(given)(output_dir = given.output_dir),
                    {*fields, *_DECISION_QUIET_ARGUMENTS, *_DECISION_UNSUPPORTED_ARGUMENTS},
                )
                ignored = [
                    field.name
                    for field in dataclasses.fields(given)
                    if field.name not in (quiet | _MLX_ALLOWED_EXTRA_ARGUMENTS) - _DECISION_UNUSED
                    and getattr(given, field.name) != getattr(default, field.name)
                ]
                if ignored:
                    warnings.warn(
                        f"Unsloth: DecisionTrainer on MLX ignores {', '.join(sorted(ignored))}."
                    )
        if strategy == "epoch":
            args.eval_steps = 0
        # transformers callbacks read arguments the MLX config has no field for, such as eval_strategy.
        for name, value in vars(given).items():
            if not (name.startswith("_") or hasattr(args, name)):
                setattr(args, name, value)
        logging = getattr(given, "logging_strategy", None)
        logging = str(getattr(logging, "value", logging) or "").lower()
        if logging == "no":
            args.logging_steps = 0
        elif logging == "epoch":
            warnings.warn(
                "Unsloth: DecisionTrainer on MLX logs every logging_steps steps, not once per epoch."
            )
        unsupported = [
            name for name in _DECISION_UNSUPPORTED_ARGUMENTS if getattr(given, name, None)
        ]
        if unsupported:
            raise NotImplementedError(
                f"Unsloth: DecisionTrainer on MLX does not support {', '.join(unsupported)}."
            )
        if getattr(args, "save_steps", 0):
            # transformers saves every 500 steps by default, so this is a warning and not a refusal.
            warnings.warn(
                "Unsloth: DecisionTrainer on MLX saves no checkpoints during training; "
                "call model.save_pretrained_merged when it finishes."
            )
            args.save_steps = 0
        return args, strategy, checkpointing

    def _decision_smoothed(items, smoothing):
        # Toward uniform over each decision's own options, as the torch trainer smooths its targets.
        def smooth(target):
            return [(1.0 - smoothing) * value + smoothing / len(target) for value in target]

        return [
            {**item, "targets": [smooth(target) for target in item["targets"]]}
            if "targets" in item
            else {**item, "target": smooth(item["target"])}
            for item in items
        ]

    class DecisionTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            train_dataset = None,
            eval_dataset = None,
            *,
            head_learning_rate: Optional[float] = None,
            tokenizer = None,
            callbacks = None,
            processing_class = None,
            **kwargs,
        ):
            label_smoothing = kwargs.pop("label_smoothing", None)
            # The torch trainer's other objectives may be passed at the value that leaves them off.
            kwargs = {
                name: value
                for name, value in kwargs.items()
                if value or name not in _DECISION_UNSUPPORTED_OBJECTIVES
            }
            if kwargs:
                raise NotImplementedError(
                    f"Unsloth: DecisionTrainer on MLX does not support {', '.join(sorted(kwargs))}."
                )
            args, eval_strategy, gradient_checkpointing = _decision_arguments(args)
            if label_smoothing is None:
                label_smoothing = getattr(args, "label_smoothing_factor", 0.0)
            # The torch trainer's loss smooths its targets when it evaluates too.
            self._smoothing = float(label_smoothing or 0.0)
            train_dataset, eval_dataset = map(self._smoothed, (train_dataset, eval_dataset))
            if gradient_checkpointing and not _is_clef(model):
                model.gradient_checkpointing = True
            processing_class = tokenizer if processing_class is None else processing_class
            if processing_class is None:
                processing_class = model._saved_temp_tokenizer
            self._eval_dataset = eval_dataset
            self._trainer = _decision_zoo().MLXDecisionTrainer(
                model,
                args,
                train_dataset,
                # Kept for evaluate(); the zoo trainer evaluates on a schedule whenever it holds one.
                None if eval_strategy == "no" else eval_dataset,
                pad_token_id = _decision_pad_token_id(processing_class),
                head_learning_rate = head_learning_rate,
                callbacks = callbacks,
                processing_class = processing_class,
            )

        def __getattr__(self, name):
            if name in ("_trainer", "_eval_dataset", "_smoothing"):
                raise AttributeError(name)
            return getattr(self._trainer, name)

        def train(self):
            return self._trainer.train()

        def _smoothed(self, items):
            return (
                _decision_smoothed(items, self._smoothing) if self._smoothing and items else items
            )

        def evaluate(self, eval_dataset = None):
            trainer = self._trainer
            eval_dataset = (
                self._eval_dataset if eval_dataset is None else self._smoothed(eval_dataset)
            )
            scheduled, trainer.eval_dataset = trainer.eval_dataset, eval_dataset
            try:
                return trainer.evaluate()
            finally:
                trainer.eval_dataset = scheduled

    def is_bfloat16_supported():
        try:
            import mlx.core as mx
            name = mx.device_info().get("device_name", "") or ""
            return not name.startswith(("Apple M1", "Apple M2"))
        except Exception:
            return True

    is_bf16_supported = is_bfloat16_supported

    def get_gpu_memory_stats():
        """Return MLX device stats, peak memory, and total memory in GiB."""
        import mlx.core as mx

        info = mx.device_info()
        total = info.get("memory_size") or info.get("max_recommended_working_set_size") or 0
        get_peak_memory = getattr(mx, "get_peak_memory", None)
        if get_peak_memory is None and hasattr(mx, "metal"):
            get_peak_memory = getattr(mx.metal, "get_peak_memory", None)
        peak = get_peak_memory() if callable(get_peak_memory) else 0
        stats = _UnslothDeviceStats(info.get("device_name", "Apple GPU"), total)
        max_memory = _bytes_to_gb(total) or 1.0
        return stats, _bytes_to_gb(peak), max_memory

    def clear_gpu_memory():
        """Clear MLX's cached GPU memory for compatibility cleanup helpers."""
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is None and hasattr(mx, "metal"):
            clear_cache = getattr(mx.metal, "clear_cache", None)
        if callable(clear_cache):
            # MLX pins buffers a live command buffer reads, but not a dropped output array.
            # Generation runs on its own streams, which a no-argument mx.synchronize()
            # would not wait on. Best effort: this helper is torch.cuda.empty_cache() on
            # MLX, called from finally arms on any thread, and synchronizing a stream
            # bound on another thread raises (mlx 0.31.2 made encoders thread local).
            _synchronize = getattr(mx, "synchronize", None)
            if callable(_synchronize):
                _drained = []
                for _module in (
                    "mlx_lm.generate",
                    "mlx_vlm.generate",
                    "mlx_vlm.generate.dispatch",
                    "mlx_vlm.generate.ar",
                    # Speculative decoding owns a second stream wired_limit never sees.
                    "mlx_vlm.speculative.common",
                ):
                    try:
                        _stream = getattr(sys.modules.get(_module), "generation_stream", None)
                        # 0.6.x aliases one object across every mlx_vlm.generate name.
                        if _stream is None or any(_stream is _seen for _seen in _drained):
                            continue
                        _drained.append(_stream)
                        _synchronize(_stream)
                    except Exception:
                        continue
                try:
                    _synchronize()
                except Exception:
                    pass
            clear_cache()

    def _patch_mlx_torch_cuda_compat_api():
        """Expose CUDA-shaped torch helpers for compatibility callers on MLX."""
        try:
            import torch
        except Exception:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and not getattr(cuda, "_unsloth_mlx_cuda_compat_api", False):

            def get_device_properties(device = None):
                """Return MLX device stats through torch.cuda's compatibility API."""
                return get_gpu_memory_stats()[0]

            def get_device_name(device = None):
                """Return the MLX device name through torch.cuda's compatibility API."""
                return get_device_properties(device).name

            def max_memory_reserved(device = None):
                """Return MLX peak memory in bytes for torch.cuda compatibility API."""
                return int(get_gpu_memory_stats()[1] * 1024 * 1024 * 1024)

            def empty_cache():
                """Clear MLX cache through torch.cuda.empty_cache()."""
                clear_gpu_memory()

            def _mlx_active_memory_bytes():
                """Current active MLX memory in bytes (not the peak high-water mark)."""
                import mlx.core as mx

                get_active = getattr(mx, "get_active_memory", None)
                if get_active is None and hasattr(mx, "metal"):
                    get_active = getattr(mx.metal, "get_active_memory", None)
                return int(get_active()) if callable(get_active) else 0

            def memory_current(device = None):
                """Return CURRENT MLX memory in bytes. torch.cuda.memory_reserved /
                memory_allocated report live usage, not the peak (that is max_*)."""
                return _mlx_active_memory_bytes()

            def mem_get_info(device = None):
                """Return (free, total) bytes for torch.cuda compatibility API.
                Free uses CURRENT active memory, not the peak high-water mark, so
                a capacity check stays accurate after a transient spike."""
                total = int(get_gpu_memory_stats()[2] * 1024 * 1024 * 1024)
                return (max(total - _mlx_active_memory_bytes(), 0), total)

            def reset_peak_memory_stats(device = None):
                """Reset MLX's peak-memory counter so a later max_memory_reserved /
                max_memory_allocated scopes to the run, not earlier model-load peaks."""
                import mlx.core as mx

                reset = getattr(mx, "reset_peak_memory", None)
                if reset is None and hasattr(mx, "metal"):
                    reset = getattr(mx.metal, "reset_peak_memory", None)
                if callable(reset):
                    reset()

            def synchronize(device = None):
                """Wait for queued MLX work when torch.cuda.synchronize() is called."""
                import mlx.core as mx

                sync = getattr(mx, "synchronize", None)
                if callable(sync):
                    sync()

            cuda.get_device_properties = get_device_properties
            cuda.get_device_name = get_device_name
            cuda.max_memory_reserved = max_memory_reserved
            cuda.max_memory_allocated = max_memory_reserved
            cuda.memory_reserved = memory_current
            cuda.memory_allocated = memory_current
            cuda.empty_cache = empty_cache
            cuda.mem_get_info = mem_get_info
            cuda.reset_peak_memory_stats = reset_peak_memory_stats
            cuda.synchronize = synchronize
            cuda.current_device = lambda: 0
            cuda.device_count = lambda: 1
            cuda.set_device = lambda device = None: None
            cuda.get_device_capability = lambda device = None: (0, 0)
            cuda.is_bf16_supported = lambda *args, **kwargs: is_bfloat16_supported()
            cuda._unsloth_mlx_cuda_compat_api = True

        tensor_to = getattr(torch.Tensor, "to", None)
        if tensor_to is not None and not getattr(tensor_to, "_unsloth_mlx_cuda_noop", False):

            def _coerce_mlx_dtype_to_torch(value):
                """Map MLX dtype objects to their torch dtype equivalents."""
                try:
                    import mlx.core as mx
                except Exception:
                    return value
                dtype_map = {
                    mx.bool_: torch.bool,
                    mx.int8: torch.int8,
                    mx.int16: torch.int16,
                    mx.int32: torch.int32,
                    mx.int64: torch.int64,
                    mx.uint8: torch.uint8,
                    mx.float16: torch.float16,
                    mx.float32: torch.float32,
                    mx.bfloat16: torch.bfloat16,
                }
                mapped = dtype_map.get(value, None)
                if mapped is not None:
                    return mapped
                dtype_name = str(value).rsplit(".", 1)[-1]
                name_map = {
                    "bool_": torch.bool,
                    "int8": torch.int8,
                    "int16": torch.int16,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "uint8": torch.uint8,
                    "float16": torch.float16,
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }
                return name_map.get(dtype_name, value)

            def mlx_tensor_to(self, *args, **kwargs):
                """Ignore CUDA device targets while preserving dtype conversions."""
                args = list(args)
                kwargs = dict(kwargs)
                removed_cuda_device = False
                if args and _is_mlx_cuda_device_target(args[0]):
                    args.pop(0)
                    removed_cuda_device = True
                if _is_mlx_cuda_device_target(kwargs.get("device", None)):
                    kwargs.pop("device", None)
                    removed_cuda_device = True
                if removed_cuda_device and not args:
                    cuda_only_kwargs = ("non_blocking", "copy", "memory_format")
                    if all(key in cuda_only_kwargs for key in kwargs):
                        return self
                if removed_cuda_device and not args and not kwargs:
                    return self
                if args:
                    args[0] = _coerce_mlx_dtype_to_torch(args[0])
                if "dtype" in kwargs:
                    kwargs["dtype"] = _coerce_mlx_dtype_to_torch(kwargs["dtype"])
                return tensor_to(self, *args, **kwargs)

            mlx_tensor_to._unsloth_mlx_cuda_noop = True
            mlx_tensor_to._unsloth_original_to = tensor_to
            torch.Tensor.to = mlx_tensor_to

        tensor_cuda = getattr(torch.Tensor, "cuda", None)
        if tensor_cuda is not None and not getattr(tensor_cuda, "_unsloth_mlx_cuda_noop", False):

            def mlx_tensor_cuda(self, *args, **kwargs):
                """Treat tensor.cuda() as a no-op on MLX."""
                return self

            mlx_tensor_cuda._unsloth_mlx_cuda_noop = True
            mlx_tensor_cuda._unsloth_original_cuda = tensor_cuda
            torch.Tensor.cuda = mlx_tensor_cuda

    _patch_mlx_torch_cuda_compat_api()

    _MLX_TRAINING_CONFIG_FIELDS = {_field.name for _field in _dataclasses.fields(MLXTrainingConfig)}
    _MLX_TRAINING_ARGUMENT_ALIASES = {
        "max_length": "max_seq_length",
    }
    _MLX_COMPAT_EXTRA_ARGUMENTS = frozenset(
        (
            "bf16",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataset_kwargs",
            "ddp_find_unused_parameters",
            "disable_tqdm",
            "eval_strategy",
            "evaluation_strategy",
            "fp16",
            "full_determinism",
            "gradient_checkpointing_kwargs",
            "hub_model_id",
            "hub_token",
            "log_level",
            "logging_strategy",
            "neftune_noise_alpha",
            "optim_args",
            "padding_free",
            "push_to_hub",
            "remove_unused_columns",
            "save_on_each_node",
            "save_safetensors",
            "save_strategy",
            "torch_compile",
        )
    )
    _MLX_IMPLEMENTED_EXTRA_ARGUMENTS = frozenset(
        (
            "image_size",
            "preserve_dataset_order",
            "warmup_ratio",
        )
    )
    _MLX_ALLOWED_EXTRA_ARGUMENTS = _MLX_COMPAT_EXTRA_ARGUMENTS | _MLX_IMPLEMENTED_EXTRA_ARGUMENTS
    _MLX_UNSUPPORTED_TASK_ARGUMENTS = frozenset(
        (
            "assistant_only_loss",
            "completion_only_loss",
        )
    )

    def _is_mlx_no_save_strategy(value):
        if hasattr(value, "value"):
            value = value.value
        strategy = str(value or "").strip().lower()
        strategy = strategy.rsplit(".", 1)[-1]
        return strategy in ("no", "none", "false")

    # Mirrors zoo's _normalize_mlx_optimizer_name; adamw_8bit is a real MLX optimizer, never collapse it.
    _MLX_ADAMW_OPTIMIZER_ALIASES = frozenset(
        (
            "paged_adamw_8bit",
            "adamw_bnb_8bit",
            "paged_adamw_32bit",
            "adamw_torch",
            "adamw_torch_fused",
            "paged_adamw",
            "adamw_32bit",
            "adamw_hf",
            "adamw_anyprecision",
            "adamw_apex_fused",
        )
    )

    def _normalize_mlx_training_value(key, value):
        if key == "eval_steps" and value is None:
            return 0
        if key == "num_train_epochs" and value is not None and not isinstance(value, bool):
            try:
                epochs = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if epochs.is_integer():
                    return int(epochs)
        if key == "lr_scheduler_type" and hasattr(value, "value"):
            return value.value
        if key != "optim":
            return value
        try:
            return _normalize_mlx_optimizer_name(value)
        except ValueError:
            # Older unsloth-zoo lacks the CUDA/TRL optimizer aliases, so map the common adamw_* names.
            opt = str(getattr(value, "value", value) or "adamw").strip().lower()
            opt = opt.rsplit(".", 1)[-1].replace("-", "_")
            if opt in _MLX_ADAMW_OPTIMIZER_ALIASES:
                return "adamw"
            raise

    def _mlx_training_argument_values(args):
        values = {}
        for field in _dataclasses.fields(MLXTrainingConfig):
            if hasattr(args, field.name):
                values[field.name] = _normalize_mlx_training_value(
                    field.name,
                    getattr(args, field.name),
                )
        for alias, target in _MLX_TRAINING_ARGUMENT_ALIASES.items():
            if target not in values and hasattr(args, alias):
                values[target if target in _MLX_ALLOWED_EXTRA_ARGUMENTS else alias] = getattr(
                    args, alias
                )
        for name in _MLX_ALLOWED_EXTRA_ARGUMENTS:
            if hasattr(args, name):
                values[name] = getattr(args, name)
        for name in _MLX_UNSUPPORTED_TASK_ARGUMENTS:
            if hasattr(args, name):
                value = getattr(args, name)
                if (
                    name == "completion_only_loss"
                    and value is not None
                    and name in _MLX_TRAINING_CONFIG_FIELDS
                ):
                    values[name] = value
                elif value is not None and value is not False:
                    values[name] = value
        if _is_mlx_no_save_strategy(values.get("save_strategy", None)):
            values["save_steps"] = 0
        return values

    def _split_mlx_trainer_kwargs(kwargs):
        trainer_kwargs = {}
        config_kwargs = {}
        ignored_kwargs = {}
        for key, value in kwargs.items():
            if key in _MLX_TRAINER_KWARGS:
                trainer_kwargs[key] = value
                continue
            target = _MLX_TRAINING_ARGUMENT_ALIASES.get(key, key)
            if target in _MLX_TRAINING_CONFIG_FIELDS or key in _MLX_ALLOWED_EXTRA_ARGUMENTS:
                config_kwargs[key] = value
            else:
                ignored_kwargs[key] = value
        return trainer_kwargs, config_kwargs, ignored_kwargs

    def _is_mlx_training_args_like(value):
        if isinstance(value, (MLXTrainingConfig, dict, str, os.PathLike)):
            return True
        return any(
            hasattr(value, name)
            for name in (
                "output_dir",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "max_steps",
                "learning_rate",
            )
        )

    def _should_use_trl_positional_schema(args):
        if len(args) < 2:
            return False
        if _is_mlx_training_args_like(args[1]):
            return True
        # TRL callers often pass explicit defaults: SFTTrainer(model, None, None, train_dataset, ...)
        return len(args) >= 3 and args[1] is None and (args[2] is None or callable(args[2]))

    def _assign_mlx_positional_kwarg(kwargs, name, value):
        if name in kwargs:
            raise TypeError(f"UnslothTrainer.__init__() got multiple values for argument {name!r}")
        kwargs[name] = value

    def _normalize_mlx_trainer_init_args(args, kwargs):
        kwargs = dict(kwargs)
        if len(args) == 0:
            return kwargs

        use_trl_schema = _should_use_trl_positional_schema(args)
        positional_names = (
            _TRL_SFT_TRAINER_POSITIONAL_KWARGS if use_trl_schema else _MLX_TRAINER_POSITIONAL_KWARGS
        )
        if len(args) > len(positional_names):
            raise TypeError(
                f"UnslothTrainer.__init__() takes at most "
                f"{len(positional_names)} positional arguments on MLX "
                f"({len(args)} given)"
            )
        for name, value in zip(positional_names, args):
            _assign_mlx_positional_kwarg(kwargs, name, value)
        return kwargs

    def _is_meaningful_mlx_extra_value(value):
        if value is None or value is False:
            return False
        if isinstance(value, (str, bytes)) and len(value) == 0:
            return False
        if isinstance(value, (dict, list, tuple, set, frozenset)) and len(value) == 0:
            return False
        return True

    def _warn_ignored_mlx_training_args(extra_kwargs):
        names = sorted(
            key
            for key, value in extra_kwargs.items()
            if (key in _MLX_COMPAT_EXTRA_ARGUMENTS and _is_meaningful_mlx_extra_value(value))
        )
        if not names:
            return
        _warnings.warn(
            "Unsloth MLX: accepting but not applying unsupported "
            "TrainingArguments kwargs: "
            f"{', '.join(names)}. These options are not implemented by "
            "MLXTrainer yet.",
            RuntimeWarning,
            stacklevel = 3,
        )

    def _is_meaningful_mlx_trainer_kwarg(key, value):
        if key == "optimizers" and value == (None, None):
            return False
        return _is_meaningful_mlx_extra_value(value)

    def _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs):
        names = sorted(
            key
            for key, value in ignored_kwargs.items()
            if _is_meaningful_mlx_trainer_kwarg(key, value)
        )
        if not names:
            return
        raise NotImplementedError(
            "Unsloth MLX: unsupported SFTTrainer kwargs cannot be ignored safely: "
            f"{', '.join(names)}. Remove these kwargs or use a supported MLX "
            "trainer configuration."
        )

    def _raise_unknown_mlx_training_args(extra_kwargs):
        names = sorted(key for key in extra_kwargs if key not in _MLX_ALLOWED_EXTRA_ARGUMENTS)
        if not names:
            return
        raise NotImplementedError(
            "Unsloth MLX: unsupported TrainingArguments/SFTConfig kwargs: "
            f"{', '.join(names)}. Remove these kwargs or use fields implemented "
            "by MLXTrainingConfig."
        )

    def _positive_mlx_context_length(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            length = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if length <= 0:
            return None
        return length

    def _positive_mlx_training_number(value):
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if number <= 0:
            return None
        return number

    def _set_mlx_cuda_style_context_length(args, length):
        args.max_seq_length = length
        args.max_length = length
        args._unsloth_mlx_max_length_value = length
        return args

    class UnslothTrainingArguments(MLXTrainingConfig):
        """MLX-compatible public training arguments for Unsloth notebooks."""

        def __init__(self, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
            elif len(args) == 1 and isinstance(args[0], (str, os.PathLike)):
                kwargs = {"output_dir": os.fspath(args[0]), **kwargs}
            elif args:
                raise TypeError(
                    "UnslothTrainingArguments on MLX accepts keyword arguments, "
                    "a dict, or a single positional output_dir."
                )

            max_length_value = kwargs.get("max_length", None)
            # Only the canonical max_seq_length marks context length explicit; TRL max_length stays a
            # compatibility alias that defers to the model's context length.
            max_seq_length_explicit = (
                _positive_mlx_context_length(kwargs.get("max_seq_length", None)) is not None
            )
            if "max_length" in kwargs and "max_seq_length" not in kwargs:
                kwargs["max_seq_length"] = kwargs["max_length"]
            elif (
                "max_length" in kwargs
                and _positive_mlx_context_length(kwargs.get("max_seq_length", None)) is not None
            ):
                max_length_value = kwargs["max_seq_length"]
            if "num_train_epochs" in kwargs and "max_steps" not in kwargs:
                kwargs["max_steps"] = -1

            dataset_order_explicit = "dataset_order" in kwargs or bool(
                kwargs.get("preserve_dataset_order", False)
            )
            append_eos_explicit = "append_eos" in kwargs
            grad_clip_explicit = any(
                name in kwargs for name in ("max_grad_norm", "max_grad_value", "max_grad_leaf_norm")
            )
            warmup_ratio = kwargs.get("warmup_ratio", None)
            warmup_steps_supplied = "warmup_steps" in kwargs
            warmup_steps_value = kwargs.get("warmup_steps", None)
            warmup_steps_explicit = False
            if warmup_steps_supplied:
                try:
                    warmup_steps_explicit = int(warmup_steps_value) > 0
                except (TypeError, ValueError):
                    warmup_steps_explicit = True
            filtered_kwargs = {}
            extra_kwargs = {}
            for key, value in kwargs.items():
                target = _MLX_TRAINING_ARGUMENT_ALIASES.get(key, key)
                if key != target and target in kwargs:
                    continue
                value = _normalize_mlx_training_value(target, value)
                if target in _MLX_UNSUPPORTED_TASK_ARGUMENTS:
                    if (
                        target == "completion_only_loss"
                        and value is not None
                        and target in _MLX_TRAINING_CONFIG_FIELDS
                    ):
                        filtered_kwargs[target] = value
                    elif _is_meaningful_mlx_extra_value(value):
                        extra_kwargs[key] = value
                    continue
                if target in _MLX_TRAINING_CONFIG_FIELDS:
                    filtered_kwargs[target] = value
                else:
                    extra_kwargs[target if target in _MLX_ALLOWED_EXTRA_ARGUMENTS else key] = value

            _raise_unknown_mlx_training_args(extra_kwargs)

            if _is_mlx_no_save_strategy(extra_kwargs.get("save_strategy", None)):
                filtered_kwargs["save_steps"] = 0

            if warmup_ratio is not None and not warmup_steps_explicit:
                import math as _math
                max_steps = filtered_kwargs.get(
                    "max_steps",
                    getattr(MLXTrainingConfig, "max_steps", 60),
                )
                try:
                    if int(max_steps) > 0:
                        filtered_kwargs["warmup_steps"] = max(
                            0,
                            _math.ceil(int(max_steps) * float(warmup_ratio)),
                        )
                except (TypeError, ValueError):
                    pass

            super().__init__(**filtered_kwargs)
            self._unsloth_mlx_dataset_order_explicit = dataset_order_explicit
            self._unsloth_mlx_append_eos_explicit = append_eos_explicit
            self._unsloth_mlx_max_seq_length_explicit = max_seq_length_explicit
            self._unsloth_mlx_max_length_value = max_length_value
            if "max_length" in kwargs:
                self.max_length = max_length_value
            self._unsloth_mlx_grad_clip_explicit = grad_clip_explicit
            self._unsloth_mlx_warmup_steps_explicit = warmup_steps_explicit
            self._unsloth_mlx_extra_args = extra_kwargs
            for key, value in extra_kwargs.items():
                setattr(self, key, value)
            _warn_ignored_mlx_training_args(extra_kwargs)

    def _resolve_mlx_cuda_style_max_seq_length(args, model = None):
        model_max_seq_length = _positive_mlx_context_length(
            getattr(model, "max_seq_length", None),
        )
        args_max_seq_length = _positive_mlx_context_length(
            getattr(args, "max_seq_length", None),
        )
        args_max_seq_length_explicit = getattr(
            args,
            "_unsloth_mlx_max_seq_length_explicit",
            None,
        )
        if args_max_seq_length_explicit is None:
            default_max_seq_length = getattr(MLXTrainingConfig, "max_seq_length", 2048)
            args_max_seq_length_explicit = (
                args_max_seq_length is not None and args_max_seq_length != default_max_seq_length
            )
        if not args_max_seq_length_explicit:
            args_max_seq_length = None

        if args_max_seq_length is None and model_max_seq_length is not None:
            args_max_seq_length = model_max_seq_length
        elif (
            args_max_seq_length is not None
            and model_max_seq_length is not None
            and args_max_seq_length > model_max_seq_length
        ):
            print(
                "Unsloth: You set `max_seq_length` as "
                f"{args_max_seq_length} but the maximum the model supports is "
                f"{model_max_seq_length}. We shall reduce it."
            )
            args_max_seq_length = model_max_seq_length

        if args_max_seq_length is not None:
            _set_mlx_cuda_style_context_length(args, args_max_seq_length)
            return args

        model_max_length = model_max_seq_length
        if model_max_length is None:
            model_max_length = _positive_mlx_context_length(
                getattr(model, "max_length", None),
            )
        if model_max_length is not None:
            _set_mlx_cuda_style_context_length(args, model_max_length)
            return args

        args_max_length = _positive_mlx_context_length(
            getattr(args, "max_length", None),
        )
        if args_max_length is None:
            args_max_length = _positive_mlx_context_length(
                getattr(args, "_unsloth_mlx_max_length_value", None),
            )
        if args_max_length is not None:
            _set_mlx_cuda_style_context_length(args, args_max_length)
            if model is not None:
                setattr(model, "max_seq_length", args_max_length)
            return args

        _set_mlx_cuda_style_context_length(args, 1024)
        return args

    def _apply_unsloth_trainer_mlx_defaults(
        args,
        model = None,
        max_seq_length_explicit = False,
    ):
        if (
            not getattr(args, "streaming", False)
            and not getattr(args, "preserve_dataset_order", False)
            and not getattr(args, "_unsloth_mlx_dataset_order_explicit", False)
        ):
            default_order = getattr(MLXTrainingConfig, "dataset_order", "default")
            if getattr(args, "dataset_order", default_order) in (None, default_order):
                args.dataset_order = "torch_randperm"

        if isinstance(args, UnslothTrainingArguments) and not getattr(
            args, "_unsloth_mlx_append_eos_explicit", False
        ):
            args.append_eos = False

        if isinstance(args, UnslothTrainingArguments) and not getattr(
            args, "_unsloth_mlx_grad_clip_explicit", False
        ):
            max_grad_norm = _positive_mlx_training_number(
                getattr(args, "max_grad_norm", None),
            )
            max_grad_value = _positive_mlx_training_number(
                getattr(args, "max_grad_value", None),
            )
            max_grad_leaf_norm = _positive_mlx_training_number(
                getattr(args, "max_grad_leaf_norm", None),
            )
            if max_grad_norm is None and max_grad_value is None and max_grad_leaf_norm is None:
                args.max_grad_norm = 1.0

        if not max_seq_length_explicit:
            _resolve_mlx_cuda_style_max_seq_length(args, model = model)
        return args

    def _coerce_mlx_training_args(args, overrides = None):
        overrides = overrides or {}
        if isinstance(args, MLXTrainingConfig) and not overrides:
            return args
        dataset_order_explicit = None
        append_eos_explicit = None
        max_seq_length_explicit = None
        max_length_value = None
        grad_clip_explicit = None
        if args is None:
            values = {}
        elif isinstance(args, dict):
            values = dict(args)
        elif isinstance(args, (str, os.PathLike)):
            values = {"output_dir": os.fspath(args)}
        else:
            dataset_order_explicit = getattr(
                args,
                "_unsloth_mlx_dataset_order_explicit",
                False,
            )
            append_eos_explicit = getattr(
                args,
                "_unsloth_mlx_append_eos_explicit",
                None,
            )
            max_seq_length_explicit = getattr(
                args,
                "_unsloth_mlx_max_seq_length_explicit",
                None,
            )
            if max_seq_length_explicit is None:
                args_max_seq_length = _positive_mlx_context_length(
                    getattr(args, "max_seq_length", None),
                )
                default_max_seq_length = getattr(MLXTrainingConfig, "max_seq_length", 2048)
                max_seq_length_explicit = (
                    args_max_seq_length is not None
                    and args_max_seq_length != default_max_seq_length
                )
            max_length_value = getattr(
                args,
                "_unsloth_mlx_max_length_value",
                getattr(args, "max_length", None),
            )
            grad_clip_explicit = getattr(
                args,
                "_unsloth_mlx_grad_clip_explicit",
                None,
            )
            values = _mlx_training_argument_values(args)
            if hasattr(args, "max_length"):
                values["max_length"] = getattr(args, "max_length")
        values.update(overrides)
        coerced = UnslothTrainingArguments(**values)
        if (
            dataset_order_explicit is not None
            and "dataset_order" not in overrides
            and "preserve_dataset_order" not in overrides
        ):
            coerced._unsloth_mlx_dataset_order_explicit = dataset_order_explicit
        if append_eos_explicit is not None and "append_eos" not in overrides:
            coerced._unsloth_mlx_append_eos_explicit = append_eos_explicit
        if (
            max_seq_length_explicit is not None
            and "max_seq_length" not in overrides
            and "max_length" not in overrides
        ):
            coerced._unsloth_mlx_max_seq_length_explicit = max_seq_length_explicit
        if max_length_value is not None and "max_length" not in overrides:
            coerced._unsloth_mlx_max_length_value = max_length_value
            coerced.max_length = max_length_value
        if (
            grad_clip_explicit is not None
            and "max_grad_norm" not in overrides
            and "max_grad_value" not in overrides
            and "max_grad_leaf_norm" not in overrides
        ):
            coerced._unsloth_mlx_grad_clip_explicit = grad_clip_explicit
        return coerced

    _MLX_TRAINER_POSITIONAL_KWARGS = (
        "model",
        "tokenizer",
        "train_dataset",
        "eval_dataset",
        "dataset_text_field",
        "max_seq_length",
        "packing",
        "data_collator",
        "args",
        "formatting_func",
        "processor",
        "callbacks",
    )
    _TRL_SFT_TRAINER_POSITIONAL_KWARGS = (
        "model",
        "args",
        "data_collator",
        "train_dataset",
        "eval_dataset",
        "processing_class",
        "compute_loss_func",
        "compute_metrics",
        "callbacks",
        "optimizers",
        "optimizer_cls_and_kwargs",
        "preprocess_logits_for_metrics",
        "peft_config",
        "formatting_func",
    )
    _MLX_TRAINER_KWARGS = frozenset(_MLX_TRAINER_POSITIONAL_KWARGS)

    def _filter_supported_mlx_trainer_kwargs(trainer_kwargs):
        """Drop inert/empty kwargs unsupported by this zoo MLXTrainer."""
        unsupported = {
            key: value
            for key, value in trainer_kwargs.items()
            if not _mlx_trainer_supports_kwarg(key)
        }
        names = sorted(
            key for key, value in unsupported.items() if _is_meaningful_mlx_extra_value(value)
        )
        if names:
            subject = ", ".join(names)
            verb = "requires" if len(names) == 1 else "require"
            raise NotImplementedError(
                "Unsloth MLX: "
                f"{subject} {verb} an unsloth-zoo build with "
                "matching MLXTrainer support. Upgrade unsloth-zoo together "
                "with unsloth."
            )
        for key in unsupported:
            trainer_kwargs.pop(key, None)
        return trainer_kwargs

    def _is_mlx_native_text_collator(collator):
        """HF pad/copy collators are redundant on MLX; match by class name."""
        for klass in type(collator).__mro__:
            name = klass.__name__
            if name in (
                "DataCollatorForSeq2Seq",
                "DataCollatorWithPadding",
                "DefaultDataCollator",
            ):
                return True
            if name == "DataCollatorForLanguageModeling":
                # Plain causal padding is fine; MLM masking changes semantics.
                return not bool(getattr(collator, "mlm", False))
        return False

    _MLX_VISION_COLLATOR_FORWARDED_KWARGS = frozenset(
        ("completion_only_loss", "formatting_func", "max_seq_length")
    )
    _MLX_VISION_COLLATOR_IMAGE_KWARGS = frozenset(("image_size", "resize"))
    _MLX_VISION_COLLATOR_POSITIONAL_KWARGS = (
        "max_seq_length",
        "formatting_func",
        "resize",
        "ignore_index",
        "train_on_responses_only",
        "instruction_part",
        "response_part",
        "force_match",
        "num_proc",
        "completion_only_loss",
        "pad_to_multiple_of",
        "resize_dimension",
        "snap_to_patch_size",
        "last_response_only",
    )
    _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS = {
        "ignore_index": -100,
        "train_on_responses_only": False,
        "instruction_part": None,
        "response_part": None,
        "force_match": True,
        "num_proc": None,
        "pad_to_multiple_of": None,
        "resize_dimension": 0,
        "snap_to_patch_size": False,
        "last_response_only": False,
    }

    def _is_default_mlx_vision_collator_value(key, value):
        """Return whether an unsupported collator value is the CUDA default."""
        if key not in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS:
            return False
        default = _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS[key]
        if default is None:
            return value is None
        if isinstance(default, bool):
            return value is default
        return value == default and type(value) is type(default)

    def _has_mlx_training_arg_value(args, key):
        """Return whether training args already carry an explicit config value."""
        if args is None or isinstance(args, (str, os.PathLike)):
            return False
        if isinstance(args, dict):
            return key in args
        return getattr(args, key, None) is not None

    def _raise_unsupported_mlx_vision_collator_kwargs(collator_kwargs):
        """Reject VLM collator kwargs that cannot be ignored safely on MLX."""
        unsupported = sorted(
            key
            for key, value in collator_kwargs.items()
            if (
                key not in _MLX_VISION_COLLATOR_FORWARDED_KWARGS
                and key not in _MLX_VISION_COLLATOR_IMAGE_KWARGS
                and (
                    (
                        key in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS
                        and not _is_default_mlx_vision_collator_value(key, value)
                    )
                    or (
                        key not in _MLX_VISION_COLLATOR_UNSUPPORTED_DEFAULTS
                        and _is_meaningful_mlx_extra_value(value)
                    )
                )
            )
        )
        if unsupported:
            raise NotImplementedError(
                "Unsloth MLX: unsupported UnslothVisionDataCollator kwargs "
                f"cannot be ignored safely: {', '.join(unsupported)}."
            )

    class UnslothTrainer(MLXTrainer):
        """Backend-aware public trainer that routes supported SFT notebooks to MLX."""

        def __init__(self, *args, **kwargs):
            kwargs = _normalize_mlx_trainer_init_args(args, kwargs)
            processing_class = kwargs.pop("processing_class", None)
            processor_from_processing_class = False
            if processing_class is not None:
                if kwargs.get("processor", None) is None:
                    kwargs["processor"] = processing_class
                    processor_from_processing_class = True
                if kwargs.get("tokenizer", None) is None:
                    kwargs["tokenizer"] = getattr(
                        processing_class,
                        "tokenizer",
                        processing_class,
                    )
            kwargs.setdefault("tokenizer", None)

            data_collator = kwargs.pop("data_collator", None)
            if data_collator is not None:
                if isinstance(data_collator, UnslothVisionDataCollator):
                    collator_processor = getattr(data_collator, "processor", None)
                    if collator_processor is not None and (
                        kwargs.get("processor", None) is None or processor_from_processing_class
                    ):
                        kwargs["processor"] = collator_processor
                        if kwargs.get("tokenizer", None) is None:
                            kwargs["tokenizer"] = getattr(
                                collator_processor,
                                "tokenizer",
                                collator_processor,
                            )
                    collator_kwargs = getattr(data_collator, "kwargs", None) or {}
                    collator_explicit_kwargs = getattr(
                        data_collator,
                        "_unsloth_mlx_explicit_kwargs",
                        set(collator_kwargs),
                    )
                    collator_image_size = collator_kwargs.get(
                        "image_size",
                        collator_kwargs.get("resize", None),
                    )
                    if isinstance(collator_image_size, list):
                        collator_image_size = tuple(collator_image_size)
                    if (
                        isinstance(collator_image_size, str)
                        and collator_image_size.lower() == "max"
                    ):
                        collator_image_size = "max"
                    if "image_size" not in kwargs and (
                        isinstance(collator_image_size, int)
                        or collator_image_size == "max"
                        or (
                            isinstance(collator_image_size, tuple)
                            and len(collator_image_size) == 2
                            and all(isinstance(x, int) for x in collator_image_size)
                        )
                    ):
                        kwargs["image_size"] = collator_image_size
                    for collator_key in _MLX_VISION_COLLATOR_FORWARDED_KWARGS:
                        collator_defaulted_value = collator_key not in collator_explicit_kwargs
                        if collator_defaulted_value and _has_mlx_training_arg_value(
                            kwargs.get("args"), collator_key
                        ):
                            continue
                        if (
                            collator_key in collator_kwargs
                            and collator_key not in kwargs
                            and collator_kwargs[collator_key] is not None
                        ):
                            kwargs[collator_key] = collator_kwargs[collator_key]
                    _raise_unsupported_mlx_vision_collator_kwargs(collator_kwargs)
                elif _is_mlx_native_text_collator(data_collator):
                    pass
                else:
                    raise NotImplementedError(
                        "Unsloth MLX: custom data_collator is not supported by "
                        "MLXTrainer. Pass the dataset directly or use the MLX "
                        "trainer's native batching path."
                    )

            trainer_kwargs, config_kwargs, ignored_kwargs = _split_mlx_trainer_kwargs(kwargs)
            _raise_unsupported_mlx_trainer_kwargs(ignored_kwargs)
            trainer_kwargs = _filter_supported_mlx_trainer_kwargs(trainer_kwargs)
            trainer_kwargs["args"] = _coerce_mlx_training_args(
                trainer_kwargs.get("args"),
                config_kwargs,
            )
            if getattr(
                trainer_kwargs["args"], "completion_only_loss", None
            ) is True and not _is_vlm_model(trainer_kwargs.get("model")):
                raise NotImplementedError(
                    "Unsloth MLX: completion_only_loss=True is only supported "
                    "for VLM training. For text SFT, call train_on_responses_only "
                    "after constructing the trainer."
                )
            if getattr(
                trainer_kwargs["args"], "train_on_completions", None
            ) is True and not _is_vlm_model(trainer_kwargs.get("model")):
                raise NotImplementedError(
                    "Unsloth MLX: train_on_completions=True is only supported "
                    "for VLM training. For text SFT, call train_on_responses_only "
                    "after constructing the trainer."
                )
            trainer_kwargs["args"] = _apply_unsloth_trainer_mlx_defaults(
                trainer_kwargs["args"],
                model = trainer_kwargs.get("model"),
                max_seq_length_explicit = (trainer_kwargs.get("max_seq_length") is not None),
            )

            super().__init__(**trainer_kwargs)
            self.processing_class = (
                processing_class
                if processing_class is not None
                else self.processor or self.tokenizer
            )
            if trainer_kwargs.get("max_seq_length") is not None:
                _set_mlx_cuda_style_context_length(
                    self.args,
                    self.args.max_seq_length,
                )
            self._unsloth_mlx_ignored_trainer_kwargs = ignored_kwargs

    class UnslothVisionDataCollator:
        def __init__(
            self,
            model = None,
            processor = None,
            *args,
            **kwargs,
        ):
            explicit_kwargs = set(kwargs)
            if len(args) > len(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS):
                raise TypeError(
                    "UnslothVisionDataCollator on MLX accepts at most "
                    f"{len(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS)} positional "
                    "options after model and processor."
                )
            for key, value in zip(_MLX_VISION_COLLATOR_POSITIONAL_KWARGS, args):
                if key in kwargs:
                    raise TypeError(
                        f"UnslothVisionDataCollator got multiple values for argument {key!r}"
                    )
                kwargs[key] = value
                explicit_kwargs.add(key)
            if "completion_only_loss" not in kwargs:
                kwargs["completion_only_loss"] = True
            self.model = model
            self.processor = processor
            self.args = ()
            self.kwargs = kwargs
            self._unsloth_mlx_explicit_kwargs = explicit_kwargs

        def __call__(self, features):
            raise NotImplementedError(
                "Unsloth: UnslothVisionDataCollator is a compatibility placeholder "
                "on MLX. Pass the dataset to UnslothTrainer; MLXTrainer performs "
                "vision batching internally."
            )

    def get_chat_template(*args, **kwargs):
        """Apply an Unsloth chat template through a lazy MLX-safe import."""
        from .chat_templates import get_chat_template as _get_chat_template
        return _get_chat_template(*args, **kwargs)

    def apply_chat_template(*args, **kwargs):
        """Format a dataset with an Unsloth chat template through a lazy import."""
        from .chat_templates import apply_chat_template as _apply_chat_template
        return _apply_chat_template(*args, **kwargs)

    def standardize_data_formats(*args, **kwargs):
        """Normalize ShareGPT-style datasets through the shared zoo helper."""
        from unsloth_zoo.dataset_utils import standardize_data_formats as _standardize_data_formats
        return _standardize_data_formats(*args, **kwargs)

    def standardize_sharegpt(*args, **kwargs):
        """Alias ShareGPT standardization to the shared dataset-format helper."""
        return standardize_data_formats(*args, **kwargs)

    def train_on_responses_only(*args, **kwargs):
        """Mask non-response tokens through the shared zoo dataset helper."""
        # Prefer the chat_templates export, which bounds the zoo's dataset.map() worker count (#2693); it is
        # None on a torch-free host, so fall back and let the zoo raise its own ImportError.
        from .chat_templates import train_on_responses_only as _train_on_responses_only

        if _train_on_responses_only is None:
            from unsloth_zoo.dataset_utils import (
                train_on_responses_only as _train_on_responses_only,
            )
        return _train_on_responses_only(*args, **kwargs)

    def _safe_mlx_trl_star_exports(_trl):
        """Return importable TRL star exports plus the MLX SFT shims."""
        exports = list(getattr(_trl, "__all__", ()))
        safe_exports = []
        for name in exports:
            try:
                getattr(_trl, name)
            except Exception:
                continue
            safe_exports.append(name)
        for name in ("SFTConfig", "SFTTrainer"):
            if name not in safe_exports:
                safe_exports.append(name)
        return safe_exports

    # Stub the trl trainers with no MLX implementation yet, so an unmigrated GRPO/DPO/ORPO notebook
    # fails with a clear message instead of crashing deep inside the real torch/CUDA trainer.
    _MLX_UNSUPPORTED_TRL_TRAINERS = (
        "GRPOTrainer",
        "DPOTrainer",
        "ORPOTrainer",
        "KTOTrainer",
        "PPOTrainer",
        "RewardTrainer",
    )

    def _make_mlx_unsupported_trl_trainer(name):
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"Unsloth: {name} is not yet supported on the MLX (Apple Silicon) "
                f"backend. Only SFT training runs on MLX today; use a CUDA/ROCm GPU "
                f"for {name}."
            )

        return type(name, (), {"__init__": __init__, "_unsloth_mlx_unsupported": True})

    class _MLXSFTConfig(UnslothTrainingArguments):
        """`trl.SFTConfig` alias that keeps TRL's default training length.

        TRL/HF SFTConfig defaults to num_train_epochs=3 (max_steps=-1); the
        native MLX config defaults to max_steps=60. An unmigrated notebook that
        builds SFTConfig without an explicit length would otherwise silently run
        60 MLX steps under this alias, so seed the TRL epoch default when neither
        max_steps nor num_train_epochs is given (epoch mode is MLX-supported).
        """

        def __init__(self, *args, **kwargs):
            keys = set(kwargs)
            if len(args) == 1 and isinstance(args[0], dict):
                keys |= set(args[0])
            if not ({"max_steps", "num_train_epochs"} & keys):
                kwargs.setdefault("num_train_epochs", 3)
            super().__init__(*args, **kwargs)

    def _install_mlx_trl_sft_shim():
        """Install MLX-backed TRL SFT shims without replacing the TRL module."""
        _trl = _sys.modules.get("trl")
        if _trl is None:
            try:
                import trl as _trl
            except ImportError:
                _trl = _types.ModuleType("trl")
                _trl.__version__ = "0.0.0+unsloth-mlx"
                _trl.__package__ = "trl"
                _trl.__path__ = []
                _trl.__spec__ = _machinery.ModuleSpec("trl", loader = None, is_package = True)
                _sys.modules["trl"] = _trl

        _trl.SFTTrainer = UnslothTrainer
        _trl.SFTConfig = _MLXSFTConfig
        # Only retarget trainers the installed trl actually exposes; idempotent, so re-importing unsloth is a no-op.
        # Names come from trl's __all__ and already materialized attrs, never a getattr probe: resolving
        # one triggers trl's lazy trainer import and pulls torch, breaking `import unsloth` on MLX.
        _trl_exports = set(getattr(_trl, "__all__", ()) or ())
        # Stub every non-SFT trainer trl exposes, not just a fixed list, so newer trainers (RLOOTrainer, ...)
        # also fail with a clear MLX message instead of importing the real torch trainer. Names come from
        # __all__ so we never resolve them (that would trigger trl's lazy import and pull torch).
        _unsupported = set(_MLX_UNSUPPORTED_TRL_TRAINERS) | {
            _n for _n in _trl_exports if _n.endswith("Trainer") and _n != "SFTTrainer"
        }
        for _name in _unsupported:
            _current = vars(_trl).get(_name)
            if getattr(_current, "_unsloth_mlx_unsupported", False):
                continue
            if _name in _trl_exports or _current is not None:
                setattr(_trl, _name, _make_mlx_unsupported_trl_trainer(_name))
        _trl.__all__ = _safe_mlx_trl_star_exports(_trl)
        _trl.__UNSLOTH_MLX_COMPAT__ = True

    def _install_mlx_unsloth_trainer_shim():
        module_name = f"{__name__}.trainer"
        _trainer = _types.ModuleType(module_name)
        _trainer.__package__ = __name__
        _trainer.__spec__ = _machinery.ModuleSpec(module_name, loader = None)
        _trainer.MLXTrainer = MLXTrainer
        _trainer.MLXTrainingConfig = MLXTrainingConfig
        _trainer.UnslothTrainer = UnslothTrainer
        _trainer.UnslothTrainingArguments = UnslothTrainingArguments
        _trainer.UnslothVisionDataCollator = UnslothVisionDataCollator
        _sys.modules[module_name] = _trainer
        globals()["trainer"] = _trainer

    _install_mlx_trl_sft_shim()
    _install_mlx_unsloth_trainer_shim()

else:
    # GPU path: load everything from _gpu_init
    from ._gpu_init import *
    from ._gpu_init import __version__

    def get_gpu_memory_stats():
        """Return CUDA/ROCm/XPU/NPU device stats, peak memory, and total memory in GiB."""
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                props = torch.xpu.get_device_properties(0)
                peak = (
                    torch.xpu.max_memory_reserved()
                    if hasattr(torch.xpu, "max_memory_reserved")
                    else torch.xpu.max_memory_allocated()
                )
                total = getattr(props, "total_memory", 0)
                return props, _bytes_to_gb(peak), _bytes_to_gb(total) or 1.0
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                peak = torch.cuda.max_memory_reserved()
                total = getattr(props, "total_memory", 0)
                return props, _bytes_to_gb(peak), _bytes_to_gb(total) or 1.0
            # Last, so no existing device changes branch. npu fell through to a fake 1 GiB.
            if hasattr(torch, "npu") and torch.npu.is_available():
                props = torch.npu.get_device_properties(0)
                peak = (
                    torch.npu.max_memory_reserved()
                    if hasattr(torch.npu, "max_memory_reserved")
                    else torch.npu.max_memory_allocated()
                )
                total = getattr(props, "total_memory", 0)
                return props, _bytes_to_gb(peak), _bytes_to_gb(total) or 1.0
        except Exception:
            pass
        stats = _UnslothDeviceStats("Unknown GPU", 0)
        return stats, 0.0, 1.0

    def clear_gpu_memory():
        """Clear cached GPU memory on CUDA, ROCm, XPU, or NPU when available."""
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
        except Exception:
            pass


# A `pip install --target` / PYTHONPATH layout makes dill pickle whole modules by value, and every
# training path here builds a datasets.Dataset, which fingerprints through dill. See
# import_fixes.fix_dill_module_by_value_pickling.
try:
    from .import_fixes import fix_dill_module_by_value_pickling as _fix_dill
    _fix_dill()
    del _fix_dill
except Exception:
    pass
