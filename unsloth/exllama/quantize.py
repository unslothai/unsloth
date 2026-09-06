# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-the-fly quantization of a HF model to the EXL3 format."""

from __future__ import annotations

import hashlib
import os
import shutil
from typing import Optional

from .config import Exl3Config, normalize_exl3_config
from .patcher import is_exl3_model_dir
from .utils import require_exllama


def _default_cache_root() -> str:
    env = os.environ.get("UNSLOTH_EXL3_CACHE")
    if env:
        return os.path.expanduser(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(os.path.expanduser(hf_home), "unsloth", "exl3")
    return os.path.join(os.path.expanduser("~"), ".cache", "unsloth", "exl3")


# ExLlamaV3's calibration corpora; some wheels omit them, so we self-heal by
# downloading them from the pinned release on FileNotFoundError.
_CAL_DATA_FILES = (
    "c4.utf8",
    "code.utf8",
    "multilingual.utf8",
    "technical.utf8",
    "wiki.utf8",
    "tiny.utf8",
)


def _calibration_data_dir() -> Optional[str]:
    try:
        import exllamav3.conversion.calibration_data as _cd
    except Exception:
        return None
    return os.path.join(os.path.dirname(os.path.abspath(_cd.__file__)), "standard_cal_data")


def ensure_calibration_data() -> bool:
    """Ensure ExLlamaV3's bundled calibration corpora are present on disk.

    Returns True if the data is available (already or after fetching), False if
    it could not be provisioned. Missing calibration data is a packaging gap in
    some prebuilt wheels, so we download the exact files from the matching
    ExLlamaV3 release tag as a fallback.
    """
    data_dir = _calibration_data_dir()
    if data_dir is None:
        return False
    os.makedirs(data_dir, exist_ok = True)

    missing = [
        f
        for f in _CAL_DATA_FILES
        if not os.path.isfile(os.path.join(data_dir, f))
        or os.path.getsize(os.path.join(data_dir, f)) == 0
    ]
    if not missing:
        return True

    from .utils import exllama_version

    version = exllama_version() or "master"
    base_urls = [
        f"https://raw.githubusercontent.com/turboderp-org/exllamav3/v{version}/"
        f"exllamav3/conversion/standard_cal_data/",
        "https://raw.githubusercontent.com/turboderp-org/exllamav3/master/"
        "exllamav3/conversion/standard_cal_data/",
    ]

    import urllib.request

    if os.environ.get("UNSLOTH_EXL3_NO_NETWORK", "").strip().lower() in ("1", "true", "yes"):
        return False
    timeout = float(os.environ.get("UNSLOTH_EXL3_DOWNLOAD_TIMEOUT", "30"))
    for filename in missing:
        fetched = False
        for base in base_urls:
            try:
                url = base + filename
                dest = os.path.join(data_dir, filename)
                with urllib.request.urlopen(url, timeout = timeout) as resp:
                    data = resp.read()
                with open(dest, "wb") as f:
                    f.write(data)
                if os.path.getsize(dest) > 0:
                    fetched = True
                    break
            except Exception:
                continue
        if not fetched:
            return False
    return True


def _safe_name(model_name: str) -> str:
    """Turn a model id / path into a filesystem-friendly slug."""
    base = os.path.basename(os.path.normpath(model_name)) or "model"
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in base)
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def ensure_vlm_preprocessor(model_dir: str) -> bool:
    """Synthesize a missing ``preprocessor_config.json`` for a Qwen-VL checkpoint.

    Some VLM checkpoints (e.g. text-focused re-saves) ship a ``vision_config``
    but omit ``preprocessor_config.json``. ExLlamaV3's Qwen3.5 handler treats
    any model with a vision config as multimodal and hard-reads that file, so
    quantization/loading crashes with a FileNotFoundError. When it is missing we
    reconstruct a valid Qwen2-VL image-processor config from the model's own
    ``vision_config`` (patch/merge/temporal sizes), so the VLM quantizes instead
    of failing. Returns True if the file exists (already or after writing).
    """
    import json

    if not os.path.isdir(model_dir):
        return False
    dest = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.isfile(dest):
        return True

    cfg = _read_model_config(model_dir)
    vc = cfg.get("vision_config")
    if not isinstance(vc, dict):
        return False  # not a VLM - nothing to synthesize

    # Only synthesize for Qwen-family VLMs (the config is Qwen-VL specific).
    arch_blob = " ".join(str(a) for a in (cfg.get("architectures") or [])).lower()
    model_type = str(cfg.get("model_type", "")).lower()
    vc_type = str(vc.get("model_type", "")).lower()
    if not ("qwen" in arch_blob or "qwen" in model_type or "qwen" in vc_type):
        return False

    patch_size = int(vc.get("patch_size", 16))
    temporal = int(vc.get("temporal_patch_size", 2))
    merge = int(vc.get("spatial_merge_size", vc.get("merge_size", 2)))
    prep = {
        # Qwen3-VL uses edge-based sizing; these are the standard Qwen3-VL bounds.
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": patch_size,
        "temporal_patch_size": temporal,
        "merge_size": merge,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }
    try:
        with open(dest, "w", encoding = "utf-8") as f:
            json.dump(prep, f, indent = 2)
        print(
            f"Unsloth: synthesized a missing preprocessor_config.json for the VLM "
            f"at {model_dir} (from its vision_config) so EXL3 can quantize it."
        )
        return True
    except Exception:
        return False


# Map a VLM ``architectures`` value to its text-only causal-LM counterpart, so a
# VLM checkpoint can be quantized/trained as a pure text model (skipping the
# vision tower) when the user only wants the language decoder.
_VLM_TO_CAUSAL_ARCH = {
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
}


def make_text_only_config(model_dir: str) -> bool:
    """Rewrite a VLM checkpoint's config to its text-only causal-LM form.

    For a Qwen3.5(-MoE) VLM whose decoder we want to quantize/train without the
    vision tower, this: (1) sets ``architectures`` to the causal-LM class,
    (2) removes ``vision_config`` (and vision token ids), and (3) flattens the
    ``text_config`` fields to the top level (the causal config reader expects
    them there). ExLlamaV3's causal Qwen3.5(-MoE) model uses the SAME
    ``model.language_model.*`` key prefix as the VL checkpoint, so the existing
    weights (and a stitched MTP layer) load unchanged. Backs up the original
    config. Returns True if a rewrite was applied.
    """
    import json
    import shutil

    cfg_path = os.path.join(model_dir, "config.json")
    try:
        with open(cfg_path, encoding = "utf-8") as f:
            raw = json.load(f)
    except Exception:
        return False
    archs = list(raw.get("architectures") or [])
    causal = next((_VLM_TO_CAUSAL_ARCH[a] for a in archs if a in _VLM_TO_CAUSAL_ARCH), None)
    if causal is None or "vision_config" not in raw:
        return False  # not a convertible VLM (or already text-only)

    tc = raw.get("text_config")
    new = dict(raw)
    if isinstance(tc, dict):
        # Flatten text_config to top level (keep top-level keys it doesn't set).
        merged = dict(raw)
        merged.update(tc)
        new = merged
    new["architectures"] = [causal]
    new["model_type"] = str(new.get("model_type", "")).replace("_text", "") or "qwen3_5_moe"
    for k in (
        "vision_config",
        "text_config",
        "vision_start_token_id",
        "vision_end_token_id",
        "image_token_id",
        "video_token_id",
    ):
        new.pop(k, None)

    if not os.path.exists(cfg_path + ".vlm_backup"):
        shutil.copy(cfg_path, cfg_path + ".vlm_backup")
    with open(cfg_path, "w", encoding = "utf-8") as f:
        json.dump(new, f, indent = 2)
    print(
        f"Unsloth: converted VLM config to text-only '{causal}' at {model_dir} "
        f"(vision tower skipped; language decoder + MTP quantized)."
    )
    return True


def resolve_exl3_cache_dir(
    model_name: str,
    config: Exl3Config,
    cache_root: Optional[str] = None,
) -> str:
    """Return the canonical output directory for a (model, config) pair."""
    root = os.path.expanduser(cache_root) if cache_root else _default_cache_root()
    return os.path.join(root, _safe_name(model_name), config.label())


def _maybe_enable_uncalibrated(cfg):
    """If ``cfg.calibrate`` is False, force ExLlamaV3's uncalibrated quant path.

    ExLlamaV3 gates data-free quantization on a per-architecture
    ``caps['uncalibrated_quantize']`` flag that most models leave unset. When the
    user opts out of calibration we wrap ``Model.__init__`` to set that cap on
    every constructed model, so the converter skips the calibration forward
    passes / reference-state capture. Returns a restore callable (or None).
    """
    if getattr(cfg, "calibrate", True):
        return None
    try:
        from exllamav3.model.model import Model
    except Exception:
        return None
    _orig_init = Model.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        try:
            self.caps["uncalibrated_quantize"] = True
            # Calibrating every expert is the dominant MoE cost; disable it too.
            if getattr(self, "calibration_all_experts", None) is not None:
                self.calibration_all_experts = False
        except Exception:
            pass

    Model.__init__ = _patched_init
    print(
        "Unsloth: EXL3 uncalibrated (data-free) quantization enabled "
        "(calibrate=False) - faster to produce; recommended for QLoRA where the "
        "frozen base is adapted by LoRA. Use calibrate=True for best-quality "
        "inference/merge checkpoints."
    )

    def _restore():
        Model.__init__ = _orig_init

    return _restore


# Config keys that expose the number of experts across the architectures
# ExLlamaV3 supports (Mixtral, Qwen3-MoE, GLM-MoE, DeepSeek, dots, Ernie, ...).
_MOE_CONFIG_KEYS = (
    "num_local_experts",
    "num_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_tok",
)


def _read_model_config(model_dir: str) -> dict:
    import json

    path = os.path.join(model_dir, "config.json")
    try:
        with open(path, "r", encoding = "utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    # Some multimodal configs nest the decoder under text_config.
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        merged = dict(cfg)
        merged.update(text_cfg)
        return merged
    return cfg


def is_moe_model(model_dir: str) -> bool:
    """Best-effort detection of a Mixture-of-Experts model from its config."""
    cfg = _read_model_config(model_dir)
    for key in _MOE_CONFIG_KEYS:
        val = cfg.get(key)
        if isinstance(val, int) and val > 1:
            return True
    return False


def _preflight_check(model_dir: str, cfg: Exl3Config) -> None:
    """Validate EXL3 compatibility before starting a (potentially long) job.

    EXL3's trellis tiling requires the quantized matmul dimensions to be
    multiples of 128. Models with an unusual ``hidden_size`` /
    ``intermediate_size`` (e.g. 576) cannot be quantized and would otherwise
    fail deep inside the converter with a bare assertion. We surface a clear,
    actionable error up front instead.
    """
    model_config = _read_model_config(model_dir)
    checks = {
        "hidden_size": model_config.get("hidden_size"),
        "intermediate_size": model_config.get("intermediate_size"),
    }
    bad = {name: val for name, val in checks.items() if isinstance(val, int) and val % 128 != 0}
    if bad:
        details = ", ".join(f"{k}={v}" for k, v in bad.items())
        raise ValueError(
            "Unsloth: this model cannot be quantized with EXL3 because its "
            f"dimensions are not multiples of 128 ({details}).\n"
            "EXL3's trellis quantization tiles weights in 16x16 blocks and "
            "requires matmul dimensions divisible by 128. Use a 16-bit LoRA "
            "(`load_in_exl3=False`, `load_in_16bit=True`) for this model, or a "
            "model whose hidden/intermediate sizes are multiples of 128."
        )


def _conversion_complete(out_dir: str) -> bool:
    """Heuristic: a finished EXL3 conversion has a config + safetensors + index."""
    if not is_exl3_model_dir(out_dir):
        return False
    has_weights = (
        any(f.endswith(".safetensors") for f in os.listdir(out_dir))
        if os.path.isdir(out_dir)
        else False
    )
    return has_weights


def quantize_to_exl3(
    model_name: str,
    config = None,
    *,
    out_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    cache_root: Optional[str] = None,
    devices: str = "0",
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    """Quantize ``model_name`` (a local HF model dir) to EXL3 and return the path.

    :param model_name:
        Path to an *unquantized* HF model directory to convert. (Hub ids should
        be resolved to a local snapshot by the caller first.)
    :param config:
        An :class:`Exl3Config`, a bitrate, a preset string, a dict, or None.
    :param out_dir:
        Explicit output directory. Defaults to a cached location keyed by the
        model name and quant config.
    :param work_dir:
        Scratch directory for the (resumable) conversion job. Defaults to
        ``<out_dir>.work``.
    :param devices:
        Comma-separated GPU indices used for the conversion, e.g. ``"0"``.
    :param overwrite:
        Re-quantize even if a completed conversion already exists.
    :returns:
        Path to the finished EXL3 checkpoint directory.
    """
    require_exllama()
    # Ensure ExLlamaV3 compatibility patches are applied even when
    # quantize_to_exl3 is called directly (not via the loader). This installs
    # the Qwen3.5 mtp KeyError guard and the grouped-mm fallback.
    from .patcher import patch_transformers_exl3

    patch_transformers_exl3()
    cfg = normalize_exl3_config(config)

    if is_exl3_model_dir(model_name):
        # Already an EXL3 checkpoint - nothing to do.
        return model_name

    # Fail fast on incompatible dimensions rather than deep inside the converter.
    _preflight_check(model_name, cfg)

    # VLM checkpoints that ship a vision_config but no preprocessor_config.json
    # crash ExLlamaV3's multimodal config reader; synthesize one so the VLM
    # (e.g. a bigger Qwen3.5) can be quantized.
    ensure_vlm_preprocessor(model_name)

    # Auto-enable parallel expert quantization for MoEs unless overridden.
    if is_moe_model(model_name) and not cfg.parallel_mode:
        cfg.parallel_mode = True
        print(
            "Unsloth: MoE model detected - enabling EXL3 parallel quantization "
            "mode for the expert layers. bitsandbytes cannot quantize MoE models "
            "under transformers 5; EXL3 handles them natively."
        )

    resolved_out = out_dir or resolve_exl3_cache_dir(model_name, cfg, cache_root)
    resolved_work = work_dir or (resolved_out.rstrip(os.sep) + ".work")

    if _conversion_complete(resolved_out) and not overwrite:
        return resolved_out

    if overwrite and os.path.isdir(resolved_out):
        shutil.rmtree(resolved_out, ignore_errors = True)
        shutil.rmtree(resolved_work, ignore_errors = True)

    os.makedirs(resolved_out, exist_ok = True)
    os.makedirs(resolved_work, exist_ok = True)

    resume = os.path.isdir(resolved_work) and bool(os.listdir(resolved_work))

    # Self-heal the calibration corpora if the installed wheel omitted them.
    if not ensure_calibration_data():
        print(
            "Unsloth: warning - could not verify ExLlamaV3 calibration data. "
            "If quantization fails with a missing .utf8 file, reinstall "
            "exllamav3 from source or download the standard_cal_data files."
        )

    from exllamav3.conversion.convert_model import prepare, main as convert_main

    # Data-free quantization (calibrate=False): skip the converter's
    # calibration passes. Good for QLoRA, where LoRA absorbs the quant error.
    _restore_uncal = _maybe_enable_uncalibrated(cfg)

    args = _build_conversion_args(
        in_dir = model_name,
        out_dir = resolved_out,
        work_dir = resolved_work,
        cfg = cfg,
        devices = devices,
        resume = resume,
        verbose = verbose,
    )

    try:
        in_args, job_state, ok, err = prepare(args)
        if not ok:
            raise RuntimeError(f"Unsloth: EXL3 quantization failed to start: {err}")
        convert_main(in_args, job_state)
    finally:
        if _restore_uncal is not None:
            _restore_uncal()

    if not _conversion_complete(resolved_out):
        raise RuntimeError(
            f"Unsloth: EXL3 quantization did not produce a complete checkpoint at "
            f"{resolved_out}."
        )
    return resolved_out


class _ConversionArgs:
    """Lightweight stand-in for the argparse Namespace the converter expects.

    ``convert_model.prepare`` accesses conversion options as attributes; we build
    an object exposing exactly those attributes with our chosen values.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __contains__(self, key):
        # ``convert_model.prepare`` probes membership with ``arg in args``.
        return key in self.__dict__


def _build_conversion_args(
    *,
    in_dir: str,
    out_dir: str,
    work_dir: str,
    cfg: Exl3Config,
    devices: str,
    resume: bool,
    verbose: bool,
) -> _ConversionArgs:
    """Assemble the converter argument namespace from an :class:`Exl3Config`."""
    return _ConversionArgs(
        in_dir = in_dir,
        out_dir = out_dir,
        work_dir = work_dir,
        shard_size = 8192,
        bits = float(cfg.bits),
        head_bits = int(cfg.head_bits),
        mtp_bits = int(cfg.mtp_bits),
        hq = bool(cfg.hq),
        resume = bool(resume),
        cal_rows = cfg.cal_rows,
        cal_cols = cfg.cal_cols,
        checkpoint_interval = 120,
        last_checkpoint_index = None,
        verbose = bool(verbose),
        devices = str(devices),
        device_ratios = "",
        image_dump = False,
        codebook = cfg.codebook,
        parallel_mode = bool(cfg.parallel_mode),
        override_anyway = False,
        out_scales = "always",
    )
