# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Native Transformers/Diffusers audio generation backends.

This module intentionally has no ML imports at module import time. The worker uses
``is_native_audio_model`` before choosing its MLX or Unsloth runtime, including on
Apple Silicon, and imports torch/transformers only after the normal version and
remote-code security gates have run.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


NATIVE_AUDIO_MODEL_IDS = {
    "bosonai/higgs-tts-2-3b-base": "higgs_tts2",
    "openmoss-team/moss-tts-local-transformer-v1.5": "moss_tts_local",
    "openmoss-team/moss-tts-nano-100m": "moss_tts_nano",
    "multimodalart/higgs-audio-v3-tts-4b-transformers": "higgs_tts3",
    "minimaxai/minimax-music3": "minimax_music3",
}

NATIVE_AUDIO_MODEL_TYPES = {
    "higgs_audio_v2": "higgs_tts2",
    "moss_tts_local": "moss_tts_local",
    "moss_tts_nano": "moss_tts_nano",
    "higgs_multimodal_qwen3": "higgs_tts3",
    "minimax_music3": "minimax_music3",
}

NATIVE_AUDIO_TYPES = frozenset(NATIVE_AUDIO_MODEL_TYPES.values())
REMOTE_CODE_AUDIO_TYPES = frozenset(("moss_tts_local", "moss_tts_nano", "higgs_tts3"))
PYTHON310_AUDIO_TYPES = frozenset(("higgs_tts2", "higgs_tts3", "minimax_music3"))
MOSS_LOCAL_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MOSS_NANO_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
MOSS_NANO_TEXT_TEMPERATURE = 1.5
MOSS_NANO_TEXT_TOP_P = 1.0
MOSS_NANO_TEXT_TOP_K = 50
HIGGS_TTS2_CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"
HIGGS_TTS3_CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"
NATIVE_AUDIO_COMPANION_REPOS = {
    "higgs_tts2": (HIGGS_TTS2_CODEC_REPO,),
    "moss_tts_local": (MOSS_LOCAL_CODEC_REPO,),
    "moss_tts_nano": (MOSS_NANO_CODEC_REPO,),
    "higgs_tts3": (HIGGS_TTS3_CODEC_REPO,),
}

_MINIMAX_MODULAR_CLASSES = (
    "MiniMaxMusic3ModularPipeline",
    "MiniMaxMusic3Blocks",
)
_MINIMAX_LEADING_LYRIC_TAGS = re.compile(r"^\s*((?:\[[^\]\r\n]+\]\s*)+)(\S.*)$")


def _minimax_lyrics_for_pipeline(lyrics: str) -> str:
    """Keep words that follow a MiniMax section tag on the same input line."""
    normalized = []
    for line in str(lyrics or "").splitlines():
        match = _MINIMAX_LEADING_LYRIC_TAGS.match(line)
        if match is None:
            normalized.append(line)
            continue
        normalized.extend(re.findall(r"\[[^\]\r\n]+\]", match.group(1)))
        normalized.append(match.group(2))
    return "\n".join(normalized)


_MINIMAX_DOWNLOAD_COMPONENTS = frozenset(
    (
        "condition_encoder",
        "language_model",
        "rvq_depth_decoder",
        "scheduler",
        "tokenizer",
        "transformer",
        "vocoder",
    )
)
_MOSS_CONFIG_COMPAT_LOCK = threading.Lock()
_MOSS_NANO_SAVE_LOCK = threading.Lock()
_MAX_AUDIO_METADATA_BYTES = 1_000_000


class _AudioMetadataTooLarge(ValueError):
    pass


def _read_local_audio_metadata(
    path: Path,
    filename: str,
    *,
    reject_oversized: bool = False,
) -> dict[str, Any]:
    metadata_path = path / filename
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("rb") as handle:
        raw = handle.read(_MAX_AUDIO_METADATA_BYTES + 1)
    if len(raw) > _MAX_AUDIO_METADATA_BYTES:
        if reject_oversized:
            raise _AudioMetadataTooLarge(
                f"{filename} exceeds the {_MAX_AUDIO_METADATA_BYTES}-byte security inspection limit."
            )
        return {}
    value = json.loads(raw.decode("utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _read_audio_metadata(
    model_name: str,
    filename: str,
    hf_token: Optional[str] = None,
    *,
    reject_oversized: bool = False,
) -> dict[str, Any]:
    """Read one bounded metadata file from a local checkpoint or Hub repo."""
    normalized = str(model_name or "").strip()
    if not normalized:
        return {}
    try:
        path = Path(normalized).expanduser()
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return _read_local_audio_metadata(
                path,
                filename,
                reject_oversized = reject_oversized,
            )

        from huggingface_hub import hf_hub_download
        from utils.hf_cache_settings import active_hf_hub_cache

        metadata_path = Path(
            hf_hub_download(
                repo_id = normalized,
                filename = filename,
                token = hf_token,
                cache_dir = active_hf_hub_cache(),
            )
        )
        return _read_local_audio_metadata(
            metadata_path.parent,
            metadata_path.name,
            reject_oversized = reject_oversized,
        )
    except _AudioMetadataTooLarge:
        raise
    except Exception as exc:
        logger.debug(
            "Could not read native audio metadata %s from %s: %s", filename, normalized, exc
        )
        return {}


def _moss_local_codec_target(model_name: str, hf_token: Optional[str] = None) -> str:
    """Resolve and freeze the codec source the publisher processor will load."""
    processor_config = _read_audio_metadata(
        model_name, "processor_config.json", hf_token, reject_oversized = True
    )
    model_config = _read_audio_metadata(model_name, "config.json", hf_token, reject_oversized = True)
    nested = processor_config.get("audio_tokenizer")
    candidates = (
        processor_config.get("audio_tokenizer_name_or_path"),
        nested.get("audio_tokenizer_name_or_path") if isinstance(nested, dict) else None,
        model_config.get("audio_tokenizer_name_or_path"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return MOSS_LOCAL_CODEC_REPO


def _higgs_tts2_codec_target(model_name: str, hf_token: Optional[str] = None) -> str:
    """Resolve the codec source that the Higgs TTS 2 processor will load."""
    audio_tokenizer_config = _read_audio_metadata(
        model_name, "audio_tokenizer_config.json", hf_token, reject_oversized = True
    )
    processor_config = _read_audio_metadata(
        model_name, "processor_config.json", hf_token, reject_oversized = True
    )
    nested = processor_config.get("audio_tokenizer")
    candidates = (
        audio_tokenizer_config.get("audio_tokenizer_name_or_path"),
        processor_config.get("audio_tokenizer_name_or_path"),
        nested.get("audio_tokenizer_name_or_path") if isinstance(nested, dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return HIGGS_TTS2_CODEC_REPO


def _higgs_tts3_codec_target(model_name: str, hf_token: Optional[str] = None) -> str:
    """Resolve the codec source that the Higgs TTS 3 remote model will load."""
    model_config = _read_audio_metadata(model_name, "config.json", hf_token, reject_oversized = True)
    candidate = model_config.get("audio_tokenizer_id")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return HIGGS_TTS3_CODEC_REPO


def native_audio_type_from_local_path(model_name: str) -> Optional[str]:
    """Recognize a local native-audio checkpoint from bounded metadata files."""
    normalized = str(model_name or "").strip()
    if not normalized:
        return None
    try:
        path = Path(normalized).expanduser()
        if path.is_file():
            path = path.parent
        config = _read_local_audio_metadata(path, "config.json")
        audio_type = NATIVE_AUDIO_MODEL_TYPES.get(str(config.get("model_type") or "").lower())
        if audio_type:
            return audio_type
        modular_index = _read_local_audio_metadata(path, "modular_model_index.json")
        classes = (
            modular_index.get("_class_name"),
            modular_index.get("_blocks_class_name"),
        )
        if classes == _MINIMAX_MODULAR_CLASSES:
            return "minimax_music3"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


def _native_audio_type(model_name: str) -> Optional[str]:
    normalized = str(model_name or "").strip()
    curated = NATIVE_AUDIO_MODEL_IDS.get(normalized.lower())
    if curated:
        return curated
    return native_audio_type_from_local_path(normalized)


def is_native_audio_model(model_name: str) -> bool:
    """Whether ``model_name`` belongs in the portable native-audio worker.

    Curated Hub IDs are answered without network access. Local checkpoints are
    recognized from small metadata files; model weights are not opened.
    """
    return _native_audio_type(model_name) is not None


def native_audio_security_targets(
    model_name: str,
    audio_type: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> list[str]:
    """Repositories whose code or weights are loaded for this audio model."""
    targets = [model_name]
    resolved_type = audio_type or _native_audio_type(model_name)
    if resolved_type == "moss_tts_local":
        targets.append(_moss_local_codec_target(model_name, hf_token))
    elif resolved_type == "higgs_tts2":
        targets.append(_higgs_tts2_codec_target(model_name, hf_token))
    elif resolved_type == "higgs_tts3":
        targets.append(_higgs_tts3_codec_target(model_name, hf_token))
    else:
        targets.extend(NATIVE_AUDIO_COMPANION_REPOS.get(resolved_type, ()))
    return targets


def _kv_config_memory_gb(config: dict[str, Any]) -> float:
    def positive_int(*names: str) -> int:
        for name in names:
            try:
                value = int(config.get(name) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0

    context = positive_int(
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_length",
        "n_positions",
        "n_ctx",
        "seq_length",
    )
    layers = positive_int("num_hidden_layers", "n_layer")
    attention_heads = positive_int("num_attention_heads", "n_head")
    kv_heads = positive_int("num_key_value_heads", "n_head") or attention_heads
    head_dim = positive_int("head_dim")
    if not head_dim:
        hidden_size = positive_int("hidden_size", "n_embd")
        if hidden_size and attention_heads:
            head_dim = hidden_size // attention_heads
    if not all((context, layers, kv_heads, head_dim)):
        return 0.0
    # One key and one value per layer. Native MOSS GPU runtimes use BF16/FP16.
    return 2 * layers * kv_heads * head_dim * context * 2 / (1024**3)


def native_audio_kv_memory_gb(
    model_name: str,
    audio_type: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> float:
    """Full-context MOSS KV footprint omitted by the generic weight estimator."""
    resolved_type = audio_type or _native_audio_type(model_name)
    if resolved_type not in ("moss_tts_local", "moss_tts_nano"):
        return 0.0
    config = _read_audio_metadata(model_name, "config.json", hf_token)
    if resolved_type == "moss_tts_local":
        candidates = (config.get("qwen3_config"), config.get("gpt2_config"))
    else:
        candidates = (config.get("gpt2_config"),)
    return sum(
        _kv_config_memory_gb(candidate) for candidate in candidates if isinstance(candidate, dict)
    )


def _moss_transformers5_config_compat(codec_source: str, token_kwargs: dict[str, Any]) -> None:
    """Import a MOSS codec config with the pre-Transformers-5 subclass contract.

    Transformers 5 turns every ``PreTrainedConfig`` subclass into a dataclass.
    The published MOSS codec configs have required fields (including
    ``sampling_rate``) after inherited default fields, which Python dataclasses
    reject before any model weights are read. The remote configs already own
    their constructors, so briefly suppressing only that automatic conversion
    restores the contract they were published against. The base hook is always
    restored before the actual model load begins.
    """
    import transformers

    try:
        major = int(str(getattr(transformers, "__version__", "0")).split(".", 1)[0])
    except (TypeError, ValueError):
        return
    if major < 5:
        return

    auto_config = getattr(transformers, "AutoConfig", None)
    base_config = getattr(transformers, "PreTrainedConfig", None)
    if auto_config is None or base_config is None:
        return
    original = base_config.__dict__.get("__init_subclass__")
    if original is None:
        return

    with _MOSS_CONFIG_COMPAT_LOCK:
        setattr(
            base_config,
            "__init_subclass__",
            classmethod(lambda cls, *args, **kwargs: None),
        )
        try:
            auto_config.from_pretrained(
                codec_source,
                trust_remote_code = True,
                **token_kwargs,
            )
        finally:
            setattr(base_config, "__init_subclass__", original)


def _repair_moss_nano_rotary_buffers(model) -> None:
    """Rebuild buffers that Transformers 5 leaves uninitialized after meta loading."""
    import torch
    for decoder_name in ("transformer", "local_transformer"):
        decoder = getattr(model, decoder_name, None)
        config = getattr(decoder, "config", None)
        if decoder is None or config is None:
            continue
        base = float(getattr(config, "rope_base", 10000.0))
        for module in decoder.modules():
            rotary = getattr(module, "rotary_emb", None)
            inv_freq = getattr(rotary, "inv_freq", None)
            if inv_freq is None:
                continue
            dimension = int(inv_freq.numel()) * 2
            rotary.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(
                        0,
                        dimension,
                        2,
                        dtype = torch.float32,
                        device = inv_freq.device,
                    )
                    / dimension
                )
            )


@contextmanager
def _seeded_torch_rng(seed: Optional[int], device: str):
    """Seed one request and restore the process RNG states afterward."""
    import torch

    if seed is None:
        yield
        return

    device_type = torch.device(device).type
    accelerator = getattr(torch, device_type, None) if device_type != "cpu" else None
    devices = list(range(accelerator.device_count())) if accelerator is not None else []
    fork_device_type = device_type if accelerator is not None else "cuda"
    with torch.random.fork_rng(devices = devices, device_type = fork_device_type):
        torch.random.default_generator.manual_seed(int(seed))
        if accelerator is not None:
            manual_seed_all = getattr(accelerator, "manual_seed_all", None)
            if manual_seed_all is not None:
                manual_seed_all(int(seed))
            else:
                accelerator.manual_seed(int(seed))
        yield


@contextmanager
def _moss_nano_soundfile_save(model):
    """Route the publisher's unconditional torchaudio save through soundfile."""
    remote_module = sys.modules.get(model.__class__.__module__)
    original_torchaudio = getattr(remote_module, "torchaudio", None)
    if remote_module is None or original_torchaudio is None:
        raise RuntimeError("MOSS TTS Nano remote module does not expose torchaudio")

    class SoundfileSaveProxy:
        def __getattr__(self, name):
            return getattr(original_torchaudio, name)

        @staticmethod
        def save(uri, source, sample_rate, *_args, **_kwargs):
            payload = _as_wav_bytes(source, sample_rate)
            if hasattr(uri, "write"):
                uri.write(payload)
            else:
                Path(uri).write_bytes(payload)

    with _MOSS_NANO_SAVE_LOCK:
        remote_module.torchaudio = SoundfileSaveProxy()
        try:
            yield
        finally:
            remote_module.torchaudio = original_torchaudio


def _native_audio_file_size(sibling: Any) -> int:
    size = getattr(sibling, "size", None)
    if size is None:
        lfs = getattr(sibling, "lfs", None)
        size = getattr(lfs, "size", None) if lfs is not None else None
    try:
        return max(0, int(size or 0))
    except (TypeError, ValueError):
        return 0


def _native_audio_file_is_cached(
    repo_id: str, filename: str, revision: Optional[str], expected_size: int
) -> bool:
    """Require a complete current-revision hit in a cache the loaders reuse."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from utils.hf_cache_settings import active_hf_hub_cache

        roots: list[Optional[str]] = [str(active_hf_hub_cache()), None]
        seen: set[Optional[str]] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            current = try_to_load_from_cache(
                repo_id,
                filename,
                cache_dir = root,
                revision = revision,
            )
            default = try_to_load_from_cache(repo_id, filename, cache_dir = root)
            if not isinstance(current, str) or not isinstance(default, str):
                continue
            current_path = Path(current)
            default_path = Path(default)
            if not current_path.is_file() or not default_path.is_file():
                continue
            try:
                if current_path.resolve() != default_path.resolve():
                    continue
                if expected_size > 0 and default_path.stat().st_size != expected_size:
                    continue
            except OSError:
                continue
            return True
    except Exception:
        return False
    return False


def _native_audio_repo_files(
    audio_type: Optional[str], checkpoint: bool, siblings: list[Any]
) -> list[Any]:
    if audio_type != "minimax_music3" or not checkpoint:
        return [s for s in siblings if str(getattr(s, "rfilename", "")).strip()]

    required = []
    for sibling in siblings:
        name = str(getattr(sibling, "rfilename", "")).strip()
        if not name:
            continue
        root = name.split("/", 1)[0]
        if (
            root in _MINIMAX_DOWNLOAD_COMPONENTS
            or name == "modular_model_index.json"
            or ("/" not in name and Path(name).suffix.lower() in (".json", ".py"))
        ):
            required.append(sibling)
    return required


def native_audio_download_plan(model_name: str, hf_token: Optional[str] = None) -> dict[str, Any]:
    """Return uncached Hub files for an Audio-page TTS load.

    Native models add their companion codec repositories and MiniMax excludes
    legacy weights its modular index never references. Other TTS repositories
    use a full snapshot, matching Chat's safe generic fallback.
    """
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ValueError("A model repository is required.")
    local_checkpoint = Path(normalized).expanduser().exists()
    audio_type = _native_audio_type(normalized)
    if audio_type in PYTHON310_AUDIO_TYPES and sys.version_info < (3, 10):
        family = "Higgs TTS" if audio_type.startswith("higgs_") else "MiniMax Music 3"
        raise ValueError(f"{family} requires Python 3.10 or newer in Studio.")
    if local_checkpoint and audio_type is None:
        return {
            "entries": [],
            "total_bytes": 0,
            "required_bytes": 0,
            "checkpoint_bytes": 0,
        }

    from huggingface_hub import HfApi

    api = HfApi(token = hf_token or None)
    entries = []
    total_bytes = 0
    required_bytes = 0
    checkpoint_bytes = 0
    targets = (
        list(dict.fromkeys(native_audio_security_targets(normalized, audio_type, hf_token)))
        if audio_type is not None
        else [normalized]
    )
    for index, repo_id in enumerate(targets):
        checkpoint = index == 0
        if checkpoint and local_checkpoint:
            continue
        info = api.model_info(repo_id, files_metadata = True)
        siblings = _native_audio_repo_files(
            audio_type,
            checkpoint,
            list(getattr(info, "siblings", None) or []),
        )
        if not siblings:
            raise ValueError(f"{repo_id} does not publish files required by its audio loader.")
        revision = str(getattr(info, "sha", "") or "") or None
        missing_files = []
        missing_bytes = 0
        repo_bytes = 0
        for sibling in siblings:
            filename = str(getattr(sibling, "rfilename", ""))
            size = _native_audio_file_size(sibling)
            repo_bytes += size
            if not _native_audio_file_is_cached(repo_id, filename, revision, size):
                missing_files.append(filename)
                missing_bytes += size
        required_bytes += repo_bytes
        if checkpoint:
            checkpoint_bytes += repo_bytes
        if missing_files:
            entries.append(
                {
                    "repo_id": repo_id,
                    "files": missing_files,
                    "bytes": missing_bytes,
                    "gguf_filename": None,
                    "checkpoint": checkpoint,
                }
            )
            total_bytes += missing_bytes
    return {
        "entries": entries,
        "total_bytes": total_bytes,
        "required_bytes": required_bytes,
        "checkpoint_bytes": checkpoint_bytes,
    }


class _CancelStoppingCriteria:
    """Small adapter constructed without inheriting until Transformers is active."""

    def __init__(self, cancel_event) -> None:
        self.cancel_event = cancel_event

    def __call__(self, _input_ids, _scores, **_kwargs) -> bool:
        return bool(self.cancel_event is not None and self.cancel_event.is_set())


def _stopping_criteria(cancel_event):
    if cancel_event is None:
        return None
    from transformers import StoppingCriteria, StoppingCriteriaList

    class EventStoppingCriteria(StoppingCriteria):
        def __init__(self, event) -> None:
            self._delegate = _CancelStoppingCriteria(event)

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            return self._delegate(input_ids, scores, **kwargs)

    return StoppingCriteriaList([EventStoppingCriteria(cancel_event)])


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Audio generation cancelled")


def _as_wav_bytes(audio, sample_rate: int) -> bytes:
    """Serialize mono ``[samples]`` or channel-first stereo to a PCM WAV."""
    import numpy as np
    import soundfile as sf

    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    array = np.asarray(audio, dtype = np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        array = array.T
    if array.ndim not in (1, 2):
        raise RuntimeError(f"Audio decoder returned unsupported shape {array.shape}")

    buffer = io.BytesIO()
    sf.write(buffer, array, int(sample_rate), format = "WAV", subtype = "PCM_16")
    return buffer.getvalue()


class NativeAudioBackend:
    """One-model backend for the five curated native audio architectures."""

    def __init__(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.models: dict[str, dict[str, Any]] = {}
        self.active_model_name: Optional[str] = None
        self.loading_models: set[str] = set()

    @staticmethod
    def _token_kwargs(hf_token: Optional[str]) -> dict[str, str]:
        token = str(hf_token or "").strip()
        return {"token": token} if token else {}

    def _dtype(self):
        import torch

        if self.device == "cuda":
            if getattr(torch.version, "hip", None):
                supports_bf16 = torch.cuda.is_bf16_supported()
            else:
                try:
                    major, _minor = torch.cuda.get_device_capability()
                    supports_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
                except Exception:
                    supports_bf16 = False
            return torch.bfloat16 if supports_bf16 else torch.float16
        if self.device == "xpu":
            supports_bf16 = getattr(torch.xpu, "is_bf16_supported", None)
            return torch.bfloat16 if supports_bf16 is None or supports_bf16() else torch.float16
        if self.device == "mps":
            return torch.float16
        return torch.float32

    @staticmethod
    def _context_length(entry: dict[str, Any], requested: int, audio_type: str) -> int:
        if audio_type == "minimax_music3":
            return 0
        model = entry.get("model")
        config = getattr(model, "config", None)
        detected = 0
        nested_configs = (
            getattr(config, "language_config", None),
            getattr(config, "qwen3_config", None),
            getattr(config, "text_config", None),
        )
        for candidate in (*nested_configs, config):
            if candidate is None:
                continue
            for name in (
                "max_position_embeddings",
                "max_sequence_length",
                "max_seq_length",
                "n_positions",
                "seq_length",
            ):
                try:
                    raw_value = (
                        candidate.get(name, 0)
                        if isinstance(candidate, dict)
                        else getattr(candidate, name, 0)
                    )
                    value = int(raw_value or 0)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    detected = value
                    break
            if detected:
                break
        requested = max(0, int(requested or 0))
        if detected and audio_type in ("moss_tts_local", "moss_tts_nano"):
            return detected
        if detected and requested:
            return min(detected, requested)
        return detected or requested

    def _move(self, model):
        model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    @staticmethod
    def _configure_moss_cuda_sdpa() -> None:
        import torch

        if getattr(torch.version, "hip", None):
            return
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)

    def load_model(
        self,
        config,
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = False,
        gpu_ids: Optional[list[int]] = None,
    ) -> bool:
        del dtype, load_in_4bit
        model_name = config.identifier
        audio_type = config.audio_type
        if audio_type not in NATIVE_AUDIO_TYPES:
            raise RuntimeError(f"Unsupported native audio architecture: {audio_type or model_name}")
        if audio_type in REMOTE_CODE_AUDIO_TYPES and not trust_remote_code:
            raise RuntimeError(
                f"Model '{model_name}' requires trust_remote_code=True before its custom "
                "Transformers classes can be loaded."
            )
        if gpu_ids is not None and len(gpu_ids) > 1:
            raise RuntimeError(
                "Native audio models currently require a single selected GPU; "
                "multi-GPU sharding is not supported yet."
            )
        if audio_type == "minimax_music3" and self.device != "cuda":
            raise RuntimeError(
                "MiniMax Music 3 currently requires an NVIDIA CUDA GPU in its official "
                "local runtime. It is not available on CPU, Apple Silicon, AMD, or Intel XPU."
            )
        if audio_type == "minimax_music3":
            import torch
            if getattr(torch.version, "hip", None):
                raise RuntimeError(
                    "MiniMax Music 3 currently requires an NVIDIA CUDA GPU; "
                    "its official local runtime does not support AMD ROCm."
                )
        if audio_type == "minimax_music3" and sys.version_info < (3, 10):
            raise RuntimeError("MiniMax Music 3 requires Python 3.10 or newer in Studio.")

        if model_name in self.models:
            self.active_model_name = model_name
            return True

        self.loading_models.add(model_name)
        try:
            source = config.path or config.identifier
            entry: dict[str, Any] = {
                "is_audio": True,
                "audio_type": audio_type,
                "has_audio_input": False,
                "model_path": source,
                "context_length": 0,
            }
            if audio_type == "higgs_tts2":
                self._load_higgs_tts2(entry, source, hf_token)
            elif audio_type == "moss_tts_local":
                self._load_moss_local(entry, source, hf_token, trust_remote_code)
            elif audio_type == "moss_tts_nano":
                self._load_moss_nano(entry, source, hf_token, trust_remote_code)
            elif audio_type == "higgs_tts3":
                self._load_higgs_tts3(entry, source, hf_token, trust_remote_code)
            elif audio_type == "minimax_music3":
                self._load_minimax_music3(entry, source, hf_token)

            entry["context_length"] = self._context_length(entry, max_seq_length, audio_type)

            self.models[model_name] = entry
            self.active_model_name = model_name
            return True
        finally:
            self.loading_models.discard(model_name)

    def _load_higgs_tts2(self, entry: dict[str, Any], source: str, hf_token: Optional[str]) -> None:
        from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

        token_kwargs = self._token_kwargs(hf_token)
        processor = AutoProcessor.from_pretrained(source, **token_kwargs)
        audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
            processor.audio_tokenizer = self._move(audio_tokenizer)
        model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
            source,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
        entry.update(model = self._move(model), processor = processor, sample_rate = 24000)

    def _load_moss_local(
        self, entry: dict[str, Any], source: str, hf_token: Optional[str], trust_remote_code: bool
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        token_kwargs = self._token_kwargs(hf_token)
        codec_source = _moss_local_codec_target(source, hf_token)
        _moss_transformers5_config_compat(codec_source, token_kwargs)
        processor = AutoProcessor.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            codec_path = codec_source,
            **token_kwargs,
        )
        audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
            processor.audio_tokenizer = audio_tokenizer.to(self.device)
        attention = "sdpa" if self.device == "cuda" else "eager"
        if self.device == "cuda":
            self._configure_moss_cuda_sdpa()
        model = AutoModel.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            attn_implementation = attention,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
        sample_rate = int(getattr(processor.model_config, "sampling_rate", 48000))
        entry.update(model = self._move(model), processor = processor, sample_rate = sample_rate)

    def _load_moss_nano(
        self, entry: dict[str, Any], source: str, hf_token: Optional[str], trust_remote_code: bool
    ) -> None:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        token_kwargs = self._token_kwargs(hf_token)
        _moss_transformers5_config_compat(MOSS_NANO_CODEC_REPO, token_kwargs)
        attention = "sdpa" if self.device == "cuda" else "eager"
        if self.device == "cuda":
            self._configure_moss_cuda_sdpa()
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            attn_implementation = attention,
            local_transformer_attn_implementation = attention,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
        _repair_moss_nano_rotary_buffers(model)
        codec = AutoModel.from_pretrained(
            MOSS_NANO_CODEC_REPO,
            trust_remote_code = trust_remote_code,
            **token_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code = trust_remote_code, **token_kwargs
        )
        entry.update(
            model = self._move(model),
            processor = tokenizer,
            audio_codec = self._move(codec),
            sample_rate = 48000,
        )

    def _load_higgs_tts3(
        self, entry: dict[str, Any], source: str, hf_token: Optional[str], trust_remote_code: bool
    ) -> None:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        token_kwargs = self._token_kwargs(hf_token)
        codec_source = _higgs_tts3_codec_target(source, hf_token)
        tokenizer = AutoTokenizer.from_pretrained(source, **token_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
        model = self._move(model)
        codec = AutoModel.from_pretrained(
            codec_source,
            trust_remote_code = trust_remote_code,
            dtype = torch.float32,
            **token_kwargs,
        )
        codec = self._move(codec)
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        model._audio_codec = codec
        sample_rate = int(getattr(model.config, "sample_rate", 24000))
        entry.update(model = model, processor = tokenizer, sample_rate = sample_rate)

    def _load_minimax_music3(
        self, entry: dict[str, Any], source: str, hf_token: Optional[str]
    ) -> None:
        from diffusers import ModularPipeline

        token_kwargs = self._token_kwargs(hf_token)
        pipeline = ModularPipeline.from_pretrained(source, **token_kwargs)
        pipeline.load_components(
            pretrained_model_name_or_path = source,
            dtype = self._dtype(),
            **token_kwargs,
        )
        pipeline.to(self.device)
        entry.update(pipeline = pipeline, sample_rate = int(pipeline.sampling_rate))

    def generate_audio_response(
        self,
        text: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        use_adapter = None,
        cancel_event = None,
        instructions: Optional[str] = None,
        language: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        del min_p, use_adapter
        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        _raise_if_cancelled(cancel_event)
        entry = self.models[self.active_model_name]
        audio_type = entry["audio_type"]
        top_k = max(0, int(top_k))

        request_seed = None if audio_type == "minimax_music3" else seed
        with _seeded_torch_rng(request_seed, self.device):
            if audio_type == "higgs_tts2":
                audio, sample_rate = self._generate_higgs_tts2(
                    entry,
                    text,
                    instructions,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    cancel_event,
                )
            elif audio_type == "moss_tts_local":
                audio, sample_rate = self._generate_moss_local(
                    entry,
                    text,
                    instructions,
                    language,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    repetition_penalty,
                    cancel_event,
                )
            elif audio_type == "moss_tts_nano":
                return self._generate_moss_nano(
                    entry,
                    text,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    repetition_penalty,
                    cancel_event,
                )
            elif audio_type == "higgs_tts3":
                audio, sample_rate = self._generate_higgs_tts3(
                    entry,
                    text,
                    temperature,
                    top_p,
                    top_k,
                    max_new_tokens,
                    cancel_event,
                )
            elif audio_type == "minimax_music3":
                audio, sample_rate = self._generate_minimax_music3(
                    entry,
                    text,
                    instructions,
                    max_new_tokens,
                    seed,
                    cancel_event,
                )
            else:
                raise RuntimeError(f"Unsupported native audio architecture: {audio_type}")

        _raise_if_cancelled(cancel_event)
        return _as_wav_bytes(audio, sample_rate), sample_rate

    def _generate_higgs_tts2(
        self, entry, text, instructions, temperature, top_p, top_k, max_new_tokens, cancel_event
    ):
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text

        processor = entry["processor"]
        model = entry["model"]
        text = neutralize_tts_prompt_text(text, "higgs_tts2")
        scene = neutralize_tts_prompt_text(
            instructions or "Audio is recorded from a quiet room.", "higgs_tts2"
        )
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Generate audio following instruction."}],
            },
            {
                "role": "scene",
                "content": [
                    {
                        "type": "text",
                        "text": scene,
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            sampling_rate = entry["sample_rate"],
            return_tensors = "pt",
        ).to(model.device)
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0,
        }
        if kwargs["do_sample"]:
            kwargs.update(temperature = float(temperature), top_p = float(top_p), top_k = int(top_k))
        stopping = _stopping_criteria(cancel_event)
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping
        outputs = model.generate(**inputs, **kwargs)
        decoded = processor.batch_decode(outputs)
        return decoded[0], entry["sample_rate"]

    def _generate_moss_local(
        self,
        entry,
        text,
        instructions,
        language,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
    ):
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text

        text = neutralize_tts_prompt_text(text, "moss_tts_local")
        if instructions is not None:
            instructions = neutralize_tts_prompt_text(instructions, "moss_tts_local")
        if language is not None:
            language = neutralize_tts_prompt_text(language, "moss_tts_local")
        processor = entry["processor"]
        batch = processor(
            [
                [
                    processor.build_user_message(
                        text = text,
                        instruction = instructions,
                        language = language,
                    )
                ]
            ],
            mode = "generation",
        )
        kwargs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0,
            "audio_temperature": max(0.0, float(temperature)),
            "audio_top_p": float(top_p),
            "audio_top_k": int(top_k),
            "audio_repetition_penalty": float(repetition_penalty),
        }
        model = entry["model"]
        cancel_hooks = []
        if cancel_event is not None:
            for target in (
                getattr(model, "transformer", None),
                getattr(model, "local_transformer", None),
            ):
                if target is not None and hasattr(target, "register_forward_pre_hook"):
                    cancel_hooks.append(
                        target.register_forward_pre_hook(
                            lambda _module, _args: _raise_if_cancelled(cancel_event)
                        )
                    )
        try:
            outputs = model.generate(**kwargs)
        finally:
            for cancel_hook in cancel_hooks:
                cancel_hook.remove()
        message = next((item for item in processor.decode(outputs) if item is not None), None)
        if message is None or not message.audio_codes_list:
            raise RuntimeError("MOSS TTS Local returned no audio")
        return message.audio_codes_list[0], entry["sample_rate"]

    def _generate_moss_nano(
        self,
        entry,
        text,
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
    ) -> Tuple[bytes, int]:
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text
        text = neutralize_tts_prompt_text(text, "moss_tts_nano")
        with tempfile.TemporaryDirectory(prefix = "unsloth-moss-nano-") as temp_dir:
            output_path = Path(temp_dir) / "speech.wav"
            model = entry["model"]
            cancel_hooks = []
            if cancel_event is not None:
                targets = (
                    getattr(model, "transformer", None),
                    getattr(model, "local_transformer", None),
                    entry["audio_codec"],
                )
                seen_targets = set()
                for target in targets:
                    if id(target) in seen_targets or not hasattr(
                        target, "register_forward_pre_hook"
                    ):
                        continue
                    seen_targets.add(id(target))
                    cancel_hooks.append(
                        target.register_forward_pre_hook(
                            lambda _module, _args: _raise_if_cancelled(cancel_event)
                        )
                    )
            try:
                with _moss_nano_soundfile_save(model):
                    result = model.inference(
                        text = text,
                        output_audio_path = output_path,
                        mode = "continuation",
                        text_tokenizer = entry["processor"],
                        audio_tokenizer = entry["audio_codec"],
                        device = self.device,
                        max_new_frames = int(max_new_tokens),
                        do_sample = float(temperature) > 0,
                        text_temperature = MOSS_NANO_TEXT_TEMPERATURE,
                        text_top_p = MOSS_NANO_TEXT_TOP_P,
                        text_top_k = MOSS_NANO_TEXT_TOP_K,
                        audio_temperature = max(0.0, float(temperature)),
                        audio_top_p = float(top_p),
                        audio_top_k = int(top_k),
                        audio_repetition_penalty = float(repetition_penalty),
                        use_kv_cache = True,
                    )
            finally:
                for cancel_hook in cancel_hooks:
                    cancel_hook.remove()
            _raise_if_cancelled(cancel_event)
            sample_rate = int(result.get("sample_rate") or entry["sample_rate"])
            return output_path.read_bytes(), sample_rate

    @staticmethod
    def _generate_higgs_tts3(entry, text, temperature, top_p, top_k, max_new_tokens, cancel_event):
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text

        text = neutralize_tts_prompt_text(text, "higgs_tts3")
        model = entry["model"]
        cancel_hook = None
        forward_model = getattr(model, "model", model)
        if cancel_event is not None and hasattr(forward_model, "register_forward_pre_hook"):
            cancel_hook = forward_model.register_forward_pre_hook(
                lambda _module, _args: _raise_if_cancelled(cancel_event)
            )
        try:
            audio = model.generate_speech(
                text,
                entry["processor"],
                max_new_tokens = int(max_new_tokens),
                temperature = max(0.0, float(temperature)),
                top_p = float(top_p),
                top_k = int(top_k),
            )
        finally:
            if cancel_hook is not None:
                cancel_hook.remove()
        return audio, entry["sample_rate"]

    def _generate_minimax_music3(
        self, entry, lyrics, instructions, max_new_tokens, seed, cancel_event
    ):
        import torch
        from core.inference.chat_template_helpers import neutralize_tts_prompt_text

        prompt = str(instructions or "").strip()
        if not prompt:
            raise RuntimeError("MiniMax Music 3 requires a music description in addition to lyrics")
        lyrics = neutralize_tts_prompt_text(lyrics, "minimax_music3")
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(seed))
        pipeline = entry["pipeline"]
        frame_rate = float(getattr(pipeline, "frame_rate", 25.0) or 25.0)
        audio_duration = float(max(1, int(max_new_tokens))) / frame_rate
        pipeline_kwargs = dict(
            prompt = prompt,
            lyrics = _minimax_lyrics_for_pipeline(lyrics),
            audio_duration = audio_duration,
            output = "audios",
        )
        if generator is not None:
            pipeline_kwargs["generator"] = generator
        cancel_hooks = []
        if cancel_event is not None:
            language_model = getattr(pipeline, "language_model", None)
            targets = (
                getattr(language_model, "model", None),
                getattr(pipeline, "rvq_depth_decoder", None),
                getattr(pipeline, "condition_encoder", None),
                getattr(pipeline, "transformer", None),
                getattr(pipeline, "vocoder", None),
            )
            seen_targets = set()
            for target in targets:
                if id(target) in seen_targets or not hasattr(target, "register_forward_pre_hook"):
                    continue
                seen_targets.add(id(target))
                cancel_hooks.append(
                    target.register_forward_pre_hook(
                        lambda _module, _args: _raise_if_cancelled(cancel_event)
                    )
                )
        try:
            audio = pipeline(**pipeline_kwargs)[0]
        finally:
            for cancel_hook in cancel_hooks:
                cancel_hook.remove()
        return audio, entry["sample_rate"]

    def unload_model(self, model_name: str) -> bool:
        entry = self.models.pop(model_name, None)
        if entry is not None:
            entry.clear()
        if self.active_model_name == model_name:
            self.active_model_name = None
        self.reset_generation_state()
        return True

    def reset_generation_state(self, caller_cancel_event = None) -> None:
        del caller_cancel_event
        gc.collect()
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "xpu" and hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            logger.debug("Could not clear native audio device cache", exc_info = True)
