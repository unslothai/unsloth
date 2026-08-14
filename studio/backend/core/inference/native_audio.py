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
import sys
import tempfile
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
MOSS_LOCAL_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MOSS_NANO_CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
HIGGS_TTS3_CODEC_REPO = "bosonai/higgs-audio-v2-tokenizer"
NATIVE_AUDIO_COMPANION_REPOS = {
    "moss_tts_local": (MOSS_LOCAL_CODEC_REPO,),
    "moss_tts_nano": (MOSS_NANO_CODEC_REPO,),
    "higgs_tts3": (HIGGS_TTS3_CODEC_REPO,),
}

_MINIMAX_MODULAR_CLASSES = (
    "MiniMaxMusic3ModularPipeline",
    "MiniMaxMusic3Blocks",
)


def _read_local_audio_metadata(path: Path, filename: str) -> dict[str, Any]:
    metadata_path = path / filename
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("rb") as handle:
        raw = handle.read(1_000_001)
    if len(raw) > 1_000_000:
        return {}
    value = json.loads(raw.decode("utf-8-sig"))
    return value if isinstance(value, dict) else {}


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
        audio_type = NATIVE_AUDIO_MODEL_TYPES.get(
            str(config.get("model_type") or "").lower()
        )
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


def native_audio_security_targets(model_name: str, audio_type: Optional[str] = None) -> list[str]:
    """Repositories whose code or weights are loaded for this audio model."""
    targets = [model_name]
    resolved_type = audio_type or _native_audio_type(model_name)
    targets.extend(NATIVE_AUDIO_COMPANION_REPOS.get(resolved_type, ()))
    return targets


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
    def _context_length(
        entry: dict[str, Any], requested: int, audio_type: str
    ) -> int:
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
        if detected and requested:
            return min(detected, requested)
        return detected or requested

    def _move(self, model):
        model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        return model

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

            entry["context_length"] = self._context_length(
                entry, max_seq_length, audio_type
            )

            self.models[model_name] = entry
            self.active_model_name = model_name
            return True
        finally:
            self.loading_models.discard(model_name)

    def _load_higgs_tts2(self, entry: dict[str, Any], source: str, hf_token: Optional[str]) -> None:
        from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

        token_kwargs = self._token_kwargs(hf_token)
        processor = AutoProcessor.from_pretrained(source, **token_kwargs)
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
        processor = AutoProcessor.from_pretrained(
            source, trust_remote_code = trust_remote_code, **token_kwargs
        )
        audio_tokenizer = getattr(processor, "audio_tokenizer", None)
        if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
            processor.audio_tokenizer = audio_tokenizer.to(self.device)
        attention = "eager" if self.device in ("cpu", "mps") else "sdpa"
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
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token_kwargs = self._token_kwargs(hf_token)
        tokenizer = AutoTokenizer.from_pretrained(source, **token_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code = trust_remote_code,
            torch_dtype = self._dtype(),
            **token_kwargs,
        )
        model = self._move(model)
        sample_rate = int(getattr(model.config, "sample_rate", 24000))
        entry.update(model = model, processor = tokenizer, sample_rate = sample_rate)

    def _load_minimax_music3(
        self, entry: dict[str, Any], source: str, hf_token: Optional[str]
    ) -> None:
        from diffusers import ModularPipeline

        token_kwargs = self._token_kwargs(hf_token)
        pipeline = ModularPipeline.from_pretrained(source, **token_kwargs)
        pipeline.load_components(dtype = self._dtype(), **token_kwargs)
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
        seed: Optional[int] = None,
    ) -> Tuple[bytes, int]:
        del min_p, use_adapter
        if not self.active_model_name or self.active_model_name not in self.models:
            raise RuntimeError("No active audio model")
        _raise_if_cancelled(cancel_event)
        entry = self.models[self.active_model_name]
        audio_type = entry["audio_type"]
        top_k = max(0, int(top_k))

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
        processor = entry["processor"]
        model = entry["model"]
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
                        "text": instructions or "Audio is recorded from a quiet room.",
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
        temperature,
        top_p,
        top_k,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
    ):
        processor = entry["processor"]
        batch = processor([[processor.build_user_message(text = text)]], mode = "generation")
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
        stopping = _stopping_criteria(cancel_event)
        if stopping is not None:
            kwargs["stopping_criteria"] = stopping
        outputs = entry["model"].generate(**kwargs)
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
        with tempfile.TemporaryDirectory(prefix = "unsloth-moss-nano-") as temp_dir:
            output_path = Path(temp_dir) / "speech.wav"
            result = entry["model"].inference(
                text = text,
                output_audio_path = output_path,
                mode = "continuation",
                text_tokenizer = entry["processor"],
                audio_tokenizer = entry["audio_codec"],
                device = self.device,
                max_new_frames = int(max_new_tokens),
                do_sample = float(temperature) > 0,
                text_temperature = max(0.0, float(temperature)),
                text_top_p = float(top_p),
                text_top_k = int(top_k),
                audio_temperature = max(0.0, float(temperature)),
                audio_top_p = float(top_p),
                audio_top_k = int(top_k),
                audio_repetition_penalty = float(repetition_penalty),
                use_kv_cache = True,
            )
            _raise_if_cancelled(cancel_event)
            sample_rate = int(result.get("sample_rate") or entry["sample_rate"])
            return output_path.read_bytes(), sample_rate

    @staticmethod
    def _generate_higgs_tts3(entry, text, temperature, top_p, top_k, max_new_tokens):
        audio = entry["model"].generate_speech(
            text,
            entry["processor"],
            max_new_tokens = int(max_new_tokens),
            temperature = max(0.0, float(temperature)),
            top_p = float(top_p),
            top_k = int(top_k),
        )
        return audio, entry["sample_rate"]

    def _generate_minimax_music3(
        self, entry, lyrics, instructions, max_new_tokens, seed, cancel_event
    ):
        import torch

        prompt = str(instructions or "").strip()
        if not prompt:
            raise RuntimeError("MiniMax Music 3 requires a music description in addition to lyrics")
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(int(seed))
        audio_duration = min(300.0, max(1.0, float(max_new_tokens) / 25.0))
        pipeline_kwargs = dict(
            prompt = prompt,
            lyrics = lyrics,
            audio_duration = audio_duration,
            output = "audios",
        )
        if generator is not None:
            pipeline_kwargs["generator"] = generator
        pipeline = entry["pipeline"]
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
