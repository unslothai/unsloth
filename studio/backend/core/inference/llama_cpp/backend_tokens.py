# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: prompt token counting and TTS."""

import contextlib
import httpx
import json
import threading

from typing import Optional
from core.inference.chat_template_helpers import build_dac_tts_prompt
from core.inference.llama_tool_schema import llama_grammar_tools
from hub.utils.hf_errors import modelscope_missing

from .core_types import _GGUF_AUDIO_READ_TIMEOUT
from .launch_support import (
    CountAborted,
    _local_ssl_context,
)
from .log import logger

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _TokenCountingMixin:
    # ── Prompt token counting ──────────────────────────────────

    def count_chat_tokens(
        self,
        messages,
        system = None,
        tools = None,
        strict: bool = False,
        chat_template_kwargs = None,
        continue_final_message: bool = False,
        should_abort = None,
        prefer_native: bool = False,
    ) -> int:
        """Count prompt tokens for a chat request via llama-server.

        Non-strict callers keep the historical best-effort behavior and get 0 when a
        count cannot be determined. Strict callers (public count_tokens endpoints)
        raise instead: the text-only fallback is an approximation, not a count.

        ``chat_template_kwargs`` reaches /apply-template unchanged, so a caller can
        price a request rendered in a non-default reasoning mode.

        ``should_abort`` is polled between the two llama-server calls. Admission is the
        caller's job; this only stops a count that was admitted while idle from spending its
        second round trip once the answer stopped mattering. Raises when it fires.

        ``prefer_native`` tries /v1/chat/completions/input_tokens first (one round trip).
        """
        if not self.is_loaded:
            if strict:
                raise RuntimeError("llama-server is not loaded")
            return 0

        def _block_text(content) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if isinstance(block.get("text"), str):
                            parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
                return "".join(parts)
            return ""

        # Normalize system into a leading message / plain text.
        system_text = ""
        if isinstance(system, str):
            system_text = system
        elif isinstance(system, list):
            system_text = _block_text(system)

        # Count the neutralized prompt: raw text budgets a prompt generation never sends (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_control_markup,
            neutralize_control_markup_in_messages,
            neutralize_tool_descriptions,
        )

        _profile = self.markup_profile
        messages = neutralize_control_markup_in_messages(messages, None, _profile)
        # The model's own profile, like its two neighbours. The curated sweep here made the
        # count describe a different prompt from the one generation sends: a Llama profile
        # has no "</think>", so generation leaves that text byte-exact while the count path
        # was turning it into "< /think>" before /apply-template (#7066).
        system_text = (
            neutralize_control_markup(system_text)
            if _profile is None
            else _profile.rewrite_control(system_text)
        )
        tools = neutralize_tool_descriptions(tools, None, _profile)

        try:
            with httpx.Client(
                timeout = 10,
                headers = self._auth_headers,
                trust_env = False,
                verify = _local_ssl_context(),
            ) as client:

                def _tokenize(text: str) -> int:
                    r = client.post(
                        f"{self.base_url}/tokenize",
                        json = {"content": text, "add_special": True},
                    )
                    if r.status_code != 200:
                        if strict:
                            raise RuntimeError("llama-server tokenizer failed")
                        return 0
                    tokens = r.json().get("tokens", [])
                    if not isinstance(tokens, list):
                        if strict:
                            raise RuntimeError("llama-server tokenizer returned invalid tokens")
                        return 0
                    return len(tokens)

                # 1. Try /apply-template to render the real chat prompt.
                from core.inference.mcp_images import prepare_image_turn_boundaries

                template_messages = prepare_image_turn_boundaries(
                    list(messages) if messages else [],
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                )
                if system_text:
                    template_messages = [
                        {"role": "system", "content": system_text}
                    ] + template_messages

                def _render_count(render_messages) -> tuple[Optional[int], bool]:
                    """(count or None, refused). Only a 500 is a template refusal; 503 = loading or no slot."""
                    try:
                        # /apply-template renders tool schemas; omitting them undercounts.
                        template_body = {"messages": render_messages}
                        if tools:
                            template_body["tools"] = llama_grammar_tools(tools)
                        # Layered over the load-time --chat-template-kwargs: only keys sent here move.
                        if chat_template_kwargs:
                            template_body["chat_template_kwargs"] = chat_template_kwargs
                        if continue_final_message:
                            template_body["continue_final_message"] = True
                            template_body["add_generation_prompt"] = False
                        if prefer_native:
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                            try:
                                native = client.post(
                                    f"{self.base_url}/v1/chat/completions/input_tokens",
                                    json = template_body,
                                )
                                if native.status_code == 200:
                                    count = native.json().get("input_tokens")
                                    if type(count) is int and count > 0:
                                        return count, False
                            except Exception:
                                pass
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                        resp = client.post(
                            f"{self.base_url}/apply-template",
                            json = template_body,
                        )
                        if resp.status_code != 200:
                            return None, resp.status_code == 500
                        prompt = resp.json().get("prompt", "")
                        if isinstance(prompt, str):
                            if should_abort is not None and should_abort():
                                raise CountAborted()
                            return _tokenize(prompt), False
                    except CountAborted:
                        # Not a template failure: swallowed, the text fallback tokenizes anyway,
                        # which is the work being declined. Must precede the generic except.
                        raise
                    except Exception:
                        pass
                    return None, False

                # Qwen3.5+ templates raise on an empty render (#12327); a lone assistant turn is
                # stripped as a prefill, so renders empty too. Retry behind an empty user turn.
                renders_empty = not template_messages or (
                    len(template_messages) == 1
                    and isinstance(template_messages[0], dict)
                    and template_messages[0].get("role") == "assistant"
                )
                with_user_turn = [{"role": "user", "content": ""}] + template_messages
                if renders_empty and self._empty_chat_render_refused:
                    count, _refused = _render_count(with_user_turn)
                else:
                    count, _refused = _render_count(template_messages)
                    if count is None and _refused and renders_empty:
                        if should_abort is not None and should_abort():
                            raise CountAborted()
                        count, _ = _render_count(with_user_turn)
                        if count is not None:
                            self._empty_chat_render_refused = True
                if count is not None:
                    return count

                # The fallback drops role markers, special tokens and tool schemas (~30% of a
                # six-turn two-tool prompt), so strict callers error rather than undercount.
                if strict:
                    raise RuntimeError("llama-server could not render the chat template")

                # 2. Fallback: concatenate plain text and tokenize. Append a
                # serialized form of the tools so they still contribute to the
                # count when /apply-template is unavailable.
                parts = []
                if system_text:
                    parts.append(system_text)
                for msg in messages or []:
                    if isinstance(msg, dict):
                        parts.append(_block_text(msg.get("content", "")))
                if tools:
                    try:
                        parts.append(json.dumps(tools, ensure_ascii = False))
                    except Exception:
                        pass
                return _tokenize("\n".join(p for p in parts if p))
        except Exception:
            if strict:
                raise
            return 0

    # ── TTS support ────────────────────────────────────────────

    def detect_audio_type(self) -> Optional[str]:
        """Detect audio/TTS codec; swallows errors (use _strict to distinguish)."""
        try:
            return self._detect_audio_type_strict()
        except Exception as e:
            logger.debug(f"Audio type detection failed: {e}")
            return None

    def codec_failure(self) -> Optional[str]:
        """The ModelScope missing-repo sentence from this thread's last load_model codec init."""
        return getattr(self._codec_failure, "message", None)

    def _apply_detected_audio(
        self,
        detected: Optional[str],
        audio_codec_path: Optional[str] = None,
    ) -> bool:
        """Apply a probed audio codec under self._lock. Returns True to continue
        the load (codec inited OK, or nothing to init), False to abort (server
        unhealthy or codec init failed). Shared by the fast-path retry and the
        main load path."""
        if detected in ("snac", "bicodec", "dac"):
            with self._lock:
                if not self._healthy:
                    return False
                try:
                    self.init_audio_codec(detected, audio_codec_path)
                    self._is_audio = True
                    self._audio_type = detected
                except Exception as exc:
                    # Surface as HTTP 500 (matches pre-PR contract).
                    logger.warning("Failed to init audio codec '%s': %s", detected, exc)
                    self._audio_probed = False
                    self._codec_failure.message = modelscope_missing(exc)
                    return False
        elif detected:
            # csm / whisper / audio_vlm: track type but keep _is_audio False --
            # GGUF TTS routing only fires for snac/bicodec/dac.
            with self._lock:
                if not self._healthy:
                    return False
                self._audio_type = detected
        # Audio input = token probe (audio_vlm/whisper) OR mmproj encoder.
        from utils.models.model_config import is_audio_input_type

        self._has_audio_input = bool(is_audio_input_type(self._audio_type)) or bool(
            self._mmproj_has_audio
        )
        return True

    def _detect_audio_type_strict(self) -> Optional[str]:
        """Codec name on match, None on non-audio, raises on transport/JSON errors."""
        if not self.is_loaded:
            return None
        with httpx.Client(timeout = 10, headers = self._auth_headers, trust_env = False) as client:

            def _detok(tid: int) -> str:
                # Non-200 means "marker not in vocab" -- keep probing.
                # Transport / JSON errors still raise.
                r = client.post(f"{self.base_url}/detokenize", json = {"tokens": [tid]})
                if r.status_code != 200:
                    return ""
                return r.json().get("content", "")

            def _tok(text: str) -> list[int]:
                r = client.post(
                    f"{self.base_url}/tokenize",
                    json = {"content": text, "add_special": False},
                )
                if r.status_code != 200:
                    return []
                return r.json().get("tokens", [])

            from utils.audio_tokens import SNAC_PROBE_TOKEN_IDS

            # Codec-specific tokens; ids past the vocab log a llama-server 500 (#10336).
            vocab_size = getattr(self, "_vocab_size", None)
            if (not vocab_size or vocab_size > max(SNAC_PROBE_TOKEN_IDS)) and all(
                "<custom_token_" in _detok(tid) for tid in SNAC_PROBE_TOKEN_IDS
            ):
                return "snac"
            if len(_tok("<|AUDIO|>")) == 1 and len(_tok("<|audio_eos|>")) == 1:
                return "csm"
            if len(_tok("<|startoftranscript|>")) == 1:
                return "whisper"
            # Gemma 3n: <audio_soft_token>; Gemma 4: <|audio|> (not csm's <|AUDIO|>).
            if len(_tok("<audio_soft_token>")) == 1 or len(_tok("<|audio|>")) == 1:
                return "audio_vlm"
            if len(_tok("<|bicodec_semantic_0|>")) == 1 and len(_tok("<|bicodec_global_0|>")) == 1:
                return "bicodec"
            if len(_tok("<|c1_0|>")) == 1 and len(_tok("<|c2_0|>")) == 1:
                return "dac"
        return None

    # Prompt format per codec: (template, stop_tokens, needs_token_ids).
    # Matches InferenceBackend._generate_snac/bicodec/dac.
    _TTS_PROMPTS = {
        "snac": (
            "<custom_token_3>{text}<|eot_id|><custom_token_4>",
            ["<custom_token_2>"],
            True,
        ),
        "bicodec": (
            "<|task_tts|><|start_content|>{text}<|end_content|><|start_global_token|>",
            ["<|im_end|>", "</s>"],
            False,
        ),
        "dac": (
            build_dac_tts_prompt("{text}"),
            ["<|im_end|>", "<|audio_end|>"],
            False,
        ),
    }

    _codec_mgr = None  # Shared AudioCodecManager instance

    @staticmethod
    def _unload_audio_codec() -> None:
        if LlamaCppBackend._codec_mgr is None:
            return
        LlamaCppBackend._codec_mgr.unload()
        LlamaCppBackend._codec_mgr = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_audio_codec(
        self,
        audio_type: str,
        audio_codec_path: Optional[str] = None,
    ) -> None:
        """Load the audio codec at model load time (mirrors the non-GGUF path)."""
        import torch
        from core.inference.audio_codecs import AudioCodecManager

        if LlamaCppBackend._codec_mgr is None:
            LlamaCppBackend._codec_mgr = AudioCodecManager()

        # A second allocation: on CUDA for a zero-offload server it would hold VRAM the
        # load is classified as not holding, which is what lets the route skip
        # arbitration and survive training.
        device = "cuda" if torch.cuda.is_available() and not self.holds_no_vram else "cpu"
        model_repo_path = audio_codec_path

        # BiCodec needs a repo with BiCodec/ weights -- download canonical SparkTTS
        if audio_type == "bicodec" and model_repo_path is None:
            from core.inference.audio_codecs import resolve_bicodec_repo_path
            from utils.utils import hf_env_offline
            model_repo_path = resolve_bicodec_repo_path(local_files_only = hf_env_offline())

        # The codec holds memory the load-time fit never priced: VRAM on CUDA, and on a
        # unified-memory host the very pool the fit sized even when it lands on the CPU.
        # Cleared first, as the model is already listed while the codec allocates.
        self._vram_fit_context_length = None
        LlamaCppBackend._codec_mgr.load_codec(audio_type, device, model_repo_path = model_repo_path)
        logger.info(f"Loaded audio codec for GGUF TTS: {audio_type}")

    def generate_audio_response(
        self,
        text: str,
        audio_type: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        cancel_event: Optional[threading.Event] = None,
        stats_holder: Optional[dict] = None,
        audio_options: Optional[dict] = None,
    ) -> tuple:
        """
        Generate TTS audio via llama-server /completion + codec decode.
        Returns (wav_bytes, sample_rate).

        ``cancel_event`` lets a Stop or a forced model swap end the request: the
        decode is one blocking POST, so a watcher closes the client out from under
        it rather than polling. Raises RuntimeError once cancelled.
        """
        if audio_type not in self._TTS_PROMPTS:
            raise RuntimeError(f"GGUF TTS does not support '{audio_type}' codec.")

        tpl, stop, need_ids = self._TTS_PROMPTS[audio_type]
        # Raw prompt, not messages: codec delimiters are the only structure to break (#7066).
        from core.inference.chat_template_helpers import (
            neutralize_tts_prompt_text,
            tts_prompt_with_voice,
        )

        prompt_text = tts_prompt_with_voice(
            neutralize_tts_prompt_text(text, audio_type), audio_type, audio_options
        )
        payload: dict = {
            "prompt": tpl.format(text = prompt_text),
            "stream": False,
            "n_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
        }
        if stop:
            payload["stop"] = stop
        if need_ids:
            payload["n_probs"] = 1

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Audio generation cancelled")

        # Scale with the request the way the subprocess path does: a flat 300s aborts a
        # legitimate long generation on a CPU-only or slow host now that the page offers
        # more tokens than the old ceiling. Scaled from this backend's own 300s floor,
        # not the subprocess one: llama.cpp answers in seconds, and /audio/speech is in
        # _INFERENCE_SUFFIXES, so a wedged server holds other_inference_request_count()
        # up for the whole window, blocking idle auto-unload and 409-ing a training start.
        # Imported here to keep the module import acyclic.
        from core.inference.orchestrator import _audio_generation_timeout

        read_timeout = _audio_generation_timeout(max_new_tokens, base = _GGUF_AUDIO_READ_TIMEOUT)
        with httpx.Client(
            timeout = httpx.Timeout(read_timeout, connect = 10),
            headers = self._auth_headers,
            trust_env = False,
            verify = _local_ssl_context(),
        ) as client:
            finished = threading.Event()
            watcher: Optional[threading.Thread] = None
            if cancel_event is not None:

                def _close_when_cancelled() -> None:
                    while not finished.wait(0.05):
                        if cancel_event.is_set():
                            # Closing mid-request makes the blocking post raise
                            # httpx.RequestError, the only way out of it.
                            with contextlib.suppress(Exception):
                                client.close()
                            return

                watcher = threading.Thread(target = _close_when_cancelled, daemon = True)
                watcher.start()
            try:
                resp = client.post(f"{self.base_url}/completion", json = payload)
            except httpx.RequestError:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Audio generation cancelled") from None
                raise
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join(timeout = 0.5)
            if resp.status_code != 200:
                raise RuntimeError(f"llama-server returned {resp.status_code}: {resp.text}")

        # The codec decode below is GPU work with no interruption point, so check here:
        # cancelling after this only wastes the decode it cannot stop.
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Audio generation cancelled")

        data = resp.json()
        if stats_holder is not None:
            stats_holder["stats"] = {"truncated": data.get("stop_type") == "limit"}
        token_ids = (
            [p["id"] for p in data.get("completion_probabilities", []) if "id" in p]
            if need_ids
            else None
        )

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return LlamaCppBackend._codec_mgr.decode(
            audio_type, device, token_ids = token_ids, text = data.get("content", "")
        )
