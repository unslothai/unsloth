# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp speech-to-text sidecar for Unsloth dictation.

Serves any audio.cpp ASR GGUF (Qwen3-ASR, Parakeet, Canary, Moonshine, Nemotron and
whatever else audio.cpp transcribes) through one ``audiocpp_server`` child, with the same lifecycle as the
whisper.cpp sidecar: loads on demand, stays warm for the keep-alive window, yields
to training, and is paused while the managed runtime is being replaced. Audio is
decoded with PyAV like every other engine and sent as 16 kHz mono WAV, since the
default server build reads WAV only.
"""

from __future__ import annotations

import json
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from core.inference import audio_cpp_files, audio_cpp_server
from core.inference.audio_cpp_models import (
    DEFAULT_AUDIO_CPP_STT_MODEL,
    is_legacy_key,
    legacy_keys,
    RECOMMENDED_STT_MODELS,
    AudioCppModel,
    AudioCppModelError,
    downloaded_models,
    parse_identifier,
    require_runnable,
    resolve,
    split_variant_ref,
)
from core.inference.audio_cpp_server import (
    AudioCppRequestCancelledError,
    AudioCppRequestError,
    AudioCppServer,
    AudioCppStartCancelledError,
    AudioCppUnavailableError,
)
from core.inference.stt_ggml_sidecar import _pcm_to_wav_bytes
from core.inference.stt_sidecar import (
    STT_KEEP_ALIVE_SECONDS,
    SttAudioDecodeError,
    SttLoadCancelledError,
    SttModelIdError,
    SttModelNotDownloadedError,
    SttTranscriptionCancelledError,
    SttUnavailableError,
    _capture_stt_hub_cache,
    _claim_stt_repository,
    _decode_audio_bounded,
    _downloaded_file_bytes,
    _HF_COMMIT_SHA,
    _prepare_stt_cache_for_http,
    _TARGET_SAMPLE_RATE,
    _training_active,
    normalize_whisper_language,
)
from hub.utils.hf_errors import modelscope_missing
from hub.utils.hf_tokens import normalize_token
from loggers import get_logger

logger = get_logger(__name__)

_TRANSCRIBE_TIMEOUT_SECONDS = 600.0


class SttEngineUnavailableError(SttUnavailableError):
    """audiocpp_server is not installed or stopped serving; the audio.cpp dictation engine is off."""


# Rows the dictation picker recommends, in order; any other audio.cpp ASR GGUF works too.
AUDIO_CPP_STT_MODELS: tuple[str, ...] = RECOMMENDED_STT_MODELS


def resolve_audio_cpp_stt_model(
    model: Optional[str],
    variant: Optional[str] = None,
    *,
    network: bool = False,
    hf_token: Optional[str] = None,
) -> AudioCppModel:
    """The audio.cpp ASR model a repo id, umbrella folder id or legacy key names.

    ``network=False`` (loads and transcriptions) resolves from the HF cache, so dictation works
    offline once downloaded; downloads resolve against the Hub.
    """
    if model is None or not str(model).strip():
        model = DEFAULT_AUDIO_CPP_STT_MODEL
    base, ref_variant = split_variant_ref(str(model).strip())
    if parse_identifier(base) is None:
        raise SttModelIdError(
            f"STT model '{model}' is not an audio GGUF the audio runtime can transcribe with."
        )
    found = resolve(base, variant or ref_variant, hf_token, network = network)
    if found is None and not network:
        raise SttModelNotDownloadedError(
            f"STT model '{base}' is not downloaded. Download it in Settings, then Voice, before loading it."
        )
    if found is None:
        raise SttModelIdError(
            f"STT model '{model}' is not an audio GGUF the audio runtime can transcribe with."
        )
    try:
        require_runnable(found, "asr")
    except AudioCppModelError as exc:
        raise SttModelIdError(str(exc)) from exc
    return found


def resolve_audio_cpp_stt_model_id(model: Optional[str]) -> str:
    """The name dictation reports for ``model``: a legacy key stays that key (Settings compares
    against it), anything else becomes its row id."""
    if model is None or not str(model).strip():
        return DEFAULT_AUDIO_CPP_STT_MODEL
    base, variant = split_variant_ref(str(model).strip())
    ref = parse_identifier(base)
    if ref is None:
        raise SttModelIdError(
            f"STT model '{model}' is not an audio GGUF the audio runtime can transcribe with."
        )
    if variant is None and is_legacy_key(base):
        return base
    return ref.id


def _reported_name(requested: Optional[str], entry: AudioCppModel) -> str:
    """How status names a model loaded through ``requested``: its legacy key, else its row id."""
    base, variant = split_variant_ref(str(requested or "").strip())
    return base if variant is None and is_legacy_key(base) else entry.id


def is_model_downloaded(model: Optional[str]) -> bool:
    try:
        return audio_cpp_files.is_downloaded(resolve_audio_cpp_stt_model(model))
    except Exception:  # noqa: BLE001 - a probe never fails its caller
        return False


def downloaded_model_ids() -> list[str]:
    """Row ids of the ASR models with a variant in the HF cache (found by header), plus every legacy
    key whose own folder and variant is downloaded, which Settings and dictation compare against."""
    try:
        ids = list(dict.fromkeys(m.id for m in downloaded_models("asr")))
    except Exception:  # noqa: BLE001 - status never fails on a cache walk
        return []
    for key in legacy_keys():
        if acm_resolve_folder(key) not in {i.lower() for i in ids}:
            continue
        try:
            model = resolve(key, network = False)
        except Exception:  # noqa: BLE001 - one unreadable key never hides the rest
            continue
        if (
            model is not None
            and model.task == "asr"
            and model.unsupported is None
            and audio_cpp_files.is_downloaded(model)
        ):
            ids.append(key)
    return ids


def acm_resolve_folder(key: str) -> str:
    """The lowercased row id a legacy key names."""
    return acm_row_id(key).lower()


def acm_row_id(model: Optional[str]) -> str:
    """The row id any STT name (legacy key, ``id:variant``, folder or repo id) refers to, else ""."""
    base, _variant = split_variant_ref(str(model or "").strip())
    ref = parse_identifier(base)
    return ref.id if ref is not None else ""


# Diagnostics only: with no engine to fall back to, one bad clip must not mark audio.cpp unavailable.
_runtime_inference_failure: Optional[str] = None
_runtime_failure_lock = threading.Lock()


def note_runtime_inference_failure(reason: str) -> None:
    global _runtime_inference_failure
    with _runtime_failure_lock:
        if _runtime_inference_failure is None:
            logger.warning("audio.cpp runtime failed to serve a transcription (%s)", reason)
        _runtime_inference_failure = reason


def clear_runtime_inference_failure() -> None:
    global _runtime_inference_failure
    with _runtime_failure_lock:
        _runtime_inference_failure = None


def runtime_inference_failure() -> Optional[str]:
    with _runtime_failure_lock:
        return _runtime_inference_failure


def is_available() -> bool:
    if not audio_cpp_server.is_available():
        return False
    try:
        import av  # noqa: F401
    except Exception:
        # No PyAV means every transcription fails on decode.
        return False
    return True


def ensure_engine_available() -> str:
    try:
        return audio_cpp_server.ensure_binary()
    except AudioCppUnavailableError as exc:
        raise SttEngineUnavailableError(str(exc)) from exc


class _AudioCppDownloadState:
    """One background download of an audio.cpp ASR model's files."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._model_id: Optional[str] = None
        self._error: Optional[str] = None
        self._total_bytes: Optional[int] = None
        self._etag: Optional[str] = None
        self._revision: Optional[str] = None
        self._hub_cache: Optional[Path] = None
        self._repo: Optional[str] = None
        self._filename: Optional[str] = None
        self._cancelled = False

    def status(self) -> dict:
        with self._lock:
            downloading = self._thread is not None and self._thread.is_alive()
            # Callers track the row they picked; a variant pick arrives folded in as ``row:variant``.
            row = split_variant_ref(self._model_id)[0] if self._model_id else None
            snapshot = {
                "downloading": downloading,
                "model": row if downloading else None,
                "error": self._error,
                "cancelled": self._cancelled,
                "cancelled_model": row if self._cancelled else None,
                "bytes_total": self._total_bytes if downloading else None,
            }
            captured = (
                self._repo,
                self._filename,
                self._etag,
                self._total_bytes,
                self._hub_cache,
                self._revision,
            )
        snapshot["bytes_done"] = self._downloaded_bytes(*captured) if downloading else None
        return snapshot

    def cancel(self) -> bool:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return False
            self._cancelled = True
            process = self._process
        if process is not None and process.poll() is None:
            from core.inference.stt_download_worker import terminate_download
            terminate_download(process)
        return True

    @staticmethod
    def _downloaded_bytes(repo, filename, etag, total, hub_cache, revision) -> Optional[int]:
        try:
            if not repo or not filename or not etag or total is None or hub_cache is None:
                return None
            return _downloaded_file_bytes(
                hub_cache = hub_cache,
                repo = repo,
                filename = filename,
                size = total,
                blob_key = etag,
                revision = revision,
            )
        except Exception:
            return None

    def start(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
    ) -> None:
        model_id = str(model_id or DEFAULT_AUDIO_CPP_STT_MODEL).strip()
        resolve_audio_cpp_stt_model_id(model_id)
        hub_cache = _capture_stt_hub_cache()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._model_id == model_id:
                    if not self._cancelled:
                        return
                    raise SttModelIdError(
                        f"'{model_id}' is still cancelling; try again in a moment."
                    )
                raise SttModelIdError(
                    f"Another dictation model ('{self._model_id}') is still downloading; "
                    "wait for it to finish."
                )
            self._model_id = model_id
            self._error = None
            self._total_bytes = None
            self._etag = None
            self._revision = None
            self._repo = None
            self._filename = None
            self._hub_cache = hub_cache
            self._cancelled = False
            self._process = None
            thread = threading.Thread(
                target = self._run, args = (model_id, hf_token, hub_cache), daemon = True
            )
            self._thread = thread
            thread.start()

    def _run(self, model_id: str, hf_token: Optional[str], hub_cache: Path) -> None:
        registry = None
        owner = None
        repo = None
        try:
            model = resolve_audio_cpp_stt_model(model_id, network = True, hf_token = hf_token)
            repo = model.repo_id
            if not repo:
                raise RuntimeError("a local model file has nothing to download")
            filename = model.gguf_file
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            try:
                meta = get_hf_file_metadata(
                    hf_hub_url(repo, filename),
                    token = normalize_token(hf_token),
                )
                total_bytes = int(meta.size or 0)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError("could not resolve the download metadata") from exc
            revision = meta.commit_hash
            etag = meta.etag
            if not isinstance(revision, str) or not _HF_COMMIT_SHA.fullmatch(revision):
                raise RuntimeError("could not resolve an immutable revision")
            if not isinstance(etag, str) or not etag or total_bytes <= 0:
                raise RuntimeError("could not resolve the file identity")
            with self._lock:
                if self._cancelled:
                    return
            registry, owner = _claim_stt_repository(repo)
            with self._lock:
                if self._cancelled:
                    return
            _prepare_stt_cache_for_http(repo, hub_cache)
            with self._lock:
                self._repo = repo
                self._filename = filename
                self._total_bytes = total_bytes
                self._etag = etag
                self._revision = revision
            from core.inference.stt_download_worker import (
                reap_download,
                spawn_download,
                terminate_download,
            )

            file_args = [arg for path in model.files for arg in ("--filename", path)]
            process = spawn_download(
                ["--repo-id", repo, "--revision", revision, *file_args],
                hf_token = normalize_token(hf_token),
                hub_cache = hub_cache,
            )
            with self._lock:
                if self._cancelled:
                    terminate_download(process)
                self._process = process
            stderr = reap_download(process)
            with self._lock:
                if self._process is process:
                    self._process = None
                cancelled = self._cancelled
            if process.returncode == 0 and not cancelled:
                if audio_cpp_files.cached_files(model, hub_cache = hub_cache) is None:
                    raise RuntimeError("downloaded file is missing from the captured cache")
                from core.inference.audio_cpp_models import forget

                forget(model.id)
                return
            with self._lock:
                if cancelled or process.returncode < 0:
                    self._cancelled = True
                    return
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            logger.warning("audio.cpp STT download failed for %s: %s", model_id, detail)
            with self._lock:
                self._error = modelscope_missing(detail) or f"Download failed for '{model_id}'."
        except SttModelIdError as exc:
            with self._lock:
                self._error = str(exc)
        except Exception as exc:
            with self._lock:
                if not self._cancelled:
                    logger.warning("audio.cpp STT download failed for %s: %s", model_id, exc)
                    self._error = modelscope_missing(exc) or f"Download failed for '{model_id}'."
        finally:
            if registry is not None and owner is not None and repo is not None:
                registry.release_repository_owner(repo, owner)


_download_state = _AudioCppDownloadState()


def start_model_download(model: Optional[str], hf_token: Optional[str] = None) -> None:
    model = get_audio_cpp_stt_sidecar().keep_loaded_variant(model)
    _download_state.start(str(model or DEFAULT_AUDIO_CPP_STT_MODEL).strip(), hf_token)


def download_status() -> dict:
    return _download_state.status()


def cancel_model_download() -> bool:
    return _download_state.cancel()


class AudioCppSttSidecar:
    """Owns one audiocpp_server serving an ASR model and proxies dictation to it."""

    def __init__(self, keep_alive_seconds: float = STT_KEEP_ALIVE_SECONDS) -> None:
        self._lock = threading.RLock()
        self._load_state_lock = threading.Lock()
        self._server: Optional[AudioCppServer] = None
        self._model_id: Optional[str] = None
        self._model: Optional[AudioCppModel] = None
        # The name the last client used for the loaded model: a legacy key (what Settings and dictation
        # save) is reported back as that key, so their string comparisons keep matching.
        self._loaded_as: Optional[str] = None
        self._forced_cpu = False
        # Where the server actually runs: training also puts it on the CPU, without the user asking.
        self._launched_cpu = False
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_generation = 0
        self._keep_alive_seconds = keep_alive_seconds
        self._loading = False
        self._loading_model: Optional[str] = None
        self._load_cancel_event: Optional[threading.Event] = None
        self._load_owner_cancel_event: Optional[threading.Event] = None
        self._starting_process: Optional[subprocess.Popen] = None
        self._update_in_progress = False

    @property
    def loaded_model(self) -> Optional[str]:
        # Lock-free: transcribe() holds _lock for the whole inference, and status polls must not queue behind it.
        if not self._server_alive():
            return None
        return self._loaded_as or self._model_id

    @property
    def loaded_variant(self) -> Optional[str]:
        model = self._model
        return model.variant.key if model is not None and self._server_alive() else None

    @property
    def device(self) -> Optional[str]:
        server = self._server
        return server.backend if server is not None and server.alive() else None

    @property
    def _gpu_disabled(self) -> Optional[bool]:
        """Whether the live server runs on the CPU backend: the fact training admission reads, not the preference."""
        server = self._server
        if server is None or not server.alive():
            return None
        return server.backend == "cpu"

    def keep_loaded_variant(self, model: Optional[str]) -> Optional[str]:
        """``model`` pinned to the variant already loaded for that same row when it names none.

        A variantless id means "this row", not "this row's default": Settings and dictation send the
        bare row id, and resolving it afresh would swap a loaded Moonshine small for the default
        tiny. An explicit variant, or another row, is left alone.
        """
        loaded = self._model
        if loaded is None or not self._server_alive() or model is None or not str(model).strip():
            return model
        base, variant = split_variant_ref(str(model).strip())
        if variant is not None:
            return model
        ref = parse_identifier(base)
        # A legacy key or sub-folder id names its own variant (``audiocpp-moonshine-tiny``).
        if ref is None or ref.variant_hint or ref.id.lower() != loaded.id.lower():
            return model
        return loaded.canonical_id

    def is_loading(self) -> bool:
        with self._load_state_lock:
            return self._loading

    @property
    def loading_model(self) -> Optional[str]:
        """The row id a load in flight is reading, so a delete of its repo can wait for it."""
        with self._load_state_lock:
            return self._loading_model if self._loading else None

    @property
    def keep_alive_seconds(self) -> float:
        return self._keep_alive_seconds

    def _server_alive(self) -> bool:
        server = self._server
        return server is not None and server.alive()

    def _cancel_idle_unload_locked(self) -> None:
        self._idle_generation += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_unload_locked()
        if not self._server_alive():
            return
        generation = self._idle_generation
        timer = threading.Timer(self._keep_alive_seconds, self._idle_unload, args = (generation,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        with self._lock:
            if generation != self._idle_generation:
                return
            logger.info("Unloading idle audio.cpp STT model %s", self._model_id)
            self._release_locked()

    def _release_locked(self) -> None:
        self._cancel_idle_unload_locked()
        server = self._server
        self._server = None
        self._model_id = None
        self._model = None
        self._loaded_as = None
        self._forced_cpu = False
        self._launched_cpu = False
        if server is not None:
            server.stop()

    def _holds_expected_model(self, expected: Optional[str]) -> bool:
        if expected is None:
            return True
        current = self._model_id
        if current is None:
            return False
        try:
            # Row ids on both sides: a legacy key and its folder id name the same model.
            return current == acm_row_id(expected)
        except Exception:  # noqa: BLE001 - an unresolvable name is not this model
            return False

    def unload(
        self,
        wait: bool = True,
        expected_model: Optional[str] = None,
    ) -> None:
        if not self._lock.acquire(blocking = wait):
            return
        try:
            if not self._holds_expected_model(expected_model):
                return
            self._release_locked()
        finally:
            self._lock.release()

    def _raise_if_update_in_progress(self) -> None:
        if self._update_in_progress:
            raise SttEngineUnavailableError(
                "The audio runtime is being updated. Try dictation again shortly."
            )

    @contextmanager
    def update_maintenance(self) -> Iterator[bool]:
        """Block new loads while the managed audio.cpp tree is replaced.

        The runtime is pinned in source and replaced only by setup, which runs while Studio is
        stopped, so nothing calls this today. It is kept so an in-app updater can reuse the
        whisper.cpp update flow unchanged.
        """
        self._update_in_progress = True
        try:
            with self._lock:
                model_was_active = self._server_alive()
                self._release_locked()
                yield model_was_active
        finally:
            self._update_in_progress = False

    def cancel_pending_load(self) -> bool:
        with self._load_state_lock:
            event = self._load_cancel_event
            if not self._loading or event is None:
                return False
            event.set()
            process = self._starting_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        return True

    def _cancel_owned_load(self, owner: threading.Event) -> bool:
        with self._load_state_lock:
            event = self._load_cancel_event
            if not self._loading or event is None or self._load_owner_cancel_event is not owner:
                return False
            event.set()
            process = self._starting_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass
        return True

    def wait_for_load_to_settle(self) -> None:
        with self._lock:
            pass

    def _ensure_model_downloaded(self, model: AudioCppModel) -> str:
        try:
            return audio_cpp_files.materialize(model)
        except FileNotFoundError:
            raise SttModelNotDownloadedError(
                f"STT model '{model.display_name}' ({model.variant.key}) is not downloaded. "
                "Download it in Settings, then Voice, before loading it."
            ) from None
        except AudioCppUnavailableError as exc:
            # A path the runtime cannot open (Windows MAX_PATH) is a runtime limit, not a missing download.
            raise SttEngineUnavailableError(str(exc)) from exc

    def load(
        self,
        model: Optional[str] = None,
        request_cancel_event: Optional[threading.Event] = None,
        device: Optional[str] = None,
    ) -> None:
        """Start (or switch) audiocpp_server for the requested audio.cpp ASR model."""
        from core.inference.audio_device import audio_device_forces_cpu

        if request_cancel_event is not None and request_cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        self._raise_if_update_in_progress()
        entry = resolve_audio_cpp_stt_model(self.keep_loaded_variant(model))
        with self._lock:
            if request_cancel_event is not None and request_cancel_event.is_set():
                raise SttTranscriptionCancelledError("Transcription cancelled.")
            self._raise_if_update_in_progress()
            ensure_engine_available()
            force_cpu = (
                self._forced_cpu
                if device is None and self._server_alive()
                else audio_device_forces_cpu(device)
            )
            if (
                self._server_alive()
                and self._model == entry
                and self._forced_cpu == force_cpu
                # A server training moved to the CPU goes back to the GPU once training ends.
                and self._launched_cpu == (force_cpu or _training_active())
            ):
                self._loaded_as = _reported_name(model, entry)
                self._schedule_idle_unload_locked()
                return
            model_path = self._ensure_model_downloaded(entry)
            cancel_event = (
                request_cancel_event if request_cancel_event is not None else threading.Event()
            )
            with self._load_state_lock:
                self._load_cancel_event = cancel_event
                self._load_owner_cancel_event = request_cancel_event
                self._loading = True
                self._loading_model = entry.id
            try:
                if cancel_event.is_set():
                    raise SttLoadCancelledError("Dictation model loading was cancelled.")
                # Decided under the loading flag: training admission reads is_loading() without the lock. During training
                # the model goes to CPU so a dictation cannot reclaim the VRAM training just freed.
                run_on_cpu = force_cpu or _training_active()
                self._release_locked()

                def _track(process: subprocess.Popen) -> None:
                    with self._load_state_lock:
                        self._starting_process = process

                try:
                    server = AudioCppServer.start(
                        entry,
                        model_path,
                        force_cpu = run_on_cpu,
                        cancel_event = cancel_event,
                        on_process = _track,
                    )
                except AudioCppStartCancelledError as exc:
                    raise SttLoadCancelledError(str(exc)) from exc
                except AudioCppUnavailableError as exc:
                    raise SttEngineUnavailableError(str(exc)) from exc
                self._server = server
                self._model_id = entry.id
                self._model = entry
                self._loaded_as = _reported_name(model, entry)
                self._forced_cpu = force_cpu
                self._launched_cpu = run_on_cpu
                self._schedule_idle_unload_locked()
            finally:
                with self._load_state_lock:
                    self._loading = False
                    self._loading_model = None
                    self._load_cancel_event = None
                    self._load_owner_cancel_event = None
                    self._starting_process = None

    def transcribe(
        self,
        audio: bytes,
        model: Optional[str] = None,
        language: Optional[str] = None,
        fast: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Transcribe encoded audio bytes via audiocpp_server. Returns {text, language, duration, model}."""
        del fast  # audio.cpp ASR families decode greedily; there is no beam knob to trade.
        self._raise_if_update_in_progress()
        ensure_engine_available()
        target = self.keep_loaded_variant(model)
        entry = resolve_audio_cpp_stt_model(target)
        lang = normalize_whisper_language(language)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        # A missing model fails before decoding so a long clip does not burn CPU only to 409. The
        # cheap probe first: materialize prunes the link farm, which a warm request does not need.
        if not audio_cpp_files.is_downloaded(entry):
            self._ensure_model_downloaded(entry)
        decoded_audio = _decode_audio_bounded(audio, cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        wav_bytes = _pcm_to_wav_bytes(decoded_audio)
        with self._lock:
            try:
                # The caller's own name, so a legacy key stays the name status reports.
                if cancel_event is None:
                    self.load(target)
                else:
                    self.load(target, request_cancel_event = cancel_event)
                text = self._post_transcription(wav_bytes, lang, cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
            except Exception:
                if cancel_event is not None and cancel_event.is_set():
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
                raise
            finally:
                self._schedule_idle_unload_locked()
        duration = (len(decoded_audio) / _TARGET_SAMPLE_RATE) if len(decoded_audio) else None
        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "model": _reported_name(model, entry),
        }

    def cancel_transcription(self, cancel_event: threading.Event) -> bool:
        already_cancelled = cancel_event.is_set()
        cancel_event.set()
        return self._cancel_owned_load(cancel_event) or not already_cancelled

    def _post_transcription(
        self, wav_bytes: bytes, lang: Optional[str], cancel_event: Optional[threading.Event]
    ) -> str:
        server = self._server
        if server is None:
            raise SttEngineUnavailableError("The audio runtime is not running.")
        fields = {"model": server.model_id}
        if lang:
            fields["language"] = lang

        def post(form: dict[str, str]) -> bytes:
            return server.post_multipart(
                "/v1/audio/transcriptions",
                form,
                "dictation.wav",
                wav_bytes,
                "audio/wav",
                timeout = _TRANSCRIBE_TIMEOUT_SECONDS,
                cancel_event = cancel_event,
            )[1]

        try:
            try:
                data = post(fields)
            except AudioCppRequestError as exc:
                if "language" not in fields or not 400 <= exc.status < 500:
                    raise
                # English-only families (Moonshine, Nemotron, Parakeet) reject a language option; a
                # dictation language preference is a hint, so transcribe without it rather than fail.
                logger.info(
                    "audio.cpp: %s rejected language %r (%s); retrying without it",
                    server.model.family,
                    fields["language"],
                    exc.detail,
                )
                data = post({k: v for k, v in fields.items() if k != "language"})
            payload = json.loads(data.decode("utf-8"))
        except AudioCppRequestCancelledError as exc:
            # The server keeps decoding the abandoned clip and would queue the next request behind it;
            # stop it, as speech generation does (the caller holds self._lock).
            self._release_locked()
            raise SttTranscriptionCancelledError("Transcription cancelled.") from exc
        except AudioCppRequestError as exc:
            if 400 <= exc.status < 500:
                # The server rejected this clip or option (e.g. an unsupported language), not a broken runtime.
                raise SttAudioDecodeError(exc.detail) from exc
            note_runtime_inference_failure(str(exc))
            raise SttEngineUnavailableError(f"The audio runtime failed: {exc.detail}") from exc
        except (AudioCppUnavailableError, ValueError) as exc:
            if cancel_event is None or not cancel_event.is_set():
                note_runtime_inference_failure(f"{type(exc).__name__}: {exc}")
            raise SttEngineUnavailableError(
                "The audio runtime did not answer the request."
            ) from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str):
            raise SttAudioDecodeError("Could not decode the audio.")
        clear_runtime_inference_failure()
        return " ".join(part.strip() for part in text.splitlines() if part.strip()).strip()


_sidecar: Optional[AudioCppSttSidecar] = None


def get_audio_cpp_stt_sidecar() -> AudioCppSttSidecar:
    global _sidecar
    if _sidecar is None:
        _sidecar = AudioCppSttSidecar()
    return _sidecar
