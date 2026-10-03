# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio the user gives the Audio page, kept for a day under the account's ``<gallery_dir>/inputs``.

Requests name an input, a gallery clip or a saved voice by id only; this module resolves the id
inside the current account, so no client ever names a file."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterable, Optional

from core.inference import audio_gallery
from loggers import get_logger
from utils.upload_limits import AUDIO_INPUT_MAX_BYTES

logger = get_logger(__name__)

MAX_SECONDS = 30 * 60
TTL_SECONDS = 24 * 60 * 60
BYTE_CAP = 2 * 1024 * 1024 * 1024
REFERENCE_MAX_SECONDS = 30.0
REFERENCE_RATE = 24000
_MAX_CHANNELS = 2
# Caps the stored rate so a 30 minute upload stays under ~350 MB however it was encoded.
MAX_RATE = 48000
_NAME_MAX = 255
_STALE_TMP_SECONDS = 60 * 60


class AudioInputError(ValueError):
    """A refusal with the HTTP status and plain reason the route sends."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def inputs_dir() -> Path:
    """``<gallery_dir>/inputs``, which ``gallery_dir`` scopes to the account for a non-owner."""
    directory = audio_gallery.gallery_dir() / "inputs"
    directory.mkdir(parents = True, exist_ok = True)
    return directory


def _valid_id(input_id: Optional[str]) -> bool:
    return bool(input_id) and bool(audio_gallery._ID_RE.match(str(input_id)))


def _now() -> float:
    return time.time()


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _clean_name(name: Optional[str]) -> str:
    """The file's own name: no folders, no control characters, bounded."""
    text = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    text = "".join(ch for ch in text if ch.isprintable()).strip()
    return text[:_NAME_MAX] or "audio"


def _sidecar(directory: Path, input_id: str) -> Path:
    return directory / f"{input_id}.json"


def _read_sidecar(path: Path) -> Optional[dict[str, Any]]:
    try:
        meta = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    required = ("name", "duration_s", "sample_rate", "channels", "sha256", "created_at")
    if not isinstance(meta, dict) or any(k not in meta for k in required):
        return None
    return meta


def _touched(meta: dict[str, Any]) -> float:
    try:
        return float(meta.get("touched_at") or meta.get("created_at_epoch") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _expired(meta: dict[str, Any], now: Optional[float] = None) -> bool:
    return _touched(meta) + TTL_SECONDS <= (now if now is not None else _now())


def _record(input_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": input_id,
        "name": meta["name"],
        "duration_s": meta["duration_s"],
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "url": f"/api/inference/audio/inputs/{input_id}/file",
        "expires_at": _iso(_touched(meta) + TTL_SECONDS),
    }


def _write_json(path: Path, meta: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(meta), encoding = "utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok = True)
        raise


def _av_open(av, source: str):
    try:
        return av.open(source, mode = "r", metadata_errors = "ignore")
    except TypeError as exc:
        if "metadata_errors" not in str(exc):
            raise
        return av.open(source, mode = "r", format = None)


def transcode(
    src: Path,
    dst: Path,
    *,
    rate: Optional[int] = None,
    mono: bool = False,
    stereo: bool = False,
    max_seconds: Optional[float] = None,
    cut: bool = False,
) -> dict[str, Any]:
    """Decode ``src`` into a 16-bit WAV at ``dst``: at ``rate`` (default the source's), mono, two
    channels (``stereo``) or at most two channels. Past ``max_seconds`` it is cut when ``cut``, else refused with 413.

    Streams frame by frame through FFmpeg's resampler (a voice reference must not alias) into a
    temporary file renamed in at the end."""
    import av

    max_seconds = MAX_SECONDS if max_seconds is None else max_seconds
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    frames = channels = ceiling = 0
    out = resampler = None

    def write(blocks) -> bool:
        nonlocal frames
        for block in blocks:
            take = min(block.samples, ceiling - frames)
            if take < block.samples and not cut:
                raise AudioInputError(
                    413, f"Audio is longer than {int(max_seconds // 60)} minutes."
                )
            out.writeframes(block.to_ndarray().reshape(-1)[: take * channels].tobytes())
            frames += take
            if cut and frames >= ceiling:
                return True
        return False

    try:
        try:
            container = _av_open(av, str(src))
        except Exception as exc:  # noqa: BLE001 - not a container PyAV knows
            raise AudioInputError(400, "This file is not audio Studio can read.") from exc
        with container:
            if not container.streams.audio:
                raise AudioInputError(400, "This file has no audio stream.")
            try:
                for frame in container.decode(container.streams.audio[0]):
                    if resampler is None:
                        rate = rate or min(int(frame.sample_rate or 0), MAX_RATE)
                        if rate <= 0:
                            raise AudioInputError(400, "This audio has no sample rate.")
                        channels = (
                            1
                            if mono
                            else 2
                            if stereo
                            else max(1, min(_MAX_CHANNELS, len(frame.layout.channels)))
                        )
                        layout = "stereo" if channels == 2 else "mono"
                        resampler = av.AudioResampler(format = "s16", layout = layout, rate = rate)
                        ceiling = int(max_seconds * rate)
                        out = wave.open(str(tmp), "wb")
                        out.setnchannels(channels)
                        out.setsampwidth(2)
                        out.setframerate(rate)
                    if write(resampler.resample(frame)):
                        break
                else:
                    if resampler is not None:
                        write(resampler.resample(None))
            except AudioInputError:
                raise
            except Exception as exc:  # noqa: BLE001 - a corrupt stream mid-file keeps what decoded
                if not frames:
                    raise AudioInputError(400, "This file is not audio Studio can read.") from exc
                logger.info("audio_inputs: decode stopped early after %d frames: %s", frames, exc)
        if out is not None:
            out.close()
            out = None
        if not frames:
            raise AudioInputError(400, "This file has no audio in it.")
        os.replace(tmp, dst)
    finally:
        if out is not None:
            out.close()
        tmp.unlink(missing_ok = True)
    return {"duration_s": round(frames / rate, 3), "sample_rate": rate, "channels": channels}


async def save_stream(
    chunks: AsyncIterable[bytes],
    name: Optional[str],
    *,
    max_bytes: Optional[int] = None,
) -> tuple[dict[str, Any], bool]:
    """Stream an upload to disk, then decode it; ``(record, created)``.

    The byte cap is checked as each chunk lands, so an oversize body is refused without being read
    whole. The same audio (sha256) uploaded again within the TTL returns the existing record."""
    import asyncio

    cap = AUDIO_INPUT_MAX_BYTES if max_bytes is None else max_bytes
    directory = inputs_dir()
    tmp = directory / f".{uuid.uuid4().hex}.upload.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > cap:
                    raise AudioInputError(413, "Audio is too large.")
                digest.update(chunk)
                f.write(chunk)
        if not size:
            raise AudioInputError(400, "Audio is empty.")
        return await asyncio.to_thread(
            _finish_upload, directory, tmp, digest.hexdigest(), _clean_name(name), size
        )
    finally:
        tmp.unlink(missing_ok = True)


def _find_by_sha(directory: Path, sha: str) -> Optional[tuple[str, dict[str, Any]]]:
    now = _now()
    for sidecar in directory.glob("*.json"):
        meta = _read_sidecar(sidecar)
        if meta is None or meta.get("sha256") != sha or _expired(meta, now):
            continue
        if (directory / f"{sidecar.stem}.wav").is_file():
            return sidecar.stem, meta
    return None


def _finish_upload(
    directory: Path, tmp: Path, sha: str, name: str, size: int
) -> tuple[dict[str, Any], bool]:
    existing = _find_by_sha(directory, sha)
    if existing is not None:
        input_id, meta = existing
        meta["touched_at"] = _now()
        _write_json(_sidecar(directory, input_id), meta)
        return _record(input_id, meta), False
    input_id = uuid.uuid4().hex
    wav = directory / f"{input_id}.wav"
    try:
        info = transcode(tmp, wav)
        now = _now()
        meta = {
            "name": name,
            **info,
            "sha256": sha,
            "bytes": size,
            "created_at": _iso(now),
            "created_at_epoch": now,
            "touched_at": now,
        }
        # The sidecar is the record's commit marker, so it lands last.
        _write_json(_sidecar(directory, input_id), meta)
    except BaseException:
        wav.unlink(missing_ok = True)
        raise
    sweep(keep = input_id)
    return _record(input_id, meta), True


def _derived(directory: Path, stem: str) -> list[Path]:
    """Prepared copies made from one input (``{id}.24000.mono....wav``)."""
    try:
        return [p for p in directory.glob(f"{stem}.*.wav") if p.name != f"{stem}.wav"]
    except OSError:
        return []


def _remove(directory: Path, input_id: str) -> bool:
    wav = directory / f"{input_id}.wav"
    existed = wav.is_file()
    for path in (wav, *_derived(directory, input_id)):
        try:
            path.unlink(missing_ok = True)
        except OSError as exc:
            logger.warning("audio_inputs.delete_failed: %s", exc)
    try:
        _sidecar(directory, input_id).unlink(missing_ok = True)
    except OSError:
        pass
    return existed


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def sweep(
    *,
    keep: Optional[str] = None,
    ttl: Optional[float] = None,
    byte_cap: Optional[int] = None,
) -> int:
    """Remove expired inputs, then the oldest beyond the byte cap; prepared copies of clips and
    voices past the TTL; stale partial uploads. ``keep`` (the input just saved) is never removed.
    Best-effort; returns the number of inputs removed."""
    ttl = TTL_SECONDS if ttl is None else ttl
    byte_cap = BYTE_CAP if byte_cap is None else byte_cap
    try:
        directory = inputs_dir()
        now = _now()
        removed = 0
        live: list[tuple[float, str, int]] = []
        for sidecar in directory.glob("*.json"):
            input_id = sidecar.stem
            meta = _read_sidecar(sidecar)
            if meta is None:
                continue
            if input_id != keep and _touched(meta) + ttl <= now:
                removed += _remove(directory, input_id)
                continue
            size = _size(directory / f"{input_id}.wav") + sum(
                _size(p) for p in _derived(directory, input_id)
            )
            live.append((_touched(meta), input_id, size))
        live.sort(reverse = True)
        running = 0
        for index, (_touched_at, input_id, size) in enumerate(live):
            running += size
            if running > byte_cap and index > 0 and input_id != keep:
                removed += _remove(directory, input_id)
        for path in directory.iterdir():
            name = path.name
            age = now - _mtime(path)
            if name.startswith(".") and name.endswith(".tmp") and age > _STALE_TMP_SECONDS:
                path.unlink(missing_ok = True)
            elif name.startswith(("c-", "v-")) and name.endswith(".wav") and age > ttl:
                path.unlink(missing_ok = True)
        # Music run folders a crash left behind.
        runs = directory / "runs"
        if runs.is_dir():
            for run in runs.iterdir():
                if run.is_dir() and now - _mtime(run) > _STALE_TMP_SECONDS:
                    shutil.rmtree(run, ignore_errors = True)
        return removed
    except Exception as exc:  # noqa: BLE001 - housekeeping never fails a request
        logger.warning("audio_inputs.sweep_failed: %s", exc)
        return 0


def input_path(input_id: str) -> Optional[Path]:
    """The canonical WAV of a live input in this account, else None."""
    if not _valid_id(input_id):
        return None
    directory = inputs_dir()
    meta = _read_sidecar(_sidecar(directory, input_id))
    if meta is None or _expired(meta):
        return None
    path = directory / f"{input_id}.wav"
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def delete(input_id: str) -> bool:
    if not _valid_id(input_id):
        return False
    directory = inputs_dir()
    if _read_sidecar(_sidecar(directory, input_id)) is None:
        return False
    return _remove(directory, input_id)


@dataclass(frozen = True)
class Source:
    kind: str  # "input" | "clip" | "voice"
    id: str
    path: Path
    name: str


def resolve_source(ref: dict[str, Any]) -> Source:
    """The canonical file behind ``{"input_id"|"clip_id"|"voice_id": id}`` in this account.

    404 for an id that is unknown, expired, unsafe or another account's."""
    kinds = [k for k in ("input_id", "clip_id", "voice_id") if ref.get(k)]
    if len(kinds) != 1:
        raise AudioInputError(400, "Name exactly one of input_id, clip_id or voice_id.")
    key = kinds[0]
    source_id = str(ref[key])
    if key == "input_id":
        path = input_path(source_id)
        if path is None:
            raise AudioInputError(404, "This reference expired. Add it again.")
        meta = _read_sidecar(_sidecar(path.parent, source_id)) or {}
        return Source("input", source_id, path, meta.get("name") or "audio")
    if key == "clip_id":
        path = audio_gallery.owned_audio_path(source_id) if _valid_id(source_id) else None
        if path is None:
            raise AudioInputError(404, "That clip is no longer in your history.")
        meta = audio_gallery._read_meta(audio_gallery._sidecar_path(source_id)) or {}
        prompt = " ".join(str(meta.get("prompt") or "").split())
        return Source("clip", source_id, path, (prompt[:60] or "Clip"))
    from core.inference import audio_voices

    path = audio_voices.voice_path(source_id)
    if path is None:
        raise AudioInputError(404, "That saved voice no longer exists.")
    voice = audio_voices.get(source_id) or {}
    return Source("voice", source_id, path, str(voice.get("name") or "Saved voice"))


def prepared_path(
    source: Source,
    rate: int,
    max_seconds: Optional[float] = None,
    stereo: bool = False,
) -> Path:
    """A cached copy of ``source`` at ``rate`` (mono, or stereo for music edits), cut to
    ``max_seconds``, in this account's inputs.

    An input's copies sit beside it as ``{id}.{rate}.{mono|stereo}[.m{n}].wav`` and go when it
    goes; a clip's or voice's as ``c-``/``v-`` files the sweep removes after the TTL."""
    prefix = {"input": "", "clip": "c-", "voice": "v-"}[source.kind]
    cap = f".m{max_seconds:g}" if max_seconds is not None else ""
    layout = "stereo" if stereo else "mono"
    dst = inputs_dir() / f"{prefix}{source.id}.{int(rate)}.{layout}{cap}.wav"
    if dst.is_file() and _mtime(dst) >= _mtime(source.path):
        if source.kind != "input":
            os.utime(dst)
        return dst
    transcode(
        source.path,
        dst,
        rate = rate,
        mono = not stereo,
        stereo = stereo,
        max_seconds = max_seconds,
        cut = True,
    )
    return dst


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        return {
            "sample_rate": rate,
            "channels": w.getnchannels(),
            "frames": w.getnframes(),
            "duration_s": round(w.getnframes() / rate, 3) if rate else 0.0,
        }


def prepare_reference(ref: dict[str, Any]) -> tuple[Source, Path]:
    """A voice reference ready for the runtime: 24 kHz mono, at most 30 s."""
    source = resolve_source(ref)
    return source, prepared_path(source, REFERENCE_RATE, REFERENCE_MAX_SECONDS)
