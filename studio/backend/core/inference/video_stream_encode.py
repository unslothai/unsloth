# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Encode a Wan clip while its VAE is still decoding.

A decoder forward hook converts each chunk to uint8 with the pipeline's exact ops, copies it to pinned memory on a
side stream and feeds a libx264 thread. The streamed mp4 is used only when its frames equal the final frames bit for
bit; otherwise the clip is encoded as before. ``UNSLOTH_STUDIO_VIDEO_STREAM_ENCODE=0`` turns it off.
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator, Optional

STREAM_ENCODE_ENV = "UNSLOTH_STUDIO_VIDEO_STREAM_ENCODE"
QUEUE_DEPTH = 8
_STOP = object()

STATS = {"streamed": 0, "fallback": 0, "last": ""}


def stream_encode_enabled() -> bool:
    return os.environ.get(STREAM_ENCODE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _note(outcome: str, reason: str, logger: Any) -> None:
    STATS[outcome] += 1
    STATS["last"] = reason
    if logger is not None:
        logger.info("video.stream_encode: %s (%s)", outcome, reason)


class StreamEncode:
    """One render's overlapped encode. Hooks are live only inside ``hooks()``; ``take`` or ``close`` ends it."""

    def __init__(
        self,
        vae: Any,
        processor: Any,
        fps: int,
        logger: Any = None,
    ) -> None:
        import torch

        self._torch = torch
        self.vae = vae
        self.processor = processor
        self.fps = int(fps)
        self.logger = logger
        patch = getattr(getattr(vae, "config", None), "patch_size", None)
        self._unpatchify = (
            getattr(sys.modules.get(type(vae).__module__), "unpatchify", None) if patch else None
        )
        self._patch = patch
        self._side = None
        self.chunks: list = []
        self.reason = ""
        self.valid = True
        self._decoding = False
        self._decoded = False
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize = QUEUE_DEPTH)
        self._failed = False
        self._thread: Optional[threading.Thread] = None
        handle = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
        handle.close()
        self.path = handle.name

    def invalidate(self, reason: str) -> None:
        if self.valid:
            self.valid = False
            self.reason = reason

    def _on_chunk(self, _module: Any, args: Any, kwargs: Any, output: Any) -> None:
        if not self.valid or not self._decoding:
            return
        torch = self._torch
        try:
            if kwargs.get("first_chunk") and self.chunks:
                # The fp16 decode's fp32 rerun (or any second pass) starts over: the chunks so far are stale.
                self.invalidate("decoder restarted")
                return
            if not torch.is_tensor(output) or output.ndim != 5 or output.shape[0] != 1:
                self.invalidate("unexpected decoder output")
                return
            if self._failed:
                self.invalidate("encoder failed")
                return
            x = output
            if self._patch:
                if self._unpatchify is None:
                    self.invalidate("no unpatchify")
                    return
                x = self._unpatchify(x, patch_size = self._patch)
            x = torch.clamp(x, min = -1.0, max = 1.0)
            x = self.processor.postprocess(x[0].permute(1, 0, 2, 3), "pt")
            x = (
                (x.to(torch.float32) * 255)
                .round_()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            device = x.device
            done = None
            if device.type == "cuda":
                if self._side is None:
                    self._side = torch.cuda.Stream(device = device)
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
                host = torch.empty(x.shape, dtype = torch.uint8, pin_memory = True)
                with torch.cuda.stream(self._side):
                    self._side.wait_event(ready)
                    host.copy_(x, non_blocking = True)
                    done = torch.cuda.Event()
                    done.record(self._side)
                x.record_stream(self._side)
            else:
                host = x.cpu() if device.type != "cpu" else x
        except Exception as exc:  # noqa: BLE001 - streaming is an optimisation; the normal encode still runs
            self.invalidate(f"chunk failed: {type(exc).__name__}")
            return
        self.chunks.append((host, done))
        if self._thread is None:
            self._thread = threading.Thread(
                target = self._encode, name = "video-stream-encode", daemon = True
            )
            self._thread.start()
        self._queue.put((host, done))

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        """Arm the decoder hook for the outermost ``vae.decode``; a tiled decode disarms it for this render."""
        vae = self.vae
        restore: list = []

        def _wrap(name: str, make: Any) -> None:
            original = getattr(vae, name, None)
            if not callable(original):
                return
            had_own = name in getattr(vae, "__dict__", {})
            setattr(vae, name, make(original))
            restore.append((name, original, had_own))

        def _decode_wrapper(original: Any) -> Any:
            def decode(*args: Any, **kwargs: Any) -> Any:
                if self._decoding:
                    return original(*args, **kwargs)
                if self._decoded:
                    self.invalidate("second decode")
                self._decoding = True
                try:
                    return original(*args, **kwargs)
                finally:
                    self._decoding = False
                    self._decoded = True

            return decode

        def _tiled_wrapper(original: Any) -> Any:
            def tiled_decode(*args: Any, **kwargs: Any) -> Any:
                self.invalidate("tiled decode")
                return original(*args, **kwargs)

            return tiled_decode

        handle = None
        try:
            _wrap("decode", _decode_wrapper)
            _wrap("tiled_decode", _tiled_wrapper)
            handle = vae.decoder.register_forward_hook(self._on_chunk, with_kwargs = True)
        except Exception:  # noqa: BLE001 - a VAE that refuses the hooks just encodes afterwards
            self.invalidate("hooks refused")
        try:
            yield
        except BaseException:
            self.close()
            raise
        finally:
            if handle is not None:
                handle.remove()
            for name, original, had_own in reversed(restore):
                try:
                    if had_own:
                        setattr(vae, name, original)
                    else:
                        delattr(vae, name)
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass

    def _encode(self) -> None:
        container = None
        try:
            import av
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                if self._failed or not self.valid:
                    continue
                host, done = item
                if done is not None:
                    done.synchronize()
                frames = host.numpy()
                if container is None:
                    container = av.open(self.path, mode = "w", format = "mp4")
                    stream = container.add_stream("libx264", rate = self.fps)
                    stream.width = int(frames.shape[2])
                    stream.height = int(frames.shape[1])
                    stream.pix_fmt = "yuv420p"
                    stream.codec_context.thread_type = "FRAME"
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format = "rgb24")):
                        container.mux(packet)
            if container is not None and self.valid and not self._failed:
                for packet in stream.encode():
                    container.mux(packet)
        except Exception:  # noqa: BLE001 - drain so the render thread never blocks on a full queue
            self._failed = True
            while self._queue.get() is not _STOP:
                pass
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:  # noqa: BLE001
                    self._failed = True

    def _join(self) -> None:
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join()
            self._thread = None

    def _matches(self, video_frames: Any) -> bool:
        torch = self._torch
        if not isinstance(video_frames, torch.Tensor) or video_frames.dtype != torch.uint8:
            self.invalidate("final frames are not uint8")
            return False
        if not self.chunks:
            self.invalidate("no chunks")
            return False
        for _, done in self.chunks:
            if done is not None:
                done.synchronize()
        streamed = [host for host, _ in self.chunks]
        if sum(len(h) for h in streamed) != len(video_frames):
            self.invalidate("frame count differs")
            return False
        import numpy as np

        final = video_frames.numpy()
        start = 0
        for host in streamed:
            # numpy, not torch.equal: torch's parallel compare costs seconds on a loaded many-core host.
            if not np.array_equal(host.numpy(), final[start : start + len(host)]):
                self.invalidate("frames differ")
                return False
            start += len(host)
        return True

    def take(self, video_frames: Any, fps: int) -> Optional[bytes]:
        """The streamed mp4 when its frames are exactly ``video_frames`` at ``fps``; None (caller encodes) otherwise."""
        try:
            if self.valid and int(fps) != self.fps:
                self.invalidate("fps differs")
            if self.valid and not self._decoded:
                self.invalidate("no decode")
            if self.valid and self._matches(video_frames):
                self._join()
                if not self._failed and self.valid:
                    data = Path(self.path).read_bytes()
                    count = _mp4_frame_count(data, video_frames.shape[1:3])
                    if count == len(video_frames):
                        _note("streamed", f"{len(video_frames)} frames", self.logger)
                        return data
                    self.invalidate(f"streamed file holds {count} frames")
                else:
                    self.invalidate(self.reason or "encoder failed")
            _note("fallback", self.reason or "unknown", self.logger)
            return None
        finally:
            self.close()

    def close(self) -> None:
        self.invalidate(self.reason or "closed")
        self._join()
        self.chunks = []
        try:
            os.unlink(self.path)
        except OSError:
            pass


def _mp4_frame_count(data: bytes, size: Any) -> Optional[int]:
    """Video packets in ``data`` when its video stream is ``size`` (H, W); None when it does not demux."""
    if not data:
        return None
    try:
        import io

        import av

        with av.open(io.BytesIO(data), mode = "r") as container:
            stream = container.streams.video[0]
            if (stream.codec_context.height, stream.codec_context.width) != tuple(int(x) for x in size):
                return None
            return sum(1 for packet in container.demux(stream) if packet.size)
    except Exception:  # noqa: BLE001 - an unreadable file is never used
        return None


_offered = threading.local()


@contextlib.contextmanager
def offer_stream(session: Optional[StreamEncode]) -> Iterator[None]:
    """Make ``session`` the mp4 source for ``VideoBackend._encode_mp4`` calls on this thread.

    The export stays the single entry point, so whatever wraps ``_encode_mp4`` still sees every clip."""
    previous = getattr(_offered, "session", None)
    _offered.session = session
    try:
        yield
    finally:
        _offered.session = previous


def claim_stream(video_frames: Any, fps: int) -> Optional[bytes]:
    """The offered session's mp4 for exactly these frames, once; None (encode normally) otherwise."""
    session = getattr(_offered, "session", None)
    if session is None:
        return None
    _offered.session = None
    return session.take(video_frames, fps)


def open_stream_encode(
    pipe: Any,
    fps: int,
    *,
    engage: bool,
    logger: Any = None,
) -> Optional[StreamEncode]:
    """A session for a resident CUDA ``AutoencoderKLWan`` with libx264 frame threads, or None."""
    if not engage or not stream_encode_enabled():
        return None
    from .video_encode import frame_threads_enabled
    from .video_nvenc import nvenc_requested

    if not frame_threads_enabled() or nvenc_requested():
        return None
    vae = getattr(pipe, "vae", None)
    processor = getattr(pipe, "video_processor", None)
    if type(vae).__name__ != "AutoencoderKLWan" or not callable(
        getattr(processor, "postprocess", None)
    ):
        return None
    decoder = getattr(vae, "decoder", None)
    if not callable(getattr(decoder, "register_forward_hook", None)):
        return None
    try:
        import av  # noqa: F401
        import torch

        if not torch.cuda.is_available() or next(decoder.parameters()).device.type != "cuda":
            return None
        return StreamEncode(vae, processor, fps, logger)
    except Exception:  # noqa: BLE001 - no PyAV, no CUDA or no parameters: encode afterwards
        return None
