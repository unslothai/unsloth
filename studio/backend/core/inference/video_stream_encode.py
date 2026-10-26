# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Encode a Wan or MiniMax-H3 clip while its VAE is still decoding.

The streamed mp4 is used only when its frames equal the final frames bit for bit, else the clip is encoded as
before. H3 declares its audio stream up front as ``encode_x264`` does and muxes the soundtrack at the end.
``UNSLOTH_STUDIO_VIDEO_STREAM_ENCODE=0`` turns it off.
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

from .host_pinning import host_pinning_allowed

STREAM_ENCODE_ENV = "UNSLOTH_STUDIO_VIDEO_STREAM_ENCODE"
# Kill switch: "0" hides the decode below the session's wrapper from chain walkers again (as before).
STREAM_CHAIN_ENV = "UNSLOTH_STUDIO_VIDEO_STREAM_ENCODE_CHAIN"
QUEUE_DEPTH = 8
_STOP = object()

STATS = {"streamed": 0, "fallback": 0, "last": ""}


def stream_encode_enabled() -> bool:
    return os.environ.get(STREAM_ENCODE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _expose_inner(wrapper: Any, original: Any) -> Any:
    """Set ``__wrapped__`` so diffusion_vae_oom's chain walkers see the untiled gate below this wrapper."""
    if os.environ.get(STREAM_CHAIN_ENV, "").strip().lower() not in ("0", "false", "no", "off"):
        wrapper.__wrapped__ = original
    return wrapper


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
        *,
        audio_sample_rate: Optional[int] = None,
    ) -> None:
        import torch

        self._torch = torch
        # the soundtrack's rate, declared before the first frame like encode_x264; None = a video-only file
        self.audio_sample_rate = int(audio_sample_rate) if audio_sample_rate else None
        # the last frames may come from the final clip at take() (H3 finishes its tail only when the decode ends)
        self._tail_ok = False
        self._container = None
        self._audio_stream = None
        self.vae = vae
        self.processor = processor
        self.fps = int(fps)
        self.logger = logger
        patch = getattr(getattr(vae, "config", None), "patch_size", None)
        self._unpatchify = (
            getattr(sys.modules.get(type(vae).__module__), "unpatchify", None) if patch else None
        )
        self._patch = patch
        # ROCm's bf16 decode (diffusion_device.install_rocm_vae_bf16_decode) runs the decoder in bf16 and hands the
        # pipeline the fp32 widening of its output, so the pipeline postprocesses fp32: the chunks must be widened the
        # same way or no frame ever matches. Read off the VAE, not ``vae.decode``: the untiled-when-fits decode (and
        # the OOM scope) wrap it from outside. NVIDIA's fp16 decode returns fp16 as is, which the chunks already are.
        self._widen = bool(getattr(vae, "_unsloth_bf16_decode_mode", None)) or bool(
            getattr(getattr(vae, "decode", None), "_unsloth_bf16_decode", False)
        )
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
            if self._widen and x.is_floating_point() and x.dtype != torch.float32:
                x = x.float()
            x = torch.clamp(x, min = -1.0, max = 1.0)
            x = self.processor.postprocess(x[0].permute(1, 0, 2, 3), "pt")
            self._push_uint8(x)
        except Exception as exc:  # noqa: BLE001 - streaming is an optimisation; the normal encode still runs
            self.invalidate(f"chunk failed: {type(exc).__name__}")

    def _push_uint8(self, x: Any) -> None:
        """(F, C, H, W) in [0, 1] -> uint8 (F, H, W, C) with the export's exact ops."""
        torch = self._torch
        x = (x.to(torch.float32) * 255).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        device = x.device
        done = None
        if device.type == "cuda":
            if self._side is None:
                self._side = torch.cuda.Stream(device = device)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            host = torch.empty(x.shape, dtype = torch.uint8, pin_memory = host_pinning_allowed(device))
            with torch.cuda.stream(self._side):
                self._side.wait_event(ready)
                host.copy_(x, non_blocking = True)
                done = torch.cuda.Event()
                done.record(self._side)
            x.record_stream(self._side)
        else:
            host = x.cpu() if device.type != "cpu" else x
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

            return _expose_inner(decode, original)

        def _tiled_wrapper(original: Any) -> Any:
            def tiled_decode(*args: Any, **kwargs: Any) -> Any:
                self.invalidate("tiled decode")
                return original(*args, **kwargs)

            return tiled_decode

        handle = None
        try:
            _wrap("decode", _decode_wrapper)
            _wrap("tiled_decode", _tiled_wrapper)
            # Disabled for Dynamo: under a compiled decode (UNSLOTH_DIFFUSION_COMPILE_VAE=1) the hook's thread, queue and
            # CUDA stream code broke the graph at each of them; as one eager island the chunks still stream.
            handle = vae.decoder.register_forward_hook(
                self._torch.compiler.disable(self._on_chunk), with_kwargs = True
            )
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
        stopped = False
        try:
            import av
            while True:
                item = self._queue.get()
                if item is _STOP:
                    stopped = True
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
                    if self.audio_sample_rate is not None:
                        from diffusers.utils.export_utils import _prepare_audio_stream
                        self._audio_stream = _prepare_audio_stream(
                            container, self.audio_sample_rate
                        )
                for frame in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format = "rgb24")):
                        container.mux(packet)
            if container is not None and self.valid and not self._failed:
                for packet in stream.encode():
                    container.mux(packet)
        except Exception:  # noqa: BLE001 - drain so the render thread never blocks on a full queue
            self._failed = True
            # A failure in the final flush comes after _STOP was taken; waiting for another would hang _join.
            while not stopped and self._queue.get() is not _STOP:
                pass
        finally:
            if container is not None:
                if self._audio_stream is not None and not self._failed and self.valid:
                    self._container = container
                else:
                    try:
                        container.close()
                    except Exception:  # noqa: BLE001
                        self._failed = True

    def _join(self) -> None:
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join()
            self._thread = None

    def _close_container(self) -> None:
        container, self._container = self._container, None
        if container is not None:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                self._failed = True

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
        count = sum(len(h) for h in streamed)
        if count > len(video_frames) or (count != len(video_frames) and not self._tail_ok):
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

    def take(
        self,
        video_frames: Any,
        fps: int,
        audio: Any = None,
        audio_sample_rate: Optional[int] = None,
    ) -> Optional[bytes]:
        """The streamed mp4 when its frames are exactly ``video_frames`` at ``fps`` and its declared audio stream is
        ``audio`` at ``audio_sample_rate``; None (caller encodes) otherwise."""
        try:
            if self.valid and int(fps) != self.fps:
                self.invalidate("fps differs")
            if self.valid and (audio is None) != (self.audio_sample_rate is None):
                self.invalidate("audio differs")
            if (
                self.valid
                and audio is not None
                and (audio_sample_rate is None or int(audio_sample_rate) != self.audio_sample_rate)
            ):
                self.invalidate("audio rate differs")
            if self.valid and not self._decoded:
                self.invalidate("no decode")
            if self.valid and self._matches(video_frames):
                streamed = sum(len(host) for host, _ in self.chunks)
                if streamed < len(video_frames):
                    self._queue.put((video_frames[streamed:], None))
                self._join()
                if audio is not None and not self._failed and self.valid:
                    try:
                        import av
                        from diffusers.utils.export_utils import _write_audio

                        if self._container is None or self._audio_stream is None:
                            raise RuntimeError("no open container")
                        _write_audio(
                            self._container, self._audio_stream, audio, self.audio_sample_rate, av
                        )
                    except Exception as exc:  # noqa: BLE001 - the stock exporter writes the clip instead
                        self.invalidate(f"audio mux failed: {type(exc).__name__}")
                    self._close_container()
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
        self._close_container()
        self.chunks = []
        try:
            os.unlink(self.path)
        except OSError:
            pass


class H3StreamEncode(StreamEncode):
    """Passive ``_decode_clip`` hook replaying H3's clip split and ``_blend`` on the side; decode output untouched."""

    def __init__(
        self,
        vae: Any,
        processor: Any,
        fps: int,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        pixel_mean = kwargs.pop("pixel_mean")
        pixel_std = kwargs.pop("pixel_std")
        super().__init__(vae, processor, fps, logger, **kwargs)
        self._tail_ok = True
        self._pixel = (tuple(float(v) for v in pixel_mean), tuple(float(v) for v in pixel_std))
        self._overlap = None
        self._clips = 0

    def _pixel_tensors(self, device: Any) -> tuple:
        """Built once: a pageable ``torch.tensor(..., device=)`` per clip would block the host on the GPU."""
        torch = self._torch
        held = getattr(self, "_pixel_on", None)
        if held is None or held[0] != device:
            pair = []
            for values in self._pixel:
                host = torch.tensor(values, dtype = torch.float32)
                if device.type == "cuda" and host_pinning_allowed(device):
                    host = host.pin_memory()
                pair.append(host.to(device, non_blocking = True).view(1, -1, 1, 1, 1))
            held = (device, tuple(pair))
            self._pixel_on = held
        return held[1]

    def _on_clip(self, clip: Any) -> None:
        if not self.valid or not self._decoding:
            return
        torch = self._torch
        try:
            if self._failed:
                self.invalidate("encoder failed")
                return
            if not torch.is_tensor(clip) or clip.ndim != 5 or clip.shape[0] != 1:
                self.invalidate("unexpected decoder output")
                return
            vae = self.vae
            chunk_frames = int(vae.tokens_chunk_size) * int(vae.temporal_compression_ratio)
            splits = int(int(vae.config.token_drop) > 0) + 1
            pre = int(vae.frame_pre_padding)
            mean, std = self._pixel_tensors(clip.device)
            self._clips += 1
            for j in range(splits):
                chunk = clip[:, :, j * chunk_frames : (j + 1) * chunk_frames][:, :, pre:]
                if j:
                    self._overlap = chunk
                    continue
                if self._overlap is not None:
                    chunk = vae._blend(self._overlap, chunk, vae.frame_overlap, dim = -3)
                    self._overlap = None
                # MiniMaxH3VideoDecodeStep's denormalise, then the uint8 export's postprocess
                x = (chunk.float() * std + mean).clamp(0, 1)
                self._push_uint8(self.processor.postprocess(x[0].permute(1, 0, 2, 3), "pt"))
        except Exception as exc:  # noqa: BLE001 - streaming is an optimisation; the normal encode still runs
            self.invalidate(f"chunk failed: {type(exc).__name__}")

    @contextlib.contextmanager
    def hooks(self) -> Iterator[None]:
        """Arm the clip hook for the outermost ``vae.decode``; a second ``_decode`` in that window disarms it."""
        vae = self.vae
        restore: list = []
        state = {"inner": 0}

        def _wrap(name: str, make: Any) -> None:
            original = getattr(vae, name, None)
            if not callable(original):
                raise AttributeError(name)
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

            return _expose_inner(decode, original)

        def _inner_wrapper(original: Any) -> Any:
            def _decode(*args: Any, **kwargs: Any) -> Any:
                if self._decoding:
                    state["inner"] += 1
                    if state["inner"] > 1:
                        # an OOM retry or a sliced batch: the clips so far belong to another pass
                        self.invalidate("decoder restarted")
                return original(*args, **kwargs)

            return _decode

        # one eager island under a compiled decode, as the Wan hook is
        on_clip = self._torch.compiler.disable(self._on_clip)

        def _clip_wrapper(original: Any) -> Any:
            def _decode_clip(*args: Any, **kwargs: Any) -> Any:
                out = original(*args, **kwargs)
                on_clip(out)
                return out

            return _decode_clip

        try:
            _wrap("decode", _decode_wrapper)
            _wrap("_decode", _inner_wrapper)
            _wrap("_decode_clip", _clip_wrapper)
        except Exception:  # noqa: BLE001 - a VAE without the clip loop just encodes afterwards
            self.invalidate("hooks refused")
        try:
            yield
        except BaseException:
            self.close()
            raise
        finally:
            for name, original, had_own in reversed(restore):
                try:
                    if had_own:
                        setattr(vae, name, original)
                    else:
                        delattr(vae, name)
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    pass


def _mp4_frame_count(data: bytes, size: Any) -> Optional[int]:
    """Video packets in ``data`` when its video stream is ``size`` (H, W); None when it does not demux."""
    if not data:
        return None
    try:
        import io

        import av
        with av.open(io.BytesIO(data), mode = "r", format = "mp4") as container:
            stream = container.streams.video[0]
            if (stream.codec_context.height, stream.codec_context.width) != tuple(
                int(x) for x in size
            ):
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


def claim_stream(
    video_frames: Any,
    fps: int,
    audio: Any = None,
    audio_sample_rate: Optional[int] = None,
) -> Optional[bytes]:
    """The offered session's mp4 for exactly these frames (and soundtrack), once; None (encode normally) otherwise."""
    session = getattr(_offered, "session", None)
    if session is None:
        return None
    _offered.session = None
    if audio is None:
        return session.take(video_frames, fps)
    return session.take(video_frames, fps, audio, audio_sample_rate)


def _decodes_on_cuda(pipe: Any, decoder: Any, torch: Any) -> bool:
    """True if resident on CUDA, or offloaded to host while the pipeline executes on CUDA."""
    if next(decoder.parameters()).device.type == "cuda":
        return True
    try:
        device = torch.device(getattr(pipe, "_execution_device", None) or "cpu")
    except Exception:  # noqa: BLE001 - no execution device: a CPU render
        return False
    return device.type == "cuda"


def open_stream_encode(
    pipe: Any,
    fps: int,
    *,
    engage: bool,
    logger: Any = None,
) -> Optional[StreamEncode]:
    """A session for an ``AutoencoderKLWan`` decoding on CUDA with libx264 frame threads, or None."""
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

        if not torch.cuda.is_available():
            return None
        if not _decodes_on_cuda(pipe, decoder, torch):
            return None
        return StreamEncode(vae, processor, fps, logger)
    except Exception:  # noqa: BLE001 - no PyAV, no CUDA or no parameters: encode afterwards
        return None


_H3_CLIP_ATTRS = (
    "tokens_chunk_size",
    "temporal_compression_ratio",
    "frame_pre_padding",
    "frame_overlap",
    "_blend",
    "_decode",
    "_decode_clip",
)


def open_h3_stream_encode(
    pipe: Any,
    fps: int,
    *,
    engage: bool,
    logger: Any = None,
) -> Optional[StreamEncode]:
    """A session for a resident CUDA ``AutoencoderKLMiniMaxH3`` whose soundtrack is muxed at the end, or None."""
    if not engage or not stream_encode_enabled():
        return None
    from .video_encode import frame_threads_enabled
    from .video_nvenc import nvenc_requested

    if not frame_threads_enabled() or nvenc_requested():
        return None
    vae = getattr(pipe, "vae", None)
    processor = getattr(pipe, "video_processor", None)
    if type(vae).__name__ != "AutoencoderKLMiniMaxH3" or not callable(
        getattr(processor, "postprocess", None)
    ):
        return None
    if not all(hasattr(vae, name) for name in _H3_CLIP_ATTRS):
        return None
    try:
        import av  # noqa: F401
        import torch
        from diffusers.utils.export_utils import _prepare_audio_stream, _write_audio  # noqa: F401

        if getattr(vae, "use_slicing", False) or not torch.cuda.is_available():
            return None
        # a VAE parked on the host between renders (pinned swap) still decodes on the execution device
        device = next(vae.decoder.parameters()).device
        if device.type != "cuda":
            device = torch.device(getattr(pipe, "_execution_device", None) or "cpu")
        if device.type != "cuda":
            return None
        rate = getattr(pipe, "audio_sampling_rate", None)
        return H3StreamEncode(
            vae,
            processor,
            fps,
            logger,
            audio_sample_rate = int(rate) if rate else None,
            pixel_mean = pipe.pixel_mean,
            pixel_std = pipe.pixel_std,
        )
    except Exception:  # noqa: BLE001 - no PyAV, no CUDA or no parameters: encode afterwards
        return None
