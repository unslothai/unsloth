# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restyle a video with Cosmos3-Nano-Sim-Transfer on a resident sd-server: frames in, Canny edge
control frames out to ``vid_gen``, one restyled clip back. Import-light: numpy, PIL and av load lazily."""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from core.inference.sd_cpp_args import SdCppModelFiles
from loggers import get_logger

logger = get_logger(__name__)

MODEL_ID = "nvidia/Cosmos3-Nano-Sim-Transfer"
# What the model was trained for (docs/cosmos3.md in the sd.cpp port).
WIDTH = 832
HEIGHT = 480
FPS = 30
MAX_FRAMES = 121
MIN_FRAMES = 5
# NVIDIA's reference control is cv2.Canny(rgb, 100, 200) on the resized RGB frames.
CANNY_LOW = 100
CANNY_HIGH = 200
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

MODEL_ENV = "UNSLOTH_COSMOS3_TRANSFER_MODEL"
VAE_ENV = "UNSLOTH_COSMOS3_TRANSFER_VAE"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_LATENT_RE = re.compile(r"cosmos3: latent frame (\d+)/(\d+)")
_DECODE_RE = re.compile(r"cosmos3: \d+ latent frames in")

_TERMINAL = ("done", "error", "cancelled")


class RestyleError(ValueError):
    """A request the restyle workflow cannot run; the message is shown to the user."""


class RestyleBusy(RuntimeError):
    pass


def snap_frame_count(count: int) -> int:
    """Largest ``4n + 1`` (Wan2.2 temporal compression) not above ``count`` or ``MAX_FRAMES``."""
    count = min(int(count), MAX_FRAMES)
    if count < MIN_FRAMES:
        raise RestyleError(f"Need at least {MIN_FRAMES} frames, got {count}.")
    return ((count - 1) // 4) * 4 + 1


def model_files(env: Optional[dict[str, str]] = None) -> SdCppModelFiles:
    env = os.environ if env is None else env
    model = (env.get(MODEL_ENV) or "").strip()
    vae = (env.get(VAE_ENV) or "").strip()
    missing = [name for name, path in ((MODEL_ENV, model), (VAE_ENV, vae)) if not path or not Path(path).is_file()]
    if missing:
        raise RestyleError(
            "Cosmos3 transfer weights not found. Set "
            + " and ".join(missing)
            + " to the merged transformer and the Wan2.2 VAE (.safetensors or .gguf)."
        )
    return SdCppModelFiles(diffusion_model = model, vae = vae)


def fit_frame(image: Any, width: int = WIDTH, height: int = HEIGHT) -> Any:
    """Scale to cover ``width`` x ``height`` and centre-crop, so the clip is never stretched."""
    from PIL import Image

    image = image.convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, round(image.width * scale)), max(height, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def frames_from_video(data: bytes, limit: int = MAX_FRAMES, fps: int = FPS) -> list[Any]:
    """Frames at ``fps`` by presentation time (held when the source is slower), up to ``limit``."""
    import av

    out: list[Any] = []
    try:
        with av.open(io.BytesIO(data)) as container:
            if not container.streams.video:
                raise RestyleError("The file has no video track.")
            stream = container.streams.video[0]
            rate = float(stream.average_rate or stream.guessed_rate or fps)
            first: Optional[float] = None
            prev = None
            for index, frame in enumerate(container.decode(stream)):
                t = frame.time if frame.time is not None else index / rate
                if first is None:
                    first = t
                t -= first
                while prev is not None and len(out) < limit and t > len(out) / fps + 1e-6:
                    out.append(prev)
                if len(out) >= limit:
                    break
                prev = frame.to_image()
            if prev is not None and len(out) < limit:
                out.append(prev)
    except RestyleError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a bad upload
        raise RestyleError(f"Could not read the video: {exc}") from exc
    return out


def frames_from_images(items: list[tuple[str, bytes]], limit: int = MAX_FRAMES) -> list[Any]:
    """A folder of frames, in file-name order like sd-cli's ``--control-video``."""
    from PIL import Image

    named = sorted(
        ((name, data) for name, data in items if name.lower().endswith(IMAGE_SUFFIXES)),
        key = lambda item: item[0],
    )
    out = []
    for name, data in named[:limit]:
        try:
            with Image.open(io.BytesIO(data)) as image:
                out.append(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise RestyleError(f"Could not read frame {Path(name).name}: {exc}") from exc
    return out


def canny_edges(rgb: Any, low: int = CANNY_LOW, high: int = CANNY_HIGH) -> Any:
    """``cv2.Canny(rgb, low, high)`` in numpy (3x3 Sobel, L1 magnitude, strongest channel), since
    OpenCV is not a Studio dependency. Returns a uint8 map of 0 / 255."""
    import numpy as np

    img = np.asarray(rgb, dtype = np.int32)
    if img.ndim == 2:
        img = img[..., None]
    h, w = img.shape[:2]
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode = "edge")
    dx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    dy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.abs(dx) + np.abs(dy)
    pick = mag.argmax(axis = -1)[..., None]
    dx = np.take_along_axis(dx, pick, -1)[..., 0].astype(np.int64)
    dy = np.take_along_axis(dy, pick, -1)[..., 0].astype(np.int64)
    mag = np.take_along_axis(mag, pick, -1)[..., 0]

    m = np.pad(mag, 1)
    c = m[1:-1, 1:-1]
    ax = np.abs(dx)
    ay = np.abs(dy) << 15
    tg22 = ax * 13573  # round(tan(22.5 deg) * 2**15), as in OpenCV
    tg67 = tg22 + (ax << 16)
    horizontal = ay < tg22
    vertical = ay > tg67
    same_sign = (dx ^ dy) >= 0
    keep = np.where(
        horizontal,
        (c > m[1:-1, :-2]) & (c >= m[1:-1, 2:]),
        np.where(
            vertical,
            (c > m[:-2, 1:-1]) & (c >= m[2:, 1:-1]),
            np.where(
                same_sign,
                (c > m[:-2, :-2]) & (c > m[2:, 2:]),
                (c > m[:-2, 2:]) & (c > m[2:, :-2]),
            ),
        ),
    )
    weak = keep & (c > low)

    # Hysteresis: grow from strong pixels through 8-connected weak ones.
    pw = w + 2
    weak_flat = np.zeros((h + 2) * pw, dtype = bool)
    weak_flat.reshape(h + 2, pw)[1:-1, 1:-1] = weak
    edge = np.zeros_like(weak_flat)
    frontier = np.flatnonzero(np.pad(weak & (c > high), 1))
    edge[frontier] = True
    offsets = np.array([-pw - 1, -pw, -pw + 1, -1, 1, pw - 1, pw, pw + 1])
    while frontier.size:
        nb = (frontier[:, None] + offsets).ravel()
        nb = np.unique(nb[weak_flat[nb] & ~edge[nb]])
        edge[nb] = True
        frontier = nb
    return edge.reshape(h + 2, pw)[1:-1, 1:-1].astype(np.uint8) * 255


def _png_b64(array: Any) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format = "PNG", compress_level = 1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def control_frames_b64(frames: list[Any]) -> list[str]:
    """Each fitted RGB frame as a 3-channel Canny PNG, base64."""
    import numpy as np

    out = []
    for frame in frames:
        edges = canny_edges(np.asarray(frame))
        out.append(_png_b64(np.repeat(edges[..., None], 3, axis = -1)))
    return out


def build_restyle_request(
    *,
    prompt: str,
    control_frames: list[str],
    seed: int,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> dict:
    """POST body for sd-server ``/sdcpp/v1/vid_gen``; the sampler is fixed by the model, so no steps or CFG."""
    if not (prompt or "").strip():
        raise RestyleError("A prompt is required.")
    frames = snap_frame_count(len(control_frames))
    return {
        "prompt": prompt.strip(),
        "width": int(width),
        "height": int(height),
        "video_frames": frames,
        "fps": int(fps),
        "seed": int(seed),
        "control_frames": list(control_frames[:frames]),
        # MJPG AVI needs no WebM build.
        "output_format": "avi",
        "output_compression": 90,
    }


def parse_progress(line: str) -> Optional[tuple[int, int]]:
    match = _LATENT_RE.search(line)
    return (int(match.group(1)), int(match.group(2))) if match else None


def encode_mp4(frames: list[Any], fps: int = FPS) -> bytes:
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as dst:
        stream = dst.add_stream("libx264", rate = fps)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(frame), format = "rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    return buf.getvalue()


class RestyleRunner:
    """One restyle at a time against a lazily started, resident sd-server."""

    def __init__(self, server_factory: Optional[Callable[[str], Any]] = None) -> None:
        self._lock = threading.Lock()
        self._server: Any = None
        self._server_files: Optional[SdCppModelFiles] = None
        self._server_factory = server_factory
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {"active": False, "phase": None}
        self._workdir: Optional[Path] = None

    def progress(self) -> dict[str, Any]:
        return dict(self._state)

    def _update(self, **fields: Any) -> None:
        self._state = {**self._state, **fields}

    def start(self, *, prompt: str, frames: list[Any], seed: Optional[int] = None) -> dict[str, Any]:
        """Validate and claim the runner; the caller runs ``run`` on a worker thread."""
        if not (prompt or "").strip():
            raise RestyleError("A prompt is required.")
        if not frames:
            raise RestyleError("No frames found in the upload.")
        count = snap_frame_count(len(frames))
        files = model_files()
        with self._lock:
            if self._state.get("active"):
                raise RestyleBusy("A restyle is already running.")
            self._cancel.clear()
            if self._workdir is not None:
                shutil.rmtree(self._workdir, ignore_errors = True)
            self._workdir = Path(tempfile.mkdtemp(prefix = "unsloth-restyle-"))
            job_id = uuid.uuid4().hex
            self._state = {
                "active": True,
                "id": job_id,
                "phase": "edges",
                "step": 0,
                "total": (count - 1) // 4 + 1,
                "num_frames": count,
                "error": None,
                "video": None,
                "started_at": time.time(),
            }
        return {
            "id": job_id,
            "prompt": prompt.strip(),
            "frames": [fit_frame(f) for f in frames[:count]],
            "seed": int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF,
            "files": files,
        }

    def media_path(self, job_id: str, kind: str) -> Optional[Path]:
        if kind not in ("input", "control", "output") or self._state.get("id") != job_id:
            return None
        path = self._workdir / f"{kind}.mp4" if self._workdir is not None else None
        return path if path is not None and path.is_file() else None

    def cancel(self) -> bool:
        if not self._state.get("active"):
            return False
        self._cancel.set()
        return True

    def unload(self) -> None:
        with self._lock:
            server, self._server, self._server_files = self._server, None, None
        if server is not None:
            server.stop()

    def _ensure_server(self, files: SdCppModelFiles) -> Any:
        if self._server is not None and self._server_files == files and self._server.is_alive():
            return self._server
        self.unload()
        if self._server_factory is not None:
            server = self._server_factory(files.diffusion_model)
        else:
            from core.inference.sd_cpp_engine import find_sd_server_binary
            from core.inference.sd_cpp_server import SdCppServer

            binary = find_sd_server_binary()
            if not binary:
                raise RestyleError(
                    "No sd-server binary found. Point SD_SERVER_PATH at a stable-diffusion.cpp "
                    "build with Cosmos3 support."
                )
            server = SdCppServer(binary)
        self._update(phase = "loading")
        server.start(files)
        self._server, self._server_files = server, files
        return server

    def run(self, job: dict[str, Any], save: Optional[Callable[[bytes, dict], Any]] = None) -> None:
        """Worker body; every outcome lands in the progress state, nothing is raised."""
        from core.inference.sd_cpp_engine import SdCppCancelled
        from core.inference.video_minimax_h3 import transcode_video_to_mp4

        workdir = self._workdir
        frames = job["frames"]
        try:
            assert workdir is not None
            (workdir / "input.mp4").write_bytes(encode_mp4(frames))
            control = control_frames_b64(frames)
            if self._cancel.is_set():
                raise SdCppCancelled("cancelled")
            payload = build_restyle_request(prompt = job["prompt"], control_frames = control, seed = job["seed"])
            server = self._ensure_server(job["files"])
            self._update(phase = "generate")
            started = time.monotonic()

            def on_line(line: str) -> None:
                hit = parse_progress(line)
                if hit is not None:
                    done, total = hit
                    elapsed = time.monotonic() - started
                    self._update(step = done, total = total, eta_seconds = elapsed / done * (total - done))
                elif _DECODE_RE.search(line):
                    self._update(phase = "decode", eta_seconds = None)

            avi = server.vid_gen(payload, on_step = on_line, cancel_event = self._cancel)
            self._update(phase = "export", eta_seconds = None)
            raw = workdir / "output.avi"
            raw.write_bytes(avi)
            mp4 = transcode_video_to_mp4(raw, fps = FPS)
            raw.unlink(missing_ok = True)
            (workdir / "output.mp4").write_bytes(mp4)
            record = None
            if save is not None:
                count = payload["video_frames"]
                record = save(
                    mp4,
                    {
                        "prompt": job["prompt"],
                        "negative_prompt": None,
                        "width": WIDTH,
                        "height": HEIGHT,
                        "num_frames": count,
                        "fps": FPS,
                        "duration_s": count / float(FPS),
                        "steps": 4,
                        "guidance": None,
                        "seed": job["seed"],
                        "has_audio": False,
                        "conditioning": "restyle",
                        "model": MODEL_ID,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            self._update(active = False, phase = "done", video = record)
        except SdCppCancelled:
            self._update(active = False, phase = "cancelled")
        except Exception as exc:  # noqa: BLE001 -- surfaced through progress
            if self._cancel.is_set():
                self._update(active = False, phase = "cancelled")
            else:
                logger.warning("restyle failed: %s", exc)
                self._update(active = False, phase = "error", error = str(exc))


runner = RestyleRunner()
