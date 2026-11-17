# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Nano-Sim-Transfer video restyle on a resident sd-server. Import-light: numpy, PIL and av load lazily."""

from __future__ import annotations

import base64
import contextlib
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
WIDTH = 832
HEIGHT = 480
FPS = 30
MAX_FRAMES = 121
MIN_FRAMES = 5
# NVIDIA's reference control is cv2.Canny(rgb, 100, 200) on the resized RGB frames.
CANNY_LOW = 100
CANNY_HIGH = 200

MODEL_ENV = "UNSLOTH_COSMOS3_TRANSFER_MODEL"
VAE_ENV = "UNSLOTH_COSMOS3_TRANSFER_VAE"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_LATENT_RE = re.compile(r"cosmos3: latent frame (\d+)/(\d+)")
_DECODE_RE = re.compile(r"cosmos3: \d+ latent frames in")

_TERMINAL = ("done", "error", "cancelled")
MEDIA_KINDS = ("input", "control", "output")
# No protocols: a concat or HLS playlist upload would otherwise open the files it names.
_NO_PROTOCOLS = {"protocol_whitelist": "none"}


class RestyleError(ValueError):
    """Shown to the user."""


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
    missing = [
        name
        for name, path in ((MODEL_ENV, model), (VAE_ENV, vae))
        if not path or not Path(path).is_file()
    ]
    if missing:
        raise RestyleError(
            "Cosmos3 transfer weights not found. Set "
            + " and ".join(missing)
            + " to the merged transformer and the Wan2.2 VAE (.safetensors or .gguf)."
        )
    return SdCppModelFiles(diffusion_model = model, vae = vae)


def fit_frame(
    image: Any,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Any:
    """Scale to cover ``width`` x ``height`` and centre-crop, so the clip is never stretched."""
    from PIL import Image

    image = image.convert("RGB")
    if image.size == (width, height):
        return image
    # Resample only the centred source box: scaling the whole frame first can allocate far past the output.
    scale = max(width / image.width, height / image.height)
    box_w, box_h = width / scale, height / scale
    left, top = (image.width - box_w) / 2, (image.height - box_h) / 2
    return image.resize(
        (width, height), Image.Resampling.BICUBIC, box = (left, top, left + box_w, top + box_h)
    )


def frames_from_video(
    data: bytes,
    limit: int = MAX_FRAMES,
    fps: int = FPS,
) -> list[Any]:
    """Frames at ``fps`` by presentation time (held when the source is slower), up to ``limit``."""
    import av
    from PIL import Image

    from core.training.diffusion_h3_clips import apply_display_rotation, display_rotation_degrees

    out: list[Any] = []
    try:
        with av.open(io.BytesIO(data), options = _NO_PROTOCOLS) as container:
            if not container.streams.video:
                raise RestyleError("The file has no video track.")
            stream = container.streams.video[0]
            rate = float(stream.average_rate or stream.guessed_rate or fps)
            first: Optional[float] = None
            last = 0.0
            prev = None
            fitted = None

            def keep() -> Any:
                # Fit only frames that are kept (a 120 fps source drops 3 in 4), and as they are kept:
                # 121 full-size 4K frames would hold about 3 GB.
                nonlocal fitted
                if fitted is None:
                    theta = display_rotation_degrees(prev, stream)
                    fitted = fit_frame(apply_display_rotation(prev.to_image(), theta, Image))
                return fitted

            for index, frame in enumerate(container.decode(stream)):
                t = frame.time if frame.time is not None else index / rate
                if first is None:
                    first = t
                t -= first
                while prev is not None and len(out) < limit and t > len(out) / fps + 1e-6:
                    out.append(keep())
                if len(out) >= limit:
                    break
                prev, fitted, last = frame, None, t
            # The final frame is held for its own interval too, so a slow source keeps its last scene.
            while (
                prev is not None
                and len(out) < limit
                and (not out or last + 1 / rate > len(out) / fps + 1e-6)
            ):
                out.append(keep())
    except RestyleError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a bad upload
        logger.info("restyle: could not decode upload: %s", exc)
        raise RestyleError(
            "Could not read this video. Try an MP4 (H.264), WebM or MOV file, or a folder of frames."
        ) from exc
    if not out:
        raise RestyleError("The video has no frames.")
    return out


def video_seconds(data: bytes) -> Optional[float]:
    import av
    try:
        with av.open(io.BytesIO(data), options = _NO_PROTOCOLS) as container:
            if container.duration:
                return container.duration / 1_000_000
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except Exception:  # noqa: BLE001
        return None
    return None


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
                out.append(fit_frame(image))
        except Exception as exc:  # noqa: BLE001
            raise RestyleError(f"Could not read frame {Path(name).name}: {exc}") from exc
    return out


def canny_edges(
    rgb: Any,
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
) -> Any:
    """``cv2.Canny(rgb, low, high)`` in numpy (3x3 Sobel, L1, strongest channel): OpenCV is not a Studio dependency."""
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


def control_frames(frames: list[Any]) -> list[Any]:
    import numpy as np
    return [np.repeat(canny_edges(np.asarray(f))[..., None], 3, axis = -1) for f in frames]


def control_frames_b64(frames: list[Any]) -> list[str]:
    return [_png_b64(edges) for edges in control_frames(frames)]


def _jpeg_b64(array: Any) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format = "JPEG", quality = 85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def preview(frames: list[Any], source_seconds: Optional[float] = None) -> dict[str, Any]:
    import numpy as np

    if not frames:
        raise RestyleError("No frames found in the upload.")
    count = snap_frame_count(len(frames))
    frame = np.asarray(fit_frame(frames[count // 2]))
    return {
        "frames": count,
        "seconds": count / float(FPS),
        "source_seconds": source_seconds,
        "width": WIDTH,
        "height": HEIGHT,
        "frame_jpeg": _jpeg_b64(frame),
        "edges_jpeg": _jpeg_b64(control_frames([frame])[0]),
    }


# The progress line ``parse_progress`` reads; only an sd.cpp build with the Cosmos3 port prints it.
_COSMOS3_MARKER = b"cosmos3: latent frame"
_MARKER_CACHE: dict[tuple[str, int, int], bool] = {}


def _file_has_marker(path: Path) -> bool:
    import mmap
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in _MARKER_CACHE:
            with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as m:
                _MARKER_CACHE[key] = m.find(_COSMOS3_MARKER) >= 0
        return _MARKER_CACHE[key]
    except (OSError, ValueError):
        return False


def binary_supports_cosmos3(binary: str) -> bool:
    """The sd-server binary, or a stable-diffusion library next to it, carries the Cosmos3 port."""
    path = Path(binary)
    candidates = [
        path,
        *sorted(path.parent.glob("*stable-diffusion*")),
        *sorted(path.parent.glob("libsd*")),
    ]
    return any(c.is_file() and _file_has_marker(c) for c in candidates)


def availability(env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    try:
        model_files(env)
    except RestyleError:
        return {
            "available": False,
            "reason": "The Cosmos3-Nano-Sim-Transfer weights are not set up on this machine.",
        }
    from core.inference.sd_cpp_engine import find_sd_server_binary

    binary = find_sd_server_binary()
    if not binary or not binary_supports_cosmos3(binary):
        return {
            "available": False,
            "reason": "No stable-diffusion.cpp server with Cosmos3 support is installed.",
        }
    return {"available": True, "reason": None}


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
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(np.asarray(frame), format = "rgb24")
            ):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    return buf.getvalue()


def validate_request(prompt: str, frames: list[Any]) -> tuple[int, SdCppModelFiles]:
    """Everything a start can refuse with 400, checked before the route takes the GPU."""
    if not (prompt or "").strip():
        raise RestyleError("A prompt is required.")
    if not frames:
        raise RestyleError("No frames found in the upload.")
    return snap_frame_count(len(frames)), model_files()


class RestyleRunner:
    def __init__(self, server_factory: Optional[Callable[[str], Any]] = None) -> None:
        self._lock = threading.Lock()
        self._server: Any = None
        self._server_files: Optional[SdCppModelFiles] = None
        self._server_factory = server_factory
        self._cancel = threading.Event()
        self._state: dict[str, Any] = {"active": False, "phase": None}
        self._owner: Optional[str] = None
        self._workdir: Optional[Path] = None
        self._loading: Any = None
        self._stopping = 0
        self._stopped = threading.Condition(self._lock)
        # Set once the clip is being published; a cancel after that point is refused, not half-honoured.
        self._publishing = False
        # Held from before an upload is read until ``start``, so a second request is refused up front.
        self._reservation: Optional[str] = None
        self._reservation_owner: Optional[str] = None

    def owned_by(self, owner: Optional[str]) -> bool:
        return self._owner is not None and self._owner == owner

    def progress(self, owner: Optional[str] = None) -> dict[str, Any]:
        """Another account sees only whether the runner is busy, never the prompt's outputs."""
        with self._lock:
            state = self._state
            pending = self._reservation is not None and (
                owner is None or self._reservation_owner == owner
            )
            if owner is not None and not self.owned_by(owner):
                return {"active": bool(state.get("active")), "phase": None, "pending": pending}
            return {**state, "pending": pending}

    def reserve(self, owner: Optional[str] = None) -> str:
        with self._lock:
            if self._state.get("active") or self._reservation is not None:
                raise RestyleBusy("A restyle is already running.")
            self._reservation = uuid.uuid4().hex
            self._reservation_owner = owner
            return self._reservation

    def release(self, reservation: str) -> None:
        with self._lock:
            if self._reservation == reservation:
                self._reservation = None

    def _update(self, **fields: Any) -> None:
        self._state = {**self._state, **fields}

    def start(
        self,
        *,
        prompt: str,
        frames: list[Any],
        seed: Optional[int] = None,
        owner: Optional[str] = None,
        reservation: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate and claim the runner; the caller runs ``run`` on a worker thread."""
        count, files = validate_request(prompt, frames)
        seed = (
            int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
        )
        from state import active_generations

        with self._lock:
            if owner is not None and active_generations.fenced(owner):
                raise RestyleError("This account is being removed.")
            if self._state.get("active") or self._reservation not in (None, reservation):
                raise RestyleBusy("A restyle is already running.")
            # Allocated before anything changes hands: a failure here leaves the old job and owner intact.
            workdir = Path(tempfile.mkdtemp(prefix = "unsloth-restyle-"))
            job_id = uuid.uuid4().hex
            self._reservation = None
            self._cancel.clear()
            self._publishing = False
            self._owner = owner
            if self._workdir is not None:
                shutil.rmtree(self._workdir, ignore_errors = True)
            self._workdir = workdir
            self._state = {
                "active": True,
                "id": job_id,
                "phase": "edges",
                "step": 0,
                "total": (count - 1) // 4 + 1,
                "num_frames": count,
                "error": None,
                "video": None,
                "seed": seed,
                "started_at": time.time(),
            }
        return {
            "id": job_id,
            "prompt": prompt.strip(),
            "frames": [fit_frame(f) for f in frames[:count]],
            "seed": seed,
            "files": files,
        }

    def media_path(
        self,
        job_id: str,
        kind: str,
        owner: Optional[str] = None,
    ) -> Optional[Path]:
        with self._lock:
            if kind not in MEDIA_KINDS or self._state.get("id") != job_id:
                return None
            if owner is not None and not self.owned_by(owner):
                return None
            path = self._workdir / f"{kind}.mp4" if self._workdir is not None else None
        return path if path is not None and path.is_file() else None

    def cancel(self, owner: Optional[str] = None) -> bool:
        with self._lock:
            if (
                not self._state.get("active")
                or self._publishing
                or (owner is not None and not self.owned_by(owner))
            ):
                return False
            self._cancel.set()
            loading = self._loading
        if loading is not None:
            loading.stop()
        return True

    def retire(self, owner: str) -> bool:
        """Cancel ``owner``'s job for account retirement; True while it still holds the runner.
        A reservation of ``owner``'s is dropped; its request then meets the fence in ``start``."""
        with self._lock:
            if self._reservation is not None and self._reservation_owner == owner:
                self._reservation = None
            held = bool(self._state.get("active")) and self.owned_by(owner)
        if held:
            self.cancel(owner)
        return held

    def _finish(self, **fields: Any) -> None:
        # The server is detached in the same step that ends the job: a restyle that claims the runner
        # next can never reuse a server this worker is about to stop. Not kept resident either: an idle
        # ~30 GB server would sit outside every other loader's VRAM accounting (stop() counts in holds_gpu).
        with self._lock:
            server = self._detach_locked()
            self._update(active = False, **fields)
        self._stop(server)
        self._release_video_claim()

    def _release_video_claim(self) -> None:
        if self._server_factory is not None:
            return
        try:
            from core.inference import gpu_arbiter
            from core.inference.video import get_video_backend

            video = get_video_backend()

            def idle() -> bool:
                return not (
                    self.holds_gpu() or video.loading_repo_ids() or video.status().get("loaded")
                )

            gpu_arbiter.release_if(gpu_arbiter.VIDEO, idle)
        except Exception as exc:  # noqa: BLE001 -- the job's outcome is already recorded
            logger.warning("restyle: could not release the VIDEO claim: %s", exc)

    def unload(self) -> None:
        with self._lock:
            server = self._detach_locked()
        self._stop(server)

    def _detach_locked(self) -> Any:
        server, self._server, self._server_files = self._server, None, None
        if server is not None:
            self._stopping += 1
        return server

    def _stop(self, server: Any) -> None:
        # Counted until stop() returns, so the managed tree stays pinned (is_alive) while the process exits.
        if server is None:
            return
        try:
            server.stop()
        finally:
            with self._lock:
                self._stopping -= 1
                self._stopped.notify_all()

    def holds_gpu(self) -> bool:
        """A job runs, a request is reserved to start one, or a detached server is still exiting."""
        with self._lock:
            return (
                bool(self._state.get("active"))
                or self._reservation is not None
                or self._stopping > 0
            )

    def unload_if_idle(self, owner: Optional[str] = None) -> bool:
        """Detach the server under the same lock ``start`` claims, so a restyle can't begin in between.
        A reservation counts as busy: that request is about to reuse the server. With ``owner``, only
        a server whose last job was ``owner``'s is detached (RestyleBusy otherwise)."""
        with self._lock:
            if self._state.get("active") or self._reservation is not None:
                return False
            if owner is not None and self._server is not None and not self.owned_by(owner):
                raise RestyleBusy("Another account's restyle model is loaded.")
            server = self._detach_locked()
        self._stop(server)
        with self._lock:
            self._stopped.wait_for(lambda: self._stopping == 0, timeout = 120)
        return True

    def is_alive(self) -> bool:
        server = self._server
        return (
            (server is not None and server.is_alive())
            or self._loading is not None
            or self._stopping > 0
        )

    def _tree_lease(self) -> Any:
        if self._server_factory is not None:
            return contextlib.nullcontext()
        from core.inference.sd_cpp_backend import _tree_reader
        from core.inference.sd_cpp_engine import find_sd_server_binary

        return _tree_reader(find_sd_server_binary(), self._cancel, "cancelled")

    def _ensure_server(self, files: SdCppModelFiles) -> Any:
        if self._server is not None and self._server_files == files and self._server.is_alive():
            return self._server
        self.unload()
        # A server another caller detached may still hold its VRAM: never launch beside it.
        with self._lock:
            self._stopped.wait_for(lambda: self._stopping == 0, timeout = 120)
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
            from core.inference.sd_cpp_backend import register_tree_holder
            from core.inference.sd_cpp_engine import is_managed_binary

            if is_managed_binary(binary):
                # A resident server pins Studio's managed sd.cpp tree: an upgrade waits (is_alive).
                register_tree_holder(self)
        self._update(phase = "loading")
        self._loading = server
        try:
            if self._cancel.is_set():
                server.stop()
            server.start(files)
        finally:
            self._loading = None
        self._server, self._server_files = server, files
        return server

    def run(
        self,
        job: dict[str, Any],
        save: Optional[Callable[[bytes, dict], Any]] = None,
    ) -> None:
        """Worker body; every outcome lands in the progress state, nothing is raised."""
        from PIL import Image

        from core.inference.sd_cpp_engine import SdCppCancelled
        from core.inference.video_minimax_h3 import transcode_video_to_mp4

        workdir = self._workdir
        frames = job["frames"]
        lease = contextlib.ExitStack()
        try:
            assert workdir is not None
            (workdir / "input.mp4").write_bytes(encode_mp4(frames))
            edges = control_frames(frames)
            (workdir / "control.mp4").write_bytes(encode_mp4([Image.fromarray(e) for e in edges]))
            control = [_png_b64(e) for e in edges]
            if self._cancel.is_set():
                raise SdCppCancelled("cancelled")
            payload = build_restyle_request(
                prompt = job["prompt"], control_frames = control, seed = job["seed"]
            )
            lease.enter_context(self._tree_lease())
            server = self._ensure_server(job["files"])
            self._update(phase = "generate")
            first: list[float] = []

            def on_line(line: str) -> None:
                hit = parse_progress(line)
                if hit is not None:
                    done, total = hit
                    now = time.monotonic()
                    if not first:
                        first.append(now)
                    eta = (now - first[0]) / (done - 1) * (total - done) if done > 1 else None
                    self._update(step = done, total = total, eta_seconds = eta)
                elif _DECODE_RE.search(line):
                    self._update(phase = "decode", eta_seconds = None)

            avi = server.vid_gen(payload, on_step = on_line, cancel_event = self._cancel)
            lease.close()
            self._update(phase = "export", eta_seconds = None)
            raw = workdir / "output.avi"
            raw.write_bytes(avi)
            mp4 = transcode_video_to_mp4(raw, fps = FPS)
            raw.unlink(missing_ok = True)
            with self._lock:
                if self._cancel.is_set():
                    raise SdCppCancelled("cancelled")
                self._publishing = True
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
                        "guidance": 1.0,
                        "seed": job["seed"],
                        "has_audio": False,
                        "conditioning": "restyle",
                        "model": MODEL_ID,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            self._finish(phase = "done", video = record)
        except SdCppCancelled:
            self._finish(phase = "cancelled")
        except Exception as exc:  # noqa: BLE001 -- surfaced through progress
            if self._cancel.is_set():
                self._finish(phase = "cancelled")
            else:
                logger.warning("restyle failed: %s", exc)
                self._finish(phase = "error", error = str(exc))
        finally:
            lease.close()


runner = RestyleRunner()


def generation_in_flight() -> bool:
    """For the desktop watchdog's busy budget; a lock-free read, like the other media modules'."""
    return bool(runner._state.get("active"))
