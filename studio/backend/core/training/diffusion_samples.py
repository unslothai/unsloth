# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fixed-seed preview images during diffusion LoRA training. A round never touches training state:
fork_rng, private noise generator, a fresh scheduler copy, and force_eager for a compiled model."""

from __future__ import annotations

import re
import secrets
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Video families are left out: a preview there is a clip with audio, not a cheap still.
SAMPLE_FAMILIES: frozenset[str] = frozenset(
    {"sdxl", "z-image", "flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "krea-2"}
)
MAX_SAMPLE_PROMPTS = 4
MAX_SAMPLE_PROMPT_CHARS = 1000
MAX_SAMPLE_RESOLUTION = 768
SAMPLES_DIRNAME = "samples"
SAMPLE_PATH_RE = re.compile(
    r"^samples/[0-9]{8}-[0-9]{6}-[0-9a-f]{6}/step-[0-9]{1,6}-[0-9]{1,2}\.png$"
)


def validate_sample_settings(
    sample_every: Any, sample_prompts: Any, family: str
) -> tuple[int, tuple[str, ...]]:
    """Normalise (sample_every, sample_prompts). Raises ValueError on a value that cannot run."""
    try:
        every = int(sample_every or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_every must be a whole number, got {sample_every!r}") from exc
    if every < 0:
        raise ValueError("sample_every must be >= 0 (0 disables sample images)")
    if isinstance(sample_prompts, str):
        sample_prompts = [sample_prompts]
    prompts = tuple(str(p).strip() for p in (sample_prompts or ()) if str(p or "").strip())
    if len(prompts) > MAX_SAMPLE_PROMPTS:
        raise ValueError(f"at most {MAX_SAMPLE_PROMPTS} sample prompts are supported")
    if any(len(p) > MAX_SAMPLE_PROMPT_CHARS for p in prompts):
        raise ValueError(f"sample prompts must be at most {MAX_SAMPLE_PROMPT_CHARS} characters")
    if every and family not in SAMPLE_FAMILIES:
        # Refused, not ignored: a silently ignored setting reads as "sampling is broken".
        raise ValueError(f"sample images are not supported for {family}; set sample_every to 0")
    return every, prompts


# How each diffusers pipeline combines the conditional (c) and empty-prompt (u) predictions, and above which scale it
# runs the second pass at all: "uncond" u + g(c - u) (SDXL, FLUX.1 true CFG, FLUX.2 Klein; on when g > 1), "cond"
# c + g(c - u) (Z-Image pipeline_z_image.py:547, Krea 2 pipeline_krea2.py:666; on when g > 0), "qwen" the "uncond"
# combination rescaled to the conditional prediction's per-token norm (pipeline_qwenimage.py:668-672; on when g > 1).
_CFG_MODES = {"z-image": "cond", "krea-2": "cond", "qwen-image": "qwen"}


@dataclass(frozen = True)
class SampleSettings:
    steps: int
    # The pipeline's own guidance_scale (true_cfg_scale for Qwen-Image / FLUX.1); 0 means one conditional pass.
    guidance: float
    cfg_mode: str = "uncond"
    # A fixed schedule shift (Krea 2 Turbo pins mu = 1.15, pipeline_krea2.py:615); None derives it.
    mu: Optional[float] = None

    @property
    def uses_cfg(self) -> bool:
        return self.guidance > (0.0 if self.cfg_mode == "cond" else 1.0)


def sample_inference_settings(family: str, base_model: str) -> SampleSettings:
    """Distilled setting for a distilled base, else a short schedule at the pipeline's default guidance."""
    name = (base_model or "").lower()
    mode = _CFG_MODES.get(family, "uncond")
    if family == "sdxl":
        return SampleSettings(4, 0.0) if "turbo" in name else SampleSettings(20, 5.0)
    if family == "z-image":
        if "de-turbo" in name or "turbo" not in name:
            return SampleSettings(28, 4.0, mode)
        return SampleSettings(8, 0.0, mode)
    if family == "flux.2-klein":
        return SampleSettings(28, 4.0) if "base" in name else SampleSettings(4, 0.0)
    if family == "flux.2-dev":
        # Guidance-distilled: the forward already feeds the 3.5 guidance embedding.
        return SampleSettings(28, 0.0)
    if family == "flux.1":
        return SampleSettings(20, 0.0)
    if family == "qwen-image":
        return SampleSettings(20, 4.0, mode)
    if family == "krea-2":
        if "turbo" in name:
            return SampleSettings(8, 0.0, mode, mu = 1.15)
        return SampleSettings(20, 4.5, mode)
    return SampleSettings(20, 0.0)


def combine_cfg(mode: str, guidance: float, cond, uncond):
    """One guided prediction, exactly as the family's pipeline forms it (see _CFG_MODES)."""
    if mode == "cond":
        return cond + guidance * (cond - uncond)
    comb = uncond + guidance * (cond - uncond)
    if mode == "qwen":
        return comb * (_token_norm(cond) / _token_norm(comb))
    return comb


def _token_norm(v):
    """Per-token norm of an unpacked Qwen latent [B,C,1,H,W]: the pipeline takes it over the last dim
    of the packed [B, H/2*W/2, C*4] sequence, i.e. over the channels of each 2x2 patch."""
    b, c, f, h, w = v.shape
    p = v.reshape(b, c, f, h // 2, 2, w // 2, 2)
    n = p.pow(2).sum(dim = (1, 4, 6), keepdim = True).sqrt()
    return n.expand_as(p).reshape(v.shape)


def sample_resolution(train_resolution: int, multiple: int = 64) -> int:
    res = min(int(train_resolution), MAX_SAMPLE_RESOLUTION)
    return max(multiple, res // multiple * multiple)


@dataclass
class SamplePlan:
    every: int
    prompts: tuple[str, ...]
    seed: int
    steps: int
    guidance: float
    resolution: int
    out_dir: Path
    tag: str
    written: list
    cfg_mode: str = "uncond"
    mu: Optional[float] = None

    @property
    def uses_cfg(self) -> bool:
        return SampleSettings(self.steps, self.guidance, self.cfg_mode).uses_cfg

    @property
    def encode_texts(self) -> list[str]:
        """Every text whose embedding a round needs: the prompts, plus the empty prompt under CFG."""
        texts = list(dict.fromkeys(self.prompts))
        if self.uses_cfg and "" not in texts:
            texts.append("")
        return texts

    def due(self, step: int, total_steps: int) -> bool:
        return step > 0 and (step % self.every == 0 or step == total_steps)

    @property
    def run_dir(self) -> Path:
        return self.out_dir / SAMPLES_DIRNAME / self.tag

    def path_for(self, step: int, index: int) -> Path:
        return self.run_dir / f"step-{int(step)}-{int(index)}.png"

    def relpath(self, path: Path) -> str:
        return path.relative_to(self.out_dir).as_posix()


def plan_samples(cfg: Any, family: str, captions: list[str]) -> Optional[SamplePlan]:
    """The run's sample plan, or None when sampling is off."""
    every = int(getattr(cfg, "sample_every", 0) or 0)
    if every <= 0 or family not in SAMPLE_FAMILIES:
        return None
    prompts = tuple(getattr(cfg, "sample_prompts", ()) or ())
    if not prompts:
        default = (getattr(cfg, "instance_prompt", None) or "").strip() or next(
            (c for c in captions if c and c.strip()), ""
        )
        prompts = (default,)
    settings = sample_inference_settings(family, getattr(cfg, "base_model", ""))
    multiple = 128 if family.startswith("flux.2") else 64
    return SamplePlan(
        every = every,
        prompts = prompts,
        seed = int(getattr(cfg, "seed", 0) or 0),
        steps = settings.steps,
        guidance = settings.guidance,
        cfg_mode = settings.cfg_mode,
        mu = settings.mu,
        resolution = sample_resolution(getattr(cfg, "resolution", 1024), multiple),
        out_dir = Path(cfg.output_dir).expanduser(),
        tag = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}",
        written = [],
    )


def sample_seed(plan: SamplePlan, index: int) -> int:
    # Bounded to int64 so manual_seed accepts any configured seed.
    return (plan.seed + 1000003 * (index + 1)) % (2**63)


@contextmanager
def isolated_sampling(device: str, compiled: bool):
    """No-grad, RNG-isolated (keeps runs bit-identical with sampling on or off), eager when compiled."""
    import torch

    devices: list = []
    device_type = "cuda"
    if device == "cuda" and torch.cuda.is_available():
        devices = [torch.cuda.current_device()]
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        devices = [torch.xpu.current_device()]
        device_type = "xpu"
    stance = nullcontext()
    set_stance = getattr(getattr(torch, "compiler", None), "set_stance", None)
    if compiled and callable(set_stance):
        stance = set_stance("force_eager")
    with torch.random.fork_rng(devices = devices, device_type = device_type), torch.no_grad(), stance:
        yield


def initial_noise(plan: SamplePlan, index: int, shape, device):
    """Fixed-seed fp32 noise from a private CPU generator; never draws from the training streams."""
    import torch

    gen = torch.Generator(device = "cpu").manual_seed(sample_seed(plan, index))
    return torch.randn(tuple(shape), generator = gen, dtype = torch.float32).to(device)


def flow_sigmas(
    scheduler_config: Any,
    family: str,
    steps: int,
    image_seq_len: int,
    mu: Optional[float] = None,
):
    """Inference sigmas from a FRESH scheduler: set_timesteps on the training one rewrites its tables."""
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    sc = sched.config
    kwargs: dict = {}
    if getattr(sc, "use_dynamic_shifting", False):
        if mu is not None:
            kwargs["mu"] = float(mu)
        elif family.startswith("flux.2"):
            from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
            kwargs["mu"] = compute_empirical_mu(image_seq_len, steps)
        else:
            base_len = sc.get("base_image_seq_len", 256)
            # pipeline_krea2.py:620 defaults the max length to 6400; every other pipeline to 4096.
            max_len = sc.get("max_image_seq_len", 6400 if family == "krea-2" else 4096)
            base_shift = sc.get("base_shift", 0.5)
            max_shift = sc.get("max_shift", 1.15)
            m = (max_shift - base_shift) / (max_len - base_len)
            kwargs["mu"] = image_seq_len * m + (base_shift - m * base_len)
    sched.set_timesteps(
        sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist(), device = "cpu", **kwargs
    )
    return sched.sigmas.to(dtype = torch.float32), float(sc.num_train_timesteps)


def to_pil(images):
    import torch
    from PIL import Image

    arr = ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(dtype = torch.uint8)
    arr = arr.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def save_round(plan: SamplePlan, step: int, images: list) -> list[dict]:
    plan.run_dir.mkdir(parents = True, exist_ok = True)
    out = []
    for i, (img, prompt) in enumerate(zip(images, plan.prompts)):
        path = plan.path_for(step, i)
        tmp = path.with_suffix(".png.tmp")
        img.save(tmp, format = "PNG")
        tmp.replace(path)
        plan.written.append(path)
        out.append({"path": plan.relpath(path), "prompt": prompt, "seed": sample_seed(plan, i)})
    return out


def discard_samples(plan: Optional[SamplePlan]) -> None:
    """Remove this run's previews (a stop-without-saving); never another run's."""
    if plan is None:
        return
    try:
        if plan.run_dir.is_dir():
            shutil.rmtree(plan.run_dir)
        plan.run_dir.parent.rmdir()
    except OSError:
        pass


def _run_dirs(base: Path, relpaths: list) -> set:
    """The per-run tag folders (``samples/<tag>``) the reported paths name; never anything else."""
    root = base / SAMPLES_DIRNAME
    dirs = set()
    for rel in relpaths:
        if isinstance(rel, str) and SAMPLE_PATH_RE.match(rel):
            dirs.add(root / rel.split("/")[1])
    return dirs


def discard_sample_paths(output_dir: Optional[str], relpaths: list) -> None:
    """Parent-side ``discard_samples`` for a child that died before cleaning up."""
    if not output_dir:
        return
    base = Path(output_dir)
    for d in _run_dirs(base, relpaths):
        try:
            if d.is_dir() and not d.is_symlink():
                shutil.rmtree(d)
        except OSError:
            pass
    try:
        (base / SAMPLES_DIRNAME).rmdir()
    except OSError:
        pass


def delete_sample_files(output_dir: Optional[str], relpaths: list) -> None:
    """Delete individual reported images (history thinning), so no file outlives its listing."""
    if not output_dir:
        return
    for rel in relpaths:
        if isinstance(rel, str) and SAMPLE_PATH_RE.match(rel):
            try:
                (Path(output_dir) / rel).unlink()
            except OSError:
                pass


class SampleRoundStopped(Exception):
    """Raised by a render whose stop check fired mid-denoise."""


def run_sample_round(
    plan: SamplePlan,
    step: int,
    render: Callable[[int, str], Any],
    on_event,
    emit,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> bool:
    """Render, save and emit each prompt; a stop keeps finished images. True when cut short."""
    t0 = time.time()
    images = []
    stopped = False
    for i, prompt in enumerate(plan.prompts):
        if stop_requested is not None and stop_requested():
            stopped = True
            break
        try:
            images.extend(to_pil(render(i, prompt)))
        except SampleRoundStopped:
            stopped = True
            break
    if images:
        entries = save_round(plan, step, images)
        emit(on_event, "sample", step = int(step), images = entries, seconds = round(time.time() - t0, 2))
    return stopped


def euler_flow_sample(
    *,
    plan: SamplePlan,
    index: int,
    latent_shape,
    image_seq_len: int,
    family: str,
    scheduler_config: Any,
    velocity: Callable[[Any, Any, Any, Any], Any],
    cond,
    uncond,
    device,
    weight_dtype,
    stop_requested: Optional[Callable[[], bool]] = None,
):
    """Flow-matching Euler: x <- x + (sigma_next - sigma) * v, with v from the training forward
    (target convention noise - latents) and the family pipeline's CFG combination."""
    import torch

    sigmas, num_train = flow_sigmas(scheduler_config, family, plan.steps, image_seq_len, plan.mu)
    x = initial_noise(plan, index, latent_shape, device)
    for i in range(len(sigmas) - 1):
        if stop_requested is not None and stop_requested():
            raise SampleRoundStopped()
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        t = torch.full((x.shape[0],), s * num_train, device = device, dtype = torch.float32)
        sig = torch.full((x.shape[0],), s, device = device, dtype = weight_dtype)
        while sig.ndim < x.ndim:
            sig = sig.unsqueeze(-1)
        xin = x.to(weight_dtype)
        v = velocity(xin, t, sig, cond).float()
        if plan.uses_cfg and uncond is not None:
            v_u = velocity(xin, t, sig, uncond).float()
            v = combine_cfg(plan.cfg_mode, plan.guidance, v, v_u)
        x = x + (s_next - s) * v
    return x


__all__ = [
    "MAX_SAMPLE_PROMPTS",
    "SAMPLE_FAMILIES",
    "SAMPLE_PATH_RE",
    "SamplePlan",
    "SampleRoundStopped",
    "combine_cfg",
    "discard_sample_paths",
    "discard_samples",
    "euler_flow_sample",
    "isolated_sampling",
    "plan_samples",
    "run_sample_round",
    "sample_inference_settings",
    "validate_sample_settings",
]
