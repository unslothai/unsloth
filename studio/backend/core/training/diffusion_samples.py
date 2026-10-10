# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fixed-seed preview images rendered during diffusion LoRA training (opt-in via ``sample_every``).

Loss does not show likeness or overfitting, so every N optimizer steps the trainer renders the same
prompts from the same seeds with the live adapter. A round never touches training state: it runs
under ``torch.random.fork_rng`` (the CPU and accelerator generators come back exactly as they were),
draws its own noise from a private ``torch.Generator``, never steps the training scheduler (a fresh
copy is built from its config), and runs a compiled transformer eagerly
(``torch.compiler.set_stance("force_eager")``) so the training graph is neither recompiled nor
replaced. Images land under ``<output_dir>/samples/<run tag>/step-<N>-<i>.png``; the tag is per
run, so a retrain into the same folder never overwrites or mixes in another run's previews.
"""

from __future__ import annotations

import re
import secrets
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Image families with a sampling path. Video families (LTX-2, MiniMax-H3) are left out: a preview there is a clip
# with an audio stream, not a cheap still.
SAMPLE_FAMILIES: frozenset[str] = frozenset(
    {"sdxl", "z-image", "flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "krea-2"}
)
MAX_SAMPLE_PROMPTS = 4
MAX_SAMPLE_PROMPT_CHARS = 1000
# Previews stay modest: training resolutions above this render at it (aspect is square either way).
MAX_SAMPLE_RESOLUTION = 768
SAMPLES_DIRNAME = "samples"
# The relative path every reported sample has; the route serves nothing else.
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
        # Refused, not ignored, like save_steps on a checkpointless family: a silently ignored setting reads as
        # "sampling is broken".
        raise ValueError(f"sample images are not supported for {family}; set sample_every to 0")
    return every, prompts


def sample_inference_settings(family: str, base_model: str) -> tuple[int, float]:
    """(steps, guidance) for a preview: the family's distilled setting when the base is a distilled
    checkpoint, else a short undistilled schedule. guidance <= 1 means a single conditional pass."""
    name = (base_model or "").lower()
    if family == "sdxl":
        return 20, 5.0
    if family == "z-image":
        if "de-turbo" in name or "turbo" not in name:
            return 28, 4.0
        return 8, 1.0
    if family == "flux.2-klein":
        return (28, 4.0) if "base" in name else (4, 1.0)
    if family == "flux.2-dev":
        # Guidance-distilled: the forward already feeds the 3.5 guidance embedding.
        return 28, 1.0
    if family == "flux.1":
        return 20, 1.0
    if family == "qwen-image":
        return 20, 4.0
    if family == "krea-2":
        return 20, 4.5
    return 20, 1.0


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

    @property
    def uses_cfg(self) -> bool:
        return self.guidance > 1.0

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
    """The run's sample plan, or None when sampling is off. Default prompt: the instance prompt,
    else the first caption."""
    every = int(getattr(cfg, "sample_every", 0) or 0)
    if every <= 0 or family not in SAMPLE_FAMILIES:
        return None
    prompts = tuple(getattr(cfg, "sample_prompts", ()) or ())
    if not prompts:
        default = (getattr(cfg, "instance_prompt", None) or "").strip() or next(
            (c for c in captions if c and c.strip()), ""
        )
        prompts = (default,)
    steps, guidance = sample_inference_settings(family, getattr(cfg, "base_model", ""))
    multiple = 128 if family.startswith("flux.2") else 64
    return SamplePlan(
        every = every,
        prompts = prompts,
        seed = int(getattr(cfg, "seed", 0) or 0),
        steps = steps,
        guidance = guidance,
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
    """No-grad, RNG-isolated and (for a compiled model) eager. The RNG fork is what keeps a run with
    sampling on bit-identical to one with it off."""
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
    """Fixed-seed fp32 noise from a private CPU generator, so it is the same on every step and every
    device and never draws from the training streams."""
    import torch

    gen = torch.Generator(device = "cpu").manual_seed(sample_seed(plan, index))
    return torch.randn(tuple(shape), generator = gen, dtype = torch.float32).to(device)


def flow_sigmas(scheduler_config: Any, family: str, steps: int, image_seq_len: int):
    """The family's inference sigma schedule (N+1 values ending at 0), from a FRESH scheduler built
    from the training one's config: set_timesteps on the training scheduler would rewrite the
    tables the loop draws its timesteps from."""
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    sc = sched.config
    kwargs: dict = {}
    if getattr(sc, "use_dynamic_shifting", False):
        if family.startswith("flux.2"):
            from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu
            kwargs["mu"] = compute_empirical_mu(image_seq_len, steps)
        else:
            base_len = sc.get("base_image_seq_len", 256)
            max_len = sc.get("max_image_seq_len", 4096)
            base_shift = sc.get("base_shift", 0.5)
            max_shift = sc.get("max_shift", 1.15)
            m = (max_shift - base_shift) / (max_len - base_len)
            kwargs["mu"] = image_seq_len * m + (base_shift - m * base_len)
    sched.set_timesteps(
        sigmas = np.linspace(1.0, 1.0 / steps, steps).tolist(), device = "cpu", **kwargs
    )
    return sched.sigmas.to(dtype = torch.float32), float(sc.num_train_timesteps)


def to_pil(images):
    """[B,3,H,W] in [-1, 1] -> PIL images."""
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


def discard_sample_paths(output_dir: Optional[str], relpaths: list) -> None:
    """Parent-side twin of ``discard_samples`` for a child that died before cleaning up: only paths
    the run itself reported, then their per-run folder if it is left empty."""
    if not output_dir:
        return
    base = Path(output_dir)
    dirs = set()
    for rel in relpaths:
        if not isinstance(rel, str) or not SAMPLE_PATH_RE.match(rel):
            continue
        p = base / rel
        try:
            p.unlink()
        except OSError:
            pass
        dirs.add(p.parent)
    for d in dirs:
        for target in (d, d.parent):
            try:
                target.rmdir()
            except OSError:
                pass


def run_sample_round(
    plan: SamplePlan, step: int, render: Callable[[int, str], Any], on_event, emit
) -> None:
    """Render every prompt via ``render(index, prompt) -> [1,3,H,W] in [-1,1]``, save, and emit one
    ``sample`` event."""
    t0 = time.time()
    images = []
    for i, prompt in enumerate(plan.prompts):
        images.extend(to_pil(render(i, prompt)))
    entries = save_round(plan, step, images)
    emit(on_event, "sample", step = int(step), images = entries, seconds = round(time.time() - t0, 2))


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
):
    """Flow-matching Euler: x <- x + (sigma_next - sigma) * v, with v from the training forward
    (target convention noise - latents) and true CFG when ``plan.guidance > 1``."""
    import torch

    sigmas, num_train = flow_sigmas(scheduler_config, family, plan.steps, image_seq_len)
    x = initial_noise(plan, index, latent_shape, device)
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        t = torch.full((x.shape[0],), s * num_train, device = device, dtype = torch.float32)
        sig = torch.full((x.shape[0],), s, device = device, dtype = weight_dtype)
        while sig.ndim < x.ndim:
            sig = sig.unsqueeze(-1)
        xin = x.to(weight_dtype)
        v = velocity(xin, t, sig, cond).float()
        if plan.uses_cfg and uncond is not None:
            v_u = velocity(xin, t, sig, uncond).float()
            v = v_u + plan.guidance * (v - v_u)
        x = x + (s_next - s) * v
    return x


__all__ = [
    "MAX_SAMPLE_PROMPTS",
    "SAMPLE_FAMILIES",
    "SAMPLE_PATH_RE",
    "SamplePlan",
    "discard_sample_paths",
    "discard_samples",
    "euler_flow_sample",
    "isolated_sampling",
    "plan_samples",
    "run_sample_round",
    "sample_inference_settings",
    "validate_sample_settings",
]
