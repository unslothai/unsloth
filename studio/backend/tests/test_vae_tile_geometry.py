# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam guard: tiled VAE decodes must overlap enough to hide tile seams at every offered size."""

from __future__ import annotations

import functools
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from torch.overrides import TorchFunctionMode

from core.inference import diffusion_families as DF
from core.inference import video_families as VF

BACKEND = Path(__file__).resolve().parents[1]
INFERENCE = BACKEND / "core" / "inference"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vae_tile_configs.json"

# Fallback floor when diffusion_vae_tiling is absent.
MIN_TILE = 32
MIN_OVERLAP = 16
# The probe holds the other axis at Studio's smallest side so only the swept axis tiles.
MIN_SIDE = 256
LTX23 = "ltx-2.3"


@dataclass(frozen = True)
class Floor:
    tile: int
    overlap: int
    edge: int
    why: str


def _module(name: str):
    try:
        return importlib.import_module(f"core.inference.{name}")
    except ImportError:
        return None


def default_floor() -> Floor:
    """Read from diffusion_vae_tiling so the guard and the fix cannot drift apart."""
    mod = _module("diffusion_vae_tiling")
    tile = int(getattr(mod, "TILE_LATENTS", MIN_TILE))
    overlap = int(getattr(mod, "OVERLAP_LATENTS", MIN_OVERLAP))
    src = "diffusion_vae_tiling" if mod is not None and hasattr(mod, "TILE_LATENTS") else "default"
    return Floor(tile, overlap, tile, src)


def ltx_floor() -> Floor:
    mod = _module("video_ltx2_vae_tiles")
    tile = int(getattr(mod, "MIN_TILE_LATENTS", 16))
    overlap = int(getattr(mod, "OVERLAP_LATENTS", 8))
    return Floor(
        tile, overlap, overlap, "LTX-2 wide tiles (video_ltx2_vae_tiles); seam bench clean"
    )


def keep_stock() -> dict[str, str]:
    return dict(getattr(_module("diffusion_vae_tiling"), "KEEP_STOCK", {}) or {})


# "VAE class@ratio" -> a smaller geometry measured seam-free. Measurement required.
SMALLER_GEOMETRY_OK: dict[str, Floor] = {
    # #12698 seam bench: worst 64 px window <= 1.88 levels, PSNR >= 53 dB (stock 2-latent overlaps: 2.3-7.7).
    "AutoencoderKLLTX2Video@32x": Floor(
        16, 8, 8, "PR #12698's 16-latent tiles, 8-latent overlaps: seam bench clean"
    ),
    "AutoencoderKLWan@8x": Floor(
        32, 8, 8, "seam bench: worst 64 px window 1.93 levels, PSNR >= 49.9 dB"
    ),
    # Untiled is worse here (20 dB vs the input photo, tiled 30 dB): Studio always decodes it tiled.
    "AutoencoderKLMiniMaxH3@16x": Floor(
        16, 4, 16, "decoder works at its 256 px tile; untiled is out of distribution"
    ),
    "AutoencoderKLHunyuanVideo15@16x": Floor(
        16, 4, 4, "video seam audit: PSNR >= 50.6 dB, no line at the boundaries"
    ),
    "AutoencoderKLWan@16x": Floor(
        16, 4, 8, "video seam audit: PSNR 49.3 dB, no line at the boundaries"
    ),
}

# family -> open seam bug below the floor. Strict xfail, so the fix landing forces the entry out.
KNOWN_SEAMS: dict[str, str] = {}

GEOMETRY_PRESERVING = {
    "diffusion_vae_fused": "batched stock tiled_decode: the stock attributes' tiles, decoded as one batch",
}


class _TileRecorder(TorchFunctionMode):
    """Every (start, stop) slice taken of the last two dims of a tensor shaped like the latent."""

    def __init__(self, hw):
        super().__init__()
        self.hw = tuple(int(v) for v in hw)
        self.axes = (set(), set())
        self.paused = False

    def __torch_function__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        kwargs = kwargs or {}
        if not self.paused and func is torch.Tensor.__getitem__ and len(args) > 1:
            self._record(args[0], args[1])
        return func(*args, **kwargs)

    def _record(self, t, idx):
        if not torch.is_tensor(t) or t.dim() < 4 or tuple(t.shape[-2:]) != self.hw:
            return
        idx = idx if isinstance(idx, tuple) else (idx,)
        if any(i is Ellipsis for i in idx):
            k = next(n for n, i in enumerate(idx) if i is Ellipsis)
            idx = idx[:k] + (slice(None),) * (t.dim() - len(idx) + 1) + idx[k + 1 :]
        if len(idx) != t.dim() or not all(isinstance(s, slice) for s in idx[-2:]):
            return
        for axis, (s, n) in enumerate(zip(idx[-2:], self.hw)):
            lo, hi, _ = s.indices(n)
            if (lo, hi) != (0, n) and hi > lo:
                self.axes[axis].add((lo, hi))


_ACTIVE: list = []


class _ShapeOnly(torch.nn.Module):
    """Zeros of the real module's output shape. The real module runs on meta for the first two spatial sizes per
    call signature; later sizes extrapolate the measured upsampling, so the sweep costs no further decoder passes."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._seen = {}

    @staticmethod
    def _sig(v, spatial: bool):
        if torch.is_tensor(v):
            return tuple(v.shape) if spatial else (tuple(v.shape[:-2]), v.dim())
        return type(v).__name__ if isinstance(v, (list, dict, tuple)) else repr(v)

    def _run(self, args, kwargs):
        meta = lambda v: v.to("meta") if torch.is_tensor(v) else v  # noqa: E731
        for rec in _ACTIVE:
            rec.paused = True
        try:
            return tuple(
                self.inner(*map(meta, args), **{k: meta(v) for k, v in kwargs.items()}).shape
            )
        finally:
            for rec in _ACTIVE:
                rec.paused = False

    def forward(self, *args, **kwargs):
        x = next(a for a in args if torch.is_tensor(a))
        key = (
            tuple(self._sig(a, False) for a in args),
            tuple(sorted((k, self._sig(v, False)) for k, v in kwargs.items())),
        )
        seen = self._seen.setdefault(key, [])
        hw = tuple(x.shape[-2:])
        known = [out for inp, out in seen if inp == hw]
        if known:
            shape = known[0]
        elif len(seen) < 2:
            shape = self._run(args, kwargs)
            seen.append((hw, shape))
            if len(seen) == 2:
                (h0, w0), o0 = seen[0]
                (h1, w1), o1 = seen[1]
                if o0[:-2] != o1[:-2] or o0[-2] * h1 != o1[-2] * h0 or o0[-1] * w1 != o1[-1] * w0:
                    seen.append(((-1, -1), None))  # not a pure upsample: always run the module
        elif any(out is None for _, out in seen):
            shape = self._run(args, kwargs)
        else:
            (h0, w0), o0 = seen[0]
            shape = (*o0[:-2], hw[0] * o0[-2] // h0, hw[1] * o0[-1] // w0)
        return x if shape == tuple(x.shape) else x.new_zeros(shape)


def tile_override_modules() -> tuple[list[str], list[str]]:
    """(override modules a load applies with ``install(vae)``, other modules that rebind ``tiled_decode``)."""
    found, other = [], []
    for path in sorted(INFERENCE.glob("*.py")):
        text = path.read_text(encoding = "utf-8", errors = "replace")
        if not re.search(r"\.tiled_decode\s*=", text):
            continue
        if re.search(r"tiled_decode\s*=\s*types\.MethodType", text) and re.search(
            r"^def install\(\s*vae\b", text, re.M
        ):
            found.append(path.stem)
        else:
            other.append(path.stem)
    return found, other


def _vae_class(name: str):
    cls = getattr(diffusers, name, None)
    if cls is None:
        for mod in ("core.inference.video_minimax_h3_vae",):
            cls = getattr(importlib.import_module(mod), name, None)
            if cls is not None:
                break
    return cls


@functools.lru_cache(maxsize = None)
def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding = "utf-8"))["families"]


def _family_configs() -> dict[str, tuple[str, dict]]:
    """family -> (kind, vae config) for every registry family, plus LTX-2.3's single-file VAE config."""
    out = {}
    fixture = _fixture()
    for f in DF._FAMILIES:
        if f.name in fixture:
            out[f.name] = ("image", fixture[f.name]["config"])
    for f in VF._FAMILIES:
        if f.name in fixture:
            out[f.name] = ("video", fixture[f.name]["config"])
    from core.inference import video_ltx2

    out[LTX23] = (
        "video",
        {"_class_name": "AutoencoderKLLTX2Video", **video_ltx2._VIDEO_VAE_CONFIG},
    )
    return out


def _sides(family: str) -> list[int]:
    fam = next((f for f in DF._FAMILIES if f.name == family), None)
    if fam is not None:
        step = max(8, int(fam.dimension_multiple))
        return list(range(MIN_SIDE, int(fam.max_output_side) + 1, step))
    fam = next(f for f in VF._FAMILIES if f.name == ("ltx-2" if family == LTX23 else family))
    return sorted({v for wh in fam.resolution_presets for v in wh})


# The loader each kind goes through: a family gets only the overrides its own loader installs.
LOADERS = {"image": "diffusion.py", "video": "video.py"}


def overrides_for(kind: str) -> list[str]:
    src = (INFERENCE / LOADERS[kind]).read_text(encoding = "utf-8", errors = "replace")
    return [m for m in tile_override_modules()[0] if re.search(rf"\.{m}\b", src)]


def build_vae(config: dict, kind: str | None = None):
    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    cls = _vae_class(config["_class_name"])
    if cls is None:
        pytest.fail(
            f"{config['_class_name']} is in neither diffusers {diffusers.__version__} nor Studio"
        )
    with torch.device("meta"):
        vae = cls.from_config(cfg)
    vae.enable_tiling()
    applied = []
    for mod in overrides_for(kind) if kind else ():
        if importlib.import_module(f"core.inference.{mod}").install(vae):
            applied.append(mod)
    for name in ("decoder", "post_quant_conv"):
        child = getattr(vae, name, None)
        if isinstance(child, torch.nn.Module):
            setattr(vae, name, _ShapeOnly(child))
    return vae, applied


def _latent_channels(config: dict) -> int:
    for key in ("z_dim", "latent_channels", "z_channels"):
        if config.get(key):
            return int(config[key])
    return 16


def probe_axis(vae, config: dict, length: int, other: int, video: bool) -> list[tuple[int, int]]:
    c = _latent_channels(config)
    for frames in (1, 2, 4, 8) if video else (None,):
        shape = (1, c, length, other) if frames is None else (1, c, frames, length, other)
        rec = _TileRecorder(shape[-2:])
        _ACTIVE.append(rec)
        try:
            with rec, torch.no_grad():
                vae.decode(torch.zeros(shape), return_dict = False)
            return sorted(rec.axes[0])
        except Exception as exc:  # noqa: BLE001 - a frame count the VAE's temporal chunking refuses
            last = exc
        finally:
            _ACTIVE.remove(rec)
    raise last


def _ratio(vae) -> int:
    ratio = getattr(vae, "spatial_compression_ratio", None)
    if ratio is None:  # AutoencoderKL-style: one 2x upsample per block after the first
        ratio = 2 ** (len(vae.config.block_out_channels) - 1)
    return int(ratio)


def _is_5d(vae) -> bool:
    import inspect
    try:
        src = inspect.getsource(type(vae)._encode)
    except (OSError, TypeError, AttributeError):
        return False
    return (
        re.search(r"num_frame|frames|shape\[2\]|(?:\w+\s*,\s*){4}\w+\s*=\s*\w+\.shape", src)
        is not None
    )


def check_axis(tiles: list[tuple[int, int]], floor: Floor) -> list[str]:
    if len(tiles) < 2:
        return []
    problems = []
    longest = max(hi - lo for lo, hi in tiles)
    shortest = min(hi - lo for lo, hi in tiles)
    overlap = min(a[1] - b[0] for a, b in zip(tiles, tiles[1:]))
    if longest < floor.tile:
        problems.append(f"{longest}-latent tiles < {floor.tile}")
    if overlap < floor.overlap:
        problems.append(f"{overlap}-latent overlap < {floor.overlap}")
    if shortest < floor.edge:
        problems.append(f"a {shortest}-latent edge tile < {floor.edge}")
    return problems


_PROBED: dict = {}


def _tile_key(vae, applied, video: bool) -> tuple:
    attrs = sorted(
        (k, repr(v)) for k, v in vars(vae).items() if k.startswith(("tile_", "use_framewise"))
    )
    return type(vae).__name__, _ratio(vae), tuple(attrs), tuple(applied), video


@functools.lru_cache(maxsize = None)
def geometry(family: str):
    """(VAE class, ratio, overrides applied, {side px: tiles}, temporal problem or None) for ``family``."""
    kind, config = _family_configs()[family]
    vae, applied = build_vae(config, kind)
    ratio = _ratio(vae)
    video = _is_5d(vae)
    other = max(1, MIN_SIDE // ratio)
    cache = _PROBED.setdefault(_tile_key(vae, applied, video), {})
    grid = {}
    for side in _sides(family):
        if side % ratio:
            continue
        length = side // ratio
        if length not in cache:
            cache[length] = probe_axis(vae, config, length, other, video)
        grid[side] = cache[length]
    temporal = None
    if getattr(vae, "use_framewise_decoding", False):
        t_ratio = int(getattr(vae, "temporal_compression_ratio", 1) or 1)
        span = int(vae.tile_sample_min_num_frames) - int(vae.tile_sample_stride_num_frames)
        if span < t_ratio:
            temporal = f"temporal tiles share {span} frames < 1 latent frame ({t_ratio})"
    return type(vae).__name__, ratio, tuple(applied), grid, temporal


def _all_families() -> list[str]:
    return [f.name for f in DF._FAMILIES] + [f.name for f in VF._FAMILIES] + [LTX23]


def test_every_family_has_a_vae_config():
    missing = [f.name for f in (*DF._FAMILIES, *VF._FAMILIES) if f.name not in _fixture()]
    assert not missing, (
        f"families without a VAE config in {FIXTURE.name}: {missing}. Add each one's vae/config.json "
        "(base repo, pinned revision) so the seam guard checks the tile geometry Studio decodes it with."
    )


def test_every_tiled_decode_patch_is_probed():
    _, other = tile_override_modules()
    unknown = [m for m in other if m not in GEOMETRY_PRESERVING]
    assert not unknown, (
        f"{unknown} rebind a VAE's tiled_decode but expose no install(vae) this guard applies: give them one "
        "(types.MethodType + def install(vae, logger=None)) or, if they keep the stock tile grid, list them in "
        "GEOMETRY_PRESERVING with the reason."
    )


def _family_params():
    return [
        pytest.param(f, marks = pytest.mark.xfail(strict = True, reason = KNOWN_SEAMS[f]))
        if f in KNOWN_SEAMS
        else f
        for f in _all_families()
    ]


@pytest.mark.parametrize("family", _family_params())
def test_tile_geometry(family):
    name, ratio, applied, grid, temporal = geometry(family)
    # A blind recorder (tiling via narrow / split) must fail, not pass with no grid.
    assert any(
        len(t) >= 2 for t in grid.values()
    ), f"{family}: {name} recorded no multi-tile grid on any canvas side; the probe no longer sees the tiling"
    kept = keep_stock()
    if name in kept:
        # because the image module keeps this VAE on diffusers' tiles on purpose, with a measurement: its reason stands
        pytest.skip(reason = f"{name} in diffusion_vae_tiling.KEEP_STOCK: {kept[name]}")
    floor = ltx_floor() if name == "AutoencoderKLLTX2Video" else None
    floor = floor or SMALLER_GEOMETRY_OK.get(f"{name}@{ratio}x", default_floor())
    bad = {side: (tiles, check_axis(tiles, floor)) for side, tiles in grid.items()}
    bad = {side: v for side, v in bad.items() if v[1]}
    msgs = []
    if bad:
        side, (tiles, problems) = min(bad.items(), key = lambda kv: abs(kv[0] - 1024))
        msgs.append(
            f"{family}: {name} ({ratio}x, overrides {list(applied) or 'none'}) decodes a {side} px side in tiles "
            f"{tiles} (latents): {', '.join(problems)}; {len(bad)} of {len(grid)} canvas sides fail "
            f"({sorted(bad)[:8]}{' ...' if len(bad) > 8 else ''}). Need tile >= {floor.tile}, overlap >= "
            f"{floor.overlap}, every tile >= {floor.edge} latents ({floor.why}). Install wide tiles for this VAE, "
            "or add it to SMALLER_GEOMETRY_OK with a measured seam-free decode."
        )
    if temporal:
        msgs.append(f"{family}: {name}: {temporal}")
    assert not msgs, "\n".join(msgs)


def test_overrides_follow_the_loader():
    assert "diffusion_vae_tiling" in overrides_for("image")
    assert "diffusion_vae_tiling" not in overrides_for("video")
    assert "video_ltx2_vae_tiles" in overrides_for("video")


def test_check_axis_flags_each_threshold():
    ok = Floor(32, 16, 16, "test")
    assert check_axis([(0, 64)], ok) == []
    assert check_axis([(0, 32), (16, 48), (32, 64)], ok) == []
    # Qwen-Image-2.1 before #12696, 1024 px
    stock = [(0, 16), (12, 28), (24, 40), (36, 52), (48, 64), (60, 64)]
    assert check_axis(stock, ok) == [
        "16-latent tiles < 32",
        "4-latent overlap < 16",
        "a 4-latent edge tile < 16",
    ]
    # 8x AutoencoderKL at 1600 px: an 8-latent sliver
    assert check_axis([(0, 128), (96, 200), (192, 200)], ok) == [
        "8-latent overlap < 16",
        "a 8-latent edge tile < 16",
    ]


def test_probe_reads_the_stock_grid():
    vae, applied = build_vae({"_class_name": "AutoencoderKLWan", "z_dim": 16})
    assert applied == []
    tiles = probe_axis(vae, {"z_dim": 16}, 64, 8, True)
    assert tiles == [(0, 32), (24, 56), (48, 64)]
