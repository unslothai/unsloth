# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seam guard: every VAE Studio can tile decodes in tiles wide enough, and overlapping enough, not to leave lines.

A VAE decoder sees several latents past a tile edge, so a tile decodes its border differently from its neighbour.
When the tiles are small, the overlap short, or the last tile a sliver with almost no context, the blend cannot
hide that and the image shows thin lines one tile long (Qwen-Image-2.1 with the Wan pixel geometry on a 16x VAE,
LTX-2.3 with 2-latent overlaps).

For every family in the image and video registries this builds the family's VAE from its real ``vae/config.json``
on the meta device (no weights), applies what a Studio load applies (``enable_tiling()`` plus every tile override
module Studio ships) and runs the real ``decode`` over every canvas side Studio offers, with the decoder replaced
by a shape-only stand-in. The tile grid is read from the latent slices the decode actually takes, so it covers
diffusers' defaults, a diffusers upgrade that changes them and Studio's own overrides alike. Off CUDA the overrides
size tiles for zero free VRAM, which is the tightest low-VRAM tier.

Thresholds, in latents, on every tiled axis: tile >= 32, neighbour overlap >= 16 (ComfyUI's decode overlap), and
no tile (the edge tile included) shorter than the overlap threshold. Where temporal tiling is on, neighbouring
temporal tiles share at least one latent frame. ``SMALLER_GEOMETRY_OK`` lists the VAEs measured seam-free at a
smaller geometry, each with its own floor and the evidence; ``KNOWN_SEAMS`` lists measured, still-open seam bugs
(strict xfail: the fix landing turns it into a failure until the entry goes).
"""

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

# The floor, when the tree has no image tile module to read it from (diffusion_vae_tiling: TILE_LATENTS,
# OVERLAP_LATENTS; its rule also wants every tile, the edge one included, at least TILE_LATENTS long).
MIN_TILE = 32
MIN_OVERLAP = 16
# Studio's smallest canvas side; the probe holds the other axis here so that only the swept axis tiles.
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
    """The image tile module's own rule (tile >= TILE_LATENTS, overlap >= OVERLAP_LATENTS, no tile shorter than
    TILE_LATENTS), so the guard and the fix cannot drift apart; 32 / 16 / 32 when the tree has no such module."""
    mod = _module("diffusion_vae_tiling")
    tile = int(getattr(mod, "TILE_LATENTS", MIN_TILE))
    overlap = int(getattr(mod, "OVERLAP_LATENTS", MIN_OVERLAP))
    src = "diffusion_vae_tiling" if mod is not None and hasattr(mod, "TILE_LATENTS") else "default"
    return Floor(tile, overlap, tile, src)


def ltx_floor() -> Floor:
    """LTX-2's floor from its tile module (MIN_TILE_LATENTS, OVERLAP_LATENTS), 16 / 8 when the tree has none."""
    mod = _module("video_ltx2_vae_tiles")
    tile = int(getattr(mod, "MIN_TILE_LATENTS", 16))
    overlap = int(getattr(mod, "OVERLAP_LATENTS", 8))
    return Floor(
        tile, overlap, overlap, "LTX-2 wide tiles (video_ltx2_vae_tiles); seam bench clean"
    )


def keep_stock() -> dict[str, str]:
    """VAE classes the image tile module deliberately leaves on diffusers' tiles, with its measured reason."""
    return dict(getattr(_module("diffusion_vae_tiling"), "KEEP_STOCK", {}) or {})


# "VAE class@ratio" -> the smaller geometry it is proven seam-free at. Only with a measurement: a tiled decode of a real
# latent against the untiled decode, at that geometry, without lines.
SMALLER_GEOMETRY_OK: dict[str, Floor] = {
    # 32x VAE: 16 latents are 512 px, the stock tile and what the tightest tier can afford. PR #12698's tiles (16
    # latents, >= 8-latent overlaps, 3-latent margin, 2-latent ramp), seam bench at the tightest tier, every preset:
    # worst 64 px window 1.88 levels (LTX-2) / 1.09 (LTX-2.3), PSNR >= 53 dB; the stock 2-latent overlaps: 2.3-7.7.
    # (read from video_ltx2_vae_tiles when present: ltx_floor)
    "AutoencoderKLLTX2Video@32x": Floor(
        16, 8, 8, "PR #12698's 16-latent tiles, 8-latent overlaps: seam bench clean"
    ),
    # Wan2.1 VAE (Wan2.2-T2V-A14B), stock 32-latent tiles with 8-latent overlaps and 8-latent edge tiles: the seam
    # bench (two photos, every preset, tightest tier, 9-frame pan) stays within 1.93 levels of the untiled decode in
    # the worst 64 px window, PSNR >= 49.9 dB, no boundary step above 1.38x its surroundings.
    "AutoencoderKLWan@8x": Floor(
        32, 8, 8, "seam bench: worst 64 px window 1.93 levels, PSNR >= 49.9 dB"
    ),
    # MiniMax-H3: a transformer decoder that works at its 256 px tile. Its untiled decode is not a reference (PSNR
    # 20 dB against the input photo, the tiled decode 30 dB), Studio always decodes it tiled, and diffusers spreads
    # the 16-latent tiles evenly with >= 4-latent overlaps and no sliver. A smaller tile or a sliver still fails.
    "AutoencoderKLMiniMaxH3@16x": Floor(
        16, 4, 16, "decoder works at its 256 px tile; untiled is out of distribution"
    ),
    # 16x video VAEs at their stock 16-latent tiles / 4-latent overlaps / 4-latent edge tiles. Video seam audit (real
    # 33-frame clip, every preset, stock tiles vs untiled): HunyuanVideo-1.5 PSNR 50.6-52.5 dB, Wan2.2-TI2V-5B 49.3 dB,
    # the boundary score on |tiled - untiled| 1.7-2.4 (no line); seam bench worst 64 px window 2.9-3.6 levels.
    "AutoencoderKLHunyuanVideo15@16x": Floor(
        16, 4, 4, "video seam audit: PSNR >= 50.6 dB, no line at the boundaries"
    ),
    "AutoencoderKLWan@16x": Floor(
        16, 4, 8, "video seam audit: PSNR 49.3 dB, no line at the boundaries"
    ),
}

# family -> a geometry below the floor that is a measured seam bug, or not yet proven seam-free. Strict xfail: the
# fix (wide tiles) or an allow-list entry backed by a measurement turns it into a failure until the entry goes.
# Empty: every image family's tiled decode meets the floor since the seam-free wide tiles (#12736).
KNOWN_SEAMS: dict[str, str] = {}

# Modules that rebind a VAE's tiled_decode without changing its tile grid (same attributes, batched).
GEOMETRY_PRESERVING = {
    "diffusion_vae_fused": "batched stock tiled_decode: the stock attributes' tiles, decoded as one batch",
}


# ---------------------------------------------------------------------------------------------------- probing
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
    """Stands in for a VAE submodule: returns zeros of the shape the real module would return (the input itself
    when the shape is unchanged, e.g. post_quant_conv). The real module runs on the meta device for the first two
    spatial sizes of each call signature; after that the output's spatial dims are the input's times the measured
    upsampling (checked equal on both runs), so sweeping canvas sides costs no further decoder passes."""

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
    return json.loads(FIXTURE.read_text())["families"]


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
    """Every canvas side, in pixels, Studio offers the family."""
    fam = next((f for f in DF._FAMILIES if f.name == family), None)
    if fam is not None:
        step = max(8, int(fam.dimension_multiple))
        return list(range(MIN_SIDE, int(fam.max_output_side) + 1, step))
    fam = next(f for f in VF._FAMILIES if f.name == ("ltx-2" if family == LTX23 else family))
    return sorted({v for wh in fam.resolution_presets for v in wh})


def build_vae(config: dict, overrides: bool = True):
    """The VAE on the meta device, set up as a Studio load sets it up (``overrides=False``: diffusers' stock tiling
    only), decoder replaced by a shape-only stand-in."""
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
    for mod in tile_override_modules()[0] if overrides else ():
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
    """Tiles (start, stop) along a ``length``-latent axis, the other axis ``other`` latents."""
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
    """Threshold violations of one axis's tiles."""
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
    """What decides the tile grid: the class, its tile attributes and the overrides installed on it."""
    attrs = sorted(
        (k, repr(v)) for k, v in vars(vae).items() if k.startswith(("tile_", "use_framewise"))
    )
    return type(vae).__name__, _ratio(vae), tuple(attrs), tuple(applied), video


@functools.lru_cache(maxsize = None)
def geometry(family: str):
    """(VAE class, ratio, overrides applied, {side px: tiles}, temporal problem or None) for ``family``."""
    kind, config = _family_configs()[family]
    vae, applied = build_vae(config)
    ratio = _ratio(vae)
    video = _is_5d(vae)
    other = max(1, MIN_SIDE // ratio)
    # families sharing a VAE geometry (FLUX.1 / Z-Image / HiDream / SDXL ...) share one probe per latent length
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


# ---------------------------------------------------------------------------------------------------- tests
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
    kept = keep_stock()
    if name in kept:
        # the image module keeps this VAE on diffusers' tiles on purpose, with a measurement: its reason stands
        pytest.skip(f"{name} in diffusion_vae_tiling.KEEP_STOCK: {kept[name]}")
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


def test_check_axis_flags_each_threshold():
    ok = Floor(32, 16, 16, "test")
    assert check_axis([(0, 64)], ok) == []
    assert check_axis([(0, 32), (16, 48), (32, 64)], ok) == []
    # Qwen-Image-2.1 on main at 1024 px: 16-latent tiles, 4-latent overlaps, a 4-latent sliver
    stock = [(0, 16), (12, 28), (24, 40), (36, 52), (48, 64), (60, 64)]
    assert check_axis(stock, ok) == [
        "16-latent tiles < 32",
        "4-latent overlap < 16",
        "a 4-latent edge tile < 16",
    ]
    # an 8x AutoencoderKL at 1600 px: 128-latent tiles, the last an 8-latent sliver inside its neighbour
    assert check_axis([(0, 128), (96, 200), (192, 200)], ok) == [
        "8-latent overlap < 16",
        "a 8-latent edge tile < 16",
    ]


def test_probe_reads_the_stock_grid():
    """The probe reports diffusers' own grid for a VAE with no override (Wan's 256 px / 192 px pixel geometry)."""
    vae, applied = build_vae({"_class_name": "AutoencoderKLWan", "z_dim": 16}, overrides = False)
    assert applied == []
    tiles = probe_axis(vae, {"z_dim": 16}, 64, 8, True)
    assert tiles == [(0, 32), (24, 56), (48, 64)]
