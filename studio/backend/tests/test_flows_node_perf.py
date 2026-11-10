# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Work the nodes no longer repeat: the border key, the init image read and the fine-tune cache lookups."""

import asyncio
import io

from core.flows.nodes import _common, training


def _looped_mask(data: bytes, tolerance: int = 40) -> bytes:
    """The per-pixel border key chroma_key replaced, as the reference."""
    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(data)).convert("RGBA")
    w, h = im.size
    work = im.convert("RGB")
    mask = Image.new("L", (w, h), 255)
    marker = (255, 0, 255)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]:  # fmt: skip
        if work.getpixel(seed) != marker:
            ImageDraw.floodfill(work, seed, marker, thresh = tolerance)
    pixels, mpx = work.load(), mask.load()
    for y in range(h):
        for x in range(w):
            if pixels[x, y] == marker:
                mpx[x, y] = 0
    return mask.tobytes()


def _subject_png() -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (96, 64), (20, 200, 30))
    draw = ImageDraw.Draw(im)
    draw.ellipse((30, 12, 70, 52), fill = (200, 40, 40))
    draw.rectangle((0, 60, 10, 63), fill = (255, 0, 255))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def test_chroma_key_matches_the_per_pixel_key():
    from PIL import Image

    data = _subject_png()
    cutout, mask = _common.chroma_key(data)
    got = Image.open(io.BytesIO(mask))
    assert got.mode == "L" and got.tobytes() == _looped_mask(data)
    assert Image.open(io.BytesIO(cutout)).getchannel("A").tobytes() == got.tobytes()


class _ImageCtx:
    def __init__(self, reads):
        self.reads = reads

    async def call(
        self,
        method,
        path,
        json = None,
        **kwargs,
    ):
        if path == "/api/inference/images/generate":
            return {"images": [{"id": "out1"}]}
        return {}

    def on_cancel(self, fn):
        pass

    def progress(self, *args, **kwargs):
        pass

    def log(
        self,
        message,
        level = "info",
    ):
        pass


def test_an_inpaint_reads_its_init_image_once(monkeypatch):
    reads = []
    png = _subject_png()

    async def fake_media_bytes(ctx, media):
        reads.append(media["ref"])
        return png, "image/png"

    async def keep(*args, **kwargs):
        return None

    monkeypatch.setattr(_common, "media_bytes", fake_media_bytes)
    monkeypatch.setattr(_common, "ensure_model", keep)
    asyncio.run(
        _common.generate_image(
            _ImageCtx(reads),
            model = None,
            prompt = "a fox",
            init_image = {"ref": "image:init"},
            mask = {"ref": "image:mask"},
        )
    )
    assert sorted(reads) == ["image:init", "image:mask"]


class _TrainCtx:
    def __init__(self, run_id):
        self.run_id = run_id
        self.caller = None
        self.params = {"base_model": "unsloth/Qwen3-4B", "training_type": "lora"}
        self.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/data"}}


def test_fine_tune_cache_lookups_run_once_per_run(monkeypatch):
    asked = []

    def revision(dataset, token):
        asked.append(("dataset", dataset["ref"]))
        return "d1"

    def model_revision(repo, token):
        asked.append(("model", repo))
        return "m1"

    def adapter_base(repo, revision, token):
        asked.append(("adapter", repo))
        return None

    monkeypatch.setattr(training, "_RUN_MEMO", {})
    monkeypatch.setattr(training, "_hub_revision", revision)
    monkeypatch.setattr(training, "_hub_model_revision", model_revision)
    monkeypatch.setattr(training, "_hub_adapter_base", adapter_base)
    first = asyncio.run(training.finetune_cache_material(_TrainCtx("run-1")))
    per_run = len(asked)
    assert per_run >= 3
    again = asyncio.run(training.finetune_cache_material(_TrainCtx("run-1")))
    assert again == first and len(asked) == per_run
    asyncio.run(training.finetune_cache_material(_TrainCtx("run-2")))
    assert len(asked) == 2 * per_run


def test_a_local_dataset_is_hashed_again_when_its_file_changes(monkeypatch, tmp_path):
    from core.flows.nodes import data

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n')
    hashed = []
    real = training._content_hash

    def counting(p):
        hashed.append(p)
        return real(p)

    monkeypatch.setattr(training, "_RUN_MEMO", {})
    monkeypatch.setattr(training, "_content_hash", counting)
    monkeypatch.setattr(data, "local_seed_path", lambda ref: str(path))
    ctx = _TrainCtx("run-1")
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "local", "ref": "uploads/rows.jsonl"}}
    one = asyncio.run(training.dataset_material(ctx))
    assert asyncio.run(training.dataset_material(ctx)) == one and len(hashed) == 1
    path.write_text('{"a": 2, "b": "longer"}\n')
    assert asyncio.run(training.dataset_material(ctx)) != one and len(hashed) == 2


def test_chroma_key_without_scipy_matches_too(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    data = _subject_png()
    from PIL import Image

    _, mask = _common.chroma_key(data)
    assert Image.open(io.BytesIO(mask)).tobytes() == _looped_mask(data)


def test_chroma_key_matches_on_noisy_borders():
    import random

    from PIL import Image
    for seed in range(6):
        rng = random.Random(seed)
        im = Image.new("RGB", (40, 30))
        im.putdata(
            [
                tuple(min(255, max(0, 120 + rng.randint(-30, 30) + (x + y) % 7)) for _ in range(3))
                if rng.random() > 0.15
                else (255, 0, 255)
                for y in range(30)
                for x in range(40)
            ]
        )
        out = io.BytesIO()
        im.save(out, "PNG")
        data = out.getvalue()
        _, mask = _common.chroma_key(data, tolerance = 25)
        assert Image.open(io.BytesIO(mask)).tobytes() == _looped_mask(data, 25), seed


def test_the_scipy_free_flood_finds_the_same_region():
    import numpy as np
    from scipy import ndimage

    rng = np.random.default_rng(0)
    for _ in range(20):
        candidates = rng.random((30, 40)) > 0.35
        y, x = np.argwhere(candidates)[0]
        labels, _ = ndimage.label(candidates)
        expected = labels == labels[y, x]
        assert (_common._flood_region(candidates, int(x), int(y)) == expected).all()
