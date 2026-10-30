# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Images Detailer (face fix) pass: FaceDetailer's crop / enlarge / feather / paste math, ADetailer's prompt and
infotext rules, the per-detection seeds, the recipe record, and the gating that refuses a model with no inpaint path
or a missing SAM 3. The inpaint render and SAM 3 are fakes here; the real ones run in the GPU end-to-end script."""

from __future__ import annotations

import types

import numpy as np
import pytest
from PIL import Image

from core.inference import diffusion_detailer as det


def test_defaults_follow_facedetailer():
    s = det.DetailerSettings()
    assert (s.targets, s.confidence, s.denoise, s.crop_factor) == ("face", 0.5, 0.5, 3.0)
    assert (s.guide_size, s.guide_size_for, s.max_size) == (512, "bbox", 1024)
    assert (s.dilation, s.feather, s.drop_size) == (10, 5, 10)
    assert s.steps is None and s.guidance is None and s.prompt is None


def test_request_defaults_match_the_module():
    from models.inference import DiffusionDetailerSpec
    spec = DiffusionDetailerSpec()
    assert det.DetailerSettings.from_request(spec) == det.DetailerSettings()


def test_crop_region_is_the_box_times_crop_factor_about_its_centre():
    # 100x80 box at (400, 300): 300x240 crop centred on (450, 340).
    assert det.crop_region(1024, 1024, (400, 300, 500, 380), 3.0) == (300, 220, 600, 460)


def test_crop_region_is_shifted_back_inside_the_image():
    # Near the top-left corner the crop keeps its size and starts at 0 (normalize_region).
    assert det.crop_region(1024, 768, (10, 5, 110, 85), 3.0) == (0, 0, 300, 240)
    assert det.crop_region(1024, 768, (950, 700, 1020, 760), 3.0) == (814, 588, 1024, 768)
    assert det.crop_region(200, 100, (50, 20, 150, 80), 3.0) == (0, 0, 200, 100)


def test_upscale_puts_the_box_short_side_at_guide_size():
    # 100x80 box in a 300x240 crop: 512 / 80 = 6.4 -> 1920x1536, capped at 1024 on the long side.
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "bbox", 1024)
    assert (w, h) == (1024, 819) and upscale == pytest.approx(6.4 * 1024 / 1920)
    # A loose cap: the box's short side lands on guide_size.
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "bbox", 4096)
    assert upscale == pytest.approx(6.4) and (w, h) == (1920, 1536)
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "crop_region", 4096)
    assert upscale == pytest.approx(512 / 240) and h == 512


def test_a_big_target_renders_at_its_own_size():
    # force_inpaint: factor <= 1 never shrinks the crop.
    assert det.upscaled_size(600, 600, 1000, 1000, 512, "bbox", 1024) == (1.0, 1000, 1000)


def test_max_size_caps_the_enlargement_but_never_shrinks_a_large_crop():
    """FaceDetailer parity (Impact-Pack core.enhance_detail, force_inpaint): when the capped factor is at or below 1,
    the crop renders at its own size, so a 2048 px crop with a large face is not shrunk to max_size."""
    assert det.upscaled_size(800, 800, 2048, 2048, 512, "bbox", 1024) == (1.0, 2048, 2048)


def test_snap_size_floors_to_the_inpaint_grid():
    assert det.snap_size(1024, 819) == (1024, 816)
    assert det.snap_size(5, 40) == (16, 32)


def _cv2():
    return pytest.importorskip("cv2")


@pytest.mark.parametrize("pixels", [10, 3, 4, -4, -7])
def test_dilate_matches_opencv(pixels):
    cv2 = _cv2()
    rng = np.random.default_rng(0)
    mask = rng.random((37, 53)) > 0.93
    kernel = np.ones((abs(pixels), abs(pixels)), np.uint8)
    op = cv2.dilate if pixels > 0 else cv2.erode
    expected = op(mask.astype(np.uint8), kernel, iterations = 1) > 0
    assert (det.dilate(mask, pixels) == expected).all()


def test_dilate_by_zero_or_one_is_a_copy():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    for px in (0, 1, -1):
        out = det.dilate(mask, px)
        assert (out == mask).all() and out is not mask


@pytest.mark.parametrize("feather,shape", [(5, (64, 80)), (3, (40, 33)), (5, (9, 30))])
def test_feather_matches_torchvision_gaussian_blur(feather, shape):
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision")
    mask = np.zeros(shape, np.float32)
    mask[shape[0] // 4 : shape[0] * 3 // 4, shape[1] // 3 : shape[1] * 2 // 3] = 1.0
    size = feather * 2 + 1
    shortest = min(shape)
    if shortest <= size:
        size = int(shortest / 2)
        size += size % 2 == 0
    expected = (
        tv.transforms.GaussianBlur(kernel_size = size, sigma = 10.0)(
            torch.from_numpy(mask)[None, None]
        )[0, 0]
        .numpy()
        .clip(0, 1)
    )
    assert np.abs(det.feather_mask(mask, feather) - expected).max() < 1e-5


def test_feather_skips_a_tiny_crop_and_zero():
    # A 3 px short side halves to a 1 px kernel, under 3: Impact-Pack skips the blur.
    mask = np.zeros((3, 9), np.float32)
    mask[1, 4] = 1.0
    assert (det.feather_mask(mask, 5) == mask).all()
    assert (det.feather_mask(np.ones((40, 40), np.float32), 0) == 1.0).all()


def test_composite_keeps_zero_alpha_pixels_byte_for_byte():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (16, 16, 3), dtype = np.uint8)
    patch = rng.integers(0, 256, (16, 16, 3), dtype = np.uint8)
    alpha = np.zeros((16, 16), np.float32)
    alpha[4:12, 4:12] = 1.0
    alpha[3, 3:13] = 0.25
    out = composite = det.composite(base, patch, alpha)
    assert (composite[alpha == 0] == base[alpha == 0]).all()
    assert (out[alpha == 1] == patch[alpha == 1]).all()
    blended = np.rint(base[3, 5] * 0.75 + patch[3, 5] * 0.25)
    assert (out[3, 5] == blended).all()


def test_targets_take_the_detailer_count_unless_named():
    assert det.target_concepts("face") == [("face", det.DEFAULT_PER_TARGET)]
    assert det.target_concepts("face:2, (hand), eyes:99") == [
        ("face", 2),
        ("hand", det.DEFAULT_PER_TARGET),
        ("eyes", det.MAX_PER_TARGET),
    ]
    assert det.segment_prompt(" face , hand:3 ") == f"face:{det.DEFAULT_PER_TARGET}, hand:3"
    assert det.target_concepts(" , ") == []
    with pytest.raises(ValueError):
        det.DetailerSettings.from_request(types.SimpleNamespace(targets = " , "))


def test_detection_prompt_follows_adetailer():
    assert det.detection_prompt(None, "a man", 0) == "a man"
    assert det.detection_prompt("", "a man", 3) == "a man"
    assert det.detection_prompt("detailed face, [PROMPT]", "a man", 0) == "detailed face, a man"
    template = "smiling [SEP] [SKIP] [SEP]  [SEP] old face"
    assert [det.detection_prompt(template, "p", i) for i in range(6)] == [
        "smiling",
        None,
        "p",
        "old face",
        "old face",
        "old face",
    ]


def test_detection_seed_is_seed_plus_index():
    assert [det.detection_seed(42, i) for i in range(3)] == [42, 43, 44]
    assert det.detection_seed(2**53 - 1, 1) == 0


def _scene(width = 256, height = 192):
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype = np.uint8))


def _face(
    box,
    width = 256,
    height = 192,
):
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), bool)
    yy, xx = np.ogrid[:height, :width]
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    mask[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1] = True
    return det.Detection("face", 0.9, box, mask)


class _FakeInpaint:
    """Paints the masked area with a seed-derived colour, as an inpaint would; records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, crop, mask, seed, prompt, negative):
        self.calls.append(
            {
                "size": crop.size,
                "mask_size": mask.size,
                "seed": seed,
                "prompt": prompt,
                "negative": negative,
            }
        )
        arr = np.array(crop)
        m = np.array(mask) > 127
        arr[m] = [(seed * 37) % 256, (seed * 91) % 256, (seed * 13) % 256]
        return Image.fromarray(arr)


def test_pass_changes_only_inside_the_feathered_masks():
    image = _scene()
    detections = [_face((40, 30, 80, 80)), _face((150, 60, 200, 120))]
    settings = det.DetailerSettings()
    fake = _FakeInpaint()
    out = det.detail_image(
        image,
        detections,
        settings,
        base_seed = 100,
        prompt = "two people",
        negative_prompt = "blurry",
        inpaint = fake,
    )
    before, after = np.array(image), np.array(out.image)
    # Where each feathered paste mask can reach: the dilated mask, blurred, inside its crop.
    reach = np.zeros(before.shape[:2], bool)
    for d in detections:
        x1, y1, x2, y2 = det.crop_region(256, 192, d.box, settings.crop_factor)
        local = det.feather_mask(
            det.dilate(d.mask[y1:y2, x1:x2], settings.dilation), settings.feather
        )
        reach[y1:y2, x1:x2] |= local > 0
    changed = (before != after).any(axis = 2)
    assert changed.any()
    assert not changed[~reach].any()
    # Sequential seeds, the image's prompts, sizes on the 16 px grid, the enlarged box short side near guide_size.
    assert [c["seed"] for c in fake.calls] == [100, 101]
    assert all(c["prompt"] == "two people" and c["negative"] == "blurry" for c in fake.calls)
    for call in fake.calls:
        w, h = call["size"]
        assert w % 16 == 0 and h % 16 == 0 and call["mask_size"] == call["size"]
        assert max(w, h) <= settings.max_size
    record = out.record
    assert [d["seed"] for d in record["detections"]] == [100, 101]
    assert record["found"] == 2 and record["denoise"] == 0.5 and record["targets"] == "face"
    assert record["detections"][0]["crop"] == list(det.crop_region(256, 192, (40, 30, 80, 80), 3.0))
    assert "prompt" not in record["detections"][0]


def test_pass_is_deterministic():
    image = _scene()
    detections = [_face((40, 30, 80, 80)), _face((150, 60, 200, 120))]
    runs = [
        det.detail_image(
            image,
            detections,
            det.DetailerSettings(),
            base_seed = 5,
            prompt = "p",
            negative_prompt = None,
            inpaint = _FakeInpaint(),
        )
        for _ in range(2)
    ]
    assert np.array_equal(np.array(runs[0].image), np.array(runs[1].image))
    assert runs[0].record["detections"] == runs[1].record["detections"]


def test_later_detections_see_earlier_results():
    """Overlapping crops: the second crop is read from the image the first paste left (Impact-Pack re-crops)."""
    image = _scene()
    seen = []

    def _inpaint(crop, mask, seed, prompt, negative):
        seen.append(np.array(crop).copy())
        return _FakeInpaint()(crop, mask, seed, prompt, negative)

    detections = [_face((60, 40, 100, 90)), _face((90, 50, 130, 100))]
    det.detail_image(
        image,
        detections,
        det.DetailerSettings(),
        base_seed = 1,
        prompt = "p",
        negative_prompt = None,
        inpaint = _inpaint,
    )
    first_colour = np.array([(1 * 37) % 256, (1 * 91) % 256, (1 * 13) % 256], np.uint8)
    assert (seen[1].reshape(-1, 3) == first_colour).all(axis = 1).any()


def test_small_skipped_and_empty_detections():
    image = _scene()
    tiny = _face((10, 10, 20, 40))  # 10 px wide: drop_size 10 drops it (the side must be > 10)
    empty = det.Detection("face", 0.8, (100, 100, 150, 150), np.zeros((192, 256), bool))
    keep = _face((150, 60, 200, 120))
    fake = _FakeInpaint()
    out = det.detail_image(
        image,
        [tiny, empty, keep],
        det.DetailerSettings(prompt = "a [SEP] b"),
        base_seed = 9,
        prompt = "p",
        negative_prompt = None,
        inpaint = fake,
    )
    # Undersized boxes are dropped before indexing (Impact-Pack's detectors); an empty mask still takes its index.
    assert [c["seed"] for c in fake.calls] == [10]
    assert fake.calls[0]["prompt"] == "b"
    assert out.record["skipped"] == 2 and out.record["detections"][0]["prompt"] == "b"


def test_undersized_detections_take_no_prompt_or_seed():
    fake = _FakeInpaint()
    det.detail_image(
        _scene(),
        [_face((10, 10, 20, 40)), _face((150, 60, 200, 120))],
        det.DetailerSettings(prompt = "a [SEP] b"),
        base_seed = 9,
        prompt = "p",
        negative_prompt = None,
        inpaint = fake,
    )
    assert [(c["seed"], c["prompt"]) for c in fake.calls] == [(9, "a")]


def test_too_many_targets_are_refused_before_the_render():
    from models.inference import DiffusionDetailerSpec

    targets = ", ".join(f"thing{i}" for i in range(17))
    with pytest.raises(ValueError, match = "At most 16"):
        det.DetailerSettings.from_request(DiffusionDetailerSpec(targets = targets))
    sixteen = ", ".join(f"thing{i}" for i in range(16))
    assert (
        det.DetailerSettings.from_request(DiffusionDetailerSpec(targets = sixteen)).targets == sixteen
    )


def test_an_inpaint_on_another_build_of_the_same_repo_is_discarded(monkeypatch):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    image = _scene()

    def _generate(**kwargs):
        # Another tab reloaded the same repo with a different GGUF file while SAM 3 ran.
        return {"images": [_scene(64, 64)], "repo_id": "org/m", "gguf_filename": "q8.gguf"}

    result = {"images": [image], "seeds": [1], "repo_id": "org/m", "gguf_filename": "q4.gguf"}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert result["images"][0] is image
    assert result["detailer"][0]["error"] == "the model changed"


def test_same_build_compares_every_build_field():
    base = {"repo_id": "r", "model_kind": "gguf", "gguf_filename": "a", "baked_loras": ["x"]}
    assert det.same_build(base, {**base, "baked_loras": ("x",), "seed": 4})
    assert det.same_build(base, {"images": []})
    for key, value in [("gguf_filename", "b"), ("transformer_quant", "int8"), ("baked_loras", [])]:
        assert not det.same_build(
            {**base, "transformer_quant": None}, {**base, "transformer_quant": None, key: value}
        )


def test_detection_holds_the_engine_generation_slot(monkeypatch):
    import threading

    from core.inference import diffusion_engine_router as router
    from core.inference import diffusion_segment

    lock = threading.Lock()
    monkeypatch.setattr(
        router, "get_active_diffusion_engine", lambda: types.SimpleNamespace(_generate_lock = lock)
    )
    held = []

    class _Seg:
        def segment(self, *a, **k):
            held.append(("segment", lock.locked()))
            return types.SimpleNamespace(
                instances = [], instance_logits = [], device = "cpu", precision = "fp32"
            )

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(
        diffusion_segment,
        "release_segmenter",
        lambda reason, wait = False, park = False: held.append(("park", lock.locked())),
    )
    monkeypatch.setattr(
        diffusion_segment, "active_variant", lambda: types.SimpleNamespace(id = "sam3.1")
    )
    det.detect([_scene()], det.DetailerSettings())
    # A queued render cannot take the slot until SAM 3 is parked.
    assert held == [("segment", True), ("park", True)] and not lock.locked()


def test_detection_registers_as_the_generation_slot_holder(monkeypatch):
    # Without a registered slot holder, generation_is_foreign() refuses the owner's cancel.
    import threading

    from core.inference import diffusion_engine_router as router
    from core.inference import diffusion_segment
    from hub.services.models import account_access
    from utils.account_context import AccountContext, bind_account, reset_account

    lock = threading.Lock()
    monkeypatch.setattr(
        router, "get_active_diffusion_engine", lambda: types.SimpleNamespace(_generate_lock = lock)
    )
    monkeypatch.setattr(account_access, "account_scope", lambda: "acct-a")
    seen = []

    class _Seg:
        def segment(self, *a, **k):
            seen.append(
                (
                    list(account_access._generation_holders.get("diffusion", [])),
                    account_access.generation_is_foreign("diffusion"),
                )
            )
            return types.SimpleNamespace(
                instances = [], instance_logits = [], device = "cpu", precision = "fp32"
            )

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(diffusion_segment, "release_segmenter", lambda *a, **k: None)
    monkeypatch.setattr(
        diffusion_segment, "active_variant", lambda: types.SimpleNamespace(id = "sam3.1")
    )
    token = bind_account(AccountContext("acct-a", "a"))
    try:
        with account_access.media_generation("diffusion"):
            # Account B is queued on the engine lock: counted, not holding.
            other = bind_account(AccountContext("acct-b", "b"))
            try:
                with account_access.media_generation("diffusion"):
                    reset_account(other)
                    other = None
                    det.detect([_scene()], det.DetailerSettings())
            finally:
                if other is not None:
                    reset_account(other)
    finally:
        reset_account(token)
    assert seen == [(["acct-a"], False)]
    assert "diffusion" not in account_access._generation_holders


def test_detection_waiting_for_the_slot_honours_stop(monkeypatch):
    import threading

    from core.inference import diffusion_engine_router as router
    from core.inference import diffusion_segment

    lock = threading.Lock()
    lock.acquire()  # a queued render denoises
    monkeypatch.setattr(
        router, "get_active_diffusion_engine", lambda: types.SimpleNamespace(_generate_lock = lock)
    )
    monkeypatch.setattr(
        diffusion_segment, "get_segmenter", lambda: pytest.fail("SAM 3 loaded beside a render")
    )
    with pytest.raises(det.DetailerCancelled):
        det.detect([_scene()], det.DetailerSettings(), stop = lambda: True)
    lock.release()


def test_detections_beyond_the_cap_are_dropped():
    image = _scene()
    detections = [_face((150, 60, 200, 120))] * (det.MAX_DETECTIONS + 3)
    fake = _FakeInpaint()
    out = det.detail_image(
        image,
        detections,
        det.DetailerSettings(),
        base_seed = 0,
        prompt = "p",
        negative_prompt = None,
        inpaint = fake,
    )
    assert len(fake.calls) == det.MAX_DETECTIONS and out.record["dropped"] == 3


def test_overrides_reach_the_record_and_the_negative_prompt():
    settings = det.DetailerSettings(
        negative_prompt = "[SKIP]x", steps = 12, guidance = 4.5, prompt = "sharp face"
    )
    fake = _FakeInpaint()
    out = det.detail_image(
        _scene(),
        [_face((150, 60, 200, 120))],
        settings,
        base_seed = 0,
        prompt = "p",
        negative_prompt = "n",
        inpaint = fake,
    )
    assert fake.calls[0]["negative"] == "[SKIP]x"
    record = out.record
    assert record["steps"] == 12 and record["guidance"] == 4.5 and record["prompt"] == "sharp face"


def _patch_detect(monkeypatch, per_image):
    monkeypatch.setattr(
        det,
        "detect",
        lambda images, settings, stop = None: (
            per_image[: len(images)],
            {"model": "sam3.1", "device": "cpu", "detect_ms": 1.0},
        ),
    )


def test_detail_result_runs_every_image_with_its_own_seed(monkeypatch):
    _patch_detect(
        monkeypatch,
        [[_face((150, 60, 200, 120))], [_face((40, 30, 80, 80)), _face((150, 60, 200, 120))]],
    )
    calls = []

    def _generate(**kwargs):
        calls.append(kwargs)
        crop = Image.open(
            __import__("io").BytesIO(
                __import__("base64").b64decode(kwargs["init_image"].split(",", 1)[1])
            )
        )
        return {"images": [crop.convert("RGB")]}

    result = {"images": [_scene(), _scene()], "seeds": [10, 20], "seed": 10}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(steps = 6),
        prompt_for = lambda i: f"prompt {i}",
        negative_prompt = None,
        steps = 9,
        guidance = 0.0,
        generate_kwargs = {"expected_load": "L", "loras": None},
    )
    assert [c["seed"] for c in calls] == [10, 20, 21]
    assert [c["prompt"] for c in calls] == ["prompt 0", "prompt 1", "prompt 1"]
    assert all(
        c["steps"] == 6 and c["strength"] == 0.5 and c["expected_load"] == "L" for c in calls
    )
    assert all(c["mask_image"].startswith("data:image/png;base64,") for c in calls)
    assert [len(r["detections"]) for r in result["detailer"]] == [1, 2]
    assert result["detailer"][0]["model"] == "sam3.1"


def test_every_inpaint_runs_as_an_auxiliary_eager_unrecorded_pass(monkeypatch):
    from core.inference import diffusion_bg_compile as bg

    _patch_detect(monkeypatch, [[_face((40, 30, 80, 80)), _face((150, 60, 230, 150))]])
    seen = []

    def _generate(**kwargs):
        seen.append(
            (bg.eager_forced(), bg._NO_RECORD.get(), getattr(bg, "auxiliary", lambda: False)())
        )
        crop = Image.open(
            __import__("io").BytesIO(
                __import__("base64").b64decode(kwargs["init_image"].split(",", 1)[1])
            )
        )
        return {"images": [crop.convert("RGB")]}

    result = {"images": [_scene()], "seeds": [3], "seed": 3}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 8,
        guidance = 0.0,
        generate_kwargs = {},
    )
    assert seen == [(True, True, True), (True, True, True)]
    assert not bg.eager_forced() and not bg._NO_RECORD.get()


def test_a_failed_pass_keeps_the_render(monkeypatch):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    image = _scene()

    def _generate(**kwargs):
        raise RuntimeError("CUDA out of memory")

    result = {"images": [image], "seeds": [1]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert result["images"][0] is image
    assert result["detailer"][0]["error"] == "CUDA out of memory"


def test_cancel_stops_the_pass(monkeypatch):
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))], [_face((150, 60, 200, 120))]])
    calls = []

    def _generate(**kwargs):
        calls.append(1)
        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    result = {"images": [_scene(), _scene()], "seeds": [1, 2]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert len(calls) == 1
    assert [r["error"] for r in result["detailer"]] == ["cancelled", "cancelled"]


def test_detection_failure_is_recorded(monkeypatch):
    def _boom(
        images,
        settings,
        stop = None,
    ):
        raise RuntimeError("SAM 3 failed")

    monkeypatch.setattr(det, "detect", _boom)
    result = {"images": [_scene()], "seeds": [1]}
    det.detail_result(
        lambda **k: None,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert result["detailer"][0]["error"] == "SAM 3 failed"


def test_detect_releases_sam3_even_on_failure(monkeypatch):
    from core.inference import diffusion_segment

    released = []

    class _Seg:
        def segment(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(
        diffusion_segment,
        "release_segmenter",
        lambda reason, wait = False, park = False: released.append((reason, wait, park)),
    )
    with pytest.raises(RuntimeError):
        det.detect([_scene()], det.DetailerSettings())
    assert released == [("detailer", True, True)]


def test_detect_builds_full_size_instance_masks(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_segment

    plane = torch.full((8, 8), -1.0)
    plane[2:6, 2:6] = 1.0
    seen = {}

    class _Seg:
        def segment(
            self,
            image,
            prompt,
            threshold,
            instance_masks,
            plane_filter = None,
            max_planes = None,
        ):
            seen.update(prompt = prompt, threshold = threshold, instance_masks = instance_masks)
            return types.SimpleNamespace(
                instances = [diffusion_segment.Instance("face", 0.9, (64.0, 48.0, 192.0, 144.0))],
                instance_logits = [plane],
                device = "cpu",
                precision = "bf16",
            )

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(
        diffusion_segment, "release_segmenter", lambda reason, wait = False, park = False: True
    )
    found, info = det.detect(
        [_scene()], det.DetailerSettings(targets = "face, hand:2", confidence = 0.3)
    )
    assert seen == {
        "prompt": f"face:{det.DEFAULT_PER_TARGET}, hand:2",
        "threshold": 0.3,
        "instance_masks": True,
    }
    # Kept at SAM 3's resolution until the detection is processed, then expanded to the image's size.
    assert found[0][0].mask is None
    mask = found[0][0].full_mask(192, 256)
    assert mask.shape == (192, 256) and mask.dtype == bool
    assert mask[96, 128] and not mask[5, 5]
    assert info["device"] == "cpu" and info["found"] == [1]


def _many_detections_segmenter(count, calls):
    import torch

    from core.inference import diffusion_segment

    class _Seg:
        def segment(
            self,
            image,
            prompt,
            threshold,
            instance_masks,
            plane_filter = None,
            max_planes = None,
        ):
            calls.append(image)
            return types.SimpleNamespace(
                instances = [
                    diffusion_segment.Instance("face", 0.9, (10.0 + i, 10.0, 60.0 + i, 70.0))
                    for i in range(count)
                ],
                instance_logits = [torch.zeros(8, 8) for _ in range(count)],
                device = "cpu",
                precision = "bf16",
            )

    return _Seg()


def test_detection_keeps_only_the_pass_cap_and_no_full_size_masks(monkeypatch):
    """16 targets x 20 detections is 320 masks; only the first MAX_DETECTIONS are kept, none expanded yet."""
    pytest.importorskip("torch")
    from core.inference import diffusion_segment

    calls = []
    monkeypatch.setattr(
        diffusion_segment, "get_segmenter", lambda: _many_detections_segmenter(320, calls)
    )
    monkeypatch.setattr(
        diffusion_segment, "release_segmenter", lambda reason, wait = False, park = False: True
    )
    found, info = det.detect([_scene(), _scene()], det.DetailerSettings())
    assert [len(f) for f in found] == [det.MAX_DETECTIONS] * 2
    assert info["found"] == [320, 320]
    assert all(d.mask is None and d.plane is not None for f in found for d in f)


def test_detail_result_records_what_detection_dropped(monkeypatch):
    found = [[_face((150, 60, 200, 120))]]
    monkeypatch.setattr(
        det,
        "detect",
        lambda images, settings, stop = None: (found, {"model": "sam3.1", "found": [40]}),
    )
    result = {"images": [_scene()], "seeds": [1]}
    det.detail_result(
        lambda **k: {"images": [Image.new("RGB", (k["width"], k["height"]))]},
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    record = result["detailer"][0]
    assert record["found"] == 40 and record["dropped"] == 39


def test_stop_during_detection_skips_the_remaining_images(monkeypatch):
    """Stop pressed while SAM 3 runs on a batch: no further image is segmented and no inpaint runs."""
    pytest.importorskip("torch")
    from core.inference import diffusion_segment

    calls, released = [], []
    segmenter = _many_detections_segmenter(1, calls)
    real_segment = segmenter.segment

    def _segment(*a, **k):
        out = real_segment(*a, **k)
        assert det.cancel_active() is True  # Stop arrives during the first image's detection
        return out

    segmenter.segment = _segment
    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: segmenter)
    monkeypatch.setattr(
        diffusion_segment,
        "release_segmenter",
        lambda reason, wait = False, park = False: released.append(park),
    )
    inpaints = []
    result = {"images": [_scene() for _ in range(4)], "seeds": [1, 2, 3, 4]}
    det.detail_result(
        lambda **k: inpaints.append(1),
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert len(calls) == 1 and inpaints == [] and released == [True]
    assert [r["error"] for r in result["detailer"]] == ["cancelled"] * 4


def test_infotext_uses_adetailer_keys():
    record = {
        **det.settings_record(det.DetailerSettings(prompt = "detailed face, smile", steps = 12)),
        "model": "sam3.1",
    }
    text = det.infotext_fields(record)
    assert text == (
        ", ADetailer model: sam3.1, ADetailer model classes: face"
        ', ADetailer prompt: "detailed face, smile"'
        ", ADetailer confidence: 0.5, ADetailer dilate erode: 10, ADetailer mask blur: 5"
        ", ADetailer denoising strength: 0.5, ADetailer inpaint only masked: True"
        ", ADetailer use separate steps: True, ADetailer steps: 12"
    )
    assert det.infotext_fields(None) == ""
    assert det.infotext_fields({**record, "error": "cancelled"}) == ""


def test_gallery_parameters_carry_the_detailer_only_when_it_ran():
    from core.inference import image_gallery

    meta = {
        "prompt": "p",
        "steps": 9,
        "guidance": 0.0,
        "seed": 1,
        "width": 512,
        "height": 512,
        "model": "m",
    }
    plain = image_gallery._params_text(meta)
    assert "ADetailer" not in plain
    record = {**det.settings_record(det.DetailerSettings()), "model": "sam3.1", "detections": []}
    text = image_gallery._params_text({**meta, "detailer": record})
    assert text.startswith(plain.split(", Version")[0])
    assert "ADetailer model: sam3.1" in text and "ADetailer denoising strength: 0.5" in text


def test_gallery_image_accepts_the_detailer_record():
    from models.inference import GalleryImage

    record = {**det.settings_record(det.DetailerSettings()), "detections": [{"seed": 1}]}
    image = GalleryImage(
        id = "x",
        url = "/u",
        prompt = "p",
        width = 64,
        height = 64,
        steps = 1,
        guidance = 0.0,
        seed = 1,
        created_at = 0.0,
        detailer = record,
    )
    assert image.detailer["targets"] == "face"


class _Backend:
    def __init__(self, workflows):
        self._status = {"loaded": True, "workflows": workflows, "repo_id": "r"}

    def status(self):
        return self._status


def _gate(
    monkeypatch,
    workflows,
    *,
    sam3 = True,
    downloaded = True,
):
    from fastapi import HTTPException

    from core.inference import diffusion_segment
    from models.inference import DiffusionDetailerSpec
    from routes import inference as routes

    monkeypatch.setattr(diffusion_segment, "transformers_supports_sam3", lambda: sam3)
    monkeypatch.setattr(diffusion_segment, "model_dir", lambda: "/sam" if downloaded else None)
    try:
        return routes._detailer_settings(_Backend(workflows), DiffusionDetailerSpec()), None
    except HTTPException as exc:
        return None, exc


def test_gate_admits_an_inpaint_model_with_sam3(monkeypatch):
    settings, error = _gate(monkeypatch, ["txt2img", "inpaint"])
    assert error is None and settings == det.DetailerSettings()


@pytest.mark.parametrize(
    "workflows,sam3,downloaded,needle",
    [
        (["txt2img", "img2img"], True, True, "inpaint pipeline"),
        (["edit"], True, True, "inpaint pipeline"),
        (["txt2img", "inpaint"], False, True, "transformers 5"),
        (["txt2img", "inpaint"], True, False, "not downloaded"),
    ],
)
def test_gate_refuses_with_400(monkeypatch, workflows, sam3, downloaded, needle):
    _, error = _gate(monkeypatch, workflows, sam3 = sam3, downloaded = downloaded)
    assert error is not None and error.status_code == 400 and needle in error.detail


def test_request_without_detailer_is_unchanged():
    from models.inference import DiffusionGenerateRequest

    request = DiffusionGenerateRequest(prompt = "p")
    assert request.detailer is None
    assert "detailer" not in request.model_dump(exclude_none = True)


def test_stop_between_two_inpaints_ends_the_pass(monkeypatch):
    """Stop pressed while no denoise runs (between two face inpaints): the next detection is not inpainted."""
    _patch_detect(
        monkeypatch,
        [[_face((40, 30, 80, 80)), _face((150, 60, 200, 120))], [_face((150, 60, 200, 120))]],
    )
    calls = []

    def _generate(**kwargs):
        import base64
        import io

        calls.append(kwargs["seed"])
        # The first inpaint finishes, then the user presses Stop before the second starts.
        assert det.cancel_active() is True
        crop = Image.open(io.BytesIO(base64.b64decode(kwargs["init_image"].split(",", 1)[1])))
        return {"images": [crop.convert("RGB")]}

    first, second = _scene(), _scene()
    result = {"images": [first, second], "seeds": [1, 2]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert calls == [1]
    assert [r["error"] for r in result["detailer"]] == ["cancelled", "cancelled"]
    assert result["images"][0] is first and result["images"][1] is second
    assert det.cancel_active() is False


def test_cancel_route_stops_a_detailer_pass(monkeypatch):
    import asyncio
    import sys

    import core.inference.diffusion_engine_router as router
    import core.inference.flux3_prompt_upsampler as pup
    from routes import inference as routes

    stopped = []
    fake = types.SimpleNamespace(cancel_active = lambda account = None: stopped.append(1) or True)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_detailer", fake)
    monkeypatch.setattr(routes.account_access, "generation_is_foreign", lambda kind: False)
    monkeypatch.setattr(routes.account_access, "generation_is_mine", lambda kind: True)
    monkeypatch.setattr(routes.account_access, "tracked_generation_account", lambda: None)
    monkeypatch.setattr(pup, "cancel_active", lambda: False)
    monkeypatch.setattr(
        router,
        "get_active_diffusion_engine",
        lambda: types.SimpleNamespace(cancel_generate = lambda: False),
    )
    out = asyncio.run(routes.cancel_diffusion_generation(target = None, current_subject = "u"))
    assert out == {"cancelled": True} and stopped == [1]


def test_route_runs_the_pass_on_a_display_id_load_and_stays_in_flight(monkeypatch, tmp_path):
    """A load shown under a display id (a local snapshot under its Hub id) is the same model, and generate-progress
    / liveness stay active from the end of the render through the Detailer and the persist."""
    import asyncio

    from core.inference import diffusion_engine_router as router
    from core.inference import image_gallery
    from models.inference import DiffusionDetailerSpec, DiffusionGenerateRequest
    from routes import inference as routes

    class _Backend:
        def status(self):
            return {
                "loaded": True,
                "workflows": ["txt2img", "inpaint"],
                "repo_id": "/models/snap",
                "display_repo_id": "org/model",
                "family": "z-image",
                "base_repo": None,
            }

        def generate(self, **kwargs):
            return {"images": [_scene()], "seed": 3, "seeds": [3], "repo_id": "org/model"}

    seen = {}

    def _detail_result(generate, result, settings, **kwargs):
        seen["in_flight"] = routes.generation_in_flight()
        result["detailer"] = [{"targets": "face", "detections": []}]

    monkeypatch.setattr(router, "get_active_diffusion_engine", lambda: _Backend())
    monkeypatch.setattr(det, "check_available", lambda status: None)
    monkeypatch.setattr(det, "detail_result", _detail_result)
    monkeypatch.setattr(routes, "_release_mask_segmenter_for_render", lambda: None)
    monkeypatch.setattr(
        image_gallery,
        "save",
        lambda image, meta: {**meta, "id": "x", "url": "/u", "created_at": 0.0},
    )
    request = DiffusionGenerateRequest(
        prompt = "p", steps = 4, guidance = 0.0, seed = 3, detailer = DiffusionDetailerSpec()
    )
    response = asyncio.run(routes._generate_diffusion_image_core(request))
    assert seen == {"in_flight": True}
    assert response.images[0].detailer == {"targets": "face", "detections": []}
    assert routes.generation_in_flight() is False


def _crop_back(kwargs):
    import base64
    import io
    return Image.open(io.BytesIO(base64.b64decode(kwargs["init_image"].split(",", 1)[1]))).convert(
        "RGB"
    )


def test_cancel_stops_only_the_callers_pass(monkeypatch):
    _patch_detect(monkeypatch, [[_face((40, 30, 80, 80)), _face((150, 60, 200, 120))]])
    calls = []

    def _generate(**kwargs):
        calls.append(kwargs["seed"])
        # Another account presses Stop: this account's pass carries on.
        assert det.cancel_active("someone-else") is False
        return {"images": [_crop_back(kwargs)]}

    result = {"images": [_scene()], "seeds": [1]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
        account = "me",
    )
    assert calls == [1, 2] and "error" not in result["detailer"][0]

    calls.clear()

    def _generate_own(**kwargs):
        calls.append(kwargs["seed"])
        assert det.cancel_active("me") is True
        return {"images": [_crop_back(kwargs)]}

    result = {"images": [_scene()], "seeds": [1]}
    det.detail_result(
        _generate_own,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
        account = "me",
    )
    assert calls == [1] and result["detailer"][0]["error"] == "cancelled"


def test_stop_after_the_last_inpaint_returns_keeps_the_render(monkeypatch):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    render = _scene()

    def _generate(**kwargs):
        out = {"images": [_crop_back(kwargs).point(lambda v: 255 - v)]}
        assert det.cancel_active() is True  # Stop lands while the patch is still being composited
        return out

    result = {"images": [render], "seeds": [1]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert result["images"][0] is render
    assert result["detailer"][0]["error"] == "cancelled"
    assert det.cancel_active() is False


def test_undersized_detections_take_no_cap_slot(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_segment

    tiny = [
        diffusion_segment.Instance("eye", 0.9, (1.0, 1.0, 5.0, 5.0))
        for _ in range(det.MAX_DETECTIONS)
    ]
    big = [diffusion_segment.Instance("face", 0.8, (10.0, 10.0, 60.0, 70.0)) for _ in range(20)]

    class _Seg:
        def segment(
            self,
            image,
            prompt,
            threshold,
            instance_masks,
            plane_filter = None,
            max_planes = None,
        ):
            return types.SimpleNamespace(
                instances = tiny + big,
                instance_logits = [torch.zeros(8, 8) for _ in tiny + big],
                device = "cpu",
                precision = "bf16",
            )

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(
        diffusion_segment, "release_segmenter", lambda reason, wait = False, park = False: True
    )
    found, info = det.detect([_scene()], det.DetailerSettings())
    assert [d.concept for d in found[0]] == ["face"] * 20
    assert info["found"] == [det.MAX_DETECTIONS + 20] and info["undersized"] == [det.MAX_DETECTIONS]


def test_detect_keeps_sam_loaded_across_the_batch(monkeypatch):
    from core.inference import diffusion_segment

    events = []
    seg = diffusion_segment.MaskSegmenter(
        keep_alive_seconds = 0
    )  # keep-alive off: every call would unload
    seg.device = "cpu"
    monkeypatch.setattr(seg, "_unload_locked", lambda reason: events.append(("unload", reason)))

    def _segment(
        image,
        prompt,
        threshold,
        instance_masks,
        plane_filter = None,
        max_planes = None,
    ):
        seg._after_call()  # what every real call ends with
        events.append(("segment", None))
        return types.SimpleNamespace(
            instances = [], instance_logits = [], device = "cpu", precision = "bf16"
        )

    seg.segment = _segment
    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: seg)
    monkeypatch.setattr(
        diffusion_segment,
        "release_segmenter",
        lambda reason, wait = False, park = False: events.append(("park", reason)),
    )
    det.detect([_scene(), _scene(), _scene()], det.DetailerSettings())
    assert events == [("segment", None)] * 3 + [("park", "detailer")]
    seg._after_call()
    assert events[-1] == ("unload", "no keep-alive")


def test_hold_still_honours_a_release_asked_mid_call(monkeypatch):
    from core.inference import diffusion_segment

    seg = diffusion_segment.MaskSegmenter()
    unloads = []
    monkeypatch.setattr(seg, "_unload_locked", lambda reason: unloads.append(reason))
    with seg.hold():
        seg._release_pending = True
        seg._after_call()
    assert unloads == ["released during a call"]


def test_inherited_negative_prompt_is_recorded_when_the_detailer_cfg_engages_it(monkeypatch):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    seen = []

    def _generate(**kwargs):
        seen.append(kwargs["negative_prompt"])
        return {"images": [_crop_back(kwargs)], "negative_prompt": kwargs["negative_prompt"]}

    # The base render's CFG was too low for the negative prompt, so its result records none.
    result = {"images": [_scene()], "seeds": [1], "negative_prompt": None}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(guidance = 5.0),
        prompt_for = lambda i: "p",
        negative_prompt = "blurry",
        steps = 4,
        guidance = 1.0,
        generate_kwargs = {},
    )
    assert seen == ["blurry"] and result["detailer"][0]["negative_prompt"] == "blurry"


def test_route_passes_sd_controls_submitted_negative_and_account(monkeypatch):
    from core.inference.diffusion_sd_controls import SdControls
    from models.inference import DiffusionGenerateRequest
    from routes import inference as routes

    captured = {}

    class _Backend:
        def status(self):
            return {"loaded": True, "repo_id": "org/m", "family": "sdxl", "base_repo": None}

        def generate(
            self,
            sd_controls = None,
            **kwargs,
        ):
            raise AssertionError("not called")

    def _detail_result(generate, result, settings, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(det, "detail_result", _detail_result)
    monkeypatch.setattr(routes.account_access, "tracked_generation_account", lambda: "acct-1")
    request = DiffusionGenerateRequest(
        prompt = "p",
        negative_prompt = "blurry",
        steps = 4,
        guidance = 1.0,
        prompt_weighting = False,
        clip_skip = 2,
        cfg_rescale = 0.7,
        variation_seed = 5,
        variation_strength = 0.3,
    )
    result = {"images": [_scene()], "repo_id": "org/m", "negative_prompt": None}
    enhancement = types.SimpleNamespace(encoded = lambda text: text)
    routes._run_detailer(_Backend(), request, result, enhancement, det.DetailerSettings())
    controls = captured["generate_kwargs"]["sd_controls"]
    assert isinstance(controls, SdControls)
    assert (controls.prompt_weighting, controls.clip_skip, controls.cfg_rescale) == (False, 2, 0.7)
    assert controls.variation_seed is None and not controls.variation_strength
    assert captured["negative_prompt"] == "blurry" and captured["account"] == "acct-1"


def test_a_low_denoise_still_schedules_one_inpaint_timestep(monkeypatch):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    calls = []

    def _generate(**kwargs):
        calls.append(kwargs)
        # SDXL inpaint keeps int(steps * strength) timesteps and refuses zero.
        if int(kwargs["steps"] * kwargs["strength"]) < 1:
            raise ValueError("num_inference_steps < 1")
        return {"images": [_crop_back(kwargs)]}

    result = {"images": [_scene()], "seeds": [1]}
    det.detail_result(
        _generate,
        result,
        det.DetailerSettings(denoise = 0.25),
        prompt_for = lambda i: "p",
        negative_prompt = None,
        steps = 3,  # SDXL Turbo's default
        guidance = 0.0,
        generate_kwargs = {},
    )
    assert not result["detailer"][0].get("error")
    assert calls[0]["steps"] == 4
    assert det.schedulable_steps(1, 0.5) == 2
    assert det.schedulable_steps(9, 0.5) == 9
    assert det.schedulable_steps(3, 1.0) == 3


def _schedulable_steps_reference(steps, denoise):
    """The original count-up search, kept as the oracle for the closed form."""
    n = max(1, int(steps))
    if denoise <= 0.0 or denoise >= 1.0:
        return n
    while int(n * denoise) < 1:
        n += 1
    return n


def test_schedulable_steps_closed_form_matches_the_search_over_a_sweep():
    denoises = [i / 1000 for i in range(10, 1001)] + [
        0.01,
        0.05,
        1 / 3,
        2 / 3,
        0.1 + 0.2,
        0.999999,
        1.0,
    ]
    for steps in (0, 1, 2, 3, 4, 7, 20, 50, 99, 100, 101, 150):
        for d in denoises:
            assert det.schedulable_steps(steps, d) == _schedulable_steps_reference(steps, d), (
                steps,
                d,
            )


def test_schedulable_steps_is_bounded_for_a_tiny_denoise():
    # The count-up search would loop 10M times here; the closed form must not depend on 1 / denoise.
    import sys

    lines = [0]

    def tracer(frame, event, arg):
        if frame.f_code is det.schedulable_steps.__code__:
            if event == "line":
                lines[0] += 1
                if lines[0] > 200:
                    raise AssertionError("schedulable_steps is not bounded")
            return tracer
        return None

    sys.settrace(tracer)
    try:
        result = det.schedulable_steps(1, 1e-7)
    finally:
        sys.settrace(None)
    assert result == 10_000_000
    assert det.schedulable_steps(1, 1e-300) >= 1
    assert det.schedulable_steps(5, float("nan")) == 5


@pytest.mark.parametrize("denoise", [1e-7, 0.001, 0.0099, 0.0, -0.5])
def test_request_rejects_a_denoise_below_the_ui_floor(denoise):
    import pydantic

    from models.inference import DiffusionDetailerSpec

    with pytest.raises(pydantic.ValidationError):
        DiffusionDetailerSpec(denoise = denoise)
    assert DiffusionDetailerSpec(denoise = 0.01).denoise == 0.01


@pytest.mark.parametrize(
    "template, expected_seen, expected_saved",
    [
        ("[PROMPT], bad teeth", ["blurry, bad teeth"], "blurry, bad teeth"),
        (" [SEP] bad teeth", ["blurry"], "blurry [SEP] bad teeth"),
        ("bad teeth", ["bad teeth"], "bad teeth"),
    ],
)
def test_a_template_inheriting_the_negative_saves_it_resolved(
    monkeypatch, template, expected_seen, expected_saved
):
    _patch_detect(monkeypatch, [[_face((150, 60, 200, 120))]])
    seen = []

    def _generate(**kwargs):
        seen.append(kwargs["negative_prompt"])
        return {"images": [_crop_back(kwargs)], "negative_prompt": kwargs["negative_prompt"]}

    result = {"images": [_scene()], "seeds": [1], "negative_prompt": None}
    settings = det.DetailerSettings(guidance = 5.0, negative_prompt = template)
    det.detail_result(
        _generate,
        result,
        settings,
        prompt_for = lambda i: "p",
        negative_prompt = "blurry",
        steps = 4,
        guidance = 0.0,
        generate_kwargs = {},
    )
    assert seen == expected_seen
    saved = result["detailer"][0]["negative_prompt"]
    assert saved == expected_saved
    # Restored with no main negative (the base recorded none), the saved template expands the same.
    assert det.detection_prompt(saved, "", 0) == expected_seen[0]
