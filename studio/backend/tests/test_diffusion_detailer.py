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


# ── geometry, against Impact-Pack's own formulas ────────────────────────────────────────────────────────────────────
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
    # Near the bottom-right it ends at the edge.
    assert det.crop_region(1024, 768, (950, 700, 1020, 760), 3.0) == (814, 588, 1024, 768)
    # Larger than the picture: the whole picture.
    assert det.crop_region(200, 100, (50, 20, 150, 80), 3.0) == (0, 0, 200, 100)


def test_upscale_puts_the_box_short_side_at_guide_size():
    # 100x80 box in a 300x240 crop: 512 / 80 = 6.4 -> 1920x1536, capped at 1024 on the long side.
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "bbox", 1024)
    assert (w, h) == (1024, 819) and upscale == pytest.approx(6.4 * 1024 / 1920)
    # A loose cap: the box's short side lands on guide_size.
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "bbox", 4096)
    assert upscale == pytest.approx(6.4) and (w, h) == (1920, 1536)
    # guide_size_for = crop_region measures the crop instead.
    upscale, w, h = det.upscaled_size(100, 80, 300, 240, 512, "crop_region", 4096)
    assert upscale == pytest.approx(512 / 240) and h == 512


def test_a_big_target_renders_at_its_own_size():
    # force_inpaint: factor <= 1 never shrinks the crop.
    assert det.upscaled_size(600, 600, 1000, 1000, 512, "bbox", 1024) == (1.0, 1000, 1000)


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


# ── prompts, targets, seeds ─────────────────────────────────────────────────────────────────────────────────────────
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


# ── the pass ────────────────────────────────────────────────────────────────────────────────────────────────────────
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
    """Paints the masked area of the enlarged crop a colour derived from the seed, as a real inpaint would only change
    the masked area; records every call."""

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
    # The seed and the [SEP] part follow the detection's index, skipped ones included (as ADetailer counts them).
    assert [c["seed"] for c in fake.calls] == [11]
    assert fake.calls[0]["prompt"] == "b"
    assert out.record["skipped"] == 2 and out.record["detections"][0]["prompt"] == "b"


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


# ── detail_result: the batch, failures, cancel ──────────────────────────────────────────────────────────────────────
def _patch_detect(monkeypatch, per_image):
    monkeypatch.setattr(
        det,
        "detect",
        lambda images, settings: (
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
    def _boom(images, settings):
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
        lambda reason, wait = False: released.append((reason, wait)),
    )
    with pytest.raises(RuntimeError):
        det.detect([_scene()], det.DetailerSettings())
    assert released == [("detailer", True)]


def test_detect_builds_full_size_instance_masks(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_segment

    plane = torch.full((8, 8), -1.0)
    plane[2:6, 2:6] = 1.0
    seen = {}

    class _Seg:
        def segment(self, image, prompt, threshold, instance_masks):
            seen.update(prompt = prompt, threshold = threshold, instance_masks = instance_masks)
            return types.SimpleNamespace(
                instances = [diffusion_segment.Instance("face", 0.9, (64.0, 48.0, 192.0, 144.0))],
                instance_logits = [plane],
                device = "cpu",
                precision = "bf16",
            )

    monkeypatch.setattr(diffusion_segment, "get_segmenter", lambda: _Seg())
    monkeypatch.setattr(diffusion_segment, "release_segmenter", lambda reason, wait = False: True)
    found, info = det.detect(
        [_scene()], det.DetailerSettings(targets = "face, hand:2", confidence = 0.3)
    )
    assert seen == {
        "prompt": f"face:{det.DEFAULT_PER_TARGET}, hand:2",
        "threshold": 0.3,
        "instance_masks": True,
    }
    mask = found[0][0].mask
    assert mask.shape == (192, 256) and mask.dtype == bool
    assert mask[96, 128] and not mask[5, 5]
    assert info["device"] == "cpu"


# ── infotext and the gallery record ─────────────────────────────────────────────────────────────────────────────────
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


# ── route gating ────────────────────────────────────────────────────────────────────────────────────────────────────
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
