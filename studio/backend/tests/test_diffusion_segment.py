# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-to-mask (SAM 3) for the Images painters: ComfyUI's prompt syntax, the keep rule, the union mask at the
source size, the lifecycle (keep-alive, release, GPU arbiter hooks) and the routes. A tiny random-weight Sam3Model
built from a shrunk config runs the real forward on the CPU; no download and no network."""

from __future__ import annotations

import base64
import io
import re
import threading
import time
import types

import pytest

from core.inference import diffusion_segment as seg


@pytest.fixture(autouse = True)
def _bf16_weights(monkeypatch):
    """Load and download the bf16 model.safetensors whatever card the test host has: the variant and lifecycle tests
    here count its bytes and files. The pre-quantized choice is covered in test_diffusion_segment_quant.py."""
    monkeypatch.setenv("UNSLOTH_SAM3_PRECISION", "bf16")
    monkeypatch.delenv("UNSLOTH_SAM3_INT8_CONVROT", raising = False)


@pytest.fixture(autouse = True)
def _no_other_gpu_owner(monkeypatch):
    """Start every test with no GPU arbiter resident.

    choose_device() runs on the CPU when another owner holds the card (_other_owner_on), and
    the arbiter's state is module-global. A test elsewhere in the same xdist worker that leaves
    a CHAT owner behind turns the card tests here into ('cpu', torch.float32). Tests that need
    an owner set one themselves.
    """
    from core.inference import gpu_arbiter

    # Restore every correlated global together: acquire_for() moves owner, account and epoch as one,
    # or later tests pair mismatched fields.
    for name, empty in (
        ("_owner", None),
        ("_owner_account", None),
        ("_prior_account", None),
        ("_residents", {}),
        ("_accounts", {}),
        ("_media_needs", {}),
    ):
        monkeypatch.setattr(gpu_arbiter, name, empty)
    for counter in ("_owner_epoch", "_handovers"):
        monkeypatch.setattr(gpu_arbiter, counter, getattr(gpu_arbiter, counter))


def _png_b64(
    width: int = 64,
    height: int = 48,
    color = (200, 30, 30),
) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _decode_mask(data_url: str):
    from PIL import Image
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


# ── prompt syntax (comfy/text_encoders/sam3_clip.py _parse_prompts) ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Sofa:2", [("Sofa", 2)]),
        ("sofa", [("sofa", 1)]),
        (
            "Sofa:2, Red lamp, Window, Pillow:3",
            [("Sofa", 2), ("Red lamp", 1), ("Window", 1), ("Pillow", 3)],
        ),
        ("(red lamp), window", [("red lamp", 1), ("window", 1)]),
        ("a : 4 ", [("a", 4)]),
        ("a:2.6", [("a", 3)]),
        # Python's round(): half to even, as ComfyUI computes it.
        ("a:2.5, b:3.5", [("a", 2), ("b", 4)]),
        ("a:0", [("a", 1)]),
        (" , sofa,, ,", [("sofa", 1)]),
        ("note: red", [("note: red", 1)]),
        ("v:2.5.1", [("v:2.5.1", 1)]),
        ("grain:5000", [("grain", seg.MAX_DETECTIONS)]),
        ("", []),
        (None, []),
        ("sofa, sofa:3", [("sofa", 1), ("sofa", 3)]),
    ],
)
def test_parse_prompt_matches_comfyui(text, expected):
    assert [(c.text, c.max_detections) for c in seg.parse_prompt(text)] == expected


def test_parse_prompt_overflowing_count_is_a_malformed_suffix():
    # float() of hundreds of digits is inf and round(inf) raises OverflowError: read it like "v:2.5.1".
    huge = "9" * 400
    assert [(c.text, c.max_detections) for c in seg.parse_prompt(f"object:{huge}, cat")] == [
        (f"object:{huge}", 1),
        ("cat", 1),
    ]


def test_parse_prompt_caps_the_concept_count():
    with pytest.raises(ValueError, match = "At most"):
        seg.parse_prompt(",".join(f"thing{i}" for i in range(seg.MAX_CONCEPTS + 1)))


# ── keep rule and mask ──────────────────────────────────────────────────────────────────────────────────────────────
def test_select_detections_thresholds_sorts_and_caps():
    import torch

    logits = torch.tensor([0.0, 3.0, -3.0, 2.0, 4.0])  # sigmoid: .5, .95, .05, .88, .98
    boxes = torch.arange(20, dtype = torch.float32).reshape(5, 4)
    masks = torch.arange(5, dtype = torch.float32)[:, None, None].expand(5, 2, 2).clone()
    scores, kept_boxes, kept_masks = seg.select_detections(
        logits, boxes, masks, max_detections = 2, threshold = 0.5
    )
    # 0.5 itself is not above the threshold; best first; at most 2.
    assert [round(v, 2) for v in scores.tolist()] == [0.98, 0.95]
    assert kept_boxes[:, 0].tolist() == [16.0, 4.0]
    assert kept_masks[:, 0, 0].tolist() == [4.0, 1.0]


def test_presence_suppresses_an_absent_concept():
    import torch

    logits = torch.tensor([1.0, 0.2])  # 0.73 and 0.55 alone: both kept without presence
    boxes = torch.zeros(2, 4)
    masks = torch.zeros(2, 2, 2)
    scores, _, _ = seg.select_detections(logits, boxes, masks, max_detections = 5, threshold = 0.5)
    assert scores.shape[0] == 2
    absent = torch.tensor([-7.0])  # sigmoid ~ 0.001
    scores, _, _ = seg.select_detections(
        logits, boxes, masks, max_detections = 5, threshold = 0.5, presence_logit = absent
    )
    assert scores.shape[0] == 0
    present = torch.tensor([7.0])
    scores, _, _ = seg.select_detections(
        logits, boxes, masks, max_detections = 5, threshold = 0.5, presence_logit = present
    )
    assert scores.shape[0] == 2


def test_union_mask_upsamples_to_the_source_size():
    import torch

    left = torch.full((1, 4, 4), -5.0)
    left[0, :, :2] = 5.0
    bottom = torch.full((1, 4, 4), -5.0)
    bottom[0, 3:, :] = 5.0
    union = seg.union_mask(torch.cat([left, bottom]), 30, 40)
    assert tuple(union.shape) == (30, 40)
    assert union[0, 0] and union[29, 39] and not union[0, 39]
    assert not seg.union_mask(torch.zeros(0, 4, 4), 7, 9).any()
    # Nothing kept: the empty union stays on the masks' device (a CUDA run ORs it into a CUDA union).
    empty = seg.union_mask(torch.zeros(0, 4, 4, device = "meta"), 7, 9)
    assert empty.device.type == "meta" and tuple(empty.shape) == (7, 9)
    png = seg.mask_to_png(union)
    from PIL import Image

    image = Image.open(io.BytesIO(png))
    assert image.mode == "L" and image.size == (40, 30)
    assert set(image.histogram()[1:255]) == {0}


def test_preprocess_is_comfyui_bilinear_1008_normalised():
    import torch
    from PIL import Image

    image = Image.new("RGB", (37, 23), (255, 0, 128))
    tensor = seg.preprocess(image, "cpu", torch.float32)
    assert tuple(tensor.shape) == (1, 3, seg.IMAGE_SIZE, seg.IMAGE_SIZE)
    assert torch.allclose(tensor[0, 0], torch.ones(1)) and torch.allclose(
        tensor[0, 1], -torch.ones(1)
    )


# ── the real forward on a tiny random Sam3Model ─────────────────────────────────────────────────────────────────────
def _tiny_config():
    from transformers import Sam3Config
    return Sam3Config(
        vision_config = {
            "backbone_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "window_size": 4,
                "global_attn_indexes": [1],
                "pretrain_image_size": 56,
                "image_size": 112,
                "patch_size": 14,
            },
            "fpn_hidden_size": 32,
            # 112 / 14 = 8 tokens a side, at the 4x, 2x and 1x levels.
            "backbone_feature_sizes": [[32, 32], [16, 16], [8, 8]],
        },
        text_config = {
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 32,
        },
        geometry_encoder_config = {
            "hidden_size": 32,
            "num_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
        },
        detr_encoder_config = {
            "hidden_size": 32,
            "num_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
        },
        detr_decoder_config = {
            "hidden_size": 32,
            "num_layers": 1,
            "num_queries": 12,
            "num_attention_heads": 2,
            "intermediate_size": 64,
        },
        mask_decoder_config = {
            "hidden_size": 32,
            "num_upsampling_stages": 3,
            "num_attention_heads": 2,
        },
    )


def test_tiny_config_is_small_but_the_real_input_is_1008():
    """The tiny model runs at 112 px to keep the CPU tests fast; the size comes from the checkpoint's config."""
    from transformers import Sam3Config

    assert seg.IMAGE_SIZE == 1008
    assert Sam3Config().vision_config.backbone_config.image_size == seg.IMAGE_SIZE


class _FakeTokenizer:
    """CLIP-shaped ids for the tiny vocab: one id per character, padded to max_length."""

    def __call__(
        self,
        texts,
        padding = None,
        max_length = 32,
        truncation = True,
        return_tensors = "pt",
    ):
        import torch

        ids = torch.zeros(len(texts), max_length, dtype = torch.long)
        mask = torch.zeros(len(texts), max_length, dtype = torch.long)
        for row, text in enumerate(texts):
            tokens = [1] + [2 + (ord(ch) % 100) for ch in text][: max_length - 2] + [127]
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1

        class _Batch(dict):
            def to(self, device):
                return _Batch({k: v.to(device) for k, v in self.items()})

        return _Batch(input_ids = ids, attention_mask = mask)


@pytest.fixture
def tiny_sam3(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Sam3Model"):
        pytest.skip("transformers without SAM 3")
    import torch

    torch.manual_seed(0)
    model = transformers.Sam3Model(_tiny_config())
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTokenizer()
    )
    return str(tmp_path)


def test_tiny_model_segments_at_source_size(tiny_sam3):
    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    image = Image.new("RGB", (64, 40), (120, 80, 40))
    # A threshold of 0 keeps every query up to each concept's cap, so the caps are what is checked.
    result = segmenter.segment(
        image, "sofa, pillow:3, sofa", threshold = 0.0, allow_gpu = False, path = tiny_sam3
    )
    assert result.device == "cpu" and (result.width, result.height) == (64, 40)
    assert [c["found"] for c in result.concepts] == [1, 3, 1]
    assert [c["max_detections"] for c in result.concepts] == [1, 3, 1]
    assert [i.concept for i in result.instances] == ["sofa", "pillow", "pillow", "pillow", "sofa"]
    for inst in result.instances:
        x1, y1, x2, y2 = inst.box
        assert 0 <= x1 <= 64 and 0 <= y1 <= 40 and x2 <= 64 and y2 <= 40
    mask = Image.open(io.BytesIO(result.mask_png))
    assert mask.mode == "L" and mask.size == (64, 40)
    body = result.to_json()
    assert body["mask"].startswith("data:image/png;base64,")
    assert set(body) == {
        "mask",
        "width",
        "height",
        "instances",
        "prompts",
        "device",
        "precision",
        "elapsed_ms",
    }
    assert body["precision"] == "bf16"
    # No keep-alive: the weights left with the call.
    assert not segmenter.loaded
    # An unreachable threshold keeps nothing: an all-black mask, still at the source size.
    result = segmenter.segment(image, "sofa", threshold = 1.0, allow_gpu = False, path = tiny_sam3)
    assert result.instances == [] and Image.open(io.BytesIO(result.mask_png)).getbbox() is None
    assert torch.is_grad_enabled()


def test_instance_masks_are_the_union_split_per_detection(tiny_sam3):
    """The Detailer's per-detection masks: one plane per instance in ``instances`` order, whose OR at the source
    size is exactly the union mask; a plain Find keeps none."""
    import numpy as np
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    image = Image.new("RGB", (64, 40), (120, 80, 40))
    plain = segmenter.segment(
        image, "sofa, pillow:3", threshold = 0.0, allow_gpu = False, path = tiny_sam3
    )
    assert plain.instance_logits == []
    result = segmenter.segment(
        image,
        "sofa, pillow:3",
        threshold = 0.0,
        allow_gpu = False,
        path = tiny_sam3,
        instance_masks = True,
    )
    assert len(result.instance_logits) == len(result.instances) == 4
    assert all(p.device.type == "cpu" and p.ndim == 2 for p in result.instance_logits)
    import torch

    union = seg.union_mask(torch.stack(result.instance_logits), 40, 64).numpy()
    png = np.array(Image.open(io.BytesIO(result.mask_png))) > 0
    assert (union == png).all()
    assert "instance_logits" not in result.to_json()


def test_detailer_budget_moves_only_the_planes_it_keeps(tiny_sam3, monkeypatch):
    """Detailer call (plane_filter + max_planes) keeps the instance-mask planes bit-identical,
    filters before capping, and moves at most max_planes off-device."""
    import torch
    from PIL import Image

    from core.inference import diffusion_detailer as det

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    image = Image.new("RGB", (64, 40), (120, 80, 40))
    # A repeat with a larger cap after a smaller one: its extra planes are only picked on the second occurrence.
    prompt = "pillow:1, sofa:2, pillow:4"
    full = segmenter.segment(
        image, prompt, threshold = 0.0, allow_gpu = False, path = tiny_sam3, instance_masks = True
    )
    assert len(full.instances) == 7
    dropped = {full.instances[1].box, full.instances[4].box}

    def keep(box):
        return box not in dropped

    # What the Detailer kept before: every plane moved, then filtered, then capped.
    cap = 3
    before = [p for p in zip(full.instances, full.instance_logits) if keep(p[0].box)][:cap]

    def _no_union(*a, **k):
        raise AssertionError("the Detailer's call built the union mask")

    monkeypatch.setattr(seg, "union_mask", _no_union)
    monkeypatch.setattr(seg, "mask_to_png", _no_union)
    plane_shape = full.instance_logits[0].shape
    real_float = torch.Tensor.float
    moved = []

    def _counting_float(self, *a, **k):
        out = real_float(self, *a, **k)
        if self.shape == plane_shape:  # a mask plane, not the (N, 4) boxes
            moved.append(out)
        return out

    monkeypatch.setattr(torch.Tensor, "float", _counting_float)
    budgeted = segmenter.segment(
        image,
        prompt,
        threshold = 0.0,
        allow_gpu = False,
        path = tiny_sam3,
        instance_masks = True,
        plane_filter = keep,
        max_planes = cap,
    )
    monkeypatch.setattr(torch.Tensor, "float", real_float)
    assert budgeted.mask_png == b""
    assert budgeted.instances == full.instances and budgeted.concepts == full.concepts
    assert len(budgeted.instance_logits) == len(budgeted.instances)
    after = [p for p in zip(budgeted.instances, budgeted.instance_logits) if p[1] is not None]
    assert len(after) == cap and len(moved) <= cap
    assert [i for i, _ in after] == [i for i, _ in before]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(after, before))
    sized = [p for p in zip(budgeted.instances, budgeted.instance_logits) if keep(p[0].box)]
    assert all(plane is not None for _, plane in sized[:cap])
    assert det.MAX_DETECTIONS >= cap


def _small_model():
    """Stands in for a loaded Sam3Model: real parameters and a buffer, one of them tied (shared storage)."""
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    model.register_buffer("table", torch.arange(16.0))
    model.register_buffer("tied", model.table)
    return model


def _gpu_loaded(
    monkeypatch,
    segmenter,
    model,
    path = "/sam3",
):
    """``segmenter`` as if ``model`` were loaded on cuda:0 (its tensors stay on the CPU), with the weights plan, the
    cache release and the device sync faked, and the host buffer pageable (no CUDA on the test host)."""
    import torch

    plan = seg.WeightsPlan("int8", path, "/sam3/w.safetensors", None, "test")
    monkeypatch.setattr(seg, "plan_weights", lambda *a, **k: plan)
    monkeypatch.setattr(seg, "_free_device_cache", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setenv(seg.ENV_PARK_PIN, "0")
    segmenter._model, segmenter._tokenizer = model, "tok"
    segmenter._path, segmenter.device, segmenter._dtype = path, "cuda:0", torch.bfloat16
    segmenter._weights_key = (plan.model_dir, plan.weights, plan.dequantize)
    segmenter.precision, segmenter.weights_file = "int8", "w.safetensors"
    return plan


def _weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_park_keeps_the_model_in_host_ram_and_the_next_load_reuses_it(monkeypatch):
    """The render (or the Detailer) parks SAM 3 instead of dropping it: its weight storages are emptied, and the next
    call on the same card gives the same module its exact bytes back with no second load."""
    import torch

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    model = _small_model()
    before = _weights(model)
    _gpu_loaded(monkeypatch, segmenter, model)
    loads = []
    monkeypatch.setattr(segmenter, "_load_weights", lambda *a, **k: loads.append(1))
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: loads.append("tok"),
        raising = False,
    )
    assert segmenter.release("image generation", wait = True, park = True) is True
    assert segmenter.parked and not segmenter.loaded and not segmenter.holds_vram()
    # Every weight storage gave its bytes up (the tied buffer once); the host buffer holds them.
    assert all(t.untyped_storage().nbytes() == 0 for t in model.state_dict().values())
    host = segmenter._parked["host"]
    # One device storage behind every weight, emptied; one host buffer holds them (each slot 256-byte aligned).
    assert host.storage.nbytes() == 0
    assert host.total >= sum(v.numel() * v.element_size() for k, v in before.items() if k != "tied")
    # A render's release while only the parked copy exists keeps it.
    assert segmenter.release("image generation", wait = True, park = True) is False
    assert segmenter.parked
    segmenter._load("/sam3", "cuda:0", torch.bfloat16)
    assert loads == []
    assert segmenter._model is model and segmenter.device == "cuda:0" and not segmenter.parked
    assert (segmenter.precision, segmenter.weights_file) == ("int8", "w.safetensors")
    after = _weights(model)
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert model.tied.data_ptr() == model.table.data_ptr()
    # The second park reuses the same host buffer (no new allocation) and round-trips the bytes again.
    buffer = host.buffer
    segmenter.release("detailer", wait = True, park = True)
    assert segmenter._parked["host"] is host and host.buffer is buffer
    assert all(t.untyped_storage().nbytes() == 0 for t in model.state_dict().values())
    segmenter._load("/sam3", "cuda:0", torch.bfloat16)
    assert all(torch.equal(before[k], v) for k, v in _weights(model).items())
    segmenter._cancel_park_timer()


def test_a_full_release_or_another_load_drops_the_parked_copy(monkeypatch):
    import torch

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    _gpu_loaded(monkeypatch, segmenter, _small_model())
    segmenter.release("detailer", wait = True, park = True)
    # Training start / a GPU handover: everything goes, the host copy too.
    assert segmenter.release("training start", wait = True) is True
    assert not segmenter.parked and not segmenter.loaded and segmenter._host is None

    _gpu_loaded(monkeypatch, segmenter, _small_model())
    segmenter.release("detailer", wait = True, park = True)
    fresh = []
    monkeypatch.setattr(
        segmenter,
        "_load_weights",
        lambda plan, path, device, dtype: (fresh.append(device) or _small_model(), plan),
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: "tok", raising = False
    )
    # The CPU fallback (fp32) is a different load: the parked GPU copy is dropped and the files are read.
    segmenter._load("/sam3", "cpu", torch.float32)
    assert fresh == ["cpu"] and not segmenter.parked


def test_park_expires_and_a_cpu_model_is_never_parked(monkeypatch):
    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    segmenter.park_seconds = 0.05
    _gpu_loaded(monkeypatch, segmenter, _small_model())
    segmenter.release("detailer", wait = True, park = True)
    deadline = time.time() + 5
    while segmenter.parked and time.time() < deadline:
        time.sleep(0.02)
    assert not segmenter.parked

    model = _small_model()
    _gpu_loaded(monkeypatch, segmenter, model)
    segmenter.device = "cpu"
    assert segmenter.release("detailer", wait = True, park = True) is True
    assert not segmenter.parked and not segmenter.loaded
    assert model[0].weight.untyped_storage().nbytes() > 0


def test_a_model_that_cannot_park_is_unloaded(monkeypatch):
    import torch

    class _Opaque(torch.Tensor):
        """A tensor subclass with no __tensor_flatten__: nothing knows its storages."""

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    model = _small_model()
    model.register_buffer("odd", torch.zeros(2).as_subclass(_Opaque))
    _gpu_loaded(monkeypatch, segmenter, model)
    assert segmenter.release("detailer", wait = True, park = True) is True
    assert not segmenter.parked and not segmenter.loaded
    # Nothing was emptied before the refusal.
    assert model[0].weight.untyped_storage().nbytes() > 0


def test_mask_planes_do_not_outlive_their_text(tiny_sam3, monkeypatch):
    import weakref

    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    real_load, real_select = segmenter._load, seg.select_detections
    planes: list = []

    def _select(*args, **kwargs):
        scores, boxes, masks = real_select(*args, **kwargs)
        planes.append(weakref.ref(masks))
        return scores, boxes, masks

    class _Watch:
        def __init__(self, model):
            self._model = model

        def __getattr__(self, name):
            return getattr(self._model, name)

        def __call__(self, **kwargs):
            # Every earlier text's kept mask planes are already OR-ed into the union and gone.
            assert all(
                ref() is None for ref in planes
            ), "an earlier concept's mask planes are still held"
            return self._model(**kwargs)

    def _load(path, device, dtype):
        real_load(path, device, dtype)
        if not isinstance(segmenter._model, _Watch):
            segmenter._model = _Watch(segmenter._model)

    monkeypatch.setattr(seg, "select_detections", _select)
    monkeypatch.setattr(segmenter, "_load", _load)
    image = Image.new("RGB", (32, 32), (120, 80, 40))
    prompt = "cat:200, dog:200, cat:2, bird:200"
    result = segmenter.segment(image, prompt, threshold = 0.0, allow_gpu = False, path = tiny_sam3)
    assert len(planes) == 3
    # The union is what every occurrence's own best-k masks give: each text's largest cap covers its smaller ones.
    monkeypatch.setattr(seg, "select_detections", real_select)
    alone = [
        segmenter.segment(image, part, threshold = 0.0, allow_gpu = False, path = tiny_sam3)
        for part in ("cat:200", "dog:200", "bird:200")
    ]
    from PIL import ImageChops

    combined = Image.open(io.BytesIO(alone[0].mask_png))
    for other in alone[1:]:
        combined = ImageChops.lighter(combined, Image.open(io.BytesIO(other.mask_png)))
    assert (
        ImageChops.difference(combined, Image.open(io.BytesIO(result.mask_png))).getbbox() is None
    )


def test_raw_outputs_do_not_outlive_their_concept(tiny_sam3, monkeypatch):
    import weakref

    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    real_load = segmenter._load
    outputs: list = []

    class _Watch:
        def __init__(self, model):
            self._model = model

        def __getattr__(self, name):
            return getattr(self._model, name)

        def __call__(self, **kwargs):
            # Every earlier concept's full output (all its mask planes) is gone before the next forward.
            assert all(ref() is None for ref in outputs), "a raw SAM 3 output is still held"
            out = self._model(**kwargs)
            outputs.append(weakref.ref(out))
            return out

    def _load(path, device, dtype):
        real_load(path, device, dtype)
        if not isinstance(segmenter._model, _Watch):
            segmenter._model = _Watch(segmenter._model)

    monkeypatch.setattr(segmenter, "_load", _load)
    image = Image.new("RGB", (32, 32), (120, 80, 40))
    result = segmenter.segment(
        image, "cat, dog, cat:3", threshold = 0.0, allow_gpu = False, path = tiny_sam3
    )
    assert len(outputs) == 2  # one forward per distinct text
    # The repeat keeps the best of the larger cap: cat:1 is cat:3's first detection.
    found = {(c["text"], c["max_detections"]): c["found"] for c in result.concepts}
    assert found[("cat", 1)] == 1 and found[("cat", 3)] == 3
    cats = [i for i in result.instances if i.concept == "cat"]
    assert cats[0] == cats[1] and cats[0].score >= cats[2].score >= cats[3].score


def test_keep_alive_then_release(tiny_sam3):
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    image = Image.new("RGB", (32, 32))
    segmenter.segment(image, "cat", allow_gpu = False, path = tiny_sam3)
    assert segmenter.loaded and segmenter._timer is not None
    assert segmenter.release("test") is True
    assert not segmenter.loaded and segmenter._timer is None
    assert segmenter.release("again") is False


def test_keep_alive_expiry_unloads(tiny_sam3):
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    segmenter.segment(Image.new("RGB", (16, 16)), "cat", allow_gpu = False, path = tiny_sam3)
    segmenter._timer.cancel()
    segmenter._expire(segmenter._release_epoch)
    assert not segmenter.loaded


def test_release_during_a_call_unloads_when_it_ends(tiny_sam3, monkeypatch):
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    entered, proceed = threading.Event(), threading.Event()
    real = segmenter._segment_on

    def _slow(*args, **kwargs):
        entered.set()
        proceed.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(segmenter, "_segment_on", _slow)
    worker = threading.Thread(
        target = segmenter.segment,
        args = (Image.new("RGB", (16, 16)), "cat"),
        kwargs = {"allow_gpu": False, "path": tiny_sam3},
    )
    worker.start()
    assert entered.wait(10)
    # Non-blocking: the call is in flight, so the release is deferred, not refused.
    assert segmenter.release("arbiter") is False
    proceed.set()
    worker.join(30)
    assert not segmenter.loaded


def test_tight_gpu_unloads_right_after_the_call(tiny_sam3, monkeypatch):
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    segmenter.segment(Image.new("RGB", (16, 16)), "cat", allow_gpu = False, path = tiny_sam3)
    assert segmenter.loaded
    # Pretend the copy sits on a card that is now tight.
    segmenter.device = "cuda:0"
    monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal: (1000, 80_000))
    with segmenter._run_lock:
        segmenter._after_call()
    assert not segmenter.loaded


def test_vram_is_tight_rule():
    assert seg.vram_is_tight(1000, 24_000)
    assert seg.vram_is_tight(10_000, 80_000)  # under 20% of the card
    assert not seg.vram_is_tight(30_000, 80_000)
    assert not seg.vram_is_tight(None, None)


def test_choose_device_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert seg.choose_device(True) == ("cpu", torch.float32)
    # Training (allow_gpu False) never takes a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert seg.choose_device(False) == ("cpu", torch.float32)


def test_choose_device_needs_room_on_the_card(monkeypatch):
    import contextlib

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: contextlib.nullcontext())
    monkeypatch.setattr(seg, "_resident_cuda_ordinal", lambda: 1)
    monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal: (20_000, 24_000))
    assert seg.choose_device(True) == ("cuda:1", torch.bfloat16)
    monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal: (1_000, 24_000))
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert seg.choose_device(True) == ("cpu", torch.float32)


def test_choose_device_credits_the_allocator_cache_parking_leaves(monkeypatch):
    # Reusable CUDA cache counts as free, so the next detection must not fall back to the CPU.
    import contextlib

    import torch

    from core.inference import diffusion_block_graph as bg
    from core.inference import diffusion_cuda_graph as cg

    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda ordinal: (2 * gib, 24 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 12 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 4 * gib)
    monkeypatch.setattr(bg, "pool_bytes", lambda device = None: 0)
    monkeypatch.setattr(cg, "live_pool_free_bytes", lambda: 0)
    monkeypatch.setattr(seg, "_resident_cuda_ordinal", lambda: 1)
    monkeypatch.setattr(seg, "_other_owner_on", lambda ordinal: None)
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert seg._cuda_free_mib(1) == (10 * 1024, 24 * 1024)
    assert seg.choose_device(True) == ("cuda:1", torch.bfloat16)
    # A live CUDA graph pool is reserved but not reusable: without it the card really is short.
    monkeypatch.setattr(cg, "live_pool_free_bytes", lambda: 7 * gib)
    assert seg._cuda_free_mib(1) == (3 * 1024, 24 * 1024)
    assert seg.choose_device(True) == ("cpu", torch.float32)
    # The credit never claims more than the card.
    monkeypatch.setattr(cg, "live_pool_free_bytes", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 90 * gib)
    assert seg._cuda_free_mib(1) == (24 * 1024, 24 * 1024)


def test_choose_device_never_shares_a_card_another_owner_holds(monkeypatch):
    import contextlib

    import torch

    from core.inference import gpu_arbiter

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(
        seg, "_cuda_free_mib", lambda ordinal: (60_000, 80_000)
    )  # plenty of room either way
    monkeypatch.setattr(seg, "_resident_cuda_ordinal", lambda: None)
    monkeypatch.setattr(gpu_arbiter, "physical_gpu_ids", lambda visible: frozenset(visible))
    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    cases = [
        ({}, "cuda:0"),  # nobody
        (
            {gpu_arbiter.DIFFUSION: frozenset({0})},
            "cuda:0",
        ),  # the image model: SAM 3 runs beside it by design
        ({gpu_arbiter.CHAT: frozenset({0})}, "cpu"),
        ({gpu_arbiter.VIDEO: frozenset({0}), gpu_arbiter.DIFFUSION: frozenset({0})}, "cpu"),
        ({gpu_arbiter.CHAT: None}, "cpu"),  # unknown GPUs: every card
        ({gpu_arbiter.CHAT: frozenset({1})}, "cuda:0"),  # another card
    ]
    for held, expected in cases:
        monkeypatch.setattr(gpu_arbiter, "residents", lambda held = held: dict(held))
        device, dtype = seg.choose_device(True)
        assert device == expected, held
        assert dtype == (torch.bfloat16 if expected != "cpu" else torch.float32)


def test_metal_is_not_shared_with_another_owner(monkeypatch):
    import torch

    from core.inference import gpu_arbiter

    if getattr(torch.backends, "mps", None) is None:
        pytest.skip("torch without an MPS backend")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    # Apple has one device: any resident but the image model counts, whatever GPU ids it reports.
    for held, expected in (
        ({}, "mps"),
        ({gpu_arbiter.DIFFUSION: None}, "mps"),
        ({gpu_arbiter.CHAT: frozenset({0})}, "cpu"),
        ({gpu_arbiter.VIDEO: None, gpu_arbiter.DIFFUSION: None}, "cpu"),
        ({gpu_arbiter.CHAT: frozenset()}, "cpu"),
    ):
        monkeypatch.setattr(gpu_arbiter, "residents", lambda held = held: dict(held))
        assert seg.choose_device(True) == (expected, torch.float32), held


def test_a_failed_load_places_no_weights(tmp_path, monkeypatch):
    import torch
    import transformers

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    placed, freed = [], []

    def _broken_tokenizer(*a, **k):
        raise OSError("no tokenizer files")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _broken_tokenizer)
    monkeypatch.setattr(
        transformers.Sam3Model, "from_pretrained", lambda *a, **k: placed.append(a) or object()
    )
    with pytest.raises(OSError):
        segmenter._load(str(tmp_path), "cuda:0", torch.bfloat16)
    assert (
        placed == [] and not segmenter.loaded
    )  # the tokenizer fails before any weights reach the card

    # Weights placed, then the load fails: they are dropped and the device cache handed back.
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())

    class _Half:
        def eval(self):
            raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(transformers.Sam3Model, "from_pretrained", lambda *a, **k: _Half())
    monkeypatch.setattr(seg, "_free_device_cache", lambda device: freed.append(device))
    with pytest.raises(RuntimeError, match = "bad checkpoint"):
        segmenter._load(str(tmp_path), "cuda:0", torch.bfloat16)
    assert freed == ["cuda:0"] and not segmenter.loaded and segmenter._tokenizer is None


def test_choose_device_reads_bf16_on_the_chosen_card(monkeypatch):
    import contextlib

    import torch

    # cuda:0 (current) has bf16, cuda:1 (the image model's card) does not.
    current = [0]

    @contextlib.contextmanager
    def _device(ordinal):
        before, current[0] = current[0], ordinal
        try:
            yield
        finally:
            current[0] = before

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", _device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: current[0] == 0)
    monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal: (20_000, 24_000))
    monkeypatch.setattr(seg, "_resident_cuda_ordinal", lambda: 1)
    assert seg.choose_device(True) == ("cuda:1", torch.float16)
    monkeypatch.setattr(seg, "_resident_cuda_ordinal", lambda: 0)
    assert seg.choose_device(True) == ("cuda:0", torch.bfloat16)
    assert current[0] == 0


def test_oom_on_the_gpu_retries_on_the_cpu(tiny_sam3, monkeypatch):
    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cuda:0", torch.bfloat16))
    real = segmenter._segment_on
    devices = []

    def _maybe_oom(image, concepts, threshold, path, device, dtype):
        devices.append(device)
        if device != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(image, concepts, threshold, path, device, dtype)

    monkeypatch.setattr(segmenter, "_segment_on", _maybe_oom)
    result = segmenter.segment(Image.new("RGB", (16, 16)), "cat", path = tiny_sam3)
    assert devices == ["cuda:0", "cpu"] and result.device == "cpu"


def test_oom_retry_frees_the_failed_attempt_before_reading_cpu_only(monkeypatch):
    import weakref

    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cuda:0", torch.bfloat16))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)

    class _CudaTensors:  # stands in for the activations the failed attempt's frame holds
        pass

    pinned: list = []
    freed: list = []

    def _segment_on(image, concepts, threshold, path, device, dtype):
        if device != "cpu":
            activations = _CudaTensors()
            pinned.append(weakref.ref(activations))
            raise torch.OutOfMemoryError("CUDA out of memory")
        assert segmenter._call_device == "cpu"
        return types.SimpleNamespace()

    def _free(device):
        # The cache is released only once the OOM's traceback no longer pins the failed attempt's tensors.
        freed.append((device, pinned[0]() is None, segmenter._call_device))

    monkeypatch.setattr(segmenter, "_segment_on", _segment_on)
    monkeypatch.setattr(seg, "_free_device_cache", _free)
    segmenter.segment(Image.new("RGB", (8, 8)), "cat", path = "unused")
    assert freed and freed[0] == ("cuda:0", True, "cuda:0")


def test_failed_cpu_retry_still_cleans_up(tiny_sam3, monkeypatch):
    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cuda:0", torch.bfloat16))
    real_load = segmenter._load

    def _oom_then_fail(image, concepts, threshold, path, device, dtype):
        if device != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory")
        real_load(path, device, dtype)  # the CPU copy is resident when preprocessing fails
        segmenter._release_pending = True  # a release that arrived mid-call
        raise RuntimeError("preprocess failed")

    monkeypatch.setattr(segmenter, "_segment_on", _oom_then_fail)
    with pytest.raises(RuntimeError, match = "preprocess failed"):
        segmenter.segment(Image.new("RGB", (16, 16)), "cat", path = tiny_sam3)
    # The pending release was honoured, not left with the weights resident and no timer.
    assert not segmenter.loaded and segmenter._release_pending is False
    assert segmenter._call_device is None

    # Without a pending release the CPU copy gets the keep-alive instead.
    def _oom_then_fail_quietly(image, concepts, threshold, path, device, dtype):
        if device != "cpu":
            raise torch.OutOfMemoryError("CUDA out of memory")
        real_load(path, device, dtype)
        raise RuntimeError("preprocess failed")

    monkeypatch.setattr(segmenter, "_segment_on", _oom_then_fail_quietly)
    with pytest.raises(RuntimeError):
        segmenter.segment(Image.new("RGB", (16, 16)), "cat", path = tiny_sam3)
    assert segmenter.loaded and segmenter._timer is not None
    segmenter.release("test")


def test_segment_errors(monkeypatch, tmp_path):
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError):
        segmenter.segment(image, " , ", path = str(tmp_path))
    with pytest.raises(ValueError):
        segmenter.segment(image, "sofa", threshold = 2.0, path = str(tmp_path))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: False)
    with pytest.raises(seg.SegmenterUnavailableError):
        segmenter.segment(image, "sofa", path = str(tmp_path))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.setattr(seg, "model_dir", lambda: None)
    with pytest.raises(seg.SegmenterNotDownloadedError):
        segmenter.segment(image, "sofa")


# ── files, status and download errors ───────────────────────────────────────────────────────────────────────────────
def test_override_dir_must_hold_a_config(tmp_path, monkeypatch):
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    assert seg.model_dir() is None  # a typo'd / empty override is not replaced by the cache
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "tokenizer.json").write_text("{}")
    assert seg.model_dir() == str(tmp_path)
    assert seg.status()["override"] is True


def test_partial_override_does_not_read_as_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    (tmp_path / "config.json").write_text("{}")
    assert seg.model_dir() is None  # config only: no weights, no tokenizer
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"w": "model-1.safetensors"}}'
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"")
    assert seg.model_dir() is None  # still no tokenizer
    (tmp_path / "vocab.json").write_text("{}")
    assert seg.model_dir() is None  # vocab.json alone: AutoTokenizer refuses it without merges.txt
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    assert seg.model_dir() == str(tmp_path)  # sharded weights + a tokenizer
    (tmp_path / "model.safetensors.index.json").unlink()
    body = seg.status()
    assert body["downloaded"] is False and body["override"] is True


def test_sharded_override_needs_every_shard(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": shards[0], "b": shards[1], "c": shards[1]}}))
    (tmp_path / shards[0]).write_bytes(b"")
    assert seg.model_dir() is None  # an interrupted copy: the second shard is missing
    assert seg.status()["downloaded"] is False
    (tmp_path / shards[1]).write_bytes(b"")
    assert seg.model_dir() == str(tmp_path)
    for broken in ("not json", json.dumps({"weight_map": {}}), json.dumps({"metadata": {}})):
        index.write_text(broken)
        assert seg.model_dir() is None


def test_status_shape(monkeypatch):
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE, raising = False)
    monkeypatch.delenv(seg.ENV_VARIANT, raising = False)
    monkeypatch.setattr(seg, "_cached_snapshot", lambda *a, **k: None)
    body = seg.status()
    assert body["variant"] == seg.DEFAULT_VARIANT == "sam3.1" and body["label"] == "SAM 3.1"
    assert body["repo"] == seg.SAM31_REPO and body["gated"] is False
    assert body["downloaded"] is False and body["loaded"] is False
    assert body["license_url"].startswith(f"https://huggingface.co/{seg.SAM31_REPO}")
    assert 1_600_000_000 < body["size_bytes"] < 1_800_000_000
    assert {"downloading", "error", "bytes_total", "bytes_done"} <= set(body["download"])


# ── variants ────────────────────────────────────────────────────────────────────────────────────────────────────────
_SHA = re.compile(r"[0-9a-f]{40}")


def test_variant_table():
    assert list(seg.SEGMENTER_VARIANTS) == ["sam3.1", "sam3"] and seg.DEFAULT_VARIANT == "sam3.1"
    sam31, sam3 = seg.SEGMENTER_VARIANTS["sam3.1"], seg.SEGMENTER_VARIANTS["sam3"]
    assert sam31.repo == seg.SAM31_REPO and re.fullmatch(r"[\w.-]+/[\w.-]+", sam31.repo)
    assert sam31.gated is False and sam3.gated is True
    assert (
        sam3.repo == "facebook/sam3" and sam3.license_url == "https://huggingface.co/facebook/sam3"
    )
    # bf16 SAM 3.1 is half of the fp32 facebook/sam3 download.
    assert 1_600_000_000 < sam31.size_bytes < 1_800_000_000 < 3_400_000_000 < sam3.size_bytes
    for variant in seg.SEGMENTER_VARIANTS.values():
        assert variant.id in seg.SEGMENTER_SPECS
        spec = seg.SEGMENTER_SPECS[variant.id]
        assert (spec.repo, spec.files, spec.revision) == (
            variant.repo,
            variant.files,
            variant.revision,
        )
        assert "model.safetensors" in variant.files and "sam3.pt" not in variant.files
        assert not any(name.endswith(".pt") for name in variant.files)
        assert variant.license_url.startswith(f"https://huggingface.co/{variant.repo}")


def test_pinned_revision_constants():
    """facebook/sam3 and unsloth/sam3.1 (bf16 and pre-quantized files) are pinned to full commit shas, never a
    short or partial one."""
    assert _SHA.fullmatch(seg.SAM3_REVISION)
    assert _SHA.fullmatch(seg.SAM31_REVISION)
    assert _SHA.fullmatch(seg.SAM31_PREQUANT_REVISION)
    for variant in seg.SEGMENTER_VARIANTS.values():
        assert variant.revision == "main" or _SHA.fullmatch(variant.revision)
        assert seg.pinned_revision(variant) == (
            None if variant.revision == "main" else variant.revision
        )
    assert seg.pinned_revision(seg.SEGMENTER_VARIANTS["sam3"]) == seg.SAM3_REVISION


def test_variant_env_choice(monkeypatch):
    for value, expected in (
        ("sam3", "sam3"),
        (" SAM3.1 ", "sam3.1"),
        ("facebook/sam3", "sam3"),
        (seg.SAM31_REPO.upper(), "sam3.1"),
        ("sam2", None),
        ("", None),
    ):
        monkeypatch.setenv(seg.ENV_VARIANT, value)
        assert seg.requested_variant_id() == expected, value


def _cached(*ids):
    return (
        lambda model_id = seg.DEFAULT_VARIANT, **_k: f"/cache/{model_id}" if model_id in ids else None
    )


def test_active_variant_prefers_the_default_and_keeps_an_existing_sam3(monkeypatch):
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE, raising = False)
    monkeypatch.delenv(seg.ENV_VARIANT, raising = False)
    monkeypatch.setattr(seg, "_cached_snapshot", _cached())
    assert seg.active_variant().id == "sam3.1" and seg.model_dir() is None
    # An install that already fetched facebook/sam3 keeps using it rather than downloading again.
    monkeypatch.setattr(seg, "_cached_snapshot", _cached("sam3"))
    assert seg.active_variant().id == "sam3" and seg.model_dir() == "/cache/sam3"
    monkeypatch.setattr(seg, "_cached_snapshot", _cached("sam3", "sam3.1"))
    assert seg.active_variant().id == "sam3.1" and seg.model_dir() == "/cache/sam3.1"
    # The env choice wins even when only the other variant is on disk.
    monkeypatch.setenv(seg.ENV_VARIANT, "sam3")
    monkeypatch.setattr(seg, "_cached_snapshot", _cached("sam3.1"))
    assert seg.active_variant().id == "sam3" and seg.model_dir() is None


def test_status_reports_gated_and_ungated_variants(monkeypatch):
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE, raising = False)
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.setattr(seg, "_cached_snapshot", _cached())
    monkeypatch.setenv(seg.ENV_VARIANT, "sam3")
    body = seg.status()
    assert (body["variant"], body["label"], body["repo"], body["gated"]) == (
        "sam3",
        "SAM 3",
        "facebook/sam3",
        True,
    )
    assert body["revision"] == seg.SAM3_REVISION and body["size_bytes"] == seg.SAM3_SIZE_BYTES
    assert body["downloaded"] is False
    monkeypatch.setenv(seg.ENV_VARIANT, "sam3.1")
    body = seg.status()
    assert (body["variant"], body["label"], body["repo"], body["gated"]) == (
        "sam3.1",
        "SAM 3.1",
        seg.SAM31_REPO,
        False,
    )
    assert body["size_bytes"] == seg.SAM31_SIZE_BYTES
    monkeypatch.setattr(seg, "_cached_snapshot", _cached("sam3.1"))
    assert seg.status()["downloaded"] is True
    # Unsupported installs still name the variant they would use, without touching the cache.
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: False)
    monkeypatch.setattr(seg, "_cached_snapshot", lambda *a, **k: pytest.fail("cache read"))
    body = seg.status()
    assert body["variant"] == "sam3.1" and body["downloaded"] is False


def test_override_still_wins_over_the_variant(tmp_path, monkeypatch):
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, str(tmp_path))
    monkeypatch.setenv(seg.ENV_VARIANT, "sam3")
    monkeypatch.setattr(seg, "_cached_snapshot", _cached("sam3", "sam3.1"))
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"{}")
    assert seg.model_dir() == str(tmp_path)
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    body = seg.status()
    assert body["override"] is True and body["downloaded"] is True and body["variant"] == "sam3"
    # With the override the cache is not consulted: a cached facebook/sam3 never relabels the override as SAM 3.
    monkeypatch.delenv(seg.ENV_VARIANT)
    monkeypatch.setattr(seg, "_cached_snapshot", lambda *a, **k: pytest.fail("cache read"))
    body = seg.status()
    assert (
        body["variant"] == seg.DEFAULT_VARIANT
        and body["gated"] is False
        and body["downloaded"] is True
    )
    assert seg.active_variant().id == seg.DEFAULT_VARIANT


def test_cached_snapshot_reads_the_variant_repo_at_its_pin(tmp_path, monkeypatch):
    from core.inference import stt_sidecar

    monkeypatch.setattr(stt_sidecar, "_read_revision_record", lambda repo: None)
    variant = seg.SEGMENTER_VARIANTS["sam3"]
    snapshot = tmp_path / "models--facebook--sam3" / "snapshots" / variant.revision
    snapshot.mkdir(parents = True)
    for name in variant.files[:-1]:
        (snapshot / name).write_text("{}")
    assert seg._cached_snapshot("sam3", hub_cache = tmp_path) is None  # no weights yet
    (snapshot / seg.SEGMENTER_WEIGHTS_FILE).write_bytes(b"")
    assert seg._cached_snapshot("sam3", hub_cache = tmp_path) == str(snapshot)
    # facebook/sam3's files are not unsloth/sam3.1's.
    assert seg._cached_snapshot("sam3.1", hub_cache = tmp_path) is None


def test_gated_download_error_names_the_license():
    class GatedRepoError(Exception):
        pass

    sam3 = seg.SEGMENTER_VARIANTS["sam3"]
    message = seg.describe_download_error("sam3", GatedRepoError("403 Client Error"))
    assert "license" in message and sam3.license_url in message and sam3.repo in message
    assert seg.describe_download_error("sam3", RuntimeError("disk full")) is None
    # An ungated variant needs no token or license: a refusal there keeps the generic message.
    assert seg.describe_download_error("sam3.1", GatedRepoError("401 Client Error")) is None


@pytest.mark.parametrize("variant_id", ["sam3.1", "sam3"])
def test_download_fetches_the_variant_files_at_its_revision(monkeypatch, tmp_path, variant_id):
    """The pinned downloader asks for exactly the variant's files, at its pinned commit (or main, recorded)."""
    from huggingface_hub import hf_hub_url

    from core.inference import stt_mtmd_sidecar as mtmd

    variant = seg.SEGMENTER_VARIANTS[variant_id]
    resolved = seg.pinned_revision(variant) or "b" * 40
    seen = []

    class _Meta:
        commit_hash = resolved
        size = 10
        etag = "e"

    def _meta(url, token = None):
        seen.append(url)
        return _Meta()

    def _stop(repo):
        seen.append(("claim", repo))
        raise RuntimeError("stop before the transfer")

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", _meta)
    monkeypatch.setattr(mtmd, "_claim_stt_repository", _stop)
    state = mtmd._MtmdDownloadState(
        specs = seg.SEGMENTER_SPECS,
        cached_paths = lambda *a, **k: None,
        resolve = seg._resolve_segmenter_id,
        label = "Mask segmenter",
        describe_error = lambda model_id, error: f"described {model_id}",
    )
    state._run(variant_id, None, hub_cache = tmp_path)
    # The first lookup resolves the revision (the pin, else main), the rest are asked at that commit.
    expected = [hf_hub_url(variant.repo, variant.files[0], revision = seg.pinned_revision(variant))]
    expected += [hf_hub_url(variant.repo, name, revision = resolved) for name in variant.files[1:]]
    if variant.revision == "main":
        assert "/resolve/main/" in expected[0]
    else:
        assert all(f"/resolve/{variant.revision}/" in url for url in expected)
    assert seen[:-1] == expected and seen[-1] == ("claim", variant.repo)
    assert state.status()["error"] == f"described {variant_id}"


def test_start_download_uses_the_active_variant(monkeypatch):
    started = []

    class _State:
        def start(self, model_id, token):
            started.append((model_id, token))
            return "dl"

    monkeypatch.setattr(seg, "_get_download_state", lambda: _State())
    monkeypatch.setattr(seg, "active_variant", lambda: seg.SEGMENTER_VARIANTS["sam3"])
    assert seg.start_download("hf_x") == "dl" and started[-1] == ("sam3", "hf_x")
    assert seg.start_download(None, "sam3.1") == "dl" and started[-1] == ("sam3.1", None)
    with pytest.raises(ValueError):
        seg.start_download(None, "sam2")


# ── GPU arbiter hooks ───────────────────────────────────────────────────────────────────────────────────────────────
class _Recorder:
    def __init__(self):
        self.calls = []
        self.parks = []

    def release_segmenter(
        self,
        reason,
        wait = False,
        handover = False,
        park = False,
    ):
        self.calls.append((reason, wait))
        self.parks.append(park)
        return True


def test_arbiter_releases_the_segmenter(monkeypatch):
    import sys

    from core.inference import gpu_arbiter

    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", recorder)
    gpu_arbiter._release_mask_segmenter("x")
    assert recorder.calls == [("x", True)]
    # Training's start path: release(DIFFUSION) whether or not an image model is resident.
    recorder.calls.clear()
    monkeypatch.setattr(gpu_arbiter, "_residents", {})
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    gpu_arbiter.release(gpu_arbiter.DIFFUSION)
    assert recorder.calls and recorder.calls[0][0] == "image GPU released"
    # Never loaded: nothing is imported or created.
    monkeypatch.delitem(sys.modules, "core.inference.diffusion_segment")
    gpu_arbiter._release_mask_segmenter("y")


def _hold_a_call(segmenter, device):
    """Run a fake call on ``device`` that holds the run lock until the returned event is set."""
    entered, proceed = threading.Event(), threading.Event()

    def _call():
        with segmenter._run_lock:
            segmenter._call_device = device
            entered.set()
            proceed.wait(10)
            segmenter._call_device = None

    worker = threading.Thread(target = _call)
    worker.start()
    assert entered.wait(10)
    return proceed, worker


def test_a_release_timing_out_as_the_call_settles_still_unloads(monkeypatch):
    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 600)
    monkeypatch.setattr(seg, "choose_device", lambda allow_gpu: ("cuda:0", torch.bfloat16))
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.setattr(seg, "_cuda_free_mib", lambda ordinal: (None, None))

    def _segment_on(image, concepts, threshold, path, device, dtype):
        segmenter._model, segmenter.device = (
            object(),
            device,
        )  # the CUDA copy stays resident after the call
        return types.SimpleNamespace()

    real_after = segmenter._after_call
    settled = threading.Event()

    def _after_call():
        real_after()  # reads the release flag (none yet) and arms the keep-alive
        settled.set()
        time.sleep(0.3)  # the call's tail: the release's wait runs out right here

    monkeypatch.setattr(segmenter, "_segment_on", _segment_on)
    monkeypatch.setattr(segmenter, "_after_call", _after_call)
    worker = threading.Thread(
        target = segmenter.segment, args = (Image.new("RGB", (8, 8)), "cat"), kwargs = {"path": "x"}
    )
    worker.start()
    assert settled.wait(10)
    # Handover waits out the tail and unloads, so no new owner loads beside resident weights.
    assert segmenter.release("handover", wait = True, timeout = 0.05, handover = True) is True
    worker.join(10)
    assert not segmenter.loaded and segmenter._timer is None and segmenter._release_pending is False
    # Codex's interleaving: the call already cleared its device and is only letting go of the lock.
    segmenter._model, segmenter.device = object(), "cuda:0"
    proceed, holder = _hold_a_call(segmenter, None)
    threading.Timer(0.2, proceed.set).start()
    try:
        assert segmenter.release("handover", wait = True, timeout = 0.05, handover = True) is True
    finally:
        proceed.set()
        holder.join(10)
    assert not segmenter.loaded


def test_handover_refuses_while_a_gpu_call_outlives_the_wait(monkeypatch):
    import sys

    from core.inference import gpu_arbiter

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    monkeypatch.setattr(seg, "_segmenter", segmenter)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", seg)
    real_release = segmenter.release
    monkeypatch.setattr(
        segmenter, "release", lambda reason, **kw: real_release(reason, **{**kw, "timeout": 0.05})
    )
    for device in ("cuda:0", ""):  # on the card, or not placed yet
        proceed, worker = _hold_a_call(segmenter, device)
        try:
            with pytest.raises(seg.SegmenterReleaseTimeoutError):
                seg.release_segmenter("x", wait = True, handover = True)
            # The arbiter aborts the transfer instead of handing over a busy card.
            with pytest.raises(RuntimeError, match = "cannot be transferred"):
                gpu_arbiter._release_mask_segmenter("GPU acquired for chat", handover = True)
            # Release / training paths stay best effort: deferred, logged, never raised.
            gpu_arbiter._release_mask_segmenter("image GPU released")
            assert segmenter._release_pending is True
        finally:
            proceed.set()
            worker.join(10)
    # A call on the CPU holds no VRAM: the handover proceeds and the release stays deferred.
    proceed, worker = _hold_a_call(segmenter, "cpu")
    try:
        assert seg.release_segmenter("x", wait = True, handover = True) is False
        gpu_arbiter._release_mask_segmenter("GPU acquired for chat", handover = True)
    finally:
        proceed.set()
        worker.join(10)
    # Idle: nothing to wait for.
    assert seg.release_segmenter("x", wait = True, handover = True) is False


def test_acquire_does_not_register_while_the_segmenter_holds_the_card(monkeypatch):
    import sys

    from core.inference import gpu_arbiter

    class _Busy:
        class SegmenterReleaseTimeoutError(RuntimeError):
            pass

        def release_segmenter(
            self,
            reason,
            wait = False,
            handover = False,
        ):
            if handover:
                raise self.SegmenterReleaseTimeoutError("still running")
            return False

    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", _Busy())
    monkeypatch.setattr(gpu_arbiter, "_residents", {})
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_release_idle_video_servers", lambda reason: None)
    registered = []
    with pytest.raises(RuntimeError, match = "cannot be transferred"):
        gpu_arbiter.acquire_for(gpu_arbiter.VIDEO, register = lambda: registered.append(1))
    assert registered == [] and gpu_arbiter.current_owner() is None


def test_a_call_queued_across_a_handover_stays_off_the_card(tiny_sam3, monkeypatch):
    import torch
    from PIL import Image

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    monkeypatch.setattr(
        seg,
        "choose_device",
        lambda allow_gpu: ("cuda:0", torch.bfloat16) if allow_gpu else ("cpu", torch.float32),
    )
    placed = []
    monkeypatch.setattr(
        segmenter, "_segment_on", lambda *a: placed.append(a[4]) or types.SimpleNamespace()
    )
    image = Image.new("RGB", (16, 16))
    for handover, cpu_only in ((True, False), (True, True), (False, False)):
        placed.clear()
        proceed, worker = _hold_a_call(segmenter, "cuda:0")
        errors = []

        def _queued():
            try:
                segmenter.segment(image, "cat", allow_gpu = not cpu_only, path = tiny_sam3)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        queued = threading.Thread(target = _queued)
        queued.start()
        deadline = time.monotonic() + 5
        while not segmenter._run_lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)  # the queued call is now blocked on _run_lock
        # Release waits on the in-flight call, then takes and drops the lock for the new owner.
        releaser = threading.Thread(
            target = lambda: segmenter.release("handover", wait = True, handover = handover)
        )
        releaser.start()
        time.sleep(0.1)
        proceed.set()
        worker.join(10)
        releaser.join(10)
        queued.join(10)
        if handover and not cpu_only:
            assert (
                placed == [] and len(errors) == 1 and isinstance(errors[0], seg.SegmenterBusyError)
            )
        elif handover:
            assert placed == ["cpu"] and errors == []  # a CPU-only call holds no VRAM
        else:
            assert placed == ["cuda:0"] and errors == []  # an ordinary release does not fence calls
    # A call that starts after the handover runs normally.
    placed.clear()
    segmenter.segment(image, "cat", path = tiny_sam3)
    assert placed == ["cuda:0"]


def test_a_call_during_a_handovers_register_stays_off_the_card(tiny_sam3, monkeypatch):
    import sys

    import torch
    from PIL import Image

    from core.inference import gpu_arbiter

    segmenter = seg.MaskSegmenter(keep_alive_seconds = 0)
    monkeypatch.setattr(seg, "_segmenter", segmenter)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", seg)
    monkeypatch.setattr(
        seg,
        "choose_device",
        lambda allow_gpu: ("cuda:0", torch.bfloat16) if allow_gpu else ("cpu", torch.float32),
    )
    placed = []
    monkeypatch.setattr(
        segmenter, "_segment_on", lambda *a: placed.append(a[4]) or types.SimpleNamespace()
    )
    monkeypatch.setattr(gpu_arbiter, "_residents", {})
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_release_idle_video_servers", lambda reason: None)
    image = Image.new("RGB", (16, 16))
    seen = {}

    def _register():
        # The segmenter's release already ran and dropped its lock; the new owner is still loading.
        seen["fenced"] = gpu_arbiter.handover_in_progress()
        with pytest.raises(seg.SegmenterBusyError):
            segmenter.segment(image, "cat", path = tiny_sam3)
        segmenter.segment(
            image, "cat", allow_gpu = False, path = tiny_sam3
        )  # the CPU is not being handed over

    gpu_arbiter.acquire_for(gpu_arbiter.VIDEO, register = _register)
    assert seen["fenced"] is True and placed == ["cpu"]
    assert gpu_arbiter.handover_in_progress() is False
    segmenter.segment(image, "cat", path = tiny_sam3)
    assert placed == ["cpu", "cuda:0"]
    # A raising register still lifts the fence.
    with pytest.raises(RuntimeError):
        gpu_arbiter.acquire_for(
            gpu_arbiter.VIDEO, register = lambda: (_ for _ in ()).throw(RuntimeError("load failed"))
        )
    assert gpu_arbiter.handover_in_progress() is False


def test_training_beside_an_off_torch_image_model_releases_the_segmenter(monkeypatch):
    import inspect
    import sys
    from types import SimpleNamespace

    import routes.training as training_routes
    from core.inference import diffusion_engine_router, gpu_arbiter

    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", recorder)
    resident = SimpleNamespace(runs_off_torch_device = True, is_loaded = True, unload = lambda: None)
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: resident)
    monkeypatch.setattr(gpu_arbiter, "release", lambda owner: None)
    monkeypatch.setattr(
        "core.inference.video.get_video_backend",
        lambda: SimpleNamespace(status = lambda: {"loaded": False}, unload = lambda: None),
    )
    monkeypatch.setattr("routes.training_vram.summarize_resident_chat", lambda: {"any": False})
    training_routes._free_gpu_for_diffusion_training()
    assert ("diffusion training start", True) in recorder.calls
    # The LLM start path keeps an off-torch image model the same way and releases the segmenter there too.
    source = inspect.getsource(training_routes.start_training)
    release = source.index('gpu_arbiter.release_mask_segmenter("training start")')
    assert release < source.index('if getattr(diffusion, "runs_off_torch_device", False) is True:')


def test_training_start_refuses_while_a_mask_call_holds_the_gpu(monkeypatch):
    import inspect
    import sys
    from types import SimpleNamespace

    import routes.training as training_routes
    from core.inference import diffusion_engine_router, gpu_arbiter

    class _StillRunning:
        class SegmenterReleaseTimeoutError(RuntimeError):
            pass

        def release_segmenter(
            self,
            reason,
            wait = False,
            handover = False,
        ):
            if handover:
                raise self.SegmenterReleaseTimeoutError("still running")
            return False

    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", _StillRunning())
    with pytest.raises(gpu_arbiter.SegmenterStillRunningError, match = "try again"):
        gpu_arbiter.release_mask_segmenter("training start")
    assert (
        gpu_arbiter.SegmenterStillRunningError.blocks_training is True
    )  # the core trainer re-raises it
    unloaded = []
    for off_torch in (True, False):
        resident = SimpleNamespace(
            runs_off_torch_device = off_torch, is_loaded = True, unload = lambda: unloaded.append(1)
        )
        monkeypatch.setattr(
            diffusion_engine_router, "get_active_diffusion_engine", lambda r = resident: r
        )
        monkeypatch.setattr(gpu_arbiter, "release", lambda owner: None)
        # A RuntimeError: the diffusion start route answers 409 instead of spawning beside SAM 3.
        with pytest.raises(gpu_arbiter.SegmenterStillRunningError):
            training_routes._free_gpu_for_diffusion_training()
    assert unloaded == []  # refused before tearing anything down
    # The LLM start path: re-raised out of its best-effort block and answered 409.
    source = inspect.getsource(training_routes.start_training)
    assert "except gpu_arbiter.SegmenterStillRunningError:\n                raise" in source
    assert source.count("SegmenterStillRunningError,") == 2


def test_no_find_starts_while_a_render_is_on_its_way(monkeypatch):
    import asyncio

    import core.inference.diffusion_engine_router as engine_router
    import routes.inference as inference_routes

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    monkeypatch.setattr(
        seg,
        "get_segmenter",
        lambda: pytest.fail("a Find ran between the render's SAM 3 release and its slot"),
    )
    seen = []

    async def _run(request):
        # The render released SAM 3 and has not taken the slot yet: a Find is refused, and frees the slot it took.
        with pytest.raises(seg.SegmenterBusyError, match = "generating"):
            seg.segment_image(_png_b64(), "sofa")
        seen.append(engine._generate_lock.locked())
        return "done"

    monkeypatch.setattr(inference_routes, "_generate_diffusion_image_run", _run)
    assert asyncio.run(inference_routes._generate_diffusion_image_core(object())) == "done"
    assert seen == [False] and seg._renders_pending == 0


def test_a_render_releases_the_segmenter_first(monkeypatch):
    import inspect
    import sys

    import routes.inference as inference_routes

    recorder = _Recorder()
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_segment", recorder)
    inference_routes._release_mask_segmenter_for_render()
    assert recorder.calls == [("image generation", True)]
    # Parked in host RAM for the next Find or Detailer pass, not dropped.
    assert recorder.parks == [True]
    # Never loaded: nothing is imported.
    monkeypatch.delitem(sys.modules, "core.inference.diffusion_segment")
    inference_routes._release_mask_segmenter_for_render()
    # Before the run, on every render (a resident model renders with no arbiter acquire).
    source = inspect.getsource(inference_routes._generate_diffusion_image_run)
    release = source.index("_release_mask_segmenter_for_render")
    assert (
        source.index("_enhance_generate_prompts(backend, request)")
        < release
        < source.index("for attempt in range(2)")
    )


def test_release_segmenter_never_creates_one(monkeypatch):
    monkeypatch.setattr(seg, "_segmenter", None)
    assert seg.release_segmenter("x") is False
    assert seg._segmenter is None


# ── render slot ─────────────────────────────────────────────────────────────────────────────────────────────────────
class _Engine:
    def __init__(self):
        self._generate_lock = threading.Lock()
        self._loading = None


def test_render_slot_refuses_while_generating_or_loading(monkeypatch):
    import core.inference.diffusion_engine_router as engine_router

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    lock, why = seg._take_render_slot()
    assert lock is engine._generate_lock and why == ""
    lock.release()
    with engine._generate_lock:
        assert seg._take_render_slot() == (None, "generating")

    class _Loading:
        error = None

    engine._loading = _Loading()
    assert seg._take_render_slot() == (None, "loading")
    engine._loading.error = "failed"
    lock, why = seg._take_render_slot()
    assert why == "" and lock is not None
    lock.release()


def test_segment_image_holds_and_frees_the_slot(monkeypatch):
    import core.inference.diffusion_engine_router as engine_router

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)

    class _Stub:
        def segment(self, image, prompt, threshold, allow_gpu):
            assert engine._generate_lock.locked()
            assert image.size == (64, 48)
            return seg.SegmentResult(mask_png = b"png", width = 64, height = 48)

    monkeypatch.setattr(seg, "get_segmenter", lambda: _Stub())
    body = seg.segment_image(_png_b64(), "sofa")
    assert body["width"] == 64 and not engine._generate_lock.locked()
    with engine._generate_lock:
        with pytest.raises(seg.SegmenterBusyError):
            seg.segment_image(_png_b64(), "sofa")
    with pytest.raises(ValueError):
        seg.segment_image("not base64 at all !!", "sofa")


# ── routes ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import core.inference.diffusion_engine_router as engine_router
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from routes.inference import studio_router

    engine = _Engine()
    monkeypatch.setattr(engine_router, "get_active_diffusion_engine", lambda: engine)
    app = FastAPI()
    app.include_router(studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app), engine


def test_segment_route_contract(client, tiny_sam3, monkeypatch):
    http, engine = client
    import routes.inference as inference_routes

    monkeypatch.setattr(seg, "model_dir", lambda: tiny_sam3)
    monkeypatch.setattr(seg, "_segmenter", seg.MaskSegmenter(keep_alive_seconds = 0))
    monkeypatch.setattr(
        inference_routes, "_training_is_active", lambda: True
    )  # CPU, no GPU in tests
    r = http.post(
        "/api/inference/images/segment",
        json = {"image": _png_b64(64, 48), "prompt": "Sofa:2, red lamp", "threshold": 0.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["width"] == 64 and body["height"] == 48 and body["device"] == "cpu"
    assert [p["text"] for p in body["prompts"]] == ["Sofa", "red lamp"]
    assert [p["max_detections"] for p in body["prompts"]] == [2, 1]
    assert len(body["instances"]) == 3
    assert all(
        set(i) == {"prompt", "score", "box"} and len(i["box"]) == 4 for i in body["instances"]
    )
    mask = _decode_mask(body["mask"])
    assert mask.size == (64, 48) and mask.mode == "L"


def test_segment_route_errors(client, monkeypatch):
    http, engine = client
    url = "/api/inference/images/segment"
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.setattr(seg, "model_dir", lambda: None)
    r = http.post(url, json = {"image": _png_b64(), "prompt": "sofa"})
    assert r.status_code == 409 and "not downloaded" in r.json()["detail"]
    r = http.post(url, json = {"image": _png_b64(), "prompt": " , , "})
    assert r.status_code == 400
    r = http.post(url, json = {"image": _png_b64(), "prompt": "sofa", "threshold": 3})
    assert r.status_code == 422
    r = http.post(url, json = {"image": "!!!", "prompt": "sofa"})
    assert r.status_code == 400
    monkeypatch.setattr(seg, "model_dir", lambda: "/nonexistent")
    with engine._generate_lock:
        r = http.post(url, json = {"image": _png_b64(), "prompt": "sofa"})
    assert r.status_code == 409 and "generating" in r.json()["detail"]
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: False)
    r = http.post(url, json = {"image": _png_b64(), "prompt": "sofa"})
    assert r.status_code == 501


def test_status_download_cancel_release_routes(client, monkeypatch):
    http, _engine = client
    from hub.services.models import account_access

    monkeypatch.setattr(account_access, "require_installation_owner", lambda: None)
    monkeypatch.setattr(
        seg, "status", lambda: {"supported": True, "downloaded": False, "override": False}
    )
    assert http.get("/api/inference/images/segment/status").json()["downloaded"] is False
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: True)
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE, raising = False)
    monkeypatch.setattr(seg, "active_variant", lambda: seg.SEGMENTER_VARIANTS["sam3.1"])
    started = {}
    monkeypatch.setattr(
        seg,
        "start_download",
        lambda token, variant: started.update(token = token, variant = variant) or "dl-1",
    )
    monkeypatch.setattr(seg, "download_status", lambda: {"downloading": True})
    r = http.post("/api/inference/images/segment/download", json = {"hf_token": "hf_x"})
    assert r.status_code == 200 and r.json()["download_id"] == "dl-1"
    assert started == {"token": "hf_x", "variant": "sam3.1"}
    body = r.json()
    assert (body["variant"], body["repo"], body["gated"]) == ("sam3.1", seg.SAM31_REPO, False)
    assert body["size_bytes"] == seg.SAM31_SIZE_BYTES and body["downloading"] is True
    monkeypatch.setenv(seg.ENV_PATH_OVERRIDE, "/somewhere")
    r = http.post("/api/inference/images/segment/download")
    assert r.status_code == 409 and seg.SAM31_REPO in r.json()["detail"]
    monkeypatch.delenv(seg.ENV_PATH_OVERRIDE)
    monkeypatch.setattr(seg, "cancel_download", lambda: True)
    r = http.post("/api/inference/images/segment/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True
    released = []
    monkeypatch.setattr(seg, "release_segmenter", lambda reason: released.append(reason) or True)
    r = http.post("/api/inference/images/segment/release")
    assert r.status_code == 200 and released == ["mask painter closed"]
    monkeypatch.setattr(seg, "transformers_supports_sam3", lambda: False)
    monkeypatch.setattr(seg, "status", lambda: {"supported": False, "reason": "old transformers"})
    r = http.post("/api/inference/images/segment/download")
    assert r.status_code == 501


def test_a_no_torch_install_reports_sam3_unsupported(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a)
    )
    assert seg.transformers_supports_sam3() is False
    body = seg.status()
    assert body["supported"] is False and "PyTorch" in body["reason"]
