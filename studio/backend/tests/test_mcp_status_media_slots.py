# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studio_status image and video slots carry what an agent needs to call the generate tools right: workflows, the defaults an omitted value resolves to, the size multiple, the image cap, the prompt tip and the video controls."""

from studio_mcp.outputs import ImageSlot, VideoSlot
from studio_mcp.tools.status import BUILDERS, _media, _placed

from .test_mcp_tools_status import PAYLOADS, _status

IMAGE_STATUS = {
    "loaded": True,
    "repo_id": "/srv/hf/qwen",
    "display_repo_id": "Qwen/Qwen-Image-2.1",
    "family": "qwen-image-2.1",
    "workflows": ["txt2img", "img2img", "edit", "reference"],
    # The status route types these as floats.
    "generation_defaults": {"steps": 8.0, "guidance": 4.0},
    "conditioning": {"dimension_multiple": 32, "max_condition_images": 10, "other": "x"},
    "prompt_hint": "Refer to inputs as Image 1, Image 2.",
}
LTX_STATUS = {
    "loaded": True,
    "repo_id": "Lightricks/LTX-2",
    "supports_last_frame": False,
    "supports_references": False,
    "defaults": {"resolution_presets": [[768, 512], [512, 768], ["x", 1], [640]]},
    "prompt_hint": None,
}
H3_STATUS = {
    "loaded": True,
    "repo_id": "MiniMax/H3",
    "supports_last_frame": True,
    "supports_references": True,
    "defaults": {"resolution_presets": [[1280, 720]]},
    "prompt_hint": "A tip.",
}


def test_image_slot_reads_workflows_defaults_and_conditioning_limits():
    slot = _media(ImageSlot, IMAGE_STATUS)
    assert slot.model == "Qwen/Qwen-Image-2.1"
    assert slot.family == "qwen-image-2.1"
    assert slot.workflows == ["txt2img", "img2img", "edit", "reference"]
    assert slot.generation_defaults == {"steps": 8, "guidance": 4.0}
    assert type(slot.generation_defaults["steps"]) is int
    assert slot.dimension_multiple == 32
    assert slot.max_condition_images == 10
    assert slot.prompt_hint == "Refer to inputs as Image 1, Image 2."


def test_image_slot_drops_malformed_fields():
    payload = {
        **IMAGE_STATUS,
        "generation_defaults": {"steps": None, "guidance": 1.0},
        "conditioning": {"dimension_multiple": True, "max_condition_images": "10"},
        "workflows": ["txt2img", 3],
        "prompt_hint": "",
    }
    slot = _media(ImageSlot, payload)
    assert slot.generation_defaults is None
    assert slot.dimension_multiple is None
    assert slot.max_condition_images is None
    assert slot.workflows == ["txt2img"]
    assert slot.prompt_hint is None


def test_video_slot_reads_controls_and_size_presets():
    ltx = _media(VideoSlot, LTX_STATUS)
    assert ltx.model == "Lightricks/LTX-2"
    assert ltx.supports_last_frame is False
    assert ltx.references is False
    assert ltx.size_presets == ["768x512", "512x768"]
    assert ltx.prompt_hint is None
    h3 = _media(VideoSlot, H3_STATUS)
    assert (h3.supports_last_frame, h3.references, h3.size_presets) == (True, True, ["1280x720"])
    assert h3.prompt_hint == "A tip."


def test_unloaded_slots_keep_their_defaults():
    assert _media(ImageSlot, {**IMAGE_STATUS, "loaded": False}) == ImageSlot()
    assert _media(VideoSlot, {**H3_STATUS, "loaded": False}) == VideoSlot()


def test_builders_still_use_media():
    # _placed builds the slot with _media, then adds where the model ended up.
    for name, slot in (("image", ImageSlot), ("video", VideoSlot)):
        assert BUILDERS[name].func is _placed
        assert BUILDERS[name].args == (slot,)


def test_studio_status_reports_the_new_fields(monkeypatch):
    overrides = {
        ("GET", "/api/inference/images/status"): IMAGE_STATUS,
        ("GET", "/api/inference/video/status"): H3_STATUS,
    }
    status = _status(monkeypatch, {**PAYLOADS, **overrides})[0]["structuredContent"]
    assert status["image"]["generation_defaults"] == {"steps": 8, "guidance": 4.0}
    assert status["image"]["max_condition_images"] == 10
    assert status["image"]["prompt_hint"] == "Refer to inputs as Image 1, Image 2."
    assert status["video"]["references"] is True
    assert status["video"]["size_presets"] == ["1280x720"]
