# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-Blackwell NVFP4 picks load INT8 (or Auto off NVIDIA); Blackwell and the kill switch keep today's behaviour."""

from __future__ import annotations

import asyncio
import types

import pytest

from core.inference import diffusion_nvfp4_route as route
from core.inference.diffusion_nvfp4_route import DeviceProfile

CAPS = [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]
DEVICES = ["cuda", "rocm", "mps", "cpu", "xpu"]
PICKS = ["dit", "te", "hosted_source", "hosted_repo", "ltx_packed", "local_unknown", "auto"]

LTX_FILE = "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"


@pytest.fixture(autouse = True)
def _nvfp4_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.delenv(route.ROUTE_ENV, raising = False)
    yield


def _profile(device, cap):
    if device == "cuda":
        return DeviceProfile("nvidia", (cap,))
    if device == "rocm":
        # gfx versions come through the same call: RDNA 3 reads 11.x, which must never count as Blackwell.
        return DeviceProfile("rocm", ((11, 0),))
    return DeviceProfile(device, ())


def _fake_device(monkeypatch, device, cap):
    monkeypatch.setattr(route, "device_profile", lambda **kw: _profile(device, cap))


def _ltx_folder(tmp_path):
    path = tmp_path / "cm" / LTX_FILE
    path.parent.mkdir(parents = True)
    path.write_bytes(b"\0" * 64)
    return str(tmp_path / "cm")


def _unknown_file(tmp_path):
    folder = tmp_path / "my_models"
    folder.mkdir(parents = True)
    (folder / "mystery-model-nvfp4.safetensors").write_bytes(b"\0" * 64)
    return str(folder), "mystery-model-nvfp4.safetensors"


def _pick(kind, tmp_path):
    if kind == "dit":
        return "image", dict(model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4")
    if kind == "te":
        return "image", dict(model_path = "Qwen/Qwen-Image-2.1", text_encoder_quant = "nvfp4")
    if kind == "hosted_source":
        # A family with a hosted NVFP4 prequant repo: the scheme decides which hosted file is fetched.
        return "image", dict(model_path = "Qwen/Qwen-Image-2.1", transformer_quant = "nvfp4")
    if kind == "hosted_repo":
        return "image", dict(model_path = "unsloth/Qwen-Image-2.1-NVFP4")
    if kind == "ltx_packed":
        return "video", dict(
            model_path = _ltx_folder(tmp_path),
            gguf_filename = LTX_FILE,
            model_kind = "single_file",
            base_repo = "Lightricks/LTX-2.5-Diffusers",
            family_override = "ltx-2.5",
        )
    if kind == "local_unknown":
        folder, name = _unknown_file(tmp_path)
        return "image", dict(model_path = folder, gguf_filename = name, model_kind = "single_file")
    return "image", dict(model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "auto")


def _expected(device, cap, kind):
    blackwell = device == "cuda" and cap[0] >= 10
    if kind in ("auto", "local_unknown") or blackwell:
        return None
    nvidia = device == "cuda"
    floor = (8, 0) if kind == "ltx_packed" else (7, 5)
    dit = "int8" if nvidia and cap >= floor else "auto"
    te = "int8" if nvidia and cap[0] >= 8 else "auto"
    if kind == "dit" or kind == "hosted_source":
        return {"transformer_quant": dit}
    if kind == "te":
        return {"text_encoder_quant": te}
    if kind == "hosted_repo":
        return {
            "model_path": "Qwen/Qwen-Image-2.1",
            "model_kind": "pipeline",
            "gguf_filename": None,
            "transformer_quant": dit,
        }
    if kind == "ltx_packed":
        return {
            "model_path": "Lightricks/LTX-2.5-Diffusers",
            "model_kind": "pipeline",
            "gguf_filename": None,
            "base_repo": None,
            "transformer_quant": dit,
        }
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", PICKS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cap", CAPS, ids = lambda c: f"sm{c[0]}{c[1]}")
def test_routing_matrix(monkeypatch, tmp_path, cap, device, kind):
    _fake_device(monkeypatch, device, cap)
    media, kwargs = _pick(kind, tmp_path)
    got = route.plan_nvfp4_route(media = media, **kwargs)
    want = _expected(device, cap, kind)
    if want is None:
        assert got is None
        return
    assert got is not None
    for key, value in want.items():
        assert got.updates[key] == value, (key, got.updates)
    assert got.reason.startswith("NVFP4 needs a Blackwell GPU; using ")
    expected_word = (
        "INT8"
        if want.get("transformer_quant", want.get("text_encoder_quant")) == "int8"
        else "Auto"
    )
    assert f"using {expected_word}" in got.reason
    if "transformer_quant" in want:
        assert got.requested["transformer_quant"] == "nvfp4"
    if kind in ("hosted_repo", "ltx_packed"):
        assert got.replaced.startswith("nvfp4:")


@pytest.mark.parametrize("kind", PICKS)
def test_kill_switch_restores_todays_request(monkeypatch, tmp_path, kind):
    _fake_device(monkeypatch, "cuda", (8, 9))
    monkeypatch.setenv(route.ROUTE_ENV, "0")
    media, kwargs = _pick(kind, tmp_path)
    assert route.plan_nvfp4_route(media = media, **kwargs) is None


def test_the_nvfp4_switch_off_keeps_the_400_for_an_nvfp4_scheme(monkeypatch, tmp_path):
    """The master switch decides whether NVFP4 exists; the route never turns its 400 into a load."""
    _fake_device(monkeypatch, "cuda", (8, 9))
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    for kind in ("dit", "te", "hosted_source", "hosted_repo"):
        media, kwargs = _pick(kind, tmp_path)
        assert route.plan_nvfp4_route(media = media, **kwargs) is None, kind
    # The packed single-file path loads today with the switch off, so it is still routed.
    media, kwargs = _pick("ltx_packed", tmp_path)
    assert route.plan_nvfp4_route(media = media, **kwargs).updates["transformer_quant"] == "int8"


def test_an_unknown_nvfp4_file_keeps_its_dit_precision_but_not_its_te_pick(
    monkeypatch, tmp_path, caplog
):
    _fake_device(monkeypatch, "cuda", (8, 9))
    folder, name = _unknown_file(tmp_path)
    with caplog.at_level("INFO", logger = route.logger.name):
        got = route.plan_nvfp4_route(
            model_path = folder,
            gguf_filename = name,
            model_kind = "single_file",
            transformer_quant = "nvfp4",
            text_encoder_quant = "nvfp4",
        )
    assert dict(got.updates) == {"text_encoder_quant": "int8"}
    assert got.kept == (name,)
    assert any("kept" in r.getMessage() and name in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "name,base",
    [
        ("FLUX.2-klein-9B-NVFP4.safetensors", "black-forest-labs/FLUX.2-klein-9B"),
        ("flux-2-klein-4b-nvfp4.safetensors", "black-forest-labs/FLUX.2-klein-4B"),
        ("Qwen-Image-2.1-NVFP4.safetensors", "Qwen/Qwen-Image-2.1"),
        # Spells a base, but no official NVFP4 file has this name: kept.
        ("qwen-image-2.1-turbo-nvfp4.safetensors", None),
        ("flux1-dev-nvfp4.safetensors", "black-forest-labs/FLUX.1-dev"),
        # Spells a variant whose canonical spelling the family never states: kept, never another model.
        ("flux1-krea-dev-nvfp4.safetensors", None),
    ],
)
def test_a_known_family_file_keeps_its_variant(monkeypatch, tmp_path, name, base):
    _fake_device(monkeypatch, "cuda", (8, 9))
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / name).write_bytes(b"\0" * 64)
    got = route.plan_nvfp4_route(
        model_path = str(folder), gguf_filename = name, model_kind = "single_file"
    )
    if base is None:
        assert got is None
    else:
        assert got.updates["model_path"] == base
        assert got.updates["transformer_quant"] == "int8"


@pytest.mark.parametrize(
    "name,override,base",
    [
        (
            "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            "ltx-2.5",
            "Lightricks/LTX-2.5-Diffusers",
        ),
        # The pack runs only its distilled DiT: a dev file is a different model.
        ("ltx-2.5-22b-dev-nvfp4.safetensors", "ltx-2.5", None),
        # The ltx-2 family spans 2.0 to 2.3 on a 2.0 pipeline.
        ("ltx-2.3-22b-distilled-nvfp4.safetensors", None, None),
        ("Wan2.2-TI2V-5B-nvfp4.safetensors", None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
    ],
)
def test_a_video_file_is_replaced_only_by_the_same_model(
    monkeypatch, tmp_path, name, override, base
):
    _fake_device(monkeypatch, "cuda", (8, 0))
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / name).write_bytes(b"\0" * 64)
    got = route.plan_nvfp4_route(
        media = "video",
        model_path = str(folder),
        gguf_filename = name,
        model_kind = "single_file",
        family_override = override,
    )
    if base is None:
        assert got is None
    else:
        assert got.updates["model_path"] == base
        assert got.updates["transformer_quant"] == "int8"


def _finetune_file(
    folder,
    name,
    base_model = None,
):
    import json

    import torch
    from safetensors.torch import save_file

    meta = {"scheme": "nvfp4", "family": "flux.2-klein"}
    if base_model:
        meta["base_model_id"] = base_model
    save_file(
        {"w": torch.tensor([42.0])},
        str(folder / name),
        metadata = {
            "unsloth_format": "unsloth_prequant_transformer_state_dict_v1",
            "unsloth_metadata": json.dumps(meta),
        },
    )


@pytest.mark.parametrize(
    "model_path,name,base_model",
    [
        ("local", "flux-2-klein-4b-my-finetune-nvfp4.safetensors", None),
        ("local", "flux-2-klein-4b-my-finetune-nvfp4.safetensors", "my-team/personal-finetune"),
        # The official file name, but the header names another base.
        ("local", "flux-2-klein-4b-nvfp4.safetensors", "my-team/personal-finetune"),
        # The official file name, but cached from someone else's repo.
        ("hf_cache", "flux-2-klein-4b-nvfp4.safetensors", None),
        ("my-team/FLUX.2-klein-4B-personal-finetune", "flux-2-klein-4b-nvfp4.safetensors", None),
        ("my-team/FLUX.2-klein-4B-personal-finetune-nvfp4", None, None),
    ],
)
def test_a_custom_nvfp4_checkpoint_keeps_its_own_weights(
    monkeypatch, tmp_path, model_path, name, base_model
):
    _fake_device(monkeypatch, "cuda", (8, 9))
    if model_path in ("local", "hf_cache"):
        folder = tmp_path / "models"
        if model_path == "hf_cache":
            folder = tmp_path / "hub" / "models--my-team--klein-ft" / "snapshots" / "abc"
        folder.mkdir(parents = True)
        _finetune_file(folder, name, base_model)
        model_path = str(folder)
    got = route.plan_nvfp4_route(
        model_path = model_path, gguf_filename = name, model_kind = "single_file"
    )
    assert got is None or "model_path" not in got.updates, dict(got.updates)


@pytest.mark.parametrize(
    "model_path,name,base",
    [
        (
            "black-forest-labs/FLUX.2-klein-4b-nvfp4",
            "flux-2-klein-4b-nvfp4.safetensors",
            "black-forest-labs/FLUX.2-klein-4B",
        ),
        ("black-forest-labs/FLUX.2-klein-9b-nvfp4", None, "black-forest-labs/FLUX.2-klein-9B"),
        ("unsloth/Z-Image-Turbo-NVFP4", None, "Tongyi-MAI/Z-Image-Turbo"),
        ("hf_cache", "flux-2-klein-4b-nvfp4.safetensors", "black-forest-labs/FLUX.2-klein-4B"),
        ("local_header", "Z-Image-Turbo-NVFP4.safetensors", "Tongyi-MAI/Z-Image-Turbo"),
    ],
)
def test_an_official_nvfp4_checkpoint_is_still_replaced(
    monkeypatch, tmp_path, model_path, name, base
):
    _fake_device(monkeypatch, "cuda", (8, 9))
    if model_path == "hf_cache":
        folder = (
            tmp_path
            / "hub"
            / "models--black-forest-labs--FLUX.2-klein-4b-nvfp4"
            / "snapshots"
            / "a"
        )
        folder.mkdir(parents = True)
        (folder / name).write_bytes(b"\0" * 64)
        model_path = str(folder)
    elif model_path == "local_header":
        folder = tmp_path / "models"
        folder.mkdir()
        _finetune_file(folder, name, "Tongyi-MAI/Z-Image-Turbo")
        model_path = str(folder)
    got = route.plan_nvfp4_route(
        model_path = model_path, gguf_filename = name, model_kind = "single_file"
    )
    assert got.updates["model_path"] == base
    assert got.updates["transformer_quant"] == "int8"


def test_every_trusted_nvfp4_file_resolves_to_a_family():
    """A vendor row whose base no family knows would be dead: it must name a base a family loads."""
    for media in ("image", "video"):
        rows = route.trusted_nvfp4_files(media)
        vendor = [r for r in route._VENDOR_NVFP4_FILES if r[0] == media]
        assert len([r for r in rows if not r[0].lower().startswith("unsloth/")]) == len(vendor)
        assert any(r[0].lower().startswith("unsloth/") for r in rows)


def test_a_family_that_denies_int8_gets_auto(monkeypatch):
    from core.inference import diffusion_transformer_quant as tq

    _fake_device(monkeypatch, "cuda", (8, 9))
    monkeypatch.setattr(tq, "_FAMILY_SCHEME_DENY", {"z-image": frozenset({"int8"})})
    got = route.plan_nvfp4_route(model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4")
    assert got.updates["transformer_quant"] == "auto"


def test_an_nvfp4_prequant_path_is_never_handed_to_int8(monkeypatch, tmp_path):
    _fake_device(monkeypatch, "cuda", (8, 0))
    art = tmp_path / "Z-Image-Turbo-NVFP4.safetensors"
    art.write_bytes(b"\0" * 64)
    got = route.plan_nvfp4_route(
        model_path = "Tongyi-MAI/Z-Image-Turbo",
        transformer_quant = "nvfp4",
        transformer_prequant_path = str(art),
    )
    assert got.updates["transformer_prequant_path"] is None
    assert got.updates["transformer_quant"] == "int8"


@pytest.mark.parametrize(
    "caps,ordinal,routes",
    [
        (((10, 0), (8, 0)), None, False),  # auto may land on the B200: keep NVFP4
        (((10, 0), (8, 0)), 1, True),  # pinned to the A100
        (((10, 0), (8, 0)), 0, False),  # pinned to the B200
        (((8, 9), (8, 9)), None, True),
        (((12, 0),), None, False),
    ],
)
def test_mixed_boxes_follow_the_card_the_load_uses(caps, ordinal, routes):
    profile = DeviceProfile("nvidia", caps)
    got = route.plan_nvfp4_route(
        model_path = "Tongyi-MAI/Z-Image-Turbo",
        transformer_quant = "nvfp4",
        gpu_ordinal = ordinal,
        profile = profile,
    )
    assert (got is not None) is routes


def test_an_ordinary_request_never_reads_the_device(monkeypatch):
    monkeypatch.setattr(route, "device_profile", lambda **kw: pytest.fail("read the device"))
    for kwargs in (
        dict(model_path = "Tongyi-MAI/Z-Image-Turbo"),
        dict(
            model_path = "Tongyi-MAI/Z-Image-Turbo",
            transformer_quant = "auto",
            text_encoder_quant = "int8",
        ),
        dict(model_path = "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_M.gguf"),
    ):
        assert route.plan_nvfp4_route(**kwargs) is None


def test_nvidia_smi_answers_without_initialising_cuda(monkeypatch):
    """A cold process reads the cards from nvidia-smi, never torch.cuda (training may be running beside it)."""
    import torch

    from utils.hardware import hardware, nvidia

    route.reset_device_profile()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a: pytest.fail("initialised CUDA")
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("asked torch"))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu130")
    monkeypatch.setattr(
        nvidia, "get_compute_capabilities", lambda: {0: (8, 9), 1: (8, 9), 2: (12, 0)}
    )
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [0, 1])
    try:
        assert route.device_profile() == DeviceProfile("nvidia", ((8, 9), (8, 9)))
    finally:
        route.reset_device_profile()
    # Mixed cards: nvidia-smi rows are not torch ordinals, so it cannot answer.
    for caps in ({0: (8, 9), 1: (12, 0)}, {0: (8, 9), 1: (7, 5)}):
        monkeypatch.setattr(nvidia, "get_compute_capabilities", lambda caps = caps: caps)
        monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [0, 1])
        assert route._smi_visible_capabilities(torch) == ()


def test_training_never_initialises_cuda_for_the_route(monkeypatch):
    """Mixed cards (nvidia-smi cannot answer) while training owns the GPU: left as is, nothing cached."""
    import torch

    route.reset_device_profile()
    monkeypatch.setattr(route, "_smi_visible_capabilities", lambda torch: ())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *a: pytest.fail("initialised CUDA")
    )
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu130")
    try:
        got = route.plan_nvfp4_route(
            model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4", allow_cuda_init = False
        )
        assert got is None
        assert route._PROFILE == []
    finally:
        route.reset_device_profile()


def test_the_route_reads_the_cards_without_cuda_while_training(monkeypatch):
    from models.inference import DiffusionLoadRequest
    from routes import inference as image_routes

    seen = {}

    def _plan(**kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(route, "plan_nvfp4_route", _plan)
    monkeypatch.setattr(image_routes, "_training_is_active", lambda: True)
    pick = DiffusionLoadRequest(model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4")
    asyncio.run(image_routes._route_nvfp4_request(pick, "image"))
    assert seen["allow_cuda_init"]() is False


def test_the_device_is_read_once(monkeypatch):
    import torch

    calls = []
    route.reset_device_profile()
    monkeypatch.setattr(route, "_smi_visible_capabilities", lambda torch: ())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: calls.append("avail") or True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: (8, 9))
    monkeypatch.setattr(torch.version, "hip", None)
    try:
        assert route.device_profile() == DeviceProfile("nvidia", ((8, 9),))
        assert route.device_profile() == DeviceProfile("nvidia", ((8, 9),))
        assert calls == ["avail"]
    finally:
        route.reset_device_profile()


@pytest.mark.parametrize(
    "hip,version",
    [
        ("7.0", "2.9.0+rocm7.0"),
        (None, "2.9.0+rocm7.0"),
    ],  # AMD SDK / Radeon wheels leave torch.version.hip unset
)
def test_rocm_never_reads_as_blackwell(monkeypatch, hip, version):
    import torch

    route.reset_device_profile()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = None: (12, 0))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch, "__version__", version)
    try:
        got = route.plan_nvfp4_route(
            model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4"
        )
        assert got.updates["transformer_quant"] == "auto"
        assert got.device == "rocm"
    finally:
        route.reset_device_profile()


def test_the_resolved_record_reports_the_rerouted_ask(monkeypatch):
    from core.inference.diffusion_auto_policy import build_resolved_record

    _fake_device(monkeypatch, "cuda", (8, 9))
    got = route.plan_nvfp4_route(
        model_path = "Qwen/Qwen-Image-2.1", transformer_quant = "nvfp4", text_encoder_quant = "nvfp4"
    )
    controls = {
        "transformer_quant": ("int8", "int8", "seeded from the hosted checkpoint unsloth/x"),
        "text_encoder_quant": ("int8", "int8", "int8 ConvRot encoder"),
        "memory_mode": (None, "none", "fits"),
    }

    @route.with_route_scope
    def _load(**kwargs):
        assert route.ROUTE_KWARG not in kwargs
        return build_resolved_record(controls)

    record = _load(_nvfp4_route = got, repo_id = "Qwen/Qwen-Image-2.1")
    tq = record["transformer_quant"]
    assert tq["value"] == "int8"
    assert tq["requested"] == "nvfp4"
    assert tq["status"] == "fell_back"
    assert tq["source"] == "explicit"
    assert tq["reason"].startswith("NVFP4 needs a Blackwell GPU; using INT8")
    assert "seeded from the hosted checkpoint" in tq["reason"]
    assert record["text_encoder_quant"]["requested"] == "nvfp4"
    assert record["memory_mode"]["status"] == "applied"
    plain = build_resolved_record(controls)
    assert plain["transformer_quant"]["requested"] == "int8"
    assert plain["transformer_quant"]["status"] == "applied"


def test_the_resolved_record_reports_a_replaced_checkpoint(monkeypatch, tmp_path):
    from core.inference.diffusion_auto_policy import build_resolved_record

    _fake_device(monkeypatch, "cuda", (8, 0))
    media, kwargs = _pick("ltx_packed", tmp_path)
    got = route.plan_nvfp4_route(media = media, **kwargs)
    record = route.annotate_resolved(
        build_resolved_record({"transformer_quant": ("int8", "int8", "hosted int8")}), got
    )
    assert record["transformer_quant"]["replaced"] == f"nvfp4:{LTX_FILE}"
    assert "from Lightricks/LTX-2.5-Diffusers instead of" in record["transformer_quant"]["reason"]


class _Seen(Exception):
    pass


@pytest.mark.parametrize("field", ["transformer_quant", "text_encoder_quant"])
@pytest.mark.parametrize("cap,want", [((8, 9), "int8"), ((10, 0), "nvfp4")])
def test_every_load_and_plan_route_reroutes_before_its_checks(monkeypatch, field, cap, want):
    """All four entries reroute after the disabled-NVFP4 and account refusals, before anything else reads the request."""
    from models.inference import DiffusionLoadRequest, VideoLoadRequest
    from routes import inference as image_routes
    from routes import video as video_routes

    _fake_device(monkeypatch, "cuda", cap)
    real = image_routes._route_nvfp4_request
    seen = []

    async def _spy(
        request,
        media = "image",
        **kwargs,
    ):
        routed, got = await real(request, media, **kwargs)
        seen.append((media, getattr(routed, field), kwargs.get("user_initiated", True)))
        raise _Seen()

    monkeypatch.setattr(image_routes, "_route_nvfp4_request", _spy)
    image = DiffusionLoadRequest(model_path = "Tongyi-MAI/Z-Image-Turbo", **{field: "nvfp4"})
    video = VideoLoadRequest(model_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers", **{field: "nvfp4"})
    for call in (
        image_routes.diffusion_download_plan(image, current_subject = "u"),
        image_routes.load_diffusion_model_gated(image, "u", user_initiated = True),
        video_routes.video_download_plan(video, current_subject = "u"),
        video_routes.load_video_model_gated(video, "u"),
    ):
        with pytest.raises(_Seen):
            asyncio.run(call)
    assert seen == [
        ("image", want, True),
        ("image", want, True),
        ("video", want, True),
        ("video", want, False),
    ]


def test_a_load_nobody_picked_by_hand_keeps_its_checkpoint(monkeypatch, tmp_path):
    """Media auto-switch tracks the model by the path it asked for, so only precision is rerouted there."""
    from models.inference import VideoLoadRequest

    _fake_device(monkeypatch, "cuda", (8, 0))
    media, kwargs = _pick("ltx_packed", tmp_path)
    pick = VideoLoadRequest(**kwargs)
    same, got = route.route_load_request(pick, media = media, replace_checkpoint = False)
    assert got is None and same is pick
    swapped, got = route.route_load_request(pick, media = media)
    assert swapped.model_path == "Lightricks/LTX-2.5-Diffusers"


def test_a_pick_with_its_own_components_keeps_its_checkpoint(monkeypatch, tmp_path):
    _fake_device(monkeypatch, "cuda", (8, 9))
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "flux-2-klein-4b-nvfp4.safetensors").write_bytes(b"\0" * 64)
    got = route.plan_nvfp4_route(
        model_path = str(folder),
        gguf_filename = "flux-2-klein-4b-nvfp4.safetensors",
        model_kind = "single_file",
        component_files = True,
    )
    assert got is None


def test_a_bare_folder_or_file_path_decides_like_its_folder_and_file(monkeypatch, tmp_path):
    _fake_device(monkeypatch, "cuda", (8, 0))
    folder = tmp_path / "ltx"
    folder.mkdir()
    name = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
    (folder / name).write_bytes(b"\0" * 64)
    pair = route.plan_nvfp4_route(media = "video", model_path = str(folder), gguf_filename = name)
    bare = route.plan_nvfp4_route(media = "video", model_path = str(folder))
    whole = route.plan_nvfp4_route(media = "video", model_path = str(folder / name))
    assert pair.updates == bare.updates == whole.updates
    assert pair.updates["model_path"] == "Lightricks/LTX-2.5-Diffusers"


def test_a_missing_local_file_is_not_a_pick(monkeypatch, tmp_path):
    _fake_device(monkeypatch, "cuda", (8, 9))
    assert not route.checkpoint_is_nvfp4(
        str(tmp_path / "nope"), "flux-2-klein-4b-nvfp4.safetensors"
    )
    assert (
        route.plan_nvfp4_route(
            model_path = str(tmp_path / "nope"), gguf_filename = "flux-2-klein-4b-nvfp4.safetensors"
        )
        is None
    )


def test_a_recorded_header_wins_over_the_name_and_pickles_are_never_opened(monkeypatch, tmp_path):
    from core.inference import diffusion_prequant

    opened = []
    monkeypatch.setattr(
        diffusion_prequant, "local_prequant_scheme", lambda path: opened.append(path) or "int8"
    )
    art = tmp_path / "my-finetune-from-nvfp4-int8.safetensors"
    art.write_bytes(b"\0" * 64)
    assert not route.checkpoint_is_nvfp4(str(art))
    pickle = tmp_path / "flux-2-klein-4b-nvfp4.pt"
    pickle.write_bytes(b"\0" * 64)
    assert route.checkpoint_is_nvfp4(str(pickle))
    assert opened == [str(art)]


def test_an_unreadable_device_is_not_cached_and_leaves_the_request(monkeypatch):
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("driver hiccup")

    route.reset_device_profile()
    monkeypatch.setattr(route, "_probe_device_profile", _boom)
    try:
        assert route.device_profile().kind == route.UNKNOWN
        assert route.device_profile().kind == route.UNKNOWN
        assert len(calls) == 2
        got = route.plan_nvfp4_route(
            model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4"
        )
        assert got is None
    finally:
        route.reset_device_profile()


@pytest.mark.parametrize(
    "cap,media,turing_on,want",
    [
        ((7, 5), "image", True, "int8"),
        ((7, 5), "image", False, "auto"),
        ((7, 5), "video", True, "auto"),  # video INT8 needs bf16
        ((8, 0), "video", True, "int8"),
        ((6, 1), "image", True, "auto"),
    ],
)
def test_the_int8_target_follows_what_the_card_runs(monkeypatch, cap, media, turing_on, want):
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setenv(tq.FP16_CARD_INT8_ENV, "1" if turing_on else "0")
    _fake_device(monkeypatch, "cuda", cap)
    repo = "Tongyi-MAI/Z-Image-Turbo" if media == "image" else "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    got = route.plan_nvfp4_route(media = media, model_path = repo, transformer_quant = "nvfp4")
    assert got.updates["transformer_quant"] == want


def test_a_selection_counts_every_card_it_places_work_on(monkeypatch):
    from core.inference import diffusion_device

    monkeypatch.setattr(diffusion_device, "resolve_selected_cuda_ordinals", lambda ids: [0, 1])
    profile = DeviceProfile("nvidia", ((10, 0), (8, 0)))
    # Primary B200 ranked by the route, A100 also selected: the A100 cannot run NVFP4.
    got = route.plan_nvfp4_route(
        model_path = "Tongyi-MAI/Z-Image-Turbo",
        transformer_quant = "nvfp4",
        gpu_ordinal = 0,
        gpu_ids = [0, 1],
        profile = profile,
    )
    assert got is not None and got.updates["transformer_quant"] == "int8"


def test_hosted_repo_lookup_is_unambiguous_and_canonical(monkeypatch):
    _fake_device(monkeypatch, "cuda", (8, 9))
    # HunyuanVideo-1.5 480p and 720p share one NVFP4 repo: nothing tells them apart, so it is kept.
    assert route.hosted_nvfp4_repo_base("unsloth/HunyuanVideo-1.5-NVFP4", "video") is None
    fam, base = route.hosted_nvfp4_repo_base(
        "unsloth/HunyuanVideo-1.5-NVFP4", "video", family_override = "hunyuanvideo-1.5-720p"
    )
    assert fam.name == "hunyuanvideo-1.5-720p" and base == fam.base_repo
    fam, base = route.hosted_nvfp4_repo_base(
        "unsloth/HunyuanVideo-1.5-NVFP4",
        "video",
        filename = "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.safetensors",
    )
    assert fam.name == "hunyuanvideo-1.5-720p"
    # A variant row's base is a lowercase key: returned only in a spelling the family states in full.
    got = route.hosted_nvfp4_repo_base("unsloth/Qwen-Image-2512-NVFP4", "image")
    assert got is None or got[1] != got[1].lower()


def test_the_route_hands_the_reroute_only_to_a_backend_that_records_it():
    from routes.inference import _nvfp4_route_kwarg

    def native(repo_id, **kwargs):
        pass

    def diffusers(repo_id, *, nvfp4_route = None):
        pass

    marker = object()
    assert _nvfp4_route_kwarg(native, marker) == {}
    assert _nvfp4_route_kwarg(diffusers, marker) == {"nvfp4_route": marker}
    # An unchanged decision is handed over too, so the backend does not decide again on another card.
    assert _nvfp4_route_kwarg(diffusers, None) == {"nvfp4_route": route.ROUTE_UNCHANGED}


@pytest.mark.parametrize(
    "backend_name,repo,base",
    [
        ("image", "unsloth/Qwen-Image-2.1-NVFP4", "Qwen/Qwen-Image-2.1"),
        ("video", "unsloth/Wan2.2-T2V-A14B-NVFP4", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
    ],
)
def test_begin_load_reroutes_a_direct_caller(monkeypatch, backend_name, repo, base):
    """Direct callers (no route) are rerouted in ``begin_load`` before validation reads the pick."""
    _fake_device(monkeypatch, "cuda", (8, 6))
    if backend_name == "image":
        from core.inference.diffusion import DiffusionBackend as Backend
    else:
        from core.inference.video import VideoBackend as Backend
    seen = {}

    def _validate(self, repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen.update(kwargs)
        raise _Seen()

    monkeypatch.setattr(Backend, "validate_load_request", _validate)
    with pytest.raises(_Seen):
        Backend().begin_load(repo, transformer_quant = "nvfp4")
    assert seen["repo_id"] == base
    assert seen.get("gguf_filename") is None
    monkeypatch.setenv(route.ROUTE_ENV, "0")
    with pytest.raises(_Seen):
        Backend().begin_load(repo, transformer_quant = "nvfp4")
    assert seen["repo_id"] == repo


_WAN_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


class _RecordingApi:
    def __init__(self, repos):
        self.repos = repos
        self.asked: list = []

    def model_info(
        self,
        repo_id,
        files_metadata = False,
        token = None,
    ):
        self.asked.append(repo_id)
        return types.SimpleNamespace(siblings = self.repos[repo_id])


def _sib(name, size):
    return types.SimpleNamespace(rfilename = name, size = size)


_WAN_FILES = [
    _sib("model_index.json", 1000),
    _sib("transformer/config.json", 1000),
    _sib("transformer/diffusion_pytorch_model.safetensors", 28_000_000_000),
    _sib("transformer_2/config.json", 1000),
    _sib("transformer_2/diffusion_pytorch_model.safetensors", 28_000_000_000),
    _sib("text_encoder/model-00001-of-00001.safetensors", 11_000_000_000),
    _sib("vae/diffusion_pytorch_model.safetensors", 500_000_000),
]
_WAN_SIBLINGS = {
    _WAN_T2V: _WAN_FILES,
    "unsloth/Wan2.2-T2V-A14B-FP8": _WAN_FILES
    + [
        _sib("Wan2.2-T2V-A14B-INT8.pt", 14_000_000_000),
        _sib("Wan2.2-T2V-A14B-INT8-2.pt", 14_000_000_000),
    ],
    "unsloth/Wan2.2-T2V-A14B-NVFP4": [
        _sib("Wan2.2-T2V-A14B-NVFP4.safetensors", 8_000_000_000),
        _sib("Wan2.2-T2V-A14B-transformer_2-NVFP4.safetensors", 8_100_000_000),
    ],
}


def test_the_rerouted_video_plan_asks_no_nvfp4_repo_but_the_kill_switch_does(monkeypatch):
    """The routed request (int8) stages no NVFP4 file; the same pick under the kill switch would stage both."""
    from models.inference import VideoLoadRequest

    import core.inference.video as video_mod
    from core.inference import diffusion_prequant
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend

    def _plan_for(request):
        api = _RecordingApi(_WAN_SIBLINGS)
        monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: api)
        monkeypatch.setattr(
            DiffusionBackend,
            "_hub_file_is_cached",
            staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kw: False),
        )
        monkeypatch.setattr(
            video_mod,
            "select_transformer_quant_scheme",
            lambda target, requested, *a, **k: requested,
        )
        monkeypatch.setattr(
            diffusion_prequant, "restricted_prequant_load_supported", lambda *a, **k: True
        )
        monkeypatch.setattr(video_mod, "_video_seed_stays_resident", lambda fam, **kw: True)
        monkeypatch.setattr(
            video_mod,
            "resolve_diffusion_device_target",
            lambda *a, **k: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
        )
        plan = VideoBackend().download_plan(
            request.model_path, model_kind = "pipeline", transformer_quant = request.transformer_quant
        )
        return api.asked, {f for e in plan["entries"] for f in e["files"]}

    _fake_device(monkeypatch, "cuda", (8, 9))
    pick = VideoLoadRequest(model_path = _WAN_T2V, transformer_quant = "nvfp4")
    routed, got = route.route_load_request(pick, media = "video")
    assert routed.transformer_quant == "int8" and got is not None
    asked, staged = _plan_for(routed)
    assert not any(repo.lower().endswith("-nvfp4") for repo in asked), asked
    assert not any("NVFP4" in f for f in staged), staged

    monkeypatch.setenv(route.ROUTE_ENV, "0")
    kept, none = route.route_load_request(pick, media = "video")
    assert none is None and kept.transformer_quant == "nvfp4"
    asked, staged = _plan_for(kept)
    assert "unsloth/Wan2.2-T2V-A14B-NVFP4" in asked


def _all_family_names():
    from core.inference.diffusion_families import _FAMILIES as image
    from core.inference.video_families import _FAMILIES as video
    return sorted({f.name for f in (*image, *video)})


@pytest.mark.parametrize("cap", CAPS, ids = lambda c: f"sm{c[0]}{c[1]}")
def test_auto_never_orders_nvfp4_below_blackwell(monkeypatch, cap):
    from core.inference import diffusion_nvfp4_gate as gate
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "_nvfp4_gate_passed", lambda *a, **k: True)
    monkeypatch.setattr(tq, "_nvfp4_gate_backend_ok", lambda *a, **k: True)
    monkeypatch.setattr(tq, "_nvfp4_backend_is", lambda *a, **k: True)
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda device = None: False)
    monkeypatch.setattr(gate, "nvfp4_gate_passed", lambda *a, **k: True)
    for family in _all_family_names():
        order = tq._auto_scheme_order(family, "cuda", cap)
        if cap[0] < 10:
            assert "nvfp4" not in order, (family, cap, order)


def test_blackwell_requests_are_returned_untouched(monkeypatch, tmp_path):
    for cap in ((10, 0), (10, 3), (11, 0), (12, 0), (12, 1)):
        _fake_device(monkeypatch, "cuda", cap)
        for kind in PICKS:
            media, kwargs = _pick(kind, tmp_path / f"{cap[0]}{cap[1]}{kind}")
            assert route.plan_nvfp4_route(media = media, **kwargs) is None, (cap, kind)
        from models.inference import DiffusionLoadRequest

        pick = DiffusionLoadRequest(
            model_path = "Tongyi-MAI/Z-Image-Turbo", transformer_quant = "nvfp4"
        )
        same, none = route.route_load_request(pick)
        assert same is pick and none is None
