# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where the pre-download guard from issue #9130 is allowed to speak, and what it does when
the Hub answers badly.

test_diffusion_predownload_memory_guard.py hand-builds a DeviceMemory, which proves the
arithmetic but assumes the classification. These drive the REAL ``snapshot_device_memory``
over a faked driver, once per platform and vendor, so "discrete VRAM and plain CPU are
untouched" is tested rather than asserted.

The second half covers the one new network read: however model_index.json comes back,
staging must fall back to the old best-effort listing rather than refuse or raise.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_memory as memory_mod
from core.inference.diffusion import DiffusionBackend, _pipeline_components_from_index
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.diffusion_memory import DeviceMemory, snapshot_device_memory

MIB = 1024 * 1024
GIB_MIB = 1024

# unsloth/FLUX.2-dev, 112.9 GB: too large for every pool below, so a machine that keeps
# loading it is one the guard genuinely never reaches.
FLUX2_DEV = [
    (f"{name}/model.safetensors", mib * MIB)
    for name, mib in (
        ("transformer", 61461),
        ("text_encoder", 45798),
        ("vae", 321),
        ("tokenizer", 16),
    )
]
# unsloth/Lumina-Image-2.0, 20 GB stored fp32: the control that must still load.
LUMINA_2 = [
    (f"{name}/model.safetensors", mib * MIB)
    for name, mib in (
        ("transformer", 9956),
        ("text_encoder", 9973),
        ("vae", 320),
        ("tokenizer", 21),
    )
]

# Every OS Unsloth ships on. The classifier reads the device and the driver's `integrated`
# flag and never sys.platform, so these are here to prove that rather than to vary it.
PLATFORMS = ("linux", "wsl", "win32", "darwin")


def _target(
    device = "cuda",
    *,
    dtype = "bfloat16",
    vendor = "amd",
):
    return DiffusionDeviceTarget(
        device = device,
        dtype = dtype,
        backend = device,
        vendor = vendor,
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = True,
        ordinal = None,
    )


def _classify(monkeypatch, *, device, integrated, total_mib, platform):
    """The real snapshot_device_memory over a faked driver, as ``platform`` would see it."""
    monkeypatch.setattr(sys, "platform", "linux" if platform == "wsl" else platform)
    if platform == "wsl":
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    else:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising = False)

    props = types.SimpleNamespace(integrated = integrated)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(
                current_device = lambda: 0,
                get_device_properties = lambda _i: props,
            ),
            xpu = None,
        ),
    )
    hardware = types.ModuleType("utils.hardware")
    hardware.trusted_mem_get_info = lambda: (total_mib * MIB, total_mib * MIB)
    monkeypatch.setitem(sys.modules, "utils.hardware", hardware)
    monkeypatch.setattr(memory_mod, "_system_memory_mib", lambda: (total_mib, total_mib))
    return snapshot_device_memory(_target(device))


def _guard(
    monkeypatch,
    snapshot,
    *,
    device = "cuda",
    dtype = "bfloat16",
):
    backend = DiffusionBackend()
    monkeypatch.setattr(
        backend, "_target_for_ordinal", lambda *_a, **_k: _target(device, dtype = dtype)
    )
    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", lambda _t: snapshot)

    def verdict(files):
        return backend.declared_footprint_shortfall(
            types.SimpleNamespace(name = "flux.2-dev", base_repo = "black-forest-labs/FLUX.2-dev"),
            "unsloth/FLUX.2-dev",
            "black-forest-labs/FLUX.2-dev",
            kind = "pipeline",
            declared_files = files,
        )

    return verdict


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("total_mib", [8 * GIB_MIB, 24 * GIB_MIB, 96 * GIB_MIB])
def test_a_discrete_card_keeps_loading_what_it_loads_today(monkeypatch, platform, total_mib):
    """Discrete VRAM has somewhere to offload to, so an oversized pipeline still loads and
    streams from host RAM. Asserted at three card sizes because the refusal must be keyed
    on the memory KIND and never on the pipeline being larger than the card."""
    snapshot = _classify(
        monkeypatch,
        device = "cuda",
        integrated = False,
        total_mib = total_mib,
        platform = platform,
    )
    assert snapshot.memory_kind == "discrete_vram"
    assert _guard(monkeypatch, snapshot)(FLUX2_DEV) is None


@pytest.mark.parametrize("platform", ["linux", "wsl", "win32"])
def test_an_integrated_gpu_is_the_one_machine_that_is_judged(monkeypatch, platform):
    """gfx1151 and the other APUs: one pool, no offload target, so the OS kills an
    oversized load outright. Refused before the download, and a model that fits is not."""
    snapshot = _classify(
        monkeypatch,
        device = "cuda",
        integrated = True,
        total_mib = 64 * GIB_MIB,
        platform = platform,
    )
    assert snapshot.memory_kind == "unified_memory"
    verdict = _guard(monkeypatch, snapshot)
    assert verdict(FLUX2_DEV) is not None
    assert verdict(LUMINA_2) is None


def test_apple_silicon_is_judged_the_same_way(monkeypatch):
    snapshot = _classify(
        monkeypatch,
        device = "mps",
        integrated = False,
        total_mib = 36 * GIB_MIB,
        platform = "darwin",
    )
    assert snapshot.memory_kind == "unified_memory"
    verdict = _guard(monkeypatch, snapshot, device = "mps")
    assert verdict(FLUX2_DEV) is not None
    assert verdict(LUMINA_2) is None


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_cpu_only_host_is_left_alone(monkeypatch, platform):
    """``system_memory`` is deliberately outside the refusal: it has swap, and it is not
    what gets killed. The pre-download check has to honour the same carve-out the load-time
    one already makes, or a CPU install starts refusing models it can page through."""
    snapshot = _classify(
        monkeypatch,
        device = "cpu",
        integrated = False,
        total_mib = 16 * GIB_MIB,
        platform = platform,
    )
    assert snapshot.memory_kind == "system_memory"
    assert _guard(monkeypatch, snapshot, device = "cpu")(FLUX2_DEV) is None


def test_an_intel_gpu_is_left_alone(monkeypatch):
    snapshot = _classify(
        monkeypatch,
        device = "xpu",
        integrated = False,
        total_mib = 16 * GIB_MIB,
        platform = "linux",
    )
    assert snapshot.memory_kind == "discrete_vram"
    assert _guard(monkeypatch, snapshot, device = "xpu")(FLUX2_DEV) is None


def test_a_driver_that_will_not_answer_is_left_alone(monkeypatch):
    """An uninitialised or absent CUDA runtime yields no totals, which is not evidence
    that anything is too large."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(
                current_device = lambda: 0,
                get_device_properties = lambda _i: (_ for _ in ()).throw(RuntimeError("no ctx")),
            ),
        ),
    )
    hardware = types.ModuleType("utils.hardware")
    hardware.trusted_mem_get_info = lambda: (_ for _ in ()).throw(RuntimeError("driver"))
    monkeypatch.setitem(sys.modules, "utils.hardware", hardware)
    snapshot = snapshot_device_memory(_target("cuda"))
    assert snapshot.memory_kind == "discrete_vram" and snapshot.total_mib is None
    assert _guard(monkeypatch, snapshot)(FLUX2_DEV) is None


@pytest.mark.parametrize("memory_mode", [None, "auto", "fast", "balanced", "low_vram"])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_no_offload_request_can_talk_a_shared_pool_into_it(monkeypatch, memory_mode, cpu_offload):
    """Offloading inside one pool frees nothing, so no requested mode may turn the refusal
    off -- and none of them may turn it ON for a model that fits either."""
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(
        diffusion_mod,
        "snapshot_device_memory",
        lambda _t: DeviceMemory("cuda", "cuda", "unified_memory", 64 * GIB_MIB, 64 * GIB_MIB),
    )

    def verdict(files, base):
        return backend.declared_footprint_shortfall(
            types.SimpleNamespace(name = "flux.2-dev", base_repo = base),
            "unsloth/FLUX.2-dev",
            base,
            kind = "pipeline",
            declared_files = files,
            memory_mode = memory_mode,
            cpu_offload = cpu_offload,
        )

    assert verdict(FLUX2_DEV, "black-forest-labs/FLUX.2-dev") is not None
    assert verdict(LUMINA_2, "Alpha-VLLM/Lumina-Image-2.0") is None


# ── the manifest read ─────────────────────────────────────────────────────────

_MANIFEST = {
    "_class_name": "FluxPipeline",
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "safety_checker": [None, None],
    "_ignore_files": ["transformer/diffusion_pytorch_model.fp16.safetensors"],
}


def _info(*, siblings = ("model_index.json",), sha = "deadbeef"):
    return types.SimpleNamespace(
        siblings = None
        if siblings is None
        else [types.SimpleNamespace(rfilename = name) for name in siblings],
        sha = sha,
    )


def _stub_manifest(
    monkeypatch,
    tmp_path,
    payload,
    *,
    raises = None,
):
    calls: list = []

    def _download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs.get("revision")))
        if raises is not None:
            raise raises
        path = tmp_path / "model_index.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    return calls


def test_the_manifest_names_the_components_and_the_revision_it_was_read_at(monkeypatch, tmp_path):
    calls = _stub_manifest(monkeypatch, tmp_path, _MANIFEST)
    selected, ignored = _pipeline_components_from_index("repo", _info(sha = "abc123"), None)
    # A component declared [None, None] is not loaded, so its files are not priced.
    assert selected == frozenset({"transformer", "text_encoder", "vae"})
    assert ignored == frozenset({"transformer/diffusion_pytorch_model.fp16.safetensors"})
    assert calls == [("repo", "model_index.json", "abc123")]


@pytest.mark.parametrize(
    "payload, siblings, raises",
    [
        ("{not json", ("model_index.json",), None),
        ("[1, 2, 3]", ("model_index.json",), None),
        ('"a string"', ("model_index.json",), None),
        ("null", ("model_index.json",), None),
        ({}, ("model_index.json",), None),
        ({"_class_name": "FluxPipeline"}, ("model_index.json",), None),
        ({"safety_checker": [None, None]}, ("model_index.json",), None),
        ({"transformer": "not-a-list"}, ("model_index.json",), None),
        ({"transformer": ["diffusers"]}, ("model_index.json",), None),
        (_MANIFEST, ("transformer/model.safetensors",), None),
        (_MANIFEST, (), None),
        (_MANIFEST, None, None),
        (_MANIFEST, ("model_index.json",), OSError("hub unreachable")),
        (_MANIFEST, ("model_index.json",), PermissionError("gated")),
    ],
    ids = [
        "invalid-json",
        "a-list",
        "a-string",
        "null",
        "empty",
        "only-private-keys",
        "every-component-disabled",
        "malformed-spec",
        "short-spec",
        "no-manifest-listed",
        "empty-listing",
        "null-listing",
        "download-fails",
        "download-401s",
    ],
)
def test_a_manifest_that_cannot_be_read_declines_instead_of_raising(
    monkeypatch, tmp_path, payload, siblings, raises
):
    """None here means staging keeps the previous best-effort listing and resident sizing
    issues no hard verdict, which is the whole fail-open contract."""
    _stub_manifest(monkeypatch, tmp_path, payload, raises = raises)
    failures: list = []
    assert (
        _pipeline_components_from_index(
            "repo", _info(siblings = siblings), None, failures_out = failures
        )
        is None
    )
    assert len(failures) == 1


def test_an_ignore_list_of_the_wrong_shape_is_tolerated(monkeypatch, tmp_path):
    _stub_manifest(monkeypatch, tmp_path, dict(_MANIFEST, _ignore_files = "not-a-list"))
    selected, ignored = _pipeline_components_from_index("repo", _info(), None)
    assert selected == frozenset({"transformer", "text_encoder", "vae"})
    assert ignored == frozenset()


# ── the two switches the plan exposes ─────────────────────────────────────────


def _plan_probe(monkeypatch, calls):
    """A download_plan whose device-dependent steps announce themselves."""
    backend = DiffusionBackend()
    # The real registry entry: download_plan reads more of it than a stub can carry.
    fam = detect_family_for_pick("unsloth/FLUX.2-dev", None, None)
    assert fam is not None
    monkeypatch.setattr(diffusion_mod, "detect_family_for_pick", lambda *_a, **_k: fam)
    monkeypatch.setattr(diffusion_mod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(diffusion_mod, "_assert_base_repo_accessible", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_mod, "flux2_pick_mismatch", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_mod, "speech_pick_refusal", lambda *_a, **_k: None)
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(
        diffusion_mod,
        "snapshot_device_memory",
        lambda _t: DeviceMemory("cuda", "cuda", "unified_memory", 64 * GIB_MIB, 64 * GIB_MIB),
    )

    def _te(*_a, **_k):
        calls.append("te_prequant")
        return {}

    def _dit(*_a, **_k):
        calls.append("dit_prequant")
        return None

    def _estimate(*_a, **kwargs):
        out = kwargs.get("file_sizes_out")
        if out is not None:
            out["unsloth/FLUX.2-dev"] = {name: size for name, size in FLUX2_DEV}
        resident = kwargs.get("resident_file_sizes_out")
        if resident is not None:
            resident.extend(FLUX2_DEV)
        return sum(size for _name, size in FLUX2_DEV), []

    monkeypatch.setattr(DiffusionBackend, "_te_prequant_plan_files", _te)
    monkeypatch.setattr(DiffusionBackend, "_dit_prequant_plan_source", _dit)
    monkeypatch.setattr(DiffusionBackend, "_estimate_download_bytes", staticmethod(_estimate))
    return backend


def test_suppressing_the_verdict_leaves_the_file_scope_alone(monkeypatch):
    """``memory_verdict=False`` is for callers that want today's plan and no refusal --
    a byte count taken over a different file list than the load will fetch is how a
    "fully downloaded" answer goes wrong."""
    calls: list = []
    backend = _plan_probe(monkeypatch, calls)
    plan = backend.download_plan("unsloth/FLUX.2-dev", model_kind = "pipeline", memory_verdict = False)
    assert plan["incompatible_reason"] is None
    assert calls == ["te_prequant", "dit_prequant"]


def test_clearing_the_probe_suppresses_the_verdict_too(monkeypatch):
    """The training route wants both off: it must not open a second CUDA context, and
    without a target it cannot resolve precision at all."""
    calls: list = []
    backend = _plan_probe(monkeypatch, calls)
    plan = backend.download_plan(
        "unsloth/FLUX.2-dev", model_kind = "pipeline", allow_device_probe = False
    )
    assert plan["incompatible_reason"] is None
    assert calls == []


def test_the_default_plan_still_refuses_an_oversized_pipeline(monkeypatch):
    calls: list = []
    backend = _plan_probe(monkeypatch, calls)
    plan = backend.download_plan("unsloth/FLUX.2-dev", model_kind = "pipeline")
    assert plan["incompatible_reason"] is not None
    assert "unified memory" in plan["incompatible_reason"]
