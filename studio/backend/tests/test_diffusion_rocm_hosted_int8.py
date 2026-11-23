# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit int8 image pick on ROCm RDNA seeds its denoiser from the hosted int8 ConvRot file read as native W8A8
Linears instead of quantising the dense shards at load. AUTO, CUDA, weight-only families and the kill switches keep
the previous routes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion as D  # noqa: E402
from core.inference import diffusion_denoiser_prequant as DP  # noqa: E402
from core.inference import diffusion_native_hosted as NH  # noqa: E402
from core.inference import diffusion_rocm_hosted_int8 as RH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference import rocm_bf16  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402
from core.inference.diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE  # noqa: E402

FLUX3 = ("black-forest-labs/flux-3-image-klein", "flux.3-klein")
QWEN21 = ("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
ZIMAGE = ("Tongyi-MAI/Z-Image-Turbo", "z-image")
H3 = ("MiniMaxAI/MiniMax-H3-Image", "minimax-h3-image")
QWEN = ("Qwen/Qwen-Image", "qwen-image")
GPU_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
ENVS = (
    RH.ROCM_IMAGE_HOSTED_INT8_ENV,
    "UNSLOTH_NATIVE_INT8_ACT",
    "UNSLOTH_NATIVE_INT8_ROT",
    "UNSLOTH_DIFFUSION_INT8_CONVROT",
)


def _fam(pair = FLUX3):
    fam = detect_family(pair[0], override = pair[1])
    assert fam is not None and fam.name == pair[1]
    return fam


def _host(
    monkeypatch,
    *,
    rocm: bool,
    ready: bool = True,
    proven: bool = True,
    arch: str = "gfx1151",
):
    asked: list = []

    def gfx_arch(_torch, ordinal):
        asked.append(ordinal)
        return arch

    monkeypatch.setattr(rocm_bf16, "_device_gfx_arch", gfx_arch)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda ordinal = None: ready)
    monkeypatch.setattr(tq, "_rdna_compile_tier_proven", lambda target: proven)
    for env in ENVS:
        monkeypatch.delenv(env, raising = False)
    return asked


@pytest.fixture
def rocm(monkeypatch):
    _host(monkeypatch, rocm = True)


@pytest.fixture
def nvidia(monkeypatch):
    _host(monkeypatch, rocm = False)


@pytest.mark.parametrize("pair", [FLUX3, QWEN21])
@pytest.mark.parametrize(
    "requested, engaged",
    [
        ("int8", True),
        ("INT8", True),
        ("auto", False),
        (None, False),
        ("fp8", False),
        ("off", False),
    ],
)
def test_rocm_gate_takes_explicit_int8_on_w8a8_families_only(rocm, pair, requested, engaged):
    reason = RH.rocm_image_hosted_int8_reason(_fam(pair), requested, GPU_BF16)
    assert (reason is not None) is engaged


def test_rocm_gate_quotes_the_family_measurement(rocm):
    assert "1.365 vs 1.418 s/step" in RH.rocm_image_hosted_int8_reason(
        _fam(FLUX3), "int8", GPU_BF16
    )
    # Qwen-Image-2.1's explicit int8 already runs W8A8 through its AUTO row.
    assert "83.3 vs 95.1 s/image" in RH.rocm_image_hosted_int8_reason(
        _fam(QWEN21), "int8", GPU_BF16
    )


def test_weight_only_families_keep_the_dense_shards(rocm, monkeypatch):
    # Z-Image runs weight-only int8 on RDNA, and the hosted file is W8A8 only.
    assert RH.rocm_image_hosted_int8_reason(_fam(ZIMAGE), "int8", GPU_BF16) is None
    # Forcing W8A8 for every family makes its runtime build W8A8 too, so the hosted file is that build.
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", "1")
    assert RH.rocm_image_hosted_int8_reason(_fam(ZIMAGE), "int8", GPU_BF16) is not None


@pytest.mark.parametrize("pair", [H3, QWEN])
def test_families_without_a_hosted_convrot_build_are_untouched(rocm, monkeypatch, pair):
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", "1")
    assert RH.rocm_image_hosted_int8_reason(_fam(pair), "int8", GPU_BF16) is None


@pytest.mark.parametrize(
    "env, value",
    [
        (RH.ROCM_IMAGE_HOSTED_INT8_ENV, "0"),
        (RH.ROCM_IMAGE_HOSTED_INT8_ENV, "off"),
        ("UNSLOTH_NATIVE_INT8_ACT", "0"),
        ("UNSLOTH_NATIVE_INT8_ROT", "0"),
        ("UNSLOTH_DIFFUSION_INT8_CONVROT", "0"),
    ],
)
def test_kill_switches_keep_the_dense_shards(rocm, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", GPU_BF16) is None


def test_cuda_and_other_hosts_are_untouched(nvidia):
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", GPU_BF16) is None
    cpu = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, ordinal = None)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", cpu) is None


@pytest.mark.parametrize("ready, proven", [(False, True), (True, False)])
def test_rocm_without_the_compiled_rdna_tier_is_untouched(monkeypatch, ready, proven):
    _host(monkeypatch, rocm = True, ready = ready, proven = proven)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", GPU_BF16) is None


@pytest.mark.parametrize("pair", [FLUX3, QWEN21])
@pytest.mark.parametrize(
    "arch, engaged",
    [("gfx1151", True), ("gfx1100", False), ("gfx1201", False), ("gfx1101", False), ("", False)],
)
def test_only_measured_archs_read_the_hosted_file(monkeypatch, pair, arch, engaged):
    # The RDNA W8A8 kernels accept every gfx11xx / gfx12xx card; only gfx1151 was measured, so the others (and an
    # unreadable arch) keep quantising the dense shards at load. The arch is the target card's, by its ordinal.
    asked = _host(monkeypatch, rocm = True, arch = arch)
    card = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 1)
    reason = RH.rocm_image_hosted_int8_reason(_fam(pair), "int8", card)
    assert (reason is not None) is engaged
    assert asked and set(asked) == {1}


def test_unreadable_arch_keeps_the_dense_shards(monkeypatch):
    _host(monkeypatch, rocm = True)

    def boom(_torch, _ordinal):
        raise RuntimeError("no properties")

    monkeypatch.setattr(rocm_bf16, "_device_gfx_arch", boom)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", GPU_BF16) is None


def test_cuda_never_asks_for_a_gfx_arch(monkeypatch):
    asked = _host(monkeypatch, rocm = False)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", GPU_BF16) is None
    assert asked == []


def test_fp16_target_is_untouched(rocm):
    fp16 = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
    assert RH.rocm_image_hosted_int8_reason(_fam(), "int8", fp16) is None


@pytest.fixture
def eligible(rocm, monkeypatch):
    monkeypatch.setattr(D, "compile_eligible", lambda *a, **k: True)


def _route(**kw):
    args = dict(kind = "pipeline", speed_mode = None, memory_mode = None, cpu_offload = False)
    args.update(kw)
    return D._rocm_hosted_int8_route_reason(_fam(), GPU_BF16, "int8", **args)


def test_route_follows_the_compiled_rdna_int8_gates(eligible, monkeypatch):
    assert _route() is not None
    assert _route(speed_mode = "default") is not None
    assert _route(speed_mode = "max") is not None
    assert _route(speed_mode = "off") is None
    assert _route(kind = "gguf") is None
    assert _route(kind = "single_file") is None
    assert _route(memory_mode = "balanced") is None
    assert _route(memory_mode = "low_vram") is None
    assert _route(cpu_offload = True) is None
    monkeypatch.setattr(D, "compile_eligible", lambda *a, **k: False)
    assert _route() is None


class _Planner:
    """Just the backend methods ``_rocm_hosted_int8_seed`` calls."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def _seeded_pipeline_plan(self, scheme, *a, **k):
        self.calls.append(scheme)
        return self.plan


def _seed(planner, **kw):
    args = dict(
        raw = "int8",
        kind = "pipeline",
        speed_mode = None,
        memory_mode = None,
        cpu_offload = False,
        base = FLUX3[0],
        repo_id = FLUX3[0],
        fetch_base = FLUX3[0],
        transformer_prequant_path = None,
        local_files_only = False,
        text_encoder_quant = None,
    )
    args.update(kw)
    return D.DiffusionBackend._rocm_hosted_int8_seed(planner, _fam(), GPU_BF16, **args)


def _flux3_source():
    from core.inference.diffusion_prequant import PrequantSource
    return PrequantSource(
        kind = "repo",
        location = "unsloth/flux-3-image-klein-FP8",
        filename = "flux-3-image-klein-INT8-ConvRot.safetensors",
        fallback_filenames = ("flux-3-image-klein-INT8.safetensors",),
    )


@pytest.fixture
def planner_host(eligible, monkeypatch):
    from core.inference import diffusion_prequant as pq

    source = _flux3_source()
    monkeypatch.setattr(pq, "hosted_prequant_refusal", lambda *a, **k: None)
    monkeypatch.setattr(D, "denoiser_prequant_source", lambda *a, **k: source)
    monkeypatch.setattr(
        D,
        "snapshot_device_memory",
        lambda target: D.DeviceMemory(
            backend = "rocm", device = "cuda", memory_kind = "unified_memory", total_mib = 1000, free_mib = 10
        ),
    )
    return source


def test_planner_seeds_int8_for_a_resident_plan(planner_host):
    planner = _Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))
    assert _seed(planner) == tq.TQ_INT8
    assert planner.calls == [tq.TQ_INT8]


def test_planner_keeps_the_dense_shards_when_the_plan_offloads(planner_host):
    planner = _Planner(types.SimpleNamespace(offload_policy = OFFLOAD_GROUP, stream_transformer = True))
    assert _seed(planner) is None


def test_planner_never_seeds_a_local_override(planner_host, monkeypatch):
    local = types.SimpleNamespace(kind = "path", location = "/models/flux3-int8.safetensors")
    monkeypatch.setattr(D, "denoiser_prequant_source", lambda *a, **k: local)
    assert _seed(_Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))) is None


@pytest.mark.parametrize(
    "name",
    [
        "flux-3-image-klein-INT8.safetensors",
        "Z-Image-Turbo-INT8.pt",
        "x-INT8-ConvRot.pt",
        "x-INT8-ConvRot-ComfyUI.safetensors",
    ],
)
def test_reader_takes_only_a_convrot_safetensors_build(monkeypatch, tmp_path, name):
    from core.inference import diffusion_prequant as pq

    path = tmp_path / name
    path.write_bytes(b"")
    monkeypatch.setattr(pq, "_resolve_checkpoint_path", lambda *a, **k: str(path))
    called = []
    monkeypatch.setattr(NH, "load_native_from_hosted", lambda *a, **k: called.append(1))
    assert RH.load_rocm_hosted_transformer(_fam(), object, FLUX3[0], _flux3_source()) is None
    assert called == []


def test_reader_opts_into_the_keep_fp32_rule(monkeypatch, tmp_path):
    from core.inference import diffusion_prequant as pq

    path = tmp_path / "flux-3-image-klein-INT8-ConvRot.safetensors"
    path.write_bytes(b"")
    monkeypatch.setattr(pq, "_resolve_checkpoint_path", lambda *a, **k: str(path))
    seen = {}
    module = types.SimpleNamespace()

    def reader(*a, **k):
        seen.update(k)
        return module

    monkeypatch.setattr(NH, "load_native_from_hosted", reader)
    assert RH.load_rocm_hosted_transformer(_fam(), object, FLUX3[0], _flux3_source()) is module
    assert seen["keep_fp32"] is True and seen["spec"]["act_int8"] is True
    assert seen["tag"] == "diffusion.rocm_hosted_int8"
    assert module._unsloth_prequant_path == str(path)


def test_planner_keeps_the_dense_shards_without_a_hosted_source(planner_host, monkeypatch):
    monkeypatch.setattr(D, "denoiser_prequant_source", lambda *a, **k: None)
    assert _seed(_Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))) is None


@pytest.mark.parametrize("raw", [None, "fp8"])
def test_planner_never_seeds_auto_or_other_schemes(planner_host, raw):
    planner = _Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))
    assert _seed(planner, raw = raw) is None
    assert planner.calls == []


def test_planner_kill_switch(planner_host, monkeypatch):
    monkeypatch.setenv(RH.ROCM_IMAGE_HOSTED_INT8_ENV, "0")
    planner = _Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))
    assert _seed(planner) is None
    assert planner.calls == []


def _seed_kwargs(
    monkeypatch,
    *,
    dense_supported: bool,
    native_result = "native-module",
):
    """Drive ``denoiser_prequant_pipe_kwargs`` with both loaders stubbed; returns (result, which loader ran)."""
    ran = []
    source = types.SimpleNamespace(kind = "repo", location = "unsloth/flux-3-image-klein-FP8")
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: source)
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: dense_supported)
    from core.inference import diffusion_prequant as pq

    def torchao(*a, **k):
        ran.append("torchao")
        return "torchao-module"

    def native(*a, **k):
        ran.append("native")
        return native_result

    monkeypatch.setattr(pq, "load_prequantized_transformer", torchao)
    monkeypatch.setattr(RH, "load_rocm_hosted_transformer", native)
    import diffusers

    monkeypatch.setattr(diffusers, "Flux3Transformer2DModel", object, raising = False)
    out = DP.denoiser_prequant_pipe_kwargs(
        _fam(),
        FLUX3[0],
        scheme = "int8",
        dtype = torch.bfloat16,
        device = "cuda",
        target = GPU_BF16,
    )
    return out, ran


def test_rocm_seed_reads_the_hosted_file_as_native_linears(rocm, monkeypatch):
    out, ran = _seed_kwargs(monkeypatch, dense_supported = False)
    assert out == {"transformer": "native-module"} and ran == ["native"]


def test_rocm_seed_kill_switch_seeds_nothing(rocm, monkeypatch):
    monkeypatch.setenv(RH.ROCM_IMAGE_HOSTED_INT8_ENV, "0")
    out, ran = _seed_kwargs(monkeypatch, dense_supported = False)
    assert out == {} and ran == []


def test_rocm_seed_failure_restores_the_dense_route(rocm, monkeypatch):
    # {} obliges the loader to re-plan at bf16 and restore the dropped shards.
    out, ran = _seed_kwargs(monkeypatch, dense_supported = False, native_result = None)
    assert out == {} and ran == ["native"]


def test_cuda_seed_keeps_the_torchao_loader(nvidia, monkeypatch):
    out, ran = _seed_kwargs(monkeypatch, dense_supported = True)
    assert out == {"transformer": "torchao-module"} and ran == ["torchao"]


def test_hosted_reader_logs_under_the_callers_tag(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "x.safetensors"
    save_file({"w": torch.zeros(2)}, str(path))
    seen = []

    class _Log:
        def info(self, fmt, *args):
            seen.append(fmt % args)

        warning = info

    assert (
        NH.load_native_from_hosted(
            str(path),
            object,
            "unused",
            spec = {"scheme": "int8", "act_int8": True},
            family = "flux.3-klein",
            logger = _Log(),
            tag = "diffusion.rocm_hosted_int8",
        )
        is None
    )
    assert seen == [
        "diffusion.rocm_hosted_int8: hosted int8 checkpoint not used (no unsloth metadata)"
    ]
    seen.clear()
    NH.load_native_from_hosted(
        str(path), object, "unused", spec = {"scheme": "int8"}, family = None, logger = _Log()
    )
    assert seen and seen[0].startswith("video.native_hosted: ")


def test_planner_and_loader_share_one_convrot_candidate_list(planner_host, monkeypatch, tmp_path):
    # A mirror holding only the plain INT8 file: the plan probes the ConvRot file, so the load must resolve it too.
    from core.inference import diffusion_prequant as pq

    fam = _fam(QWEN21)
    repo = "unsloth/Qwen-Image-2.1-FP8"
    mirror = tmp_path / "mirror"
    (mirror / repo).mkdir(parents = True)
    (mirror / repo / "Qwen-Image-2.1-INT8.safetensors").write_bytes(b"")
    monkeypatch.setenv(pq.PREQUANT_MIRROR_ENV, str(mirror))
    source = pq.resolve_prequant_source(fam, "int8", base_repo = QWEN21[0])
    assert source is not None and "Qwen-Image-2.1-INT8.safetensors" in source.candidate_filenames
    probed = []
    monkeypatch.setattr(
        pq, "hosted_prequant_refusal", lambda src, *a, **k: probed.append(src.candidate_filenames)
    )
    monkeypatch.setattr(D, "denoiser_prequant_source", lambda *a, **k: source)
    planner = _Planner(types.SimpleNamespace(offload_policy = OFFLOAD_NONE))
    kw = dict(base = QWEN21[0], repo_id = QWEN21[0], fetch_base = QWEN21[0])
    assert (
        D.DiffusionBackend._rocm_hosted_int8_seed(
            planner,
            fam,
            GPU_BF16,
            raw = "int8",
            kind = "pipeline",
            speed_mode = None,
            memory_mode = None,
            cpu_offload = False,
            transformer_prequant_path = None,
            local_files_only = False,
            text_encoder_quant = None,
            **kw,
        )
        == tq.TQ_INT8
    )
    assert probed == [("Qwen-Image-2.1-INT8-ConvRot.safetensors",)]
    fetched = []
    staged = tmp_path / "Qwen-Image-2.1-INT8-ConvRot.safetensors"

    def download(src, name, *a, **k):
        fetched.append(name)
        return str(staged)

    monkeypatch.setattr(pq, "_download_checkpoint_name", download)
    read = []
    monkeypatch.setattr(
        NH, "load_native_from_hosted", lambda path, *a, **k: read.append(path) or object()
    )
    assert RH.load_rocm_hosted_transformer(fam, object, QWEN21[0], source) is not None
    assert fetched == list(probed[0]) and read == [str(staged)]
