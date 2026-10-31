# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AUTO resident runtime-quantized video DiT (HunyuanVideo-1.5 on a 22-24 GB card) and the local prequant cache.

Before: an auto load quantised the dense shards on every load (no hosted int8 for the family) and never wrote or
read the cache. Now: the resident torchao build is copied to host RAM in the load, written after publish, and a
returning auto load whose own pick is that scheme seeds it instead of reading + quantising the dense shards.

Hermetic, CPU only: a tiny random HunyuanVideo15Transformer3DModel saved as a local base dir, quantised through
``quantize_transformer``'s torchao branch, a private cache root per test.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("safetensors")
diffusers = pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import prequant_safetensors as PS  # noqa: E402
from core.inference import video as vid  # noqa: E402

FAMILY = "hunyuanvideo-1.5"
CLS = "HunyuanVideo15Transformer3DModel"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    for name in (C.ENV_MODE, C.ENV_MAX_GB, C.ENV_ASYNC, vid.VIDEO_RESIDENT_QUANT_CACHE_ENV):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.delenv("UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH", raising = False)
    C._HEADER_MEMO.clear()
    getattr(C, "_UNREACHABLE", {}).clear()
    # Local base dirs only: never ask the Hub whether a snapshot moved.
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    yield root
    C.cancel_background_writes(timeout = 10.0)


def _fam(base: Path, **over) -> types.SimpleNamespace:
    kw = dict(
        name = FAMILY,
        transformer_class = CLS,
        base_repo = str(base),
        is_moe = False,
        modular_workflow = None,
        prequant_repos = (),
    )
    kw.update(over)
    return types.SimpleNamespace(**kw)


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    model = diffusers.HunyuanVideo15Transformer3DModel(
        in_channels = 9,
        out_channels = 4,
        num_attention_heads = 4,
        attention_head_dim = 128,  # 512-wide linears: over the 512 min_features floor, so they quantise
        num_layers = 1,
        num_refiner_layers = 1,
        mlp_ratio = 2.0,
        text_embed_dim = 64,
        text_embed_2_dim = 32,
        image_embed_dim = 32,
        rope_axes_dim = (16, 56, 56),
        target_size = 64,
    )
    return model.to(torch.bfloat16).eval()


def _write_base(tmp_path: Path, seed: int = 0, name: str = "base"):
    model = _tiny_model(seed)
    base = tmp_path / name
    model.save_pretrained(str(base / "transformer"), safe_serialization = True)
    return base, model


def _flattenable(scheme: str) -> bool:
    try:
        from torchao.quantization import quantize_

        from core.inference.diffusion_transformer_quant import _make_quant_config

        m = torch.nn.Sequential(torch.nn.Linear(512, 512)).to(torch.bfloat16)
        quantize_(m, _make_quant_config(scheme))
        helpers = PS._torchao_helpers()
        if helpers is None:
            return False
        helpers[0](m.state_dict())
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope = "module")
def scheme() -> str:
    """int8 (the resident auto pick) where this torchao writes it (>= 0.18, Int8Tensor); else fp8; else skip."""
    for candidate in ("int8", "fp8"):
        if _flattenable(candidate):
            return candidate
    pytest.skip("this torchao cannot write a quantized build as safetensors on the CPU")


def _quantize(model, scheme: str, monkeypatch):
    """``quantize_transformer``'s own torchao branch (what a resident CUDA load runs), with the device gate opened."""
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    monkeypatch.setattr(tq, "native_quant_scheme", lambda *a, **k: None)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = model)
    assert tq.quantize_transformer(pipe, target, mode = "auto", family = FAMILY) == scheme
    assert tq.transformer_is_quantised(model)
    return model


def _flat_equal(a: dict, b: dict) -> list:
    fa, _ = C._flatten(a)
    fb, _ = C._flatten(b)
    assert set(fa) == set(fb)
    bad = []
    for k in fa:
        x, y = fa[k], fb[k]
        x = x.tensor() if isinstance(x, C._FlatRef) else x
        y = y.tensor() if isinstance(y, C._FlatRef) else y
        if x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            bad.append(k)
            continue
        if not torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8), y.contiguous().reshape(-1).view(torch.uint8)
        ):
            bad.append(k)
    return bad


def _views(model):
    return [types.SimpleNamespace(transformer = model)]


def _bf16_target():
    return types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


# ------------------------------------------------------------------------------------------------ round trip
def test_the_deferred_resident_write_reads_back_bit_identically(env, tmp_path, scheme, monkeypatch):
    from core.inference.diffusion_prequant import PrequantSource, load_prequantized_transformer

    base, model = _write_base(tmp_path)
    fam = _fam(base)
    fresh = _quantize(model, scheme, monkeypatch)

    pending = vid._save_runtime_quant_to_cache(
        fam, _views(fresh), base = str(base), source = str(base), defer = True
    )
    assert hasattr(pending, "start"), "the auto resident write must be deferred to after publish"
    # Nothing is published before start(): a load that fails before publish leaves no entry.
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    assert pending.start() and pending.wait(120)
    path = C.lookup(fam, scheme, base_repo = str(base))
    assert path is not None and Path(path).is_file()
    # Keyed exactly like the explicit save (same source fingerprint, scheme, filters, toolchain).
    key, _parts = C._key_for(fam, scheme, str(base), "transformer", base_id = str(base))
    assert Path(path).parent.name == key

    loaded = load_prequantized_transformer(
        diffusers.HunyuanVideo15Transformer3DModel,
        str(base),
        PrequantSource(kind = "path", location = path, filename = None),
        device = "cpu",
        dtype = torch.bfloat16,
        scheme = scheme,
        min_features = 512,
        component = "transformer",
        local_files_only = True,
    )
    assert loaded is not None
    assert _flat_equal(fresh.state_dict(), loaded.state_dict()) == []
    assert {n for n, _ in fresh.named_buffers()} == {n for n, _ in loaded.named_buffers()}
    # And the denoiser computes the same bits as the build it replaces.
    g = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states = torch.randn(1, 9, 1, 4, 4, generator = g).to(torch.bfloat16),
        timestep = torch.tensor([500.0]).to(torch.bfloat16),
        encoder_hidden_states = torch.randn(1, 8, 64, generator = g).to(torch.bfloat16),
        encoder_attention_mask = torch.ones(1, 8, dtype = torch.bool),
        encoder_hidden_states_2 = torch.randn(1, 4, 32, generator = g).to(torch.bfloat16),
        encoder_attention_mask_2 = torch.ones(1, 4, dtype = torch.bool),
        image_embeds = torch.zeros(1, 4, 32).to(torch.bfloat16),
        return_dict = False,
    )
    with torch.no_grad():
        want = fresh(**inputs)[0]
        got = loaded(**inputs)[0]
    assert torch.equal(want.contiguous().view(torch.uint8), got.contiguous().view(torch.uint8))


def test_a_returning_auto_load_seeds_exactly_that_build(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    fam = _fam(base)
    monkeypatch.setattr(vid, "select_transformer_quant_scheme", lambda *a, **k: scheme)
    kw = dict(target = _bf16_target(), requested = "auto", base_repo = str(base), speed_mode = None)
    # First load: no entry, nothing to seed (and auto's pick is never asked for).
    assert vid._video_auto_denoiser_scheme(fam, **kw) is None

    _quantize(model, scheme, monkeypatch)
    pending = vid._save_runtime_quant_to_cache(fam, _views(model), base = str(base), source = str(base), defer = True)
    assert pending.start() and pending.wait(120)
    assert vid._video_auto_denoiser_scheme(fam, **kw) == scheme

    from core.inference.video_denoiser_prequant import denoiser_prequant_sources

    sources = denoiser_prequant_sources(fam, scheme, str(base))
    assert sources and all(vid._is_prequant_cache_source(s) for s in sources.values())


# ------------------------------------------------------------------------------------------------ decision
def _stub_decision(monkeypatch, *, cached = ("int8",), pick = "int8"):
    import core.inference.diffusion_prequant as pq
    import core.inference.video_denoiser_prequant as vdp

    calls = {"select": 0}

    def _sources(fam, scheme, base_repo, *, hf_token = None):
        if scheme not in cached:
            return None
        return {"transformer": types.SimpleNamespace(kind = "path", location = f"/c/{scheme}/model.safetensors")}

    def _select(*a, **k):
        calls["select"] += 1
        return pick

    monkeypatch.setattr(vdp, "denoiser_prequant_sources", _sources)
    monkeypatch.setattr(vid, "_is_prequant_cache_source", lambda src: str(src.location).startswith("/c/"))
    monkeypatch.setattr(vid, "select_transformer_quant_scheme", _select)
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme = None, *a: True)
    return calls


def _decide(fam = None, **over):
    kw = dict(target = _bf16_target(), requested = "auto", base_repo = "org/base", speed_mode = None)
    kw.update(over)
    return vid._video_cached_resident_scheme(fam or _fam(Path("/nonexistent")), **kw)


def test_auto_replays_only_its_own_pick(env, monkeypatch):
    _stub_decision(monkeypatch)
    assert _decide() == "int8"
    # An explicit int8 entry must not turn auto's fp8 pick into int8.
    _stub_decision(monkeypatch, cached = ("int8",), pick = "fp8")
    assert _decide() is None
    _stub_decision(monkeypatch, cached = ("int8",), pick = None)
    assert _decide() is None


def test_no_entry_asks_nothing_and_changes_nothing(env, monkeypatch):
    calls = _stub_decision(monkeypatch, cached = ())
    assert _decide() is None
    assert calls["select"] == 0


def test_explicit_unset_speed_off_modular_moe_and_non_bf16_never_replay(env, monkeypatch):
    _stub_decision(monkeypatch)
    assert _decide(requested = "int8") is None  # explicit schemes keep their own seed path
    assert _decide(requested = None) is None  # unset never quantises at runtime
    assert _decide(requested = "off") is None
    assert _decide(speed_mode = "off") is None
    assert _decide(speed_mode = " OFF ") is None
    assert _decide(_fam(Path("/x"), modular_workflow = "fl2va")) is None
    assert _decide(_fam(Path("/x"), is_moe = True)) is None
    assert _decide(target = types.SimpleNamespace(device = "cuda", dtype = torch.float16)) is None
    assert _decide() == "int8"


def test_a_hosted_or_operator_source_is_not_a_cache_replay(env, monkeypatch):
    import core.inference.video_denoiser_prequant as vdp

    _stub_decision(monkeypatch)
    monkeypatch.setattr(
        vdp,
        "denoiser_prequant_sources",
        lambda fam, scheme, base, hf_token = None: {
            "transformer": types.SimpleNamespace(kind = "repo", location = "org/hosted")
        },
    )
    assert _decide() is None


def test_an_install_that_cannot_open_the_entry_keeps_the_dense_build(env, monkeypatch):
    import core.inference.diffusion_prequant as pq

    _stub_decision(monkeypatch)
    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda scheme = None, *a: False)
    assert _decide() is None


def test_kill_switches(env, monkeypatch):
    _stub_decision(monkeypatch)
    assert vid._resident_quant_cache_enabled()
    monkeypatch.setenv(vid.VIDEO_RESIDENT_QUANT_CACHE_ENV, "0")
    assert not vid._resident_quant_cache_enabled()
    assert _decide() is None
    monkeypatch.delenv(vid.VIDEO_RESIDENT_QUANT_CACHE_ENV)
    monkeypatch.setenv(C.ENV_MODE, "off")
    assert not vid._resident_quant_cache_enabled()
    assert _decide() is None


def test_the_hosted_seed_wins_and_the_cache_is_only_a_fallback(env, monkeypatch):
    seen = []
    monkeypatch.setattr(vid, "_video_hosted_denoiser_scheme", lambda fam, **kw: "nvfp4")
    monkeypatch.setattr(vid, "_video_cached_resident_scheme", lambda fam, **kw: seen.append(1) or "int8")
    kw = dict(target = None, requested = "auto", base_repo = "b", speed_mode = None)
    assert vid._video_auto_denoiser_scheme(None, **kw) == "nvfp4" and seen == []
    monkeypatch.setattr(vid, "_video_hosted_denoiser_scheme", lambda fam, **kw: None)
    assert vid._video_auto_denoiser_scheme(None, **kw) == "int8" and seen == [1]


# ------------------------------------------------------------------------------------------------ host snapshot
class _FakeDeviceTensor:
    """Reports a CUDA device; ``.to('cpu')`` hands back the real host tensor (a CPU-only host has no device)."""

    def __init__(self, t):
        self._t = t
        self.device = types.SimpleNamespace(type = "cuda")
        self.dtype = t.dtype
        self.shape = t.shape

    def numel(self):
        return self._t.numel()

    def element_size(self):
        return self._t.element_size()

    def detach(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self._t.clone()


def _job(tmp_path, flat):
    return C._WriteJob("k" * 32, tmp_path / "k", flat, {}, {}, sum(C._tensor_bytes(t) for t in flat.values()), 0.0, None)


def test_the_snapshot_moves_every_device_tensor_to_the_host(env, tmp_path, monkeypatch):
    a, b = torch.randn(8, 8).to(torch.bfloat16), torch.randint(-128, 127, (16,), dtype = torch.int8)
    job = _job(tmp_path, {"a": _FakeDeviceTensor(a), "b": b})
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (64 << 30, 32 << 30))
    assert C._defer_blocker(job, host_snapshot = True) is None
    assert all(getattr(t, "device", None).type == "cpu" for t in job.tensors())
    assert torch.equal(job.flat["a"], a) and torch.equal(job.flat["b"], b)
    # Without host_snapshot a device build is never deferred (the image path's rule, unchanged).
    job2 = _job(tmp_path, {"a": _FakeDeviceTensor(a)})
    assert "not the host" in C._defer_blocker(job2)


def test_a_snapshot_that_does_not_fit_host_ram_is_refused(env, tmp_path, monkeypatch):
    a = torch.zeros(1024, 1024, dtype = torch.bfloat16)
    job = _job(tmp_path, {"a": _FakeDeviceTensor(a)})
    monkeypatch.setattr(C, "_host_memory_bytes", lambda: (8 << 30, 1 << 30))
    why = C._defer_blocker(job, host_snapshot = True)
    assert why is not None and "host RAM short" in why
    assert isinstance(job.flat["a"], _FakeDeviceTensor)  # nothing copied


def test_a_resident_write_that_cannot_defer_is_skipped_not_written_in_the_load(
    env, tmp_path, scheme, monkeypatch
):
    base, model = _write_base(tmp_path)
    fam = _fam(base)
    _quantize(model, scheme, monkeypatch)
    monkeypatch.setattr(C, "_defer_blocker", lambda job, **kw: "another background cache write is running")
    assert vid._save_runtime_quant_to_cache(fam, _views(model), base = str(base), source = str(base), defer = True) is None
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    # ASYNC=0 is the operator asking for the synchronous write: honoured.
    monkeypatch.setenv(C.ENV_ASYNC, "0")
    monkeypatch.setattr(C, "_defer_blocker", lambda job, **kw: f"{C.ENV_ASYNC}=0")
    out = vid._save_runtime_quant_to_cache(fam, _views(model), base = str(base), source = str(base), defer = True)
    assert isinstance(out, str) and C.lookup(fam, scheme, base_repo = str(base)) == out


def test_a_discarded_write_releases_its_slot_and_leaves_no_entry(env, tmp_path, scheme, monkeypatch):
    base, model = _write_base(tmp_path)
    fam = _fam(base)
    _quantize(model, scheme, monkeypatch)
    pending = vid._save_runtime_quant_to_cache(fam, _views(model), base = str(base), source = str(base), defer = True)
    owner = types.SimpleNamespace(_resident_quant_write = (7, pending))
    vid._discard_resident_quant_write(owner, 8)  # another load's token: untouched
    assert owner._resident_quant_write is not None and not pending.done()
    vid._discard_resident_quant_write(owner, 7)
    assert owner._resident_quant_write is None and pending.done()
    assert C.lookup(fam, scheme, base_repo = str(base)) is None
    assert not any(not p.done() for p in C._BACKGROUND)


def test_publish_starts_only_this_loads_write_and_teardown_finishes_it(env):
    events = []

    class _W:
        def start(self):
            events.append("start")
            return True

        def wait(self, timeout = None):
            events.append("wait")
            return True

        def cancel(self):
            events.append("cancel")

        def discard(self):
            events.append("discard")

    w = _W()
    owner = types.SimpleNamespace(_resident_quant_write = (1, w))
    pipe = types.SimpleNamespace()
    vid._start_resident_quant_write(owner, pipe, 2)
    assert events == [] and owner._resident_quant_write is not None
    vid._start_resident_quant_write(owner, pipe, 1)
    assert events == ["start"] and owner._resident_quant_write is None
    assert getattr(pipe, vid._RESIDENT_QUANT_WRITE_ATTR) is w
    vid._finish_resident_quant_write(pipe)
    assert events == ["start", "wait"] and getattr(pipe, vid._RESIDENT_QUANT_WRITE_ATTR) is None


# ------------------------------------------------------------------------------------------------ wiring
def test_the_load_writes_auto_resident_builds_deferred_and_keeps_explicit_writes_inline():
    src = inspect.getsource(vid.VideoBackend.load_pipeline)
    explicit_at = src.index("_save_runtime_quant_to_cache(")
    assert "elif transformer_quant_pinned is not None:" in src[explicit_at - 200 : explicit_at]
    assert "defer = True" not in src[explicit_at : explicit_at + 400]
    auto_at = src.index("_save_runtime_quant_to_cache(", explicit_at + 1)
    guard = src[auto_at - 700 : auto_at]
    for needle in (
        "not video_offload",
        "native_scheme is None",
        "normalize_transformer_quant(transformer_quant) == TQ_AUTO",
        "_resident_quant_cache_enabled()",
    ):
        assert needle in guard, needle
    assert "defer = True" in src[auto_at : auto_at + 400]
    assert "source = _base_local_dir or fetch_base," in src[auto_at : auto_at + 400]
    assert "_start_resident_quant_write(self, pipe, _load_token)" in src
    assert src.index("_start_resident_quant_write(") > src.index('"video.loaded: %s (%s, %s, offload=%s')


def test_a_failed_load_discards_and_an_unload_finishes_the_write():
    backend_src = inspect.getsource(vid.VideoBackend)
    assert "_discard_resident_quant_write(self, token)" in backend_src
    assert '_finish_resident_quant_write(getattr(state, "pipe", None))' in backend_src


def test_a_cache_seed_says_so_in_the_log_and_the_resolved_source():
    src = inspect.getsource(vid.VideoBackend.load_pipeline)
    assert "read the resident %s denoiser for %s from the local prequant" in src
    assert "read back from this machine's prequant cache" in src
