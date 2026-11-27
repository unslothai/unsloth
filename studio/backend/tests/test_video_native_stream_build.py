# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fresh offloaded two-expert Wan load whose dense experts would not fit host RAM builds the native int8 per shard
into the prequant cache: same bytes as the in-place runtime build, never a dense expert in RAM."""

from __future__ import annotations

import gc
import sys
import types
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("safetensors")
pytest.importorskip("accelerate")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_native_cache as NC  # noqa: E402
from core.inference import diffusion_prequant_cache as C  # noqa: E402
from core.inference import diffusion_small_host as SH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference import video_native_stream as VS  # noqa: E402
from core.inference.diffusion_native_quant import is_native_linear, quantise_host_linears_on  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

FAMILY = "wan2.2-t2v-a14b"
BASE = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
COMPONENTS = ("transformer", "transformer_2")
CLS = diffusers.WanTransformer3DModel
CUDA = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)


def _family():
    return detect_video_family(BASE, override = FAMILY)


def _tiny_config() -> dict:
    # 2 x 256 = 512 wide: every attention / FFN Linear passes the 512 filter and the g256 ConvRot divides it.
    return dict(
        patch_size = (1, 2, 2),
        num_attention_heads = 2,
        attention_head_dim = 256,
        in_channels = 4,
        out_channels = 4,
        text_dim = 64,
        freq_dim = 32,
        ffn_dim = 1024,
        num_layers = 2,
        cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads",
        eps = 1e-6,
        rope_max_seq_len = 64,
    )


def _write_base(root: Path, dtype, sharded: bool) -> str:
    """A two-expert Wan base like the real A14B repo: fp32 experts on disk (``dtype``), plus a model_index.json."""
    kwargs = {"max_shard_size": "2MB"} if sharded else {}
    for seed, component in enumerate(COMPONENTS):
        torch.manual_seed(seed)
        model = CLS(**_tiny_config()).to(dtype)
        model.save_pretrained(str(root / component), safe_serialization = True, **kwargs)
    (root / "model_index.json").write_text(
        '{"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"], '
        '"transformer_2": ["diffusers", "WanTransformer3DModel"], "scheduler": ["diffusers", "X"], '
        '"tokenizer": ["transformers", "T"], "text_encoder": [null, null]}'
    )
    return str(root)


@pytest.fixture(params = ["fp32-single", "fp32-sharded", "bf16-single"])
def base_dir(tmp_path, request):
    dtype = torch.bfloat16 if request.param.startswith("bf16") else torch.float32
    return _write_base(
        tmp_path / "Wan2.2-T2V-A14B-Diffusers", dtype, request.param.endswith("sharded")
    )


@pytest.fixture
def fp32_dir(tmp_path):
    return _write_base(tmp_path / "Wan2.2-T2V-A14B-Diffusers", torch.float32, True)


@pytest.fixture(autouse = True)
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(C.ENV_DIR, str(root))
    for env in (C.ENV_MODE, C.ENV_ASYNC, NC.STREAM_BUILD_ENV, VS.VIDEO_NATIVE_STREAM_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(C, "_hub_source_stale", lambda shards, logger = None: False)
    return root


def _spec(act_int8 = True):
    return NC.native_spec("int8", FAMILY, act_int8 = act_int8)


def _runtime_build(
    base_dir,
    component,
    act_int8 = True,
    device = None,
):
    """Today's offloaded load: dense bf16 from_pretrained, then the in-place native pass (host linears on ``device``)."""
    model = CLS.from_pretrained(base_dir, subfolder = component, torch_dtype = torch.bfloat16)
    view = types.SimpleNamespace(transformer = model)
    with quantise_host_linears_on(device):
        assert (
            tq._quantize_native(
                view,
                "int8",
                family = FAMILY,
                min_features = tq.DEFAULT_MIN_LINEAR_FEATURES,
                act_int8 = act_int8,
                logger = None,
            )
            == "int8"
        )
    return model


def _streamed(
    base_dir,
    component,
    act_int8 = True,
    device = None,
    **kwargs,
):
    with quantise_host_linears_on(device):
        return NC.build_from_shards(
            CLS,
            base_dir,
            _spec(act_int8),
            family = FAMILY,
            dtype = kwargs.pop("dtype", torch.bfloat16),
            component = component,
            cast_floats = kwargs.pop("cast_floats", True),
            **kwargs,
        )


def _assert_same_module(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k
    ba, bb = (
        dict(a.named_buffers()),
        dict(b.named_buffers()),
    )  # non-persistent too (rope tables, rot_h)
    assert ba.keys() == bb.keys()
    for k in ba:
        assert ba[k].dtype == bb[k].dtype and torch.equal(ba[k], bb[k]), k
    for (na, ma), (nb, mb) in zip(a.named_modules(), b.named_modules()):
        assert na == nb and type(ma).__name__ == type(mb).__name__, na
        assert ma.extra_repr() == mb.extra_repr(), na
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for k in pa:
        assert pa[k].dtype == pb[k].dtype and pa[k].requires_grad == pb[k].requires_grad, k


def _forward(model):
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(1, 4, 3, 8, 8, generator = g).to(torch.bfloat16)
    text = torch.randn(1, 6, 64, generator = g).to(torch.bfloat16)
    with torch.no_grad():
        return model(hidden, torch.tensor([500]), text, return_dict = False)[0]


# -- decision ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dense, available, reserve, forced, disabled, engaged",
    [
        (
            58_600,
            51_000,
            5_300,
            False,
            False,
            True,
        ),  # A14B experts alone on the 53 GB Colab L4 host
        (69_000, 29_000, 3_100, False, False, True),  # A14B pipeline on the 31 GB Kaggle T4 host
        (
            20_000,
            51_000,
            5_300,
            False,
            False,
            False,
        ),  # Wan2.2-TI2V-5B: the dense load fits, unchanged
        (46_000, 51_000, 5_300, False, False, True),  # fits only without the reserve: engage
        (46_000, 51_000, 5_300, False, True, False),  # kill switch
        (20_000, 51_000, 5_300, True, False, True),  # forced
        (None, 51_000, 5_300, False, False, False),  # unreadable size: the dense load, as before
        (58_600, None, 5_300, False, False, False),  # unreadable host: the dense load, as before
        (58_600, 51_000, 5_300, True, True, False),  # off wins over force
    ],
)
def test_decision(dense, available, reserve, forced, disabled, engaged):
    got, why = VS.decide(
        dense, available_mib = available, reserve_mib = reserve, forced = forced, disabled = disabled
    )
    assert got is engaged and why


def test_env_switches(monkeypatch):
    for value, off, on in (
        ("0", True, False),
        ("off", True, False),
        ("1", False, True),
        ("", False, False),
    ):
        monkeypatch.setenv(VS.VIDEO_NATIVE_STREAM_ENV, value)
        assert (VS.stream_disabled(), VS.stream_forced()) == (off, on)


def test_dense_size_prices_loaded_dtype_and_keep_in_fp32(fp32_dir, tmp_path):
    total, sizes = VS.dense_load_mib(fp32_dir, COMPONENTS, cache_dir = None, transformer_cls = CLS)
    assert set(sizes) == set(COMPONENTS)
    model = CLS.from_pretrained(fp32_dir, subfolder = "transformer", torch_dtype = torch.bfloat16)
    loaded = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    assert sizes["transformer"] == -(-loaded // (1024 * 1024))
    assert total == sum(sizes.values())
    # a companion with weights counts, one already injected does not; a missing expert is "unknown"
    te = Path(fp32_dir) / "text_encoder"
    te.mkdir()
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(1024, 1024)}, str(te / "model.safetensors"))
    idx = Path(fp32_dir) / "model_index.json"
    idx.write_text(
        idx.read_text().replace(
            '"text_encoder": [null, null]', '"text_encoder": ["transformers", "T"]'
        )
    )
    with_te, sizes = VS.dense_load_mib(fp32_dir, COMPONENTS, cache_dir = None, transformer_cls = CLS)
    assert sizes["text_encoder"] == 2 and with_te == total + 2
    assert (
        VS.dense_load_mib(
            fp32_dir, COMPONENTS, cache_dir = None, transformer_cls = CLS, skip = ("text_encoder",)
        )[0]
        == total
    )
    assert (
        VS.dense_load_mib(str(tmp_path / "nope"), COMPONENTS, cache_dir = None, transformer_cls = CLS)[
            0
        ]
        is None
    )


# -- bit identity ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("act_int8", [True, False])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_per_shard_build_is_bit_identical_to_the_runtime_build(base_dir, act_int8, device):
    for component in COMPONENTS:
        ref = _runtime_build(base_dir, component, act_int8, device)
        got = _streamed(base_dir, component, act_int8, device)
        assert got is not None
        assert got._unsloth_runtime_quant_info == {"scheme": "int8", "native": True}
        assert NC.build_spec(got, FAMILY) == _spec(act_int8)
        _assert_same_module(ref, got)
        # the deliberate fp32 linears (_keep_in_fp32_modules) stay dense fp32, as in the dense load
        kept = [n for n, m in got.named_modules() if isinstance(m, torch.nn.Linear)]
        assert kept and all(
            dict(got.named_modules())[n].weight.dtype == dict(ref.named_modules())[n].weight.dtype
            for n in kept
        )
        assert sum(is_native_linear(m) for m in got.modules()) > 0
        assert torch.equal(_forward(ref), _forward(got))


def test_cache_entries_are_byte_identical_to_the_runtime_entries(fp32_dir, tmp_path, monkeypatch):
    fam = _family()
    for component in COMPONENTS:
        entries = []
        for arm, build in (("ref", _runtime_build), ("streamed", _streamed)):
            monkeypatch.setenv(C.ENV_DIR, str(tmp_path / f"cache_{arm}"))
            out = NC.save(
                build(fp32_dir, component),
                fam,
                _spec(),
                base_repo = BASE,
                source_base = fp32_dir,
                component = component,
            )
            assert isinstance(out, str)
            entries.append(Path(out))
        assert entries[0].parent.name == entries[1].parent.name
        assert entries[0].read_bytes() == entries[1].read_bytes()


def test_default_mode_still_declines_fp32_and_keep_in_fp32(fp32_dir):
    # The image path's call (no cast_floats) keeps its old contract.
    assert _streamed(fp32_dir, "transformer", cast_floats = False) is None


def test_mover_builds_every_quantised_linear(fp32_dir, monkeypatch):
    from core.inference import diffusion_native_quant as NQ

    seen = []
    real = NQ._HostLinearMover.build

    def spy(self, linear, make):
        seen.append(str(self.device))
        return real(self, linear, make)

    monkeypatch.setattr(NQ._HostLinearMover, "build", spy)
    got = _streamed(fp32_dir, "transformer", device = "cpu")
    assert got is not None
    assert len(seen) == sum(is_native_linear(m) for m in got.modules()) and set(seen) == {"cpu"}


def test_oom_is_raised_not_swallowed(fp32_dir, monkeypatch):
    from core.inference import diffusion_native_quant as NQ

    real = NQ.native_linear_class()

    class Boom(real):
        def __init__(self, *a, **k):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    monkeypatch.setattr(NQ, "native_linear_class", lambda: Boom)
    with pytest.raises(torch.OutOfMemoryError):
        _streamed(fp32_dir, "transformer", reraise_oom = True)
    # the image path's call keeps returning None
    assert _streamed(fp32_dir, "transformer", cast_floats = False, reraise_oom = False) is None


# -- the load-time route -----------------------------------------------------------------------------------------


def _small_host(
    monkeypatch,
    available_mib = 1,
    total_mib = 53 * 1024,
):
    monkeypatch.setattr(SH, "host_ram_mib", lambda: (total_mib, available_mib))


def _route(base_dir, **kwargs):
    from core.inference import video as V
    return V._video_native_stream_build(
        _family(),
        kwargs.pop("spec", _spec()),
        target = kwargs.pop("target", CUDA),
        dtype = torch.bfloat16,
        base = BASE,
        source = base_dir,
        config_kwargs = {},
        **kwargs,
    )


def _refuse_dense(monkeypatch):
    """The dense expert load must never run on this route."""

    def refuse(*a, **k):
        raise AssertionError("dense from_pretrained of a denoiser")

    monkeypatch.setattr(CLS, "from_pretrained", refuse)


@pytest.fixture
def no_dense(monkeypatch):
    _refuse_dense(monkeypatch)


class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, msg, *args):
        self.lines.append(str(msg) % args if args else str(msg))

    info = warning = debug = _add


def test_small_host_builds_every_expert_into_the_cache_and_the_returning_load_reads_it(
    fp32_dir, monkeypatch
):
    from core.inference import video as V

    refs = {c: _runtime_build(fp32_dir, c, act_int8 = True) for c in COMPONENTS}
    _small_host(monkeypatch)
    _refuse_dense(monkeypatch)
    log = _Log()
    monkeypatch.setattr(V, "logger", log)
    got = _route(fp32_dir)
    assert tuple(got) == COMPONENTS
    for component, module in got.items():
        assert module._unsloth_runtime_quant_info["cache_path"]
        _assert_same_module(refs[component], module)
    text = "\n".join(log.lines)
    assert "video.native_stream: engaged=True" in text
    assert "building the native int8 transformer/transformer_2" in text
    assert (
        "2 native int8 denoiser(s) ready" in text
        and "(2 from the local prequant cache, 0 kept in memory)" in text
    )
    # The returning load: the existing read serves both experts.
    hits = V._read_native_quant_from_cache(
        _family(), _spec(), base = BASE, source = fp32_dir, config_kwargs = {}
    )
    assert set(hits) == set(COMPONENTS)


def test_one_expert_at_a_time(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch)
    real = NC.build_from_shards
    alive: list = []

    def build(*a, **k):
        gc.collect()
        assert all(ref() is None for ref in alive), "the previous expert's build is still alive"
        module = real(*a, **k)
        alive.append(weakref.ref(module))
        return module

    monkeypatch.setattr(NC, "build_from_shards", build)
    assert tuple(_route(fp32_dir)) == COMPONENTS
    assert len(alive) == 2


def test_fits_host_keeps_the_dense_load(fp32_dir, monkeypatch, cache_root):
    _small_host(monkeypatch, available_mib = 10**7)
    assert _route(fp32_dir) == {}
    assert not cache_root.exists() or not any(cache_root.iterdir())


def test_kill_switches(fp32_dir, monkeypatch, cache_root):
    _small_host(monkeypatch)
    for env in (VS.VIDEO_NATIVE_STREAM_ENV, NC.STREAM_BUILD_ENV):
        monkeypatch.setenv(env, "0")
        assert _route(fp32_dir) == {}
        monkeypatch.delenv(env)
    assert not cache_root.exists() or not any(cache_root.iterdir())


def test_forced_streams_on_a_big_host(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch, available_mib = 10**7)
    monkeypatch.setenv(VS.VIDEO_NATIVE_STREAM_ENV, "1")
    assert tuple(_route(fp32_dir)) == COMPONENTS


def test_not_cuda_or_no_spec_is_a_no_op(fp32_dir, monkeypatch):
    _small_host(monkeypatch)
    assert _route(fp32_dir, target = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)) == {}
    assert _route(fp32_dir, spec = None) == {}


def test_unwritable_cache_keeps_the_built_experts(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch)
    monkeypatch.setenv(C.ENV_MODE, "0")
    got = _route(fp32_dir)
    assert tuple(got) == COMPONENTS
    assert all("cache_path" not in m._unsloth_runtime_quant_info for m in got.values())


def test_a_cached_expert_from_an_interrupted_build_is_reused(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch)
    first = _streamed(fp32_dir, "transformer")
    assert NC.save(
        first, _family(), _spec(), base_repo = BASE, source_base = fp32_dir, component = "transformer"
    )
    calls = []
    real = NC.build_from_shards
    monkeypatch.setattr(
        NC, "build_from_shards", lambda *a, **k: calls.append(k["component"]) or real(*a, **k)
    )
    assert tuple(_route(fp32_dir)) == COMPONENTS
    assert calls == ["transformer_2"]


def test_a_failed_expert_build_keeps_the_dense_load(fp32_dir, monkeypatch):
    _small_host(monkeypatch)
    monkeypatch.setattr(NC, "build_from_shards", lambda *a, **k: None)
    assert _route(fp32_dir) == {}


def test_a_cancel_raises(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch)
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) > 5:
            raise RuntimeError("Video load was cancelled or superseded.")

    with pytest.raises(RuntimeError, match = "cancelled"):
        _route(fp32_dir, cancel_check = cancel)


def test_an_entry_that_does_not_read_back_is_rebuilt_in_memory(fp32_dir, monkeypatch, no_dense):
    _small_host(monkeypatch)
    monkeypatch.setattr(NC, "load", lambda *a, **k: None)
    got = _route(fp32_dir)
    assert tuple(got) == COMPONENTS
    assert all("cache_path" not in m._unsloth_runtime_quant_info for m in got.values())
