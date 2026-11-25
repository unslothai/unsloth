# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-card Wan VAE tiled decode on CPU: bitwise the stock tiled decode, gated off for anything it cannot mirror."""

from __future__ import annotations

import math
import types

import pytest
import torch

from core.inference import video_vae_multigpu as vm

wan = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")


def _vae(patch_size = None):
    kw = dict(
        base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    )
    if patch_size is not None:
        kw.update(
            patch_size = patch_size,
            is_residual = True,
            in_channels = 12,
            out_channels = 12,
            z_dim = 4,
            scale_factor_spatial = 16,
        )
    torch.manual_seed(0)
    vae = wan.AutoencoderKLWan(**kw).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.2)
    vae.enable_tiling(
        tile_sample_min_height = 32,
        tile_sample_min_width = 32,
        tile_sample_stride_height = 24,
        tile_sample_stride_width = 24,
    )
    return vae


def _latents(
    vae,
    frames = 3,
    h = 9,
    w = 11,
):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, vae.config.z_dim, frames, h, w, generator = g)


@pytest.mark.parametrize("patch_size", [None, 2])
def test_split_tiles_match_the_stock_tiled_decode_bitwise(patch_size):
    try:
        vae = _vae(patch_size)
    except TypeError:
        pytest.skip("this diffusers has no patch_size Wan VAE")
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
        from core.inference.diffusion_multigpu import build_replica

        replica = build_replica(vae, torch.device("cpu"))
        out = vm.split_tiled_decode(vae, z, torch.device("cpu"), replica = replica)
        built = vm.split_tiled_decode(vae, z, torch.device("cpu"))
    assert out.shape == ref.shape
    assert torch.equal(out, ref) and torch.equal(built, ref)


def _copy(vae):
    import copy
    return copy.deepcopy(vae)


def test_the_mirrored_body_is_the_installed_diffusers_one():
    # A diffusers that changes tiled_decode turns the split off (fingerprint mismatch) instead of drifting from it.
    assert vm.eligible(_vae()) is None
    assert vm.body_fingerprint(wan.AutoencoderKLWan.tiled_decode) in vm._KNOWN_BODIES


def test_a_changed_body_or_another_vae_is_refused():
    vae = _vae()

    class Other(torch.nn.Module):
        pass

    assert "not the Wan VAE" in vm.eligible(Other())

    def tiled_decode(
        self,
        z,
        return_dict = True,
    ):
        return None

    vae.__class__ = type(
        "AutoencoderKLWan", (wan.AutoencoderKLWan,), {"tiled_decode": tiled_decode}
    )
    assert "differs" in vm.eligible(vae)


def test_install_routes_tiled_decode_and_falls_back_off_cuda(monkeypatch):
    vae = _vae()
    pipe = types.SimpleNamespace(vae = vae)
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
    assert vm.install(pipe, 1)
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]
    assert torch.equal(out, ref)
    assert vm.stats(pipe)["one_card"] == 1 and vm.stats(pipe)["split"] == 0
    assert "CUDA" in vm.stats(pipe)["last"]
    monkeypatch.setenv(vm.ENV, "0")
    assert not vm.install(types.SimpleNamespace(vae = _vae()), 1)
    vm.uninstall(pipe)
    assert "tiled_decode" not in vae.__dict__ and vm.stats(pipe) is None


def test_hooks_stay_with_the_source_while_the_copy_is_built():
    vae = _vae()
    calls = []
    handle = vae.decoder.register_forward_hook(lambda m, a, o: calls.append(1))
    with vm._without_hooks(vae):
        assert not vae.decoder._forward_hooks
        replica = _copy(vae)
    assert vae.decoder._forward_hooks and not replica.decoder._forward_hooks
    handle.remove()


def test_partner_free_memory_counts_reusable_cache_and_the_need_scales_with_the_tile(monkeypatch):
    fake = types.SimpleNamespace(
        cuda = types.SimpleNamespace(mem_get_info = lambda o: (1 << 30, 16 << 30)), device = torch.device
    )
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: 3 << 30)
    assert vm._partner_free_mib(fake, 1) == 4096
    vae = _vae()
    vae.tile_sample_min_height = vae.tile_sample_min_width = 256
    base = vm.need_mib(vae)
    vae.tile_sample_min_height = vae.tile_sample_min_width = 512
    grown = vm.need_mib(vae) - base
    want = vm._NEED_MARGIN * vm._WORKING_MIB_PER_TILE_PX * (512 * 512 - 256 * 256)
    assert abs(grown - want) <= 1


def test_a_copy_holding_tensors_on_another_card_is_detected():
    m = torch.nn.Linear(2, 2)
    m.cache = torch.zeros(1)
    assert vm._stray_tensors(m, torch.device("cpu")) == []
    m.cache = types.SimpleNamespace()
    assert vm._stray_tensors(m, torch.device("cuda", 1)) == []


@pytest.mark.parametrize("failing", ["load", "partner"])
def test_a_failing_card_still_releases_the_copy(monkeypatch, failing):
    # A raising card must still free the partner copy before the stock retry.
    import gc
    import weakref

    from core.inference import diffusion_multigpu as dm

    vae = _vae()
    z = _latents(vae)
    built = []

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("card out of memory")

    def fake_build(src, device):
        copy = _vae()
        if failing == "partner":
            copy.post_quant_conv = Boom()
        built.append(weakref.ref(copy))
        return copy

    monkeypatch.setattr(dm, "build_replica", fake_build)
    if failing == "load":
        vae.post_quant_conv = Boom()
    with torch.no_grad(), pytest.raises(RuntimeError, match = "card out of memory") as info:
        vm.split_tiled_decode(vae, z, torch.device("cpu"))
    gc.collect()
    assert info.tb is not None
    assert built and built[0]() is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)
def test_split_tiles_keep_the_callers_autocast_on_two_cards():
    vae = _vae().to("cuda:0")
    z = _latents(vae).to("cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype = torch.float16):
        ref = vae.tiled_decode(z, return_dict = False)[0]
        out = vm.split_tiled_decode(vae, z, torch.device("cuda:1"))
    assert out.dtype == ref.dtype
    assert torch.equal(out, ref)


def test_partner_tiles_never_run_the_sources_patched_forwards():
    # Speed-path closures over source blocks must be rebuilt against the copy.
    import threading

    vae = _vae()
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
    ran = []

    def patch(block):
        stock = type(block).forward.__get__(block)

        def fast(
            *args,
            _block = block,
            **kwargs,
        ):
            ran.append((threading.current_thread().name, id(_block)))
            return stock(*args, **kwargs)

        def forward(*args, **kwargs):
            return fast(*args, **kwargs)

        block.forward = forward

    blocks = [m for m in vae.decoder.modules() if type(m).__name__.endswith("ResidualBlock")]
    assert blocks
    for block in blocks:
        patch(block)
    source = {id(b) for b in blocks}
    with torch.no_grad():
        out = vm.split_tiled_decode(vae, z, torch.device("cpu"))
    partner = [i for name, i in ran if name == "unsloth-vae-split"]
    assert partner and not (set(partner) & source)
    assert torch.equal(out, ref)


def test_a_copy_whose_forward_reaches_the_source_is_detected():
    from core.inference.diffusion_multigpu import build_replica

    vae = _vae()
    block = next(m for m in vae.decoder.modules() if type(m).__name__.endswith("ResidualBlock"))
    block.forward = lambda x, *a, _b = block, **k: type(_b).forward(_b, x, *a, **k)
    copy = build_replica(vae, torch.device("cpu"))
    assert any("captured" in s for s in vm._stray_tensors(copy, torch.device("cpu")))
    assert vm._rebind_forwards(vae, copy) >= 1
    assert vm._stray_tensors(copy, torch.device("cpu")) == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)
def test_split_tiles_with_the_fused_wan_forwards_on_two_cards(monkeypatch):
    import threading

    from core.inference import diffusion_vae_fused as fused

    vae = _vae().to("cuda:0")
    if not fused.install_wan_vae(vae, encoder = False):
        pytest.skip("fused Wan VAE path unavailable here")
    if vm.eligible(vae) is not None:
        pytest.skip(vm.eligible(vae))
    z = _latents(vae).to("cuda:0")
    threads = []
    count = fused._count
    monkeypatch.setattr(
        fused, "_count", lambda name: (threads.append(threading.current_thread().name), count(name))
    )
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
        threads.clear()
        out = vm.split_tiled_decode(vae, z, torch.device("cuda:1"))
    assert "unsloth-vae-split" in threads and any(t != "unsloth-vae-split" for t in threads)
    # the partner card can differ by an ulp (Kaggle 2x T4: 4.5e-7 at scale 0.85 in fp32)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 2e-6)


@pytest.mark.parametrize(
    "partner_probe, want",
    [
        (("T4", 15360, 15000, 90.0), None),  # computing for another process
        (("T4", 15360, 9000, 0.0), None),  # another process holds 6 GiB on it
        (("L4", 23034, 23000, 0.0), None),  # a different card model
        (("T4", 15360, 15000, 0.0), 1),  # idle, same model: lent
    ],
)
def test_automatic_placement_lends_only_an_idle_same_model_card(monkeypatch, partner_probe, want):
    from core.inference import diffusion_multigpu as dm

    name, total, free, util = partner_probe
    probes = {
        0: dm.DeviceProbe(0, "T4", 15360, 4000, None),
        1: dm.DeviceProbe(1, name, total, free, util),
    }
    monkeypatch.setattr(dm, "probe_devices", lambda ordinals: {o: probes[o] for o in ordinals})
    monkeypatch.setattr(dm, "install_compile_recorder", lambda: False)
    opened = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: opened.append(a) or (0, 0))
    automatic = dm.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    pipe = types.SimpleNamespace(vae = _vae())
    with dm.selection_scope(automatic):
        installed = vm.install(pipe, 1)
    assert installed is (want is not None)
    assert (vm.stats(pipe) or {}).get("partner") == want
    assert not opened
    explicit = types.SimpleNamespace(vae = _vae())
    with dm.selection_scope(dm.MultiGpuSelection(primary = 0, secondaries = (1,))):
        assert vm.install(explicit, 1) and vm.stats(explicit)["partner"] == 1
    monkeypatch.setenv(vm.ENV, "force")
    forced = types.SimpleNamespace(vae = _vae())
    with dm.selection_scope(automatic):
        assert vm.install(forced, 1) and vm.stats(forced)["partner"] == 1


def test_the_copy_is_built_before_the_source_decode_fills_its_caches(monkeypatch):
    from core.inference import diffusion_multigpu as dm

    vae = _vae()
    z = _latents(vae)
    seen = []
    real = dm.build_replica

    def build(src, device):
        import time

        time.sleep(0.5)
        seen.append(any(t is not None for t in (getattr(src, "_feat_map", None) or [])))
        return real(src, device)

    monkeypatch.setattr(dm, "build_replica", build)
    with torch.no_grad():
        vae.clear_cache()
        out = vm.split_tiled_decode(vae, z, torch.device("cpu"))
        ref = vae.tiled_decode(z, return_dict = False)[0]
    assert seen == [False]
    assert torch.equal(out, ref)


def test_a_partner_setup_failure_releases_the_caller(monkeypatch):
    import contextlib
    import threading

    vae = _vae()
    z = _latents(vae)
    real_null = contextlib.nullcontext

    class Ctx(types.SimpleNamespace):
        pass

    def nullcontext(*a, **k):
        if threading.current_thread().name == "unsloth-vae-split":
            raise RuntimeError("partner setup failed")
        return real_null(*a, **k)

    monkeypatch.setattr(
        vm,
        "contextlib",
        Ctx(nullcontext = nullcontext, ExitStack = contextlib.ExitStack, suppress = contextlib.suppress),
    )
    box = {}

    def call():
        try:
            with torch.no_grad():
                vm.split_tiled_decode(vae, z, torch.device("cpu"))
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target = call, daemon = True)
    t.start()
    t.join(timeout = 60)
    assert not t.is_alive(), "the caller hung waiting for the partner's copy"
    assert "partner setup failed" in str(box.get("error"))


def _self_referencing_decode(vae):
    inner = vae.decode

    def untiled_when_fits(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        finally:
            if vae.__dict__.get("decode") is not untiled_when_fits:
                vae.decode = untiled_when_fits

    vae.decode = untiled_when_fits
    return untiled_when_fits


def test_a_self_referencing_decode_wrapper_is_rebound_to_the_copy():
    from core.inference.diffusion_multigpu import build_replica

    vae = _vae()
    _self_referencing_decode(vae)
    copy = build_replica(vae, torch.device("cpu"))
    assert any("captured" in s for s in vm._stray_tensors(copy, torch.device("cpu")))
    assert vm._rebind_forwards(vae, copy) >= 1
    assert vm._stray_tensors(copy, torch.device("cpu")) == []
    fn = copy.__dict__["decode"]
    cells = {id(c.cell_contents) for c in fn.__closure__}
    assert id(fn) in cells and id(copy) in cells and id(vae) not in cells
    src = vae.__dict__["decode"]
    assert id(vae) in {id(c.cell_contents) for c in src.__closure__}


def test_mutually_recursive_forwards_are_rebound_together():
    vae = _vae()
    block = next(m for m in vae.decoder.modules() if type(m).__name__.endswith("ResidualBlock"))

    def make(b):
        def first(x, *a, **k):
            return second(x, *a, **k)

        def second(x, *a, **k):
            if x is None:
                return first(x)
            return type(b).forward(b, x, *a, **k)

        return first

    block.forward = make(block)
    from core.inference.diffusion_multigpu import build_replica

    copy = build_replica(vae, torch.device("cpu"))
    vm._rebind_forwards(vae, copy)
    assert vm._stray_tensors(copy, torch.device("cpu")) == []


def test_an_unusable_copy_never_starts_the_source_tiles(monkeypatch):
    # A refused copy must fail the split before the load card decodes any tile.
    vae = _vae()
    z = _latents(vae)
    import time

    def stray(module, device):
        time.sleep(0.5)
        return ["x.y (captured AutoencoderKLWan)"]

    monkeypatch.setattr(vm, "_stray_tensors", stray)
    calls = []
    real = type(vae.post_quant_conv).forward

    def counted(self, *a, **k):
        if self is vae.post_quant_conv:
            calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(type(vae.post_quant_conv), "forward", counted)
    with torch.no_grad(), pytest.raises(RuntimeError, match = "holds tensors on another card"):
        vm.split_tiled_decode(vae, z, torch.device("cpu"))
    assert calls == []


def _fake_cuda_split(monkeypatch, free_mib, total_mib, need):
    vae = types.SimpleNamespace(
        post_quant_conv = types.SimpleNamespace(parameters = lambda: []), named_modules = lambda: []
    )
    z = types.SimpleNamespace(device = torch.device("cuda", 0))
    monkeypatch.setattr(vm, "_partner_free_mib", lambda torch_, partner: free_mib)
    monkeypatch.setattr(vm, "need_mib", lambda v, z = None: need)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda d = None: (free_mib << 20, total_mib << 20)
    )
    return vae, z, {"partner": 1}


def test_the_decode_time_gate_keeps_the_partner_cards_reserve(monkeypatch):
    # 80% Budget on 24 GiB keeps 4915 MiB free: 6144 free leaves 1229 usable, short of a 4600 MiB copy.
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda base: int(base * 0.2))
    vae, z, state = _fake_cuda_split(monkeypatch, 6144, 24576, 4600)
    monkeypatch.setattr(vm, "split_tiled_decode", lambda *a, **k: pytest.fail("split admitted"))
    dec, why = vm._try_split(vae, z, state, None)
    assert dec is None and "4915 MiB kept free" in why
    monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda base: 0)
    vae, z, state = _fake_cuda_split(monkeypatch, 5000, 24576, 4600)
    dec, why = vm._try_split(vae, z, state, None)
    assert dec is None and "1024 MiB kept free" in why
    monkeypatch.setattr(vm, "split_tiled_decode", lambda *a, **k: "decoded")
    vae, z, state = _fake_cuda_split(monkeypatch, 5700, 24576, 4600)
    assert vm._try_split(vae, z, state, None) == ("decoded", "split")


def test_a_sticky_cuda_error_is_not_retried_on_one_card(monkeypatch):
    vae, z, state = _fake_cuda_split(monkeypatch, 20000, 24576, 4600)

    def sticky(*a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(vm, "split_tiled_decode", sticky)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        vm._try_split(vae, z, state, None)

    def oom(*a, **k):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 806.00 MiB")

    monkeypatch.setattr(vm, "split_tiled_decode", oom)
    dec, why = vm._try_split(vae, z, state, None)
    assert dec is None and "OutOfMemoryError" in why


def test_cleanup_runs_every_step_when_the_load_stream_wait_raises(monkeypatch):
    import gc

    calls = []
    lost = RuntimeError("CUDA error: an illegal memory access was encountered")

    class Stream:
        def __init__(
            self,
            name,
            error = None,
        ):
            self.name, self.error = name, error

        def synchronize(self):
            calls.append(self.name)
            if self.error is not None:
                raise self.error

    monkeypatch.setattr(gc, "collect", lambda *a: calls.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    import contextlib

    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    decoded, box = {(0, 0): object()}, {"copy": object()}
    out = vm._cleanup(torch, True, Stream("caller", lost), Stream("side"), None, decoded, box)
    assert out is lost and vm.sticky_cuda_error(out)
    assert decoded == {} and box == {}
    assert calls == ["side", "caller", "collect"]
    calls.clear()
    assert vm._cleanup(torch, True, Stream("caller"), Stream("side"), None, {}, {}) is None
    assert calls == ["side", "caller", "collect", "empty_cache"]


def test_a_lost_context_in_the_cleanup_replaces_an_ordinary_split_failure(monkeypatch):
    # Partner OOM then a sticky illegal address on the load stream: re-raised, no one-card retry.
    lost = RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(vm, "_cleanup", lambda *a, **k: lost)
    vae = _vae()
    z = _latents(vae)

    def build(src, device):
        raise torch.OutOfMemoryError("CUDA out of memory on the partner")

    from core.inference import diffusion_multigpu as dm

    monkeypatch.setattr(dm, "build_replica", build)
    with torch.no_grad(), pytest.raises(RuntimeError, match = "illegal memory access") as info:
        vm.split_tiled_decode(vae, z, torch.device("cpu"))
    assert isinstance(info.value.__context__, torch.OutOfMemoryError)
    monkeypatch.setattr(
        dm, "build_replica", lambda src, device: dm.__dict__["_copy_module"](src, device)
    )
    monkeypatch.setattr(vm, "_cleanup", lambda *a, **k: ValueError("cleanup"))
    with torch.no_grad(), pytest.raises(ValueError, match = "cleanup"):
        vm.split_tiled_decode(vae, z, torch.device("cpu"))


# Kaggle 2x T4 Wan 2.2 TI2V 5B 1280x704x49 partner peaks per tile, (reserved, allocated) MiB.
_WAN_WEIGHTS_MIB = 1629
_MEASURED_PEAKS = {(256, 256): (2656, 2422), (704, 496): (6370, 5307), (704, 704): (None, 6371)}


def _wan_sized(monkeypatch, h, w):
    vae = types.SimpleNamespace(tile_sample_min_height = h, tile_sample_min_width = w)
    monkeypatch.setattr(vm, "_module_mib", lambda module: _WAN_WEIGHTS_MIB)
    return vae


@pytest.mark.parametrize("tile", sorted(_MEASURED_PEAKS))
def test_the_split_estimate_covers_every_measured_partner_peak(monkeypatch, tile):
    reserved, allocated = _MEASURED_PEAKS[tile]
    need = vm.need_mib(_wan_sized(monkeypatch, *tile))
    assert need >= 1.15 * allocated
    if reserved is not None:
        assert need >= 1.15 * reserved


def test_a_12_gib_partner_admits_the_704x496_split_and_a_smaller_one_does_not(monkeypatch):
    # The 12 GiB-capped T4 partner read 11664 - 11696 MiB free at the decode (1024 MiB kept free).
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda base: 0)
    need = vm.need_mib(_wan_sized(monkeypatch, 704, 496))
    vae, z, state = _fake_cuda_split(monkeypatch, 11664, 12288, need)
    monkeypatch.setattr(vm, "split_tiled_decode", lambda *a, **k: "decoded")
    assert vm._try_split(vae, z, state, None) == ("decoded", "split")
    vae, z, state = _fake_cuda_split(monkeypatch, 7600, 8192, need)
    dec, why = vm._try_split(vae, z, state, None)
    assert dec is None and "kept free" in why
    vae, z, state = _fake_cuda_split(monkeypatch, 5185, 15360, need)
    assert vm._try_split(vae, z, state, None)[0] is None


def test_the_split_estimate_adds_the_partner_tiles_frames_beyond_the_fitted_clip(monkeypatch):
    import torch

    vae = _wan_sized(monkeypatch, 704, 496)
    vae.config = types.SimpleNamespace(scale_factor_temporal = 4)
    vae.spatial_compression_ratio = 16
    vae.tile_sample_stride_height, vae.tile_sample_stride_width = 576, 368
    vae.parameters = lambda: iter([torch.empty(1, dtype = torch.float16)])
    fitted = vm.need_mib(vae)
    # 1280x704 latents at 16x: 44 x 80; 49 frames = 13 latent frames is the fitted clip
    assert vm.need_mib(vae, torch.empty(1, 48, 13, 44, 80)) == fitted
    longer = vm.need_mib(vae, torch.empty(1, 48, 31, 44, 80))  # 121 frames
    partner_tiles = (len(range(0, 44, 36)) * len(range(0, 80, 23))) // 2
    extra = (121 - 49) * 3 * 704 * 496 * 2 * 2 * partner_tiles / 2**20
    assert longer == int(
        math.ceil(1.25 * (_WAN_WEIGHTS_MIB + 169 + 3714 / 283648 * 704 * 496 + extra))
    )
    assert vm.need_mib(vae, object()) == fitted


def test_partner_sized_tiles_fit_the_partner_and_leave_the_vae_as_it_was(monkeypatch):
    vae = _wan_sized(monkeypatch, 704, 704)
    vae.tile_sample_stride_height, vae.tile_sample_stride_width = 704, 640
    before = tuple(getattr(vae, a) for a in vm._TILE_ATTRS)
    usable = 5185 - 1024  # the uncapped partner beside a CFG copy
    tiles = vm.partner_sized_tiles(vae, None, usable)
    assert tuple(getattr(vae, a) for a in vm._TILE_ATTRS) == before
    h, sh, w, sw = tiles
    assert (h - sh, w - sw) == (64, 64) and min(h, w) >= 256
    sized = _wan_sized(monkeypatch, h, w)
    assert vm.need_mib(sized) <= usable
    bigger = _wan_sized(monkeypatch, h + 32, w + 32)
    assert vm.need_mib(bigger) > usable
    assert vm.partner_sized_tiles(vae, None, 1000) is None  # nothing at the 256 px floor fits


def test_partner_sized_split_matches_the_stock_decode_at_those_tiles(monkeypatch):
    vae = _vae()
    z = _latents(vae)
    tiles = (32, 16, 32, 16)
    with torch.no_grad():
        vae.clear_cache()
        with vm._tiles(vae, tiles):
            ref = vae.tiled_decode(z, return_dict = False)[0]
        before = tuple(getattr(vae, a) for a in vm._TILE_ATTRS)
        with vm._tiles(vae, tiles):
            out = vm.split_tiled_decode(vae, z, torch.device("cpu"))
    assert tuple(getattr(vae, a) for a in vm._TILE_ATTRS) == before
    assert torch.equal(out, ref)


def _automatic_wan_split(
    monkeypatch,
    free_mib,
    total_mib,
    automatic = True,
):
    # A 1280x704 Wan decode at the stock wide plan (704x704, stride 704x640) reaching _try_split without a GPU.
    import core.inference.diffusion_memory as dmem
    import core.inference.diffusion_multigpu as dmg

    monkeypatch.setattr(dmem, "_user_budget_reserve_mib", lambda base: 0)
    monkeypatch.setattr(
        dmg, "current_selection", lambda: types.SimpleNamespace(automatic = automatic)
    )
    monkeypatch.delenv(vm.PARTNER_TILES_ENV, raising = False)
    vae = types.SimpleNamespace(
        post_quant_conv = types.SimpleNamespace(parameters = lambda: []),
        named_modules = lambda: [],
        tile_sample_min_height = 704,
        tile_sample_stride_height = 704,
        tile_sample_min_width = 704,
        tile_sample_stride_width = 640,
    )
    monkeypatch.setattr(vm, "_module_mib", lambda module: _WAN_WEIGHTS_MIB)
    z = types.SimpleNamespace(device = torch.device("cuda", 0))
    monkeypatch.setattr(vm, "_partner_free_mib", lambda torch_, partner: free_mib)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda d = None: (free_mib << 20, total_mib << 20)
    )
    seen = []

    def split(v, z_, partner):
        seen.append(tuple(getattr(v, a) for a in vm._TILE_ATTRS))
        return "decoded"

    monkeypatch.setattr(vm, "split_tiled_decode", split)
    return vae, z, {"partner": 1, "automatic": automatic}, seen


_STOCK_WIDE = (704, 704, 704, 640)


def test_partner_sized_tiles_engage_under_automatic_when_the_stock_split_does_not_fit(monkeypatch):
    # The uncapped 2x T4 with a CFG copy on the partner: 5185 MiB free, the stock 704x704 split needs ~10360.
    vae, z, state, seen = _automatic_wan_split(monkeypatch, 5185, 15360)
    out, why = vm._try_split(vae, z, state, None)
    assert out == "decoded" and "partner-sized tiles" in why
    (tiles,) = seen
    assert tiles != _STOCK_WIDE and tiles[0] - tiles[1] == tiles[2] - tiles[3] == 64
    sized = _wan_sized(monkeypatch, tiles[0], tiles[2])
    assert vm.need_mib(sized) <= 5185 - 1024
    assert tuple(getattr(vae, a) for a in vm._TILE_ATTRS) == _STOCK_WIDE  # restored


def test_the_stock_split_keeps_its_tiles_when_it_fits(monkeypatch):
    # The 12 GiB-capped partner fits the stock 704x704 split: partner-sized tiles never engage (bit-identical).
    vae, z, state, seen = _automatic_wan_split(monkeypatch, 11692, 12288)
    assert vm._try_split(vae, z, state, None) == ("decoded", "split")
    assert seen == [_STOCK_WIDE]


def test_partner_sized_tiles_decline_when_even_the_floor_does_not_fit(monkeypatch):
    vae, z, state, seen = _automatic_wan_split(monkeypatch, 3000, 15360)
    out, why = vm._try_split(vae, z, state, None)
    assert out is None and "kept free" in why and seen == []


def test_partner_sized_tiles_opt_out_and_explicit_selection(monkeypatch):
    vae, z, state, seen = _automatic_wan_split(monkeypatch, 5185, 15360)
    monkeypatch.setenv(vm.PARTNER_TILES_ENV, "0")
    assert vm._try_split(vae, z, state, None)[0] is None and seen == []
    # an explicit card pick keeps the stock tiles unless forced
    vae, z, state, seen = _automatic_wan_split(monkeypatch, 5185, 15360, automatic = False)
    assert vm._try_split(vae, z, state, None)[0] is None and seen == []
    monkeypatch.setenv(vm.PARTNER_TILES_ENV, "1")
    assert vm._try_split(vae, z, state, None)[0] == "decoded" and len(seen) == 1


def test_the_load_time_selection_decides_partner_tiles_on_the_render_thread(monkeypatch):
    # The selection is a context variable of the load; the decode runs on the render thread where it is unset, so
    # install records whether the load was Automatic and the decode reads that record.
    import core.inference.diffusion_multigpu as dmg

    vae = _vae()
    pipe = types.SimpleNamespace(vae = vae)
    monkeypatch.setattr(vm, "eligible", lambda v: None)
    monkeypatch.setattr(vm, "_automatic_partner", lambda v, p: (int(p), "selected"))
    monkeypatch.setattr(dmg, "current_selection", lambda: types.SimpleNamespace(automatic = True))
    assert vm.install(pipe, 1)
    assert vae.__dict__[vm._ATTR]["automatic"] is True
    vm.uninstall(pipe)
    monkeypatch.setattr(dmg, "current_selection", lambda: types.SimpleNamespace(automatic = False))
    assert vm.install(pipe, 1)
    assert vae.__dict__[vm._ATTR]["automatic"] is False
    vm.uninstall(pipe)
    # at decode time the selection is gone: the recorded flag still engages the partner tiles
    w, z, state, seen = _automatic_wan_split(monkeypatch, 5185, 15360)
    monkeypatch.setattr(dmg, "current_selection", lambda: None)
    out, why = vm._try_split(w, z, state, None)
    assert out == "decoded" and "partner-sized tiles" in why


def _installed_wan_behind_a_real_load(monkeypatch):
    # The Studio sequence: the split is installed inside an Automatic load (the selection is a context variable of
    # the load scope), the load returns and resets it, and the render runs later on another thread. Only CUDA memory,
    # device discovery and the decode math are simulated.
    import threading

    import core.inference.diffusion_memory as dmem
    import core.inference.diffusion_multigpu as dmg

    attrs = vm._TILE_ATTRS
    stock_calls, split_calls = [], []

    class Vae:
        spatial_compression_ratio = 16
        config = types.SimpleNamespace(scale_factor_temporal = 4)
        post_quant_conv = types.SimpleNamespace(parameters = lambda: ())

        def parameters(self):
            return iter((torch.empty(0, dtype = torch.float16),))

        def named_modules(self):
            return ()

        def tiled_decode(
            self,
            latents,
            return_dict = True,
        ):
            stock_calls.append(tuple(getattr(self, a) for a in attrs))
            return ("stock",)

    vae = Vae()
    for attr, value in zip(attrs, _STOCK_WIDE):
        setattr(vae, attr, value)
    pipe = types.SimpleNamespace(vae = vae)
    z = types.SimpleNamespace(device = torch.device("cuda", 0), shape = (1, 48, 13, 44, 80))
    automatic = dmg.MultiGpuSelection(0, (1,), automatic = True)
    cards = {i: dmg.DeviceProbe(i, "Tesla T4", 15360, 15360, 0) for i in (0, 1)}
    monkeypatch.setenv(vm.ENV, "1")
    monkeypatch.setenv(dmg.MULTIGPU_ENV, "1")
    monkeypatch.delenv(vm.PARTNER_TILES_ENV, raising = False)
    monkeypatch.setattr(dmg, "automatic_selection", lambda *a, **k: automatic)
    monkeypatch.setattr(dmg, "install_compile_recorder", lambda *a, **k: None)
    monkeypatch.setattr(dmg, "probe_devices", lambda *a, **k: cards)
    monkeypatch.setattr(vm, "eligible", lambda v: None)
    monkeypatch.setattr(vm, "_module_mib", lambda m: _WAN_WEIGHTS_MIB)
    monkeypatch.setattr(vm, "_partner_free_mib", lambda t, p: 5185)
    monkeypatch.setattr(dmem, "_user_budget_reserve_mib", lambda base: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d = None: (5185 << 20, 15360 << 20))

    def split(model, latents, partner):
        split_calls.append(tuple(getattr(model, a) for a in attrs))
        return "split"

    monkeypatch.setattr(vm, "split_tiled_decode", split)

    @dmg.with_load_selection
    def load(_backend, **kwargs):
        assert dmg.current_selection() == automatic
        assert vm.install(pipe, 1)

    load(None, _gpu_ordinals = dmg.AUTOMATIC_ORDINALS, gpu_ordinal = 0)
    assert dmg.current_selection() is None  # the load scope reset it

    def render():
        out = []
        worker = threading.Thread(
            target = lambda: out.append(vae.tiled_decode(z, return_dict = False)[0])
        )
        worker.start()
        worker.join()
        return out[0]

    return vae, pipe, render, stock_calls, split_calls


def test_partner_sized_tiles_engage_on_a_render_after_an_automatic_load(monkeypatch):
    vae, pipe, render, stock_calls, split_calls = _installed_wan_behind_a_real_load(monkeypatch)
    assert render() == "split"
    (tiles,) = split_calls
    assert tiles != _STOCK_WIDE and stock_calls == []
    st = vm.stats(pipe)
    assert st["plan"] == "resized split" and st["tiles"] == tiles and st["split"] == 1
    assert tuple(getattr(vae, a) for a in vm._TILE_ATTRS) == _STOCK_WIDE
    # the opt-out is still read at decode time
    monkeypatch.setenv(vm.PARTNER_TILES_ENV, "0")
    assert render() == "stock"
    st = vm.stats(pipe)
    assert st["plan"] == "one card" and st["tiles"] == _STOCK_WIDE and "kept free" in st["last"]


def test_an_oom_in_the_partner_sized_split_falls_back_to_one_card_and_is_recorded(monkeypatch):
    # #19220's contract: an ordinary split failure (a partner OOM) decodes on one card at the stock tiles, which is
    # also what ran before partner-sized tiles for this shortfall. It is recorded, and that size is not retried.
    vae, pipe, render, stock_calls, split_calls = _installed_wan_behind_a_real_load(monkeypatch)

    def oom(model, latents, partner):
        split_calls.append(tuple(getattr(model, a) for a in vm._TILE_ATTRS))
        raise torch.OutOfMemoryError("CUDA out of memory on the partner")

    monkeypatch.setattr(vm, "split_tiled_decode", oom)
    assert render() == "stock"
    assert stock_calls == [_STOCK_WIDE] and len(split_calls) == 1
    st = vm.stats(pipe)
    assert st["plan"] == "one card" and st["resized_failures"] == 1
    assert "partner-sized split" in st["last"] and "OutOfMemoryError" in st["last"]
    assert render() == "stock" and len(split_calls) == 1  # the size that ran out is not tried again


def test_stats_record_the_stock_split_plan(monkeypatch):
    vae, pipe, render, stock_calls, split_calls = _installed_wan_behind_a_real_load(monkeypatch)
    monkeypatch.setattr(vm, "_partner_free_mib", lambda t, p: 11692)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d = None: (11692 << 20, 12288 << 20))
    assert render() == "split" and split_calls == [_STOCK_WIDE]
    st = vm.stats(pipe)
    assert st["plan"] == "split" and st["tiles"] == _STOCK_WIDE and st["last"] == "split"
