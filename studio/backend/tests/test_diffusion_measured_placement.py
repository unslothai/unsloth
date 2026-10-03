# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured-activation placement and partial denoiser residency (``diffusion_memory.py``), plus the Qwen-Image-2.1
prefix K/V compaction (``diffusion_qwenimage21.py``).

The planner cases use the estimates a real Qwen-Image-2.1 auto load logged (int8 DiT + fp8 text encoder): on a 24 GB
card the flat plan read ``safe_device_budget_mib 21432, resident_required_mib 29870`` and streamed the text encoder
on every prompt, although the loaded weights are 6922 + 8959 + 644 MiB. CPU-only; the loaded sizes and the torchao
check are stubbed. The partial-residency mechanics need a CUDA device and real diffusers group offloading.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

# Loaded storage of Qwen-Image-2.1 auto (int8 transformer, fp8 text encoder, bf16 VAE), MiB.
Q21_LOADED = {
    "transformer": (6922, "dit"),
    "text_encoder": (8959, "text_encoder"),
    "vae": (644, "other"),
}
# The flat plan's inputs for that load (family table sizes, flat 8192 headroom).
Q21_FLAT = dict(model_dense_mib = 19630, companion_dense_mib = 12182, text_encoder_dense_mib = 10847)


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    """A discrete CUDA snapshot whose safe budget is exactly ``budget_mib``."""
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _flat_plan(
    budget_mib: int,
    total_mib: int,
    mode = None,
):
    plan = dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "qwen-image-2.1"
        ),
        requested_mode = mode,
        **Q21_FLAT,
    )
    if plan.offload_policy == dm.OFFLOAD_MODEL:
        # what torchao_offload_plan makes of whole-module offload for a streamable int8 denoiser
        plan = dm.torchao_streaming_plan(plan)
    return plan


@pytest.fixture
def q21_pipe(monkeypatch):
    monkeypatch.delenv(
        getattr(dm, "MEASURED_ACTIVATION_ENV", "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION"),
        raising = False,
    )
    monkeypatch.delenv(
        getattr(dm, "PARTIAL_RESIDENT_ENV", "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT"), raising = False
    )
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(Q21_LOADED), raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    return object()


def _refine(
    pipe,
    plan,
    family = "qwen-image-2.1",
    speed = "default",
):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


def test_measured_headroom_values(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION", raising = False)
    # 1849 MiB measured worst phase x 1.15, rounded up to 256 MiB; linear in pixels
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "default") == 2304
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "max") == 2304
    assert (
        dm.measured_image_runtime_mib("qwen-image-2.1", "default", width = 2048, height = 2048) == 8704
    )
    # never below the 1 MP figure for a smaller canvas
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "default", width = 512, height = 512) == 2304
    # unmeasured family / eager tiers keep the flat estimate
    assert dm.measured_image_runtime_mib("flux", "default") is None
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "off") is None
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "eager") is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION", "0")
    assert dm.measured_image_runtime_mib("qwen-image-2.1", "default") is None


def test_24gb_keeps_transformer_and_text_encoder_resident(q21_pipe):
    plan = _flat_plan(21432, 24576)
    # the reported flat decision: DiT resident, text encoder streamed on every prompt
    assert plan.estimates["resident_required_mib"] == 29870
    assert (
        plan.offload_policy == dm.OFFLOAD_GROUP
        and plan.stream_text_encoders
        and not plan.stream_transformer
    )
    new = _refine(q21_pipe, plan)
    # the flat hooks stay (so an oversized request can stream again); the whole encoder is kept resident in them
    assert new.offload_policy == dm.OFFLOAD_GROUP
    assert new.stream_text_encoders and not new.stream_transformer
    assert new.resident_transformer_mib is None
    assert (
        new.resident_text_encoder_mib
        == 21432 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644 - 6922
        >= 8959
    )
    assert new.estimates["measured_runtime_headroom_mib"] == 2304
    # never under-reserved: loaded weights + measured peak x margin + base overhead within the safe budget
    assert 6922 + 8959 + 644 + 2304 + dm.DEFAULT_BASE_OVERHEAD_MIB <= 21432


def test_16gb_keeps_transformer_resident_streams_encoder(q21_pipe):
    plan = _flat_plan(13638, 16376)
    assert (
        plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_transformer
    )  # flat: every DiT block streamed
    new = _refine(q21_pipe, plan)
    assert new.offload_policy == dm.OFFLOAD_GROUP
    assert new.stream_text_encoders and new.stream_transformer
    # every DiT group kept resident through its hooks, the rest of the room holds encoder layers
    assert new.resident_transformer_mib == 6922


def test_l4_class_card_keeps_most_of_the_encoder_resident(q21_pipe):
    """An L4 reports 23034 MiB: its 10% reserve leaves the whole encoder ~900 MiB short, so the DiT stays resident
    and only the encoder layers past the budget stream."""
    budget = 20000
    new = _refine(q21_pipe, _flat_plan(budget, 23034))
    assert new.offload_policy == dm.OFFLOAD_GROUP and new.stream_text_encoders
    room = budget - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644 - 6922
    assert new.resident_text_encoder_mib == room and 0 < room < 8959
    assert new.as_public_dict()["resident_text_encoder_mib"] == room


def test_16gb_encoder_room_and_kill_switch(q21_pipe, monkeypatch):
    new = _refine(q21_pipe, _flat_plan(13638, 16376))
    assert new.resident_text_encoder_mib == 13638 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644 - 6922
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", "0")
    plan = _flat_plan(13638, 16376)
    assert _refine(q21_pipe, plan) is plan


def test_12gb_partial_residency_sized_from_budget(q21_pipe):
    plan = _flat_plan(9550, 12288)
    assert plan.offload_policy == dm.OFFLOAD_STREAMING
    new = _refine(q21_pipe, plan)
    assert new.offload_policy == dm.OFFLOAD_STREAMING
    room = 9550 - 2304 - dm.DEFAULT_BASE_OVERHEAD_MIB - 644
    assert new.resident_transformer_mib == room
    assert 0 < room < 6922
    assert new.as_public_dict()["resident_transformer_mib"] == room


def test_no_room_keeps_flat_plan(q21_pipe):
    plan = _flat_plan(4000, 8188)
    assert _refine(q21_pipe, plan) is plan


@pytest.mark.parametrize(
    "total,budget", [(24576, 21432), (16376, 13638), (12288, 9550), (8188, 5450)]
)
def test_kill_switch_restores_flat_plan(q21_pipe, monkeypatch, total, budget):
    plan = _flat_plan(budget, total)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION", "0")
    assert _refine(q21_pipe, plan) is plan


def test_partial_kill_switch_only_drops_partial_tier(q21_pipe, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", "0")
    for total, budget in ((12288, 9550), (16376, 13638), (24576, 21432)):
        plan = _flat_plan(budget, total)
        assert _refine(q21_pipe, plan) is plan


@pytest.mark.parametrize(
    "family,speed,mode",
    [
        ("flux", "default", None),
        ("qwen-image-2.1", "off", None),
        ("qwen-image-2.1", "default", "balanced"),
    ],
)
def test_unmeasured_or_explicit_untouched(q21_pipe, family, speed, mode):
    plan = _flat_plan(21432, 24576, mode = mode)
    assert _refine(q21_pipe, plan, family = family, speed = speed) is plan


def test_non_torchao_and_unified_untouched(q21_pipe, monkeypatch):
    plan = _flat_plan(21432, 24576)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    assert _refine(q21_pipe, plan) is plan
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    unified = dm.replace(
        plan, device_memory = dm.DeviceMemory("cuda", "cuda", "unified_memory", 23000, 24576)
    )
    assert _refine(q21_pipe, unified) is unified


def test_never_moves_away_from_residency(q21_pipe):
    resident = _flat_plan(200000, 200000)
    assert resident.offload_policy == dm.OFFLOAD_NONE
    assert _refine(q21_pipe, resident) is resident
    model = dm.replace(_flat_plan(9550, 12288), offload_policy = dm.OFFLOAD_MODEL)
    assert _refine(q21_pipe, model) is model


@pytest.mark.parametrize("budget", list(range(3000, 30001, 250)))
def test_every_tier_fits_measured_need(q21_pipe, budget):
    """Whatever the refinement picks, the bytes it keeps on the device plus the measured peak x margin plus the base
    overhead fit the safe budget (the reserve below it is untouched), and it is never slower than the flat pick."""
    total = max(8188, budget + 3000)
    plan = _flat_plan(budget, total)
    new = _refine(q21_pipe, plan)
    head = 2304 + dm.DEFAULT_BASE_OVERHEAD_MIB
    if new is plan:
        return
    # the policy and stream flags are the flat plan's: only resident rooms are added
    assert (new.offload_policy, new.stream_transformer, new.stream_text_encoders) == (
        plan.offload_policy,
        plan.stream_transformer,
        plan.stream_text_encoders,
    )
    kept = (
        644
        + (6922 if not new.stream_transformer else int(new.resident_transformer_mib or 0))
        + int(new.resident_text_encoder_mib or 0)
        + (0 if new.stream_text_encoders or new.offload_policy == dm.OFFLOAD_STREAMING else 8959)
    )
    assert int(new.resident_transformer_mib or 0) <= 6922
    assert kept + head <= budget


def test_compact_layer_cache_frees_full_storage(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_qwenimage21 as q21

    full_k = torch.randn(1, 40, 4, 8)
    full_v = torch.randn(1, 40, 4, 8)
    cache = types.SimpleNamespace(k = full_k[:, :6], v = full_v[:, :6])
    assert cache.k.untyped_storage().nbytes() > cache.k.numel() * cache.k.element_size()
    q21._compact_layer_cache(cache)
    for got, ref in ((cache.k, full_k[:, :6]), (cache.v, full_v[:, :6])):
        assert got.untyped_storage().nbytes() == got.numel() * got.element_size()
        assert torch.equal(got, ref)
    owned = cache.k
    q21._compact_layer_cache(cache)
    assert cache.k is owned  # already compact: untouched
    monkeypatch.setenv("UNSLOTH_DIFFUSION_Q21_COMPACT_KV", "0")
    assert not q21.compact_kv_enabled()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_Q21_COMPACT_KV")
    assert q21.compact_kv_enabled()


def _cuda_offload_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only: diffusers group offloading with a copy stream")
    pytest.importorskip("diffusers.hooks")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(64, 1024)
            self.blocks = torch.nn.ModuleList(
                torch.nn.Linear(1024, 1024) for _ in range(6)
            )  # ~4 MiB each
            self.proj_out = torch.nn.Linear(1024, 64)

        def forward(self, x):
            x = self.proj_in(x)
            for block in self.blocks:
                x = torch.nn.functional.gelu(block(x))
            return self.proj_out(x)

    torch.manual_seed(0)
    return torch, Net()


@pytest.mark.parametrize("keep_blocks", [0, 2, 6])
def test_partial_residency_bit_identical_and_placed(keep_blocks, monkeypatch):
    torch, net = _cuda_offload_model()
    from diffusers.hooks import apply_group_offloading

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)
    x = torch.randn(3, 64)
    ref = net.to("cuda")(x.cuda()).cpu()
    net.to("cpu")
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
    )
    mib = 1024 * 1024
    top = sum(
        p.numel() * p.element_size() for m in (net.proj_in, net.proj_out) for p in m.parameters()
    )
    block = sum(p.numel() * p.element_size() for p in net.blocks[0].parameters())
    # whole MiB, rounded up: room for the top-level group and keep_blocks blocks, short of one more block
    room = -(-(top + keep_blocks * block) // mib) if keep_blocks else 0
    kept = dm._keep_groups_resident(net, room, "cuda")
    for _ in range(3):  # first forward traces the prefetch order, later ones prefetch
        out = net(x.cuda()).cpu()
        assert torch.equal(out, ref)
    placed = [next(b.parameters()).device.type for b in net.blocks]
    want = ["cuda"] * min(keep_blocks, 6) + ["cpu"] * (6 - min(keep_blocks, 6))
    if keep_blocks:
        assert kept > 0
        assert next(net.proj_in.parameters()).device.type == "cuda"
        assert placed[:keep_blocks] == want[:keep_blocks]
    assert placed[keep_blocks:] == want[keep_blocks:]


def test_partial_residency_kill_switch(monkeypatch):
    torch, net = _cuda_offload_model()
    from diffusers.hooks import apply_group_offloading

    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
    )
    monkeypatch.setenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", "0")
    assert dm._keep_groups_resident(net, 10_000, "cuda") == 0
    assert all(next(b.parameters()).device.type == "cpu" for b in net.blocks)


def _family_names():
    from core.inference.diffusion_families import supported_family_names
    return supported_family_names()


@pytest.mark.parametrize("speed", ["off", "eager", "default", "max"])
@pytest.mark.parametrize(
    "budget,total", [(21432, 24576), (13638, 16376), (9550, 12288), (5450, 8188)]
)
def test_only_measured_family_moves(q21_pipe, speed, budget, total):
    """Every other supported family keeps its flat plan byte for byte, on every speed tier and budget."""
    names = _family_names()
    assert "qwen-image-2.1" in names and len(names) > 5
    plan = _flat_plan(budget, total)
    for family in names:
        new = _refine(q21_pipe, plan, family = family, speed = speed)
        if family == "qwen-image-2.1" and speed in ("default", "max"):
            continue
        assert new is plan, family


def test_resident_group_keeps_copy_stream_wait(monkeypatch):
    """A streamed group that prefetched its successor skips its own wait and relies on the successor's onload_ to
    synchronize the copy stream. A resident successor must still do that wait (else the streamed group computes on
    weights in flight), and its offload must free nothing."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers.hooks")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)

    class Stream:
        waits = 0

        def synchronize(self):
            Stream.waits += 1

    module = types.SimpleNamespace()
    stream = Stream()
    groups = [
        types.SimpleNamespace(
            modules = [],
            parameters = [],
            buffers = [],
            offload_leader = object(),
            stream = stream,
            cpu_param_dict = {},
            offload_to_disk_path = None,
        )
        for _ in range(3)
    ]
    monkeypatch.setattr(dm, "_offload_groups", lambda m: groups)
    dm._keep_groups_resident(module, 1, "cpu")
    assert all(getattr(g, "_unsloth_resident", False) for g in groups)
    # every group resident: nothing is ever queued on the copy stream, so there is nothing to wait for
    for g in groups:
        g.onload_()
        g.offload_()
    assert Stream.waits == 0
    # one group of the module streams again (an oversized request): the resident ones wait for it again
    module._unsloth_stream_state["streamed"] = 1
    for g in groups:
        g.onload_()
        g.offload_()
    assert Stream.waits == 3


def test_oversized_request_streams_resident_groups_and_restores_them(monkeypatch):
    """A request past the measured reserve streams the kept groups again (the flat plan's placement), stays
    bit-identical, and the groups are resident again afterwards."""
    torch, net = _cuda_offload_model()
    from diffusers.hooks import apply_group_offloading

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)
    x = torch.randn(3, 64)
    ref = net.to("cuda")(x.cuda()).cpu()
    net.to("cpu")
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
    )
    pipe = types.SimpleNamespace(components = {"transformer": net})
    assert dm._keep_groups_resident(net, 1024, "cuda") > 0
    assert all(next(b.parameters()).device.type == "cuda" for b in net.blocks)
    assert torch.equal(net(x.cuda()).cpu(), ref)

    restore = dm.release_resident_groups(pipe, 1024)
    assert restore is not None
    # blocks first, from the last; the top-level group goes last
    assert all(next(b.parameters()).device.type == "cpu" for b in net.blocks)
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    assert all(next(b.parameters()).device.type == "cpu" for b in net.blocks)

    restore()
    assert all(next(b.parameters()).device.type == "cuda" for b in net.blocks)
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    assert dm.release_resident_groups(types.SimpleNamespace(components = {}), 1024) is None


def test_partial_release_frees_only_what_the_request_needs(monkeypatch):
    torch, net = _cuda_offload_model()
    from diffusers.hooks import apply_group_offloading

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
    )
    dm._keep_groups_resident(net, 1024, "cuda")
    pipe = types.SimpleNamespace(components = {"transformer": net})
    restore = dm.release_resident_groups(pipe, 1)  # one ~4 MiB block covers 1 MiB
    placed = [next(b.parameters()).device.type for b in net.blocks]
    assert placed == ["cuda"] * 5 + ["cpu"]
    restore()
    assert all(next(b.parameters()).device.type == "cuda" for b in net.blocks)


def test_measured_request_extra(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION", raising = False)
    pipe = types.SimpleNamespace(_unsloth_measured_reserve = (2304, "qwen-image-2.1", "default"))
    # the default request is what the reserve was sized for
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    assert dm.measured_request_extra_mib(pipe, width = 512, height = 512) == 0
    # a bigger canvas, a batch and reference images each need more
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 2048) == 8704 - 2304
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, batch_size = 2) > 0
    one_ref = int(1024 * 1024 * 0.32)
    assert dm.measured_request_extra_mib(
        pipe, width = 1024, height = 1024, condition_pixels = one_ref
    ) == int(8192 * 0.32 * 1.15)
    # no measured placement: never releases
    assert dm.measured_request_extra_mib(object(), width = 2048, height = 2048) == 0


def test_generate_releases_and_restores_resident_groups():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    assert "restore_resident = release_resident_groups(state.pipe, extra_mib, logger)" in src
    finally_at = src.index("if restore_resident is not None:")
    assert "restore_resident()" in src[finally_at : finally_at + 200]
    assert "pipe._unsloth_measured_reserve = (" in src


def test_torchao_groups_stay_on_device_after_release_and_restore(monkeypatch):
    """An oversized request streams the pinned groups of a torchao int8 DiT, then restore pins them again. The int8
    wrapper can still report the onload device after a streamed offload while its data and scales sit on the host;
    restore must judge it by the inner tensors, else the next request computes on a host weight."""
    torch, _ = _cuda_offload_model()
    pytest.importorskip("torchao")
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)
    cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config = False)
    if not hasattr(cfg, "version"):
        pytest.skip("torchao without config versions")
    cfg.version = 2
    torch.manual_seed(0)
    net = (
        torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.Sequential(*[torch.nn.Linear(512, 512) for _ in range(3)]),
        )
        .cuda()
        .to(torch.bfloat16)
    )
    quantize_(net[1], cfg)
    x = torch.randn(64, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = net(x)
    net.to("cpu")
    kwargs = dict(
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
    )
    apply_group_offloading(net, **dm._torchao_group_offload_kwargs(net, kwargs, [0]))
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    assert dm._keep_groups_resident(net, 1024, "cuda") > 0
    with torch.no_grad():
        assert torch.equal(net(x), ref)
        restore = dm.release_resident_groups(pipe, 1024)
        assert restore is not None
        for _ in range(2):
            assert torch.equal(net(x), ref)
        restore()
        for lin in net[1]:
            # torchao 0.14 (torch <= 2.9) keeps the v1 layout, nesting its int8 data one subclass deeper
            assert set(_inner_devices(lin.weight)) == {"cuda"}
        assert torch.equal(net(x), ref)


def _inner_devices(tensor):
    names, _ = tensor.__tensor_flatten__()
    out = []
    for name in names:
        inner = getattr(tensor, name)
        out += (
            _inner_devices(inner) if hasattr(inner, "__tensor_flatten__") else [inner.device.type]
        )
    return out


def test_eager_offload_hooks_install_is_idempotent():
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    dm.install_group_offload_hooks_eager()
    hooks = (
        (go.GroupOffloadingHook, "pre_forward"),
        (go.GroupOffloadingHook, "post_forward"),
        (go.LayerExecutionTrackerHook, "pre_forward"),
        (go.LazyPrefetchGroupOffloadingHook, "post_forward"),
    )
    patched = [cls.__dict__[name] for cls, name in hooks]
    assert all(getattr(fn, "_unsloth_eager", False) for fn in patched)
    assert dm.install_group_offload_hooks_eager() is False
    assert [cls.__dict__[name] for cls, name in hooks] == patched


def test_eager_offload_hooks_kill_switch(monkeypatch, traced_offload_hooks):
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    monkeypatch.setenv(dm.EAGER_OFFLOAD_HOOKS_ENV, "0")
    assert dm.install_group_offload_hooks_eager() is False
    assert not getattr(go.GroupOffloadingHook.__dict__["pre_forward"], "_unsloth_eager", False)


def test_failed_release_keeps_the_copy_stream_wait(monkeypatch):
    groups = [
        types.SimpleNamespace(
            stream = object(), _unsloth_resident = True, _unsloth_resident_bytes = 1 << 20
        )
        for _ in range(2)
    ]
    module = types.SimpleNamespace(_unsloth_resident_room = 1, _unsloth_stream_state = {"streamed": 0})
    monkeypatch.setattr(dm, "_offload_groups", lambda m: groups)

    def boom(group):
        raise RuntimeError("offload callback failed")

    monkeypatch.setattr(dm, "_release_group", boom)
    pipe = types.SimpleNamespace(components = {"transformer": module})
    assert dm.release_resident_groups(pipe, 4) is None
    assert module._unsloth_stream_state["streamed"] >= 1


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason = "stream group offload needs a CUDA device"
)
def test_compiled_blocks_under_group_offload_skip_hook_tracing():
    import torch

    pytest.importorskip("diffusers.hooks.group_offloading")
    from diffusers.hooks import apply_group_offloading
    from torch._dynamo.utils import counters

    dm.install_group_offload_hooks_eager()
    torch.manual_seed(0)
    make = lambda: torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.GELU())
    blocks = torch.nn.ModuleList([make() for _ in range(6)])
    plain = [make().cuda() for _ in blocks]
    for ref, block in zip(plain, blocks):
        ref.load_state_dict(block.state_dict())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = Model()
    apply_group_offloading(
        model,
        onload_device = torch.device("cuda"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
    )
    x = torch.randn(64, 256, device = "cuda")
    with torch.no_grad():
        for block in [*model.blocks, *plain]:
            block.compile()
        expected = x
        for ref in plain:
            expected = ref(expected)
        torch._dynamo.reset()
        counters.clear()
        outs = [model(x) for _ in range(3)]
    for out in outs:
        torch.testing.assert_close(out, expected, rtol = 0, atol = 1e-6)
    # Traced hooks add a graph per hook variant; eager hooks leave only the block graph.
    assert counters["stats"]["unique_graphs"] <= 1
