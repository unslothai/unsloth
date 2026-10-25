# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image loads pin their streamed text encoders after the load returns, and pin the prefix that fits the budget."""

import ast
import pathlib
import threading
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_CUDA = torch.cuda.is_available()


@pytest.fixture
def budget(monkeypatch):
    for name in (
        dm.GROUP_OFFLOAD_PIN_ENV,
        dm.PARTIAL_ENCODER_PIN_ENV,
        dm.ENCODER_BACKGROUND_PIN_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)

    def set_budget(mib):
        monkeypatch.setattr(dm, "_pin_budget_mib", lambda: mib)

    set_budget(4387)
    return set_budget


def test_the_leftover_is_the_budget_after_a_pinned_transformer(budget):
    # the Kaggle T4 klein loads: nothing pinned at 4387 MiB, the transformer alone at 9868 MiB
    assert dm._streamed_pin_leftover_mib(7680, False) == 4387
    budget(9868)
    assert dm._streamed_pin_leftover_mib(7680, True) == 9868 - 7680


@pytest.mark.parametrize("mib", [None, 0, dm._PARTIAL_PIN_MIN_MIB - 1])
def test_no_leftover_without_a_useful_budget(budget, mib):
    budget(mib)
    assert dm._streamed_pin_leftover_mib(0, False) == 0


@pytest.mark.parametrize(
    "env, value",
    [
        (dm.PARTIAL_ENCODER_PIN_ENV, "0"),
        (dm.GROUP_OFFLOAD_PIN_ENV, "0"),
        (dm.GROUP_OFFLOAD_PIN_ENV, "1"),
    ],
)
def test_the_kill_switch_and_a_forced_plan_leave_nothing_to_split(budget, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert dm._streamed_pin_leftover_mib(0, False) == 0


def test_no_leftover_on_an_integrated_gpu_or_capped_pinned_memory(budget, monkeypatch):
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: True)
    assert dm._streamed_pin_leftover_mib(0, False) == 0
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: False)
    assert dm._streamed_pin_leftover_mib(0, False) == 0


@pytest.fixture
def wanted(monkeypatch):
    monkeypatch.delenv(dm.ENCODER_BACKGROUND_PIN_ENV, raising = False)
    monkeypatch.delenv(dm.BACKGROUND_PIN_ENV, raising = False)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: True)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: True)
    pipe = types.SimpleNamespace()
    dm.request_background_encoder_pins(pipe)
    return pipe


def test_a_requested_pipe_defers_its_encoder_pins(wanted):
    assert dm._encoder_background_pin_wanted(wanted)
    assert not dm._encoder_background_pin_wanted(types.SimpleNamespace())
    video = types.SimpleNamespace()
    dm.request_background_pins(video)
    assert not dm._encoder_background_pin_wanted(video)


@pytest.mark.parametrize("env", ["ENCODER_BACKGROUND_PIN_ENV", "BACKGROUND_PIN_ENV"])
def test_the_kill_switches_restore_the_up_front_pin(wanted, monkeypatch, env):
    monkeypatch.setenv(getattr(dm, env), "0")
    assert not dm._encoder_background_pin_wanted(wanted)


def test_no_deferral_without_the_registered_pin_or_on_an_integrated_gpu(wanted, monkeypatch):
    # the allocator's pin_memory() rounds every tensor to a power of two, past the weight-byte budget
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: False)
    assert not dm._encoder_background_pin_wanted(wanted)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda: True)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *a, **k: False)
    assert not dm._encoder_background_pin_wanted(wanted)


def test_a_registered_pin_is_priced_at_page_rounded_bytes_not_powers_of_two():
    module = torch.nn.Linear(1500, 1000, bias = False).to(torch.bfloat16)  # 3,000,000 bytes
    assert dm._module_host_mib(module) == 4  # the allocator's 4 MiB block
    assert dm._module_registered_mib(module) == 3


def test_the_load_starts_and_waits_for_the_encoder_pins(monkeypatch):
    calls = []
    monkeypatch.setattr(dm, "start_background_pins", lambda pipe: calls.append("start") or 1)
    monkeypatch.setattr(dm, "finish_background_pins", lambda pipe: calls.append("finish") or 0.0)
    dm.finish_encoder_pins_at_load(types.SimpleNamespace())
    assert calls == ["start", "finish"]
    assert dm.finish_encoder_pins_at_load(None) == 0.0


def _fake_groups(sizes):
    groups = []
    for n in sizes:
        weight = torch.nn.Parameter(torch.zeros(n, dtype = torch.uint8), requires_grad = False)
        groups.append(types.SimpleNamespace(cpu_param_dict = {weight: weight.data}))
    return groups


def test_a_capped_pinner_pins_the_prefix_that_fits_and_never_blocks_the_rest(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: getattr(self, "_fake_pin", False))

    def pin_memory(self):
        out = self.clone()
        out._fake_pin = True
        return out

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    # registered copies are page-rounded: 1 + 2 + 1 + 1 pages
    groups = _fake_groups([100, 5000, 300, 50])
    sources = [dict(g.cpu_param_dict) for g in groups]
    pinner = dm._GroupPinner(
        torch.nn.Module(), groups, torch.device("cpu"), byte_cap = 3 * 4096 + 100
    )
    assert [id(g) for g in pinner._capped_groups()] == [id(groups[0]), id(groups[1])]
    pinner._run()
    assert pinner.pinned == 5100
    for group, before in zip(groups, sources):
        assert pinner._done[id(group)].is_set()
        for tensor, src in group.cpu_param_dict.items():
            pinned = getattr(src, "_fake_pin", False)
            assert pinned is (group in groups[:2])
            assert tensor.data_ptr() == src.data_ptr()
            if not pinned:
                assert src is before[tensor]


def test_an_onload_past_the_cap_does_not_wait_for_the_prefix(monkeypatch):
    groups = _fake_groups([10, 1000])
    release = threading.Event()

    def slow_pin(self):
        release.wait(5)  # the prefix group is still copying
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: False)
    monkeypatch.setattr(torch.Tensor, "pin_memory", slow_pin)
    pinner = dm._GroupPinner(torch.nn.Module(), groups, torch.device("cpu"), byte_cap = 4096)
    pinner.start()
    try:
        assert pinner._done[id(groups[1])].wait(2)
        assert not pinner._done[id(groups[0])].is_set()
    finally:
        release.set()
        pinner.join(5)
    assert pinner._done[id(groups[0])].is_set()


def _calls(tree, name):
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == name
    ]


def test_the_image_load_requests_starts_and_stops_the_encoder_pins():
    tree = ast.parse((_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8"))
    request = _calls(tree, "request_background_encoder_pins")
    apply = _calls(tree, "apply_memory_plan")
    assert request and apply and min(c.lineno for c in request) < min(c.lineno for c in apply)
    starts = _calls(tree, "start_background_pins")
    assert starts and min(c.lineno for c in starts) > min(c.lineno for c in apply)
    assert _calls(tree, "finish_encoder_pins_at_load")
    unload = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_unload_locked"
    )
    stops = _calls(unload, "stop_background_pins")
    releases = _calls(unload, "release_pinned_host_memory")
    assert stops and releases and stops[0].lineno < releases[0].lineno


class _Pipe:
    pass


def _encoder():
    torch.manual_seed(0)
    # 2 MiB of bf16 weight per Linear, one leaf group each
    return torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(4)]).to(torch.bfloat16)


def _tiny_pipe():
    pipe = _Pipe()
    pipe.transformer = torch.nn.Linear(1024, 1024).to(torch.bfloat16)
    pipe.text_encoder = _encoder()
    pipe.components = {"transformer": pipe.transformer, "text_encoder": pipe.text_encoder}
    return pipe


def _host_copies(module):
    return [t for g in dm._offload_groups(module) for t in g.cpu_param_dict.values()]


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
@pytest.mark.parametrize("cap_mib", [None, 0])
def test_a_streamed_encoder_pins_after_the_load_and_encodes_identically(
    budget, monkeypatch, cap_mib
):
    pytest.importorskip("diffusers")
    monkeypatch.delenv(dm.BACKGROUND_PIN_ENV, raising = False)
    x = torch.randn(4, 1024, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        ref = _encoder().to("cuda")(x)
    budget(1 << 20)
    pipe = _tiny_pipe()
    dm.request_background_encoder_pins(pipe)
    if cap_mib == 0:
        # whole pin refused, 5 MiB left: the first two 2 MiB groups fit, the third does not
        monkeypatch.setattr(dm, "_streamed_pin_plan", lambda *a, **k: (True, False))
        monkeypatch.setattr(dm, "_streamed_pin_leftover_mib", lambda *a: 5)
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    hosts = _host_copies(pipe.text_encoder)
    assert hosts and not any(t.is_pinned() for t in hosts)
    assert dm.start_background_pins(pipe) == 1
    with torch.no_grad():
        out = pipe.text_encoder(x)
    torch.cuda.synchronize()
    dm.finish_background_pins(pipe)
    assert torch.equal(out, ref)
    pinned = [t.is_pinned() for t in _host_copies(pipe.text_encoder)]
    if cap_mib is None:
        assert all(pinned)
    else:
        assert pinned == [True] * 4 + [False] * 4  # weight + bias of the first two groups, in order
    with torch.no_grad():
        assert torch.equal(pipe.text_encoder(x), ref)
    dm.stop_background_pins(pipe)


@pytest.mark.skipif(not _CUDA, reason = "needs a CUDA device")
def test_the_kill_switch_pins_the_encoder_into_slabs_during_the_apply(budget, monkeypatch):
    pytest.importorskip("diffusers")
    monkeypatch.setenv(dm.ENCODER_BACKGROUND_PIN_ENV, "0")
    budget(1 << 20)
    pipe = _tiny_pipe()
    dm.request_background_encoder_pins(pipe)
    assert dm._apply_group_offload(
        pipe, "cuda", None, stream_text_encoders = True, stream_transformer = False
    )
    assert dm.start_background_pins(pipe) == 0
    assert all(t.is_pinned() for t in _host_copies(pipe.text_encoder))


def test_an_uncommitted_load_stops_the_pinners_it_started():
    import ast
    import inspect

    from core.inference import diffusion

    tree = ast.parse(inspect.getsource(diffusion))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = {
            node.func.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        if "start_background_pins" not in calls:
            continue
        rollbacks = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == "state_committed"
        ]
        assert rollbacks, fn.name
        stopped = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "stop_background_pins"
            for rollback in rollbacks
            for node in ast.walk(rollback)
        )
        assert (
            stopped
        ), f"{fn.name} starts encoder pinners but its uncommitted-load rollback never stops them"
