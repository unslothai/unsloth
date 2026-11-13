# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engine-specific bounds on operating-system-assigned HTTP ports."""

from contextlib import nullcontext

import pytest

from core.inference import managed_engine


@pytest.mark.parametrize(
    "name,ports,expected",
    [
        ("vllm", [60000] * 100, 60000),
        ("sglang", [60000, 8123], 8123),
        ("sglang", [60000] * 100, None),
    ],
)
def test_only_sglang_reserves_space_for_its_derived_port(
    monkeypatch, real_engine_free_port, name, ports, expected
):
    assigned = iter(ports)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def bind(self, _):
            pass

        def getsockname(self):
            return "127.0.0.1", next(assigned)

    monkeypatch.setattr(managed_engine.socket, "socket", Socket)
    monkeypatch.setattr(managed_engine, "engine_lease", lambda _: nullcontext())
    monkeypatch.setattr(managed_engine, "installed", lambda _: {"host": "wsl"})
    engine = managed_engine.ManagedEngine(name)
    selected = []

    def launch(*args):
        selected.append(args[-1])
        raise RuntimeError("test reached engine launch")

    monkeypatch.setattr(engine, "_wsl_command", launch)
    message = (
        "test reached engine launch" if expected else "Could not allocate an inference server port"
    )
    with pytest.raises(RuntimeError, match = message):
        engine.start("model", 2048, [0], {})
    assert selected == ([] if expected is None else [expected])


def test_the_suite_hands_engines_ports_no_other_process_can_be_given():
    """conftest keeps engine ports out of the ephemeral range: a port released there can be handed
    to any bind(0) on the runner before the engine binds it, and that listener then answers /health."""
    for _ in range(3):
        port = managed_engine._free_port()
        assert 21000 <= port < 31000, port


@pytest.mark.parametrize("workers", [1, 4, 16, 40, 41, 64, 192, 512])
def test_every_xdist_worker_gets_its_own_engine_port_block(engine_port_block, workers):
    """A shared block lets one worker bind a port another just released and answer its /health."""
    taken = set()
    for index in range(workers):
        base, size = engine_port_block(index, workers)
        block = set(range(base, base + size))
        assert size >= 1 and not (block & taken), (index, base, size)
        assert 21000 <= base and base + size <= 31000, (index, base, size)
        taken |= block


@pytest.mark.parametrize("workers, crashed", [(2, 1), (4, 0), (40, 17), (64, 63)])
def test_a_replacement_xdist_worker_takes_the_crashed_workers_slot(
    engine_port_block, engine_port_slot_hooks, workers, crashed
):
    """xdist names any crash's replacement gw<count>; it must take the dead worker's block, not a survivor's."""
    import types

    configure_node, node_down = engine_port_slot_hooks

    def node(worker_id):
        return types.SimpleNamespace(gateway = types.SimpleNamespace(id = worker_id), workerinput = {})

    def start(worker_id):
        started = node(worker_id)
        configure_node(started)
        return started.workerinput["engine_port_slot"]

    slots = {f"gw{index}": start(f"gw{index}") for index in range(workers)}
    node_down(node(f"gw{crashed}"), None)
    del slots[f"gw{crashed}"]
    replacement = start(f"gw{workers}")
    assert replacement == crashed
    base, size = engine_port_block(replacement, workers)
    for slot in slots.values():
        other, other_size = engine_port_block(slot, workers)
        assert base + size <= other or other + other_size <= base, (slot, base, other)


@pytest.mark.parametrize("workers", [2, 4, 16])
def test_a_replacement_without_a_controller_slot_avoids_every_survivor(engine_port_block, workers):
    """`pytest studio -n N` never loads the slot hooks on the controller, so gw<count> falls back to its id."""
    base, size = engine_port_block(workers, workers)
    for index in range(workers):
        other, other_size = engine_port_block(index, workers)
        assert base + size <= other or other + other_size <= base, (index, base, other)
