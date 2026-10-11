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
def test_only_sglang_reserves_space_for_its_derived_port(monkeypatch, name, ports, expected):
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
