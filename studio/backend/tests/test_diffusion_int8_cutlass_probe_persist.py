# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The prebuilt CUTLASS tiles' bit-exact probe runs once per card + toolchain + library + code (CPU, fakes only).

A returning L4 load spent 1.3-1.5 s re-proving the same tiles before its first image. The verdict persists through
``diffusion_probe_cache.Namespace`` (card uuid, driver, torch / CUDA / Triton / torchao, env) plus these modules'
sources and the library's bytes; a missing, stale or malformed verdict probes again, and a failure persists only
when it repeats."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_cutlass as cut

torch = pytest.importorskip("torch")


class _Store:
    def __init__(self):
        self.table = {}

    def get(self, card, key):
        return self.table.get((card, key))

    def put(self, card, key, value):
        self.table[(card, key)] = value
        return True


@pytest.fixture
def env(monkeypatch):
    store = _Store()
    cut._PROBED.clear()
    monkeypatch.setattr(cut, "_probe_store", lambda: store)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    monkeypatch.setattr(cut, "library", lambda: object())
    monkeypatch.setattr(cut, "tiles", lambda fam: tuple(range(4)))
    monkeypatch.setattr(cut, "_alloc_failures", lambda index: 0)
    probes = []
    verdict = {0: True, 1: False, 2: True, 3: True}
    monkeypatch.setattr(cut, "_probe_tile", lambda i, fam, t, pairs: probes.append(t) or verdict[t])
    yield store, probes, verdict
    cut._PROBED.clear()


def test_first_process_probes_and_persists_then_the_next_reads_it(env):
    store, probes, _ = env
    assert cut.probed_tiles(0) == (0, 2, 3)
    assert probes == [0, 1, 2, 3, 1]  # the failing tile is asked twice before its verdict persists
    assert store.table[("cuda:0", "tiles|80")] == {"n": 4, "ok": [0, 2, 3]}
    cut._PROBED.clear()
    probes.clear()
    assert cut.probed_tiles(0) == (0, 2, 3) and probes == []


def test_a_flapping_failure_is_not_persisted(env):
    store, probes, verdict = env
    calls = {"n": 0}

    def flaky(i, fam, t, pairs):
        probes.append(t)
        if t == 1:
            calls["n"] += 1
            return calls["n"] > 1  # fails, then passes
        return True

    cut._probe_tile = flaky  # monkeypatch fixture restores via its own setattr record
    assert cut.probed_tiles(0) == (0, 2, 3)  # this process keeps the conservative verdict
    assert store.table == {}


@pytest.mark.parametrize(
    "counts", [[0, 1], [0, None], [None, None]], ids = ["oom", "sticky", "unreadable"]
)
def test_an_oom_or_a_broken_context_during_the_probe_is_not_persisted(env, monkeypatch, counts):
    """A full card (out-of-memory inside the probe) or a sticky launch error fails tiles for this process only."""
    store, probes, _ = env
    seen = iter(counts)
    monkeypatch.setattr(cut, "_alloc_failures", lambda index: next(seen))
    assert cut.probed_tiles(0) == (0, 2, 3)  # this process keeps the conservative verdict
    assert store.table == {}
    cut._PROBED.clear()
    probes.clear()
    monkeypatch.setattr(cut, "_alloc_failures", lambda index: 0)
    assert cut.probed_tiles(0) == (0, 2, 3) and probes[:4] == [
        0,
        1,
        2,
        3,
    ]  # probed again, then persisted
    assert store.table[("cuda:0", "tiles|80")] == {"n": 4, "ok": [0, 2, 3]}


def test_stale_or_malformed_verdicts_probe_again(env, monkeypatch):
    store, probes, _ = env
    for bad in ({"n": 5, "ok": [0]}, {"n": 4, "ok": [9]}, {"n": 4, "ok": "0"}, ["x"]):
        store.table[("cuda:0", "tiles|80")] = bad
        cut._PROBED.clear()
        probes.clear()
        assert cut.probed_tiles(0) == (0, 2, 3) and probes[:4] == [0, 1, 2, 3]


def test_the_key_covers_the_library_and_the_code(monkeypatch):
    from core.inference import diffusion_int8_gemm as g8

    assert set(cut.PROBE_PERSIST_SOURCES) == {"diffusion_int8_cutlass.py", "diffusion_int8_gemm.py"}
    monkeypatch.setattr(cut, "library_path", lambda: "/lib/x.so")
    monkeypatch.setattr(g8, "_sources_digest", lambda lib: f"digest-of-{lib}")
    assert cut._probe_extra() == {"abi": cut.ABI_VERSION, "lib": "digest-of-/lib/x.so"}
    monkeypatch.setattr(cut, "library_path", lambda: None)
    with pytest.raises(RuntimeError):
        cut._probe_extra()  # no library: the Namespace treats a raising extra as "never persisted"
    cut._PROBE_STORE[0] = None
    ns = cut._probe_store()
    assert ns.name == "int8_cutlass_tiles" and ns.sources == cut.PROBE_PERSIST_SOURCES
    assert ns.extra is cut._probe_extra
    cut._PROBE_STORE[0] = None


def test_a_real_namespace_misses_on_another_fingerprint(monkeypatch, tmp_path):
    from core.inference import diffusion_probe_cache as pc

    monkeypatch.setattr(pc, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    fp = {"gpu_uuid": "GPU-1", "driver": "580"}
    monkeypatch.setattr(pc, "fingerprint", lambda card: dict(fp))
    monkeypatch.setattr(cut, "_probe_extra", lambda: {"abi": 1, "lib": "L1"})
    ns = pc.Namespace(
        "int8_cutlass_tiles", cut.PROBE_PERSIST_SOURCES, extra = lambda: cut._probe_extra()
    )
    assert ns.put("cuda:0", "tiles|80", {"n": 4, "ok": [0]})
    fresh = pc.Namespace(
        "int8_cutlass_tiles", cut.PROBE_PERSIST_SOURCES, extra = lambda: cut._probe_extra()
    )
    assert fresh.get("cuda:0", "tiles|80") == {"n": 4, "ok": [0]}
    fp["driver"] = "590"  # a driver update
    assert (
        pc.Namespace(
            "int8_cutlass_tiles", cut.PROBE_PERSIST_SOURCES, extra = lambda: cut._probe_extra()
        ).get("cuda:0", "tiles|80")
        is None
    )
    fp["driver"] = "580"
    monkeypatch.setattr(cut, "_probe_extra", lambda: {"abi": 1, "lib": "L2"})  # a new library build
    assert (
        pc.Namespace(
            "int8_cutlass_tiles", cut.PROBE_PERSIST_SOURCES, extra = lambda: cut._probe_extra()
        ).get("cuda:0", "tiles|80")
        is None
    )
