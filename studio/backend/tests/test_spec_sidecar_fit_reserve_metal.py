# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metal: Auto drops a sidecar whose reserve stops the target fitting; explicit DSpark keeps it."""

from __future__ import annotations

import pytest

from . import _sidecar_fit_helpers as h
from ._sidecar_fit_helpers import GIB, lc, metal_cmd

B = lc.LlamaCppBackend


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _embd(
    monkeypatch,
    value,
    default = False,
):
    """Excluded NextN blocks forfeit the pageable-embedding discount; the drafter may engage."""
    calls = []

    def embd(
        self,
        *a,
        mtp_may_engage = default,
        **k,
    ):
        calls.append(mtp_may_engage)
        return value(mtp_may_engage)

    monkeypatch.setattr(B, "_metal_demand_paged_embedding_bytes", embd)
    return calls


def _drafter_free_discount(mtp):
    return 0 if mtp else 3 * GIB


# Every GGUF here reads 4 GiB; the target with compute and soft overhead prices ~9.2 GiB,
# so 12 GiB holds the target alone but not target + sidecar, and 16 GiB holds both.
class TestMetalLaunch:
    @pytest.mark.parametrize(
        "extras, kept",
        [
            ([], False),
            # Only one block on the card: not dropped for the whole file.
            (["--spec-draft-ngl", "1"], True),
            (["--cpu-moe-draft"], True),
            # Its weights allocate on the RPC host, so Metal's budget is no reason to drop it.
            (["--rpc", "192.0.2.1:50052", "--device", "MTL0", "--device-draft", "RPC0"], True),
        ],
        ids = [
            "test_auto_drops_a_sidecar_that_does_not_fit",
            "test_metal_auto_keeps_a_partially_offloaded_drafter",
            "test_metal_auto_keeps_a_drafter_with_its_experts_on_cpu",
            "test_metal_auto_keeps_a_drafter_pinned_to_another_device",
        ],
    )
    def test_auto_at_12_gib(self, tmp_path, monkeypatch, extras, kept):
        cmd, backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = extras)
        assert ("--model-draft" in cmd) is kept
        assert (backend.spec_fallback_reason == "drafter_no_vram") is not kept

    def test_explicit_dspark_keeps_it_with_the_reserve(self, tmp_path, monkeypatch):
        cmd, _backend, path = metal_cmd(tmp_path, monkeypatch, "dspark", budget_gib = 12)
        assert h.last(cmd, "--model-draft") == path
        assert h.margin(cmd) >= 1024 + 4 * 1024

    def test_auto_keeps_a_sidecar_that_fits(self, tmp_path, monkeypatch):
        cmd, backend, path = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 16)
        assert h.last(cmd, "--model-draft") == path
        assert h.margin(cmd) >= 1024 + 4 * 1024
        assert backend.spec_fallback_reason is None


@pytest.mark.parametrize(
    "budget_gib, kw, discount",
    [
        # The flat draft reserve and the draft compute charge go with the sidecar.
        (12, {"kv_per_token": 128 * 1024}, None),
        # Within 224 MiB of the target-only footprint the drafter's compute must not keep it.
        (9.1, {}, None),
        # The discount decided with the drafter engaged is decided again once it is dropped.
        (9.1, {}, _drafter_free_discount),
    ],
    ids = [
        "test_metal_auto_drop_releases_the_drafter_charges",
        "test_metal_target_only_bound_leaves_out_the_drafter_compute",
        "test_the_metal_auto_drop_recomputes_the_embedding_discount",
    ],
)
def test_a_metal_auto_drop_matches_a_load_with_speculation_off(
    tmp_path, monkeypatch, budget_gib, kw, discount
):
    def run(mode):
        if discount is not None:
            _embd(monkeypatch, discount)
        return metal_cmd(tmp_path, monkeypatch, mode, budget_gib = budget_gib, **kw)[0]

    auto, off = run("auto"), run("off")
    assert "--model-draft" not in auto and "--model-draft" not in off
    assert h.last(auto, "-c") == h.last(off, "-c")


def test_metal_auto_drop_prices_the_target_with_the_drafter_free_discount(tmp_path, monkeypatch):
    _embd(monkeypatch, _drafter_free_discount, default = True)
    cmd, _backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 7.5)
    assert "--model-draft" not in cmd


def test_a_discount_the_drop_unlocks_is_logged(tmp_path, monkeypatch):
    _embd(monkeypatch, _drafter_free_discount)
    messages = h.info_messages(monkeypatch)
    cmd, _backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 9.1)
    assert "--model-draft" not in cmd
    assert any("of input embeddings stay in the CPU file mapping" in m for m in messages)


def test_the_draft_free_probe_waits_for_a_drafter_that_does_not_fit(tmp_path, monkeypatch):
    """16 GiB holds both, so the draft-free discount (a cold llama-fit-params run) is skipped."""
    calls = _embd(monkeypatch, lambda mtp: 0)
    cmd, _backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 16)
    assert "--model-draft" in cmd
    assert False not in calls


def test_metal_auto_keeps_a_user_owned_bare_model_draft(tmp_path, monkeypatch):
    """A bare user --model-draft still loads (has_dft()), so Auto must keep the reserve."""
    custom = tmp_path / "custom-dspark-Q8_0.gguf"
    custom.write_bytes(b"draft")
    cmd, backend, _ = metal_cmd(
        tmp_path, monkeypatch, "auto", budget_gib = 12, extra_args = ["--model-draft", str(custom)]
    )
    assert str(custom) in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


@pytest.mark.parametrize("pin, reserved", [("MTL0", True), ("mtl0", True), ("Vulkan0", False)])
def test_a_metal_draft_pin_is_reserved_only_on_the_metal_device(
    tmp_path, monkeypatch, pin, reserved
):
    cmd, _backend, _ = metal_cmd(
        tmp_path, monkeypatch, "dspark", budget_gib = 16, extra_args = ["--device-draft", pin]
    )
    assert "--model-draft" in cmd
    assert (h.margin(cmd) >= 1024 + 4 * 1024) is reserved


def test_every_context_gate_prices_the_draft_micro_batch(tmp_path, monkeypatch):
    """The Metal context search takes the draft callback directly, not through the wrapper
    that reads --spec-draft-n-ubatch."""
    seen = []
    real = B._estimate_mtp_overhead_bytes

    def spy(self, n_ctx, *args, **kwargs):
        seen.append(kwargs.get("n_ubatch"))
        return real(self, n_ctx, *args, **kwargs)

    monkeypatch.setattr(B, "_estimate_mtp_overhead_bytes", spy)
    cmd, _backend, _ = metal_cmd(
        tmp_path,
        monkeypatch,
        "dspark",
        budget_gib = 16,
        kv_per_token = 32 * 1024,
        extra_args = ["--spec-draft-n-ubatch", "32"],
    )
    assert "--model-draft" in cmd
    # The first call only asks whether the reserve can be sized at all.
    assert len(seen) > 2 and set(seen[1:]) == {32}, seen


@pytest.mark.parametrize("budget_gib, wired_gib", [(20, 20), (15, 20), (20, 18), (20, 0), (30, 18)])
def test_metal_auto_drops_a_sidecar_an_explicit_context_cannot_hold(
    tmp_path, monkeypatch, budget_gib, wired_gib
):
    """An explicit context launches verbatim: 65,536 tokens at 128 KiB hold the target (Off
    loads), not target + sidecar, so Auto drops the sidecar instead of refusing."""
    kw = dict(n_ctx = 65_536, kv_per_token = 128 * 1024, budget_gib = budget_gib, wired_gib = wired_gib)
    off, _, _ = metal_cmd(tmp_path, monkeypatch, "off", **kw)
    assert h.last(off, "-c") == "65536"
    auto, backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", **kw)
    assert "--model-draft" not in auto
    assert backend.spec_fallback_reason == "drafter_no_vram"
    assert h.last(auto, "-c") == "65536"


def test_metal_auto_keeps_a_sidecar_an_explicit_context_holds(tmp_path, monkeypatch):
    cmd, backend, _ = metal_cmd(
        tmp_path, monkeypatch, "auto", n_ctx = 4096, kv_per_token = 1024, budget_gib = 20
    )
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


def test_metal_auto_keeps_the_sidecar_when_kv_cannot_be_estimated(tmp_path, monkeypatch):
    """With no KV estimate the target alone is unproven to fit (the discrete drop's guard)."""
    cmd, backend, _ = metal_cmd(tmp_path, monkeypatch, "auto", budget_gib = 10, can_estimate_kv = False)
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason != "drafter_no_vram"


def test_an_unmeasured_model_is_reported_once_per_load(monkeypatch):
    import test_metal_explicit_context_guard as metal_guard

    messages = h.info_messages(monkeypatch)
    case = metal_guard.TestWhichLoadsLeaveTheEmbeddingsInTheMapping()
    with lc._load_probe_memo_scope():
        assert case._bytes(monkeypatch, measured = None) == 0
        assert case._bytes(monkeypatch, measured = None) == 0
    assert sum("could not measure how llama.cpp maps" in m for m in messages) == 1
