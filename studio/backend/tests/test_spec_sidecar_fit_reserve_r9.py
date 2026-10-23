# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ninth-round cells for the DSpark/DFlash sidecar reserve.

A draft pinned to any card of a Studio-chosen Vulkan list is colocated with the target (only
--split-mode none narrows it to the main card), a draft missing either tensor borrows it, and a
draft with its own head keeps its logits graph on the card its own split gives that head.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r2 as _r2  # noqa: E402
import test_spec_sidecar_fit_reserve_r4 as _r4  # noqa: E402
import test_spec_sidecar_fit_reserve_r7 as _r7  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402

_FLOOR = 1024 + 8 * 1024
_TWO = ((0, 24_000, 24_000), (1, 24_000, 24_000))


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)


@pytest.mark.parametrize(
    "extras, pin, reserved",
    [
        ([], "Vulkan1", True),
        ([], "Vulkan0", True),
        (["--split-mode", "none"], "Vulkan1", False),
        (["--split-mode", "none"], "Vulkan0", True),
    ],
)
def test_a_pin_on_any_studio_chosen_vulkan_card_is_on_the_target(tmp_path, extras, pin, reserved):
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO, vulkan = True)
    cmd = _r2._placement._launch(
        backend,
        gguf,
        speculative_type = "dspark",
        dspark_draft_path = _r4._sidecar(tmp_path),
        gpu_ids = [0, 1],
        extra_args = [*extras, "--device-draft", pin],
    )["cmd"]
    assert (_r4._margin(cmd) >= _FLOOR) is reserved


def _draft(monkeypatch, tmp_path, *, tensors):
    path = tmp_path / "draft.gguf"
    path.write_bytes(b"draft")
    import utils.models.gguf_metadata as gm
    import utils.models.model_config as mc

    monkeypatch.setattr(mc, "colocated_split_shards", lambda p: ([p], True))
    monkeypatch.setattr(gm, "_gguf_shard_has_named_tensor", lambda shard, name: name in tensors)
    return str(path)


@pytest.mark.parametrize(
    "tensors, borrows, borrows_output",
    [
        # Upstream DFlash borrows the target's head when a draft ships none, and no marker
        # tells a build that ties it to the draft's embeddings apart, so it stays borrowed.
        ({"token_embd.weight"}, True, True),
        ({"output.weight"}, True, False),
        (set(), True, True),
        ({"token_embd.weight", "output.weight"}, False, False),
    ],
)
def test_a_draft_missing_a_tensor_borrows_it(
    monkeypatch, tmp_path, tensors, borrows, borrows_output
):
    path = _draft(monkeypatch, tmp_path, tensors = tensors)
    assert _lc._sidecar_borrows_target_tensors(path) is borrows
    assert _lc._sidecar_borrows_output(path) is borrows_output


def test_an_unread_draft_borrows_the_output():
    assert _lc._sidecar_borrows_output(None) is True


def test_a_draft_with_its_own_head_keeps_its_graph_on_its_own_output_card(tmp_path, monkeypatch):
    """A 3:2 split puts all three draft blocks on card 0 and the draft's own output layer
    (3 / 4 = 0.75) on card 1. Borrowing the one-block target's head would put the graph on
    card 0 (r7); owning it moves the graph to card 1."""
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(_lc, "_sidecar_borrows_output", lambda path: False)
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    backend._mtp_draft_compute_bytes = lambda *a, **k: 3 * 1024**3
    _r7._three_block_drafter(backend)
    backend._n_layers = 1
    used, out = _r7._margins(
        _r7._dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"])
    )
    assert out >= 1024 + 3 * 1024


@pytest.mark.parametrize("order, per_card", [("FASTEST_FIRST", False), ("PCI_BUS_ID", True)])
def test_per_card_shares_need_a_provable_device_order(tmp_path, monkeypatch, order, per_card):
    """nvidia-smi keys free memory by PCI order; under FASTEST_FIRST the child enumerates its
    own way, so a positional list could put the large card's share on the small one."""
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    B = _lc.LlamaCppBackend
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    backend, gguf = _r2._cuda_backend(tmp_path, memory = ((0, 80_000, 80_000), (1, 8_000, 8_000)))
    backend._arch_gate_survivors = lambda binary = None: []
    margins = _r7._margins(_r7._dspark(backend, gguf, tmp_path))
    assert (len(margins) == 2) is per_card


def _owned_head_margins(tmp_path, monkeypatch, head_gib):
    from types import SimpleNamespace

    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(_lc, "_sidecar_borrows_output", lambda path: False)
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    _r7._three_block_drafter(backend)
    backend._n_layers = 1
    read = backend._tensor_spill_layout

    def layout(path, **kw):
        if "dspark" in str(path):
            return SimpleNamespace(complete = True, lm_head_bytes = int(head_gib * 1024**3))
        return read(path, **kw)

    backend._tensor_spill_layout = layout
    return _r7._margins(_r7._dspark(backend, gguf, tmp_path, extra_args = ["--tensor-split", "3,2"]))


def test_an_owned_head_is_reserved_on_its_output_card(tmp_path, monkeypatch):
    """A 3:2 split puts the three draft blocks on card 0 and the draft's own output layer on
    card 1, so its head's bytes move there; the total stays the same."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    no_head = _owned_head_margins(tmp_path / "a", monkeypatch, 0)
    head = _owned_head_margins(tmp_path / "b", monkeypatch, 2)
    assert head[1] - no_head[1] >= 2000
    assert no_head[0] - head[0] >= 2000


@pytest.mark.parametrize(
    "embedding, extras, endpoint",
    [
        (False, (), "/completion"),
        (True, (), "/v1/embeddings"),
        (False, ("--embeddings",), "/v1/embeddings"),
    ],
)
def test_an_embedding_server_is_probed_through_embeddings(monkeypatch, embedding, extras, endpoint):
    """--embedding restricts llama-server to embeddings, so a /completion probe would never
    reach the poisoned compute path and the child would never be retired."""
    from unittest.mock import patch

    backend = _lc.LlamaCppBackend()
    backend._port = 1
    backend._extra_args = list(extras)
    monkeypatch.setattr(_lc.LlamaCppBackend, "is_embedding_gguf", property(lambda self: embedding))
    resp = type(
        "Resp",
        (),
        {"status_code": 500, "json": lambda self: {"error": {"message": "Compute error."}}},
    )()
    with patch.object(_lc.httpx, "post", return_value = resp, create = True) as post:
        assert backend._metal_compute_failure_confirmed() is True
    assert post.call_args.args[0].endswith(endpoint)


@pytest.mark.parametrize(
    "args, env, want",
    [
        # No -c: the pool is n_parallel * per-slot limit (tools/server/server.cpp).
        (["--kv-unified-per-slot", "2048"], {}, 4 * 2048),
        (["--kv-unified-per-slot=65536"], {}, 4 * 65536),
        ([], {"LLAMA_ARG_KV_UNIFIED_PER_SLOT": "2048"}, 4 * 2048),
        # An explicit -c wins.
        (["-c", "16384", "--kv-unified-per-slot", "2048"], {}, 16384),
        # -c 0 asks for the full window; a later --fit-ctx clears that.
        (["-c", "0", "--kv-unified-per-slot", "2048"], {}, 131072),
        (["-c", "0", "--fit-ctx", "8192", "--kv-unified-per-slot", "2048"], {}, 4 * 2048),
        ([], {}, 131072),
    ],
)
def test_the_per_slot_limit_sizes_the_priced_context(args, env, want):
    assert _lc._sidecar_child_ctx(args, env, 131072, 4, True) == want


@pytest.mark.parametrize(
    "args, want",
    [
        (["--split_mode=tensor"], "tensor"),
        (["--split_mode", "row"], "row"),
        (["-sm", "layer"], "layer"),
    ],
)
def test_the_underscore_split_mode_spelling_is_read(args, want):
    assert _lc._extra_args_split_mode(args, {}) == want


def test_one_card_keeps_its_own_emitted_base_under_the_reserve():
    """A lowered budget sizes the host-wide scalar by the largest card; a child masked to a
    smaller one carries that card's own margin, which the reserve sits on."""
    flags = _lc._sidecar_reserve_margin_flags(
        ["--model-draft", "/d.gguf", "--fit", "on"],
        [],
        {},
        4000,
        5000,
        base_by_device = [1000],
    )
    assert flags == ["--fit-target", "6000"]


@pytest.mark.parametrize(
    "extras, caps_fit_target, fit_sized",
    [
        ([], True, True),
        (["--rpc", "10.0.0.2:50052"], True, False),
        (["--device", "CUDA0,CUDA1"], True, False),
        ([], False, False),
    ],
)
def test_fit_sized_is_reported_only_where_the_spawn_can_price(
    tmp_path, monkeypatch, extras, caps_fit_target, fit_sized
):
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    caps = dict(_r2._CAPS, supports_fit_target = caps_fit_target)
    backend.probe_server_capabilities = lambda _binary = None: dict(caps)
    seen = {}
    build = backend._build_speculative_flags

    def spy(**kw):
        seen["fit_sized"] = kw.get("dspark_fit_sized")
        return build(**kw)

    backend._build_speculative_flags = spy
    _r7._dspark(backend, gguf, tmp_path, extra_args = extras)
    assert seen["fit_sized"] is fit_sized


def test_planning_drops_an_inherited_split_mode_the_launch_clears(tmp_path, monkeypatch):
    """A layer launch clears an inherited non-layer LLAMA_ARG_SPLIT_MODE before spawning, so
    planning must not narrow the target to the main card on it: the draft pinned to CUDA1
    shares that card with the target and needs the reserve there."""
    monkeypatch.setattr(_lc, "_metal_capable_host", lambda: False)
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0,CUDA1")
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "none")
    backend, gguf = _r2._cuda_backend(tmp_path, memory = _TWO)
    backend._arch_gate_survivors = lambda binary = None: []
    cmd = _r7._dspark(backend, gguf, tmp_path, extra_args = ["--device-draft", "CUDA1"])
    assert max(_r7._margins(cmd)) >= _FLOOR


# Only a common library the server actually links speaks for its fitter.

_MARKER = b"measuring the extra model next to an allocation-free target context"


def _elf(path, payload = b""):
    path.write_bytes(b"\x7fELF" + payload + b"\x00")
    return path


def test_a_static_server_beside_a_stale_common_library_keeps_the_reserve(tmp_path):
    binary = _elf(tmp_path / "llama-server")
    _elf(tmp_path / "libllama-common.so.0", _MARKER)
    assert not _lc._fit_measures_borrowed_drafts(str(binary))


def test_the_common_library_is_followed_through_the_server_impl_library(tmp_path):
    """b11734-style layout: llama-server links libllama-server-impl.so, which links
    libllama-common.so.0, where fit.cpp lives."""
    binary = _elf(tmp_path / "llama-server", b"libllama-server-impl.so")
    _elf(tmp_path / "libllama-server-impl.so", b"libllama-common.so.0")
    _elf(tmp_path / "libllama-common.so.0", _MARKER)
    assert _lc._fit_measures_borrowed_drafts(str(binary))
    # An impl library that does not link it leaves a stale sibling unread.
    _elf(tmp_path / "libllama-server-impl.so")
    assert not _lc._fit_measures_borrowed_drafts(str(binary))


@pytest.mark.parametrize(
    "linked",
    ["libllama-common.so.0", "libllama-common.0.dylib", "llama-common.dll"],
)
def test_a_server_that_names_its_common_library_is_read_through_it(tmp_path, linked):
    binary = _elf(tmp_path / "llama-server", linked.encode())
    _elf(tmp_path / linked, _MARKER)
    assert _lc._fit_measures_borrowed_drafts(str(binary))
