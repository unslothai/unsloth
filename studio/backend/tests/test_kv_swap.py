# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for KV-cache checkpoint swapping (``core.inference.kv_swap``).

The numbers asserted here are the ones measured against llama-server b10715 and written
down in ``plans/impl_v2_slot_swap.md``; the header layout and the resume rule in
particular are properties of that build, not guesses.
"""

from __future__ import annotations

import struct

import pytest

from core.inference.kv_swap import (
    DEFAULT_DRAFT_MARGIN,
    PAUSED,
    RUNNING,
    SLOT_FILE_MAGIC,
    SWAP_STREAK_PROTECT,
    KvSwapController,
    KvSwapError,
    default_buffer_tokens,
    get_kv_swap_controller,
    kv_swap_chunk_tokens,
    kv_swap_enabled,
    kv_swap_force_every,
    parse_slot_file_tokens,
    reset_kv_swap_controllers,
    resume_prompt_is_valid,
)


# --------------------------------------------------------------------------- helpers

def write_slot_file(path, tokens, *, magic = SLOT_FILE_MAGIC, version = 3, count = None,
                    truncate_tokens = False):
    """Write a slot file in llama-server's ``ggsq`` layout."""
    count = len(tokens) if count is None else count
    blob = struct.pack("<IIIiI I", magic, version, len(tokens) + 4, -1, 1, count)
    body = struct.pack(f"<{len(tokens)}i", *tokens) if tokens else b""
    if truncate_tokens:
        body = body[:len(body) // 2]
    path.write_bytes(blob + body + b"\x00" * 64)
    return path


class FakeServer:
    """Stands in for llama-server's slot state endpoints."""

    def __init__(self):
        self.calls = []
        self.saved = {}
        self.fail_on = set()
        self.save_returns = None

    def post(self, *, slot, action, body):
        self.calls.append((slot, action, dict(body)))
        if action in self.fail_on:
            raise RuntimeError(f"{action} failed")
        if action == "save":
            name = body.get("filename")
            n = self.save_returns if self.save_returns is not None else 140
            self.saved[name] = n
            return {"id_slot": slot, "filename": name, "n_saved": n}
        if action == "erase":
            return {"id_slot": slot, "n_erased": 140}
        if action == "restore":
            name = body.get("filename")
            return {"id_slot": slot, "filename": name, "n_restored": self.saved.get(name, 0)}
        raise AssertionError(action)


def make_controller(server = None, **kwargs):
    params = dict(n_ctx = 8192, n_parallel = 4, draft_n_max = 2, save_dir = None)
    params.update(kwargs)
    return KvSwapController("http://test", http_post = (server or FakeServer()).post, **params)


# ----------------------------------------------------------------- the file header

def test_parse_slot_file_returns_exact_tokens(tmp_path):
    tokens = [248045, 846, 198, 814, 20139]
    path = write_slot_file(tmp_path / "ok.bin", tokens)
    assert parse_slot_file_tokens(path) == tokens


def test_parse_slot_file_rejects_bad_magic(tmp_path):
    path = write_slot_file(tmp_path / "bad.bin", [1, 2, 3], magic = 0xDEADBEEF)
    with pytest.raises(KvSwapError, match = "magic"):
        parse_slot_file_tokens(path)


def test_parse_slot_file_rejects_unknown_version(tmp_path):
    path = write_slot_file(tmp_path / "v9.bin", [1, 2, 3], version = 9)
    with pytest.raises(KvSwapError, match = "version"):
        parse_slot_file_tokens(path)


def test_parse_slot_file_rejects_truncated_token_block(tmp_path):
    path = tmp_path / "short.bin"
    path.write_bytes(struct.pack("<IIIiII", SLOT_FILE_MAGIC, 3, 104, -1, 1, 100) + b"\x00" * 8)
    with pytest.raises(KvSwapError, match = "of 100 tokens"):
        parse_slot_file_tokens(path)


def test_parse_slot_file_rejects_short_header(tmp_path):
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"\x00" * 8)
    with pytest.raises(KvSwapError, match = "truncated"):
        parse_slot_file_tokens(path)


def test_parse_slot_file_rejects_missing_file(tmp_path):
    with pytest.raises(KvSwapError, match = "could not be read"):
        parse_slot_file_tokens(tmp_path / "nope.bin")


# ------------------------------------------------------------------- the resume rule

def test_resume_prompt_must_strictly_extend_the_save():
    saved = [1, 2, 3, 4]
    # A proper extension is the only shape that lands on the restored cells.
    assert resume_prompt_is_valid(saved, [1, 2, 3, 4, 5]) is True
    # Equal length re-prefills on the real server, so it is not "valid" here either.
    assert resume_prompt_is_valid(saved, [1, 2, 3, 4]) is False
    assert resume_prompt_is_valid(saved, [1, 2, 3]) is False
    assert resume_prompt_is_valid(saved, [1, 2, 9, 4, 5]) is False
    assert resume_prompt_is_valid([], [1, 2]) is False


# ------------------------------------------------------------------------- the buffer

def test_default_buffer_covers_the_speculative_lead_on_every_slot():
    assert default_buffer_tokens(4, 2) == 4 * (2 + DEFAULT_DRAFT_MARGIN) == 20
    assert default_buffer_tokens(1, 0) == DEFAULT_DRAFT_MARGIN
    assert default_buffer_tokens(0, 0) == DEFAULT_DRAFT_MARGIN


def test_buffer_env_override(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP_BUFFER", "64")
    assert default_buffer_tokens(4, 2) == 64
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP_BUFFER", "0")
    assert default_buffer_tokens(4, 2) == 0


def test_budget_is_context_minus_buffer():
    controller = make_controller()
    assert controller.buffer_tokens == 20
    assert controller.budget == 8192 - 20


# --------------------------------------------------------------------------- env knobs

def test_env_knobs(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_KV_SWAP", raising = False)
    assert kv_swap_enabled() is True
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP", "0")
    assert kv_swap_enabled() is False
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP", "nonsense")
    assert kv_swap_enabled() is True

    monkeypatch.delenv("UNSLOTH_LLAMA_KV_SWAP_CHUNK", raising = False)
    assert kv_swap_chunk_tokens() == 512
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP_CHUNK", "1024")
    assert kv_swap_chunk_tokens() == 1024

    monkeypatch.delenv("UNSLOTH_LLAMA_KV_SWAP_EVERY", raising = False)
    assert kv_swap_force_every() == 0
    monkeypatch.setenv("UNSLOTH_LLAMA_KV_SWAP_EVERY", "200")
    assert kv_swap_force_every() == 200


# --------------------------------------------------------------------------- accounting

def test_resident_counts_prompt_plus_generated_and_drops_when_paused():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 0)
    controller.update("a", generated_tokens = 50)
    chat = controller.get("a")
    assert chat.resident == 150
    assert controller.resident_total() == 150
    chat.state = PAUSED
    assert chat.resident == 0
    assert chat.size == 150
    assert controller.resident_total() == 0


def test_finish_removes_the_chat():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 10, slot = 0)
    controller.finish("a")
    assert controller.get("a") is None
    assert controller.resident_total() == 0


def test_reconcile_trusts_the_server_over_the_stream():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 1)
    controller.update("a", generated_tokens = 111)
    # The server holds 4 more than the stream emitted: the in-flight speculative batch.
    controller.reconcile([{"id": 1, "n_prompt_tokens": 215}])
    assert controller.get("a").resident == 215
    assert controller.get("a").generated_tokens == 115


def test_reconcile_ignores_slots_it_does_not_own():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 1)
    controller.reconcile([{"id": 3, "n_prompt_tokens": 900}, {"id": 1, "n_prompt_tokens": 0}])
    assert controller.get("a").resident == 100


def test_active_count_arms_chunking_only_above_one():
    controller = make_controller()
    assert controller.active_count() == 0
    controller.admit("a", prompt_tokens = 10, slot = 0)
    assert controller.active_count() == 1
    controller.admit("b", prompt_tokens = 10, slot = 1)
    assert controller.active_count() == 2


# ------------------------------------------------------------------------------ policy

def _four_chats(controller, sizes):
    for i, (name, size) in enumerate(sizes):
        controller.admit(name, prompt_tokens = size, slot = i)


def test_plan_does_nothing_while_it_fits():
    controller = make_controller(n_ctx = 8192)
    _four_chats(controller, [("a", 100), ("b", 200)])
    decision = controller.plan()
    assert decision.victims == []
    assert decision.reason == "fits"
    assert decision.keep == "b"


def test_plan_keeps_the_largest_and_swaps_newest_first():
    controller = make_controller(n_ctx = 1000)   # budget 980
    for i, (name, size) in enumerate([("a", 500), ("b", 300), ("c", 300)]):
        controller.admit(name, prompt_tokens = size, slot = i)
        controller.get(name).admitted_at = 100.0 + i   # a oldest, c newest
    decision = controller.plan()
    assert decision.keep == "a"                 # most tokens
    assert decision.victims == ["c"]            # newest first, and one is enough
    assert decision.reason == "pressure"


def test_plan_never_swaps_the_last_one_standing():
    controller = make_controller(n_ctx = 100)   # budget 80, nothing fits
    controller.admit("a", prompt_tokens = 500, slot = 0)
    controller.get("a").admitted_at = 1.0
    controller.admit("b", prompt_tokens = 400, slot = 1)
    controller.get("b").admitted_at = 2.0
    decision = controller.plan()
    assert decision.victims == ["b"]
    assert decision.keep == "a"
    # And with a single chat there is nobody to swap at all.
    controller.finish("b")
    assert controller.plan().victims == []
    assert controller.plan().reason == "single-chat"


def test_plan_protects_a_chat_on_a_swap_streak_while_another_candidate_exists():
    controller = make_controller(n_ctx = 1000)  # budget 980
    for i, (name, size) in enumerate([("keep", 500), ("streak", 300), ("fresh", 300)]):
        controller.admit(name, prompt_tokens = size, slot = i)
        controller.get(name).admitted_at = 100.0 + i
    # "streak" is the newest-but-one; make it the one that has been hit repeatedly.
    controller.get("streak").swap_streak = SWAP_STREAK_PROTECT
    controller.get("streak").admitted_at = 200.0   # newest, so normally chosen first
    decision = controller.plan()
    assert decision.victims == ["fresh"]           # protection pushed "streak" behind it


def test_plan_uses_a_protected_chat_when_it_is_the_only_candidate():
    controller = make_controller(n_ctx = 1000)
    controller.admit("keep", prompt_tokens = 700, slot = 0)
    controller.get("keep").admitted_at = 1.0
    controller.admit("streak", prompt_tokens = 400, slot = 1)
    controller.get("streak").admitted_at = 2.0
    controller.get("streak").swap_streak = SWAP_STREAK_PROTECT + 5
    assert controller.plan().victims == ["streak"]


def test_plan_accounts_for_an_incoming_chat():
    controller = make_controller(n_ctx = 1000)   # budget 980
    controller.admit("a", prompt_tokens = 500, slot = 0)
    controller.get("a").admitted_at = 1.0
    controller.admit("b", prompt_tokens = 400, slot = 1)
    controller.get("b").admitted_at = 2.0
    assert controller.plan().victims == []       # 900 fits
    assert controller.plan(incoming = 300).victims == ["b"]


def test_plan_swaps_several_when_one_is_not_enough():
    controller = make_controller(n_ctx = 1000)
    for i, name in enumerate(["a", "b", "c", "d"]):
        controller.admit(name, prompt_tokens = 400 if name == "a" else 300, slot = i)
        controller.get(name).admitted_at = 100.0 + i
    decision = controller.plan()
    assert decision.keep == "a"
    assert decision.victims == ["d", "c"]        # newest first until it fits


# ------------------------------------------------------------------------- swap moves

def test_swap_out_saves_then_erases_and_frees_the_slot():
    server = FakeServer()
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 2)
    controller.update("a", generated_tokens = 40)
    chat = controller.swap_out("a")
    assert [c[1] for c in server.calls] == ["save", "erase"]
    assert server.calls[0][0] == 2 and server.calls[1][0] == 2
    assert chat.state == PAUSED
    assert chat.slot is None
    assert chat.saved_tokens == 140
    assert chat.swap_streak == 1 and chat.swaps_total == 1
    assert controller.resident_total() == 0      # the cells are free
    assert controller.swaps_out == 1


def test_swap_out_does_not_erase_when_the_save_reports_nothing():
    server = FakeServer()
    server.save_returns = 0
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 1)
    with pytest.raises(KvSwapError, match = "saved 0 tokens"):
        controller.swap_out("a")
    assert [c[1] for c in server.calls] == ["save"]     # no erase: content is never lost
    assert controller.get("a").state == RUNNING


def test_swap_out_propagates_a_failed_save_without_erasing():
    server = FakeServer()
    server.fail_on.add("save")
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 1)
    with pytest.raises(RuntimeError):
        controller.swap_out("a")
    assert [c[1] for c in server.calls] == ["save"]
    assert controller.get("a").state == RUNNING


def test_swap_out_requires_a_pinned_slot():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 10, slot = None)
    with pytest.raises(KvSwapError, match = "not pinned"):
        controller.swap_out("a")


def test_swap_in_restores_into_any_free_slot():
    server = FakeServer()
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 2)
    controller.swap_out("a")
    chat = controller.swap_in("a", 0)            # a different id than it left
    assert server.calls[-1][0] == 0
    assert server.calls[-1][1] == "restore"
    assert chat.state == RUNNING and chat.slot == 0
    assert controller.swaps_in == 1


def test_swap_in_rejects_a_short_restore_and_stays_paused():
    server = FakeServer()
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 2)
    controller.swap_out("a")
    server.saved[controller.get("a").filename] = 3     # server came back with fewer cells
    with pytest.raises(KvSwapError, match = "restored 3 of 140"):
        controller.swap_in("a", 0)
    assert controller.get("a").state == PAUSED         # still recoverable via fallback


def test_swap_in_keeps_the_checkpoint_when_the_call_raises():
    server = FakeServer()
    controller = make_controller(server)
    controller.admit("a", prompt_tokens = 100, slot = 2)
    controller.swap_out("a")
    server.fail_on.add("restore")
    with pytest.raises(RuntimeError):
        controller.swap_in("a", 0)
    assert controller.get("a").state == PAUSED
    assert controller.get("a").filename is not None


def test_swap_in_requires_a_checkpoint():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 10, slot = 0)
    with pytest.raises(KvSwapError, match = "no checkpoint"):
        controller.swap_in("a", 1)


def test_controller_without_transport_raises_rather_than_silently_skipping():
    controller = KvSwapController("k", n_ctx = 100, n_parallel = 1)
    controller.admit("a", prompt_tokens = 10, slot = 0)
    with pytest.raises(KvSwapError, match = "no HTTP transport"):
        controller.swap_out("a")


# ---------------------------------------------------------------------------- fallback

def test_fall_back_drops_the_checkpoint_and_unlinks_the_file(tmp_path):
    server = FakeServer()
    controller = make_controller(server, save_dir = str(tmp_path))
    controller.admit("a", prompt_tokens = 100, slot = 1)
    controller.swap_out("a")
    name = controller.get("a").filename
    (tmp_path / name).write_bytes(b"x")
    controller.fall_back("a")
    assert controller.get("a").filename is None
    assert not (tmp_path / name).exists()
    assert controller.fallbacks == 1


def test_fall_back_on_an_unknown_chat_is_counted_and_harmless():
    controller = make_controller()
    controller.fall_back("ghost")
    assert controller.fallbacks == 1


def test_sweep_unlinks_every_checkpoint(tmp_path):
    server = FakeServer()
    controller = make_controller(server, save_dir = str(tmp_path))
    for i, name in enumerate(["a", "b"]):
        controller.admit(name, prompt_tokens = 100, slot = i)
        controller.swap_out(name)
        (tmp_path / controller.get(name).filename).write_bytes(b"x")
    assert controller.sweep() == 2
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------- progress + streaks

def test_a_full_chunk_without_a_swap_clears_the_streak():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 0)
    controller.get("a").swap_streak = 2
    controller.note_progress("a")
    assert controller.get("a").swap_streak == 0


def test_note_resume_records_the_prefill_the_resume_cost():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 0)
    controller.note_resume("a", 1)
    assert controller.get("a").last_resume_prompt_n == 1


# ---------------------------------------------------------------------------- registry

def test_registry_reuses_and_updates(monkeypatch):
    reset_kv_swap_controllers()
    first = get_kv_swap_controller("http://a", n_ctx = 4096, n_parallel = 2)
    again = get_kv_swap_controller("http://a", n_ctx = 8192, n_parallel = 4)
    assert first is again
    assert again.n_ctx == 8192 and again.n_parallel == 4
    other = get_kv_swap_controller("http://b", n_ctx = 4096, n_parallel = 2)
    assert other is not first
    reset_kv_swap_controllers()


def test_snapshot_reports_the_live_shape():
    controller = make_controller()
    controller.admit("a", prompt_tokens = 100, slot = 0)
    controller.admit("b", prompt_tokens = 200, slot = 1)
    snap = controller.snapshot()
    assert snap["running"] == 2
    assert snap["resident"] == 300
    assert snap["budget"] == 8192 - 20
    assert snap["swaps_out"] == 0
