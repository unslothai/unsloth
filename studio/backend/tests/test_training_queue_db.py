# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from storage import studio_db


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _enqueue(item_id: str, model: str = "unsloth/Qwen3-0.6B") -> dict:
    return studio_db.enqueue_queue_item(
        id = item_id,
        request_json = json.dumps({"model_name": model, "hf_token": "hf_secret"}),
        model_name = model,
        dataset_summary = "test_dataset.jsonl",
        subject = "tester",
    )


def test_enqueue_assigns_increasing_positions():
    a = _enqueue("q1")
    b = _enqueue("q2")
    c = _enqueue("q3")
    assert a["status"] == "pending"
    assert a["position"] < b["position"] < c["position"]


def test_enqueue_stores_via_api_key_flag():
    api_item = studio_db.enqueue_queue_item(
        id = "q_api",
        request_json = "{}",
        model_name = "unsloth/Qwen3-0.6B",
        dataset_summary = "test_dataset.jsonl",
        subject = "api",
        via_api_key = True,
    )
    assert api_item["via_api_key"] == 1
    assert _enqueue("q_ui")["via_api_key"] == 0


def test_next_pending_returns_lowest_position():
    _enqueue("q1")
    _enqueue("q2")
    item = studio_db.next_pending_queue_item()
    assert item is not None
    assert item["id"] == "q1"
    # Runner-internal accessor keeps the frozen request.
    assert "hf_secret" in item["request_json"]


def test_next_pending_skips_non_pending():
    _enqueue("q1")
    _enqueue("q2")
    assert studio_db.update_queue_item_status("q1", "starting", expected_status = "pending")
    item = studio_db.next_pending_queue_item()
    assert item["id"] == "q2"


def test_count_pending():
    assert studio_db.count_pending_queue_items() == 0
    _enqueue("q1")
    _enqueue("q2")
    assert studio_db.count_pending_queue_items() == 2
    studio_db.update_queue_item_status("q1", "running")
    assert studio_db.count_pending_queue_items() == 1


def test_move_swaps_adjacent_pending_neighbors():
    _enqueue("q1")
    _enqueue("q2")
    _enqueue("q3")
    assert studio_db.move_queue_item("q3", "up")
    order = [i["id"] for i in studio_db.list_queue_items(statuses = ("pending",))]
    assert order == ["q1", "q3", "q2"]
    assert studio_db.move_queue_item("q3", "up")
    order = [i["id"] for i in studio_db.list_queue_items(statuses = ("pending",))]
    assert order == ["q3", "q1", "q2"]


def test_move_noop_at_edges():
    _enqueue("q1")
    _enqueue("q2")
    assert not studio_db.move_queue_item("q1", "up")
    assert not studio_db.move_queue_item("q2", "down")
    order = [i["id"] for i in studio_db.list_queue_items(statuses = ("pending",))]
    assert order == ["q1", "q2"]


def test_move_skips_non_pending_neighbor():
    _enqueue("q1")
    _enqueue("q2")
    _enqueue("q3")
    # q2 is running: moving q3 up should hop over it and swap with q1.
    studio_db.update_queue_item_status("q2", "running")
    assert studio_db.move_queue_item("q3", "up")
    pending = [i["id"] for i in studio_db.list_queue_items(statuses = ("pending",))]
    assert pending == ["q3", "q1"]


def test_move_rejects_missing_or_non_pending_item():
    _enqueue("q1")
    assert not studio_db.move_queue_item("missing", "up")
    studio_db.update_queue_item_status("q1", "running")
    assert not studio_db.move_queue_item("q1", "down")
    with pytest.raises(ValueError):
        studio_db.move_queue_item("q1", "sideways")


def test_delete_only_pending():
    _enqueue("q1")
    _enqueue("q2")
    studio_db.update_queue_item_status("q1", "running")
    assert not studio_db.delete_queue_item_if_pending("q1")
    assert studio_db.delete_queue_item_if_pending("q2")
    assert studio_db.get_queue_item("q2") is None
    assert studio_db.get_queue_item("q1") is not None


def test_guarded_transition_rejects_unexpected_status():
    _enqueue("q1")
    assert studio_db.update_queue_item_status("q1", "starting", expected_status = "pending")
    # Second identical transition fails: item is no longer pending.
    assert not studio_db.update_queue_item_status("q1", "starting", expected_status = "pending")
    assert studio_db.get_queue_item("q1")["status"] == "starting"


def test_terminal_transition_can_redact_request_json():
    _enqueue("q1")
    redacted = json.dumps({"model_name": "unsloth/Qwen3-0.6B", "hf_token": "***"})
    assert studio_db.update_queue_item_status(
        "q1",
        "skipped",
        error_message = "dataset missing",
        finished_at = "2026-07-05T00:00:00+00:00",
        request_json = redacted,
    )
    item = studio_db.get_queue_item("q1")
    assert item["status"] == "skipped"
    assert "hf_secret" not in item["request_json"]


def test_list_finished_returns_newest_first():
    for i in range(3):
        _enqueue(f"q{i}")
        studio_db.update_queue_item_status(
            f"q{i}", "skipped", finished_at = f"2026-07-05T00:00:0{i}+00:00"
        )
    finished = studio_db.list_finished_queue_items(limit = 2)
    assert [i["id"] for i in finished] == ["q2", "q1"]


def test_enqueue_enforces_cap_in_insert():
    # The cap is checked inside the insert transaction, so racing enqueues
    # can't both slip past a stale count.
    assert _enqueue("q1") is not None
    assert (
        studio_db.enqueue_queue_item(
            id = "q2",
            request_json = "{}",
            model_name = "unsloth/Qwen3-0.6B",
            dataset_summary = "test_dataset.jsonl",
            subject = "tester",
            max_pending = 1,
        )
        is None
    )
    assert studio_db.get_queue_item("q2") is None
    assert studio_db.count_pending_queue_items() == 1
    # Non-pending items don't count against the cap.
    studio_db.update_queue_item_status("q1", "running")
    assert (
        studio_db.enqueue_queue_item(
            id = "q3",
            request_json = "{}",
            model_name = "unsloth/Qwen3-0.6B",
            dataset_summary = "test_dataset.jsonl",
            subject = "tester",
            max_pending = 1,
        )
        is not None
    )


def test_paused_flag_round_trip():
    assert studio_db.get_queue_paused() == (False, None)
    studio_db.set_queue_paused(True, "restart")
    assert studio_db.get_queue_paused() == (True, "restart")
    studio_db.set_queue_paused(True, "user")
    assert studio_db.get_queue_paused() == (True, "user")
    studio_db.set_queue_paused(False)
    assert studio_db.get_queue_paused() == (False, None)
