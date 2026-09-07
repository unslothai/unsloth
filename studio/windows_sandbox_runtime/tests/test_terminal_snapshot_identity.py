# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Durable copied-Terminal identity ownership, not launch qualification."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import secrets
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from core.inference.windows_sandbox import terminal_native as lpac
from core.inference.windows_sandbox import identity
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason = "Windows identity journal path and profile contracts"
)


@pytest.fixture
def schema():
    recipe = identity.InvocationRecipe("unsloth.studio." + "a" * 32, 123, 456)
    reader = identity.RuntimeReaderRecipe("C:/runtime-store", "b" * 32, "c" * 64)
    value = identity._payload(
        recipe,
        "S-1-15-2-unit",
        "creating",
        "C:/work",
        reader = reader,
        terminal_snapshot = True,
    )
    return recipe, reader, value


def test_v6_schema_is_closed_and_returns_mandatory_reader(schema):
    recipe, reader, value = schema
    assert value == {
        "version": 6,
        "state": "creating",
        "moniker": recipe.moniker,
        "owner_pid": recipe.owner_pid,
        "owner_created": recipe.owner_created,
        "sid": "S-1-15-2-unit",
        "workdir": "C:/work",
        "profile_folder": None,
        "reader": asdict(reader),
        "purpose": "terminal",
    }
    assert "runtime_roots" not in value
    assert identity._validate(value, recipe, value["sid"]) == reader


@pytest.mark.parametrize(
    "change",
    ["version", "purpose", "missing_reader", "null_reader", "changed_reader", "runtime_roots"],
)
def test_v6_schema_rejects_version_purpose_reader_and_extra_fields(schema, change):
    recipe, _reader, original = schema
    unchanged = deepcopy(original)
    value = deepcopy(original)
    if change == "version":
        value["version"] = 5
    elif change == "purpose":
        value["purpose"] = "qualification"
    elif change == "missing_reader":
        value.pop("reader")
    elif change == "null_reader":
        value["reader"] = None
    elif change == "changed_reader":
        value["reader"]["digest"] = "not-a-digest"
    else:
        value["runtime_roots"] = ["C:/Git/bin"]

    with pytest.raises(WindowsRuntimeError):
        identity._validate(value, recipe, original["sid"])
    assert original == unchanged


def test_snapshot_variant_rejects_private_roots_and_missing_reader(schema):
    recipe, reader, _value = schema
    common = (recipe, "S-1-15-2-unit", "creating", "C:/work")
    with pytest.raises(WindowsRuntimeError, match = "only its workdir and durable reader"):
        identity._payload(*common, terminal_snapshot = True)
    with pytest.raises(WindowsRuntimeError, match = "only its workdir and durable reader"):
        identity._payload(*common, reader = reader, private_workdir = True, terminal_snapshot = True)
    with pytest.raises(WindowsRuntimeError, match = "only its workdir and durable reader"):
        identity._payload(
            *common,
            reader = reader,
            terminal_roots = (),
            terminal_snapshot = True,
        )

    reservation = identity.InvocationReservation(recipe)
    with pytest.raises(WindowsRuntimeError, match = "only its workdir and durable reader"):
        reservation.create_terminal_snapshot("C:/work", reader = None)
    assert reservation.started is reservation.owned is False


def test_prior_journal_versions_keep_their_existing_shapes(schema):
    recipe, reader, _value = schema
    common = (recipe, "S-1-15-2-unit", "creating", "C:/work")
    assert identity._payload(*common)["version"] == 2
    assert identity._payload(*common, reader = reader)["version"] == 3
    private = identity._payload(
        recipe,
        common[1],
        "creating",
        None,
        reader = reader,
        private_workdir = True,
    )
    assert private["version"] == 4 and private["purpose"] == "qualification"
    terminal = identity._payload(*common, terminal_roots = ("C:/Git/bin",))
    assert terminal["version"] == 5 and terminal["runtime_roots"] == ["C:/Git/bin"]


def test_native_v6_intent_precedes_profile_and_reader_recovery_precedes_cleanup(
    tmp_path, monkeypatch
):
    base = tmp_path / "journals"
    workdir = tmp_path / "work"
    base.mkdir()
    workdir.mkdir()
    monkeypatch.setattr(lpac, "_manifest_root", lambda: str(base))
    store = RuntimeContentStore(tmp_path / "terminal-store")
    # This synthetic generation has no live read lease. The test proves only
    # durable journal/profile recovery ordering, not snapshot-reader ACLs.
    reader = identity.RuntimeReaderRecipe(
        str(store.root), secrets.token_hex(16), secrets.token_hex(32)
    )
    reservation = identity.InvocationReservation(identity.InvocationRecipe.new())
    api = lpac._api().userenv
    original_create = api.CreateAppContainerProfile
    creating = []

    def create(*args):
        record = identity._read(reservation.path)
        assert record["version"] == 6 and record["state"] == "creating"
        assert record["purpose"] == "terminal" and record["reader"] == asdict(reader)
        assert "runtime_roots" not in record
        assert identity._profile_path(record["sid"]) is None
        creating.append(record)
        return original_create(*args)

    monkeypatch.setattr(api, "CreateAppContainerProfile", create)
    try:
        actual = reservation.create_terminal_snapshot(workdir, reader = reader)
        ready = identity._read(reservation.path)
        assert len(creating) == 1 and ready["version"] == 6 and ready["state"] == "ready"
        assert ready["reader"] == asdict(reader) and "runtime_roots" not in ready

        events = []
        original_recover = identity._recover_runtime_reader
        original_cleanup = type(actual).cleanup

        def recover(*args):
            events.append("reader")
            return original_recover(*args)

        def cleanup(owner):
            events.append("profile")
            return original_cleanup(owner)

        monkeypatch.setattr(identity, "_recover_runtime_reader", recover)
        monkeypatch.setattr(type(actual), "cleanup", cleanup)
        reservation.cleanup()

        assert events == ["reader", "profile"]
        assert actual.cleaned and not reservation.path.exists()
        assert identity._profile_path(actual.sid_string) is None
    finally:
        reservation.cleanup()
