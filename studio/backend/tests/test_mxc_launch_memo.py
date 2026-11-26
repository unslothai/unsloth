# SPDX-License-Identifier: AGPL-3.0-only

"""One tool launch asks for the MXC pin and the grant deny-lists many times: each is read once per launch, never across."""

from __future__ import annotations

import json

import pytest

from core.inference import mxc_runtime, os_sandbox
from utils.scan_memo import scan_memo


@pytest.fixture
def windows_pin_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv(mxc_runtime.PIN_ENV, raising = False)
    monkeypatch.setattr(mxc_runtime.sys, "platform", "win32")
    release = (1, 26100, 1)
    monkeypatch.setattr(mxc_runtime, "_windows_release", lambda: release)
    monkeypatch.setattr(mxc_runtime, "builtin_container_supported", lambda _release: True)
    reads = {"marker": 0, "legacy": 0}
    real_demoted = mxc_runtime.demoted_by_probe

    def demoted(release = None):
        reads["marker"] += 1
        return real_demoted(release)

    def legacy():
        reads["legacy"] += 1
        return False

    monkeypatch.setattr(mxc_runtime, "demoted_by_probe", demoted)
    monkeypatch.setattr(mxc_runtime, "_legacy_still_installed", legacy)

    def demote():
        path = mxc_runtime._demoted_marker_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(
            json.dumps(
                {
                    "windows": mxc_runtime._release_label(release),
                    "from": mxc_runtime.PRIMARY_PIN.version,
                }
            ),
            encoding = "utf-8",
        )

    return reads, demote


def test_one_launch_reads_the_pin_inputs_once(windows_pin_inputs):
    reads, _demote = windows_pin_inputs
    with scan_memo():
        pins = {mxc_runtime.selected_pin() for _ in range(30)}
    assert pins == {mxc_runtime.PRIMARY_PIN}
    assert reads == {"marker": 1, "legacy": 1}


def test_each_launch_rereads_the_pin_so_a_new_marker_applies_to_the_next(windows_pin_inputs):
    reads, demote = windows_pin_inputs
    with scan_memo():
        assert mxc_runtime.selected_pin() is mxc_runtime.PRIMARY_PIN
    demote()
    with scan_memo():
        assert mxc_runtime.selected_pin() is mxc_runtime.LEGACY_PIN
        assert mxc_runtime.selected_pin() is mxc_runtime.LEGACY_PIN
    assert reads["marker"] == 2


def test_outside_a_launch_every_ask_is_live(windows_pin_inputs):
    # acquire_runtime and the spawn-time fingerprint run outside the scope: they always pick afresh.
    reads, demote = windows_pin_inputs
    assert mxc_runtime.selected_pin() is mxc_runtime.PRIMARY_PIN
    demote()
    assert mxc_runtime.selected_pin() is mxc_runtime.LEGACY_PIN
    assert reads["marker"] == 2


def test_a_forced_pin_changed_inside_a_launch_is_honoured(windows_pin_inputs, monkeypatch):
    with scan_memo():
        assert mxc_runtime.selected_pin() is mxc_runtime.PRIMARY_PIN
        monkeypatch.setenv(mxc_runtime.PIN_ENV, mxc_runtime.LEGACY_PIN.version)
        assert mxc_runtime.selected_pin() is mxc_runtime.LEGACY_PIN


@pytest.mark.parametrize(
    "name",
    ["_hf_home_roots_uncached", "_never_a_model_library_uncached", "_studio_state_roots_uncached"],
)
def test_grant_deny_lists_are_computed_once_per_launch_and_again_next_launch(monkeypatch, name):
    public = {
        "_hf_home_roots_uncached": os_sandbox._hf_home_roots,
        "_never_a_model_library_uncached": os_sandbox._never_a_model_library,
        "_studio_state_roots_uncached": os_sandbox.studio_state_roots,
    }[name]
    real = getattr(os_sandbox, name)
    calls = []

    def counted():
        calls.append(1)
        return real()

    monkeypatch.setattr(os_sandbox, name, counted)
    with scan_memo():
        first = {public() for _ in range(5)}
    assert len(calls) == 1 and len(first) == 1
    with scan_memo():
        public()
    assert len(calls) == 2
    public()
    public()
    assert len(calls) == 4


def test_a_studio_home_changed_inside_a_launch_is_seen(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "a"))
    with scan_memo():
        first = os_sandbox.studio_state_roots()
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "b"))
        second = os_sandbox.studio_state_roots()
    assert first != second
    assert any(path.endswith("b") for path in second)


def test_a_token_planted_between_launches_denies_the_next_launch(monkeypatch, tmp_path):
    import utils.hf_cache_settings as hf_settings

    home = tmp_path / "hf"
    home.mkdir()
    monkeypatch.delenv("HF_TOKEN_PATH", raising = False)
    monkeypatch.setattr(hf_settings, "known_hf_cache_homes", lambda: [home])
    real = os_sandbox.os.path.realpath(str(home))
    with scan_memo():
        assert real not in os_sandbox._hf_home_roots()
        (home / "token").write_text("x", encoding = "utf-8")
        assert real not in os_sandbox._hf_home_roots()
    with scan_memo():
        assert real in os_sandbox._hf_home_roots()
