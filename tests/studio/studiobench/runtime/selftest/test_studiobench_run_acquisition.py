# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A run releases the Studios it acquired, and records which ones they were.

THE SIDES ARE ACQUIRED ONE AFTER THE OTHER. In a self-managed A/B the base is installed, launched
and already serving before the treatment's clone begins, so a failure in the treatment's setup used
to unwind past a base that nothing ever stopped: the cleanup lives in the `finally` under the cells
and a failure during setup never reaches it. The same hole swallowed the two setup steps that leave
by RETURNING -- the health check and the development-build gate -- with both Studios up.

An abandoned Studio is not idle. `launch_studio` detaches the server with `setsid -f`, so it keeps
the port; Studio's own launcher then ABORTS rather than binding when it finds one of its own servers
there (`studio/backend/run.py`, `_resolve_port` with `avoid_own_studio`), the retry's server exits,
and `wait_for_healthz` takes its 200 from the STALE process. That run measures the build the
previous attempt installed while `run_meta` records the ref this one asked for.

WHICH SERVER THE TREATMENT WAS is the second thing here, and it is asserted through the same drive
because it is a property of the row a real run writes. An attached base is identified by its URL;
the treatment carried only the label typed after `--ab`, so `--attach A --attach-b B --ab fix`
resumed happily against `--attach-b C` and reported B's measurements as C's result.

`run()` itself is driven, with the seams that leave this process stubbed at the boundary they cross:
the installer, the launcher, the health check, the login, the browser and the cell runner. Nothing
below them is re-implemented -- the acquisition loop, the gate, the identity and the payload are the
shipped ones.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import studiobench.__main__ as sb  # noqa: E402
from studiobench import pacer as pacer_mod  # noqa: E402
from studiobench.runtime import browser as browser_mod  # noqa: E402
from studiobench.runtime import bundle_guard, lifecycle  # noqa: E402
from studiobench.runtime import session as session_mod  # noqa: E402
from studiobench.runtime.lifecycle import StudioAuth, StudioInstall  # noqa: E402
from studiobench.runtime.types import Paths  # noqa: E402


class _Verdict:
    def __init__(self, production: bool) -> None:
        self.production = production
        self.bundle_type = 0
        self.reason = "production build" if production else "development server"

    def as_dict(self) -> dict:
        return {"production": self.production, "reason": self.reason}


class _Bundle:
    # WHAT `browser.launch` WOULD HAVE RESOLVED on the machine running this test. `run_meta`
    # records the engine and `requested_identity` resolves the same way, so a stub that names a
    # fixed one would make a legitimate resume look like an engine change off Linux and macOS.
    engine = browser_mod.default_engine()[0]
    engine_note = "stubbed for this test"
    browser = context = page = cdp = None

    def close(self) -> None:
        pass


class _Pacer:
    base_url = "http://127.0.0.1:65535"

    def __init__(self) -> None:
        self.state = types.SimpleNamespace(model_ids = [])

    def start(self):
        return self

    def stop(self) -> None:
        pass


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """Every seam `run()` reaches outside this process, stubbed where it leaves it.

    The returned dict is both the knobs (which ref fails to install, whether /healthz answers,
    whether the bundle is a production build) and the record (what was installed, launched and
    stopped, and what the Studios looked like at the moment the first cell ran).
    """

    state = {
        "installed": [],
        "launched": [],
        "stopped": [],
        "stopped_when_the_cells_ran": None,
        "install_fails_for": None,
        "healthy": True,
        "production": True,
        "out": tmp_path / "out",
    }

    def fake_install(ref, home, *args, **kwargs):
        if ref == state["install_fails_for"]:
            raise RuntimeError(f"install.sh for {ref} exited 1")
        state["installed"].append(ref)
        return StudioInstall(home = Path(home), repo = Path(home).parent / "repo", branch = ref)

    def fake_launch(install, port, log_path, *args, **kwargs):
        install.port = port
        install.pid = 90_000 + port
        install.bootstrap_password = "secret"
        state["launched"].append(install)
        return install

    class _Runner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def run(self, cell, plan):
            if state["stopped_when_the_cells_ran"] is None:
                state["stopped_when_the_cells_ran"] = list(state["stopped"])
            return {
                "cell_id": cell.cell_id,
                "completed": True,
                "actions": [{"action": "keystroke", "ran": True}],
            }

    monkeypatch.setattr(lifecycle, "install_studio", fake_install)
    monkeypatch.setattr(lifecycle, "launch_studio", fake_launch)
    monkeypatch.setattr(lifecycle, "stop_studio", lambda install: state["stopped"].append(install))
    monkeypatch.setattr(lifecycle, "wait_for_healthz", lambda *a, **k: state["healthy"])
    monkeypatch.setattr(
        lifecycle,
        "authenticate",
        lambda base_url, username, password: StudioAuth(
            access_token = "t",
            refresh_token = "r",
            base_url = base_url,
            username = username,
            password = password,
        ),
    )
    monkeypatch.setattr(lifecycle, "register_provider", lambda *a, **k: "provider-1")
    monkeypatch.setattr(bundle_guard, "check_bundle", lambda url: _Verdict(state["production"]))
    monkeypatch.setattr(browser_mod, "launch", lambda *a, **k: _Bundle())
    monkeypatch.setattr(
        browser_mod,
        "install_wall_clock_watchdog",
        lambda *a, **k: types.SimpleNamespace(cancel = lambda: None),
    )
    monkeypatch.setattr(pacer_mod, "Pacer", _Pacer)
    monkeypatch.setattr(session_mod, "CellRunner", _Runner)
    return state


def _args(state, *extra):
    return sb.parse_args(
        ["--tier", "quick", "--rungs", "1K", "--out", str(state["out"])] + list(extra)
    )


def _rows(state):
    path = Paths.under(state["out"]).payload_jsonl
    return [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line]


# ── the sides a failed setup leaves behind ──────────────────────────────────────────────────


def test_the_base_studio_is_stopped_when_the_treatment_install_fails(studio):
    """The reported leak: the base is up and serving while the treatment's clone and build run."""

    studio["install_fails_for"] = "pr-9296"

    with pytest.raises(RuntimeError):
        sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296")

    assert [i.branch for i in studio["launched"]] == ["main"]
    assert [i.branch for i in studio["stopped"]] == ["main"]


def test_every_studio_is_stopped_when_one_of_them_never_answers_healthz(studio):
    """A setup step that fails by RETURNING leaves the same abandoned server behind."""

    studio["healthy"] = False

    assert sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296") == 2
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


def test_every_studio_is_stopped_when_the_gate_refuses_a_development_build(studio):
    studio["production"] = False

    assert sb.run(_args(studio, "--branch", "main", "--ab", "pr-9296"), ab_ref = "pr-9296") == 3
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


def test_keep_studio_still_leaves_a_failed_setup_running(studio):
    """The control on the guard itself: `--keep-studio` asks for exactly this leak."""

    studio["install_fails_for"] = "pr-9296"

    with pytest.raises(RuntimeError):
        sb.run(
            _args(studio, "--branch", "main", "--ab", "pr-9296", "--keep-studio"),
            ab_ref = "pr-9296",
        )

    assert [i.branch for i in studio["launched"]] == ["main"]
    assert studio["stopped"] == []


def test_a_studio_the_caller_attached_is_never_stopped(studio):
    """The other control: this harness only stops what it launched itself."""

    studio["healthy"] = False
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )

    assert sb.run(args, ab_ref = "fix") == 2
    assert studio["stopped"] == []


def test_a_run_that_reaches_its_cells_stops_the_studios_once_at_the_end(studio):
    """The control that matters: an ordinary run still gets two live Studios and still cleans up.

    The guard is a `finally` over the whole of setup, so the failure it must not have is stopping
    the Studios on the way IN. `stopped_when_the_cells_ran` is read inside the cell runner.
    """

    args = _args(studio, "--branch", "main", "--ab", "pr-9296", "--reps", "2")

    assert sb.run(args, ab_ref = "pr-9296") == 0
    assert studio["stopped_when_the_cells_ran"] == []
    assert [i.branch for i in studio["stopped"]] == ["main", "pr-9296"]


# ── which server the treatment was ──────────────────────────────────────────────────────────


def test_an_attached_ab_records_the_treatment_url_it_measured(studio):
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310/",
        "--attach-b",
        "http://127.0.0.1:5311/",
        "--ab",
        "fix",
    )

    assert sb.run(args, ab_ref = "fix") == 0

    plan = [r for r in _rows(studio) if r.get("row_type") == "ab_plan"]
    assert len(plan) == 1
    assert plan[0]["treatment_url"] == "http://127.0.0.1:5311"


def test_a_self_managed_ab_records_no_treatment_url(studio):
    """A treatment this run installed is identified by its ref; the port it landed on is not it."""

    args = _args(studio, "--branch", "main", "--ab", "pr-9296")

    assert sb.run(args, ab_ref = "pr-9296") == 0

    plan = [r for r in _rows(studio) if r.get("row_type") == "ab_plan"]
    assert plan[0]["treatment_url"] == ""


def test_a_resume_pointed_at_another_treatment_studio_is_refused(studio):
    """End to end, over the payload the run above actually wrote: same `--ab` label, second URL."""

    paths = Paths.under(studio["out"])
    first = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )
    assert sb.run(first, ab_ref = "fix") == 0

    again = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5312",
        "--ab",
        "fix",
        "--resume",
    )
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    with pytest.raises(SystemExit) as excinfo:
        sb.prepare_payload(
            paths,
            sb.requested_identity(again, "fix", corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )

    message = str(excinfo.value)
    assert "treatment_url" in message
    assert "5311" in message and "5312" in message


def test_the_same_treatment_studio_still_resumes(studio):
    """The control: the identity this payload was recorded under has to keep resuming."""

    paths = Paths.under(studio["out"])
    args = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311",
        "--ab",
        "fix",
    )
    assert sb.run(args, ab_ref = "fix") == 0
    corpus_hash = _rows(studio)[0]["corpus_hash"]

    resumed = _args(
        studio,
        "--attach",
        "http://127.0.0.1:5310",
        "--attach-b",
        "http://127.0.0.1:5311/",
        "--ab",
        "fix",
        "--resume",
    )

    assert (
        sb.prepare_payload(
            paths,
            sb.requested_identity(resumed, "fix", corpus_hash),
            resume = True,
            log = lambda *_a: None,
        )
        is None
    )


# ── one Studio, two sides: the attached null control ────────────────────────────────────────


class _ProviderBackend:
    """`lifecycle.register_provider`'s contract, per origin: idempotent by DISPLAY NAME.

    The real one deletes every existing provider whose `display_name` matches before creating the
    replacement, so registering the pacer twice against ONE Studio destroys the id the first
    registration handed out. Modelled rather than stubbed to a constant, because that deletion is
    the whole failure.
    """

    def __init__(self) -> None:
        self.live: dict = {}
        self.registrations: list = []
        self.deleted: list = []
        self._issued = 0

    def register(self, base_url, auth, provider):
        origin = base_url.rstrip("/")
        rows = self.live.setdefault(origin, {})
        for pid, name in list(rows.items()):
            if name == provider.name:
                del rows[pid]
                self.deleted.append((origin, pid))
        self._issued += 1
        pid = f"provider-{self._issued}"
        rows[pid] = provider.name
        provider.id = pid
        self.registrations.append((origin, pid))
        return pid


def _capture_init_scripts(monkeypatch) -> list:
    captured: list = []

    def fake_launch(
        engine,
        *args,
        init_scripts = None,
        **kwargs,
    ):
        captured.extend(init_scripts or [])
        return _Bundle()

    monkeypatch.setattr(browser_mod, "launch", fake_launch)
    return captured


def _selected_provider_ids(scripts: list, origin: str) -> set:
    """The provider ids the seed scripts for `origin` actually SELECT, out of their checkpoints."""

    import re

    ids = set()
    for script in scripts:
        if json.dumps(origin) not in script:
            continue
        for match in re.finditer(r"external::([^:]+)::", script):
            ids.add(match.group(1))
    return ids


def test_an_attached_null_control_registers_one_provider_for_the_one_studio(studio, monkeypatch):
    """`--attach U --attach-b U` is TWO SIDES ON ONE STUDIO, which `is_null_control` accepts.

    Registering per side registered the pacer twice against that single backend, and the second
    registration deleted the id the base side's seed script had already captured. Both scripts are
    scoped to the same origin, Playwright does not define the order init scripts run in, and
    `StudioAuth.rotate` re-adds them mid-run -- so the base could boot every cell with a DELETED
    provider selected, which renders as "No longer offered" and throws `Connection not found`
    without ever asking for a completion.
    """

    backend = _ProviderBackend()
    monkeypatch.setattr(lifecycle, "register_provider", backend.register)
    scripts = _capture_init_scripts(monkeypatch)
    url = "http://127.0.0.1:5310"

    args = _args(studio, "--attach", url, "--attach-b", url, "--branch", "main", "--ab", "main")
    assert sb.run(args, ab_ref = "main") == 0

    # THE SYMPTOM FIRST. Whatever the bookkeeping below says, the failure a cell actually meets is
    # a seed script that SELECTS a provider the backend no longer has, so that is what this pins:
    # every id named by a script scoped to this origin is still live there, and no live id is
    # unnamed. Asserting only "nothing was deleted" would pass a future fix that deleted and then
    # re-seeded both scripts with the survivor, and would fail a fix that never selects at all.
    selected = _selected_provider_ids(scripts, url)
    assert selected == set(backend.live[url])
    assert selected, "the one Studio must still have a provider selected"
    # And the registration itself: one backend, one registration, nothing destroyed.
    assert backend.deleted == []
    assert backend.registrations == [(url, "provider-1")]


def test_two_attached_studios_still_get_a_provider_each(studio, monkeypatch):
    """The control: two DIFFERENT origins are two backends and must each be registered."""

    backend = _ProviderBackend()
    monkeypatch.setattr(lifecycle, "register_provider", backend.register)
    scripts = _capture_init_scripts(monkeypatch)
    base_url, treatment_url = "http://127.0.0.1:5310", "http://127.0.0.1:5311"

    args = _args(studio, "--attach", base_url, "--attach-b", treatment_url, "--ab", "fix")
    assert sb.run(args, ab_ref = "fix") == 0

    assert backend.registrations == [(base_url, "provider-1"), (treatment_url, "provider-2")]
    assert _selected_provider_ids(scripts, base_url) == {"provider-1"}
    assert _selected_provider_ids(scripts, treatment_url) == {"provider-2"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
