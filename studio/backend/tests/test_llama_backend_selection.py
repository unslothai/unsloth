# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_llama_prebuilt.py: naming a llama.cpp backend, and keeping that choice.

The picker in Settings > System, `UNSLOTH_LLAMA_CPP_BACKEND`, and
`--llama-backend` are three spellings of one thing: a backend request, resolved
here and recorded in the install marker so every later entry point -- setup.sh,
`unsloth studio update`, the desktop updater -- installs the same backend without
being told again.

Network and host detection are stubbed; no GPU or internet needed.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

ilp = importlib.import_module("install_llama_prebuilt")

FORK = ilp.DEFAULT_PUBLISHED_REPO


@pytest.fixture(autouse = True)
def _no_ambient_backend_env(monkeypatch):
    """A backend exported in the developer's shell would override every case here."""
    for name in ("UNSLOTH_LLAMA_CPP_BACKEND", "UNSLOTH_FORCE_VULKAN"):
        monkeypatch.delenv(name, raising = False)


def _marker(tmp_path: Path, **fields) -> Path:
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b9925", **fields}), encoding = "utf-8"
    )
    return tmp_path


def _choice(install_kind: str, name: str = "bundle.tar.gz") -> ilp.AssetChoice:
    return ilp.AssetChoice(
        repo = FORK,
        tag = "b9925",
        name = name,
        url = "https://example/bundle",
        source_label = "published",
        install_kind = install_kind,
    )


# ── The vocabulary ──


def test_every_install_kind_the_installer_can_select_names_a_backend():
    """The marker describes each install by backend, so a bundle kind with no
    mapping would install as an unknown one the picker cannot show or re-assert.

    Reads the kinds out of the module source rather than a hand-kept list, so
    adding a bundle family without extending INSTALL_KIND_BACKENDS fails here
    instead of silently shipping an undescribed install.
    """
    source = Path(ilp.__file__).read_text(encoding = "utf-8")
    selected = set(re.findall(r'install_kind = "([a-z0-9-]+)"', source))
    # The validate-install CLI documents kinds in help text; only assignments count.
    assert selected, "no install_kind assignments found -- has the pattern changed?"
    unmapped = sorted(selected - set(ilp.INSTALL_KIND_BACKENDS))
    assert unmapped == [], f"install kinds with no backend mapping: {unmapped}"


@pytest.mark.parametrize(
    "kind, backend",
    [
        ("linux-cuda", "cuda"),
        ("linux-arm64-cuda", "cuda"),
        ("windows-cuda", "cuda"),
        ("linux-rocm", "rocm"),
        ("windows-hip", "rocm"),
        ("windows-rocm", "rocm"),
        ("linux-vulkan", "vulkan"),
        ("windows-vulkan", "vulkan"),
        ("linux-cpu", "cpu"),
        ("linux-arm64", "cpu"),
        ("windows-arm64", "cpu"),
        ("macos-arm64", "metal"),
        ("sycl-someday", None),
    ],
)
def test_install_kind_maps_to_its_accelerator(kind, backend):
    assert ilp.backend_for_install_kind(kind) == backend


def test_install_kinds_for_backend_is_the_inverse():
    assert ilp.install_kinds_for_backend("vulkan") == ilp.VULKAN_INSTALL_KINDS
    assert "linux-cuda" in ilp.install_kinds_for_backend("cuda")
    assert ilp.install_kinds_for_backend(None) == frozenset()


def test_windows_rocm_bundle_satisfies_a_rocm_request():
    choice = _choice("windows-rocm", "llama-b9925-windows-rocm-gfx1100.zip")
    plan = ilp.InstallReleasePlan("latest", "b9925", "b9925", [choice], SimpleNamespace())

    filtered = ilp._backend_only_release_plans([plan], "rocm")

    assert filtered[0].attempts == [choice]


# ── Reading a choice back off an install ──


@pytest.mark.parametrize(
    "marker, expected",
    [
        # Nothing recorded: an ordinary detected install, which must keep detecting.
        ({"asset": "app-b1-linux-x64-cuda12.tar.gz", "force_cpu": False}, "auto"),
        # The two overrides older installers could record.
        ({"asset": "app-b1-linux-x64-cpu.tar.gz", "force_cpu": True}, "cpu"),
        ({"asset": "vulkan.tar.gz", "llama_backend": "vulkan"}, "vulkan"),
        # Automatic Windows-AMD Vulkan routing: detected, not chosen.
        ({"asset": "win-vulkan.zip", "llama_backend": "auto"}, "auto"),
        # Pre-#7188: no llama_backend key at all, so the asset is the only evidence.
        ({"asset": "llama-b1-bin-ubuntu-vulkan-x64.tar.gz"}, "vulkan"),
        # Written by this build.
        ({"backend": "rocm", "backend_request": "rocm"}, "rocm"),
        ({"backend": "cuda", "backend_request": "auto"}, "auto"),
        # A choice from a newer Studio is returned verbatim, never as "auto":
        # "auto" would license this build to re-detect over it.
        ({"backend": "sycl", "backend_request": "sycl"}, "sycl"),
        ({"asset": "x.tar.gz", "llama_backend": "sycl"}, "sycl"),
        # A non-string records no readable choice at all, so detection applies.
        ({"backend": "cuda", "backend_request": 7}, "auto"),
        ({"asset": "x.tar.gz", "llama_backend": 7}, "auto"),
    ],
)
def test_persisted_backend_request_reads_old_and_new_markers(tmp_path, marker, expected):
    assert ilp.persisted_backend_request(_marker(tmp_path, **marker)) == expected


def test_persisted_backend_request_without_an_install(tmp_path):
    # No marker records no choice, which is detection, not an unreadable one.
    assert ilp.persisted_backend_request(None) == "auto"
    assert ilp.persisted_backend_request(tmp_path) == "auto"


# ── Precedence ──


def test_the_flag_outranks_the_environment_and_the_install(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cuda")
    install = _marker(tmp_path, backend_request = "cpu")
    assert ilp.effective_backend_request("vulkan", install_dir = install) == ("vulkan", True)


def test_the_environment_outranks_the_install(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "cuda")
    install = _marker(tmp_path, backend_request = "cpu")
    assert ilp.effective_backend_request(None, install_dir = install) == ("cuda", True)


def test_an_explicit_auto_clears_a_recorded_choice(monkeypatch, tmp_path):
    # How the picker's "Automatic" entry gets back to detection.
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "auto")
    install = _marker(tmp_path, backend_request = "vulkan")
    assert ilp.effective_backend_request(None, install_dir = install) == ("auto", True)


def test_an_explicit_auto_suppresses_the_legacy_vulkan_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "auto")
    monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")
    install = _marker(tmp_path, backend_request = "vulkan")
    assert ilp.effective_backend_request(None, install_dir = install) == ("auto", True)


def test_legacy_force_vulkan_still_outranks_a_recorded_choice(monkeypatch, tmp_path):
    # The legacy environment override outranks the stored choice.
    monkeypatch.setenv("UNSLOTH_FORCE_VULKAN", "1")
    install = _marker(tmp_path, backend_request = "cpu")
    assert ilp.effective_backend_request(None, install_dir = install) == ("vulkan", True)


def test_a_recorded_choice_applies_when_nobody_names_one(tmp_path):
    # The whole point: no env, no flag, and the install still comes back Vulkan.
    install = _marker(tmp_path, backend_request = "vulkan")
    assert ilp.effective_backend_request(None, install_dir = install) == ("vulkan", False)


def test_an_unknown_environment_value_falls_through_to_the_install(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_BACKEND", "banana")
    install = _marker(tmp_path, backend_request = "vulkan")
    assert ilp.effective_backend_request(None, install_dir = install) == ("vulkan", False)


# ── What gets recorded ──


@pytest.mark.parametrize(
    "request_backend, kind, expected",
    [
        ("vulkan", "linux-vulkan", "vulkan"),
        ("cpu", "linux-cpu", "cpu"),
        ("auto", "linux-cuda", "auto"),
        (None, "linux-cuda", "auto"),
        # macOS cannot persist requests that all resolve to its universal Metal build.
        ("cpu", "macos-arm64", "auto"),
        ("vulkan", "linux-cpu", "auto"),
    ],
)
def test_only_a_request_the_install_honours_is_recorded(request_backend, kind, expected):
    assert ilp.persisted_marker_backend_request(request_backend, _choice(kind)) == expected


def test_macos_backend_resolver_only_offers_automatic_metal(monkeypatch):
    host = ilp.HostInfo(
        system = "Darwin",
        machine = "arm64",
        is_windows = False,
        is_linux = False,
        is_macos = True,
        is_x86_64 = False,
        is_arm64 = True,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
        macos_version = (15, 5),
    )
    seen = []
    monkeypatch.setattr(ilp, "detect_host", lambda: host)
    monkeypatch.setattr(ilp, "load_prebuilt_metadata", lambda install_dir: None)

    def _select(**kwargs):
        seen.append(kwargs["backend"])
        return ilp.BackendSelection(
            backend = "auto",
            host = host,
            published_repo = FORK,
            published_release_tag = "",
            requested_tag = "latest",
            release_plans = [
                ilp.InstallReleasePlan(
                    "latest",
                    "b9925",
                    "b9925",
                    [_choice("macos-arm64")],
                    SimpleNamespace(),
                )
            ],
            persist_llama_backend = None,
            persist_rocm_gfx = None,
        )

    monkeypatch.setattr(ilp, "select_backend_install", _select)
    args = SimpleNamespace(
        published_repo = FORK,
        published_release_tag = "",
        has_rocm = False,
        rocm_gfx = None,
    )

    payload = ilp.resolve_backends_payload("latest", args = args)

    assert seen == ["auto"]
    assert [entry["backend"] for entry in payload["backends"]] == ["auto"]
    assert payload["backends"][0]["resolved_backend"] == "metal"


def test_selection_payload_reports_the_backend_a_plan_would_install():
    primary = _choice("linux-cuda", "cuda13.tar.gz")
    fallback = _choice("linux-cuda", "cuda12.tar.gz")
    selection = SimpleNamespace(
        choice = primary,
        published_repo = FORK,
        release_plans = [
            SimpleNamespace(
                release_tag = "b9925",
                llama_tag = "b9925",
                attempts = [primary, fallback],
            )
        ],
    )

    payload = ilp._selection_payload(selection)

    assert payload["asset"] == "cuda13.tar.gz"
    assert payload["backend"] == "cuda"


def test_explicit_rocm_reprobes_and_suppresses_cuda_on_a_mixed_host(monkeypatch):
    automatic_host = ilp.HostInfo(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = "/usr/bin/nvidia-smi",
        driver_cuda_version = (13, 0),
        compute_caps = ["120"],
        visible_cuda_devices = None,
        has_physical_nvidia = True,
        has_usable_nvidia = True,
    )
    mixed_host = ilp.dataclasses_replace(
        automatic_host,
        has_rocm = True,
        rocm_gfx_target = "gfx1100",
        rocm_gfx_targets = ["gfx1100"],
    )
    probes = []

    def _detect_host(*, probe_rocm_with_nvidia = False):
        probes.append(probe_rocm_with_nvidia)
        return mixed_host if probe_rocm_with_nvidia else automatic_host

    monkeypatch.setattr(ilp, "detect_host", _detect_host)

    route = ilp.route_backend_request(
        backend = "rocm",
        published_repo = FORK,
        published_release_tag = "",
        host = automatic_host,
    )

    assert probes == [True]
    assert route.host.has_usable_nvidia is False
    assert route.host.has_physical_nvidia is False
    assert route.host.has_rocm is True
    assert route.host.rocm_gfx_target == "gfx1100"


def test_backend_resolver_does_not_turn_a_switch_into_a_version_update(monkeypatch):
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: SimpleNamespace(is_macos = False, system = "Linux", machine = "x86_64"),
    )
    monkeypatch.setattr(
        ilp, "load_prebuilt_metadata", lambda install_dir: {"release_tag": "b9900-mix-old"}
    )
    seen_pins = []

    def _unavailable(**kwargs):
        seen_pins.append(kwargs["published_release_tag"])
        raise ilp.BackendUnavailable("release no longer publishes this backend")

    monkeypatch.setattr(ilp, "select_backend_install", _unavailable)
    args = SimpleNamespace(
        published_repo = FORK,
        published_release_tag = "",
        has_rocm = False,
        rocm_gfx = None,
    )

    payload = ilp.resolve_backends_payload("latest", args = args, install_dir = Path("unused"))

    assert payload["pinned_release_tag"] == "b9900-mix-old"
    assert seen_pins == ["b9900-mix-old"] * len(ilp.REQUESTABLE_BACKENDS)
    assert not any(entry["available"] for entry in payload["backends"])


def test_backend_resolver_rejects_cross_repository_switches(monkeypatch):
    host = SimpleNamespace(is_macos = False, system = "Linux", machine = "aarch64")
    monkeypatch.setattr(ilp, "detect_host", lambda: host)
    monkeypatch.setattr(
        ilp,
        "load_prebuilt_metadata",
        lambda install_dir: {"release_tag": "b9900-mix-old", "published_repo": FORK},
    )

    def _upstream_selection(**kwargs):
        choice = ilp.AssetChoice(
            repo = ilp.UPSTREAM_REPO,
            tag = "b9999",
            name = "llama-b9999-bin-ubuntu-vulkan-arm64.tar.gz",
            url = "https://example/upstream-vulkan",
            source_label = "upstream",
            install_kind = "linux-vulkan",
        )
        plan = ilp.InstallReleasePlan("latest", "b9999", "b9999", [choice], SimpleNamespace())
        return ilp.BackendSelection(
            backend = kwargs["backend"],
            host = host,
            published_repo = ilp.UPSTREAM_REPO,
            published_release_tag = "",
            requested_tag = "latest",
            release_plans = [plan],
            persist_llama_backend = "vulkan",
            persist_rocm_gfx = None,
        )

    monkeypatch.setattr(ilp, "select_backend_install", _upstream_selection)
    args = SimpleNamespace(
        published_repo = FORK,
        published_release_tag = "",
        has_rocm = False,
        rocm_gfx = None,
    )

    payload = ilp.resolve_backends_payload("latest", args = args, install_dir = Path("unused"))

    assert not any(entry["available"] for entry in payload["backends"])
    assert all(entry["reason"] == "no_prebuilt" for entry in payload["backends"])


def test_backend_resolver_fails_when_every_option_hits_an_unexpected_error(monkeypatch):
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: SimpleNamespace(is_macos = False, system = "Linux", machine = "x86_64"),
    )
    monkeypatch.setattr(ilp, "load_prebuilt_metadata", lambda install_dir: None)

    def _offline(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(ilp, "select_backend_install", _offline)
    args = SimpleNamespace(
        published_repo = FORK,
        published_release_tag = "",
        has_rocm = False,
        rocm_gfx = None,
    )

    with pytest.raises(RuntimeError, match = "could not resolve any"):
        ilp.resolve_backends_payload("latest", args = args)


def test_metadata_records_both_the_backend_and_the_choice(tmp_path):
    checksums = ilp.ApprovedReleaseChecksums(
        repo = FORK,
        release_tag = "b9925",
        upstream_tag = "b9925",
        source_repo = FORK,
        source_repo_url = f"https://github.com/{FORK}",
    )
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = _choice("linux-vulkan", "app-b9925-linux-x64-vulkan.tar.gz"),
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        llama_backend = "vulkan",
        backend_request = "vulkan",
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["backend"] == "vulkan"
    assert marker["backend_request"] == "vulkan"
    # The superseded field stays, so an older Studio keeps re-asserting Vulkan.
    assert marker["llama_backend"] == "vulkan"


def test_a_detected_install_records_its_backend_but_no_choice(tmp_path):
    checksums = ilp.ApprovedReleaseChecksums(
        repo = FORK,
        release_tag = "b9925",
        upstream_tag = "b9925",
        source_repo = FORK,
        source_repo_url = f"https://github.com/{FORK}",
    )
    ilp.write_prebuilt_metadata(
        tmp_path,
        requested_tag = "latest",
        llama_tag = "b9925",
        release_tag = "b9925",
        choice = _choice("linux-cuda", "app-b9925-linux-x64-cuda12.tar.gz"),
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
    )
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    # Describes the install for the picker...
    assert marker["backend"] == "cuda"
    # ...without pinning it, so the next update re-detects as it always has.
    assert marker["backend_request"] == "auto"


# ── Applying a request to an install ──


def _stub_selection(
    monkeypatch,
    *,
    available,
    install_kind = "linux-cuda",
):
    """Record which backend the install path asked for, and answer for it."""
    seen = []
    # Avoid the unrelated DiffusionGemma backfill download.
    monkeypatch.setattr(ilp, "diffusion_visual_server_backfill_needed", lambda *a, **k: False)

    def _select(
        *,
        backend,
        llama_tag,
        published_repo,
        published_release_tag,
        route = None,
        **_,
    ):
        seen.append(backend)
        if backend in available:
            plan = ilp.InstallReleasePlan(
                "latest",
                "b9925",
                "b9925",
                [_choice(install_kind)],
                SimpleNamespace(ggml_tree = None, repo = FORK),
            )
            return ilp.BackendSelection(
                backend = backend,
                host = route.host if route is not None else ilp.detect_host(),
                published_repo = published_repo,
                published_release_tag = published_release_tag,
                requested_tag = "latest",
                release_plans = [plan],
                persist_llama_backend = None,
                persist_rocm_gfx = None,
            )
        raise ilp.BackendUnavailable(f"no {backend} prebuilt bundle attempts were available")

    monkeypatch.setattr(ilp, "select_backend_install", _select)
    return seen


def test_an_install_applies_the_choice_its_marker_recorded(monkeypatch, tmp_path):
    seen = _stub_selection(monkeypatch, available = {"vulkan"}, install_kind = "linux-vulkan")
    monkeypatch.setattr(ilp, "existing_install_matches_plan", lambda *a, **k: True)
    _marker(tmp_path, backend = "vulkan", backend_request = "vulkan")

    ilp.install_prebuilt(tmp_path, "latest", FORK, "")

    assert seen == ["vulkan"]


def test_an_update_refuses_to_replace_an_unknown_recorded_choice(monkeypatch, tmp_path):
    seen = _stub_selection(monkeypatch, available = {"auto"})
    marker_path = _marker(tmp_path, backend = "sycl", backend_request = "sycl")

    with pytest.raises(SystemExit) as raised:
        ilp.install_prebuilt(tmp_path, "latest", FORK, "")

    # EXIT_ERROR, not the source fallback: a source build would pick its own
    # backend, which is the outcome refusing this update exists to prevent.
    assert raised.value.code == ilp.EXIT_ERROR
    assert seen == []
    marker = json.loads((marker_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["backend_request"] == "sycl"


def test_a_recorded_choice_this_host_cannot_serve_falls_back_to_detection(monkeypatch, tmp_path):
    # Re-detect after hardware invalidates a stored choice.
    seen = _stub_selection(monkeypatch, available = {"auto"})
    monkeypatch.setattr(ilp, "existing_install_matches_plan", lambda *a, **k: True)
    _marker(tmp_path, backend = "rocm", backend_request = "rocm")

    ilp.install_prebuilt(tmp_path, "latest", FORK, "")

    assert seen == ["rocm", "auto"]
    marker = json.loads((tmp_path / "UNSLOTH_PREBUILT_INFO.json").read_text())
    assert marker["backend_request"] == "auto"


def test_a_named_backend_this_host_cannot_serve_fails_instead(monkeypatch, tmp_path):
    # An explicit request must not silently install another backend.
    seen = _stub_selection(monkeypatch, available = {"auto"})
    _marker(tmp_path, backend = "cuda", backend_request = "auto")

    with pytest.raises(SystemExit) as raised:
        ilp.install_prebuilt(tmp_path, "latest", FORK, "", llama_backend = "vulkan")

    # Both the UI and setup need a specific fail-closed result.
    assert raised.value.code == ilp.EXIT_BACKEND_UNAVAILABLE
    assert seen == ["vulkan"]


@pytest.mark.parametrize(
    "backend_request,expected_exit",
    [("cpu", ilp.EXIT_BACKEND_UNAVAILABLE), ("auto", ilp.EXIT_FALLBACK)],
)
def test_only_automatic_selection_can_source_fallback_after_candidate_failure(
    monkeypatch, tmp_path, backend_request, expected_exit
):
    _stub_selection(monkeypatch, available = {backend_request}, install_kind = "linux-cpu")
    _marker(tmp_path, backend = "cpu", backend_request = backend_request)
    monkeypatch.setattr(ilp, "existing_install_matches_plan", lambda *a, **k: False)
    monkeypatch.setattr(ilp, "diffusion_visual_server_backfill_needed", lambda *a, **k: False)
    monkeypatch.setattr(ilp, "resolve_validation_model", lambda probe: probe)
    monkeypatch.setattr(
        ilp,
        "validate_prebuilt_attempts",
        lambda *a, **k: (_ for _ in ()).throw(ilp.PrebuiltFallback("candidate failed")),
    )
    monkeypatch.setattr(ilp, "collect_system_report", lambda *a, **k: "report")

    with pytest.raises(SystemExit) as raised:
        ilp.install_prebuilt(tmp_path, "latest", FORK, "")

    assert raised.value.code == expected_exit


def test_backend_unavailable_uses_the_prebuilt_rollback_path():
    assert issubclass(ilp.BackendUnavailable, ilp.PrebuiltFallback)
