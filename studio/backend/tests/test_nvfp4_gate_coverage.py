# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every hosted NVFP4 denoiser Studio references is either gated or kept away from auto, and a gate record only
covers the flashinfer kernels it was measured (or shown bit-identical) on."""

from __future__ import annotations

import json
import types

import pytest

import core.inference.diffusion_nvfp4_gate as gate
import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_families import _FAMILIES as FAMILIES, family_prequant_repo
from core.inference.diffusion_nvfp4_gate import load_gate_records, nvfp4_gate_passed
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES
from core.inference.video_families import (
    detect_video_family,
    video_family_prequant_schemes,
)

_FAILED_VIDEO_REPOS = (
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
)
_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_ZIMAGE = "Tongyi-MAI/Z-Image-Turbo"


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")


def _blackwell(monkeypatch, backend = "flashinfer"):
    import core.inference.diffusion_nvfp4_ops as ops

    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(tq, "_capability", lambda *a, **k: (10, 0))
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *a, **k: False)
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, **kw: True)
    monkeypatch.setattr(ops, "select_nvfp4_backend", lambda device = None: backend)
    return types.SimpleNamespace(device = "cuda:0", dtype = "bfloat16")


def _hosted(scheme):
    return True


def _image_nvfp4_rows():
    rows = []
    for fam in FAMILIES:
        bases = {fam.base_repo} | {
            entry[0]
            for entry in (getattr(fam, "prequant_variant_repos", ()) or ())
            if entry[1] == "nvfp4"
        }
        for base in sorted(b for b in bases if b):
            repo = family_prequant_repo(fam, "nvfp4", base)
            if repo:
                rows.append((fam.name, base, repo))
    return rows


def test_every_hosted_image_nvfp4_build_has_a_passing_gate_record_for_that_repo(enabled):
    rows = _image_nvfp4_rows()
    assert rows
    recorded = {
        (r["family"], r["repo_id"]) for r in load_gate_records() if r.get("all_pass") is True
    }
    for family, base, repo in rows:
        assert nvfp4_gate_passed(family, base), (family, base, repo)
        assert (family, repo) in recorded, (family, repo)


def test_no_family_can_auto_pick_nvfp4_without_a_gate_record():
    for name, row in tq._FAMILY_AUTO_PREFER.items():
        if "nvfp4" in row.schemes:
            assert row.gated, name


def test_the_ungated_a14b_build_never_leads_auto_but_stays_an_explicit_pick(monkeypatch, enabled):
    target = _blackwell(monkeypatch)
    fam = detect_video_family(_A14B)
    assert "nvfp4" in video_family_prequant_schemes(fam)
    assert (
        tq.select_transformer_quant_scheme(
            target, "auto", family = fam.name, base_repo = _A14B, has_prequant = _hosted
        )
        == "int8"
    )
    assert (
        tq.select_transformer_quant_scheme(
            target, "nvfp4", family = fam.name, base_repo = _A14B, has_prequant = _hosted
        )
        == "nvfp4"
    )


@pytest.mark.parametrize("repo", _FAILED_VIDEO_REPOS)
def test_video_builds_that_failed_the_gate_are_denied_and_not_advertised(
    monkeypatch, enabled, repo
):
    target = _blackwell(monkeypatch)
    fam = detect_video_family(repo)
    assert fam is not None
    assert "nvfp4" not in video_family_prequant_schemes(fam)
    for requested in ("nvfp4", "auto"):
        assert (
            tq.select_transformer_quant_scheme(
                target, requested, family = fam.name, base_repo = fam.base_repo, has_prequant = _hosted
            )
            != "nvfp4"
        ), (repo, requested)
    assert tq.family_denies_scheme(fam.name, "nvfp4", fam.base_repo)
    reason = tq.explain_unusable_scheme(fam.name, "nvfp4", fam.base_repo)
    # A measured failure, not an unmeasured base: the refusal must not say nothing was measured.
    assert "measured and failed the NVFP4 accuracy gate" in reason
    assert "LPIPS" in reason
    assert "nothing has measured" not in reason


def test_every_video_family_with_a_hosted_nvfp4_build_is_denied_or_kept_out_of_auto():
    for fam in VIDEO_FAMILIES:
        hosted = any(entry[0] == "nvfp4" for entry in (fam.prequant_repos or ()))
        if not hosted:
            continue
        prefer = getattr(fam.profile, "auto_prefer", None)
        denied = "nvfp4" in fam.profile.scheme_deny
        assert denied or prefer is None or prefer.gated, fam.name


def _gate_file(tmp_path, *records):
    path = tmp_path / "nvfp4_gate_record.json"
    path.write_text(json.dumps({"version": 1, "records": list(records)}), encoding = "utf-8")
    return path


def _zimage_record(**versions):
    return {
        "family": "z-image",
        "base_repo": _ZIMAGE,
        "policy_id": "zimg_rg76_v1",
        "policy_version": 1,
        "checkpoint_sha256": "a" * 64,
        "all_pass": True,
        "backend": "flashinfer",
        "versions": versions,
    }


def test_the_shipped_records_cover_the_flashinfer_studio_installs():
    from core.inference.diffusion_nvfp4_install import FLASHINFER_VERSION

    records = [r for r in load_gate_records() if r.get("backend") == "flashinfer"]
    assert records
    for record in records:
        assert FLASHINFER_VERSION in gate.record_flashinfer_versions(record), record["repo_id"]


def test_a_flashinfer_the_record_names_or_lists_bit_identical_is_covered(tmp_path):
    path = _gate_file(
        tmp_path, _zimage_record(flashinfer = "0.6.18.post1", flashinfer_bit_identical = ["0.6.6"])
    )
    for version in ("0.6.18.post1", "0.6.6"):
        assert (
            gate.nvfp4_gate_flashinfer_mismatch("z-image", _ZIMAGE, installed = version, path = path)
            is None
        )


def test_an_unchecked_flashinfer_is_named_in_the_mismatch(tmp_path):
    path = _gate_file(
        tmp_path, _zimage_record(flashinfer = "0.6.18.post1", flashinfer_bit_identical = ["0.6.6"])
    )
    message = gate.nvfp4_gate_flashinfer_mismatch("z-image", _ZIMAGE, installed = "0.7.0", path = path)
    assert message and "0.7.0" in message and "0.6.18.post1" in message and "0.6.6" in message


def test_no_installed_flashinfer_or_no_recorded_version_is_not_a_mismatch(tmp_path):
    path = _gate_file(tmp_path, _zimage_record(flashinfer = "0.6.6"))
    assert gate.nvfp4_gate_flashinfer_mismatch("z-image", _ZIMAGE, installed = "", path = path) is None
    bare = _gate_file(tmp_path, _zimage_record())
    assert (
        gate.nvfp4_gate_flashinfer_mismatch("z-image", _ZIMAGE, installed = "0.7.0", path = bare)
        is None
    )


def test_auto_drops_the_gated_head_on_an_unchecked_flashinfer_but_explicit_still_loads(
    monkeypatch, enabled, tmp_path
):
    target = _blackwell(monkeypatch)
    path = _gate_file(tmp_path, _zimage_record(flashinfer = "0.6.6"))
    monkeypatch.setattr(gate, "GATE_RECORD_PATH", path)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.6.6")
    assert tq._nvfp4_gate_backend_ok("z-image", _ZIMAGE, "cuda:0") is True
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    assert tq._nvfp4_gate_backend_ok("z-image", _ZIMAGE, "cuda:0") is False
    assert (
        tq.select_transformer_quant_scheme(
            target, "nvfp4", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
        )
        == "nvfp4"
    )


def test_the_mismatch_warning_is_logged_once(monkeypatch, tmp_path, caplog):
    path = _gate_file(tmp_path, _zimage_record(flashinfer = "0.6.6"))
    monkeypatch.setattr(gate, "GATE_RECORD_PATH", path)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    monkeypatch.setattr(
        gate,
        "_passing_records",
        lambda family, base_repo, path = None: (_zimage_record(flashinfer = "0.6.6"),),
    )
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert gate.warn_flashinfer_mismatch_once("z-image", _ZIMAGE) is not None
    assert sum("flashinfer 0.7.0" in r.getMessage() for r in caplog.records) == 1


def test_the_installed_version_read_never_raises():
    value = gate.installed_flashinfer_version()
    assert value is None or isinstance(value, str)


def test_an_ungated_base_that_was_never_measured_keeps_the_generic_refusal(enabled):
    reason = tq.explain_unusable_scheme("qwen-image", "nvfp4", "Qwen/Qwen-Image")
    assert "no reviewed NVFP4 accuracy-gate record" in reason
    assert "measured and failed" not in reason


def _engage(
    monkeypatch,
    backend = "flashinfer",
    layers = 4,
):
    """``_engage_nvfp4_backend`` with the conversion stubbed: ``layers`` move only when the backend is flashinfer."""
    import core.inference.diffusion_nvfp4_linear as lin
    import core.inference.diffusion_nvfp4_ops as ops
    import core.inference.diffusion_prequant as pq

    monkeypatch.setattr(ops, "select_nvfp4_backend", lambda device = None: backend)
    monkeypatch.setattr(
        lin,
        "convert_nvfp4_backend",
        lambda transformer, metadata, chosen, logger = None: layers if chosen == "flashinfer" else 0,
    )

    def run(
        family,
        base,
        metadata = None,
    ):
        return pq._engage_nvfp4_backend(
            object(), metadata or {}, "cuda:0", family = family, base = base
        )

    return run


def _gate_warnings(caplog):
    return [r.getMessage() for r in caplog.records if "nvfp4.gate" in r.getMessage()]


@pytest.mark.parametrize(
    ("family", "base"),
    [("z-image", _ZIMAGE), ("qwen-image", "Qwen/Qwen-Image-2512")],
)
def test_an_unchecked_flashinfer_warns_once_when_an_nvfp4_load_engages_it(
    monkeypatch, enabled, caplog, family, base
):
    """z-image has no nvfp4 deny entry (its record only feeds auto); qwen-image does. Both warn, once, at the load."""
    target = _blackwell(monkeypatch)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    assert family not in tq._FAMILY_SCHEME_DENY or "nvfp4" in tq._FAMILY_SCHEME_DENY[family]
    engage = _engage(monkeypatch)
    with caplog.at_level("WARNING"):
        assert (
            tq.select_transformer_quant_scheme(
                target, "nvfp4", family = family, base_repo = base, has_prequant = _hosted
            )
            == "nvfp4"
        )
        assert not _gate_warnings(caplog)
        for _ in range(2):
            assert engage(family, base) == 4
    hits = [m for m in _gate_warnings(caplog) if "flashinfer 0.7.0" in m]
    assert len(hits) == 1, _gate_warnings(caplog)


def test_planning_and_validation_selects_do_not_use_up_the_load_warning(
    monkeypatch, enabled, caplog
):
    """Scheme selection runs for planning, the pre-eviction gate and validation, none of which loads kernels."""
    target = _blackwell(monkeypatch)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    with caplog.at_level("WARNING"):
        for unproven_ok in (False, True, False):
            tq.select_transformer_quant_scheme(
                target,
                "nvfp4",
                family = "z-image",
                base_repo = _ZIMAGE,
                has_prequant = _hosted,
                unproven_ok = unproven_ok,
            )
    assert not _gate_warnings(caplog)
    assert gate._WARNED == set()
    with caplog.at_level("WARNING"):
        _engage(monkeypatch)("z-image", _ZIMAGE)
    assert len([m for m in _gate_warnings(caplog) if "flashinfer 0.7.0" in m]) == 1


def test_a_covered_flashinfer_loads_explicit_nvfp4_silently(monkeypatch, enabled, caplog):
    target = _blackwell(monkeypatch)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.6.6")
    with caplog.at_level("WARNING"):
        assert (
            tq.select_transformer_quant_scheme(
                target, "nvfp4", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
            )
            == "nvfp4"
        )
        assert _engage(monkeypatch)("z-image", _ZIMAGE) == 4
    assert not _gate_warnings(caplog)


@pytest.mark.parametrize("backend", ["torchao", "raises"])
def test_an_explicit_nvfp4_pick_on_torchao_never_warns_about_flashinfer(
    monkeypatch, enabled, caplog, backend
):
    """UNSLOTH_NVFP4_BACKEND=torchao or a failed flashinfer preflight runs torchao kernels: the flashinfer coverage
    warning would name kernels this load does not use."""
    import core.inference.diffusion_nvfp4_ops as ops

    target = _blackwell(monkeypatch)

    def _select(device = None):
        if backend == "raises":
            raise RuntimeError("preflight blew up")
        return backend

    monkeypatch.setattr(ops, "select_nvfp4_backend", _select)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    with caplog.at_level("WARNING"):
        assert (
            tq.select_transformer_quant_scheme(
                target, "nvfp4", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
            )
            == "nvfp4"
        )
        if backend == "torchao":
            assert _engage(monkeypatch, backend = "torchao")("z-image", _ZIMAGE) == 0
    assert not _gate_warnings(caplog)


def test_a_load_that_moves_no_layer_onto_flashinfer_never_warns(monkeypatch, enabled, caplog):
    """A checkpoint without baked activation scales stays on torchao even when flashinfer is selected."""
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    with caplog.at_level("WARNING"):
        assert _engage(monkeypatch, layers = 0)("z-image", _ZIMAGE) == 0
    assert not _gate_warnings(caplog)
    assert gate._WARNED == set()


def test_the_load_warning_keys_on_the_checkpoint_base_not_the_fetch_mirror(
    monkeypatch, enabled, caplog
):
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    with caplog.at_level("WARNING"):
        _engage(monkeypatch)(
            "z-image", "someone/z-image-mirror", metadata = {"base_model_id": _ZIMAGE}
        )
    assert len([m for m in _gate_warnings(caplog) if _ZIMAGE in m]) == 1


def test_the_backend_probe_is_skipped_by_an_explicit_select(monkeypatch, enabled):
    import core.inference.diffusion_nvfp4_ops as ops

    target = _blackwell(monkeypatch)
    probed = []
    monkeypatch.setattr(
        ops, "select_nvfp4_backend", lambda device = None: probed.append(device) or "flashinfer"
    )
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    assert (
        tq.select_transformer_quant_scheme(
            target, "nvfp4", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
        )
        == "nvfp4"
    )
    assert probed == []


def test_auto_rejects_an_unchecked_flashinfer_silently_and_the_explicit_load_still_warns(
    monkeypatch, enabled, caplog
):
    """Auto falls back to the ladder without the warning (it never runs those kernels), and does not use up the
    once-only warning a later explicit nvfp4 load on the same base owes the user."""
    target = _blackwell(monkeypatch)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: "0.7.0")
    with caplog.at_level("WARNING"):
        assert tq._nvfp4_gate_backend_ok("z-image", _ZIMAGE, "cuda:0") is False
        tq.select_transformer_quant_scheme(
            target, "auto", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
        )
    assert not _gate_warnings(caplog)
    engage = _engage(monkeypatch)
    with caplog.at_level("WARNING"):
        assert (
            tq.select_transformer_quant_scheme(
                target, "nvfp4", family = "z-image", base_repo = _ZIMAGE, has_prequant = _hosted
            )
            == "nvfp4"
        )
        engage("z-image", _ZIMAGE)
    assert len([m for m in _gate_warnings(caplog) if "flashinfer 0.7.0" in m]) == 1


def test_an_unreadable_flashinfer_version_is_not_covered_when_flashinfer_runs(tmp_path):
    path = _gate_file(tmp_path, _zimage_record(flashinfer = "0.6.6"))
    message = gate.nvfp4_gate_flashinfer_mismatch(
        "z-image", _ZIMAGE, installed = "", path = path, backend_active = True
    )
    assert message and "could not be read" in message and "0.6.6" in message
    bare = _gate_file(tmp_path, _zimage_record())
    assert (
        gate.nvfp4_gate_flashinfer_mismatch(
            "z-image", _ZIMAGE, installed = "", path = bare, backend_active = True
        )
        is None
    )


def test_auto_drops_the_gated_head_when_the_flashinfer_version_is_unreadable(
    monkeypatch, enabled, caplog
):
    _blackwell(monkeypatch)
    monkeypatch.setattr(gate, "_WARNED", set())
    monkeypatch.setattr(gate, "installed_flashinfer_version", lambda: None)
    with caplog.at_level("WARNING"):
        assert tq._nvfp4_gate_backend_ok("z-image", _ZIMAGE, "cuda:0") is False
    assert not _gate_warnings(caplog)
    with caplog.at_level("WARNING"):
        _engage(monkeypatch)("z-image", _ZIMAGE)
    assert len([m for m in _gate_warnings(caplog) if "could not be read" in m]) == 1


def test_the_installed_version_falls_back_to_an_imported_module(monkeypatch):
    import importlib.metadata as md
    import sys

    def _missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", _missing)
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(__version__ = "0.6.6"))
    assert gate.installed_flashinfer_version() == "0.6.6"
    monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace())
    assert gate.installed_flashinfer_version() is None
