# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The checked-in NVFP4 gate record, and the verdict the runtime reads out of it."""

from __future__ import annotations

import dataclasses
import json

import pytest

import core.inference.diffusion_nvfp4_policy as policy_mod
from core.inference.diffusion_families import canonical_base, detect_family
from core.inference.diffusion_nvfp4_gate import (
    GATE_RECORD_PATH,
    GATE_RECORD_VERSION,
    RECORD_KEY_FIELDS,
    load_gate_records,
    nvfp4_gate_backend,
    nvfp4_gate_backends,
    nvfp4_gate_passed,
    nvfp4_gate_record,
)
from core.inference.diffusion_nvfp4_policy import policy_by_id, resolve_policy

ZIMAGE_BASE = "Tongyi-MAI/Z-Image-Turbo"


def _record(**overrides):
    record = {
        "family": "z-image",
        "base_repo": ZIMAGE_BASE,
        "policy_id": "zimg_rg76_v1",
        "policy_version": 1,
        "checkpoint_sha256": "a" * 64,
        "repo_id": "unsloth/Z-Image-Turbo-NVFP4",
        "filename": "Z-Image-Turbo-NVFP4.pt",
        "gptq": False,
        "all_pass": True,
        "num_pairs": 28,
        "num_passed": 28,
        "backend": "flashinfer",
    }
    record.update(overrides)
    return record


def _gate_file(
    tmp_path,
    *records,
    version = GATE_RECORD_VERSION,
):
    path = tmp_path / "nvfp4_gate_record.json"
    path.write_text(json.dumps({"version": version, "records": list(records)}), encoding = "utf-8")
    return path


def test_the_shipped_gate_record_parses_and_declares_this_schema():
    document = json.loads(GATE_RECORD_PATH.read_text(encoding = "utf-8"))
    assert document["version"] == GATE_RECORD_VERSION
    assert isinstance(document["records"], list)
    assert len(load_gate_records()) == len(document["records"])


def test_every_shipped_record_names_a_policy_this_commit_resolves():
    for record in load_gate_records():
        policy = policy_by_id(record["policy_id"])
        assert policy is not None, record["policy_id"]
        assert int(record["policy_version"]) == int(policy.version), record
        resolved = resolve_policy(record["family"], record["base_repo"])
        assert resolved is not None and resolved.policy_id == policy.policy_id, record
        assert set(RECORD_KEY_FIELDS) <= set(record), record


def test_every_shipped_record_names_a_checkpoint_the_family_hosts():
    records = load_gate_records()
    if not records:
        pytest.skip("no gate records yet: nvfp4 is out of the auto ladder for every family")
    for record in records:
        fam = detect_family(record["base_repo"])
        assert fam is not None, record
        hosted = {repo.lower() for _scheme, repo in fam.prequant_repos}
        hosted.update(repo.lower() for _base, _scheme, repo in fam.prequant_variant_repos)
        assert str(record["repo_id"]).lower() in hosted, record
        assert len(str(record["checkpoint_sha256"])) == 64, record


def test_the_shipped_record_leaves_every_family_ungated():
    if load_gate_records():
        pytest.skip("a gate record has been added; the ladder tests cover its effect")
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE) is False
    assert nvfp4_gate_passed("qwen-image", "Qwen/Qwen-Image") is False
    assert nvfp4_gate_passed("flux.1", "black-forest-labs/FLUX.1-schnell") is False


def test_no_record_at_all_reads_false(tmp_path):
    empty = _gate_file(tmp_path)
    assert nvfp4_gate_record("z-image", ZIMAGE_BASE, path = empty) is None
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = empty) is False


def test_a_matching_record_reads_true_and_canonicalises_the_base(tmp_path):
    path = _gate_file(tmp_path, _record())
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is True
    assert nvfp4_gate_passed("Z-Image", ZIMAGE_BASE.lower(), path = path) is True
    mirrors = [
        mirror
        for mirror in ("unsloth/Z-Image-Turbo",)
        if canonical_base(mirror).lower() == ZIMAGE_BASE.lower()
    ]
    for mirror in mirrors:
        assert nvfp4_gate_passed("z-image", mirror, path = path) is True


def test_the_backend_a_verdict_was_measured_on_is_readable(tmp_path):
    path = _gate_file(tmp_path, _record())
    assert nvfp4_gate_backend("z-image", ZIMAGE_BASE, path = path) == "flashinfer"
    assert nvfp4_gate_backend("Z-Image", ZIMAGE_BASE, path = path) == "flashinfer"
    for record in (_record(all_pass = False), _record(backend = None), _record(backend = "")):
        assert (
            nvfp4_gate_backend("z-image", ZIMAGE_BASE, path = _gate_file(tmp_path, record)) is None
        ), record
    assert nvfp4_gate_backend("z-image", ZIMAGE_BASE, path = _gate_file(tmp_path)) is None
    assert (
        nvfp4_gate_backend(
            "z-image", ZIMAGE_BASE, path = _gate_file(tmp_path, _record(backend = "TorchAO "))
        )
        == "torchao"
    )


def test_two_artifacts_gated_on_different_backends_both_read_as_covered(tmp_path):
    path = _gate_file(
        tmp_path,
        _record(),
        _record(checkpoint_sha256 = "b" * 64, gptq = True, backend = "torchao"),
    )
    assert nvfp4_gate_backends("z-image", ZIMAGE_BASE, path = path) == ("flashinfer", "torchao")
    assert nvfp4_gate_backends("Z-Image", ZIMAGE_BASE, path = path) == ("flashinfer", "torchao")
    assert nvfp4_gate_backends("z-image", ZIMAGE_BASE, path = _gate_file(tmp_path)) == ()
    assert (
        nvfp4_gate_backends(
            "z-image", ZIMAGE_BASE, path = _gate_file(tmp_path, _record(all_pass = False))
        )
        == ()
    )


def test_every_shipped_record_names_the_backend_it_was_measured_on():
    for record in load_gate_records():
        assert str(record.get("backend") or "").strip(), record
        assert (
            nvfp4_gate_backend(record["family"], record["base_repo"])
            == str(record["backend"]).strip().lower()
        ), record


def test_a_failed_run_reads_false(tmp_path):
    path = _gate_file(tmp_path, _record(all_pass = False))
    assert nvfp4_gate_record("z-image", ZIMAGE_BASE, path = path) is not None
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is False


def test_a_recorded_failure_does_not_mask_a_later_checkpoint_that_passed(tmp_path):
    # --allow-fail records failures, so a pass can sit behind a failure in file order.
    path = _gate_file(
        tmp_path,
        _record(all_pass = False, checkpoint_sha256 = "b" * 64, backend = "torchao"),
        _record(),
    )
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is True
    assert nvfp4_gate_backend("z-image", ZIMAGE_BASE, path = path) == "flashinfer"


def test_a_policy_version_bump_invalidates_the_verdict(tmp_path, monkeypatch):
    path = _gate_file(tmp_path, _record())
    bumped = dataclasses.replace(policy_mod.ZIMG_RG76, version = 2)
    monkeypatch.setattr(
        policy_mod,
        "NVFP4_POLICIES",
        (bumped,) + tuple(p for p in policy_mod.NVFP4_POLICIES if p.policy_id != bumped.policy_id),
    )
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is False


def test_a_record_at_a_superseded_policy_no_longer_gates_its_base(tmp_path):
    # The artifact it measured declares a policy this build no longer resolves, so the loader
    # refuses it; the record must not keep nvfp4 in the auto ladder on its behalf.
    path = _gate_file(tmp_path, _record(policy_id = "zimg_f8mod_toq34_v1"))
    assert nvfp4_gate_record("z-image", ZIMAGE_BASE, path = path) is not None
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is False
    flux = _record(
        family = "flux.1",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        policy_id = "flux_mod_single_v1",
    )
    path = _gate_file(tmp_path, flux)
    assert nvfp4_gate_passed("flux.1", "black-forest-labs/FLUX.1-schnell", path = path) is False
    path = _gate_file(tmp_path, dict(flux, policy_id = "flux_r420_v1"))
    assert nvfp4_gate_passed("flux.1", "black-forest-labs/FLUX.1-schnell", path = path) is True


def test_a_record_for_another_base_or_family_is_not_inherited(tmp_path):
    path = _gate_file(tmp_path, _record())
    assert nvfp4_gate_passed("z-image", "some-org/Z-Image-Fork", path = path) is False
    assert nvfp4_gate_passed("z-image", None, path = path) is False
    assert nvfp4_gate_passed("flux.1", ZIMAGE_BASE, path = path) is False


def test_a_record_pinned_to_another_policy_id_is_not_returned(tmp_path):
    path = _gate_file(tmp_path, _record())
    assert nvfp4_gate_record("z-image", ZIMAGE_BASE, "qwen_p02_v1", path = path) is None
    assert nvfp4_gate_record("z-image", ZIMAGE_BASE, "zimg_rg76_v1", path = path) is not None


def test_an_unreadable_or_absent_file_is_no_evidence_not_an_error(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_gate_records(missing) == ()
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = missing) is False
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding = "utf-8")
    assert load_gate_records(broken) == ()
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = broken) is False


def test_a_rewritten_file_is_re_read_rather_than_served_from_the_cache(tmp_path):
    path = _gate_file(tmp_path)
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is False
    _gate_file(tmp_path, _record())
    assert nvfp4_gate_passed("z-image", ZIMAGE_BASE, path = path) is True
