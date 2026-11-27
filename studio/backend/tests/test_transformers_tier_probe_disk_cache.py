# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tier probe verdicts for local checkpoints persist across worker processes."""

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import utils.transformers_version as tv


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fake sidecars with dist-info versions, a counting probe and a fresh process cache."""
    versions = {"default": "5.5.0", "530": "5.3.0", "550": "5.5.0", "510": "5.10.2"}
    dirs = {}
    for tier in ("530", "550", "510"):
        d = tmp_path / f"venv_{tier}"
        d.mkdir()
        dirs[tier] = str(d)

    def set_version(tier, version):
        versions[tier] = version
        if tier == "default":
            return
        for old in Path(dirs[tier]).glob("transformers-*.dist-info"):
            for f in old.iterdir():
                f.unlink()
            old.rmdir()
        if version is not None:
            info = Path(dirs[tier]) / f"transformers-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: transformers\nVersion: {version}\n\nbody\n"
            )

    for tier in ("530", "550", "510"):
        set_version(tier, versions[tier])
    monkeypatch.setattr(
        tv,
        "_probe_tier_venvs",
        lambda: {
            "default": ("", lambda: True),
            "530": (dirs["530"], lambda: True),
            "550": (dirs["550"], lambda: True),
            "510": (dirs["510"], lambda: True),
        },
    )
    monkeypatch.setattr(tv, "_probe_tier_order", lambda: ("530", "550", "510"))
    real_version = getattr(tv, "_tier_transformers_version", None)
    monkeypatch.setattr(
        tv,
        "_tier_transformers_version",
        lambda d: versions["default"] if d == "" else real_version(d),
        raising = False,
    )
    monkeypatch.delenv("UNSLOTH_DISABLE_TIER_PROBE", raising = False)
    monkeypatch.setattr(tv, "_probe_tier_cache", {})
    state = {"calls": [], "results": {}}

    def fake_probe(target_dir, model_name, hf_token):
        tier = next(t for t, (d, _) in tv._probe_tier_venvs().items() if d == target_dir)
        state["calls"].append(tier)
        return state["results"].get(tier, True)

    monkeypatch.setattr(tv, "_probe_autoconfig", fake_probe)
    state["set_version"] = set_version
    return state


def _model(tmp_path, cfg = None):
    d = tmp_path / "merged"
    d.mkdir(exist_ok = True)
    (d / "config.json").write_text(
        json.dumps(cfg or {"model_type": "llama", "transformers_version": "5.1.0"})
    )
    return str(d)


def _cache_file() -> Path:
    from utils.paths.storage_roots import cache_root
    return cache_root() / "transformers_tier_probe.json"


def _new_process(monkeypatch):
    monkeypatch.setattr(tv, "_probe_tier_cache", {})


def _probe(model, **kw):
    kw.setdefault("include_default", True)
    kw.setdefault("floor", "default")
    return tv._probe_tier(model, None, "test", **kw)


def test_second_process_reuses_verdict(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    assert _probe(model) == "default"
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert env["calls"] == ["default"]


def test_escalated_verdict_persists(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    env["results"] = {"default": False, "530": False}
    assert _probe(model) == "550"
    _new_process(monkeypatch)
    assert _probe(model) == "550"
    assert env["calls"] == ["default", "530", "550"]


def test_floor_verdict_persists_after_full_walk(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    env["results"] = {t: False for t in ("default", "530", "550", "510")}
    assert _probe(model) == "default"
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert len(env["calls"]) == 4


def test_rewritten_config_reprobes(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    assert _probe(model) == "default"
    _model(tmp_path, {"model_type": "llama", "transformers_version": "5.2.0"})
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert len(env["calls"]) == 2


def test_sidecar_upgrade_reprobes(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    env["results"] = {"default": False, "530": False}
    assert _probe(model) == "550"
    env["set_version"]("530", "5.3.1")
    _new_process(monkeypatch)
    assert _probe(model) == "550"
    assert len(env["calls"]) == 6


def test_default_upgrade_reprobes(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    assert _probe(model) == "default"
    env["set_version"]("default", "5.6.0")
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert len(env["calls"]) == 2


def test_unwalked_tier_upgrade_keeps_verdict(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    assert _probe(model) == "default"
    env["set_version"]("510", "5.11.0")
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert len(env["calls"]) == 1


def test_probe_mode_is_part_of_key(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    assert _probe(model) == "default"
    _new_process(monkeypatch)
    assert tv._probe_tier(model, None, "tokenizer") == "530"
    assert env["calls"] == ["default", "530"]


def test_transient_not_persisted(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    env["results"] = {"default": None}
    assert _probe(model) == "default"
    _new_process(monkeypatch)
    env["results"] = {}
    assert _probe(model) == "default"
    assert env["calls"] == ["default", "default"]


def test_skipped_tier_not_persisted(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    venvs = tv._probe_tier_venvs()
    venvs["530"] = (venvs["530"][0], lambda: False)
    monkeypatch.setattr(tv, "_probe_tier_venvs", lambda: venvs)
    env["results"] = {"default": False}
    assert _probe(model) == "550"
    _new_process(monkeypatch)
    assert _probe(model) == "550"
    assert env["calls"] == ["default", "550", "default", "550"]


def test_missing_sidecar_version_not_persisted(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    env["set_version"]("530", None)
    env["results"] = {"default": False}
    assert _probe(model) == "530"
    _new_process(monkeypatch)
    assert _probe(model) == "530"
    assert len(env["calls"]) == 4


def test_hub_id_not_persisted(env, monkeypatch):
    assert _probe("org/not-a-local-dir") == "default"
    _new_process(monkeypatch)
    assert _probe("org/not-a-local-dir") == "default"
    assert len(env["calls"]) == 2
    assert not _cache_file().exists()


def test_config_changed_during_probe_not_persisted(env, monkeypatch, tmp_path):
    model = _model(tmp_path)
    original = tv._probe_autoconfig

    def rewriting_probe(target_dir, model_name, hf_token):
        _model(tmp_path, {"model_type": "llama", "transformers_version": "5.9.0"})
        return original(target_dir, model_name, hf_token)

    monkeypatch.setattr(tv, "_probe_autoconfig", rewriting_probe)
    assert _probe(model) == "default"
    assert not _cache_file().exists()


@pytest.mark.parametrize("payload", ["{not json", "[1, 2]", '{"x": {"tier": 5}}', ""])
def test_corrupt_cache_file_falls_back_to_probe(env, monkeypatch, tmp_path, payload):
    path = _cache_file()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(payload)
    model = _model(tmp_path)
    assert _probe(model) == "default"
    _new_process(monkeypatch)
    assert _probe(model) == "default"
    assert env["calls"] == ["default"]
    assert isinstance(json.loads(path.read_text()), dict)


def test_cache_is_bounded(env, monkeypatch, tmp_path):
    monkeypatch.setattr(tv, "_PROBE_DISK_CACHE_MAX", 3, raising = False)
    for i in range(5):
        d = tmp_path / f"m{i}"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": "llama", "i": i}))
        _probe(str(d))
    assert len(json.loads(_cache_file().read_text())) == 3


def test_disabled_probe_never_touches_cache(env, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_DISABLE_TIER_PROBE", "1")
    assert _probe(_model(tmp_path)) == "default"
    assert env["calls"] == []
    assert not _cache_file().exists()


def test_tier_version_reads_dist_info(tmp_path):
    info = tmp_path / "transformers-5.5.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: transformers\nVersion: 5.5.0\n")
    assert tv._tier_transformers_version(str(tmp_path)) == "5.5.0"
    assert tv._tier_transformers_version(str(tmp_path / "missing")) is None


def test_cached_tier_whose_sidecar_is_unavailable_walks_on(env, monkeypatch, tmp_path):
    # A damaged sidecar that cannot be repaired now (offline, rebuild lock held, latest in a
    # worker) keeps its dist-info; a live probe skips it and settles on the next tier.
    model = _model(tmp_path)
    env["results"] = {"default": False, "530": False}
    assert _probe(model) == "550"
    _new_process(monkeypatch)
    venvs = dict(tv._probe_tier_venvs())
    venvs["550"] = (venvs["550"][0], lambda: False)
    monkeypatch.setattr(tv, "_probe_tier_venvs", lambda: venvs)
    assert _probe(model) == "510"
    assert env["calls"] == ["default", "530", "550", "default", "530", "510"]
