# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The upgrade preflight the Train tab runs before it starts a worker.

A training start on a model whose ``model_type`` no installed transformers ships used
to be accepted and then killed at model load ("... is not supported yet in
transformers==5.3.0"). Chat asks first, through /validate; training could not reuse that
route (it resolves a ModelConfig, picks a GPU placement and runs the chat coexistence
guard), so it asks here instead.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

MODEL = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
UPGRADE = {
    "model_type": "muse_glimmer",
    "pypi_version": "5.15.0",
    "supported_in_pypi": True,
    "supported_in_main": True,
}


def _route():
    pytest.importorskip("fastapi", reason = "inference stack not installed")
    return pytest.importorskip("routes.inference", reason = "inference stack not installed")


def _stub(
    monkeypatch,
    *,
    upgrade = None,
    latest_tier = False,
    trust_remote_code = False,
):
    """Answer the three preflights the route composes, and nothing else."""
    inf_mod = _route()
    import utils.transformers_latest as latest_mod
    import utils.transformers_version as tv

    monkeypatch.setattr(
        inf_mod, "_requires_trust_remote_code_for_model", lambda *a, **k: trust_remote_code
    )
    monkeypatch.setattr(
        inf_mod, "_hf_offline_if_unreachable", lambda: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(latest_mod, "check_upgrade_for_model", lambda *a, **k: upgrade)
    monkeypatch.setattr(tv, "latest_tier_active_for", lambda *a, **k: latest_tier)
    monkeypatch.setattr(
        "utils.models.model_config.get_base_model_from_lora_identifier", lambda *a, **k: None
    )
    return inf_mod


def _call(
    inf_mod,
    model = MODEL,
    hf_token = None,
):
    from models.inference import TransformersUpgradeCheckRequest
    return asyncio.run(
        inf_mod.check_transformers_upgrade_route(
            TransformersUpgradeCheckRequest(model_name = model, hf_token = hf_token),
            "tester",
        )
    )


def test_installable_upgrade_is_reported_with_its_version(monkeypatch):
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE)
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is True
    assert response.transformers_upgrade.model_type == "muse_glimmer"
    assert response.transformers_upgrade.pypi_version == "5.15.0"
    # The install lands the model on the latest sidecar, and that sidecar trains 16-bit.
    assert response.forces_16bit is True
    assert response.latest_tier_active is False


def test_dev_only_upgrade_does_not_claim_16bit(monkeypatch):
    # Unsloth never installs a transformers dev build, so nothing about the run changes.
    inf_mod = _stub(
        monkeypatch,
        upgrade = {**UPGRADE, "supported_in_pypi": False},
    )
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is True
    assert response.forces_16bit is False


def test_already_routed_model_reports_16bit_without_an_upgrade(monkeypatch):
    # The second run on a provisioned sidecar: nothing to install, still no 4-bit. The
    # Configure preview reads "QLoRA - 4-bit" without this, understating the run's VRAM.
    inf_mod = _stub(monkeypatch, upgrade = None, latest_tier = True)
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is False
    assert response.transformers_upgrade is None
    assert response.latest_tier_active is True
    assert response.forces_16bit is True


def test_supported_model_needs_nothing(monkeypatch):
    inf_mod = _stub(monkeypatch, upgrade = None, latest_tier = False)
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is False
    assert response.forces_16bit is False
    assert response.model_name == MODEL


def test_custom_code_fallback_is_reported(monkeypatch):
    # Feeds the dialog's "continue with custom code" way out, exactly as /validate does.
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, trust_remote_code = True)
    assert _call(inf_mod).requires_trust_remote_code is True


def test_a_failing_preflight_never_fails_the_start(monkeypatch):
    # This gate is additive. If it raised, it would block starts that work today.
    inf_mod = _stub(monkeypatch)
    import utils.transformers_latest as latest_mod
    import utils.transformers_version as tv

    def _boom(*args, **kwargs):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(latest_mod, "check_upgrade_for_model", _boom)
    monkeypatch.setattr(tv, "latest_tier_active_for", _boom)
    monkeypatch.setattr(inf_mod, "_requires_trust_remote_code_for_model", _boom)

    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is False
    assert response.forces_16bit is False


def test_route_is_off_the_openai_compatible_mount():
    # /v1 is the OpenAI-compatible surface; a Studio preflight has no business there.
    inf_mod = _route()
    paths = {route.path for route in inf_mod.studio_router.routes}
    assert "/transformers-upgrade-check" in paths
    assert "/transformers-upgrade-check" not in {route.path for route in inf_mod.router.routes}
