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
    inspected = None,
):
    """Answer the three preflights the route composes, and nothing else.

    ``inspected`` collects every target the preflights were pointed at, so a test can
    assert WHICH copy of the model was read.
    """
    inf_mod = _route()
    import utils.transformers_latest as latest_mod
    import utils.transformers_version as tv

    def _record(target):
        if inspected is not None:
            inspected.append(target)

    def _trust_remote_code(target, *args, **kwargs):
        _record(target)
        return trust_remote_code

    def _check_upgrade(target, *args, **kwargs):
        _record(target)
        return upgrade

    def _latest_tier(target, *args, **kwargs):
        _record(target)
        return latest_tier

    monkeypatch.setattr(inf_mod, "_requires_trust_remote_code_for_model", _trust_remote_code)
    monkeypatch.setattr(
        inf_mod, "_hf_offline_if_unreachable", lambda: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(latest_mod, "check_upgrade_for_model", _check_upgrade)
    monkeypatch.setattr(tv, "latest_tier_active_for", _latest_tier)
    monkeypatch.setattr(
        "utils.models.model_config.get_base_model_from_lora_identifier", lambda *a, **k: None
    )
    return inf_mod


def _call(
    inf_mod,
    model = MODEL,
    hf_token = None,
    **fields,
):
    from models.inference import TransformersUpgradeCheckRequest
    return asyncio.run(
        inf_mod.check_transformers_upgrade_route(
            TransformersUpgradeCheckRequest(model_name = model, hf_token = hf_token, **fields),
            "tester",
        )
    )


def test_installable_upgrade_is_reported_with_its_version(monkeypatch):
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE)
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is True
    assert response.transformers_upgrade.model_type == "muse_glimmer"
    assert response.transformers_upgrade.pypi_version == "5.15.0"
    assert response.forces_16bit is True
    assert response.latest_tier_active is False


def test_dev_only_upgrade_does_not_claim_16bit(monkeypatch):
    inf_mod = _stub(
        monkeypatch,
        upgrade = {**UPGRADE, "supported_in_pypi": False},
    )
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is True
    assert response.forces_16bit is False


def test_already_routed_model_reports_16bit_without_an_upgrade(monkeypatch):
    # The second run on a provisioned sidecar: nothing to install, still no 4-bit, so the preview understates VRAM.
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
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, trust_remote_code = True)
    assert _call(inf_mod).requires_trust_remote_code is True


def test_a_merely_offered_upgrade_keeps_4bit_when_custom_code_can_load_it(monkeypatch):
    # Taking the "continue with custom code" way out installs nothing, so the worker runs on the
    # current transformers and loads bnb 4-bit; claiming 16-bit would oversize the run's VRAM.
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, trust_remote_code = True)
    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is True
    assert response.forces_16bit is False


def test_an_active_sidecar_forces_16bit_even_with_custom_code(monkeypatch):
    # No install to decline: the sidecar already routes this model, and it trains 16-bit whatever the repo ships.
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, trust_remote_code = True, latest_tier = True)
    assert _call(inf_mod).forces_16bit is True


def test_a_failing_preflight_never_fails_the_start(monkeypatch):
    # This gate is additive: if it raised, it would block starts that work today.
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


def _cached_snapshot(
    monkeypatch,
    root,
    repo_id = "org/model",
    commit = "commit-a",
):
    """A real HF-layout cache entry: the pin resolvers validate the layout AND the root."""
    from hub.utils import hf_cache_state

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kwargs: [root])
    snapshot = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / commit
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}", encoding = "utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_a_pinned_snapshot_is_what_gets_inspected(monkeypatch, tmp_path):
    # The gate used to be handed the Hub identifier for a cached model, while the remote-code gate and
    # the worker both load the pinned snapshot, and a repo's current config.json says nothing about
    # the snapshot this run opens.
    inspected: list = []
    inf_mod = _stub(monkeypatch, upgrade = None, inspected = inspected)
    snapshot = _cached_snapshot(monkeypatch, tmp_path)

    response = _call(
        inf_mod,
        model = "org/model",
        model_snapshot_path = str(snapshot),
        model_snapshot_repo_id = "org/model",
        prefer_local_cache = True,
    )

    assert inspected, "the route must inspect something"
    assert all(target == str(snapshot) for target in inspected), inspected
    assert response.model_name == "org/model"


def test_a_selected_cache_directory_resolves_to_its_snapshot(monkeypatch, tmp_path):
    # prefer_local_cache with no exact pin, the scan route's second branch: the cache dir resolves to its snapshot.
    inspected: list = []
    inf_mod = _stub(monkeypatch, upgrade = None, inspected = inspected)
    snapshot = _cached_snapshot(monkeypatch, tmp_path)

    _call(
        inf_mod,
        model = "org/model",
        prefer_local_cache = True,
        model_local_path = str(snapshot.parent.parent),
    )

    assert all(target == str(snapshot) for target in inspected), inspected


def test_the_lora_base_is_resolved_from_the_pinned_snapshot(monkeypatch, tmp_path):
    # The worker resolves a LoRA's base from its load target and the scan route does the same, so
    # reading the Hub identifier instead asks the current adapter_config.json which base to judge
    # while the run loads the pinned snapshot's.
    resolved_from: list = []
    inf_mod = _stub(monkeypatch, upgrade = None)
    snapshot = _cached_snapshot(monkeypatch, tmp_path)

    def _base(identifier, *args, **kwargs):
        resolved_from.append(identifier)
        return None

    monkeypatch.setattr("utils.models.model_config.get_base_model_from_lora_identifier", _base)

    _call(
        inf_mod,
        model = "org/model",
        model_snapshot_path = str(snapshot),
        model_snapshot_repo_id = "org/model",
        prefer_local_cache = True,
    )

    assert resolved_from == [str(snapshot)], resolved_from


def test_a_known_cached_model_with_no_path_still_resolves_its_snapshot(monkeypatch, tmp_path):
    # A cached inventory row can carry a null cachePath while the Train tab still sends
    # prefer_local_cache, which is why _resolve_model_snapshot searches every cache root; requiring a
    # path here judged those selections on the repo's current architecture.
    inspected: list = []
    inf_mod = _stub(monkeypatch, upgrade = None, inspected = inspected)
    snapshot = _cached_snapshot(monkeypatch, tmp_path)
    from hub.utils import hf_cache_state

    monkeypatch.setattr(
        hf_cache_state, "iter_repo_cache_dirs", lambda *a, **k: [snapshot.parent.parent]
    )

    _call(inf_mod, model = "org/model", prefer_local_cache = True)

    assert all(target == str(snapshot) for target in inspected), inspected


def test_an_unpinned_model_is_still_checked_by_identifier(monkeypatch):
    inspected: list = []
    inf_mod = _stub(monkeypatch, upgrade = None, inspected = inspected)
    _call(inf_mod)
    assert all(target == MODEL for target in inspected), inspected


def test_an_unresolvable_pin_falls_back_to_the_identifier(monkeypatch, tmp_path):
    # _model_config_inspection_target 404s for a vanished snapshot, and this preflight is additive.
    inspected: list = []
    inf_mod = _stub(monkeypatch, upgrade = None, inspected = inspected)

    _call(
        inf_mod,
        model = "org/model",
        prefer_local_cache = True,
        model_snapshot_path = str(tmp_path / "models--org--model" / "snapshots" / "gone"),
        model_snapshot_repo_id = "org/model",
    )

    assert all(target == "org/model" for target in inspected), inspected


def test_an_exact_4bit_resume_is_flagged_before_the_install_is_offered(monkeypatch):
    # effective_training_load_in_4bit RAISES for this config once the latest sidecar routes the model,
    # and that sidecar is a persistent overlay, so consenting to the install on the way into a resume
    # strands the checkpoint for good.
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, trust_remote_code = True)
    monkeypatch.setattr(
        "storage.studio_db.get_run",
        lambda run_id: {"config_json": {"load_in_4bit": True}} if run_id == "run-42" else None,
    )
    monkeypatch.setattr(
        "core.training.provenance.exact_resume_resource_requirements",
        lambda config: (True, True),
    )

    assert _call(inf_mod, resume_run_id = "run-42").install_breaks_exact_resume is True
    assert _call(inf_mod).install_breaks_exact_resume is False
    assert _call(inf_mod, resume_run_id = "missing").install_breaks_exact_resume is False


def test_an_already_active_sidecar_is_not_blamed_on_the_install(monkeypatch):
    # The overlay is already installed, so the resume is refused whatever this route answers.
    inf_mod = _stub(monkeypatch, upgrade = UPGRADE, latest_tier = True)
    monkeypatch.setattr(
        "storage.studio_db.get_run", lambda run_id: {"config_json": {"load_in_4bit": True}}
    )
    monkeypatch.setattr(
        "core.training.provenance.exact_resume_resource_requirements",
        lambda config: (True, True),
    )

    assert _call(inf_mod, resume_run_id = "run-42").install_breaks_exact_resume is False


def test_route_is_off_the_openai_compatible_mount():
    inf_mod = _route()
    paths = {route.path for route in inf_mod.studio_router.routes}
    assert "/transformers-upgrade-check" in paths
    assert "/transformers-upgrade-check" not in {route.path for route in inf_mod.router.routes}


# The tests above prove the gate fires; these pin the far more common case where it must not.


def test_an_old_client_sends_the_identifier_alone():
    # Every added field must be optional, or an older frontend fails validation on a payload it once took.
    from models.inference import TransformersUpgradeCheckRequest

    request = TransformersUpgradeCheckRequest(model_name = MODEL)
    assert request.prefer_local_cache is False
    assert (request.model_local_path, request.model_snapshot_path) == (None, None)
    assert (request.model_snapshot_repo_id, request.resume_run_id) == (None, None)


def test_a_minimal_response_reads_as_the_pre_gate_behaviour():
    # What an older client sees and a newer one falls back to: no upgrade, no precision claim, no refusal.
    from models.inference import TransformersUpgradeCheckResponse

    response = TransformersUpgradeCheckResponse(model_name = MODEL)
    assert response.requires_transformers_upgrade is False
    assert response.requires_trust_remote_code is False
    assert response.latest_tier_active is False
    assert response.forces_16bit is False
    assert response.install_breaks_exact_resume is False


@pytest.mark.parametrize(
    "latest_tier,installable,custom_code,expected",
    [
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, True),
        (False, True, True, False),
        (True, False, False, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, True),
    ],
)
def test_forces_16bit_over_every_combination(
    monkeypatch, latest_tier, installable, custom_code, expected
):
    # The preview draws its VRAM claim from this field, so a wrong cell is a wrong number. Exhaustive.
    upgrade = None
    if installable or custom_code:
        upgrade = {
            **UPGRADE,
            "supported_in_pypi": installable,
            "pypi_version": "5.15.0" if installable else None,
        }
    inf_mod = _stub(
        monkeypatch, upgrade = upgrade, latest_tier = latest_tier, trust_remote_code = custom_code
    )
    assert _call(inf_mod).forces_16bit is expected


@pytest.mark.parametrize(
    "failure",
    [
        OSError("network is unreachable"),
        TimeoutError("timed out"),
        ValueError("malformed config.json"),
        KeyError("architectures"),
    ],
)
def test_a_failing_preflight_never_escapes_the_route(monkeypatch, failure):
    # This route runs in front of every start, so a raise here fails a start for a model that loads fine.
    inf_mod = _route()
    import utils.transformers_latest as latest_mod
    import utils.transformers_version as tv

    def _boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(inf_mod, "_requires_trust_remote_code_for_model", _boom)
    monkeypatch.setattr(
        inf_mod, "_hf_offline_if_unreachable", lambda: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(latest_mod, "check_upgrade_for_model", _boom)
    monkeypatch.setattr(tv, "latest_tier_active_for", _boom)
    monkeypatch.setattr(
        "utils.models.model_config.get_base_model_from_lora_identifier", lambda *a, **k: None
    )

    response = _call(inf_mod)
    assert response.requires_transformers_upgrade is False
    assert response.forces_16bit is False


def test_the_route_is_behind_authentication():
    import inspect

    inf_mod = _route()
    subject = inspect.signature(inf_mod.check_transformers_upgrade_route).parameters[
        "current_subject"
    ]
    assert subject.default is not inspect.Parameter.empty
