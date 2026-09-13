# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached-path authorization for training starts.

Companion to test_training_ambient_hf_token.py, which covers the token plumbing. These cover the
path-to-repository mapping the model and dataset legs authorize against, where a miss is fail-OPEN:
``cached_repo_ref_for_path`` returning ``None`` means the caller skips the access check entirely.
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _make_repo(
    root: Path,
    dir_name: str,
    sha: str = "abc",
) -> Path:
    """A cache repo laid out as huggingface_hub writes it, snapshot symlinks included."""
    repo = root / dir_name
    snapshot = repo / "snapshots" / sha
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True, exist_ok = True)
    blobs.mkdir(parents = True, exist_ok = True)
    for index, name in enumerate(("config.json", "model.safetensors")):
        blob = blobs / f"blob{index}"
        blob.write_text("{}")
        link = snapshot / name
        if not link.exists():
            try:
                link.symlink_to(Path("..") / ".." / "blobs" / f"blob{index}")
            except (OSError, NotImplementedError):
                # Windows without Developer Mode refuses symlinks; the real cache falls back to
                # copies there, so a plain file is the faithful layout rather than a skip.
                link.write_text("{}")
    return snapshot


@pytest.fixture(autouse = True)
def restore_environ():
    """The child-side scrub writes os.environ directly, which monkeypatch cannot undo for keys it
    never saw. Snapshot the whole mapping so one test's scrub is not the next one's premise."""
    import os

    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [root])
    return root


def test_snapshot_symlink_into_blobs_still_names_its_repo(cache_root):
    """resolve(strict=True) lands in blobs/, which is under the same repo dir."""
    from hub.utils.hf_cache_state import cached_repo_ref_for_path

    snapshot = _make_repo(cache_root, "models--Org--Private-Model")
    assert cached_repo_ref_for_path(snapshot / "model.safetensors") == (
        "Org/Private-Model",
        "model",
    )


def test_cached_private_dataset_path_is_attributed_to_the_dataset_repo(cache_root):
    """A dataset snapshot passed where a model is expected must still be authorized.

    A dataset repo can hold config.json plus weights, so the model leg accepted it; matching only
    "models--" returned None and skipped the access check on the operator's private dataset.
    """
    from hub.utils.hf_cache_state import cached_repo_id_for_path, cached_repo_ref_for_path

    snapshot = _make_repo(cache_root, "datasets--org--private-data")
    assert cached_repo_ref_for_path(snapshot) == ("org/private-data", "dataset")
    # The repo_type-restricted view still answers only for its own type.
    assert cached_repo_id_for_path(snapshot) is None
    assert cached_repo_id_for_path(snapshot, "dataset") == "org/private-data"


def test_cached_space_path_is_attributed(cache_root):
    from hub.utils.hf_cache_state import cached_repo_ref_for_path
    snapshot = _make_repo(cache_root, "spaces--org--private-space")
    assert cached_repo_ref_for_path(snapshot) == ("org/private-space", "space")


def test_a_decoy_repo_dir_inside_a_snapshot_does_not_hide_the_real_repo(cache_root):
    """Repo file paths are arbitrary, so a private snapshot may contain "models--foo--bar".

    Committing to the DEEPEST match and giving up when its parent was not a cache root returned
    None for a path plainly inside models--org--private, skipping the check.
    """
    from hub.utils.hf_cache_state import cached_repo_ref_for_path

    snapshot = _make_repo(cache_root, "models--org--private")
    decoy = snapshot / "models--foo--bar"
    decoy.mkdir()
    (decoy / "config.json").write_text("{}")

    assert cached_repo_ref_for_path(decoy) == ("org/private", "model")
    assert cached_repo_ref_for_path(decoy / "config.json") == ("org/private", "model")


def test_a_symlink_from_outside_cannot_launder_a_private_snapshot(cache_root, tmp_path):
    """resolve(strict=True) runs first, so the link's target is what gets authorized."""
    from hub.utils.hf_cache_state import cached_repo_ref_for_path

    snapshot = _make_repo(cache_root, "models--org--private")
    link = tmp_path / "looks_harmless"
    try:
        link.symlink_to(snapshot, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem does not allow symlinks")
    assert cached_repo_ref_for_path(link) == ("org/private", "model")


def test_a_single_segment_repo_id_round_trips(cache_root):
    """models--gpt2 has no separator at all."""
    from hub.utils.hf_cache_state import cached_repo_ref_for_path
    assert cached_repo_ref_for_path(_make_repo(cache_root, "models--gpt2")) == ("gpt2", "model")


def test_a_repo_in_a_second_cache_root_is_attributed(monkeypatch, tmp_path):
    """A box whose cache has moved keeps the old HF_HOME in the remembered roots."""
    from hub.utils import hf_cache_state

    first = tmp_path / "hub"
    second = tmp_path / "hub2"
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(
        hf_cache_state,
        "hf_cache_roots",
        lambda scan_errors = None: [first, second],
    )
    snapshot = _make_repo(second, "models--other--secret")
    assert hf_cache_state.cached_repo_ref_for_path(snapshot) == ("other/secret", "model")


def test_a_prefix_collision_sibling_of_a_cache_root_is_not_attributed(monkeypatch, tmp_path):
    """ "hub_evil" is not "hub": containment is decided by samefile, not by name."""
    from hub.utils import hf_cache_state

    root = tmp_path / "hub"
    root.mkdir()
    sibling = tmp_path / "hub_evil"
    sibling.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda scan_errors = None: [root])
    assert hf_cache_state.cached_repo_ref_for_path(_make_repo(sibling, "models--org--p")) is None


def test_a_lookalike_outside_every_cache_root_is_not_attributed(cache_root, tmp_path):
    """Not the operator's cache, so there is nothing of theirs to protect."""
    from hub.utils.hf_cache_state import cached_repo_ref_for_path

    outside = _make_repo(tmp_path / "elsewhere", "models--org--private")
    assert cached_repo_ref_for_path(outside) is None
    assert cached_repo_ref_for_path(cache_root) is None


def test_an_unreadable_cache_root_authorizes_rather_than_skips(monkeypatch, tmp_path):
    """Fail closed: the unscannable root may be the one this path belongs to."""
    from hub.utils import hf_cache_state

    missing = tmp_path / "gone"

    def roots(scan_errors = None):
        if scan_errors is not None:
            scan_errors.append(OSError("unreadable"))
        return [missing]

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", roots)
    snapshot = _make_repo(tmp_path / "elsewhere", "models--org--private")
    assert hf_cache_state.cached_repo_ref_for_path(snapshot) == ("org/private", "model")


def test_repo_ids_round_trip_through_the_cache_dir_name(cache_root):
    """ "--" is forbidden inside a repo id, so the split is unambiguous.

    Guards the mapping against a "fix" for a name that cannot exist: huggingface_hub's own
    validate_repo_id rejects "org/my--model", and _scan_cached_repo parses the same way.
    """
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    from hub.utils.hf_cache_state import cached_repo_ref_for_path

    for repo_id in ("org/private", "Org/Private-Model", "org/name_with_underscores"):
        validate_repo_id(repo_id)
        snapshot = _make_repo(cache_root, "models--" + repo_id.replace("/", "--"))
        assert cached_repo_ref_for_path(snapshot) == (repo_id, "model")

    with pytest.raises(HFValidationError):
        validate_repo_id("org/my--model")


def test_local_dataset_paths_in_the_cache_require_caller_authorization(monkeypatch, cache_root):
    """local_datasets takes any readable path, including one inside the operator's cache."""
    import routes.training as training_routes
    from fastapi import HTTPException

    snapshot = _make_repo(cache_root, "datasets--org--private-data")
    target = str(snapshot / "config.json")

    seen = {}

    def refuse(
        hf_token,
        *,
        repo_id,
        is_cached,
        repo_type = "model",
        offline = False,
    ):
        seen["repo_id"] = repo_id
        seen["repo_type"] = repo_type
        assert is_cached() is True
        return hf_token is False

    monkeypatch.setattr(training_routes, "cached_read_refused", refuse)

    # An API key with no token of its own: refused, and told which repository to get access to.
    with pytest.raises(HTTPException) as error:
        training_routes._refuse_unauthorized_cached_local_paths([target], False)
    assert error.value.status_code == 422
    assert seen == {"repo_id": "org/private-data", "repo_type": "dataset"}

    # A UI session entitled to the saved login is unaffected.
    training_routes._refuse_unauthorized_cached_local_paths([target], None)


def test_local_dataset_paths_outside_the_cache_are_left_alone(monkeypatch, cache_root, tmp_path):
    import routes.training as training_routes

    plain = tmp_path / "my_data.jsonl"
    plain.write_text("{}")

    def refuse(*args, **kwargs):  # pragma: no cover - must not be consulted
        raise AssertionError("a path outside the Hub cache must not be authorized against a repo")

    monkeypatch.setattr(training_routes, "cached_read_refused", refuse)
    training_routes._refuse_unauthorized_cached_local_paths([str(plain)], False)


def test_diffusion_child_drops_the_saved_login_for_a_tokenless_api_key(monkeypatch):
    """The diffusion child is spawned, so the scrub must happen inside it, before any Hub import."""
    from core.training import diffusion_training_service as service

    for key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.setenv(key, "operator-saved-login")
    monkeypatch.delenv("HF_TOKEN_PATH", raising = False)
    monkeypatch.setattr(service, "_run_diffusion_child", lambda **kwargs: None)

    service._default_target(event_queue = None, stop_queue = None, config = {"allow_ambient": False})

    import os

    assert "HF_TOKEN" not in os.environ
    assert "HF_HUB_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ
    assert os.environ["HF_TOKEN_PATH"] == os.devnull
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_diffusion_child_forwards_the_callers_own_token(monkeypatch):
    from core.training import diffusion_training_service as service

    monkeypatch.setenv("HF_TOKEN", "operator-saved-login")
    monkeypatch.setattr(service, "_run_diffusion_child", lambda **kwargs: None)

    service._default_target(
        event_queue = None,
        stop_queue = None,
        config = {"allow_ambient": False, "hf_token": "  caller-own-token  "},
    )

    import os

    # Trimmed, and the operator's credential is gone rather than sitting beside it.
    assert os.environ["HF_TOKEN"] == "caller-own-token"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


def test_diffusion_child_leaves_a_studio_session_alone(monkeypatch):
    """No allow_ambient, or True: the saved login is exactly what a UI start should use."""
    from core.training import diffusion_training_service as service

    monkeypatch.setenv("HF_TOKEN", "operator-saved-login")
    monkeypatch.setattr(service, "_run_diffusion_child", lambda **kwargs: None)

    import os

    # conftest points HF_TOKEN_PATH at the isolated cache for every test, so the claim is that it
    # is left where it was rather than redirected to devnull.
    token_path_before = os.environ.get("HF_TOKEN_PATH")
    for config in ({}, {"allow_ambient": True}):
        service._default_target(event_queue = None, stop_queue = None, config = config)
        assert os.environ["HF_TOKEN"] == "operator-saved-login"
        assert os.environ.get("HF_TOKEN_PATH") == token_path_before
        assert os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN") != "1"


def test_a_cached_diffusion_base_requires_caller_authorization(monkeypatch, cache_root):
    """Scrubbing the child environment does not protect a base that is ALREADY cached.

    `_preflight_gated_base` returns early for a local path and `_assert_trusted_base_model` accepts
    any real pipeline directory, so a tokenless API key could name the operator's cached private
    base and `from_pretrained` would read it off disk without asking for a credential.
    """
    import routes.training as training_routes
    from fastapi import HTTPException

    snapshot = _make_repo(cache_root, "models--org--private-sdxl")
    seen = {}

    def refuse(
        hf_token,
        *,
        repo_id,
        is_cached,
        repo_type = "model",
        offline = False,
    ):
        seen["repo_id"] = repo_id
        return hf_token is False and is_cached()

    monkeypatch.setattr(training_routes, "cached_read_refused", refuse)

    with pytest.raises(HTTPException) as error:
        training_routes._refuse_unauthorized_cached_local_paths(
            [str(snapshot), ""],
            False,
            "model",
        )
    assert error.value.status_code == 422
    assert error.value.detail["code"] == "hf_model_access_denied"
    assert seen["repo_id"] == "org/private-sdxl"

    # A Hub id for a base this disk does not hold has nothing to disclose.
    training_routes._refuse_unauthorized_cached_local_paths(
        ["black-forest-labs/FLUX.2-klein-4B", ""],
        False,
        "model",
    )


def test_a_remote_named_cached_base_is_authorized_when_the_hub_probe_fails_open(
    monkeypatch, cache_root
):
    """`org/private` names no path, and the HEAD that would cover it is best-effort.

    `_preflight_gated_base` treats an unreachable Hub as "not a denial", so on an offline box the
    scrubbed child still loaded the cached private copy. The disk is asked directly instead.
    """
    import routes.training as training_routes
    from fastapi import HTTPException

    _make_repo(cache_root, "models--org--private-sdxl")
    seen = {}

    def refuse(
        hf_token,
        *,
        repo_id,
        is_cached,
        repo_type = "model",
        offline = False,
    ):
        seen["repo_id"] = repo_id
        seen["repo_type"] = repo_type
        seen["is_cached"] = is_cached()
        return hf_token is False and seen["is_cached"]

    monkeypatch.setattr(training_routes, "cached_read_refused", refuse)

    with pytest.raises(HTTPException) as error:
        training_routes._refuse_unauthorized_cached_local_paths(
            ["org/private-sdxl"],
            False,
            "model",
        )
    assert error.value.status_code == 422
    assert seen == {"repo_id": "org/private-sdxl", "repo_type": "model", "is_cached": True}

    # A repo that is NOT on this disk has nothing to leak, so it is not refused.
    training_routes._refuse_unauthorized_cached_local_paths(
        ["org/never-downloaded"],
        False,
        "model",
    )
    assert seen["is_cached"] is False

    # A bare model name (no owner) is not a repo id; it must not be probed as one.
    seen.clear()
    training_routes._refuse_unauthorized_cached_local_paths(["gpt2", ""], False, "model")
    assert seen == {}


def test_the_worker_rebuilds_the_anonymous_sentinel_rather_than_none():
    """`or None` handed an API key the AMBIENT caller class inside the worker.

    The environment scrub only makes the worker's network traffic anonymous. Every disk-cache guard
    downstream discriminates on the ``False`` sentinel, and ``cache_reads_authorized(None)`` is
    True, so the operator's cached private weights stayed readable for the repos the route never
    authorized: a LoRA checkpoint's base, sibling scan targets, a fallback load target.
    """
    from core.training.worker import _worker_hf_token
    from hub.utils.hf_tokens import AmbientAuthorizedToken, cache_reads_authorized, is_anonymous

    # A tokenless API key: the sentinel, and it authorizes no cache read.
    token = _worker_hf_token({"allow_ambient": False})
    assert is_anonymous(token)
    assert cache_reads_authorized(token, repo_id = "org/private") is False

    # A tokenless UI session keeps the ambient login, exactly as before.
    assert _worker_hf_token({"allow_ambient": True}) is None
    assert _worker_hf_token({}) is None

    # An API key's own token stays a plain str, so it is probed rather than trusted outright.
    own = _worker_hf_token({"allow_ambient": False, "hf_token": "  hf_caller  "})
    assert own == "hf_caller"
    assert not isinstance(own, AmbientAuthorizedToken)

    # A UI session's own token is entitled to ambient and must not be demoted by the trim.
    ui = _worker_hf_token({"allow_ambient": True, "hf_token": " hf_ui "})
    assert isinstance(ui, AmbientAuthorizedToken) and ui == "hf_ui"
    assert cache_reads_authorized(ui, repo_id = "org/private") is True


def test_the_worker_no_longer_launders_the_sentinel_through_or_none():
    """A guard against the tempting `config.get("hf_token") or None` simplification coming back."""
    import inspect

    from core.training import worker

    source = inspect.getsource(worker)
    assert 'config.get("hf_token") or None' not in source


def test_an_interrupted_download_is_not_evidence_of_a_cached_read(monkeypatch, cache_root):
    """A repo DIRECTORY with no usable snapshot has disclosed nothing, so refusing it protects
    nothing and costs a legitimate caller its model.

    Reachable because the anonymous rescue in cached_read_refused needs /auth-check, which an
    HF_ENDPOINT mirror need not serve: there the refusal stands and a PUBLIC model is rejected
    with hf_model_access_denied for bytes that were never on disk.
    """
    from hub.utils.hf_cache_state import repo_cache_has_usable_snapshot

    metadata = ("config.json", "adapter_config.json")

    def usable(repo_id: str) -> bool:
        return repo_cache_has_usable_snapshot("model", repo_id, metadata)

    # Interrupted: the repo dir exists, nothing under it does.
    (cache_root / "models--org--interrupted").mkdir()
    assert usable("org/interrupted") is False

    # Present but empty: snapshots/<rev> with no metadata for the load to consume.
    (cache_root / "models--org--partial" / "snapshots" / "abc").mkdir(parents = True)
    assert usable("org/partial") is False

    # A real one still counts, so the guard has not been blunted.
    snapshot = _make_repo(cache_root, "models--org--usable")
    assert (snapshot / "config.json").exists()
    assert usable("org/usable") is True

    # Never downloaded at all: nothing to disclose and nothing to refuse.
    assert usable("org/absent") is False


def test_an_unreadable_repo_directory_still_counts_as_cached(monkeypatch, cache_root):
    """The predicate answers a guard, so its own failure must not open what it guards."""
    import os

    from hub.utils.hf_cache_state import repo_cache_has_usable_snapshot

    repo = cache_root / "models--org--locked"
    (repo / "snapshots").mkdir(parents = True)
    os.chmod(repo / "snapshots", 0o000)
    try:
        if os.access(repo / "snapshots", os.R_OK):
            pytest.skip("running as a user that ignores directory permissions")
        assert repo_cache_has_usable_snapshot("model", "org/locked", ("config.json",)) is True
    finally:
        os.chmod(repo / "snapshots", 0o755)


def test_the_diffusion_config_tolerates_the_new_policy_key():
    """allow_ambient rides the raw config dict; the trainer dataclass must ignore it."""
    from core.training.diffusion_train_common import _config_from_dict

    cfg = _config_from_dict(
        {
            "base_model": "org/sdxl",
            "data_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "allow_ambient": False,
        }
    )
    assert not hasattr(cfg, "allow_ambient")
