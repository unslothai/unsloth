# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sample-sigmas grid route (Qwen-Image-2.1-Turbo) on both transports."""

from __future__ import annotations

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_comfy_components as comfy_components
from core.inference import sd_cpp_backend as bk
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.sd_cpp_args import SdCppModelFiles
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

from .test_sd_cpp_backend import _FakeEngine, _FakeServer

TURBO_GRID = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
REPO = "Abiray/Qwen-Image-2.1-Turbo-GGUF"
GGUF = "qwen_image_2.1_turbo_Q4_K_M.gguf"
FAM = detect_family_for_pick(REPO, GGUF, None)
FILES = SdCppModelFiles(
    diffusion_model = f"/m/{GGUF}",
    vae = "/m/qwen_image_2.1_vae_bf16.safetensors",
    llm = "/m/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
)


def _state(
    *,
    grid,
    mode = "oneshot",
    server = None,
):
    return bk._SdState(
        repo_id = REPO,
        base_repo = FAM.base_repo,
        family = FAM,
        device = "cpu",
        files = FILES,
        sampling_method = FAM.sd_cpp_sampling_method,
        flow_shift = FAM.sd_cpp_flow_shift,
        sample_sigmas = grid,
        mode = mode,
        server = server,
    )


def _server_body(grid, steps):
    server = _FakeServer("sd-server")
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    b._state = _state(grid = grid, mode = "server", server = server)
    b.generate(prompt = "p", steps = steps, guidance = 1.0, seed = 3)
    return server.payloads[-1]


def _oneshot_extra_args(grid, steps):
    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _state(grid = grid)
    b.generate(prompt = "p", steps = steps, guidance = 1.0, seed = 3)
    return list(engine.calls[-1][3].get("extra_args") or [])


def test_family_is_qwen_image_21_without_a_flow_shift():
    assert FAM is not None and FAM.base_repo == "Qwen/Qwen-Image-2.1"
    # Custom sigmas replace sd.cpp's schedule outright, so there is no shift for them to fight.
    assert FAM.sd_cpp_flow_shift is None


def test_server_body_carries_the_grid_and_terminal_zero():
    body = _server_body(TURBO_GRID, 8)
    assert body["sample_params"]["custom_sigmas"] == [*TURBO_GRID, 0.0]
    assert body["sample_params"]["sample_steps"] == 8


def test_server_body_without_a_grid_is_unchanged():
    with_none = _server_body(None, 8)
    assert "custom_sigmas" not in with_none["sample_params"]
    with_grid = _server_body(TURBO_GRID, 8)
    with_grid["sample_params"].pop("custom_sigmas")
    assert with_grid == with_none


def test_oneshot_argv_carries_the_grid_and_terminal_zero():
    extra = _oneshot_extra_args(TURBO_GRID, 8)
    sigmas = [float(x) for x in extra[extra.index("--sigmas") + 1].split(",")]
    assert sigmas == [*TURBO_GRID, 0.0]


def test_oneshot_argv_without_a_grid_is_unchanged():
    assert "--sigmas" not in _oneshot_extra_args(None, 8)
    with_grid = _oneshot_extra_args(TURBO_GRID, 8)
    i = with_grid.index("--sigmas")
    assert with_grid[:i] + with_grid[i + 2 :] == _oneshot_extra_args(None, 8)


@pytest.mark.parametrize("steps", [4, 12])
def test_other_step_counts_resample_along_the_grid(steps):
    sigmas = _server_body(TURBO_GRID, steps)["sample_params"]["custom_sigmas"]
    # sd.cpp takes the step count from len(sigmas) - 1, so it must match the request.
    assert len(sigmas) == steps + 1
    assert sigmas[0] == 1.0 and sigmas[-2] == pytest.approx(TURBO_GRID[-1]) and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def _patch_hub(monkeypatch, *, card_base, indexes):
    calls = {"card": 0, "index": []}

    def _card(repo_id, token):
        calls["card"] += 1
        return card_base

    def _index(base, **kw):
        calls["index"].append((base, kw.get("local_files_only")))
        if base not in indexes:
            raise FileNotFoundError(base)
        return indexes[base]

    monkeypatch.setattr(diffusion_mod, "_hf_base_model", _card)
    monkeypatch.setattr(comfy_components, "read_model_index", _index)
    return calls


INDEXES = {
    "Qwen/Qwen-Image-2.1-Turbo": {"sample_sigmas": list(TURBO_GRID)},
    "Qwen/Qwen-Image-2.1": {"_class_name": "QwenImage21Pipeline"},
}


def test_community_gguf_finds_the_grid_through_its_card_base(monkeypatch):
    _patch_hub(monkeypatch, card_base = "Qwen/Qwen-Image-2.1-Turbo", indexes = INDEXES)
    grid, base = bk._base_sample_sigmas(
        REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = False
    )
    assert grid == TURBO_GRID
    # Reported as the load's base, so the Images form resets to Turbo's 8 steps, not 2.1's 25.
    assert base == "Qwen/Qwen-Image-2.1-Turbo"


def test_base_without_a_grid_and_untrusted_tags_give_none(monkeypatch):
    _patch_hub(monkeypatch, card_base = "Qwen/Qwen-Image-2.1", indexes = INDEXES)
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = False
        )[0]
        is None
    )
    # An untrusted card tag is dropped, as on the diffusers route.
    calls = _patch_hub(
        monkeypatch,
        card_base = "someone/evil",
        indexes = {**INDEXES, "someone/evil": {"sample_sigmas": list(TURBO_GRID)}},
    )
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = False
        )[0]
        is None
    )
    assert calls["index"] == [("Qwen/Qwen-Image-2.1", False)]


def test_explicit_base_and_cache_only_loads_skip_the_card(monkeypatch):
    calls = _patch_hub(monkeypatch, card_base = "Qwen/Qwen-Image-2.1-Turbo", indexes = INDEXES)
    assert (
        bk._base_sample_sigmas(
            REPO,
            "Qwen/Qwen-Image-2.1-Turbo",
            None,
            family = FAM.name,
            explicit_base = True,
            local_files_only = False,
        )[0]
        == TURBO_GRID
    )
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = True
        )[0]
        is None
    )
    assert calls["card"] == 0
    assert calls["index"][-1] == ("Qwen/Qwen-Image-2.1", True)


@pytest.mark.parametrize("raw", [[1.0, 1.2], [0.5, 0.9], "x", [float("nan")]])
def test_invalid_or_unreadable_grids_give_none(monkeypatch, raw):
    _patch_hub(monkeypatch, card_base = None, indexes = {"Qwen/Qwen-Image-2.1": {"sample_sigmas": raw}})
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = False
        )[0]
        is None
    )
    _patch_hub(monkeypatch, card_base = None, indexes = {})
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = FAM.name, explicit_base = False, local_files_only = False
        )[0]
        is None
    )


def test_other_families_never_read_the_card_or_index(monkeypatch):
    calls = _patch_hub(monkeypatch, card_base = "Qwen/Qwen-Image-2.1-Turbo", indexes = INDEXES)
    assert (
        bk._base_sample_sigmas(
            REPO, FAM.base_repo, None, family = "flux", explicit_base = False, local_files_only = False
        )[0]
        is None
    )
    assert calls["card"] == 0 and calls["index"] == []


def test_a_cache_only_reload_follows_the_picks_own_name(monkeypatch):
    calls = _patch_hub(monkeypatch, card_base = "Qwen/Qwen-Image-2.1-Turbo", indexes = INDEXES)
    kw = dict(family = FAM.name, explicit_base = False, local_files_only = True)
    # The OpenAI route's auto-switch reloads cache-only with no base: a Turbo-named pick keeps its grid and base.
    grid, base = bk._base_sample_sigmas(
        REPO, FAM.base_repo, None, named_base = "Qwen/Qwen-Image-2.1-Turbo", **kw
    )
    assert grid == TURBO_GRID and base == "Qwen/Qwen-Image-2.1-Turbo"
    # A plain 2.1 file from the same repo gets 2.1's schedule, whatever an earlier Turbo load resolved.
    assert bk._base_sample_sigmas(REPO, FAM.base_repo, None, **kw) == (None, FAM.base_repo)
    assert calls["card"] == 0


def test_a_local_turbo_gguf_without_a_card_takes_the_named_grid(monkeypatch):
    calls = _patch_hub(monkeypatch, card_base = None, indexes = INDEXES)
    grid, base = bk._base_sample_sigmas(
        REPO,
        FAM.base_repo,
        None,
        family = FAM.name,
        explicit_base = False,
        local_files_only = False,
        named_base = "Qwen/Qwen-Image-2.1-Turbo",
    )
    assert grid == TURBO_GRID and base == "Qwen/Qwen-Image-2.1-Turbo"
    assert calls["index"][-1][0] == "Qwen/Qwen-Image-2.1-Turbo"


@pytest.mark.parametrize(
    "gguf, base_repo, staged",
    [
        (GGUF, None, True),
        ("qwen_image_2.1_Q4_K_M.gguf", None, False),
        (GGUF, "Qwen/Qwen-Image-2.1", False),
        ("model-Q4_K_M.gguf", "Qwen/Qwen-Image-2.1-Turbo", True),
        ("model-Q4_K_M.gguf", "qwen/qwen-image-2.1-turbo", True),
    ],
)
def test_the_native_plan_stages_a_named_variants_index(monkeypatch, gguf, base_repo, staged):
    # Locality approves a cache-only load from this plan, so the index the load reads for its grid is in it.
    from core.inference.diffusion import DiffusionBackend

    repo = "someone/Qwen-Image-2.1-GGUF"
    b = SdCppDiffusionBackend(engine = _FakeEngine())
    monkeypatch.setattr(b, "_asset_specs", lambda *a, **k: [(repo, gguf, "diffusion_model")])
    monkeypatch.setattr(b, "_flux2_inner_dim", lambda *a, **k: None)
    monkeypatch.setattr(bk, "_assert_pick_is_not_speech", lambda *a, **k: None)
    monkeypatch.setattr(bk, "_fetch_repo_map", lambda specs, token: {repo: repo})
    monkeypatch.setattr(b, "_preflight_companion_repos", lambda *a, **k: None)
    monkeypatch.setattr(b, "_plan_file_sizes", lambda by_repo, token: {})
    monkeypatch.setattr(
        DiffusionBackend, "_hub_file_is_loadable", staticmethod(lambda *a, **k: False)
    )
    plan = b.download_plan(repo, gguf_filename = gguf, base_repo = base_repo, model_kind = "gguf")
    index = [e for e in plan["entries"] if e["repo_id"].lower() == "qwen/qwen-image-2.1-turbo"]
    assert (
        (index == [{**index[0], "files": ["model_index.json"], "checkpoint": False}])
        if staged
        else index == []
    )


def test_a_card_resolved_grid_base_is_linked_to_the_pick(monkeypatch):
    # begin_load linked only the family default; the base the card named holds the index a cache-only reload needs.
    from hub.utils import companion_assets

    from .test_sd_cpp_backend import _shared_setup_1

    links = []
    monkeypatch.setattr(
        companion_assets, "record_companion_link", lambda c, b: links.append((c, b))
    )
    monkeypatch.setattr(
        bk, "_base_sample_sigmas", lambda *a, **k: (TURBO_GRID, "Qwen/Qwen-Image-2.1-Turbo")
    )
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)
    b = SdCppDiffusionBackend()
    _shared_setup_1(b, _FakeEngine(), monkeypatch)
    b._load_token = 1
    b._run_load(
        repo_id = "someone/community-gguf",
        gguf_filename = "model-Q4_K_M.gguf",
        base = FAM.base_repo,
        fam = FAM,
        hf_token = None,
        _load_token = 1,
    )
    assert b._state is not None and b._state.sample_sigmas == TURBO_GRID
    assert ("someone/community-gguf", "Qwen/Qwen-Image-2.1-Turbo") in links


def test_a_cache_only_grid_read_falls_back_to_the_default_cache_root(monkeypatch):
    # The plan's locality check and _fetch_assets accept a copy in either root; the grid read must too.
    roots = []

    def _index(base, **kw):
        roots.append(kw.get("cache_dir"))
        if kw.get("cache_dir") is not None:
            raise FileNotFoundError(base)
        return {"sample_sigmas": list(TURBO_GRID)}

    monkeypatch.setattr(comfy_components, "read_model_index", _index)
    monkeypatch.setattr(diffusion_mod, "hub_cache_dir", lambda: "/live")
    grid, base = bk._base_sample_sigmas(
        REPO,
        FAM.base_repo,
        None,
        family = FAM.name,
        explicit_base = False,
        local_files_only = True,
        named_base = "Qwen/Qwen-Image-2.1-Turbo",
    )
    assert grid == TURBO_GRID and base == "Qwen/Qwen-Image-2.1-Turbo"
    assert roots == ["/live", None]
