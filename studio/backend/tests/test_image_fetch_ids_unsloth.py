# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Hub repo an IMAGE load can fetch resolves to ``unsloth/*``.

Walks the family table (bases, named variants, prequant / TE checkpoints, sd-cli VAE and text encoders, train and
deploy bases), the frontend image catalog (every artifact row and the base each GGUF / bnb row takes its companions
from) and fails on any fetch id outside ``unsloth/*``. Fully offline: the cache probe is forced to "not cached", so the
answer is what a fresh install fetches, not what this host happens to hold.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference import diffusion_families as df
from core.inference.diffusion_families import (
    _FAMILIES,
    _UNGATED_MIRROR_PAIRS,
    canonical_base,
    detect_family,
    mirror_repo,
    prefer_ungated_mirror,
    sd_cpp_text_encoder_candidates,
    upstream_is_gated,
)

# The ONLY vendor repos an image load may still fetch, each with the licence clause that forbids a mirror.
_LICENCE_BLOCKED: dict[str, str] = {
    # Tencent Hunyuan Community License: "Territory" means the worldwide territory, excluding the territory of the
    # European Union, United Kingdom and South Korea. A public Hub mirror cannot honour a territorial limit.
    "hunyuanvideo-community/hunyuanimage-2.1-diffusers": "territory-restricted licence",
}

# Third-party DENOISER artifacts the catalog still offers. Not a companion fetch and not licence-blocked: there is
# simply no unsloth GGUF of FLUX.1 Krea dev yet. Its companions already come from unsloth/FLUX.1-Krea-dev.
_THIRD_PARTY_DENOISER_ROWS: frozenset[str] = frozenset({"quantstack/flux.1-krea-dev-gguf"})

# The base_model card tag of every quantised image repo Studio offers or supports: the companions (VAE, text
# encoders, tokenizer, scheduler) of a GGUF / bnb pick come from THIS repo, which no family table names for the
# card-tag-only variants (Edit, Edit-2509, Z-Image). Read off the Hub cards; the catalog test below fails when the
# catalog gains a quantised row this map does not cover.
_CARD_BASES: dict[str, str] = {
    "unsloth/Z-Image-Turbo-GGUF": "Tongyi-MAI/Z-Image-Turbo",
    "unsloth/Z-Image-Turbo-unsloth-bnb-4bit": "Tongyi-MAI/Z-Image-Turbo",
    "unsloth/Z-Image-GGUF": "Tongyi-MAI/Z-Image",
    "unsloth/Qwen-Image-2.1-GGUF": "Qwen/Qwen-Image-2.1",
    "unsloth/Qwen-Image-2.1-Turbo-GGUF": "Qwen/Qwen-Image-2.1-Turbo",
    "unsloth/Qwen-Image-2512-GGUF": "Qwen/Qwen-Image-2512",
    "unsloth/Qwen-Image-2512-unsloth-bnb-4bit": "Qwen/Qwen-Image-2512",
    "unsloth/Qwen-Image-GGUF": "Qwen/Qwen-Image",
    "unsloth/Qwen-Image-Edit-GGUF": "Qwen/Qwen-Image-Edit",
    "unsloth/Qwen-Image-Edit-2509-GGUF": "Qwen/Qwen-Image-Edit-2509",
    "unsloth/Qwen-Image-Edit-2511-GGUF": "Qwen/Qwen-Image-Edit-2511",
    "unsloth/Qwen-Image-Layered-GGUF": "Qwen/Qwen-Image-Layered",
    "unsloth/FLUX.1-schnell-GGUF": "black-forest-labs/FLUX.1-schnell",
    "unsloth/FLUX.1-dev-GGUF": "black-forest-labs/FLUX.1-dev",
    "QuantStack/FLUX.1-Krea-dev-GGUF": "black-forest-labs/FLUX.1-Krea-dev",
    "unsloth/FLUX.1-Kontext-dev-GGUF": "black-forest-labs/FLUX.1-Kontext-dev",
    "unsloth/FLUX.2-dev-GGUF": "black-forest-labs/FLUX.2-dev",
    "unsloth/FLUX.2-klein-4B-GGUF": "black-forest-labs/FLUX.2-klein-4B",
    "unsloth/FLUX.2-klein-9B-GGUF": "black-forest-labs/FLUX.2-klein-9B",
    "unsloth/FLUX.2-klein-base-4B-GGUF": "black-forest-labs/FLUX.2-klein-base-4B",
    "unsloth/FLUX.2-klein-base-9B-GGUF": "black-forest-labs/FLUX.2-klein-base-9B",
    "unsloth/flux-3-image-klein-GGUF": "black-forest-labs/flux-3-image-klein",
    "unsloth/flux-3-image-klein-base-GGUF": "black-forest-labs/flux-3-image-klein-base",
    "unsloth/MOSS-Image-SFT-GGUF": "OpenMOSS-Team/MOSS-Image-SFT",
    "unsloth/MOSS-Image-Base-GGUF": "OpenMOSS-Team/MOSS-Image-Base",
}

# These have no separate mirror repo: the official diffusers folders live inside the model's existing unsloth
# repo (-FP8 where one exists, else -GGUF).
_NEW_PAIRS = (
    ("Qwen/Qwen-Image-2.1-Turbo", "unsloth/Qwen-Image-2.1-Turbo-FP8"),
    ("Qwen/Qwen-Image-Edit", "unsloth/Qwen-Image-Edit-GGUF"),
    ("Qwen/Qwen-Image-Edit-2509", "unsloth/Qwen-Image-Edit-2509-GGUF"),
    ("Qwen/Qwen-Image-Layered", "unsloth/Qwen-Image-Layered-GGUF"),
    ("Tongyi-MAI/Z-Image", "unsloth/Z-Image-GGUF"),
    ("OpenMOSS-Team/MOSS-Image-SFT", "unsloth/MOSS-Image-SFT-FP8"),
    ("OpenMOSS-Team/MOSS-Image-Base", "unsloth/MOSS-Image-Base-FP8"),
)

# Repacks built by ComfyUI (or its unsloth/* copies). No image load may fetch from one, not even through the
# legacy-cache fallback.
_COMFY_MADE: frozenset[str] = frozenset(
    {
        "unsloth/flux-text-encoders",
        "unsloth/qwen-image-comfyui",
        "unsloth/flux.2-vae",
        "unsloth/flux.2-dev-comfyui",
        "unsloth/z-image-turbo-comfyui",
        "unsloth/flux.2-klein-9b-comfyui",
    }
)


def _is_comfy(repo: str) -> bool:
    r = repo.lower()
    return (
        r in _COMFY_MADE
        or r.startswith(("comfy-org/", "comfyanonymous/"))
        or "comfyui" in r
        or "comfy" in r.split("/")[0]
    )


@pytest.fixture(autouse = True)
def _fresh_install(monkeypatch):
    """Nothing is cached and the opt-out is off: the fetch id a fresh install uses."""
    monkeypatch.setattr(df, "_upstream_is_cached", lambda repo_id, files = None, **_k: False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def _fetch_id(repo: str) -> str:
    return prefer_ungated_mirror(repo)


def _allowed(repo: str) -> bool:
    if _is_comfy(repo):
        return False
    return repo.lower().startswith("unsloth/") or repo.lower() in _LICENCE_BLOCKED


def _family_fetch_ids() -> list[tuple[str, str, str, str]]:
    """(family, role, table id, fetch id) for every repo the image family table can hand a fetch site."""
    rows: list[tuple[str, str, str, str]] = []

    def add(fam, role, repo):
        if repo:
            rows.append((fam.name, role, repo, _fetch_id(repo)))

    for fam in _FAMILIES:
        add(fam, "base_repo", fam.base_repo)
        for repo in fam.named_variant_bases:
            add(fam, "named_variant", repo)
        for _scheme, repo in fam.prequant_repos:
            add(fam, "prequant", repo)
        for base, _scheme, repo in fam.prequant_variant_repos:
            add(fam, "prequant_variant_base", base)
            add(fam, "prequant_variant", repo)
        for _scheme, _component, repo, *_built_as in fam.te_prequant_repos:
            add(fam, "te_prequant", repo)
        for repo in fam.train_base_repos:
            add(fam, "train_base", repo)
        add(fam, "deploy_base", fam.deploy_base_repo)
        for training, inference in fam.deploy_base_repos:
            add(fam, "deploy_base", training)
            add(fam, "deploy_base", inference)
        for base, _config in fam.mlx_model_config_by_base:
            add(fam, "mlx_base", base)
    return rows


def _sd_cpp_fetch_ids() -> list[tuple[str, str, str, str]]:
    """The sd-cli VAE and every text-encoder candidate, through the backend's own per-repo fetch map."""
    from core.inference.sd_cpp_backend import _fetch_repo_map

    rows: list[tuple[str, str, str, str]] = []
    for fam in _FAMILIES:
        specs = []
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        specs.extend(sd_cpp_text_encoder_candidates(fam))
        fetch = _fetch_repo_map(specs, None)
        rows.extend((fam.name, f"sd_cpp:{kind}", repo, fetch[repo]) for repo, _f, kind in specs)
    return rows


def test_new_mirror_pairs_round_trip():
    for upstream, mirror in _NEW_PAIRS:
        assert (upstream, mirror) in _UNGATED_MIRROR_PAIRS
        assert mirror_repo(upstream) == mirror
        assert mirror_repo(upstream.lower()) == mirror
        assert canonical_base(mirror) == upstream
        # Ungated upstreams: the mirror only keeps the fetch in unsloth/*, it must not override a user's cache.
        assert not upstream_is_gated(upstream)
        assert _fetch_id(upstream) == mirror


def test_every_image_family_fetch_id_is_unsloth():
    bad = [row for row in _family_fetch_ids() if not _allowed(row[3])]
    assert not bad, "image family fetch ids outside unsloth/*:\n" + "\n".join(map(str, bad))


def test_every_sd_cpp_asset_fetch_id_is_unsloth():
    rows = _sd_cpp_fetch_ids()
    assert rows
    bad = [row for row in rows if not _allowed(row[3])]
    assert not bad, "sd-cli asset fetch ids outside unsloth/*:\n" + "\n".join(map(str, bad))


def test_moss_image_components_come_from_its_fp8_repos():
    fam = detect_family("OpenMOSS-Team/MOSS-Image-SFT")
    assert fam is not None and fam.name == "moss-image"
    assert fam.base_repo == "OpenMOSS-Team/MOSS-Image-SFT"
    for variant in ("SFT", "Base"):
        upstream, mirror = (
            f"OpenMOSS-Team/MOSS-Image-{variant}",
            f"unsloth/MOSS-Image-{variant}-FP8",
        )
        assert _fetch_id(upstream) == mirror
        assert df.mirror_hosts_artifacts(mirror)
    sd_cpp = {r[1]: r[3] for r in _sd_cpp_fetch_ids() if r[0] == "moss-image"}
    assert sd_cpp["sd_cpp:vae"] == "unsloth/MOSS-Image-SFT-FP8"
    assert fam.sd_cpp_tokenizer == ("unsloth/MOSS-Image-SFT-FP8", "tokenizer/tokenizer.json")


def test_layered_sd_cpp_vae_comes_from_the_mirror():
    """Layered's sd-cli VAE is read from its own base, so the base mirror is what keeps it inside unsloth/*."""
    fam = detect_family("", override = "qwen-image-layered")
    assert fam is not None and fam.sd_cpp_vae
    rows = [r for r in _sd_cpp_fetch_ids() if r[0] == "qwen-image-layered" and r[1] == "sd_cpp:vae"]
    assert rows and rows[0][3] == "unsloth/Qwen-Image-Layered-GGUF"


def _image_catalog() -> str:
    text = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/features/model-picker/components/model-selector/model-catalog.ts"
    ).read_text(encoding = "utf-8")
    images = text.split("export const IMAGE_CATALOG", 1)[1].split("export const VIDEO_CATALOG", 1)[
        0
    ]
    # Comments name withdrawn repos (the old QuantStack HunyuanImage GGUF); only live code counts.
    return re.sub(r"^\s*//.*$", "", images, flags = re.M)


def _catalog_rows() -> list[tuple[str, str]]:
    images = _image_catalog()
    rows = []
    for helper in ("gguf", "bf16Pipeline", "bf16Mirror", "fp8Pipeline", "bnb4bit", "bf16Single"):
        rows.extend((helper, repo) for repo in re.findall(helper + r'\(\s*"([^"]+)"', images))
    rows.extend(("prequantRepo", repo) for repo in re.findall(r'prequantRepo:\s*"([^"]+)"', images))
    assert len(rows) > 20, rows
    return rows


def _catalog_fetch_id(helper: str, repo: str) -> str:
    # bf16Mirror rows already load the unsloth copy; the rest reach the backend by their own id.
    if helper == "bf16Mirror":
        return f"unsloth/{repo.split('/', 1)[1]}"
    return _fetch_id(repo)


def test_every_image_catalog_row_fetches_from_unsloth():
    bad = []
    for helper, repo in _catalog_rows():
        fetch = _catalog_fetch_id(helper, repo)
        if helper == "bf16Mirror":
            # The row's id is only usable if the backend maps it back to the upstream it copies.
            assert mirror_repo(repo) == fetch, repo
        if _allowed(fetch) or repo.lower() in _THIRD_PARTY_DENOISER_ROWS:
            continue
        bad.append((helper, repo, fetch))
    assert not bad, f"image catalog rows fetching outside unsloth/*: {bad}"


def test_every_quantised_catalog_row_takes_its_companions_from_unsloth():
    quantised = {repo for helper, repo in _catalog_rows() if helper in ("gguf", "bnb4bit")}
    unmapped = sorted(
        r for r in quantised if r not in _CARD_BASES and not r.lower().startswith("ideogram-ai/")
    )
    assert (
        not unmapped
    ), f"add the base_model card tag of these catalog rows to _CARD_BASES: {unmapped}"
    bad = []
    for repo, base in _CARD_BASES.items():
        fam = detect_family(repo)
        assert fam is not None, repo
        assert detect_family(base) is fam, (repo, base)
        fetch = _fetch_id(base)
        if not _allowed(fetch):
            bad.append((repo, base, fetch))
    assert not bad, f"companion bases fetched outside unsloth/*: {bad}"


def test_licence_allowlist_is_still_needed():
    """The allowlist cannot outlive its reason: each entry must still be a fetch id with no mirror."""
    ids = {row[2].lower() for row in _family_fetch_ids()} | {r.lower() for _h, r in _catalog_rows()}
    for repo in _LICENCE_BLOCKED:
        assert repo in ids, f"{repo} is no longer fetched; drop it from _LICENCE_BLOCKED"
        assert mirror_repo(repo) is None, f"{repo} is mirrored now; drop it from _LICENCE_BLOCKED"
    rows = {r.lower() for _h, r in _catalog_rows()}
    for repo in _THIRD_PARTY_DENOISER_ROWS:
        assert (
            repo in rows
        ), f"the catalog no longer offers {repo}; drop it from _THIRD_PARTY_DENOISER_ROWS"


def test_no_image_asset_names_a_comfy_repack():
    """Neither the tables nor the fetch map may name a ComfyUI-built file, and the legacy-cache fallback must not be
    able to turn an image asset back into one."""
    rows = _family_fetch_ids() + _sd_cpp_fetch_ids()
    bad = [r for r in rows if _is_comfy(r[2]) or _is_comfy(r[3])]
    assert not bad, bad
    image_assets = {r[2].lower() for r in rows} | {r[3].lower() for r in rows}
    for key, legacy in df._SD_CPP_LEGACY_SOURCES.items():
        assert key not in image_assets, (key, legacy)
        assert key not in _COMFY_MADE, key


def test_sd_cpp_text_encoders_are_unsloth_ggufs_or_official_files():
    """Every sd-cli text encoder is an unsloth GGUF or an official upstream file (through its unsloth mirror)."""
    for fam, role, repo, fetch in _sd_cpp_fetch_ids():
        if role == "sd_cpp:vae":
            continue
        assert fetch.lower().startswith("unsloth/") and not _is_comfy(fetch), (
            fam,
            role,
            repo,
            fetch,
        )


def test_every_sampling_grid_index_is_read_from_an_unsloth_repo(monkeypatch):
    """The native route reads a variant base's model_index.json for its sample_sigmas grid outside the asset table,
    so it needs its own walk: every base a grid family can name must resolve to an unsloth repo on a fresh install."""
    from core.inference import diffusion_families as families
    from core.inference import sd_cpp_backend as bk

    monkeypatch.setattr(families, "_upstream_is_cached", lambda *a, **k: False)
    bases = {
        b
        for fam in families._FAMILIES
        if fam.name in bk._SAMPLE_SIGMAS_FAMILIES
        for b in (fam.base_repo, *(getattr(fam, "named_variant_bases", ()) or ()))
        if b
    }
    assert bases
    for base in sorted(bases):
        assert bk._grid_fetch_repo(base, None).lower().startswith("unsloth/"), base
