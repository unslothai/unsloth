# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion backend.

The family helpers are pure functions, tested directly. The backend lifecycle is
exercised with ``torch`` / ``diffusers`` stubbed via ``sys.modules`` so no real
GPU, weights, or network access is needed (sub-second, CI-friendly).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
    DiffusionModelReplacedError,
    _LoadState,
    _base_file_downloaded,
    _clamp_max_side,
    _resolve_base_repo,
    _resolve_diffusion_compute_dtype,
)

# diffusion.py imports these lazily, so pull them in under the real torch before the fake-torch fixtures land.
import core.inference.diffusion_eager_patches  # noqa: E402,F401
import core.inference.diffusion_arch_patches  # noqa: E402,F401
from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    _GATED_MIRROR_PAIRS,
    _MIRROR_PAIRS,
    _UNGATED_MIRROR_PAIRS,
    assert_flux2_gguf_matches_base,
    canonical_base,
    detect_family,
    family_prequant_repo,
    load_identity,
    mirror_repo,
    pipeline_available_family_names,
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    sd_cpp_companion_only_repo_ids,
    supported_family_names,
    upstream_is_gated,
)


@pytest.fixture(autouse = True)
def _unmeasured_torchao(monkeypatch):
    """Pin "no measured torchao" so the installed release does not decide the offload tiers."""
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: None)
    # Studio's diffusers pin, so tests that opt into a measured torchao do not depend on the runner.
    monkeypatch.setattr(diffusion_memory, "_installed_diffusers_version", lambda: (0, 40))


def test_clamp_max_side_bounds_oversized_init():
    from PIL import Image

    out = _clamp_max_side(Image.new("RGB", (4096, 3072)), 2048)
    assert out.size == (2048, 1536)
    assert _clamp_max_side(Image.new("RGB", (1000, 4000)), 2048).size == (512, 2048)
    small = Image.new("RGB", (768, 512))
    assert _clamp_max_side(small, 2048) is small


def test_detect_family_from_repo_id():
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").name == "z-image"
    assert detect_family("unsloth/Z-Image-GGUF").name == "z-image"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").name == "flux.1"
    klein = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    assert klein.name == "flux.2-klein"
    assert klein.pipeline_class == "Flux2KleinPipeline"
    assert klein.cfg_kwarg == "guidance_scale"
    assert detect_family("unsloth/FLUX.2-klein-9B-GGUF").name == "flux.2-klein"
    dev = detect_family("unsloth/FLUX.2-dev-GGUF")
    assert dev.name == "flux.2-dev"
    assert dev.pipeline_class == "Flux2Pipeline"
    assert dev.base_repo == "black-forest-labs/FLUX.2-dev"
    assert detect_family("black-forest-labs/FLUX.2-dev").name == "flux.2-dev"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").cfg_kwarg == "true_cfg_scale"
    assert detect_family("unsloth/Z-Image-GGUF").cfg_kwarg == "guidance_scale"
    edit = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
    assert edit.name == "qwen-image-edit"
    assert edit.pipeline_class == "QwenImageEditPlusPipeline"
    assert edit.edit is True
    assert detect_family("unsloth/Qwen-Image-Edit-2509-GGUF").name == "qwen-image-edit"
    kontext = detect_family("unsloth/FLUX.1-Kontext-dev-GGUF")
    assert kontext.name == "flux.1-kontext"
    assert kontext.pipeline_class == "FluxKontextPipeline"
    assert kontext.edit is True
    assert kontext.cfg_kwarg == "guidance_scale"
    assert detect_family("unsloth/FLUX.1-dev-GGUF").name == "flux.1"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    krea2 = detect_family("krea/Krea-2-Turbo")
    assert krea2.name == "krea-2"
    assert krea2.pipeline_class == "Krea2Pipeline"
    assert krea2.transformer_class == "Krea2Transformer2DModel"
    assert krea2.cfg_kwarg == "guidance_scale"
    assert krea2.fp16_incompatible is True
    assert krea2.sd_cpp_text_encoders == ()
    assert detect_family("meta-llama/Llama-3-8B") is None


def test_detect_family_matches_reject_and_alias_by_segment():
    assert detect_family("/models/edited/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Z-Image-Edition-GGUF").name == "z-image"
    assert detect_family("/models/kontextual/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").name == "qwen-image-edit"
    assert detect_family("unsloth/FLUX.1-Kontext-dev-GGUF").name == "flux.1-kontext"
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF").name == "qwen-image-layered"
    assert detect_family("unsloth/FLUX.1-dev-Layered-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-2512-Inpaint") is None


def test_detect_family_edit_keyword_scoped_to_basename():
    from core.inference.diffusion_families import detect_family_for_pick

    assert detect_family("/models/edit") is None
    assert detect_family_for_pick("/models/edit", "Z-Image-Turbo-Q4.gguf").name == "z-image"
    assert detect_family_for_pick("/models/inpaint", "qwen-image-2512-Q4.gguf").name == "qwen-image"
    assert detect_family_for_pick("/models/misc", "Z-Image-Turbo-Layered-Q4.gguf") is None
    assert detect_family_for_pick("/models/misc", "Qwen-Image-Layered-Q4.gguf").name == (
        "qwen-image-layered"
    )


def test_detect_family_override():
    assert detect_family("local/path", override = "z-image").name == "z-image"
    assert detect_family("local/path", override = "zimage").name == "z-image"
    assert detect_family("local/path", override = "not-a-family") is None


def test_supported_family_names():
    names = supported_family_names()
    for expected in ("flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "z-image", "krea-2"):
        assert expected in names
    for name in names:
        assert detect_family("some/unknown-repo", override = name) is not None


def test_pipeline_available_names_filter_the_selector_without_importing(monkeypatch):
    # Status polls call this; importing pipeline classes there raced the loader's own import.
    import importlib.util

    import core.inference.diffusion_families as families

    blocked = {"krea-2", "flux.2-klein"}
    monkeypatch.setattr(families, "family_selectable", lambda fam: fam.name not in blocked)
    assert set(supported_family_names()) - set(pipeline_available_family_names()) == blocked

    def _strict(*_a, **_k):
        raise AssertionError("the selector must stay import-free")

    monkeypatch.undo()
    monkeypatch.setattr(families, "assert_pipeline_class_available", _strict)
    real_find_spec = importlib.util.find_spec
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "diffusers" else real_find_spec(name, *a, **k),
    )
    assert "flux.1" in pipeline_available_family_names()

    monkeypatch.setitem(sys.modules, "diffusers", None)
    assert pipeline_available_family_names() == ()


def test_resolve_base_repo():
    fam = detect_family("x", override = "z-image")
    assert resolve_base_repo(fam, None) == fam.base_repo
    assert resolve_base_repo(fam, "   ") == fam.base_repo
    assert resolve_base_repo(fam, "custom/base") == "custom/base"


def _no_cache(monkeypatch):
    """Report every upstream as uncached, so local files cannot mask the mirror decision."""
    monkeypatch.setattr(
        "core.inference.diffusion_families._upstream_is_cached", lambda repo_id, files = None: False
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def _all_cached(monkeypatch):
    """The opposite: every upstream already satisfies the load, so nothing is swapped."""
    monkeypatch.setattr(
        "core.inference.diffusion_families._upstream_is_cached", lambda repo_id, files = None: True
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def test_gated_mirror_table_round_trips():
    """Both directions, exact case: canonical_base must hand back a real repo id."""
    assert len(_MIRROR_PAIRS) == 25
    for upstream, mirror in _MIRROR_PAIRS:
        assert mirror_repo(upstream) == mirror
        assert canonical_base(mirror) == upstream
        assert mirror_repo(upstream.upper()) == mirror
    # HunyuanImage 2.1's licence excludes the EU, UK and South Korea, so it cannot be mirrored.
    hunyuan = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    assert mirror_repo(hunyuan) is None
    assert canonical_base(hunyuan) == hunyuan


def test_only_the_genuinely_gated_half_reads_as_gated():
    """Redirecting a fetch and needing credentials are different questions.

    Most of the table is mirrored to keep the fetch inside ``unsloth/*``, not to route around a
    gate, and callers that override a user's cache must key on the gate rather than on "a mirror
    exists". Klein base-4B is the one that makes this concrete: a default trainable base, which
    is mirrored, and which the Hub serves anonymously.
    """
    assert len(_GATED_MIRROR_PAIRS) == 12
    assert len(_UNGATED_MIRROR_PAIRS) == 13
    for upstream, _mirror in _GATED_MIRROR_PAIRS:
        assert upstream_is_gated(upstream), upstream
        assert upstream_is_gated(upstream.upper()), upstream
    for upstream, _mirror in _UNGATED_MIRROR_PAIRS:
        assert not upstream_is_gated(upstream), upstream
    assert mirror_repo("black-forest-labs/FLUX.2-klein-base-4B")
    assert not upstream_is_gated("black-forest-labs/FLUX.2-klein-base-4B")
    assert not upstream_is_gated("hunyuanvideo-community/HunyuanImage-2.1-Diffusers")
    assert not upstream_is_gated(None)


def test_no_mirror_is_a_companion_only_repo():
    """A mirror substitutes for the WHOLE base, so it must never be a components-only repo.

    ``prefer_ungated_mirror`` also fires on a plain bf16 pick, where the transformer is read from
    the base, so a mirror pointing at a repo with no denoiser turns a working load into a
    missing-weights error. The companion-only set is exactly that list of repos.
    """
    companions = sd_cpp_companion_only_repo_ids()
    for _upstream, mirror in _MIRROR_PAIRS:
        assert mirror.lower() not in companions, mirror


# Bases whose unsloth mirror is not on the Hub yet; a mirror row for a missing repo would 404.
_MIRRORS_NOT_YET_PUBLISHED: frozenset[str] = frozenset()


def test_every_third_party_bf16_pipeline_the_catalog_offers_is_mirrored():
    """Lookup is by exact id, so a variant the catalog offers is silently missed until listed.

    Adding a family's flagship is not enough: HiDream ships Full, Dev and Fast, and FLUX.2 klein
    ships 4B and base-4B. Each is its own repo id, so each needs its own row or the pick keeps
    fetching tens of GB from the vendor while the change claims to have stopped that. Read the
    catalog rather than restating the table, so a newly offered variant fails here instead of
    quietly bypassing the mirrors.
    """
    catalog = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/features/model-picker/components/model-selector/model-catalog.ts"
    ).read_text(encoding = "utf-8")
    # Image side only: slice at VIDEO_CATALOG so new video entries do not fail this.
    images = catalog.split("export const IMAGE_CATALOG", 1)[1].split(
        "export const VIDEO_CATALOG", 1
    )[0]
    offered = set(re.findall(r'bf16Pipeline\(\s*"([^"]+)"', images))
    mirrored = {u.lower() for u, _m in _MIRROR_PAIRS}
    missing = sorted(
        repo
        for repo in offered
        if not repo.lower().startswith("unsloth/")
        and "hunyuan" not in repo.lower()
        and repo.lower() not in mirrored
        and repo.lower() not in _MIRRORS_NOT_YET_PUBLISHED
    )
    assert not missing, f"catalog offers these vendor bases with no unsloth mirror: {missing}"
    for repo in _MIRRORS_NOT_YET_PUBLISHED:
        assert (
            repo not in mirrored
        ), f"{repo} is in the mirror table now; drop it from _MIRRORS_NOT_YET_PUBLISHED"
        assert repo in {
            o.lower() for o in offered
        }, f"the catalog no longer offers {repo}; drop it from _MIRRORS_NOT_YET_PUBLISHED"


def test_the_qwen_2512_mirror_covers_the_card_tag_route(monkeypatch):
    """#8001: the 2512 companions come from a repo the family table never names.

    ``unsloth/Qwen-Image-2512-GGUF`` carries ``base_model: Qwen/Qwen-Image-2512`` and
    ``_resolve_base_repo`` trusts that tag, so the fetch lands on the vendor repo whatever the
    family default says. The mirror is the only thing that redirects it.
    """
    _no_cache(monkeypatch)
    assert mirror_repo("Qwen/Qwen-Image-2512") == "unsloth/Qwen-Image-2512"
    assert prefer_ungated_mirror("Qwen/Qwen-Image-2512") == "unsloth/Qwen-Image-2512"
    assert mirror_repo("Qwen/Qwen-Image") == "unsloth/Qwen-Image"
    assert canonical_base("unsloth/Qwen-Image-2512") == "Qwen/Qwen-Image-2512"


def test_prefer_ungated_mirror_swaps_gated_bases(monkeypatch):
    _no_cache(monkeypatch)
    for upstream, mirror in _MIRROR_PAIRS:
        assert prefer_ungated_mirror(upstream) == mirror
    hunyuan = "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    assert prefer_ungated_mirror(hunyuan) == hunyuan


def test_prefer_ungated_mirror_declines(monkeypatch):
    """Each decline path lands on the upstream id, i.e. exactly today's behaviour."""
    gated = "black-forest-labs/FLUX.1-dev"

    _no_cache(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert prefer_ungated_mirror(gated) == gated

    _all_cached(monkeypatch)
    assert prefer_ungated_mirror(gated) == gated


def test_the_opt_out_maps_a_direct_mirror_pick_back_to_its_upstream(monkeypatch):
    """The picker lists mirror ids, so the opt-out must also undo a mirror picked directly."""
    mirror, upstream = "unsloth/FLUX.1-dev", "black-forest-labs/FLUX.1-dev"
    _no_cache(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert prefer_ungated_mirror(mirror) == upstream
    _all_cached(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    for files in (None, [], ["model_index.json"], ["vae/diffusion_pytorch_model.safetensors"]):
        assert prefer_ungated_mirror(mirror, files = files) == upstream
    _no_cache(monkeypatch)
    assert prefer_ungated_mirror(mirror) == mirror


def test_a_local_base_directory_is_never_mirrored(monkeypatch, tmp_path):
    """A path that exists on disk is not a Hub id, so it must survive the swap untouched.

    A user can clone a base into a relative dir named exactly like the vendor id. The loaders
    resolve such a base locally (``Path(base).exists()``), but several take that branch after the
    swap, so rewriting it would send the load to the Hub and ignore the on-disk files.
    """
    gated = "black-forest-labs/FLUX.1-dev"
    _no_cache(monkeypatch)
    assert mirror_repo(gated) == "unsloth/FLUX.1-dev"
    assert prefer_ungated_mirror(gated) == "unsloth/FLUX.1-dev"

    local = tmp_path / gated
    (local / "vae").mkdir(parents = True)
    (local / "model_index.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert prefer_ungated_mirror(gated) == gated
    assert prefer_ungated_mirror(gated, files = ["model_index.json"]) == gated
    assert prefer_ungated_mirror(str(local)) == str(local)


def test_mirrored_base_still_trips_the_flux2_shape_guard():
    """The regression the two-helper split exists for.

    The guard fails OPEN on an unmapped base, so a mirror id reaching ``_FLUX2_BASE_INNER_DIM``
    would silence it. Assert the RAISE: a disabled guard passes any weaker check.
    """
    fam = detect_family("x", override = "flux.2-klein")
    assert fam is not None and fam.name.startswith("flux.2")

    def _reader_for(inner_dim):
        class _Reader:
            def __init__(self, _path):
                self.tensors = [
                    type(
                        "T",
                        (),
                        {"name": "double_stream_modulation_img.lin.weight", "shape": (inner_dim,)},
                    )()
                ]

        return _Reader

    import gguf

    original = gguf.GGUFReader
    try:
        gguf.GGUFReader = _reader_for(3072)
        for base in ("black-forest-labs/FLUX.2-klein-9B", "unsloth/FLUX.2-klein-9B"):
            with pytest.raises(ValueError, match = "klein"):
                assert_flux2_gguf_matches_base(fam, base, "some-klein-4b.gguf")
        gguf.GGUFReader = _reader_for(4096)
        for base in ("black-forest-labs/FLUX.2-klein-9B", "unsloth/FLUX.2-klein-9B"):
            assert assert_flux2_gguf_matches_base(fam, base, "some-klein-9b.gguf") is None
    finally:
        gguf.GGUFReader = original


def test_family_prequant_repo_accepts_either_id():
    """prequant_variant_repos is keyed on upstream ids, so a mirror must hit the same entry."""
    fam = detect_family("x", override = "flux.1")
    for scheme in ("int8", "fp8"):
        upstream = family_prequant_repo(fam, scheme, "black-forest-labs/FLUX.1-dev")
        mirrored = family_prequant_repo(fam, scheme, "unsloth/FLUX.1-dev")
        assert upstream == mirrored == "unsloth/FLUX.1-dev-FP8"


def _fake_hub_cache(
    monkeypatch,
    tmp_path,
    repo_id,
    files,
    *,
    revision = "abc123",
    ref = None,
):
    """Lay out ``files`` as a cached snapshot revision of ``repo_id`` and point the live cache
    setting at it, so the mirror decision reads a tree the test controls. ``ref`` writes
    refs/main, as huggingface_hub does for a branch download."""
    root = tmp_path / f"models--{repo_id.replace('/', '--')}"
    rev = root / "snapshots" / revision
    for name in files:
        path = rev / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
    rev.mkdir(parents = True, exist_ok = True)
    if ref is not None:
        (root / "refs").mkdir(parents = True, exist_ok = True)
        (root / "refs" / "main").write_text(ref, encoding = "utf-8")
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)


def test_a_superseded_cached_revision_does_not_disable_the_mirror(monkeypatch, tmp_path):
    """Only the revision refs/main names can satisfy a gated fetch: on a 401 the HEAD call fails
    and hf_hub_download resolves refs/<revision> to ONE commit, returning its pointer or
    re-raising. A complete but superseded revision must therefore not read as cached."""
    from core.inference.diffusion_families import _upstream_is_cached

    gated = "black-forest-labs/FLUX.1-dev"
    wanted = ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]
    _fake_hub_cache(monkeypatch, tmp_path, gated, wanted, revision = "old")
    _fake_hub_cache(monkeypatch, tmp_path, gated, ["model_index.json"], revision = "new", ref = "new")
    assert _upstream_is_cached(gated, wanted) is False
    assert prefer_ungated_mirror(gated, files = wanted) == "unsloth/FLUX.1-dev"

    _fake_hub_cache(monkeypatch, tmp_path, gated, wanted, revision = "old", ref = "old")
    assert _upstream_is_cached(gated, wanted) is True
    assert prefer_ungated_mirror(gated, files = wanted) == gated


def test_a_stray_upstream_file_does_not_disable_the_mirror(monkeypatch, tmp_path):
    """The decline is "the load is satisfiable from cache", not "some blob exists".

    An interrupted (or previously tokened) pull leaves a config behind. Treating that as cached
    pinned every later load to the gated upstream and re-raised the 401 the mirror removes.
    """
    from core.inference.diffusion_families import _upstream_is_cached

    gated = "black-forest-labs/FLUX.1-dev"
    _fake_hub_cache(monkeypatch, tmp_path, gated, ["model_index.json", "vae/config.json"])
    assert _upstream_is_cached(gated) is False
    assert prefer_ungated_mirror(gated) == "unsloth/FLUX.1-dev"

    _fake_hub_cache(
        monkeypatch,
        tmp_path,
        gated,
        ["model_index.json", "vae/diffusion_pytorch_model.safetensors"],
    )
    assert _upstream_is_cached(gated) is True
    assert prefer_ungated_mirror(gated) == gated

    wanted = ["vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"]
    assert _upstream_is_cached(gated, wanted) is False
    assert prefer_ungated_mirror(gated, files = wanted) == "unsloth/FLUX.1-dev"
    assert prefer_ungated_mirror(gated, files = wanted[:1]) == gated


def test_a_repack_split_across_the_two_cache_roots_still_counts(monkeypatch, tmp_path):
    """A pair split by a cache-folder change is held by neither root alone, but IS reusable.

    The callers that pass ``other_root`` fetch with ``reuse_other_cache_root``, which resolves
    each file through whichever root holds it. Asking each root for the whole set therefore calls
    a split pair absent and re-pulls several GB the two roots already have between them (offline,
    it fails outright). Reachable with an interrupted download either side of the change: the file
    fetched before it stays in the old root, the one fetched after lands in the new one.
    """
    from huggingface_hub import constants

    from core.inference.diffusion_families import _upstream_is_cached, prefer_cached_legacy_source

    repack = "Comfy-Org/z_image_turbo"
    mirror = "unsloth/Z-Image-Turbo-ComfyUI"
    first, second = "split_files/vae/ae.safetensors", "split_files/text_encoders/te.safetensors"
    live, other = tmp_path / "live", tmp_path / "other"

    def seed(root, name):
        rev = root / f"models--{repack.replace('/', '--')}" / "snapshots" / ("d" * 40)
        (rev / name).parent.mkdir(parents = True, exist_ok = True)
        (rev / name).write_bytes(b"x")
        refs = root / f"models--{repack.replace('/', '--')}" / "refs"
        refs.mkdir(parents = True, exist_ok = True)
        (refs / "main").write_text("d" * 40, encoding = "utf-8")

    seed(other, first)
    seed(live, second)
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(live))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))

    wanted = (first, second)
    assert _upstream_is_cached(repack, wanted, other_root = True) is True
    assert prefer_cached_legacy_source(mirror, wanted) == repack

    assert _upstream_is_cached(repack, wanted) is False

    assert (
        _upstream_is_cached(repack, (*wanted, "split_files/absent.safetensors"), other_root = True)
        is False
    )


def test_the_two_root_union_does_not_relax_the_revision_rule(monkeypatch, tmp_path):
    """Per-file across roots, whole-set within one: a superseded revision contributes nothing.

    Only the revision refs/main names can satisfy a fetch, and that stays true per root. Without
    the split the union would let an old complete revision in one root paper over the new
    incomplete one in the other.
    """
    from huggingface_hub import constants

    from core.inference.diffusion_families import _upstream_is_cached

    repack = "Comfy-Org/z_image_turbo"
    first, second = "split_files/vae/ae.safetensors", "split_files/text_encoders/te.safetensors"
    live, other = tmp_path / "live", tmp_path / "other"

    def seed(root, name, revision, ref):
        base = root / f"models--{repack.replace('/', '--')}"
        rev = base / "snapshots" / revision
        (rev / name).parent.mkdir(parents = True, exist_ok = True)
        (rev / name).write_bytes(b"x")
        (base / "refs").mkdir(parents = True, exist_ok = True)
        (base / "refs" / "main").write_text(ref, encoding = "utf-8")

    seed(other, first, "old", ref = "new")
    seed(live, second, "old", ref = "new")
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(live))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))

    assert _upstream_is_cached(repack, (first, second), other_root = True) is False


def test_the_union_never_borrows_across_revisions_inside_one_root(monkeypatch, tmp_path):
    """Split across ROOTS is reusable; split across SNAPSHOTS of one root is not.

    A commit-pinned download leaves no refs/main, so every snapshot is a candidate. Answering the
    set name by name would then let an old snapshot complete a newer one inside the same root,
    which no fetch can do: a fetch that lands in a root lands in ONE revision of it. Studio never
    pins a revision itself, but the cache is shared with anything else that does.
    """
    from huggingface_hub import constants

    from core.inference.diffusion_families import _upstream_is_cached

    repack = "Comfy-Org/z_image_turbo"
    first, second = "split_files/vae/ae.safetensors", "split_files/text_encoders/te.safetensors"
    live, other = tmp_path / "live", tmp_path / "other"
    other.mkdir()

    def seed(root, name, revision):
        rev = root / f"models--{repack.replace('/', '--')}" / "snapshots" / revision
        (rev / name).parent.mkdir(parents = True, exist_ok = True)
        (rev / name).write_bytes(b"x")

    seed(live, first, "a" * 40)
    seed(live, second, "b" * 40)
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(live))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))

    assert _upstream_is_cached(repack, (first, second), other_root = True) is False

    seed(live, second, "a" * 40)
    assert _upstream_is_cached(repack, (first, second), other_root = True) is True


def test_te_prequant_equivalence_group_accepts_a_mirrored_base():
    """The T5-XXL artifact is shared across the FLUX.1 releases through an equivalence group of
    UPSTREAM ids. A mirrored base is a different string, so without normalising it the pre-cast
    encoder is refused and the load falls back to the dense multi-GB download."""
    from core.inference.diffusion_te_prequant import te_base_equivalent

    ckpt = "black-forest-labs/FLUX.1-schnell"
    assert te_base_equivalent(ckpt, "black-forest-labs/FLUX.1-dev") is True
    assert te_base_equivalent(ckpt, "unsloth/FLUX.1-dev") is True
    assert te_base_equivalent("unsloth/FLUX.1-schnell", "unsloth/FLUX.1-dev") is True
    assert te_base_equivalent(ckpt, "unsloth/FLUX.2-dev") is False


def test_resolve_local_gguf_child(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    assert resolve_local_gguf_child(tmp_path, "model.gguf") == (tmp_path / "model.gguf").resolve()
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "/etc/passwd")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "../secret.gguf")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "..\\secret.gguf")
    with pytest.raises(FileNotFoundError):
        resolve_local_gguf_child(tmp_path, "missing.gguf")


def test_resolve_local_gguf_child_blocks_symlink_escape(tmp_path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"secret")
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        (repo / "model.gguf").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(repo, "model.gguf")


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"torch.{self._name}"

    __str__ = __repr__


class _FakeGenerator:
    def __init__(self, device = None) -> None:
        self.device = device
        self.manual = None

    def seed(self) -> int:
        return 4242

    def manual_seed(self, value: int):
        self.manual = value
        return self


class _FakeImage:
    """Stand-in for a generated PIL image (the route persists it; here we only
    count how many come back)."""


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None
        self.offloaded = False
        self.sequential_offloaded = False
        self.vae_tiled = False
        self.vae_sliced = False
        self.last_kwargs = None

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self, device = None) -> None:
        self.offloaded = True
        self.offload_device = device

    def enable_sequential_cpu_offload(self, device = None) -> None:
        self.sequential_offloaded = True
        self.offload_device = device

    def enable_vae_tiling(self) -> None:
        self.vae_tiled = True

    def enable_vae_slicing(self) -> None:
        self.vae_sliced = True

    # Explicit signature (not just **kwargs) so generate()'s signature-gated guards fire.
    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        cfg_trunc_ratio = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "callback_on_step_end": callback_on_step_end,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            "cfg_trunc_ratio": cfg_trunc_ratio,
            **kwargs,
        }
        # Mirror diffusers batching: a prompt list gives one image each, fanned out by num_images_per_prompt.
        n = kwargs.get("num_images_per_prompt", 1)
        if isinstance(prompt, list):
            n *= len(prompt)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakePipeline:
    last: dict = {}
    last_single_file: dict = {}

    @classmethod
    def from_pretrained(cls, base, **kwargs):
        _FakePipeline.last = {"base": base, **kwargs}
        return _FakePipe()

    @classmethod
    def from_single_file(cls, path, **kwargs):
        _FakePipeline.last_single_file = {"path": path, **kwargs}
        return _FakePipe()


class _FakeTransformer:
    last: dict = {}

    @classmethod
    def from_single_file(cls, path, **kwargs):
        _FakeTransformer.last = {"path": path, **kwargs}
        return object()


class _FakeImg2ImgPipe:
    """An img2img pipeline call: records the image-conditioned kwargs. Its signature
    declares image/strength but NOT width/height, mirroring real img2img pipelines
    (which derive the output size from the input image)."""

    last_kwargs: dict = {}

    def __call__(
        self,
        *,
        prompt = None,
        image = None,
        strength = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        _FakeImg2ImgPipe.last_kwargs = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            **kwargs,
        }
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakeImg2ImgPipeline:
    built_from: object = None
    from_pipe_kwargs: dict = {}
    recast_dtype: object = None

    def to(self, *args, **kwargs):
        _FakeImg2ImgPipeline.recast_dtype = kwargs.get("dtype")
        return self

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeImg2ImgPipeline.built_from = base_pipe
        _FakeImg2ImgPipeline.from_pipe_kwargs = kwargs
        _FakeImg2ImgPipeline.recast_dtype = None
        # from_pipe's terminal cast, which is what makes the call site's class choice observable.
        cls().to(dtype = kwargs.get("dtype") or kwargs.get("torch_dtype") or "float32")
        return _FakeImg2ImgPipe()


class _FakeInpaintPipe:
    """An inpaint pipeline call: records image + mask_image + strength. Real inpaint
    pipelines take both an init image and a grayscale mask and derive output size from
    the input, so width/height are not in its signature."""

    last_kwargs: dict = {}

    def __call__(
        self,
        *,
        prompt = None,
        image = None,
        mask_image = None,
        strength = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        _FakeInpaintPipe.last_kwargs = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask_image,
            "strength": strength,
            **kwargs,
        }
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakeInpaintPipeline:
    built_from: object = None
    recast_dtype: object = None

    def to(self, *args, **kwargs):
        _FakeInpaintPipeline.recast_dtype = kwargs.get("dtype")
        return self

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.built_from = base_pipe
        _FakeInpaintPipeline.recast_dtype = None
        cls().to(dtype = kwargs.get("dtype") or kwargs.get("torch_dtype") or "float32")
        return _FakeInpaintPipe()


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = _FakeDtype("bfloat16")
    torch.float16 = _FakeDtype("float16")
    torch.float32 = _FakeDtype("float32")
    torch.Generator = _FakeGenerator
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)
    torch.inference_mode = lambda: contextlib.nullcontext()
    torch.no_grad = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    diffusers.ZImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.ZImageInpaintPipeline = _FakeInpaintPipeline
    diffusers.QwenImagePipeline = _FakePipeline
    diffusers.QwenImageTransformer2DModel = _FakeTransformer
    diffusers.QwenImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.QwenImageInpaintPipeline = _FakeInpaintPipeline
    diffusers.QwenImageEditPlusPipeline = _FakePipeline
    diffusers.Ideogram4Pipeline = _FakePipeline
    diffusers.Ideogram4Transformer2DModel = _FakeTransformer
    diffusers.Lumina2Pipeline = _FakePipeline
    diffusers.Lumina2Transformer2DModel = _FakeTransformer
    diffusers.StableDiffusionXLPipeline = _FakePipeline
    diffusers.UNet2DConditionModel = _FakeTransformer
    diffusers.StableDiffusionXLImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.StableDiffusionXLInpaintPipeline = _FakeInpaintPipeline

    monkeypatch.setattr(
        "core.inference.diffusion.load_ideogram4_pipeline",
        lambda repo_id, dtype, hf_token = None, check_cancelled = None: _FakePipe(),
    )

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: None)
    _FakePipeline.last = {}
    _FakePipeline.last_single_file = {}
    _FakeTransformer.last = {}
    _FakeImg2ImgPipeline.built_from = None
    _FakeImg2ImgPipe.last_kwargs = {}
    _FakeInpaintPipeline.built_from = None
    _FakeInpaintPipe.last_kwargs = {}
    yield


_LOAD_DEFAULTS = dict(gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image")


def _write_pipeline(
    root,
    class_name = "TestPipeline",
    **components,
):
    weight = components.pop("weight", "diffusion_pytorch_model.safetensors")
    (root / "transformer").mkdir(parents = True, exist_ok = True)
    manifest = {"_class_name": class_name, "transformer": ["diffusers", "Transformer2DModel"]}
    (root / "model_index.json").write_text(json.dumps({**manifest, **components}))
    (root / "transformer" / "config.json").write_text("{}")
    (root / "transformer" / weight).write_bytes(b"x")


def _load_into(backend, tmp_path, **overrides):
    """``load_pipeline`` on ``tmp_path`` over the z-image defaults; writes no checkpoint file."""
    return backend.load_pipeline(str(tmp_path), **{**_LOAD_DEFAULTS, **overrides})


def _loaded_backend(tmp_path, **overrides):
    """A backend loaded off a stub checkpoint written into ``tmp_path``.

    ``overrides`` replace the z-image defaults and are forwarded to ``load_pipeline``.
    """
    filename = overrides.get("gguf_filename", _LOAD_DEFAULTS["gguf_filename"])
    (tmp_path / filename).write_bytes(b"weights")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path, **overrides)
    return backend


def test_generate_refuses_when_the_model_was_replaced_since_the_snapshot(fake_runtime, tmp_path):
    """The guard in isolation: a snapshot naming another model is refused, typed (#9448)."""
    backend = _loaded_backend(tmp_path)
    st = backend.status()
    loaded = load_identity(st["repo_id"], st["base_repo"], st["family"])

    stale = load_identity("other/model", st["base_repo"], st["family"])
    with pytest.raises(DiffusionModelReplacedError) as replaced:
        backend.generate(prompt = "stale request", expected_load = stale)
    assert replaced.value.expected == stale
    assert replaced.value.actual == loaded

    gen = backend.generate(prompt = "fresh request", expected_load = loaded, steps = 4)
    assert len(gen["images"]) == 1

    gen2 = backend.generate(prompt = "legacy caller", steps = 4)
    assert len(gen2["images"]) == 1


def test_generate_refuses_a_replacement_that_committed_while_it_waited(fake_runtime, tmp_path):
    """The reported interleaving end to end (#9448).

    A load drops its teardown fence for the whole construction of the new model while still
    holding the generation lock, so a generate arriving there used to block, then denoise on
    the NEW model with the snapshot's steps/guidance.
    """
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for d in (old_dir, new_dir):
        d.mkdir()
        (d / "model.gguf").write_bytes(b"weights")
    load_kwargs = dict(gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image")

    backend = DiffusionBackend()
    backend.load_pipeline(str(old_dir), **load_kwargs)
    st = backend.status()
    snapshot = load_identity(st["repo_id"], st["base_repo"], st["family"])

    reached, release = threading.Event(), threading.Event()
    original = _FakeTransformer.from_single_file.__func__

    def _parked(cls, path, **kwargs):
        reached.set()
        assert release.wait(30)
        return original(cls, path, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_FakeTransformer, "from_single_file", classmethod(_parked))
        loader = threading.Thread(
            target = backend.load_pipeline, args = (str(new_dir),), kwargs = load_kwargs, daemon = True
        )
        loader.start()
        assert reached.wait(30)
        assert backend._teardown_waiters == 0

        outcome = {}

        def _generate():
            try:
                outcome["ok"] = backend.generate(
                    prompt = "a sloth", steps = 9, guidance = 0.0, expected_load = snapshot
                )
            except BaseException as exc:  # noqa: BLE001 (the exception IS the assertion)
                outcome["err"] = exc

        gen = threading.Thread(target = _generate, daemon = True)
        gen.start()
        gen.join(1.0)
        assert gen.is_alive()

        release.set()
        loader.join(30)
        gen.join(30)

    assert not gen.is_alive()
    assert backend.status()["repo_id"] == str(new_dir)
    assert "ok" not in outcome, "denoised on the replacement with the snapshot's parameters"
    assert isinstance(outcome["err"], DiffusionModelReplacedError)
    assert (outcome["err"].expected.repo_id, outcome["err"].actual.repo_id) == (
        str(old_dir),
        str(new_dir),
    )


def test_the_same_path_reloaded_under_a_different_base_is_a_replacement(fake_runtime, tmp_path):
    """repo_id is not a load identity (#9448).

    base_repo and family_override are settable per load, so one local checkpoint reloads as a
    different model. Pinning the path alone let a FLUX.1-dev request reach a schnell pipeline.
    """
    backend = _loaded_backend(tmp_path, base_repo = "black-forest-labs/FLUX.1-dev")
    st = backend.status()
    snapshot = load_identity(st["repo_id"], st["base_repo"], st["family"])

    _load_into(backend, tmp_path, base_repo = "black-forest-labs/FLUX.1-schnell")
    assert backend.status()["repo_id"] == snapshot.repo_id
    with pytest.raises(DiffusionModelReplacedError) as replaced:
        backend.generate(prompt = "p", steps = 28, guidance = 3.5, expected_load = snapshot)
    assert replaced.value.actual.base_repo == "black-forest-labs/FLUX.1-schnell"


def test_load_generate_unload_gguf(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()

    status = _load_into(
        backend, tmp_path, hf_token = "hf_secret", display_repo_id = "Org/pinned-z-image"
    )
    assert status["loaded"] is True
    assert status["family"] == "z-image"
    assert status["base_repo"] == "base/repo"
    assert status["device"] == "cpu"
    assert status["dtype"] == "float32"
    assert status["cpu_offload"] is False
    assert _FakeTransformer.last["path"] == str((tmp_path / "model.gguf").resolve())
    assert _FakeTransformer.last["subfolder"] == "transformer"
    assert _FakeTransformer.last["token"] == "hf_secret"
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last

    gen = backend.generate(
        prompt = "a sloth", negative_prompt = "blurry", width = 512, height = 512, steps = 4, guidance = 3.0
    )
    assert gen["seed"] == 4242
    assert gen["repo_id"] == "Org/pinned-z-image"
    assert len(gen["images"]) == 1
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 3.0 and call["true_cfg_scale"] is None
    assert call["negative_prompt"] == "blurry"
    assert callable(call["callback_on_step_end"])

    gen2 = backend.generate(prompt = "again", seed = 99)
    assert gen2["seed"] == 99

    batch = backend.generate(prompt = "batch", seed = 7, batch_size = 3)
    assert len(batch["images"]) == 3 and batch["seed"] == 7
    assert batch["seeds"] == [7, 8, 9]

    assert backend.unload()["loaded"] is False
    assert backend.is_loaded is False


def test_gguf_status_reports_selected_quant_instead_of_only_compute_dtype(fake_runtime, tmp_path):
    filename = "z-image-turbo-Q8_0.gguf"
    (tmp_path / filename).write_bytes(b"weights")
    backend = DiffusionBackend()

    status = _load_into(backend, tmp_path, gguf_filename = filename)

    assert status["dtype"] == "float32"
    assert status["gguf_variant"] == "Q8_0"
    assert status["gguf_filename"] == filename
    assert backend.unload()["gguf_variant"] is None


def test_generate_progress_active_during_setup(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend(tmp_path, hf_token = "hf_secret")

    seen = {}

    def fake_apply(self, state, loras, cancel):
        seen["progress"] = self.generate_progress()

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", fake_apply)

    assert backend.generate_progress()["active"] is False

    gen = backend.generate(prompt = "a sloth", steps = 4)
    assert len(gen["images"]) == 1

    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 4
    assert seen["progress"]["step"] == 0

    assert backend.generate_progress()["active"] is False


def test_generate_progress_cleared_on_setup_error(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend(tmp_path, hf_token = "hf_secret")

    def boom(self, state, loras, cancel):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", boom)

    with pytest.raises(RuntimeError, match = "setup failed"):
        backend.generate(prompt = "a sloth", steps = 4)

    assert backend.generate_progress()["active"] is False


def test_generate_progress_active_through_compile_cache_save(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = _loaded_backend(tmp_path, hf_token = "hf_secret")

    seen = {}

    def fake_save(ctx, *, logger = None):
        seen["progress"] = backend.generate_progress()
        return True

    monkeypatch.setattr(dmod.compile_cache, "register_shape", lambda *a, **k: None)
    monkeypatch.setattr(dmod.compile_cache, "save_async", fake_save)

    gen = backend.generate(prompt = "a sloth", steps = 4)
    assert len(gen["images"]) == 1
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 4
    assert backend.generate_progress()["active"] is False


def test_dense_speed_auto_defers_compile_to_third_generation(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(
        dmod, "apply_attention_backend", lambda pipe, backend, logger = None, target = None: backend
    )
    monkeypatch.setattr(
        dmod,
        "select_attention_backend",
        lambda target, requested, speed_active = False, family = None, speed_unset = False: (
            "_native_cudnn" if speed_active else None
        ),
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_into(
        backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    assert status["speed_mode"] == "off"
    assert status["resolved"]["speed_mode"]["value"] == "deferred"
    assert status["resolved"]["speed_mode"]["source"] == "auto"

    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    assert backend.status()["speed_mode"] == "off"
    backend.generate(prompt = "three")
    status3 = backend.status()
    assert status3["speed_mode"] == "default"
    assert "compiled" in status3["speed_optims"]
    assert status3["attention_backend"] == "_native_cudnn"
    assert status3["resolved"]["speed_mode"]["value"] == "default"

    backend.unload()
    status_off = _load_into(
        backend,
        tmp_path,
        gguf_filename = "model.safetensors",
        family_override = "qwen-image",
        speed_mode = "off",
    )
    assert status_off["resolved"]["speed_mode"]["value"] == "off"
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    assert backend.status()["speed_mode"] == "off"
    backend.unload()


def test_deferred_speed_stays_off_when_only_an_explicit_tier_may_compile(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    seen = []
    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod, "fp16_compile_explicit_only", lambda target: seen.append(target) or True
    )
    engaged = []
    monkeypatch.setattr(
        DiffusionBackend, "_engage_deferred_speed", lambda self, state: engaged.append(1)
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_into(
        backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    assert seen
    assert status["resolved"]["speed_mode"]["value"] == "off"
    for p in ("one", "two", "three"):
        backend.generate(prompt = p)
    assert engaged == []
    assert backend.status()["speed_mode"] == "off"
    backend.unload()


def test_deferred_speed_skips_when_lora_requested(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)
    monkeypatch.setattr(DiffusionBackend, "_apply_loras", lambda self, state, loras, cancel: None)

    backend = _loaded_backend(
        tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    backend.generate(prompt = "three", loras = [("adapter", 1.0)])
    assert engaged == []
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_skips_while_adapter_attached(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)

    def fake_apply(self, state, loras, cancel):
        specs = [(i, w) for (i, w) in (loras or []) if w != 0]
        state.pipe._unsloth_loras = tuple(specs)

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", fake_apply)

    backend = _loaded_backend(
        tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    backend.generate(prompt = "one", loras = [("adapter", 1.0)])
    backend.generate(prompt = "two", loras = [("adapter", 1.0)])
    backend.generate(prompt = "three")
    assert engaged == []
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_refreshes_the_cuda_graph_entry(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    graphs_on = {"value": True}
    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {
            "compiled": k.get("speed_mode") == "default",
            "cuda_graph": k.get("speed_mode") == "default" and graphs_on["value"],
        },
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image")
    assert backend.status()["resolved"]["cuda_graph"]["value"] == "off"
    for p in ("one", "two", "three"):
        backend.generate(prompt = p)
    status = backend.status()
    assert "cuda_graph" in status["speed_optims"]
    assert status["resolved"]["cuda_graph"]["value"] == "on"
    assert "captured" in status["resolved"]["cuda_graph"]["reason"]

    backend.unload()
    graphs_on["value"] = False
    _load_into(backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image")
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    status = backend.status()
    assert "cuda_graph" not in status["speed_optims"]
    assert status["resolved"]["cuda_graph"]["value"] == "off"
    backend.unload()


def test_deferred_speed_preserves_explicit_attention(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(
        dmod, "apply_attention_backend", lambda pipe, backend, logger = None, target = None: backend
    )

    def fake_select(
        target,
        requested,
        speed_active = False,
        family = None,
        speed_unset = False,
    ):
        if requested in (None, "", "auto"):
            return "_native_cudnn" if speed_active else None
        if str(requested).lower() in ("native", "sdpa"):
            return None
        return requested

    monkeypatch.setattr(dmod, "select_attention_backend", fake_select)
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    _load_into(
        backend,
        tmp_path,
        gguf_filename = "model.safetensors",
        family_override = "qwen-image",
        attention_backend = "native",
    )
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    backend.generate(prompt = "three")
    status = backend.status()
    assert status["speed_mode"] == "default"
    assert "compiled" in status["speed_optims"]
    assert status["attention_backend"] is None
    assert status["resolved"]["attention_backend"]["value"] == "native"
    assert status["resolved"]["attention_backend"]["source"] == "explicit"

    backend.unload()
    _load_into(backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image")
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    assert backend.status()["attention_backend"] == "_native_cudnn"
    backend.unload()


def _tiny_png_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 30, 30)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_generate_img2img_uses_from_pipe(fake_runtime, tmp_path):
    """An init_image routes generate() through the family's img2img pipeline, built via
    Pipeline.from_pipe around the loaded pipe (no reload), with image + strength passed
    and width/height dropped (the img2img pipe derives size from the input image)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    assert backend.status()["workflows"] == ["txt2img", "img2img", "upscale", "inpaint", "outpaint"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a car at sunset",
        steps = 4,
        guidance = 0.0,
        seed = 3,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert len(out["images"]) == 1
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    # from_pipe gets no dtype, via a class that drops its terminal cast, so modules keep theirs.
    assert "torch_dtype" not in _FakeImg2ImgPipeline.from_pipe_kwargs
    assert "dtype" not in _FakeImg2ImgPipeline.from_pipe_kwargs
    assert _FakeImg2ImgPipeline.recast_dtype is None
    call = _FakeImg2ImgPipe.last_kwargs
    assert call["image"] is not None
    assert call["strength"] == 0.5
    assert "width" not in call and "height" not in call

    backend.generate(prompt = "plain", steps = 4, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


class _Component:
    """Records the dtype it is left at; a quantized one refuses a cast, as ModelMixin does."""

    def __init__(
        self,
        dtype,
        quantized = False,
    ):
        self.dtype = dtype
        self.is_quantized = quantized

    def to(
        self,
        device = None,
        dtype = None,
    ):
        if dtype is not None:
            if self.is_quantized:
                raise ValueError("Casting a quantized model to a new `dtype` is unsupported.")
            self.dtype = dtype
        return self


class _Resident:
    """The loaded text-to-image pipeline the workflow pipes are built from."""

    def __init__(self, *, quantized_transformer):
        self.components = {
            "text_encoder": _Component("bfloat16"),
            "transformer": _Component("bfloat16", quantized = quantized_transformer),
            "vae": _Component("bfloat16"),
        }

    def dtypes(self):
        return {name: c.dtype for name, c in self.components.items()}


class _RecastingPipeline:
    """``from_pipe`` as every diffusers Unsloth can install implements it: reuse the resident
    components, then cast them in name order to float32 unless the caller named a dtype."""

    seen: dict = {}
    recasts = True

    def __init__(self, **components):
        self.components = components
        for name, component in components.items():
            setattr(self, name, component)

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        for name in sorted(self.components):
            component = self.components[name]
            if hasattr(component, "to"):
                component.to(dtype = dtype)
        return self

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _RecastingPipeline.seen = dict(kwargs)
        new = cls(**dict(base_pipe.components, **kwargs))
        if cls.recasts:
            new.to(dtype = kwargs.get("dtype") or kwargs.get("torch_dtype") or "float32")
        return new


class _PreservingPipeline(_RecastingPipeline):
    """``from_pipe`` once upstream keeps the loaded dtype instead of defaulting to float32."""

    recasts = False


def _torch_with_dtype(monkeypatch):
    """A torch stub whose ``dtype`` is a real class, so ``isinstance`` means something."""
    torch = types.ModuleType("torch")

    class dtype:  # noqa: N801 -- mirrors torch.dtype
        pass

    torch.dtype = dtype
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_no_recast_class_drops_every_shape_of_dtype_cast(monkeypatch):
    """Every form of dtype ``.to()`` accepts is ignored; every form of device is forwarded."""
    from core.inference.diffusion import _no_recast_pipeline_class

    torch = _torch_with_dtype(monkeypatch)
    fp32 = torch.dtype()

    class _Recorder:
        def __init__(self):
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self

    pipe = _no_recast_pipeline_class(_Recorder)()

    assert pipe.to(fp32) is pipe
    assert pipe.to(dtype = fp32) is pipe
    assert pipe.calls == []

    pipe.to("cuda")
    pipe.to("cuda", fp32)
    pipe.to(device = "cuda", dtype = fp32)
    assert pipe.calls == [(("cuda",), {}), (("cuda",), {}), ((), {"device": "cuda"})]


def test_no_recast_class_is_cached_and_keeps_identity():
    """One subclass per pipeline class, and it still passes as the family's own class --
    which the rest of the backend and the diffusers internals go on assuming."""
    from core.inference.diffusion import _no_recast_pipeline_class

    cls = _no_recast_pipeline_class(_RecastingPipeline)
    assert _no_recast_pipeline_class(_RecastingPipeline) is cls
    assert issubclass(cls, _RecastingPipeline)
    assert cls.__name__ == _RecastingPipeline.__name__


@pytest.mark.parametrize("pipeline_cls", [_RecastingPipeline, _PreservingPipeline])
@pytest.mark.parametrize("quantized_transformer", [True, False])
def test_from_pipe_no_recast_leaves_every_component_at_its_loaded_dtype(
    pipeline_cls, quantized_transformer
):
    """The build succeeds and no component moves off bfloat16.

    The two quantization cases fail differently against a recasting from_pipe: a quantized
    denoiser makes the cast raise, and since components are cast in name order the text
    encoder is float32 already by then, so catching the error is not a fix; unquantized
    raises nothing at all and the whole pipeline is silently doubled in place. The two
    pipeline classes cover a from_pipe that recasts and one that has stopped, so an upstream
    fix landing under Unsloth cannot change the outcome."""
    from core.inference.diffusion import DiffusionBackend

    resident = _Resident(quantized_transformer = quantized_transformer)
    before = resident.dtypes()

    pipe = DiffusionBackend._from_pipe_no_recast(resident, pipeline_cls)

    assert resident.dtypes() == before == {n: "bfloat16" for n in before}
    assert pipe.transformer is resident.components["transformer"]
    assert pipe.vae is resident.components["vae"]


def test_from_pipe_no_recast_names_no_dtype_and_forwards_extras():
    """The helper names no dtype, and passes a ControlNet along."""
    from core.inference.diffusion import DiffusionBackend

    resident = _Resident(quantized_transformer = True)
    DiffusionBackend._from_pipe_no_recast(resident, _RecastingPipeline)
    assert _RecastingPipeline.seen == {}

    controlnet = _Component("bfloat16")
    pipe = DiffusionBackend._from_pipe_no_recast(
        resident, _RecastingPipeline, controlnet = controlnet
    )
    assert _RecastingPipeline.seen == {"controlnet": controlnet}
    assert pipe.controlnet is controlnet


def test_from_pipe_no_recast_does_not_swallow_errors():
    """A real assembly failure must surface: catching the quantized-cast error would hide
    that from_pipe had already cast every component ahead of the one that refused."""
    from core.inference.diffusion import DiffusionBackend

    class _Broken:
        @classmethod
        def from_pipe(cls, base_pipe, **kwargs):
            raise ValueError("Casting a quantized model to a new `dtype` is unsupported")

    with pytest.raises(ValueError, match = "Casting a quantized model"):
        DiffusionBackend._from_pipe_no_recast(_Resident(quantized_transformer = True), _Broken)


def test_generate_img2img_unsupported_family_raises(fake_runtime, tmp_path, monkeypatch):
    """A family with no image-conditioning at all (no img2img/inpaint/edit/reference) rejects
    an init_image with a clear error rather than failing deep in the pipeline."""
    from core.inference.diffusion_families import DiffusionFamily

    plain = DiffusionFamily(
        name = "plain-test",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "base/repo",
    )
    monkeypatch.setattr(
        "core.inference.diffusion.detect_family_for_pick",
        lambda repo_id, gguf_filename = None, override = None: plain,
    )
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path, family_override = None)
    assert backend.status()["workflows"] == ["txt2img"]
    with pytest.raises(ValueError, match = "img2img"):
        backend.generate(prompt = "x", steps = 4, init_image = _tiny_png_b64())


def test_generate_rejects_conditioning_without_init_image(fake_runtime, tmp_path):
    """mask / upscale / reference all need an input image; without one they must raise a
    clear ValueError rather than silently degrading to txt2img."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path)
    with pytest.raises(ValueError, match = "mask_image requires"):
        backend.generate(prompt = "x", steps = 4, mask_image = _mask_b64(64))
    with pytest.raises(ValueError, match = "upscale requires"):
        backend.generate(prompt = "x", steps = 4, upscale = 2.0)
    with pytest.raises(ValueError, match = "reference_images require"):
        backend.generate(prompt = "x", steps = 4, reference_images = [_tiny_png_b64()])


def test_generate_rejects_reference_on_unsupported_family(fake_runtime, tmp_path):
    """A non-reference family rejects reference_images instead of silently dropping them."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    with pytest.raises(ValueError, match = "Reference images are not supported"):
        backend.generate(
            prompt = "x",
            steps = 4,
            init_image = _tiny_png_b64(),
            reference_images = [_tiny_png_b64()],
        )


def test_generate_upscale_enlarges_and_low_strength(fake_runtime, tmp_path):
    """An init_image + upscale factor routes generate() through the family's img2img
    pipeline (hires fix): the source is enlarged to size*factor (rounded to /16) before the
    denoise, the strength defaults low, and the factor is capped so a huge value can't OOM."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    assert "upscale" in backend.status()["workflows"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a crisp photo",
        steps = 4,
        guidance = 0.0,
        seed = 3,
        init_image = _tiny_png_b64(),
        upscale = 2.0,
    )
    assert len(out["images"]) == 1
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    call = _FakeImg2ImgPipe.last_kwargs
    assert call["image"].size == (128, 128)
    assert call["strength"] == 0.35

    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 99.0,
    )
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (256, 256)

    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 1.5,
        strength = 0.2,
    )
    assert _FakeImg2ImgPipe.last_kwargs["strength"] == 0.2
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (96, 96)


def _png_b64(side: int) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (10, 20, 30)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_rejects_oversized(fake_runtime, tmp_path):
    """An input image larger than the per-side cap is rejected with a clear error (protects
    img2img / inpaint / reference from decompression-bomb / OOM inputs), not a 500."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    with pytest.raises(ValueError, match = "too large"):
        backend.generate(prompt = "x", steps = 4, init_image = _png_b64(4112))


def test_upscale_output_is_capped(fake_runtime, tmp_path):
    """Upscale bounds the absolute output side to 2048 even when input*factor exceeds it, so a
    large upload at 4x can't OOM the VAE/transformer."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 4.0)
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (2048, 2048)


def _mask_b64(side: int) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("L", (side, side), 0)
    for y in range(side // 4, 3 * side // 4):
        for x in range(side // 4, 3 * side // 4):
            img.putpixel((x, y), 255)
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_img2img_snaps_non_multiple_of_16(fake_runtime, tmp_path):
    """An odd-sized img2img upload (not divisible by 16) is auto-resized to the nearest
    multiple of 16 so the pipeline's divisibility check passes instead of erroring."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png_b64(186), strength = 0.5)
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (192, 192)


def test_inpaint_snaps_image_and_mask_together(fake_runtime, tmp_path):
    """Inpaint snaps the odd-sized input to /16 AND resizes the mask to match, so the image
    and mask stay aligned (a mismatch would crash the inpaint pipeline)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _png_b64(186),
        mask_image = _mask_b64(186),
        strength = 0.5,
    )
    assert _FakeInpaintPipe.last_kwargs["image"].size == (192, 192)
    assert _FakeInpaintPipe.last_kwargs["mask_image"].size == (192, 192)


def test_generate_reference_uses_loaded_pipe_at_slider_size(fake_runtime, tmp_path):
    """A reference family (FLUX.2-klein) advertises txt2img + reference, and a generate with
    an init_image passes it as the loaded pipe's `image` arg (no from_pipe, no strength) while
    the output size stays the REQUESTED slider size (the pipe resizes the reference itself)."""
    import diffusers

    diffusers.Flux2KleinPipeline = _FakePipeline
    diffusers.Flux2KleinInpaintPipeline = _FakeInpaintPipeline
    diffusers.Flux2Transformer2DModel = _FakeTransformer
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path, family_override = "flux.2-klein")
    assert backend.status()["workflows"] == ["txt2img", "reference", "inpaint"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a portrait in this style",
        steps = 6,
        guidance = 4.0,
        seed = 5,
        width = 768,
        height = 512,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert len(out["images"]) == 1
    call = loaded_pipe.last_kwargs
    assert call["image"] is not None
    assert call["width"] == 768 and call["height"] == 512
    assert "strength" not in call
    assert "mask_image" not in call
    assert call["guidance_scale"] == 4.0

    backend.generate(
        prompt = "combine these",
        steps = 6,
        seed = 9,
        width = 1024,
        height = 1024,
        init_image = _tiny_png_b64(),
        reference_images = [_tiny_png_b64(), _tiny_png_b64()],
    )
    img_arg = loaded_pipe.last_kwargs["image"]
    assert isinstance(img_arg, list) and len(img_arg) == 3

    backend.generate(
        prompt = "repaint here",
        steps = 6,
        seed = 2,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.8,
    )
    assert _FakeInpaintPipeline.built_from is loaded_pipe
    assert _FakeInpaintPipe.last_kwargs["mask_image"] is not None
    assert _FakeInpaintPipe.last_kwargs["strength"] == 0.8

    backend.generate(prompt = "just text", steps = 6, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


def _tiny_mask_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("L", (64, 64), 0)
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), 255)
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_generate_inpaint_uses_from_pipe(fake_runtime, tmp_path):
    """An init_image + mask_image routes generate() through the family's inpaint pipeline,
    built via Pipeline.from_pipe around the loaded pipe (no reload), with the decoded image
    + mask + strength passed through and width/height dropped (size derives from the input)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a red door",
        steps = 4,
        guidance = 0.0,
        seed = 5,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.7,
    )
    assert len(out["images"]) == 1
    assert _FakeInpaintPipeline.built_from is loaded_pipe
    assert _FakeInpaintPipeline.recast_dtype is None
    assert _FakeImg2ImgPipeline.built_from is None
    call = _FakeInpaintPipe.last_kwargs
    assert call["image"] is not None and call["mask_image"] is not None
    assert call["strength"] == 0.7
    assert "width" not in call and "height" not in call


def test_image_conditioned_passes_image_size_not_slider(fake_runtime, tmp_path):
    """When the workflow pipe DOES accept width/height, an image-conditioned call must pass
    the INPUT IMAGE's size, never the txt2img slider size -- otherwise a non-slider-sized
    input (e.g. a 1536px outpaint canvas with a 1024 slider) mismatches the latents
    ("tensor a (128) must match tensor b (192)"). Covers Transform + Extend with any size."""
    import base64
    import io

    from PIL import Image

    class _SizePipe:
        last: dict = {}

        def __call__(
            self,
            *,
            prompt = None,
            image = None,
            strength = None,
            width = None,
            height = None,
            negative_prompt = None,
            callback_on_step_end = None,
            guidance_scale = None,
            true_cfg_scale = None,
            **kwargs,
        ):
            _SizePipe.last = {"width": width, "height": height}
            n = kwargs.get("num_images_per_prompt", 1)
            return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])

    class _SizePipeline:
        @classmethod
        def from_pipe(cls, base_pipe, **kwargs):
            return _SizePipe()

    import diffusers

    diffusers.ZImageImg2ImgPipeline = _SizePipeline
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (96, 64), (10, 20, 30)).save(buf, format = "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    backend.generate(prompt = "x", steps = 4, width = 1024, height = 1024, init_image = b64, strength = 0.5)
    assert _SizePipe.last == {"width": 96, "height": 64}


def test_compile_shape_dims_follow_workflow():
    """_compile_shape_dims mirrors generate()'s width/height derivation: slider size for
    txt2img / reference / controlnet, the input image's size for the image-conditioned
    workflows (whose forward runs at init_pil.size, whatever the sliders say)."""
    from PIL import Image

    from core.inference.diffusion import _compile_shape_dims

    img = Image.new("RGB", (96, 64), (10, 20, 30))
    assert _compile_shape_dims("txt2img", None, 1024, 512) == (1024, 512)
    assert _compile_shape_dims("reference", img, 1024, 512) == (1024, 512)
    assert _compile_shape_dims("controlnet", None, 768, 768) == (768, 768)
    for wf in ("img2img", "inpaint", "upscale", "edit"):
        assert _compile_shape_dims(wf, img, 1024, 512) == (96, 64)


def test_register_shape_uses_actual_forward_dims(fake_runtime, tmp_path, monkeypatch):
    """The static compile-cache manifest must record the dims the forward ACTUALLY ran
    at: an image-conditioned generate derives its output size from the input image, so
    registering the slider values would mark a never-compiled shape as covered while the
    truly-used shape never re-dirties/saves the bundle (warm restarts keep paying its
    compile)."""
    from core.inference import diffusion as diff

    registered: list = []
    monkeypatch.setattr(
        diff.compile_cache,
        "register_shape",
        lambda ctx, shape, *, static: registered.append(tuple(shape)),
    )
    monkeypatch.setattr(diff.compile_cache, "save_async", lambda ctx, *, logger = None: True)
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)
    backend.generate(prompt = "x", steps = 4, width = 1024, height = 512, seed = 1)
    assert registered[-1] == (1024, 512, 1)
    backend.generate(
        prompt = "x",
        steps = 4,
        width = 1024,
        height = 512,
        seed = 1,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert registered[-1] == (64, 64, 1)


def test_edit_family_uses_own_pipeline_and_requires_image(fake_runtime, tmp_path):
    """An instruction-editing family (Qwen-Image-Edit) exposes only the 'edit' workflow,
    runs the image through its OWN loaded pipeline (no from_pipe), and rejects a call with
    no input image."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend, tmp_path, base_repo = "Qwen/Qwen-Image-Edit-2511", family_override = "qwen-image-edit"
    )
    assert backend.status()["workflows"] == ["edit"]
    loaded_pipe = backend._state.pipe

    out = backend.generate(
        prompt = "make it night",
        steps = 8,
        guidance = 4.0,
        seed = 1,
        init_image = _tiny_png_b64(),
    )
    assert len(out["images"]) == 1
    assert backend._state.pipe is loaded_pipe
    assert _FakeImg2ImgPipeline.built_from is None and _FakeInpaintPipeline.built_from is None
    assert loaded_pipe.last_kwargs.get("image") is not None

    with pytest.raises(ValueError, match = "image"):
        backend.generate(prompt = "make it night", steps = 8)


@pytest.mark.parametrize(
    "gguf_filename, expected",
    [
        ("qwen-image-edit-2509-Q6_K.gguf", {"zero_cond_t": False}),
        ("qwen_image_edit_2509_Q4_K_M.gguf", {"zero_cond_t": False}),
        ("qwen-image-edit-Q4_K_M.gguf", {"zero_cond_t": False}),
        ("qwen-image-edit-2511-Q4_K_M.gguf", {}),
        ("model.gguf", {}),
    ],
)
def test_qwen_edit_gguf_builds_on_its_variant_config(
    fake_runtime, tmp_path, gguf_filename, expected
):
    """2509 / original Edit override the 2511 companion's zero_cond_t; 2511 and unnamed files do not."""
    (tmp_path / gguf_filename).write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend,
        tmp_path,
        gguf_filename = gguf_filename,
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        family_override = "qwen-image-edit",
    )
    assert _FakeTransformer.last["config"].endswith("/Qwen-Image-Edit-2511")
    assert {k: v for k, v in _FakeTransformer.last.items() if k == "zero_cond_t"} == expected


def test_non_qwen_edit_gguf_gets_no_config_override(fake_runtime, tmp_path):
    (tmp_path / "qwen-image-2512-Q4_K_M.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend, tmp_path, gguf_filename = "qwen-image-2512-Q4_K_M.gguf", family_override = "qwen-image"
    )
    assert "zero_cond_t" not in _FakeTransformer.last


def test_qwen_edit_display_repo_id_names_the_variant(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    _load_into(
        DiffusionBackend(),
        tmp_path,
        gguf_filename = "model.gguf",
        display_repo_id = "unsloth/Qwen-Image-Edit-2509-GGUF",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        family_override = "qwen-image-edit",
    )
    assert _FakeTransformer.last.get("zero_cond_t") is False


def _qwen_edit_dense_route(backend, monkeypatch):
    from core.inference import diffusion as dmod

    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("base") if "base" in k else a[2])
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    return attempted


@pytest.mark.parametrize(
    "gguf_filename, takes_dense",
    [("qwen-image-edit-2509-Q6_K.gguf", False), ("qwen-image-edit-2511-Q4_K_M.gguf", True)],
)
def test_qwen_edit_variant_gguf_never_runs_the_2511_dense_transformer(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback, gguf_filename, takes_dense
):
    backend = DiffusionBackend()
    attempted = _qwen_edit_dense_route(backend, monkeypatch)
    (tmp_path / gguf_filename).write_bytes(b"x")
    _load_into(
        backend,
        tmp_path,
        gguf_filename = gguf_filename,
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        family_override = "qwen-image-edit",
        transformer_quant = "fp8",
    )
    assert bool(attempted) is takes_dense
    assert _FakeTransformer.last["path"].endswith(gguf_filename)


def test_qwen_edit_variant_gguf_refuses_a_pinned_quant(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    attempted = _qwen_edit_dense_route(backend, monkeypatch)
    (tmp_path / "qwen-image-edit-2509-Q6_K.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "different transformer"):
        _load_into(
            backend,
            tmp_path,
            gguf_filename = "qwen-image-edit-2509-Q6_K.gguf",
            base_repo = "Qwen/Qwen-Image-Edit-2511",
            family_override = "qwen-image-edit",
            transformer_quant = "fp8",
        )
    assert attempted == []


def test_qwen_edit_variant_gguf_with_baked_loras_fails_instead_of_silent_drop(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    attempted = _qwen_edit_dense_route(backend, monkeypatch)
    (tmp_path / "qwen-image-edit-2509-Q6_K.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "LoRA adapters could not be applied"):
        _load_into(
            backend,
            tmp_path,
            gguf_filename = "qwen-image-edit-2509-Q6_K.gguf",
            base_repo = "Qwen/Qwen-Image-Edit-2511",
            family_override = "qwen-image-edit",
            loras = [("adapter", 1.0)],
        )
    assert attempted == []


def test_load_pipeline_kind_uses_from_pretrained(fake_runtime):
    """A full-pipeline (no single-file) load on an unsloth/* repo builds the pipe with
    pipeline_cls.from_pretrained(repo_id) -- NO single-file transformer build, NO GGUF
    quant config -- so an embedded bnb-4bit config is reloaded by diffusers itself."""
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit", family_override = "z-image"
    )
    assert status["loaded"] is True
    assert status["family"] == "z-image"
    assert _FakePipeline.last["base"] == "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
    assert "transformer" not in _FakePipeline.last
    assert _FakeTransformer.last == {}


def test_load_single_file_safetensors_no_gguf_config(fake_runtime, tmp_path):
    """A single-file *.safetensors transformer is built with from_single_file WITHOUT the
    GGUF dequant config (it carries its own dtype), then assembled from the base repo."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_into(
        backend, tmp_path, gguf_filename = "model.safetensors", family_override = "qwen-image"
    )
    assert status["loaded"] is True
    assert _FakeTransformer.last["path"] == str((tmp_path / "model.safetensors").resolve())
    assert _FakeTransformer.last["subfolder"] == "transformer"
    assert "quantization_config" not in _FakeTransformer.last
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last


def test_load_sdxl_pipeline_from_pretrained(fake_runtime):
    """SDXL as a full pipeline (no single-file name) loads via pipeline_cls.from_pretrained
    on the allowlisted official base repo -- no U-Net single-file build, no GGUF config.
    A U-Net family must NOT try to build a transformer from a single file."""
    backend = DiffusionBackend()
    status = backend.load_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
    assert status["loaded"] is True
    assert status["family"] == "sdxl"
    assert _FakePipeline.last["base"] == "unsloth/stable-diffusion-xl-base-1.0"
    assert status["base_repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "transformer" not in _FakePipeline.last
    assert _FakeTransformer.last == {}
    assert _FakePipeline.last_single_file == {}


def test_load_sdxl_single_file_uses_pipeline_from_single_file(fake_runtime, tmp_path):
    """A single-file SDXL *.safetensors is the WHOLE pipeline: it must load via
    pipeline_cls.from_single_file(path, config=base), NOT transformer_cls.from_single_file
    (UNet2DConditionModel has no companion-transformer assembly here)."""
    (tmp_path / "sdxl.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_into(
        backend, tmp_path, gguf_filename = "sdxl.safetensors", base_repo = None, family_override = "sdxl"
    )
    assert status["loaded"] is True
    assert status["family"] == "sdxl"
    assert _FakePipeline.last_single_file["path"] == str((tmp_path / "sdxl.safetensors").resolve())
    assert _FakePipeline.last_single_file["config"] == "unsloth/stable-diffusion-xl-base-1.0"
    assert status["base_repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert _FakeTransformer.last == {}


def test_load_sdxl_allowlisted_turbo_repo_is_trusted(fake_runtime):
    """The official sdxl-turbo repo is on the non-GGUF allowlist, so a full-pipeline load
    is permitted even though it is not under unsloth/*."""
    backend = DiffusionBackend()
    status = backend.load_pipeline("stabilityai/sdxl-turbo")
    assert status["loaded"] is True
    assert status["family"] == "sdxl"


def test_load_pipeline_rejects_non_unsloth_repo(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("randomorg/Z-Image-bnb-4bit", family_override = "z-image")


def test_validate_refuses_a_pipeline_pick_of_a_hosted_prequant_repo(fake_runtime):
    from core.inference.diffusion_families import _FAMILIES, prequant_only_repo_ids

    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "Qwen/Qwen-Image-2.1") as excinfo:
        backend.validate_load_request("unsloth/Qwen-Image-2.1-FP8")
    message = str(excinfo.value)
    assert "transformer precision to fp8 or int8" in message
    assert "text encoder precision to fp8" in message
    ids = prequant_only_repo_ids()
    assert "unsloth/qwen-image-2.1-fp8" in ids
    assert not any(fam.base_repo.lower() in ids for fam in _FAMILIES)


def test_load_sdxl_rejects_untrusted_repo(fake_runtime):
    """A random non-allowlisted, non-unsloth repo is still rejected for a full pipeline
    load even when it detects as SDXL -- the allowlist is exact-match only."""
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("randomorg/my-sdxl-merge", family_override = "sdxl")


def test_validate_gates_untrusted_base_repo(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "base_repo"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = "evil/companions",
        )
    # A local base_repo without model_index.json would evict the resident model, then fail.
    bad_base = tmp_path / "bare-base"
    bad_base.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(bad_base),
        )
    (tmp_path / "model_index.json").write_text("{")
    with pytest.raises(ValueError, match = "valid model_index.json"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(tmp_path),
        )
    _write_pipeline(tmp_path, scheduler = ["diffusers", "FlowMatchEulerDiscreteScheduler"])
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text("{}")
    fam = backend.validate_load_request(
        "unsloth/Qwen-Image-2512-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = str(tmp_path),
    )
    assert fam is not None
    with pytest.raises(FileNotFoundError, match = "valid model_index.json"):
        backend.validate_load_request(str(tmp_path), family_override = "qwen-image")


def test_validate_accepts_config_only_local_base_for_whole_pipeline_single_file(
    fake_runtime, tmp_path
):
    backend = DiffusionBackend()
    (tmp_path / "model.safetensors").write_bytes(b"checkpoint")
    base = tmp_path / "sdxl-config"
    (base / "unet").mkdir(parents = True)
    (base / "unet" / "config.json").write_text("{}")
    manifest = {
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
    }
    (base / "model_index.json").write_text(json.dumps(manifest))

    fam = backend.validate_load_request(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
        base_repo = str(base),
        family_override = "sdxl",
    )
    assert fam.single_file_is_pipeline is True
    with pytest.raises(FileNotFoundError, match = "valid model_index.json"):
        backend.validate_load_request(str(base), family_override = "sdxl")


def test_validate_accepts_custom_local_components(fake_runtime, tmp_path):
    manifest = {
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["local_extensions", "CustomModel"],
        "scheduler": ["local_extensions", "CustomScheduler"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(manifest))
    for name, asset in (
        ("unet", "custom_weights.safetensors"),
        ("scheduler", "custom_schedule.json"),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / asset).write_text("{}")

    backend = DiffusionBackend()
    assert backend.validate_load_request(str(tmp_path), family_override = "sdxl").name == "sdxl"
    status = backend.load_pipeline(str(tmp_path), family_override = "sdxl", speed_mode = "off")
    assert status["loaded"] is True
    assert _FakePipeline.last["base"] == str(tmp_path)


def test_resolve_local_single_file(tmp_path):
    from core.inference.diffusion import resolve_local_single_file

    d = tmp_path / "solo"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d)) == "model.safetensors"

    (d / "model_index.json").write_text("{}")
    assert resolve_local_single_file(str(d)) is None

    d2 = tmp_path / "shards"
    d2.mkdir()
    (d2 / "a.safetensors").write_bytes(b"w")
    (d2 / "b.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d2)) is None
    assert resolve_local_single_file(str(tmp_path / "empty-nonexistent")) is None
    assert resolve_local_single_file("unsloth/Qwen-Image-2512-GGUF") is None

    adapter = tmp_path / "flux-style-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter)) is None
    adapter2 = tmp_path / "z-image-lora"
    adapter2.mkdir()
    (adapter2 / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter2)) is None


def test_resolve_base_repo_drops_untrusted_card_tag(monkeypatch):
    # The base_model card tag is attacker-controlled: an untrusted tag uses the family default.
    import core.inference.diffusion as dmod

    fam = detect_family("unsloth/FLUX.1-dev-GGUF")
    monkeypatch.setattr(dmod, "_hf_base_model", lambda repo_id, hf_token: "attacker/evil-pipeline")
    assert _resolve_base_repo("attacker/flux.1-evil-GGUF", None, fam, None) == fam.base_repo
    monkeypatch.setattr(
        dmod, "_hf_base_model", lambda repo_id, hf_token: "black-forest-labs/FLUX.1-dev"
    )
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", None, fam, None)
        == "black-forest-labs/FLUX.1-dev"
    )
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", "unsloth/custom-base", fam, None)
        == "unsloth/custom-base"
    )


def test_resolve_base_repo_maps_a_mirrored_card_tag_back_to_the_vendor_id(monkeypatch):
    """A card tag can now name a mirror and clear the trust bar. This value is
    status()["base_repo"] and a trained adapter's default base_model, so it must be the vendor
    id; only the fetch sites see the mirror."""
    import core.inference.diffusion as dmod

    fam = detect_family("unsloth/FLUX.1-dev-GGUF")
    monkeypatch.setattr(dmod, "_hf_base_model", lambda repo_id, hf_token: "unsloth/FLUX.1-dev")
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", None, fam, None)
        == "black-forest-labs/FLUX.1-dev"
    )
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", "unsloth/FLUX.1-dev", fam, None)
        == "unsloth/FLUX.1-dev"
    )


def test_detect_family_routes_layered_to_its_own_pipeline():
    assert (
        detect_family("unsloth/Qwen-Image-Layered-GGUF").pipeline_class
        == "QwenImageLayeredPipeline"
    )
    assert detect_family("unsloth/qwen_image_layered").name == "qwen-image-layered"
    assert detect_family("unsloth/FLUX.1-Layered") is None


def test_failed_load_rolls_back_eager_patches(fake_runtime, tmp_path, monkeypatch):
    """A load failure AFTER the eager patches install but BEFORE the _LoadState commit must
    roll the process-wide patches back, so the next bit-identical `off` load is not
    contaminated (the asymmetric-cleanup bug the reviewers flagged)."""
    from core.inference import diffusion as diff_mod
    from core.inference import diffusion_eager_patches as ep

    (tmp_path / "model.gguf").write_bytes(b"x")
    ep.uninstall_patches()

    def _boom(*_a, **_k):
        raise RuntimeError("placement boom")

    monkeypatch.setattr(diff_mod, "apply_memory_plan", _boom)
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            family_override = "z-image",
            base_repo = "base/repo",
            speed_mode = "eager",
        )
    assert ep.is_installed() is False
    assert backend.is_loaded is False


def test_cpu_offload_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_into(backend, tmp_path, cpu_offload = True)
    assert status["cpu_offload"] is False


def test_low_vram_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_into(backend, tmp_path, memory_mode = "low_vram")
    assert status["cpu_offload"] is False


def test_generate_without_load_raises(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.generate(prompt = "x")


def test_failed_load_restores_backend_flags(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()

    restored: list = []
    cleared: list = []
    monkeypatch.setattr(
        "core.inference.diffusion.restore_backend_flags", lambda snap: restored.append(snap)
    )
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: cleared.append(True))
    monkeypatch.setattr(
        "core.inference.diffusion.apply_memory_plan",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )

    with pytest.raises(RuntimeError, match = "out of memory"):
        _load_into(backend, tmp_path, speed_mode = "max")
    assert restored, "restore_backend_flags was not called on the failed-load path"
    assert cleared, "clear_gpu_cache was not called on the failed-load path (VRAM leak)"
    assert backend._state is None and backend.is_loaded is False


def test_resolve_base_repo_prefers_caller_then_hf_tag_then_fallback(monkeypatch):
    from core.inference import diffusion
    from core.inference.diffusion_families import detect_family

    fam = detect_family("unsloth/Qwen-Image-2512-GGUF")
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: "Qwen/Qwen-Image-2512")
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "my/base", fam, None)
        == "my/base"
    )
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", None, fam, None)
        == "Qwen/Qwen-Image-2512"
    )
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: None)
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "  ", fam, None)
        == fam.base_repo
    )


def test_load_without_gguf_raises():
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("some-org/Z-Image-bnb-4bit")


def test_load_unknown_family_raises():
    backend = DiffusionBackend()
    with pytest.raises(ValueError):
        backend.load_pipeline("some/unrecognised-repo", gguf_filename = "x.gguf")


# load_progress state machine (no threads / network / real cache)

from core.inference.diffusion import _LoadingState  # noqa: E402


def test_load_progress_idle_and_ready():
    backend = DiffusionBackend()
    assert backend.load_progress()["phase"] is None
    backend._state = _LoadState(object(), None, "r", "b", "cpu", "float32", False)
    assert backend.load_progress()["phase"] == "ready"


def test_load_progress_error():
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", error = "boom")
    p = backend.load_progress()
    assert p["phase"] == "error" and p["error"] == "boom"


def test_load_progress_downloading_then_finalizing(monkeypatch):
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", expected_bytes = 1000)

    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 150))
    p = backend.load_progress()
    assert p["phase"] == "downloading"
    assert p["bytes_downloaded"] == 300
    assert abs(p["fraction"] - 0.3) < 1e-9

    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 500))
    assert backend.load_progress()["phase"] == "finalizing"


def test_load_progress_counts_only_the_selected_prequant_file_in_its_artifact_repo(monkeypatch):
    # unsloth/Qwen-Image-FP8 holds one checkpoint per scheme; count only the selected file.
    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "Qwen/Qwen-Image",
        base_repo = "Qwen/Qwen-Image",
        expected_bytes = 1000,
    )
    backend._loading.asset_repos = ("unsloth/Qwen-Image-FP8",)
    backend._loading.asset_files = (("unsloth/Qwen-Image-FP8", "Qwen-Image-INT8.pt", 400, 600),)
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 0))
    monkeypatch.setattr(DiffusionBackend, "_cache_file_bytes", staticmethod(lambda repo, f: 0))

    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo: 700 if repo == "unsloth/Qwen-Image-FP8" else 500),
    )
    p = backend.load_progress()
    assert p["phase"] == "downloading"
    assert p["bytes_downloaded"] == 600

    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_file_bytes",
        staticmethod(lambda repo, f: 400 if f == "Qwen-Image-INT8.pt" else 0),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo: 1000 if repo == "unsloth/Qwen-Image-FP8" else 600),
    )
    assert backend.load_progress()["phase"] == "finalizing"
    assert backend.load_progress()["bytes_downloaded"] == 1000


def test_load_progress_reports_a_prequant_file_that_was_already_cached(monkeypatch):
    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "Qwen/Qwen-Image",
        base_repo = "Qwen/Qwen-Image",
        expected_bytes = 1000,
    )
    backend._loading.asset_repos = ("unsloth/Qwen-Image-FP8",)
    backend._loading.asset_files = (("unsloth/Qwen-Image-FP8", "Qwen-Image-INT8.pt", 400, 1000),)
    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo: 1000 if repo == "unsloth/Qwen-Image-FP8" else 600),
    )
    monkeypatch.setattr(DiffusionBackend, "_cache_file_bytes", staticmethod(lambda repo, f: 400))
    assert backend.load_progress()["phase"] == "finalizing"


def test_load_progress_counts_a_mirrored_pipeline_repo_once(monkeypatch):
    # base_repo == repo_id: count once, or the upstream's stale partials peg the bar at 100%.
    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "black-forest-labs/FLUX.1-dev",
        base_repo = "black-forest-labs/FLUX.1-dev",
        expected_bytes = 1000,
    )
    backend._loading.fetch_repo = "unsloth/FLUX.1-dev"
    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo: 600 if repo.startswith("black-forest-labs/") else 500),
    )
    p = backend.load_progress()
    assert p["bytes_downloaded"] == 500
    assert p["phase"] == "downloading"


def test_load_progress_still_sums_a_gguf_pick_and_its_separate_base(monkeypatch):
    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        base_repo = "black-forest-labs/FLUX.1-dev",
        expected_bytes = 1000,
    )
    backend._loading.fetch_repo = "unsloth/FLUX.1-dev"
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 150))
    assert backend.load_progress()["bytes_downloaded"] == 300


def test_base_file_downloaded_excludes_undownloaded():
    assert _base_file_downloaded("model_index.json")
    assert _base_file_downloaded("text_encoder/model-00001-of-00003.safetensors")
    assert _base_file_downloaded("vae/diffusion_pytorch_model.safetensors")
    # Excluded: the GGUF supplies the transformer, and docs/assets are never fetched, so counting them would peg the bar short of 100%.
    assert not _base_file_downloaded(
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
    )
    assert not _base_file_downloaded("assets/Z-Image-Gallery.pdf")
    assert not _base_file_downloaded("README.md")
    assert not _base_file_downloaded(".gitattributes")


def test_load_progress_fraction_clamped(monkeypatch):
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", expected_bytes = 1000)
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 900))
    p = backend.load_progress()
    assert p["phase"] == "finalizing"
    assert p["fraction"] == 1.0
    assert p["bytes_downloaded"] == 1000


def test_estimate_eta():
    from core.inference.diffusion import _estimate_eta

    assert _estimate_eta(8, 1, first_step_at = 100.0, now = 100.0) is None
    assert _estimate_eta(8, 0, first_step_at = 0.0, now = 100.0) is None
    assert _estimate_eta(8, 4, first_step_at = 100.0, now = 103.0) == 4.0
    assert _estimate_eta(8, 8, first_step_at = 100.0, now = 107.0) == 0.0


def test_generate_qwen_uses_true_cfg_scale(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path, base_repo = "Qwen/Qwen-Image", family_override = "qwen-image")
    backend.generate(prompt = "a sloth", guidance = 4.0)
    call = backend._state.pipe.last_kwargs
    assert call["true_cfg_scale"] == 4.0 and call["guidance_scale"] is None


def _load_ideogram(backend, tmp_path):
    _write_pipeline(tmp_path, "Ideogram4Pipeline", weight = "pytorch_model.bin")
    backend.load_pipeline(str(tmp_path), family_override = "ideogram-4")


def test_ideogram_rejects_single_file_and_gguf_kinds(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        _load_into(backend, tmp_path, base_repo = None, family_override = "ideogram-4")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        _load_into(
            backend,
            tmp_path,
            gguf_filename = "model.safetensors",
            base_repo = None,
            family_override = "ideogram-4",
            model_kind = "single_file",
        )


def test_generate_ideogram_defaults_keep_recommended_schedule(fake_runtime, tmp_path):
    # Ideogram 4 defaults to guidance_schedule (valid only at 48 steps) and rejects guidance_scale with it, so drop the constant at the defaults.
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 7.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] is None
    assert "guidance_schedule" not in call


def test_generate_ideogram_custom_guidance_nulls_schedule(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 20, guidance = 5.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 5.0
    assert "guidance_schedule" in call and call["guidance_schedule"] is None


def _load_lumina(backend, tmp_path):
    _write_pipeline(tmp_path, "Lumina2Pipeline")
    backend.load_pipeline(str(tmp_path), family_override = "lumina-2")


def test_generate_lumina2_passes_cfg_trunc_ratio(fake_runtime, tmp_path):
    # The card recipe truncates the CFG double-forward to the first quarter; the pipeline default (1.0) applies it everywhere.
    backend = DiffusionBackend()
    _load_lumina(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 50, guidance = 4.0)
    call = backend._state.pipe.last_kwargs
    assert call["cfg_trunc_ratio"] == 0.25
    assert call["guidance_scale"] == 4.0


def test_generate_other_family_never_passes_cfg_trunc_ratio(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"weights")
    _load_into(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 9, guidance = 0.0)
    call = backend._state.pipe.last_kwargs
    assert call["cfg_trunc_ratio"] is None


def test_begin_load_rejects_concurrent(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setattr("core.inference.diffusion._hf_base_model", lambda *a, **k: None)
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    # Drain the worker by identity, not by diffing threading.enumerate(): an enumerated thread
    # may not be joinable yet (ident is set before _started).
    holding = threading.Event()
    entered = threading.Event()
    worker = []

    def held(self, **kwargs):
        worker.append(threading.current_thread())
        entered.set()
        assert holding.wait(timeout = 30), "the test never released the load worker"

    monkeypatch.setattr(DiffusionBackend, "load_pipeline", held)
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    assert entered.wait(timeout = 30), "the load worker never reached load_pipeline"
    with pytest.raises(RuntimeError):
        backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    # begin_load's thread is fire-and-forget; left running it loads under a later test's patches.
    holding.set()
    worker[0].join(timeout = 5)


def test_unload_cancels_in_flight_load(fake_runtime):
    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    token = 7
    backend._load_token = token
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend._load_token = token + 1
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )


def test_superseded_load_does_not_cancel_live_generation(fake_runtime):
    import threading as _threading

    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    live_cancel = _threading.Event()
    backend._active_generate_cancel = live_cancel
    token = 11
    backend._load_token = token + 1
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )
    assert not live_cancel.is_set()


def test_pick_dtype_bf16_only_on_ampere(fake_runtime, monkeypatch):
    # BF16 only on Ampere+ (cc >= 8); pre-Ampere cards must fall back to FP16.
    torch = sys.modules["torch"]
    backend = DiffusionBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.float16)


def test_unload_sets_cancel_event(fake_runtime):
    backend = DiffusionBackend()
    assert not backend._cancel_event.is_set()
    backend.unload()
    assert backend._cancel_event.is_set()


@pytest.mark.parametrize("reason", ["generation", "resident"])
def test_rejected_unload_preserves_pending_work(fake_runtime, monkeypatch, reason):
    from fastapi import HTTPException
    from core.inference.gpu_arbiter import GpuBusyForAnotherAccountError
    from hub.services.models import account_access

    backend = DiffusionBackend()
    pending = object()
    backend._loading = pending
    active = threading.Event()
    backend._active_generate_cancel = active
    backend._active_generate_account = "bob" if reason == "generation" else "alice"

    def refuse(*args):
        raise HTTPException(status_code = 404, detail = "Model not found")

    monkeypatch.setattr(account_access, "require_resident_control", refuse)
    error = GpuBusyForAnotherAccountError if reason == "generation" else HTTPException
    with pytest.raises(error):
        backend.unload(expected_account = "alice")
    assert not active.is_set() and not backend._cancel_event.is_set()
    assert backend._load_token == backend._unload_waiters == backend._teardown_waiters == 0
    assert backend._loading is pending


def test_generation_waits_for_unload_before_teardown_reservation(fake_runtime, monkeypatch):
    backend = DiffusionBackend()
    parked, release, attempted, started = (threading.Event() for _ in range(4))
    real_lock = backend._lock
    torn_down = []
    observed = []
    errors = []

    class GateLock:
        def __enter__(self):
            if threading.current_thread() is ejector:
                parked.set()
                assert release.wait(5)
            real_lock.acquire()

        def __exit__(self, *args):
            real_lock.release()
            if threading.current_thread() is generator:
                attempted.set()

    monkeypatch.setattr(backend, "_lock", GateLock())
    monkeypatch.setattr(backend, "_unload_locked", lambda: torn_down.append(True))

    def eject():
        try:
            backend.unload()
        except Exception as exc:
            errors.append(str(exc))

    def generate():
        with backend._generation_slot(threading.Event()):
            observed.append(bool(torn_down))
            started.set()

    ejector = threading.Thread(target = eject, daemon = True)
    generator = threading.Thread(target = generate, daemon = True)
    ejector.start()
    try:
        assert parked.wait(5)
        generator.start()
        assert attempted.wait(5)
        assert not started.wait(0.1), "generation started before the accepted eject"
    finally:
        release.set()
        ejector.join(5)
        if generator.ident is not None:
            generator.join(5)
    assert not errors and not ejector.is_alive() and not generator.is_alive()
    assert observed == [True]
    assert backend._unload_waiters == backend._teardown_waiters == 0


@pytest.mark.parametrize(
    "phase",
    [
        "validation",
        "preinstall",
        "resolution",
        "memory_plan",
        "dense_plan",
        "transformer_error",
        "pipeline_error",
        "dense_error",
        "dense_fallback",
        "transformer",
        "pipeline",
        "dense",
        "attention_error",
        "cache_error",
        "attention",
        "step_cache",
        "compile_cache",
        "speed",
        "quantize",
        "component_plan",
        "conditioning",
        "placement",
        "publication",
    ],
)
@pytest.mark.parametrize("account_scoped", [False, True])
def test_unload_cancels_pipeline_construction(
    fake_runtime, tmp_path, monkeypatch, phase, account_scoped
):
    import gc
    import weakref
    from core.inference import diffusion as diff_mod
    from core.inference import diffusion_eager_patches as ep

    (tmp_path / "model.gguf").write_bytes(b"weights")
    # ep.is_installed() is process-global and other tests leave the patches installed.
    ep.uninstall_patches()
    backend = DiffusionBackend()
    entered, release = threading.Event(), threading.Event()
    outcome = {}
    live = weakref.WeakSet()
    reclaimed = []
    pipelines = []
    transformer_calls = []
    setup_calls = []

    def tracked():
        value = _FakePipe()
        value.cycle = value
        live.add(value)
        return value

    def reclaim():
        gc.collect()
        reclaimed.append((len(live), backend._transition_owns_slot, backend._lock.locked()))

    def park(value):
        entered.set()
        assert release.wait(5)
        return value

    def transformer(cls, *args, **kwargs):
        transformer_calls.append(True)
        value = tracked()
        if phase == "transformer_error":
            park(None)
            raise RuntimeError("setup failed")
        return park(value) if phase == "transformer" else value

    def pipeline(cls, *args, **kwargs):
        value = tracked()
        value.transformer = kwargs["transformer"]
        value.text_encoder = kwargs["text_encoder"]
        pipelines.append(weakref.ref(value))
        if phase == "pipeline_error":
            park(None)
            raise RuntimeError("setup failed")
        return park(value) if phase == "pipeline" else value

    def dense(*args, **kwargs):
        value = tracked()
        value.transformer = tracked()
        pipelines.append(weakref.ref(value))
        if phase in ("dense_error", "dense_fallback"):
            park(None)
            raise RuntimeError("setup failed")
        return park(value), "int8"

    def load():
        try:
            outcome["loaded"] = _load_into(
                backend,
                tmp_path,
                speed_mode = "default" if phase == "compile_cache" else "eager",
                transformer_quant = None
                if phase == "dense_fallback"
                else "int8"
                if phase.startswith("dense")
                else "off",
            )
        except Exception as exc:
            outcome["error"] = str(exc)

    with monkeypatch.context() as mp:
        mp.setattr(diff_mod, "clear_gpu_cache", reclaim)
        mp.setattr(_FakeTransformer, "from_single_file", classmethod(transformer))
        mp.setattr(_FakePipeline, "from_pretrained", classmethod(pipeline))
        mp.setattr(diff_mod, "te_prequant_pipe_kwargs", lambda *a, **k: {"text_encoder": tracked()})
        if phase in ("attention", "step_cache", "compile_cache", "speed"):
            if phase == "compile_cache":
                mp.setattr(diff_mod, "compile_eligible", lambda *a, **k: True)
                mp.setattr(diff_mod.compile_cache, "begin", lambda **k: None)
            for stage, owner, name in (
                ("attention", diff_mod, "apply_attention_backend"),
                ("step_cache", diff_mod, "apply_step_cache"),
                ("compile_cache", diff_mod.compile_cache, "begin"),
                ("speed", diff_mod, "apply_speed_optims"),
                ("quantize", diff_mod, "quantize_text_encoders"),
            ):
                original = getattr(owner, name)

                def setup(
                    *args,
                    _stage = stage,
                    _original = original,
                    **kwargs,
                ):
                    setup_calls.append(_stage)
                    result = _original(*args, **kwargs)
                    return park(result) if phase == _stage else result

                mp.setattr(owner, name, setup)
        if phase.startswith("dense"):
            mp.setattr(diff_mod, "dense_transformer_supported", lambda target: True)
            mp.setattr(diff_mod, "select_transformer_quant_scheme", lambda *a, **k: "int8")
            # Z-Image declares a rotated INT8 artifact; treat it as cached so the dense attempt is reached.
            mp.setattr(diff_mod, "_uncached_prequant_repo", lambda *a, **k: None)
            mp.setattr(
                backend,
                "_dense_transformer_resident_bytes",
                lambda *a, **k: park(0) if phase == "dense_plan" else 0,
            )
            mp.setattr(backend, "_load_dense_quant_pipeline", dense)
        elif phase in ("resolution", "memory_plan"):
            name = "_resolve_gguf_path" if phase == "resolution" else "_plan_memory"
            original = getattr(backend, name)
            mp.setattr(backend, name, lambda *a, **k: park(original(*a, **k)))
        elif phase in ("validation", "preinstall"):
            if phase == "validation":
                original = backend.validate_load_request
                mp.setattr(
                    backend, "validate_load_request", lambda *a, **k: park(original(*a, **k))
                )
            else:
                mp.setattr(diff_mod, "select_attention_backend", lambda *a, **k: "test")
                mp.setattr(
                    diff_mod, "_ensure_attention_backend_installed", lambda *a, **k: park(None)
                )
        elif phase in ("attention_error", "cache_error"):

            def fail_setup(*args, **kwargs):
                park(None)
                raise RuntimeError("setup failed")

            name = "apply_attention_backend" if phase == "attention_error" else "apply_step_cache"
            mp.setattr(diff_mod, name, fail_setup)
        elif phase in ("component_plan", "conditioning"):
            owner, name = (
                (diff_mod, "refine_memory_plan_for_components")
                if phase == "component_plan"
                else (diff_mod.cond_cache, "install")
            )
            original = getattr(owner, name)
            mp.setattr(owner, name, lambda *a, **k: park(original(*a, **k)))
            place = diff_mod.apply_memory_plan

            def placement(*args, **kwargs):
                setup_calls.append("placement")
                return place(*args, **kwargs)

            mp.setattr(diff_mod, "apply_memory_plan", placement)
        elif phase in ("quantize", "placement", "publication"):
            name = {
                "quantize": "quantize_text_encoders",
                "placement": "apply_memory_plan",
                "publication": "_LoadState",
            }[phase]
            original = getattr(diff_mod, name)

            def parked(*args, **kwargs):
                return park(original(*args, **kwargs))

            mp.setattr(diff_mod, name, parked)

        loader = threading.Thread(target = load, daemon = True)
        ejector = threading.Thread(
            target = backend.unload,
            kwargs = {"expected_account": diff_mod.current_account_id()} if account_scoped else {},
            daemon = True,
        )
        loader.start()
        try:
            assert entered.wait(5), "load did not reach the blocked construction stage"
            ejector.start()
            assert backend._cancel_event.wait(2), "eject could not signal during construction"
            if phase in ("validation", "preinstall"):
                ejector.join(5)
                assert not ejector.is_alive()
            else:
                assert ejector.is_alive(), "teardown must wait for the constructor to unwind"
        finally:
            release.set()
            loader.join(5)
            if ejector.ident is not None:
                ejector.join(5)

    assert not loader.is_alive() and not ejector.is_alive()
    assert "loaded" not in outcome, "a cancelled pipeline was published as ready"
    assert (
        "setup failed" if phase.endswith("_error") and phase != "dense_error" else "cancelled"
    ) in outcome["error"]
    assert not backend.is_loaded
    assert not ep.is_installed()
    assert backend._teardown_waiters == 0
    if phase in ("validation", "preinstall"):
        assert not live and not reclaimed
    else:
        assert reclaimed and all(item == (0, True, True) for item in reclaimed), reclaimed
    if phase in ("resolution", "memory_plan", "dense_plan"):
        assert not transformer_calls and not pipelines, "cancelled planning constructed weights"
    if phase == "dense_fallback":
        assert not transformer_calls, "cancelled dense attempt started a GGUF fallback"
    if phase == "transformer":
        assert not pipelines, "cancelled load still constructed its companions"
    if phase in ("attention", "step_cache", "compile_cache", "speed"):
        assert setup_calls[-1] == phase, setup_calls
    if phase in ("component_plan", "conditioning"):
        assert not setup_calls, "cancelled setup still placed the pipeline"

    _load_into(backend, tmp_path)
    pipe = backend._state.pipe
    for _ in range(2):
        assert len(backend.generate(prompt = "a sloth", steps = 2)["images"]) == 1
        assert backend._state.pipe is pipe
    backend.unload()


@pytest.mark.parametrize(
    "phase",
    [
        "index",
        "tokenizer",
        "tokenizer_error",
        "encoder_config",
        "text_encoder",
        "scheduler",
        "vae",
        "transformer",
        "pipeline",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_krea_component_load_honors_eject(fake_runtime, tmp_path, monkeypatch, phase, cancel):
    import gc
    import weakref
    from core.inference import diffusion as diff_mod, diffusion_krea2 as krea

    backend = DiffusionBackend()
    calls, ejectors, reclaimed = [], [], []
    live = weakref.WeakSet()
    _write_pipeline(tmp_path)

    def record(name, value):
        calls.append(name)
        if name == phase and cancel:
            ejector = threading.Thread(target = backend.unload, daemon = True)
            ejectors.append(ejector)
            ejector.start()
            assert backend._cancel_event.wait(5), "eject could not signal during assembly"
        return value

    def component(name):
        value = _FakePipe()
        value.cycle = value
        live.add(value)
        return record(name, value)

    def pretrained(name):
        return types.SimpleNamespace(from_pretrained = lambda *a, **k: component(name))

    def tokenizer(*args, **kwargs):
        if phase == "tokenizer_error" and "extra_special_tokens" not in kwargs:
            record("tokenizer_error", None)
            raise ValueError("tokenizer config requires the compatibility fallback")
        return component("tokenizer_retry" if phase == "tokenizer_error" else "tokenizer")

    def reclaim():
        gc.collect()
        reclaimed.append(len(live))

    diffusers = sys.modules["diffusers"]
    diffusers.Krea2Pipeline = lambda **kwargs: component("pipeline")
    diffusers.Krea2Transformer2DModel = pretrained("transformer")
    diffusers.FlowMatchEulerDiscreteScheduler = pretrained("scheduler")
    diffusers.AutoencoderKLQwenImage = pretrained("vae")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer = types.SimpleNamespace(from_pretrained = tokenizer),
            AutoConfig = types.SimpleNamespace(
                from_pretrained = lambda *a, **k: record("encoder_config", types.SimpleNamespace())
            ),
            Qwen3VLModel = pretrained("text_encoder"),
        ),
    )
    monkeypatch.setattr(krea, "_load_model_index", lambda *a, **k: record("index", {}))
    monkeypatch.setattr(diff_mod, "clear_gpu_cache", reclaim)
    expected = [
        "index",
        "tokenizer",
        "encoder_config",
        "text_encoder",
        "scheduler",
        "vae",
        "transformer",
        "pipeline",
    ]
    if phase == "tokenizer_error":
        expected[1:2] = ["tokenizer_error", "tokenizer_retry"]
    try:
        if cancel:
            with pytest.raises(RuntimeError, match = "cancelled"):
                backend.load_pipeline(
                    str(tmp_path), family_override = "krea-2", local_files_only = True
                )
            assert calls == expected[: expected.index(phase) + 1]
            assert not backend.is_loaded
            assert reclaimed and not any(reclaimed), reclaimed
        else:
            assert backend.load_pipeline(
                str(tmp_path), family_override = "krea-2", local_files_only = True
            )["loaded"]
            assert calls == expected
            for _ in range(2):
                assert backend.generate(prompt = "a sloth", steps = 2)["images"]
    finally:
        for ejector in ejectors:
            ejector.join(5)
            assert not ejector.is_alive()
        backend.unload()


@pytest.mark.parametrize("kind", ["pipeline", "gguf", "single_file"])
@pytest.mark.parametrize(
    "family,phase", [("hidream-i1", "te4"), ("hidream-i1", "encoder"), ("z-image", "encoder")]
)
@pytest.mark.parametrize("cancel", [False, True])
def test_eager_encoder_cancellation_stops_pipeline_build(
    fake_runtime, tmp_path, monkeypatch, kind, family, phase, cancel
):
    import gc
    import weakref
    from core.inference import diffusion as mod

    backend = DiffusionBackend()
    calls, ejectors, reclaimed = [], [], []
    live = weakref.WeakSet()
    _write_pipeline(tmp_path)
    filename = "model.gguf" if kind == "gguf" else "model.safetensors"
    (tmp_path / filename).write_bytes(b"weights")
    sys.modules["diffusers"].HiDreamImagePipeline = _FakePipeline
    sys.modules["diffusers"].HiDreamImageTransformer2DModel = _FakeTransformer

    def prepare(name, key):
        calls.append(name)
        value = _FakePipe()
        value.cycle = value
        live.add(value)
        if cancel and phase == name:
            thread = threading.Thread(target = backend.unload, daemon = True)
            ejectors.append(thread)
            thread.start()
            assert backend._cancel_event.wait(5)
        return {key: value}

    def pipeline(*args, **kwargs):
        calls.append("pipeline")
        return _FakePipe()

    def reclaim():
        gc.collect()
        reclaimed.append(len(live))

    monkeypatch.setattr(mod, "hidream_te4_kwargs", lambda *a, **k: prepare("te4", "text_encoder_4"))
    monkeypatch.setattr(
        mod, "te_prequant_pipe_kwargs", lambda *a, **k: prepare("encoder", "text_encoder")
    )
    monkeypatch.setattr(_FakePipeline, "from_pretrained", staticmethod(pipeline))
    monkeypatch.setattr(mod, "clear_gpu_cache", reclaim)
    expected = (["te4"] if family == "hidream-i1" else []) + ["encoder", "pipeline"]
    kwargs = dict(family_override = family, local_files_only = True, transformer_quant = "off")
    if kind != "pipeline":
        kwargs.update(gguf_filename = filename, base_repo = str(tmp_path))
    try:
        if cancel:
            with pytest.raises(RuntimeError, match = "cancelled"):
                backend.load_pipeline(str(tmp_path), **kwargs)
            assert calls == expected[: expected.index(phase) + 1]
            assert not backend.is_loaded
            assert reclaimed and not any(reclaimed)
        else:
            assert backend.load_pipeline(str(tmp_path), **kwargs)["loaded"]
            assert calls == expected
            assert backend.generate(prompt = "a sloth", steps = 2)["images"]
    finally:
        for thread in ejectors:
            thread.join(5)
            assert not thread.is_alive()
        backend.unload()


@pytest.mark.parametrize(
    "phase",
    [
        "mask",
        "text_encoder",
        "tokenizer",
        "transformer",
        "unconditional_transformer",
        "vae",
        "scheduler",
        "pipeline",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_ideogram_component_load_honors_eject(fake_runtime, tmp_path, monkeypatch, phase, cancel):
    import gc
    import weakref
    from core.inference import diffusion as mod, diffusion_ideogram4 as ideogram

    backend = DiffusionBackend()
    calls, ejectors, reclaimed = [], [], []
    live = weakref.WeakSet()
    _write_pipeline(tmp_path, weight = "pytorch_model.bin")

    def component(name):
        calls.append(name)
        value = _FakePipe()
        value.cycle = value
        live.add(value)
        if cancel and phase == name:
            thread = threading.Thread(target = backend.unload, daemon = True)
            ejectors.append(thread)
            thread.start()
            assert backend._cancel_event.wait(5)
        return value

    def reclaim():
        gc.collect()
        reclaimed.append(len(live))

    monkeypatch.setattr(mod, "load_ideogram4_pipeline", ideogram.load_ideogram4_pipeline)
    monkeypatch.setattr(ideogram, "_patch_create_causal_mask", lambda: component("mask"))
    monkeypatch.setattr(
        ideogram, "load_ideogram4_text_encoder", lambda *a, **k: component("text_encoder")
    )
    monkeypatch.setattr(ideogram, "load_krea2_tokenizer", lambda *a, **k: component("tokenizer"))
    monkeypatch.setattr(
        ideogram,
        "load_ideogram4_transformer",
        lambda repo, subfolder, *a, **k: component(subfolder),
    )
    diffusers = sys.modules["diffusers"]
    diffusers.Ideogram4Pipeline = lambda **kwargs: component("pipeline")
    diffusers.AutoencoderKLFlux2 = types.SimpleNamespace(
        from_pretrained = lambda *a, **k: component("vae")
    )
    diffusers.FlowMatchEulerDiscreteScheduler = types.SimpleNamespace(
        from_pretrained = lambda *a, **k: component("scheduler")
    )
    monkeypatch.setattr(mod, "clear_gpu_cache", reclaim)
    expected = [
        "mask",
        "text_encoder",
        "tokenizer",
        "transformer",
        "unconditional_transformer",
        "vae",
        "scheduler",
        "pipeline",
    ]
    try:
        if cancel:
            with pytest.raises(RuntimeError, match = "cancelled"):
                backend.load_pipeline(
                    str(tmp_path), family_override = "ideogram-4", local_files_only = True
                )
            assert calls == expected[: expected.index(phase) + 1]
            assert not backend.is_loaded
            assert reclaimed and not any(reclaimed)
        else:
            assert backend.load_pipeline(
                str(tmp_path), family_override = "ideogram-4", local_files_only = True
            )["loaded"]
            assert calls == expected
            assert backend.generate(prompt = "a sloth", steps = 2)["images"]
    finally:
        for thread in ejectors:
            thread.join(5)
            assert not thread.is_alive()
        backend.unload()


@pytest.mark.parametrize("phase", ["gpu", "validation", "precision"])
def test_begin_load_remembers_an_eject_during_preflight(fake_runtime, tmp_path, monkeypatch, phase):
    from core.inference import diffusion as diff_mod

    backend = DiffusionBackend()
    entered, release, dispatched = threading.Event(), threading.Event(), threading.Event()
    errors = []
    (tmp_path / "model.gguf").write_bytes(b"weights")
    monkeypatch.setattr(backend, "_run_load", lambda **kwargs: dispatched.set())

    def load():
        return backend.begin_load(
            str(tmp_path), gguf_filename = "model.gguf", family_override = "z-image", gpu_ids = [0]
        )

    def invoke():
        try:
            load()
        except Exception as exc:
            errors.append(str(exc))

    with monkeypatch.context() as mp:
        owner, name = {
            "gpu": (diff_mod, "resolve_selected_cuda_ordinal"),
            "validation": (backend, "validate_load_request"),
            "precision": (backend, "assert_precision_available"),
        }[phase]
        original = getattr(owner, name)
        if phase == "gpu":
            mp.setattr(
                diff_mod,
                "resolve_diffusion_device_target",
                lambda: types.SimpleNamespace(device = "cuda"),
            )

        def parked(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return 0 if phase == "gpu" else original(*args, **kwargs)

        mp.setattr(owner, name, parked)
        worker = threading.Thread(target = invoke, daemon = True)
        worker.start()
        try:
            assert entered.wait(5)
            backend.unload()
            assert backend._unload_waiters == 0
        finally:
            release.set()
            worker.join(5)

    assert not worker.is_alive()
    assert errors and "cancelled" in errors[0], errors
    assert not dispatched.is_set()
    assert backend._loading is None
    load()
    assert dispatched.wait(5)
    backend.unload()


@pytest.mark.parametrize("load_method", ["begin_load", "load_pipeline"])
def test_replacement_load_waits_for_every_unload(fake_runtime, tmp_path, monkeypatch, load_method):
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"weights")
    parked = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    real_lock = backend._lock
    errors = []
    dispatched = threading.Event()
    monkeypatch.setattr(backend, "_run_load", lambda **kwargs: dispatched.set())

    class GateLock:
        def __init__(self):
            self.waited = set()

        def __enter__(self):
            name = threading.current_thread().name
            if name.startswith("eject-") and name not in self.waited:
                self.waited.add(name)
                index = int(name[-1])
                parked[index].set()
                assert release[index].wait(5)
            real_lock.acquire()

        def __exit__(self, *args):
            real_lock.release()

    monkeypatch.setattr(backend, "_lock", GateLock())

    def eject():
        try:
            backend.unload()
        except Exception as exc:
            errors.append(str(exc))

    def load():
        return getattr(backend, load_method)(
            str(tmp_path),
            gguf_filename = "model.gguf",
            base_repo = "base/repo",
            family_override = "z-image",
        )

    # The replacement waits for every pending eject rather than being refused with a 409.
    replacement = {}

    def replacement_load():
        try:
            load()
            replacement["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            replacement["error"] = repr(exc)

    ejectors = [threading.Thread(target = eject, name = f"eject-{i}", daemon = True) for i in range(2)]
    for thread in ejectors:
        thread.start()
    waiter = None
    try:
        assert all(event.wait(5) for event in parked)
        waiter = threading.Thread(target = replacement_load, name = "replacement", daemon = True)
        waiter.start()
        waiter.join(0.5)
        assert waiter.is_alive() and not replacement and not dispatched.is_set()
        release[0].set()
        ejectors[0].join(5)
        assert not ejectors[0].is_alive()
        waiter.join(0.5)
        assert waiter.is_alive() and not replacement, "one eject left, the load must still wait"
    finally:
        for event in release:
            event.set()
        for thread in ejectors:
            thread.join(5)

    assert not errors and all(not thread.is_alive() for thread in ejectors)
    waiter.join(5)
    assert not waiter.is_alive()
    assert replacement.get("ok"), replacement
    assert backend._teardown_waiters == 0
    assert backend._unload_waiters == 0
    if load_method == "begin_load":
        assert dispatched.wait(5)
        backend._loading = None
        _load_into(backend, tmp_path)
    assert backend.generate(prompt = "after eject", steps = 2)["images"]
    backend.unload()


def test_prefetch_aborts_when_cancelled(tmp_path):
    backend = DiffusionBackend()
    backend._cancel_event.set()
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "Cancelled"):
        backend._prefetch_files(
            str(tmp_path),
            "model.gguf",
            "Tongyi-MAI/Z-Image-Turbo",
            ["vae/diffusion_pytorch_model.safetensors"],
            None,
        )


def test_each_load_owns_its_cancel_event(fake_runtime, monkeypatch, tmp_path):
    # unload() drops _loading, so a replacement starts while the old worker still prefetches;
    # a shared Event would let the replacement's clear() un-cancel it.
    backend = DiffusionBackend()
    started = threading.Event()
    seen: list[threading.Event] = []

    def _capture(**kwargs):
        seen.append(kwargs["_cancel_event"])
        started.set()

    monkeypatch.setattr(backend, "_run_load", _capture)
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    assert started.wait(5)
    first = seen[0]

    backend.unload()
    assert first.is_set()

    started.clear()
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q8_0.gguf")
    assert started.wait(5)
    second = seen[1]

    assert second is not first, "each load needs its own event, not a clear() of the shared one"
    assert not second.is_set()
    assert first.is_set(), "the superseded worker's event must stay set"
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "Cancelled"):
        backend._prefetch_files(
            str(tmp_path),
            "model.gguf",
            "Tongyi-MAI/Z-Image-Turbo",
            ["vae/diffusion_pytorch_model.safetensors"],
            None,
            cancel_event = first,
        )


def test_prefetch_downloads_gguf_and_base(monkeypatch, tmp_path):
    backend = DiffusionBackend()
    calls: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: (calls.append((repo, fn)), f"/cache/{fn}")[1],
    )
    backend._prefetch_files(
        "unsloth/Z-Image-Turbo-GGUF",
        "model.gguf",
        "base/repo",
        ["vae/x.safetensors", "text_encoder/y.safetensors"],
        "hf_tok",
    )
    assert ("unsloth/Z-Image-Turbo-GGUF", "model.gguf") in calls
    assert ("base/repo", "vae/x.safetensors") in calls
    assert ("base/repo", "text_encoder/y.safetensors") in calls
    calls.clear()
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend._prefetch_files(str(tmp_path), "model.gguf", "base/repo", ["vae/x.safetensors"], None)
    assert all(repo != str(tmp_path) for repo, _ in calls)
    assert ("base/repo", "vae/x.safetensors") in calls


def test_prefetch_pulls_companions_from_the_ungated_mirror(monkeypatch, tmp_path):
    """The companions are why a gated base blocked a GGUF pick, so this is the call that has to
    move. The file names are identical either way, so the list passes through untouched."""
    backend = DiffusionBackend()
    calls: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: (calls.append((repo, fn)), f"/cache/{fn}")[1],
    )
    _no_cache(monkeypatch)
    backend._prefetch_files(
        "unsloth/FLUX.1-dev-GGUF",
        "flux1-dev-Q4_K_M.gguf",
        "black-forest-labs/FLUX.1-dev",
        ["vae/diffusion_pytorch_model.safetensors"],
        None,
    )
    assert ("unsloth/FLUX.1-dev", "vae/diffusion_pytorch_model.safetensors") in calls
    assert all(repo != "black-forest-labs/FLUX.1-dev" for repo, _ in calls)
    assert ("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf") in calls

    calls.clear()
    _all_cached(monkeypatch)
    backend._prefetch_files(
        "unsloth/FLUX.1-dev-GGUF",
        None,
        "black-forest-labs/FLUX.1-dev",
        ["vae/diffusion_pytorch_model.safetensors"],
        None,
    )
    assert ("black-forest-labs/FLUX.1-dev", "vae/diffusion_pytorch_model.safetensors") in calls


def test_single_file_load_reads_config_and_companions_from_the_mirror(
    fake_runtime, tmp_path, monkeypatch
):
    """``config=`` is a REPO FETCH that runs BEFORE the mirrored pipeline load, so a gated id left
    there 401s an anonymous user first and the swap below is never reached."""
    diffusers = sys.modules["diffusers"]
    diffusers.FluxPipeline = _FakePipeline
    diffusers.FluxTransformer2DModel = _FakeTransformer
    _no_cache(monkeypatch)
    (tmp_path / "model.gguf").write_bytes(b"weights")

    status = DiffusionBackend().load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "black-forest-labs/FLUX.1-dev",
        family_override = "flux.1",
    )

    assert _FakeTransformer.last["config"] == "unsloth/FLUX.1-dev"
    assert _FakePipeline.last["base"] == "unsloth/FLUX.1-dev"
    assert status["base_repo"] == "black-forest-labs/FLUX.1-dev"


def test_pipeline_kind_assembles_krea_and_ideogram_from_the_mirror(fake_runtime, monkeypatch):
    """Both per-component loaders fetch EVERY component from the id handed to them, so a gated
    pipeline pick must arrive already swapped."""
    from core.inference import diffusion as dmod

    diffusers = sys.modules["diffusers"]
    diffusers.Krea2Pipeline = _FakePipeline
    diffusers.Krea2Transformer2DModel = _FakeTransformer
    _no_cache(monkeypatch)

    seen: dict[str, str] = {}

    def _krea(base, dtype, **kwargs):
        seen["krea"] = base
        return _FakePipe()

    def _ideogram(
        repo_id,
        dtype,
        hf_token = None,
        check_cancelled = None,
    ):
        seen["ideogram"] = repo_id
        return _FakePipe()

    monkeypatch.setattr(dmod, "load_krea2_pipeline", _krea)
    monkeypatch.setattr(dmod, "load_ideogram4_pipeline", _ideogram)

    krea = DiffusionBackend().load_pipeline("krea/Krea-2-Turbo", model_kind = "pipeline")
    assert seen["krea"] == "unsloth/Krea-2-Turbo"
    assert krea["repo_id"] == "krea/Krea-2-Turbo"

    ideogram = DiffusionBackend().load_pipeline("ideogram-ai/ideogram-4-fp8", model_kind = "pipeline")
    assert seen["ideogram"] == "unsloth/ideogram-4-fp8"
    assert ideogram["repo_id"] == "ideogram-ai/ideogram-4-fp8"


def test_dense_quant_pulls_the_transformer_from_the_mirror(monkeypatch):
    """The dense fallback downloads the base repo's transformer/ shards. With a nonzero baked LoRA
    the GGUF fallback is refused, so a 401 here fails the load outright."""
    _no_cache(monkeypatch)
    from core.inference import diffusion as dmod

    seen: dict = {}

    class _Transformer:
        @staticmethod
        def from_pretrained(base, **kwargs):
            seen["dense"] = base
            return object()

    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(dmod, "quantize_transformer", lambda pipe, target, **kw: "fp8")
    monkeypatch.setattr(
        DiffusionBackend,
        "_assemble_pipe",
        staticmethod(lambda *a, **k: seen.setdefault("assembled", _FakePipe())),
    )

    _pipe, scheme = DiffusionBackend()._load_dense_quant_pipeline(
        _Transformer,
        _FakePipeline,
        "black-forest-labs/FLUX.1-dev",
        "cuda:0",
        "bf16",
        None,
        types.SimpleNamespace(device = "cuda:0"),
        "fp8",
        fam = detect_family("x", override = "flux.1"),
    )
    assert scheme == "fp8"
    assert seen["dense"] == "unsloth/FLUX.1-dev"


def test_load_progress_and_delete_guard_follow_the_mirrored_companion(monkeypatch):
    """Bytes land under the mirror, so scanning the upstream reports a companion download of zero
    and leaves the repo being written to deletable."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    backend._loading = dmod._LoadingState(
        repo_id = "unsloth/FLUX.1-dev-GGUF",
        base_repo = "black-forest-labs/FLUX.1-dev",
        fetch_repo = "unsloth/FLUX.1-dev",
        expected_bytes = 10,
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo_id: 4 if repo_id == "unsloth/FLUX.1-dev" else 0),
    )

    assert backend.load_progress()["bytes_downloaded"] == 4
    assert backend.loading_repo_ids() == (
        "unsloth/FLUX.1-dev-GGUF",
        "black-forest-labs/FLUX.1-dev",
        "unsloth/FLUX.1-dev",
    )


def test_plan_memory_sizes_the_mirrored_companion_cache(monkeypatch, tmp_path):
    """A companion total of zero reads as "unknown", which picks resident placement and OOMs."""
    monkeypatch.setattr(
        DiffusionBackend,
        "_companion_cache_bytes",
        staticmethod(
            lambda base, staged = None, load_dtype = None: (
                8 * 1024 * 1024 if base == "unsloth/FLUX.1-dev" else 0
            )
        ),
    )
    seen: dict = {}
    monkeypatch.setattr(
        "core.inference.diffusion.plan_diffusion_memory",
        lambda **kwargs: seen.update(kwargs) or types.SimpleNamespace(offload_policy = "none"),
    )
    from core.inference.diffusion_device import resolve_diffusion_device_target

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1024)

    DiffusionBackend()._plan_memory(
        resolve_diffusion_device_target(),
        str(gguf),
        "black-forest-labs/FLUX.1-dev",
        detect_family("x", override = "flux.1"),
        None,
        False,
        fetch_base = "unsloth/FLUX.1-dev",
    )
    assert seen["companion_dense_mib"] == 8


def test_zimage_is_fp16_incompatible():
    # Only families whose activations overflow fp16 carry the guard: Z-Image, and Qwen-Image (NaN latents, black).
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Z-Image-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.2-klein-4B-GGUF").fp16_incompatible is False


def test_resolve_compute_dtype_promotes_fp16_for_zimage(fake_runtime, monkeypatch):
    from core.inference import diffusion_fp16_guard as guard

    torch = sys.modules["torch"]
    z = detect_family("unsloth/Z-Image-GGUF")
    q = detect_family("unsloth/FLUX.1-schnell-GGUF")
    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: True)
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float16
    monkeypatch.setenv(guard.FP16_GUARD_ENV, "0")
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    assert _resolve_diffusion_compute_dtype(z, torch.bfloat16) is torch.bfloat16
    assert _resolve_diffusion_compute_dtype(z, torch.float32) is torch.float32
    assert _resolve_diffusion_compute_dtype(q, torch.float16) is torch.float16
    assert _resolve_diffusion_compute_dtype(None, torch.float16) is torch.float16


def test_load_promotes_fp16_to_fp32_for_zimage_only(fake_runtime, monkeypatch, tmp_path):
    from core.inference import diffusion_fp16_guard as guard

    torch = sys.modules["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")

    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: True)
    kept = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
    )
    assert kept["dtype"] == "float16"
    assert str(_FakeTransformer.last["torch_dtype"]) == "torch.float16"

    monkeypatch.setenv(guard.FP16_GUARD_ENV, "0")
    z = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
    )
    assert z["device"] == "cuda" and z["dtype"] == "float32"
    assert str(_FakeTransformer.last["torch_dtype"]) == "torch.float32"

    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "FluxPipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "FluxTransformer2DModel", _FakeTransformer, raising = False)
    q = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "flux.1"
    )
    assert q["dtype"] == "float16"


def test_bad_mode_strings_fail_before_eviction(fake_runtime):
    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = object(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )
    for kwargs in (
        {"transformer_quant": "int7"},
        {"speed_mode": "warp"},
        {"attention_backend": "bogus"},
        {"transformer_cache": "bogus"},
        {"text_encoder_quant": "fp3"},
    ):
        with pytest.raises(ValueError):
            backend.load_pipeline("unsloth/Z-Image-GGUF", gguf_filename = "m.gguf", **kwargs)
        assert backend._state is not None


def test_generate_lock_split_keeps_status_and_unload_responsive(fake_runtime):
    import threading

    backend = DiffusionBackend()
    started = threading.Event()
    release = threading.Event()

    class _BlockingPipe:
        def __call__(self, **kwargs):
            started.set()
            release.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = _BlockingPipe(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 4)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target = _run)
    t.start()
    assert started.wait(5)

    assert backend.status()["loaded"] is True
    assert backend.generate_progress()["active"] is True

    cancel_ref = backend._active_generate_cancel
    assert cancel_ref is not None

    releaser = threading.Thread(target = lambda: (cancel_ref.wait(5), release.set()))
    releaser.start()
    backend.unload()
    releaser.join(5)
    assert cancel_ref.is_set()
    assert backend.status()["loaded"] is False

    t.join(5)
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()
    assert backend._active_generate_cancel is None


def test_callback_cancellation_interrupts_denoise(fake_runtime):
    import threading

    backend = DiffusionBackend()
    at_step0 = threading.Event()
    resume = threading.Event()

    class _SteppingPipe:
        def __init__(self) -> None:
            self._interrupt = False
            self.steps_run = 0

        def __call__(
            self,
            *,
            callback_on_step_end = None,
            num_inference_steps = 8,
            **kwargs,
        ):
            for i in range(num_inference_steps):
                if self._interrupt:
                    break
                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, 0.0, {})
                self.steps_run = i + 1
                if i == 0:
                    at_step0.set()
                    resume.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    pipe = _SteppingPipe()
    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = pipe,
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 8)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target = _run)
    t.start()
    assert at_step0.wait(5)
    assert backend._active_generate_cancel is not None
    backend._active_generate_cancel.set()
    resume.set()
    t.join(5)
    assert pipe._interrupt is True
    assert pipe.steps_run < 8
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()


def test_validate_load_request(tmp_path):
    backend = DiffusionBackend()
    assert backend.validate_load_request("unsloth/Z-Image-Turbo-unsloth-bnb-4bit").name == "z-image"
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image-bnb-4bit")
    with pytest.raises(ValueError, match = "single-file"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf")
    with pytest.raises(ValueError, match = "pipeline"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-bnb-4bit", gguf_filename = "q.gguf", model_kind = "pipeline"
        )
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image", gguf_filename = "model.safetensors")
    with pytest.raises(ValueError, match = "family"):
        backend.validate_load_request("meta/Llama-3", gguf_filename = "q.gguf")
    with pytest.raises(ValueError, match = r"\.gguf"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "README.md")
    assert (
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf").name
        == "z-image"
    )
    with pytest.raises(ValueError, match = ".gguf"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "model.safetensors", model_kind = "gguf"
        )
    with pytest.raises(ValueError, match = "gguf"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-FP8", gguf_filename = "q.gguf", model_kind = "single_file"
        )
    with pytest.raises(ValueError, match = "GGUF"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "pipeline")
    with pytest.raises(FileNotFoundError):
        backend.validate_load_request(
            str(tmp_path), gguf_filename = "missing.gguf", family_override = "z-image"
        )
    (tmp_path / "m.gguf").write_bytes(b"x")
    assert (
        backend.validate_load_request(
            str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
        ).name
        == "z-image"
    )
    with pytest.raises(FileNotFoundError):
        backend.validate_load_request(
            "/tmp/unsloth-definitely-missing-model",
            gguf_filename = "m.gguf",
            family_override = "z-image",
        )


def test_replacement_load_waits_for_inflight_generation(fake_runtime, tmp_path):
    # A superseding load must signal the in-flight generation and wait for _generate_lock before allocating, so two pipelines never sit in VRAM.
    import threading

    backend = DiffusionBackend()
    started = threading.Event()
    release = threading.Event()

    class _BlockingPipe:
        def __call__(self, **kwargs):
            started.set()
            release.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = _BlockingPipe(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    gen_out: dict = {}

    def _gen():
        try:
            backend.generate(prompt = "p", steps = 4)
        except Exception as exc:  # noqa: BLE001
            gen_out["exc"] = exc

    gt = threading.Thread(target = _gen)
    gt.start()
    assert started.wait(5)

    (tmp_path / "m.gguf").write_bytes(b"x")
    load_done = threading.Event()

    def _load():
        _load_m(backend, tmp_path)
        load_done.set()

    lt = threading.Thread(target = _load)
    lt.start()

    assert not load_done.wait(0.5)
    assert backend._active_generate_cancel is not None
    assert backend._active_generate_cancel.is_set()

    release.set()
    gt.join(5)
    assert load_done.wait(5)
    assert "exc" in gen_out and "cancelled" in str(gen_out["exc"]).lower()
    assert backend.status()["loaded"] is True
    assert backend.status()["repo_id"] == str(tmp_path)


def test_load_reports_memory_plan_fields_on_cpu(fake_runtime, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = _load_m(backend, tmp_path)
    assert status["offload_policy"] == "none"
    assert status["cpu_offload"] is False
    assert status["vae_tiling"] is True
    assert status["memory_mode"] == "auto"
    pipe = backend._state.pipe
    assert pipe.moved_to == "cpu" and pipe.vae_tiled and pipe.vae_sliced


@pytest.fixture
def allow_precision_fallback(monkeypatch):
    """Restore the pre-P1-2 behaviour where a DECLINED explicit precision silently loaded the GGUF.

    The tests below are about which PLANNING path ran, not about the precision contract, and the
    strict default now stops the load before their assertions can look at it. The refusal itself
    is covered by test_explicit_transformer_quant_refuses_instead_of_loading_the_gguf."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")


def _force_cuda_target(backend, monkeypatch):
    """Drive the loader down the CUDA (offload-capable) path under the stub."""
    torch = sys.modules["torch"]
    monkeypatch.setattr(backend, "_pick_device_and_dtype", lambda: ("cuda", torch.bfloat16))


def _mps_target(torch):
    """An Apple/MPS device target: no model offload, no compile, no pinned transfer."""
    from core.inference.diffusion_device import DiffusionDeviceTarget
    return DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )


def _fake_zimage_hub(monkeypatch):
    """The Z-Image GGUF / base / FP8 trio the prequant tests resolve against."""
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/Z-Image-GGUF": [_FakeSibling("Z-Image-Turbo-Q4_K_M.gguf", 4 * GB)],
            "Tongyi-MAI/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS,
            "unsloth/Z-Image-Turbo-FP8": [_FakeSibling("Z-Image-Turbo-FP8.pt", 6 * GB)],
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: "Tongyi-MAI/Z-Image-Turbo"
    )


def _cuda_backend(tmp_path, monkeypatch):
    """A CUDA-target backend with an ``m.gguf`` stub checkpoint written into ``tmp_path``."""
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    return backend


def _load_m(backend, tmp_path, **kwargs):
    """``load_pipeline`` on the ``m.gguf`` z-image stub, with per-test overrides."""
    return backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", **kwargs
    )


def test_load_memory_mode_balanced_streams_or_falls_back(fake_runtime, tmp_path, monkeypatch):
    backend = _cuda_backend(tmp_path, monkeypatch)
    status = _load_m(backend, tmp_path, memory_mode = "balanced")
    assert status["offload_policy"] in ("group", "model") and status["cpu_offload"] is True
    assert status["memory_mode"] == "balanced"
    assert backend._state.pipe.offloaded is True


def test_load_memory_mode_low_vram_engages_model_offload(fake_runtime, tmp_path, monkeypatch):
    backend = _cuda_backend(tmp_path, monkeypatch)
    status = _load_m(backend, tmp_path, memory_mode = "low_vram")
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True
    pipe = backend._state.pipe
    assert pipe.offloaded is True and pipe.moved_to is None


def test_load_refines_component_placement_after_text_encoder_quantization(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod
    from core.inference.diffusion_precision import TEQuantOutcome

    backend = _cuda_backend(tmp_path, monkeypatch)
    seen = {"quantized": False, "refined": False}

    def _quantize(*args, **kwargs):
        seen["quantized"] = True
        # The loader reads `.mode` off the report; a None mode means the encoders were left dense.
        return TEQuantOutcome(None)

    def _refine(pipe, plan):
        assert seen["quantized"] is True
        assert pipe is not None and plan.offload_policy == "model"
        seen["refined"] = True
        return plan

    monkeypatch.setattr(dmod, "quantize_text_encoders", _quantize)
    monkeypatch.setattr(dmod, "refine_memory_plan_for_components", _refine)
    _load_m(backend, tmp_path, memory_mode = "low_vram")
    assert seen == {"quantized": True, "refined": True}


def test_load_explicit_cpu_offload_engages_model_offload_on_cuda(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _cuda_backend(tmp_path, monkeypatch)
    status = _load_m(backend, tmp_path, cpu_offload = True)
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True


def test_load_speed_mode_gguf_auto_defaults_and_explicit(
    fake_runtime, tmp_path, allow_precision_fallback
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_m(backend, tmp_path)
    assert status["speed_mode"] == "default"
    status_off = _load_m(backend, tmp_path, speed_mode = "off")
    assert status_off["speed_mode"] == "off" and status_off["speed_optims"] == []
    status2 = _load_m(backend, tmp_path, speed_mode = "max")
    assert status2["speed_mode"] == "max"
    assert status2["text_encoder_quant"] is None
    status3 = _load_m(backend, tmp_path, text_encoder_quant = "nvfp4")
    assert status3["text_encoder_quant"] is None
    resolved_te = status3["resolved"]["text_encoder_quant"]
    assert resolved_te["requested"] == "nvfp4" and resolved_te["value"] == "off"
    assert resolved_te["status"] == "unsupported"


def test_load_fast_mode_stays_resident_on_cuda(fake_runtime, tmp_path, monkeypatch):
    backend = _cuda_backend(tmp_path, monkeypatch)
    status = _load_m(backend, tmp_path, memory_mode = "fast")
    assert status["offload_policy"] == "none" and status["cpu_offload"] is False
    assert backend._state.pipe.moved_to == "cuda"


def _stub_dense_quant(monkeypatch, *, scheme = "fp8"):
    """Force the dense+quant branch hermetically: a supported dense source, a
    from_pretrained on the fake transformer, and a quantizer that engages `scheme`.
    Returns a dict recording the dense-loader / quantizer calls."""
    from core.inference import diffusion as dmod

    calls: dict = {"from_pretrained": 0, "quantize": 0, "quant_mode": None}

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        calls["from_pretrained"] += 1
        calls["fp_kwargs"] = {"base": base, **kwargs}
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: scheme
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    def _quantize(pipe, target, *, mode, **kw):
        calls["quantize"] += 1
        calls["quant_mode"] = mode
        return scheme

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    return calls


@pytest.mark.parametrize(
    "stage",
    [
        "prequant",
        "prequant_miss",
        "transformer",
        "encoder",
        "pipeline",
        "placement",
        "lora_resolve",
        "lora_first",
        "lora_last",
        "lora_set",
        "quantize",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_dense_build_cancellation_boundaries(fake_runtime, tmp_path, monkeypatch, stage, cancel):
    import gc
    import weakref
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _stub_dense_quant(monkeypatch, scheme = "int8")
    monkeypatch.setattr(backend, "_dense_transformer_resident_bytes", lambda *a, **k: 0)
    (tmp_path / "model.gguf").write_bytes(b"weights")
    entered, release = threading.Event(), threading.Event()
    calls, outcome = [], {}
    live = weakref.WeakSet()

    def record(name, value = None):
        calls.append(name)
        if name == stage:
            entered.set()
            assert release.wait(5)
        return value

    def transformer(*args, **kwargs):
        value = _FakePipe()
        live.add(value)
        return record("transformer", value)

    class Pipe(_FakePipe):
        def to(self, device):
            return record("placement", super().to(device))

        def load_lora_weights(self, path, adapter_name):
            record("lora_first" if adapter_name == "first" else "lora_last")

        def set_adapters(self, *args, **kwargs):
            record("lora_set")

    def pipeline(cls, base, **kwargs):
        value = Pipe()
        value.transformer = kwargs["transformer"]
        live.add(value)
        return record("pipeline", value)

    monkeypatch.setattr(
        _FakeTransformer, "from_pretrained", staticmethod(transformer), raising = False
    )
    monkeypatch.setattr(_FakePipeline, "from_pretrained", classmethod(pipeline))
    monkeypatch.setattr(dmod, "te_prequant_pipe_kwargs", lambda *a, **k: record("encoder", {}))
    monkeypatch.setattr(
        backend,
        "_resolve_lora_set",
        lambda *a, **k: record("lora_resolve", (("first", "one", 1.0), ("last", "two", 0.5))),
    )
    monkeypatch.setattr(dmod, "quantize_transformer", lambda *a, **k: record("quantize", "int8"))
    prequant = stage.startswith("prequant")
    if prequant:
        monkeypatch.setattr(dmod, "resolve_prequant_source", lambda *a, **k: object())
        monkeypatch.setattr(
            dmod,
            "load_prequantized_transformer",
            lambda *a, **k: record(stage, None if stage == "prequant_miss" else _FakePipe()),
        )

    def load():
        try:
            outcome["result"] = _load_into(
                backend,
                tmp_path,
                transformer_quant = "int8",
                local_files_only = True,
                loras = None if prequant else [("one", 1.0), ("two", 0.5)],
            )
        except Exception as exc:
            outcome["error"] = str(exc)

    loader = threading.Thread(target = load, daemon = True)
    ejector = threading.Thread(target = backend.unload, daemon = True)
    loader.start()
    try:
        assert entered.wait(5), outcome
        if cancel:
            ejector.start()
            assert backend._cancel_event.wait(2)
    finally:
        release.set()
        loader.join(5)
        if ejector.ident is not None:
            ejector.join(5)
    assert not loader.is_alive() and not ejector.is_alive()
    assert backend._unload_waiters == backend._teardown_waiters == 0
    if cancel:
        assert "cancelled" in outcome.get("error", ""), outcome
        assert calls[-1] == stage, calls
        assert not backend.is_loaded
        gc.collect()
        assert not live
    else:
        assert "error" not in outcome, outcome
        assert backend.is_loaded
        assert backend.status()["transformer_quant"] == "int8"
        assert calls.count("quantize") == (0 if stage == "prequant" else 1)
        backend.unload()


def test_default_load_autos_dense_gate_and_falls_back(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    consulted = {"n": 0}

    def _supported(*a, **k):
        consulted["n"] += 1
        return False

    monkeypatch.setattr(dmod, "dense_transformer_supported", _supported)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_m(backend, tmp_path)
    assert consulted["n"] >= 1
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_explicit_off_load_skips_dense_quant_path(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod,
        "dense_transformer_supported",
        lambda *a, **k: pytest.fail("dense path must not run with an explicit off"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_m(backend, tmp_path, transformer_quant = "none")
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_speed_off_load_suppresses_auto_dtype_quant(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod,
        "dense_transformer_supported",
        lambda *a, **k: pytest.fail("dense path must not run under an explicit Speed=off"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = _load_m(backend, tmp_path, speed_mode = "off")
    assert status["transformer_quant"] is None
    assert status["speed_mode"] == "off"
    assert _FakeTransformer.last["path"]


def test_transformer_quant_dense_path_engaged(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert status["transformer_quant"] == "fp8"
    # No speed_mode was given, but a quantized transformer is ~30x slower eager, so it is promoted to `default`.
    assert status["speed_mode"] == "default"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1
    assert calls["quant_mode"] == "fp8"
    assert calls["fp_kwargs"]["subfolder"] == "transformer"
    assert _FakeTransformer.last == {}
    assert backend._state.pipe.moved_to == "cuda"
    assert status["offload_policy"] == "none"


def test_transformer_quant_prequant_path_engaged(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    prequant_obj = object()
    loaded: dict = {"n": 0}

    def _load_prequant(transformer_cls, base, source, **kw):
        loaded["n"] += 1
        loaded["scheme"] = kw.get("scheme")
        return prequant_obj

    monkeypatch.setattr(dmod, "load_prequantized_transformer", _load_prequant)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense from_pretrained must not run when a prequant checkpoint loads")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    monkeypatch.setattr(
        dmod,
        "quantize_transformer",
        lambda *a, **k: pytest.fail("quantize_transformer must not run on the prequant path"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        transformer_prequant_path = str(tmp_path / "zimage_fp8.pt"),
    )
    assert status["transformer_quant"] == "fp8"
    assert loaded["n"] == 1 and loaded["scheme"] == "fp8"
    assert _FakePipeline.last.get("transformer") is prequant_obj
    assert _FakeTransformer.last == {}


def test_transformer_quant_prequant_load_fails_falls_back_to_dense(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    monkeypatch.setattr(dmod, "load_prequantized_transformer", lambda *a, **k: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert status["transformer_quant"] == "fp8"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1
    assert _FakeTransformer.last == {}


def test_prequant_failure_never_pulls_unprefetched_dense_shards(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    # The prefetch skips transformer/ shards when a prequant is expected, so a dense fallback would
    # pull them under the load lock after eviction; refuse it for the GGUF build.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    monkeypatch.setattr(dmod, "load_prequantized_transformer", lambda *a, **k: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8", _transformer_prefetched = False)
    assert calls["from_pretrained"] == 0
    assert calls["quantize"] == 0
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_run_load_flags_the_transformer_prefetched_from_the_staged_file_list(monkeypatch):
    # load_pipeline reads what the prefetch actually staged; a failed size estimate stages none.
    seen: list[bool] = []
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: "Tongyi-MAI/Z-Image-Turbo"
    )
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "load_pipeline",
        lambda self, **kw: seen.append(kw["_transformer_prefetched"]),
    )
    cases = (
        (["model_index.json", "vae/config.json"], False),
        (
            ["model_index.json", "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"],
            True,
        ),
        ([], False),
    )
    for base_files, _expected in cases:
        monkeypatch.setattr(
            DiffusionBackend,
            "_estimate_download_bytes",
            staticmethod(lambda *a, _files = base_files, **k: (0, _files)),
        )
        DiffusionBackend()._run_load(
            repo_id = "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q8_0.gguf",
            model_kind = "gguf",
        )
    assert seen == [expected for _files, expected in cases]


def test_run_load_counts_a_complete_local_base_as_staged(monkeypatch, tmp_path):
    # A local base dir has no Hub listing yet holds its shards: an empty list is not a refusal.
    local = tmp_path / "Z-Image-Turbo"
    (local / "transformer").mkdir(parents = True)
    (local / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    seen: list[bool] = []
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: str(local))
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", staticmethod(lambda *a, **k: {})
    )
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_estimate_download_bytes",
        staticmethod(lambda *a, **k: (0, [])),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "load_pipeline",
        lambda self, **kw: seen.append(kw["_transformer_prefetched"]),
    )
    DiffusionBackend()._run_load(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q8_0.gguf",
        model_kind = "gguf",
    )
    assert seen == [True]
    (local / "transformer" / "diffusion_pytorch_model.safetensors").unlink()
    from core.inference import diffusion as dmod

    assert dmod._local_base_transformer_present(str(local)) is False
    assert dmod._local_base_transformer_present("Tongyi-MAI/Z-Image-Turbo") is False
    assert dmod._local_base_transformer_present(None) is False


def test_the_widening_decision_is_taken_on_the_repo_listing(monkeypatch):
    # Only the base repo's listing says which repo the fetch resolves to and if shards are cached.
    import types

    from core.inference import diffusion as dmod

    siblings = [
        types.SimpleNamespace(rfilename = name, size = 1)
        for name in (
            "model_index.json",
            "vae/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        )
    ]

    class _Api:
        def model_info(self, repo_id, **kw):
            return types.SimpleNamespace(siblings = siblings, sha = "abc")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    calls: list = []

    def _decide(companions, transformer_files):
        calls.append((tuple(companions), tuple(transformer_files)))
        return len(calls) == 1

    widened = DiffusionBackend._estimate_download_bytes(
        "unsloth/Z-Image-Turbo-GGUF",
        None,
        "Tongyi-MAI/Z-Image-Turbo",
        None,
        include_transformer = _decide,
    )[1]
    narrow = DiffusionBackend._estimate_download_bytes(
        "unsloth/Z-Image-Turbo-GGUF",
        None,
        "Tongyi-MAI/Z-Image-Turbo",
        None,
        include_transformer = _decide,
    )[1]
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert all(not f.startswith("transformer/") for f in calls[0][0])
    assert calls[0][1] == (
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    )
    assert any(f.startswith("transformer/") for f in widened)
    assert all(not f.startswith("transformer/") for f in narrow)


def test_a_cached_prequant_survives_the_resolvers_free_disk_gate(
    fake_runtime, tmp_path, monkeypatch
):
    # resolve_dense_quant_candidate's None is download-sized; a cached prequant downloads nothing.
    from core.inference import diffusion as dmod

    _stub_hosted_prequant(monkeypatch, cached = True)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert len(_dense_calls(calls, backend)) == 1


def _stub_declining_dense_quant(backend, monkeypatch):
    """Reach the dense fast path, then have the quantiser decline (the NVIDIA scenario: FP8 asked
    for, transformer FP8 disabled at runtime, the Q4_K_M GGUF loaded instead)."""
    from core.inference import diffusion as dmod

    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(dmod, "quantize_transformer", lambda pipe, target, **kw: None)


def test_declined_explicit_precision_reports_the_ask_and_the_outcome(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend = DiffusionBackend()
    _stub_declining_dense_quant(backend, monkeypatch)
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"x")
    status = _load_into(
        backend,
        tmp_path,
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        base_repo = None,
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["requested"] == "fp8"
    assert resolved["value"] == "off"
    assert resolved["source"] == "explicit"
    assert resolved["status"] == "fell_back"
    assert "build failed" in resolved["reason"]


def test_auto_precision_still_falls_back_silently(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path)
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["source"] == "auto"
    assert resolved["requested"] is None
    assert resolved["status"] == "applied"
    assert resolved["value"] == "off"


def test_explicit_transformer_quant_refuses_instead_of_loading_the_gguf(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    _stub_declining_dense_quant(backend, monkeypatch)
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError) as excinfo:
        _load_into(
            backend,
            tmp_path,
            gguf_filename = "z-image-turbo-Q4_K_M.gguf",
            base_repo = None,
            transformer_quant = "fp8",
        )
    message = str(excinfo.value)
    assert "transformer_quant='fp8' could not be used" in message
    assert "build failed" in message
    assert "Auto" in message and "Off" in message
    assert backend.status()["loaded"] is False
    assert backend._state is None


def test_explicit_off_is_honored_not_reported_as_a_fallback(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = DiffusionBackend().load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "none",
    )
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["source"] == "explicit" and resolved["requested"] == "none"
    assert resolved["value"] == "off" and resolved["status"] == "applied"


def test_begin_load_refuses_an_explicit_precision_this_host_cannot_run(
    fake_runtime, tmp_path, monkeypatch
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend.begin_load(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            model_kind = "gguf",
            transformer_quant = "fp8",
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)
    assert "CUDA GPU in bf16" in str(excinfo.value)
    assert backend.load_progress()["phase"] is None


def test_a_refusal_caused_by_a_broken_torchao_says_so_instead_of_blaming_the_gpu(
    fake_runtime, tmp_path, monkeypatch
):
    # A torch/torchao import skew must not read as a GPU limit: pip fixes one, not the other.
    from core.inference import diffusion as dmod
    import core.inference.diffusion_transformer_quant as tq

    backend = _cuda_backend(tmp_path, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(dmod, "select_transformer_quant_scheme", lambda *a, **k: None)
    monkeypatch.setattr(
        tq, "_TORCHAO_UNAVAILABLE", ("ImportError: cannot import name 'ScalingType'",)
    )

    with pytest.raises(RuntimeError) as excinfo:
        backend.begin_load(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            model_kind = "gguf",
            transformer_quant = "fp8",
        )
    message = str(excinfo.value)
    assert "transformer_quant='fp8' could not be used" in message
    assert "cannot import name 'ScalingType'" in message
    assert "not a limit of the GPU" in message
    assert "is not usable for family" not in message


def test_begin_load_refuses_an_explicit_text_encoder_quant_this_host_cannot_run(
    fake_runtime, tmp_path, monkeypatch
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "text_encoder_quant='fp8' could not be used"):
        DiffusionBackend().begin_load(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            model_kind = "gguf",
            text_encoder_quant = "fp8",
        )


def test_explicit_text_encoder_quant_refuses_when_nothing_engaged(
    fake_runtime, tmp_path, monkeypatch
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError) as excinfo:
        DiffusionBackend().load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            text_encoder_quant = "nvfp4",
        )
    assert "text_encoder_quant='nvfp4' could not be used" in str(excinfo.value)


def test_explicit_text_encoder_quant_refuses_a_partial_cast(fake_runtime, tmp_path, monkeypatch):
    # Partial engagement (one encoder cast, its sibling not) must also refuse.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_precision import TEQuantOutcome

    (tmp_path / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(
        dmod,
        "quantize_text_encoders",
        lambda *a, **k: TEQuantOutcome(
            "fp8",
            "'fp8' engaged on text_encoder but text_encoder_2 stayed dense",
            "fell_back",
            True,
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        DiffusionBackend().load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            text_encoder_quant = "fp8",
        )
    message = str(excinfo.value)
    assert "text_encoder_quant='fp8' could not be used" in message
    assert "text_encoder_2" in message
    assert "Auto" not in message


def test_text_encoder_int8_downgrade_is_reported_not_refused(fake_runtime, tmp_path, monkeypatch):
    # int8 without a measured keep-bf16 schedule becomes fp8: warn via the record, do not stop.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_precision import TEQuantOutcome

    monkeypatch.setattr(
        dmod,
        "quantize_text_encoders",
        lambda pipe, target, **kw: TEQuantOutcome(
            "fp8", "int8 has no measured keep-bf16 schedule for family 'z-image'", "fell_back"
        ),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = DiffusionBackend().load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        text_encoder_quant = "int8",
    )
    assert status["loaded"] is True
    assert status["text_encoder_quant"] == "fp8"
    resolved = status["resolved"]["text_encoder_quant"]
    assert resolved["requested"] == "int8"
    assert resolved["value"] == "fp8"
    assert resolved["status"] == "fell_back"
    assert "keep-bf16 schedule" in resolved["reason"]


def test_begin_load_never_refuses_auto(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()

    # Hold the worker so `loaded` cannot be True yet; otherwise the assertion races the thread.
    release = threading.Event()
    entered = threading.Event()
    worker: dict = {}

    def _blocked_run_load(self, **kwargs):
        worker["thread"] = threading.current_thread()
        entered.set()
        release.wait(30)

    monkeypatch.setattr(DiffusionBackend, "_run_load", _blocked_run_load)

    started = backend.begin_load(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        model_kind = "gguf",
        transformer_quant = "auto",
    )
    assert started["loaded"] is False
    assert entered.wait(30), "begin_load never started the load thread"
    # Checked from the caller: an assert failing on a non-main thread does not fail the test.
    assert worker["thread"].is_alive(), "begin_load waited for the load instead of returning"
    release.set()
    worker["thread"].join(30)


def test_transformer_quant_falls_back_to_gguf_on_failure(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(dmod, "quantize_transformer", lambda pipe, target, **kw: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_transformer_quant_skipped_when_plan_offloads(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not load when the plan offloads")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8", memory_mode = "low_vram")
    assert status["transformer_quant"] is None
    assert status["offload_policy"] == "model"
    assert _FakeTransformer.last["path"]


def test_dense_quant_skipped_when_dense_transformer_does_not_fit(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base, staged_dir = None: 40 * 1024**3),
    )
    orig_plan = DiffusionBackend._plan_memory

    def plan_wrap(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        if transformer_resident_override_mib is not None:
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", plan_wrap)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not load when it won't fit resident")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert status["transformer_quant"] is None
    assert status["offload_policy"] == "none"
    assert _FakeTransformer.last["path"]


def test_dense_quant_prequant_proceeds_but_forbids_dense_fallback(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: "prequant/path")
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base, staged_dir = None: 999 * 1024**3),
    )
    dense_refit_ran = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        # Scoped to this backend: _plan_memory is patched on the CLASS and stray loads run on daemons.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(True)
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert dense_refit_ran == [True]
    assert attempted == [False]


@pytest.mark.parametrize(
    "unreachable,expected_mib,expected_fallback",
    [
        (("fp8",), 28_561, True),
        ((), 22_930, False),
    ],
)
def test_dense_quant_replan_sizes_an_unreachable_prequant_as_dense(
    fake_runtime,
    tmp_path,
    monkeypatch,
    allow_precision_fallback,
    unreachable,
    expected_mib,
    expected_fallback,
):
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )

    def _candidate(*, force_dense = False, **_kw):
        if force_dense:
            return types.SimpleNamespace(
                transient_transformer_mib = 28_561, companions_mib = 1, prequant = False, scheme = "fp8"
            )
        return types.SimpleNamespace(
            transient_transformer_mib = 22_930, companions_mib = 1, prequant = True, scheme = "fp8"
        )

    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", _candidate)
    sized: list = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        sized.append(transformer_resident_override_mib)
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_m(backend, tmp_path, transformer_quant = "fp8", _prequant_unreachable = unreachable)
    assert sized == [expected_mib]
    assert attempted == [expected_fallback]


def test_dense_quant_unreachable_prequant_does_not_skip_the_dense_decline(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: "prequant/path")
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base, staged_dir = None: 999 * 1024**3),
    )
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        if transformer_resident_override_mib is not None and self is backend:
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8", _prequant_unreachable = ("fp8",))
    assert attempted == []
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_dense_quant_replan_retries_once_on_transient_free_undercount(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    # A transient foreign allocation can make an empty card look full, so the loader retries once.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 33_831, companions_mib = 46_157, prequant = True
        ),
    )
    replan_calls = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        replan_calls.append(True)
        if len(replan_calls) == 1:
            return types.SimpleNamespace(
                offload_policy = "model",
                estimates = {"resident_required_mib": 90_228, "safe_device_budget_mib": 40_000},
                device_memory = types.SimpleNamespace(
                    total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
                ),
                reasons = ("companions exceed budget",),
            )
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_m(backend, tmp_path, transformer_quant = "int8")
    assert replan_calls == [True, True]
    assert attempted == [False]


def test_dense_quant_replan_no_retry_when_capacity_truly_short(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 33_831, companions_mib = 46_157, prequant = True
        ),
    )
    replan_calls = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        replan_calls.append(True)
        return types.SimpleNamespace(
            offload_policy = "model",
            estimates = {"resident_required_mib": 150_000, "safe_device_budget_mib": 40_000},
            device_memory = types.SimpleNamespace(
                total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
            ),
            reasons = ("companions exceed budget",),
        )

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_m(backend, tmp_path, transformer_quant = "int8")
    assert replan_calls == [True]


def _decline_dense_quant(backend, monkeypatch, tmp_path):
    """Configure the harness so the dense-quant fast path is declined for capacity
    (mirrors test_dense_quant_replan_no_retry_when_capacity_truly_short)."""
    import dataclasses

    from core.inference import diffusion as dmod

    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 33_831, companions_mib = 46_157, prequant = False
        ),
    )
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        return types.SimpleNamespace(
            offload_policy = "model",
            estimates = {"resident_required_mib": 150_000, "safe_device_budget_mib": 40_000},
            device_memory = types.SimpleNamespace(
                total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
            ),
            reasons = ("companions exceed budget",),
        )

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    (tmp_path / "m.gguf").write_bytes(b"x")


def test_declined_dense_with_baked_loras_fails_instead_of_silent_drop(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    _decline_dense_quant(backend, monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match = "LoRA adapters could not be applied"):
        _load_m(backend, tmp_path, transformer_quant = "int8", loras = [("adapter", 1.0)])


def test_declined_dense_without_loras_still_falls_back_to_gguf(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend = DiffusionBackend()
    _decline_dense_quant(backend, monkeypatch, tmp_path)
    result = _load_m(backend, tmp_path, transformer_quant = "int8", loras = [("adapter", 0.0)])
    assert result is not None
    assert backend.status()["transformer_quant"] is None


def test_an_uncompilable_gguf_baking_loras_keeps_the_dense_build(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _decline_dense_quant(backend, monkeypatch, tmp_path)
    monkeypatch.setattr(dmod, "family_compiles_regionally", lambda _fam: False)
    monkeypatch.setattr(dmod, "_plan_proves_resident", lambda _plan: True)
    with pytest.raises(RuntimeError, match = "LoRA adapters could not be applied"):
        _load_m(backend, tmp_path, loras = [("adapter", 1.0)])


class _BakePipe:
    def __init__(self):
        self.calls: list = []

    def load_lora_weights(
        self,
        path,
        adapter_name = None,
    ):
        self.calls.append(("load", path, adapter_name))

    def set_adapters(
        self,
        names,
        adapter_weights = None,
    ):
        self.calls.append(("set", tuple(names), tuple(adapter_weights)))


def test_dense_quant_lora_bake_attaches_before_quantize(fake_runtime, monkeypatch):
    # Adapters attach before quantize_transformer: post-quant torchao dispatch TypeErrors.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    prequant_consulted = []
    monkeypatch.setattr(
        dmod,
        "resolve_prequant_source",
        lambda *a, **k: prequant_consulted.append(True) or None,
    )
    order: list = []

    class FakeTransformerCls:
        @staticmethod
        def from_pretrained(*a, **k):
            order.append("dense_load")
            return object()

    pipe = _BakePipe()
    monkeypatch.setattr(DiffusionBackend, "_assemble_pipe", staticmethod(lambda *a, **k: pipe))
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(lambda specs, **k: (("sloth", "/adapters/sloth.safetensors", 0.8),)),
    )

    def fake_quantize(p, target, **k):
        order.append("quantize")
        assert any(c[0] == "load" for c in p.calls), "adapters must attach before quantize"
        return "int8"

    monkeypatch.setattr(dmod, "quantize_transformer", fake_quantize)
    got_pipe, scheme = backend._load_dense_quant_pipeline(
        FakeTransformerCls,
        object,
        "base/repo",
        "cuda",
        "bf16",
        None,
        types.SimpleNamespace(device = "cuda", dtype = "bf16"),
        "int8",
        fam = types.SimpleNamespace(name = "z-image"),
        lora_specs = [("sloth", 0.8)],
    )
    assert scheme == "int8"
    assert prequant_consulted == []
    assert order == ["dense_load", "quantize"]
    assert pipe.calls[0] == ("load", "/adapters/sloth.safetensors", "sloth")
    assert pipe.calls[1] == ("set", ("sloth",), (0.8,))
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 0.8),)
    assert pipe._unsloth_loras_baked is True


def _quant_lora_state(pipe, quant = "int8"):
    return types.SimpleNamespace(
        pipe = pipe,
        transformer_quant = quant,
        kind = "gguf",
        family = types.SimpleNamespace(name = "z-image"),
        hf_token = None,
        speed_optims = ("compiled",),
    )


def test_apply_loras_quant_unbaked_requires_reload(monkeypatch):
    # Topology is frozen after quantize_ + compile, so a new adapter needs a reload (clean 400).
    backend = DiffusionBackend()
    pipe = _BakePipe()
    with pytest.raises(ValueError, match = "Reload the model with the adapter selection"):
        backend._apply_loras(_quant_lora_state(pipe), [("sloth", 1.0)], threading.Event())
    backend._apply_loras(_quant_lora_state(pipe), [], threading.Event())
    assert pipe.calls == []


def test_apply_loras_quant_baked_matrix(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(
            lambda specs, **k: tuple((i, f"/adapters/{i}.safetensors", w) for (i, w) in specs)
        ),
    )

    def baked_pipe():
        pipe = _BakePipe()
        pipe._unsloth_loras = (("sloth", "/adapters/sloth.safetensors", 0.8),)
        pipe._unsloth_loras_baked = True
        return pipe

    ev = threading.Event()
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 0.8)], ev)
    assert pipe.calls == []
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 1.4)], ev)
    assert pipe.calls == [("set", ("sloth",), (1.4,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 1.4),)
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert pipe.calls == [("set", ("sloth",), (0.0,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 0.0),)
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert len(pipe.calls) == 1
    pipe = baked_pipe()
    with pytest.raises(ValueError, match = "Reload the model with the new adapter selection"):
        backend._apply_loras(_quant_lora_state(pipe), [("other", 1.0)], ev)


class _GraphHandle:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1
        return self


def test_a_failed_lora_switch_drops_the_captured_graphs(monkeypatch):
    """A failed switch records an empty applied set, so the later "none requested" reset never fires."""
    backend = DiffusionBackend()
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(
            lambda specs, **k: tuple((i, f"/adapters/{i}.safetensors", w) for (i, w) in specs)
        ),
    )

    class _FailingPipe(_BakePipe):
        def load_lora_weights(
            self,
            path,
            adapter_name = None,
        ):
            raise RuntimeError("size mismatch for the adapter")

        def unload_lora_weights(self):
            self.calls.append(("unload",))

    pipe = _FailingPipe()
    pipe._unsloth_loras = (("sloth", "/adapters/sloth.safetensors", 1.0),)
    handle = _GraphHandle()
    state = types.SimpleNamespace(
        pipe = pipe,
        transformer_quant = None,
        kind = "dense",
        family = types.SimpleNamespace(name = "z-image"),
        hf_token = None,
        speed_optims = ("cuda_graph",),
        cuda_graphs = (handle,),
    )

    with pytest.raises(ValueError, match = "Failed to apply LoRA"):
        backend._apply_loras(state, [("other", 1.0)], threading.Event())

    assert pipe._unsloth_loras == ()
    assert handle.resets == 1
    backend._apply_loras(state, [], threading.Event())
    assert handle.resets == 1


def test_baked_lora_names_survive_being_disabled_at_generate_time(monkeypatch):
    # A baked load's applied set is always empty (zero weights dropped), so record baked separately.
    from core.inference.diffusion import _active_lora_pairs, _baked_lora_names

    backend = DiffusionBackend()
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(
            lambda specs, **k: tuple((i, f"/adapters/{i}.safetensors", w) for (i, w) in specs)
        ),
    )
    pipe = _BakePipe()
    pipe._unsloth_loras = (("sloth", "/adapters/sloth.safetensors", 0.8),)
    pipe._unsloth_loras_baked = True

    backend._apply_loras(_quant_lora_state(pipe), [], threading.Event())
    assert _active_lora_pairs(pipe) == []
    assert _baked_lora_names(pipe) == ["sloth"]

    plain = _BakePipe()
    plain._unsloth_loras = (("sloth", "/adapters/sloth.safetensors", 0.8),)
    assert _baked_lora_names(plain) == []
    assert _active_lora_pairs(plain) == [("sloth", 0.8)]


def test_assemble_pipe_routes_krea2_per_component(monkeypatch):
    # krea ships transformers-5.x configs and no top-level tokenizer files, so the quant fast path must assemble per-component.
    from core.inference import diffusion as dmod

    calls: dict = {}

    class Pipe:
        def to(self, device):
            calls["device"] = device
            return self

    def fake_loader(
        base,
        dtype,
        hf_token = None,
        transformer = None,
        text_encoder = None,
        # Spelled out, not **kwargs: pins the exact production signature incl. local_files_only.
        local_files_only = False,
        check_cancelled = None,
    ):
        assert callable(check_cancelled)
        calls["base"] = base
        calls["transformer"] = transformer
        calls["local_files_only"] = local_files_only
        return Pipe()

    monkeypatch.setattr(dmod, "load_krea2_pipeline", fake_loader)
    # Pin the mirror decision, else the assertion below reads the developer's real HF cache.
    _no_cache(monkeypatch)

    class ExplodingPipeline:
        @staticmethod
        def from_pretrained(*a, **k):
            raise AssertionError("krea-2 must not go through Pipeline.from_pretrained")

    marker = object()
    pipe = dmod.DiffusionBackend._assemble_pipe(
        ExplodingPipeline,
        "krea/Krea-2-Turbo",
        marker,
        "bf16",
        None,
        "cuda:0",
        fam = types.SimpleNamespace(name = "krea-2"),
    )
    assert isinstance(pipe, Pipe)
    assert calls == {
        "base": "unsloth/Krea-2-Turbo",
        "transformer": marker,
        "device": "cuda:0",
        "local_files_only": False,
    }


def test_dense_quant_unusable_prequant_path_runs_dense_refit(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base, staged_dir = None: 999 * 1024**3),
    )
    dense_refit_ran = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        # Scoped to this backend: _plan_memory is patched on the CLASS and stray loads run on daemons.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(True)
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    monkeypatch.setattr(
        DiffusionBackend, "_load_dense_quant_pipeline", lambda self, *a, **k: (None, None)
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        transformer_prequant_path = str(tmp_path / "not-allowlisted.pt"),
    )
    assert dense_refit_ran == [True]
    assert backend.status()["loaded"] is True


def test_transformer_quant_unsupported_scheme_skips_dense_download(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: None
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not download when the scheme is unsupported")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = _load_m(backend, tmp_path, transformer_quant = "fp8")
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_base_file_downloaded_include_transformer_flag():
    from core.inference.diffusion import _base_file_downloaded

    assert _base_file_downloaded("transformer/diffusion_pytorch_model-00001.safetensors") is False
    assert (
        _base_file_downloaded(
            "transformer/diffusion_pytorch_model-00001.safetensors", include_transformer = True
        )
        is True
    )
    assert _base_file_downloaded("assets/teaser.png", include_transformer = True) is False
    assert _base_file_downloaded("README.md", include_transformer = True) is False


def test_dense_quant_prefetch_capacity_gate(fake_runtime, monkeypatch):
    # Widening fetches multi-GB shards, so compare steady_total against TOTAL device capacity.
    from core.inference import diffusion as dmod
    from core.inference import diffusion_memory as dmem

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Qwen-Image-GGUF")

    def candidate_with(steady):
        return lambda **kw: types.SimpleNamespace(prequant = False, steady_total_mib = steady)

    monkeypatch.setattr(
        dmem,
        "snapshot_device_memory",
        lambda target: types.SimpleNamespace(
            total_mib = 24_564, free_mib = 24_000, memory_kind = "discrete_vram"
        ),
    )
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", candidate_with(39_900))
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is False
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", candidate_with(12_000))
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is True
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(prequant = False),
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is True


def test_dense_quant_prefetch_needed_gates(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")

    seen: list = []

    def fake_candidate(
        *,
        fam,
        target,
        requested,
        base_repo = None,
        prequant_path = None,
        force_dense = False,
        logger = None,
    ):
        seen.append(requested)
        return types.SimpleNamespace(prequant = False)

    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", fake_candidate)
    _stub_dense_transformer_cached(monkeypatch, cached = True)

    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is True
    assert seen[-1] == "fp8"
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert seen[-1] == "auto"
    # balanced / low_vram force offload whatever the footprint, so they must not widen.
    before = len(seen)
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "balanced"}
        )
        is False
    )
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "low_vram"}
        )
        is False
    )
    assert (
        backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8", "cpu_offload": True})
        is False
    )
    assert len(seen) == before
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast"}
        )
        is True
    )
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast", "cpu_offload": True}
        )
        is True
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "none"}) is False
    assert (
        backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8", "speed_mode": "off"})
        is False
    )
    monkeypatch.setattr(
        dmod, "resolve_dense_quant_candidate", lambda **kw: types.SimpleNamespace(prequant = True)
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False


_HOSTED_PREQUANT = types.SimpleNamespace(
    kind = "repo",
    location = "unsloth/Z-Image-Turbo-FP8",
    filename = "Z-Image-Turbo-FP8.pt",
    fallback_filenames = ("transformer_fp8.pt",),
)


def _stub_hosted_prequant(monkeypatch, *, cached: bool):
    """Resolve the family's hosted fp8 checkpoint, present or absent from the cache."""
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: _HOSTED_PREQUANT)
    monkeypatch.setattr(dmod, "prequant_checkpoint_cached", lambda source, **kw: cached)


def _stub_dense_transformer_cached(monkeypatch, *, cached: bool):
    """Answer "are the base repo's dense transformer/ shards already on disk?" without a cache.

    Same rule as the hosted prequant above, applied to the base repo's own shards: uncached, an
    auto quant must not buy a second denoiser for a GGUF pick."""
    from core.inference import diffusion as dmod
    monkeypatch.setattr(dmod, "_dense_transformer_cached", lambda *a, **k: cached)


def _stub_dense_candidate(monkeypatch, *, prequant: bool):
    """Pin what the fast path would open: a PRE-QUANT checkpoint, or the base repo's dense shards.

    ``resolve_dense_quant_candidate`` is the resolver both the plan and the load re-plan against,
    so pinning it here pins the same answer for both."""
    from core.inference import diffusion as dmod
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            prequant = prequant,
            steady_total_mib = 1,
            transient_transformer_mib = 1,
            companions_mib = 1,
        ),
    )


def _spy_dense_quant(monkeypatch):
    """Record every dense/prequant fast-path build and keep it from running.

    Keyed by backend INSTANCE: the patch is class-level and an earlier test's begin_load can leave
    a daemon thread still loading, so a bare count is not this test's. Read via ``_dense_calls``."""
    calls: list = []

    def _record(self, *a, **k):
        calls.append((self, k.get("prequant_path")))
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", _record)
    return calls


def _dense_calls(calls, backend):
    """The recorded fast-path builds belonging to ``backend`` alone."""
    return [prequant_path for (owner, prequant_path) in calls if owner is backend]


def test_auto_quant_declines_an_uncached_hosted_prequant(fake_runtime, tmp_path, monkeypatch):
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    status = _load_m(backend, tmp_path)

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]


def test_auto_quant_takes_a_hosted_prequant_that_is_already_cached(
    fake_runtime, tmp_path, monkeypatch
):
    _stub_hosted_prequant(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path)

    assert len(_dense_calls(calls, backend)) == 1


@pytest.mark.parametrize("loras", [[("adapter", 0.0)], [("a", 0.0), ("b", 0.0)]])
def test_all_zero_weight_loras_do_not_look_like_a_bake(loras, fake_runtime, tmp_path, monkeypatch):
    # Weight 0 is disabled, so a truthy lora list alone must not count as a bake.
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, loras = loras)

    assert _dense_calls(calls, backend) == []


def test_a_weighted_lora_is_still_treated_as_a_bake(fake_runtime, tmp_path, monkeypatch):
    _stub_hosted_prequant(monkeypatch, cached = False)
    _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match = "LoRA adapters could not be applied"):
        _load_m(backend, tmp_path, loras = [("adapter", 0.8)])


def test_an_explicit_quant_request_still_downloads_the_hosted_prequant(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, transformer_quant = "fp8")

    assert len(_dense_calls(calls, backend)) == 1


def test_a_baked_lora_load_is_unaffected_by_the_prequant_cache(fake_runtime, tmp_path, monkeypatch):
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match = "LoRA"):
        _load_m(backend, tmp_path, loras = [("adapter", 1.0)])

    assert len(_dense_calls(calls, backend)) == 1


@pytest.mark.parametrize(
    "lora_specs, consults_prequant",
    [([("adapter", 0.0)], True), ([("adapter", 0.8)], False), (None, True)],
)
def test_the_dense_builder_skips_the_prequant_only_for_a_real_bake(
    lora_specs, consults_prequant, fake_runtime, monkeypatch
):
    import contextlib

    from core.inference import diffusion as dmod

    consulted: list = []
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(
        dmod, "resolve_prequant_source", lambda *a, **k: consulted.append(1) or None
    )
    backend = DiffusionBackend()
    target = _force_cuda_target(backend, monkeypatch)

    with contextlib.suppress(Exception):
        backend._load_dense_quant_pipeline(
            object(),
            object(),
            "Tongyi-MAI/Z-Image-Turbo",
            "cuda",
            None,
            None,
            target,
            "fp8",
            None,
            fam = detect_family("unsloth/Z-Image-GGUF"),
            lora_specs = lora_specs,
        )

    assert bool(consulted) is consults_prequant


def test_the_plan_does_not_force_a_dense_bake_for_disabled_adapters(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-GGUF")
    forced: list = []
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: forced.append(kw.get("force_dense")) or types.SimpleNamespace(prequant = True),
    )
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_transformer_cached(monkeypatch, cached = True)

    backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.0)]})
    backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.8)]})

    assert forced == [False, True]


def test_the_plan_reads_pydantic_lora_specs_as_the_load_reads_tuples(fake_runtime, monkeypatch):
    # LoraSpec unpacks as (field, value) pairs, unlike the (id, weight) tuples /images/load sends.
    from core.inference import diffusion as dmod
    from models.inference import LoraSpec

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-GGUF")
    consulted: list = []
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: consulted.append(kw) or types.SimpleNamespace(prequant = True),
    )
    _stub_hosted_prequant(monkeypatch, cached = False)

    assert (
        backend._dense_quant_prefetch_needed(fam, {"loras": [LoraSpec(id = "adapter", weight = 0)]})
        is False
    )
    assert consulted == []

    backend._dense_quant_prefetch_needed(fam, {"loras": [LoraSpec(id = "adapter", weight = 0.8)]})
    assert consulted != []


def test_the_plan_reads_zero_weight_loras_exactly_as_the_load_does(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-GGUF")
    consulted: list = []
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: consulted.append(kw) or types.SimpleNamespace(prequant = True),
    )
    _stub_hosted_prequant(monkeypatch, cached = False)

    for loras in ([("adapter", 0.0)], [("a", 0.0), ("b", 0.0)]):
        consulted.clear()
        assert backend._dense_quant_prefetch_needed(fam, {"loras": loras}) is False
        assert consulted == []

    consulted.clear()
    backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.8)]})
    assert consulted != []


def test_dense_quant_prefetch_declines_with_the_load(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-GGUF")
    consulted: list = []
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: consulted.append(kw) or types.SimpleNamespace(prequant = True),
    )

    _stub_dense_transformer_cached(monkeypatch, cached = True)

    _stub_hosted_prequant(monkeypatch, cached = False)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    assert consulted == []
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False
    _stub_hosted_prequant(monkeypatch, cached = True)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    assert len(consulted) == 2


def test_auto_quant_declines_an_uncached_dense_base(fake_runtime, monkeypatch):
    # An auto quant never downloads a second transformer for a GGUF pick.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Qwen-Image-GGUF")
    consulted: list = []
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: consulted.append(kw) or types.SimpleNamespace(prequant = False),
    )
    _stub_hosted_prequant(monkeypatch, cached = True)

    _stub_dense_transformer_cached(monkeypatch, cached = False)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    assert consulted == []

    _stub_dense_transformer_cached(monkeypatch, cached = True)
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert len(consulted) == 1


def test_an_explicit_transformer_quant_still_buys_the_dense_base(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Qwen-Image-GGUF")
    monkeypatch.setattr(
        dmod, "resolve_dense_quant_candidate", lambda **kw: types.SimpleNamespace(prequant = False)
    )
    _stub_dense_transformer_cached(monkeypatch, cached = False)

    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is True
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is True
    assert backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.8)]}) is True


def test_the_load_declines_when_the_prefetch_skipped_the_dense_shards(
    fake_runtime, tmp_path, monkeypatch
):
    # Unstaged shards would be fetched under the load lock after eviction, past 100% progress.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    status = _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]

    calls.clear()
    backend2 = DiffusionBackend()
    _force_cuda_target(backend2, monkeypatch)
    backend2.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = True,
    )
    assert len(_dense_calls(calls, backend2)) == 1


def test_an_unstaged_transformer_still_takes_a_CACHED_prequant(fake_runtime, tmp_path, monkeypatch):
    # A cached prequant stages no transformer/ shards because it replaces them, not as a decline.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert len(_dense_calls(calls, backend)) == 1


def test_an_unstaged_prequant_load_still_forbids_the_dense_fallback(
    fake_runtime, tmp_path, monkeypatch
):
    # With no transformer/ staged, a failed prequant load must raise, not materialise dense bf16.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    seen: list = []

    def _record(self, *a, **k):
        seen.append(k.get("allow_dense_fallback"))
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", _record)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert seen == [False]


def test_an_uncached_prequant_still_declines_before_the_candidate_is_asked(
    fake_runtime, tmp_path, monkeypatch
):
    _stub_hosted_prequant(monkeypatch, cached = False)
    _stub_dense_candidate(monkeypatch, prequant = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    status = _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert _dense_calls(calls, backend) == []
    assert status["transformer_quant"] is None


def test_a_resolver_with_no_answer_reads_as_the_dense_base(fake_runtime, tmp_path, monkeypatch):
    # None with no size entry means no basis, not a prequant: read it as dense to match the plan.
    from core.inference import diffusion as dmod

    _stub_hosted_prequant(monkeypatch, cached = True)
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert _dense_calls(calls, backend) == []


def test_a_raising_resolver_reads_as_the_dense_base(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    def _boom(**kw):
        raise RuntimeError("resolver is on fire")

    _stub_hosted_prequant(monkeypatch, cached = True)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", _boom)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    status = _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True


def test_the_plan_and_the_load_agree_on_a_cached_prequant(fake_runtime, tmp_path, monkeypatch):
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    _stub_dense_transformer_cached(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    (tmp_path / "m.gguf").write_bytes(b"x")

    assert backend._dense_quant_prefetch_needed(fam, {"base_repo": "Tongyi-MAI/Z-Image-Turbo"}) is (
        False
    )
    _load_m(backend, tmp_path, _transformer_prefetched = False)
    assert len(_dense_calls(calls, backend)) == 1


def test_dense_transformer_cached_asks_the_repo_the_fetch_will_use(
    fake_runtime, monkeypatch, tmp_path
):
    # prefer_ungated_mirror decides from the widened listing, so the probe follows it, not a union.
    from core.inference import diffusion as dmod

    asked: list = []

    def _holds(repo_id, files):
        asked.append((repo_id, tuple(files)))
        return repo_id == "black-forest-labs/FLUX.2-dev"

    monkeypatch.setattr(dmod, "cache_holds_files", _holds)
    monkeypatch.setattr(
        dmod,
        "prefer_ungated_mirror",
        lambda base, *a, files = None: (
            base if files and "vae/config.json" in files else "unsloth/FLUX.2-dev"
        ),
    )

    shards = ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",)
    assert (
        dmod._dense_transformer_cached(
            "black-forest-labs/FLUX.2-dev",
            companion_files = ("vae/config.json",),
            transformer_files = shards,
        )
        is True
    )
    assert (
        dmod._dense_transformer_cached(
            "black-forest-labs/FLUX.2-dev",
            companion_files = ("text_encoder/model.safetensors",),
            transformer_files = shards,
        )
        is False
    )
    assert asked[0][0] == "black-forest-labs/FLUX.2-dev"
    assert asked[1][0] == "unsloth/FLUX.2-dev"


def test_dense_transformer_cached_follows_the_mirror_the_widened_fetch_picks(
    fake_runtime, monkeypatch
):
    # The upstream wins only when its cache holds the WHOLE widened fetch.
    from core.inference import diffusion as dmod

    companions = ("vae/config.json", "text_encoder/model.safetensors")
    shards = ("transformer/diffusion_pytorch_model-00001-of-00001.safetensors",)
    upstream_cache = set(companions)
    mirror_cache = set(shards)

    monkeypatch.setattr(
        dmod,
        "prefer_ungated_mirror",
        lambda base, *a, files = None: (
            base if files and set(files) <= upstream_cache else "unsloth/FLUX.2-dev"
        ),
    )
    monkeypatch.setattr(
        dmod,
        "cache_holds_files",
        lambda repo_id, files: set(files)
        <= (mirror_cache if repo_id == "unsloth/FLUX.2-dev" else upstream_cache),
    )

    assert (
        dmod._dense_transformer_cached(
            "black-forest-labs/FLUX.2-dev",
            companion_files = companions,
            transformer_files = shards,
        )
        is True
    )


def test_dense_transformer_cached_requires_every_shard(fake_runtime, monkeypatch, tmp_path):
    # A cancelled pull leaves partial shards; a hit there would download tens of GB more.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_families import cache_holds_files

    resident = {"transformer/model-00001-of-00002.safetensors"}
    monkeypatch.setattr(
        dmod,
        "cache_holds_files",
        lambda repo_id, files: bool(files) and set(files) <= resident,
    )
    both = (
        "transformer/model-00001-of-00002.safetensors",
        "transformer/model-00002-of-00002.safetensors",
    )
    assert dmod._dense_transformer_cached("Qwen/Qwen-Image-Edit-2511", transformer_files = both) is (
        False
    )
    resident.add("transformer/model-00002-of-00002.safetensors")
    assert dmod._dense_transformer_cached("Qwen/Qwen-Image-Edit-2511", transformer_files = both) is (
        True
    )
    assert dmod._dense_transformer_cached("Qwen/Qwen-Image-Edit-2511") is False
    assert dmod._dense_transformer_cached(None, transformer_files = both) is False
    assert dmod._dense_transformer_cached("  ", transformer_files = both) is False
    assert cache_holds_files("Qwen/Qwen-Image-Edit-2511", ()) is False


def test_dense_transformer_cached_survives_an_unreadable_cache(fake_runtime, monkeypatch):
    from core.inference import diffusion as dmod

    def _boom(repo_id, files):
        raise OSError("cache is on fire")

    monkeypatch.setattr(dmod, "cache_holds_files", _boom)
    assert (
        dmod._dense_transformer_cached(
            "Qwen/Qwen-Image-Edit-2511",
            transformer_files = ("transformer/model.safetensors",),
        )
        is False
    )


def test_status_names_the_gguf_quant_that_actually_ran(fake_runtime, tmp_path):
    # dtype is the compute dtype (bf16 on every CUDA load); gguf_variant names the opened file.
    backend = DiffusionBackend()
    (tmp_path / "z-image-turbo-Q8_0.gguf").write_bytes(b"x")
    _load_into(backend, tmp_path, gguf_filename = "z-image-turbo-Q8_0.gguf", base_repo = None)
    status = backend.status()
    assert status["model_kind"] == "gguf"
    assert status["transformer_quant"] is None
    assert status["gguf_variant"] == "Q8_0"


def test_status_reports_the_dense_build_when_it_replaced_the_gguf(
    fake_runtime, tmp_path, monkeypatch
):
    # When the dense fast path took over, the .gguf is never opened, so prefer transformer_quant.
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    _stub_dense_quant(monkeypatch, scheme = "fp8")
    (tmp_path / "z-image-turbo-Q8_0.gguf").write_bytes(b"x")
    status = _load_into(
        backend,
        tmp_path,
        gguf_filename = "z-image-turbo-Q8_0.gguf",
        base_repo = None,
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    assert backend.status()["transformer_quant"] == "fp8"


def test_status_names_the_nvfp4_kernel_backend_that_actually_ran():
    from types import SimpleNamespace

    import core.inference.diffusion as dmod

    class _FlashInferLinear:
        a_gsf = 1.0

    _FlashInferLinear.__name__ = "NVFP4FlashInferLinear"

    def _state(quant, modules, **kw):
        denoiser = SimpleNamespace(modules = lambda: iter(modules), **kw)
        return SimpleNamespace(
            transformer_quant = quant,
            pipe = SimpleNamespace(transformer = denoiser),
            family = SimpleNamespace(denoiser_attr = "transformer"),
        )

    assert dmod._transformer_quant_backend(_state("nvfp4", [_FlashInferLinear()])) == "flashinfer"
    assert dmod._transformer_quant_backend(_state("nvfp4", [object()])) == "torchao"
    assert (
        dmod._transformer_quant_backend(
            _state("nvfp4", [object()], _unsloth_nvfp4_backend = "flashinfer")
        )
        == "flashinfer"
    )
    for scheme in ("fp8", "int8", "mxfp8", None):
        assert dmod._transformer_quant_backend(_state(scheme, [_FlashInferLinear()])) is None

    def _boom():
        raise RuntimeError("no")

    hostile = SimpleNamespace(
        transformer_quant = "nvfp4",
        pipe = SimpleNamespace(transformer = SimpleNamespace(modules = _boom)),
        family = SimpleNamespace(denoiser_attr = "transformer"),
    )
    assert dmod._transformer_quant_backend(hostile) is None


def test_the_unloaded_status_declares_the_quant_backend_key():
    status = DiffusionBackend().status()
    assert "transformer_quant_backend" in status
    assert status["transformer_quant_backend"] is None


def test_diffusion_status_response_declares_the_quant_backend():
    from models.inference import DiffusionStatusResponse

    resp = DiffusionStatusResponse(loaded = True, transformer_quant_backend = "flashinfer")
    assert resp.model_dump()["transformer_quant_backend"] == "flashinfer"
    assert DiffusionStatusResponse(loaded = True).model_dump()["transformer_quant_backend"] is None


def test_status_carries_no_gguf_variant_when_nothing_is_loaded():
    assert DiffusionBackend().status()["gguf_variant"] is None


def test_diffusion_status_response_carries_resolved():
    # The backend records per-control auto-policy provenance on state.resolved, so the response model must declare the field or Pydantic drops it.
    from models.inference import DiffusionStatusResponse

    rec = {"transformer_quant": {"value": "fp8", "source": "auto", "reason": "blackwell"}}
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    assert resp.model_dump()["resolved"] == {
        "transformer_quant": {
            "value": "fp8",
            "requested": None,
            "source": "auto",
            "status": "applied",
            "reason": "blackwell",
            "artifact": None,
            "replaced": None,
        }
    }
    assert DiffusionStatusResponse(loaded = False).resolved is None


def test_diffusion_status_response_carries_requested_precision():
    from models.inference import DiffusionStatusResponse

    rec = {
        "transformer_quant": {
            "value": "off",
            "requested": "fp8",
            "source": "explicit",
            "status": "fell_back",
            "reason": "the dense bf16 transformer does not fit resident",
        }
    }
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    assert resp.model_dump()["resolved"] == {
        "transformer_quant": {**rec["transformer_quant"], "artifact": None, "replaced": None}
    }


def test_diffusion_status_response_carries_the_prequant_artifact():
    from models.inference import DiffusionStatusResponse

    rec = {
        "transformer_quant": {
            "value": "fp8",
            "source": "auto",
            "reason": "seeded",
            "artifact": "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-FP8.pt",
        }
    }
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    dumped = resp.model_dump()["resolved"]["transformer_quant"]
    assert dumped["artifact"] == "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-FP8.pt"


def test_diffusion_status_response_carries_gguf_variant():
    from models.inference import DiffusionStatusResponse
    assert DiffusionStatusResponse(loaded = True, gguf_variant = "Q8_0").gguf_variant == "Q8_0"


def test_companion_cache_bytes_local_dir_excludes_transformer(tmp_path):
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "model.safetensors").write_bytes(b"y" * 50)
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"z" * 9999)
    (tmp_path / "model_index.json").write_bytes(b"{}")
    total = DiffusionBackend._companion_cache_bytes(str(tmp_path))
    assert total == 150


def test_plan_memory_dense_replan_does_not_double_count_prefetched_transformer(monkeypatch):
    # Prefetched transformer/ shards share the blob cache _companion_cache_bytes sums: no double count.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

    backend = DiffusionBackend()
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 40000, 40960),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 4000)
    monkeypatch.setattr(
        DiffusionBackend,
        "_companion_cache_bytes",
        staticmethod(lambda base: (8000 + 24000) * 1024 * 1024),
    )
    fam = types.SimpleNamespace(name = "z-image")
    plan = backend._plan_memory(
        target,
        None,
        "org/base",
        fam,
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 12000,
        companion_override_mib = 8000,
    )
    # 12000 + 8000 + 4000 + 2048 overhead = 26048 MiB, fits the ~36 GiB budget. A double-count would have exceeded it and offloaded.
    assert plan.offload_policy == OFFLOAD_NONE


def test_plan_memory_pipeline_replan_prices_the_quantised_transformer(monkeypatch):
    """A pipeline re-plan reads the quant-size overrides instead of the whole-repo cache.

    The pipeline branch sizes the repo as one download, the bf16 footprint the re-plan exists to
    replace; ignoring the overrides returned the bf16 plan unchanged, so an offloaded pipeline
    could never reach the fast path.
    """
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

    backend = DiffusionBackend()
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 40000, 40960),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 4000)
    monkeypatch.setattr(
        DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: (24000 + 8000) * 1024 * 1024)
    )
    plan = backend._plan_memory(
        target,
        None,
        "org/base",
        types.SimpleNamespace(name = "z-image", base_repo = "org/base"),
        None,
        False,
        kind = "pipeline",
        repo_id = "org/base",
        transformer_resident_override_mib = 12000,
        companion_override_mib = 8000,
        text_encoder_override_mib = 6000,
    )
    assert plan.offload_policy == OFFLOAD_NONE


def _split_cache_roots(
    tmp_path,
    monkeypatch,
    *,
    register_root = False,
):
    """Unsloth's live cache root and a second one holding what a mid-session cache-folder change
    left behind, both empty. ``register_root`` makes the second dir huggingface_hub's import-time
    constant, the root ``cache_dir = None`` resolves to; without it the constant points at a third
    empty dir, so ``other`` is reachable only as an explicit staged snapshot. Either way the test
    never sees the developer's real cache."""
    from huggingface_hub import constants as hf_constants

    from core.inference import diffusion as dmod

    live = tmp_path / "live-hub"
    other = tmp_path / "other-hub"
    unused = tmp_path / "import-time-hub"
    for path in (live, other, unused):
        path.mkdir(exist_ok = True)
    monkeypatch.setattr(dmod, "hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(other if register_root else unused))
    return live, other


def _hub_blob(root, repo_id, name, mib):
    """A completed download's blob under ``root``. Named after the file's etag, so the same file
    carries the same name in every root it was ever downloaded into. Sparse: costs no disk."""
    blob = root / f"models--{repo_id.replace('/', '--')}" / "blobs" / name
    blob.parent.mkdir(parents = True, exist_ok = True)
    with open(blob, "wb") as fh:
        fh.truncate(mib * 1024 * 1024)
    return blob


def _hub_snapshot_file(root, repo_id, rev, rel, blob_name):
    """The pointer a finished download leaves at ``snapshots/<rev>/<rel>``: a symlink at the blob
    named after the file's etag, exactly as hf_hub_download creates it once the blob is complete."""
    repo = root / f"models--{repo_id.replace('/', '--')}"
    pointer = repo / "snapshots" / rev / rel
    pointer.parent.mkdir(parents = True, exist_ok = True)
    pointer.symlink_to(repo / "blobs" / blob_name)
    return pointer


def _hub_ref(root, repo_id, rev):
    """``refs/main`` -> the commit this root currently serves. hf_hub_download writes it as soon as
    it resolves the revision, i.e. BEFORE the first byte of that revision lands."""
    ref = root / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    ref.parent.mkdir(parents = True, exist_ok = True)
    ref.write_text(rev)
    return ref


def _sparse_snapshot_file(root, repo_id, rev, rel, mib):
    """One file of ``rev`` present in ``root``'s snapshot. Sparse, so the size costs no disk."""
    path = root / f"models--{repo_id.replace('/', '--')}" / "snapshots" / rev / rel
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as fh:
        fh.truncate(mib * 1024 * 1024)
    return path


def _safetensors_with_params(path, numel):
    """A safetensors shard whose JSON header declares ``numel`` elements. Only the header is read
    (_safetensors_param_count never touches tensor data), so no payload is written."""
    import json

    header = json.dumps({"w": {"dtype": "F32", "shape": [numel], "data_offsets": [0, 4]}}).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little"))
        fh.write(header)
    return path


def _other_root_base_snapshot(
    tmp_path,
    monkeypatch,
    *,
    register_root = False,
):
    """A base repo cached ONLY under the other cache root, with Unsloth's live root empty: what
    a mid-session cache-folder change leaves behind, handed back as ``_base_local_dir``. Sparse."""
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = register_root)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    for rel, mib in (
        ("text_encoder/model.safetensors", 150),
        ("vae/diffusion_pytorch_model.safetensors", 50),
        ("transformer/diffusion_pytorch_model.safetensors", 4096),
    ):
        path = snapshot / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, "wb") as fh:
            fh.truncate(mib * 1024 * 1024)
    return snapshot


def _small_card(monkeypatch):
    """A 5 GiB-free card + a fixed runtime headroom, so the plan's arithmetic is exact:
    budget = 5000 - max(2048, 5120*0.10) = 2952 MiB, resident margin 0.85 * 2952 = 2509 MiB."""
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import DeviceMemory

    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 5000, 5120),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 100)
    return types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)


def test_plan_memory_budgets_companions_from_the_other_root_snapshot(monkeypatch, tmp_path):
    # _companion_cache_bytes only looks under hub_cache_dir(), so the import-time root sizes as 0.
    from core.inference.diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE

    snapshot = _other_root_base_snapshot(tmp_path, monkeypatch)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    fam = types.SimpleNamespace(name = "flux.1")

    def _plan(**kw):
        return backend._plan_memory(
            target,
            None,
            "bfl/base",
            fam,
            None,
            False,
            kind = "gguf",
            transformer_resident_override_mib = 300,
            **kw,
        )

    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 0
    blind = _plan()
    assert blind.estimates["companion_dense_mib"] is None
    # 300 + 0 + 100 + 2048 = 2448 <= 2509: resident, and the 200 MiB of companions arrive unbudgeted.
    assert blind.offload_policy == OFFLOAD_NONE

    plan = _plan(base_local_dir = str(snapshot))
    assert plan.estimates["companion_dense_mib"] == 200
    # 300 + 200 + 100 + 2048 = 2648 > 2509, and the 2348 MiB group floor fits: stream the transformer.
    assert plan.offload_policy == OFFLOAD_GROUP


def test_plan_memory_sizes_a_pipeline_load_from_the_other_root_snapshot(monkeypatch, tmp_path):
    from core.inference.diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE

    snapshot = _other_root_base_snapshot(tmp_path, monkeypatch)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    fam = types.SimpleNamespace(name = "flux.1", base_repo = "unrelated/repo")

    def _plan(**kw):
        return backend._plan_memory(
            target, None, "bfl/base", fam, None, False, kind = "pipeline", repo_id = "bfl/base", **kw
        )

    blind = _plan()
    assert blind.estimates["model_dense_mib"] is None
    assert blind.offload_policy == OFFLOAD_NONE

    plan = _plan(base_local_dir = str(snapshot))
    assert plan.estimates["model_dense_mib"] == 4296
    # Group, not whole-module: the planner now gets the companion split, so the group floor fits.
    assert plan.estimates["companion_dense_mib"] == 200
    assert plan.offload_policy == OFFLOAD_GROUP


def test_plan_memory_keeps_companions_a_partial_staged_snapshot_omits(monkeypatch, tmp_path):
    from core.inference.diffusion_memory import OFFLOAD_GROUP

    snapshot = _other_root_base_snapshot(tmp_path, monkeypatch, register_root = True)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    manifest_only = snapshot.parent / ("c" * 40)
    manifest_only.mkdir(parents = True)
    (manifest_only / "model_index.json").write_bytes(b"{}")

    plan = backend._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        base_local_dir = str(manifest_only),
    )
    # The registered root still holds 150 + 50 MiB of companions, so 300 + 200 + 100 + 2048 = 2648
    # clears the 2509 MiB resident margin and the 2348 MiB group floor fits.
    assert plan.estimates["companion_dense_mib"] == 200
    assert plan.offload_policy == OFFLOAD_GROUP


def test_load_progress_counts_a_checkpoint_the_other_cache_root_already_holds(
    tmp_path, monkeypatch
):
    from core.inference.diffusion import _LoadingState

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    _hub_blob(other, "org/gguf", "a" * 64, 400)
    _hub_blob(other, "org/base", "b" * 64, 100)
    _hub_blob(live, "org/base", "b" * 64, 100)

    assert DiffusionBackend._cache_bytes("org/gguf") == 400 * 1024 * 1024
    assert DiffusionBackend._cache_bytes("org/base") == 100 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/gguf", base_repo = "org/base", expected_bytes = 500 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "finalizing"
    assert progress["fraction"] == 1.0


def test_load_progress_ignores_a_revision_the_moved_root_has_superseded(tmp_path, monkeypatch):
    # blobs/ is append-only: summing it counts superseded revisions too.
    from core.inference.diffusion import _LoadingState

    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    _hub_blob(other, "org/pipe", "a1", 2000)
    _hub_blob(other, "org/pipe", "a2", 2000)
    _hub_snapshot_file(other, "org/pipe", old_rev, "transformer/shard-1.safetensors", "a1")
    _hub_snapshot_file(other, "org/pipe", old_rev, "vae/diffusion_pytorch_model.safetensors", "a2")
    _hub_ref(other, "org/pipe", new_rev)
    _hub_blob(other, "org/pipe", "b1", 2000)
    _hub_snapshot_file(other, "org/pipe", new_rev, "transformer/shard-1.safetensors", "b1")
    _hub_blob(other, "org/pipe", "b2.incomplete", 200)

    assert DiffusionBackend._cache_bytes("org/pipe") == 2200 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/pipe", base_repo = None, expected_bytes = 4000 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"
    assert progress["fraction"] == 0.55


def test_load_progress_counts_one_logical_file_across_roots_at_two_revisions(tmp_path, monkeypatch):
    # Each root has its own refs/main, so one shard has different etags; key by logical path.
    from core.inference.diffusion import _LoadingState

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    _hub_blob(other, "org/pipe", "a1", 1000)
    _hub_snapshot_file(other, "org/pipe", old_rev, "transformer/shard-1.safetensors", "a1")
    _hub_ref(other, "org/pipe", old_rev)
    _hub_blob(live, "org/pipe", "b1", 1000)
    _hub_snapshot_file(live, "org/pipe", new_rev, "transformer/shard-1.safetensors", "b1")
    _hub_ref(live, "org/pipe", new_rev)

    assert DiffusionBackend._cache_bytes("org/pipe") == 1000 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/pipe", base_repo = None, expected_bytes = 2000 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"
    assert progress["fraction"] == 0.5


def test_companion_bytes_union_a_base_the_prefetch_split_across_roots(tmp_path, monkeypatch):
    # Roots can hold disjoint parts of one revision, so sizing off the larger one under-budgets.
    from core.inference.diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    rev = "a" * 40
    _sparse_snapshot_file(other, "bfl/base", rev, "text_encoder/model.safetensors", 200)
    _sparse_snapshot_file(live, "bfl/base", rev, "vae/diffusion_pytorch_model.safetensors", 150)

    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 350 * 1024 * 1024

    target = _small_card(monkeypatch)
    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 100,
    )
    assert plan.estimates["companion_dense_mib"] == 350
    # 100 + 350 + 100 + 2048 = 2598 > the 2509 MiB resident margin.
    assert plan.offload_policy != OFFLOAD_NONE
    assert plan.offload_policy == OFFLOAD_GROUP


def test_companion_bytes_skip_a_superseded_revision_in_the_same_root(tmp_path, monkeypatch):
    # Merged per FILE, so only refs/main's revision is read (repacked shards would double count).
    live, _other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _sparse_snapshot_file(live, "bfl/base", old_rev, f"text_encoder/{shard}", 100)
    _sparse_snapshot_file(live, "bfl/base", new_rev, "text_encoder/model.safetensors", 200)
    _hub_ref(live, "bfl/base", new_rev)

    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 200 * 1024 * 1024


def test_plan_memory_sizes_a_pipeline_split_across_both_cache_roots(monkeypatch, tmp_path):
    from core.inference.diffusion_memory import OFFLOAD_MODEL

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    fam = types.SimpleNamespace(name = "flux.1", base_repo = "unrelated/repo")
    _hub_blob(live, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "b" * 64, 4000)

    def _plan(**kw):
        return backend._plan_memory(
            target, None, "bfl/base", fam, None, False, kind = "pipeline", repo_id = "bfl/base", **kw
        )

    plan = _plan()
    # 300 + 4000, each counted once (per-root sum would be 4600, live root alone 300).
    assert plan.estimates["model_dense_mib"] == 4300
    assert plan.offload_policy == OFFLOAD_MODEL

    # The gated preflight can stage a manifest-only snapshot, so a staged dir is a floor only.
    manifest_only = other / "models--bfl--base" / "snapshots" / ("c" * 40)
    manifest_only.mkdir(parents = True)
    (manifest_only / "model_index.json").write_bytes(b"{}")
    assert _plan(base_local_dir = str(manifest_only)).estimates["model_dense_mib"] == 4300


def test_dense_transformer_bytes_read_the_other_root_and_treat_the_snapshot_as_a_floor(
    tmp_path, monkeypatch
):
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    _safetensors_with_params(
        snapshot / "transformer" / "diffusion_pytorch_model.safetensors", 1_000_000
    )

    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base") == 2_000_000
    assert (
        DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(snapshot)) == 2_000_000
    )
    bare = tmp_path / "companions-only-snapshot"
    bare.mkdir()
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(bare)) == 2_000_000
    # The cache folder can change mid-prefetch; the resolved snapshot is still where shards live.
    stale = tmp_path / "stale-root" / "models--bfl--base" / "snapshots" / ("b" * 40)
    _safetensors_with_params(
        stale / "transformer" / "diffusion_pytorch_model.safetensors", 3_000_000
    )
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(stale)) == 6_000_000


@pytest.mark.parametrize("staged", [False, True])
def test_dense_fit_check_runs_for_a_base_the_live_cache_root_does_not_hold(
    fake_runtime, tmp_path, monkeypatch, staged, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    # Pin the base id so the fixture's cache, not the ambient one, decides the mirror swap.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    root = tmp_path / "stale-root" if staged else other
    shards = root / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / ("a" * 40)
    _safetensors_with_params(
        shards / "transformer" / "diffusion_pytorch_model.safetensors",
        6 * 1024**3,
    )
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    dense_refit_ran = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        # Scoped to this backend: _plan_memory is patched on the CLASS and stray loads run on daemons.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(transformer_resident_override_mib)
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    monkeypatch.setattr(
        DiffusionBackend, "_load_dense_quant_pipeline", lambda self, *a, **k: (None, None)
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_m(
        backend, tmp_path, transformer_quant = "fp8", _base_local_dir = str(shards) if staged else None
    )
    assert dense_refit_ran == [12288]
    assert backend.status()["loaded"] is True


def test_the_dense_builder_reads_transformer_from_the_hub_id_not_the_staged_snapshot(
    fake_runtime, tmp_path, monkeypatch
):
    # Sizing reads the staged snapshot but the load does not: diffusers treats a local dir as
    # terminal, so a partial snapshot would hard-fail instead of re-downloading.
    import contextlib

    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda *a, **k: None)
    _no_cache(monkeypatch)
    snapshot = tmp_path / "other-hub" / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / "abc"
    (snapshot / "transformer").mkdir(parents = True)
    seen: list = []

    class _Transformer:
        @classmethod
        def from_pretrained(cls, source, **kw):
            seen.append(source)
            raise RuntimeError("the source is the subject; the build cannot complete here")

    backend = DiffusionBackend()
    target = _force_cuda_target(backend, monkeypatch)
    with contextlib.suppress(Exception):
        backend._load_dense_quant_pipeline(
            _Transformer,
            object(),
            "Tongyi-MAI/Z-Image-Turbo",
            "cuda",
            None,
            None,
            target,
            "fp8",
            None,
            fam = detect_family("unsloth/Z-Image-GGUF"),
            base_local_dir = str(snapshot),
        )
    assert seen == ["unsloth/Z-Image-Turbo"]


def test_reset_step_cache_helper_is_best_effort():
    # reset_stateful_hooks lives only on HookRegistry; CacheMixin exposes _reset_stateful_cache.
    calls = []
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: calls.append("real"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(
            _reset_stateful_cache = lambda: calls.append("real"),
            reset_stateful_hooks = lambda: calls.append("fallback"),
        )
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(reset_stateful_hooks = lambda: calls.append("fallback"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["fallback"]
    DiffusionBackend._reset_step_cache(types.SimpleNamespace())
    DiffusionBackend._reset_step_cache(types.SimpleNamespace(transformer = object()))


def test_generate_resets_step_cache_only_when_engaged(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    resets = []
    backend._state.pipe.transformer = types.SimpleNamespace(
        _reset_stateful_cache = lambda: resets.append(True)
    )
    backend.generate(prompt = "a sloth")
    assert resets == []
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    backend.generate(prompt = "a sloth")
    backend.generate(prompt = "another sloth")
    assert resets == [True, True]


def test_prefetch_returns_snapshot_dir_for_manifest(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: f"/cache/snap/{fn}",
    )
    root = backend._prefetch_files(
        "base/repo", None, "base/repo", ["model_index.json", "vae/x.safetensors"], None
    )
    # str(Path(...).parent) uses the platform separator, so the bare literal failed on Windows.
    assert root == str(Path("/cache/snap"))
    assert (
        backend._prefetch_files("base/repo", None, "base/repo", ["vae/x.safetensors"], None) is None
    )


def test_pipeline_load_uses_predownloaded_dir(fake_runtime, tmp_path):
    # With a prefetched snapshot, from_pretrained must get the local dir: its own hub sweep would re-download the root singles (24 GB per FLUX.1).
    backend = DiffusionBackend()
    backend.load_pipeline(
        "unsloth/Qwen-Image-2512-bnb-4bit",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert _FakePipeline.last["base"] == str(tmp_path)
    backend.unload()


def test_unload_mid_render_releases_the_pipeline(fake_runtime, monkeypatch):
    import threading
    import weakref

    backend = DiffusionBackend()
    # Skip the real hardware probe (nvidia-smi can take >10 s on a busy host) so step 0 arrives promptly.
    monkeypatch.setattr(
        backend, "_pick_device_and_dtype", lambda: ("cpu", sys.modules["torch"].float32)
    )
    at_step0 = threading.Event()
    resume = threading.Event()

    class _SteppingPipe:
        def __init__(self) -> None:
            self._interrupt = False

        def __call__(
            self,
            *,
            callback_on_step_end = None,
            num_inference_steps = 8,
            **kwargs,
        ):
            for i in range(num_inference_steps):
                if self._interrupt:
                    break
                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, 0.0, {})
                if i == 0:
                    at_step0.set()
                    resume.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    pipe = _SteppingPipe()
    pipe_ref = weakref.ref(pipe)
    backend._state = _LoadState(
        pipe = pipe,
        family = detect_family("unsloth/Z-Image-GGUF"),
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )
    del pipe
    cleared_with_pipe_gone = []
    monkeypatch.setattr(
        "core.inference.diffusion.clear_gpu_cache",
        lambda: cleared_with_pipe_gone.append(pipe_ref() is None),
    )

    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 8)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target = _run)
    t.start()
    assert at_step0.wait(5)
    u = threading.Thread(target = backend.unload)
    u.start()
    assert backend._active_generate_cancel.wait(5)
    resume.set()
    t.join(5)
    u.join(5)
    assert "cancelled" in str(out["exc"]).lower()
    assert pipe_ref() is None, "the cancelled render's traceback still pins the pipeline"
    assert True in cleared_with_pipe_gone


@pytest.mark.parametrize("transition_ends_first", [False, True])
def test_replacing_load_mid_render_releases_the_pipeline(
    fake_runtime, monkeypatch, transition_ends_first
):
    import threading
    import weakref

    from core.inference import diffusion as diffusion_mod

    backend = DiffusionBackend()
    # Skip the real hardware probe (nvidia-smi can take >10 s on a busy host) so step 0 arrives promptly.
    monkeypatch.setattr(
        backend, "_pick_device_and_dtype", lambda: ("cpu", sys.modules["torch"].float32)
    )
    at_step0 = threading.Event()
    resume = threading.Event()
    teardown_cleared = threading.Event()
    render_done = threading.Event()

    class _SteppingPipe:
        def __init__(self) -> None:
            self._interrupt = False

        def __call__(
            self,
            *,
            callback_on_step_end = None,
            num_inference_steps = 8,
            **kwargs,
        ):
            for i in range(num_inference_steps):
                if self._interrupt:
                    break
                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, 0.0, {})
                if i == 0:
                    at_step0.set()
                    resume.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    pipe = _SteppingPipe()
    pipe_ref = weakref.ref(pipe)
    backend._state = _LoadState(
        pipe = pipe,
        family = detect_family("unsloth/Z-Image-GGUF"),
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )
    del pipe
    cleared_with_pipe_gone = []

    def _clear():
        cleared_with_pipe_gone.append(pipe_ref() is None)
        teardown_cleared.set()

    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", _clear)
    real_clear_frames = diffusion_mod._clear_exception_frames

    transition_done = threading.Event()

    def _late_clear_frames(exc):
        (transition_done if transition_ends_first else teardown_cleared).wait(5)
        real_clear_frames(exc)

    monkeypatch.setattr(diffusion_mod, "_clear_exception_frames", _late_clear_frames)
    backend._load_token += 1
    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 8)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc
        render_done.set()

    def _replacing_load():
        with backend._lock:
            with backend._generation_cancel_lock:
                backend._active_generate_cancel.set()
            backend._reserve_teardown_locked()
        with backend._model_transition_slot():
            with backend._lock:
                try:
                    backend._unload_locked()
                finally:
                    backend._release_teardown_locked()
            if not transition_ends_first:
                render_done.wait(5)
        transition_done.set()

    t = threading.Thread(target = _run)
    t.start()
    assert at_step0.wait(5)
    ld = threading.Thread(target = _replacing_load)
    ld.start()
    assert backend._active_generate_cancel.wait(5)
    resume.set()
    t.join(5)
    ld.join(5)
    assert "cancelled" in str(out["exc"]).lower()
    assert pipe_ref() is None
    assert cleared_with_pipe_gone[0] is False, "the forced order did not happen"
    assert True in cleared_with_pipe_gone


def test_unload_waits_for_in_flight_denoise_before_teardown():
    import threading

    backend = DiffusionBackend()

    denoise_active = {"v": False}
    teardown_saw = []

    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    started = threading.Event()
    finish = threading.Event()

    def _denoise():
        with backend._generate_lock:
            denoise_active["v"] = True
            started.set()
            cancel.wait(2.0)
            finish.wait(2.0)
            denoise_active["v"] = False

    def _fake_unload_locked():
        teardown_saw.append(denoise_active["v"])

    backend._unload_locked = _fake_unload_locked

    d = threading.Thread(target = _denoise)
    d.start()
    assert started.wait(2.0)

    unloaded = threading.Event()

    def _unload():
        backend.unload()
        unloaded.set()

    u = threading.Thread(target = _unload)
    u.start()
    assert cancel.wait(2.0)
    assert teardown_saw == []
    assert not unloaded.wait(0.3)

    finish.set()
    d.join(2.0)
    u.join(2.0)
    assert unloaded.is_set()
    assert teardown_saw == [False]


def _load_zimage_backend(tmp_path):
    backend = _loaded_backend(tmp_path)
    return backend


def test_generate_seed_list_uses_one_generator_per_image(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "a sloth", seeds = [11, 22, 33, 44])
    assert len(out["images"]) == 4
    assert out["seeds"] == [11, 22, 33, 44]
    assert out["seed"] == 11
    call = backend._state.pipe.last_kwargs
    assert call["prompt"] == "a sloth"
    assert call["num_images_per_prompt"] == 4
    assert [g.manual for g in call["generator"]] == [11, 22, 33, 44]


def test_generate_prompt_list_one_image_per_prompt(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "fallback", prompts = ["a", "b", "c"], seed = 100)
    assert len(out["images"]) == 3
    assert out["seeds"] == [100, 101, 102]
    call = backend._state.pipe.last_kwargs
    assert call["prompt"] == ["a", "b", "c"]
    assert call["num_images_per_prompt"] == 1
    assert [g.manual for g in call["generator"]] == [100, 101, 102]


def test_generate_prompt_list_with_matching_seed_list(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "fallback", prompts = ["a", "b"], seeds = [5, 6])
    assert out["seeds"] == [5, 6]
    with pytest.raises(ValueError, match = "same length"):
        backend.generate(prompt = "fallback", prompts = ["a", "b"], seeds = [5])


def test_generate_single_image_keeps_scalar_generator(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "one", seed = 5)
    call = backend._state.pipe.last_kwargs
    assert not isinstance(call["generator"], list)
    assert call["generator"].manual == 5
    assert call["num_images_per_prompt"] == 1
    assert out["seeds"] == [5]


def test_generate_batched_seed_matches_solo_replay(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    backend.generate(prompt = "p", seeds = [3, 9])
    batched = [g.manual for g in backend._state.pipe.last_kwargs["generator"]]
    backend.generate(prompt = "p", seed = 9)
    solo = backend._state.pipe.last_kwargs["generator"].manual
    assert batched[1] == solo == 9


def test_generate_prompt_list_rejected_off_txt2img(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    with pytest.raises(ValueError, match = "text-to-image only"):
        backend.generate(prompt = "x", prompts = ["a", "b"], init_image = _tiny_png_b64())


class _CountingPipe(_FakePipe):
    """Records each forward's image count; optionally OOMs above ``max_images``."""

    def __init__(self, max_images = None):
        super().__init__()
        self.batch_attempts = []
        self.max_images = max_images

    def __call__(
        self,
        *,
        prompt = None,
        **kwargs,
    ):
        n = kwargs.get("num_images_per_prompt", 1)
        if isinstance(prompt, list):
            n *= len(prompt)
        self.batch_attempts.append(n)
        if self.max_images is not None and n > self.max_images:
            raise _FakeOutOfMemoryError("CUDA out of memory. Tried to allocate everything")
        return super().__call__(prompt = prompt, **kwargs)


# Structural stand-in for torch.cuda.OutOfMemoryError (matched by class name).
_FakeOutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})


def test_generate_explicit_batch_size_caps_per_forward(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    pipe = _CountingPipe()
    object.__setattr__(backend._state, "pipe", pipe)
    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4], batch_size = 2)
    assert pipe.batch_attempts == [2, 2]
    assert len(out["images"]) == 4
    assert out["seeds"] == [1, 2, 3, 4]


def test_generate_oom_backoff_halves_the_batch(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    pipe = _CountingPipe(max_images = 2)
    object.__setattr__(backend._state, "pipe", pipe)
    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert pipe.batch_attempts == [4, 2, 2]
    assert len(out["images"]) == 4
    assert out["seeds"] == [1, 2, 3, 4]


def test_generate_oom_backoff_drops_the_batch_shaped_graphs(fake_runtime, tmp_path):
    """A captured entry is batch-shaped and empty_cache() cannot reclaim it, so the retry OOMs too."""
    backend = _load_zimage_backend(tmp_path)
    pipe = _CountingPipe(max_images = 2)
    object.__setattr__(backend._state, "pipe", pipe)

    class _Handle:
        def __init__(self):
            self.reset_after = []

        def reset(self):
            self.reset_after.append(len(pipe.batch_attempts))
            return self

        def set_bypass(self, on):
            return self

    handle = _Handle()
    object.__setattr__(backend._state, "cuda_graphs", (handle,))

    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4])

    assert pipe.batch_attempts == [4, 2, 2]
    assert len(out["images"]) == 4 and out["seeds"] == [1, 2, 3, 4]
    assert handle.reset_after == [1]


class _BoomPipe(_CountingPipe):
    """Fails every forward with a NON-OOM error (must not trigger backoff)."""

    def __call__(
        self,
        *,
        prompt = None,
        **kwargs,
    ):
        self.batch_attempts.append(kwargs.get("num_images_per_prompt", 1))
        raise RuntimeError("shape mismatch")


def test_generate_single_image_oom_drops_the_graphs_before_raising(fake_runtime, tmp_path):
    """A one-image OOM raises instead of splitting, so the graphs must be dropped on the way out."""
    backend = _load_zimage_backend(tmp_path)
    pipe = _CountingPipe(max_images = 0)
    object.__setattr__(backend._state, "pipe", pipe)

    class _Handle:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1
            return self

        def set_bypass(self, on):
            return self

    handle = _Handle()
    object.__setattr__(backend._state, "cuda_graphs", (handle,))

    with pytest.raises(RuntimeError, match = "out of memory"):
        backend.generate(prompt = "p", seed = 1)

    assert pipe.batch_attempts == [1]
    assert handle.resets == 1, "the graphs stayed pinned across the raise"


def test_generate_non_oom_error_leaves_the_graphs_alone(fake_runtime, tmp_path):
    """Only an OOM justifies throwing away working graphs; a shape mismatch does not."""
    backend = _load_zimage_backend(tmp_path)
    pipe = _BoomPipe()
    object.__setattr__(backend._state, "pipe", pipe)

    class _Handle:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1
            return self

        def set_bypass(self, on):
            return self

    handle = _Handle()
    object.__setattr__(backend._state, "cuda_graphs", (handle,))

    with pytest.raises(RuntimeError, match = "shape mismatch"):
        backend.generate(prompt = "p", seeds = [1, 2])
    assert handle.resets == 0


def test_generate_non_oom_error_is_not_retried(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    pipe = _BoomPipe()
    object.__setattr__(backend._state, "pipe", pipe)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert pipe.batch_attempts == [4]


def test_generate_reclaims_model_offload_memory_once_after_success(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = _load_zimage_backend(tmp_path)
    trace = []

    def fake_reclaim(policy, logger = None):
        if policy == "model":
            trace.append("reclaim")
            return True
        return False

    monkeypatch.setattr(dmod, "reclaim_offload_host_memory", fake_reclaim, raising = False)
    monkeypatch.setattr(dmod.compile_cache, "register_shape", lambda *a, **k: None)
    monkeypatch.setattr(dmod.compile_cache, "save_async", lambda *a, **k: trace.append("save"))

    object.__setattr__(backend._state, "offload_policy", "model")
    object.__setattr__(backend._state, "pipe", _CountingPipe(max_images = 2))
    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert len(out["images"]) == 4
    assert trace == ["save", "reclaim"]

    for policy in ("none", "group", "streaming", "sequential"):
        object.__setattr__(backend._state, "offload_policy", policy)
        object.__setattr__(backend._state, "pipe", _FakePipe())
        backend.generate(prompt = policy)
    assert trace.count("reclaim") == 1

    object.__setattr__(backend._state, "offload_policy", "model")
    object.__setattr__(backend._state, "pipe", _BoomPipe())
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        backend.generate(prompt = "failed")

    class _CancellingPipe(_FakePipe):
        def __call__(
            self,
            *,
            prompt = None,
            **kwargs,
        ):
            assert backend.cancel_generate() is True
            return super().__call__(prompt = prompt, **kwargs)

    object.__setattr__(backend._state, "pipe", _CancellingPipe())
    with pytest.raises(RuntimeError, match = DIFFUSION_CANCELLED_MSG):
        backend.generate(prompt = "cancelled")
    assert trace.count("reclaim") == 1


def test_generate_broadcasts_negative_prompt_across_a_mixed_prompt_batch(fake_runtime, tmp_path):
    # encode_prompt requires matching lengths to avoid batch-1 embeds with batch-N latents.
    backend = _load_zimage_backend(tmp_path)
    backend.generate(
        prompt = "fallback",
        prompts = ["a", "b", "c"],
        negative_prompt = "blurry",
        guidance = 0.5,
    )
    call = backend._state.pipe.last_kwargs
    assert call["prompt"] == ["a", "b", "c"]
    assert call["negative_prompt"] == ["blurry", "blurry", "blurry"]
    # empty negatives remain omitted instead of expanding to [""] * n.
    backend.generate(prompt = "fallback", prompts = ["a", "b"])
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None


def test_generate_keeps_a_scalar_negative_prompt_off_the_list_paths(fake_runtime, tmp_path):
    # scalar prompts require scalar negatives on uniform-prompt and single-image paths.
    backend = _load_zimage_backend(tmp_path)
    backend.generate(prompt = "a sloth", seeds = [1, 2, 3], negative_prompt = "blurry", guidance = 0.5)
    assert backend._state.pipe.last_kwargs["prompt"] == "a sloth"
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"
    backend.generate(prompt = "a sloth", seed = 1, negative_prompt = "blurry", guidance = 0.5)
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"


class _TracingPipe(_CountingPipe):
    def __init__(
        self,
        trace,
        max_images = None,
    ):
        super().__init__(max_images = max_images)
        self.trace = trace

    def __call__(
        self,
        *,
        prompt = None,
        **kwargs,
    ):
        n = kwargs.get("num_images_per_prompt", 1)
        if isinstance(prompt, list):
            n *= len(prompt)
        self.trace.append(("call", n))
        return super().__call__(prompt = prompt, **kwargs)


def test_generate_resets_the_step_cache_before_an_oom_retry(fake_runtime, tmp_path):
    # A raising forward skips maybe_free_model_hooks(), leaving a stale FBCache residual.
    backend = _load_zimage_backend(tmp_path)
    trace: list = []
    pipe = _TracingPipe(trace, max_images = 2)
    pipe.transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: trace.append(("reset",)))
    object.__setattr__(backend._state, "pipe", pipe)
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert len(out["images"]) == 4 and out["seeds"] == [1, 2, 3, 4]
    assert trace == [
        ("reset",),
        ("call", 4),
        ("reset",),
        ("call", 2),
        ("reset",),
        ("call", 2),
    ]


def test_generate_resets_the_step_cache_before_every_chunk(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    trace: list = []
    pipe = _TracingPipe(trace)
    pipe.transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: trace.append(("reset",)))
    object.__setattr__(backend._state, "pipe", pipe)
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    backend.generate(prompt = "p", seeds = [1, 2, 3], batch_size = 2)
    assert trace == [("reset",), ("call", 2), ("reset",), ("call", 1)]


class _FakeSibling:
    def __init__(self, rfilename, size):
        self.rfilename = rfilename
        self.size = size


class _FakeInfo:
    def __init__(
        self,
        siblings,
        sha = None,
    ):
        self.siblings = siblings
        self.sha = sha


GB = 1024**3
_FLUX_BASE_SIBLINGS = [
    _FakeSibling("model_index.json", 1000),
    _FakeSibling("flux1-dev.safetensors", 24 * GB),
    _FakeSibling("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 8 * GB),
    _FakeSibling("text_encoder/model.safetensors", 2 * GB),
    _FakeSibling("text_encoder/model.fp16.safetensors", 1 * GB),
    _FakeSibling("vae/diffusion_pytorch_model.safetensors", 300),
    _FakeSibling("assets/gallery.pdf", 5000),
    _FakeSibling("README.md", 200),
]
_FLUX_BASE_SIBLINGS_BY_NAME = {s.rfilename: s.size for s in _FLUX_BASE_SIBLINGS}


def _fake_hf_api(
    monkeypatch,
    repos,
    shas = None,
):
    """Point HfApi.model_info at a canned sibling list per repo id."""

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo(repos[repo_id], (shas or {}).get(repo_id))

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    # Never let a developer's real Unsloth cache leak into these hermetic plan tests.
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: False),
    )


def _no_dense_prefetch(monkeypatch):
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )


def _fake_flux_hub(monkeypatch):
    """The FLUX.1-dev GGUF + base-repo pair the download-plan tests resolve against."""
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": [_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)],
            "black-forest-labs/FLUX.1-dev": _FLUX_BASE_SIBLINGS,
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo",
        lambda *a, **k: "black-forest-labs/FLUX.1-dev",
    )
    _no_dense_prefetch(monkeypatch)


def _flux_download_plan(**kwargs):
    return DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf", **kwargs
    )


def test_download_plan_scopes_the_base_repo_files(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)

    plan = _flux_download_plan()

    # The base entry names the MIRROR: a gated id would 401 an anonymous user at staging.
    assert [e["repo_id"] for e in plan["entries"]] == [
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-dev",
    ]
    checkpoint, base = plan["entries"]
    assert checkpoint["files"] == ["flux1-dev-Q4_K_M.gguf"]
    assert checkpoint["bytes"] == 7 * GB
    assert checkpoint["checkpoint"] is True
    assert base["checkpoint"] is False
    assert "flux1-dev.safetensors" not in base["files"]
    assert not any(f.startswith("transformer/") for f in base["files"])
    assert not any(f.startswith("assets/") for f in base["files"])
    assert "model_index.json" in base["files"]
    assert "text_encoder/model.safetensors" in base["files"]
    assert base["bytes"] < 24 * GB
    assert plan["total_bytes"] == checkpoint["bytes"] + base["bytes"]
    assert plan["required_bytes"] == plan["total_bytes"]
    assert plan["checkpoint_bytes"] == 7 * GB


def test_download_plan_omits_a_cached_gguf_but_keeps_missing_companions(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename, revision = None, expected_size = None, **kwargs: repo_id
            == "unsloth/FLUX.1-dev-GGUF"
            and filename == "flux1-dev-Q4_K_M.gguf"
        ),
    )

    plan = _flux_download_plan()

    assert [entry["repo_id"] for entry in plan["entries"]] == ["unsloth/FLUX.1-dev"]
    assert "text_encoder/model.safetensors" in plan["entries"][0]["files"]
    assert plan["total_bytes"] == plan["entries"][0]["bytes"]
    assert plan["required_bytes"] == 7 * GB + plan["total_bytes"]
    assert plan["checkpoint_bytes"] == 7 * GB


def test_download_plan_stages_but_does_not_count_a_file_an_older_snapshot_holds(monkeypatch):
    """README-only commit on a no-symlink cache: the GGUF is still staged but counts no bytes."""
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    asked = []

    def reusable(repo_id, names, revision, declared_sizes, hf_token):
        asked.append((repo_id, tuple(names)))
        return {"flux1-dev-Q4_K_M.gguf"} if repo_id == "unsloth/FLUX.1-dev-GGUF" else set()

    monkeypatch.setattr(DiffusionBackend, "_reusable_from_older_snapshot", staticmethod(reusable))

    plan = _flux_download_plan()

    checkpoint, base = plan["entries"]
    assert checkpoint["repo_id"] == "unsloth/FLUX.1-dev-GGUF"
    assert checkpoint["files"] == ["flux1-dev-Q4_K_M.gguf"]
    assert checkpoint["bytes"] == 0
    assert checkpoint["checkpoint"] is True
    assert base["bytes"] > 0
    assert plan["total_bytes"] == base["bytes"]
    assert plan["checkpoint_bytes"] == 7 * GB
    assert ("unsloth/FLUX.1-dev-GGUF", ("flux1-dev-Q4_K_M.gguf",)) in asked


def test_reusable_from_older_snapshot_reads_the_live_cache_without_hashing(monkeypatch, tmp_path):
    from hub.utils import snapshot_reuse

    old, new = "1" * 40, "2" * 40
    same, changed = b"s" * 4096, b"c" * 4096
    snap = tmp_path / "models--unsloth--Qwen-Image-2.1-FP8" / "snapshots" / old
    (snap / "vae").mkdir(parents = True)
    (snap / "text_encoder.safetensors").write_bytes(same)
    (snap / "vae" / "vae.safetensors").write_bytes(changed)
    digests = {
        old: {"text_encoder.safetensors": "a" * 64, "vae/vae.safetensors": "b" * 64},
        new: {"text_encoder.safetensors": "a" * 64, "vae/vae.safetensors": "c" * 64},
    }
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        snapshot_reuse,
        "hub_remote_digests",
        lambda repo_type, repo_id, token: lambda commit, paths: {
            p: digests[commit][p] for p in paths
        },
    )
    monkeypatch.setattr(snapshot_reuse, "file_digest", lambda *a, **k: pytest.fail("plan hashed"))

    found = DiffusionBackend._reusable_from_older_snapshot(
        "unsloth/Qwen-Image-2.1-FP8",
        ["text_encoder.safetensors", "vae/vae.safetensors"],
        new,
        {"text_encoder.safetensors": len(same), "vae/vae.safetensors": len(changed)},
        None,
    )

    assert found == {"text_encoder.safetensors"}
    assert not (snap.parent / new).exists()
    assert (
        DiffusionBackend._reusable_from_older_snapshot(
            "unsloth/Qwen-Image-2.1-FP8", ["text_encoder.safetensors"], None, {}, None
        )
        == set()
    )


def test_reusable_from_older_snapshot_targets_the_main_ref_when_unpinned(monkeypatch, tmp_path):
    from hub.utils import snapshot_reuse

    old, new = "1" * 40, "2" * 40
    encoder = b"s" * 4096
    repo = tmp_path / "models--unsloth--Qwen-Image-2.1-FP8"
    (repo / "snapshots" / old).mkdir(parents = True)
    (repo / "snapshots" / new / "vae").mkdir(parents = True)
    (repo / "snapshots" / old / "te.safetensors").write_bytes(encoder)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(new)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    head = {"main": "a" * 64}
    asked = []

    def digests(repo_type, repo_id, token):
        def lookup(commit, paths):
            asked.append(commit)
            return {p: head.get(commit, "a" * 64) for p in paths}

        return lookup

    monkeypatch.setattr(snapshot_reuse, "hub_remote_digests", digests)

    def reusable():
        return DiffusionBackend._reusable_from_older_snapshot(
            "unsloth/Qwen-Image-2.1-FP8",
            ["te.safetensors"],
            None,
            {"te.safetensors": len(encoder)},
            None,
        )

    assert reusable() == {"te.safetensors"}
    assert asked[0] == "main"
    head["main"] = "b" * 64
    assert reusable() == set()


def test_reusable_from_older_snapshot_keeps_the_anonymous_token_sentinel(monkeypatch, tmp_path):
    from hub.utils import snapshot_reuse

    tokens = []
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        snapshot_reuse,
        "hub_remote_digests",
        lambda repo_type, repo_id, token: tokens.append(token) or (lambda commit, paths: {}),
    )

    for token in (False, "", None, "hf_x"):
        DiffusionBackend._reusable_from_older_snapshot(
            "unsloth/Qwen-Image-2.1-FP8", ["te.safetensors"], "2" * 40, {"te.safetensors": 1}, token
        )

    assert tokens == [False, None, None, "hf_x"]


def test_download_plan_is_empty_when_every_required_file_is_cached(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _all_cached(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: True),
    )

    plan = _flux_download_plan()

    assert plan["entries"] == []
    assert plan["total_bytes"] == 0
    assert plan["required_bytes"] > 7 * GB
    assert plan["checkpoint_bytes"] == 7 * GB


def test_download_plan_sizes_the_checkpoint_when_the_base_is_the_same_repo(monkeypatch):
    # A combined repo keys both Hub lookups on one id: merge, do not overwrite or list twice.
    combined = "unsloth/Combined-Image-GGUF"
    _fake_hf_api(
        monkeypatch,
        {
            combined: [
                _FakeSibling("model-Q4_K_M.gguf", 7 * GB),
                _FakeSibling("model_index.json", 1000),
                _FakeSibling("text_encoder/model.safetensors", 2 * GB),
                _FakeSibling("vae/diffusion_pytorch_model.safetensors", 300),
            ],
        },
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: combined)
    _no_dense_prefetch(monkeypatch)
    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(
        combined, gguf_filename = "model-Q4_K_M.gguf", base_repo = combined
    )

    assert len(plan["entries"]) == 1, "one repo, one scoped job"
    entry = plan["entries"][0]
    assert entry["gguf_filename"] == "model-Q4_K_M.gguf"
    assert entry["files"].count("model-Q4_K_M.gguf") == 1
    assert plan["checkpoint_bytes"] == 7 * GB
    assert entry["bytes"] == plan["required_bytes"] == 7 * GB + 2 * GB + 1300

    assert entry["checkpoint"] is True

    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(
            lambda _repo, files, _revision = None, _declared_sizes = None: (
                set(files) if files == ["model-Q4_K_M.gguf"] else set()
            )
        ),
    )
    warming = DiffusionBackend().download_plan(
        combined, gguf_filename = "model-Q4_K_M.gguf", base_repo = combined
    )
    assert warming["entries"][0]["files"] == entry["files"]

    assert warming["entries"][0]["checkpoint"] is False


def _write_hub_cache(
    root,
    repo_id,
    filename,
    sha,
    size,
    *,
    symlink = True,
    set_main = True,
):
    """The tree hf_hub_download leaves behind: blobs/ + snapshots/<sha>/ + refs/main."""
    import os

    repo_dir = (root / f"models--{repo_id.replace('/', '--')}").resolve()
    blobs, snaps, refs = repo_dir / "blobs", repo_dir / "snapshots" / sha, repo_dir / "refs"
    for d in (blobs, (snaps / filename).parent, refs):
        d.mkdir(parents = True, exist_ok = True)
    blob = blobs / f"etag-{sha}"
    blob.write_bytes(b"\0" * size)
    target = snaps / filename
    if symlink:
        os.symlink(os.path.relpath(blob, target.parent), target)
    else:
        import shutil
        shutil.copyfile(blob, target)
    if set_main:
        (refs / "main").write_text(sha)


@pytest.mark.parametrize("symlink", [True, False], ids = ["posix_links", "windows_copies"])
def test_a_cached_file_survives_an_unrelated_commit_to_its_repo(tmp_path, monkeypatch, symlink):
    # model_info().sha is the REPO head (moves on a README commit), so the declared size wins.
    repo, name = "black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"
    _write_hub_cache(tmp_path, repo, name, "a" * 40, 4096, symlink = symlink)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "unused"))

    assert DiffusionBackend._hub_file_is_cached(repo, name, "a" * 40, 4096)
    assert DiffusionBackend._hub_file_is_cached(
        repo, name, "b" * 40, 4096
    ), "an unrelated repo commit must not invalidate a file the plan sized identically"
    assert not DiffusionBackend._hub_file_is_cached(
        repo, name, "b" * 40, 9999
    ), "a republished file has a different declared size and must be fetched through the manager"
    assert DiffusionBackend._hub_file_is_cached(repo, name, "b" * 40, 0)


@pytest.mark.parametrize("symlink", [True, False], ids = ["posix_links", "windows_copies"])
def test_an_explicit_current_snapshot_does_not_hide_a_stale_main_ref(
    tmp_path, monkeypatch, symlink
):
    # The loader follows refs/main (A) while the planner sizes B, so a same-size B hit is not cached.
    repo, name = "black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"
    stale, current = "a" * 40, "b" * 40
    _write_hub_cache(tmp_path, repo, name, stale, 4096, symlink = symlink)
    _write_hub_cache(
        tmp_path,
        repo,
        name,
        current,
        4096,
        symlink = symlink,
        set_main = False,
    )
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "unused"))

    assert not DiffusionBackend._hub_file_is_cached(repo, name, current, 4096)

    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    (repo_dir / "refs" / "main").write_text(current)
    assert DiffusionBackend._hub_file_is_cached(repo, name, current, 4096)


@pytest.mark.parametrize("symlink", [True, False], ids = ["posix_links", "windows_copies"])
def test_a_damaged_file_is_restaged_even_under_the_pinned_revision(tmp_path, monkeypatch, symlink):
    # The right commit is not proof of the right bytes: corroborate with the declared size.
    repo, name = "black-forest-labs/FLUX.1-dev", "text_encoder/model.safetensors"
    _write_hub_cache(tmp_path, repo, name, "a" * 40, 1024, symlink = symlink)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "unused"))

    assert not DiffusionBackend._hub_file_is_cached(
        repo, name, "a" * 40, 4096
    ), "the pinned commit is right but the bytes are not, so it must be restaged"
    assert DiffusionBackend._hub_file_is_cached(repo, name, "a" * 40, 1024)
    assert DiffusionBackend._hub_file_is_cached(repo, name, "a" * 40, 0)


def test_download_plan_probes_the_cache_at_the_revision_it_sized(monkeypatch):
    # An unpinned probe answers from the LOCAL main ref, so a republished companion reads as
    # present and the loader fetches it inline, outside the download manager.
    seen = []
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": [_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)],
            "black-forest-labs/FLUX.1-dev": _FLUX_BASE_SIBLINGS,
        },
        shas = {"unsloth/FLUX.1-dev-GGUF": "abc123", "black-forest-labs/FLUX.1-dev": "def456"},
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo",
        lambda *a, **k: "black-forest-labs/FLUX.1-dev",
    )
    _no_dense_prefetch(monkeypatch)
    # A mirror swap drops the pin: the vendor's commit means nothing in the mirror repo.
    _all_cached(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename, revision = None, expected_size = None, **kwargs: bool(
                seen.append(revision)
            )
        ),
    )

    DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert set(seen) == {"abc123", "def456"}


def test_download_plan_decides_the_widening_from_the_base_listing(monkeypatch):
    # The plan's gate is DEFERRED like _run_load's: eagerly it runs before the base listing exists.
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": [_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)],
            "black-forest-labs/FLUX.1-dev": _FLUX_BASE_SIBLINGS,
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo",
        lambda *a, **k: "black-forest-labs/FLUX.1-dev",
    )
    _no_cache(monkeypatch)

    seen: list[tuple] = []

    def _gate(
        self,
        fam,
        kwargs,
        *,
        companion_files = None,
        transformer_files = None,
    ):
        seen.append((tuple(companion_files or ()), tuple(transformer_files or ())))
        return bool(transformer_files)

    monkeypatch.setattr(DiffusionBackend, "_dense_quant_prefetch_needed", _gate)

    plan = _flux_download_plan()

    assert seen, "the deferred gate was never called with the base listing"
    companions, transformer_files = seen[-1]
    assert transformer_files == ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors",)
    assert "text_encoder/model.safetensors" in companions
    assert not any(f.startswith("transformer/") for f in companions)
    base = next(e for e in plan["entries"] if not e["repo_id"].endswith("-GGUF"))
    assert "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" in base["files"]


def test_download_plan_pipeline_kind_is_one_entry(monkeypatch):
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _FLUX_BASE_SIBLINGS})

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert len(plan["entries"]) == 1
    files = plan["entries"][0]["files"]
    assert any(f.startswith("transformer/") for f in files)
    assert "flux1-dev.safetensors" not in files
    assert "text_encoder/model.fp16.safetensors" not in files


def test_download_plan_flags_a_mirrored_pipeline_as_the_checkpoint(monkeypatch):
    # Gated pipelines stage from the mirror, so only the planner can flag the selected model.
    gated = "black-forest-labs/FLUX.1-dev"
    mirror = "unsloth/FLUX.1-dev"
    _fake_hf_api(monkeypatch, {gated: _FLUX_BASE_SIBLINGS, mirror: _FLUX_BASE_SIBLINGS})
    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(gated, model_kind = "pipeline")

    assert len(plan["entries"]) == 1
    entry = plan["entries"][0]
    assert entry["repo_id"] == mirror != gated
    assert entry["checkpoint"] is True


def test_download_plan_is_empty_for_a_local_path(tmp_path, monkeypatch):
    local = tmp_path / "my-model"
    (local / "transformer").mkdir(parents = True)
    (local / "model_index.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: str(local))
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )

    plan = DiffusionBackend().download_plan(str(local), gguf_filename = "weights.gguf")
    assert plan["entries"] == []
    assert plan["required_bytes"] == 0
    assert plan["checkpoint_bytes"] == 0


def test_download_plan_stages_the_precast_encoder_instead_of_the_dense_one(monkeypatch):
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": [_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)],
            "black-forest-labs/FLUX.1-dev": _FLUX_BASE_SIBLINGS,
            "unsloth/FLUX.1-schnell-FP8": [
                _FakeSibling("text_encoder_2-fp8.pt", 1 * GB),
                _FakeSibling("README.md", 100),
            ],
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo",
        lambda *a, **k: "black-forest-labs/FLUX.1-dev",
    )
    _no_dense_prefetch(monkeypatch)
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_sources",
        lambda fam, *, te_quant_mode, target, **_kwargs: (
            {
                "text_encoder_2": types.SimpleNamespace(
                    kind = "repo",
                    location = "unsloth/FLUX.1-schnell-FP8",
                    filename = "text_encoder_2-fp8.pt",
                )
            }
            if te_quant_mode == "fp8"
            else {}
        ),
    )

    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        text_encoder_quant = "fp8",
    )
    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert "unsloth/FLUX.1-schnell-FP8" in by_repo
    assert by_repo["unsloth/FLUX.1-schnell-FP8"]["files"] == ["text_encoder_2-fp8.pt"]
    base = by_repo["unsloth/FLUX.1-dev"]
    assert not any(
        f.startswith("text_encoder_2/") and f.endswith(".safetensors") for f in base["files"]
    )
    assert "text_encoder/model.safetensors" in base["files"]
    assert "model_index.json" in base["files"]
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_keeps_the_dense_encoder_without_an_fp8_request(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _all_cached(monkeypatch)
    plan = _flux_download_plan()
    base = next(e for e in plan["entries"] if e["repo_id"] == "black-forest-labs/FLUX.1-dev")
    assert "text_encoder/model.safetensors" in base["files"]
    assert len(plan["entries"]) == 2


def test_download_plan_keeps_the_dense_encoder_when_the_precast_repo_is_unavailable(monkeypatch):
    _fake_flux_hub(monkeypatch)
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_sources",
        lambda fam, *, te_quant_mode, target: {
            "text_encoder": types.SimpleNamespace(
                kind = "repo", location = "unsloth/does-not-exist", filename = "te-fp8.pt"
            )
        },
    )
    _no_cache(monkeypatch)
    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        text_encoder_quant = "fp8",
    )
    base = next(e for e in plan["entries"] if e["repo_id"] == "unsloth/FLUX.1-dev")
    assert "text_encoder/model.safetensors" in base["files"]
    assert not any(e["repo_id"] == "unsloth/does-not-exist" for e in plan["entries"])


_ZIMAGE_BASE_SIBLINGS = [
    _FakeSibling("model_index.json", 1000),
    _FakeSibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 12 * GB),
    _FakeSibling("text_encoder/model.safetensors", 8 * GB),
    _FakeSibling("vae/diffusion_pytorch_model.safetensors", 300),
]
_ZIMAGE_BASE_SIBLINGS_BY_NAME = {s.rfilename: s.size for s in _ZIMAGE_BASE_SIBLINGS}


_QWEN_EDIT_Q6 = "qwen-image-edit-2511-Q6_K.gguf"
_QWEN_EDIT_BASE_SIBLINGS = [
    _FakeSibling("model_index.json", 516),
    _FakeSibling("transformer/diffusion_pytorch_model-00001-of-00005.safetensors", 9_973_578_592),
    _FakeSibling("transformer/diffusion_pytorch_model-00002-of-00005.safetensors", 9_987_326_072),
    _FakeSibling("transformer/diffusion_pytorch_model-00003-of-00005.safetensors", 9_987_307_440),
    _FakeSibling("transformer/diffusion_pytorch_model-00004-of-00005.safetensors", 9_930_685_712),
    _FakeSibling("transformer/diffusion_pytorch_model-00005-of-00005.safetensors", 982_130_472),
    _FakeSibling("text_encoder/model-00001-of-00004.safetensors", 4_968_243_304),
    _FakeSibling("text_encoder/model-00002-of-00004.safetensors", 4_991_495_816),
    _FakeSibling("text_encoder/model-00003-of-00004.safetensors", 4_932_751_040),
    _FakeSibling("text_encoder/model-00004-of-00004.safetensors", 1_691_924_384),
    _FakeSibling("vae/diffusion_pytorch_model.safetensors", 253_806_966),
    _FakeSibling("processor/merges.txt", 1_671_853),
    _FakeSibling("processor/tokenizer.json", 11_421_896),
    _FakeSibling("processor/vocab.json", 2_776_833),
]


def test_qwen_edit_q6_auto_stays_gguf_but_explicit_quant_requests_dense_transformer(
    fake_runtime, tmp_path, monkeypatch
):
    """Cover the reported live Q6 shape and its explicit-quant causal control."""
    from core.inference import diffusion as dmod
    from core.inference import diffusion_memory as dmem

    checkpoint_repo = "unsloth/Qwen-Image-Edit-2511-GGUF"
    base_repo = "Qwen/Qwen-Image-Edit-2511"
    _fake_hf_api(
        monkeypatch,
        {
            checkpoint_repo: [_FakeSibling(_QWEN_EDIT_Q6, 16_852_417_120)],
            base_repo: _QWEN_EDIT_BASE_SIBLINGS,
        },
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: base_repo)
    _split_cache_roots(tmp_path, monkeypatch)
    _no_cache(monkeypatch)

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(prequant = False, steady_total_mib = 39_900),
    )
    monkeypatch.setattr(
        dmem,
        "snapshot_device_memory",
        lambda target: types.SimpleNamespace(
            total_mib = 81_920, free_mib = 80_000, memory_kind = "discrete_vram"
        ),
    )

    auto = backend.download_plan(checkpoint_repo, gguf_filename = _QWEN_EDIT_Q6)
    auto_base = next(e for e in auto["entries"] if e["gguf_filename"] is None)
    auto_transformer = [f for f in auto_base["files"] if f.startswith("transformer/")]
    assert auto_transformer == []
    assert 16_000_000_000 < auto_base["bytes"] < 18_000_000_000

    explicit = backend.download_plan(
        checkpoint_repo, gguf_filename = _QWEN_EDIT_Q6, transformer_quant = "int8"
    )
    explicit_base = next(e for e in explicit["entries"] if e["gguf_filename"] is None)
    explicit_transformer = [f for f in explicit_base["files"] if f.startswith("transformer/")]
    assert len(explicit_transformer) == 5
    assert 55_000_000_000 < explicit_base["bytes"] < 60_000_000_000


def test_download_plan_stages_no_second_denoiser_for_an_uncached_prequant(monkeypatch):
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/Z-Image-GGUF": [_FakeSibling("Z-Image-Turbo-Q4_K_M.gguf", 4 * GB)],
            "Tongyi-MAI/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS,
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo", lambda *a, **k: "Tongyi-MAI/Z-Image-Turbo"
    )
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf"
    )

    assert [e["repo_id"] for e in plan["entries"]] == [
        "unsloth/Z-Image-GGUF",
        "unsloth/Z-Image-Turbo",
    ]
    checkpoint, base = plan["entries"]
    assert checkpoint["files"] == ["Z-Image-Turbo-Q4_K_M.gguf"]
    assert not any(f.endswith(".pt") for e in plan["entries"] for f in e["files"])
    assert not any(f.startswith("transformer/") for f in base["files"])
    assert "text_encoder/model.safetensors" in base["files"]
    assert plan["total_bytes"] == 4 * GB + base["bytes"] < 17 * GB


def test_download_plan_counts_the_hosted_prequant_in_the_required_footprint(monkeypatch):
    _fake_zimage_hub(monkeypatch)
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )

    staged = {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]}
    assert ("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-FP8.pt") in staged
    assert plan["required_bytes"] == plan["total_bytes"]
    baseline = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf"
    )
    assert plan["required_bytes"] - baseline["required_bytes"] == 6 * GB
    prequant = next(e for e in plan["entries"] if e["repo_id"] == "unsloth/Z-Image-Turbo-FP8")
    assert prequant["checkpoint"] is False


def test_download_plan_counts_a_cached_lower_auto_prequant(monkeypatch):
    from core.inference import diffusion as dmod

    source = types.SimpleNamespace(
        kind = "repo",
        location = "unsloth/Qwen-Image-FP8",
        filename = "Qwen-Image-INT8.pt",
        fallback_filenames = ("transformer_int8.pt",),
    )
    _fake_hf_api(
        monkeypatch,
        {
            "unsloth/Qwen-Image-GGUF": [_FakeSibling("Qwen-Image-Q4_K_M.gguf", 4 * GB)],
            "Qwen/Qwen-Image": _ZIMAGE_BASE_SIBLINGS,
            source.location: [_FakeSibling(source.filename, 6 * GB)],
        },
    )
    monkeypatch.setattr(dmod, "_resolve_base_repo", lambda *a, **k: "Qwen/Qwen-Image")
    monkeypatch.setattr(dmod, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.auto_scheme_candidates",
        lambda *a, **k: ("fp8", "int8"),
    )
    monkeypatch.setattr(
        dmod,
        "usable_prequant_source",
        lambda fam, scheme, **kw: source if scheme == "int8" else None,
    )
    monkeypatch.setattr(dmod, "prequant_checkpoint_cached", lambda *a, **k: True)
    # Whether this install can open an int8 pickle is a separate question (torchao 0.18); pin it.
    readable = {"answer": True}
    monkeypatch.setattr(
        "core.inference.diffusion_prequant.restricted_prequant_load_supported",
        lambda *a, **k: readable["answer"],
    )

    def plans():
        plan = DiffusionBackend().download_plan(
            "unsloth/Qwen-Image-GGUF",
            gguf_filename = "Qwen-Image-Q4_K_M.gguf",
            text_encoder_quant = "off",
        )
        baseline = DiffusionBackend().download_plan(
            "unsloth/Qwen-Image-GGUF",
            gguf_filename = "Qwen-Image-Q4_K_M.gguf",
            text_encoder_quant = "off",
            speed_mode = "off",
        )
        return plan, baseline

    plan, baseline = plans()
    assert plan["required_bytes"] - baseline["required_bytes"] == 6 * GB
    assert any(source.filename in entry["files"] for entry in plan["entries"])

    readable["answer"] = False
    plan, baseline = plans()
    assert plan["required_bytes"] == baseline["required_bytes"]
    assert not any(source.filename in entry["files"] for entry in plan["entries"])


def test_download_plan_omits_the_prequant_under_a_definite_offload_policy(monkeypatch):
    # Balanced / low_vram offload by MODE, so the load keeps the GGUF and never fetches the prequant.
    _fake_zimage_hub(monkeypatch)
    _stub_hosted_prequant(monkeypatch, cached = True)

    for kwargs in (
        {"memory_mode": "balanced"},
        {"memory_mode": "low_vram"},
        {"cpu_offload": True},
    ):
        plan = DiffusionBackend().download_plan(
            "unsloth/Z-Image-GGUF",
            gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
            transformer_quant = "fp8",
            **kwargs,
        )
        assert plan["required_bytes"] == plan["total_bytes"], kwargs
        assert not any(f.endswith("-FP8.pt") for e in plan["entries"] for f in e["files"]), kwargs

    resident = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert any(f.endswith("-FP8.pt") for e in resident["entries"] for f in e["files"])


def test_download_plan_omits_the_prequant_for_an_auto_pick_at_speed_off(monkeypatch):
    # load_pipeline forces an AUTO quant to "off" under Speed="off", which normalizes to None and
    # skips the fast path, so nothing is fetched and the footprint must not claim it.
    _fake_zimage_hub(monkeypatch)
    _stub_hosted_prequant(monkeypatch, cached = True)

    auto = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf", speed_mode = "off"
    )
    assert not any(f.endswith("-FP8.pt") for e in auto["entries"] for f in e["files"])

    explicit = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        speed_mode = "off",
        transformer_quant = "fp8",
    )
    assert any(f.endswith("-FP8.pt") for e in explicit["entries"] for f in e["files"])


def test_download_plan_omits_a_prequant_an_auto_pick_would_decline(monkeypatch):
    _fake_zimage_hub(monkeypatch)
    _stub_hosted_prequant(monkeypatch, cached = False)
    monkeypatch.setattr(
        "core.inference.diffusion._uncached_prequant_repo",
        lambda *a, **k: "unsloth/Z-Image-Turbo-FP8",
    )

    plan = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf"
    )

    assert plan["required_bytes"] == plan["total_bytes"]


def test_download_plan_for_a_pipeline_kind_ignores_the_prequant_cache(monkeypatch):
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _ZIMAGE_BASE_SIBLINGS})
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert any(f.startswith("transformer/") for f in plan["entries"][0]["files"])


def test_download_plan_restages_a_base_split_across_both_cache_roots(monkeypatch):
    # A base split across old and new cache roots hides one half from from_pretrained; restage it.
    _fake_hf_api(monkeypatch, {"unsloth/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS})
    old_root_only = {"model_index.json"}

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
        **kwargs,
    ):
        asks_live = roots is not None and roots != (None,)
        return (filename not in old_root_only) if asks_live else (filename in old_root_only)

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    plan = DiffusionBackend().download_plan("unsloth/Z-Image-Turbo", model_kind = "pipeline")

    entry = plan["entries"][0]
    assert set(entry["files"]) == set(_ZIMAGE_BASE_SIBLINGS_BY_NAME)
    assert entry["bytes"] == sum(_ZIMAGE_BASE_SIBLINGS_BY_NAME[n] for n in old_root_only)


def test_download_plan_restages_the_old_root_half_when_other_files_are_missing_too(monkeypatch):
    _fake_hf_api(monkeypatch, {"unsloth/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS})
    old_root_only = {"model_index.json"}
    absent = {"vae/diffusion_pytorch_model.safetensors"}

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
        **kwargs,
    ):
        if filename in absent:
            return False
        asks_live = roots is not None and roots != (None,)
        return (filename not in old_root_only) if asks_live else (filename in old_root_only)

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    plan = DiffusionBackend().download_plan("unsloth/Z-Image-Turbo", model_kind = "pipeline")

    entry = plan["entries"][0]
    assert set(entry["files"]) == set(_ZIMAGE_BASE_SIBLINGS_BY_NAME)
    assert entry["bytes"] == sum(_ZIMAGE_BASE_SIBLINGS_BY_NAME[n] for n in old_root_only | absent)


def test_download_plan_stages_a_file_a_stale_live_copy_shadows(monkeypatch):
    # reuse_other_cache_root switches roots only when the live lookup finds nothing: a stale copy wins.
    _fake_hf_api(monkeypatch, {"unsloth/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS})
    shadowed = {"model_index.json"}

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
        **kwargs,
    ):
        asks_live = roots is not None and roots != (None,)
        if filename in shadowed:
            return expected_size is None if asks_live else True
        return asks_live

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    plan = DiffusionBackend().download_plan("unsloth/Z-Image-Turbo", model_kind = "pipeline")

    entry = plan["entries"][0]
    assert set(entry["files"]) == set(_ZIMAGE_BASE_SIBLINGS_BY_NAME)
    assert entry["bytes"] == sum(_ZIMAGE_BASE_SIBLINGS_BY_NAME[n] for n in shadowed)


def test_download_plan_stages_nothing_for_a_base_wholly_in_the_other_root(monkeypatch):
    _fake_hf_api(monkeypatch, {"unsloth/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS})

    def probe(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
        **kwargs,
    ):
        return roots is None or roots == (None,)

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    plan = DiffusionBackend().download_plan("unsloth/Z-Image-Turbo", model_kind = "pipeline")

    assert plan["entries"] == [], "a base living entirely in one root is already loadable"


def test_download_plan_declines_an_unrecognised_gguf_instead_of_raising(monkeypatch):
    # A repo matching no family plans no work rather than 500ing the route.
    _fake_hf_api(monkeypatch, {})

    plan = DiffusionBackend().download_plan(
        "someone/mixed-gguf-collection",
        gguf_filename = "totally-unknown-thing-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert plan == {"entries": [], "total_bytes": 0, "required_bytes": 0, "checkpoint_bytes": 0}


def test_download_plan_still_plans_an_unrecognised_gguf_given_an_explicit_base(monkeypatch):
    _fake_hf_api(
        monkeypatch,
        {
            "someone/mixed-gguf-collection": [
                _FakeSibling("totally-unknown-thing-Q4_K_M.gguf", 4_000)
            ],
            "unsloth/Z-Image-Turbo": _ZIMAGE_BASE_SIBLINGS,
        },
    )
    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(
        "someone/mixed-gguf-collection",
        gguf_filename = "totally-unknown-thing-Q4_K_M.gguf",
        model_kind = "gguf",
        base_repo = "unsloth/Z-Image-Turbo",
    )

    assert plan["entries"], "an explicit base still has a companion set to stage"


def test_unload_fences_queued_generations_while_it_waits(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)

    seen: list[int] = []
    real_unload_locked = backend._unload_locked

    def _record_then_unload():
        seen.append(backend._teardown_waiters)
        real_unload_locked()

    backend._unload_locked = _record_then_unload
    backend.unload()

    assert seen == [1]
    assert backend._teardown_waiters == 0


def test_a_raising_unload_still_drains_the_teardown_fence(fake_runtime, tmp_path, monkeypatch):
    # clear_gpu_cache() raises on a sticky CUDA fault; the fence must still come down.
    from core.inference import diffusion as diffusion_module

    backend = _loaded_backend(tmp_path)

    real_clear = diffusion_module.clear_gpu_cache

    def _sticky(*_args, **_kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", _sticky)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        backend.unload()
    assert backend._teardown_waiters == 0, "a failed teardown must not leave the fence up"

    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", real_clear)
    _load_into(backend, tmp_path)
    assert backend.generate(prompt = "after", steps = 2)["images"]


def test_unload_returns_freed_host_pages_after_the_gpu_cache(fake_runtime, tmp_path, monkeypatch):
    # The trim must run after clear_gpu_cache() (which runs gc) and with the state gone.
    from core.inference import diffusion as diffusion_module

    backend = _loaded_backend(tmp_path)
    order = []
    real_clear = diffusion_module.clear_gpu_cache

    def _clear(*args, **kwargs):
        order.append("clear")
        return real_clear(*args, **kwargs)

    def _trim(logger = None):
        order.append(("trim", backend._state is None))
        return True

    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", _clear)
    monkeypatch.setattr(diffusion_module, "reclaim_host_memory", _trim)
    assert backend.unload()["loaded"] is False
    assert order == ["clear", ("trim", True)]
    backend.unload()
    assert order == ["clear", ("trim", True)]


class _RecordingGate(threading.Event):
    """Teardown gate that reports every time a generation parks on it."""

    def __init__(self, parked: threading.Event):
        super().__init__()
        self._parked = parked

    def wait(self, timeout = None):
        self._parked.set()
        return super().wait(timeout)


class _AdmissionHookLock:
    """Lock wrapper that pauses generation after atomic admission releases state."""

    def __init__(self, backend, on_admitted):
        self._lock = threading.Lock()
        self._backend = backend
        self._on_admitted = on_admitted
        self._fired = False

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()
        if (
            not self._fired
            and threading.current_thread().name == "generation-under-test"
            and self._backend._active_generate_cancel is not None
        ):
            self._fired = True
            self._on_admitted()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_generation_waits_for_all_pending_teardowns(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend(tmp_path)
    assert backend.generate(prompt = "before", steps = 2)["images"]

    parked = threading.Event()
    denoise_entered = threading.Event()
    pipe_type = type(backend._state.pipe)
    real_call = pipe_type.__call__

    def record_denoise(self, *args, **kwargs):
        denoise_entered.set()
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(pipe_type, "__call__", record_denoise)
    backend._teardown_drained = _RecordingGate(parked)
    with backend._lock:
        backend._reserve_teardown_locked()
        backend._reserve_teardown_locked()

    outcome: dict = {}
    worker = threading.Thread(
        target = lambda: outcome.setdefault("result", backend.generate(prompt = "during", steps = 2)),
        daemon = True,
    )
    worker.start()
    assert parked.wait(5), "generation did not yield to the pending teardown"

    parked.clear()
    with backend._lock:
        backend._release_teardown_locked()
    assert parked.wait(5), "generation did not re-park behind the final teardown"
    assert not backend._teardown_drained.is_set(), "the gate opened with a reservation live"
    assert not denoise_entered.is_set(), "generation denoised before every teardown drained"

    with backend._lock:
        backend._release_teardown_locked()
    worker.join(5)
    assert not worker.is_alive(), "generation did not resume after the teardown drained"
    assert denoise_entered.is_set()
    assert outcome["result"]["images"]


def test_cancel_wakes_generation_waiting_for_replacement(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend(tmp_path)

    replacement_build_started = threading.Event()
    allow_replacement_commit = threading.Event()
    real_from_single_file = _FakeTransformer.from_single_file

    def blocking_from_single_file(cls, path, **kwargs):
        replacement_build_started.set()
        assert allow_replacement_commit.wait(5), "replacement load was not released"
        return real_from_single_file(path, **kwargs)

    monkeypatch.setattr(
        _FakeTransformer, "from_single_file", classmethod(blocking_from_single_file)
    )

    load_outcome: dict = {}

    def replace_model():
        try:
            load_outcome["result"] = backend.load_pipeline(
                str(tmp_path),
                gguf_filename = "model.gguf",
                base_repo = "base/repo",
                family_override = "z-image",
            )
        except BaseException as exc:  # noqa: BLE001 - surface worker failures in the test thread
            load_outcome["error"] = exc

    loader = threading.Thread(target = replace_model, daemon = True)
    loader.start()
    assert replacement_build_started.wait(5), load_outcome

    outcome: dict = {}

    def generate():
        try:
            backend.generate(prompt = "cancel while queued", steps = 2)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target = generate, daemon = True)
    worker.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if backend._queued_generate_cancels:
                break
        time.sleep(0.01)
    else:
        allow_replacement_commit.set()
        pytest.fail("generation was not published while waiting for replacement")
    assert backend.cancel_generate() is True
    worker.join(5)
    assert not worker.is_alive(), "cancelled generation waited for replacement to finish"
    assert outcome["error"] == DIFFUSION_CANCELLED_MSG
    assert not backend._queued_generate_cancels
    assert backend.generate_progress()["active"] is False
    assert loader.is_alive(), "replacement unexpectedly finished before the queued cancel"

    allow_replacement_commit.set()
    loader.join(5)
    assert not loader.is_alive(), "replacement load did not finish"
    assert "error" not in load_outcome, load_outcome


def test_cancel_stops_every_generation_queued_behind_teardown(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    with backend._lock:
        backend._reserve_teardown_locked()

    errors: list[str] = []

    def generate(prompt):
        try:
            backend.generate(prompt = prompt, steps = 2)
        except RuntimeError as exc:
            errors.append(str(exc))

    workers = [
        threading.Thread(target = generate, args = (f"queued-{index}",), daemon = True)
        for index in range(2)
    ]
    for worker in workers:
        worker.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if len(backend._queued_generate_cancels) == 2:
                break
        time.sleep(0.01)
    else:
        pytest.fail("both generations did not register for queued cancellation")

    try:
        assert backend.cancel_generate() is True
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        assert errors == [DIFFUSION_CANCELLED_MSG, DIFFUSION_CANCELLED_MSG]
        assert not backend._queued_generate_cancels
    finally:
        with backend._lock:
            if backend._teardown_waiters:
                backend._release_teardown_locked()


class _SlotYieldLock:
    """Generation-lock wrapper that reports the release yielding the slot to a teardown."""

    def __init__(self, lock, yielded):
        self._lock = lock
        self._yielded = yielded

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()
        self._yielded.set()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_cancel_reaches_a_queued_generation_while_a_load_holds_the_state_lock(
    fake_runtime, tmp_path
):
    # Condition.wait() reacquires _lock before returning, so Stop must not wait on a Condition over it.
    backend = _loaded_backend(tmp_path)
    with backend._lock:
        backend._reserve_teardown_locked()

    yielded = threading.Event()
    backend._generate_lock = _SlotYieldLock(backend._generate_lock, yielded)

    outcome: dict = {}

    def generate():
        try:
            backend.generate(prompt = "queued", steps = 2)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target = generate, daemon = True)
    worker.start()
    assert yielded.wait(5), "generation did not yield the slot to the teardown"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if backend._queued_generate_cancels:
                break
        time.sleep(0.01)
    else:
        pytest.fail("the queued generation never published a cancel event")

    with backend._lock:
        assert backend.cancel_generate() is True
        worker.join(5)
        assert not worker.is_alive(), "Stop could not reach the queued generation"
    assert outcome["error"] == DIFFUSION_CANCELLED_MSG
    assert not backend._queued_generate_cancels

    with backend._lock:
        backend._release_teardown_locked()


def test_cancel_reaches_a_waiter_once_the_generation_it_queued_behind_exits(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend(tmp_path)

    denoising = threading.Event()
    release_active = threading.Event()
    calls: list[int] = []
    real_call = _FakePipe.__call__

    def _call(self, **kwargs):
        first = not calls
        calls.append(1)
        if first:
            denoising.set()
            assert release_active.wait(5), "the active generation was never released"
        return real_call(self, **kwargs)

    monkeypatch.setattr(_FakePipe, "__call__", _call)

    outcomes: dict = {}

    def generate(key, prompt):
        try:
            outcomes[key] = backend.generate(prompt = prompt, steps = 2)
        except RuntimeError as exc:
            outcomes[key] = exc

    active = threading.Thread(target = generate, args = ("active", "active"), daemon = True)
    active.start()
    assert denoising.wait(5), "the first generation never started denoising"
    waiter = threading.Thread(target = generate, args = ("waiter", "waiter"), daemon = True)
    waiter.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if backend._queued_generate_cancels:
                break
        time.sleep(0.01)
    else:
        release_active.set()
        pytest.fail("the waiting request was never published")

    with backend._lock:
        backend._reserve_teardown_locked()
    release_active.set()
    active.join(5)
    assert not active.is_alive()

    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with backend._generation_cancel_lock:
                if backend._active_generate_cancel is None:
                    break
            time.sleep(0.01)
        else:
            pytest.fail("the active generation never deregistered")

        assert backend.cancel_generate() is True, "Stop did not reach the waiting request"
        waiter.join(5)
        assert not waiter.is_alive(), "the waiting request stayed queued through the teardown"
        assert isinstance(outcomes["waiter"], RuntimeError)
        assert str(outcomes["waiter"]) == DIFFUSION_CANCELLED_MSG
        assert not backend._queued_generate_cancels
    finally:
        with backend._lock:
            if backend._teardown_waiters:
                backend._release_teardown_locked()
        waiter.join(5)


def test_cancel_spares_a_serialized_request_through_the_active_epilogue(
    fake_runtime, tmp_path, monkeypatch
):
    # The active generation drops its cancel event before its epilogue but still owns the slot.
    backend = _loaded_backend(tmp_path)

    from core.inference import diffusion as diffusion_module

    queued = threading.Event()
    stop_answered: list[bool] = []
    fired: list[int] = []
    real_baked = diffusion_module._baked_lora_names

    def _baked(pipe):
        if not fired:
            fired.append(1)
            if queued.wait(5):
                stop_answered.append(backend.cancel_generate())
        return real_baked(pipe)

    monkeypatch.setattr(diffusion_module, "_baked_lora_names", _baked)

    outcomes: dict = {}

    def generate(key, prompt):
        try:
            outcomes[key] = backend.generate(prompt = prompt, steps = 2)
        except RuntimeError as exc:
            outcomes[key] = exc

    active = threading.Thread(target = generate, args = ("active", "active"), daemon = True)
    active.start()
    serialized = threading.Thread(target = generate, args = ("serialized", "serialized"), daemon = True)
    serialized.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if backend._queued_generate_cancels:
                break
        time.sleep(0.01)
    queued.set()

    active.join(5)
    serialized.join(5)
    assert not active.is_alive() and not serialized.is_alive()
    assert stop_answered == [False], "Stop claimed a generation that had already committed"
    assert isinstance(outcomes["active"], dict), outcomes["active"]
    assert isinstance(outcomes["serialized"], dict), outcomes["serialized"]
    assert outcomes["serialized"]["images"]


class _FirstAcquireHookLock:
    """Generation-lock wrapper that runs a hook before the first acquisition attempt."""

    def __init__(self, lock, on_first_acquire):
        self._lock = lock
        self._on_first_acquire = on_first_acquire
        self._fired = False

    def acquire(self, *args, **kwargs):
        if not self._fired:
            self._fired = True
            self._on_first_acquire()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_stop_reaches_a_queued_generation_before_its_first_lock_attempt(fake_runtime, tmp_path):
    # Publish the cancel event before the first timed acquire, or Stop misses a 100 ms window.
    backend = _loaded_backend(tmp_path)
    with backend._lock:
        backend._reserve_teardown_locked()

    stop_answered: list[bool] = []
    backend._generate_lock = _FirstAcquireHookLock(
        backend._generate_lock, lambda: stop_answered.append(backend.cancel_generate())
    )

    outcome: dict = {}

    def generate():
        try:
            backend.generate(prompt = "queued", steps = 2)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target = generate, daemon = True)
    worker.start()
    try:
        worker.join(5)
        assert not worker.is_alive(), "the queued generation did not unwind"
        assert stop_answered == [True], "Stop did not see the request before it queued"
        assert outcome["error"] == DIFFUSION_CANCELLED_MSG
        assert not backend._queued_generate_cancels
    finally:
        with backend._lock:
            if backend._teardown_waiters:
                backend._release_teardown_locked()
        worker.join(5)


class _SlotHandoffLock:
    """Generation-lock wrapper that pauses a waiter after it receives the slot."""

    def __init__(self, lock, handed_off, admit_waiter):
        self._lock = lock
        self._handed_off = handed_off
        self._admit_waiter = admit_waiter

    def acquire(self, *args, **kwargs):
        acquired = self._lock.acquire(*args, **kwargs)
        if acquired and threading.current_thread().name == "serialized-waiter":
            self._handed_off.set()
            assert self._admit_waiter.wait(5), "the waiter was not allowed to register"
        return acquired

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_cancel_spares_a_serialized_request_during_slot_handoff(
    fake_runtime, tmp_path, monkeypatch
):
    # A waiter can own _generate_lock before moving its cancel event to the active slot.
    backend = _loaded_backend(tmp_path)

    denoising = threading.Event()
    release_active = threading.Event()
    handed_off = threading.Event()
    admit_waiter = threading.Event()
    calls: list[int] = []
    real_call = _FakePipe.__call__

    def _call(self, **kwargs):
        first = not calls
        calls.append(1)
        if first:
            denoising.set()
            assert release_active.wait(5), "the active generation was never released"
        return real_call(self, **kwargs)

    monkeypatch.setattr(_FakePipe, "__call__", _call)
    backend._generate_lock = _SlotHandoffLock(backend._generate_lock, handed_off, admit_waiter)

    outcomes: dict = {}

    def generate(key, prompt):
        try:
            outcomes[key] = backend.generate(prompt = prompt, steps = 2)
        except RuntimeError as exc:
            outcomes[key] = exc

    active = threading.Thread(
        target = generate,
        args = ("active", "active"),
        name = "active-generation",
        daemon = True,
    )
    active.start()
    assert denoising.wait(5), "the first generation never started denoising"
    serialized = threading.Thread(
        target = generate,
        args = ("serialized", "serialized"),
        name = "serialized-waiter",
        daemon = True,
    )
    serialized.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with backend._generation_cancel_lock:
            if backend._queued_generate_cancels:
                break
        time.sleep(0.01)
    else:
        pytest.fail("the serialized request was never published")

    release_active.set()
    assert handed_off.wait(5), "the serialized waiter never received the slot"
    assert backend.cancel_generate() is False
    admit_waiter.set()

    active.join(5)
    serialized.join(5)
    assert not active.is_alive() and not serialized.is_alive()
    assert isinstance(outcomes["active"], dict), outcomes["active"]
    assert isinstance(outcomes["serialized"], dict), outcomes["serialized"]
    assert outcomes["serialized"]["images"]


class _SlotContentionLock:
    """Generation-lock wrapper that reports a failed (contended) acquisition."""

    def __init__(self, lock, contended):
        self._lock = lock
        self._contended = contended

    def acquire(self, *args, **kwargs):
        acquired = self._lock.acquire(*args, **kwargs)
        if not acquired:
            self._contended.set()
        return acquired

    def release(self):
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_cancel_spares_a_request_only_serialized_behind_the_active_one(
    fake_runtime, tmp_path, monkeypatch
):
    # /images/generate and /v1/images/generations can both be in flight; Stop must not fail the queued one.
    backend = _loaded_backend(tmp_path)

    denoising = threading.Event()
    release_active = threading.Event()
    contended = threading.Event()
    calls: list[int] = []
    real_call = _FakePipe.__call__

    def _call(self, **kwargs):
        first = not calls
        calls.append(1)
        if first:
            denoising.set()
            assert release_active.wait(5), "the active generation was never released"
        return real_call(self, **kwargs)

    monkeypatch.setattr(_FakePipe, "__call__", _call)
    backend._generate_lock = _SlotContentionLock(backend._generate_lock, contended)

    outcomes: dict = {}

    def generate(key, prompt):
        try:
            outcomes[key] = backend.generate(prompt = prompt, steps = 2)
        except RuntimeError as exc:
            outcomes[key] = exc

    active = threading.Thread(target = generate, args = ("active", "active"), daemon = True)
    active.start()
    assert denoising.wait(5), "the first generation never started denoising"
    serialized = threading.Thread(target = generate, args = ("serialized", "serialized"), daemon = True)
    serialized.start()
    assert contended.wait(5), "the second request never queued on the generation lock"

    assert backend.cancel_generate() is True
    release_active.set()
    active.join(5)
    serialized.join(5)
    assert not active.is_alive() and not serialized.is_alive()

    assert isinstance(outcomes["active"], RuntimeError)
    assert str(outcomes["active"]) == DIFFUSION_CANCELLED_MSG
    assert isinstance(outcomes["serialized"], dict), outcomes["serialized"]
    assert outcomes["serialized"]["images"]


def test_admission_registers_cancel_before_teardown_can_reserve(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)

    start_teardown = threading.Event()
    teardown_reserved = threading.Event()
    saw_active_cancel: list[bool] = []

    def after_admission():
        start_teardown.set()
        assert teardown_reserved.wait(5), "teardown did not reserve after admission"

    backend._lock = _AdmissionHookLock(backend, after_admission)

    def teardown():
        assert start_teardown.wait(5), "generation never reached admission"
        with backend._lock:
            with backend._generation_cancel_lock:
                cancel = backend._active_generate_cancel
                saw_active_cancel.append(cancel is not None)
                if cancel is not None:
                    cancel.set()
            backend._reserve_teardown_locked()
            teardown_reserved.set()
        with backend._generate_lock:
            with backend._lock:
                try:
                    backend._unload_locked()
                finally:
                    backend._release_teardown_locked()

    teardown_worker = threading.Thread(target = teardown, daemon = True)
    teardown_worker.start()

    outcome: dict = {}

    def generate():
        try:
            backend.generate(prompt = "atomic admission", steps = 2)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    generation_worker = threading.Thread(target = generate, name = "generation-under-test", daemon = True)
    generation_worker.start()
    generation_worker.join(5)
    teardown_worker.join(5)

    assert not generation_worker.is_alive(), "generation ignored teardown cancellation"
    assert not teardown_worker.is_alive(), "teardown remained blocked behind generation"
    assert saw_active_cancel == [True]
    assert outcome["error"] == DIFFUSION_CANCELLED_MSG
    assert backend._state is None
    assert backend._teardown_waiters == 0


def test_generation_reports_not_loaded_after_waiting_for_unload(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)

    parked = threading.Event()
    backend._teardown_drained = _RecordingGate(parked)
    with backend._lock:
        backend._reserve_teardown_locked()

    outcome: dict = {}

    def generate():
        try:
            backend.generate(prompt = "during", steps = 2)
        except RuntimeError as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target = generate, daemon = True)
    worker.start()
    assert parked.wait(5), "generation did not wait for unload"

    with backend._lock:
        backend._unload_locked()
        backend._release_teardown_locked()
    worker.join(5)
    assert not worker.is_alive(), "generation remained blocked after unload"
    assert outcome["error"] == "No diffusion model is loaded."


def test_a_superseding_load_fences_queued_generations_too(fake_runtime, tmp_path):
    # begin_load frees the old pipeline behind the same barrier, so it needs the same fence: a queued generation would otherwise run on the pipe being dropped.
    backend = _loaded_backend(tmp_path)

    seen: list[int] = []
    real_unload_locked = backend._unload_locked

    def _record_then_unload():
        seen.append(backend._teardown_waiters)
        real_unload_locked()

    backend._unload_locked = _record_then_unload
    _load_into(backend, tmp_path)

    assert seen == [1]
    assert backend._teardown_waiters == 0


def test_auto_retries_a_lower_scheme_that_has_a_prequant(monkeypatch):
    # Qwen-Image: auto picks fp8 but only int8 is published; the retry walks the ladder for a cached rung.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.auto_scheme_candidates",
        lambda target, family = None, **_kw: ("fp8", "mxfp8", "int8"),
    )
    have = {"int8"}
    monkeypatch.setattr(
        "core.inference.diffusion.usable_prequant_source",
        lambda fam, scheme, path_override = None, base_repo = None: (
            types.SimpleNamespace(kind = "repo", location = f"unsloth/{scheme}")
            if scheme in have
            else None
        ),
    )
    cached = {"int8"}
    monkeypatch.setattr(
        "core.inference.diffusion.prequant_checkpoint_cached",
        lambda source, cache_dir = None: source.location.rsplit("/", 1)[-1] in cached,
    )
    fam = types.SimpleNamespace(name = "qwen-image")
    retry = DiffusionBackend._auto_prequant_retry_scheme(
        object(),
        fam,
        "auto",
        "fp8",
        base_repo = "Qwen/Qwen-Image",
        path_override = None,
        loras = None,
    )
    assert retry == "int8"

    # GGUF picks are cached-only and _uncached_prequant_repo only sees the winner, so check here too.
    cached.clear()
    assert (
        DiffusionBackend._auto_prequant_retry_scheme(
            object(),
            fam,
            "auto",
            "fp8",
            base_repo = "Qwen/Qwen-Image",
            path_override = None,
            loras = None,
        )
        is None
    )
    monkeypatch.setattr(
        "core.inference.diffusion.usable_prequant_source",
        lambda fam, scheme, path_override = None, base_repo = None: (
            types.SimpleNamespace(kind = "path", location = "/tmp/int8.pt")
            if scheme == "int8"
            else None
        ),
    )
    assert (
        DiffusionBackend._auto_prequant_retry_scheme(
            object(),
            fam,
            "auto",
            "fp8",
            base_repo = "Qwen/Qwen-Image",
            path_override = None,
            loras = None,
        )
        == "int8"
    )

    assert (
        DiffusionBackend._auto_prequant_retry_scheme(
            object(),
            fam,
            "fp8",
            "fp8",
            base_repo = "Qwen/Qwen-Image",
            path_override = None,
            loras = None,
        )
        is None
    )

    monkeypatch.setattr(
        "core.inference.diffusion.usable_prequant_source",
        lambda fam, scheme, path_override = None, base_repo = None: None,
    )
    assert (
        DiffusionBackend._auto_prequant_retry_scheme(
            object(),
            fam,
            "auto",
            "fp8",
            base_repo = "Qwen/Qwen-Image",
            path_override = None,
            loras = None,
        )
        is None
    )


def test_the_retry_never_climbs_above_the_scheme_auto_already_chose(monkeypatch):
    # Rungs ABOVE the winner were already rejected by the ladder, so never offer them back.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.auto_scheme_candidates",
        lambda target, family = None: ("fp8", "mxfp8", "int8"),
    )
    monkeypatch.setattr(
        "core.inference.diffusion.usable_prequant_source",
        lambda fam, scheme, path_override = None, base_repo = None: (
            types.SimpleNamespace(kind = "path", location = "/tmp/fp8.pt") if scheme == "fp8" else None
        ),
    )
    assert (
        DiffusionBackend._auto_prequant_retry_scheme(
            object(),
            types.SimpleNamespace(name = "qwen-image"),
            "auto",
            "mxfp8",
            base_repo = "Qwen/Qwen-Image",
            path_override = None,
            loras = None,
        )
        is None
    )


def test_the_offload_retry_runs_when_the_auto_winner_had_no_candidate_at_all(
    fake_runtime, tmp_path, monkeypatch
):
    # The winner can yield no estimate (disk gate) while a lower cached rung would load resident.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: "int8")
    )

    resolved = []

    def fake_resolve(**kw):
        resolved.append(kw.get("requested"))
        if len(resolved) == 1:
            return None
        return types.SimpleNamespace(
            transient_transformer_mib = 12_000, companions_mib = 8_000, prequant = True
        )

    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", fake_resolve)

    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_into(
        backend,
        tmp_path,
        gguf_filename = "m.gguf",
        base_repo = None,
        family_override = "qwen-image",
        transformer_quant = "auto",
    )

    assert len(resolved) == 2
    assert attempted == [False]


def test_the_resident_retry_runs_when_the_dense_shards_were_never_staged(
    fake_runtime, tmp_path, monkeypatch
):
    # A capacity-declined prefetch reads 0 resident bytes; that must not skip the retry.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_dense_transformer_resident_bytes", lambda self, *a, **k: 0
    )
    monkeypatch.setattr(
        DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: "int8")
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 12_000, companions_mib = 8_000, prequant = True
        ),
    )

    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    seen = []

    def fake_dense_load(self, *a, **k):
        seen.append(a[7] if len(a) > 7 else k.get("transformer_quant"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_into(
        backend,
        tmp_path,
        gguf_filename = "m.gguf",
        base_repo = None,
        family_override = "qwen-image",
        transformer_quant = "auto",
        _transformer_prefetched = False,
    )
    assert seen == ["int8"]


def test_the_resident_retry_declines_a_rung_that_does_not_plan_resident(
    fake_runtime, tmp_path, monkeypatch
):
    # Existence is not fit: an int8 checkpoint can outweigh the Q4 GGUF, so the rung is replanned.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_dense_transformer_resident_bytes", lambda self, *a, **k: 0
    )
    monkeypatch.setattr(
        DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: "int8")
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 900_000, companions_mib = 8_000, prequant = True
        ),
    )

    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "none")
        return dataclasses.replace(real, offload_policy = "model")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    attempted = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_load_dense_quant_pipeline",
        lambda self, *a, **k: attempted.append(True),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    _load_into(
        backend,
        tmp_path,
        gguf_filename = "m.gguf",
        base_repo = None,
        family_override = "qwen-image",
        transformer_quant = "auto",
        _transformer_prefetched = False,
    )
    assert attempted == []


def test_an_auto_pick_that_retried_a_lower_rung_is_still_badged_auto():
    # build_resolved_record treats anything but None/""/"auto" as explicit, so pass the original.
    from core.inference.diffusion_auto_policy import build_resolved_record

    record = build_resolved_record({"transformer_quant": ("auto", "int8", "retried")})
    assert record["transformer_quant"]["source"] == "auto"
    assert record["transformer_quant"]["value"] == "int8"
    explicit = build_resolved_record({"transformer_quant": ("int8", "int8", "requested")})
    assert explicit["transformer_quant"]["source"] == "explicit"


def test_download_plan_skips_files_already_in_the_cache(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(lambda repo_id, files, revision = None, declared_sizes = None: set(files)),
    )

    plan = _flux_download_plan()

    assert plan["entries"] == []
    assert plan["total_bytes"] == 0


def test_download_plan_stages_only_what_the_cache_is_missing(monkeypatch):
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(
            lambda repo_id, files, revision = None, declared_sizes = None: (
                set() if repo_id.endswith("-GGUF") else set(files)
            )
        ),
    )

    plan = _flux_download_plan()

    assert [e["repo_id"] for e in plan["entries"]] == ["unsloth/FLUX.1-dev-GGUF"]
    assert plan["entries"][0]["files"] == ["flux1-dev-Q4_K_M.gguf"]
    assert plan["total_bytes"] == 7 * GB


def _seed_cache_file(
    root,
    repo_id,
    filename,
    sha,
    *,
    size = None,
    dangling = False,
):
    """Write ``filename`` into a real HF cache layout at ``sha``, refs/main pointing there."""
    import os as _os

    repo = root / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents = True, exist_ok = True)
    (repo / "refs" / "main").write_text(sha)
    target = repo / "snapshots" / sha / filename
    target.parent.mkdir(parents = True, exist_ok = True)
    if dangling:
        # A snapshot entry is a symlink into blobs/; a pruned blob leaves the link but no bytes.
        try:
            _os.symlink(repo / "blobs" / "gone", target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this host")
    else:
        target.write_bytes(b"x")
        if size is not None:
            with target.open("r+b") as handle:
                handle.truncate(size)
    return target


def _two_cache_roots(monkeypatch, tmp_path):
    """(live, other) roots wired up the way a mid-session cache-folder change leaves them."""
    from huggingface_hub import constants as hf_constants

    live = tmp_path / "live"
    other = tmp_path / "other"
    live.mkdir()
    other.mkdir()
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(other))
    return live, other


def test_files_already_cached_needs_a_revision_to_drop_anything(monkeypatch, tmp_path):
    """No commit, no verdict: try_to_load_from_cache would otherwise resolve the cache's OWN
    refs/main, stale on a republished repo, and the loader would pull the new blob outside the
    panel."""
    live, _other = _two_cache_roots(monkeypatch, tmp_path)
    sha = "a" * 40
    _seed_cache_file(live, "unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", sha)

    probe = DiffusionBackend._files_already_cached
    assert probe("unsloth/FLUX.1-dev-GGUF", ["flux1-dev-Q4_K_M.gguf"]) == set()
    assert probe("unsloth/FLUX.1-dev-GGUF", ["flux1-dev-Q4_K_M.gguf"], sha) == {
        "flux1-dev-Q4_K_M.gguf"
    }


def test_files_already_cached_ignores_a_superseded_revision(monkeypatch, tmp_path):
    """The blob is on disk under the old commit and refs/main still names it, but the plan asked
    about the commit the Hub just reported, so the file stays staged."""
    live, _other = _two_cache_roots(monkeypatch, tmp_path)
    _seed_cache_file(live, "unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", "a" * 40)

    assert (
        DiffusionBackend._files_already_cached(
            "unsloth/FLUX.1-dev-GGUF", ["flux1-dev-Q4_K_M.gguf"], "b" * 40
        )
        == set()
    )


def test_files_already_cached_skips_unusable_hits(monkeypatch, tmp_path):
    """A dangling symlink is a path but not usable bytes, and an absent file is simply missing, so
    neither can complete the live root's set."""
    live, _other = _two_cache_roots(monkeypatch, tmp_path)
    sha = "c" * 40
    repo = "black-forest-labs/FLUX.1-dev"
    _seed_cache_file(live, repo, "model_index.json", sha)
    _seed_cache_file(live, repo, "vae/diffusion_pytorch_model.safetensors", sha)
    _seed_cache_file(live, repo, "text_encoder/model.safetensors", sha, dangling = True)

    probe = DiffusionBackend._files_already_cached
    files = ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]
    assert probe(repo, files, sha, {name: 1 for name in files}) == set(files)
    assert probe(repo, files, sha, {files[0]: 1, files[1]: 2}) == set()
    assert probe(repo, [*files, "text_encoder/model.safetensors"], sha) == set()
    assert probe(repo, [*files, "scheduler/scheduler_config.json"], sha) == set()


def test_files_already_cached_takes_the_whole_set_from_the_fallback_root(monkeypatch, tmp_path):
    """The other root still counts, as a WHOLE: every diffusion fetch passes reuse_other_cache_root,
    so _prefetch_files resolves every file there and hands from_pretrained that snapshot."""
    _live, other = _two_cache_roots(monkeypatch, tmp_path)
    sha = "c" * 40
    repo = "black-forest-labs/FLUX.1-dev"
    files = ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]
    for name in files:
        _seed_cache_file(other, repo, name, sha)

    assert DiffusionBackend._files_already_cached(repo, files, sha) == set(files)


def test_files_already_cached_refuses_a_set_split_over_two_roots(monkeypatch, tmp_path):
    """Never a per-file union. Neither root holds a complete snapshot, so _prefetch_files returns
    None instead of a snapshot dir and from_pretrained falls back to the hub id pinned to
    hub_cache_dir(), which cannot see the fallback root: calling this repo cached would refetch
    that root's share inline, outside the Downloads panel's progress and its disk preflight."""
    live, other = _two_cache_roots(monkeypatch, tmp_path)
    sha = "c" * 40
    repo = "black-forest-labs/FLUX.1-dev"
    _seed_cache_file(live, repo, "model_index.json", sha)
    _seed_cache_file(other, repo, "vae/diffusion_pytorch_model.safetensors", sha)
    files = ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]

    assert DiffusionBackend._files_already_cached(repo, files, sha) == set()
    _seed_cache_file(live, repo, "vae/diffusion_pytorch_model.safetensors", sha)
    assert DiffusionBackend._files_already_cached(repo, files, sha) == set(files)


def test_download_plan_stages_a_repo_split_across_two_cache_roots(monkeypatch, tmp_path):
    """End to end: a base repo half in the live root and half in the import-time one keeps its row.
    Dropping it tells the panel there is nothing to fetch and the load pulls the rest itself."""
    live, other = _two_cache_roots(monkeypatch, tmp_path)
    base_sha = "9" * 40
    base = "black-forest-labs/FLUX.1-dev"
    _fake_hf_api_with_shas(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": ([_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)], "8" * 40),
            base: (_FLUX_BASE_SIBLINGS, base_sha),
        },
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: base)
    _no_dense_prefetch(monkeypatch)
    _all_cached(monkeypatch)
    staged = [
        name
        for name in _FLUX_BASE_SIBLINGS_BY_NAME
        if _base_file_downloaded(name, include_transformer = False)
    ]
    assert len(staged) > 1
    _seed_cache_file(live, base, staged[0], base_sha)
    for name in staged[1:]:
        _seed_cache_file(other, base, name, base_sha)

    plan = _flux_download_plan()

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert base in by_repo, "a split base repo must keep its row"
    assert by_repo[base]["files"] == staged
    assert by_repo[base]["bytes"] == sum(_FLUX_BASE_SIBLINGS_BY_NAME[n] for n in staged)
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_drops_a_repo_the_fallback_root_holds_whole(monkeypatch, tmp_path):
    """The other half of the rule, so the split guard cannot become "always stage": a repo left
    complete in the import-time root after a cache-folder change still leaves the plan."""
    _live, other = _two_cache_roots(monkeypatch, tmp_path)
    base_sha, gguf_sha = "9" * 40, "8" * 40
    base = "black-forest-labs/FLUX.1-dev"
    _fake_hf_api_with_shas(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": ([_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)], gguf_sha),
            base: (_FLUX_BASE_SIBLINGS, base_sha),
        },
    )
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: base)
    _no_dense_prefetch(monkeypatch)
    _all_cached(monkeypatch)
    for name, size in _FLUX_BASE_SIBLINGS_BY_NAME.items():
        if _base_file_downloaded(name, include_transformer = False):
            _seed_cache_file(other, base, name, base_sha, size = size)
    _seed_cache_file(
        other,
        "unsloth/FLUX.1-dev-GGUF",
        "flux1-dev-Q4_K_M.gguf",
        gguf_sha,
        size = 7 * GB,
    )

    plan = _flux_download_plan()

    assert plan["entries"] == []
    assert plan["total_bytes"] == 0
    assert plan["incompatible_reason"] is None
    assert plan["required_bytes"] > 0
    assert plan["checkpoint_bytes"] == 7 * GB


def test_files_already_cached_survives_an_unreadable_root(monkeypatch, tmp_path):
    """A cache we cannot read is not a verdict: the first root raising must not lose the second
    root's hit, nor abort the files after it."""
    live, other = _two_cache_roots(monkeypatch, tmp_path)
    sha = "d" * 40
    _seed_cache_file(other, "unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf", sha)
    import huggingface_hub

    real = huggingface_hub.try_to_load_from_cache

    def _boom(
        repo_id,
        filename,
        cache_dir = None,
        **kwargs,
    ):
        if str(cache_dir) == str(live):
            raise OSError("unreadable")
        return real(repo_id, filename, cache_dir = cache_dir, **kwargs)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _boom)

    assert DiffusionBackend._files_already_cached(
        "unsloth/FLUX.1-dev-GGUF", ["flux1-dev-Q4_K_M.gguf"], sha
    ) == {"flux1-dev-Q4_K_M.gguf"}


def test_download_plan_stages_a_half_cached_repo_whole(monkeypatch):
    """Dropped only when ALL of it is cached: a shrinking file list would 409 a second pick sharing
    this base, since every diffusion entry rides the one "@diffusion" scope slot and
    download_registry refuses a claim whose scoped_files differ from the live job's."""
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(
            lambda repo_id, files, revision = None, declared_sizes = None: (
                set()
                if repo_id.endswith("-GGUF")
                else {n for n in files if n != "vae/diffusion_pytorch_model.safetensors"}
            )
        ),
    )

    plan = _flux_download_plan()

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert set(by_repo) == {"unsloth/FLUX.1-dev-GGUF", "unsloth/FLUX.1-dev"}
    base = by_repo["unsloth/FLUX.1-dev"]
    assert "vae/diffusion_pytorch_model.safetensors" in base["files"]
    assert "model_index.json" in base["files"]
    assert base["bytes"] == sum(
        size for name, size in _FLUX_BASE_SIBLINGS_BY_NAME.items() if name in base["files"]
    )
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])
    assert all(e["files"] for e in plan["entries"])


def test_download_plan_files_do_not_shrink_as_a_repo_warms(monkeypatch):
    """The scope-slot invariant, stated directly: for one pick, a repo's staged file list is the
    same whether none or some of it is on disk. Only all-cached removes the entry."""

    def _plan(cached_for_base):
        mp = pytest.MonkeyPatch()
        try:
            _fake_hf_api(
                mp,
                {
                    "unsloth/FLUX.1-dev-GGUF": [_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)],
                    "black-forest-labs/FLUX.1-dev": _FLUX_BASE_SIBLINGS,
                },
            )
            mp.setattr(
                "core.inference.diffusion._resolve_base_repo",
                lambda *a, **k: "black-forest-labs/FLUX.1-dev",
            )
            mp.setattr(
                DiffusionBackend,
                "_dense_quant_prefetch_needed",
                lambda self, fam, kwargs, **_kw: False,
            )
            _no_cache(mp)
            mp.setattr(
                DiffusionBackend,
                "_files_already_cached",
                staticmethod(
                    lambda repo_id, files, revision = None, declared_sizes = None: (
                        set() if repo_id.endswith("-GGUF") else set(cached_for_base(files))
                    )
                ),
            )
            return DiffusionBackend().download_plan(
                "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
            )
        finally:
            mp.undo()

    cold = _plan(lambda files: [])
    half = _plan(lambda files: files[:1])
    most = _plan(lambda files: files[:-1])
    warm = _plan(lambda files: files)

    base_files = {e["repo_id"]: e["files"] for e in cold["entries"]}["unsloth/FLUX.1-dev"]
    for plan in (half, most):
        staged = {e["repo_id"]: e["files"] for e in plan["entries"]}
        assert staged["unsloth/FLUX.1-dev"] == base_files
    assert "unsloth/FLUX.1-dev" not in {e["repo_id"] for e in warm["entries"]}


class _ShaInfo(_FakeInfo):
    """A model_info that carries a commit, as the real one does."""

    def __init__(self, siblings, sha):
        super().__init__(siblings)
        self.sha = sha


def _fake_hf_api_with_shas(monkeypatch, repos):
    """_fake_hf_api, but each repo also reports the commit its listing describes."""

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            siblings, sha = repos[repo_id]
            return _ShaInfo(siblings, sha)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())


def test_download_plan_pins_each_probe_to_the_commit_it_just_read(monkeypatch):
    """Each probe gets the sha its own model_info reported, the MIRROR at ITS commit: a mirror is a
    separate repo with its own history, so the vendor's sha would never hit and a cached mirror
    would re-stage in full."""
    gguf_sha, vendor_sha, mirror_sha = "f" * 40, "d" * 40, "e" * 40
    _fake_hf_api_with_shas(
        monkeypatch,
        {
            "unsloth/FLUX.1-dev-GGUF": ([_FakeSibling("flux1-dev-Q4_K_M.gguf", 7 * GB)], gguf_sha),
            "black-forest-labs/FLUX.1-dev": (_FLUX_BASE_SIBLINGS, vendor_sha),
            "unsloth/FLUX.1-dev": (_FLUX_BASE_SIBLINGS, mirror_sha),
        },
    )
    monkeypatch.setattr(
        "core.inference.diffusion._resolve_base_repo",
        lambda *a, **k: "black-forest-labs/FLUX.1-dev",
    )
    _no_dense_prefetch(monkeypatch)
    _no_cache(monkeypatch)
    seen: list = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(
            lambda repo_id, files, revision = None, declared_sizes = None: seen.append(
                (repo_id, revision)
            )
            or set()
        ),
    )

    DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert ("unsloth/FLUX.1-dev-GGUF", gguf_sha) in seen
    assert ("unsloth/FLUX.1-dev", mirror_sha) in seen
    assert vendor_sha not in [rev for _repo, rev in seen]


def test_download_plan_skips_nothing_when_the_hub_reports_no_commit(monkeypatch):
    """An old huggingface_hub, or a listing without a sha, must not fall back to the cache's own
    refs/main: no commit is no verdict, so the pick stages exactly as it did before #8154."""
    _fake_flux_hub(monkeypatch)
    _no_cache(monkeypatch)
    revisions: list = []
    real = DiffusionBackend._files_already_cached

    def _spy(
        repo_id,
        files,
        revision = None,
        declared_sizes = None,
    ):
        revisions.append(revision)
        return real(repo_id, files, revision, declared_sizes)

    monkeypatch.setattr(DiffusionBackend, "_files_already_cached", staticmethod(_spy))

    plan = _flux_download_plan()

    assert revisions and all(rev is None for rev in revisions)
    assert {e["repo_id"] for e in plan["entries"]} == {
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-dev",
    }


# Unified-memory hosts SIGKILL instead of raising OOM (mps disables the high-watermark limit).


def _unified_snapshot(total_gib):
    from core.inference.diffusion_memory import DeviceMemory
    total = total_gib * 1024
    return lambda target: DeviceMemory("mps", "mps", "unified_memory", int(total * 0.80), total)


def _oversized_gguf(
    monkeypatch,
    tmp_path,
    total_gib,
    *,
    resident_mib = 24 * 1024,
):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    monkeypatch.setattr(
        "core.inference.diffusion.settled_snapshot_device_memory", _unified_snapshot(total_gib)
    )
    monkeypatch.setattr(
        "core.inference.diffusion.estimate_gguf_resident_mib", lambda storage: resident_mib
    )
    return DiffusionBackend()


def test_unified_memory_refuses_an_oversized_image_load(fake_runtime, monkeypatch, tmp_path):
    backend = _oversized_gguf(monkeypatch, tmp_path, 16)
    with pytest.raises(RuntimeError) as excinfo:
        _load_into(backend, tmp_path)
    message = str(excinfo.value)
    assert "z-image" in message
    assert "unified memory" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message
    assert backend.status()["loaded"] is False


def test_unified_memory_allows_an_image_load_that_fits(fake_runtime, monkeypatch, tmp_path):
    backend = _oversized_gguf(monkeypatch, tmp_path, 128)
    status = _load_into(backend, tmp_path)
    assert status["loaded"] is True


def test_unified_memory_image_refusal_is_overridable(fake_runtime, monkeypatch, tmp_path):
    from core.inference.diffusion_memory import UNIFIED_OVERSIZE_ENV

    backend = _oversized_gguf(monkeypatch, tmp_path, 16)
    monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, "1")
    status = _load_into(backend, tmp_path)
    assert status["loaded"] is True


def test_discrete_vram_image_load_is_unaffected_by_the_refusal(fake_runtime, monkeypatch, tmp_path):
    from core.inference.diffusion_memory import DeviceMemory

    (tmp_path / "model.gguf").write_bytes(b"weights")
    monkeypatch.setattr(
        "core.inference.diffusion.settled_snapshot_device_memory",
        lambda target: DeviceMemory("cuda", "cuda", "discrete_vram", 13_107, 16_384),
    )
    monkeypatch.setattr(
        "core.inference.diffusion.estimate_gguf_resident_mib", lambda storage: 24 * 1024
    )
    status = DiffusionBackend().load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    assert status["loaded"] is True


def _plan_with_weights(mib):
    from core.inference.diffusion_memory import DeviceMemory, MemoryPlan
    return MemoryPlan(
        requested_mode = "auto",
        offload_policy = "none",
        vae_tiling = False,
        vae_slicing = False,
        device_memory = DeviceMemory("mps", "mps", "unified_memory", 32_768, 65_536),
        estimates = {"model_dense_mib": mib, "safe_device_budget_mib": 24_000},
    )


def test_the_resident_size_table_never_shrinks_a_local_checkpoint(fake_runtime, monkeypatch):
    """The table is keyed on UPSTREAM ids, so a local directory can only reach the coarse family
    entry -- and a family with more than one size under it (a local FLUX.2-klein 9B against
    klein's 4B default) would be re-sized to less than half what it loads, walking straight past
    the refusal into the OS killer. On disk is the measured truth for a local path."""
    import torch

    from core.inference.diffusion_families import detect_family

    target = _mps_target(torch)
    fam = detect_family("black-forest-labs/FLUX.2-klein-9B")
    backend = DiffusionBackend()
    measured = 34_000

    local = str(Path.cwd())
    plan = _plan_with_weights(measured)
    kept = backend._resident_sized_plan(plan, fam, local, target, "pipeline")
    assert kept.estimates["model_dense_mib"] == measured

    lowered = backend._resident_sized_plan(
        plan, fam, "black-forest-labs/FLUX.2-klein-4B", target, "pipeline"
    )
    assert lowered.estimates["model_dense_mib"] < measured


@pytest.mark.parametrize("configured", [True, False])
def test_the_resident_size_table_trusts_only_a_configured_cache_snapshot(
    fake_runtime, tmp_path, monkeypatch, configured
):
    import torch

    cache_root = tmp_path / "hub"
    snapshot = cache_root / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    known = [cache_root] if configured else [tmp_path / "other-hub"]
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: known)
    sized = DiffusionBackend()._resident_sized_plan(
        _plan_with_weights(40_000),
        detect_family("Tongyi-MAI/Z-Image-Turbo"),
        str(snapshot),
        _mps_target(torch),
        "pipeline",
    )
    assert (sized.estimates["model_dense_mib"] < 40_000) is configured


def test_speed_off_is_not_reported_as_a_staging_failure(fake_runtime, tmp_path, monkeypatch):
    """An explicit Speed=off rewrites an auto request to "off" and the plan stages no
    transformer/ on purpose. Reading that expected absence as a decline told the caller their
    automatic quant had failed for want of shards, when what actually happened is the bit-exact
    GGUF they asked for."""
    _stub_hosted_prequant(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = _cuda_backend(tmp_path, monkeypatch)

    status = _load_m(backend, tmp_path, speed_mode = "off", _transformer_prefetched = False)

    assert _dense_calls(calls, backend) == []
    resolved = status.get("resolved", {}).get("transformer_quant", {})
    assert "not staged" not in str(resolved.get("reason") or "")


def test_a_cached_lower_rung_survives_the_unstaged_decline(fake_runtime, tmp_path, monkeypatch):
    """Auto's winner having no hosted prequant does not mean there is none to open. fp8 winning
    while only an int8 checkpoint is published is what the retry below exists for, and declining
    on the winner alone set dense_declined and skipped straight past it to the GGUF for a
    checkpoint already on disk."""
    from core.inference import diffusion as dmod

    def _reason(retry):
        _stub_hosted_prequant(monkeypatch, cached = True)
        monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
        monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
        monkeypatch.setattr(
            DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: retry)
        )
        backend = DiffusionBackend()
        _force_cuda_target(backend, monkeypatch)
        (tmp_path / "m.gguf").write_bytes(b"x")
        status = _load_m(backend, tmp_path, _transformer_prefetched = False)
        return str(status.get("resolved", {}).get("transformer_quant", {}).get("reason") or "")

    marker = "an auto quant never downloads a second transformer"
    assert marker not in _reason("int8")
    assert marker in _reason(None)


def test_the_cache_probe_reads_the_root_the_dense_load_will_use():
    """Tempting to count the import-time root, since _prefetch_files would not re-fetch from it.
    But the consumer of this verdict is the dense fast path, and that calls from_pretrained
    pinned to hub_cache_dir(), so a hit in the other root widens the plan and then downloads the
    whole transformer again after eviction -- the exact outcome the check exists to prevent."""
    import inspect

    from core.inference.diffusion_families import cache_holds_files

    src = inspect.getsource(cache_holds_files)
    assert "other_root" not in src.split('"""')[-1]


def test_variant_hint_carries_both_the_repo_id_and_the_base():
    # The base carries the distilled marker (Tongyi-MAI/Z-Image-Turbo), so keep both ids.
    from core.inference.diffusion import _image_variant_hint
    from core.inference.diffusion_memory import estimate_image_runtime_mib

    hint = _image_variant_hint(
        "z-image", "Z-Image-Q4_K_S.gguf", "unsloth/Z-Image-GGUF", "Tongyi-MAI/Z-Image-Turbo"
    )
    assert "unsloth/Z-Image-GGUF" in hint and "Tongyi-MAI/Z-Image-Turbo" in hint
    assert estimate_image_runtime_mib(width = None, height = None, family = hint) == 6963
    assert estimate_image_runtime_mib(width = None, height = None, family = "") == 8192


def test_variant_hint_is_deduplicated_and_order_stable():
    from core.inference.diffusion import _image_variant_hint

    assert (
        _image_variant_hint("z-image", None, "Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image-Turbo")
        == "z-image Tongyi-MAI/Z-Image-Turbo"
    )
    assert _image_variant_hint("z-image", "  ", None, None) == "z-image"
    assert _image_variant_hint(None, None, None, None) == ""


def _base_snapshot_with_sizes(tmp_path, monkeypatch, sizes):
    """A base repo cached only under the other cache root, with the given ``{relative path: MiB}``."""
    _live, other = _split_cache_roots(tmp_path, monkeypatch)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    for rel, mib in sizes.items():
        path = snapshot / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        with open(path, "wb") as fh:
            fh.truncate(mib * 1024 * 1024)
    return snapshot


def test_text_encoder_cache_bytes_is_a_subset_of_the_companion_total(tmp_path, monkeypatch):
    # Subtracted from the companion total, so it must come off the same walk or go negative.
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 150,
            "text_encoder_2/model.safetensors": 90,
            "vae/diffusion_pytorch_model.safetensors": 50,
            "transformer/diffusion_pytorch_model.safetensors": 4096,
        },
    )
    sizes = DiffusionBackend._local_dir_text_encoder_sizes(snapshot)
    assert sorted(sizes) == ["text_encoder/model.safetensors", "text_encoder_2/model.safetensors"]
    assert DiffusionBackend._text_encoder_cache_bytes(str(snapshot)) == 240 * 1024 * 1024
    assert DiffusionBackend._companion_cache_bytes(str(snapshot)) == 290 * 1024 * 1024


def test_plan_memory_hands_the_planner_the_text_encoder_split(monkeypatch, tmp_path):
    # 150 MiB of encoders inside a 200 MiB companion total, so the streamed-encoder floor is the
    # VAE (50) + headroom (100) + overhead (2048).
    from core.inference.diffusion_memory import OFFLOAD_GROUP

    snapshot = _other_root_base_snapshot(tmp_path, monkeypatch)
    target = _small_card(monkeypatch)

    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        base_local_dir = str(snapshot),
    )
    assert plan.estimates["companion_dense_mib"] == 200
    assert plan.estimates["text_encoder_dense_mib"] == 150
    assert plan.estimates["group_floor_streamed_te_mib"] == 2198
    assert plan.estimates["resident_transformer_floor_mib"] == 2498
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_text_encoders is True and plan.stream_transformer is False


def test_plan_memory_streams_the_text_encoders_instead_of_offloading_everything(
    monkeypatch, tmp_path
):
    # A text encoder that alone busts the group floor; the VAE-only floor of 2198 fits 2952 MiB.
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 2800,
            "vae/diffusion_pytorch_model.safetensors": 50,
        },
    )
    target = _small_card(monkeypatch)
    from core.inference.diffusion_memory import OFFLOAD_GROUP

    def _plan(**kw):
        return DiffusionBackend()._plan_memory(
            target,
            None,
            "bfl/base",
            types.SimpleNamespace(name = "flux.1"),
            None,
            False,
            kind = "gguf",
            transformer_resident_override_mib = 300,
            base_local_dir = str(snapshot),
            **kw,
        )

    plan = _plan()
    # group floor 2850 + 100 + 2048 = 4998, over the 2952 budget: this is the whole-module case.
    assert plan.estimates["group_floor_mib"] == 4998
    assert plan.estimates["group_floor_streamed_te_mib"] == 2198
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders is True


def test_plan_memory_keeps_the_split_on_the_dense_candidate_path(monkeypatch, tmp_path):
    from core.inference.diffusion_memory import OFFLOAD_GROUP

    _base_snapshot_with_sizes(tmp_path, monkeypatch, {})
    target = _small_card(monkeypatch)

    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        companion_override_mib = 2850,
        text_encoder_override_mib = 2800,
    )
    assert plan.estimates["text_encoder_dense_mib"] == 2800
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders is True


def test_dense_quant_estimate_carries_the_text_encoder_share():
    # The override above is only as good as the table it comes from: companions minus text encoders
    # has to be the VAE and nothing else, or the streamed floor is wrong on every dense candidate.
    from core.inference.diffusion_auto_policy import estimate_dense_quant

    estimate = estimate_dense_quant(types.SimpleNamespace(name = "z-image"), "int8")
    assert estimate.steady_transformer_mib == 6451
    assert estimate.companions_mib == 7820
    assert estimate.text_encoders_mib == 7629
    assert estimate.companions_mib - estimate.text_encoders_mib == 191


# Load-time plans budget 1024x1024; on Windows WDDM an overrun spills to system RAM silently,
# so generate() re-checks with the real dimensions.

# The reported card: free 15,870 of 16,305 MiB, so the safe budget is 13,822 MiB.
_ROCM_16G = (15_870, 16_305)


def _loaded_backend_on_a_16g_card(
    tmp_path,
    monkeypatch,
    *,
    base_repo = "base/repo",
):
    """A loaded GGUF pipeline, then a 16 GB discrete-CUDA memory snapshot. Patched AFTER the load
    so the load itself still plans against the fixture's CPU target and is unaffected."""
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import DeviceMemory

    backend = _loaded_backend(tmp_path, base_repo = base_repo)
    free, total = _ROCM_16G
    snapshot = lambda target, **kw: DeviceMemory("cuda", "cuda", "discrete_vram", free, total)
    monkeypatch.setattr(dmod, "settled_snapshot_device_memory", snapshot)
    # The guard reads the RECLAIMABLE snapshot, so pin that one too.
    monkeypatch.setattr(dmod, "reclaimable_snapshot_device_memory", snapshot)
    return backend


def test_generate_refuses_a_resolution_whose_activations_cannot_fit(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)
    message = str(excinfo.value)
    assert "1088x1920" in message
    assert "smaller resolution" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_GENERATE" in message

    assert len(backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 4)["images"]) == 1


def test_generate_guard_measures_the_input_image_not_the_sliders(
    fake_runtime, tmp_path, monkeypatch
):
    # inpaint / upscale / edit take OUTPUT size from the upload, not the width/height kwargs.
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        backend.generate(
            prompt = "a sloth",
            width = 1024,
            height = 1024,
            steps = 4,
            init_image = _png_b64(2048),
            mask_image = _mask_b64(2048),
        )
    message = str(excinfo.value)
    assert "2048x2048" in message
    assert "Upload a smaller source image" in message
    assert "smaller resolution" not in message


def test_transform_fits_the_upload_to_the_sliders_instead_of_refusing(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)

    out = backend.generate(
        prompt = "a sloth",
        width = 1024,
        height = 1024,
        steps = 4,
        init_image = _png_b64(2048),
    )
    assert len(out["images"]) == 1
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (1024, 1024)


def test_generate_guard_uses_the_hint_the_load_planned_with(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend_on_a_16g_card(
        tmp_path, monkeypatch, base_repo = "Tongyi-MAI/Z-Image-Turbo"
    )
    assert "Tongyi-MAI/Z-Image-Turbo" in backend._state.variant_hint

    # 1280^2: 12,800 + 2048 = 14,848 > 13,822 undiscounted; distilled 10,880 + 2048 = 12,928 fits.
    assert len(backend.generate(prompt = "a sloth", width = 1280, height = 1280, steps = 4)["images"]) == 1
    with pytest.raises(ValueError, match = "1088x1920"):
        backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)


def test_generate_guard_fails_open_when_the_probe_raises(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)

    def _boom(target, **kw):
        raise RuntimeError("mem_get_info exploded")

    monkeypatch.setattr(dmod, "reclaimable_snapshot_device_memory", _boom)
    assert len(backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)["images"]) == 1


def test_generate_guard_env_override(fake_runtime, tmp_path, monkeypatch):
    from core.inference.diffusion_memory import OVERSIZED_GENERATE_ENV

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    monkeypatch.setenv(OVERSIZED_GENERATE_ENV, "1")
    assert len(backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)["images"]) == 1


class _TilingVae:
    """A VAE that can tile, like every diffusers AutoencoderKL*: records its saver calls."""

    tile_sample_min_height = 256
    tile_sample_min_width = 256

    def __init__(self) -> None:
        self.use_tiling = False
        self.use_slicing = False
        self.calls: list = []

    def enable_tiling(self):
        self.calls.append("enable_tiling")
        self.use_tiling = True

    def disable_tiling(self):
        self.calls.append("disable_tiling")
        self.use_tiling = False

    def enable_slicing(self):
        self.calls.append("enable_slicing")
        self.use_slicing = True

    def disable_slicing(self):
        self.calls.append("disable_slicing")
        self.use_slicing = False


def _upscale_with_tiling_vae(backend, monkeypatch, **kw):
    """Run a 1024 -> 2048 Upscale whose img2img pipe shares a tiling VAE; returns (vae, tiled_during_call)."""
    from core.inference import diffusion as dmod

    vae = _TilingVae()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: False)
    seen = {}
    real_call = _FakeImg2ImgPipe.__call__

    def _spy(self, **kwargs):
        seen["tiled"] = vae.use_tiling
        seen["sliced"] = vae.use_slicing
        return real_call(self, **kwargs)

    monkeypatch.setattr(_FakeImg2ImgPipe, "__call__", _spy)
    out = backend.generate(
        prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0, **kw
    )
    return out, vae, seen


def test_generate_upscale_that_was_refused_runs_with_the_vae_tiled(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match = "2048x2048"):
        backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0)
    out, vae, seen = _upscale_with_tiling_vae(backend, monkeypatch)
    assert len(out["images"]) == 1
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (2048, 2048)
    assert seen == {"tiled": True, "sliced": True}
    assert not vae.use_tiling and not vae.use_slicing
    assert vae.calls == ["enable_tiling", "enable_slicing", "disable_tiling", "disable_slicing"]


def test_generate_upscale_that_fits_is_not_tiled(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    from core.inference import diffusion as dmod

    vae = _TilingVae()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: False)
    backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(512), upscale = 2.0)
    assert vae.calls == []


def test_generate_upscale_restores_the_vae_when_the_render_fails(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    from core.inference import diffusion as dmod

    vae = _TilingVae()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: False)

    def _boom(self, **kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(_FakeImg2ImgPipe, "__call__", _boom)
    with pytest.raises(RuntimeError, match = "decode failed"):
        backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0)
    assert not vae.use_tiling and not vae.use_slicing


def test_generate_upscale_on_math_only_attention_still_refuses(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", _TilingVae(), raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: True)
    with pytest.raises(ValueError) as excinfo:
        backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0)
    message = str(excinfo.value)
    assert "Upload a smaller source image" in message
    assert "Allow oversized generations" in message


class _UnslicedTilingVae(_TilingVae):
    """Tiles at 1024 (AutoencoderKL) but its enable_slicing() leaves slicing off."""

    tile_sample_min_height = 1024
    tile_sample_min_width = 1024

    def enable_slicing(self):
        self.calls.append("enable_slicing")


@pytest.mark.parametrize("vae_cls", [_TilingVae, _UnslicedTilingVae])
def test_generate_windows_batch_prices_tiles_at_the_batch_without_slicing(
    fake_runtime, tmp_path, monkeypatch, vae_cls
):
    from core.inference import diffusion as dmod

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    vae = vae_cls()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: False)
    monkeypatch.setattr(dmod.sys, "platform", "win32")
    kw = dict(prompt = "a sloth", steps = 4, init_image = _png_b64(1024), upscale = 2.0, seeds = [1, 2])
    if vae_cls is _TilingVae:
        assert len(backend.generate(**kw)["images"]) == 2
    else:
        with pytest.raises(ValueError, match = "2048x2048 at a batch of 2"):
            backend.generate(**kw)
    assert not vae.use_tiling and not vae.use_slicing


class _BrokenTilingVae(_TilingVae):
    """A VAE that claims to tile but whose enable_tiling() raises."""

    def enable_tiling(self):
        self.calls.append("enable_tiling")
        raise RuntimeError("tiling unsupported")


def test_generate_upscale_refuses_when_the_vae_tiling_does_not_engage(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    vae = _BrokenTilingVae()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    monkeypatch.setattr(dmod, "_quadratic_attention", lambda target, backend = None: False)
    calls = []
    real_call = _FakeImg2ImgPipe.__call__

    def _spy(self, **kwargs):
        calls.append(kwargs)
        return real_call(self, **kwargs)

    monkeypatch.setattr(_FakeImg2ImgPipe, "__call__", _spy)
    with pytest.raises(ValueError) as excinfo:
        backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0)
    assert "2048x2048" in str(excinfo.value)
    assert "even with tiled VAE decoding" not in str(excinfo.value)
    assert calls == []
    assert not vae.use_slicing and not vae.use_tiling
    out = backend.generate(
        prompt = "a sloth",
        steps = 4,
        seed = 1,
        init_image = _png_b64(1024),
        upscale = 2.0,
        allow_oversized = True,
    )
    assert len(out["images"]) == 1


def test_quadratic_attention_follows_the_engaged_backend(monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "sdpa_math_only", lambda target: True)
    assert dmod._quadratic_attention(object()) is True
    assert dmod._quadratic_attention(object(), None) is True
    assert dmod._quadratic_attention(object(), "native") is True
    for engaged in ("aiter", "sage", "xformers", "_native_cudnn", "flash"):
        assert dmod._quadratic_attention(object(), engaged) is False


def test_generate_upscale_with_an_engaged_backend_on_a_math_only_device_runs_tiled(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    monkeypatch.setattr(dmod, "sdpa_math_only", lambda target: True)
    vae = _TilingVae()
    monkeypatch.setattr(_FakeImg2ImgPipe, "vae", vae, raising = False)
    with pytest.raises(ValueError, match = "2048x2048"):
        backend.generate(prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0)
    object.__setattr__(backend._state, "attention_backend", "aiter")
    out = backend.generate(
        prompt = "a sloth", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 2.0
    )
    assert len(out["images"]) == 1
    assert vae.calls[:2] == ["enable_tiling", "enable_slicing"]


def test_generate_allow_oversized_runs_a_refused_request(fake_runtime, tmp_path, monkeypatch):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    out = backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4, allow_oversized = True)
    assert len(out["images"]) == 1


def test_generate_guard_leaves_a_large_batch_to_the_oom_backoff(
    fake_runtime, tmp_path, monkeypatch
):
    # The chunk loop halves failed batches to singletons, so the guard budgets one image.
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    out = backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 4, batch_size = 8)
    assert len(out["images"]) == 8


def test_dense_quant_candidate_replan_prices_the_streamed_encoder_tier(
    fake_runtime, tmp_path, monkeypatch
):
    # This path uses the family table's overrides, so the text-encoder share must be threaded too.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 6_451,
            companions_mib = 7_820,
            text_encoders_mib = 7_629,
            prequant = True,
        ),
    )
    seen = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        seen.append(k.get("text_encoder_override_mib"))
        return dataclasses.replace(real, offload_policy = "model")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    monkeypatch.setattr(
        DiffusionBackend,
        "_load_dense_quant_pipeline",
        lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("test: stop after reaching the fast path")
        ),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "transformer_quant='int8'"):
        _load_m(backend, tmp_path, transformer_quant = "int8")
    assert seen and all(value == 7_629 for value in seen)


def test_the_activation_guard_budgets_the_real_batch_on_windows(monkeypatch):
    """The singleton floor rests on the OOM backoff halving a failed forward, and under WDDM
    there is no OOM to catch: the driver serves the overflow from system RAM, the desktop stops
    responding and nothing recovers it. So Windows budgets the largest chunk it will actually
    run, while every other platform keeps the batch-32 fast path it measures today."""
    from core.inference import diffusion as dmod

    chunks = [[object()] * 8, [object()] * 3]
    monkeypatch.setattr(dmod.sys, "platform", "linux")
    assert dmod._activation_guard_batch(chunks) == 1
    assert dmod._activation_guard_batch([]) == 1
    monkeypatch.setattr(dmod.sys, "platform", "win32")
    assert dmod._activation_guard_batch(chunks) == 8
    assert dmod._activation_guard_batch([[object()]]) == 1
    assert dmod._activation_guard_batch([]) == 1


# All five UI surfaces funnel through generate(), so cancel is covered for each workflow.


def _stepping_call(record):
    """A pipeline __call__ that actually steps, so a cancel can be observed mid-denoise.

    The fake pipes return immediately, which cannot distinguish "the sampler stopped" from
    "the sampler finished". This mirrors diffusers: invoke callback_on_step_end each step and
    break out when the callback sets ``_interrupt``, exactly as the real denoise loop does."""

    def _call(
        self,
        *,
        callback_on_step_end = None,
        **kwargs,
    ):
        record["steps_run"] = 0
        self._interrupt = False
        for index in range(record["total_steps"]):
            if callback_on_step_end is not None:
                callback_on_step_end(self, index, index, {})
            record["steps_run"] = index + 1
            record["reached"].set()
            if getattr(self, "_interrupt", False):
                break
            record["resume"].wait(5)
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])

    return _call


@pytest.mark.parametrize(
    "surface,gen_kwargs",
    [
        ("create", {}),
        ("transform", {"init_image": _tiny_png_b64(), "strength": 0.5}),
        # Extend (outpaint) sends the padded canvas down the same inpaint path.
        ("inpaint", {"init_image": _tiny_png_b64(), "mask_image": _mask_b64(64)}),
        ("extend", {"init_image": _tiny_png_b64(), "mask_image": _mask_b64(64)}),
        ("upscale", {"init_image": _tiny_png_b64(), "upscale": 2.0}),
    ],
)
def test_cancel_generate_stops_every_workflow(
    fake_runtime, tmp_path, monkeypatch, surface, gen_kwargs
):
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)

    record = {
        "total_steps": 8,
        "steps_run": 0,
        "reached": threading.Event(),
        "resume": threading.Event(),
    }
    stepping = _stepping_call(record)
    for cls in (_FakePipe, _FakeImg2ImgPipe, _FakeInpaintPipe):
        monkeypatch.setattr(cls, "__call__", stepping)

    assert backend.cancel_generate() is False

    outcome: dict = {}

    def _run():
        try:
            outcome["result"] = backend.generate(
                prompt = "a sloth", steps = record["total_steps"], **gen_kwargs
            )
        except BaseException as exc:  # noqa: BLE001 -- the test asserts on the exact type/text
            outcome["error"] = exc

    worker = threading.Thread(target = _run, daemon = True)
    worker.start()
    assert record["reached"].wait(5), f"{surface}: the denoise never started"

    assert backend.cancel_generate() is True
    record["resume"].set()
    worker.join(10)
    assert not worker.is_alive(), f"{surface}: the denoise did not unwind"

    assert "result" not in outcome, f"{surface}: a cancelled run still produced images"
    assert isinstance(outcome["error"], RuntimeError)
    assert str(outcome["error"]) == DIFFUSION_CANCELLED_MSG
    assert record["steps_run"] < record["total_steps"], (
        f"{surface}: ran {record['steps_run']}/{record['total_steps']} steps, so the cancel "
        "never reached the sampler"
    )
    assert backend.generate_progress()["active"] is False
    assert backend.cancel_generate() is False


def test_cancel_generate_lands_at_the_next_step_boundary(fake_runtime, tmp_path, monkeypatch):
    # Best-effort at the NEXT step boundary: a cancel during step 1 must not let step 3 run.
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)

    seen: list[int] = []

    def _call(
        self,
        *,
        callback_on_step_end = None,
        **kwargs,
    ):
        self._interrupt = False
        for index in range(20):
            if callback_on_step_end is not None:
                callback_on_step_end(self, index, index, {})
            seen.append(index)
            if getattr(self, "_interrupt", False):
                break
            if index == 1:
                backend.cancel_generate()
        return types.SimpleNamespace(images = [_FakeImage()])

    monkeypatch.setattr(_FakePipe, "__call__", _call)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "x", steps = 20)
    assert seen == [0, 1, 2]


def test_cancel_generate_during_the_post_denoise_save_still_cancels(
    fake_runtime, tmp_path, monkeypatch
):
    # Stop during the compile-cache handoff must unwind as cancelled, not return images.
    from core.inference import diffusion_compile_cache as compile_cache

    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)

    def _save(ctx, logger = None):
        assert backend.cancel_generate() is True

    monkeypatch.setattr(compile_cache, "save_async", _save)

    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate(prompt = "x", steps = 2)


def test_a_completed_generation_stops_advertising_itself_as_cancellable(
    fake_runtime, tmp_path, monkeypatch
):
    # Final check and deregistration share cancel_generate's lock, so Stop cannot answer true then lose.
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = _loaded_backend(tmp_path)

    from core.inference import diffusion as diffusion_module

    seen: list[bool] = []
    real_baked = diffusion_module._baked_lora_names

    def _baked(pipe):
        seen.append(backend.cancel_generate())
        return real_baked(pipe)

    monkeypatch.setattr(diffusion_module, "_baked_lora_names", _baked)

    out = backend.generate(prompt = "x", steps = 2)
    assert out["images"]
    assert seen == [False]


def test_cancel_generate_is_a_no_op_without_a_load(fake_runtime):
    assert DiffusionBackend().cancel_generate() is False


def test_unified_memory_declines_a_prequant_that_outweighs_the_gguf(
    fake_runtime, monkeypatch, tmp_path
):
    """A GGUF pick on unified memory can still be upsized by the dense fast path: the hosted
    fp8/int8 artifact is roughly 0.55x bf16 against a Q4's ~0.3x, so it can be twice the file that
    just passed the load-level refusal. The planner returns 'none' for any size on unified memory,
    so the OFFLOAD_NONE gate cannot catch that, and the prequant path skips the dense-size check
    (it never builds dense). Without an explicit size the load materialises it and is OS-killed."""
    from core.inference import diffusion as dmod
    from core.inference.diffusion_auto_policy import DenseQuantEstimate

    backend = _oversized_gguf(monkeypatch, tmp_path, 32, resident_mib = 8 * 1024)
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **kw: "unsloth/Z-Image-FP8")
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: DenseQuantEstimate(
            scheme = "fp8",
            steady_transformer_mib = 40 * 1024,
            transient_transformer_mib = 40 * 1024,
            companions_mib = 2 * 1024,
            prequant = True,
        ),
    )

    status = _load_into(backend, tmp_path, base_repo = None, model_kind = "gguf")
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "off"
    assert "unified memory" in (resolved["reason"] or "")


def test_unified_memory_keeps_a_prequant_that_fits(fake_runtime, monkeypatch, tmp_path):
    from core.inference import diffusion as dmod
    from core.inference.diffusion_auto_policy import DenseQuantEstimate

    backend = _oversized_gguf(monkeypatch, tmp_path, 32, resident_mib = 8 * 1024)
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda *a, **kw: "unsloth/Z-Image-FP8")
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: DenseQuantEstimate(
            scheme = "fp8",
            steady_transformer_mib = 6 * 1024,
            transient_transformer_mib = 6 * 1024,
            companions_mib = 2 * 1024,
            prequant = True,
        ),
    )
    calls: list = []

    def _record(self, *a, **kw):
        calls.append("built")
        # Raising here keeps the stub out of pipeline assembly; the loader's own handler falls
        # back to the GGUF, and reaching this line at all is the assertion.
        raise RuntimeError("stub")

    monkeypatch.setattr(dmod.DiffusionBackend, "_load_dense_quant_pipeline", _record)

    _load_into(backend, tmp_path, base_repo = None, model_kind = "gguf")
    assert calls == ["built"], "a prequant that fits must still reach the dense fast path"


def test_the_resident_size_table_prices_a_pre_cast_encoder_at_its_real_size(
    fake_runtime, monkeypatch
):
    """The table's encoder term is the dense one, and a pick that takes its encoder PRE-CAST from a
    hosted fp8 checkpoint loads roughly 0.65x of it. Budgeting the dense figure against a hard
    refusal turns tens of GB the pipeline never materialises into a rejected load."""
    import torch

    from core.inference import diffusion as dmod
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_te_prequant import TE_PREQUANT_BUDGET_SCALE

    target = _mps_target(torch)
    fam = detect_family("Tongyi-MAI/Z-Image-Turbo")
    base = "Tongyi-MAI/Z-Image-Turbo"
    backend = DiffusionBackend()
    plan = _plan_with_weights(200_000)

    dense = backend._resident_sized_plan(plan, fam, base, target, "pipeline")
    monkeypatch.setattr(dmod, "family_bf16_components_gb", dmod.family_bf16_components_gb)
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_sources",
        lambda fam, te_quant_mode = None, target = None, **_kwargs: {"text_encoder": object()},
    )
    precast = backend._resident_sized_plan(
        plan, fam, base, target, "pipeline", text_encoder_quant = "fp8"
    )
    dense_mib = dense.estimates["model_dense_mib"]
    precast_mib = precast.estimates["model_dense_mib"]
    assert precast_mib < dense_mib, "a pre-cast encoder must lower the refusal's weight term"
    transformer_gb, encoders_gb, _vae = dmod.family_bf16_components_gb(fam, base)
    saved_gb = (dense_mib - precast_mib) * (1024.0 * 1024.0) / (1000.0**3)
    assert saved_gb == pytest.approx(encoders_gb * (1.0 - TE_PREQUANT_BUDGET_SCALE), rel = 0.02)


def test_the_resident_size_table_never_shrinks_an_unrecognised_remote_variant(fake_runtime):
    """Same hole as the local-path one, reached from the Hub: a fine-tune or a renamed mirror that
    the family detector still matches by name is NOT an exact key in the size table, so it falls
    through to the family entry -- and for a family carrying two sizes that entry is the smaller
    one. A 9B derivative lowered to the 4B number walks straight past the refusal."""
    import torch

    from core.inference.diffusion_families import detect_family

    target = _mps_target(torch)
    fam = detect_family("black-forest-labs/FLUX.2-klein-9B")
    backend = DiffusionBackend()
    measured = 34_000
    plan = _plan_with_weights(measured)

    kept = backend._resident_sized_plan(
        plan, fam, "someone/FLUX.2-klein-9B-anime-tune", target, "pipeline"
    )
    assert kept.estimates["model_dense_mib"] == measured
    override = backend._resident_sized_plan(
        plan, fam, "black-forest-labs/FLUX.2-klein-9B", target, "pipeline"
    )
    assert override.estimates["model_dense_mib"] < measured
    default = backend._resident_sized_plan(plan, fam, fam.base_repo, target, "pipeline")
    assert default.estimates["model_dense_mib"] < measured


def test_a_whole_pipeline_single_file_is_not_charged_for_cached_companions(fake_runtime):
    """An SDXL-style single file carries the U-Net, VAE and text encoders itself and the base repo
    is read for config only, but the plan still adds the base's cached companion weights. As an
    offload hint that is conservative; as a hard refusal it rejects a checkpoint that fits, and
    only for users who happen to have loaded the full pipeline before."""
    import torch

    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_memory import DeviceMemory, MemoryPlan

    target = _mps_target(torch)
    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    assert fam.single_file_is_pipeline, "this test is about the SDXL-shaped families"
    plan = MemoryPlan(
        requested_mode = "auto",
        offload_policy = "none",
        vae_tiling = False,
        vae_slicing = False,
        device_memory = DeviceMemory("mps", "mps", "unified_memory", 32_768, 65_536),
        estimates = {
            "model_dense_mib": 14_000,
            "companion_dense_mib": 7_000,
            "safe_device_budget_mib": 10_000,
        },
    )
    sized = DiffusionBackend()._resident_sized_plan(
        plan, fam, "stabilityai/stable-diffusion-xl-base-1.0", target, "single_file"
    )
    assert sized.estimates["model_dense_mib"] == 7_000


def test_the_prequant_fit_check_prices_a_pre_cast_text_encoder(fake_runtime, monkeypatch):
    """``DenseQuantEstimate.companions_mib`` is always the DENSE encoder plus the VAE, but the
    assembly this check is sizing is handed ``text_encoder_quant`` and injects the pre-cast
    encoder. Refusing on the dense figure declines a prequant that fits on bytes never
    materialised -- for FLUX.2-dev's Mistral-24B that is tens of GB. The load-level resident plan
    already applies te_prequant_budget_scale; this is the same scale on the same estimate."""
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import detect_family

    fam = detect_family("black-forest-labs/FLUX.2-dev")
    assert fam is not None and fam.te_prequant_repos, "the fixture family lost its pre-cast repo"
    encoders = 48_000
    candidate = types.SimpleNamespace(companions_mib = encoders + 400, text_encoders_mib = encoders)

    base = "black-forest-labs/FLUX.2-dev"
    seen: dict = {}

    def _scale(_fam, *, te_quant_mode, target, base):
        seen.update(mode = te_quant_mode, target = target, base = base)
        return 0.5 if te_quant_mode == "fp8" else 1.0

    monkeypatch.setattr("core.inference.diffusion_te_prequant.te_prequant_budget_scale", _scale)
    scaled = DiffusionBackend._precast_scaled_companions_mib(candidate, fam, base, object(), "fp8")
    assert scaled == 24_400
    assert seen["mode"] == "fp8"
    assert seen["base"] == base

    assert (
        DiffusionBackend._precast_scaled_companions_mib(candidate, fam, base, object(), None)
        == candidate.companions_mib
    )
    no_split = types.SimpleNamespace(companions_mib = 1234, text_encoders_mib = 0)
    assert (
        DiffusionBackend._precast_scaled_companions_mib(no_split, fam, base, object(), "fp8")
        == 1234
    )
    empty = types.SimpleNamespace(companions_mib = None)
    assert (
        DiffusionBackend._precast_scaled_companions_mib(empty, fam, base, object(), "fp8") is None
    )


def test_an_offload_memory_request_is_not_reported_as_unstaged_shards(
    fake_runtime, tmp_path, monkeypatch
):
    """balanced, low_vram and the legacy cpu_offload flag name their policy outright, so the plan
    omits transformer/ because the dense build is skipped, not because bytes were missing.
    Reporting a second-denoiser refusal told the caller the wrong thing about their own setting.
    """
    marker = "an auto quant never downloads a second transformer"
    for request in (
        {"memory_mode": "balanced"},
        {"memory_mode": "low_vram"},
        {"cpu_offload": True},
    ):
        _stub_hosted_prequant(monkeypatch, cached = True)
        backend = DiffusionBackend()
        _force_cuda_target(backend, monkeypatch)
        (tmp_path / "m.gguf").write_bytes(b"x")
        status = _load_m(backend, tmp_path, _transformer_prefetched = False, **request)
        reason = str(status.get("resolved", {}).get("transformer_quant", {}).get("reason") or "")
        assert marker not in reason, request


def test_an_unsupported_host_is_not_told_its_shards_are_unstaged(
    fake_runtime, tmp_path, monkeypatch
):
    """The unsupported-device checks above the decline run for an EXPLICIT scheme only, so on
    CPU/MPS, non-bf16 CUDA or a stubbed torchao an AUTO request reached the unstaged-shards branch
    and the badge told the user the base transformer/ shards were not staged. True and irrelevant:
    caching them cannot enable a quant this host cannot run. The load itself is unchanged -- the
    dense re-plan is already gated on dense_transformer_supported -- so what this pins is the
    reason, on the commonest path there is (every Mac and CPU GGUF load)."""
    from core.inference import diffusion as dmod

    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = False)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: False)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = _load_m(backend, tmp_path, _transformer_prefetched = False)

    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    reason = ((status.get("resolved") or {}).get("transformer_quant") or {}).get("reason") or ""
    assert "shards are not staged" not in reason, reason

    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: None
    )
    backend2 = DiffusionBackend()
    _force_cuda_target(backend2, monkeypatch)
    status2 = backend2.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )
    reason2 = ((status2.get("resolved") or {}).get("transformer_quant") or {}).get("reason") or ""
    assert "shards are not staged" not in reason2, reason2

    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: "fp8"
    )
    _stub_hosted_prequant(monkeypatch, cached = False)
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    backend3 = DiffusionBackend()
    _force_cuda_target(backend3, monkeypatch)
    status3 = backend3.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )
    reason3 = ((status3.get("resolved") or {}).get("transformer_quant") or {}).get("reason") or ""
    assert "shards are not staged" in reason3, reason3


def test_generation_in_flight_tracks_a_generation(fake_runtime, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import core.inference.diffusion as diffusion_mod

    backend = _loaded_backend(tmp_path)
    monkeypatch.setattr(diffusion_mod, "_diffusion_backend", backend)

    seen = {}

    def fake_apply(self, state, loras, cancel):
        seen["in_flight"] = diffusion_mod.generation_in_flight()

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", fake_apply)

    assert diffusion_mod.generation_in_flight() is False
    backend.generate(prompt = "a sloth", steps = 4)
    assert (
        seen["in_flight"] is True
    ), "liveness cannot tell this backend from a dead one while it renders an image"
    assert diffusion_mod.generation_in_flight() is False


def test_generation_in_flight_never_builds_a_backend(fake_runtime, monkeypatch):
    import core.inference.diffusion as diffusion_mod

    monkeypatch.setattr(diffusion_mod, "_diffusion_backend", None)
    monkeypatch.setattr(
        diffusion_mod,
        "DiffusionBackend",
        lambda *a, **k: pytest.fail("liveness constructed a diffusion backend"),
    )
    assert diffusion_mod.generation_in_flight() is False


def test_the_download_plan_resolves_the_same_nvfp4_rung_the_load_does(monkeypatch):
    from types import SimpleNamespace

    import core.inference.diffusion as dmod
    from core.inference import diffusion_nvfp4_ops as ops
    from core.inference import diffusion_transformer_quant as tq

    fam = detect_family("black-forest-labs/FLUX.1-schnell")
    assert fam is not None and fam.name == "flux.1"
    base = "black-forest-labs/FLUX.1-schnell"

    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(tq, "_capability", lambda: (10, 0))
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda device = None: False)
    monkeypatch.setattr(
        tq,
        "_scheme_supported",
        lambda scheme, device, unproven_ok = False: scheme not in ("int8", "fp8"),
    )
    monkeypatch.setattr(dmod, "prequant_checkpoint_cached", lambda source, **kw: False)
    monkeypatch.setattr(ops, "select_nvfp4_backend", lambda device = None: "flashinfer")

    target = SimpleNamespace(device = "cuda", dtype = None)
    assert (
        dmod._planned_quant_scheme(fam, target, "auto", base_repo = base, prequant_path = None)
        == "nvfp4"
    )
    assert (
        dmod._uncached_prequant_repo(fam, target, "auto", base_repo = base, prequant_path = None)
        == "unsloth/FLUX.1-schnell-NVFP4"
    )


class _FakeDenoiser:
    """A denoiser whose parameters expose a dtype to the quantisation gate."""

    def __init__(self, dtype = "torch.bfloat16") -> None:
        self._params = [types.SimpleNamespace(dtype = dtype)]

    def parameters(self, recurse = True):
        return iter(self._params)


def _init_with_denoiser(dtype):
    """A ``_FakePipe.__init__`` whose transformer reports ``dtype``, for the gate's dtype walk."""
    real_init = _FakePipe.__init__

    def _init(self):
        real_init(self)
        self.transformer = _FakeDenoiser(dtype)

    return _init


def _stub_pipeline_dense_quant(
    backend,
    monkeypatch,
    *,
    engages = "fp8",
    denoisers = ("transformer",),
):
    """Stub a capable CUDA host and record transformer quantisation calls."""
    from core.inference import diffusion as dmod

    calls: list = []
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **kw: "fp8"
    )
    real_init = _FakePipe.__init__

    def _init(self):
        real_init(self)
        for attr in denoisers:
            setattr(self, attr, _FakeDenoiser())

    monkeypatch.setattr(_FakePipe, "__init__", _init)

    def _quantize(pipe, target, **kwargs):
        calls.append({"pipe": pipe, "transformer": pipe.transformer, **kwargs})
        return engages

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    return calls


def test_a_pipeline_pick_quantises_its_transformer_in_place(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] == "fp8"
    assert len(calls) == 1
    assert calls[0]["transformer"] is backend._state.pipe.transformer
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "fp8"
    assert resolved["source"] == "auto" and resolved["status"] == "applied"
    backend.unload()


def test_a_pipeline_pick_keeps_bf16_when_the_scheme_declines(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, engages = None)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "off" and resolved["source"] == "auto"
    assert resolved["status"] == "applied"
    backend.unload()


def test_a_pipeline_pick_refuses_an_explicit_scheme_that_did_not_engage(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, engages = None)
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512",
            model_kind = "pipeline",
            transformer_quant = "fp8",
            _base_local_dir = str(tmp_path),
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)


def _offload_plan(
    offload_policy,
    budget_mib = 1_000_000,
    runtime_headroom_mib = 0,
):
    real_plan = DiffusionBackend._plan_memory

    def _plan(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        return dataclasses.replace(
            plan,
            offload_policy = offload_policy,
            estimates = {
                **plan.estimates,
                "safe_device_budget_mib": budget_mib,
                "runtime_headroom_mib": runtime_headroom_mib,
                "base_overhead_mib": 0,
            },
        )

    return _plan


def test_a_pipeline_pick_quantises_under_whole_module_offload(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _offload_plan("model"))
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls != []
    assert status["transformer_quant"] is not None
    backend.unload()


@pytest.mark.parametrize(
    ("offload_policy", "expected"),
    [("none", "inference_mode"), ("model", "no_grad"), ("group", "no_grad")],
)
def test_an_offloaded_quantised_transformer_renders_outside_inference_mode(
    fake_runtime, tmp_path, monkeypatch, offload_policy, expected
):
    """torchao tensors cannot change device under inference_mode, so offloaded quant renders use no_grad."""
    import torch

    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(diffusion_memory, "_torchao_stream_pinnable", lambda plan, *_a: True)
    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _offload_plan(offload_policy))
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] is not None
    used = []

    def _mode(name):
        @contextlib.contextmanager
        def _cm():
            used.append(name)
            yield

        return _cm

    monkeypatch.setattr(torch, "inference_mode", _mode("inference_mode"))
    monkeypatch.setattr(torch, "no_grad", _mode("no_grad"))
    backend.generate(prompt = "p", steps = 2)
    assert used == [expected]
    backend.unload()


def test_the_offload_replan_prices_a_precast_text_encoder_once(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion_te_prequant as te_prequant

    real_plan = DiffusionBackend._plan_memory

    class _Module:
        def __init__(self, mib):
            self._t = types.SimpleNamespace(numel = lambda: mib * 1024 * 1024, element_size = lambda: 1)

        def parameters(self, recurse = True):
            return [self._t]

        def buffers(self, recurse = True):
            return []

    monkeypatch.setattr(
        sys.modules["torch"], "nn", types.SimpleNamespace(Module = _Module), raising = False
    )

    def _replan_te(scale, held_mib = None):
        monkeypatch.setattr(te_prequant, "te_prequant_budget_scale", lambda *a, **k: scale)
        backend = DiffusionBackend()
        _stub_pipeline_dense_quant(backend, monkeypatch)
        if held_mib is not None:
            monkeypatch.setattr(
                _FakePipe, "components", {"text_encoder": _Module(held_mib)}, raising = False
            )
        seen = []

        def _plan(self, *args, **kwargs):
            plan = real_plan(self, *args, **kwargs)
            if kwargs.get("transformer_resident_override_mib") is not None:
                seen.append(kwargs.get("text_encoder_override_mib"))
            return dataclasses.replace(plan, offload_policy = "model")

        monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
        )
        backend.unload()
        assert seen, "the quant replan never ran"
        return seen[-1]

    table_te = _replan_te(1.0)
    precast_te = _replan_te(0.65)
    assert precast_te == int(table_te * 0.65)
    assert _replan_te(0.65, held_mib = table_te // 2) == precast_te


@pytest.mark.parametrize("offload_policy", ["group", "sequential"])
def test_a_pipeline_pick_stays_dense_under_streamed_offload(
    fake_runtime, tmp_path, monkeypatch, offload_policy
):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _offload_plan(offload_policy))
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert (
        "hooks torchao weights do not survive" in status["resolved"]["transformer_quant"]["reason"]
    )
    backend.unload()


@pytest.mark.parametrize(
    ("budget_mib", "runtime_headroom_mib", "companion_mib"),
    [(1, 0, None), (1_000_000, 1_000_000, None), (1_000_000, 0, 2_000_000)],
)
def test_a_pipeline_pick_stays_dense_when_the_quantised_transformer_exceeds_the_budget(
    fake_runtime, tmp_path, monkeypatch, budget_mib, runtime_headroom_mib, companion_mib
):
    """Quantised transformer (plus runtime headroom) and every text encoder must fit unstreamed."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(dmod, "largest_streamable_companion_mib", lambda pipe: companion_mib)
    monkeypatch.setattr(
        DiffusionBackend,
        "_plan_memory",
        _offload_plan("model", budget_mib = budget_mib, runtime_headroom_mib = runtime_headroom_mib),
    )
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert "not known to fit" in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


@pytest.mark.parametrize("torchao_version", [(0, 17), (0, 18)])
def test_a_pipeline_pick_quantises_under_streamed_group_offload_on_a_measured_torchao(
    fake_runtime, tmp_path, monkeypatch, torchao_version
):
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: torchao_version)
    monkeypatch.setattr(diffusion_memory, "_torchao_stream_pinnable", lambda plan, *_a: True)
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _offload_plan("group"))
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    backend.unload()


@pytest.mark.parametrize(
    ("budget_mib", "runtime_headroom_mib", "companion_mib"),
    [(1, 0, None), (1_000_000, 1_000_000, None), (1_000_000, 0, 2_000_000)],
)
def test_a_quantised_transformer_too_big_to_onload_whole_streams_instead(
    fake_runtime, tmp_path, monkeypatch, budget_mib, runtime_headroom_mib, companion_mib
):
    from core.inference import diffusion as dmod
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(diffusion_memory, "_torchao_stream_pinnable", lambda plan, *_a: True)
    # diffusion.py binds its own reference; without this the host's real pin budget decides.
    monkeypatch.setattr(dmod, "_torchao_stream_pinnable", lambda plan, *_a: True)
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(dmod, "largest_streamable_companion_mib", lambda pipe: companion_mib)
    placed: list = []

    def _apply(pipe, plan, **_kwargs):
        placed.append(plan.offload_policy)
        return plan.offload_policy, plan.vae_tiling

    monkeypatch.setattr(dmod, "apply_memory_plan", _apply)
    monkeypatch.setattr(
        DiffusionBackend,
        "_plan_memory",
        _offload_plan("model", budget_mib = budget_mib, runtime_headroom_mib = runtime_headroom_mib),
    )
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    assert placed == ["streaming"] and status["offload_policy"] == "streaming"
    backend.unload()


def test_an_offloaded_pipeline_replans_against_the_quantised_size(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference.diffusion_memory import OFFLOAD_NONE

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    overrides: list = []
    real_plan = DiffusionBackend._plan_memory

    def _plan(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        override = kwargs.get("transformer_resident_override_mib")
        overrides.append(override)
        policy = OFFLOAD_NONE if override is not None else "model"
        return dataclasses.replace(plan, offload_policy = policy)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert any(override is not None for override in overrides)
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    assert status["offload_policy"] == OFFLOAD_NONE
    backend.unload()


def test_a_declined_quant_gives_back_the_bf16_placement(fake_runtime, tmp_path, monkeypatch):
    from core.inference.diffusion_memory import OFFLOAD_NONE

    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, engages = None)
    real_plan = DiffusionBackend._plan_memory

    def _plan(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        resident = kwargs.get("transformer_resident_override_mib") is not None
        return dataclasses.replace(plan, offload_policy = OFFLOAD_NONE if resident else "model")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] is None
    assert status["offload_policy"] == "model"
    backend.unload()


def test_a_pipeline_pick_stays_dense_when_the_speed_mode_will_not_compile(
    fake_runtime, tmp_path, monkeypatch
):
    """Eager torchao is far slower than the bf16 it replaces, so an uncompilable load keeps bf16."""
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        speed_mode = "eager",
        _base_local_dir = str(tmp_path),
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert "eager" in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def test_speed_off_keeps_the_bit_exact_bf16_pipeline(fake_runtime, tmp_path, monkeypatch):
    """Speed=Off asks for bit-exact output, so an auto quant is rewritten to off before it runs."""
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        speed_mode = "off",
        _base_local_dir = str(tmp_path),
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert status["resolved"]["transformer_quant"]["value"] == "off"
    backend.unload()


def test_a_pipeline_pick_stays_dense_when_this_process_cannot_compile(
    fake_runtime, tmp_path, monkeypatch
):
    """A Windows install without Triton, or TORCHDYNAMO_DISABLE, must not quantise either."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(dmod, "compile_eligible", lambda target, **kw: False)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert "compile" in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def test_an_uncompilable_pipeline_refuses_an_explicit_scheme(fake_runtime, tmp_path, monkeypatch):
    """A pinned scheme fails closed rather than landing a slower-than-bf16 build."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(dmod, "compile_eligible", lambda target, **kw: False)
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512",
            model_kind = "pipeline",
            transformer_quant = "fp8",
            _base_local_dir = str(tmp_path),
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)


@pytest.mark.parametrize("speed_mode", [None, "default", "max"])
def test_a_compiling_speed_mode_still_quantises(fake_runtime, tmp_path, monkeypatch, speed_mode):
    """The guard must not cost the default path: speed unset is upgraded to a compile.

    "off" is absent on purpose: not uncompilable, but the bit-exact request that rewrites an auto
    quant to off well before this guard, which the test below pins.
    """
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        speed_mode = speed_mode,
        _base_local_dir = str(tmp_path),
    )
    assert status["transformer_quant"] == "fp8"
    assert len(calls) == 1
    backend.unload()


def test_a_pipeline_pick_bakes_its_adapters_before_quantising(fake_runtime, tmp_path, monkeypatch):
    """Adapters are baked before torchao replaces their dense base layers."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    order: list = []
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        lambda self, specs, **kw: [("a", "/loras/a.safetensors", 0.8)],
    )
    real_init = _FakePipe.__init__

    def _init(self):
        real_init(self)
        self.transformer = _FakeDenoiser()

    monkeypatch.setattr(_FakePipe, "__init__", _init)

    def _quantize(pipe, target, **kwargs):
        order.append("quantize")
        return "int8"

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)

    def _load_lora(
        self,
        path,
        adapter_name = None,
    ):
        order.append(f"bake:{adapter_name}")

    monkeypatch.setattr(_FakePipe, "load_lora_weights", _load_lora, raising = False)
    monkeypatch.setattr(
        _FakePipe,
        "set_adapters",
        lambda self, names, adapter_weights = None: None,
        raising = False,
    )
    backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        loras = [("a", 0.8)],
        _base_local_dir = str(tmp_path),
    )
    assert order == ["bake:a", "quantize"]
    assert backend._state.pipe._unsloth_loras_baked is True
    backend.unload()


def test_a_unet_pipeline_is_never_quantised(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch, denoisers = ())
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert "UNet" in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("torch.uint8", "uint8"),  # bitsandbytes packs NF4 into uint8 storage
        ("torch.float8_e4m3fn", "float8_e4m3fn"),
        ("torch.float16", "float16"),
    ],
)
def test_an_already_quantised_pipeline_is_never_requantised(
    fake_runtime, tmp_path, monkeypatch, dtype, expected
):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(_FakePipe, "__init__", _init_with_denoiser(dtype))
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert expected in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def test_a_blocked_pipeline_still_refuses_an_explicit_scheme(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, denoisers = ())
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512",
            model_kind = "pipeline",
            transformer_quant = "fp8",
            _base_local_dir = str(tmp_path),
        )
    assert "transformer_quant='fp8' could not be used" in str(excinfo.value)
    assert "UNet" in str(excinfo.value)


def test_every_denoiser_is_quantised_or_none_is(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(
        backend, monkeypatch, denoisers = ("transformer", "unconditional_transformer")
    )
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] == "fp8"
    pipe = backend._state.pipe
    assert [call["transformer"] for call in calls] == [
        pipe.transformer,
        pipe.unconditional_transformer,
    ]
    backend.unload()


def test_a_partially_converted_transformer_fails_the_load(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, engages = None)
    monkeypatch.setattr(dmod, "transformer_is_quantised", lambda module: True)
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
        )
    assert "neither dense nor usable" in str(excinfo.value)


def test_a_clean_decline_keeps_the_dense_transformer(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch, engages = None)
    monkeypatch.setattr(dmod, "transformer_is_quantised", lambda module: False)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert status["transformer_quant"] is None
    assert status["resolved"]["transformer_quant"]["status"] == "applied"
    backend.unload()


def test_a_raw_fp8_pipeline_is_not_quantised_a_second_time(fake_runtime, tmp_path, monkeypatch):
    """A local fp8 checkpoint is widened to bf16 on load; quantising it again compounds loss.

    Non-GGUF loads are gated to unsloth/* or a LOCAL path, so the reachable shape is a user pointing
    at their own fp8 conversion. test_diffusion_transformer_quant.py covers the header parse against
    real safetensors; this is the loader wiring that stamps every denoiser for the blocker.
    """
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    seen: list = []

    def _stored(local_dir):
        seen.append(local_dir)
        return "fp8"

    monkeypatch.setattr(dmod, "stored_denoiser_precision", _stored)
    status = backend.load_pipeline(
        "unsloth/Qwen-Image-2512",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert seen == [str(tmp_path)]
    assert calls == []
    assert status["transformer_quant"] is None
    reason = status["resolved"]["transformer_quant"]["reason"]
    assert "fp8" in reason and "widened to bf16" in reason
    backend.unload()


def test_a_bf16_pipeline_is_unaffected_by_the_header_probe(fake_runtime, tmp_path, monkeypatch):
    """The probe must not cost the ordinary official-pipeline case."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(dmod, "stored_denoiser_precision", lambda local_dir: None)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    backend.unload()


def test_a_local_fp8_directory_is_scanned_through_the_load_base(
    fake_runtime, tmp_path, monkeypatch
):
    """A local diffusers dir stages nothing, so the probe must read the base the load reads."""
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    seen: list = []

    def _stored(local_dir):
        seen.append(local_dir)
        return "fp8" if local_dir else None

    monkeypatch.setattr(dmod, "stored_denoiser_precision", _stored)
    status = backend.load_pipeline(
        "unsloth/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = None
    )
    assert seen and seen[0]
    assert calls == []
    assert status["transformer_quant"] is None
    assert "fp8" in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def test_a_prequant_repo_missing_its_artifact_marks_the_plan_incomplete(monkeypatch):
    """The repo answers and holds NEITHER the primary nor the fallback name. That is not "no
    prequant is used": this pick is configured to use one and the dense shards are already excluded
    for it, so the plan would name no transformer source at all while calling itself complete."""
    from types import SimpleNamespace

    import core.inference.diffusion as diffusion_mod

    source = SimpleNamespace(
        kind = "repo",
        location = "unsloth/some-prequant",
        filename = "transformer_fp8.safetensors",
        fallback_filenames = ("transformer.fp8.safetensors",),
    )
    monkeypatch.setattr(diffusion_mod, "usable_prequant_source", lambda *a, **k: source)
    monkeypatch.setattr(diffusion_mod, "select_transformer_quant_scheme", lambda *a, **k: "fp8")
    monkeypatch.setattr(
        DiffusionBackend, "_target_for_ordinal", lambda self, fam, ordinal: SimpleNamespace()
    )
    _fake_hf_api(
        monkeypatch,
        {"unsloth/some-prequant": [_FakeSibling("README.md", 10)]},
    )

    failures: list = []
    got = DiffusionBackend()._dit_prequant_plan_source(
        SimpleNamespace(name = "flux.2-klein"),
        "gguf",
        None,
        {"transformer_quant": "fp8"},
        failures,
    )

    assert got is None
    assert (
        failures
    ), "a configured prequant that is not in its repo left the plan calling itself complete"
    assert "prequant artifact missing" in str(failures[0])


def test_generate_runs_the_pipeline_through_the_render_thread(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as diff_mod

    names = []

    def run(name, fn):
        names.append(name)
        return fn()

    monkeypatch.setattr(diff_mod.render_thread, "run", run)
    backend = _loaded_backend(tmp_path)
    assert len(backend.generate(prompt = "a sloth", steps = 2)["images"]) == 1
    assert names == ["diffusion"]


class _StopAfterInstallGate(Exception):
    """Raised just past the FlashInfer pre-install hop."""


def _hub_refusal(cls):
    # response is optional in huggingface_hub 0.x but required in 1.x; a stub works on either.
    return cls(
        "401 Client Error. Repository Not Found for url: "
        "https://huggingface.co/api/models/unsloth/Z-Image-Turbo-NVFP4",
        response = types.SimpleNamespace(headers = {}, request = None),
    )


def _nvfp4_install_probe(
    monkeypatch,
    *,
    listing = None,
    refusal = None,
    cached = False,
):
    """Record FlashInfer installs and Hub listings; ``refusal`` is raised instead of ``listing``."""
    import core.inference.diffusion as diffusion_mod
    import core.inference.diffusion_prequant as prequant_mod
    from core.inference import diffusion_nvfp4_install as inst

    installs: list = []
    listed: list = []

    def _ensure(device, **kwargs):
        installs.append((device, kwargs.get("local_files_only")))
        return True, "installed flashinfer for NVFP4"

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            listed.append(repo_id)
            if refusal is not None:
                raise refusal
            return _FakeInfo(list(listing or []))

    monkeypatch.setattr(inst, "ensure_flashinfer_for_nvfp4", _ensure)
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr(
        prequant_mod, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.setattr(diffusion_mod, "prequant_checkpoint_cached", lambda *a, **k: cached)
    monkeypatch.setattr(
        DiffusionBackend,
        "_target_for_ordinal",
        lambda self, fam, ordinal: types.SimpleNamespace(device = "cuda", dtype = None, ordinal = 0),
    )
    monkeypatch.setattr(diffusion_mod, "apply_diffusion_device_ordinal", lambda target: None)
    monkeypatch.setattr(diffusion_mod, "select_attention_backend", lambda *a, **k: None)

    def _stop(self):
        raise _StopAfterInstallGate()

    monkeypatch.setattr(DiffusionBackend, "_reserve_teardown_locked", _stop)
    return installs, listed


def _load_to_the_install_gate(
    repo_id = "Tongyi-MAI/Z-Image-Turbo",
    family = "z-image",
    **overrides,
):
    kwargs = dict(
        model_kind = "pipeline",
        family_override = family,
        transformer_quant = "nvfp4",
        _fetch_base = repo_id,
    )
    kwargs.update(overrides)
    with pytest.raises(_StopAfterInstallGate):
        DiffusionBackend().load_pipeline(repo_id, **kwargs)


@pytest.mark.parametrize("error", ["RepositoryNotFoundError", "GatedRepoError"])
def test_an_nvfp4_checkpoint_the_hub_refuses_installs_no_flashinfer(
    fake_runtime, monkeypatch, error
):
    import huggingface_hub.errors as hub_errors

    installs, listed = _nvfp4_install_probe(
        monkeypatch, refusal = _hub_refusal(getattr(hub_errors, error))
    )
    _load_to_the_install_gate()
    assert installs == []
    assert listed == ["unsloth/Z-Image-Turbo-NVFP4"]


def test_an_nvfp4_repo_missing_the_checkpoint_installs_no_flashinfer(fake_runtime, monkeypatch):
    installs, listed = _nvfp4_install_probe(monkeypatch, listing = [_FakeSibling("README.md", 10)])
    _load_to_the_install_gate()
    assert installs == []
    assert listed == ["unsloth/Z-Image-Turbo-NVFP4"]


def test_a_family_with_no_hosted_nvfp4_checkpoint_installs_no_flashinfer(fake_runtime, monkeypatch):
    installs, listed = _nvfp4_install_probe(monkeypatch)
    _load_to_the_install_gate("Qwen/Qwen-Image", family = "qwen-image")
    assert installs == []
    assert listed == [], "nothing hosted, so there is nothing to ask the Hub about"


def test_a_gguf_nvfp4_load_whose_checkpoint_the_hub_refuses_installs_no_flashinfer(
    fake_runtime, monkeypatch, tmp_path
):
    from huggingface_hub.errors import RepositoryNotFoundError

    installs, listed = _nvfp4_install_probe(
        monkeypatch, refusal = _hub_refusal(RepositoryNotFoundError)
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    with pytest.raises(_StopAfterInstallGate):
        DiffusionBackend().load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            base_repo = "Tongyi-MAI/Z-Image-Turbo",
            family_override = "z-image",
            transformer_quant = "nvfp4",
            _fetch_base = "Tongyi-MAI/Z-Image-Turbo",
        )
    assert installs == []
    assert listed == ["unsloth/Z-Image-Turbo-NVFP4"]


def test_a_reachable_nvfp4_checkpoint_still_installs_flashinfer(fake_runtime, monkeypatch):
    installs, listed = _nvfp4_install_probe(
        monkeypatch, listing = [_FakeSibling("Z-Image-Turbo-NVFP4.safetensors", 6 * GB)]
    )
    _load_to_the_install_gate()
    assert listed == ["unsloth/Z-Image-Turbo-NVFP4"]
    assert installs == [("cuda", False)]


def test_with_the_nvfp4_switch_off_a_reachable_checkpoint_installs_nothing(
    fake_runtime, monkeypatch
):
    # conftest sets UNSLOTH_NVFP4_DIFFUSION=1; unset for the shipped default.
    monkeypatch.delenv("UNSLOTH_NVFP4_DIFFUSION", raising = False)
    installs, listed = _nvfp4_install_probe(
        monkeypatch, listing = [_FakeSibling("Z-Image-Turbo-NVFP4.safetensors", 6 * GB)]
    )
    _load_to_the_install_gate(transformer_quant = None, _pipeline_prequant_planned = "nvfp4")
    assert installs == []
    assert listed == [], "no Hub request to a *-NVFP4 repo while the switch is off"
    with pytest.raises(ValueError, match = "NVFP4 is disabled in this build"):
        DiffusionBackend().load_pipeline(
            "Tongyi-MAI/Z-Image-Turbo",
            model_kind = "pipeline",
            family_override = "z-image",
            transformer_quant = "nvfp4",
            _fetch_base = "Tongyi-MAI/Z-Image-Turbo",
        )
    assert installs == [] and listed == []


def test_a_plan_that_settled_nvfp4_installs_without_asking_the_hub_again(fake_runtime, monkeypatch):
    installs, listed = _nvfp4_install_probe(
        monkeypatch, refusal = RuntimeError("no request expected")
    )
    _load_to_the_install_gate(transformer_quant = None, _pipeline_prequant_planned = "nvfp4")
    assert listed == []
    assert installs == [("cuda", False)]


@pytest.mark.parametrize("reserved_gb, installs_expected", [(0, False), (60, True)])
def test_a_settled_nvfp4_seed_the_live_memory_plan_would_drop_installs_no_flashinfer(
    fake_runtime, monkeypatch, reserved_gb, installs_expected
):
    import core.inference.diffusion as diffusion_mod
    from core.inference.diffusion_memory import DeviceMemory

    installs, listed = _nvfp4_install_probe(
        monkeypatch, refusal = RuntimeError("no request expected")
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_target_for_ordinal",
        lambda self, fam, ordinal: types.SimpleNamespace(
            device = "cuda", dtype = None, ordinal = 0, supports_model_cpu_offload = True
        ),
    )
    monkeypatch.setattr(
        diffusion_mod,
        "snapshot_device_memory",
        lambda target: DeviceMemory("cuda", "cuda", "discrete_vram", 2 * 1024, 80 * 1024),
    )
    sys.modules["torch"].cuda.memory_reserved = lambda: reserved_gb * 1024**3
    _load_to_the_install_gate(transformer_quant = None, _pipeline_prequant_planned = "nvfp4")
    assert listed == []
    assert installs == ([("cuda", False)] if installs_expected else [])


def test_an_nvfp4_lora_bake_installs_no_flashinfer(fake_runtime, monkeypatch):
    installs, listed = _nvfp4_install_probe(
        monkeypatch, listing = [_FakeSibling("Z-Image-Turbo-NVFP4.safetensors", 6 * GB)]
    )
    _load_to_the_install_gate(loras = [("some/lora", 1.0)])
    assert installs == []


@pytest.mark.parametrize("cached", [False, True])
def test_an_offline_nvfp4_load_asks_only_the_cache(fake_runtime, monkeypatch, cached):
    installs, listed = _nvfp4_install_probe(
        monkeypatch, refusal = RuntimeError("an offline load made a Hub request"), cached = cached
    )
    _load_to_the_install_gate(local_files_only = True)
    assert listed == []
    assert installs == ([("cuda", True)] if cached else [])


def test_plan_memory_prices_the_hosted_precast_text_encoder(monkeypatch, tmp_path):
    snapshot = _base_snapshot_with_sizes(
        tmp_path, monkeypatch, {"vae/diffusion_pytorch_model.safetensors": 50}
    )
    target = _small_card(monkeypatch)
    seen = {}

    def _precast(
        fam,
        base,
        tgt,
        text_encoder_quant,
        staged_dir = None,
    ):
        seen["quant"] = text_encoder_quant
        return (2800, ("text_encoder",), True) if text_encoder_quant == "fp8" else None

    monkeypatch.setattr(DiffusionBackend, "_precast_text_encoder_mib", staticmethod(_precast))

    def _plan(**kw):
        return DiffusionBackend()._plan_memory(
            target,
            None,
            "bfl/base",
            types.SimpleNamespace(name = "flux.1"),
            None,
            False,
            kind = "gguf",
            transformer_resident_override_mib = 300,
            base_local_dir = str(snapshot),
            **kw,
        )

    before = _plan()
    assert before.estimates["companion_dense_mib"] == 50
    assert before.estimates["text_encoder_dense_mib"] is None
    after = _plan(text_encoder_quant = "fp8")
    assert seen["quant"] == "fp8"
    assert after.estimates["text_encoder_dense_mib"] == 2800
    assert after.estimates["companion_dense_mib"] == 2850
    assert after.estimates["model_dense_mib"] == 3150


def test_plan_memory_prices_cached_dense_shards_over_a_cached_precast(monkeypatch, tmp_path):
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 4000,
            "vae/diffusion_pytorch_model.safetensors": 50,
        },
    )
    target = _small_card(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_precast_text_encoder_mib",
        staticmethod(lambda *a, **k: (2600, ("text_encoder",), True)),
    )
    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        base_local_dir = str(snapshot),
        text_encoder_quant = "fp8",
    )
    assert plan.estimates["text_encoder_dense_mib"] == 4000
    assert plan.estimates["companion_dense_mib"] == 4050
    assert plan.estimates["model_dense_mib"] == 4350


def test_cached_dense_shards_keep_the_24g_denoiser_resident(monkeypatch, tmp_path):
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import OFFLOAD_GROUP, DeviceMemory

    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 16689,
            "vae/diffusion_pytorch_model.safetensors": 1288,
        },
    )
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 23000, 24576),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 8192)
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    monkeypatch.setattr(
        DiffusionBackend,
        "_precast_text_encoder_mib",
        staticmethod(lambda *a, **k: (8959, ("text_encoder",), True)),
    )
    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "qwen-image"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 7650,
        base_local_dir = str(snapshot),
        text_encoder_quant = "fp8",
    )
    assert plan.estimates["text_encoder_dense_mib"] == 16689
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_transformer is False and plan.stream_text_encoders is True


def _flux_like_plan(tmp_path, monkeypatch, precast):
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 235,
            "text_encoder_2/model.safetensors": 4000,
            "vae/diffusion_pytorch_model.safetensors": 50,
        },
    )
    target = _small_card(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend, "_precast_text_encoder_mib", staticmethod(lambda *a, **k: precast)
    )
    return DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        base_local_dir = str(snapshot),
        text_encoder_quant = "fp8",
    )


def test_plan_memory_swaps_only_the_encoder_the_precast_checkpoint_replaces(monkeypatch, tmp_path):
    plan = _flux_like_plan(tmp_path, monkeypatch, (4200, ("text_encoder_2",), True))
    assert plan.estimates["text_encoder_dense_mib"] == 235 + 4200
    assert plan.estimates["companion_dense_mib"] == 50 + 235 + 4200


def test_plan_memory_never_prices_an_uncached_precast_below_the_dense_shards(monkeypatch, tmp_path):
    plan = _flux_like_plan(tmp_path, monkeypatch, (2600, ("text_encoder_2",), False))
    assert plan.estimates["text_encoder_dense_mib"] == 235 + 4000


def test_a_table_priced_pipeline_does_not_add_the_precast_encoder_on_top(monkeypatch, tmp_path):
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "transformer/diffusion_pytorch_model.safetensors": 13500,
            "vae/diffusion_pytorch_model.safetensors": 1288,
        },
    )
    target = _small_card(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_precast_text_encoder_mib",
        staticmethod(lambda *a, **k: (8959, ("text_encoder",), True)),
    )
    fam = types.SimpleNamespace(name = "qwen-image-2.1", base_repo = "bfl/base")

    def _plan(quant):
        return (
            DiffusionBackend()
            ._plan_memory(
                target,
                None,
                "bfl/base",
                fam,
                None,
                False,
                kind = "pipeline",
                repo_id = "bfl/base",
                base_local_dir = str(snapshot),
                text_encoder_quant = quant,
            )
            .estimates
        )

    dense, precast = _plan(None), _plan("fp8")
    assert dense["text_encoder_dense_mib"] == 16689
    for key in ("model_dense_mib", "companion_dense_mib", "text_encoder_dense_mib"):
        assert precast[key] == dense[key]


def test_plan_memory_leaves_a_callers_companion_override_alone(monkeypatch, tmp_path):
    target = _small_card(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend, "_precast_text_encoder_mib", staticmethod(lambda *a, **k: 2600)
    )
    plan = DiffusionBackend()._plan_memory(
        target,
        None,
        "bfl/base",
        types.SimpleNamespace(name = "flux.1"),
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 300,
        companion_override_mib = 700,
        text_encoder_override_mib = 600,
        text_encoder_quant = "fp8",
    )
    assert plan.estimates["companion_dense_mib"] == 700
    assert plan.estimates["text_encoder_dense_mib"] == 600


def _q21_precast_cache(tmp_path, monkeypatch, *, mib):
    """unsloth/Qwen-Image-2.1-FP8 cached under the live root, holding only the pre-cast encoder."""
    live, _other = _split_cache_roots(tmp_path, monkeypatch)
    repo = live / "models--unsloth--Qwen-Image-2.1-FP8"
    rev = "b" * 40
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(rev)
    snap = repo / "snapshots" / rev
    snap.mkdir(parents = True)
    if mib:
        with open(snap / "Qwen-Image-2.1-text_encoder-FP8.safetensors", "wb") as fh:
            fh.truncate(mib * 1024 * 1024)
    return snap


def _bf16_cuda_target():
    import torch
    return types.SimpleNamespace(
        device = "cuda", backend = "cuda", dtype = torch.bfloat16, supports_model_cpu_offload = True
    )


def test_precast_text_encoder_mib_reads_the_cached_checkpoint(monkeypatch, tmp_path):
    from core.inference.diffusion_families import detect_family

    fam = detect_family("Qwen/Qwen-Image-2.1")
    assert fam is not None and fam.name == "qwen-image-2.1"
    _q21_precast_cache(tmp_path, monkeypatch, mib = 8959)
    target = _bf16_cuda_target()
    assert DiffusionBackend._precast_text_encoder_mib(
        fam, "Qwen/Qwen-Image-2.1", target, "fp8"
    ) == (
        8959,
        ("text_encoder",),
        True,
    )
    assert (
        DiffusionBackend._precast_text_encoder_mib(fam, "Qwen/Qwen-Image-2.1", target, None) is None
    )
    assert (
        DiffusionBackend._precast_text_encoder_mib(fam, "Qwen/Qwen-Image-2.1", target, "none")
        is None
    )


def test_precast_text_encoder_mib_prices_hidreams_standalone_fourth_encoder(monkeypatch, tmp_path):
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_te_prequant import te_candidate_filenames, te_prequant_sources

    fam = detect_family("HiDream-ai/HiDream-I1-Full")
    assert fam is not None and fam.name == "hidream-i1"
    target = _bf16_cuda_target()
    source = te_prequant_sources(
        fam, te_quant_mode = "fp8", target = target, components = ("text_encoder_4",)
    )["text_encoder_4"]
    live, _other = _split_cache_roots(tmp_path, monkeypatch)
    repo = live / ("models--" + source.location.replace("/", "--"))
    rev = "c" * 40
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(rev)
    snap = repo / "snapshots" / rev
    snap.mkdir(parents = True)
    with open(snap / te_candidate_filenames(source)[0], "wb") as fh:
        fh.truncate(7700 * 1024 * 1024)
    assert DiffusionBackend._precast_text_encoder_mib(
        fam, "HiDream-ai/HiDream-I1-Full", target, "fp8"
    ) == (7700, ("text_encoder_4",), True)

    (snap / te_candidate_filenames(source)[0]).unlink()
    from core.inference.diffusion_hidream import HIDREAM_LLAMA_BF16_BYTES
    from core.inference.diffusion_te_prequant import TE_PREQUANT_BUDGET_SCALE

    assert DiffusionBackend._precast_text_encoder_mib(
        fam, "HiDream-ai/HiDream-I1-Full", target, "fp8"
    ) == (
        int(HIDREAM_LLAMA_BF16_BYTES * TE_PREQUANT_BUDGET_SCALE) // (1024 * 1024),
        ("text_encoder_4",),
        False,
    )


def test_precast_text_encoder_mib_prices_an_uncached_checkpoint_from_the_family_table(
    monkeypatch, tmp_path
):
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_te_prequant import TE_PREQUANT_BUDGET_SCALE

    fam = detect_family("Qwen/Qwen-Image-2.1")
    _q21_precast_cache(tmp_path, monkeypatch, mib = 0)
    got = DiffusionBackend._precast_text_encoder_mib(
        fam, "Qwen/Qwen-Image-2.1", _bf16_cuda_target(), "fp8"
    )
    assert got == (
        int(17.5 * 1000**3 * TE_PREQUANT_BUDGET_SCALE) // (1024 * 1024),
        ("text_encoder",),
        False,
    )
    assert got[0] > 8959


def _stub_amd_weight_only_host(backend, monkeypatch):
    """ROCm / torchao-stub host; records the quantise calls."""
    from core.inference import diffusion as dmod
    from core.inference import diffusion_transformer_quant as tq

    calls: list = []
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: False)
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: False)
    monkeypatch.setattr(dmod, "native_quant_host", lambda target: True)
    monkeypatch.setattr(tq, "native_quant_host", lambda target: True)
    real_init = _FakePipe.__init__

    def _init(self):
        real_init(self)
        self.transformer = _FakeDenoiser()

    monkeypatch.setattr(_FakePipe, "__init__", _init)

    def _quantize(
        pipe,
        target,
        *,
        mode,
        family = None,
        **kwargs,
    ):
        scheme = tq.native_quant_scheme(target, mode, family = family)
        if scheme is None:
            raise AssertionError("the torchao path must not run on an AMD weight-only host")
        calls.append({"module": pipe.transformer, "scheme": scheme})
        return scheme

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    monkeypatch.setattr(dmod, "transformer_is_quantised", lambda module: bool(calls))
    return calls


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_an_explicit_scheme_on_amd_runs_weight_only(fake_runtime, tmp_path, monkeypatch, scheme):
    backend = DiffusionBackend()
    calls = _stub_amd_weight_only_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = scheme,
        _base_local_dir = str(tmp_path),
    )
    assert [c["scheme"] for c in calls] == [scheme]
    assert calls[0]["module"] is backend._state.pipe.transformer
    assert status["transformer_quant"] == scheme
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == scheme
    assert resolved["source"] == "explicit" and resolved["status"] == "applied"
    assert "weight-only" in resolved["reason"] and "bf16 compute" in resolved["reason"]
    assert "requires compile" not in status["resolved"]["speed_mode"]["reason"]
    backend.unload()


def test_auto_on_amd_stays_bf16(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls = _stub_amd_weight_only_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert status["resolved"]["transformer_quant"]["value"] == "off"
    backend.unload()


def test_an_explicit_scheme_on_nvidia_keeps_the_torchao_path_and_its_wording(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        _base_local_dir = str(tmp_path),
    )
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    assert "weight-only" not in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def _stub_nvidia_offload_host(
    backend,
    monkeypatch,
    *,
    engages = "int8",
):
    from core.inference import diffusion as dmod
    from core.inference import diffusion_transformer_quant as tq

    calls = _stub_pipeline_dense_quant(backend, monkeypatch, engages = engages)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_: mode
    )
    monkeypatch.setattr(tq, "native_quant_host", lambda target: False)
    monkeypatch.setattr(tq, "native_offload_host", lambda target: True)
    monkeypatch.setattr(dmod, "native_offload_host", lambda target: True)
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ACT", raising = False)
    reasons: list = []

    def _reason(module, scheme):
        reasons.append(scheme)
        return f"W8A8: {scheme} (stub)"

    monkeypatch.setattr(dmod, "native_quant_reason", _reason)
    return calls, reasons


@pytest.mark.parametrize(
    "memory",
    [{"memory_mode": "balanced"}, {"memory_mode": "low_vram"}, {"cpu_offload": True}],
)
def test_an_explicit_int8_under_offload_on_nvidia_runs_native_w8a8(
    fake_runtime, tmp_path, monkeypatch, memory
):
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        _base_local_dir = str(tmp_path),
        **memory,
    )
    assert len(calls) == 1
    assert calls[0]["offload"] is True and calls[0]["act_int8"] is True
    assert status["offload_policy"] != "none"
    assert status["transformer_quant"] == "int8"
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "int8" and resolved["status"] == "applied"
    assert resolved["reason"].startswith("W8A8") and reasons == ["int8"]
    backend.unload()


def test_the_act_kill_switch_keeps_nvidia_offload_native_but_weight_only(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    calls, _ = _stub_nvidia_offload_host(backend, monkeypatch)
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", "0")
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        memory_mode = "balanced",
        _base_local_dir = str(tmp_path),
    )
    assert calls[0]["offload"] is True and calls[0]["act_int8"] is False
    assert status["transformer_quant"] == "int8"
    backend.unload()


def test_an_explicit_fp8_under_group_offload_engages_on_a_measured_torchao(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(diffusion_memory, "_torchao_stream_pinnable", lambda plan, *_a: True)
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch, engages = "fp8")
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        memory_mode = "balanced",
        _base_local_dir = str(tmp_path),
    )
    assert len(calls) == 1 and "offload" not in calls[0] and reasons == []
    assert status["transformer_quant"] == "fp8"
    backend.unload()


@pytest.mark.parametrize(
    "memory",
    [{"memory_mode": "balanced"}, {"memory_mode": "low_vram"}, {"cpu_offload": True}],
)
def test_an_explicit_int8_under_offload_stays_native_on_a_measured_torchao(
    fake_runtime, tmp_path, monkeypatch, memory
):
    from core.inference import diffusion_memory

    monkeypatch.setattr(diffusion_memory, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(diffusion_memory, "_torchao_stream_pinnable", lambda plan, *_a: True)
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        _base_local_dir = str(tmp_path),
        **memory,
    )
    assert len(calls) == 1 and calls[0]["offload"] is True
    assert status["transformer_quant"] == "int8" and reasons == ["int8"]
    backend.unload()


@pytest.mark.parametrize("fallback", ["0", "1"])
def test_an_explicit_fp8_under_offload_on_nvidia_is_still_declined(
    fake_runtime, tmp_path, monkeypatch, fallback
):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", fallback)
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch, engages = "fp8")
    kwargs = dict(
        model_kind = "pipeline",
        transformer_quant = "fp8",
        memory_mode = "balanced",
        _base_local_dir = str(tmp_path),
    )
    if fallback == "0":
        with pytest.raises(RuntimeError, match = "hooks torchao weights do not survive"):
            backend.load_pipeline("Qwen/Qwen-Image-2512", **kwargs)
    else:
        status = backend.load_pipeline("Qwen/Qwen-Image-2512", **kwargs)
        assert status["transformer_quant"] is None
        assert (
            "hooks torchao weights do not survive"
            in status["resolved"]["transformer_quant"]["reason"]
        )
        backend.unload()
    assert calls == [] and reasons == []


def test_an_explicit_int8_resident_on_nvidia_keeps_torchao(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        memory_mode = "fast",
        _base_local_dir = str(tmp_path),
    )
    assert status["offload_policy"] == "none"
    assert len(calls) == 1 and "offload" not in calls[0] and "act_int8" not in calls[0]
    assert status["transformer_quant"] == "int8"
    assert reasons == []
    assert "W8A8" not in status["resolved"]["transformer_quant"]["reason"]
    backend.unload()


def test_auto_under_offload_on_nvidia_never_goes_native(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    calls, reasons = _stub_nvidia_offload_host(backend, monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        memory_mode = "balanced",
        _base_local_dir = str(tmp_path),
    )
    assert calls == [] and reasons == []
    assert status["transformer_quant"] is None
    backend.unload()


def test_a_resident_nvidia_int8_that_cannot_compile_still_declines(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls, _ = _stub_nvidia_offload_host(backend, monkeypatch)
    monkeypatch.setattr(
        dmod, "_pipeline_quant_uncompilable_reason", lambda *a, **k: "no compile here (stub)"
    )
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        memory_mode = "fast",
        _base_local_dir = str(tmp_path),
    )
    assert calls == []
    assert status["transformer_quant"] is None
    assert status["resolved"]["transformer_quant"]["reason"] == "no compile here (stub)"
    backend.unload()


def test_an_offloaded_nvidia_int8_ignores_the_compile_requirement(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    calls, _ = _stub_nvidia_offload_host(backend, monkeypatch)
    monkeypatch.setattr(
        dmod, "_pipeline_quant_uncompilable_reason", lambda *a, **k: "no compile here (stub)"
    )
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "int8",
        memory_mode = "low_vram",
        _base_local_dir = str(tmp_path),
    )
    assert len(calls) == 1 and calls[0]["offload"] is True
    assert status["transformer_quant"] == "int8"
    backend.unload()


def _record_step_cache(
    monkeypatch,
    *,
    supported = True,
    engages = True,
):
    calls = {"apply": [], "toggle": []}

    def _apply(pipe, *, mode, **kwargs):
        calls["apply"].append(mode)
        return mode if (mode == "fbcache" and engages) else None

    def _toggle(pipe, *, steps, **kwargs):
        calls["toggle"].append(steps)
        return "fbcache" if steps >= 20 else None

    monkeypatch.setattr("core.inference.diffusion.apply_step_cache", _apply)
    monkeypatch.setattr("core.inference.diffusion.maybe_toggle_step_cache", _toggle)
    monkeypatch.setattr(
        "core.inference.diffusion.step_cache_supported", lambda pipe, logger = None: supported
    )
    return calls


@pytest.mark.parametrize("speed_mode", [None, "off", "eager", "default"])
def test_image_step_cache_auto_stays_off_below_max(fake_runtime, tmp_path, monkeypatch, speed_mode):
    calls = _record_step_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image", speed_mode = speed_mode)
    status = backend.status()
    assert calls["apply"] == [None]
    assert status["transformer_cache"] is None
    assert status["resolved"]["transformer_cache"]["source"] == "auto"
    assert "max speed tier" in status["resolved"]["transformer_cache"]["reason"]
    assert backend._state.cache_auto is False
    backend.generate(prompt = "a sloth", steps = 30)
    assert calls["toggle"] == []
    assert backend.status()["transformer_cache"] is None


def test_image_step_cache_auto_engages_on_max(fake_runtime, tmp_path, monkeypatch):
    calls = _record_step_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image", speed_mode = "max")
    assert calls["apply"] == ["fbcache"]
    assert backend.status()["transformer_cache"] == "fbcache"
    assert backend._state.cache_auto is True
    backend.generate(prompt = "a sloth", steps = 8)
    assert calls["toggle"] == [8]
    assert backend.status()["transformer_cache"] is None


def test_image_explicit_step_cache_is_honoured_on_the_default_tier(
    fake_runtime, tmp_path, monkeypatch
):
    calls = _record_step_cache(monkeypatch)
    backend = _loaded_backend(
        tmp_path, family_override = "qwen-image", speed_mode = "default", transformer_cache = "fbcache"
    )
    assert calls["apply"] == ["fbcache"]
    assert backend.status()["transformer_cache"] == "fbcache"
    assert backend._state.cache_auto is False
    backend.generate(prompt = "a sloth", steps = 8)
    assert calls["toggle"] == []


def test_image_auto_toggle_armed_only_where_the_cache_can_engage(
    fake_runtime, tmp_path, monkeypatch
):
    _record_step_cache(monkeypatch, engages = False)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image", speed_mode = "max")
    assert backend._state.cache_auto is False
    assert (
        backend.status()["resolved"]["transformer_cache"]["reason"]
        == "auto: model does not support step caching"
    )
    backend.unload()

    turbo = dict(
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        speed_mode = "max",
    )
    for supported in (False, True):
        _record_step_cache(monkeypatch, supported = supported)
        backend = _loaded_backend(tmp_path, **turbo)
        assert backend.status()["transformer_cache"] is None
        assert backend._state.cache_auto is supported
        backend.unload()


def test_image_auto_below_max_names_an_uncacheable_model(fake_runtime, tmp_path, monkeypatch):
    _record_step_cache(monkeypatch, supported = False)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image", speed_mode = "default")
    assert (
        backend.status()["resolved"]["transformer_cache"]["reason"]
        == "auto: model does not support step caching"
    )


def test_unload_drains_pinned_host_memory_after_the_pipeline_is_gone(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as diffusion_module

    backend = _loaded_backend(tmp_path)
    calls: list = []
    monkeypatch.setattr(
        diffusion_module, "clear_gpu_cache", lambda: calls.append(("clear", backend._state))
    )
    monkeypatch.setattr(
        diffusion_module,
        "release_pinned_host_memory",
        lambda: calls.append(("host", backend._state)),
    )
    backend.unload()
    assert calls == [("clear", None), ("host", None)]


def test_unload_drains_pinned_host_memory_even_when_gpu_cleanup_raises(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as diffusion_module

    backend = _loaded_backend(tmp_path)
    drained: list = []

    def _sticky():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", _sticky)
    monkeypatch.setattr(
        diffusion_module, "release_pinned_host_memory", lambda: drained.append(True)
    )
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        backend.unload()
    assert drained == [True]


def test_status_reports_cuda_graph_off_once_every_armed_step_ran_eager():
    backend = DiffusionBackend()
    handle = types.SimpleNamespace(
        cache = {},
        stats = {"captures": 0, "replays": 0, "eager_calls": 0, "refused_object": 0},
        poisoned = False,
        capture_error = None,
    )
    resolved = {
        "cuda_graph": {
            "value": "on",
            "requested": None,
            "source": "auto",
            "status": "applied",
            "reason": "denoiser step captured per input shape, replayed bit-identically",
        }
    }
    backend._state = _LoadState(
        pipe = object(),
        family = detect_family("unsloth/Z-Image-GGUF"),
        repo_id = "r",
        base_repo = "b",
        device = "cuda",
        dtype = "bfloat16",
        cpu_offload = False,
        speed_optims = ("compiled", "cuda_graph"),
        resolved = resolved,
        cuda_graphs = (handle,),
    )

    st = backend.status()
    assert st["resolved"]["cuda_graph"]["value"] == "on"
    assert st["speed_optims"] == ["compiled", "cuda_graph"]

    handle.stats.update(eager_calls = 25, refused_object = 25)
    st = backend.status()
    assert st["resolved"]["cuda_graph"]["value"] == "off"
    assert st["resolved"]["cuda_graph"]["reason"] == (
        "armed, but all 25 denoiser call(s) so far ran eager (25 with a non-tensor argument)"
    )
    assert st["speed_optims"] == ["compiled"]
    assert resolved["cuda_graph"]["value"] == "on"

    handle.stats.update(captures = 1, replays = 24)
    st = backend.status()
    assert st["resolved"]["cuda_graph"]["value"] == "on"
    assert st["speed_optims"] == ["compiled", "cuda_graph"]


def _resident_transformer(plan):
    """``plan`` as the tier that keeps the denoiser resident and streams only the text encoders."""
    return dataclasses.replace(
        plan, offload_policy = "group", stream_text_encoders = True, stream_transformer = False
    )


def _record_placement(monkeypatch):
    from core.inference import diffusion as dmod

    placed: list = []

    def _apply(pipe, plan, **kwargs):
        placed.append(plan)
        return plan.offload_policy, False

    monkeypatch.setattr(dmod, "apply_memory_plan", _apply)
    return placed


def test_a_pipeline_quantises_where_only_the_encoders_stream(fake_runtime, tmp_path, monkeypatch):
    """torchao weights fit once the encoders stream, so the transformer is converted and placed once."""
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    real_plan = DiffusionBackend._plan_memory

    def _plan(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        if kwargs.get("transformer_resident_override_mib") is not None:
            return _resident_transformer(plan)
        return dataclasses.replace(plan, offload_policy = "group")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
    placed = _record_placement(monkeypatch)
    status = backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        _base_local_dir = str(tmp_path),
    )
    assert len(calls) == 1
    assert status["transformer_quant"] == "fp8"
    assert status["offload_policy"] == "group"
    assert placed and placed[-1].stream_transformer is False
    backend.unload()


def test_a_pipeline_quant_replan_prices_the_encoder_the_load_opens(
    fake_runtime, tmp_path, monkeypatch
):
    """The in-place replan prices encoders like the other candidate replans, not the family table."""
    backend = DiffusionBackend()
    _stub_pipeline_dense_quant(backend, monkeypatch)
    priced = {"companion_override_mib": 1234, "text_encoder_override_mib": 567}
    monkeypatch.setattr(
        DiffusionBackend,
        "_candidate_companion_overrides",
        staticmethod(lambda *args, **kwargs: dict(priced)),
    )
    real_plan = DiffusionBackend._plan_memory
    seen = []

    def _plan(self, *args, **kwargs):
        plan = real_plan(self, *args, **kwargs)
        if kwargs.get("transformer_resident_override_mib") is not None:
            seen.append(kwargs)
            return _resident_transformer(plan)
        return dataclasses.replace(plan, offload_policy = "group")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
    _record_placement(monkeypatch)
    backend.load_pipeline(
        "Qwen/Qwen-Image-2512",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        _base_local_dir = str(tmp_path),
    )
    assert seen
    assert seen[-1]["companion_override_mib"] == 1234
    assert seen[-1]["text_encoder_override_mib"] == 567
    backend.unload()


def test_a_pipeline_whose_quantised_plan_still_streams_the_transformer_stays_dense(
    fake_runtime, tmp_path, monkeypatch
):
    backend = DiffusionBackend()
    calls = _stub_pipeline_dense_quant(backend, monkeypatch)
    real_plan = DiffusionBackend._plan_memory

    def _plan(self, *args, **kwargs):
        return dataclasses.replace(real_plan(self, *args, **kwargs), offload_policy = "group")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan)
    _record_placement(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Qwen/Qwen-Image-2512",
            model_kind = "pipeline",
            transformer_quant = "fp8",
            _base_local_dir = str(tmp_path),
        )
    assert calls == []
    assert "hooks torchao weights do not survive" in str(excinfo.value)


def _gguf_candidate_backend(monkeypatch, tmp_path, *, initial_policy, candidate_plan):
    """GGUF load planned as ``initial_policy`` whose candidate plans as ``candidate_plan(real_plan)``."""
    from core.inference import diffusion as dmod

    backend = _cuda_backend(tmp_path, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_: "int8"
    )
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: "prequant/path")
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 7_000,
            steady_transformer_mib = 7_000,
            companions_mib = 18_000,
            text_encoders_mib = 16_700,
            prequant = True,
        ),
    )
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = initial_policy)
        return candidate_plan(real)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted: list = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    return backend, attempted


def test_a_gguf_pick_builds_the_quant_where_only_the_encoders_stream(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend, attempted = _gguf_candidate_backend(
        monkeypatch, tmp_path, initial_policy = "group", candidate_plan = _resident_transformer
    )
    _load_m(backend, tmp_path, transformer_quant = "int8")
    assert attempted == [False]


def test_a_gguf_pick_declines_the_quant_where_the_transformer_would_stream(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend, attempted = _gguf_candidate_backend(
        monkeypatch,
        tmp_path,
        initial_policy = "group",
        candidate_plan = lambda real: dataclasses.replace(real, offload_policy = "group"),
    )
    status = _load_m(backend, tmp_path, transformer_quant = "int8")
    assert attempted == []
    assert status["transformer_quant"] is None


def test_a_resident_gguf_plan_sizes_the_prequant_that_replaces_it(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    """The INT8 artifact outgrows the GGUF, so it loads under its own plan rather than the GGUF's."""
    sized: list = []

    def _candidate(real):
        plan = _resident_transformer(real)
        sized.append(plan)
        return plan

    backend, attempted = _gguf_candidate_backend(
        monkeypatch, tmp_path, initial_policy = "none", candidate_plan = _candidate
    )
    placed = _record_placement(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_load_dense_quant_pipeline",
        lambda self, *a, **k: attempted.append(k.get("allow_dense_fallback")) or (None, None),
    )
    _load_m(backend, tmp_path, transformer_quant = "int8")
    assert sized, "the prequant was never sized"
    assert attempted == [False]
    assert placed[-1].offload_policy in ("none", "group")


def test_a_resident_gguf_plan_declines_a_prequant_that_would_stream(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend, attempted = _gguf_candidate_backend(
        monkeypatch,
        tmp_path,
        initial_policy = "none",
        candidate_plan = lambda real: dataclasses.replace(real, offload_policy = "group"),
    )
    status = _load_m(backend, tmp_path, transformer_quant = "int8")
    assert attempted == []
    assert status["transformer_quant"] is None
    assert (
        "torchao tensors cannot be offloaded" in (status["resolved"]["transformer_quant"]["reason"])
    )


def test_candidate_overrides_price_the_encoder_the_load_opens(monkeypatch):
    candidate = types.SimpleNamespace(companions_mib = 18_000, text_encoders_mib = 16_000)
    monkeypatch.setattr(
        DiffusionBackend,
        "_precast_scaled_companions_mib",
        staticmethod(lambda cand, fam, base, target, teq: 2_000 + int(16_000 * 0.65)),
    )
    overrides = DiffusionBackend._candidate_companion_overrides(candidate, None, "b", None, "fp8")
    assert overrides == {"companion_override_mib": 12_400, "text_encoder_override_mib": 10_400}
    bare = types.SimpleNamespace(companions_mib = 18_000, text_encoders_mib = 0)
    assert (
        DiffusionBackend._candidate_companion_overrides(bare, None, "b", None, None)[
            "text_encoder_override_mib"
        ]
        == 0
    )


_Q4 = "m-Q4_K_M.gguf"


def _gguf_offload_swap_setup(
    monkeypatch,
    tmp_path,
    *,
    scheme = "int8",
    policy = "model",
):
    """A CUDA GGUF pick whose plan and every quantised re-plan offload, with a cached hosted checkpoint."""
    import dataclasses

    from core.inference import diffusion as dmod

    (tmp_path / _Q4).write_bytes(b"x")
    (tmp_path / "m-Q8_0.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_GGUF_OFFLOAD_PREQUANT", raising = False)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None, **_kw: scheme
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            scheme = scheme, transient_transformer_mib = 7_000, companions_mib = 8_000, prequant = True
        ),
    )
    source = types.SimpleNamespace(
        kind = "repo", location = "unsloth/Z-FP8", filename = "Z-INT8.safetensors"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda *a, **k: source)
    monkeypatch.setattr(dmod, "torchao_offload_plan", lambda plan, s, **k: plan)
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(self, *a, **k):
        return dataclasses.replace(orig_plan(self, *a, **k), offload_policy = policy)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    calls: list = []

    def fake_dense_load(self, *a, **k):
        calls.append(k)
        pipe = _FakePipe()
        pipe.transformer = types.SimpleNamespace(_unsloth_prequant_path = "/hub/Z-INT8.safetensors")
        return pipe, scheme

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    return backend, calls


def _load_gguf(
    backend,
    tmp_path,
    name = _Q4,
    **kwargs,
):
    return backend.load_pipeline(
        str(tmp_path), gguf_filename = name, family_override = "z-image", **kwargs
    )


def test_offloading_gguf_pick_loads_the_cached_int8_checkpoint(fake_runtime, tmp_path, monkeypatch):
    backend, calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)
    status = _load_gguf(backend, tmp_path)
    assert len(calls) == 1
    assert calls[0]["allow_dense_fallback"] is False
    assert calls[0]["seed_plan"].offload_policy == "model"
    assert status["transformer_quant"] == "int8"
    assert status["offload_policy"] == "model"
    tq = status["resolved"]["transformer_quant"]
    assert tq["value"] == "int8" and tq["source"] == "auto"
    assert tq["artifact"] == "prequant:unsloth/Z-FP8/Z-INT8.safetensors"
    assert tq["replaced"] == f"gguf:{_Q4}"
    assert "Q4_K_M GGUF pick was replaced by unsloth/Z-FP8/Z-INT8.safetensors" in tq["reason"]


def test_gguf_offload_swap_kill_switch_keeps_the_gguf(fake_runtime, tmp_path, monkeypatch):
    backend, calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_OFFLOAD_PREQUANT", "0")
    status = _load_gguf(backend, tmp_path)
    assert calls == []
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"].endswith(_Q4)
    assert "replaced" not in status["resolved"]["transformer_quant"]


def test_gguf_offload_swap_keeps_a_q8_pick_on_auto(fake_runtime, tmp_path, monkeypatch):
    backend, calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)
    status = _load_gguf(backend, tmp_path, name = "m-Q8_0.gguf")
    assert calls == []
    assert status["transformer_quant"] is None
    reason = status["resolved"]["transformer_quant"]["reason"]
    assert "Q8_0" in reason and "at least as accurate" in reason


@pytest.mark.parametrize("request_", ["off", "none"])
def test_gguf_offload_swap_never_overrides_precision_off(
    fake_runtime, tmp_path, monkeypatch, request_
):
    backend, calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)
    status = _load_gguf(backend, tmp_path, transformer_quant = request_)
    assert calls == []
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"].endswith(_Q4)


def test_gguf_offload_swap_falls_back_to_the_gguf_when_the_build_fails(
    fake_runtime, tmp_path, monkeypatch
):
    backend, _calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)

    def boom(self, *a, **k):
        raise RuntimeError("checkpoint unreadable")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", boom)
    status = _load_gguf(backend, tmp_path)
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"].endswith(_Q4)
    assert "replaced" not in status["resolved"]["transformer_quant"]


def test_gguf_offload_swap_on_mps_keeps_the_gguf(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend, calls = _gguf_offload_swap_setup(monkeypatch, tmp_path)
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        backend, "_target_for_ordinal", lambda fam, ordinal = None: _mps_target(torch)
    )
    from core.inference.diffusion_transformer_quant import dense_transformer_supported

    monkeypatch.setattr(dmod, "dense_transformer_supported", dense_transformer_supported)
    status = _load_gguf(backend, tmp_path)
    assert calls == []
    assert status["transformer_quant"] is None


@pytest.mark.parametrize(
    "policy, stream, env, placed_on_host",
    [
        ("streaming", True, None, True),
        ("group", True, None, True),
        ("model", True, None, True),
        ("none", True, None, False),
        ("group", False, None, False),
        ("streaming", True, "0", False),  # UNSLOTH_DIFFUSION_PREQUANT_SEED_ON_HOST=0
    ],
)
def test_gguf_route_prequant_seed_lands_on_the_host_when_the_plan_offloads(
    fake_runtime, monkeypatch, policy, stream, env, placed_on_host
):
    from core.inference import diffusion as dmod

    if env is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_PREQUANT_SEED_ON_HOST", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_PREQUANT_SEED_ON_HOST", env)
    monkeypatch.setattr(dmod, "_planned_quant_scheme", lambda *a, **k: "int8")
    monkeypatch.setattr(
        dmod,
        "resolve_prequant_source",
        lambda *a, **k: types.SimpleNamespace(kind = "repo", location = "o/r", filename = "f"),
    )
    loaded: dict = {}

    def fake_load(*a, **k):
        loaded.update(k)
        return object()

    monkeypatch.setattr(dmod, "load_prequantized_transformer", fake_load)
    assembled: list = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_assemble_pipe",
        staticmethod(
            lambda pcls, base, tr, dtype, tok, device, *a, **k: assembled.append(device)
            or _FakePipe()
        ),
    )
    _pipe, scheme = DiffusionBackend()._load_dense_quant_pipeline(
        object,
        object,
        "base/repo",
        "cuda:0",
        "bf16",
        None,
        types.SimpleNamespace(device = "cuda", dtype = "bf16"),
        "int8",
        fam = types.SimpleNamespace(name = "qwen-image-2.1"),
        seed_plan = types.SimpleNamespace(offload_policy = policy, stream_transformer = stream),
    )
    assert scheme == "int8"
    assert loaded["placement_device"] == ("cpu" if placed_on_host else None)
    assert assembled == ["cpu" if placed_on_host else "cuda:0"]


def test_diffusion_status_response_keeps_the_gguf_a_swap_replaced():
    # Pydantic drops undeclared keys.
    from models.inference import DiffusionStatusResponse

    rec = {
        "transformer_quant": {
            "value": "int8",
            "source": "auto",
            "reason": "the Q4_K_M GGUF pick was replaced",
            "artifact": "prequant:o/r/f.safetensors",
            "replaced": "gguf:m-Q4_K_M.gguf",
        }
    }
    dumped = DiffusionStatusResponse(loaded = True, resolved = rec).model_dump()["resolved"][
        "transformer_quant"
    ]
    assert dumped["replaced"] == "gguf:m-Q4_K_M.gguf"
    assert dumped["artifact"] == "prequant:o/r/f.safetensors"


class _T5WordTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens = True,
        **_,
    ):
        return {"input_ids": [5] * len(text.split()) + ([1] if add_special_tokens else [])}


class _FluxFakePipe(_FakePipe):
    def __init__(self):
        super().__init__()
        self.tokenizer_2 = _T5WordTokenizer()

    def __call__(
        self,
        *,
        prompt = None,
        max_sequence_length = 512,
        **kwargs,
    ):
        return super().__call__(prompt = prompt, max_sequence_length = max_sequence_length, **kwargs)


def test_generate_passes_flux1_t5_length_like_comfy(fake_runtime, tmp_path, monkeypatch):
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "FluxPipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "FluxTransformer2DModel", _FakeTransformer, raising = False)
    _no_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "flux.1")
    pipe = _FluxFakePipe()
    object.__setattr__(backend._state, "pipe", pipe)
    backend.generate(prompt = "a sloth on a branch", steps = 4, guidance = 0.0)
    assert pipe.last_kwargs["max_sequence_length"] == 256
    backend.generate(prompt = " ".join(["w"] * 320), steps = 4, guidance = 0.0)
    assert pipe.last_kwargs["max_sequence_length"] == 512


def test_generate_leaves_t5_length_alone_off_flux1(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path, family_override = "z-image")
    pipe = _FluxFakePipe()
    object.__setattr__(backend._state, "pipe", pipe)
    backend.generate(prompt = "a sloth", steps = 4, guidance = 0.0)
    assert pipe.last_kwargs["max_sequence_length"] == 512


def test_qwen_true_cfg_gets_an_empty_negative_like_comfy(fake_runtime, tmp_path, monkeypatch):
    """A blank negative must not silently turn Qwen-Image true CFG off."""
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "QwenImagePipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _FakeTransformer, raising = False)
    _no_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image")
    backend.generate(prompt = "a sloth", steps = 4, guidance = 4.0)
    call = backend._state.pipe.last_kwargs
    assert call["true_cfg_scale"] == 4.0 and call["negative_prompt"] == ""
    # true CFG engages only above guidance 1 and preserves explicit negatives when engaged.
    backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 4.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"
    backend.generate(prompt = "a sloth", steps = 4, guidance = 1.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None


def test_guidance_scale_families_get_no_injected_negative(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)  # z-image owns guidance_scale CFG handling
    backend.generate(prompt = "a sloth", steps = 4, guidance = 4.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None


def test_flux_neither_receives_nor_reports_a_negative_prompt(fake_runtime, tmp_path, monkeypatch):
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "FluxPipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "FluxTransformer2DModel", _FakeTransformer, raising = False)
    _no_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "flux.1")
    assert backend.status()["supports_negative_prompt"] is False
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 3.5)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None
    assert out["negative_prompt"] is None


def test_qwen_reports_no_negative_when_true_cfg_is_off(fake_runtime, tmp_path, monkeypatch):
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "QwenImagePipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _FakeTransformer, raising = False)
    _no_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image")
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 1.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None
    assert out["negative_prompt"] is None
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 4.0)
    assert out["negative_prompt"] == "blurry"


def test_sdxl_reports_no_negative_when_cfg_is_off(fake_runtime, tmp_path):
    backend = _loaded_backend(
        tmp_path,
        gguf_filename = "sdxl.safetensors",
        base_repo = None,
        family_override = "sdxl",
    )
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 1.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None
    assert out["negative_prompt"] is None
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 1.5)
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"
    assert out["negative_prompt"] == "blurry"


def test_cfg_family_reports_the_negative_prompt_it_applied(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    assert backend.status()["supports_negative_prompt"] is True
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 0.0)
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None
    assert out["negative_prompt"] is None
    out = backend.generate(prompt = "a sloth", negative_prompt = "blurry", steps = 4, guidance = 0.5)
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"
    assert out["negative_prompt"] == "blurry"


class _IdeogramScheduleFakePipe(_FakePipe):
    def __call__(
        self,
        *,
        prompt = None,
        mu = 0.0,
        std = 1.5,
        guidance_schedule = "card",
        **kwargs,
    ):
        return super().__call__(
            prompt = prompt, mu = mu, std = std, guidance_schedule = guidance_schedule, **kwargs
        )


def test_generate_ideogram_defaults_follow_comfy_template(fake_runtime, tmp_path):
    """Ideogram 4 ComfyUI preset; other guidance stays constant, explicit 48 / 7 keeps the card taper."""
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    pipe = _IdeogramScheduleFakePipe()
    object.__setattr__(backend._state, "pipe", pipe)
    backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 20, guidance = 7.0)
    call = pipe.last_kwargs
    assert call["guidance_scale"] is None
    assert call["guidance_schedule"] == [7.0] * 17 + [3.0] * 3
    assert (call["mu"], call["std"]) == (0.0, 1.75)
    backend.generate(prompt = "a sloth", width = 512, height = 512, steps = 20, guidance = 7.0)
    assert pipe.last_kwargs["guidance_schedule"] == [7.0] * 14 + [3.0] * 6
    backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 20, guidance = 5.0)
    call = pipe.last_kwargs
    assert call["guidance_scale"] == 5.0 and call["guidance_schedule"] is None
    assert (call["mu"], call["std"]) == (0.0, 1.75)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 7.0)
    call = pipe.last_kwargs
    assert call["guidance_schedule"] == "card" and (call["mu"], call["std"]) == (0.0, 1.5)


def test_generate_ideogram_step_count_picks_its_comfy_preset(fake_runtime, tmp_path):
    # 48 steps at another guidance keeps the Quality preset (std 1.5, the pipeline default), 12 is Turbo.
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    pipe = _IdeogramScheduleFakePipe()
    object.__setattr__(backend._state, "pipe", pipe)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 5.0)
    call = pipe.last_kwargs
    assert call["guidance_scale"] == 5.0 and (call["mu"], call["std"]) == (0.0, 1.5)
    backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 12, guidance = 7.0)
    call = pipe.last_kwargs
    assert (call["mu"], call["std"]) == (0.5, 1.75)
    assert len(call["guidance_schedule"]) == 12 and call["guidance_schedule"][-1] == 3.0


class _ShiftSchedulerConfig(dict):
    pass


class _ShiftFakeScheduler:
    def __init__(self, **config):
        self.config = _ShiftSchedulerConfig(config)

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(**{**config, **overrides})


class _QwenShiftFakePipeline(_FakePipeline):
    @classmethod
    def from_pretrained(cls, base, **kwargs):
        pipe = super().from_pretrained(base, **kwargs)
        pipe.scheduler = _ShiftFakeScheduler(
            shift = 1.0, use_dynamic_shifting = True, shift_terminal = 0.02
        )
        return pipe


def test_qwen_load_samples_at_comfy_static_shift(fake_runtime, tmp_path, monkeypatch):
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "QwenImagePipeline", _QwenShiftFakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "QwenImageTransformer2DModel", _FakeTransformer, raising = False)
    _no_cache(monkeypatch)
    backend = _loaded_backend(tmp_path, family_override = "qwen-image")
    cfg = backend._state.pipe.scheduler.config
    assert (cfg["shift"], cfg["use_dynamic_shifting"], cfg["shift_terminal"]) == (3.1, False, None)
