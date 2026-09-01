# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion backend.

The family helpers are pure functions, tested directly. The backend lifecycle is
exercised with ``torch`` / ``diffusers`` stubbed via ``sys.modules`` so no real
GPU, weights, or network access is needed (sub-second, CI-friendly).
"""

from __future__ import annotations

import contextlib
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
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    sd_cpp_companion_only_repo_ids,
    supported_family_names,
    upstream_is_gated,
)




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
    # Reject keywords and short aliases match whole path segments, not substrings.
    assert detect_family("/models/edited/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Z-Image-Edition-GGUF").name == "z-image"
    assert detect_family("/models/kontextual/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").name == "qwen-image-edit"
    assert detect_family("unsloth/FLUX.1-Kontext-dev-GGUF").name == "flux.1-kontext"
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-2512-Inpaint") is None


def test_detect_family_edit_keyword_scoped_to_basename():
    from core.inference.diffusion_families import detect_family_for_pick

    assert detect_family("/models/edit") is None  # the dir alone is ambiguous
    assert detect_family_for_pick("/models/edit", "Z-Image-Turbo-Q4.gguf").name == "z-image"
    assert detect_family_for_pick("/models/inpaint", "qwen-image-2512-Q4.gguf").name == "qwen-image"
    assert detect_family_for_pick("/models/misc", "Qwen-Image-Layered-Q4.gguf") is None


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
    assert len(_MIRROR_PAIRS) == 24
    for upstream, mirror in _MIRROR_PAIRS:
        assert mirror_repo(upstream) == mirror
        assert canonical_base(mirror) == upstream
        assert mirror_repo(upstream.upper()) == mirror
    # A base with no mirror is left alone. HunyuanImage 2.1 is deliberate: its licence excludes the
    # EU, UK and South Korea from distribution, which a public Hub repo cannot honour.
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
    assert len(_UNGATED_MIRROR_PAIRS) == 12
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
    # Image side only: the video bases are ungated AND out of this table's scope.
    images = catalog.split("export const IMAGE_CATALOG", 1)[1].split(
        "export const VIDEO_CATALOG", 1
    )[0]
    offered = set(re.findall(r'bf16Pipeline\(\s*"([^"]+)"', images))
    mirrored = {u.lower() for u, _m in _MIRROR_PAIRS}
    # Anything under unsloth/ is already ours, and the deliberate Hunyuan exception is recorded
    # beside the table with the territorial reason it cannot be mirrored.
    missing = sorted(
        repo
        for repo in offered
        if not repo.lower().startswith("unsloth/")
        and "hunyuan" not in repo.lower()
        and repo.lower() not in mirrored
    )
    assert not missing, f"catalog offers these vendor bases with no unsloth mirror: {missing}"


def test_the_qwen_2512_mirror_covers_the_card_tag_route(monkeypatch):
    """#8001: the 2512 companions come from a repo the family table never names.

    ``unsloth/Qwen-Image-2512-GGUF`` carries ``base_model: Qwen/Qwen-Image-2512`` and
    ``_resolve_base_repo`` trusts that tag, so the fetch lands on the vendor repo whatever the
    family default says. The mirror is the only thing that redirects it.
    """
    _no_cache(monkeypatch)
    assert mirror_repo("Qwen/Qwen-Image-2512") == "unsloth/Qwen-Image-2512"
    assert prefer_ungated_mirror("Qwen/Qwen-Image-2512") == "unsloth/Qwen-Image-2512"
    # The family default is a DIFFERENT repo with its own mirror, so the redirect here is the card
    # tag's own and cannot be mistaken for the fallback doing the work.
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

    seed(other, first)  # fetched before the cache-folder change
    seed(live, second)  # fetched after it
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(live))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(other))

    wanted = (first, second)
    assert _upstream_is_cached(repack, wanted, other_root = True) is True
    assert prefer_cached_legacy_source(mirror, wanted) == repack

    # The live root alone is still the live root alone: a from_pretrained pinned to it cannot see
    # the other one, so the default probe must NOT count the split.
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
        lambda repo_id, dtype, hf_token = None: _FakePipe(),
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


def test_generate_refuses_when_the_model_was_replaced_since_the_snapshot(fake_runtime, tmp_path):
    """The guard in isolation: a snapshot naming another model is refused, typed (#9448)."""
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    st = backend.status()  # the route's pre-generation read
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
        assert backend._teardown_waiters == 0  # the window under test is genuinely open

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
        assert gen.is_alive()  # blocked on _generate_lock, which the load holds

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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "black-forest-labs/FLUX.1-dev",
        family_override = "z-image",
    )
    st = backend.status()
    snapshot = load_identity(st["repo_id"], st["base_repo"], st["family"])

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        family_override = "z-image",
    )
    assert backend.status()["repo_id"] == snapshot.repo_id  # repo_id alone sees no change
    with pytest.raises(DiffusionModelReplacedError) as replaced:
        backend.generate(prompt = "p", steps = 28, guidance = 3.5, expected_load = snapshot)
    assert replaced.value.actual.base_repo == "black-forest-labs/FLUX.1-schnell"


def test_load_generate_unload_gguf(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
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
    assert gen["repo_id"] == str(tmp_path)  # echoed so the route can record the model
    assert len(gen["images"]) == 1  # PIL images handed to the route for persistence
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

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = filename,
        base_repo = "base/repo",
        family_override = "z-image",
    )

    assert status["dtype"] == "float32"  # compute dtype is a separate concern
    assert status["gguf_variant"] == "Q8_0"
    assert backend.unload()["gguf_variant"] is None


def test_generate_progress_active_during_setup(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )

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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )

    def boom(self, state, loras, cancel):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", boom)

    with pytest.raises(RuntimeError, match = "setup failed"):
        backend.generate(prompt = "a sloth", steps = 4)

    assert backend.generate_progress()["active"] is False


def test_generate_progress_active_through_compile_cache_save(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )

    seen = {}

    def fake_save(ctx, *, logger = None):
        seen["progress"] = backend.generate_progress()
        return True

    monkeypatch.setattr(dmod.compile_cache, "register_shape", lambda *a, **k: None)
    monkeypatch.setattr(dmod.compile_cache, "save", fake_save)

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
        lambda target, requested, speed_active = False: ("_native_cudnn" if speed_active else None),
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    assert status["speed_mode"] == "off"
    assert status["resolved"]["speed_mode"]["value"] == "deferred"
    assert status["resolved"]["speed_mode"]["source"] == "auto"

    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    assert backend.status()["speed_mode"] == "off"  # first two stay exact eager
    backend.generate(prompt = "three")
    status3 = backend.status()
    assert status3["speed_mode"] == "default"
    assert "compiled" in status3["speed_optims"]
    assert status3["attention_backend"] == "_native_cudnn"
    assert status3["resolved"]["speed_mode"]["value"] == "default"

    backend.unload()
    status_off = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
        speed_mode = "off",
    )
    assert status_off["resolved"]["speed_mode"]["value"] == "off"
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    assert backend.status()["speed_mode"] == "off"
    backend.unload()


def test_deferred_speed_skips_when_lora_requested(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False  # mirror the real helper: engage once, then clear

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)
    monkeypatch.setattr(DiffusionBackend, "_apply_loras", lambda self, state, loras, cancel: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
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

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    backend.generate(prompt = "one", loras = [("adapter", 1.0)])
    backend.generate(prompt = "two", loras = [("adapter", 1.0)])
    backend.generate(prompt = "three")
    assert engaged == []
    backend.generate(prompt = "four")
    assert len(engaged) == 1


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
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
        attention_backend = "native",
    )
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    backend.generate(prompt = "three")  # deferred profile engages here
    status = backend.status()
    assert status["speed_mode"] == "default"
    assert "compiled" in status["speed_optims"]
    assert status["attention_backend"] is None
    assert status["resolved"]["attention_backend"]["value"] == "native"
    assert status["resolved"]["attention_backend"]["source"] == "explicit"

    backend.unload()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    assert "torch_dtype" not in _FakeImg2ImgPipeline.from_pipe_kwargs
    assert "dtype" not in _FakeImg2ImgPipeline.from_pipe_kwargs
    assert _FakeImg2ImgPipeline.recast_dtype is None
    call = _FakeImg2ImgPipe.last_kwargs
    assert call["image"] is not None
    assert call["strength"] == 0.5
    assert "width" not in call and "height" not in call  # img2img derives size from image

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
    assert _RecastingPipeline.seen == {}  # neither torch_dtype nor dtype

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
    backend.load_pipeline(str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo")
    assert backend.status()["workflows"] == ["txt2img"]
    with pytest.raises(ValueError, match = "img2img"):
        backend.generate(prompt = "x", steps = 4, init_image = _tiny_png_b64())


def test_generate_rejects_conditioning_without_init_image(fake_runtime, tmp_path):
    """mask / upscale / reference all need an input image; without one they must raise a
    clear ValueError rather than silently degrading to txt2img."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    with pytest.raises(ValueError, match = "mask_image requires"):
        backend.generate(prompt = "x", steps = 4, mask_image = _mask_b64(64))
    with pytest.raises(ValueError, match = "upscale requires"):
        backend.generate(prompt = "x", steps = 4, upscale = 2.0)
    with pytest.raises(ValueError, match = "reference_images require"):
        backend.generate(prompt = "x", steps = 4, reference_images = [_tiny_png_b64()])


def test_generate_rejects_reference_on_unsupported_family(fake_runtime, tmp_path):
    """A non-reference family rejects reference_images instead of silently dropping them."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    assert "upscale" in backend.status()["workflows"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a crisp photo",
        steps = 4,
        guidance = 0.0,
        seed = 3,
        init_image = _tiny_png_b64(),
        upscale = 2.0,  # 64 -> 128, no explicit strength
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
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (256, 256)  # 64 * 4 (capped)

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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    with pytest.raises(ValueError, match = "too large"):
        backend.generate(prompt = "x", steps = 4, init_image = _png_b64(4112))  # > 4096/side


def test_upscale_output_is_capped(fake_runtime, tmp_path):
    """Upscale bounds the absolute output side to 2048 even when input*factor exceeds it, so a
    large upload at 4x can't OOM the VAE/transformer."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png_b64(186), strength = 0.5)
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (192, 192)


def test_inpaint_snaps_image_and_mask_together(fake_runtime, tmp_path):
    """Inpaint snaps the odd-sized input to /16 AND resizes the mask to match, so the image
    and mask stay aligned (a mismatch would crash the inpaint pipeline)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "flux.2-klein",
    )
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
    assert call["width"] == 768 and call["height"] == 512  # OUTPUT size = sliders, not input
    assert "strength" not in call  # reference conditioning has no strength
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
    assert isinstance(img_arg, list) and len(img_arg) == 3  # primary + 2 extras

    backend.generate(
        prompt = "repaint here",
        steps = 6,
        seed = 2,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.8,
    )
    assert _FakeInpaintPipeline.built_from is loaded_pipe  # built via from_pipe off the load
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    assert _FakeInpaintPipeline.recast_dtype is None  # reused modules, never recast (#9186)
    assert _FakeImg2ImgPipeline.built_from is None
    call = _FakeInpaintPipe.last_kwargs
    assert call["image"] is not None and call["mask_image"] is not None
    assert call["strength"] == 0.7
    assert "width" not in call and "height" not in call  # inpaint derives size from image


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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    buf = io.BytesIO()
    Image.new("RGB", (96, 64), (10, 20, 30)).save(buf, format = "PNG")  # non-square, non-slider
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
    monkeypatch.setattr(diff.compile_cache, "save", lambda ctx, *, logger = None: True)
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
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
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        family_override = "qwen-image-edit",
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
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
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "sdxl.safetensors", family_override = "sdxl"
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
    bad_base = tmp_path / "bare-base"
    bad_base.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(bad_base),
        )
    (tmp_path / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(
        "unsloth/Qwen-Image-2512-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = str(tmp_path),
    )
    assert fam is not None


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


def test_detect_family_rejects_layered():
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF") is None
    assert detect_family("unsloth/qwen_image_layered") is None


def test_failed_load_rolls_back_eager_patches(fake_runtime, tmp_path, monkeypatch):
    """A load failure AFTER the eager patches install but BEFORE the _LoadState commit must
    roll the process-wide patches back, so the next bit-identical `off` load is not
    contaminated (the asymmetric-cleanup bug the reviewers flagged)."""
    from core.inference import diffusion as diff_mod
    from core.inference import diffusion_eager_patches as ep

    (tmp_path / "model.gguf").write_bytes(b"x")
    ep.uninstall_patches()  # clean slate

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
            speed_mode = "eager",  # != off -> installs the shared patches
        )
    assert ep.is_installed() is False  # rolled back by the load-failure finally
    assert backend.is_loaded is False


def test_cpu_offload_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        base_repo = "base/repo",
        cpu_offload = True,
    )
    assert status["cpu_offload"] is False


def test_low_vram_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        base_repo = "base/repo",
        memory_mode = "low_vram",
    )
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
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            family_override = "z-image",
            base_repo = "base/repo",
            speed_mode = "max",
        )
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



from core.inference.diffusion import _LoadingState, _LoadState  # noqa: E402


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
    assert p["bytes_downloaded"] == 300  # summed across repo + base
    assert abs(p["fraction"] - 0.3) < 1e-9

    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 500))
    assert backend.load_progress()["phase"] == "finalizing"  # 1000/1000


def test_load_progress_counts_a_mirrored_pipeline_repo_once(monkeypatch):
    # base_repo == repo_id, so count once. Summing adds the upstream's stale partial blobs -- the
    # very thing that selects the mirror -- to the mirror's live bytes, pegging the bar at 100%.
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
    assert p["bytes_downloaded"] == 500  # the mirror alone, not 1100
    assert p["phase"] == "downloading"  # not "finalizing"


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
    # Excluded: the GGUF supplies the transformer, and docs/assets are never fetched, so counting
    # them would peg the bar short of 100%.
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
    p = backend.load_progress()  # summed 1800 > expected 1000
    assert p["phase"] == "finalizing"
    assert p["fraction"] == 1.0
    assert p["bytes_downloaded"] == 1000  # clamped to the estimate


def test_estimate_eta():
    from core.inference.diffusion import _estimate_eta

    assert _estimate_eta(8, 1, first_step_at = 100.0, now = 100.0) is None
    assert _estimate_eta(8, 0, first_step_at = 0.0, now = 100.0) is None
    assert _estimate_eta(8, 4, first_step_at = 100.0, now = 103.0) == 4.0
    assert _estimate_eta(8, 8, first_step_at = 100.0, now = 107.0) == 0.0


def test_generate_qwen_uses_true_cfg_scale(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Qwen/Qwen-Image",
        family_override = "qwen-image",
    )
    backend.generate(prompt = "a sloth", guidance = 4.0)
    call = backend._state.pipe.last_kwargs
    assert call["true_cfg_scale"] == 4.0 and call["guidance_scale"] is None


def _load_ideogram(backend, tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    backend.load_pipeline(str(tmp_path), family_override = "ideogram-4")


def test_ideogram_rejects_single_file_and_gguf_kinds(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        backend.load_pipeline(
            str(tmp_path), gguf_filename = "model.gguf", family_override = "ideogram-4"
        )
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.safetensors",
            model_kind = "single_file",
            family_override = "ideogram-4",
        )


def test_generate_ideogram_defaults_keep_recommended_schedule(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 7.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] is None  # not passed: the pipe default engages
    assert "guidance_schedule" not in call


def test_generate_ideogram_custom_guidance_nulls_schedule(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 20, guidance = 5.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 5.0
    assert "guidance_schedule" in call and call["guidance_schedule"] is None


def _load_lumina(backend, tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    backend.load_pipeline(str(tmp_path), family_override = "lumina-2")


def test_generate_lumina2_passes_cfg_trunc_ratio(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    _load_lumina(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 50, guidance = 4.0)
    call = backend._state.pipe.last_kwargs
    assert call["cfg_trunc_ratio"] == 0.25
    assert call["guidance_scale"] == 4.0


def test_generate_other_family_never_passes_cfg_trunc_ratio(fake_runtime, tmp_path):
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    monkeypatch.setattr(
        DiffusionBackend, "load_pipeline", lambda self, **k: __import__("time").sleep(0.2)
    )
    before = set(threading.enumerate())
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    with pytest.raises(RuntimeError):
        backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    # Drain the worker while the stubs above still make it exit in 0.2s: begin_load's thread is
    # fire-and-forget, so left running it runs the REAL load_pipeline under another test's patches.
    for thread in set(threading.enumerate()) - before:
        thread.join(timeout = 5)


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
    backend._active_generate_cancel = live_cancel  # a generation from the CURRENT model
    token = 11
    backend._load_token = token + 1  # this load has already been superseded
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )
    assert not live_cancel.is_set()  # the live generation was left untouched


def test_pick_dtype_bf16_only_on_ampere(fake_runtime, monkeypatch):
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
    backend = DiffusionBackend()
    started = threading.Event()
    seen: list[threading.Event] = []

    def _capture(**kwargs):
        seen.append(kwargs["_cancel_event"])
        started.set()

    monkeypatch.setattr(backend, "_run_load", _capture)  # skip the download thread's work
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    assert started.wait(5)
    first = seen[0]

    backend.unload()  # cancels the worker holding `first`
    assert first.is_set()

    started.clear()
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q8_0.gguf")
    assert started.wait(5)
    second = seen[1]

    assert second is not first, "each load needs its own event, not a clear() of the shared one"
    assert not second.is_set()  # the replacement load starts uncancelled
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # Second arg is the staged snapshot dir, unused here: what this pins is that the MIRROR id
        # is the one sized.
        staticmethod(
            lambda base, staged = None: 8 * 1024 * 1024 if base == "unsloth/FLUX.1-dev" else 0
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
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Z-Image-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.2-klein-4B-GGUF").fp16_incompatible is False


def test_resolve_compute_dtype_promotes_fp16_for_zimage(fake_runtime):
    torch = sys.modules["torch"]
    z = detect_family("unsloth/Z-Image-GGUF")
    q = detect_family("unsloth/Qwen-Image-GGUF")
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    assert _resolve_diffusion_compute_dtype(z, torch.bfloat16) is torch.bfloat16
    assert _resolve_diffusion_compute_dtype(z, torch.float32) is torch.float32
    assert _resolve_diffusion_compute_dtype(q, torch.float16) is torch.float16
    assert _resolve_diffusion_compute_dtype(None, torch.float16) is torch.float16


def test_load_promotes_fp16_to_fp32_for_zimage_only(fake_runtime, monkeypatch, tmp_path):
    torch = sys.modules["torch"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")

    z = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
    )
    assert z["device"] == "cuda" and z["dtype"] == "float32"
    assert str(_FakeTransformer.last["torch_dtype"]) == "torch.float32"

    q = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "qwen-image"
    )
    assert q["dtype"] == "float16"  # fp16-compatible family keeps fp16 on pre-Ampere


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
    assert started.wait(5)  # the denoise is in flight, holding only _generate_lock

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
                if self._interrupt:  # diffusers' interrupt protocol
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
    assert at_step0.wait(5)  # step 0's callback ran with no cancel pending
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
    assert started.wait(5)  # generation in flight, holding _generate_lock

    (tmp_path / "m.gguf").write_bytes(b"x")
    load_done = threading.Event()

    def _load():
        backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
        load_done.set()

    lt = threading.Thread(target = _load)
    lt.start()

    assert not load_done.wait(0.5)
    assert backend._active_generate_cancel is not None
    assert backend._active_generate_cancel.is_set()

    release.set()  # the blocked denoise returns; generate() sees cancel and raises
    gt.join(5)
    assert load_done.wait(5)  # only now does the replacement allocate
    assert "exc" in gen_out and "cancelled" in str(gen_out["exc"]).lower()
    assert backend.status()["loaded"] is True
    assert backend.status()["repo_id"] == str(tmp_path)




def test_load_reports_memory_plan_fields_on_cpu(fake_runtime, tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
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


def test_load_memory_mode_balanced_streams_or_falls_back(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "balanced"
    )
    assert status["offload_policy"] in ("group", "model") and status["cpu_offload"] is True
    assert status["memory_mode"] == "balanced"
    assert backend._state.pipe.offloaded is True  # model-offload fallback engaged


def test_load_memory_mode_low_vram_engages_model_offload(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "low_vram"
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True
    pipe = backend._state.pipe
    assert pipe.offloaded is True and pipe.moved_to is None  # offload owns placement


def test_load_refines_component_placement_after_text_encoder_quantization(
    fake_runtime, tmp_path, monkeypatch
):
    from core.inference import diffusion as dmod
    from core.inference.diffusion_precision import TEQuantOutcome

    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    seen = {"quantized": False, "refined": False}

    def _quantize(*args, **kwargs):
        seen["quantized"] = True
        # The real pass reports what it did and the loader reads `.mode` off that report, so a bare
        # None is a shape production can no longer return; a None mode means encoders left dense.
        return TEQuantOutcome(None)

    def _refine(pipe, plan):
        assert seen["quantized"] is True
        assert pipe is not None and plan.offload_policy == "model"
        seen["refined"] = True
        return plan

    monkeypatch.setattr(dmod, "quantize_text_encoders", _quantize)
    monkeypatch.setattr(dmod, "refine_memory_plan_for_components", _refine)
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        memory_mode = "low_vram",
    )
    assert seen == {"quantized": True, "refined": True}


def test_load_explicit_cpu_offload_engages_model_offload_on_cuda(
    fake_runtime, tmp_path, monkeypatch
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", cpu_offload = True
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True


def test_load_speed_mode_gguf_auto_defaults_and_explicit(
    fake_runtime, tmp_path, allow_precision_fallback
):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["speed_mode"] == "default"
    status_off = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "off"
    )
    assert status_off["speed_mode"] == "off" and status_off["speed_optims"] == []
    status2 = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "max"
    )
    assert status2["speed_mode"] == "max"
    assert status2["text_encoder_quant"] is None
    status3 = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        text_encoder_quant = "nvfp4",
    )
    # Under the CPU stub nvfp4 is unsupported, so it engages nothing; the legacy escape hatch is set,
    # while the strict default refuses the load (see the refuses_when_nothing_engaged test).
    assert status3["text_encoder_quant"] is None
    resolved_te = status3["resolved"]["text_encoder_quant"]
    assert resolved_te["requested"] == "nvfp4" and resolved_te["value"] == "off"
    assert resolved_te["status"] == "unsupported"


def test_load_fast_mode_stays_resident_on_cuda(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "fast"
    )
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: scheme
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    def _quantize(pipe, target, *, mode, **kw):
        calls["quantize"] += 1
        calls["quant_mode"] = mode
        return scheme

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    return calls


def test_default_load_autos_dense_gate_and_falls_back(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    consulted = {"n": 0}

    def _supported(*a, **k):
        consulted["n"] += 1
        return False

    monkeypatch.setattr(dmod, "dense_transformer_supported", _supported)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "none",
    )
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        speed_mode = "off",
    )
    assert status["transformer_quant"] is None
    assert status["speed_mode"] == "off"
    assert _FakeTransformer.last["path"]


def test_transformer_quant_dense_path_engaged(fake_runtime, tmp_path, monkeypatch):
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    assert status["speed_mode"] == "default"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1
    assert calls["quant_mode"] == "fp8"
    assert calls["fp_kwargs"]["subfolder"] == "transformer"  # dense transformer subfolder
    assert _FakeTransformer.last == {}
    assert backend._state.pipe.moved_to == "cuda"
    assert status["offload_policy"] == "none"


def test_transformer_quant_prequant_path_engaged(fake_runtime, tmp_path, monkeypatch):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1  # dense path ran
    assert _FakeTransformer.last == {}  # GGUF not used


def test_prequant_failure_never_pulls_unprefetched_dense_shards(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    # The prefetch skips the base repo's transformer/ shards whenever a prequant is expected, so a
    # failed prequant fetch would send from_pretrained after them inside the load lock, after
    # eviction and past a 100% progress report.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    monkeypatch.setattr(dmod, "load_prequantized_transformer", lambda *a, **k: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        _transformer_prefetched = False,
    )
    assert calls["from_pretrained"] == 0  # no dense shard pull under the lock
    assert calls["quantize"] == 0
    assert status["transformer_quant"] is None  # dropped to the GGUF build
    assert _FakeTransformer.last["path"]


def test_run_load_flags_the_transformer_prefetched_from_the_staged_file_list(monkeypatch):
    # The gate is only as good as its input, so load_pipeline reads what the prefetch ACTUALLY
    # staged; a failed size estimate returns no base files and must close the fallback the same way.
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
        ([], False),  # size estimate failed: nothing staged, so nothing may be materialised
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
    # A base given as a local diffusers DIRECTORY has no Hub listing (model_info raises on a path) yet
    # its shards are already there, so reading that empty list as "the plan refused the shards"
    # declines the fast path for weights the user already has.
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
    # The widening turns on which repo the fetch resolves to and on whether EVERY transformer shard
    # is cached, and only the base repo's listing answers either, so the estimate takes a callable
    # and hands it that listing, split either side of transformer/.
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
    # resolve_dense_quant_candidate returns None from a gate sized for the DOWNLOAD a dense build
    # would make, and a prequant already on disk downloads nothing; reading that None as "dense"
    # sent a ready checkpoint to the GGUF over bytes it was never going to fetch.
    from core.inference import diffusion as dmod

    _stub_hosted_prequant(monkeypatch, cached = True)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert len(_dense_calls(calls, backend)) == 1




def _stub_declining_dense_quant(backend, monkeypatch):
    """Reach the dense fast path, then have the quantiser decline (the NVIDIA scenario: FP8 asked
    for, transformer FP8 disabled at runtime, the Q4_K_M GGUF loaded instead)."""
    from core.inference import diffusion as dmod

    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
    # The headline of P1-2: FP8 requested on a Q4_K_M GGUF, transformer FP8 declined.
    backend = DiffusionBackend()
    _stub_declining_dense_quant(backend, monkeypatch)
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        family_override = "z-image",
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
    # `auto` delegates the choice, so a decline is the ladder working as designed: no refusal, and
    # the record stays a plain "Auto: OFF" with nothing requested.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
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
    # Same decline, strict default: the load stops with an actionable reason rather than producing
    # images at a precision nobody asked for, and nothing is left half-loaded.
    backend = DiffusionBackend()
    _stub_declining_dense_quant(backend, monkeypatch)
    (tmp_path / "z-image-turbo-Q4_K_M.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "z-image-turbo-Q4_K_M.gguf",
            family_override = "z-image",
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
    # The host-level impossibilities are caught BEFORE the background load starts, so the route can
    # answer 409 instead of evicting a working model and failing several GB later.
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
    # Measured on a B200 whose torchao could not import: every explicit scheme was refused as "not
    # usable for family ... on this GPU".
    from core.inference import diffusion as dmod
    import core.inference.diffusion_transformer_quant as tq

    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    # The device clears the dense-path bar; the SCHEME still comes back None, which is the exact
    # shape of a host whose torchao cannot load its kernels.
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
    # The text encoder is the other half of "the requested precision": the CPU stub cannot cast it,
    # and that used to return None with no log line at all.
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
    # An encoder mode that cast NOTHING leaves a dense bf16 encoder the caller did not ask for, so
    # the load stops.
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
    # int8 without a measured keep-bf16 schedule becomes fp8: the encoder IS quantised, just not the
    # way asked, so this WARNS through the resolved record instead of stopping the load.
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

    # Hold the worker thread at its first instruction, so `loaded` CANNOT be True yet. begin_load
    # documents "Returns at once" and the assertion below used to race the daemon thread for it;
    # blocking the worker makes the claim unraceable.
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
    assert started["loaded"] is False  # returned without waiting on the load
    assert entered.wait(30), "begin_load never started the load thread"
    # The load is still in flight, checked from the caller: an assertion that fails on a non-main
    # thread does not fail the test, so that version passed against a begin_load that joined.
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        memory_mode = "low_vram",
    )
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] is None  # dense quant skipped
    assert status["offload_policy"] == "none"  # GGUF loaded resident, not offloaded
    assert _FakeTransformer.last["path"]


def test_dense_quant_prequant_proceeds_but_forbids_dense_fallback(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # Scoped to this backend: begin_load runs on a daemon thread and _plan_memory is patched on
        # the CLASS, so counting every instance lets a stray load land in this assertion.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(True)
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        return None, None  # fall through to GGUF; we only assert the path was reached

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert dense_refit_ran == [True]  # the re-check runs (it gates the fallback)...
    assert attempted == [False]  # ...fast path still attempted, dense fallback forbidden


def test_dense_quant_replan_retries_once_on_transient_free_undercount(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "int8",
    )
    assert replan_calls == [True, True]  # declined once, retried once
    assert attempted == [False]  # fast path attempted; prequant-sized plan forbids dense fallback


def test_dense_quant_replan_no_retry_when_capacity_truly_short(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "int8",
    )
    assert replan_calls == [True]  # genuine capacity shortfall: declined without a retry


def _decline_dense_quant(backend, monkeypatch, tmp_path):
    """Configure the harness so the dense-quant fast path is declined for capacity
    (mirrors test_dense_quant_replan_no_retry_when_capacity_truly_short)."""
    import dataclasses

    from core.inference import diffusion as dmod

    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            transformer_quant = "int8",
            loras = [("adapter", 1.0)],
        )


def test_declined_dense_without_loras_still_falls_back_to_gguf(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    backend = DiffusionBackend()
    _decline_dense_quant(backend, monkeypatch, tmp_path)
    result = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "int8",
        loras = [("adapter", 0.0)],
    )
    assert result is not None
    assert backend.status()["transformer_quant"] is None  # GGUF-as-is fallback


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
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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


def test_baked_lora_names_survive_being_disabled_at_generate_time(monkeypatch):
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
        # Spelled out rather than swallowed by **kwargs: this double pins the production signature,
        # so the keyword that keeps the no-download promise has to be one _assemble_pipe really
        # passes.
        local_files_only = False,
    ):
        calls["base"] = base
        calls["transformer"] = transformer
        calls["local_files_only"] = local_files_only
        return Pipe()

    monkeypatch.setattr(dmod, "load_krea2_pipeline", fake_loader)
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
        # Default here (a direct call), but PASSED rather than left to the loader's own default: the
        # parameter this test binds is what an API-initiated load flips.
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # Scoped to this backend: begin_load runs on a daemon thread and _plan_memory is patched on
        # the CLASS, so counting every instance asserted [True, True] on slower CI runners.
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: None
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not download when the scheme is unsupported")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
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
    # This test is about the mode/policy gates, so keep the "second denoiser" verdicts out of it:
    # the base shards are on disk, so an auto quant reaches the candidate like an explicit one.
    _stub_dense_transformer_cached(monkeypatch, cached = True)

    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is True
    assert seen[-1] == "fp8"
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert seen[-1] == "auto"
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
    fallback_filename = "transformer_fp8.pt",
)


def _stub_hosted_prequant(monkeypatch, *, cached: bool):
    """Resolve the family's hosted fp8 checkpoint, present or absent from the cache."""
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
    # The reported bug: picking unsloth/Z-Image-GGUF fetched the GGUF and THEN a 6.29 GB hosted fp8
    # checkpoint that became the denoiser, so the GGUF was never used.
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # the GGUF the user picked


def test_auto_quant_takes_a_hosted_prequant_that_is_already_cached(
    fake_runtime, tmp_path, monkeypatch
):
    _stub_hosted_prequant(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")

    assert len(_dense_calls(calls, backend)) == 1


@pytest.mark.parametrize("loras", [[("adapter", 0.0)], [("a", 0.0), ("b", 0.0)]])
def test_all_zero_weight_loras_do_not_look_like_a_bake(loras, fake_runtime, tmp_path, monkeypatch):
    # Weight 0 is disabled everywhere else, so plain truthiness on the list would call this a bake,
    # skip the decline and fetch the dense companion for a request that applies no adapter.
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", loras = loras
    )

    assert _dense_calls(calls, backend) == []


def test_a_weighted_lora_is_still_treated_as_a_bake(fake_runtime, tmp_path, monkeypatch):
    # The other direction, so the zero-weight fix does not turn every bake into a GGUF load: a real
    # adapter still takes the dense route.
    _stub_hosted_prequant(monkeypatch, cached = False)
    _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    with pytest.raises(RuntimeError, match = "LoRA adapters could not be applied"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            loras = [("adapter", 0.8)],
        )


def test_an_explicit_quant_request_still_downloads_the_hosted_prequant(
    fake_runtime, tmp_path, monkeypatch, allow_precision_fallback
):
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )

    assert len(_dense_calls(calls, backend)) == 1


def test_a_baked_lora_load_is_unaffected_by_the_prequant_cache(fake_runtime, tmp_path, monkeypatch):
    _stub_hosted_prequant(monkeypatch, cached = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    with pytest.raises(RuntimeError, match = "LoRA"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            loras = [("adapter", 1.0)],
        )

    assert len(_dense_calls(calls, backend)) == 1


@pytest.mark.parametrize(
    "lora_specs, consults_prequant",
    [([("adapter", 0.0)], True), ([("adapter", 0.8)], False), (None, True)],
)
def test_the_dense_builder_skips_the_prequant_only_for_a_real_bake(
    lora_specs, consults_prequant, fake_runtime, monkeypatch
):
    # This builder's own gate read the raw list as truthy and built the dense transformer for
    # adapters that apply nothing: the rule holds here too, not just at the decline.
    import contextlib

    from core.inference import diffusion as dmod

    consulted: list = []
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(
        dmod, "resolve_prequant_source", lambda *a, **k: consulted.append(1) or None
    )
    backend = DiffusionBackend()
    target = _force_cuda_target(backend, monkeypatch)

    with contextlib.suppress(Exception):  # the build cannot complete here; the gate is the subject
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
    # The candidate sizing passed force_dense on the raw list, staging base transformer/ shards
    # while the load ran the cached prequant.
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
    # /images/load sends (id, weight) tuples but download-plan passes LoraSpec models, whose
    # unpacking yields (field, value), so a plain (_lid, w) unpack binds w to ("weight", 0.0) and
    # reads a disabled adapter as an active bake.
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
    assert consulted == []  # declined on the cache verdict, never sized as a bake

    backend._dense_quant_prefetch_needed(fam, {"loras": [LoraSpec(id = "adapter", weight = 0.8)]})
    assert consulted != []


def test_the_plan_reads_zero_weight_loras_exactly_as_the_load_does(fake_runtime, monkeypatch):
    # The plan and the load must agree on what a bake is: gating the prefetch on the raw list stages
    # transformer/ shards for a load that runs the GGUF.
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
    # The plan must call it as the loader does: a declined prequant means the GGUF runs, so the
    # prefetch must not widen to the base transformer/ shards.
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
    # The reported bug: picking Qwen-Image-Edit-2511-GGUF Q6_K fetched the 16.85 GB GGUF and then
    # started a 57.72 GB pull of the base repo.
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
    _stub_hosted_prequant(monkeypatch, cached = True)  # not the verdict under test

    _stub_dense_transformer_cached(monkeypatch, cached = False)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    assert consulted == []

    _stub_dense_transformer_cached(monkeypatch, cached = True)
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert len(consulted) == 1


def test_an_explicit_transformer_quant_still_buys_the_dense_base(fake_runtime, monkeypatch):
    # The decline is for the AUTO ladder only: asking for int8/fp8 by name is opting in to the dense
    # build, so an uncached base must not silently downgrade that request to the GGUF.
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
    # The plan and the load must agree. With the shards unstaged, the fast path would fetch them
    # under the load lock after eviction, where unload cannot preempt it and progress already
    # reported 100%.
    _stub_hosted_prequant(monkeypatch, cached = True)  # not the verdict under test
    _stub_dense_candidate(monkeypatch, prequant = False)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # the GGUF the user picked

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
    # A cached pre-quant stages no transformer/ shards because the checkpoint REPLACES them, not
    # because a download was refused.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert len(_dense_calls(calls, backend)) == 1


def test_an_unstaged_prequant_load_still_forbids_the_dense_fallback(
    fake_runtime, tmp_path, monkeypatch
):
    # With no transformer/ staged, a FAILED prequant load has to raise rather than materialise the
    # dense bf16 transformer.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    seen: list = []

    def _record(self, *a, **k):
        seen.append(k.get("allow_dense_fallback"))
        return None, None

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", _record)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert seen == [False]


def test_an_uncached_prequant_still_declines_before_the_candidate_is_asked(
    fake_runtime, tmp_path, monkeypatch
):
    # An UNCACHED hosted pre-quant is still a second multi-GB denoiser, so a decline keyed on the
    # candidate alone would hand the download straight back.
    _stub_hosted_prequant(monkeypatch, cached = False)
    _stub_dense_candidate(monkeypatch, prequant = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert _dense_calls(calls, backend) == []
    assert status["transformer_quant"] is None


def test_a_resolver_with_no_answer_reads_as_the_dense_base(fake_runtime, tmp_path, monkeypatch):
    # None with no size entry is "no basis at all", not "a prequant"; the plan declines to stage
    # transformer/ in that case too.
    from core.inference import diffusion as dmod

    _stub_hosted_prequant(monkeypatch, cached = True)
    # No prequant source at all, so the None hides no cached checkpoint.
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert _dense_calls(calls, backend) == []


def test_a_raising_resolver_reads_as_the_dense_base(fake_runtime, tmp_path, monkeypatch):
    # A probe that cannot answer must not license loading shards nobody staged, nor take the load down with it.
    from core.inference import diffusion as dmod

    def _boom(**kw):
        raise RuntimeError("resolver is on fire")

    _stub_hosted_prequant(monkeypatch, cached = True)
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", _boom)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True


def test_the_plan_and_the_load_agree_on_a_cached_prequant(fake_runtime, tmp_path, monkeypatch):
    # The plan declining to stage transformer/ and the load still taking the fast path is the
    # disagreement this branch prevents.
    _stub_hosted_prequant(monkeypatch, cached = True)
    _stub_dense_candidate(monkeypatch, prequant = True)
    # Dense shards on disk too, so the plan declines for the one reason under test: a prequant needs
    # no transformer/.
    _stub_dense_transformer_cached(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    (tmp_path / "m.gguf").write_bytes(b"x")

    assert backend._dense_quant_prefetch_needed(fam, {"base_repo": "Tongyi-MAI/Z-Image-Turbo"}) is (
        False
    )
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )
    assert len(_dense_calls(calls, backend)) == 1


def test_dense_transformer_cached_asks_the_repo_the_fetch_will_use(
    fake_runtime, monkeypatch, tmp_path
):
    # prefer_ungated_mirror picks from the WIDENED listing (companions plus shards), because shards
    # resident under the upstream id spare nothing when the fetch resolves to the mirror.
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
    # Judging the mirror decision on the companions alone declined the fast path for weights already
    # on disk, one repo over.
    from core.inference import diffusion as dmod

    companions = ("vae/config.json", "text_encoder/model.safetensors")
    shards = ("transformer/diffusion_pytorch_model-00001-of-00001.safetensors",)
    upstream_cache = set(companions)
    mirror_cache = set(shards)

    # The real rule: keep the upstream only when its cache holds every file about to be fetched.
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
    # A cancelled pull leaves partial files; reading that as a cache hit downloads the REST of the
    # transformer for a pick whose GGUF is already on disk.
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
    # An unreadable cache must read as "not cached" rather than raise out of the download plan.
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
    # Reported bug: a Q8_0 GGUF showed "BF16" because dtype is the pipeline COMPUTE dtype;
    # gguf_variant names the file actually opened.
    backend = DiffusionBackend()
    (tmp_path / "z-image-turbo-Q8_0.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "z-image-turbo-Q8_0.gguf",
        family_override = "z-image",
    )
    status = backend.status()
    assert status["model_kind"] == "gguf"
    assert status["transformer_quant"] is None
    assert status["gguf_variant"] == "Q8_0"


def test_status_reports_the_dense_build_when_it_replaced_the_gguf(
    fake_runtime, tmp_path, monkeypatch
):
    # A GGUF pick taken over by the dense fast path denoises with a torchao build of the BASE
    # transformer, so the row must prefer transformer_quant.
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    _stub_dense_quant(monkeypatch, scheme = "fp8")
    (tmp_path / "z-image-turbo-Q8_0.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "z-image-turbo-Q8_0.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    assert backend.status()["transformer_quant"] == "fp8"


def test_status_carries_no_gguf_variant_when_nothing_is_loaded():
    # The unloaded payload must declare every key the loaded one does, or the row keeps the previous
    # model's quant after an eject.
    assert DiffusionBackend().status()["gguf_variant"] is None


def test_diffusion_status_response_carries_resolved():
    # The response model must declare resolved, or Pydantic drops the backend's per-control
    # provenance.
    from models.inference import DiffusionStatusResponse

    rec = {"transformer_quant": {"value": "fp8", "source": "auto", "reason": "blackwell"}}
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    # requested/status are additive with defaults, so a record from an older backend still parses.
    assert resp.model_dump()["resolved"] == {
        "transformer_quant": {
            "value": "fp8",
            "requested": None,
            "source": "auto",
            "status": "applied",
            "reason": "blackwell",
        }
    }
    assert DiffusionStatusResponse(loaded = False).resolved is None


def test_diffusion_status_response_carries_requested_precision():
    # A DECLINED explicit precision must survive the API boundary, so the UI can say "you asked for
    # fp8, the GGUF ran".
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
    assert resp.model_dump()["resolved"] == rec


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
    # The prefetched transformer/ shards land in the same blob cache _companion_cache_bytes sums, so
    # reading it would double-count.
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
    # The cache is inflated by the prefetched transformer; consulting it would offload.
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
    # 12000 + 8000 + 4000 + 2048 = 26048 MiB fits the budget; a double-count would have offloaded.
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
        # Excluded from the companion total: on a GGUF load the single file supplies the
        # transformer.
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
    # _companion_cache_bytes resolves a hub id under hub_cache_dir() ONLY, so a base served from the
    # import-time root budgets as zero and the plan OOMs on companions it never counted.
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
    assert blind.offload_policy == OFFLOAD_NONE

    plan = _plan(base_local_dir = str(snapshot))
    assert plan.estimates["companion_dense_mib"] == 200
    assert plan.offload_policy == OFFLOAD_GROUP


def test_plan_memory_sizes_a_pipeline_load_from_the_other_root_snapshot(monkeypatch, tmp_path):
    # Same hole on the full-pipeline branch: a repo served from the other root sizes as unknown, so
    # a 4 GiB pipeline that does not fit stays resident.
    from core.inference.diffusion_memory import OFFLOAD_MODEL, OFFLOAD_NONE

    snapshot = _other_root_base_snapshot(tmp_path, monkeypatch)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    # base_repo deliberately unequal to repo_id: the narrow-base size table is a different path.
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
    assert plan.offload_policy == OFFLOAD_MODEL


def test_plan_memory_keeps_companions_a_partial_staged_snapshot_omits(monkeypatch, tmp_path):
    # The preflight's snapshot can hold the manifest alone, so preferring it over the hub-id scan
    # budgets 0 for companions a root does hold.
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
    assert plan.estimates["companion_dense_mib"] == 200
    assert plan.offload_policy == OFFLOAD_GROUP


def test_load_progress_counts_a_checkpoint_the_other_cache_root_already_holds(
    tmp_path, monkeypatch
):
    # Counting only the live root leaves `downloaded` at 0 against a nonzero estimate, so the UI
    # shows a healthy load stalled near 0%.
    from core.inference.diffusion import _LoadingState

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    _hub_blob(other, "org/gguf", "a" * 64, 400)
    # The base is mirrored in both roots: one etag, one file, so it must not count twice.
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
    # blobs/ is append-only, so summing the whole dir counts a superseded revision's full copy and
    # load_progress reports "finalizing" for the rest of the pull.
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

    # A's bytes are not this load's; the .incomplete one is, so the bar still moves inside a shard.
    assert DiffusionBackend._cache_bytes("org/pipe") == 2200 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/pipe", base_repo = None, expected_bytes = 4000 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"
    assert progress["fraction"] == 0.55


def test_load_progress_counts_one_logical_file_across_roots_at_two_revisions(tmp_path, monkeypatch):
    # Each root serves its OWN refs/main, so one logical shard has a different etag in each and an
    # etag key sums both copies.
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
    # reuse_other_cache_root resolves EACH file through whichever root holds it, so disjoint PARTS
    # of one revision must be summed rather than sized off the larger.
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
    # Off the larger half alone the plan stays resident and the other root's 150 MiB arrives unbudgeted.
    assert plan.offload_policy != OFFLOAD_NONE
    assert plan.offload_policy == OFFLOAD_GROUP


def test_companion_bytes_skip_a_superseded_revision_in_the_same_root(tmp_path, monkeypatch):
    # The merge is per FILE and only refs/main's revision is read, else a repo that repacked its
    # shards would count both namings.
    live, _other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _sparse_snapshot_file(live, "bfl/base", old_rev, f"text_encoder/{shard}", 100)
    _sparse_snapshot_file(live, "bfl/base", new_rev, "text_encoder/model.safetensors", 200)
    _hub_ref(live, "bfl/base", new_rev)

    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 200 * 1024 * 1024


def test_plan_memory_sizes_a_pipeline_split_across_both_cache_roots(monkeypatch, tmp_path):
    # A prefetch split across roots hands back NO snapshot, so the plan falls back to a cache scan
    # scoped to the live root and misses shards from_pretrained will load.
    from core.inference.diffusion_memory import OFFLOAD_MODEL

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    # base_repo deliberately unequal to repo_id: the narrow-base size table is a different path.
    fam = types.SimpleNamespace(name = "flux.1", base_repo = "unrelated/repo")
    _hub_blob(live, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "b" * 64, 4000)

    def _plan(**kw):
        return backend._plan_memory(
            target, None, "bfl/base", fam, None, False, kind = "pipeline", repo_id = "bfl/base", **kw
        )

    plan = _plan()
    # Each file counted once: a per-root sum would read 4600, the live root alone 300.
    assert plan.estimates["model_dense_mib"] == 4300
    assert plan.offload_policy == OFFLOAD_MODEL

    # The gated preflight excuses a base off ONE probe file, so preferring a staged dir outright
    # would size this 4.2 GiB pipeline at 0.
    manifest_only = other / "models--bfl--base" / "snapshots" / ("c" * 40)
    manifest_only.mkdir(parents = True)
    (manifest_only / "model_index.json").write_bytes(b"{}")
    assert _plan(base_local_dir = str(manifest_only)).estimates["model_dense_mib"] == 4300


def test_dense_transformer_bytes_read_the_other_root_and_treat_the_snapshot_as_a_floor(
    tmp_path, monkeypatch
):
    # A base held only under the import-time root reads 0 on the live one, and a 0 skips the fit
    # re-check, so the dense build lands under a GGUF-sized plan.
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    _safetensors_with_params(
        snapshot / "transformer" / "diffusion_pytorch_model.safetensors", 1_000_000
    )

    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base") == 2_000_000
    assert (
        DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(snapshot)) == 2_000_000
    )
    # The staged snapshot is a floor, never a replacement: it can carry companions alone.
    bare = tmp_path / "companions-only-snapshot"
    bare.mkdir()
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(bare)) == 2_000_000
    # A staged dir under NEITHER current root: the cache folder can change again mid-prefetch, and
    # the resolved snapshot is still where the load reads shards.
    stale = tmp_path / "stale-root" / "models--bfl--base" / "snapshots" / ("b" * 40)
    _safetensors_with_params(
        stale / "transformer" / "diffusion_pytorch_model.safetensors", 3_000_000
    )
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(stale)) == 6_000_000


@pytest.mark.parametrize("staged", [False, True])
def test_dense_fit_check_runs_for_a_base_the_live_cache_root_does_not_hold(
    fake_runtime, tmp_path, monkeypatch, staged, allow_precision_fallback
):
    # With no usable prequant the loader materialises the dense bf16 transformer, so the fit
    # re-check must run; for a moved cache the shards sit outside the live root, which reads 0.
    from core.inference import diffusion as dmod

    # Pin the base id so the fixture's cache dir decides the swap rather than the ambient cache.
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # Scoped to this backend: begin_load runs on a daemon thread and _plan_memory is patched on
        # the CLASS.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(transformer_resident_override_mib)
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
        _base_local_dir = str(shards) if staged else None,
    )
    assert dense_refit_ran == [12288]
    assert backend.status()["loaded"] is True


def test_the_dense_builder_reads_transformer_from_the_hub_id_not_the_staged_snapshot(
    fake_runtime, tmp_path, monkeypatch
):
    # Sizing reads the staged snapshot; the LOAD deliberately does not. diffusers treats a local dir
    # as terminal, so a partial snapshot would turn a working load into a hard failure.
    import contextlib

    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda *a, **k: None)
    _no_cache(monkeypatch)
    snapshot = tmp_path / "other-hub" / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / "abc"
    # transformer/ present but shardless: the shape an is_dir() guard would wave through.
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
    # A hub id, not the snapshot path; pinned so the ambient cache cannot decide which mirror.
    assert seen == ["unsloth/Z-Image-Turbo"]


def test_reset_step_cache_helper_is_best_effort():
    # reset_stateful_hooks lives only on the HookRegistry, so the old lookup was a silent no-op:
    # prefer the real CacheMixin hook.
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
    # FBCache residuals survive on the resident transformer, so generate() must reset first, but
    # only when a cache is engaged.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    resets = []
    # A genuine transformer exposes the diffusers CacheMixin entry point, not reset_stateful_hooks.
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
    # from_pretrained must get the local dir: its own hub sweep would re-download the root singles (24 GB per FLUX.1).
    backend = DiffusionBackend()
    backend.load_pipeline(
        "unsloth/Qwen-Image-2512-bnb-4bit",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert _FakePipeline.last["base"] == str(tmp_path)
    backend.unload()


def test_unload_waits_for_in_flight_denoise_before_teardown():
    # Regression: unload() must wait for a running denoise to exit before _unload_locked() tears
    # down process-wide state it depends on.
    import threading

    backend = DiffusionBackend()

    denoise_active = {"v": False}
    teardown_saw = []

    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    started = threading.Event()
    finish = threading.Event()

    # _generate_lock is the only lock a real denoise holds for its whole body.
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
    # unload must NOT have torn down yet: it is blocked on the denoise's _generate_lock.
    assert teardown_saw == []
    assert not unloaded.wait(0.3)

    finish.set()
    d.join(2.0)
    u.join(2.0)
    assert unloaded.is_set()
    assert teardown_saw == [False]




def _load_zimage_backend(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    # Per-image reproducibility: image i of a batched call uses the generator seed a solo replay of
    # that image uses.
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


def test_generate_non_oom_error_is_not_retried(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    pipe = _BoomPipe()
    object.__setattr__(backend._state, "pipe", pipe)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert pipe.batch_attempts == [4]


def test_generate_broadcasts_negative_prompt_across_a_mixed_prompt_batch(fake_runtime, tmp_path):
    # A prompt list needs a matching negative list: encode_prompt asserts equal lengths, else
    # batch-1 embeds meet batch-N latents.
    backend = _load_zimage_backend(tmp_path)
    backend.generate(prompt = "fallback", prompts = ["a", "b", "c"], negative_prompt = "blurry")
    call = backend._state.pipe.last_kwargs
    assert call["prompt"] == ["a", "b", "c"]
    assert call["negative_prompt"] == ["blurry", "blurry", "blurry"]
    backend.generate(prompt = "fallback", prompts = ["a", "b"])
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None


def test_generate_keeps_a_scalar_negative_prompt_off_the_list_paths(fake_runtime, tmp_path):
    # Uniform-prompt and single-image forwards pass a SCALAR prompt, so the negative prompt must
    # stay scalar too.
    backend = _load_zimage_backend(tmp_path)
    backend.generate(prompt = "a sloth", seeds = [1, 2, 3], negative_prompt = "blurry")
    assert backend._state.pipe.last_kwargs["prompt"] == "a sloth"
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"
    backend.generate(prompt = "a sloth", seed = 1, negative_prompt = "blurry")
    assert backend._state.pipe.last_kwargs["negative_prompt"] == "blurry"


class _TracingPipe(_CountingPipe):
    """Appends ``("call", n)`` to a shared trace so resets can be interleaved with forwards."""

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
    # A forward that raises skips maybe_free_model_hooks(), so its FBCache residual would stale the
    # halved retry.
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
# A FLUX-shaped base repo: the packaged root single and the transformer shards a plain
# snapshot_download would drag in and the loader never opens.
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
    # Never let a developer's real Unsloth cache make an entry disappear from these hermetic tests.
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: False),
    )


def test_download_plan_scopes_the_base_repo_files(monkeypatch):
    # The plan's file list must match what the loader reads: a full snapshot adds the 24 GB root
    # single and the shards the GGUF replaces.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    # The base entry names the MIRROR: staged before the loader runs, so a gated id here 401s an
    # anonymous user and the downstream swap is never reached.
    assert [e["repo_id"] for e in plan["entries"]] == [
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-dev",
    ]
    checkpoint, base = plan["entries"]
    assert checkpoint["files"] == ["flux1-dev-Q4_K_M.gguf"]
    assert checkpoint["bytes"] == 7 * GB
    # Only the planner can name the selected entry, so it says so outright: the companion base is
    # assets, not the pick.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert [entry["repo_id"] for entry in plan["entries"]] == ["unsloth/FLUX.1-dev"]
    assert "text_encoder/model.safetensors" in plan["entries"][0]["files"]
    assert plan["total_bytes"] == plan["entries"][0]["bytes"]
    # Cache state changes the pending download, never the selector's declared footprint.
    assert plan["required_bytes"] == 7 * GB + plan["total_bytes"]
    assert plan["checkpoint_bytes"] == 7 * GB


def test_download_plan_is_empty_when_every_required_file_is_cached(monkeypatch):
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    _all_cached(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: True),
    )

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert plan["entries"] == []
    assert plan["total_bytes"] == 0
    assert plan["required_bytes"] > 7 * GB
    assert plan["checkpoint_bytes"] == 7 * GB


def test_download_plan_sizes_the_checkpoint_when_the_base_is_the_same_repo(monkeypatch):
    # A combined repo keys both Hub lookups on one id: overwriting left the checkpoint at 0 bytes,
    # and listing it twice double counted the footprint.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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
    # model_info().sha is the REPO head, so a README commit moves it while every weight stays byte
    # identical; the size the plan declared has the last word.
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
    # Nothing declared to compare against: trust the ref rather than call a present file missing.
    assert DiffusionBackend._hub_file_is_cached(repo, name, "b" * 40, 0)


@pytest.mark.parametrize("symlink", [True, False], ids = ["posix_links", "windows_copies"])
def test_an_explicit_current_snapshot_does_not_hide_a_stale_main_ref(
    tmp_path, monkeypatch, symlink
):
    # The planner sizes revision B but the loader follows refs/main to A, so a same-size B hit must
    # not make the planner omit the file.
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
    # Naming the right commit is not proof the bytes are right: the copy layout keeps no symlink, so
    # the pinned probe must corroborate with the declared size or the load consumes a damaged entry.
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
    # An unpinned probe answers from the LOCAL main ref, so a republished companion reads present
    # and the loader fetches it outside the download manager.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    # Upstream serves this pick, so both entries keep the id the sizes were read from; a mirror swap
    # deliberately drops the pin instead.
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
    # The gate is DEFERRED exactly as _run_load defers it: called eagerly it sees no base listing
    # and always declines, so the plan scopes narrower than the load it describes.
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert seen, "the deferred gate was never called with the base listing"
    companions, transformer_files = seen[-1]
    assert transformer_files == ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors",)
    assert "text_encoder/model.safetensors" in companions
    assert not any(f.startswith("transformer/") for f in companions)
    base = next(e for e in plan["entries"] if not e["repo_id"].endswith("-GGUF"))
    assert "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" in base["files"]


def test_download_plan_pipeline_kind_is_one_entry(monkeypatch):
    # A pipeline load has no separate checkpoint repo: the repo IS the pipeline.
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _FLUX_BASE_SIBLINGS})

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert len(plan["entries"]) == 1
    files = plan["entries"][0]["files"]
    assert any(f.startswith("transformer/") for f in files)
    assert "flux1-dev.safetensors" not in files
    assert "text_encoder/model.fp16.safetensors" not in files


def test_download_plan_flags_a_mirrored_pipeline_as_the_checkpoint(monkeypatch):
    # A gated pipeline is STAGED from its ungated mirror, so deriving the label by comparing the two
    # ids made every file of the selected model read as "Required assets".
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
    # An fp8 text-encoder request loads a hosted PRE-CAST checkpoint, so the plan must stage that
    # file, not the base repo's dense encoder shards.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    # The pick resolves one hosted pre-cast encoder for text_encoder_2 (flux.1 hosts its T5-XXL).
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
    # text_encoder_2 dense weights go; text_encoder stays (no hosted artifact), as do the non-weight
    # files the pre-cast loader meta-inits from.
    assert not any(
        f.startswith("text_encoder_2/") and f.endswith(".safetensors") for f in base["files"]
    )
    assert "text_encoder/model.safetensors" in base["files"]
    assert "model_index.json" in base["files"]
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_keeps_the_dense_encoder_without_an_fp8_request(monkeypatch):
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    # An upstream that already satisfies the load keeps its id, so the plan stages the cache the
    # user already paid for.
    _all_cached(monkeypatch)
    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )
    base = next(e for e in plan["entries"] if e["repo_id"] == "black-forest-labs/FLUX.1-dev")
    assert "text_encoder/model.safetensors" in base["files"]
    assert len(plan["entries"]) == 2


def test_download_plan_keeps_the_dense_encoder_when_the_precast_repo_is_unavailable(monkeypatch):
    # A gated / renamed / unpublished artifact must NOT cost the dense encoder: the load falls back
    # to it, so the plan stages it.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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
    # Live Hub sizes: the five dense shards are the extra ~40.9 GB this regression prevents.
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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
    # A declined prequant stages neither its .pt nor the base transformer/ shards the dense build
    # wanted.
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

    # The base entry names the MIRROR: it is staged before the loader runs, so it must be the repo
    # the manager pulls from.
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
    # An explicit fp8 request loads the hosted prequant INSTEAD of the excluded base shards, so
    # leaving it out under-reports required_bytes by the whole denoiser.
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
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )

    # Counted AND staged, else the load fetches it inline under the load lock, past the manager's
    # progress, cancel and disk preflight.
    staged = {(e["repo_id"], f) for e in plan["entries"] for f in e["files"]}
    assert ("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-FP8.pt") in staged
    assert plan["required_bytes"] == plan["total_bytes"]
    # Measured against the same pick without the quant, so the delta is exactly the prequant
    # checkpoint.
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
        fallback_filename = "transformer_int8.pt",
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

    assert plan["required_bytes"] - baseline["required_bytes"] == 6 * GB
    assert any(source.filename in entry["files"] for entry in plan["entries"])


def test_download_plan_omits_the_prequant_under_a_definite_offload_policy(monkeypatch):
    # Balanced and low_vram offload BY MODE, which no replan can clear, so the load keeps the GGUF
    # and counting the hosted checkpoint overstates the footprint.
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

    # A resident-capable pick still pays for it, so the gate cannot just zero everything.
    resident = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        transformer_quant = "fp8",
    )
    assert any(f.endswith("-FP8.pt") for e in resident["entries"] for f in e["files"])


def test_download_plan_omits_the_prequant_for_an_auto_pick_at_speed_off(monkeypatch):
    # load_pipeline forces an AUTO quant to "off" under Speed="off", so nothing is fetched and the
    # footprint must not claim it.
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
    _stub_hosted_prequant(monkeypatch, cached = True)

    auto = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF", gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf", speed_mode = "off"
    )
    assert not any(f.endswith("-FP8.pt") for e in auto["entries"] for f in e["files"])

    # An EXPLICIT quant is not forced off, so it still loads the prequant and still pays for it.
    explicit = DiffusionBackend().download_plan(
        "unsloth/Z-Image-GGUF",
        gguf_filename = "Z-Image-Turbo-Q4_K_M.gguf",
        speed_mode = "off",
        transformer_quant = "fp8",
    )
    assert any(f.endswith("-FP8.pt") for e in explicit["entries"] for f in e["files"])


def test_download_plan_omits_a_prequant_an_auto_pick_would_decline(monkeypatch):
    # Auto runs the GGUF as-is rather than download an uncached hosted checkpoint, so only an
    # explicit request pays for it.
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
    # Only a GGUF pick is restricted: a full pipeline's transformer IS the repo's.
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _ZIMAGE_BASE_SIBLINGS})
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert any(f.startswith("transformer/") for f in plan["entries"][0]["files"])


def test_download_plan_restages_a_base_split_across_both_cache_roots(monkeypatch):
    # A base half in the old cache root is invisible to from_pretrained (the assembly is pinned to
    # hub_cache_dir()), so staging nothing there makes the load re-pull it inline.
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
        # roots=(live,) asks the active root; roots=(None,) asks huggingface_hub's import-time one.
        asks_live = roots is not None and roots != (None,)
        return (filename not in old_root_only) if asks_live else (filename in old_root_only)

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(probe))

    plan = DiffusionBackend().download_plan("unsloth/Z-Image-Turbo", model_kind = "pipeline")

    entry = plan["entries"][0]
    assert set(entry["files"]) == set(_ZIMAGE_BASE_SIBLINGS_BY_NAME)
    assert entry["bytes"] == sum(_ZIMAGE_BASE_SIBLINGS_BY_NAME[n] for n in old_root_only)


def test_download_plan_restages_the_old_root_half_when_other_files_are_missing_too(monkeypatch):
    # Mixed case: files absent everywhere land in the LIVE root when staged, so the repo straddles
    # both even though nothing looked split at plan time.
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
    # The live root holds an OLD copy under the right name, so reuse_other_cache_root never switches
    # and the stale copy wins; treating that as cached reports an empty plan for an unusable base.
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
            # Present in the live root but the wrong bytes; correct in the other root.
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
    # A repo matching no family resolves no companions and the fallback raised on None, 500ing the
    # picker's plan request; planning no work is the honest answer.
    _fake_hf_api(monkeypatch, {})

    plan = DiffusionBackend().download_plan(
        "someone/mixed-gguf-collection",
        gguf_filename = "totally-unknown-thing-Q4_K_M.gguf",
        model_kind = "gguf",
    )

    assert plan == {"entries": [], "total_bytes": 0, "required_bytes": 0, "checkpoint_bytes": 0}


def test_download_plan_still_plans_an_unrecognised_gguf_given_an_explicit_base(monkeypatch):
    # An explicit base_repo supplies what family detection could not, so the pick must still plan.
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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

    seen: list[int] = []
    real_unload_locked = backend._unload_locked

    def _record_then_unload():
        # Sampled while unload holds both locks: exactly the window a queued generation could slip
        # through.
        seen.append(backend._teardown_waiters)
        real_unload_locked()

    backend._unload_locked = _record_then_unload
    backend.unload()

    assert seen == [1]
    assert backend._teardown_waiters == 0


def test_a_raising_unload_still_drains_the_teardown_fence(fake_runtime, tmp_path, monkeypatch):
    # _unload_locked ends in clear_gpu_cache(), which raises on a sticky CUDA fault; without the
    # finally the fence stayed up forever, refusing every later generation.
    from core.inference import diffusion as diffusion_module

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

    real_clear = diffusion_module.clear_gpu_cache

    def _sticky(*_args, **_kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", _sticky)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        backend.unload()
    assert backend._teardown_waiters == 0, "a failed teardown must not leave the fence up"

    # The next load and generation must not be fenced out by the teardown that blew up.
    monkeypatch.setattr(diffusion_module, "clear_gpu_cache", real_clear)
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    assert backend.generate(prompt = "after", steps = 2)["images"]


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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    # A generation parked behind a teardown must not need _lock to notice Stop: Condition.wait()
    # reacquires the lock before returning, blocking the request for the length of the load.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    # The active generation drops its cancel event at the last-word check while it still owns the
    # slot, so "no cancel event" there does not mean "nothing owns the slot".
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

    from core.inference import diffusion as diffusion_module

    queued = threading.Event()
    stop_answered: list[bool] = []
    fired: list[int] = []
    real_baked = diffusion_module._baked_lora_names

    def _baked(pipe):
        # Runs in the epilogue, after the last-word check cleared _active_generate_cancel and before
        # _generation_slot releases the lock.
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
    # Publishing the cancel event only after the first timed acquisition failed left a 100 ms hole:
    # Stop answered false and the generation still ran once the load finished.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
    # In the _generate_lock handoff a waiter owns the lock before moving its cancel event, and Stop
    # must not mistake that ordinary serialized waiter for one blocked by model replacement.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    # POST /images/generate and POST /v1/images/generations both call generate() with no busy guard,
    # so the second simply waits; Stop must not fail it with the cancel sentinel.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

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
    # begin_load frees the old pipeline behind the same barrier, so it needs the same fence.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

    seen: list[int] = []
    real_unload_locked = backend._unload_locked

    def _record_then_unload():
        seen.append(backend._teardown_waiters)
        real_unload_locked()

    backend._unload_locked = _record_then_unload
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )

    assert seen == [1]
    assert backend._teardown_waiters == 0




def test_auto_retries_a_lower_scheme_that_has_a_prequant(monkeypatch):
    # Qwen-Image: auto picks fp8 but only an int8 DiT checkpoint is published, so the pick used to
    # drop to GGUF; the retry walks the rest of auto's ladder for a rung that HAS one.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.auto_scheme_candidates",
        lambda target, family = None: ("fp8", "mxfp8", "int8"),
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

    # For a GGUF pick the policy is cached-only; _uncached_prequant_repo only ever sees auto's
    # winner, so without this check the retry would smuggle an uncached repo past it.
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
    # A local override is the operator's own file, so it costs no bytes and needs no cache hit.
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

    # An EXPLICIT scheme is never swapped: same contract as select_transformer_quant_scheme.
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

    # Nothing below the winner has a checkpoint -> no retry, and the caller declines dense.
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
    # Rungs ABOVE the winner were already rejected by the ladder, so offering one back would hand
    # the loader a scheme auto itself would not pick.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.auto_scheme_candidates",
        lambda target, family = None: ("fp8", "mxfp8", "int8"),
    )
    # fp8 (above the chosen mxfp8) has a prequant; int8 (below) does not.
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
    # The replan helper used to be defined inside the `candidate is not None` block, so reaching the
    # retry from here raised UnboundLocalError under the load lock.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: "int8")
    )

    resolved = []

    def fake_resolve(**kw):
        resolved.append(kw.get("requested"))
        # The winner has no candidate; the retried rung does.
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
            # Initial GGUF plan wants offload, which is the branch the retry lives in.
            return dataclasses.replace(real, offload_policy = "model")
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "qwen-image",
        transformer_quant = "auto",
    )

    assert len(resolved) == 2
    # A prequant-sized plan never licenses the unbudgeted dense bf16 build as a fallback.
    assert attempted == [False]


def test_the_resident_retry_runs_when_the_dense_shards_were_never_staged(
    fake_runtime, tmp_path, monkeypatch
):
    # The prefetch capacity gate declines the base shards so _dense_transformer_resident_bytes reads
    # 0; treating that as "no information" skipped the retry and dropped back to GGUF.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *a, **k: None)
    # The winner has no usable checkpoint, and no shards were staged, so no dense build either.
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
        # The GGUF plan is RESIDENT here: this is the other branch from the offload retry.
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    seen = []

    def fake_dense_load(self, *a, **k):
        seen.append(a[7] if len(a) > 7 else k.get("transformer_quant"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "qwen-image",
        transformer_quant = "auto",
        _transformer_prefetched = False,
    )
    assert seen == ["int8"]


def test_the_resident_retry_declines_a_rung_that_does_not_plan_resident(
    fake_runtime, tmp_path, monkeypatch
):
    # Existence is not fit: an int8 checkpoint can outweigh the Q4 GGUF that planned resident, so
    # without the replan the loader moves it onto CUDA under a GGUF-sized budget.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # The retried rung does NOT fit.
        return dataclasses.replace(real, offload_policy = "model")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)

    attempted = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_load_dense_quant_pipeline",
        lambda self, *a, **k: attempted.append(True),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "qwen-image",
        transformer_quant = "auto",
        _transformer_prefetched = False,
    )
    # Declined: neither the winner nor the retried rung is loadable, so the GGUF stands.
    assert attempted == []


def test_an_auto_pick_that_retried_a_lower_rung_is_still_badged_auto():
    from core.inference.diffusion_auto_policy import build_resolved_record

    record = build_resolved_record({"transformer_quant": ("auto", "int8", "retried")})
    assert record["transformer_quant"]["source"] == "auto"
    assert record["transformer_quant"]["value"] == "int8"
    explicit = build_resolved_record({"transformer_quant": ("int8", "int8", "requested")})
    assert explicit["transformer_quant"]["source"] == "explicit"


def test_download_plan_skips_files_already_in_the_cache(monkeypatch):
    # Entries are what the Downloads panel lists, so a model fully on disk must plan nothing.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_files_already_cached",
        staticmethod(lambda repo_id, files, revision = None, declared_sizes = None: set(files)),
    )

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert plan["entries"] == []
    assert plan["total_bytes"] == 0


def test_download_plan_stages_only_what_the_cache_is_missing(monkeypatch):
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert [e["repo_id"] for e in plan["entries"]] == ["unsloth/FLUX.1-dev-GGUF"]
    assert plan["entries"][0]["files"] == ["flux1-dev-Q4_K_M.gguf"]
    # The cached base is gone from the total too, or the progress bar waits on bytes nobody fetches.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    # incompatible_reason rides in the same envelope, and None is "nothing known to be wrong".
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert set(by_repo) == {"unsloth/FLUX.1-dev-GGUF", "unsloth/FLUX.1-dev"}
    base = by_repo["unsloth/FLUX.1-dev"]
    # The whole scoped list, not just the missing VAE: the cached file is still staged.
    assert "vae/diffusion_pytorch_model.safetensors" in base["files"]
    assert "model_index.json" in base["files"]
    assert base["bytes"] == sum(
        size for name, size in _FLUX_BASE_SIBLINGS_BY_NAME.items() if name in base["files"]
    )
    # Derived, so the panel's rows and the bar it drives can never disagree.
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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
    monkeypatch.setattr(
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs, **_kw: False
    )
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

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert revisions and all(rev is None for rev in revisions)
    assert {e["repo_id"] for e in plan["entries"]} == {
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-dev",
    }


# The planner is shared with video, so the image loader needs the same guard: an oversized image
# checkpoint SIGKILLs on a Mac (the mps target disables the MPS allocator's high-watermark limit).


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
    # The 7-byte fake checkpoint sizes to 1 MiB; stand in a realistic resident footprint.
    monkeypatch.setattr(
        "core.inference.diffusion.estimate_gguf_resident_mib", lambda storage: resident_mib
    )
    return DiffusionBackend()


def test_unified_memory_refuses_an_oversized_image_load(fake_runtime, monkeypatch, tmp_path):
    backend = _oversized_gguf(monkeypatch, tmp_path, 16)
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            base_repo = "base/repo",
            family_override = "z-image",
        )
    message = str(excinfo.value)
    assert "z-image" in message
    assert "unified memory" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message
    assert backend.status()["loaded"] is False


def test_unified_memory_allows_an_image_load_that_fits(fake_runtime, monkeypatch, tmp_path):
    backend = _oversized_gguf(monkeypatch, tmp_path, 128)
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    assert status["loaded"] is True


def test_unified_memory_image_refusal_is_overridable(fake_runtime, monkeypatch, tmp_path):
    from core.inference.diffusion_memory import UNIFIED_OVERSIZE_ENV

    backend = _oversized_gguf(monkeypatch, tmp_path, 16)
    monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, "1")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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

    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_families import detect_family

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
    fam = detect_family("black-forest-labs/FLUX.2-klein-9B")
    backend = DiffusionBackend()
    measured = 34_000

    local = str(Path.cwd())
    plan = _plan_with_weights(measured)
    kept = backend._resident_sized_plan(plan, fam, local, target, "pipeline")
    assert kept.estimates["model_dense_mib"] == measured

    # A hub id the table recognises still gets the substitution: the fp32-shard case (Z-Image,
    # Lumina) this exists for.
    lowered = backend._resident_sized_plan(
        plan, fam, "black-forest-labs/FLUX.2-klein-4B", target, "pipeline"
    )
    assert lowered.estimates["model_dense_mib"] < measured


def test_speed_off_is_not_reported_as_a_staging_failure(fake_runtime, tmp_path, monkeypatch):
    """An explicit Speed=off rewrites an auto request to "off" and the plan stages no
    transformer/ on purpose. Reading that expected absence as a decline told the caller their
    automatic quant had failed for want of shards, when what actually happened is the bit-exact
    GGUF they asked for."""
    _stub_hosted_prequant(monkeypatch, cached = True)
    calls = _spy_dense_quant(monkeypatch)
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    (tmp_path / "m.gguf").write_bytes(b"x")

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        speed_mode = "off",
        _transformer_prefetched = False,
    )

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
        # No prequant for the WINNER and no dense candidate: only the retry can rescue this.
        monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
        monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
        monkeypatch.setattr(
            DiffusionBackend, "_auto_prequant_retry_scheme", staticmethod(lambda *a, **k: retry)
        )
        backend = DiffusionBackend()
        _force_cuda_target(backend, monkeypatch)
        (tmp_path / "m.gguf").write_bytes(b"x")
        status = backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            _transformer_prefetched = False,
        )
        return str(status.get("resolved", {}).get("transformer_quant", {}).get("reason") or "")

    marker = "an auto quant never downloads a second transformer"
    # A cached lower rung exists, so this branch must not fire and close the path.
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
    # `repo_id or base` dropped the base whenever a repo id existed, and the base is where the
    # distilled marker lives, so the 0.85 discount never fired for GGUF loads.
    from core.inference.diffusion import _image_variant_hint
    from core.inference.diffusion_memory import estimate_image_runtime_mib

    hint = _image_variant_hint(
        "z-image", "Z-Image-Q4_K_S.gguf", "unsloth/Z-Image-GGUF", "Tongyi-MAI/Z-Image-Turbo"
    )
    assert "unsloth/Z-Image-GGUF" in hint and "Tongyi-MAI/Z-Image-Turbo" in hint
    # Worth ~1.2 GB of headroom on a card where 1.2 GB decides the offload tier.
    assert estimate_image_runtime_mib(width = None, height = None, family = hint) == 6963
    assert estimate_image_runtime_mib(width = None, height = None, family = "") == 8192


def test_variant_hint_is_deduplicated_and_order_stable():
    # A pipeline load passes the same id as repo and base; the hint must not repeat it, and the
    # order must be a pure function of the load.
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
    # The planner SUBTRACTS this from the companion total, so it has to come off the same walk or
    # the difference can go negative.
    snapshot = _base_snapshot_with_sizes(
        tmp_path,
        monkeypatch,
        {
            "text_encoder/model.safetensors": 150,
            "text_encoder_2/model.safetensors": 90,
            "vae/diffusion_pytorch_model.safetensors": 50,
            # Never a companion on a GGUF load: the single file supplies the transformer.
            "transformer/diffusion_pytorch_model.safetensors": 4096,
        },
    )
    sizes = DiffusionBackend._local_dir_text_encoder_sizes(snapshot)
    assert sorted(sizes) == ["text_encoder/model.safetensors", "text_encoder_2/model.safetensors"]
    assert DiffusionBackend._text_encoder_cache_bytes(str(snapshot)) == 240 * 1024 * 1024
    assert DiffusionBackend._companion_cache_bytes(str(snapshot)) == 290 * 1024 * 1024


def test_plan_memory_hands_the_planner_the_text_encoder_split(monkeypatch, tmp_path):
    # 150 MiB of encoders inside a 200 MiB companion total, so the streamed-encoder floor is the VAE
    # plus headroom plus overhead.
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
    # The companions fit as they are, so the cheaper tier still wins and nothing streams.
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders is False


def test_plan_memory_streams_the_text_encoders_instead_of_offloading_everything(
    monkeypatch, tmp_path
):
    # 8081's shape: a text encoder that alone busts the group floor.
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
    assert plan.estimates["group_floor_mib"] == 4998
    assert plan.estimates["group_floor_streamed_te_mib"] == 2198
    assert plan.offload_policy == OFFLOAD_GROUP and plan.stream_text_encoders is True


def test_plan_memory_keeps_the_split_on_the_dense_candidate_path(monkeypatch, tmp_path):
    # The dense int8 candidate replan overrides the companion total from the family component table,
    # so without a matching text-encoder override it keeps landing on whole-module offload.
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
    # Companions minus text encoders has to be the VAE and nothing else, or the streamed floor is
    # wrong on every dense candidate.
    from core.inference.diffusion_auto_policy import estimate_dense_quant

    estimate = estimate_dense_quant(types.SimpleNamespace(name = "z-image"), "int8")
    assert estimate.steady_transformer_mib == 6451
    assert estimate.companions_mib == 7820
    assert estimate.text_encoders_mib == 7629
    assert estimate.companions_mib - estimate.text_encoders_mib == 191


# The load-time plan budgets the 1024x1024 default; at 1088x1920 the pass needs ~2x that, and on
# Windows WDDM the overrun does not raise (the driver serves it from RAM), so generate() re-checks.

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

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = base_repo,
        family_override = "z-image",
    )
    free, total = _ROCM_16G
    snapshot = lambda target, **kw: DeviceMemory("cuda", "cuda", "discrete_vram", free, total)
    monkeypatch.setattr(dmod, "settled_snapshot_device_memory", snapshot)
    # The generate-time guard reads the RECLAIMABLE snapshot, so pin that one too or it falls
    # through to the host's real card.
    monkeypatch.setattr(dmod, "reclaimable_snapshot_device_memory", snapshot)
    return backend


def test_generate_refuses_a_resolution_whose_activations_cannot_fit(
    fake_runtime, tmp_path, monkeypatch
):
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)
    message = str(excinfo.value)
    # ValueError, so /images/generate answers 400 with this text rather than a 500 or nothing at
    # all.
    assert "1088x1920" in message
    assert "smaller resolution" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_GENERATE" in message

    assert len(backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 4)["images"]) == 1


def test_generate_guard_measures_the_input_image_not_the_sliders(
    fake_runtime, tmp_path, monkeypatch
):
    # inpaint / upscale / edit take their OUTPUT size from the uploaded image, so reading the
    # width/height kwargs would check a frame this call never renders.
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
    # ...and must not advise changing the Resolution control, which cannot move that number for a
    # workflow whose canvas comes from the upload.
    assert "Upload a smaller source image" in message
    assert "smaller resolution" not in message


def test_transform_fits_the_upload_to_the_sliders_instead_of_refusing(
    fake_runtime, tmp_path, monkeypatch
):
    # Reported: Transform refused at 2048x2048 however small the sliders were set, because img2img
    # sized from the upload alone.
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
    # The distilled discount has to apply at generate time too, or a turbo model is budgeted 18%
    # high and refused where it would have run.
    backend = _loaded_backend_on_a_16g_card(
        tmp_path, monkeypatch, base_repo = "Tongyi-MAI/Z-Image-Turbo"
    )
    assert "Tongyi-MAI/Z-Image-Turbo" in backend._state.variant_hint

    # Undiscounted, 1280x1280 exceeds the 13,822 MiB budget and would be refused; with the distilled
    # discount it goes straight through.
    assert len(backend.generate(prompt = "a sloth", width = 1280, height = 1280, steps = 4)["images"]) == 1
    # The reported 1088x1920 needs 13,872 MiB even discounted, so it is still refused.
    with pytest.raises(ValueError, match = "1088x1920"):
        backend.generate(prompt = "a sloth", width = 1088, height = 1920, steps = 4)


def test_generate_guard_fails_open_when_the_probe_raises(fake_runtime, tmp_path, monkeypatch):
    # A broken memory probe must never cost a user a generation that would have worked.
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


def test_generate_guard_leaves_a_large_batch_to_the_oom_backoff(
    fake_runtime, tmp_path, monkeypatch
):
    # The chunk loop halves a failed multi-image forward down to singletons, so the guard budgets
    # one image -- the case no backoff can rescue.
    backend = _loaded_backend_on_a_16g_card(tmp_path, monkeypatch)
    out = backend.generate(prompt = "a sloth", width = 1024, height = 1024, steps = 4, batch_size = 8)
    assert len(out["images"]) == 8


def test_dense_quant_candidate_replan_prices_the_streamed_encoder_tier(
    fake_runtime, tmp_path, monkeypatch
):
    # 8081's plan came from THIS path, not the cache walk, so the text-encoder share must be
    # threaded through the same override or the streamed-encoder tier is unreachable.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
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
            # Initial GGUF plan: force offload so the candidate replan branch is entered.
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
    # The spy forces offload on every replan, so the EXPLICIT int8 ends in the strict-precision
    # refusal; immaterial, as the asserts all happen before it.
    with pytest.raises(RuntimeError, match = "transformer_quant='int8'"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            transformer_quant = "int8",
        )
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
    # An empty job list is still a valid batch of one, never a zero passed to the estimator.
    assert dmod._activation_guard_batch([]) == 1


# The denoise loop honoured the cancel event but only unload() could set it.


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
        # Extend pads the canvas client-side and sends the result down the SAME inpaint path, so
        # inpaint covers both surfaces.
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
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )

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
    # Progress state is cleared on every exit, so the page does not stay stuck at "generating".
    assert backend.generate_progress()["active"] is False
    assert backend.cancel_generate() is False


def test_cancel_generate_lands_at_the_next_step_boundary(fake_runtime, tmp_path, monkeypatch):
    # The contract is best-effort at the NEXT step boundary: a cancel raised during step 1 must not
    # let step 3 run. documents.
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )

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
    # The cancel event stays registered through the compile-cache save, so a Stop landing there must
    # not be answered cancelled and then contradicted by returned images.
    from core.inference import diffusion_compile_cache as compile_cache

    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )

    def _save(ctx, logger = None):
        # Stop pressed while the bundle is being written: the route's cancel reaches the SAME event,
        # which must still be registered here.
        assert backend.cancel_generate() is True

    monkeypatch.setattr(compile_cache, "save", _save)

    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate(prompt = "x", steps = 2)


def test_a_completed_generation_stops_advertising_itself_as_cancellable(
    fake_runtime, tmp_path, monkeypatch
):
    # The final check and the deregistration are one critical section under the lock cancel_generate
    # takes, so Stop cannot answer true for a generation that then returns images.
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )

    from core.inference import diffusion as diffusion_module

    seen: list[bool] = []
    real_baked = diffusion_module._baked_lora_names

    def _baked(pipe):
        # Runs after the final check, while the old code still had the event registered.
        seen.append(backend.cancel_generate())
        return real_baked(pipe)

    monkeypatch.setattr(diffusion_module, "_baked_lora_names", _baked)

    out = backend.generate(prompt = "x", steps = 2)
    assert out["images"]
    assert seen == [False]


def test_cancel_generate_is_a_no_op_without_a_load(fake_runtime):
    # The route calls this unconditionally, so an idle backend must answer False, not raise.
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    # A hosted pre-cast checkpoint IS available, which is what skips the dense-size check.
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

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        model_kind = "gguf",
    )
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert resolved["value"] == "off"
    assert "unified memory" in (resolved["reason"] or "")


def test_unified_memory_keeps_a_prequant_that_fits(fake_runtime, monkeypatch, tmp_path):
    # This guard only ever removes a candidate the device cannot hold, never one it can.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_auto_policy import DenseQuantEstimate

    backend = _oversized_gguf(monkeypatch, tmp_path, 32, resident_mib = 8 * 1024)
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
        # Raising here keeps the stub out of pipeline assembly; reaching this line at all is the
        # assertion.
        raise RuntimeError("stub")

    monkeypatch.setattr(dmod.DiffusionBackend, "_load_dense_quant_pipeline", _record)

    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        model_kind = "gguf",
    )
    assert calls == ["built"], "a prequant that fits must still reach the dense fast path"


def test_the_resident_size_table_prices_a_pre_cast_encoder_at_its_real_size(
    fake_runtime, monkeypatch
):
    """The table's encoder term is the dense one, and a pick that takes its encoder PRE-CAST from a
    hosted fp8 checkpoint loads roughly 0.65x of it. Budgeting the dense figure against a hard
    refusal turns tens of GB the pipeline never materialises into a rejected load."""
    import torch

    from core.inference import diffusion as dmod
    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_te_prequant import TE_PREQUANT_BUDGET_SCALE

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
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
    # Only the ENCODER term moves: transformer and VAE are untouched by a text-encoder quant.
    transformer_gb, encoders_gb, _vae = dmod.family_bf16_components_gb(fam, base)
    saved_gb = (dense_mib - precast_mib) * (1024.0 * 1024.0) / (1000.0**3)
    assert saved_gb == pytest.approx(encoders_gb * (1.0 - TE_PREQUANT_BUDGET_SCALE), rel = 0.02)


def test_the_resident_size_table_never_shrinks_an_unrecognised_remote_variant(fake_runtime):
    """Same hole as the local-path one, reached from the Hub: a fine-tune or a renamed mirror that
    the family detector still matches by name is NOT an exact key in the size table, so it falls
    through to the family entry -- and for a family carrying two sizes that entry is the smaller
    one. A 9B derivative lowered to the 4B number walks straight past the refusal."""
    import torch

    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_families import detect_family

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
    fam = detect_family("black-forest-labs/FLUX.2-klein-9B")
    backend = DiffusionBackend()
    measured = 34_000
    plan = _plan_with_weights(measured)

    kept = backend._resident_sized_plan(
        plan, fam, "someone/FLUX.2-klein-9B-anime-tune", target, "pipeline"
    )
    assert kept.estimates["model_dense_mib"] == measured
    # The two recognised shapes still get it: an explicit per-base override, and the family default.
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

    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_memory import DeviceMemory, MemoryPlan

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = False,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
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
    # The VAE share is never scaled, and a candidate with no split degrades to the dense total.
    no_split = types.SimpleNamespace(companions_mib = 1234, text_encoders_mib = 0)
    assert (
        DiffusionBackend._precast_scaled_companions_mib(no_split, fam, base, object(), "fp8")
        == 1234
    )
    # An estimate with no companions at all stays None, which _plan_memory reads as "no override".
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
        status = backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "m.gguf",
            family_override = "z-image",
            _transformer_prefetched = False,
            **request,
        )
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
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        _transformer_prefetched = False,
    )

    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    reason = ((status.get("resolved") or {}).get("transformer_quant") or {}).get("reason") or ""
    assert "shards are not staged" not in reason, reason

    # A scheme the device rules out (torchao stub, family deny list) is the same case.
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: None
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
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
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
    import core.inference.diffusion as diffusion_mod

    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
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
