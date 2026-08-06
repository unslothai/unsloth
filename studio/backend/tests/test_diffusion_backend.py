# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion backend.

The family helpers are pure functions, tested directly. The backend lifecycle is
exercised with ``torch`` / ``diffusers`` stubbed via ``sys.modules`` so no real
GPU, weights, or network access is needed (sub-second, CI-friendly).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import types
from pathlib import Path

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
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
    _GATED_MIRROR_PAIRS,
    assert_flux2_gguf_matches_base,
    canonical_base,
    detect_family,
    family_prequant_repo,
    mirror_repo,
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
)


# Pure family helpers


def test_clamp_max_side_bounds_oversized_init():
    # img2img / inpaint take their size from the upload, so _clamp_max_side bounds the longest side to 2048, keeping aspect.
    from PIL import Image

    # 12MP landscape: longest side clamped to 2048, 4:3 kept.
    out = _clamp_max_side(Image.new("RGB", (4096, 3072)), 2048)
    assert out.size == (2048, 1536)
    # A portrait upload clamps on height.
    assert _clamp_max_side(Image.new("RGB", (1000, 4000)), 2048).size == (512, 2048)
    # Already within bound: returned unchanged.
    small = Image.new("RGB", (768, 512))
    assert _clamp_max_side(small, 2048) is small


def test_detect_family_from_repo_id():
    # Detection is by architecture, so Turbo/full and schnell/dev share a family.
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").name == "z-image"
    assert detect_family("unsloth/Z-Image-GGUF").name == "z-image"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").name == "flux.1"
    # FLUX.2-klein is its own pipeline (Qwen3 TE), distinct from FLUX.1.
    klein = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    assert klein.name == "flux.2-klein"
    assert klein.pipeline_class == "Flux2KleinPipeline"
    assert klein.cfg_kwarg == "guidance_scale"
    # Both klein sizes share the one family (base repo resolved per-variant).
    assert detect_family("unsloth/FLUX.2-klein-9B-GGUF").name == "flux.2-klein"
    # FLUX.2-dev is the Mistral-based Flux2Pipeline, a distinct family from klein.
    dev = detect_family("unsloth/FLUX.2-dev-GGUF")
    assert dev.name == "flux.2-dev"
    assert dev.pipeline_class == "Flux2Pipeline"
    assert dev.base_repo == "black-forest-labs/FLUX.2-dev"
    assert detect_family("black-forest-labs/FLUX.2-dev").name == "flux.2-dev"
    # Qwen-Image guides via true_cfg_scale, not guidance_scale.
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").cfg_kwarg == "true_cfg_scale"
    assert detect_family("unsloth/Z-Image-GGUF").cfg_kwarg == "guidance_scale"
    # Qwen-Image-Edit is a supported editing family; the most specific match beats qwen-image.
    edit = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
    assert edit.name == "qwen-image-edit"
    assert edit.pipeline_class == "QwenImageEditPlusPipeline"
    assert edit.edit is True
    assert detect_family("unsloth/Qwen-Image-Edit-2509-GGUF").name == "qwen-image-edit"
    # FLUX Kontext is supported: "kontext" is un-rejected and must beat "flux.1".
    kontext = detect_family("unsloth/FLUX.1-Kontext-dev-GGUF")
    assert kontext.name == "flux.1-kontext"
    assert kontext.pipeline_class == "FluxKontextPipeline"
    assert kontext.edit is True
    assert kontext.cfg_kwarg == "guidance_scale"
    # A plain FLUX.1 checkpoint stays on flux.1, not kontext.
    assert detect_family("unsloth/FLUX.1-dev-GGUF").name == "flux.1"
    # A plain Qwen-Image checkpoint stays on the base family, not edit.
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    # Krea 2 (diffusers >= 0.39): bf16-only single-stream DiT, no GGUF/sd.cpp mapping.
    krea2 = detect_family("krea/Krea-2-Turbo")
    assert krea2.name == "krea-2"
    assert krea2.pipeline_class == "Krea2Pipeline"
    assert krea2.transformer_class == "Krea2Transformer2DModel"
    assert krea2.cfg_kwarg == "guidance_scale"
    assert krea2.fp16_incompatible is True
    assert krea2.sd_cpp_text_encoders == ()
    assert detect_family("meta-llama/Llama-3-8B") is None


def test_detect_family_matches_reject_and_alias_by_segment():
    # Reject keywords and short aliases match whole path segments, not substrings, so unrelated words cannot misroute.
    assert detect_family("/models/edited/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Z-Image-Edition-GGUF").name == "z-image"
    assert detect_family("/models/kontextual/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    # Supported edit families still resolve (edit / kontext are whole tokens).
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").name == "qwen-image-edit"
    assert detect_family("unsloth/FLUX.1-Kontext-dev-GGUF").name == "flux.1-kontext"
    # Unsupported variants sharing only a base arch keyword still reject.
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-2512-Inpaint") is None


def test_detect_family_edit_keyword_scoped_to_basename():
    from core.inference.diffusion_families import detect_family_for_pick

    # Only the model id / filename basename is scanned, so a parent dir named `edit`/`inpaint` cannot poison a pick.
    assert detect_family("/models/edit") is None  # the dir alone is ambiguous
    assert detect_family_for_pick("/models/edit", "Z-Image-Turbo-Q4.gguf").name == "z-image"
    assert detect_family_for_pick("/models/inpaint", "qwen-image-2512-Q4.gguf").name == "qwen-image"
    # A genuinely unsupported variant keyword in the FILENAME still rejects.
    assert detect_family_for_pick("/models/misc", "Qwen-Image-Layered-Q4.gguf") is None


def test_detect_family_override():
    assert detect_family("local/path", override = "z-image").name == "z-image"
    assert detect_family("local/path", override = "zimage").name == "z-image"
    assert detect_family("local/path", override = "not-a-family") is None


def test_supported_family_names():
    names = supported_family_names()
    # The unknown-model error lists these, so the key families must be present.
    for expected in ("flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "z-image", "krea-2"):
        assert expected in names
    # Every listed name is a valid family_override (round-trips through detect_family).
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
    assert len(_GATED_MIRROR_PAIRS) == 12
    for upstream, mirror in _GATED_MIRROR_PAIRS:
        assert mirror_repo(upstream) == mirror
        assert canonical_base(mirror) == upstream
        # Case-insensitive in, since a card tag may carry any casing.
        assert mirror_repo(upstream.upper()) == mirror
    # An ungated base is left alone.
    assert mirror_repo("Qwen/Qwen-Image") is None
    assert canonical_base("Qwen/Qwen-Image") == "Qwen/Qwen-Image"


def test_prefer_ungated_mirror_swaps_gated_bases(monkeypatch):
    _no_cache(monkeypatch)
    for upstream, mirror in _GATED_MIRROR_PAIRS:
        assert prefer_ungated_mirror(upstream) == mirror
    # Ungated bases are untouched.
    assert prefer_ungated_mirror("Qwen/Qwen-Image") == "Qwen/Qwen-Image"


def test_prefer_ungated_mirror_declines(monkeypatch):
    """Each decline path lands on the upstream id, i.e. exactly today's behaviour."""
    gated = "black-forest-labs/FLUX.1-dev"

    # 1. explicit opt-out
    _no_cache(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    assert prefer_ungated_mirror(gated) == gated

    # 2. already on disk: switching would re-pull tens of GiB
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
    # The table still knows this id: only the local path stops the swap.
    assert mirror_repo(gated) == "unsloth/FLUX.1-dev"
    assert prefer_ungated_mirror(gated) == "unsloth/FLUX.1-dev"

    local = tmp_path / gated
    (local / "vae").mkdir(parents = True)
    (local / "model_index.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert prefer_ungated_mirror(gated) == gated
    assert prefer_ungated_mirror(gated, files = ["model_index.json"]) == gated
    # An absolute local path never matched the table, and still does not.
    assert prefer_ungated_mirror(str(local)) == str(local)


def test_mirrored_base_still_trips_the_flux2_shape_guard():
    """The regression the two-helper split exists for.

    The guard fails OPEN on an unmapped base, so a mirror id reaching ``_FLUX2_BASE_INNER_DIM``
    would silence it. Assert the RAISE: a disabled guard passes any weaker check.
    """
    # "flux.2-klein", not "flux.2": no family has the bare name, and a None family returns early.
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
        # A 4B GGUF (inner_dim 3072) against the 9B base still raises through the mirror id.
        gguf.GGUFReader = _reader_for(3072)
        for base in ("black-forest-labs/FLUX.2-klein-9B", "unsloth/FLUX.2-klein-9B"):
            with pytest.raises(ValueError, match = "klein"):
                assert_flux2_gguf_matches_base(fam, base, "some-klein-4b.gguf")
        # A matching pair stays silent through the mirror id too.
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
    # Old revision complete, refs/main moved on to a revision holding only the manifest.
    _fake_hub_cache(monkeypatch, tmp_path, gated, wanted, revision = "old")
    _fake_hub_cache(monkeypatch, tmp_path, gated, ["model_index.json"], revision = "new", ref = "new")
    assert _upstream_is_cached(gated, wanted) is False
    assert prefer_ungated_mirror(gated, files = wanted) == "unsloth/FLUX.1-dev"

    # refs/main pointing at the complete revision is the ordinary cached case.
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
    # 1. debris only: not usable, so the mirror stands.
    _fake_hub_cache(monkeypatch, tmp_path, gated, ["model_index.json", "vae/config.json"])
    assert _upstream_is_cached(gated) is False
    assert prefer_ungated_mirror(gated) == "unsloth/FLUX.1-dev"

    # 2. a real weight file: the user's cache is worth keeping, so no swap.
    _fake_hub_cache(
        monkeypatch,
        tmp_path,
        gated,
        ["model_index.json", "vae/diffusion_pytorch_model.safetensors"],
    )
    assert _upstream_is_cached(gated) is True
    assert prefer_ungated_mirror(gated) == gated

    # 3. with the file list in hand the test is exact: a missing companion still swaps.
    wanted = ["vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"]
    assert _upstream_is_cached(gated, wanted) is False
    assert prefer_ungated_mirror(gated, files = wanted) == "unsloth/FLUX.1-dev"
    assert prefer_ungated_mirror(gated, files = wanted[:1]) == gated


def test_te_prequant_equivalence_group_accepts_a_mirrored_base():
    """The T5-XXL artifact is shared across the FLUX.1 releases through an equivalence group of
    UPSTREAM ids. A mirrored base is a different string, so without normalising it the pre-cast
    encoder is refused and the load falls back to the dense multi-GB download."""
    from core.inference.diffusion_te_prequant import te_base_equivalent

    ckpt = "black-forest-labs/FLUX.1-schnell"
    assert te_base_equivalent(ckpt, "black-forest-labs/FLUX.1-dev") is True
    assert te_base_equivalent(ckpt, "unsloth/FLUX.1-dev") is True
    assert te_base_equivalent("unsloth/FLUX.1-schnell", "unsloth/FLUX.1-dev") is True
    # A base outside the verified group is still refused, mirrored or not.
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


# Stubbed runtime for backend lifecycle


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
        # SDXL-style single-file: the WHOLE pipeline comes from one .safetensors file.
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

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeImg2ImgPipeline.built_from = base_pipe
        _FakeImg2ImgPipeline.from_pipe_kwargs = kwargs
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

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.built_from = base_pipe
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
    # generate() wraps the pipe call in torch.inference_mode(); a no-op CM here.
    torch.inference_mode = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    diffusers.ZImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.ZImageInpaintPipeline = _FakeInpaintPipeline
    # Qwen-Image too, to exercise the true_cfg_scale path.
    diffusers.QwenImagePipeline = _FakePipeline
    diffusers.QwenImageTransformer2DModel = _FakeTransformer
    diffusers.QwenImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.QwenImageInpaintPipeline = _FakeInpaintPipeline
    # Qwen-Image-Edit: its own pipeline is the loaded one.
    diffusers.QwenImageEditPlusPipeline = _FakePipeline
    # Ideogram 4, for its guidance_scale/guidance_schedule pairing. It loads only as a full pipeline (two DiTs), so stub the assembly.
    diffusers.Ideogram4Pipeline = _FakePipeline
    diffusers.Ideogram4Transformer2DModel = _FakeTransformer
    # Lumina 2, for the cfg_trunc_ratio special case.
    diffusers.Lumina2Pipeline = _FakePipeline
    diffusers.Lumina2Transformer2DModel = _FakeTransformer
    # SDXL: a U-Net family whose single-file checkpoint is the whole pipeline, so the pipeline class has from_single_file.
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
    # Stub clear_gpu_cache (imported by reference) so unload skips hardware detection.
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: None)
    _FakePipeline.last = {}
    _FakePipeline.last_single_file = {}
    _FakeTransformer.last = {}
    _FakeImg2ImgPipeline.built_from = None
    _FakeImg2ImgPipe.last_kwargs = {}
    _FakeInpaintPipeline.built_from = None
    _FakeInpaintPipe.last_kwargs = {}
    yield


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
    # Transformer built from the local GGUF, pipeline assembled from the base repo.
    assert _FakeTransformer.last["path"] == str((tmp_path / "model.gguf").resolve())
    assert _FakeTransformer.last["subfolder"] == "transformer"
    # The token reaches the (possibly gated) base config fetch and the pipeline.
    assert _FakeTransformer.last["token"] == "hf_secret"
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last

    gen = backend.generate(
        prompt = "a sloth", negative_prompt = "blurry", width = 512, height = 512, steps = 4, guidance = 3.0
    )
    assert gen["seed"] == 4242  # random seed reported back
    assert gen["repo_id"] == str(tmp_path)  # echoed so the route can record the model
    assert len(gen["images"]) == 1  # PIL images handed to the route for persistence
    # z-image guides via guidance_scale; the signature-gated negative_prompt and the step callback both land.
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 3.0 and call["true_cfg_scale"] is None
    assert call["negative_prompt"] == "blurry"
    assert callable(call["callback_on_step_end"])

    gen2 = backend.generate(prompt = "again", seed = 99)
    assert gen2["seed"] == 99

    # batch_size yields that many images, seeded base..base+2 so each replays alone.
    batch = backend.generate(prompt = "batch", seed = 7, batch_size = 3)
    assert len(batch["images"]) == 3 and batch["seed"] == 7
    assert batch["seeds"] == [7, 8, 9]

    assert backend.unload()["loaded"] is False
    assert backend.is_loaded is False


def test_generate_progress_active_during_setup(fake_runtime, tmp_path, monkeypatch):
    # Active must be published the moment the lock is held, before the slow setup that _apply_loras runs in.
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

    # Active was published during setup, with the requested step total and step 0.
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 4
    assert seen["progress"]["step"] == 0

    assert backend.generate_progress()["active"] is False


def test_generate_progress_cleared_on_setup_error(fake_runtime, tmp_path, monkeypatch):
    # A setup failure skips the inner finally, so the outer finally must clear the published progress.
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
    # The compile-cache save runs before the route persists the image, so progress must stay active through it.
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
    # Still active while the compile-cache save ran.
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 4
    assert backend.generate_progress()["active"] is False


def test_dense_speed_auto_defers_compile_to_third_generation(fake_runtime, tmp_path, monkeypatch):
    # Speed unset: dense models stay eager for two generations, then the 3rd engages `default` and upgrades attention.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(dmod, "apply_attention_backend", lambda pipe, backend, logger = None: backend)
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

    # An explicit "off" is pinned: no deferral, still eager after 3 generations.
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
    # A compiled transformer rejects LoRA, so the deferral must skip while a LoRA is requested.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False  # mirror the real helper: engage once, then clear

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)
    # Stub LoRA loading (covered elsewhere) so no adapter file is needed.
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
    # 3rd generation requests a LoRA: the deferral must be skipped (pipe stays LoRA-capable).
    backend.generate(prompt = "three", loras = [("adapter", 1.0)])
    assert engaged == []
    # 4th generation without a LoRA: the deferral now engages (the guard is LoRA-specific).
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_skips_while_adapter_attached(fake_runtime, tmp_path, monkeypatch):
    # Defer even on a no-LoRA generation while a prior adapter is attached: _apply_loras runs after the engage, so compiling would bake it in.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)

    # Track the attached set on the pipe, mirroring the real _apply_loras marker.
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
    # Gens 1-2 attach an adapter, so it is still resident going into gen 3.
    backend.generate(prompt = "one", loras = [("adapter", 1.0)])
    backend.generate(prompt = "two", loras = [("adapter", 1.0)])
    # Gen 3 requests NO LoRA but the adapter is still attached, so defer.
    backend.generate(prompt = "three")
    assert engaged == []
    # Gen 3's _apply_loras([]) cleared the adapter; gen 4 is genuinely LoRA-free, so engage.
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_preserves_explicit_attention(fake_runtime, tmp_path, monkeypatch):
    # Speed=Auto with Attention pinned must keep that choice when the 3rd generation engages.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(dmod, "apply_attention_backend", lambda pipe, backend, logger = None: backend)

    # A select mock that HONORS an explicit request: only an unset request upgrades to cuDNN.
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
    assert status["speed_mode"] == "default"  # the compile profile still engaged
    assert "compiled" in status["speed_optims"]
    # The pinned "native" survived: NOT silently upgraded to cuDNN.
    assert status["attention_backend"] is None
    assert status["resolved"]["attention_backend"]["value"] == "native"
    assert status["resolved"]["attention_backend"]["source"] == "explicit"

    # Control: on auto the same deferral does upgrade to cuDNN, so the assertion above is not vacuous.
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
    # The family advertises its image-conditioned workflows for UI gating (upscale rides img2img).
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
    # from_pipe was handed the loaded text-to-image pipe (component reuse, no reload).
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    # ...with torch_dtype=None so from_pipe skips the float32 recast that crashes on torchao weights.
    assert _FakeImg2ImgPipeline.from_pipe_kwargs.get("torch_dtype", "MISSING") is None
    call = _FakeImg2ImgPipe.last_kwargs
    assert call["image"] is not None  # decoded source image passed through
    assert call["strength"] == 0.5
    assert "width" not in call and "height" not in call  # img2img derives size from image

    # A txt2img call after it still uses the base pipe (no image kwarg).
    backend.generate(prompt = "plain", steps = 4, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


def test_generate_img2img_unsupported_family_raises(fake_runtime, tmp_path, monkeypatch):
    """A family with no image-conditioning at all (no img2img/inpaint/edit/reference) rejects
    an init_image with a clear error rather than failing deep in the pipeline."""
    from core.inference.diffusion_families import DiffusionFamily

    # A synthetic txt2img-only family: no img2img/inpaint pipeline, not edit, not reference.
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
    # Upscale rides the img2img pipeline, so it is advertised alongside img2img.
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
    # Reuses the resident modules via from_pipe (no reload, no extra VRAM).
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    call = _FakeImg2ImgPipe.last_kwargs
    # The image handed to the pipe is the ENLARGED source (64 * 2 = 128, already /16).
    assert call["image"].size == (128, 128)
    # Strength defaults to the hires-fix value when the caller sends none.
    assert call["strength"] == 0.35

    # The factor caps at 4x so a large request cannot blow up the VAE/transformer.
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 99.0,
    )
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (256, 256)  # 64 * 4 (capped)

    # An explicit strength overrides the hires-fix default.
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 1.5,
        strength = 0.2,
    )
    assert _FakeImg2ImgPipe.last_kwargs["strength"] == 0.2
    # 64 * 1.5 = 96, already a multiple of 16.
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
    # 1024 * 4 = 4096 -> clamped to 2048 (longest side), still a multiple of 16.
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
    # 186 / 16 = 11.625 -> round to 12 -> 192.
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
    # FLUX.2-klein: txt2img + reference (own pipe) + inpaint (dedicated pipe). No img2img class, so no img2img/upscale.
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
    assert call["image"] is not None  # reference handed to the loaded pipe
    assert call["width"] == 768 and call["height"] == 512  # OUTPUT size = sliders, not input
    assert "strength" not in call  # reference conditioning has no strength
    assert "mask_image" not in call
    # Guidance flows via guidance_scale (FLUX.2 default behaviour).
    assert call["guidance_scale"] == 4.0

    # Multi-reference: extra reference_images are combined with init_image into a LIST.
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

    # Branch ordering: an init image plus a mask on a reference family routes to inpaint.
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

    # Without an init image the same family does plain txt2img (no image arg).
    backend.generate(prompt = "just text", steps = 6, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


def _tiny_mask_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    # A grayscale mask: white square (repaint) on black (keep).
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
    # The inpaint pipe (not img2img) was selected and built from the loaded pipe.
    assert _FakeInpaintPipeline.built_from is loaded_pipe
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
    # The pipe got the IMAGE's 96x64, not the 1024x1024 slider.
    assert _SizePipe.last == {"width": 96, "height": 64}


def test_compile_shape_dims_follow_workflow():
    """_compile_shape_dims mirrors generate()'s width/height derivation: slider size for
    txt2img / reference / controlnet, the input image's size for the image-conditioned
    workflows (whose forward runs at init_pil.size, whatever the sliders say)."""
    from PIL import Image

    from core.inference.diffusion import _compile_shape_dims

    img = Image.new("RGB", (96, 64), (10, 20, 30))
    assert _compile_shape_dims("txt2img", None, 1024, 512) == (1024, 512)
    # reference generates at the slider size even though an init image is present.
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
    # txt2img registers the requested slider size.
    backend.generate(prompt = "x", steps = 4, width = 1024, height = 512, seed = 1)
    assert registered[-1] == (1024, 512, 1)
    # img2img runs at the INPUT image's 64x64; the 1024x512 slider must not be recorded.
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
    # Edit families advertise only the edit workflow.
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
    # The loaded pipe handled it directly: no from_pipe img2img/inpaint was built.
    assert backend._state.pipe is loaded_pipe
    assert _FakeImg2ImgPipeline.built_from is None and _FakeInpaintPipeline.built_from is None
    assert loaded_pipe.last_kwargs.get("image") is not None

    # An edit model with no input image fails fast with a clear message.
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
    # from_pretrained pointed at the repo itself (it IS its own base), with no transformer.
    assert _FakePipeline.last["base"] == "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
    assert "transformer" not in _FakePipeline.last
    # The GGUF single-file build path was never taken.
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
    # No GGUF quant config on the safetensors path (the GGUF path sets one).
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
    assert _FakePipeline.last["base"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "transformer" not in _FakePipeline.last
    # Neither single-file path (transformer-only nor whole-pipeline) was taken.
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
    # The whole-pipeline single-file path was taken with the base repo as config.
    assert _FakePipeline.last_single_file["path"] == str((tmp_path / "sdxl.safetensors").resolve())
    assert _FakePipeline.last_single_file["config"] == "stabilityai/stable-diffusion-xl-base-1.0"
    # The transformer-only single-file build was NOT taken.
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
    # A companion base_repo also loads via from_pretrained, so it must clear the same trust bar.
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "base_repo"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = "evil/companions",
        )
    # A local base_repo that is not a diffusers pipeline is rejected here: from_pretrained needs model_index.json, so it would otherwise evict the resident model then fail.
    bad_base = tmp_path / "bare-base"
    bad_base.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(bad_base),
        )
    # A local base_repo that IS a real pipeline dir passes the gate.
    (tmp_path / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(
        "unsloth/Qwen-Image-2512-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = str(tmp_path),
    )
    assert fam is not None


def test_resolve_local_single_file(tmp_path):
    # A bare single-file safetensors dir resolves to that checkpoint, so an On-Device "pipeline" pick becomes a single_file load.
    from core.inference.diffusion import resolve_local_single_file

    d = tmp_path / "solo"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d)) == "model.safetensors"

    # A real diffusers pipeline dir loads as a pipeline unchanged.
    (d / "model_index.json").write_text("{}")
    assert resolve_local_single_file(str(d)) is None

    # Ambiguous (two checkpoints) or empty dirs leave the load unchanged.
    d2 = tmp_path / "shards"
    d2.mkdir()
    (d2 / "a.safetensors").write_bytes(b"w")
    (d2 / "b.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d2)) is None
    assert resolve_local_single_file(str(tmp_path / "empty-nonexistent")) is None
    # A remote repo id (not a local dir) -> None.
    assert resolve_local_single_file("unsloth/Qwen-Image-2512-GGUF") is None

    # A PEFT adapter folder is not a base checkpoint, even with a family-token name: from_single_file would fail after eviction.
    adapter = tmp_path / "flux-style-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter)) is None
    # A bare adapter_model.safetensors is likewise not treated as the sole checkpoint.
    adapter2 = tmp_path / "z-image-lora"
    adapter2.mkdir()
    (adapter2 / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter2)) is None


def test_resolve_base_repo_drops_untrusted_card_tag(monkeypatch):
    # With no base_repo the base comes from the GGUF repo's base_model tag, which is attacker-controlled, so an untrusted tag must fall back to the family default.
    import core.inference.diffusion as dmod

    fam = detect_family("unsloth/FLUX.1-dev-GGUF")
    # A malicious card tag is ignored, so the family default base is used.
    monkeypatch.setattr(dmod, "_hf_base_model", lambda repo_id, hf_token: "attacker/evil-pipeline")
    assert _resolve_base_repo("attacker/flux.1-evil-GGUF", None, fam, None) == fam.base_repo
    # A trusted (allowlisted) card tag is still honoured, so variant resolution is not regressed.
    monkeypatch.setattr(
        dmod, "_hf_base_model", lambda repo_id, hf_token: "black-forest-labs/FLUX.1-dev"
    )
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", None, fam, None)
        == "black-forest-labs/FLUX.1-dev"
    )
    # An explicit trusted base_repo wins over the card tag; an untrusted one is caught earlier at validate_load_request.
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
    # An EXPLICIT base_repo is the caller's own choice and is honoured verbatim.
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", "unsloth/FLUX.1-dev", fam, None)
        == "unsloth/FLUX.1-dev"
    )


def test_detect_family_rejects_layered():
    # Qwen-Image-Layered needs a dedicated pipeline (additional_t_cond), so reject at load, not at the first step.
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

    # apply_memory_plan runs AFTER the patches are installed, before _LoadState commits.
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
    # No CUDA in the stub, so offload is not engaged.
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
    # No CUDA in the stub, so offload is not engaged regardless of the request.
    assert status["cpu_offload"] is False


def test_generate_without_load_raises(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.generate(prompt = "x")


def test_failed_load_restores_backend_flags(fake_runtime, tmp_path, monkeypatch):
    # A failure after apply_speed_optims (here an OOM) must restore the global TF32 / cudnn flags and commit no partial state.
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
    # Caller's explicit base wins and the HF tag is not consulted.
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "my/base", fam, None)
        == "my/base"
    )
    # No caller base: the repo's base_model tag (the variant base) is used.
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", None, fam, None)
        == "Qwen/Qwen-Image-2512"
    )
    # No caller base and no tag: the family fallback.
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: None)
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "  ", fam, None)
        == fam.base_repo
    )


def test_load_without_gguf_raises():
    backend = DiffusionBackend()
    # No gguf_filename means a full-pipeline load, gated to unsloth/*, so a non-unsloth repo is rejected up front.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("some-org/Z-Image-bnb-4bit")


def test_load_unknown_family_raises():
    backend = DiffusionBackend()
    with pytest.raises(ValueError):
        backend.load_pipeline("some/unrecognised-repo", gguf_filename = "x.gguf")


# load_progress state machine (no threads / network / real cache)

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
    # 600 stale upstream bytes from the interrupted pull, 500 real ones into the mirror.
    monkeypatch.setattr(
        DiffusionBackend,
        "_cache_bytes",
        staticmethod(lambda repo: 600 if repo.startswith("black-forest-labs/") else 500),
    )
    p = backend.load_progress()
    assert p["bytes_downloaded"] == 500  # the mirror alone, not 1100
    assert p["phase"] == "downloading"  # not "finalizing"


def test_load_progress_still_sums_a_gguf_pick_and_its_separate_base(monkeypatch):
    # A base that is a DIFFERENT repo from the pick is a separate download, so still summed.
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
    # Counted: the pipeline manifest + component subfolders from_pretrained fetches.
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
    # The cache scan can exceed the estimate, so the reported fraction must still clamp to 1.0.
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", expected_bytes = 1000)
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 900))
    p = backend.load_progress()  # summed 1800 > expected 1000
    assert p["phase"] == "finalizing"
    assert p["fraction"] == 1.0
    assert p["bytes_downloaded"] == 1000  # clamped to the estimate


def test_estimate_eta():
    from core.inference.diffusion import _estimate_eta

    # No rate yet until a step has elapsed since the first.
    assert _estimate_eta(8, 1, first_step_at = 100.0, now = 100.0) is None
    assert _estimate_eta(8, 0, first_step_at = 0.0, now = 100.0) is None
    # 3 steps in 3s since the first: 1s/step, 4 steps left, so ~4s.
    assert _estimate_eta(8, 4, first_step_at = 100.0, now = 103.0) == 4.0
    # Last step: 0 remaining.
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
    # Qwen-Image's distilled guidance is off; the real CFG must land on true_cfg_scale.
    call = backend._state.pipe.last_kwargs
    assert call["true_cfg_scale"] == 4.0 and call["guidance_scale"] is None


def _load_ideogram(backend, tmp_path):
    # Ideogram 4 loads only as a full pipeline (the loader is stubbed), so a local dir is enough.
    (tmp_path / "model_index.json").write_text("{}")
    backend.load_pipeline(str(tmp_path), family_override = "ideogram-4")


def test_ideogram_rejects_single_file_and_gguf_kinds(fake_runtime, tmp_path):
    # Ideogram 4 needs two DiTs, so transformer-only single-file and GGUF loads are rejected up front.
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
    # Ideogram 4 defaults to guidance_schedule (valid only at 48 steps) and rejects guidance_scale with it, so drop the constant at the defaults.
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 7.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] is None  # not passed: the pipe default engages
    assert "guidance_schedule" not in call


def test_generate_ideogram_custom_guidance_nulls_schedule(fake_runtime, tmp_path):
    # A non-default request sets guidance_scale and explicitly nulls guidance_schedule (both set raises).
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 20, guidance = 5.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 5.0
    assert "guidance_schedule" in call and call["guidance_schedule"] is None


def _load_lumina(backend, tmp_path):
    # Lumina 2 loads through the GENERIC pipeline path, so a local pipeline dir is enough here.
    (tmp_path / "model_index.json").write_text("{}")
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
    # The kwarg is family-gated, not just signature-gated, so another family accepting it must not inherit Lumina's constant.
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
    # The worker resolves the base + downloads, both over the network; stub them so this is offline.
    monkeypatch.setattr("core.inference.diffusion._hf_base_model", lambda *a, **k: None)
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    # Block the spawned worker so the load stays "in progress".
    monkeypatch.setattr(
        DiffusionBackend, "load_pipeline", lambda self, **k: __import__("time").sleep(0.2)
    )
    before = set(threading.enumerate())
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    with pytest.raises(RuntimeError):
        backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    # Drain the worker while the stubs above still make it exit in 0.2s: begin_load's thread is
    # fire-and-forget, so left running it outlives this test and then runs the REAL load_pipeline
    # inside whatever test is current, under that test's patches.
    for thread in set(threading.enumerate()) - before:
        thread.join(timeout = 5)


def test_unload_cancels_in_flight_load(fake_runtime):
    # An unload (or arbiter eviction) mid-download must cancel the worker: load_pipeline sees the bumped token and aborts.
    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    token = 7
    backend._load_token = token
    with pytest.raises(RuntimeError, match = "cancelled"):
        # Simulate the worker reaching load_pipeline after unload bumped the token.
        backend._load_token = token + 1
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )


def test_superseded_load_does_not_cancel_live_generation(fake_runtime):
    # A superseded load must bail without signalling the current model's in-flight generation.
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
    # BF16 only on Ampere+ (cc >= 8); pre-Ampere cards must fall back to FP16.
    torch = sys.modules["torch"]
    backend = DiffusionBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.float16)


def test_unload_sets_cancel_event(fake_runtime):
    # unload signals an in-flight download (which runs without the lock) to abort.
    backend = DiffusionBackend()
    assert not backend._cancel_event.is_set()
    backend.unload()
    assert backend._cancel_event.is_set()


def test_prefetch_aborts_when_cancelled(tmp_path):
    # A prefetch interrupted by unload raises instead of pulling the whole base, so the load can be preempted.
    backend = DiffusionBackend()
    backend._cancel_event.set()
    # Local gguf path so the transformer download is skipped; the base loop hits the cancel check on its first file.
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
    # unload() cancels the in-flight download and drops _loading together, so a replacement load is admitted while that worker is still in _prefetch_files. One shared Event let the replacement's clear() un-cancel the dead worker.
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
    # And the superseded worker's prefetch bails on ITS event, not on the live one.
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
    # Hub repo: the GGUF transformer and each base file are fetched.
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
    # Local GGUF path: the transformer download is skipped, base still fetched.
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
    # The GGUF repo is never rewritten.
    assert ("unsloth/FLUX.1-dev-GGUF", "flux1-dev-Q4_K_M.gguf") in calls

    # An upstream already on disk keeps its cache.
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
    # Invisible: only the bytes moved, so the API still reports the id the user picked.
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
    # BOTH ids: the upstream is what status reports, the mirror is where the download writes.
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
        # Second arg is the staged snapshot dir, unused here: what this pins is that the
        # MIRROR id is the one sized.
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


# fp16-incompatible guard + dtype promotion


def test_zimage_is_fp16_incompatible():
    # Only Z-Image-class families carry the guard (their activations overflow fp16).
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Z-Image-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.2-klein-4B-GGUF").fp16_incompatible is False


def test_resolve_compute_dtype_promotes_fp16_for_zimage(fake_runtime):
    torch = sys.modules["torch"]
    z = detect_family("unsloth/Z-Image-GGUF")
    q = detect_family("unsloth/Qwen-Image-GGUF")
    # Z-Image: fp16 promotes to fp32; bf16 / fp32 pass through unchanged.
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    assert _resolve_diffusion_compute_dtype(z, torch.bfloat16) is torch.bfloat16
    assert _resolve_diffusion_compute_dtype(z, torch.float32) is torch.float32
    # An fp16-compatible family (and None) keep fp16.
    assert _resolve_diffusion_compute_dtype(q, torch.float16) is torch.float16
    assert _resolve_diffusion_compute_dtype(None, torch.float16) is torch.float16


def test_load_promotes_fp16_to_fp32_for_zimage_only(fake_runtime, monkeypatch, tmp_path):
    torch = sys.modules["torch"]
    # Pre-Ampere CUDA resolves to fp16, so the guard must promote Z-Image (and only Z-Image) to fp32 or it renders black.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")

    z = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
    )
    assert z["device"] == "cuda" and z["dtype"] == "float32"
    # The promoted dtype reaches the transformer build (and thus the quant config).
    assert str(_FakeTransformer.last["torch_dtype"]) == "torch.float32"

    q = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "qwen-image"
    )
    assert q["dtype"] == "float16"  # fp16-compatible family keeps fp16 on pre-Ampere


def test_bad_mode_strings_fail_before_eviction(fake_runtime):
    # Every mode normalizer that can raise runs BEFORE the load evicts the previous pipeline.
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


# Lock split + mid-denoise cancellation


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

    # status() / generate_progress() must NOT block behind the denoise.
    assert backend.status()["loaded"] is True
    assert backend.generate_progress()["active"] is True

    cancel_ref = backend._active_generate_cancel
    assert cancel_ref is not None

    # unload() signals this generation's cancel then waits for the denoise to exit; release the pipe once the cancel lands, standing in for a step callback.
    releaser = threading.Thread(target = lambda: (cancel_ref.wait(5), release.set()))
    releaser.start()
    backend.unload()
    releaser.join(5)
    assert cancel_ref.is_set()
    assert backend.status()["loaded"] is False

    t.join(5)
    # The cancelled generation raised instead of returning an evicted image, and deregistered its cancel before unload() returned.
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
    # Simulate an eviction / superseding load signalling THIS generation's cancel.
    assert backend._active_generate_cancel is not None
    backend._active_generate_cancel.set()
    resume.set()
    t.join(5)
    # The next step callback saw the cancel, flipped pipe._interrupt and broke the loop, so no partial image came back.
    assert pipe._interrupt is True
    assert pipe.steps_run < 8
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()


def test_validate_load_request(tmp_path):
    backend = DiffusionBackend()
    # No filename + unsloth repo -> a full-pipeline load (allowed for unsloth/*).
    assert backend.validate_load_request("unsloth/Z-Image-Turbo-unsloth-bnb-4bit").name == "z-image"
    # No filename + non-unsloth repo -> a pipeline load, gated to unsloth/* -> rejected.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image-bnb-4bit")
    # An explicit gguf/single_file kind still requires a single-file name.
    with pytest.raises(ValueError, match = "single-file"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf")
    # A pipeline kind must NOT carry a single-file name.
    with pytest.raises(ValueError, match = "pipeline"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-bnb-4bit", gguf_filename = "q.gguf", model_kind = "pipeline"
        )
    # A single-file safetensors load is also gated to unsloth/* repos.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image", gguf_filename = "model.safetensors")
    with pytest.raises(ValueError, match = "family"):
        backend.validate_load_request("meta/Llama-3", gguf_filename = "q.gguf")
    # A family-looking repo with a non-GGUF single-file name is rejected before the route evicts chat.
    with pytest.raises(ValueError, match = r"\.gguf"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "README.md")
    assert (
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf").name
        == "z-image"
    )
    # A kind/extension mismatch fails fast, before the route evicts chat and from_single_file fails in the background.
    with pytest.raises(ValueError, match = ".gguf"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "model.safetensors", model_kind = "gguf"
        )
    with pytest.raises(ValueError, match = "gguf"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-FP8", gguf_filename = "q.gguf", model_kind = "single_file"
        )
    # A remote "*-GGUF" repo loaded as a full pipeline has no manifest, so reject it before the GPU handoff.
    with pytest.raises(ValueError, match = "GGUF"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "pipeline")
    # A local path with a missing child fails here (before any GPU/network work).
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
    # A path-shaped repo_id that does not exist is rejected here, not treated as remote and failed in the background.
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
    assert started.wait(5)  # generation in flight, holding _generate_lock

    (tmp_path / "m.gguf").write_bytes(b"x")
    load_done = threading.Event()

    def _load():
        backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
        load_done.set()

    lt = threading.Thread(target = _load)
    lt.start()

    # The load must not finish while the generation holds _generate_lock: it has signalled the cancel and waits to allocate.
    assert not load_done.wait(0.5)
    assert backend._active_generate_cancel is not None
    assert backend._active_generate_cancel.is_set()

    release.set()  # the blocked denoise returns; generate() sees cancel and raises
    gt.join(5)
    assert load_done.wait(5)  # only now does the replacement allocate
    assert "exc" in gen_out and "cancelled" in str(gen_out["exc"]).lower()
    assert backend.status()["loaded"] is True
    assert backend.status()["repo_id"] == str(tmp_path)


# ── Phase 2A: memory policy wiring (load -> planner -> placement) ──────────────


def test_load_reports_memory_plan_fields_on_cpu(fake_runtime, tmp_path):
    # The default stub resolves to a CPU target: no offload possible, VAE tiling on, and status carries the new fields.
    (tmp_path / "m.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["offload_policy"] == "none"
    assert status["cpu_offload"] is False
    assert status["vae_tiling"] is True
    assert status["memory_mode"] == "auto"
    pipe = backend._state.pipe
    assert pipe.moved_to == "cpu" and pipe.vae_tiled and pipe.vae_sliced


def _force_cuda_target(backend, monkeypatch):
    """Drive the loader down the CUDA (offload-capable) path under the stub."""
    torch = sys.modules["torch"]
    monkeypatch.setattr(backend, "_pick_device_and_dtype", lambda: ("cuda", torch.bfloat16))


def test_load_memory_mode_balanced_streams_or_falls_back(fake_runtime, tmp_path, monkeypatch):
    # balanced requests streamed group offload; with no diffusers.hooks the stub falls back to whole-module offload and reports it.
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
    # low_vram offloads every component; whole-module offload is the robust path and engages directly.
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "low_vram"
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True
    pipe = backend._state.pipe
    assert pipe.offloaded is True and pipe.moved_to is None  # offload owns placement


def test_load_explicit_cpu_offload_engages_model_offload_on_cuda(
    fake_runtime, tmp_path, monkeypatch
):
    # cpu_offload=True with no mode: auto would stay resident under the stub, but the explicit flag forces whole-module offload.
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", cpu_offload = True
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True


def test_load_speed_mode_gguf_auto_defaults_and_explicit(fake_runtime, tmp_path):
    # No speed_mode on a GGUF model resolves to auto `default`; compile is CUDA-only, so nothing engages on this CPU stub.
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["speed_mode"] == "default"
    # An explicit "off" opts back into the bit-identical path (engages nothing).
    status_off = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "off"
    )
    assert status_off["speed_mode"] == "off" and status_off["speed_optims"] == []
    # An explicit speed_mode threads through to status (engaged optims are GPU-verified).
    status2 = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "max"
    )
    assert status2["speed_mode"] == "max"
    # Text-encoder quant defaults off; a requested mode threads through (engagement is GPU-verified).
    assert status2["text_encoder_quant"] is None
    status3 = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        text_encoder_quant = "nvfp4",
    )
    # Under the CPU stub nvfp4 is unsupported, so it engages nothing.
    assert status3["text_encoder_quant"] is None


def test_load_fast_mode_stays_resident_on_cuda(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "fast"
    )
    assert status["offload_policy"] == "none" and status["cpu_offload"] is False
    assert backend._state.pipe.moved_to == "cuda"


# ── transformer quant (opt-in dense fast path) ────────────────────────────────


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
    # Resolve the scheme without the GPU smoke probe, and configure no pre-quant checkpoint so the dense materialise+quantise branch runs.
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
    # An unset dtype follows the hardware ladder: the dense gate is consulted, and a device without dense support falls back to GGUF.
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
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used


def test_explicit_off_load_skips_dense_quant_path(fake_runtime, tmp_path, monkeypatch):
    # An EXPLICIT "none" pins running the GGUF as-is: the dense gate is never consulted.
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
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used


def test_speed_off_load_suppresses_auto_dtype_quant(fake_runtime, tmp_path, monkeypatch):
    # An explicit Speed="off" load with an unset dtype must stay GGUF-as-is: the auto default must not promote it to a quantized + compiled build.
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
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used, not a dense build


def test_transformer_quant_dense_path_engaged(fake_runtime, tmp_path, monkeypatch):
    # transformer_quant + a CUDA resident plan: load the dense transformer from the base repo, place it, quantise it, report the scheme.
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
    # No speed_mode was given, but a quantized transformer is ~30x slower eager, so it is promoted to `default`.
    assert status["speed_mode"] == "default"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1
    assert calls["quant_mode"] == "fp8"
    assert calls["fp_kwargs"]["subfolder"] == "transformer"  # dense transformer subfolder
    # The GGUF single-file path was NOT used for the transformer.
    assert _FakeTransformer.last == {}
    # quantize ran on-device: the dense pipe was placed on cuda (before compile).
    assert backend._state.pipe.moved_to == "cuda"
    assert status["offload_policy"] == "none"


def test_transformer_quant_prequant_path_engaged(fake_runtime, tmp_path, monkeypatch):
    # A configured pre-quant checkpoint loads the quantized transformer directly, so dense from_pretrained and quantize_transformer go unused.
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
    # The pre-quantized transformer object was assembled into the pipeline...
    assert _FakePipeline.last.get("transformer") is prequant_obj
    # ...and the GGUF single-file path was not used.
    assert _FakeTransformer.last == {}


def test_transformer_quant_prequant_load_fails_falls_back_to_dense(
    fake_runtime, tmp_path, monkeypatch
):
    # A prequant source whose load returns None must fall back to dense materialise+quantise, not straight to GGUF.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    # Override the no-prequant default: a source resolves, but its load fails.
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
    fake_runtime, tmp_path, monkeypatch
):
    # The prefetch skips the base repo's transformer/ shards whenever a prequant checkpoint is expected, so a failed prequant fetch
    # would send from_pretrained after them inside the load lock, after eviction, unpreemptable, past a 100% progress report.
    # With the shards unstaged the dense fallback must be refused for the GGUF build.
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
    assert _FakeTransformer.last["path"]  # ...which loaded


def test_run_load_flags_the_transformer_prefetched_from_the_staged_file_list(monkeypatch):
    # The gate is only as good as its input, so load_pipeline reads what the prefetch ACTUALLY staged off the returned file
    # list. A failed size estimate returns no base files and must close the fallback the same way.
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


def test_transformer_quant_falls_back_to_gguf_on_failure(fake_runtime, tmp_path, monkeypatch):
    # A dense/quant failure (here quantize returns None) must fall back to the GGUF build, not error.
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
    assert status["transformer_quant"] is None  # fell back
    assert _FakeTransformer.last["path"]  # GGUF from_single_file used


def test_transformer_quant_skipped_when_plan_offloads(fake_runtime, tmp_path, monkeypatch):
    # The dense bf16 transformer only fits resident, so when the plan would offload (low_vram) the fast path is skipped.
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
    assert _FakeTransformer.last["path"]  # GGUF path used


def test_dense_quant_skipped_when_dense_transformer_does_not_fit(
    fake_runtime, tmp_path, monkeypatch
):
    # The GGUF fits resident but the dense bf16 transformer does not: skip the fast path up front and load GGUF resident, not evicted then offloaded.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    # A scheme resolves and there is no prequant, so the dense-fit re-check runs against a will-not-fit dense transformer.
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
        # GGUF budget fits (real plan -> none); the dense-transformer preflight does not.
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
    assert _FakeTransformer.last["path"]  # GGUF path used


def test_dense_quant_prequant_proceeds_but_forbids_dense_fallback(
    fake_runtime, tmp_path, monkeypatch
):
    # With a prequant checkpoint a dense misfit must NOT decline the fast path, but the dense re-check still gates the in-loader fallback.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    # usable_ (not resolve_): the re-check only honours a source the loader would accept, so the fake presents a usable one.
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: "prequant/path")
    # Large dense shards cached: if the re-check ran, it would wrongly decline the fast path.
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
        # Scoped to this backend: begin_load runs on a daemon thread and _plan_memory is patched
        # on the CLASS, so counting every instance lets a stray load land in this assertion.
        if transformer_resident_override_mib is not None and self is backend:
            dense_refit_ran.append(True)
            # GGUF budget fits (real plan -> none); the dense-transformer preflight does not.
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
    fake_runtime, tmp_path, monkeypatch
):
    # A transient foreign allocation makes an empty card look full and the replan declines resident, but the candidate fits total capacity, so the loader retries once on a fresh settled snapshot.
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
            # Initial GGUF plan: force offload so the candidate replan branch is entered.
            return dataclasses.replace(real, offload_policy = "model")
        replan_calls.append(True)
        if len(replan_calls) == 1:
            # First replan: the transient undercount. Required fits total capacity, so a retry must follow.
            return types.SimpleNamespace(
                offload_policy = "model",
                estimates = {"resident_required_mib": 90_228, "safe_device_budget_mib": 40_000},
                device_memory = types.SimpleNamespace(
                    total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
                ),
                reasons = ("companions exceed budget",),
            )
        # Retry: the transient cleared; resident.
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


def test_dense_quant_replan_no_retry_when_capacity_truly_short(fake_runtime, tmp_path, monkeypatch):
    # When the candidate does NOT fit total capacity, the decline is real: no retry.
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
    # transformer_quant + adapters with the dense build declined: the GGUF fallback cannot bake adapters, so completing would silently generate without them behind an HTTP success.
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


def test_declined_dense_without_loras_still_falls_back_to_gguf(fake_runtime, tmp_path, monkeypatch):
    # The plain decline (no adapters requested) keeps the silent GGUF fallback: weight-0 adapters count as "none".
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
    # A LoRA bake skips the prequant shortcut, attaches the adapters before quantize_transformer (post-quant torchao dispatch TypeErrors), and marks the pipe baked.
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
    assert prequant_consulted == []  # prequant shortcut skipped for the bake
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
    # A quantized pipe built without adapters cannot take one at generation time (topology frozen after quantize_ + compile), so return a clean 400 telling the client to reload.
    backend = DiffusionBackend()
    pipe = _BakePipe()
    with pytest.raises(ValueError, match = "Reload the model with the adapter selection"):
        backend._apply_loras(_quant_lora_state(pipe), [("sloth", 1.0)], threading.Event())
    # ...but a no-adapter generation on the same pipe stays a plain no-op.
    backend._apply_loras(_quant_lora_state(pipe), [], threading.Event())
    assert pipe.calls == []


def test_apply_loras_quant_baked_matrix(monkeypatch):
    # Baked pipe: the same set is a no-op; a weight-only change calls set_adapters; empty scales all to 0 (the quantized base); a different set errors.
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
    # same set: no-op
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 0.8)], ev)
    assert pipe.calls == []
    # weight-only change: live set_adapters + marker update
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 1.4)], ev)
    assert pipe.calls == [("set", ("sloth",), (1.4,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 1.4),)
    # empty: scale everything to 0 (quantized base output), marker keeps paths
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert pipe.calls == [("set", ("sloth",), (0.0,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 0.0),)
    # empty again after zeroing: no further calls
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert len(pipe.calls) == 1
    # different adapter set: topology change -> reload error
    pipe = baked_pipe()
    with pytest.raises(ValueError, match = "Reload the model with the new adapter selection"):
        backend._apply_loras(_quant_lora_state(pipe), [("other", 1.0)], ev)


def test_baked_lora_names_survive_being_disabled_at_generate_time(monkeypatch):
    # A generate with no `loras` zeroes every baked adapter and _active_lora_pairs drops zero-weight entries, so a baked
    # load's APPLIED set is always empty. Baked-and-disabled is not never-baked, so record it separately.
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

    # A non-baked pipe reports no bake, whatever is attached at generate time.
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
    ):
        calls["base"] = base
        calls["transformer"] = transformer
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
    # _assemble_pipe reads base only to FETCH, so the loader gets the ungated mirror.
    assert calls == {"base": "unsloth/Krea-2-Turbo", "transformer": marker, "device": "cuda:0"}


def test_dense_quant_unusable_prequant_path_runs_dense_refit(fake_runtime, tmp_path, monkeypatch):
    # A request prequant path the loader refuses resolves to no usable source, so the dense-fit re-check must run and decline up front instead of OOMing after eviction.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    # The real usable_prequant_source refuses a non-allowlisted path (tested elsewhere); None pins that outcome here.
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
        # Scoped to this backend: begin_load runs on a daemon thread and _plan_memory is patched
        # on the CLASS, so counting every instance asserted [True, True] on slower CI runners.
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
    # Unusable path -> no prequant shortcut -> the dense fit re-check ran.
    assert dense_refit_ran == [True]
    assert backend.status()["loaded"] is True


def test_transformer_quant_unsupported_scheme_skips_dense_download(
    fake_runtime, tmp_path, monkeypatch
):
    # An explicit unsupported scheme must fail BEFORE materialising the multi-GB dense transformer, then fall back to GGUF.
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
    assert status["transformer_quant"] is None  # fell back to GGUF
    assert _FakeTransformer.last["path"]  # GGUF from_single_file used


def test_base_file_downloaded_include_transformer_flag():
    # Default: transformer/ shards are the GGUF's job, so they are excluded; the dense transformer-quant path opts them back in.
    from core.inference.diffusion import _base_file_downloaded

    assert _base_file_downloaded("transformer/diffusion_pytorch_model-00001.safetensors") is False
    assert (
        _base_file_downloaded(
            "transformer/diffusion_pytorch_model-00001.safetensors", include_transformer = True
        )
        is True
    )
    # The flag must not admit anything else that is normally excluded.
    assert _base_file_downloaded("assets/teaser.png", include_transformer = True) is False
    assert _base_file_downloaded("README.md", include_transformer = True) is False


def test_dense_quant_prefetch_capacity_gate(fake_runtime, monkeypatch):
    # On a device too small for even the candidate post-quant resident set, widening would fetch multi-GB shards only to run the GGUF as-is, so the gate compares steady_total against TOTAL capacity.
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
    # int8 qwen steady (~22 GB DiT + 17 GB companions) cannot fit a 24 GB card.
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", candidate_with(39_900))
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is False
    # A candidate that fits total capacity still widens.
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", candidate_with(12_000))
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is True
    # Unknown sizes keep the old behaviour (widen: the loader may still take the dense path).
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(prequant = False),
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "int8"}) is True


def test_dense_quant_prefetch_needed_gates(fake_runtime, monkeypatch):
    # The transformer/ prefetch widens exactly when load_pipeline takes the dense-quant path, deferring to resolve_dense_quant_candidate. An explicit Speed="off" load never widens.
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
        # A real (non-prequant) dense-quant candidate: the loader takes the dense build needing the base repo bf16 shards.
        return types.SimpleNamespace(prequant = False)

    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", fake_candidate)

    # Explicit fp8 widens; the resolved mode is threaded through to the candidate resolver.
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is True
    assert seen[-1] == "fp8"
    # UNSET defaults to the hardware ladder, so it widens, threading auto.
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert seen[-1] == "auto"
    # A definite-offload policy forces offload whatever the candidate's footprint, so balanced / low_vram must NOT widen and pull shards the GGUF path never uses.
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
    # The gate short-circuits BEFORE resolving the candidate (no wasted resolve).
    assert len(seen) == before
    # An explicit memory_mode still consults the candidate: fast/auto can flip resident, so they widen when it is dense-viable.
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast"}
        )
        is True
    )
    # A cpu_offload flag is overridden by an explicit resident memory_mode, so it still widens.
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast", "cpu_offload": True}
        )
        is True
    )
    # An explicit off pins running the GGUF as-is, so never widen.
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "none"}) is False
    # An explicit Speed="off" (bit-exact) load suppresses the dense path, so never widen.
    assert (
        backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8", "speed_mode": "off"})
        is False
    )
    # A prequant candidate loads the small pre-quantized checkpoint, not the base dense shards, so the widened prefetch must NOT fire.
    monkeypatch.setattr(
        dmod, "resolve_dense_quant_candidate", lambda **kw: types.SimpleNamespace(prequant = True)
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False
    # No viable candidate (unsupported scheme / no disk room) never widens; the disk guard averts filling the cache volume.
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False


# ── an auto quant never buys a second denoiser for a GGUF pick ────────────────


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

    # The fast path never ran, so nothing fetched a second transformer.
    assert _dense_calls(calls, backend) == []
    assert status["loaded"] is True
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # the GGUF the user picked


def test_auto_quant_takes_a_hosted_prequant_that_is_already_cached(
    fake_runtime, tmp_path, monkeypatch
):
    # Free shortcuts are still taken: dense+torchao beats per-matmul dequant and costs no bytes.
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
    # adapter still takes the dense route, which this runtime reports rather than silently drops.
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
    fake_runtime, tmp_path, monkeypatch
):
    # Only the AUTO-derived case is restricted: asking for fp8 asks for the artifact serving it.
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
    # A LoRA bake needs the DENSE transformer and the GGUF fallback cannot carry the adapters.
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
    # while the load ran the cached prequant. Both must read the same rule.
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
    # Cached, so the decline does not fire and sizing is actually reached.
    _stub_hosted_prequant(monkeypatch, cached = True)

    backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.0)]})
    backend._dense_quant_prefetch_needed(fam, {"loras": [("adapter", 0.8)]})

    assert forced == [False, True]


def test_the_plan_reads_pydantic_lora_specs_as_the_load_reads_tuples(fake_runtime, monkeypatch):
    # /images/load sends (id, weight) tuples but /images/download-plan passes LoraSpec models,
    # whose unpacking yields (field, value) pairs, so a plain (_lid, w) unpack binds w to
    # ("weight", 0.0) and reads a disabled adapter as an active bake.
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

    # A weighted spec is still a bake, so the normalisation did not disable the candidate path.
    backend._dense_quant_prefetch_needed(fam, {"loras": [LoraSpec(id = "adapter", weight = 0.8)]})
    assert consulted != []


def test_the_plan_reads_zero_weight_loras_exactly_as_the_load_does(fake_runtime, monkeypatch):
    # The plan and the load must agree on what a bake is: gating the prefetch on the raw list
    # stages transformer/ shards for a load that runs the GGUF. The return value alone does not
    # show it, so this asserts the decline happened on the cache verdict, BEFORE sizing.
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

    # A real bake still sizes the dense build, so the fix did not disable the candidate path.
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

    _stub_hosted_prequant(monkeypatch, cached = False)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    # Declined on the cache verdict alone, BEFORE any candidate sizing.
    assert consulted == []
    # Cached, or an explicit request: the candidate decides as before.
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False
    _stub_hosted_prequant(monkeypatch, cached = True)
    assert backend._dense_quant_prefetch_needed(fam, {}) is False
    assert len(consulted) == 2


def test_diffusion_status_response_carries_resolved():
    # The backend records per-control auto-policy provenance on state.resolved, so the response model must declare the field or Pydantic drops it.
    from models.inference import DiffusionStatusResponse

    rec = {"transformer_quant": {"value": "fp8", "source": "auto", "reason": "blackwell"}}
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    # The typed field coerces the record into DiffusionResolvedControl objects; the serialized form must round-trip.
    assert resp.model_dump()["resolved"] == rec
    # Absent by default (nothing resolved / native engine).
    assert DiffusionStatusResponse(loaded = False).resolved is None


def test_companion_cache_bytes_local_dir_excludes_transformer(tmp_path):
    # A local diffusers base: sum the on-disk VAE / text-encoder weights, excluding transformer/ (the GGUF supplies it) and non-weight files.
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "model.safetensors").write_bytes(b"y" * 50)
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"z" * 9999)
    (tmp_path / "model_index.json").write_bytes(b"{}")  # non-weight file, ignored
    total = DiffusionBackend._companion_cache_bytes(str(tmp_path))
    assert total == 150  # vae + text_encoder only; transformer/ and json excluded


def test_plan_memory_dense_replan_does_not_double_count_prefetched_transformer(monkeypatch):
    # The prefetched transformer/ shards land in the same blob cache _companion_cache_bytes sums, so reading it would double-count. The re-plan must still stay resident.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

    backend = DiffusionBackend()
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    # 40 GiB card: fits transformer + real companions + headroom, but not a second copy of the bf16 transformer.
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 40000, 40960),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 4000)
    # The cache is inflated by the prefetched transformer; if the re-plan consulted it the plan would offload.
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
        transformer_resident_override_mib = 12000,  # int8 candidate transient (~half bf16)
        companion_override_mib = 8000,  # auto-policy text-encoder + VAE estimate
    )
    # 12000 + 8000 + 4000 + 2048 overhead = 26048 MiB, fits the ~36 GiB budget. A double-count would have exceeded it and offloaded.
    assert plan.offload_policy == OFFLOAD_NONE


def _split_cache_roots(
    tmp_path,
    monkeypatch,
    *,
    register_root = False,
):
    """Studio's live cache root and a second one holding what a mid-session cache-folder change
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
    """A base repo cached ONLY under the other cache root, with Studio's live root empty: what
    a mid-session cache-folder change leaves behind, handed back as ``_base_local_dir``. Sparse."""
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = register_root)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    for rel, mib in (
        ("text_encoder/model.safetensors", 150),
        ("vae/diffusion_pytorch_model.safetensors", 50),
        # Excluded from the companion total: on a GGUF load the single file supplies the transformer.
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
    # _companion_cache_bytes resolves a hub id under hub_cache_dir() ONLY, so a base served from
    # the import-time root budgets as zero while from_pretrained loads it off that snapshot: the
    # auto plan stays resident and OOMs on companions it never counted.
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

    # The live root holds none of it, so the hub-id scan is blind to 200 MiB of real companions.
    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 0
    blind = _plan()
    assert blind.estimates["companion_dense_mib"] is None
    # 300 + 0 + 100 + 2048 = 2448 <= 2509: resident, and the 200 MiB of companions arrive unbudgeted.
    assert blind.offload_policy == OFFLOAD_NONE

    plan = _plan(base_local_dir = str(snapshot))
    # transformer/ stays excluded; only the VAE + text encoder count.
    assert plan.estimates["companion_dense_mib"] == 200
    # 300 + 200 + 100 + 2048 = 2648 > 2509, and the 2348 MiB group floor fits: stream the transformer.
    assert plan.offload_policy == OFFLOAD_GROUP


def test_plan_memory_sizes_a_pipeline_load_from_the_other_root_snapshot(monkeypatch, tmp_path):
    # Same hole on the full-pipeline branch, where the whole repo IS the base: _cache_bytes walks
    # the live root's blobs, so a repo served from the other root sizes as unknown and a 4 GiB
    # pipeline that does not fit stays resident.
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
    # A pipeline load keeps transformer/: 4096 + 150 + 50 = 4296 MiB, well past the 2509 MiB margin.
    assert plan.estimates["model_dense_mib"] == 4296
    assert plan.offload_policy == OFFLOAD_MODEL


def test_plan_memory_keeps_companions_a_partial_staged_snapshot_omits(monkeypatch, tmp_path):
    # Same floor rule for the companion total: the preflight's snapshot can hold the manifest
    # alone, so preferring it over the hub-id scan budgets 0 for the companions a root does hold.
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
    # After a cache-folder change the multi-GB checkpoint can be served entirely from
    # huggingface_hub's import-time root. Counting only the live root leaves `downloaded` at 0
    # against a nonzero estimate, so the UI shows a healthy load stalled near 0% throughout.
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
    # blobs/ is append-only: a republished repo keeps the superseded revision's blobs forever under
    # different etags, so summing the whole dir counts that stale full copy on top of the live
    # partial one and load_progress reports "finalizing" for the rest of a multi-GB pull.
    from core.inference.diffusion import _LoadingState

    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    # Revision A, complete: what this root downloaded before the cache folder changed.
    _hub_blob(other, "org/pipe", "a1", 2000)
    _hub_blob(other, "org/pipe", "a2", 2000)
    _hub_snapshot_file(other, "org/pipe", old_rev, "transformer/shard-1.safetensors", "a1")
    _hub_snapshot_file(other, "org/pipe", old_rev, "vae/diffusion_pytorch_model.safetensors", "a2")
    # The repo was republished and the per-file root reuse serves this load through the same root:
    # refs/main moves to B first, then B's shards land one at a time.
    _hub_ref(other, "org/pipe", new_rev)
    _hub_blob(other, "org/pipe", "b1", 2000)
    _hub_snapshot_file(other, "org/pipe", new_rev, "transformer/shard-1.safetensors", "b1")
    _hub_blob(other, "org/pipe", "b2.incomplete", 200)  # 200 of the second 2000 MiB shard

    # 2000 done + 200 in flight out of a 4000 MiB revision. A's bytes are not this load's; the
    # .incomplete one is, so the bar still moves inside a shard instead of freezing per shard.
    assert DiffusionBackend._cache_bytes("org/pipe") == 2200 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/pipe", base_repo = None, expected_bytes = 4000 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"
    assert progress["fraction"] == 0.55


def test_load_progress_counts_one_logical_file_across_roots_at_two_revisions(tmp_path, monkeypatch):
    # Each root serves its OWN refs/main, so after a republish the same logical shard has a
    # different blob etag in each: an etag key never collides and both copies are summed, roughly
    # doubling the count, though only one of them is ever loaded.
    from core.inference.diffusion import _LoadingState

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    # The stale root still names revision A; the live root has moved to B.
    _hub_blob(other, "org/pipe", "a1", 1000)
    _hub_snapshot_file(other, "org/pipe", old_rev, "transformer/shard-1.safetensors", "a1")
    _hub_ref(other, "org/pipe", old_rev)
    _hub_blob(live, "org/pipe", "b1", 1000)
    _hub_snapshot_file(live, "org/pipe", new_rev, "transformer/shard-1.safetensors", "b1")
    _hub_ref(live, "org/pipe", new_rev)

    # One shard at one logical path, so one count -- not 2000.
    assert DiffusionBackend._cache_bytes("org/pipe") == 1000 * 1024 * 1024

    backend = DiffusionBackend()
    backend._loading = _LoadingState(
        repo_id = "org/pipe", base_repo = None, expected_bytes = 2000 * 1024 * 1024
    )
    progress = backend.load_progress()
    assert progress["phase"] == "downloading"  # not "finalizing" off a phantom second copy
    assert progress["fraction"] == 0.5


def test_companion_bytes_union_a_base_the_prefetch_split_across_roots(tmp_path, monkeypatch):
    # reuse_other_cache_root resolves EACH file through whichever root holds it, so a moved cache
    # can leave the text encoder in the old snapshot while the VAE lands in the live one. Those
    # snapshots are disjoint PARTS of one revision, so sizing off the larger one under-budgets.
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
    # 100 + 350 + 100 + 2048 = 2598 > the 2509 MiB resident margin, so stream the transformer.
    # Off the larger half alone (200) it reads 2448 and stays resident, and the 150 MiB the other
    # root holds arrives unbudgeted on a card with nothing left for it.
    assert plan.offload_policy != OFFLOAD_NONE
    assert plan.offload_policy == OFFLOAD_GROUP


def test_companion_bytes_skip_a_superseded_revision_in_the_same_root(tmp_path, monkeypatch):
    # The merge is per FILE, so a repo that repacked its shards between revisions would count both
    # namings and force offload on a base that fits. Only refs/main's revision is read.
    live, _other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    old_rev, new_rev = "a" * 40, "b" * 40
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        _sparse_snapshot_file(live, "bfl/base", old_rev, f"text_encoder/{shard}", 100)
    _sparse_snapshot_file(live, "bfl/base", new_rev, "text_encoder/model.safetensors", 200)
    _hub_ref(live, "bfl/base", new_rev)

    # 200, not the 400 that merging both revisions' disjoint file names would report.
    assert DiffusionBackend._companion_cache_bytes("bfl/base") == 200 * 1024 * 1024


def test_plan_memory_sizes_a_pipeline_split_across_both_cache_roots(monkeypatch, tmp_path):
    # A prefetch split across roots hands back NO snapshot (one missing the rest of the files would
    # fail the load), so the plan falls back to the cache scan. Scoped to the live root that reads a
    # fraction of the repo, and from_pretrained then loads shards the plan never budgeted.
    from core.inference.diffusion_memory import OFFLOAD_MODEL

    live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    target = _small_card(monkeypatch)
    backend = DiffusionBackend()
    # base_repo deliberately unequal to repo_id: the narrow-base size table is a different path.
    fam = types.SimpleNamespace(name = "flux.1", base_repo = "unrelated/repo")
    # Downloaded into the live root, and again mirrored in both: same etag, so one file.
    _hub_blob(live, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "a" * 64, 300)
    _hub_blob(other, "bfl/base", "b" * 64, 4000)

    def _plan(**kw):
        return backend._plan_memory(
            target, None, "bfl/base", fam, None, False, kind = "pipeline", repo_id = "bfl/base", **kw
        )

    plan = _plan()
    # 300 + 4000, each counted once: a per-root sum would read 4600, the live root alone 300 (and
    # 300 + 100 + 2048 fits the 2509 MiB margin, so the 4.2 GiB repo would have stayed resident).
    assert plan.estimates["model_dense_mib"] == 4300
    assert plan.offload_policy == OFFLOAD_MODEL

    # The gated preflight excuses a base off ONE probe file, so its snapshot can hold the manifest
    # alone: preferring a staged dir outright would size this 4.2 GiB pipeline at 0.
    manifest_only = other / "models--bfl--base" / "snapshots" / ("c" * 40)
    manifest_only.mkdir(parents = True)
    (manifest_only / "model_index.json").write_bytes(b"{}")
    assert _plan(base_local_dir = str(manifest_only)).estimates["model_dense_mib"] == 4300


def test_dense_transformer_bytes_read_the_other_root_and_treat_the_snapshot_as_a_floor(
    tmp_path, monkeypatch
):
    # The dense-quant preflight sizes the bf16 transformer to decide whether to re-check the fit. A
    # base held only under the import-time root reads 0 on the live one, and a 0 skips the check, so
    # the dense build lands under a plan sized for the GGUF.
    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    snapshot = other / "models--bfl--base" / "snapshots" / ("a" * 40)
    _safetensors_with_params(
        snapshot / "transformer" / "diffusion_pytorch_model.safetensors", 1_000_000
    )

    # bf16 on load: 2 bytes/param, reached through the hub id or the staged snapshot alike.
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base") == 2_000_000
    assert (
        DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(snapshot)) == 2_000_000
    )
    # The staged snapshot is a floor, never a replacement: it can carry companions alone, and
    # letting that erase the shards a root does hold would skip the fit check again.
    bare = tmp_path / "companions-only-snapshot"
    bare.mkdir()
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(bare)) == 2_000_000
    # A staged dir under NEITHER current root: the cache folder can change again while a multi-GB
    # prefetch runs, and the snapshot it already resolved is still where the load reads the shards.
    stale = tmp_path / "stale-root" / "models--bfl--base" / "snapshots" / ("b" * 40)
    _safetensors_with_params(
        stale / "transformer" / "diffusion_pytorch_model.safetensors", 3_000_000
    )
    assert DiffusionBackend._dense_transformer_resident_bytes("bfl/base", str(stale)) == 6_000_000


@pytest.mark.parametrize("staged", [False, True])
def test_dense_fit_check_runs_for_a_base_the_live_cache_root_does_not_hold(
    fake_runtime, tmp_path, monkeypatch, staged
):
    # End of the same hole, at the load: with no usable prequant the loader materialises the dense
    # bf16 transformer, so the fit re-check has to run, and it only runs when the size lookup finds
    # the shards. For a moved cache they sit under the import-time root (staged=False) or in the
    # already-resolved snapshot a second move strands outside both roots (staged=True), and the
    # live root reads 0 for either.
    from core.inference import diffusion as dmod

    _live, other = _split_cache_roots(tmp_path, monkeypatch, register_root = True)
    root = tmp_path / "stale-root" if staged else other
    shards = root / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots" / ("a" * 40)
    _safetensors_with_params(
        shards / "transformer" / "diffusion_pytorch_model.safetensors",
        6 * 1024**3,  # 6G params -> 12 GiB resident at bf16
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
        # Scoped to this backend: begin_load runs on a daemon thread, so an earlier test's load can
        # still be in flight here and _plan_memory is patched on the CLASS.
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
    # Sizing reads the staged snapshot; the LOAD deliberately does not. diffusers treats a local
    # directory as terminal (_get_model_file raises instead of falling back to the hub) and a
    # sharded load raises per missing shard, so a partial snapshot -- what a cancelled prefetch
    # leaves -- would turn a working load into a hard failure. The hub id costs a re-download
    # instead, or 401s into the GGUF fallback, which is what main does today.
    import contextlib

    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda *a, **k: None)
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
    assert seen == ["Tongyi-MAI/Z-Image-Turbo"]


def test_reset_step_cache_helper_is_best_effort():
    # Prefer the real CacheMixin hook (_reset_stateful_cache): reset_stateful_hooks lives only on the HookRegistry, so the old lookup was a silent no-op.
    calls = []
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: calls.append("real"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    # _reset_stateful_cache wins when both are present.
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(
            _reset_stateful_cache = lambda: calls.append("real"),
            reset_stateful_hooks = lambda: calls.append("fallback"),
        )
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    # Falls back to reset_stateful_hooks for a transformer that exposes only that.
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(reset_stateful_hooks = lambda: calls.append("fallback"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["fallback"]
    # No transformer, or one without either hook, is a silent no-op (never raises).
    DiffusionBackend._reset_step_cache(types.SimpleNamespace())
    DiffusionBackend._reset_step_cache(types.SimpleNamespace(transformer = object()))


def test_generate_resets_step_cache_only_when_engaged(fake_runtime, tmp_path):
    # FBCache residuals survive on the resident transformer, so generate() must reset first, but only when a cache is engaged.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    resets = []
    # Use the real diffusers CacheMixin entry point; a genuine transformer exposes this, not reset_stateful_hooks.
    backend._state.pipe.transformer = types.SimpleNamespace(
        _reset_stateful_cache = lambda: resets.append(True)
    )
    # No cache engaged (transformer_cache is None) -> reset must NOT run.
    backend.generate(prompt = "a sloth")
    assert resets == []
    # Engage a cache; every subsequent generation resets the stateful cache first.
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    backend.generate(prompt = "a sloth")
    backend.generate(prompt = "another sloth")
    assert resets == [True, True]


def test_prefetch_returns_snapshot_dir_for_manifest(monkeypatch):
    # The prefetched manifest directory is the local snapshot root; a config-only base list returns None so the hub id stays in use.
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


def test_unload_waits_for_in_flight_denoise_before_teardown():
    # Regression: unload() must wait for a running denoise to exit before _unload_locked() tears down process-wide state it depends on.
    import threading

    backend = DiffusionBackend()

    denoise_active = {"v": False}
    teardown_saw = []  # records denoise_active at the moment _unload_locked runs

    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    started = threading.Event()
    finish = threading.Event()

    # _generate_lock is the only lock a real denoise holds for its whole body.
    def _denoise():
        with backend._generate_lock:
            denoise_active["v"] = True
            started.set()
            cancel.wait(2.0)  # unload signals this
            finish.wait(2.0)  # the test lets us finish
            denoise_active["v"] = False  # about to release _generate_lock

    def _fake_unload_locked():
        teardown_saw.append(denoise_active["v"])

    backend._unload_locked = _fake_unload_locked  # instance attr shadows the method

    d = threading.Thread(target = _denoise)
    d.start()
    assert started.wait(2.0)  # denoise holds _generate_lock

    unloaded = threading.Event()

    def _unload():
        backend.unload()
        unloaded.set()

    u = threading.Thread(target = _unload)
    u.start()
    assert cancel.wait(2.0)  # unload has signalled the denoise and is now waiting on _generate_lock
    # unload must NOT have torn down yet: it is blocked on the denoise's _generate_lock.
    assert teardown_saw == []
    assert not unloaded.wait(0.3)

    finish.set()  # let the denoise release _generate_lock
    d.join(2.0)
    u.join(2.0)
    assert unloaded.is_set()
    # Teardown ran exactly once, and only AFTER the denoise had exited.
    assert teardown_saw == [False]


# Batched generation (prompt/seed lists, per-image generators, OOM backoff)


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
    assert out["seed"] == 11  # base seed = first per-image seed
    call = backend._state.pipe.last_kwargs
    # A uniform prompt is encoded ONCE and fanned out; each image gets its own generator.
    assert call["prompt"] == "a sloth"
    assert call["num_images_per_prompt"] == 4
    assert [g.manual for g in call["generator"]] == [11, 22, 33, 44]


def test_generate_prompt_list_one_image_per_prompt(fake_runtime, tmp_path):
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "fallback", prompts = ["a", "b", "c"], seed = 100)
    assert len(out["images"]) == 3
    assert out["seeds"] == [100, 101, 102]  # derived from the base seed
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
    # The single-image call shape is the bit-identical reference: scalar prompt, one scalar generator, num_images_per_prompt=1.
    backend = _load_zimage_backend(tmp_path)
    out = backend.generate(prompt = "one", seed = 5)
    call = backend._state.pipe.last_kwargs
    assert not isinstance(call["generator"], list)
    assert call["generator"].manual == 5
    assert call["num_images_per_prompt"] == 1
    assert out["seeds"] == [5]


def test_generate_batched_seed_matches_solo_replay(fake_runtime, tmp_path):
    # Per-image reproducibility: image i of a batched call uses the exact generator seed a solo replay of that image uses.
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
    # The full batch OOMs once, then both halves run; images + seeds stay complete.
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
    assert pipe.batch_attempts == [4]  # no backoff retries on a non-OOM error


def test_generate_broadcasts_negative_prompt_across_a_mixed_prompt_batch(fake_runtime, tmp_path):
    # A prompt list needs a matching negative list: encode_prompt asserts equal lengths, and pipes that encode the negative separately would build batch-1 embeds against batch-N latents.
    backend = _load_zimage_backend(tmp_path)
    backend.generate(prompt = "fallback", prompts = ["a", "b", "c"], negative_prompt = "blurry")
    call = backend._state.pipe.last_kwargs
    assert call["prompt"] == ["a", "b", "c"]
    assert call["negative_prompt"] == ["blurry", "blurry", "blurry"]
    # An empty negative prompt is still omitted entirely (never sent as [""] * n).
    backend.generate(prompt = "fallback", prompts = ["a", "b"])
    assert backend._state.pipe.last_kwargs["negative_prompt"] is None


def test_generate_keeps_a_scalar_negative_prompt_off_the_list_paths(fake_runtime, tmp_path):
    # Uniform-prompt and single-image forwards pass a SCALAR prompt, so the negative prompt must stay scalar too.
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
    # A forward that raises skips maybe_free_model_hooks(), so its FBCache residual stays on the transformer and the halved retry would trip over the stale batch-4 residual.
    backend = _load_zimage_backend(tmp_path)
    trace: list = []
    pipe = _TracingPipe(trace, max_images = 2)
    pipe.transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: trace.append(("reset",)))
    object.__setattr__(backend._state, "pipe", pipe)
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    out = backend.generate(prompt = "p", seeds = [1, 2, 3, 4])
    assert len(out["images"]) == 4 and out["seeds"] == [1, 2, 3, 4]
    # Every forward, including both post-OOM retries, is preceded by a reset.
    assert trace == [
        ("reset",),
        ("call", 4),
        ("reset",),
        ("call", 2),
        ("reset",),
        ("call", 2),
    ]


def test_generate_resets_the_step_cache_before_every_chunk(fake_runtime, tmp_path):
    # Same guarantee for an explicit per-forward cap (no OOM involved).
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
    def __init__(self, siblings):
        self.siblings = siblings


GB = 1024**3
# A FLUX-shaped base repo: the packaged root single and the transformer shards a plain snapshot_download would drag in and the loader never opens.
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


def _fake_hf_api(monkeypatch, repos):
    """Point HfApi.model_info at a canned sibling list per repo id."""

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _FakeInfo(repos[repo_id])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    # Download-plan tests describe their cache state explicitly. Never let a developer's real
    # Studio cache make an entry disappear from these otherwise hermetic tests.
    monkeypatch.setattr(
        DiffusionBackend, "_hub_file_is_cached", staticmethod(lambda repo_id, filename: False)
    )


def test_download_plan_scopes_the_base_repo_files(monkeypatch):
    # The plan drives the Hub download manager, so its file list must match what the loader reads: a full snapshot adds the 24 GB root single and the shards the GGUF replaces.
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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
    )
    _no_cache(monkeypatch)

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    # The base entry names the MIRROR: staged before the loader runs, so a gated id here 401s an
    # anonymous user at staging and the swap downstream is never reached.
    assert [e["repo_id"] for e in plan["entries"]] == [
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-dev",
    ]
    checkpoint, base = plan["entries"]
    assert checkpoint["files"] == ["flux1-dev-Q4_K_M.gguf"]
    assert checkpoint["bytes"] == 7 * GB
    assert "flux1-dev.safetensors" not in base["files"]
    assert not any(f.startswith("transformer/") for f in base["files"])
    assert not any(f.startswith("assets/") for f in base["files"])
    assert "model_index.json" in base["files"]
    assert "text_encoder/model.safetensors" in base["files"]
    # Sized per repo, so each download job gets its own expected bytes.
    assert base["bytes"] < 24 * GB
    assert plan["total_bytes"] == checkpoint["bytes"] + base["bytes"]


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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
    )
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename: repo_id == "unsloth/FLUX.1-dev-GGUF"
            and filename == "flux1-dev-Q4_K_M.gguf"
        ),
    )

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert [entry["repo_id"] for entry in plan["entries"]] == ["unsloth/FLUX.1-dev"]
    assert "text_encoder/model.safetensors" in plan["entries"][0]["files"]
    assert plan["total_bytes"] == plan["entries"][0]["bytes"]


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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
    )
    _all_cached(monkeypatch)
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename: True),
    )

    plan = DiffusionBackend().download_plan(
        "unsloth/FLUX.1-dev-GGUF", gguf_filename = "flux1-dev-Q4_K_M.gguf"
    )

    assert plan == {"entries": [], "total_bytes": 0}


def test_download_plan_pipeline_kind_is_one_entry(monkeypatch):
    # A pipeline load has no separate checkpoint repo: the repo IS the pipeline.
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _FLUX_BASE_SIBLINGS})

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert len(plan["entries"]) == 1
    files = plan["entries"][0]["files"]
    # The pipeline keeps its own transformer, but still drops fp16 twins and the root single.
    assert any(f.startswith("transformer/") for f in files)
    assert "flux1-dev.safetensors" not in files
    assert "text_encoder/model.fp16.safetensors" not in files


def test_download_plan_is_empty_for_a_local_path(tmp_path, monkeypatch):
    # Nothing to stage: the files are already on disk.
    local = tmp_path / "my-model"
    (local / "transformer").mkdir(parents = True)
    (local / "model_index.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr("core.inference.diffusion._resolve_base_repo", lambda *a, **k: str(local))
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )

    plan = DiffusionBackend().download_plan(str(local), gguf_filename = "weights.gguf")
    assert plan["entries"] == []


def test_download_plan_stages_the_precast_encoder_instead_of_the_dense_one(monkeypatch):
    # An fp8 text-encoder request loads a hosted PRE-CAST checkpoint, so the plan must stage that file, not the base repo's dense encoder shards (tens of GB, never opened).
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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
    )
    # The pick resolves one hosted pre-cast encoder for text_encoder_2 (flux.1 hosts its T5-XXL).
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_sources",
        lambda fam, *, te_quant_mode, target: (
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
    # text_encoder_2 dense weights are gone; text_encoder stays (no hosted artifact), as do the non-weight files the pre-cast loader meta-inits from.
    assert not any(
        f.startswith("text_encoder_2/") and f.endswith(".safetensors") for f in base["files"]
    )
    assert "text_encoder/model.safetensors" in base["files"]
    assert "model_index.json" in base["files"]
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_keeps_the_dense_encoder_without_an_fp8_request(monkeypatch):
    # No fp8 request -> no hosted checkpoint -> the dense encoder is exactly as before.
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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
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
    # A gated / renamed / unpublished artifact must NOT cost the dense encoder: the load falls back to it, so the plan stages it.
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
        DiffusionBackend, "_dense_quant_prefetch_needed", lambda self, fam, kwargs: False
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


def test_download_plan_stages_no_second_denoiser_for_an_uncached_prequant(monkeypatch):
    # The plan drives the download manager, so it must agree with the load: a declined prequant
    # stages neither its .pt nor the base transformer/ shards the dense build wanted.
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
        "Tongyi-MAI/Z-Image-Turbo",
    ]
    checkpoint, base = plan["entries"]
    assert checkpoint["files"] == ["Z-Image-Turbo-Q4_K_M.gguf"]
    # No hosted checkpoint and no dense shards: the companions are all the base repo owes.
    assert not any(f.endswith(".pt") for e in plan["entries"] for f in e["files"])
    assert not any(f.startswith("transformer/") for f in base["files"])
    assert "text_encoder/model.safetensors" in base["files"]
    assert plan["total_bytes"] == 4 * GB + base["bytes"] < 17 * GB


def test_download_plan_for_a_pipeline_kind_ignores_the_prequant_cache(monkeypatch):
    # Only a GGUF pick is restricted: a full pipeline's transformer IS the repo's.
    _fake_hf_api(monkeypatch, {"unsloth/some-pipeline": _ZIMAGE_BASE_SIBLINGS})
    _stub_hosted_prequant(monkeypatch, cached = False)

    plan = DiffusionBackend().download_plan("unsloth/some-pipeline", model_kind = "pipeline")

    assert any(f.startswith("transformer/") for f in plan["entries"][0]["files"])


# ── teardown fence ────────────────────────────────────────────────────────────


def test_unload_fences_queued_generations_while_it_waits(fake_runtime, tmp_path):
    # A queued generation holds no cancel event, so unload's signal cannot reach it, and Python locks are not FIFO, so it could get in ahead and denoise after the eject.
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
        # Sampled while unload holds both locks: exactly the window a queued generation could slip through.
        seen.append(backend._teardown_waiters)
        real_unload_locked()

    backend._unload_locked = _record_then_unload
    backend.unload()

    assert seen == [1]  # the fence was up for the whole wait
    assert backend._teardown_waiters == 0  # and released once the pipeline was gone


def test_a_raising_unload_still_drains_the_teardown_fence(fake_runtime, tmp_path, monkeypatch):
    # _unload_locked ends in clear_gpu_cache(), whose CUDA branch raises on a sticky fault. Without the finally the fence stayed up forever, refusing every later generation.
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


def test_generation_refuses_while_a_teardown_is_waiting(fake_runtime, tmp_path):
    # The fence's effect: with a teardown waiting on _generate_lock, a generation that wins the lock refuses instead of denoising on a pipeline being freed.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    assert backend.generate(prompt = "before", steps = 2)["images"]

    backend._teardown_waiters = 1
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate(prompt = "during", steps = 2)
    # Still loaded: the refusal is about the pending teardown, not a missing model.
    assert backend._state is not None

    backend._teardown_waiters = 0
    assert backend.generate(prompt = "after", steps = 2)["images"]


def test_a_superseding_load_fences_queued_generations_too(fake_runtime, tmp_path):
    # begin_load frees the old pipeline behind the same barrier, so it needs the same fence: a queued generation would otherwise run on the pipe being dropped.
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
