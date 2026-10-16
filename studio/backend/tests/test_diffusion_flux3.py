# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Image Klein: family registration, the unreleased-diffusers gate, the modular-repo download
plan, and the classic-pipeline face over the modular blocks. CPU only, no network, no FLUX.3 diffusers."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from core.inference import diffusion_families as families
from core.inference.diffusion import (
    _explicit_pipeline_components,
    _is_trusted_diffusion_repo,
    _pipeline_components_from_index,
    _pipeline_file_downloaded,
    _pipeline_selected_file,
)
from core.inference.diffusion_auto_policy import family_bf16_components_gb
from core.inference.diffusion_families import (
    default_generation_params,
    detect_family,
    detect_family_by_pipeline_class,
    detect_family_by_pipeline_index,
    family_pipeline_available,
    pipeline_class_requirement,
)
from core.inference.diffusion_flux3 import (
    FLUX3_BASE_BLOCKS,
    FLUX3_COMPONENTS,
    FLUX3_DISTILLED_BLOCKS,
    FLUX3_KLEIN_BASE_FAMILY_NAME,
    FLUX3_KLEIN_FAMILY_NAME,
    _Flux3KleinPipelineMixin,
    flux3_blocks_class,
    is_flux3_family,
)

DISTILLED = "black-forest-labs/flux-3-image-klein"
BASE = "black-forest-labs/flux-3-image-klein-base"


# ── detection ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "repo_id, name",
    [
        (DISTILLED, FLUX3_KLEIN_FAMILY_NAME),
        (BASE, FLUX3_KLEIN_BASE_FAMILY_NAME),
        ("someone/flux3-klein-finetune", FLUX3_KLEIN_FAMILY_NAME),
        ("someone/flux3-klein-base-lora-merge", FLUX3_KLEIN_BASE_FAMILY_NAME),
        ("/models/FLUX-3-Image-Klein", FLUX3_KLEIN_FAMILY_NAME),
    ],
)
def test_detect_family_flux3(repo_id, name):
    fam = detect_family(repo_id)
    assert fam is not None and fam.name == name
    assert is_flux3_family(fam)
    assert fam.transformer_class == "Flux3Transformer2DModel"


def test_the_prompt_upsampler_is_not_an_image_model():
    # flux-3-image-klein-pup is a Qwen3.5 LLM: matching it to the image family would try to build a DiT from it.
    assert detect_family("black-forest-labs/flux-3-image-klein-pup") is None


def test_flux2_and_flux1_detection_unchanged():
    assert detect_family("black-forest-labs/FLUX.2-klein-4B").name == "flux.2-klein"
    assert detect_family("black-forest-labs/FLUX.2-klein-base-9B").name == "flux.2-klein"
    assert detect_family("black-forest-labs/FLUX.1-dev").name == "flux.1"


def test_override_names():
    assert detect_family("x", "flux.3-klein").name == FLUX3_KLEIN_FAMILY_NAME
    assert detect_family("x", "flux.3-klein-base").name == FLUX3_KLEIN_BASE_FAMILY_NAME
    assert FLUX3_KLEIN_FAMILY_NAME in families.supported_family_names()
    assert FLUX3_KLEIN_BASE_FAMILY_NAME in families.supported_family_names()


def test_family_records():
    distilled = detect_family(DISTILLED)
    base = detect_family(BASE)
    for fam in (distilled, base):
        # One text-to-image call takes up to 10 reference images (the BFL reference's MAX_REFERENCES).
        assert fam.reference and fam.max_condition_images == 10
        assert not fam.edit and not fam.unified_edit
        assert fam.dimension_multiple == 16
        # A single file / GGUF denoiser assembles around the base's modular companions.
        assert not fam.pipeline_only
        assert fam.supports_torch_compile
        assert fam.condition_pixel_weight == 0.11
        assert fam.img2img_pipeline_class is None and fam.inpaint_pipeline_class is None
    # The flow-map checkpoint samples without CFG; the base runs CFG with a negative prompt.
    assert distilled.uses_negative_prompt is False
    assert base.uses_negative_prompt is True


def test_modular_index_class_names_identify_the_family(tmp_path):
    assert (
        detect_family_by_pipeline_class("Flux3DistilledModularPipeline").name
        == FLUX3_KLEIN_FAMILY_NAME
    )
    assert (
        detect_family_by_pipeline_class("Flux3ModularPipeline").name == FLUX3_KLEIN_BASE_FAMILY_NAME
    )
    # An HF-cache snapshot leaf is a commit hash: the modular index is the only evidence.
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux3DistilledModularPipeline",
                "_blocks_class_name": FLUX3_DISTILLED_BLOCKS,
            }
        )
    )
    assert detect_family_by_pipeline_index(str(tmp_path)).name == FLUX3_KLEIN_FAMILY_NAME


def test_official_repos_are_trusted():
    assert _is_trusted_diffusion_repo(DISTILLED)
    assert _is_trusted_diffusion_repo(BASE)


def test_generation_defaults():
    # ComfyUI (comfy/supported_models.py Flux3) and the BFL reference: distilled 16 steps CFG 1 (edit template, BFL), base 25 at 3.5 (CFG template).
    assert default_generation_params(DISTILLED) == (16, 1.0)
    assert default_generation_params(BASE) == (25, 3.5)
    assert default_generation_params("flux3_klein_base_bf16.safetensors") == (25, 3.5)
    assert default_generation_params("black-forest-labs/FLUX.2-klein-4B") == (4, 1.0)


def test_bf16_component_table():
    for repo in (DISTILLED, BASE):
        transformer_gb, encoders_gb, vae_gb = family_bf16_components_gb(detect_family(repo))
        assert 10.5 <= transformer_gb <= 11.5  # 5.55B params
        assert 8.5 <= encoders_gb <= 9.5  # Qwen3-VL-4B
        assert vae_gb <= 0.5


# ── unreleased diffusers gate ────────────────────────────────────────────────
def test_minimum_is_the_unreleased_next_release():
    for cls in ("Flux3ModularPipeline", "Flux3DistilledModularPipeline"):
        minimum, _ = pipeline_class_requirement(cls)
        assert minimum == "0.42.0"
        assert minimum in families._UNRELEASED_MIN_DIFFUSERS
    message = families._too_old_message("Flux3DistilledModularPipeline", "flux.3-klein", "0.41.0")
    assert "has not been released yet" in message


def _fake_diffusers(monkeypatch, root, version):
    module = types.ModuleType("diffusers")
    module.__version__ = version
    module.__spec__ = SimpleNamespace(submodule_search_locations = [str(root)])
    monkeypatch.setitem(sys.modules, "diffusers", module)


def test_listing_probes_the_module_file_not_the_version(tmp_path, monkeypatch):
    # Main and the FLUX.3 branch both report 0.41.0.dev0, exactly what the 0.41.0 release without it reads as.
    fam = detect_family(DISTILLED)
    _fake_diffusers(monkeypatch, tmp_path, "0.41.0.dev0")
    assert family_pipeline_available(fam) is False
    target = tmp_path / "modular_pipelines" / "flux3" / "modular_pipeline.py"
    target.parent.mkdir(parents = True)
    target.write_text("")
    assert family_pipeline_available(fam) is True
    assert family_pipeline_available(detect_family(BASE)) is True


def test_listing_falls_back_to_the_version_when_the_package_cannot_be_located(monkeypatch):
    monkeypatch.setattr(families, "_diffusers_package_has_file", lambda relative: None)
    monkeypatch.setattr(families, "_installed_diffusers_version", lambda: "0.42.0")
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    assert family_pipeline_available(detect_family(DISTILLED)) is True
    monkeypatch.setattr(families, "_installed_diffusers_version", lambda: "0.41.0")
    assert family_pipeline_available(detect_family(DISTILLED)) is False


# ── download plan for a modular repo ─────────────────────────────────────────
_REPO_FILES = (
    "modular_model_index.json",
    "config.json",
    "manifest.json",
    "model.safetensors",  # the BFL single file, 11 GB: never fetched
    "ae.safetensors",  # the BFL autoencoder: never fetched
    "README.md",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/tokenizer.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "variants/fp8r/model.safetensors",
    "variants/nvfp4/model.safetensors",
)


def test_modular_repo_download_plan_reads_only_the_component_folders(monkeypatch):
    fam = detect_family(DISTILLED)
    assert _explicit_pipeline_components(fam) == FLUX3_COMPONENTS
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = f) for f in _REPO_FILES], sha = "x")
    selection = _pipeline_components_from_index(
        DISTILLED, info, None, explicit_components = _explicit_pipeline_components(fam)
    )
    assert selection == (frozenset(FLUX3_COMPONENTS), frozenset())
    picked = {
        f
        for f in _REPO_FILES
        if _pipeline_file_downloaded(f) and _pipeline_selected_file(f, selection)
    }
    assert "modular_model_index.json" in picked
    assert not picked & {
        "model.safetensors",
        "ae.safetensors",
        "config.json",
        "manifest.json",
        "README.md",
    }
    assert not any(f.startswith("variants/") for f in picked)
    assert {f.split("/", 1)[0] for f in picked if "/" in f} == set(FLUX3_COMPONENTS)


def test_a_conventional_repo_without_any_index_still_refuses():
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "transformer/config.json")], sha = "x")
    failures = []
    assert (
        _pipeline_components_from_index(
            "a/b", info, None, explicit_components = None, failures_out = failures
        )
        is None
    )
    assert failures
    # Without an assembler naming its folders, a modular index alone is not enough either.
    info = SimpleNamespace(
        siblings = [SimpleNamespace(rfilename = "modular_model_index.json")], sha = "x"
    )
    assert _pipeline_components_from_index("a/b", info, None, explicit_components = None) is None


# ── the classic pipeline face ────────────────────────────────────────────────
def test_blocks_class_from_index_or_flow_map():
    assert flux3_blocks_class({"_blocks_class_name": FLUX3_BASE_BLOCKS}) == FLUX3_BASE_BLOCKS
    assert (
        flux3_blocks_class({"_blocks_class_name": FLUX3_DISTILLED_BLOCKS}) == FLUX3_DISTILLED_BLOCKS
    )
    flow_map = SimpleNamespace(config = SimpleNamespace(flow_map = True))
    cfg = SimpleNamespace(config = SimpleNamespace(flow_map = False))
    assert flux3_blocks_class({}, flow_map) == FLUX3_DISTILLED_BLOCKS
    assert flux3_blocks_class({}, cfg) == FLUX3_BASE_BLOCKS


class _FakeScheduler:
    def step(
        self,
        noise_pred,
        t,
        latents,
        return_dict = False,
    ):
        return (latents + 1,)


class _FakeModular:
    """Records each call and drives scheduler.step once per denoise step, like the FLUX.3 loop."""

    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.transformer = None
        self.calls = []

    def __call__(
        self,
        output = None,
        **kwargs,
    ):
        self.calls.append(kwargs)
        latents = 0
        for i in range(kwargs["num_inference_steps"]):
            latents = self.scheduler.step(None, 1000 - i, latents)[0]
        return [f"img:{kwargs['prompt']}"] * kwargs["num_images_per_prompt"]


class _Face(_Flux3KleinPipelineMixin):
    def __init__(self, distilled = True):
        self.config = SimpleNamespace(
            blocks_class_name = FLUX3_DISTILLED_BLOCKS if distilled else FLUX3_BASE_BLOCKS
        )
        self._interrupt = False
        self.mp = _FakeModular()
        self.guidance = []

    def modular_pipeline(self):
        return self.mp

    def _set_guidance(self, mp, guidance_scale):
        self.guidance.append(guidance_scale)

    def progress_bar(self, total = None):
        return SimpleNamespace(close = lambda: None)


def test_one_prompt_many_images_is_one_call():
    face = _Face()
    out = face(prompt = "a", num_images_per_prompt = 3, num_inference_steps = 2, negative_prompt = "x")
    assert out.images == ["img:a"] * 3
    assert len(face.mp.calls) == 1
    # The distilled blocks take no negative prompt.
    assert "negative_prompt" not in face.mp.calls[0]


def test_prompt_and_generator_lists_split_into_single_prompt_calls():
    face = _Face(distilled = False)
    out = face(
        prompt = ["a", "b"], generator = ["g1", "g2"], num_inference_steps = 1, negative_prompt = "n"
    )
    assert out.images == ["img:a", "img:b"]
    assert [c["generator"] for c in face.mp.calls] == ["g1", "g2"]
    assert all(
        c["num_images_per_prompt"] == 1 and c["negative_prompt"] == "n" for c in face.mp.calls
    )
    out = face(prompt = "same", generator = ["g1", "g2", "g3"], num_inference_steps = 1)
    assert out.images == ["img:same"] * 3
    with pytest.raises(ValueError):
        face(prompt = ["a", "b"], generator = ["g1"], num_inference_steps = 1)


def test_step_callback_and_interrupt():
    face = _Face()
    seen = []

    def _cb(pipe, i, t, kwargs):
        seen.append((i, kwargs["latents"]))
        return kwargs

    face(prompt = "a", num_inference_steps = 3, callback_on_step_end = _cb, guidance_scale = 2.0)
    assert seen == [(0, 1), (1, 2), (2, 3)]
    assert face.guidance == [2.0]
    # The wrapper is gone after the call: the scheduler's own step is back.
    assert "step" not in face.mp.scheduler.__dict__

    def _cancel(pipe, i, t, kwargs):
        pipe._interrupt = True
        return kwargs

    out = face(prompt = "a", num_inference_steps = 5, callback_on_step_end = _cancel)
    assert out.images == []
    assert "step" not in face.mp.scheduler.__dict__


def test_a_prompt_must_be_text():
    with pytest.raises(ValueError):
        _Face()(prompt = None, num_inference_steps = 1)


def test_the_memory_plan_prices_component_folders_only(tmp_path):
    # The FLUX.3 repo also carries variants/fp8r and variants/nvfp4 transformer packages (6.3 + 5.3 GB) that no
    # from_pretrained reads; pricing them planned 31.7 GB for a 19.8 GB pipeline.
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from core.inference.diffusion import DiffusionBackend

    for rel in ("transformer", "variants/fp8r", "variants/nvfp4"):
        (tmp_path / rel).mkdir(parents = True)
    save_file(
        {"w": torch.zeros(100, dtype = torch.float32)},
        str(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    save_file(
        {"w": torch.zeros(100, dtype = torch.bfloat16)},
        str(tmp_path / "variants" / "fp8r" / "model.safetensors"),
    )
    save_file(
        {"w": torch.zeros(100, dtype = torch.bfloat16)},
        str(tmp_path / "variants" / "nvfp4" / "model.safetensors"),
    )
    save_file({"w": torch.zeros(100, dtype = torch.bfloat16)}, str(tmp_path / "model.safetensors"))
    sizes = DiffusionBackend._local_dir_weight_sizes(
        tmp_path, exclude_transformer = False, load_dtype = torch.bfloat16
    )
    assert sizes == {"transformer/diffusion_pytorch_model.safetensors": 200}


def test_references_are_resized_to_one_megapixel_like_comfyui():
    from PIL import Image

    from core.inference.diffusion_flux3 import resize_reference

    # ComfyUI ReferenceMultiLatent ["resize", 1]: round(side * sqrt(2**20 / (w * h))), up or down.
    assert resize_reference(Image.new("RGB", (4000, 3000))).size == (1182, 887)
    assert resize_reference(Image.new("RGB", (512, 512))).size == (1024, 1024)
    same = Image.new("RGB", (1024, 1024))
    assert resize_reference(same) is same


def test_reference_noise_is_the_bfl_layout():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_flux3 import reference_noise

    noise = reference_noise(7, 64, 96)
    assert noise.shape == (1, 128, 4, 6) and noise.dtype == torch.bfloat16
    tokens = torch.randn((4 * 6, 128), generator = torch.Generator().manual_seed(7)).to(
        torch.bfloat16
    )
    # Token (row r, col c) is channel vector noise[0, :, r, c].
    assert torch.equal(noise[0, :, 1, 2], tokens[1 * 6 + 2])


def test_a_seeded_sized_call_gets_reference_noise_and_resized_references():
    torch = pytest.importorskip("torch")
    from PIL import Image

    face = _Face()
    gen = torch.Generator().manual_seed(5)
    face(
        prompt = "a",
        height = 64,
        width = 32,
        generator = gen,
        num_inference_steps = 1,
        image = [Image.new("RGB", (2048, 2048)), Image.new("RGB", (256, 128))],
    )
    call = face.mp.calls[-1]
    assert call["latents"].shape == (1, 128, 4, 2)
    assert [im.size for im in call["image"]] == [(1024, 1024), (1448, 724)]
    # No size (the blocks pick it) or several images per generator: diffusers draws the noise.
    face(prompt = "a", generator = gen, num_inference_steps = 1)
    assert "latents" not in face.mp.calls[-1]
    face(
        prompt = "a",
        height = 64,
        width = 32,
        generator = gen,
        num_images_per_prompt = 2,
        num_inference_steps = 1,
    )
    assert "latents" not in face.mp.calls[-1]


# ── quantisation keep-list, memory rows, single file / GGUF checks ───────────
_KEEP = ("modulation", "timestep_embed", "norm_out", "txt_mode_blocks", "context_embedder")


@pytest.mark.parametrize("scheme", ["int8", "fp8", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("family", [FLUX3_KLEIN_FAMILY_NAME, FLUX3_KLEIN_BASE_FAMILY_NAME])
def test_keep_list_applies_to_every_scheme(scheme, family):
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme
    tokens = exclude_tokens_for_scheme(scheme, family)
    assert set(_KEEP) <= set(tokens)


def test_other_families_scheme_exclusions_unchanged():
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme
    for family in ("flux.1", "flux.2-klein", "qwen-image", "z-image", None):
        for scheme in ("fp8", "mxfp8", "nvfp4"):
            assert exclude_tokens_for_scheme(scheme, family) == ()


def test_keep_list_hits_the_measured_layers_only():
    from core.inference.diffusion_transformer_quant import exclude_tokens_for_scheme

    tokens = exclude_tokens_for_scheme("fp8", FLUX3_KLEIN_FAMILY_NAME)
    kept = [
        "early_stream_modulations.image.linear",
        "single_stream_modulations.txt.linear",
        "timestep_embedder.linear_1",
        "target_timestep_embedder.linear_2",
        "norm_out.linear",
        "txt_mode_blocks.3.attn.to_q",
        "context_embedder",
    ]
    quantised = [
        "mode_blocks.image.0.attn.to_q",
        "mode_blocks.image_cond.4.ff.linear_in",
        "single_transformer_blocks.27.attn.to_out.0",
        "single_transformer_blocks.0.ff.linear_out",
    ]
    assert all(any(t in name for t in tokens) for name in kept)
    assert not any(any(t in name for t in tokens) for name in quantised)


def test_measured_memory_rows():
    from core.inference import diffusion_memory as mem

    assert mem._MEASURED_IMAGE_ACTIVATION_MIB["flux.3-klein"] == (
        (301, 901, 2_437, 2_437, 3_359),
        (301, 901, 2_437, 2_437, 3_359),
    )
    assert mem._MEASURED_DENSE_EAGER_PEAK_MIB["flux.3-klein"] == (2437, 10600)
    # The base runs CFG and is unmeasured: flat estimate.
    assert "flux.3-klein-base" not in mem._MEASURED_IMAGE_ACTIVATION_MIB


def test_variant_mismatch_and_bfl_quant_packages_are_refused():
    from core.inference.diffusion_families import flux3_variant_mismatch_reason as why

    distilled = [
        "emb_in.image.weight",
        "ref_time_in.in_layer.weight",
        "single_blocks.0.q_proj.weight",
    ]
    base = ["emb_in.image.weight", "time_in.in_layer.weight", "single_blocks.0.q_proj.weight"]
    assert why("flux.3-klein", "d.gguf", distilled) is None
    assert why("flux.3-klein-base", "b.gguf", base) is None
    assert "flux.3-klein-base" in why("flux.3-klein", "b.gguf", base)
    assert "flux.3-klein" in why("flux.3-klein-base", "d.gguf", distilled)
    # diffusers key names and a ComfyUI prefix are read the same way.
    assert (
        why("flux.3-klein", "x.safetensors", ["target_timestep_embedder.linear_1.weight"]) is None
    )
    assert (
        why("flux.3-klein", "x.safetensors", ["model.diffusion_model.ref_time_in.in_layer.weight"])
        is None
    )
    bfl_nvfp4 = distilled + [
        "single_blocks.0.q_proj.weight_scale",
        "single_blocks.0.q_proj.weight_scale_2",
    ]
    assert "variants/fp8r or variants/nvfp4" in why("flux.3-klein", "model.safetensors", bfl_nvfp4)
    # BFL's ComfyUI files (flux-3-image-klein-comfy) declare their quantization; the ComfyUI loader reads them.
    assert (
        why(
            "flux.3-klein", "flux3_image_klein_sd_nvfp4.safetensors", bfl_nvfp4, comfy_declared = True
        )
        is None
    )
    # ... but the variant still has to match.
    assert "flux.3-klein" in why(
        "flux.3-klein-base", "sd_int8.safetensors", bfl_nvfp4, comfy_declared = True
    )
    # Unreadable file or another family: fail open.
    assert why("flux.3-klein", "x.gguf", None) is None
    assert why("flux.2-klein", "x.gguf", base) is None


def test_gguf_denoiser_reports_its_compute_dtype():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_flux3 import _pin_compute_dtype

    class _Packed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, dtype = torch.uint8), requires_grad = False)

        @property
        def dtype(
            self,
        ):  # diffusers' get_parameter_dtype: first float param, else first float buffer, else last
            for p in self.parameters():
                if p.is_floating_point():
                    return p.dtype
            for b in self.buffers():
                if b.is_floating_point():
                    return b.dtype
            return torch.uint8

    m = _Packed()
    assert m.dtype == torch.uint8
    _pin_compute_dtype(m, torch.bfloat16)
    assert m.dtype == torch.bfloat16
    assert "_unsloth_compute_dtype" not in m.state_dict()


def test_nvfp4_is_denied_for_flux3_only():
    from core.inference import diffusion_transformer_quant as tq

    assert "nvfp4" in tq._FAMILY_SCHEME_DENY[FLUX3_KLEIN_FAMILY_NAME]
    assert "nvfp4" in tq._FAMILY_SCHEME_DENY[FLUX3_KLEIN_BASE_FAMILY_NAME]
    assert "flux.2-klein" not in tq._FAMILY_SCHEME_DENY


def _mx_linear(torch):
    """A Linear whose weight class is named like torchao's MXTensor and whose forward asserts a contiguous input,
    the way ``to_mx`` does."""

    class MXTensor(torch.Tensor):
        pass

    class _Lin(torch.nn.Linear):
        def forward(self, x):
            assert x.is_contiguous(), "unsupported"
            return super().forward(x)

    lin = _Lin(8, 8, bias = False)
    lin.weight = torch.nn.Parameter(lin.weight.detach().as_subclass(MXTensor), requires_grad = False)
    return lin


def test_contiguous_guard_feeds_mx_linears_a_contiguous_activation():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_transformer_quant import (
        apply_zero_row_guard,
        guard_contiguous_activations,
    )

    lin = _mx_linear(torch)
    broadcast = torch.randn(1, 1, 8).expand(2, 4, 8)
    with pytest.raises(AssertionError):
        lin(broadcast)
    model = torch.nn.Sequential(lin, torch.nn.Linear(8, 8))
    # Installed by the shared post-quantisation hook every path calls; only the MX Linear is hooked, once.
    apply_zero_row_guard(model, "mxfp8", "flux.3-klein")
    assert guard_contiguous_activations(model) == 0
    out = lin(broadcast)
    assert torch.allclose(out, lin(broadcast.contiguous()))
    assert not getattr(model[1], "_unsloth_contig_hook", False)


def test_python_310_gets_a_reason_it_can_act_on(monkeypatch):
    # Diffusers main (and so the pinned FLUX.3 build) needs Python 3.11; the installer skips it below that, so the
    # refusal must not send a 3.10 user to reinstall or fetch an archive that cannot install.
    monkeypatch.setattr(families.sys, "version_info", (3, 10, 14, "final", 0))
    message = families._too_old_message("Flux3DistilledModularPipeline", "flux.3-klein", "0.41.0")
    assert "Python 3.11" in message and "3.10" in message
    assert "archive" not in message
    monkeypatch.setattr(families.sys, "version_info", (3, 12, 1, "final", 0))
    message = families._too_old_message("Flux3DistilledModularPipeline", "flux.3-klein", "0.41.0")
    assert "Python 3.11" not in message and "has not been released yet" in message


# ── hosted artifacts ─────────────────────────────────────────────────────────
def _family(name):
    return next(f for f in families._FAMILIES if f.name == name)


def test_prequant_repos_point_each_variant_at_its_own_fp8_repo():
    distilled = _family(FLUX3_KLEIN_FAMILY_NAME)
    base = _family(FLUX3_KLEIN_BASE_FAMILY_NAME)
    for scheme in ("int8", "fp8"):
        assert (
            families.family_prequant_repo(distilled, scheme, DISTILLED)
            == "unsloth/flux-3-image-klein-FP8"
        )
        assert (
            families.family_prequant_repo(base, scheme, BASE)
            == "unsloth/flux-3-image-klein-base-FP8"
        )
    # nvfp4 is denied for both families
    assert families.family_prequant_repo(distilled, "nvfp4", DISTILLED) is None
    assert families.family_prequant_repo(base, "nvfp4", BASE) is None


@pytest.mark.parametrize("variant", ["distilled", "base"])
@pytest.mark.parametrize("scheme, stem", [("int8", "INT8"), ("fp8", "FP8")])
def test_prequant_filenames_resolve_to_the_built_safetensors(scheme, stem, variant):
    from core.inference.diffusion_prequant import candidate_filenames_of, resolve_prequant_source

    name, base, model = {
        "distilled": (FLUX3_KLEIN_FAMILY_NAME, DISTILLED, "flux-3-image-klein"),
        "base": (FLUX3_KLEIN_BASE_FAMILY_NAME, BASE, "flux-3-image-klein-base"),
    }[variant]
    source = resolve_prequant_source(_family(name), scheme, base_repo = base)
    assert source.kind == "repo" and source.location == f"unsloth/{model}-FP8"
    names = candidate_filenames_of(source)
    wanted = f"{model}-{stem}.safetensors"
    # first after any ComfyUI twin (int8 tries a "-ComfyUI" name first, which neither repo hosts)
    assert [n for n in names if "-ComfyUI" not in n][0] == wanted
    # The safetensors artifact is tried before the legacy pickle name.
    assert names.index(wanted) < names.index(f"{model}-{stem}.pt")


def _comfy_int8_file(path):
    """A minimal ComfyUI int8_convrot file: one quantised Linear, no Studio metadata (BFL's own layout)."""
    import torch
    from safetensors.torch import save_file

    conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
    save_file(
        {
            "blocks.0.qkv.weight": torch.zeros(48, 16, dtype = torch.int8),
            "blocks.0.qkv.weight_scale": torch.ones(48, 1),
            "blocks.0.qkv.comfy_quant": torch.tensor(list(conf.encode()), dtype = torch.uint8),
        },
        str(path),
    )
    return str(path)


def test_the_base_int8_slot_holds_a_comfy_file_and_is_read_through_the_comfy_loader(
    tmp_path, monkeypatch
):
    """The base repo's INT8 file is BFL's ComfyUI int8_convrot denoiser, not a Studio checkpoint: the header
    probe must not refuse it for lacking Studio metadata, and the loader must hand it to the ComfyUI quant
    loader with the base family (which picks the layers and splits the fused q/k/v)."""
    import torch

    from core.inference import diffusion_comfy_quant as cq
    from core.inference import diffusion_prequant as pq

    hub = tmp_path / "hub"
    snap = hub / "models--unsloth--flux-3-image-klein-base-FP8" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (hub / "models--unsloth--flux-3-image-klein-base-FP8" / "refs").mkdir()
    (hub / "models--unsloth--flux-3-image-klein-base-FP8" / "refs" / "main").write_text("abc")
    path = _comfy_int8_file(snap / "flux-3-image-klein-base-INT8.safetensors")
    assert cq.scan_comfy_quant(path) is not None

    source = pq.resolve_prequant_source(
        _family(FLUX3_KLEIN_BASE_FAMILY_NAME), "int8", base_repo = BASE
    )
    assert (
        pq.hosted_prequant_refusal(
            source, "int8", base = BASE, cache_dir = str(hub), local_files_only = True
        )
        is None
    )

    seen = {}

    def _comfy_load(cls, file, **kw):
        seen.update(kw, path = file)
        return torch.nn.Linear(16, 16)

    monkeypatch.setattr(cq, "load_comfy_prequant", _comfy_load)
    monkeypatch.setattr(pq, "hosted_prequant_refusal", lambda *a, **k: None)
    model = pq.load_prequantized_transformer(
        object,
        BASE,
        source,
        device = "cpu",
        dtype = torch.bfloat16,
        scheme = "int8",
        cache_dir = str(hub),
        local_files_only = True,
        family = FLUX3_KLEIN_BASE_FAMILY_NAME,
    )
    assert model is not None, pq.last_prequant_failure()
    assert seen["path"] == path
    assert seen["family"] == FLUX3_KLEIN_BASE_FAMILY_NAME and seen["scheme"] == "int8"


def test_operator_mirror_serves_the_hosted_name(tmp_path, monkeypatch):
    from core.inference import diffusion_prequant as dp

    folder = tmp_path / "unsloth" / "flux-3-image-klein-FP8"
    folder.mkdir(parents = True)
    (folder / "flux-3-image-klein-INT8.safetensors").write_bytes(b"x")
    monkeypatch.setenv(dp.PREQUANT_MIRROR_ENV, str(tmp_path))
    found = dp.prequant_mirror_path(
        "unsloth/flux-3-image-klein-FP8",
        "flux-3-image-klein-INT8.safetensors",
        readable = lambda n: True,
    )
    assert found == str(folder / "flux-3-image-klein-INT8.safetensors")


def test_quantized_gguf_pick_assembles_through_the_modular_loader(monkeypatch):
    # A GGUF pick with a transformer quant builds through _assemble_pipe; FLUX.3 has no model_index.json, so
    # from_pretrained would fail there and the load would fall back to the GGUF.
    from core.inference import diffusion as d

    calls = {}

    class _Pipe:
        def to(self, device):
            calls["device"] = device
            return self

    def fake_load(repo, dtype, **kw):
        calls.update(repo = repo, **kw)
        return _Pipe()

    class _NoFromPretrained:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("FLUX.3 must not assemble via from_pretrained")

    monkeypatch.setattr(d, "load_flux3_pipeline", fake_load)
    sentinel = object()
    pipe = d.DiffusionBackend._assemble_pipe(
        _NoFromPretrained,
        DISTILLED,
        sentinel,
        "bf16",
        None,
        "cuda",
        base_local_dir = "/staged/klein",
        fam = _family(FLUX3_KLEIN_FAMILY_NAME),
        fetch_base = DISTILLED,
    )
    assert isinstance(pipe, _Pipe)
    assert calls["repo"] == "/staged/klein" and calls["transformer"] is sentinel
    assert calls["device"] == "cuda"


# ── generation splitting and local file checks ───────────────────────────────
def test_split_calls_pair_each_prompt_with_its_own_negative():
    face = _Face(distilled = False)
    face(
        prompt = ["a", "b"],
        generator = ["g1", "g2"],
        num_inference_steps = 1,
        negative_prompt = ["na", "nb"],
    )
    assert [c["negative_prompt"] for c in face.mp.calls] == ["na", "nb"]
    face = _Face(distilled = False)
    face(prompt = "same", generator = ["g1", "g2"], num_inference_steps = 1, negative_prompt = ["n"])
    assert [c["negative_prompt"] for c in face.mp.calls] == ["n", "n"]
    with pytest.raises(ValueError):
        _Face(distilled = False)(
            prompt = ["a", "b", "c"],
            generator = ["g1", "g2", "g3"],
            num_inference_steps = 1,
            negative_prompt = ["x", "y"],
        )


def test_every_flux3_image_is_its_own_generation_chunk():
    # One chunk per image, so each image gets its own encode / denoise phase and text-encoder release.
    from core.inference.diffusion import _generation_chunks

    jobs = [("a", 1), ("a", 2), ("b", 3)]
    for name in (FLUX3_KLEIN_FAMILY_NAME, FLUX3_KLEIN_BASE_FAMILY_NAME):
        assert _generation_chunks(_family(name), jobs, 1) == [[j] for j in jobs]
    # Other families keep one forward for the whole list.
    other = next(f for f in families._FAMILIES if f.name == "flux.2-klein")
    assert _generation_chunks(other, jobs, 1) == [jobs]


def _modular_base(tmp_path):
    root = tmp_path / "klein"
    index = {
        "_class_name": "Flux3DistilledModularPipeline",
        "_blocks_class_name": FLUX3_DISTILLED_BLOCKS,
    }
    for component, lib, cls in (
        ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration"),
        ("transformer", "diffusers", "Flux3Transformer2DModel"),
        ("vae", "diffusers", "AutoencoderKLFlux2"),
    ):
        folder = root / component
        folder.mkdir(parents = True)
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"\0" * 16)
        index[component] = [lib, cls, {"subfolder": component, "type_hint": [lib, cls]}]
    # The processor the loader reads from text_encoder/, as in the official repos.
    for asset in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (root / "text_encoder" / asset).write_text("{}")
    (root / "modular_model_index.json").write_text(json.dumps(index))
    return root


def _flux3_installed(monkeypatch):
    # The CPU suite runs on a released diffusers without the FLUX.3 blocks; the checks below come after that gate.
    from core.inference import diffusion as d
    monkeypatch.setattr(d, "assert_pipeline_class_available", lambda *a, **k: None)


def test_a_local_modular_base_is_accepted_for_a_gguf_denoiser(tmp_path, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    _flux3_installed(monkeypatch)
    base = _modular_base(tmp_path)
    ggufs = tmp_path / "ggufs"
    ggufs.mkdir()
    (ggufs / "flux-3-image-klein-Q4_K_M.gguf").write_bytes(b"GGUF")
    fam = DiffusionBackend().validate_load_request(
        str(ggufs),
        gguf_filename = "flux-3-image-klein-Q4_K_M.gguf",
        family_override = FLUX3_KLEIN_FAMILY_NAME,
        model_kind = "gguf",
        base_repo = str(base),
    )
    assert fam.name == FLUX3_KLEIN_FAMILY_NAME


def test_supplied_encoder_or_vae_files_are_refused_for_flux3(tmp_path, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    _flux3_installed(monkeypatch)
    ggufs = tmp_path / "ggufs"
    ggufs.mkdir()
    (ggufs / "flux-3-image-klein-Q4_K_M.gguf").write_bytes(b"GGUF")
    (ggufs / "vae.safetensors").write_bytes(b"\0" * 16)
    for extra in ({"vae_file": "vae.safetensors"}, {"text_encoder_files": ["vae.safetensors"]}):
        with pytest.raises(ValueError, match = "not supported for 'flux.3-klein'"):
            DiffusionBackend().validate_load_request(
                str(ggufs),
                gguf_filename = "flux-3-image-klein-Q4_K_M.gguf",
                family_override = FLUX3_KLEIN_FAMILY_NAME,
                model_kind = "gguf",
                **extra,
            )


# ── LoRA loader ──────────────────────────────────────────────────────────────
def test_the_face_carries_the_flux3_lora_loader(monkeypatch):
    # Studio's LoRA path calls load_lora_weights / set_adapters / unload_lora_weights on the pipe; the pinned
    # Diffusers build ships them for FLUX.3 as Flux3LoraLoaderMixin (transformer-only). Fake modules, so the
    # test does not depend on which diffusers this environment has.
    from core.inference import diffusion_flux3 as f3

    class DiffusionPipeline:
        pass

    class Flux3LoraLoaderMixin:
        _lora_loadable_modules = ["transformer"]

        def load_lora_weights(self, *a, **k):
            return "loaded"

        def set_adapters(self, *a, **k):
            return "set"

        def unload_lora_weights(self):
            return "unloaded"

    fake = types.ModuleType("diffusers")
    fake.DiffusionPipeline = DiffusionPipeline
    loaders = types.ModuleType("diffusers.loaders")
    loaders.Flux3LoraLoaderMixin = Flux3LoraLoaderMixin
    fake.loaders = loaders
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setitem(sys.modules, "diffusers.loaders", loaders)
    monkeypatch.setattr(f3, "_PIPELINE_CLASS", None)
    cls = f3.flux3_pipeline_class()
    assert issubclass(cls, Flux3LoraLoaderMixin) and issubclass(cls, DiffusionPipeline)
    for name in ("load_lora_weights", "set_adapters", "unload_lora_weights"):
        assert callable(getattr(cls, name, None)), name
    # The face's own __call__ still wins over anything a loader mixin defines.
    assert cls.__call__ is f3._Flux3KleinPipelineMixin.__call__
    # A build without the loader still assembles (no LoRA methods, as before).
    del loaders.Flux3LoraLoaderMixin
    monkeypatch.setattr(f3, "_PIPELINE_CLASS", None)
    assert not hasattr(f3.flux3_pipeline_class(), "load_lora_weights")


# ── local validation and variant refusal ─────────────────────────────────────
def test_local_validation_checks_the_folders_the_flux3_loader_reads(tmp_path, monkeypatch):
    # A spec naming another source (repo / folder) is ignored by load_flux3_pipeline, which reads
    # <root>/text_encoder, /transformer and /vae; validation must check those folders, not the named source.
    from core.inference.diffusion import DiffusionBackend

    _flux3_installed(monkeypatch)
    base = _modular_base(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "text_encoder"
    elsewhere.mkdir(parents = True)
    (elsewhere / "config.json").write_text("{}")
    (elsewhere / "model.safetensors").write_bytes(b"\0" * 16)
    index = json.loads((base / "modular_model_index.json").read_text())
    index["text_encoder"][2].update(pretrained_model_name_or_path = str(tmp_path / "elsewhere"))
    (base / "modular_model_index.json").write_text(json.dumps(index))
    for f in (base / "text_encoder").iterdir():
        f.unlink()
    (base / "text_encoder").rmdir()
    with pytest.raises(FileNotFoundError, match = "no complete valid"):
        DiffusionBackend().validate_load_request(
            str(base), family_override = FLUX3_KLEIN_FAMILY_NAME, model_kind = "pipeline"
        )
    ggufs = tmp_path / "ggufs"
    ggufs.mkdir()
    (ggufs / "flux-3-image-klein-Q4_K_M.gguf").write_bytes(b"GGUF")
    with pytest.raises(ValueError, match = "not a diffusers pipeline directory"):
        DiffusionBackend().validate_load_request(
            str(ggufs),
            gguf_filename = "flux-3-image-klein-Q4_K_M.gguf",
            family_override = FLUX3_KLEIN_FAMILY_NAME,
            model_kind = "gguf",
            base_repo = str(base),
        )


def test_the_variant_is_refused_before_the_resident_model_is_unloaded(tmp_path, monkeypatch):
    # The route preflight (before it takes the GPU) reads the header, so a base checkpoint under the distilled
    # family is refused while the current pipeline is still loaded.
    from safetensors.torch import save_file
    import torch
    from core.inference.diffusion import DiffusionBackend

    _flux3_installed(monkeypatch)
    base = _modular_base(tmp_path)
    picks = tmp_path / "picks"
    picks.mkdir()
    save_file({"time_in.in_layer.weight": torch.zeros(2, 2)}, str(picks / "klein-base.safetensors"))
    save_file({"ref_time_in.in_layer.weight": torch.zeros(2, 2)}, str(picks / "klein.safetensors"))
    be = DiffusionBackend()
    fam = _family(FLUX3_KLEIN_FAMILY_NAME)
    with pytest.raises(ValueError, match = "flux.3-klein-base"):
        be.preflight_base_access(
            str(picks),
            fam,
            gguf_filename = "klein-base.safetensors",
            model_kind = "single_file",
            base_repo = str(base),
            allow_network = False,
        )
    be.preflight_base_access(
        str(picks),
        fam,
        gguf_filename = "klein.safetensors",
        model_kind = "single_file",
        base_repo = str(base),
        allow_network = False,
    )


def test_gguf_header_tensor_names(tmp_path):
    gguf = pytest.importorskip("gguf")
    import numpy as np
    from core.inference.diffusion_families import gguf_tensor_names_from_header

    path = tmp_path / "t.gguf"
    writer = gguf.GGUFWriter(str(path), "flux")
    writer.add_tensor("ref_time_in.in_layer.weight", np.zeros((4, 4), dtype = np.float32))
    writer.add_tensor("emb_in.image.weight", np.zeros((4, 4), dtype = np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert gguf_tensor_names_from_header(path.read_bytes()[:4096]) == [
        "ref_time_in.in_layer.weight",
        "emb_in.image.weight",
    ]
    assert gguf_tensor_names_from_header(b"") is None


# ── staged dirs and family routing ───────────────────────────────────────────
def test_every_flux3_assembly_reads_the_staged_dir_and_names_the_family():
    # Each FLUX.3 assembly in the loader takes the staged companion dir first (a plan that found the companions under
    # another cache root must not re-resolve them) and hands the family to the variant backstop.
    import ast
    import inspect
    from core.inference import diffusion as d

    tree = ast.parse(inspect.getsource(d))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "load_flux3_pipeline"
    ]
    assert len(calls) == 3
    for call in calls:
        assert "base_local_dir" in ast.unparse(call.args[0]), ast.unparse(call.args[0])
        assert "family_name" in {k.arg for k in call.keywords}


@pytest.mark.parametrize(
    "family, index_blocks, refused",
    [
        (FLUX3_KLEIN_FAMILY_NAME, FLUX3_DISTILLED_BLOCKS, False),
        (FLUX3_KLEIN_FAMILY_NAME, FLUX3_BASE_BLOCKS, True),
        (FLUX3_KLEIN_BASE_FAMILY_NAME, FLUX3_BASE_BLOCKS, False),
        (FLUX3_KLEIN_BASE_FAMILY_NAME, FLUX3_DISTILLED_BLOCKS, True),
    ],
)
def test_a_pipeline_of_the_other_variant_is_refused_before_eviction(
    tmp_path, monkeypatch, family, index_blocks, refused
):
    from core.inference.diffusion import DiffusionBackend

    _flux3_installed(monkeypatch)
    root = _modular_base(tmp_path)
    index = json.loads((root / "modular_model_index.json").read_text())
    index["_blocks_class_name"] = index_blocks
    index["_class_name"] = (
        "Flux3DistilledModularPipeline"
        if index_blocks == FLUX3_DISTILLED_BLOCKS
        else "Flux3ModularPipeline"
    )
    (root / "modular_model_index.json").write_text(json.dumps(index))
    run = lambda: DiffusionBackend().preflight_base_access(  # noqa: E731
        str(root), _family(family), model_kind = "pipeline", allow_network = False
    )
    if refused:
        with pytest.raises(ValueError, match = "Load it with family"):
            run()
    else:
        run()


def test_the_loader_backstop_refuses_a_block_graph_of_the_other_family():
    from core.inference.diffusion_flux3 import flux3_family_blocks_mismatch, flux3_index_family

    assert (
        flux3_family_blocks_mismatch(FLUX3_KLEIN_FAMILY_NAME, FLUX3_DISTILLED_BLOCKS, "r") is None
    )
    assert "flux.3-klein-base" in flux3_family_blocks_mismatch(
        FLUX3_KLEIN_FAMILY_NAME, FLUX3_BASE_BLOCKS, "r"
    )
    assert flux3_family_blocks_mismatch(None, FLUX3_BASE_BLOCKS, "r") is None
    assert (
        flux3_index_family({"_class_name": "Flux3ModularPipeline"}) == FLUX3_KLEIN_BASE_FAMILY_NAME
    )
    assert (
        flux3_index_family({"_blocks_class_name": FLUX3_DISTILLED_BLOCKS})
        == FLUX3_KLEIN_FAMILY_NAME
    )
    assert flux3_index_family({}) is None


def test_canonical_family_names_get_the_flux3_recipe():
    from core.inference.diffusion import _generation_defaults_for

    assert default_generation_params("flux.3-klein") == (16, 1.0)
    assert default_generation_params("flux.3-klein-base") == (25, 3.5)
    # An opaque local pipeline names no variant; the resident recipe falls back to the family.
    for name, want in (
        (FLUX3_KLEIN_FAMILY_NAME, (16, 1.0)),
        (FLUX3_KLEIN_BASE_FAMILY_NAME, (25, 3.5)),
    ):
        got = _generation_defaults_for(
            "/models/my-model", None, "/models/my-model", family = _family(name)
        )
        assert (got["steps"], got["guidance"]) == want


def test_status_lists_flux3_as_a_modular_image_family(monkeypatch):
    from core.inference import diffusion_engine_router as router
    from models.inference import DiffusionStatusResponse

    monkeypatch.setattr(
        router,
        "_supported_family_capabilities",
        lambda: ("flux.2-klein", FLUX3_KLEIN_FAMILY_NAME, FLUX3_KLEIN_BASE_FAMILY_NAME),
    )
    out = router.annotate_status({"loaded": False, "repo_id": None})
    assert out["modular_families"] == [FLUX3_KLEIN_FAMILY_NAME, FLUX3_KLEIN_BASE_FAMILY_NAME]
    # The response model keeps it (a dropped field reaches the UI as no modular families at all).
    assert DiffusionStatusResponse(**out).modular_families == out["modular_families"]


# ── upstream manifest change (2026-10-16): the official manifests list the optional prompt upsampler ─────────────
def _official_manifest(variant, with_upsampler):
    """The official modular_model_index.json before (2026-10-15) and after (2026-10-16) BFL added the prompt
    upsampler components; same shape, abridged to what the checks read."""
    repo = DISTILLED if variant == "distilled" else BASE
    index = {
        "_blocks_class_name": FLUX3_DISTILLED_BLOCKS
        if variant == "distilled"
        else FLUX3_BASE_BLOCKS,
        "_class_name": "Flux3DistilledModularPipeline"
        if variant == "distilled"
        else "Flux3ModularPipeline",
        "_diffusers_version": "0.41.0.dev0",
    }

    def entry(lib, cls, source, subfolder):
        return [
            lib,
            cls,
            {
                "pretrained_model_name_or_path": source,
                "revision": None,
                "subfolder": subfolder,
                "type_hint": [lib, cls],
                "variant": None,
            },
        ]

    index["text_encoder"] = entry(
        "transformers", "Qwen3VLForConditionalGeneration", repo, "text_encoder"
    )
    index["tokenizer"] = entry("transformers", "Qwen3VLProcessor", repo, "text_encoder")
    index["transformer"] = entry("diffusers", "Flux3Transformer2DModel", repo, "transformer")
    index["vae"] = entry("diffusers", "AutoencoderKLFlux2", repo, "vae")
    if with_upsampler:
        pup = "black-forest-labs/flux-3-image-klein-pup"
        index["prompt_upsampler"] = entry(
            "transformers", "Qwen3_5ForConditionalGeneration", pup, ""
        )
        index["prompt_upsampler_processor"] = entry("transformers", "AutoProcessor", pup, "")
    return index


def _official_layout(tmp_path, variant, with_upsampler):
    """A local copy of the official repo: the three component folders and the manifest, no upsampler (it lives in
    another repo)."""
    root = tmp_path / f"{variant}-{with_upsampler}"
    for component in ("text_encoder", "transformer", "vae"):
        folder = root / component
        folder.mkdir(parents = True)
        (folder / "config.json").write_text("{}")
        (folder / "model.safetensors").write_bytes(b"\0" * 16)
    (root / "text_encoder" / "tokenizer.json").write_text("{}")
    (root / "text_encoder" / "tokenizer_config.json").write_text("{}")
    (root / "text_encoder" / "preprocessor_config.json").write_text("{}")
    (root / "modular_model_index.json").write_text(
        json.dumps(_official_manifest(variant, with_upsampler))
    )
    return root


_MANIFEST_CASES = [(v, u) for v in ("distilled", "base") for u in (False, True)]


@pytest.mark.parametrize("variant, with_upsampler", _MANIFEST_CASES)
def test_local_validation_ignores_the_optional_prompt_upsampler(
    tmp_path, monkeypatch, variant, with_upsampler
):
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import local_pipeline_components_are_complete
    from core.inference.diffusion_flux3 import FLUX3_COMPONENT_FOLDERS

    _flux3_installed(monkeypatch)
    family = FLUX3_KLEIN_FAMILY_NAME if variant == "distilled" else FLUX3_KLEIN_BASE_FAMILY_NAME
    root = _official_layout(tmp_path, variant, with_upsampler)
    assert local_pipeline_components_are_complete(
        root, "modular_model_index.json", fixed_component_folders = FLUX3_COMPONENT_FOLDERS
    )
    # A full pipeline pick, and the same folder as the companion base of a GGUF.
    DiffusionBackend().validate_load_request(
        str(root), family_override = family, model_kind = "pipeline"
    )
    ggufs = tmp_path / f"ggufs-{variant}-{with_upsampler}"
    ggufs.mkdir()
    (ggufs / "klein.gguf").write_bytes(b"GGUF")
    DiffusionBackend().validate_load_request(
        str(ggufs),
        gguf_filename = "klein.gguf",
        family_override = family,
        model_kind = "gguf",
        base_repo = str(root),
    )
    # The variant read off the manifest is unchanged by the extra entries.
    DiffusionBackend().preflight_base_access(
        str(root), _family(family), model_kind = "pipeline", allow_network = False
    )
    # A component the loader DOES read is still required.
    for f in (root / "vae").iterdir():
        f.unlink()
    assert not local_pipeline_components_are_complete(
        root, "modular_model_index.json", fixed_component_folders = FLUX3_COMPONENT_FOLDERS
    )


@pytest.mark.parametrize("variant, with_upsampler", _MANIFEST_CASES)
def test_locality_does_not_require_the_upsampler_repo(tmp_path, variant, with_upsampler):
    # The media auto-switch / cached-snapshot check: an official snapshot is ready without the ~19 GB pup repo.
    from core.inference.media_locality import _pipeline_components_present

    root = _official_layout(tmp_path, variant, with_upsampler)
    assert _pipeline_components_present(root)
    for f in (root / "transformer").iterdir():
        f.unlink()
    assert not _pipeline_components_present(root)


@pytest.mark.parametrize("with_upsampler", [False, True])
def test_the_download_plan_never_selects_the_upsampler(with_upsampler):
    info = SimpleNamespace(
        siblings = [
            SimpleNamespace(rfilename = n)
            for n in (
                "modular_model_index.json",
                "text_encoder/config.json",
                "transformer/config.json",
                "vae/config.json",
            )
        ],
        sha = "x",
    )
    selected, _ignored = _pipeline_components_from_index(
        DISTILLED,
        info,
        None,
        explicit_components = _explicit_pipeline_components(_family(FLUX3_KLEIN_FAMILY_NAME)),
    )
    assert selected == frozenset(FLUX3_COMPONENTS)


# ── review round 5 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("missing", ["text_encoder", "vae", "transformer", "tokenizer"])
def test_every_fixed_folder_is_required_even_when_the_manifest_omits_it(tmp_path, missing):
    # load_flux3_pipeline opens text_encoder/ (encoder and processor), transformer/ and vae/ whatever the manifest
    # lists, so a manifest that drops an entry must not let its missing folder through.
    from core.inference.diffusion_families import local_pipeline_components_are_complete
    from core.inference.diffusion_flux3 import FLUX3_COMPONENT_FOLDERS

    root = _official_layout(tmp_path, "distilled", True)
    index = json.loads((root / "modular_model_index.json").read_text())
    index.pop(missing)
    (root / "modular_model_index.json").write_text(json.dumps(index))
    check = lambda **kw: local_pipeline_components_are_complete(  # noqa: E731
        root, "modular_model_index.json", fixed_component_folders = FLUX3_COMPONENT_FOLDERS, **kw
    )
    assert check()  # the folders are all there: an omitted entry alone is fine
    folder = FLUX3_COMPONENT_FOLDERS[missing][0]
    if missing == "transformer":
        # A GGUF supplies the denoiser's weights, but not its config.
        (root / folder / "model.safetensors").unlink()
        assert not check()
        assert check(excluded_components = ("transformer",))
    for f in (root / folder).iterdir():
        f.unlink()
    assert not check()
    assert not check(excluded_components = ("transformer",))


def test_the_base_access_probe_reads_the_modular_manifest(tmp_path, monkeypatch):
    # A cached gated FLUX.3 base with no usable token: the probe must recognise the cached snapshot by the manifest
    # FLUX.3 actually ships, or the load is refused although every file is local.
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError
    from core.inference import diffusion as d

    cache = tmp_path / "hub"
    snap = cache / "models--black-forest-labs--flux-3-image-klein" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "modular_model_index.json").write_text("{}")
    (cache / "models--black-forest-labs--flux-3-image-klein" / "refs").mkdir()
    (cache / "models--black-forest-labs--flux-3-image-klein" / "refs" / "main").write_text("abc")
    monkeypatch.setattr(d, "hub_cache_dir", lambda: str(cache))

    class _Api:
        def model_info(self, *a, **k):
            raise GatedRepoError(
                "gated", response = SimpleNamespace(status_code = 401, headers = {}, request = None)
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert d._base_manifest_name(_family(FLUX3_KLEIN_FAMILY_NAME)) == "modular_model_index.json"
    assert d._base_manifest_name(_family("flux.2-klein")) == "model_index.json"
    d._assert_base_repo_accessible(
        DISTILLED, None, d._base_manifest_name(_family(FLUX3_KLEIN_FAMILY_NAME))
    )
    # Every call site in the loader names the manifest instead of the conventional default.
    import ast
    import inspect

    calls = [
        n
        for n in ast.walk(ast.parse(inspect.getsource(d.DiffusionBackend)))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_assert_base_repo_accessible"
    ]
    assert len(calls) == 3
    for call in calls:
        assert len(call.args) >= 3 and "_base_manifest_name" in ast.unparse(call.args[2])


def test_restoring_skipped_shards_fetches_the_modular_manifest(monkeypatch, tmp_path):
    from core.inference.diffusion import DiffusionBackend

    seen = {}
    be = DiffusionBackend()

    def fake_prefetch(repo_id, gguf, base, files, token, **kw):
        seen["files"] = list(files)
        return str(tmp_path)

    monkeypatch.setattr(be, "_prefetch_files", fake_prefetch)
    out = be._restore_skipped_transformer_shards(
        DISTILLED,
        DISTILLED,
        ["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"],
        None,
        base_local_dir = str(tmp_path),
        manifest = "modular_model_index.json",
    )
    assert out == str(tmp_path)
    assert (
        seen["files"][-1] == "modular_model_index.json" and "model_index.json" not in seen["files"]
    )


def test_the_prefetch_hands_back_the_snapshot_of_a_modular_manifest(monkeypatch, tmp_path):
    # The staged dir is recognised by FLUX.3's own manifest, so the restore and the assemblies can load off it.
    import utils.hf_xet_fallback as xf
    from core.inference.diffusion import DiffusionBackend

    snap = tmp_path / "snapshots" / "abc"

    def fake_download(repo, name, token, **kw):
        path = snap / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text("{}")
        return str(path)

    monkeypatch.setattr(xf, "hf_hub_download_with_xet_fallback", fake_download)
    be = DiffusionBackend()
    root = be._prefetch_files(
        DISTILLED,
        None,
        DISTILLED,
        ["modular_model_index.json", "vae/config.json", "text_encoder/config.json"],
        None,
        fetch_base = DISTILLED,
    )
    assert root == str(snap)


# ── review round 6: the download plan for FLUX.3 companions ─────────────────────
_OFFICIAL_LISTING = (
    "LICENSE.md",
    "README.md",
    "config.json",
    "manifest.json",
    "model.safetensors",
    "modular_model_index.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/tokenizer.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "variants/fp8r/model.safetensors",
    "variants/nvfp4/model.safetensors",
)


def _fake_hub(monkeypatch, repo_files):
    import huggingface_hub
    from core.inference.diffusion_families import mirror_repo

    # A gated BFL base is fetched from its byte-identical unsloth mirror, which the Hub lists the same way.
    repo_files = {
        **{mirror_repo(r): f for r, f in repo_files.items() if mirror_repo(r)},
        **repo_files,
    }

    class _Api:
        def __init__(self, *a, **k):
            pass

        def model_info(self, repo, *a, **k):
            if repo not in repo_files:
                from huggingface_hub.errors import RepositoryNotFoundError
                raise RepositoryNotFoundError(
                    repo, response = SimpleNamespace(status_code = 404, headers = {}, request = None)
                )
            return SimpleNamespace(
                siblings = [SimpleNamespace(rfilename = f, size = 10) for f in repo_files[repo]],
                sha = "abc",
                gated = False,
                private = False,
            )

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)


@pytest.mark.parametrize(
    "kind, name", [("gguf", "klein.gguf"), ("single_file", "klein.safetensors")]
)
def test_companion_staging_is_scoped_to_the_folders_flux3_loads(tmp_path, monkeypatch, kind, name):
    # A GGUF / single file stages the official repo's text encoder, VAE, transformer config and the modular manifest
    # (the staged-snapshot marker), never BFL's variants/ packages or the root single file.
    from safetensors.torch import save_file
    import torch
    from core.inference import diffusion as d

    _flux3_installed(monkeypatch)
    _fake_hub(monkeypatch, {DISTILLED: _OFFICIAL_LISTING})
    monkeypatch.setattr(d, "hub_cache_dir", lambda: str(tmp_path / "cold"))
    picks = tmp_path / "picks"
    picks.mkdir()
    if kind == "gguf":
        (picks / name).write_bytes(b"GGUF")
    else:
        save_file({"ref_time_in.in_layer.weight": torch.zeros(2, 2)}, str(picks / name))
    plan = d.DiffusionBackend().download_plan(
        str(picks),
        gguf_filename = name,
        base_repo = DISTILLED,
        model_kind = kind,
        family_override = FLUX3_KLEIN_FAMILY_NAME,
        allow_device_probe = False,
        memory_verdict = False,
        discover_text_encoders = False,
    )
    # Staged from the ungated unsloth mirror of the gated BFL repo, never from BFL itself.
    assert {e["repo_id"] for e in plan["entries"]} & {DISTILLED, "unsloth/flux-3-image-klein-FP8"} == {
        "unsloth/flux-3-image-klein-FP8"
    }
    files = sorted(
        f
        for e in plan["entries"]
        if e["repo_id"] == "unsloth/flux-3-image-klein-FP8"
        for f in e["files"]
    )
    assert files == [
        "modular_model_index.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/tokenizer.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]


def test_a_cold_gguf_plan_never_asks_the_gated_vendor_repo(tmp_path, monkeypatch):
    # The companions are staged from unsloth/flux-3-image-klein-FP8, so their listing comes from there too: an
    # anonymous cold load asks black-forest-labs/* nothing, not even model_info.
    import huggingface_hub
    from core.inference import diffusion as d

    _flux3_installed(monkeypatch)
    _fake_hub(monkeypatch, {DISTILLED: _OFFICIAL_LISTING})
    asked = []
    listing = huggingface_hub.HfApi.model_info

    def model_info(self, repo, *a, **k):
        asked.append(repo)
        return listing(self, repo, *a, **k)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(d, "hub_cache_dir", lambda: str(tmp_path / "cold"))
    picks = tmp_path / "picks"
    picks.mkdir()
    (picks / "klein.gguf").write_bytes(b"GGUF")
    plan = d.DiffusionBackend().download_plan(
        str(picks),
        gguf_filename = "klein.gguf",
        base_repo = DISTILLED,
        model_kind = "gguf",
        family_override = FLUX3_KLEIN_FAMILY_NAME,
        allow_device_probe = False,
        memory_verdict = False,
        discover_text_encoders = False,
    )
    assert "unsloth/flux-3-image-klein-FP8" in asked
    assert not [r for r in asked if r.startswith("black-forest-labs/")]
    assert "text_encoder/config.json" in next(
        e["files"] for e in plan["entries"] if e["repo_id"] == "unsloth/flux-3-image-klein-FP8"
    )


def test_the_flux3_manifest_check_reads_the_mirror_not_the_gated_repo(tmp_path, monkeypatch):
    import huggingface_hub
    from core.inference import diffusion_compat as compat

    asked = []
    manifest = tmp_path / "modular_model_index.json"
    manifest.write_text(json.dumps(_official_manifest("distilled", False)))

    def hf_hub_download(repo, filename, **k):
        asked.append(repo)
        return str(manifest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(compat, "_local_gguf_path", lambda *a: None)
    assert compat._flux3_pipeline_index(DISTILLED, None, True) is not None
    assert asked == ["unsloth/flux-3-image-klein-FP8"]


def test_the_dense_cache_probe_sees_only_the_flux3_transformer_folder(tmp_path, monkeypatch):
    # The AUTO dense fast path asks whether the base transformer is cached; BFL's variants/ packages are not part of
    # it, or a cached dense transformer would read as missing and the GGUF would be kept.
    import torch
    from core.inference import diffusion as d

    _flux3_installed(monkeypatch)
    _fake_hub(monkeypatch, {DISTILLED: _OFFICIAL_LISTING})
    monkeypatch.setattr(d, "hub_cache_dir", lambda: str(tmp_path / "cold"))
    seen = []

    def _needed(
        self,
        fam,
        kwargs,
        *,
        companion_files = None,
        transformer_files = None,
        **_,
    ):
        seen.append(tuple(transformer_files or ()))
        return False

    monkeypatch.setattr(d.DiffusionBackend, "_dense_quant_prefetch_needed", _needed)
    picks = tmp_path / "picks"
    picks.mkdir()
    (picks / "klein.gguf").write_bytes(b"GGUF")
    d.DiffusionBackend().download_plan(
        str(picks),
        gguf_filename = "klein.gguf",
        base_repo = DISTILLED,
        model_kind = "gguf",
        family_override = FLUX3_KLEIN_FAMILY_NAME,
        allow_device_probe = True,
        memory_verdict = False,
        discover_text_encoders = False,
    )
    assert seen, "the dense prefetch gate was never asked"
    for files in seen:
        assert files == ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",)


def test_the_flux3_single_file_transformer_config_comes_from_the_staged_base():
    # The transformer config is read from the staged companions, like the assembly after it, not re-resolved
    # through fetch_base (an offline load from another cache root would fail there).
    import inspect
    from core.inference import diffusion as d

    src = inspect.getsource(d.DiffusionBackend)
    assert '"config": (_base_local_dir or fetch_base)\n' in src
    assert (
        "if is_flux3_family(fam)\n"
        in src.split('"config": (_base_local_dir or fetch_base)', 1)[1][:200]
    )


def test_the_download_plan_reports_a_flux3_pick_of_the_other_variant(tmp_path, monkeypatch):
    # The Images page stages the plan before it calls load, so the plan carries the same verdict as the load.
    from safetensors.torch import save_file
    import torch
    from core.inference import diffusion as d

    _flux3_installed(monkeypatch)
    _fake_hub(monkeypatch, {BASE: _OFFICIAL_LISTING})
    monkeypatch.setattr(d, "hub_cache_dir", lambda: str(tmp_path / "cold"))
    picks = tmp_path / "picks"
    picks.mkdir()
    save_file({"ref_time_in.in_layer.weight": torch.zeros(2, 2)}, str(picks / "klein.safetensors"))
    plan = d.DiffusionBackend().download_plan(
        str(picks),
        gguf_filename = "klein.safetensors",
        base_repo = BASE,
        model_kind = "single_file",
        family_override = FLUX3_KLEIN_BASE_FAMILY_NAME,
        allow_device_probe = False,
        memory_verdict = False,
        discover_text_encoders = False,
    )
    assert "Load it with family flux.3-klein" in (plan["incompatible_reason"] or "")


def test_single_file_config_flow_map_follows_family():
    # A base single file on distilled companions (the gated base repo's TE / VAE are the distilled ones) must not
    # inherit flow_map=True from the companion config, and a distilled file on base companions must keep it.
    from core.inference.diffusion_families import transformer_config_overrides_for

    base_file = "diffusion_models/flux3_image_klein_bf16.safetensors"
    assert transformer_config_overrides_for(detect_family(BASE), base_file, DISTILLED) == {
        "flow_map": False
    }
    assert transformer_config_overrides_for(detect_family(DISTILLED), base_file, BASE) == {
        "flow_map": True
    }
    assert "flow_map" not in transformer_config_overrides_for(
        detect_family("black-forest-labs/FLUX.2-klein-4B"), "flux-2-klein-4b-Q4_K_M.gguf"
    )


def test_single_file_base_checkpoint_loads_on_distilled_config(tmp_path):
    # End to end through diffusers' from_single_file: a base checkpoint (no target-time embedder) with a distilled
    # config dir. Without the override the target-time embedder stays on meta and the load dies.
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "Flux3Transformer2DModel"):
        pytest.skip("diffusers without FLUX.3")
    import torch
    from safetensors.torch import save_file

    from core.inference.diffusion_families import transformer_config_overrides_for

    cls = diffusers.Flux3Transformer2DModel
    cfg = dict(
        attention_head_dim = 16,
        axes_dims_rope = [4, 4, 4, 4],
        cond_in_channels = 8,
        in_channels = 8,
        joint_attention_dim = 32,
        mlp_ratio = 2.0,
        num_attention_heads = 2,
        num_layers = 1,
        num_single_layers = 1,
        timestep_channels = 16,
    )
    cfg_dir = tmp_path / "distilled" / "transformer"
    cls.from_config({**cfg, "flow_map": True}).save_config(cfg_dir)
    base_model = cls.from_config({**cfg, "flow_map": False})
    assert getattr(base_model, "target_timestep_embedder", None) is None
    ckpt = tmp_path / "base.safetensors"
    save_file({k: v.contiguous() for k, v in base_model.state_dict().items()}, str(ckpt))
    kwargs = {
        "config": str(tmp_path / "distilled"),
        "subfolder": "transformer",
        "torch_dtype": torch.float32,
        **transformer_config_overrides_for(
            detect_family(BASE), "flux3_image_klein_bf16.safetensors"
        ),
    }
    loaded = cls.from_single_file(str(ckpt), **kwargs)
    assert loaded.config.flow_map is False
    assert not any(p.is_meta for p in loaded.parameters())


def test_comfy_declared_quant_files_pass_the_variant_check(tmp_path):
    # BFL ships its ComfyUI int8 files with per-layer .comfy_quant tensors and its NVFP4 files with a
    # _quantization_metadata header; its variants/ packages have neither and stay refused.
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from core.inference.diffusion_families import (
        assert_flux3_checkpoint_matches_family,
        flux3_comfy_declared,
    )

    base = {"emb_in.image.weight": torch.zeros(2, 2), "time_in.in_layer.weight": torch.zeros(2, 2)}
    scales = {
        "single_blocks.0.qkv.weight": torch.zeros(4, 4, dtype = torch.int8),
        "single_blocks.0.qkv.weight_scale": torch.ones(4, 1),
    }
    int8 = tmp_path / "flux3_image_klein_int8_convrot.safetensors"
    save_file(
        {
            **base,
            **scales,
            "single_blocks.0.qkv.comfy_quant": torch.tensor(
                list(b'{"format":"int8_tensorwise"}'), dtype = torch.uint8
            ),
        },
        str(int8),
    )
    nvfp4 = tmp_path / "flux3_image_klein_nvfp4.safetensors"
    save_file({**base, **scales}, str(nvfp4), metadata = {"_quantization_metadata": '{"layers": {}}'})
    package = tmp_path / "model.safetensors"
    save_file({**base, **scales}, str(package), metadata = {"quantization": "nvfp4"})

    fam = detect_family("x", "flux.3-klein-base")
    assert_flux3_checkpoint_matches_family(fam, int8)
    assert_flux3_checkpoint_matches_family(fam, nvfp4)
    with pytest.raises(ValueError, match = "variants/fp8r or variants/nvfp4"):
        assert_flux3_checkpoint_matches_family(fam, package)
    with pytest.raises(ValueError, match = "flux.3-klein-base"):
        assert_flux3_checkpoint_matches_family(detect_family("x", "flux.3-klein"), int8)
    assert flux3_comfy_declared(["a.weight"], {"_quantization_metadata": "{}"})
    assert not flux3_comfy_declared(["a.weight", "a.weight_scale"], {"quantization": "fp8r"})


def test_pick_check_reads_comfy_declaration_from_hub_header(monkeypatch):
    # The pre-eviction check reads a remote pick's header: the metadata table must reach the variant check.
    from types import SimpleNamespace as NS

    import huggingface_hub

    from core.inference import diffusion_compat as compat

    names = [
        "emb_in.image.weight",
        "time_in.in_layer.weight",
        "single_blocks.0.q_proj.weight",
        "single_blocks.0.q_proj.weight_scale",
        "single_blocks.0.q_proj.weight_scale_2",
    ]

    class _Api:
        def __init__(self, *a, **k):
            pass

        def parse_safetensors_file_metadata(
            self,
            repo_id,
            filename,
            token = None,
        ):
            md = (
                {"_quantization_metadata": "{}"}
                if repo_id.endswith("-comfy")
                else {"quantization": "nvfp4"}
            )
            return NS(tensors = {n: None for n in names}, metadata = md)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(compat, "_local_gguf_path", lambda *a, **k: None)
    fam = detect_family("x", "flux.3-klein-base")
    compat.assert_flux3_pick_compatible(
        fam,
        "black-forest-labs/flux-3-image-klein-comfy",
        "diffusion_models/flux3_image_klein_nvfp4.safetensors",
    )
    with pytest.raises(ValueError, match = "variants/fp8r or variants/nvfp4"):
        compat.assert_flux3_pick_compatible(
            fam, "black-forest-labs/flux-3-image-klein", "variants/nvfp4/model.safetensors"
        )


def test_comfy_fused_qkv_is_split_into_bfl_projections():
    # BFL's ComfyUI int8 FLUX.3 files fuse q/k/v ("<block>.qkv.weight"); the loader converts row tags, so the split
    # must hand each projection exactly its third of the rows, and drop the comfy_attention.config hint.
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "Flux3Transformer2DModel"):
        pytest.skip("diffusers without FLUX.3")
    from core.inference.diffusion_comfy_quant import _mapping

    fn, _sfm = _mapping(diffusers.Flux3Transformer2DModel)
    tags = torch.arange(12, dtype = torch.float64).view(-1, 1).expand(12, 4)
    out = fn(
        checkpoint = {
            "single_blocks.0.qkv.weight": tags.clone(),
            "single_blocks.0.comfy_attention.config": torch.zeros(3, dtype = torch.uint8),
            "content_mode_blocks.image.1.q_proj.weight": torch.ones(2, 2),
        }
    )
    pre = "single_transformer_blocks.0.attn."
    assert out[pre + "to_q.weight"][:, 0].tolist() == [0, 1, 2, 3]
    assert out[pre + "to_k.weight"][:, 0].tolist() == [4, 5, 6, 7]
    assert out[pre + "to_v.weight"][:, 0].tolist() == [8, 9, 10, 11]
    assert "mode_blocks.image.1.attn.to_q.weight" in out
    assert not any("comfy_attention" in k or "qkv" in k for k in out)


def test_flux3_family_decides_generation_defaults_over_file_names():
    # BFL's ComfyUI base files (flux3_image_klein_{bf16,int8_convrot,nvfp4}.safetensors) name no variant that reads as
    # base; under the base family (which the variant check has matched to the checkpoint) they get the base recipe.
    from core.inference.diffusion import _generation_defaults_for

    base, dist = detect_family("x", "flux.3-klein-base"), detect_family("x", "flux.3-klein")
    comfy = "black-forest-labs/flux-3-image-klein-comfy"
    for name in ("bf16", "int8_convrot", "nvfp4"):
        f = f"diffusion_models/flux3_image_klein_{name}.safetensors"
        assert _generation_defaults_for(comfy, f, BASE, None, base) == {
            "steps": 25,
            "guidance": 3.5,
        }
        assert _generation_defaults_for("/local/dir", f, DISTILLED, None, base) == {
            "steps": 25,
            "guidance": 3.5,
        }
    sd = "diffusion_models/flux3_image_klein_sd_bf16.safetensors"
    assert _generation_defaults_for(comfy, sd, DISTILLED, None, dist) == {
        "steps": 16,
        "guidance": 1.0,
    }
    # Not FLUX.3: names still lead (FLUX.1's base is schnell, a dev file keeps dev).
    flux1 = detect_family("black-forest-labs/FLUX.1-dev")
    assert (
        _generation_defaults_for(
            "x", "flux1-dev-Q4_K_M.gguf", "black-forest-labs/FLUX.1-schnell", None, flux1
        )["steps"]
        == 20
    )


def _seeded_snapshot(tmp_path, repo = "black-forest-labs/flux-3-image-klein"):
    """What a download planned around a hosted pre-quantized seed leaves in the HF cache: the official layout with
    only the transformer config, no dense transformer shards."""
    snapshot = tmp_path / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.parent.mkdir(parents = True)
    _official_layout(tmp_path, "distilled", True).rename(snapshot)
    (snapshot / "transformer" / "model.safetensors").unlink()
    (snapshot.parent.parent / "refs").mkdir()
    (snapshot.parent.parent / "refs" / "main").write_text("abc123")
    return snapshot


def test_only_the_snapshot_main_names_is_loaded_by_repo_id(tmp_path):
    # A load by repo id resolves refs/main: a commit-pinned snapshot (no refs/main) or one main no longer names keeps
    # its path, or the switch would load a different snapshot than the one the picker chose.
    from core.inference.media_locality import flux3_seeded_cache_repo

    snapshot = _seeded_snapshot(tmp_path)
    assert flux3_seeded_cache_repo(snapshot) == "black-forest-labs/flux-3-image-klein"
    (snapshot.parent.parent / "refs" / "main").write_text("def456")
    assert flux3_seeded_cache_repo(snapshot) is None
    (snapshot.parent.parent / "refs" / "main").unlink()
    assert flux3_seeded_cache_repo(snapshot) is None


def test_a_seeded_flux3_snapshot_is_planned_against_its_repo(tmp_path, monkeypatch):
    from core.inference import gpu_arbiter as arb
    from core.inference import media_locality as locality
    from core.inference.media_model_index import MediaModelPick

    snapshot = _seeded_snapshot(tmp_path)
    seen = []

    class _Planner:
        def __init__(self, entries):
            self.entries = entries

        def download_plan(self, model_path, **kwargs):
            seen.append(model_path)
            return {"total_bytes": 0 if not self.entries else 5, "entries": self.entries}

    monkeypatch.setattr(locality, "plan_gpu_ordinal", lambda: None)
    pick = MediaModelPick("black-forest-labs/flux-3-image-klein", str(snapshot))
    # seed cached: nothing to fetch, and the planner was asked about the repo the switch loads, not the folder
    monkeypatch.setattr(locality, "planners_for", lambda owner, p: [_Planner([])])
    assert locality.missing_download_bytes(arb.DIFFUSION, pick) == 0
    assert seen == ["black-forest-labs/flux-3-image-klein"]
    # seed not cached: sized like any hub pick instead of refused unsized
    monkeypatch.setattr(
        locality, "planners_for", lambda owner, p: [_Planner([{"repo_id": "unsloth/x"}])]
    )
    assert locality.missing_download_bytes(arb.DIFFUSION, pick) == 5
    assert locality.flux3_seeded_cache_repo(snapshot) == "black-forest-labs/flux-3-image-klein"


def test_only_a_seeded_flux3_snapshot_is_mapped_to_its_repo(tmp_path):
    from core.inference.media_locality import flux3_seeded_cache_repo

    snapshot = _seeded_snapshot(tmp_path)
    # a plain folder (not a cache snapshot) has no repo to load from
    plain = tmp_path / "plain"
    _official_layout(tmp_path, "base", False).rename(plain)
    (plain / "transformer" / "model.safetensors").unlink()
    assert flux3_seeded_cache_repo(plain) is None
    # the transformer config is still read from the snapshot, so it must be there
    (snapshot / "transformer" / "config.json").unlink()
    assert flux3_seeded_cache_repo(snapshot) is None
    other = _seeded_snapshot(tmp_path / "o")
    (other / "vae" / "model.safetensors").unlink()
    assert flux3_seeded_cache_repo(other) is None
    manifest = _seeded_snapshot(tmp_path / "m")
    (manifest / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline"})
    )
    assert flux3_seeded_cache_repo(manifest) is None


def test_a_snapshot_staged_for_the_int8_encoder_seed_is_still_a_seeded_snapshot(tmp_path, monkeypatch):
    # te_quant_auto is int8 ConvRot on both families, so a normally staged snapshot holds neither the dense
    # transformer shards nor the dense text-encoder shards. Rejecting it read as UNSIZED_MISSING and blocked the
    # media auto-switch; the planner, asked of the repo, decides whether the encoder seed is cached.
    from core.inference import gpu_arbiter as arb
    from core.inference import media_locality as locality
    from core.inference.media_model_index import MediaModelPick

    snapshot = _seeded_snapshot(tmp_path)
    (snapshot / "text_encoder" / "model.safetensors").unlink()
    assert locality.flux3_seeded_cache_repo(snapshot) == "black-forest-labs/flux-3-image-klein"
    seen = []

    class _Planner:
        def download_plan(self, model_path, **kwargs):
            seen.append(model_path)
            return {"total_bytes": 0, "entries": []}

    monkeypatch.setattr(locality, "plan_gpu_ordinal", lambda: None)
    monkeypatch.setattr(locality, "planners_for", lambda owner, p: [_Planner()])
    pick = MediaModelPick("black-forest-labs/flux-3-image-klein", str(snapshot))
    assert locality.missing_download_bytes(arb.DIFFUSION, pick) == 0
    assert seen == ["black-forest-labs/flux-3-image-klein"]
    # the encoder config and the tokenizer are still read from the snapshot
    (snapshot / "text_encoder" / "tokenizer.json").unlink()
    assert locality.flux3_seeded_cache_repo(snapshot) is None
    other = _seeded_snapshot(tmp_path / "c")
    (other / "text_encoder" / "model.safetensors").unlink()
    (other / "text_encoder" / "config.json").unlink()
    assert locality.flux3_seeded_cache_repo(other) is None


def test_the_auto_switch_loads_a_seeded_flux3_snapshot_by_repo(tmp_path, monkeypatch):
    import asyncio

    import routes.inference as route
    from core.inference import gpu_arbiter as arb
    from core.inference import media_auto_switch as mas
    from core.inference.media_model_index import MediaModelPick

    seen = []

    async def _load(
        request,
        subject,
        user_initiated = True,
    ):
        seen.append((request.model_path, request.display_repo_id, user_initiated))

    monkeypatch.setattr(route, "load_diffusion_model_gated", _load)
    snapshot = _seeded_snapshot(tmp_path)
    full = tmp_path / "full"
    _official_layout(tmp_path, "base", False).rename(full)
    repo = "black-forest-labs/flux-3-image-klein"
    asyncio.run(mas._start_load(arb.DIFFUSION, MediaModelPick(repo, str(snapshot)), "s"))
    asyncio.run(mas._start_load(arb.DIFFUSION, MediaModelPick("local/full", str(full)), "s"))
    assert seen == [(repo, repo, False), (str(full), "local/full", False)]


def test_flux3_reference_edits_use_the_measured_activations(monkeypatch):
    import torch

    from core.inference import diffusion as d

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    nvidia = SimpleNamespace(backend = "cuda", vendor = "nvidia", dtype = torch.bfloat16)
    distilled = _family("flux.3-klein")
    assert distilled.reference and not tuple(getattr(distilled, "reference_resolutions", ()) or ())
    act = d._calibrated_activation(distilled, nvidia)
    assert act is not None and act == d.calibrated_image_activation("flux.3-klein", max_speed = True)
    # the base runs CFG (two denoiser batches) and has no measured row: it keeps the flat estimate
    assert d._calibrated_activation(_family("flux.3-klein-base"), nvidia) is None
