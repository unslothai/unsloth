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


def test_prequant_repos_point_at_the_fp8_repo_for_the_distilled_model_only():
    distilled = _family(FLUX3_KLEIN_FAMILY_NAME)
    for scheme in ("int8", "fp8"):
        assert (
            families.family_prequant_repo(distilled, scheme, DISTILLED)
            == "unsloth/flux-3-image-klein-FP8"
        )
    # nvfp4 is denied for the family, and the base model has no hosted artifact yet.
    assert families.family_prequant_repo(distilled, "nvfp4", DISTILLED) is None
    base = _family(FLUX3_KLEIN_BASE_FAMILY_NAME)
    for scheme in ("int8", "fp8", "nvfp4"):
        assert families.family_prequant_repo(base, scheme, BASE) is None


@pytest.mark.parametrize("scheme, stem", [("int8", "INT8"), ("fp8", "FP8")])
def test_prequant_filenames_resolve_to_the_built_safetensors(scheme, stem):
    from core.inference.diffusion_prequant import candidate_filenames_of, resolve_prequant_source

    source = resolve_prequant_source(_family(FLUX3_KLEIN_FAMILY_NAME), scheme, base_repo = DISTILLED)
    assert source.kind == "repo" and source.location == "unsloth/flux-3-image-klein-FP8"
    names = candidate_filenames_of(source)
    wanted = f"flux-3-image-klein-{stem}.safetensors"
    assert wanted in names
    # The safetensors artifact is tried before the legacy pickle name.
    assert names.index(wanted) < names.index(f"flux-3-image-klein-{stem}.pt")


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


# ── review round 1 ───────────────────────────────────────────────────────────
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


# ── review round 2 ───────────────────────────────────────────────────────────
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


# ── review round 3 ───────────────────────────────────────────────────────────
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


# ── review round 4 ───────────────────────────────────────────────────────────
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

    assert flux3_family_blocks_mismatch(FLUX3_KLEIN_FAMILY_NAME, FLUX3_DISTILLED_BLOCKS, "r") is None
    assert "flux.3-klein-base" in flux3_family_blocks_mismatch(
        FLUX3_KLEIN_FAMILY_NAME, FLUX3_BASE_BLOCKS, "r"
    )
    assert flux3_family_blocks_mismatch(None, FLUX3_BASE_BLOCKS, "r") is None
    assert flux3_index_family({"_class_name": "Flux3ModularPipeline"}) == FLUX3_KLEIN_BASE_FAMILY_NAME
    assert flux3_index_family({"_blocks_class_name": FLUX3_DISTILLED_BLOCKS}) == FLUX3_KLEIN_FAMILY_NAME
    assert flux3_index_family({}) is None


def test_canonical_family_names_get_the_flux3_recipe():
    from core.inference.diffusion import _generation_defaults_for

    assert default_generation_params("flux.3-klein") == (16, 1.0)
    assert default_generation_params("flux.3-klein-base") == (25, 3.5)
    # An opaque local pipeline names no variant; the resident recipe falls back to the family.
    for name, want in ((FLUX3_KLEIN_FAMILY_NAME, (16, 1.0)), (FLUX3_KLEIN_BASE_FAMILY_NAME, (25, 3.5))):
        got = _generation_defaults_for("/models/my-model", None, "/models/my-model", family = _family(name))
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
