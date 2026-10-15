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
    assert detect_family_by_pipeline_class("Flux3DistilledModularPipeline").name == FLUX3_KLEIN_FAMILY_NAME
    assert detect_family_by_pipeline_class("Flux3ModularPipeline").name == FLUX3_KLEIN_BASE_FAMILY_NAME
    # An HF-cache snapshot leaf is a commit hash: the modular index is the only evidence.
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "Flux3DistilledModularPipeline", "_blocks_class_name": FLUX3_DISTILLED_BLOCKS})
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
        f for f in _REPO_FILES if _pipeline_file_downloaded(f) and _pipeline_selected_file(f, selection)
    }
    assert "modular_model_index.json" in picked
    assert not picked & {"model.safetensors", "ae.safetensors", "config.json", "manifest.json", "README.md"}
    assert not any(f.startswith("variants/") for f in picked)
    assert {f.split("/", 1)[0] for f in picked if "/" in f} == set(FLUX3_COMPONENTS)


def test_a_conventional_repo_without_any_index_still_refuses():
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "transformer/config.json")], sha = "x")
    failures = []
    assert (
        _pipeline_components_from_index("a/b", info, None, explicit_components = None, failures_out = failures)
        is None
    )
    assert failures
    # Without an assembler naming its folders, a modular index alone is not enough either.
    info = SimpleNamespace(siblings = [SimpleNamespace(rfilename = "modular_model_index.json")], sha = "x")
    assert _pipeline_components_from_index("a/b", info, None, explicit_components = None) is None


# ── the classic pipeline face ────────────────────────────────────────────────
def test_blocks_class_from_index_or_flow_map():
    assert flux3_blocks_class({"_blocks_class_name": FLUX3_BASE_BLOCKS}) == FLUX3_BASE_BLOCKS
    assert flux3_blocks_class({"_blocks_class_name": FLUX3_DISTILLED_BLOCKS}) == FLUX3_DISTILLED_BLOCKS
    flow_map = SimpleNamespace(config = SimpleNamespace(flow_map = True))
    cfg = SimpleNamespace(config = SimpleNamespace(flow_map = False))
    assert flux3_blocks_class({}, flow_map) == FLUX3_DISTILLED_BLOCKS
    assert flux3_blocks_class({}, cfg) == FLUX3_BASE_BLOCKS


class _FakeScheduler:
    def step(self, noise_pred, t, latents, return_dict = False):
        return (latents + 1,)


class _FakeModular:
    """Records each call and drives scheduler.step once per denoise step, like the FLUX.3 loop."""

    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.transformer = None
        self.calls = []

    def __call__(self, output = None, **kwargs):
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
    out = face(prompt = ["a", "b"], generator = ["g1", "g2"], num_inference_steps = 1, negative_prompt = "n")
    assert out.images == ["img:a", "img:b"]
    assert [c["generator"] for c in face.mp.calls] == ["g1", "g2"]
    assert all(c["num_images_per_prompt"] == 1 and c["negative_prompt"] == "n" for c in face.mp.calls)
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
    save_file({"w": torch.zeros(100, dtype = torch.float32)}, str(tmp_path / "transformer" / "diffusion_pytorch_model.safetensors"))
    save_file({"w": torch.zeros(100, dtype = torch.bfloat16)}, str(tmp_path / "variants" / "fp8r" / "model.safetensors"))
    save_file({"w": torch.zeros(100, dtype = torch.bfloat16)}, str(tmp_path / "variants" / "nvfp4" / "model.safetensors"))
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
    tokens = torch.randn((4 * 6, 128), generator = torch.Generator().manual_seed(7)).to(torch.bfloat16)
    # Token (row r, col c) is channel vector noise[0, :, r, c].
    assert torch.equal(noise[0, :, 1, 2], tokens[1 * 6 + 2])


def test_a_seeded_sized_call_gets_reference_noise_and_resized_references():
    torch = pytest.importorskip("torch")
    from PIL import Image

    face = _Face()
    gen = torch.Generator().manual_seed(5)
    face(prompt = "a", height = 64, width = 32, generator = gen, num_inference_steps = 1,
         image = [Image.new("RGB", (2048, 2048)), Image.new("RGB", (256, 128))])
    call = face.mp.calls[-1]
    assert call["latents"].shape == (1, 128, 4, 2)
    assert [im.size for im in call["image"]] == [(1024, 1024), (1448, 724)]
    # No size (the blocks pick it) or several images per generator: diffusers draws the noise.
    face(prompt = "a", generator = gen, num_inference_steps = 1)
    assert "latents" not in face.mp.calls[-1]
    face(prompt = "a", height = 64, width = 32, generator = gen, num_images_per_prompt = 2, num_inference_steps = 1)
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

    distilled = ["emb_in.image.weight", "ref_time_in.in_layer.weight", "single_blocks.0.q_proj.weight"]
    base = ["emb_in.image.weight", "time_in.in_layer.weight", "single_blocks.0.q_proj.weight"]
    assert why("flux.3-klein", "d.gguf", distilled) is None
    assert why("flux.3-klein-base", "b.gguf", base) is None
    assert "flux.3-klein-base" in why("flux.3-klein", "b.gguf", base)
    assert "flux.3-klein" in why("flux.3-klein-base", "d.gguf", distilled)
    # diffusers key names and a ComfyUI prefix are read the same way.
    assert why("flux.3-klein", "x.safetensors", ["target_timestep_embedder.linear_1.weight"]) is None
    assert why("flux.3-klein", "x.safetensors", ["model.diffusion_model.ref_time_in.in_layer.weight"]) is None
    bfl_nvfp4 = distilled + ["single_blocks.0.q_proj.weight_scale", "single_blocks.0.q_proj.weight_scale_2"]
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
        def dtype(self):  # diffusers' get_parameter_dtype: first float param, else first float buffer, else last
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
    from core.inference.diffusion_transformer_quant import apply_zero_row_guard, guard_contiguous_activations

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


def test_smoke_probe_feeds_a_broadcast_activation(monkeypatch):
    torch = pytest.importorskip("torch")
    import core.inference.diffusion_transformer_quant as tq

    seen = []

    class _Rec(torch.nn.Linear):
        def forward(self, x):
            seen.append(x.is_contiguous())
            return super().forward(x)

    monkeypatch.setattr(torch.nn, "Linear", _Rec)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    fake_ao = types.ModuleType("torchao.quantization")
    fake_ao.quantize_ = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_ao)
    monkeypatch.setattr(tq, "_make_quant_config", lambda *a, **k: None)
    assert tq._run_smoke_probe("fp8", "cpu") is True
    # Two forwards: the zero-row matrix and a stride-0 broadcast.
    assert seen == [True, False]


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
