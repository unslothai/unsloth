# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image: the in-tree port against the official ``pipeline_moss_image.py`` (kept as test data), bit for bit on
a tiny random model, plus the scheduler, noise, family row and availability gate. CPU only; the model tests skip
without diffusers (or a diffusers without the FLUX.2 VAE)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

REFERENCE = Path(__file__).with_name("data") / "moss_image" / "pipeline_moss_image_reference.py.txt"


def _needs_model_stack():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    if not hasattr(diffusers, "AutoencoderKLFlux2"):
        pytest.skip("diffusers without AutoencoderKLFlux2")
    return torch


@pytest.fixture(scope = "module")
def moss():
    _needs_model_stack()
    from core.inference import diffusion_moss_image
    return diffusion_moss_image


@pytest.fixture(scope = "module")
def ref():
    _needs_model_stack()
    loader = importlib.machinery.SourceFileLoader("moss_image_reference", str(REFERENCE))
    spec = importlib.util.spec_from_loader("moss_image_reference", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_TINY = dict(dim = 256, ffn_dim = 512, num_heads = 2, num_layers = 24, text_dim = 64)


@pytest.fixture(scope = "module")
def transformers_pair(moss, ref):
    import torch

    torch.manual_seed(0)
    reference = ref.MOSSImageTransformer(**_TINY)
    for p in reference.parameters():
        torch.nn.init.normal_(p, std = 0.02)
    reference = reference.to(torch.bfloat16).eval()
    ours = moss.MOSSImageTransformer(**_TINY).to(torch.bfloat16).eval()
    missing, unexpected = ours.load_state_dict(reference.state_dict(), strict = True)
    assert not missing and not unexpected
    return reference, ours


def _inputs(
    batch = 1,
    lengths = (13,),
    h = 4,
    w = 6,
    seed = 1,
):
    import torch

    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(batch, 128, h, w, generator = g)
    length = max(lengths)
    embeds = torch.randn(batch, length, _TINY["text_dim"], generator = g).to(torch.bfloat16)
    mask = torch.zeros(batch, length, dtype = torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return latents, embeds, mask


def test_state_dict_keys_match_the_reference(transformers_pair):
    reference, ours = transformers_pair
    assert list(reference.state_dict()) == list(ours.state_dict())


@pytest.mark.parametrize("skip", [None, [22], [0, 7, 23]])
def test_forward_is_bitwise_the_reference(transformers_pair, skip):
    import torch

    reference, ours = transformers_pair
    latents, embeds, _ = _inputs()
    t = torch.tensor(731.0)
    with torch.no_grad():
        want = reference(latents, t, embeds, skip_layers = skip).sample
        got = ours(latents, t, embeds, skip_layers = skip).sample
        refined = ours(latents, t, refined_text = ours.refine_text(embeds), skip_layers = skip).sample
    assert torch.equal(want, got)
    assert torch.equal(want, refined)


def test_masked_batch_is_bitwise_the_reference(transformers_pair):
    import torch

    reference, ours = transformers_pair
    latents, embeds, mask = _inputs(batch = 2, lengths = (5, 11))
    with torch.no_grad():
        want = reference(latents, torch.tensor(500.0), embeds, mask).sample
        got = ours(latents, torch.tensor(500.0), embeds, mask).sample
    assert torch.equal(want, got)


def test_unmasked_forward_takes_no_host_sync(transformers_pair, monkeypatch):
    """No mask: no ``torch.all`` check and no boolean indexing, so the forward can be CUDA-graph captured."""
    import torch

    _, ours = transformers_pair
    latents, embeds, _ = _inputs()

    def _boom(*args, **kwargs):
        raise AssertionError("host-syncing mask check ran")

    monkeypatch.setattr(torch, "all", _boom)
    with torch.no_grad():
        ours(latents, torch.tensor(10.0), embeds, return_dict = False)


def test_rope_tables_are_cached_and_rebuilt_on_moves(transformers_pair):
    import torch

    _, ours = transformers_pair
    a = ours.rope.tables(4, 6, 9, "cpu", torch.bfloat16)
    b = ours.rope.tables(4, 6, 9, "cpu", torch.bfloat16)
    assert a is b
    ours.float().to(torch.bfloat16)
    assert ours.rope.bands.dtype == torch.float32
    assert ours.rope.tables(4, 6, 9, "cpu", torch.bfloat16) is not a


def test_refined_text_and_mask_are_exclusive(transformers_pair):
    import torch

    _, ours = transformers_pair
    latents, embeds, mask = _inputs()
    with pytest.raises(ValueError):
        ours(latents, torch.tensor(1.0))
    with pytest.raises(ValueError):
        ours(latents, torch.tensor(1.0), embeds, refined_text = ours.refine_text(embeds))
    with pytest.raises(ValueError):
        ours(
            latents,
            torch.tensor(1.0),
            refined_text = ours.refine_text(embeds),
            encoder_attention_mask = mask,
        )


@pytest.mark.parametrize("steps", [1, 2, 8, 28, 50])
def test_training_grid_matches_the_reference_scheduler(moss, ref, steps):
    import torch

    want = ref.MOSSImageScheduler()
    got = moss.MOSSImageScheduler(sigma_schedule = "training_grid")
    want.set_timesteps(steps)
    got.set_timesteps(steps)
    assert torch.equal(want.sigmas, got.sigmas)
    assert torch.equal(want.timesteps, got.timesteps)


def _comfy_sgm_uniform(steps, shift):
    """comfy.samplers.normal_scheduler(sgm=True) over ModelSamplingDiscreteFlow(shift, multiplier=1), restated."""
    import torch

    def sigma(t):
        return shift * t / (1 + (shift - 1) * t)

    grid = sigma(torch.arange(1, 1001, 1) / 1000)
    timesteps = torch.linspace(grid[-1], grid[0], steps + 1)[:-1]
    return [float(sigma(ts)) for ts in timesteps] + [0.0]


@pytest.mark.parametrize("steps", [4, 20, 50])
def test_default_schedule_is_comfyui_sgm_uniform(moss, monkeypatch, steps):
    import torch

    monkeypatch.delenv(moss.SIGMAS_ENV, raising = False)
    sched = moss.MOSSImageScheduler()
    sched.set_timesteps(steps)
    want = torch.tensor(_comfy_sgm_uniform(steps, 5.0), dtype = torch.float32)
    assert sched.sigmas.shape == want.shape
    assert torch.allclose(sched.sigmas, want, atol = 1e-7, rtol = 0)
    assert float(sched.sigmas[0]) == 1.0 and float(sched.sigmas[-1]) == 0.0
    assert torch.allclose(sched.timesteps, want[:-1] * 1000, atol = 1e-3, rtol = 0)


def test_env_selects_the_training_grid(moss, ref, monkeypatch):
    import torch

    monkeypatch.setenv(moss.SIGMAS_ENV, "training_grid")
    got = moss.MOSSImageScheduler()
    want = ref.MOSSImageScheduler()
    got.set_timesteps(50)
    want.set_timesteps(50)
    assert torch.equal(got.sigmas, want.sigmas)


def test_repo_scheduler_config_loads(moss, tmp_path):
    config = {
        "_class_name": "MOSSImageScheduler",
        "_diffusers_version": "0.38.0",
        "base_image_seq_len": 256,
        "num_train_timesteps": 1000,
        "shift": 5.0,
        "training_shift": 1.0,
        "use_dynamic_shifting": False,
    }
    (tmp_path / "scheduler_config.json").write_text(json.dumps(config))
    sched = moss.MOSSImageScheduler.from_pretrained(str(tmp_path))
    assert sched.config.shift == 5.0 and sched.resolved_sigma_schedule() in moss.SIGMA_SCHEDULES


def test_cpu_generator_noise_is_the_reference_draw(moss):
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    shape = (2, 128, 4, 6)
    want = randn_tensor(
        shape,
        generator = torch.Generator("cpu").manual_seed(5),
        device = torch.device("cpu"),
        dtype = torch.float32,
    )
    got = moss.comfy_noise(shape, torch.Generator("cpu").manual_seed(5), "cpu")
    assert torch.equal(want, got)
    gens = [torch.Generator("cpu").manual_seed(s) for s in (1, 2)]
    want = randn_tensor(
        shape,
        generator = [torch.Generator("cpu").manual_seed(s) for s in (1, 2)],
        device = torch.device("cpu"),
        dtype = torch.float32,
    )
    assert torch.equal(want, moss.comfy_noise(shape, gens, "cpu"))


def test_accelerator_generator_draws_comfyui_cpu_noise_from_its_seed(moss):
    import torch

    class _FakeCudaGenerator:
        device = torch.device("cuda", 0)

        def initial_seed(self):
            return 42

    got = moss.comfy_noise((1, 128, 4, 4), _FakeCudaGenerator(), "cpu")
    want = torch.randn((1, 128, 4, 4), generator = torch.Generator("cpu").manual_seed(42))
    assert torch.equal(got, want)
    with pytest.raises(ValueError):
        moss.comfy_noise((3, 128, 4, 4), [_FakeCudaGenerator(), _FakeCudaGenerator()], "cpu")


@pytest.fixture(scope = "module")
def pipelines(moss, ref, transformers_pair):
    import torch
    from diffusers import AutoencoderKLFlux2

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels = [32, 32, 32, 32],
        decoder_block_out_channels = [16, 16, 16, 16],
        norm_num_groups = 8,
        layers_per_block = 1,
        latent_channels = 32,
    ).eval()
    reference, ours = transformers_pair
    ref_pipe = ref.MOSSImagePipeline(reference, ref.MOSSImageScheduler(), vae = vae)
    our_pipe = moss.MOSSImagePipeline(
        ours, moss.MOSSImageScheduler(sigma_schedule = "training_grid"), vae = vae
    )
    return ref_pipe, our_pipe


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"guidance_scale": 1.0},
        {"uncond_skip_layers": []},
        {"slg_layers": [3], "slg_scale": 2.0, "slg_start": 0.2, "slg_stop": 0.8},
        {"num_images_per_prompt": 2},
    ],
    ids = ["cfg_skip22", "no_cfg", "plain_cfg", "slg", "two_images"],
)
def test_pipeline_is_bitwise_the_reference(pipelines, kwargs):
    import torch

    ref_pipe, our_pipe = pipelines
    _, embeds, _ = _inputs(lengths = (9,), seed = 3)
    _, negative, _ = _inputs(lengths = (4,), seed = 4)
    outs = []
    for pipe in (ref_pipe, our_pipe):
        pipe.set_progress_bar_config(disable = True)
        outs.append(
            pipe(
                prompt_embeds = embeds,
                negative_prompt_embeds = negative,
                height = 64,
                width = 96,
                num_inference_steps = 3,
                generator = torch.Generator("cpu").manual_seed(0),
                output_type = "pt",
                **kwargs,
            ).images
        )
    assert torch.equal(outs[0], outs[1])


def test_text_cache_off_is_bitwise_the_cached_render(moss, pipelines, monkeypatch):
    import torch

    _, our_pipe = pipelines
    _, embeds, _ = _inputs(lengths = (9,), seed = 3)
    _, negative, _ = _inputs(lengths = (4,), seed = 4)

    def run():
        return our_pipe(
            prompt_embeds = embeds,
            negative_prompt_embeds = negative,
            height = 32,
            width = 32,
            num_inference_steps = 2,
            generator = torch.Generator("cpu").manual_seed(0),
            output_type = "latent",
        ).images

    cached = run()
    monkeypatch.setenv(moss.TEXT_CACHE_ENV, "0")
    assert torch.equal(cached, run())


def test_step_callback_and_interrupt(pipelines):
    import torch

    _, our_pipe = pipelines
    _, embeds, _ = _inputs(lengths = (9,), seed = 3)
    seen = []

    def on_step(pipe, i, t, kw):
        seen.append(i)
        assert set(kw) == {"latents"}
        if i == 1:
            pipe._interrupt = True
        return kw

    our_pipe(
        prompt_embeds = embeds,
        guidance_scale = 1.0,
        height = 32,
        width = 32,
        num_inference_steps = 4,
        output_type = "latent",
        callback_on_step_end = on_step,
    )
    assert seen == [0, 1]


def test_skip_layers_reach_the_transformer_as_a_tuple(pipelines, monkeypatch):
    """CUDA graphs key plain values; a frozenset kwarg would make every uncond call uncapturable."""
    import torch

    _, our_pipe = pipelines
    _, embeds, _ = _inputs(lengths = (9,), seed = 3)
    calls = []
    forward = our_pipe.transformer.forward

    def spy(*args, **kwargs):
        calls.append(kwargs.get("skip_layers"))
        assert kwargs.get("return_dict") is False
        return forward(*args, **kwargs)

    monkeypatch.setattr(our_pipe.transformer, "forward", spy)
    _, negative, _ = _inputs(lengths = (4,), seed = 4)
    our_pipe(
        prompt_embeds = embeds,
        negative_prompt_embeds = negative,
        height = 32,
        width = 32,
        num_inference_steps = 1,
        output_type = "latent",
    )
    assert calls == [None, (22,)]


def test_family_detects_both_repos_and_a_local_pipeline(tmp_path):
    from core.inference import diffusion_families as df

    for repo in ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base"):
        fam = df.detect_family(repo)
        assert fam is not None and fam.name == "moss-image", repo
    local = tmp_path / "f972bd4b4b9e"
    local.mkdir()
    (local / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline"}))
    fam = df.detect_family_for_pick(str(local))
    assert fam is not None and fam.name == "moss-image"


def test_family_row_defaults_and_trust():
    from core.inference import diffusion_families as df
    from core.inference.family_profile import trusted_repo_ids

    fam = df.detect_family("OpenMOSS-Team/MOSS-Image-SFT")
    assert fam.native_transformer_files_only and not fam.pipeline_only
    assert fam.fp16_incompatible and fam.uses_negative_prompt
    assert (
        fam.pipeline_class == "MOSSImagePipeline"
        and fam.transformer_class == "MOSSImageTransformer"
    )
    for repo in ("OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base", "moss-image"):
        assert df.default_generation_params(repo) == (50, 3.0), repo
    trusted = trusted_repo_ids(df._FAMILIES)
    assert {"openmoss-team/moss-image-sft", "openmoss-team/moss-image-base"} <= trusted


def test_in_tree_pipeline_refuses_old_transformers(monkeypatch):
    from core.inference import diffusion_families as df

    fam = df.detect_family("OpenMOSS-Team/MOSS-Image-SFT")
    monkeypatch.setattr(df, "_installed_transformers_version", lambda: "4.57.6")
    assert df.family_pipeline_available(fam) is False
    with pytest.raises(ValueError, match = r"transformers >= 5\.3"):
        df.assert_pipeline_class_available(fam.pipeline_class, fam.name)
    monkeypatch.setattr(df, "_installed_transformers_version", lambda: "5.5.0")
    monkeypatch.setattr(df, "_diffusers_package_has_file", lambda rel: True)
    assert df.family_pipeline_available(fam) is True
    df.assert_pipeline_class_available(fam.pipeline_class, fam.name)


def test_prompt_length_policy_arms_the_moss_blocks():
    from core.inference import diffusion_dynamic_text as dt
    assert (
        dt.PROMPT_LENGTH_POLICY["MOSSImageTransformer"]
        == dt._FAMILY_SOURCES["MOSSImageTransformer"]
    )
    assert "L['x']" in dt._FAMILY_SOURCES["MOSSImageTransformer"]


def test_in_tree_classes_resolve(moss):
    from core.inference.diffusion import _family_model_class

    import diffusers

    assert _family_model_class(diffusers, "MOSSImagePipeline") is moss.MOSSImagePipeline
    assert _family_model_class(diffusers, "MOSSImageTransformer") is moss.MOSSImageTransformer
    assert moss.MOSSImageTransformer._repeated_blocks == ["_Block", "_TextBlock"]
    assert moss.MOSS_IMAGE_FAMILY_NAME == "moss-image"


def test_drop_vision_tower(moss):
    import torch

    class _Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(2, 2)
            self.language_model = torch.nn.Linear(2, 2)

    enc = _Encoder()
    assert moss.drop_vision_tower(enc) is True
    assert enc.visual is None and "visual" not in dict(enc.named_children())
    assert moss.drop_vision_tower(enc) is False


def test_loader_never_reads_repo_code(moss):
    """The assembly names its components itself; no trust_remote_code / custom_pipeline anywhere in the module."""
    source = Path(moss.__file__).read_text(encoding = "utf-8")
    assert "trust_remote_code" not in source.replace("no ``trust_remote_code``", "")
    assert "custom_pipeline =" not in source and "custom_pipeline=" not in source


def _callback_block_source() -> str:
    """The pipeline's step-end callback block, wrapped as a function of the names it reads."""
    import ast
    import textwrap

    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "diffusion_moss_image.py").read_text(encoding = "utf-8")
    )
    pipe = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MOSSImagePipeline"
    )
    call = next(n for n in pipe.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    block = next(
        n
        for n in ast.walk(call)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "callback_on_step_end is not None"
    )
    return (
        "def run(self, step_index, timestep, latents, callback_on_step_end, callback_on_step_end_tensor_inputs):\n"
        + textwrap.indent(ast.unparse(block), "    ")
        + "\n    return latents\n"
        + "seen = []\n"
        + "out = run(None, 0, 1, 'L', lambda p, i, t, kw: seen.append(dict(kw)), ['latents'])\n"
        + "assert seen == [{'latents': 'L'}] and out == 'L', (seen, out)\n"
    )


def _old_pythons():
    import shutil

    found = []
    for name in ("python3.9", "python3.10", "python3.11"):
        path = os.environ.get("MOSS_OLD_PYTHON") if name == "python3.11" else None
        path = path or shutil.which(name)
        if path:
            found.append(path)
    return found


def test_step_callback_reads_tensors_from_the_call_scope():
    """Before Python 3.12 a comprehension's ``locals()`` is its own scope, so ``locals()["latents"]`` raised KeyError."""
    import subprocess

    import ast

    tree = ast.parse(
        (_BACKEND / "core" / "inference" / "diffusion_moss_image.py").read_text(encoding = "utf-8")
    )
    comps = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    for comp in (n for n in ast.walk(tree) if isinstance(n, comps)):
        calls = [
            c for c in ast.walk(comp) if isinstance(c, ast.Call) and ast.unparse(c.func) == "locals"
        ]
        assert not calls, f"locals() inside a comprehension at line {comp.lineno}"
    src = _callback_block_source()
    exec(compile(src, "<moss-callback>", "exec"), {})
    for exe in _old_pythons():
        proc = subprocess.run([exe, "-I", "-c", src], capture_output = True, text = True, timeout = 60)
        assert proc.returncode == 0, f"{exe}: {proc.stderr}"


def test_step_callback_latents_are_bchw(pipelines):
    _, our_pipe = pipelines
    _, embeds, _ = _inputs(lengths = (9,), seed = 3)
    shapes = []

    def on_step(pipe, i, t, kw):
        shapes.append(tuple(kw["latents"].shape))
        return kw

    our_pipe(
        prompt_embeds = embeds,
        guidance_scale = 1.0,
        height = 32,
        width = 48,
        num_inference_steps = 2,
        output_type = "latent",
        callback_on_step_end = on_step,
    )
    assert shapes == [(1, 128, 2, 3)] * 2
