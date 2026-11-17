# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MOSS-Image on an fp16-only card (T4): the float32 promotion must not widen the 9B encoder and the DiT in host RAM
before anything quantises them. The in-tree loader takes the small-host route like a diffusers pipeline: routed
components load at their stored bf16 and the DiT runs on the route's int8 weights. CPU only."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
if not hasattr(diffusers, "AutoencoderKLFlux2"):
    pytest.skip("diffusers without AutoencoderKLFlux2", allow_module_level = True)

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.diffusion_moss_image as moss  # noqa: E402
import core.inference.diffusion_small_host as sh  # noqa: E402

# OpenMOSS-Team/MOSS-Image-SFT safetensors headers: Qwen3.5-9B encoder (vision tower included) and the 5B DiT, bf16.
MOSS_SFT = {
    "text_encoder": sh.StoredComponent("text_encoder", 16013, "bfloat16"),
    "transformer": sh.StoredComponent("transformer", 9706, "bfloat16"),
    "vae": sh.StoredComponent("vae", 239, "bfloat16"),
}


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV, sh.SPILL_ENV):
        monkeypatch.delenv(name, raising = False)


@pytest.mark.parametrize(
    "total, available",
    [
        # Colab T4 High-RAM: the float32 load was OOM-killed at 49.3 GB anon RSS while the DiT loaded.
        (51 * 1024, 49855),
        # Colab T4: OOM-killed at 11.4 GB while the encoder loaded.
        (12976, 10964),
    ],
)
def test_colab_t4_hosts_take_the_route_and_fit_it(total, available):
    d = sh.decide_small_host(
        MOSS_SFT,
        torch.float32,
        device = "cuda",
        host_total_mib = total,
        host_available_mib = available,
    )
    assert d.engaged and d.refuse is None
    assert d.dense_host_mib > available  # what the float32 load needs
    assert d.storage_dtypes == {"text_encoder": "bfloat16", "transformer": "bfloat16"}
    assert d.route_host_mib + sh.host_reserve_mib(total) <= available


def _moss_branch() -> ast.If:
    """The ``elif fam.name == MOSS_IMAGE_FAMILY_NAME`` branch of the pipeline load that assembles the pipeline."""
    tree = ast.parse((_BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "fam.name == MOSS_IMAGE_FAMILY_NAME"
            and any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "load_moss_image_pipeline"
                for n in ast.walk(node)
            )
            and any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "self._stop_load_prefetch"
                for n in ast.walk(node)
            )
        ):
            return node
    raise AssertionError("MOSS-Image pipeline branch not found")


def test_pipeline_branch_decides_the_route_and_loads_stored_dtypes():
    branch = _moss_branch()
    body = ast.Module(body = branch.body, type_ignores = [])
    calls = {ast.unparse(n.func): n for n in ast.walk(body) if isinstance(n, ast.Call)}
    assert "self._small_host_decision" in calls, "MOSS skips the small-host decision"
    decide = calls["self._small_host_decision"]
    assert {k.arg for k in decide.keywords} >= {"lora_active", "dense_denoiser", "fetch"}
    load = calls["load_moss_image_pipeline"]
    assert "small_host" in ast.unparse(
        next(k.value for k in load.keywords if k.arg == "storage_dtypes")
    )
    # Decided before the load: the route needs the components at their stored dtype.
    assert decide.lineno < load.lineno
    # The assignment targets the name the shared ``_apply_small_host_route`` call after the branch reads.
    assigns = [
        n
        for n in ast.walk(body)
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and n.value is decide
    ]
    assert [ast.unparse(t) for a in assigns for t in a.targets] == ["small_host"]


def test_loader_reads_routed_components_at_their_stored_dtype(monkeypatch):
    seen = {}

    class _Module(torch.nn.Module):
        def eval(self):
            return self

    def _te(source, dtype, **common):
        seen["text_encoder"] = dtype
        return _Module()

    def _dit(
        repo,
        subfolder = None,
        torch_dtype = None,
        **common,
    ):
        seen[subfolder] = torch_dtype
        return _Module()

    def _vae(
        repo,
        subfolder = None,
        torch_dtype = None,
        **common,
    ):
        seen[subfolder] = torch_dtype
        return _Module()

    monkeypatch.setattr(moss, "assert_transformers_supported", lambda: None)
    monkeypatch.setattr(
        moss,
        "_tokenizer_class",
        lambda repo, common: types.SimpleNamespace(from_pretrained = lambda *a, **k: object()),
    )
    monkeypatch.setattr(moss, "load_moss_image_text_encoder", _te)
    monkeypatch.setattr(moss.MOSSImageTransformer, "from_pretrained", _dit)
    monkeypatch.setattr(diffusers.AutoencoderKLFlux2, "from_pretrained", _vae)
    monkeypatch.setattr(moss.MOSSImageScheduler, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(moss, "MOSSImagePipeline", lambda **kw: types.SimpleNamespace(**kw))

    moss.load_moss_image_pipeline(
        "repo",
        torch.float32,
        storage_dtypes = {"text_encoder": "bfloat16", "transformer": "bfloat16"},
    )
    assert seen == {
        "text_encoder": torch.bfloat16,
        "transformer": torch.bfloat16,
        "vae": torch.float32,
    }
    seen.clear()
    moss.load_moss_image_pipeline("repo", torch.float32)
    assert seen == {
        "text_encoder": torch.float32,
        "transformer": torch.float32,
        "vae": torch.float32,
    }


def test_route_int8_weights_run_the_moss_dit(monkeypatch):
    """The route swaps each large Linear for ``Int8WeightLinear`` (no ``weight``); the DiT reads its compute dtype
    from ``scale`` and the float32 output stays within int8 rounding of the dense float32 DiT."""
    monkeypatch.setattr(sh, "_INT8_MIN_ELEMENTS", 1)
    torch.manual_seed(0)
    tiny = dict(dim = 256, ffn_dim = 512, num_heads = 2, num_layers = 4, text_dim = 64)
    model = moss.MOSSImageTransformer(**tiny).to(torch.bfloat16).eval()
    dense = moss.MOSSImageTransformer(**tiny).eval()
    dense.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    latents = torch.randn(1, 128, 4, 4)
    text = torch.randn(1, 7, 64)
    t = torch.tensor([500.0])
    with torch.no_grad():
        want = dense(latents, t, encoder_hidden_states = text).sample
    stats = sh.quantize_int8_weight_(model, compute_dtype = torch.float32, work_device = "cpu")
    assert stats["linears"] > 0 and stats["dense_linears"] == 0
    assert isinstance(model.layers[0].proj["q"], sh.int8_linear_class())
    with torch.no_grad():
        got = model(latents, t, encoder_hidden_states = text).sample
    assert got.dtype == torch.float32 and torch.isfinite(got).all()
    rel = float((got - want).norm() / want.norm())
    assert rel < 0.05, rel
