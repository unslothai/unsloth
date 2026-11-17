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

# OpenMOSS-Team/MOSS-Image-SFT safetensors headers: Qwen3.5-9B encoder (vision tower included, 248320 x 4096 token
# embedding) and the 5B DiT, bf16.
MOSS_SFT = {
    "text_encoder": sh.StoredComponent(
        "text_encoder", 16013, "bfloat16", 0, "transformers", embed_mib = 1940
    ),
    "transformer": sh.StoredComponent("transformer", 9706, "bfloat16", 0, "diffusers"),
    "vae": sh.StoredComponent("vae", 239, "bfloat16", 0, "diffusers"),
}
T4_HM = (51 * 1024, 49855)
T4 = (12976, 10964)
ROOMY_DISK = 200 * 1024


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV, sh.SPILL_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(total, available, **kw):
    return sh.decide_small_host(
        MOSS_SFT,
        torch.float32,
        device = "cuda",
        host_total_mib = total,
        host_available_mib = available,
        **kw,
    )


def test_colab_t4_high_ram_takes_the_route_in_ram():
    # The float32 load was OOM-killed at 49.3 GB anon RSS while the DiT loaded; the route measured 23 GB peak.
    d = _decide(*T4_HM, spill_free_mib = ROOMY_DISK)
    assert d.engaged and d.refuse is None and not d.int8_spill
    assert d.dense_host_mib > T4_HM[1]
    assert d.storage_dtypes == {"text_encoder": "bfloat16", "transformer": "bfloat16"}


def test_route_counts_the_float32_token_embedding():
    """``prepare_streamed_encoder_`` converts the embedding on the host: 3880 MiB in float32, measured as 3.79 GiB of
    float32 encoder tensors on the route."""
    d = _decide(*T4_HM)
    assert d.route_host_mib == 9706 // 2 + 1940 * 2


def test_colab_t4_spills_the_int8_dit_or_refuses_cleanly():
    # In RAM the route left 0.6 GB free (10.7 GB anon of 12.7 GB) and the VM stalled; it must spill or refuse.
    spilled = _decide(*T4, spill_free_mib = ROOMY_DISK)
    assert spilled.engaged and spilled.int8_spill and spilled.refuse is None
    assert spilled.spill_route_host_mib == 1940 * 2 + sh._SPILL_TRANSIENT_MIB
    no_disk = _decide(*T4)
    assert no_disk.engaged and not no_disk.int8_spill
    assert no_disk.refuse and "Not enough system RAM" in no_disk.refuse


def test_same_width_embedding_keeps_the_fp16_route_size():
    """float16 compute copies the embedding at its stored width: the reserve covers it, as before."""
    d = sh.decide_small_host(
        MOSS_SFT,
        torch.float16,
        device = "cuda",
        host_total_mib = T4[0],
        host_available_mib = T4[1],
        spill_free_mib = ROOMY_DISK,
    )
    assert d.route_host_mib == 9706 // 2
    assert not d.int8_spill and d.refuse is None


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


def test_dense_route_layerwise_cast_runs_the_moss_dit():
    """With ``transformer_quant`` off, the route keeps the DiT dense: bf16 Linears widened to float32 by a layerwise
    casting hook. Inputs the DiT casts itself (the time embedding) must use that compute dtype, not the stored bf16."""
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
    assert sh.prepare_streamed_encoder_(model, torch.float32) > 0
    lin = model.time_embedding[0]
    assert lin.weight.dtype == torch.bfloat16 and moss._compute_dtype(lin) == torch.float32
    with torch.no_grad():
        got = model(latents, t, encoder_hidden_states = text).sample
    assert got.dtype == torch.float32 and torch.isfinite(got).all()
    assert float((got - want).norm() / want.norm()) < 1e-5
