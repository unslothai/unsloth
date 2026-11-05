# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stock MPS SDPA builds the full 16384-token score matrix, so the processors cap queries at 512."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_qwenimage21_math as bounded

LARGE = 2048
TOKENS = (LARGE // 16) ** 2


def _device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _reset_peak(device):
    """Bytes the allocator holds now; MPS has no peak counter, so its cache (kept between calls) stands in."""
    _sync(device)
    if device == "mps":
        torch.mps.empty_cache()
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return None


def _peak(device):
    _sync(device)
    if device == "mps":
        return torch.mps.driver_allocated_memory()
    if device == "cuda":
        return torch.cuda.max_memory_allocated()
    return None


@pytest.fixture
def tiny_pipe():
    pipeline_cls = getattr(diffusers, "QwenImage21Pipeline", None)
    if pipeline_cls is None:
        pytest.skip("this diffusers has no QwenImage21Pipeline")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download("hf-internal-testing/tiny-qwenimage21-pipe")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny pipeline not reachable: {exc}")
    device = _device()
    pipe = pipeline_cls.from_pretrained(path, torch_dtype = torch.float32).to(device)
    pipe.set_progress_bar_config(disable = True)
    return pipe, device


def _render(pipe, size):
    with torch.inference_mode():
        return pipe(
            prompt = "a lighthouse on a cliff at sunset",
            height = size,
            width = size,
            num_inference_steps = 2,
            generator = torch.Generator("cpu").manual_seed(0),
            output_type = "pt",
        ).images


@pytest.mark.allow_network
def test_large_render_is_bounded_finite_and_matches_stock(tiny_pipe, monkeypatch):
    pipe, device = tiny_pipe
    # Below any real budget, so the large render splits at the 512-row floor on every device.
    monkeypatch.setenv(bounded.SCORE_BUDGET_ENV, "0")
    reference = _render(pipe, 256).cpu()
    # The target Studio resolves on Apple Silicon; install never touches the device itself.
    assert bounded.install(pipe, SimpleNamespace(device = "mps", backend = "mps"))
    assert bounded.bounded_math_attention(pipe)
    torch.testing.assert_close(_render(pipe, 256).cpu(), reference, atol = 2e-5, rtol = 1e-4)

    rows = []
    original = bounded._bounded_attention

    def record(query, key, value, dispatch, **kwargs):
        def checked(q, k, v, **args):
            rows.append((q.shape[1], k.shape[1]))
            return dispatch(q, k, v, **args)

        return original(query, key, value, checked, **kwargs)

    monkeypatch.setattr(bounded, "_bounded_attention", record)
    # The denoise peak, read when the decode starts: the VAE at 2048x2048 is not what this measures.
    peaks = []
    decode = pipe.vae.decode

    def measured_decode(*args, **kwargs):
        peaks.append(_peak(device))
        return decode(*args, **kwargs)

    monkeypatch.setattr(pipe.vae, "decode", measured_decode)
    base = _reset_peak(device)
    image = _render(pipe, LARGE)
    assert image.shape[-2:] == (LARGE, LARGE)
    assert bool(torch.isfinite(image).all())
    assert float(image.std()) > 0
    assert max(q for q, _ in rows) == bounded.QUERY_CHUNK_SIZE
    assert max(k for _, k in rows) >= TOKENS
    # One full fp32 score matrix for 2 heads is 2 * 16384^2 * 4 bytes = 2 GiB; every bounded call stays near 64 MiB.
    full_scores = 2 * TOKENS * TOKENS * 4
    if base is not None:
        grown = peaks[0] - base
        print(
            f"{device}: denoise peak {grown / 2**20:.0f} MiB over the loaded pipeline at {LARGE}x{LARGE}"
        )
        assert grown < full_scores / 4


@pytest.mark.skipif(_device() != "mps", reason = "measures torch's MPS SDPA")
def test_mps_sdpa_ignores_kernel_selection_and_materialises_scores():
    """Why Studio cannot read MPS's attention memory off the SDPA probe (a record of the torch it runs on)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, 2, 4096, 16, device = "mps")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        flash = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.mps.synchronize()
    base = _reset_peak("mps")
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    grown = _peak("mps") - base
    scores = 2 * 4096 * 4096 * 4
    print(
        f"torch {torch.__version__}: FLASH-only SDPA ran; {grown / 2**20:.0f} MiB for {scores / 2**20:.0f} MiB of scores"
    )
    assert flash.shape == q.shape
    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 13):  # 2.13 adds a fused MPS prefill kernel
        assert grown >= scores // 2
