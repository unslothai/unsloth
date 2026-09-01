"""Regression tests for the gfx10 bf16 gate. See issue 7922."""

from pathlib import Path

import pytest

from unsloth.device_type import arch_lacks_bf16


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"


@pytest.mark.parametrize(
    "arch",
    ["gfx1010", "gfx1012", "gfx1030", "gfx1031", "gfx1032:sramecc-:xnack-", "GFX1036", " gfx1030 "],
)
def test_gfx10_lacks_bf16(arch):
    assert arch_lacks_bf16(arch) is True


@pytest.mark.parametrize(
    "arch",
    ["gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx90a", "gfx942", "gfx908"],
)
def test_newer_rdna_and_cdna_keep_bf16(arch):
    assert arch_lacks_bf16(arch) is False


@pytest.mark.parametrize("arch", ["", None, "unknown"])
def test_unreadable_arch_does_not_disable_bf16(arch):
    """A failed probe must not disable bf16 on a card that has it."""
    assert arch_lacks_bf16(arch) is False


def test_one_unreadable_device_keeps_the_others(monkeypatch):
    """One wedged device must not discard the gfx10 reading beside it, or that card
    keeps bf16 and the #7922 crash comes back. Only an unreadable device COUNT
    empties the list, and torch's own answer then stands."""
    import types

    import unsloth.device_type as dt

    if not hasattr(dt, "torch"):
        pytest.skip("device_type stub or MLX host; the real HIP probe is not loaded")

    class _Props:
        gcnArchName = "gfx1032"

    def _props(i):
        if i == 1:
            raise RuntimeError("device wedged")
        return _Props()

    monkeypatch.setattr(
        dt,
        "torch",
        types.SimpleNamespace(
            cuda = types.SimpleNamespace(device_count = lambda: 2, get_device_properties = _props)
        ),
    )
    assert dt.hip_visible_archs() == ["gfx1032"]

    def _count_raises():
        raise RuntimeError("no HIP runtime")

    monkeypatch.setattr(
        dt,
        "torch",
        types.SimpleNamespace(cuda = types.SimpleNamespace(device_count = _count_raises)),
    )
    assert dt.hip_visible_archs() == []


def test_gpu_init_gates_on_every_visible_device():
    """Guards against narrowing the gate back to device 0."""
    source = GPU_INIT.read_text(encoding = "utf-8")
    hip_branch = source.split('elif DEVICE_TYPE == "hip":', 1)[1].split("\nelif ", 1)[0]
    assert "arch_lacks_bf16" in hip_branch
    assert "hip_visible_archs()" in hip_branch
    assert "get_device_properties(0)" not in hip_branch


def test_model_utils_uses_the_patched_hip_probe():
    source = MODEL_UTILS.read_text(encoding = "utf-8")
    hip_branch = source.split('elif DEVICE_TYPE == "hip":', 1)[1].split("\nelif ", 1)[0]
    assert "SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()" in hip_branch
    assert "SUPPORTS_BFLOAT16 = True" not in hip_branch
