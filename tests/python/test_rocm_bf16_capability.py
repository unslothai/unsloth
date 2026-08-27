"""gfx10 (RDNA 1/2) reports bf16 support it does not have. See issue 7922.

The gate lives in `unsloth.device_type.arch_lacks_bf16`, which is a pure string check, so
these run on any host: no GPU, no ROCm torch, no import of the HIP branch itself.
"""

from pathlib import Path

import pytest

from unsloth.device_type import arch_lacks_bf16


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"


# RDNA 1 is gfx101x, RDNA 2 is gfx103x. Neither has native bf16 arithmetic.
@pytest.mark.parametrize(
    "arch",
    ["gfx1010", "gfx1012", "gfx1030", "gfx1031", "gfx1032:sramecc-:xnack-", "GFX1036", " gfx1030 "],
)
def test_gfx10_lacks_bf16(arch):
    assert arch_lacks_bf16(arch) is True


# gfx11 is RDNA 3, gfx12 is RDNA 4, gfx9 is CDNA. All have it, and none may be caught by a
# prefix match that is one character too short.
@pytest.mark.parametrize(
    "arch",
    ["gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx90a", "gfx942", "gfx908"],
)
def test_newer_rdna_and_cdna_keep_bf16(arch):
    assert arch_lacks_bf16(arch) is False


@pytest.mark.parametrize("arch", ["", None, "unknown"])
def test_unreadable_arch_does_not_disable_bf16(arch):
    """A failed probe must not turn bf16 off on a card that has it. torch's answer stands."""
    assert arch_lacks_bf16(arch) is False


def test_gpu_init_gates_on_every_visible_device():
    """SUPPORTS_BFLOAT16 is process-wide, so one gfx10 in the set has to disable it for all."""
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
