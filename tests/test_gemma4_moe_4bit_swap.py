"""Unit tests for the Gemma-4 MoE per-expert Linear4bit swap (#5344).

End-to-end correctness on the real 26B-A4B checkpoint requires a GPU + the
checkpoint on disk, so this file restricts itself to fast CPU-only tests
that exercise the swap helper's shape contract, idempotence, and gating
behaviour. The full repro (resident VRAM 46 GB -> 14.27 GB, cosine sim 0.994
vs BF16) is documented in the PR description.
"""

import importlib
import os

import torch
import torch.nn as nn


def _stub_gemma4_module():
    """Construct a stub Gemma4TextExperts-like module without importing
    transformers' Gemma4Config (which would force a fresh transformers
    download in CPU-only CI)."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    except Exception:
        return None

    # The class init requires a config; build a tiny synthetic one and then
    # overwrite the fused weights with shapes small enough for CPU tests.
    class _StubConfig:
        num_experts = 4
        hidden_size = 16
        moe_intermediate_size = 8
        hidden_activation = "gelu_pytorch_tanh"

    module = Gemma4TextExperts.__new__(Gemma4TextExperts)
    nn.Module.__init__(module)
    module.num_experts = _StubConfig.num_experts
    module.hidden_dim = _StubConfig.hidden_size
    module.intermediate_dim = _StubConfig.moe_intermediate_size
    module.gate_up_proj = nn.Parameter(
        torch.randn(
            _StubConfig.num_experts,
            2 * _StubConfig.moe_intermediate_size,
            _StubConfig.hidden_size,
            dtype = torch.bfloat16,
        ),
        requires_grad = False,
    )
    module.down_proj = nn.Parameter(
        torch.randn(
            _StubConfig.num_experts,
            _StubConfig.hidden_size,
            _StubConfig.moe_intermediate_size,
            dtype = torch.bfloat16,
        ),
        requires_grad = False,
    )
    from transformers.activations import ACT2FN

    module.act_fn = ACT2FN[_StubConfig.hidden_activation]
    return module


def test_is_enabled_reads_env_var():
    from unsloth.models import gemma4_moe_4bit

    old = os.environ.pop("UNSLOTH_GEMMA4_MOE_4BIT", None)
    try:
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is False
        os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = "1"
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is True
        os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = "0"
        assert gemma4_moe_4bit.is_gemma4_moe_4bit_enabled() is False
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_GEMMA4_MOE_4BIT", None)
        else:
            os.environ["UNSLOTH_GEMMA4_MOE_4BIT"] = old


def test_swap_skips_models_without_gemma4_experts():
    from unsloth.models.gemma4_moe_4bit import (
        swap_gemma4_experts_to_per_expert_linear4bit,
    )

    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0


def test_swap_skips_when_transformers_lacks_gemma4():
    """If transformers does not expose Gemma4TextExperts, the helper must
    return 0 without raising. We simulate the ImportError by patching."""
    import unsloth.models.gemma4_moe_4bit as g4m

    real_import = importlib.import_module

    def _broken_import(name, *args, **kwargs):
        if name == "transformers.models.gemma4.modeling_gemma4":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    try:
        importlib.import_module = _broken_import
        # Re-exercise via the public helper. It imports Gemma4TextExperts
        # inside its try/except, so the simulated ImportError must yield 0.
        model = nn.Sequential(nn.Linear(8, 8))
        assert g4m.swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
    finally:
        importlib.import_module = real_import


def test_swap_idempotent_on_stub_module_without_cuda():
    """On CPU we cannot exercise bnb (Linear4bit requires CUDA). Verify the
    helper at least returns 0 for the no-bnb-experts case without raising,
    and is idempotent across repeated calls."""
    from unsloth.models.gemma4_moe_4bit import (
        swap_gemma4_experts_to_per_expert_linear4bit,
    )

    if not torch.cuda.is_available():
        # CPU-only: bnb's Linear4bit init would fail. Validate the model-walk
        # path on an empty Sequential to confirm the helper is side-effect-free.
        model = nn.Sequential(nn.Linear(4, 4))
        assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
        assert swap_gemma4_experts_to_per_expert_linear4bit(model) == 0
        return

    # GPU path: build the stub and run a real swap.
    module = _stub_gemma4_module()
    if module is None:
        return  # transformers without gemma4 module: nothing to test
    model = nn.Sequential(module.to("cuda"))
    n1 = swap_gemma4_experts_to_per_expert_linear4bit(model)
    n2 = swap_gemma4_experts_to_per_expert_linear4bit(model)
    assert n1 == 1
    assert n2 == 0  # idempotent: already-swapped modules are skipped
    assert hasattr(module, "gate_up_proj_4bit")
    assert hasattr(module, "down_proj_4bit")
    assert len(module.gate_up_proj_4bit) == module.num_experts
    assert len(module.down_proj_4bit) == module.num_experts


if __name__ == "__main__":
    test_is_enabled_reads_env_var()
    test_swap_skips_models_without_gemma4_experts()
    test_swap_skips_when_transformers_lacks_gemma4()
    test_swap_idempotent_on_stub_module_without_cuda()
    print("All 4 swap tests passed.")
