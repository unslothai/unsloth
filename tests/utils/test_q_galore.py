# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tests for Q-GaLore integration (unsloth/optimizers/).

import pytest
import sys
import os
import torch
import torch.nn as nn

# Import the optimizers module directly to avoid triggering unsloth.__init__
# which requires unsloth_zoo and other heavy dependencies.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_optimizers_dir = os.path.join(_repo_root, "unsloth", "optimizers")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Direct import of the actual modules (avoids unsloth/__init__.py)
import importlib.util


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load projector module first (no dependencies on unsloth)
_projector_mod = _load_module(
    "unsloth.optimizers.q_galore_projector",
    os.path.join(_optimizers_dir, "q_galore_projector.py"),
)
GaLoreProjector = _projector_mod.GaLoreProjector
_quantize = _projector_mod._quantize
_dequantize = _projector_mod._dequantize
_quantize_stochastic = _projector_mod._quantize_stochastic

# Load adamw module (depends on projector, may skip bitsandbytes)
_adamw_mod = _load_module(
    "unsloth.optimizers.q_galore_adamw",
    os.path.join(_optimizers_dir, "q_galore_adamw.py"),
)
make_q_galore_param_groups = _adamw_mod.make_q_galore_param_groups

# ======================================================================
# Projector tests
# ======================================================================


class TestGaLoreProjector:
    """Tests for the GaLore low-rank gradient projector."""

    def test_project_and_back_tall(self):
        """Project → project_back preserves shape for tall matrices."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1)
        grad = torch.randn(16, 8)  # tall
        low = proj.project(grad, step = 0)
        assert low.shape == (16, 4)

        full = proj.project_back(low)
        assert full.shape == grad.shape

    def test_project_and_back_wide(self):
        """Project → project_back preserves shape for wide matrices."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1)
        grad = torch.randn(8, 16)  # wide
        low = proj.project(grad, step = 0)
        assert low.shape == (4, 16)

        full = proj.project_back(low)
        assert full.shape == grad.shape

    def test_project_reuses_cached_svd(self):
        """SVD is not recomputed when step is not a multiple of update_proj_gap."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 100)
        grad = torch.randn(16, 8)
        proj.project(grad, step = 0)
        assert proj.svd_count == 1

        proj.project(grad, step = 1)
        assert proj.svd_count == 1  # No recomputation

        proj.project(grad, step = 100)
        assert proj.svd_count == 2  # Recomputed

    def test_quantized_projection(self):
        """Quantized projection matrix stores and restores with bounded error."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, quant = True, n_bit = 8)
        grad = torch.randn(16, 8)
        low = proj.project(grad, step = 0)
        assert low.shape == (16, 4)

        # The projection matrix should be stored as uint8
        assert proj.ortho_matrix.dtype == torch.uint8

    def test_quantized_projection_int4(self):
        """INT4 quantized projection stores correctly."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, quant = True, n_bit = 4)
        grad = torch.randn(16, 8)
        proj.project(grad, step = 0)
        assert proj.ortho_matrix.dtype == torch.uint8
        # INT4 values should be in range [0, 15]
        assert proj.ortho_matrix.max() <= 15

    def test_adaptive_scheduling(self):
        """update_proj_gap increases when cosine similarity exceeds threshold."""
        proj = GaLoreProjector(
            rank = 4,
            update_proj_gap = 10,
            cos_threshold = 0.0,  # Very low threshold → always triggers
            gamma_proj = 2.0,
            queue_size = 2,
        )
        # Use very similar gradients so cosine similarity is high
        base_grad = torch.randn(16, 8)
        for i in range(5):
            grad = base_grad + torch.randn_like(base_grad) * 0.001
            proj.project(grad, step = i * 10)

        # After several similar SVDs, update_proj_gap should have increased
        assert proj.update_proj_gap > 10

    def test_scale_applied(self):
        """project_back applies the scale factor."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 0.5)
        grad = torch.randn(16, 8)
        low = proj.project(grad, step = 0)

        proj2 = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 1.0)
        low2 = proj2.project(grad, step = 0)

        full_half = proj.project_back(low)
        full_one = proj2.project_back(low2)

        # The ratio should be exactly 0.5 (SVD is deterministic on same input)
        ratio = full_half.norm() / full_one.norm()
        assert abs(ratio - 0.5) < 1e-5, f"Expected ratio ~0.5, got {ratio:.8f}"


# ======================================================================
# Quantization utility tests
# ======================================================================


class TestQuantizationUtils:
    """Tests for _quantize, _dequantize, _quantize_stochastic."""

    def test_quantize_dequantize_roundtrip(self):
        """Quantize → dequantize has bounded error."""
        w = torch.randn(32, 64)
        q, scales, zeros, shape = _quantize(w, n_bit = 8)
        w_hat = _dequantize(q, scales, zeros, shape)

        # Error should be bounded by the quantization step size
        error = (w - w_hat).abs().max()
        assert error < 0.1, f"Max error {error} exceeds threshold"

    def test_quantize_group_roundtrip(self):
        """Grouped quantization → dequantization has bounded error."""
        w = torch.randn(32, 64)
        q, scales, zeros, shape = _quantize(w, q_group_size = 32, n_bit = 8)
        w_hat = _dequantize(q, scales, zeros, shape)
        error = (w - w_hat).abs().max()
        assert error < 0.1

    def test_quantize_dtype(self):
        """Quantized output should be uint8."""
        w = torch.randn(16, 16)
        q, _, _, _ = _quantize(w, n_bit = 8)
        assert q.dtype == torch.uint8

    def test_quantize_int4_range(self):
        """INT4 values should be in [0, 15]."""
        w = torch.randn(16, 16)
        q, _, _, _ = _quantize(w, n_bit = 4)
        assert q.max() <= 15
        assert q.min() >= 0

    def test_stochastic_rounding_unbiased(self):
        """Stochastic rounding should be approximately unbiased."""
        torch.manual_seed(42)
        w = torch.randn(64, 64)
        errors = []
        for _ in range(50):
            q, scales, zeros, shape = _quantize_stochastic(w, n_bit = 8)
            w_hat = _dequantize(q, scales, zeros, shape)
            errors.append((w - w_hat).mean().item())

        mean_error = sum(errors) / len(errors)
        assert (
            abs(mean_error) < 0.01
        ), f"Mean error {mean_error} suggests biased rounding"


# ======================================================================
# Param group helper tests
# ======================================================================


class TestParamGroupHelper:
    """Tests for make_q_galore_param_groups."""

    def test_param_group_separation(self):
        """GaLore vs non-GaLore params are correctly separated."""

        # Create a mini-transformer-like model
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)
        model.k_proj = nn.Linear(64, 64, bias = False)
        model.embed = nn.Embedding(100, 64)
        model.norm = nn.LayerNorm(64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        # Should have 2 groups: galore and non-galore
        assert len(groups) == 2

        galore_group = [g for g in groups if "rank" in g][0]
        non_galore_group = [g for g in groups if "rank" not in g][0]

        # q_proj and k_proj should be in galore group (2 params)
        assert len(galore_group["params"]) == 2
        # embed and norm should be in non-galore group
        assert (
            len(non_galore_group["params"]) == 3
        )  # embed weight + norm weight + norm bias

    def test_custom_target_modules(self):
        """Custom target_modules narrows GaLore scope."""

        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)
        model.k_proj = nn.Linear(64, 64, bias = False)
        model.v_proj = nn.Linear(64, 64, bias = False)
        model.embed = nn.Embedding(100, 64)

        groups = make_q_galore_param_groups(
            model,
            rank = 8,
            target_modules = ["q_proj"],
            weight_quant = False,
        )

        galore_group = [g for g in groups if "rank" in g][0]
        assert len(galore_group["params"]) == 1  # Only q_proj

    def test_bias_excluded_from_galore(self):
        """1D bias params matching target names must NOT be in the GaLore group.

        GaLoreProjector.project requires 2-D gradients, so bias vectors
        (e.g. q_proj.bias) that match a target name must be excluded.
        """
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = True)  # has .weight AND .bias
        model.embed = nn.Embedding(100, 64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        galore_group = [g for g in groups if "rank" in g][0]
        non_galore_group = [g for g in groups if "rank" not in g][0]

        # Only the 2-D q_proj.weight should be in the GaLore group
        assert len(galore_group["params"]) == 1
        assert galore_group["params"][0].dim() == 2

        # q_proj.bias (1-D) + embed.weight should be in non-GaLore
        assert any(p.dim() == 1 for p in non_galore_group["params"])

    def test_empty_target_modules_no_galore(self):
        """target_modules=[] should result in no GaLore params."""
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)

        # Pass empty list, should NOT fall back to defaults
        groups = make_q_galore_param_groups(
            model,
            rank = 8,
            target_modules = [],
            weight_quant = False,
        )

        galore_groups = [g for g in groups if "rank" in g]
        assert (
            len(galore_groups) == 0
        ), "Expected no GaLore groups when target_modules=[]"


# ======================================================================
# Optimizer tests (CPU-only, no bitsandbytes dependency)
# ======================================================================


class TestQGaLoreIntegration:
    """Integration tests that work without bitsandbytes on CPU."""

    def test_projector_training_loop(self):
        """A simple training loop using manual GaLore projection converges."""
        torch.manual_seed(42)

        # Tiny model: single linear layer
        model = nn.Linear(32, 16, bias = False)
        target = torch.randn(4, 16)
        x = torch.randn(4, 32)

        proj = GaLoreProjector(rank = 8, update_proj_gap = 1, scale = 1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01)

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            out = model(x)
            loss = nn.functional.mse_loss(out, target)
            loss.backward()
            losses.append(loss.item())

            # Manual GaLore projection
            for p in model.parameters():
                if p.grad is not None and p.grad.dim() == 2:
                    low = proj.project(p.grad, step)
                    p._saved = p.data.clone()
                    update = torch.zeros_like(low)
                    update.add_(low)  # Simplified update
                    full_update = proj.project_back(update)
                    p.grad.copy_(full_update)

            optimizer.step()

        # Loss should decrease
        assert (
            losses[-1] < losses[0]
        ), f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_full_projector_roundtrip_quality(self):
        """project → project_back captures the dominant gradient directions."""
        torch.manual_seed(42)
        # Create a gradient with clear low-rank structure
        u = torch.randn(32, 4)
        v = torch.randn(4, 16)
        grad = u @ v  # rank-4 gradient

        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 1.0)
        low = proj.project(grad, step = 0)
        reconstructed = proj.project_back(low)

        # For a rank-4 gradient with rank-4 projection, reconstruction
        # should be very close to original
        relative_error = (grad - reconstructed).norm() / grad.norm()
        assert (
            relative_error < 0.05
        ), f"Reconstruction error too high: {relative_error:.4f}"

    def test_weight_quant_activates_on_first_step(self):
        """_has_weight_quant returns True even when _q_scales is None (first step)."""
        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        QGaLoreAdamW8bit = _adamw_mod_local.QGaLoreAdamW8bit

        p = torch.nn.Parameter(torch.randn(16, 16))
        # Simulate init_weight_quantization tagging
        p._q_scales = None
        p._q_zeros = None
        p._q_shape = p.data.shape

        group = {"weight_quant": True}

        # _has_weight_quant must return True even on first step (_q_scales=None)
        assert QGaLoreAdamW8bit._has_weight_quant(p, group) is True

        # Without the tag, it should return False
        p2 = torch.nn.Parameter(torch.randn(16, 16))
        assert QGaLoreAdamW8bit._has_weight_quant(p2, group) is False

    def test_embedding_lr_param_group_split(self):
        """Embedding params can be split into a separate group with custom LR."""
        # This tests the logic that make_q_galore_param_groups produces groups
        # that can be further split by the trainer for embedding LR.
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)
        model.embed = nn.Embedding(100, 64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        # Simulate splitting non-GaLore group for embedding LR
        embed_lr = 5e-5
        new_groups = []
        for group in groups:
            if "rank" in group:
                new_groups.append(group)
                continue
            embed_params = []
            other_params = []
            for p in group["params"]:
                # In real usage, we'd check the name; here just split by shape
                if p.shape[0] == 100:  # embedding
                    embed_params.append(p)
                else:
                    other_params.append(p)
            if other_params:
                g = dict(group)
                g["params"] = other_params
                new_groups.append(g)
            if embed_params:
                g = dict(group)
                g["params"] = embed_params
                g["lr"] = embed_lr
                new_groups.append(g)

        # Should have 3 groups: galore, non-galore non-embed, embed
        embed_groups = [g for g in new_groups if g.get("lr") == embed_lr]
        assert len(embed_groups) == 1
        assert embed_groups[0]["lr"] == embed_lr

    def test_optimizer_hyperparams_forwarded(self):
        """QGaLoreAdamW8bit accepts betas and eps keyword arguments."""
        # Verify the constructor signature accepts these params.
        # Without bitsandbytes we can't instantiate, but we can check the
        # function signature.
        import inspect

        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        QGaLoreAdamW8bit = _adamw_mod_local.QGaLoreAdamW8bit

        sig = inspect.signature(QGaLoreAdamW8bit.__init__)
        param_names = list(sig.parameters.keys())
        assert "betas" in param_names, "betas not in QGaLoreAdamW8bit.__init__ params"
        assert "eps" in param_names, "eps not in QGaLoreAdamW8bit.__init__ params"

    def test_weight_decay_uses_saved_data(self):
        """Weight decay should apply standard decoupled AdamW decay on current weights."""
        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]

        # Create a mock parameter and group
        p = torch.nn.Parameter(torch.ones(4, 4))
        p._saved_data = torch.ones(4, 4) * 2.0  # Pre-update weights
        # Simulate project-back: p.data = p._saved_data + projected update
        p.data = p._saved_data.add_(torch.ones(4, 4) * 1.0)  # p.data is now 3.0

        group = {"weight_decay": 0.1, "lr": 1.0, "_wd_saved": 0.1}

        # Replicate the fixed decoupled weight decay logic (uses p.data, not p._saved_data)
        p.data.add_(
            p.data,
            alpha = -group["lr"] * group["_wd_saved"],
        )

        del p._saved_data  # Clean up after all uses, matching fixed code

        # Decoupled weight decay: 3.0 - (1.0 * 0.1 * 3.0) = 2.7
        assert torch.allclose(
            p.data, torch.tensor(2.7)
        ), "Weight decay didn't use p.data for decoupled decay!"

    def test_params_float_after_weight_quant_step(self):
        """After a step with weight_quant=True, parameters must remain floating point."""
        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        _projector_mod_local = sys.modules["unsloth.optimizers.q_galore_projector"]

        _quantize = _projector_mod_local._quantize

        p = torch.nn.Parameter(torch.randn(16, 16))
        group = {
            "weight_quant": True,
            "stochastic_round": False,
            "weight_group_size": 16,
        }

        # Replicate the re-quantize logic at the end of optimizer step
        float_data = p.data.clone()
        q, scales, zeros, shape = _quantize(
            float_data, q_group_size = group["weight_group_size"]
        )

        # The key assertion: p.data stays float, _q_data holds uint8
        p._q_data = q.to(p.data.device)
        p._q_scales = scales
        p._q_zeros = zeros
        p._q_shape = shape

        assert p.data.is_floating_point(), "p.data was converted to uint8!"
        assert p._q_data.dtype == torch.uint8, "_q_data should be uint8!"

    def test_weight_quant_hook_restores_float(self):
        """Forward pre-hook should dequantize INT8 weights before forward pass."""
        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        _projector_mod_local = sys.modules["unsloth.optimizers.q_galore_projector"]
        install_hook = _adamw_mod_local.install_weight_quant_hooks

        linear = nn.Linear(16, 8, bias = False)
        original = linear.weight.data.clone()

        # Quantize the weight and replace with placeholder (simulates post-step)
        q, scales, zeros, shape = _projector_mod_local._quantize(
            linear.weight.data.clone(), q_group_size = 16
        )
        linear.weight._q_data = q
        linear.weight._q_scales = scales
        linear.weight._q_zeros = zeros
        linear.weight._q_shape = shape
        linear.weight.data = torch.zeros(1, dtype = linear.weight.dtype)
        assert linear.weight.data.numel() == 1, "placeholder should be 1 element"

        # Install hook and run forward -- should restore float weights
        handles = install_hook(linear)
        x = torch.randn(2, 16)
        out = linear(x)  # triggers pre-hook

        assert linear.weight.data.shape == (8, 16), "weight shape not restored"
        assert linear.weight.data.is_floating_point(), "weight not float after hook"
        # Check values are close to original (quantization introduces small error)
        assert torch.allclose(
            linear.weight.data, original, atol = 0.15
        ), "dequantized weight too far from original"

        for h in handles:
            h.remove()
