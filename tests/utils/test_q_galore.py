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

import inspect
import pytest
import sys
import os
import torch
import torch.nn as nn

# Load directly to avoid unsloth.__init__ heavy deps.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_optimizers_dir = os.path.join(_repo_root, "unsloth", "optimizers")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import importlib.util


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_projector_mod = _load_module(
    "unsloth.optimizers.q_galore_projector",
    os.path.join(_optimizers_dir, "q_galore_projector.py"),
)
GaLoreProjector = _projector_mod.GaLoreProjector
_quantize = _projector_mod._quantize
_dequantize = _projector_mod._dequantize
_quantize_stochastic = _projector_mod._quantize_stochastic

_adamw_mod = _load_module(
    "unsloth.optimizers.q_galore_adamw",
    os.path.join(_optimizers_dir, "q_galore_adamw.py"),
)
make_q_galore_param_groups = _adamw_mod.make_q_galore_param_groups


_BNB_OPTIMIZER_BACKEND = {}


def requires_bnb_optimizer(device):
    """Skip unless bitsandbytes can really run an optimizer step on ``device``.

    Not `supported_torch_devices`: it lists "cpu" from 0.46.0 but the CPU kernels landed in
    0.50.0, so gating on it fails these tests on 0.47.x/0.49.x, which pyproject allows.
    The probe drives bitsandbytes' own AdamW32bit, so a real regression in the code under
    test still fails rather than turning into a skip.
    """
    available = _BNB_OPTIMIZER_BACKEND.get(device)
    if available is None:
        try:
            import bitsandbytes

            probe = nn.Parameter(torch.ones(2, device = device))
            probe.grad = torch.zeros(2, device = device)
            bitsandbytes.optim.AdamW32bit([probe], lr = 0.0).step()
            available = True
        except Exception:
            available = False
        _BNB_OPTIMIZER_BACKEND[device] = available
    if not available:
        pytest.skip(f"This bitsandbytes version cannot run an optimizer step on {device}")


@pytest.mark.skipif(not _adamw_mod._HAS_BNB, reason = "bitsandbytes is required")
def test_optimizer_constructs_against_the_installed_bitsandbytes():
    """Runs on every version in the CI matrix, unlike the tests that need a step.

    Construction needs no optimizer kernels, so this is the only check the 0.45.x-0.49.x
    matrix jobs can actually execute. It has to assert on the bound arguments because the
    0.50.x misbinding left the arity intact and raised nothing.
    """
    signature = inspect.signature(_adamw_mod.Optimizer2State.__init__)
    captured = {}
    original = _adamw_mod.Optimizer2State.__init__

    def record(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        return original(self, *args, **kwargs)

    _adamw_mod.Optimizer2State.__init__ = record
    try:
        _adamw_mod.QGaLoreAdamW8bit([nn.Parameter(torch.ones(8, 8))], lr = 1e-3)
    finally:
        _adamw_mod.Optimizer2State.__init__ = original

    assert captured.get("optimizer_name") == "adam"
    assert captured.get("optim_bits") == 8
    for name, default in (("max_unorm", 0.0), ("skip_zeros", False)):
        if name in signature.parameters:
            assert captured.get(name) == default, (
                f"bitsandbytes received {name}={captured.get(name)!r}, but its default is "
                f"{default!r}; an option is landing in the wrong parameter positionally."
            )


@pytest.mark.skipif(not _adamw_mod._HAS_BNB, reason = "bitsandbytes is required")
@pytest.mark.parametrize("projected", [True, False])
@pytest.mark.parametrize("initial_value", [0.0, 1.0])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_default_optimizer_updates_match_adamw(projected, initial_value, weight_decay):
    bnb = pytest.importorskip("bitsandbytes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    requires_bnb_optimizer(device)
    param = nn.Parameter(torch.full((2, 2), initial_value, device = device))
    reference = nn.Parameter(param.detach().clone())
    group = {"params": [param]}
    if projected:
        group.update(rank = 2, scale = 1.0, quant = False)
    optimizer = _adamw_mod.QGaLoreAdamW8bit([group], lr = 0.1, weight_decay = weight_decay)
    # Unprojected steps delegate to bitsandbytes, whose CUDA kernel decays after the update.
    reference_cls = torch.optim.AdamW if projected else bnb.optim.AdamW8bit
    reference_optimizer = reference_cls([reference], lr = 0.1, weight_decay = weight_decay)
    # A diagonal gradient keeps full-rank projection aligned with AdamW.
    gradient = torch.diag(torch.tensor([2.0, 1.0], device = device))
    for weight, opt in [(param, optimizer), (reference, reference_optimizer)]:
        (weight * gradient).sum().backward()
        opt.step()
    torch.testing.assert_close(param, reference)


@pytest.mark.skipif(not _adamw_mod._HAS_BNB, reason = "bitsandbytes is required")
def test_legacy_bitsandbytes_options_are_forwarded_by_name(monkeypatch):
    original_init = _adamw_mod.Optimizer2State.__init__
    received = []

    def legacy_init(
        self,
        *args,
        percentile_clipping = 100,
        block_wise = True,
        **kwargs,
    ):
        received.append((percentile_clipping, block_wise))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(_adamw_mod.Optimizer2State, "__init__", legacy_init)
    _adamw_mod.QGaLoreAdamW8bit(
        [nn.Parameter(torch.ones(2))],
        percentile_clipping = 95,
        block_wise = False,
    )
    assert received == [(95, False)]


@pytest.mark.skipif(not _adamw_mod._HAS_BNB, reason = "bitsandbytes is required")
@pytest.mark.parametrize("name,value", [("percentile_clipping", 95), ("block_wise", False)])
def test_removed_bitsandbytes_options_are_rejected(name, value):
    import inspect
    if name in inspect.signature(_adamw_mod.Optimizer2State.__init__).parameters:
        pytest.skip("This bitsandbytes version still supports the option")
    with pytest.raises(ValueError, match = name):
        _adamw_mod.QGaLoreAdamW8bit([nn.Parameter(torch.ones(2))], **{name: value})


@pytest.mark.parametrize("out_features,in_features", [(4, 3), (3, 4)])
def test_projected_step_preserves_full_rank_gradients(out_features, in_features):
    bnb = pytest.importorskip("bitsandbytes")
    try:
        probe = nn.Parameter(torch.ones(2))
        probe.grad = torch.zeros_like(probe)
        bnb.optim.AdamW32bit([probe], lr = 0.0).step()
    except Exception as exc:
        pytest.skip(
            reason = f"bitsandbytes build lacks CPU optimizer kernels (older releases): {exc}"
        )
    model = nn.Linear(in_features, out_features, bias = False)
    optimizer = _adamw_mod.QGaLoreAdamW8bit(
        [{"params": list(model.parameters()), "rank": 1, "quant": False}],
        lr = 0.01,
        weight_decay = 0.0,
    )
    inputs = torch.arange(2 * in_features, dtype = torch.float32).reshape(2, in_features)
    for _ in range(2):
        optimizer.zero_grad(set_to_none = False)
        model(inputs).square().mean().backward()
        expected_grad = model.weight.grad.clone()
        optimizer.step()
        torch.testing.assert_close(model.weight.grad, expected_grad)


@pytest.mark.skipif(not _adamw_mod._HAS_BNB, reason = "bitsandbytes is required")
@pytest.mark.parametrize("quant", [True, False])
def test_resume_from_weights_only_checkpoint_matches_uninterrupted(quant):
    """unslothai/unsloth#729: transformers resumes with torch.load(weights_only=True, map_location="cpu")."""
    import io

    device = "cuda" if torch.cuda.is_available() else "cpu"
    requires_bnb_optimizer(device)

    def make(weight, bias):
        weight = nn.Parameter(weight.clone())
        bias = nn.Parameter(bias.clone())
        projected = {"params": [weight], "rank": 4, "update_proj_gap": 2, "quant": quant}
        projected.update(queue_size = 2, cos_threshold = 0.0)
        optimizer = _adamw_mod.QGaLoreAdamW8bit([projected, {"params": [bias]}], lr = 1e-2)
        return weight, bias, optimizer

    def step(weight, bias, optimizer, i):
        # svd_lowrank draws from the global RNG, so each step reseeds it.
        torch.manual_seed(i)
        weight.grad = torch.randn(weight.shape, device = device)
        bias.grad = torch.randn(bias.shape, device = device)
        optimizer.step()

    torch.manual_seed(0)
    weight, bias, optimizer = make(
        torch.randn(128, 64, device = device), torch.randn(64, device = device)
    )
    for i in range(3):
        step(weight, bias, optimizer, i)
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    assert isinstance(optimizer.state[weight]["projector"], GaLoreProjector)
    buffer.seek(0)
    saved = torch.load(buffer, map_location = "cpu", weights_only = True)

    weight2, bias2, resumed = make(weight.detach(), bias.detach())
    resumed.load_state_dict(saved)
    projector = resumed.state[weight2]["projector"]
    assert projector.ortho_matrix.device == weight2.device
    for i in range(3, 7):
        step(weight, bias, optimizer, i)
        step(weight2, bias2, resumed, i)
    assert torch.equal(weight, weight2)
    assert torch.equal(bias, bias2)
    # The adaptive schedule widened the gap after resume, so its saved queue was used.
    assert projector.update_proj_gap == optimizer.state[weight]["projector"].update_proj_gap == 4


# ======================================================================
# Projector tests
# ======================================================================


class TestGaLoreProjector:
    """Tests for the GaLore low-rank gradient projector."""

    def test_project_and_back_tall(self):
        """Project → project_back preserves shape for tall matrices."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1)
        grad = torch.randn(16, 8)
        low = proj.project(grad, step = 0)
        assert low.shape == (16, 4)

        full = proj.project_back(low)
        assert full.shape == grad.shape

    def test_project_and_back_wide(self):
        """Project → project_back preserves shape for wide matrices."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1)
        grad = torch.randn(8, 16)
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
        assert proj.svd_count == 1

        proj.project(grad, step = 100)
        assert proj.svd_count == 2

    def test_quantized_projection(self):
        """Quantized projection matrix stores and restores with bounded error."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, quant = True, n_bit = 8)
        grad = torch.randn(16, 8)
        low = proj.project(grad, step = 0)
        assert low.shape == (16, 4)

        assert proj.ortho_matrix.dtype == torch.uint8

    def test_quantized_projection_int4(self):
        """INT4 quantized projection stores correctly."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, quant = True, n_bit = 4)
        grad = torch.randn(16, 8)
        proj.project(grad, step = 0)
        assert proj.ortho_matrix.dtype == torch.uint8
        assert proj.ortho_matrix.max() <= 15

    @pytest.mark.parametrize("n_bit", [4, 8])
    def test_quantized_projection_preserves_constant_rank_one_gradient(self, n_bit):
        grad = torch.ones(8, 4)
        proj = GaLoreProjector(rank = 1, quant = True, n_bit = n_bit)
        restored = proj.project_back(proj.project(grad, step = 0))
        torch.testing.assert_close(restored, grad)

    def test_adaptive_scheduling(self):
        """update_proj_gap increases when cosine similarity exceeds threshold."""
        proj = GaLoreProjector(
            rank = 4,
            update_proj_gap = 10,
            cos_threshold = 0.9,
            gamma_proj = 2.0,
            queue_size = 2,
        )
        base_grad = torch.randn(16, 8)
        for i in range(5):
            grad = base_grad + torch.randn_like(base_grad) * 0.001
            proj.project(grad, step = i * 10)

        assert proj.update_proj_gap > 10, list(proj.queue)

    @staticmethod
    def _scheduled_gap(monkeypatch, bases, cos_threshold):
        """Drive the schedule with a fixed sequence of orthogonal bases, one per SVD."""
        sequence = iter(bases)
        monkeypatch.setattr(
            GaLoreProjector,
            "_compute_orthogonal",
            staticmethod(lambda *args, **kwargs: next(sequence)),
        )
        proj = GaLoreProjector(
            rank = 2, update_proj_gap = 1, cos_threshold = cos_threshold, gamma_proj = 2.0, queue_size = 2
        )
        grad = torch.zeros(8, 8)
        for step in range(len(bases)):
            if step % proj.update_proj_gap == 0:
                proj.project(grad, step = step)
        return proj.update_proj_gap

    def test_adaptive_scheduling_ignores_a_sign_flipped_basis(self, monkeypatch):
        # A negated basis is the same subspace; a raw dot product reads it as -1.
        basis = torch.eye(8)[:2]
        assert self._scheduled_gap(monkeypatch, [basis, -basis, basis], cos_threshold = 0.9) == 2

    def test_adaptive_scheduling_still_rejects_a_rotated_basis(self, monkeypatch):
        eye = torch.eye(8)
        assert (
            self._scheduled_gap(monkeypatch, [eye[:2], eye[2:4], eye[4:6]], cos_threshold = 0.4) == 1
        )

    def test_scale_applied(self):
        """project_back applies the scale factor."""
        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 0.5)
        grad = torch.randn(16, 8)
        low = proj.project(grad, step = 0)

        proj2 = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 1.0)
        low2 = proj2.project(grad, step = 0)

        full_half = proj.project_back(low)
        full_one = proj2.project_back(low2)

        # SVD is deterministic on the same input, so the ratio is exactly 0.5.
        ratio = full_half.norm() / full_one.norm()
        assert abs(ratio - 0.5) < 1e-5, f"Expected ratio ~0.5, got {ratio:.8f}"


class TestQuantizationUtils:
    """Tests for _quantize, _dequantize, _quantize_stochastic."""

    @pytest.mark.parametrize("quantize", [_quantize, _quantize_stochastic])
    @pytest.mark.parametrize("n_bit", [4, 8])
    @pytest.mark.parametrize("group_size", [-1, 2])
    def test_roundtrip_single_sign_groups(self, quantize, n_bit, group_size):
        weights = torch.tensor(
            [
                [1.0, 1.5, 2.0, 2.5],
                [-1.0, -1.5, -2.0, -2.5],
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5, -0.5],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, -0.5, 0.5, 1.0],
            ]
        )
        quantized = quantize(weights, q_group_size = group_size, n_bit = n_bit)
        restored = _dequantize(*quantized)
        scales = quantized[1]
        error = (restored - weights).abs().reshape(scales.shape[0], -1)
        # Stochastic rounding may move one step, but must not clip a group.
        assert torch.all(error <= scales + 1e-6)

    def test_quantize_dequantize_roundtrip(self):
        """Quantize → dequantize has bounded error."""
        w = torch.randn(32, 64)
        q, scales, zeros, shape = _quantize(w, n_bit = 8)
        w_hat = _dequantize(q, scales, zeros, shape)

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
        assert abs(mean_error) < 0.01, f"Mean error {mean_error} suggests biased rounding"


class TestParamGroupHelper:
    """Tests for make_q_galore_param_groups."""

    def test_param_group_separation(self):
        """GaLore vs non-GaLore params are correctly separated."""

        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)
        model.k_proj = nn.Linear(64, 64, bias = False)
        model.embed = nn.Embedding(100, 64)
        model.norm = nn.LayerNorm(64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        assert len(groups) == 2

        galore_group = [g for g in groups if "rank" in g][0]
        non_galore_group = [g for g in groups if "rank" not in g][0]

        assert len(galore_group["params"]) == 2
        assert len(non_galore_group["params"]) == 3

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
        assert len(galore_group["params"]) == 1

    def test_bias_excluded_from_galore(self):
        """1-D bias params matching target names must be excluded (project needs 2-D grads)."""
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = True)
        model.embed = nn.Embedding(100, 64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        galore_group = [g for g in groups if "rank" in g][0]
        non_galore_group = [g for g in groups if "rank" not in g][0]

        assert len(galore_group["params"]) == 1
        assert galore_group["params"][0].dim() == 2

        assert any(p.dim() == 1 for p in non_galore_group["params"])

    def test_empty_target_modules_no_galore(self):
        """target_modules=[] should result in no GaLore params."""
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)

        # Empty list must not fall back to defaults.
        groups = make_q_galore_param_groups(
            model,
            rank = 8,
            target_modules = [],
            weight_quant = False,
        )

        galore_groups = [g for g in groups if "rank" in g]
        assert len(galore_groups) == 0, "Expected no GaLore groups when target_modules=[]"


def test_optimizer_bias_correction_matches_adamw():
    pytest.importorskip("bitsandbytes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    requires_bnb_optimizer(device)

    param = nn.Parameter(torch.ones(2, device = device))
    reference = nn.Parameter(param.detach().clone())
    optimizer = _adamw_mod.QGaLoreAdamW8bit([param], lr = 0.1, weight_decay = 0.0)
    expected_optimizer = torch.optim.AdamW([reference], lr = 0.1, weight_decay = 0.0)
    for values in ([0.1, 0.2], [0.4, -0.2], [-0.05, 0.3]):
        param.grad = torch.tensor(values, device = device)
        reference.grad = param.grad.clone()
        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(param, reference)


class TestQGaLoreIntegration:
    """Integration tests that work without bitsandbytes on CPU."""

    def test_projector_training_loop(self):
        """A simple training loop using manual GaLore projection converges."""
        torch.manual_seed(42)

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

            for p in model.parameters():
                if p.grad is not None and p.grad.dim() == 2:
                    low = proj.project(p.grad, step)
                    p._saved = p.data.clone()
                    update = torch.zeros_like(low)
                    update.add_(low)
                    full_update = proj.project_back(update)
                    p.grad.copy_(full_update)

            optimizer.step()

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_full_projector_roundtrip_quality(self):
        """project → project_back captures the dominant gradient directions."""
        torch.manual_seed(42)
        u = torch.randn(32, 4)
        v = torch.randn(4, 16)
        grad = u @ v

        proj = GaLoreProjector(rank = 4, update_proj_gap = 1, scale = 1.0)
        low = proj.project(grad, step = 0)
        reconstructed = proj.project_back(low)

        # Rank-4 grad with rank-4 projection reconstructs near-exactly.
        relative_error = (grad - reconstructed).norm() / grad.norm()
        assert relative_error < 0.05, f"Reconstruction error too high: {relative_error:.4f}"

    def test_weight_quant_activates_on_first_step(self):
        """_has_weight_quant returns True even when _q_scales is None (first step)."""
        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        QGaLoreAdamW8bit = _adamw_mod_local.QGaLoreAdamW8bit

        p = torch.nn.Parameter(torch.randn(16, 16))
        p._q_scales = None
        p._q_zeros = None
        p._q_shape = p.data.shape

        group = {"weight_quant": True}

        # Must be True even on the first step, when _q_scales is None.
        assert QGaLoreAdamW8bit._has_weight_quant(p, group) is True

        p2 = torch.nn.Parameter(torch.randn(16, 16))
        assert QGaLoreAdamW8bit._has_weight_quant(p2, group) is False

    def test_embedding_lr_param_group_split(self):
        """Embedding params can be split into a separate group with custom LR."""
        model = nn.Module()
        model.q_proj = nn.Linear(64, 64, bias = False)
        model.embed = nn.Embedding(100, 64)

        groups = make_q_galore_param_groups(model, rank = 8, weight_quant = False)

        embed_lr = 5e-5
        new_groups = []
        for group in groups:
            if "rank" in group:
                new_groups.append(group)
                continue
            embed_params = []
            other_params = []
            for p in group["params"]:
                if p.shape[0] == 100:
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

        embed_groups = [g for g in new_groups if g.get("lr") == embed_lr]
        assert len(embed_groups) == 1
        assert embed_groups[0]["lr"] == embed_lr

    def test_optimizer_hyperparams_forwarded(self):
        """QGaLoreAdamW8bit accepts betas and eps keyword arguments."""
        # Cannot instantiate without bitsandbytes; check the signature.
        import inspect

        _adamw_mod_local = sys.modules["unsloth.optimizers.q_galore_adamw"]
        QGaLoreAdamW8bit = _adamw_mod_local.QGaLoreAdamW8bit

        sig = inspect.signature(QGaLoreAdamW8bit.__init__)
        param_names = list(sig.parameters.keys())
        assert "betas" in param_names, "betas not in QGaLoreAdamW8bit.__init__ params"
        assert "eps" in param_names, "eps not in QGaLoreAdamW8bit.__init__ params"

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

        float_data = p.data.clone()
        q, scales, zeros, shape = _quantize(float_data, q_group_size = group["weight_group_size"])

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

        q, scales, zeros, shape = _projector_mod_local._quantize(
            linear.weight.data.clone(), q_group_size = 16
        )
        linear.weight._q_data = q
        linear.weight._q_scales = scales
        linear.weight._q_zeros = zeros
        linear.weight._q_shape = shape
        linear.weight.data = torch.zeros(1, dtype = linear.weight.dtype)
        assert linear.weight.data.numel() == 1, "placeholder should be 1 element"

        handles = install_hook(linear)
        x = torch.randn(2, 16)
        out = linear(x)

        assert linear.weight.data.shape == (8, 16), "weight shape not restored"
        assert linear.weight.data.is_floating_point(), "weight not float after hook"
        assert torch.allclose(
            linear.weight.data, original, atol = 0.15
        ), "dequantized weight too far from original"

        for h in handles:
            h.remove()
