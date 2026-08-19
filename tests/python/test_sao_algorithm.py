"""SAO (arXiv:2607.07508) algorithm core: DIS calibration and skip-observation GAE.

Pure-tensor tests, hand-computed expectations, no model or GPU involved.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _load_sao():
    """Import the SAO core, falling back to a by-path load.

    `unsloth.models` pulls in the full loader stack; the algorithm core needs
    nothing but torch, so a CPU-only runner still gets to exercise it.
    """
    try:
        from unsloth.models import sao
        return sao
    except Exception:
        import importlib.util
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "unsloth",
            "models",
            "sao.py",
        )
        spec = importlib.util.spec_from_file_location("_unsloth_sao", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


_sao = _load_sao()

SAOConfig = _sao.SAOConfig
dis_calibrate = _sao.dis_calibrate
freeze_critic_attention = _sao.freeze_critic_attention
sao_policy_loss = _sao.sao_policy_loss
sao_value_loss = _sao.sao_value_loss
skip_observation_gae = _sao.skip_observation_gae


def test_dis_calibrate_masks_outside_window_and_passes_inside():
    ratios = torch.tensor([[0.5, 0.7, 1.0, 3.0, 6.0, 0.69, 6.01]])
    out = dis_calibrate(ratios, eps_low = 0.3, eps_high = 5.0)
    expected = torch.tensor([[0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(out, expected)


def test_dis_calibrate_boundaries_are_exclusive():
    ratios = torch.tensor([0.7, 6.0])
    assert torch.allclose(dis_calibrate(ratios, 0.3, 5.0), torch.zeros(2))


def test_dis_calibrate_rejects_non_positive_epsilons():
    with pytest.raises(ValueError):
        dis_calibrate(torch.ones(2), eps_low = 0.0, eps_high = 5.0)


def test_skip_observation_gae_matches_standard_gae_without_observations():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.25, 0.75]])
    mask = torch.ones(1, 3)
    gamma, lam = 1.0, 0.95

    adv, ret = skip_observation_gae(rewards, values, mask, gamma = gamma, lam = lam)

    d2 = 1.0 + gamma * 0.0 - 0.75
    a2 = d2
    d1 = 0.0 + gamma * 0.75 - 0.25
    a1 = d1 + gamma * lam * a2
    d0 = 0.0 + gamma * 0.25 - 0.5
    a0 = d0 + gamma * lam * a1
    assert torch.allclose(adv, torch.tensor([[a0, a1, a2]]), atol = 1e-6)
    assert torch.allclose(ret, adv + values, atol = 1e-6)


def test_skip_observation_gae_bridges_across_observation_span():
    # positions 1 and 2 are tool-observation tokens the policy did not generate.
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 9.0, -9.0, 0.25, 0.75]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0]])
    gamma, lam = 1.0, 0.95

    adv, _ = skip_observation_gae(rewards, values, mask, gamma = gamma, lam = lam)

    a4 = 1.0 - 0.75
    a3 = (0.0 + gamma * 0.75 - 0.25) + gamma * lam * a4
    # bridges 0 -> 3, so the observation values 9.0 / -9.0 never enter delta.
    a0 = (0.0 + gamma * 0.25 - 0.5) + gamma * lam * a3
    assert torch.allclose(adv, torch.tensor([[a0, 0.0, 0.0, a3, a4]]), atol = 1e-6)


def test_skip_observation_gae_carries_reward_off_skipped_positions():
    rewards = torch.tensor([[0.0, 2.0, 0.0]])
    values = torch.tensor([[0.0, 5.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    adv, _ = skip_observation_gae(rewards, values, mask, gamma = 1.0, lam = 0.95)
    # the environment reward at the skipped position lands on token 0's delta.
    assert torch.allclose(adv[0, 0], torch.tensor(2.0), atol = 1e-6)
    assert adv[0, 1] == 0.0


def test_skip_observation_gae_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        skip_observation_gae(torch.zeros(1, 3), torch.zeros(1, 4), torch.ones(1, 3))


def test_policy_loss_ignores_tokens_masked_by_dis():
    policy = torch.tensor([[-1.0, -1.0]], requires_grad = True)
    # token 1 has ratio exp(-1 - (-9)) = e^8, far outside the window.
    rollout = torch.tensor([[-1.0, -9.0]])
    advantages = torch.tensor([[1.0, 100.0]])
    mask = torch.ones(1, 2)

    loss = sao_policy_loss(policy, rollout, advantages, mask, 0.3, 5.0)
    loss.backward()
    assert policy.grad[0, 1].item() == 0.0
    assert policy.grad[0, 0].item() != 0.0


def test_policy_loss_averages_over_action_tokens_only():
    policy = torch.tensor([[-1.0, -1.0]])
    rollout = torch.tensor([[-1.0, -1.0]])
    advantages = torch.tensor([[2.0, 2.0]])
    both = sao_policy_loss(policy, rollout, advantages, torch.ones(1, 2))
    one = sao_policy_loss(policy, rollout, advantages, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(both, one, atol = 1e-6)
    assert torch.allclose(both, torch.tensor(2.0), atol = 1e-6)


def test_value_loss_is_masked_mean_squared_error():
    values = torch.tensor([[1.0, 5.0]])
    returns = torch.tensor([[0.0, 0.0]])
    loss = sao_value_loss(values, returns, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(loss, torch.tensor(1.0), atol = 1e-6)


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = torch.nn.Linear(4, 4)


def test_freeze_critic_attention_freezes_only_attention_parameters():
    block = _Block()
    frozen = freeze_critic_attention(block)
    assert frozen == 2
    assert not block.self_attn.q_proj.weight.requires_grad
    assert block.mlp.weight.requires_grad


def test_sao_config_validates_hyperparameters():
    with pytest.raises(ValueError):
        SAOConfig(output_dir = "sao", value_updates_per_policy_update = 0)
    with pytest.raises(ValueError):
        SAOConfig(output_dir = "sao", eps_low = -0.1)
    cfg = SAOConfig(output_dir = "sao")
    assert (cfg.eps_low, cfg.eps_high) == (0.3, 5.0)
    assert cfg.value_updates_per_policy_update == 2
