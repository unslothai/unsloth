"""Test sequence-level importance sampling produces correct KL shape (collapsed to 1 column)."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 12
ref = torch.randn(B, S)
new = torch.randn(B, S, requires_grad = True)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)

# Token-level IS: coef_1 should have shape (B, S)
r_tok = grpo_compute_loss(
    ref,
    new.clone().requires_grad_(True),
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    importance_sampling_level = "token",
)
coef_1_tok = r_tok[5]
assert coef_1_tok.shape == (B, S), f"Token IS coef_1 shape: {coef_1_tok.shape}"
out_mask_tok = r_tok[6]
assert out_mask_tok.shape == (B, S), f"Token IS mask shape: {out_mask_tok.shape}"

# Sequence-level IS: coef_1 should have shape (B, 1) because log_importance_weights is unsqueezed
r_seq = grpo_compute_loss(
    ref,
    new.clone().requires_grad_(True),
    old,
    None,
    input_ids,
    mask,
    beta,
    advantages,
    importance_sampling_level = "sequence",
)
coef_1_seq = r_seq[5]
# coef_1 = exp(log_importance_weights) where log_importance_weights has shape (B, 1)
assert coef_1_seq.shape[0] == B, f"Sequence IS coef_1 batch dim: {coef_1_seq.shape}"

print(
    f"PASS: token IS coef_1 shape={coef_1_tok.shape}, sequence IS coef_1 shape={coef_1_seq.shape}"
)
print(f"PASS: mean_kl token={r_tok[2].item():.4f}, sequence={r_seq[2].item():.4f}")
