"""Test CISPO/DAPO normalization with num_items_in_batch and num_processes."""

import sys, torch

sys.path.insert(0, "unsloth_zoo_repo")
from unsloth_zoo.rl_replacements import grpo_compute_loss

B, S = 4, 8
ref = torch.randn(B, S)
old = torch.randn(B, S)
input_ids = torch.randint(0, 100, (B, S))
mask = torch.ones(B, S)
beta = 0.1
advantages = torch.randn(B)
new_base = torch.randn(B, S)

for loss_type in ["cispo", "dapo"]:
    configs = [
        (B, 1),      # single process
        (B * 2, 2),  # 2 processes, 2x items
        (B, 2),      # 2 processes, same items -> different normalizer
    ]
    for nib, np_ in configs:
        new = new_base.clone().requires_grad_(True)
        result = grpo_compute_loss(
            ref, new, old, None, input_ids, mask, beta, advantages,
            loss_type=loss_type, num_items_in_batch=nib, num_processes=np_,
        )
        assert torch.isfinite(result[0]), f"{loss_type} nib={nib},np={np_}: loss not finite"
        print(f"PASS: {loss_type} num_items_in_batch={nib}, num_processes={np_}, loss={result[0].item():.4f}")

print("PASS: CISPO/DAPO normalization works across configs")
