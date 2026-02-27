#!/usr/bin/env python3
import torch
import torch.func

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

def my_func(x):
    return (x ** 2).sum()

x = torch.randn(4, requires_grad=True, device=device)

try:
    print("Testing torch.func.grad_and_value on MPS...")
    grad_fn = torch.func.grad_and_value(my_func)
    grad, val = grad_fn(x)
    print(f"  Value: {val.item()}")
    print(f"  Grad: {grad}")
    print("✅ torch.func works!")
except Exception as e:
    print(f"❌ torch.func failed: {e}")

print("-" * 30)

print("Testing UnslothFusedLoss logic on MPS...")
# Simulate accumulate_chunk logic
try:
    accumulated_loss = torch.zeros(1, device=device)[0]
    
    def loss_fn(y_pred, y_true):
        return torch.nn.functional.cross_entropy(y_pred, y_true)

    logits = torch.randn(2, 4, requires_grad=True, device=device)
    labels = torch.randint(0, 4, (2,), device=device)

    # Manual chunk simulation
    (chunk_grad_input,), (chunk_loss,) = torch.func.grad_and_value(
        loss_fn,
        argnums=(0,),
        has_aux=False,
    )(logits, labels)
    
    accumulated_loss.add_(chunk_loss)
    print(f"  Chunk Loss: {chunk_loss.item()}")
    print(f"  Chunk Grad Input: {chunk_grad_input}")
    print("✅ UnslothFusedLoss logic seems to work!")
except Exception as e:
    print(f"❌ UnslothFusedLoss logic failed: {e}")
