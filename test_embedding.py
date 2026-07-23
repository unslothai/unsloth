import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import mlx_to_torch

arr = mx.array([[1, 2, 3], [4, 5, 6]], dtype = mx.int32)
tensor = mlx_to_torch(arr)
print("Tensor:", tensor)
print("Tensor device:", tensor.device)

emb = torch.nn.Embedding(10, 5).to("mps")
try:
    res = emb(tensor.to("mps"))
    print("Success")
except Exception as e:
    print("Error:", e)

# Test with a clone
tensor2 = tensor.clone()
try:
    res = emb(tensor2.to("mps"))
    print("Success with clone")
except Exception as e:
    print("Error with clone:", e)
