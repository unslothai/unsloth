import torch 
from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

x = torch.randn(1, 126, 51200, device='cuda')
y = torch.randn(1, 126, device='cuda')

fast_cross_entropy_loss(logits=x,labels=y)