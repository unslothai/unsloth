import torch


def check_all(expected, actual, atol=1e-6, rtol=1e-6):
    for e, a in zip(expected, actual):
        print(f"{torch.allclose(e, a, atol=atol, rtol=rtol)}: {(e - a).abs().max()}")
