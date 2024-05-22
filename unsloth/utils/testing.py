import torch


def check_all(expected, actual, names, atol=1e-6, rtol=1e-6, verbose=True):
    if verbose:
        print()
    for name, e, a in zip(names, expected, actual):
        if verbose:
            print(
                f"{name}: {torch.allclose(e, a, atol=atol, rtol=rtol)}: {(e - a).abs().max()}"
            )
        assert torch.allclose(
            e, a, atol=atol, rtol=rtol
        ), f"{name}: {(e - a).abs().max()}"
