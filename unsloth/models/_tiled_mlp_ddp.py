# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import torch
from unsloth_zoo.gradient_checkpointing import set_device_states
from unsloth_zoo.tiled_mlp import TiledMLP, torch_amp_custom_bwd

__all__ = ["patch_tiled_mlp_for_ddp"]


class _TiledMLPWithParams(torch.autograd.Function):
    # TiledMLP runs a nested .backward() per chunk, so DDP's grad hooks fire once per chunk ("marked ready twice", or a
    # partial grad all-reduced), and its params are invisible to find_unused_parameters. Taking them as inputs and
    # returning their summed grads lets autograd accumulate each exactly once.
    @staticmethod
    def forward(
        ctx, mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen, *params
    ):
        ctx.params = params
        return TiledMLP.forward(
            ctx, mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen
        )

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output, *args):
        x = ctx.saved_tensors[0]
        H = x.shape[-1]
        rng_devices = ctx.fwd_devices if ctx.preserve_rng_state and ctx.had_device_in_fwd else []
        with torch.random.fork_rng(
            devices = rng_devices, enabled = ctx.preserve_rng_state, device_type = ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(
                        ctx.fwd_devices, ctx.fwd_device_states, device_type = ctx.device_type
                    )
            x_gradients = torch.zeros_like(x, memory_format = torch.preserve_format)
            param_grads = [None] * len(ctx.params)
            extra_outputs = []
            for x_split, grad_split, x_grad in zip(
                torch.split(x.view(-1, H), ctx.split_sizes, dim = 0),
                torch.split(grad_output.reshape(-1, H), ctx.split_sizes, dim = 0),
                torch.split(x_gradients.view(-1, H), ctx.split_sizes, dim = 0),
            ):
                x_split = x_split.unsqueeze(0).requires_grad_(True)
                with torch.enable_grad():
                    outputs = TiledMLP.handle_output(ctx.mlp_forward(x_split), extra_outputs)
                grads = torch.autograd.grad(
                    outputs,
                    (x_split, *ctx.params),
                    grad_split.unsqueeze(0),
                    allow_unused = True,
                )
                x_grad.copy_(grads[0].squeeze(0))
                for i, g in enumerate(grads[1:]):
                    if g is not None:
                        param_grads[i] = g if param_grads[i] is None else param_grads[i].add_(g)
        return (None, None, x_gradients, None, None, None, *param_grads)


def _apply_with_params(mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen):
    params = [p for p in mlp_module.parameters() if p.requires_grad]
    return _TiledMLPWithParams.apply(
        mlp_forward, mlp_module, x, preserve_rng_state, num_shards, max_flat_qlen, *params
    )


def patch_tiled_mlp_for_ddp():
    TiledMLP.apply = staticmethod(_apply_with_params)
