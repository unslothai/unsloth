import logging
from contextlib import contextmanager
from functools import partial

import torch

from unsloth.utils.logging import setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def cuda_profiler_wrapper(torch_profiler, warmup=5, rep=5):
    def wrapper(fn):
        def inner(*args, **kwargs):
            for _ in range(warmup):
                fn(*args, **kwargs)
                torch_profiler.step()
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(rep):
                fn(*args, **kwargs)
                torch_profiler.step()
            torch.cuda.cudart().cudaProfilerStop()

        return inner

    return wrapper


class CudaRangeHooks:
    @staticmethod
    def pre_hook(module, args, prefix=""):
        message = f"{prefix}_PRE_{module.__class__.__qualname__}"
        # logger.debug(f"Range pre-hook: {message}")
        torch.profiler.record_function(f"{message}")
        torch.cuda.nvtx.range_push(message)

    @staticmethod  # define register_full_backward_hook function
    def post_hook(module, args, output):
        # logger.debug(f"Range post-hook: {module.__class__.__qualname__}")
        torch.profiler.record_function(f"POST_{module.__class__.__qualname__}")
        torch.cuda.nvtx.range_pop()


@contextmanager
def cuda_nvtx_range_context():
    forward_pre_hook = partial(CudaRangeHooks.pre_hook, prefix="FORWARD")
    backward_pre_hook = partial(CudaRangeHooks.pre_hook, prefix="BACKWARD")
    handles = [
        torch.nn.modules.module.register_module_forward_pre_hook(forward_pre_hook),
        torch.nn.modules.module.register_module_forward_hook(CudaRangeHooks.post_hook),
        torch.nn.modules.module.register_module_full_backward_pre_hook(
            backward_pre_hook
        ),
        torch.nn.modules.module.register_module_full_backward_hook(
            CudaRangeHooks.post_hook
        ),
    ]

    yield
    for handle in handles:
        handle.remove()


def trace_handler(p, outdir="./"):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(f"{outdir}/trace.json")


@contextmanager
def torch_profiler_context(
    record_shapes=False,
    with_stack=False,
    warmup=5,
    active=5,
    outdir="./",
):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
        on_trace_ready=partial(trace_handler, outdir=outdir),
    ) as prof:
        yield prof
