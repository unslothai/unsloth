# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engine-specific launch policy and selected-device memory budgeting."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen = True)
class EngineAdapter:
    name: str
    module: str
    model_option: str
    context_option: str
    memory_option: str
    extra_args: tuple[str, ...] = ()
    exact_token_count: bool = False

    def environment(self, tensor_parallel_size):
        if self.name == "vllm" and tensor_parallel_size > 1:
            return {"VLLM_HOST_IP": "127.0.0.1"}
        if self.name == "sglang" and tensor_parallel_size > 1:
            # SGLang otherwise rejects GPUs whose free capacities differ by
            # more than 10%, including idle 24/48 GiB cards. Studio budgets
            # every device and SGLang sizes KV pools from the minimum free
            # memory across ranks, so equal capacities are not required.
            return {"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0"}
        return {}

    def command(
        self,
        python,
        model,
        port,
        key,
        context,
        memory_fraction,
        tensor_parallel_size = 1,
    ):
        parallel_args = ["--tensor-parallel-size", str(tensor_parallel_size)]
        if tensor_parallel_size > 1:
            # Keep vLLM on this machine. SGLang must probe peer access before
            # choosing its custom collective, including PCIe-only GPU pairs.
            parallel_args += (
                ["--distributed-executor-backend", "mp"]
                if self.name == "vllm"
                else ["--enable-p2p-check"]
            )
        return [
            python,
            "-I",
            "-m",
            self.module,
            self.model_option,
            model,
            self.context_option,
            str(context),
            self.memory_option,
            str(memory_fraction),
            *parallel_args,
            *self.extra_args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            key,
            "--served-model-name",
            model,
            "--load-format",
            "safetensors",
        ]

    def progress(self, line):
        lowered = line.lower()
        if "loading model weights" in lowered or "loading safetensors" in lowered:
            return "loading_weights"
        if any(word in lowered for word in ("cuda graph", "compil", "warmup")):
            return "warming_up"
        return None


ADAPTERS = {
    "vllm": EngineAdapter(
        "vllm",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "--max-model-len",
        "--gpu-memory-utilization",
        ("--generation-config", "vllm"),
        True,
    ),
    "sglang": EngineAdapter(
        "sglang",
        "sglang.launch_server",
        "--model-path",
        "--context-length",
        "--mem-fraction-static",
        ("--attention-backend", "triton", "--sampling-backend", "pytorch"),
    ),
}


def launch_arguments(
    engine,
    python,
    model,
    port,
    key,
    context,
    memory_fraction = 0.8,
    tensor_parallel_size = 1,
):
    try:
        adapter = ADAPTERS[engine]
    except KeyError:
        raise ValueError("Unknown inference engine") from None
    return adapter.command(python, model, port, key, context, memory_fraction, tensor_parallel_size)


def gpu_memory_fraction(gpu_ids: list[int]) -> float:
    """Budget every selected physical GPU after the previous resident is stopped.

    Reserve at least 512 MiB for driver allocations. An unreadable device is an
    actionable failure, never permission to fall back to a larger engine default.
    """
    from utils.vram_budget_settings import get_vram_budget_fraction
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            timeout = 5,
            check = True,
        )
        rows = result.stdout.strip().splitlines()
        if len(rows) != len(gpu_ids):
            raise ValueError("Could not measure every selected GPU")
        fraction = get_vram_budget_fraction()
        for row in rows:
            total, free = (float(value.strip()) for value in row.split(","))
            if total <= 0 or free <= 512 or free > total:
                raise ValueError("Insufficient available GPU memory")
            # The engines accept one fraction for all ranks. The most
            # constrained GPU must bound that fraction, not the first GPU.
            fraction = min(fraction, (free - 512) / total)
        if fraction < 0.05:
            raise ValueError("Insufficient available GPU memory")
        # Round down, so the reservation cannot exceed the measured free memory.
        return int(fraction * 1000) / 1000
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not reserve memory on every selected GPU. Free GPU memory or select fewer GPUs and retry."
        ) from exc
