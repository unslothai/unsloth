# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engine-specific launch policy and selected-device memory budgeting."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import json
from pathlib import Path


def tool_parser_for_template(template, engine):
    """Select a native output parser from the model's own tool syntax."""
    if not isinstance(template, str) or "tools" not in template:
        return None
    # Specific nested formats precede the generic JSON tool-call envelope.
    formats = (
        (("<tool_call>", "<function="), "qwen3_coder", "qwen3_coder"),
        (("<tool_call>", "<arg_key>"), "glm45", "glm"),
        (("<tool_call>", "arguments"), "hermes", "qwen"),
        (("[TOOL_CALLS]",), "mistral", "mistral"),
        (("<|python_tag|>",), "llama3_json", "llama3"),
        (("<|channel|>", "<|message|>"), "openai", "gpt-oss"),
        (("<|tool_calls_section_begin|>",), "kimi_k2", "kimi_k2"),
        (("<minimax:tool_call>",), "minimax_m2", "minimax-m2"),
        (("<｜tool▁calls▁begin｜>",), "deepseek_v3", "deepseekv3"),
    )
    for markers, vllm, sglang in formats:
        if all(marker in template for marker in markers):
            return vllm if engine == "vllm" else sglang
    return None


@dataclass(frozen = True)
class EngineAdapter:
    name: str
    module: str
    model_option: str
    context_option: str
    memory_option: str
    extra_args: tuple[str, ...] = ()
    exact_token_count: bool = False

    def environment(self, gpu_count):
        if self.name == "vllm" and gpu_count > 1:
            return {"VLLM_HOST_IP": "127.0.0.1"}
        if self.name == "sglang" and gpu_count > 1:
            # SGLang otherwise rejects GPUs whose free capacities differ by
            # more than 10%, including idle 24/48 GiB cards. Studio budgets
            # every device and SGLang sizes KV pools from the minimum free
            # memory across ranks, so equal capacities are not required.
            return {
                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
                # CuTe RMSNorm can compile for rank zero's architecture on every
                # rank. The CUDA implementation supports mixed Ampere/Ada groups.
                "FLASHINFER_USE_CUDA_NORM": "1",
            }
        return {}

    def command(
        self,
        python,
        model,
        port,
        key,
        context,
        memory_fraction,
        gpu_count = 1,
        options = None,
        trust_remote_code = False,
    ):
        options = options or {}
        tool_args = []
        if options.get("tool_parser"):
            tool_args = ["--tool-call-parser", options["tool_parser"]]
            if self.name == "vllm":
                tool_args.append("--enable-auto-tool-choice")
        mode = options.get("parallelism", "tensor")
        # Exactly one parallel dimension spans the selected devices;
        # the others stay at one.
        parallel_args = ["--tensor-parallel-size", str(gpu_count if mode == "tensor" else 1)]
        if mode != "tensor":
            flag = "--pipeline-parallel-size" if mode == "pipeline" else "--data-parallel-size"
            parallel_args += [flag, str(gpu_count)]
        if gpu_count > 1:
            parallel_args += (
                [
                    "--data-parallel-backend"
                    if mode == "data"
                    else "--distributed-executor-backend",
                    "mp",
                ]
                if self.name == "vllm"
                else ["--enable-p2p-check"]
            )
        precision = options.get("precision", "auto")
        precision_args = []
        if precision in ("bf16", "fp16"):
            precision_args = ["--dtype", "bfloat16" if precision == "bf16" else "float16"]
        elif self.name == "vllm" and (
            precision in ("int8", "fp8") or (precision == "int4" and mode == "pipeline")
        ):
            # Native online INT8 only converts MoE experts; online FP8 and
            # BitsAndBytes INT4 pipeline loads can return invalid output.
            # Use the native TorchAO loader for those configurations.
            config = {
                "_type": {
                    "int4": "Int4WeightOnlyConfig",
                    "int8": "Int8WeightOnlyConfig",
                    "fp8": "Float8WeightOnlyConfig",
                }[precision],
                "_version": 1 if precision == "int8" else 2,
                "_data": {"set_inductor_config": True},
            }
            if precision == "int4":
                config["_data"].update(
                    {
                        "group_size": 32,
                        "int4_packing_format": {
                            "_type": "Int4PackingFormat",
                            "_data": "TILE_PACKED_TO_4D",
                        },
                        "int4_choose_qparams_algorithm": {
                            "_type": "Int4ChooseQParamsAlgorithm",
                            "_data": "HQQ",
                        },
                    }
                )
            precision_args = [
                "--quantization",
                "torchao",
                "--hf-overrides",
                json.dumps({"quantization_config_dict_json": json.dumps(config)}),
            ]
        elif precision == "int4":
            precision_args = (
                ["--quantization", "bitsandbytes"]
                if self.name == "vllm"
                else ["--torchao-config", "int4wo-32"]
            )
        elif precision == "int8":
            precision_args = ["--torchao-config", "int8wo"]
        elif precision == "fp8":
            precision_args = (
                ["--torchao-config", "fp8wo"]
                if options.get("disable_cuda_graph")
                else ["--quantization", "fp8"]
            )
        if options.get("disable_cuda_graph"):
            precision_args.extend(
                ["--enforce-eager"]
                if self.name == "vllm"
                else ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
            )
        if self.name == "sglang":
            entrypoint = [
                str(Path(__file__).with_name("sglang_server.py")),
                self.model_option,
                model,
            ]
        elif mode == "data":
            # The native serve CLI owns data-parallel worker/API orchestration.
            entrypoint = ["-m", "vllm.entrypoints.cli.main", "serve", model]
        else:
            entrypoint = ["-m", self.module, self.model_option, model]
        return [
            python,
            "-I",
            *entrypoint,
            self.context_option,
            str(context),
            self.memory_option,
            str(memory_fraction),
            *parallel_args,
            *precision_args,
            *tool_args,
            *(["--trust-remote-code"] if trust_remote_code else []),
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
            options.get("load_format", "auto"),
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
