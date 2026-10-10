# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engine-specific launch policy and selected-device memory budgeting."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

from utils.subprocess_compat import windows_hidden_subprocess_kwargs
import json
from pathlib import Path


def tool_parser_for_template(template, engine):
    """Select a native output parser from the model's own tool syntax."""
    if not isinstance(template, str) or "tools" not in template:
        return None
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


def _release_at_least(version: str | None, floor: str) -> bool:
    from packaging.version import InvalidVersion, Version
    try:
        return version is not None and Version(version) >= Version(floor)
    except InvalidVersion:
        return False


@dataclass(frozen = True)
class EngineAdapter:
    name: str
    module: str
    model_option: str
    context_option: str
    memory_option: str
    extra_args: tuple[str, ...] = ()
    exact_token_count: bool = False

    def key_environment(self, key):
        """The API key travels in the environment, never argv, which any local user can read."""
        return {"VLLM_API_KEY" if self.name == "vllm" else "UNSLOTH_ENGINE_API_KEY": key}

    def environment(self, gpu_count):
        if self.name == "vllm" and gpu_count > 1:
            return {"VLLM_HOST_IP": "127.0.0.1"}
        if self.name == "sglang" and gpu_count > 1:
            # Else SGLang rejects GPUs whose free memory differs >10%; KV pools size from the minimum anyway.
            return {
                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
                # CuTe RMSNorm compiles for rank zero's arch only; CUDA path handles mixed Ampere/Ada.
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
        served_model_name = None,
    ):
        options = options or {}
        tool_args = []
        if options.get("tool_parser"):
            tool_args = ["--tool-call-parser", options["tool_parser"]]
            if self.name == "vllm":
                tool_args.append("--enable-auto-tool-choice")
        mode = options.get("parallelism", "tensor")
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
        legacy_torchao = not (
            self.name == "sglang" and _release_at_least(options.get("engine_version"), "0.5.18")
        )
        if not legacy_torchao and precision in ("int8", "int4"):
            # SGLang 0.5.18 removed --torchao-config, the only load-time INT8 / INT4 path it had.
            raise ValueError(
                f"SGLang {options.get('engine_version')} cannot convert weights to "
                f"{precision.upper()} when loading. Choose FP8 or Model default, or use vLLM."
            )
        if precision in ("bf16", "fp16"):
            precision_args = ["--dtype", "bfloat16" if precision == "bf16" else "float16"]
        elif self.name == "vllm" and (
            precision in ("int8", "fp8") or (precision == "int4" and mode == "pipeline")
        ):
            # Native online INT8 converts only MoE experts; online FP8 / BNB INT4 return invalid output.
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
                if options.get("disable_cuda_graph") and legacy_torchao
                else ["--quantization", "fp8"]
            )
        if self.name == "sglang" and "--torchao-config" in precision_args:
            # SGLang's Blackwell default (cutedsl TGV GEMM) has no TorchAO tensor-subclass dispatch.
            precision_args.extend(["--bf16-gemm-backend", "torch"])
        if options.get("disable_cuda_graph"):
            precision_args.extend(
                ["--enforce-eager"]
                if self.name == "vllm"
                else ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
                if legacy_torchao
                # 0.5.18 folded both into per-phase backends and removed the piecewise flag.
                else [
                    "--cuda-graph-backend-decode",
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                ]
            )
        if self.name == "sglang":
            entrypoint = [
                str(Path(__file__).with_name("sglang_server.py")),
                self.model_option,
                model,
            ]
        else:
            launcher = str(Path(__file__).with_name("vllm_server.py"))
            entrypoint = (
                [launcher, "vllm.entrypoints.cli.main", "serve", model]
                if mode == "data"
                else [launcher, self.module, self.model_option, model]
            )
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
            *(
                ["--max-num-seqs", str(options["max_num_seqs"])]
                if self.name == "vllm" and options.get("max_num_seqs")
                else []
            ),
            *(["--trust-remote-code"] if trust_remote_code else []),
            *self.extra_args,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            served_model_name or model,
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


def memory_reserve_mib(engine: str, options: dict | None) -> int:
    """MiB left outside the engine's budget, for what each engine allocates past it. On a 24 GB L4
    with Qwen2.5-0.5B, vLLM's sampler warmup failed at 512 MiB and passed from 1536 (it scales with
    the vocabulary, so larger ones need more); SGLang's CUDA graphs and NCCL buffers died mid-request
    at 512 and passed from 2048. vLLM with TorchAO weights overruns by 2.4-4.8 GiB more (B200).
    The warmup also grows with the card: vLLM raises its default max_num_seqs on large GPUs, and a
    96 GB RTX PRO 6000 ran out of memory in the sampler warmup with 3 GiB left, so
    ``gpu_memory_fraction`` keeps at least ``RESERVE_SHARE`` of each card free as well."""
    options = options or {}
    precision = options.get("precision", "auto")
    torchao = engine == "vllm" and (
        precision in ("int8", "fp8")
        or (precision == "int4" and options.get("parallelism", "tensor") == "pipeline")
    )
    return (3072 if engine == "vllm" else 4096) + (6144 if torchao else 0)


RESERVE_SHARE = 0.06


def gpu_memory_fraction(
    gpu_ids: list[int],
    reserve_mib: int = 512,
    reserve_share: float = 0.0,
) -> float:
    """Budget every selected physical GPU after the previous resident is stopped.

    Reserve the larger of ``reserve_mib`` and ``reserve_share`` of each card for allocations the
    engine does not budget. An unreadable
    device is an actionable failure, never permission to fall back to a larger engine default.
    """
    from utils.hardware.nvidia import _nvidia_smi_executable
    from utils.vram_budget_settings import get_vram_budget_fraction

    try:
        result = subprocess.run(
            [
                _nvidia_smi_executable(),
                "--id",
                ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            **windows_hidden_subprocess_kwargs(),
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
            check = True,
        )
        rows = result.stdout.strip().splitlines()
        if len(rows) != len(gpu_ids):
            raise ValueError("Could not measure every selected GPU")
        fraction = get_vram_budget_fraction()
        for row in rows:
            total, free = (float(value.strip()) for value in row.split(","))
            reserve = max(reserve_mib, reserve_share * total)
            if total <= 0 or free <= reserve or free > total:
                raise ValueError("Insufficient available GPU memory")
            # One fraction for all ranks: the most constrained GPU bounds it.
            fraction = min(fraction, (free - reserve) / total)
        if fraction < 0.05:
            raise ValueError("Insufficient available GPU memory")
        return int(fraction * 1000) / 1000
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not reserve memory on every selected GPU. Free GPU memory or select fewer GPUs and retry."
        ) from exc
