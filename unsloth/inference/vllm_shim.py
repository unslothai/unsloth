"""vLLM-API surface for the flex inference backend.

`FlexEngine.generate` / `.chat` return :class:`RequestOutput` objects with the
same attribute shape (`prompt_token_ids`, `outputs[i].token_ids`,
`outputs[i].text`, `outputs[i].logprobs`) that vLLM's `LLM.generate` does, so
TRL's GRPO trainer — which reads `output.prompt_token_ids` and iterates
`output.outputs[i].token_ids` (trl/trainer/grpo_trainer.py:1274-1279) — does
not care which backend produced the result.

`save_lora` / `load_lora` mirror `unsloth_zoo.vllm_utils.save_lora` /
`load_lora` (vllm_utils.py:2389-2628). The only backend-visible difference is
the `LoRARequest` class: we emit our shim, not vllm's, so the GRPO patch in
`unsloth/models/rl.py:1880` passes our object straight to `FlexEngine.generate`.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# vLLM result objects
# ---------------------------------------------------------------------------


@dataclass
class CompletionOutput:
    """vLLM `CompletionOutput` stand-in.

    TRL reads ``.token_ids`` (list of int) and ``.logprobs`` (optional list of
    dict[int, Logprob]); we populate both."""

    index: int = 0
    text: str = ""
    token_ids: list = field(default_factory = list)
    cumulative_logprob: Optional[float] = None
    logprobs: Optional[list] = None
    finish_reason: Optional[str] = "stop"
    stop_reason: Optional[str] = None


@dataclass
class RequestOutput:
    """vLLM `RequestOutput` stand-in.

    TRL reads ``.prompt_token_ids`` (list of int) and iterates
    ``.outputs`` (list[CompletionOutput])."""

    request_id: str = ""
    prompt: str = ""
    prompt_token_ids: list = field(default_factory = list)
    outputs: list = field(default_factory = list)
    finished: bool = True


# ---------------------------------------------------------------------------
# LoRARequest stand-in
# ---------------------------------------------------------------------------
#
# The vLLM + unsloth_zoo combo imports ``vllm.lora.request.LoRARequest`` at
# call time. When the flex backend is selected we never hit that import; we
# pass our own dataclass that carries the same three attributes the caller
# reads: ``lora_name``, ``lora_int_id``, and ``lora_tensors`` +
# ``lora_config`` (for the in-memory LoRA fast path).


@dataclass
class LoRARequest:
    lora_name: str = ""
    lora_int_id: int = 0
    lora_path: Optional[str] = None
    lora_tensors: Optional[dict] = None  # dict[str, torch.Tensor]
    lora_config: Any = None


# ---------------------------------------------------------------------------
# save_lora / load_lora — drop-in for unsloth_zoo.vllm_utils equivalents
# ---------------------------------------------------------------------------

_LORA_REQUEST_ID: Optional[int] = None


def save_lora(model, save_directory, *args, **kwargs):
    """Dump the PEFT LoRA tensors (``.lora_A.`` / ``.lora_B.``) into a
    PEFT-compatible directory. Mirrors
    ``unsloth_zoo.vllm_utils.save_lora`` (vllm_utils.py:2389-2397) byte-for-byte
    so existing callers (e.g. TRL's GRPOTrainer patch) are untouched."""
    state_dict = model.state_dict()
    dtype = model.get_input_embeddings().weight.dtype
    state_dict = {
        k: v.to(dtype)
        for k, v in state_dict.items()
        if ".lora_A." in k or ".lora_B." in k
    }
    kwargs["state_dict"] = state_dict
    model.save_pretrained(save_directory = save_directory, *args, **kwargs)


def _get_peft_config(save_directory):
    """Late-imported to keep the `peft` dep optional at module-load time."""
    from peft import PeftConfig

    return PeftConfig.from_pretrained(save_directory)


def load_lora(
    model,
    save_directory,
    load_tensors: bool = False,
    lora_request_id: Optional[int] = None,
):
    """Build a :class:`LoRARequest` the flex backend can consume.

    Mirrors ``unsloth_zoo.vllm_utils.load_lora`` (vllm_utils.py:2574-2628):
    increments a module-level counter so each request gets a fresh
    ``lora_int_id``, writes the PEFT adapter config to ``save_directory`` on
    first call (or when ``load_tensors=True``) and captures the current
    state-dict LoRA tensors so the engine can merge them without an extra
    disk round-trip."""
    global _LORA_REQUEST_ID
    if _LORA_REQUEST_ID is None:
        _LORA_REQUEST_ID = 1
    if lora_request_id is None:
        lora_request_id = _LORA_REQUEST_ID
    if not os.path.exists(save_directory) or lora_request_id == 1:
        if load_tensors:
            model.peft_config["default"].save_pretrained(save_directory)
        elif not os.path.exists(save_directory):
            raise OSError(
                f"Unsloth: LoRA filepath = {save_directory} does not exist!"
            )

    if load_tensors:
        peft_config = _get_peft_config(save_directory)
        state_dict = model.state_dict()
        state_dict = {
            k.replace(".default", ""): v
            for k, v in state_dict.items()
            if ".lora_A." in k or ".lora_B." in k
        }
        req = LoRARequest(
            lora_name = str(lora_request_id),
            lora_int_id = lora_request_id,
            lora_tensors = state_dict,
            lora_config = peft_config,
        )
    else:
        req = LoRARequest(
            lora_name = str(lora_request_id),
            lora_int_id = lora_request_id,
            lora_path = save_directory,
        )
    _LORA_REQUEST_ID += 1
    return req


# Partial-applied variants, mirroring ``patch_peft_fast_inference``:
#   model.save_lora = functools.partial(save_lora, model)
#   model.load_lora = functools.partial(load_lora, model)
# are attached by the caller (unsloth/models/_utils.py:2707-2708).

__all__ = [
    "CompletionOutput",
    "RequestOutput",
    "LoRARequest",
    "save_lora",
    "load_lora",
]
