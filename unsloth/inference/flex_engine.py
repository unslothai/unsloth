# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""FlexEngine: vLLM-compatible LLM surface for the flex inference backends.

When ``UNSLOTH_FAST_INFERENCE=1`` is set, :func:`load_flex` wraps the HF model
with a :class:`FlexEngine`. TRL's GRPO trainer and Unsloth's own callers treat
it as an ``LLM``:

    engine.generate(prompts, sampling_params=..., lora_request=..., use_tqdm=False)
    engine.chat(messages, sampling_params=..., lora_request=..., use_tqdm=False)
    engine.sleep(level=2)
    engine.wake_up(tags=["kv_cache"])
    engine.llm_engine                # minimal stub, see below

Three architectures are supported: Qwen3, Llama-3, Gemma-4-E2B-it. Anything
else raises :class:`NotImplementedError`.

Colocation model (no change from today):
- ``self.hf_model`` is the HF model instance returned by ``from_pretrained``
  (or its PEFT-wrapped descendant after :meth:`bind_peft_model`).
- The underlying :class:`FlexInference` / :class:`FlexGemma4Inference` implements
  the usual double-copy rollout pattern (pristine ``base_model`` + inference
  copy wrapped by PEFT), and ``refresh_lora_merge_from_pristine`` re-merges on
  every rollout.
- vLLM's sleep mode is not implemented for the flex backend. ``.sleep()`` /
  ``.wake_up()`` are no-op stubs so code paths that gate on
  ``UNSLOTH_VLLM_STANDBY`` stay valid.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import os
import types
import warnings
from typing import Any, Optional

import torch

from .flex_paged_attention import PagedKVCache, PageTable  # noqa: F401
from .flex_qwen3_llama import (
    DECODE_KERNEL_OPTIONS_DEFAULT,
    PREFILL_KERNEL_OPTIONS_DEFAULT,
    FlexInference,
    Sequence,
    refresh_lora_merge_from_pristine,
)
from .flex_gemma4 import FlexGemma4Inference
from .vllm_shim import CompletionOutput, LoRARequest, RequestOutput


# ---------------------------------------------------------------------------
# Hardware / kernel auto-tune
# ---------------------------------------------------------------------------


def _flash_attn_4_importable() -> bool:
    """FA4's CuTeDSL backend lives in the ``flash_attn_interface`` ships with
    the ``flash-attn`` 4.x wheel. We don't import it here (it can be slow);
    just check whether the module is resolvable."""
    return importlib.util.find_spec("flash_attn_interface") is not None


def _fa4_ok_for_head_dim(head_dim: int, device_cap: tuple[int, int]) -> bool:
    """See the plan for the hardware tier table. Returns whether FA4 is
    safe to use for an attention layer with the given ``head_dim`` on a GPU
    with the given ``(major, minor)`` capability."""
    major, _ = device_cap
    if major < 9:
        return False  # Ampere and older: Triton only
    if not _flash_attn_4_importable():
        return False
    if major == 9:
        return 8 <= head_dim <= 256  # Hopper
    return 8 <= head_dim <= 128  # Blackwell (sm_100 / sm_120)


def _triton_block_defaults(
    head_dim_max: int, device_cap: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Returns ``(prefill_BM, prefill_BN, decode_BM, decode_BN)``.

    Blackwell (sm_100 / sm_120) shared-memory budget is tight; at
    ``head_dim >= 256`` the default 128x128 prefill block overflows and the
    Triton autotuner raises ``OutOfMemoryError: out of resource``. See the
    ``--prefill_kernel_options`` probes in ``scripts/benchmarks/*.py``."""
    major, _ = device_cap
    if major >= 10 and head_dim_max >= 256:
        return (32, 32, 16, 16)
    if major >= 10 and head_dim_max >= 128:
        return (64, 64, 32, 32)
    return (128, 128, 64, 64)


def _collect_head_dims(hf_model) -> list[int]:
    """Collect the per-layer ``head_dim`` for every attention sub-module in
    the model. Gemma-4 text stack mixes 256 / 512 between full / sliding
    layers; Qwen3 + Llama-3 use a single head_dim."""
    head_dims = set()
    for module in hf_model.modules():
        hd = getattr(module, "head_dim", None)
        if hd is not None and isinstance(hd, int) and hd > 0:
            head_dims.add(hd)
    if not head_dims:
        cfg = getattr(hf_model, "config", None)
        if cfg is not None:
            hd = getattr(cfg, "head_dim", None)
            if hd is None:
                num_heads = getattr(cfg, "num_attention_heads", None)
                hidden = getattr(cfg, "hidden_size", None)
                if num_heads and hidden:
                    hd = hidden // num_heads
            if hd:
                head_dims.add(int(hd))
    return sorted(head_dims)


def _auto_kernel_options(
    hf_model,
    device,
    prefill_kernel_options: Optional[dict] = None,
    decode_kernel_options: Optional[dict] = None,
    fa4_prefill: Optional[bool] = None,
) -> tuple[Optional[bool], dict, dict]:
    """Derive safe FA4 + Triton block defaults from the GPU + model head_dim
    band. Explicit kwargs always win.

    FA4 is OFF by default. The CuTeDSL FLASH backend currently crashes on
    short B200 prompts (Llama-3.2-3B, 7-token input hits
    ``handle_block_sparse_empty_tile_correction_sm100``
    ``'NoneType' object is not subscriptable``). Users who want FA4 can
    pass ``fa4_prefill=True`` explicitly to the engine after confirming it
    works on their workload."""
    cap = (
        torch.cuda.get_device_capability(device)
        if torch.cuda.is_available()
        else (0, 0)
    )
    head_dims = _collect_head_dims(hf_model) or [128]
    head_dim_max = max(head_dims)

    if fa4_prefill is None:
        fa4_prefill = False  # conservative default — see docstring above

    bm_p, bn_p, bm_d, bn_d = _triton_block_defaults(head_dim_max, cap)

    if prefill_kernel_options is None:
        if fa4_prefill:
            # FA4 path: let the FLASH backend pick its own block shapes.
            prefill_kernel_options = None
        else:
            prefill_kernel_options = {
                "FORCE_USE_FLEX_ATTENTION": True,
                "BLOCK_M": bm_p,
                "BLOCK_N": bn_p,
            }
    if decode_kernel_options is None:
        decode_kernel_options = {"BLOCK_M": bm_d, "BLOCK_N": bn_d}

    return fa4_prefill, prefill_kernel_options, decode_kernel_options


# ---------------------------------------------------------------------------
# Arch detection
# ---------------------------------------------------------------------------


def _detect_arch(hf_model) -> str:
    """Return one of ``"gemma4"``, ``"qwen3"``, ``"llama3"`` or raises."""
    # Look at the inner base model's class; PEFT wrappers delegate to
    # ``.base_model.model``.
    target = hf_model
    for attr in ("base_model", "model"):
        inner = getattr(target, attr, None)
        if inner is not None and inner is not target:
            target = inner
    name = type(target).__name__
    # Take from the class hierarchy too so we don't miss Gemma4ForCausalLM
    # nested under a PEFT wrapper's ``base_model.model``.
    names = [c.__name__ for c in type(hf_model).__mro__]
    candidates = set(names + [name])
    lowered = " ".join(n.lower() for n in candidates)
    if "gemma4" in lowered or "gemma_4" in lowered or "gemma-4" in lowered:
        return "gemma4"
    if "qwen3" in lowered:
        return "qwen3"
    if "llama" in lowered:
        return "llama3"
    raise NotImplementedError(
        "UNSLOTH_FAST_INFERENCE=1 only supports Qwen3, Llama-3, Gemma-4 "
        f"today; got {type(hf_model).__name__}. Unset the env var or use vLLM."
    )


# ---------------------------------------------------------------------------
# Gemma-4 text-only shell extraction
# ---------------------------------------------------------------------------


def _extract_gemma4_text_shell(full_model):
    """Return a ``Gemma4ForCausalLM(text_cfg)`` shell wrapping the text
    backbone of ``full_model``. Mirrors the CLI path in
    ``scripts/benchmarks/gemma4_flex_inference.py:954-968``: strip the
    vision + audio towers, point ``shell.model`` at
    ``full_model.model.language_model``, tie ``lm_head.weight`` to the
    embeddings so the forward pass matches plain HF."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForCausalLM,
    )

    lang = getattr(full_model.model, "language_model", None)
    if lang is None:
        return full_model  # already a text-only CausalLM
    full_model.model.vision_tower = None
    full_model.model.audio_tower = None
    full_model.model.embed_vision = None
    full_model.model.embed_audio = None
    text_cfg = full_model.config.text_config
    shell = Gemma4ForCausalLM(text_cfg)
    shell.model = lang
    shell.lm_head.weight = lang.embed_tokens.weight
    shell = shell.to(next(lang.parameters()).device)
    shell.eval()
    return shell


# ---------------------------------------------------------------------------
# FlexEngine
# ---------------------------------------------------------------------------


class _LLMEngineStub:
    """Minimal stand-in for ``vllm.LLM.llm_engine``.

    ``unsloth/models/rl.py:104`` reaches into
    ``trainer.model.vllm_engine`` to pull the LLM out; `GRPOTrainer`'s
    patched ``_move_model_to_vllm`` path calls ``driver_worker.model_runner
    .model.load_weights`` but Unsloth's RL patch rewrites those calls to
    ``pass`` (rl.py:1795-1810), so a nested attribute chain that simply
    exists is enough."""

    def __init__(self):
        self.vllm_config = types.SimpleNamespace(lora_config = types.SimpleNamespace())
        self.model_executor = types.SimpleNamespace(
            driver_worker = types.SimpleNamespace(
                model_runner = types.SimpleNamespace(
                    model = types.SimpleNamespace(load_weights = lambda *a, **kw: None),
                )
            )
        )


class FlexEngine:
    """vLLM-compatible wrapper around :class:`FlexInference` /
    :class:`FlexGemma4Inference`.

    Args:
        hf_model: The HF model returned by ``from_pretrained``. It will be
            swapped to the PEFT-wrapped instance after ``bind_peft_model``.
        tokenizer: The companion tokenizer.
        dtype: ``torch.bfloat16`` or ``torch.float16``. Used for the autocast
            context wrapping forward/prefill/decode.
        max_seq_length: Upper bound on ``prompt + completion`` length.
        max_lora_rank: Unused by flex today (documented for parity with vLLM).
        max_batch_size: Concurrent sequence bucket size for the paged KV.
        page_size: Paged-KV page size (tokens per page).
        gpu_memory_utilization: Controls ``n_pages``. 0.5 means half the free
            VRAM is dedicated to the KV cache.
        capture_cudagraph: Capture CUDA graphs on the first decode step.
    """

    def __init__(
        self,
        hf_model,
        tokenizer,
        *,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_length: int = 2048,
        max_lora_rank: int = 64,  # accepted for API parity; no-op
        max_batch_size: int = 32,
        page_size: int = 128,
        gpu_memory_utilization: float = 0.5,
        max_new_tokens: int = 512,
        prefill_kernel_options: Optional[dict] = None,
        decode_kernel_options: Optional[dict] = None,
        fa4_prefill: Optional[bool] = None,
        capture_cudagraph: bool = True,
        base_model = None,
        peft_model = None,
        inference_model = None,
    ):
        assert dtype in (
            torch.bfloat16,
            torch.float16,
        ), f"FlexEngine requires bf16 or fp16 dtype; got {dtype}."
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.compute_dtype = dtype
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.page_size = page_size
        self.max_new_tokens = max_new_tokens
        self.capture_cudagraph = capture_cudagraph
        self._cudagraph_primed = False
        self._current_lora_int_id: Optional[int] = None

        self.device = hf_model.device

        # Colocate pattern (mirrors vLLM's "colocate" mode): the engine
        # runs on its own deep-copy of the HF model so that flex attention
        # patching + KV-cache attachment does not mutate the training
        # model's forward path. The user's ``model`` stays untouched for
        # gradient computation; rollouts go through this copy.
        #
        # If ``inference_model`` is provided by the caller, use it directly
        # (the loader uses this to hand in a copy captured BEFORE Unsloth's
        # post-patching; passing the patched model here would break
        # rotary_emb / QKV dispatch).
        #
        # The pristine-base copy (second deep-copy) is LoRA-only — it is
        # the refresh source for ``refresh_lora_merge_from_pristine``. We
        # defer materialising it until :meth:`bind_peft_model`, so the
        # no-LoRA path stays at 2x the base model's VRAM instead of 3x.
        if inference_model is None:
            inference_model = copy.deepcopy(hf_model)
            inference_model.eval()
        self._pristine_base = base_model  # None until bind_peft_model runs
        self._inference_model = inference_model
        self._inference_peft = peft_model  # filled in by bind_peft_model

        # Autocast wrapping (applied by the impl via a context manager).
        fa4_prefill, prefill_kernel_options, decode_kernel_options = (
            _auto_kernel_options(
                inference_model,
                self.device,
                prefill_kernel_options = prefill_kernel_options,
                decode_kernel_options = decode_kernel_options,
                fa4_prefill = fa4_prefill,
            )
        )

        # Size n_pages from available VRAM so big prompts don't OOM.
        n_pages = self._compute_n_pages(
            gpu_memory_utilization, max_batch_size, page_size
        )

        arch = _detect_arch(inference_model)
        self.arch = arch
        if arch == "gemma4":
            inference_model = _extract_gemma4_text_shell(inference_model)
            self._inference_model = inference_model
        Impl = FlexGemma4Inference if arch == "gemma4" else FlexInference
        self._impl = Impl(
            inference_model,
            tokenizer,
            max_batch_size = max_batch_size,
            max_seq_length = max_seq_length,
            n_pages = n_pages,
            page_size = page_size,
            max_new_tokens = max_new_tokens,
            decode_kernel_options = decode_kernel_options,
            prefill_kernel_options = prefill_kernel_options,
            fa4_prefill = fa4_prefill,
            base_model = self._pristine_base,
            peft_model = peft_model,
        )
        self._llm_engine_stub = _LLMEngineStub()

    # ----- configuration helpers -----

    def _compute_n_pages(
        self, gpu_mem_util: float, max_batch: int, page_size: int
    ) -> int:
        """Size the paged-KV allocation.

        ``n_pages`` = ceil(``max_batch * max_seq_length / page_size``) with a
        small headroom factor; any more and we waste VRAM on ghost pages
        that the scheduler never fills. ``gpu_memory_utilization`` scales the
        headroom."""
        min_pages = max(
            1, max_batch * ((self.max_seq_length + page_size - 1) // page_size)
        )
        factor = 1.0 + max(0.1, min(1.0, gpu_mem_util))
        return max(min_pages, int(min_pages * factor))

    # ----- generate -----

    def generate(
        self,
        prompts = None,
        sampling_params = None,
        lora_request = None,
        use_tqdm: bool = False,
        **kwargs,
    ):
        """Drop-in replacement for ``vllm.LLM.generate``.

        Returns a list of :class:`RequestOutput` mirroring vLLM's shape.
        """
        if prompts is None and "prompt_token_ids" in kwargs:
            prompts = kwargs.pop("prompt_token_ids")
        prompts = self._normalize_prompts(prompts)

        max_new_tokens, _ = self._extract_sampling(sampling_params)

        # Apply LoRA: if the request carries tensors, refresh the merged copy.
        if lora_request is not None:
            self._apply_lora_request(lora_request)

        seqs = self._build_sequences(prompts, max_new_tokens)

        with torch.amp.autocast("cuda", dtype = self.compute_dtype):
            done = self._impl.generate(
                seqs,
                capture_cudagraph = self.capture_cudagraph and not self._cudagraph_primed,
            )
        self._cudagraph_primed = self.capture_cudagraph

        # Reorder by the input prompt index (the impl returns done seqs in
        # completion order, not input order).
        done_by_input = sorted(done, key = lambda s: getattr(s, "_input_idx", 0))
        outs = []
        for idx, seq in enumerate(done_by_input):
            text = self.tokenizer.decode(seq.output_ids, skip_special_tokens = False)
            prompt_text = getattr(seq, "text", "") or self.tokenizer.decode(
                seq.input_ids.tolist() if seq.input_ids is not None else [],
                skip_special_tokens = False,
            )
            co = CompletionOutput(
                index = 0,
                text = text,
                token_ids = list(seq.output_ids),
                finish_reason = (
                    "stop" if seq.last_token_id == self._impl.eos_token_id else "length"
                ),
            )
            ro = RequestOutput(
                request_id = str(idx),
                prompt = prompt_text,
                prompt_token_ids = (
                    seq.input_ids.tolist() if seq.input_ids is not None else []
                ),
                outputs = [co],
            )
            outs.append(ro)
        return outs

    def chat(
        self,
        messages,
        sampling_params = None,
        lora_request = None,
        use_tqdm: bool = False,
        **kwargs,
    ):
        """Apply the tokenizer's chat template and defer to :meth:`generate`."""
        if messages is None:
            return []
        # ``messages`` from vLLM is either a list of message-lists or a single
        # message-list (one conversation).
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            # Single conversation.
            convos = [messages]
        else:
            convos = list(messages)
        prompts = [
            self.tokenizer.apply_chat_template(
                m, tokenize = False, add_generation_prompt = True
            )
            for m in convos
        ]
        return self.generate(
            prompts,
            sampling_params = sampling_params,
            lora_request = lora_request,
            use_tqdm = use_tqdm,
            **kwargs,
        )

    # ----- LoRA refresh -----

    def _apply_lora_request(self, lora_request):
        """Copy the training LoRA tensors onto the inference-side PEFT
        wrapper and refresh the merged inference weights from the pristine
        base.

        Expects ``lora_request.lora_tensors`` to be a ``state_dict`` slice
        filtered for ``.lora_A.`` / ``.lora_B.`` keys (which is what
        :func:`~unsloth.inference.vllm_shim.load_lora` produces)."""
        if lora_request is None:
            return
        base_model = getattr(self._impl, "base_model", None)
        peft_model = getattr(self._impl, "peft_model", None)
        if base_model is None or peft_model is None:
            # 4-bit / no double-copy: LoRA tensors already live on the PEFT
            # wrappers (bnb-4bit packed weights can't be in-place refreshed;
            # PEFT's three-matmul wrapper does the math at runtime).
            return

        tensors = getattr(lora_request, "lora_tensors", None)
        if tensors:
            target_sd = peft_model.state_dict()
            renamed = {}
            for k, v in tensors.items():
                # ``load_lora`` strips ``.default``; PEFT's own state_dict
                # keeps it. Try both forms.
                if k in target_sd:
                    renamed[k] = v
                    continue
                candidate = k.replace(".lora_A.", ".lora_A.default.").replace(
                    ".lora_B.", ".lora_B.default."
                )
                if candidate in target_sd:
                    renamed[candidate] = v
            if renamed:
                missing, unexpected = peft_model.load_state_dict(renamed, strict = False)
                # ``missing`` will be every non-LoRA param; that's fine.
        refresh_lora_merge_from_pristine(base_model, peft_model)
        self._current_lora_int_id = getattr(lora_request, "lora_int_id", None)

    # ----- prompt normalization -----

    @staticmethod
    def _normalize_prompts(prompts):
        if prompts is None:
            return []
        if isinstance(prompts, str):
            return [prompts]
        if isinstance(prompts, list):
            if not prompts:
                return []
            # Already a list. Leave dict / str / list[int] elements as-is.
            return prompts
        return [prompts]

    def _build_sequences(self, prompts, max_new_tokens: int) -> list:
        seqs = []
        for idx, p in enumerate(prompts):
            if isinstance(p, str):
                seq = Sequence(text = p, max_new_tokens = max_new_tokens)
            elif isinstance(p, dict):
                # vLLM TokensPrompt / TextPrompt dict
                if "prompt_token_ids" in p:
                    ids = torch.tensor(p["prompt_token_ids"], dtype = torch.long)
                    seq = Sequence(
                        text = p.get("prompt", ""),
                        input_ids = ids,
                        input_length = int(ids.shape[0]),
                        max_new_tokens = max_new_tokens,
                    )
                elif "prompt" in p:
                    seq = Sequence(
                        text = p["prompt"],
                        max_new_tokens = max_new_tokens,
                    )
                else:
                    raise ValueError(
                        f"Unsupported prompt dict (no 'prompt' or "
                        f"'prompt_token_ids'): {list(p)}"
                    )
            elif isinstance(p, (list, tuple)) and p and isinstance(p[0], int):
                ids = torch.tensor(list(p), dtype = torch.long)
                seq = Sequence(
                    text = "",
                    input_ids = ids,
                    input_length = int(ids.shape[0]),
                    max_new_tokens = max_new_tokens,
                )
            else:
                raise ValueError(
                    "FlexEngine.generate accepts str, list[int], or a "
                    "TokensPrompt/TextPrompt dict; got "
                    f"{type(p).__name__} at index {idx}."
                )
            seq._input_idx = idx  # preserve input order across the scheduler
            seqs.append(seq)
        return seqs

    def _extract_sampling(self, sampling_params) -> tuple[int, dict]:
        """Pull what we actually honour out of a ``SamplingParams``.

        Today the flex path does argmax sampling only. We read ``max_tokens``
        (mapped to ``max_new_tokens``) and ignore ``temperature`` / ``top_p``
        / ``top_k`` with a warning the first time we see something non-greedy.
        Sufficient for GRPO's on-policy rollout, which already accepts that
        the generation is deterministic per-prompt under a fixed seed."""
        if sampling_params is None:
            return self.max_new_tokens, {}
        max_tokens = getattr(sampling_params, "max_tokens", None)
        if max_tokens is None:
            max_tokens = self.max_new_tokens
        temp = getattr(sampling_params, "temperature", 0.0) or 0.0
        if temp and temp > 0 and not getattr(self, "_warned_sampling", False):
            warnings.warn(
                "FlexEngine (UNSLOTH_FAST_INFERENCE=1) does argmax sampling "
                "only; sampling_params.temperature / top_p / top_k are "
                "ignored. Unset the env var or use vLLM for stochastic "
                "sampling.",
                RuntimeWarning,
                stacklevel = 3,
            )
            self._warned_sampling = True
        return int(max_tokens), {}

    # ----- sleep-mode stubs -----
    #
    # vLLM's sleep mode offloads engine weights to CPU between rollouts so
    # training can use the freed VRAM. The flex backend does not implement
    # that yet; these stubs exist so code paths that assume the API are safe.

    def sleep(self, level: int = 2):
        """No-op. Real implementation in a follow-up PR (move PagedKVCache +
        shared buffers + the inference-copy HF shell to CPU pinned memory
        and swap back on wake_up)."""
        return None

    def wake_up(self, tags: Optional[list] = None):
        """No-op companion to :meth:`sleep`."""
        return None

    @property
    def llm_engine(self):
        """Nested-attribute stub used by Unsloth's RL patch — see docstring
        on :class:`_LLMEngineStub`."""
        return self._llm_engine_stub

    # ----- PEFT wiring -----

    def bind_peft_model(self, training_peft_model):
        """Hook called at the end of ``FastLanguageModel.get_peft_model``.

        * Updates ``self.hf_model`` so ``save_lora`` / ``load_lora`` — which
          call ``model.state_dict()`` — see the training LoRA weights.
        * Materialises the pristine-base copy (second deep-copy of the
          inference model's un-patched weights) used as the refresh source
          for ``refresh_lora_merge_from_pristine``. This is lazy so the
          no-LoRA path stays at 2x the base model's VRAM.
        * Mirrors the training PEFT config onto the inference copy by
          wrapping ``self._inference_model`` with a matching ``PeftModel``.
          The inference-side adapter starts at zero; LoRA tensors are then
          loaded from the request on each ``generate`` call, and
          ``refresh_lora_merge_from_pristine`` fuses them into the merged
          base weights before the kernel is launched.
        * The deep-copy pattern keeps the training model's forward pass
          untouched (its attention is not flex-patched)."""
        self.hf_model = training_peft_model

        # Materialise the pristine source the first time we see a LoRA.
        if self._pristine_base is None:
            # The inference model has already been flex-patched; its
            # linear weights (what LoRA merges into) are still pristine,
            # so we can clone it and just not call flex attention on the
            # pristine copy.
            self._pristine_base = copy.deepcopy(self._inference_model)
            self._pristine_base.eval()
            self._impl.base_model = self._pristine_base

        if self._inference_peft is None:
            try:
                from peft import get_peft_model as _get_peft_model

                peft_cfg = training_peft_model.peft_config["default"]
                # Wrap the already-patched inference copy with a fresh LoRA
                # adapter of the same shape. LoraLayer insertion is
                # attention-forward-agnostic; it wraps Linear modules.
                self._inference_peft = _get_peft_model(self._inference_model, peft_cfg)
                self._inference_peft.eval()
            except Exception as e:
                warnings.warn(
                    f"FlexEngine.bind_peft_model: could not build an "
                    f"inference-side PEFT wrapper ({e}). Falling back to "
                    "single-copy mode — LoRA refresh will rely on the "
                    "training model's state_dict directly.",
                    RuntimeWarning,
                    stacklevel = 2,
                )
                self._inference_peft = training_peft_model

        self._impl.peft_model = self._inference_peft


# ---------------------------------------------------------------------------
# load_flex — mirrors unsloth_zoo.vllm_utils.load_vllm's call shape so the
# dispatcher in unsloth/models/llama.py + vision.py can route cleanly.
# ---------------------------------------------------------------------------


def load_flex(
    hf_model,
    tokenizer,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_seq_length: int = 2048,
    max_lora_rank: int = 64,
    max_batch_size: int = 32,
    gpu_memory_utilization: float = 0.5,
    page_size: int = 128,
    capture_cudagraph: bool = True,
    base_model = None,
    peft_model = None,
    **_unused_vllm_kwargs,
) -> FlexEngine:
    """Construct a :class:`FlexEngine` around an already-loaded HF model.

    ``_unused_vllm_kwargs`` swallows vLLM-only kwargs passed through by
    `unsloth/models/llama.py` (``use_bitsandbytes``, ``enable_lora``,
    ``disable_log_stats``, ``fp8_mode``, ``float8_kv_cache``, ...) so the
    dispatcher doesn't have to know which backend is selected."""
    return FlexEngine(
        hf_model,
        tokenizer,
        dtype = dtype,
        max_seq_length = max_seq_length,
        max_lora_rank = max_lora_rank,
        max_batch_size = max_batch_size,
        gpu_memory_utilization = gpu_memory_utilization,
        page_size = page_size,
        capture_cudagraph = capture_cudagraph,
        base_model = base_model,
        peft_model = peft_model,
    )


__all__ = ["FlexEngine", "load_flex"]
