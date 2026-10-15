# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speculative-decoding drafter trained after the fine-tune.

Models with a native MTP head retrain it (llama.cpp `draft-mtp`, vLLM `method="mtp"`); models without one get
an EAGLE-3 drafter in the speculators layout (llama.cpp `draft-eagle3`, vLLM `method="eagle3"`). A fresh MTP
head on an architecture without an MTP graph would run in neither engine, so it is never offered.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import time
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

DRAFTER_MANIFEST = "drafter.json"
EAGLE3_DIRNAME = "eagle3"
MTP_HEAD_FILE = "mtp_head.safetensors"
DEFAULT_LR = {"mtp": 1e-4, "eagle3": 3e-4}

# Families vLLM serves with its MTP proposer (vllm/config/speculative.py, 0.31.0).
_VLLM_MTP_FAMILIES = {
    "qwen3_5",
    "qwen3_next",
    "glm4_moe",
    "glm4_moe_lite",
    "deepseek_v3",
    "deepseek_v32",
    "glm_moe_dsa",
    "hy_v3",
    "nemotron_h",
    "step3p5",
    "exaone_moe",
}
# EAGLE-3 targets: vLLM `SupportsEagle3` models, and llama.cpp archs that expose layer inputs
# (`t_layer_inp`; gemma3 / mistral3 do not). Used only when unsloth_zoo has no eagle3_supported().
_EAGLE3_VLLM = {
    "llama",
    "mistral",
    "mistral3",
    "qwen2",
    "qwen3",
    "qwen3_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_next",
    "gemma4_text",
    "gpt_oss",
    "minimax_m2",
    "nemotron_h",
}
_EAGLE3_LLAMA_CPP = {
    "llama",
    "mistral",
    "qwen3",
    "qwen3_moe",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_next",
    "gemma4_text",
    "gemma3n_text",
    "gpt_oss",
    "minimax_m2",
    "lfm2",
    "nemotron_h",
}
_QUANT_SUFFIX = re.compile(
    r"-(unsloth-)?bnb-4bit$|-(unsloth-)?(fp8|awq|gptq)(-int4)?$", re.IGNORECASE
)


def _text_config(model_name: str, hf_token: Optional[str] = None):
    from transformers import AutoConfig

    source = model_name
    adapter_config = os.path.join(model_name, "adapter_config.json")
    if os.path.isfile(adapter_config):
        with open(adapter_config, encoding = "utf-8") as f:
            source = json.load(f).get("base_model_name_or_path") or model_name
    config = AutoConfig.from_pretrained(source, token = hf_token)
    return config.get_text_config() if hasattr(config, "get_text_config") else config, source


def _mtp_source_candidates(source: str) -> list[str]:
    """The checkpoint itself, then the original repo a quantized re-upload came from."""
    candidates = [source]
    stripped = _QUANT_SUFFIX.sub("", source)
    if stripped != source:
        candidates.append(stripped)
    return candidates


def _find_mtp_source(source: str, config) -> Optional[str]:
    from unsloth_zoo.speculative.mtp import find_mtp_tensor_names
    for candidate in _mtp_source_candidates(source):
        try:
            if find_mtp_tensor_names(candidate, config = config):
                return candidate
        except Exception as exc:  # unreadable index, offline, gated
            logger.debug("No MTP tensors readable from %s: %s", candidate, exc)
    return None


def _mtp_family(config):
    try:
        from unsloth_zoo.speculative.mtp import mtp_family_name
    except ImportError:
        return None, None
    name = mtp_family_name(config)
    if name is None:
        return None, None
    status = "drafts"
    if name != "qwen3_5":
        from unsloth_zoo.speculative.mtp_families import MTP_FAMILIES
        family = MTP_FAMILIES.get(name)
        status = getattr(family, "llama_cpp", "unverified")
    return name, status


def eagle3_support(config) -> tuple[bool, bool, Optional[str]]:
    """(vllm, llama_cpp, reason) for an EAGLE-3 drafter on this target."""
    try:
        from unsloth_zoo.speculative import eagle3_supported
    except ImportError:
        eagle3_supported = None
    model_type = getattr(config, "model_type", None)
    vllm = model_type in _EAGLE3_VLLM
    llama_cpp = model_type in _EAGLE3_LLAMA_CPP
    if eagle3_supported is not None:
        ok, reason = eagle3_supported(config)
        if not ok:
            return False, False, reason
    if not (vllm or llama_cpp):
        return False, False, f"`{model_type}` has no EAGLE-3 drafter support in vLLM or llama.cpp."
    return vllm, llama_cpp, None


_CAPABILITY_CACHE: dict = {}
_CAPABILITY_TTL_S = 600
_TOKEN_ENV = "UNSLOTH_DRAFTER_PROBE_TOKEN"
_PROBE = """
import json, os, sys
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
sys.path.insert(0, os.getcwd())
token = os.environ.pop({env!r}, "")
token = False if token == "0" else (token or None)
from core.training.drafter import capability_in_process
print("DRAFTER_CAPABILITY " + json.dumps(capability_in_process(sys.argv[1], token)))
"""


def drafter_capability(model_name: str, hf_token = None) -> dict:
    """The capability, probed in a child process: importing unsloth_zoo in the API server races its
    background torch warm-up (import deadlock in torch._dynamo) and patches the server process."""
    import subprocess
    import sys

    key = (model_name, hf_token is False, bool(hf_token))
    hit = _CAPABILITY_CACHE.get(key)
    if hit and time.time() - hit[0] < _CAPABILITY_TTL_S:
        return dict(hit[1])
    env = dict(os.environ)
    env[_TOKEN_ENV] = "0" if hf_token is False else (hf_token or "")
    result = None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE.format(env = _TOKEN_ENV), model_name],
            cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            env = env,
            capture_output = True,
            text = True,
            timeout = 300,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("DRAFTER_CAPABILITY "):
                result = json.loads(line[len("DRAFTER_CAPABILITY ") :])
        if result is None:
            logger.warning("Drafter capability probe failed: %s", (proc.stderr or "")[-2000:])
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Drafter capability probe failed: %r", exc)
    if result is None:
        return {
            "model_name": model_name,
            "has_native_mtp": False,
            "strategy": None,
            "supported": False,
            "reason": "Could not check this model for drafter support.",
            "vllm": False,
            "llama_cpp": False,
        }
    _CAPABILITY_CACHE[key] = (time.time(), result)
    return dict(result)


def capability_in_process(model_name: str, hf_token: Optional[str] = None) -> dict:
    """What the Train tab can offer for this model. Head presence comes from the checkpoint's tensor index."""
    result = {
        "model_name": model_name,
        "has_native_mtp": False,
        "strategy": None,
        "supported": False,
        "reason": None,
        "vllm": False,
        "llama_cpp": False,
    }
    try:
        import unsloth_zoo.speculative.eagle3  # noqa: F401
        import unsloth_zoo.speculative.mtp  # noqa: F401
    except ImportError as exc:
        # An older unsloth_zoo would also misread a native-MTP model as EAGLE-3-only.
        logger.warning("Drafter trainers unavailable: %r", exc)
        result["reason"] = (
            f"Drafter training needs a newer unsloth_zoo (pip install -U unsloth_zoo): {exc}"
        )[:400]
        return result
    try:
        config, source = _text_config(model_name, hf_token)
    except Exception as exc:
        result["reason"] = f"Could not read the model config: {exc}"
        return result
    family, status = _mtp_family(config)
    mtp_source = _find_mtp_source(source, config) if family is not None else None
    if mtp_source is not None:
        result.update(
            has_native_mtp = True,
            strategy = "mtp",
            supported = True,
            vllm = family in _VLLM_MTP_FAMILIES,
            llama_cpp = status in ("drafts", "partial"),
        )
        if not result["llama_cpp"]:
            result["reason"] = "llama.cpp cannot draft with this model's MTP head yet; vLLM can."
        return result
    vllm, llama_cpp, reason = eagle3_support(config)
    if vllm or llama_cpp:
        result.update(strategy = "eagle3", supported = True, vllm = vllm, llama_cpp = llama_cpp)
        if not llama_cpp:
            result["reason"] = "llama.cpp cannot run an EAGLE-3 drafter for this model; vLLM can."
        elif not vllm:
            result["reason"] = "vLLM cannot run an EAGLE-3 drafter for this model; llama.cpp can."
        return result
    result["reason"] = reason
    return result


def resolve_strategy(requested: str, capability: dict) -> tuple[Optional[str], Optional[str]]:
    """(strategy, refusal reason)."""
    if not capability.get("supported"):
        return None, capability.get("reason") or "No drafter is available for this model."
    if requested in (None, "auto"):
        return capability["strategy"], None
    if requested == "mtp" and not capability.get("has_native_mtp"):
        return None, "This model has no native MTP head; use EAGLE-3."
    if requested == "eagle3" and capability.get("strategy") != "eagle3":
        # A native-MTP model can still take an EAGLE-3 drafter when the target supports it.
        return None, "This model trains its native MTP head; EAGLE-3 is for models without one."
    return requested, None


# ---------------------------------------------------------------- examples


def _row_examples(row: dict, max_len: int) -> list[tuple[list[int], list[int]]]:
    ids = list(row["input_ids"])
    if "labels" in row and row["labels"] is not None:
        labels = list(row["labels"])
    elif row.get("completion_mask") is not None:
        labels = [t if m else -100 for t, m in zip(ids, row["completion_mask"])]
    elif row.get("assistant_masks") is not None:
        labels = [t if m else -100 for t, m in zip(ids, row["assistant_masks"])]
    else:
        labels = list(ids)
    # Packed rows restart position_ids at every example; split there so no draft crosses a boundary.
    positions = row.get("position_ids")
    cuts = [0]
    if positions is not None:
        cuts = [i for i, p in enumerate(positions) if p == 0] or [0]
    cuts.append(len(ids))
    out = []
    for start, end in zip(cuts, cuts[1:]):
        seg_ids, seg_labels = ids[start:end][:max_len], labels[start:end][:max_len]
        if len(seg_ids) >= 4 and any(t != -100 for t in seg_labels):
            out.append((seg_ids, seg_labels))
    return out


def collect_examples(train_dataset, max_examples: int, max_len: int) -> list[dict]:
    examples = []
    for row in train_dataset:
        for ids, labels in _row_examples(row, max_len):
            examples.append({"input_ids": ids, "labels": labels})
            if len(examples) >= max_examples:
                return examples
    return examples


def _assistant_header(tokenizer) -> list[int]:
    """Token ids the chat template puts between a user turn and the answer ("<|im_start|>assistant\n", ...)."""
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    turn = [{"role": "user", "content": "x"}]
    try:
        with_prompt = tokenizer.apply_chat_template(
            turn, add_generation_prompt = True, tokenize = False
        )
        without = tokenizer.apply_chat_template(turn, add_generation_prompt = False, tokenize = False)
    except Exception:
        return []
    if not with_prompt.startswith(without):
        return []
    return text_tokenizer(with_prompt[len(without) :], add_special_tokens = False)["input_ids"]


def _response_markers(tokenizer) -> list[list[int]]:
    """Token ids that can open an answer: the chat template's assistant header, then every response
    marker Studio's completion masking knows (rows rendered with a dataset template such as Alpaca's
    "### Response:" carry no chat header)."""
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    markers = []
    header = _assistant_header(tokenizer)
    if header:
        markers.append(header)
    try:
        from utils.datasets.model_mappings import TEMPLATE_TO_RESPONSES_MAPPER
        texts = {
            m.get("response") for m in TEMPLATE_TO_RESPONSES_MAPPER.values() if isinstance(m, dict)
        }
    except Exception:
        texts = set()
    for text in sorted(t for t in texts if isinstance(t, str) and t.strip()):
        try:
            ids = text_tokenizer(text, add_special_tokens = False)["input_ids"]
        except Exception:
            continue
        if ids and ids not in markers:
            markers.append(ids)
    return markers


def _marker_end(ids: list[int], marker: list[int]) -> Optional[int]:
    """End of the marker's last occurrence; shorter prefixes cover a marker whose tail tokenizes into
    the answer, or a template that adds a thinking block after the header."""
    for width in range(len(marker), 0, -1):
        part = marker[:width]
        for start in range(len(ids) - width, 0, -1):
            if ids[start : start + width] == part:
                return start + width
        if width <= 2:
            break
    return None


def _prompt_end(ids: list[int], labels: list[int], markers) -> Optional[int]:
    """Where the answer starts: the first trained label, or (rows trained on the whole sequence) after
    the last response marker found."""
    first = next((i for i, t in enumerate(labels) if t != -100), None)
    if first:
        return first
    if markers and isinstance(markers[0], int):
        markers = [markers]
    ends = [e for e in (_marker_end(ids, m) for m in markers or []) if e and e < len(ids)]
    return max(ends) if ends else None


def regenerate_examples(
    model,
    tokenizer,
    examples: list[dict],
    max_new_tokens: int = 512,
    batch_size: int = 8,
    should_stop: Callable[[], bool] = lambda: False,
) -> list[dict]:
    """The fine-tuned model's own greedy answers to each example's prompt (the drafter imitates the model)."""
    import torch

    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = text_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = text_tokenizer.eos_token_id
    eos_id = text_tokenizer.eos_token_id
    markers = _response_markers(tokenizer)
    prompts = []
    for ex in examples:
        end = _prompt_end(ex["input_ids"], ex["labels"], markers)
        if end:  # needs a prompt to answer
            prompts.append(ex["input_ids"][:end])
    out = []
    device = next(model.parameters()).device
    for b in range(0, len(prompts), batch_size):
        if should_stop():
            break
        batch = prompts[b : b + batch_size]
        width = max(map(len, batch))
        ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in batch], device = device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch], device = device)
        with torch.no_grad():
            gen = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = pad_id,
            )
        for prompt, row in zip(batch, gen[:, width:].tolist()):
            if eos_id in row:
                row = row[: row.index(eos_id) + 1]
            row = [t for t in row if t != pad_id or t == eos_id]
            if row:
                out.append({"input_ids": prompt + row, "labels": [-100] * len(prompt) + row})
    return out


# ---------------------------------------------------------------- phase


def _progress_callback(event_queue, strategy: str, should_stop: Callable[[], bool]):
    from transformers import TrainerCallback
    class _DrafterProgress(TrainerCallback):
        def _emit(
            self,
            state,
            logs = None,
        ):
            logs = logs or {}
            loss = logs.get("loss", logs.get("mtp_loss_0"))
            top1 = logs.get(
                "mtp_top1_0", logs.get("eagle3_acc_0", logs.get("top1", logs.get("acc_0")))
            )
            event_queue.put(
                {
                    "type": "drafter_progress",
                    "strategy": strategy,
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    "loss": loss,
                    "top1": top1,
                    "state": "training",
                    "ts": time.time(),
                }
            )
            event_queue.put(
                {
                    "type": "status",
                    "ts": time.time(),
                    "message": f"Training drafter ({_LABEL[strategy]}): step {state.global_step}/{state.max_steps}",
                }
            )

        def on_train_begin(self, args, state, control, **kwargs):
            self._emit(state)

        def on_log(
            self,
            args,
            state,
            control,
            logs = None,
            **kwargs,
        ):
            self._emit(state, logs)

        def on_step_end(self, args, state, control, **kwargs):
            if should_stop():
                control.should_training_stop = True

    return _DrafterProgress()


_LABEL = {"mtp": "MTP head", "eagle3": "EAGLE-3"}


def _training_args(cls, output_dir: str, cfg: dict, strategy: str, seed: int, extra: dict):
    import torch
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return cls(
        output_dir = output_dir,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1,
        num_train_epochs = float(cfg.get("epochs") or 1.0),
        learning_rate = float(cfg.get("learning_rate") or DEFAULT_LR[strategy]),
        lr_scheduler_type = "cosine",
        warmup_steps = 10,
        logging_steps = 5,
        save_strategy = "no",
        bf16 = bf16,
        fp16 = not bf16 and torch.cuda.is_available(),
        report_to = "none",
        seed = seed,
        weight_decay = 0.0,
        **extra,
    )


def _free_sft_trainer(trainer) -> Any:
    """Drop the SFT optimizer / scheduler / dataloaders; return the train dataset."""
    sft = getattr(trainer, "trainer", None)
    dataset = getattr(sft, "train_dataset", None)
    if sft is not None:
        for attr in ("optimizer", "lr_scheduler", "train_dataset", "eval_dataset"):
            try:
                setattr(sft, attr, None)
            except Exception:
                pass
        trainer.trainer = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return dataset


def write_manifest(output_dir: str, manifest: dict) -> str:
    path = os.path.join(output_dir, DRAFTER_MANIFEST)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding = "utf-8") as f:
        json.dump(manifest, f, indent = 1)
    os.replace(tmp, path)
    return path


def read_manifest(directory: Optional[str]) -> Optional[dict]:
    if not directory:
        return None
    path = os.path.join(directory, DRAFTER_MANIFEST)
    try:
        with open(path, encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("strategy") in ("mtp", "eagle3") else None


def run_drafter_phase(
    trainer, config: dict, output_dir: str, event_queue, send_status: Callable[[Any, str], None]
) -> Optional[dict]:
    """Train the drafter after a successful fine-tune. Never raises: a failure is a warning and the
    fine-tune stays saved. Returns the manifest written, or None."""
    cfg = config.get("drafter") or {}
    if not cfg.get("enabled"):
        return None
    should_stop = lambda: bool(getattr(trainer, "should_stop", False))
    strategy = None
    eagle_dir = os.path.join(output_dir, EAGLE3_DIRNAME)
    try:
        if getattr(trainer, "is_vlm", False) or getattr(trainer, "is_audio_vlm", False):
            raise RuntimeError(
                "drafter training is not available for vision or audio fine-tunes yet"
            )
        model, tokenizer = trainer.model, trainer.tokenizer
        model_name = config.get("model_name") or getattr(trainer, "model_name", "")
        text_config = _model_text_config(model)
        family, status = _mtp_family(text_config)
        capability = (
            capability_in_process(model_name, config.get("hf_token") or None) if model_name else {}
        )
        strategy, refusal = resolve_strategy(cfg.get("strategy", "auto"), capability)
        if strategy is None:
            raise RuntimeError(refusal)
        send_status(event_queue, f"Training drafter ({_LABEL[strategy]}): preparing examples")
        max_len = int(config.get("max_seq_length") or 2048)
        dataset = _free_sft_trainer(trainer)
        if dataset is None:
            raise RuntimeError("the fine-tune's training examples are not available")
        examples = collect_examples(dataset, int(cfg.get("max_examples") or 2000), max_len)
        del dataset
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        data_source = "dataset"
        if cfg.get("data_source") == "regenerate":
            markers = _response_markers(tokenizer)
            found = sum(
                bool(_prompt_end(ex["input_ids"], ex["labels"], markers)) for ex in examples
            )
            if not found and examples:
                logger.warning(
                    "Drafter regenerate found no prompts: markers=%s first row ids=%s labels=%s",
                    markers,
                    examples[0]["input_ids"][:48],
                    examples[0]["labels"][:48],
                )
            send_status(
                event_queue,
                f"Training drafter ({_LABEL[strategy]}): generating the model's answers to {found} prompts",
            )
            regenerated = regenerate_examples(model, tokenizer, examples, should_stop = should_stop)
            if regenerated:
                examples = regenerated
                data_source = "regenerate"
            else:
                send_status(
                    event_queue,
                    f"Training drafter ({_LABEL[strategy]}): no prompts found to answer, using the dataset's answers",
                )
        if not examples:
            raise RuntimeError("no usable training examples")
        if should_stop():
            return None
        from datasets import Dataset

        train_dataset = Dataset.from_list(examples)
        seed = int(config.get("random_seed") or 3407)
        steps = int(cfg.get("draft_steps") or 3)
        ckpt_dir = os.path.join(output_dir, "drafter_tmp")
        callback = _progress_callback(event_queue, strategy, should_stop)
        if strategy == "mtp":
            from unsloth import FastModel, MTPTrainingArguments, UnslothMTPTrainer

            source = _find_mtp_source(capability_source(model_name, config), text_config)
            head = FastModel.get_mtp_head(model, source = source, random_init = source is None)
            args = _training_args(
                MTPTrainingArguments, ckpt_dir, cfg, strategy, seed, {"mtp_num_steps": steps}
            )
            drafter_trainer = UnslothMTPTrainer(
                model = head,
                target_model = model,
                args = args,
                train_dataset = train_dataset,
                processing_class = tokenizer,
            )
        else:
            from unsloth_zoo.speculative import (
                Eagle3TrainingArguments,
                UnslothEagle3Trainer,
                build_eagle3_drafter,
                build_eagle3_vocab_map,
            )

            vocab = int(getattr(text_config, "vocab_size"))
            d2t, t2d = build_eagle3_vocab_map(
                [e["input_ids"] for e in examples], vocab, draft_vocab_size = min(32000, vocab)
            )
            head = build_eagle3_drafter(model, d2t, t2d)
            args = _training_args(Eagle3TrainingArguments, ckpt_dir, cfg, strategy, seed, {})
            drafter_trainer = UnslothEagle3Trainer(
                model = head,
                target_model = model,
                args = args,
                train_dataset = train_dataset,
                processing_class = tokenizer,
            )
        drafter_trainer.add_callback(callback)
        result = drafter_trainer.train()
        shutil.rmtree(ckpt_dir, ignore_errors = True)
        if should_stop():
            # Stop during this phase keeps the fine-tune and drops only the drafter.
            _discard(output_dir, strategy, head = head, model = model)
            return None
        send_status(event_queue, f"Training drafter ({_LABEL[strategy]}): saving")
        metrics = {
            k: v
            for k, v in (getattr(result, "metrics", None) or {}).items()
            if isinstance(v, (int, float))
        }
        history = [h for h in drafter_trainer.state.log_history if isinstance(h, dict)]
        for entry in reversed(history):
            for key, value in entry.items():
                if (
                    key.startswith(
                        ("mtp_top1_", "mtp_loss_", "eagle3_acc_", "eagle3_loss_", "acc_", "top1")
                    )
                    and key not in metrics
                ):
                    metrics[key] = value
        base_model = capability_source(model_name, config)
        manifest = {
            "version": 1,
            "strategy": strategy,
            "base_model": base_model,
            "adapter_dir": output_dir,
            "draft_steps": steps,
            "examples": len(examples),
            "data_source": data_source,
            "metrics": metrics,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if strategy == "mtp":
            from unsloth.save import _save_mtp_head_beside_adapter, _stitch_mtp_head
            if os.path.isfile(os.path.join(output_dir, "adapter_config.json")):
                _save_mtp_head_beside_adapter(model, output_dir)
                manifest["files"] = [MTP_HEAD_FILE]
            else:
                _stitch_mtp_head(model, output_dir)
                manifest["files"] = []
            manifest.update(
                family = family,
                llama_cpp_status = status,
                vllm = {"method": "mtp", "num_speculative_tokens": steps},
                llama_cpp = {"spec_type": "draft-mtp", "spec_draft_n_max": steps},
            )
        else:
            from unsloth_zoo.speculative import save_eagle3_drafter

            shutil.rmtree(eagle_dir, ignore_errors = True)
            save_eagle3_drafter(
                eagle_dir, head, verifier_name_or_path = base_model, speculative_tokens = steps
            )
            manifest["files"] = [EAGLE3_DIRNAME]
            manifest["aux_layer_ids"] = list(getattr(head, "aux_layer_ids", None) or [])
            manifest.update(
                vllm = {"method": "eagle3", "num_speculative_tokens": steps},
                llama_cpp = {"spec_type": "draft-eagle3", "spec_draft_n_max": steps},
            )
        write_manifest(output_dir, manifest)
        event_queue.put(
            {
                "type": "drafter_progress",
                "strategy": strategy,
                "step": None,
                "total_steps": None,
                "loss": metrics.get("train_loss"),
                "top1": metrics.get("mtp_top1_0", metrics.get("eagle3_acc_0")),
                "state": "done",
                "ts": time.time(),
            }
        )
        return manifest
    except Exception as exc:
        logger.warning("Drafter training failed: %s", exc, exc_info = True)
        _discard(output_dir, strategy)
        event_queue.put(
            {
                "type": "warning",
                "ts": time.time(),
                "message": f"Drafter training failed, the fine-tuned model is saved without it: {exc}",
            }
        )
        event_queue.put(
            {
                "type": "drafter_progress",
                "strategy": strategy,
                "step": None,
                "total_steps": None,
                "loss": None,
                "top1": None,
                "state": "failed",
                "ts": time.time(),
            }
        )
        return None
    finally:
        gc.collect()


def capability_source(model_name: str, config: dict) -> str:
    """The base checkpoint id (an adapter folder resolves to its base)."""
    adapter_config = os.path.join(model_name or "", "adapter_config.json")
    if model_name and os.path.isfile(adapter_config):
        try:
            with open(adapter_config, encoding = "utf-8") as f:
                return json.load(f).get("base_model_name_or_path") or model_name
        except (OSError, ValueError):
            pass
    return model_name


def _model_text_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return None
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _discard(
    output_dir: str,
    strategy: Optional[str],
    head = None,
    model = None,
) -> None:
    """Remove anything this phase may have written; the fine-tune itself is untouched."""
    if model is not None:
        try:
            object.__delattr__(model, "_unsloth_mtp_head")
        except AttributeError:
            pass
    for name in (EAGLE3_DIRNAME, "drafter_tmp", DRAFTER_MANIFEST, MTP_HEAD_FILE):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors = True)
        elif os.path.isfile(path):
            os.remove(path)
