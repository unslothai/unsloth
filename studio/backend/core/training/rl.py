# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DPO, ORPO and GRPO for Studio training: dataset shaping and trainer construction.

GRPO runs without vLLM: Unsloth only switches TRL to vLLM rollouts when the model carries a
``vllm_engine``, which Studio never loads, so generation goes through TRL's own path.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any, Callable, Optional

OBJECTIVES = ("sft", "dpo", "orpo", "grpo")
PREFERENCE_OBJECTIVES = ("dpo", "orpo")
RL_ROLES = {
    "dpo": ("prompt", "chosen", "rejected", "system"),
    "orpo": ("prompt", "chosen", "rejected", "system"),
    "grpo": ("prompt", "answer", "system"),
}
_REQUIRED_ROLES = {
    "dpo": ("prompt", "chosen", "rejected"),
    "orpo": ("prompt", "chosen", "rejected"),
    "grpo": ("prompt",),
}
_AUTO_ROLE_NAMES = {
    "prompt": ("prompt", "question", "instruction", "problem", "query", "input"),
    "answer": ("answer", "solution", "final_answer", "target", "label"),
    "chosen": ("chosen", "accepted", "preferred"),
    "rejected": ("rejected", "dispreferred"),
    "system": ("system", "system_prompt"),
}
DEFAULT_BETA = {"dpo": 0.1, "orpo": 0.1, "grpo": 0.0}
# variant -> (loss_type, importance_sampling_level). dapo is TRL's and Unsloth's default; gspo is
# sequence-level ratios with the dr_grpo normaliser, as in the Unsloth vision GRPO notebooks.
GRPO_VARIANTS = {
    "dapo": ("dapo", "token"),
    "dr_grpo": ("dr_grpo", "token"),
    "bnpo": ("bnpo", "token"),
    "grpo": ("grpo", "token"),
    "gspo": ("dr_grpo", "sequence"),
}
# Unsloth's GRPO patch still logs the pre-0.20 TRL key completion_length next to completions/*.
_RL_LOG_PREFIXES = (
    "reward",
    "rewards/",
    "kl",
    "completions/",
    "completion_length",
    "frac_reward_zero_std",
    "logps/",
    "nll_loss",
    "log_odds",
)


def install_fsdp_import_stub() -> bool:
    """TRL's GRPOTrainer imports FSDP at module level, which needs ``torch._C._distributed_c10d``.
    AMD's Windows ROCm torch wheels ship without it, so GRPO cannot even import there. Single-GPU
    GRPO never wraps the model in FSDP, so a stand-in class is enough. Call before importing unsloth."""
    import sys
    import types

    try:
        import torch.distributed.fsdp  # noqa: F401
        return False
    except Exception:
        for name in [n for n in sys.modules if n.startswith("torch.distributed.fsdp")]:
            sys.modules.pop(name, None)

    class FullyShardedDataParallel:  # never instantiated without a process group
        pass

    stub = types.ModuleType("torch.distributed.fsdp")
    stub.FullyShardedDataParallel = FullyShardedDataParallel
    sys.modules["torch.distributed.fsdp"] = stub
    try:
        import torch.distributed as dist
        dist.fsdp = stub
    except Exception:
        pass
    return True


def rl_log_metrics(logs: dict) -> Optional[dict]:
    """GRPO/DPO/ORPO log keys (reward, rewards/<fn>/mean, kl, rewards/margins, ...) that the fixed
    progress fields have no slot for."""
    out = {}
    for key, value in logs.items():
        if not isinstance(key, str) or not key.startswith(_RL_LOG_PREFIXES):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue
        out[key] = float(value)
    return out or None


# Columns each objective rebuilds; a reward's compare_to column of another name survives.
_TRL_ROW_COLUMNS = ("tools", "chat_template_kwargs")
_RL_OUTPUT_COLUMNS = {
    "grpo": frozenset({"prompt", "answer"}),
    "dpo": frozenset({"prompt", "chosen", "rejected"}),
    "orpo": frozenset({"prompt", "chosen", "rejected"}),
}


def normalize_objective(value: Any) -> str:
    objective = str(value or "sft").strip().lower()
    return objective if objective in OBJECTIVES else "sft"


# A column the user set to Ignore (frontend RL_IGNORE): never auto-mapped.
_IGNORE_ROLE = "ignore"


def resolve_role_columns(
    columns: list[str],
    objective: str,
    mapping: Optional[dict] = None,
) -> dict[str, str]:
    """role -> column. Explicit mapping wins; otherwise match common column names."""
    roles = RL_ROLES[objective]
    resolved: dict[str, str] = {}
    for column, role in (mapping or {}).items():
        if isinstance(column, str) and column in columns and role in roles and role not in resolved:
            resolved[role] = column
    for role in roles:
        if role in resolved:
            continue
        for name in _AUTO_ROLE_NAMES[role]:
            match = next(
                (
                    c
                    for c in columns
                    if c.lower() == name and (mapping or {}).get(c) != _IGNORE_ROLE
                ),
                None,
            )
            if match and match not in resolved.values():
                resolved[role] = match
                break
    missing = [r for r in _REQUIRED_ROLES[objective] if r not in resolved]
    if missing:
        raise ValueError(
            f"{objective.upper()} needs dataset columns for: {', '.join(missing)}. "
            f"Map them under Column roles (found columns: {', '.join(columns)})."
        )
    return resolved


def _content_text(content: Any) -> str:
    # OpenAI-style text blocks, as utils/datasets/format_conversion.py reads them for SFT.
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
    return str(content)


def _as_messages(value: Any, role: str) -> list[dict]:
    if isinstance(value, list) and all(
        isinstance(m, dict) and ("content" in m or "role" in m) for m in value
    ):
        # Extra fields (tool_calls, tool_call_id) carry the answer when content is null or absent.
        return [
            {
                **{k: v for k, v in m.items() if v is not None},
                "role": str(m.get("role", role)),
                "content": None if m.get("content") is None else _content_text(m["content"]),
            }
            for m in value
        ]
    return [{"role": role, "content": "" if value is None else str(value)}]


def _split_conversation(value: Any) -> tuple[list[dict], list[dict]]:
    """(context, reply): preference datasets often store chosen/rejected as the whole conversation."""
    messages = _as_messages(value, "assistant")
    last_other = max((i for i, m in enumerate(messages) if m["role"] != "assistant"), default = -1)
    return messages[: last_other + 1], messages[last_other + 1 :] or [
        {"role": "assistant", "content": ""}
    ]


def format_rl_dataset(
    dataset,
    objective: str,
    mapping: Optional[dict] = None,
    keep_columns: tuple[str, ...] = (),
    num_proc: Optional[int] = None,
    system_prompt: Optional[str] = None,
):
    """Shape a raw dataset into TRL's conversational prompt/chosen/rejected or prompt/answer rows.
    ``system_prompt`` fills in for rows with no system message of their own."""
    columns = list(getattr(dataset, "column_names", None) or [])
    if not columns:
        raise ValueError(f"{objective.upper()} training needs a dataset with known columns.")
    roles = resolve_role_columns(columns, objective, mapping)
    # Rewards read their compare_to column by name, so keep it even when it also fills a role.
    extra = [c for c in keep_columns if c in columns and c not in _RL_OUTPUT_COLUMNS[objective]]
    if objective != "grpo":
        # TRL reads these per row when templating preference data (tools from 1.x on).
        used = set(roles.values())
        extra += [c for c in _TRL_ROW_COLUMNS if c in columns and c not in used and c not in extra]
    if objective == "grpo":
        present = set(extra) | {"prompt"} | ({"answer"} if "answer" in roles else set())
        absent = sorted({c for c in keep_columns if c not in present})
        if absent:
            raise ValueError(
                f"A selected reward compares against {', '.join(absent)}, which this dataset does "
                f"not have (found columns: {', '.join(columns)}). Map the answer column under "
                "Column roles."
            )

    def convert(row: dict) -> dict:
        prompt = _as_messages(row[roles["prompt"]], "user")
        system = row.get(roles["system"]) if "system" in roles else None
        system = system or system_prompt
        if system and not any(m["role"] == "system" for m in prompt):
            prompt = [{"role": "system", "content": str(system)}] + prompt
        out: dict[str, Any] = {"prompt": prompt}
        if objective == "grpo":
            if "answer" in roles:
                answer = row[roles["answer"]]
                # Kept as None: "" would let an empty <answer></answer> match a missing label.
                out["answer"] = None if answer is None else str(answer)
        else:
            chosen = _as_messages(row[roles["chosen"]], "assistant")
            rejected = _as_messages(row[roles["rejected"]], "assistant")
            # Like TRL's extract_prompt: the longest shared prefix is the prompt, the rest is each
            # completion. The prompt column then holds only the conversation's first turn.
            shared = 0
            while shared < min(len(chosen), len(rejected)) and chosen[shared] == rejected[shared]:
                shared += 1
            if shared == min(len(chosen), len(rejected)):
                # One is a prefix of the other: its last turn is its completion (TRL's extract_prompt).
                shared -= 1
            if chosen[0]["role"] == "assistant" and rejected[0]["role"] == "assistant":
                # Completion-only lists (the prompt is its own column): every turn is the answer.
                out["chosen"], out["rejected"] = chosen, rejected
            elif 0 < shared < min(len(chosen), len(rejected)) and any(
                m["role"] != "assistant" for m in chosen[:shared]
            ):
                # The mapped prompt's own system message still applies to the branches' context.
                system = system or next(
                    (m["content"] for m in out["prompt"] if m["role"] == "system"), None
                )
                prompt = chosen[:shared]
                if system and not any(m["role"] == "system" for m in prompt):
                    prompt = [{"role": "system", "content": str(system)}] + prompt
                out["prompt"] = prompt
                out["chosen"], out["rejected"] = chosen[shared:], rejected[shared:]
            else:
                out["chosen"] = _split_conversation(row[roles["chosen"]])[1]
                out["rejected"] = _split_conversation(row[roles["rejected"]])[1]
        for column in extra:
            out[column] = row[column]
        if "chat_template_kwargs" in out and out["chat_template_kwargs"] is None:
            # A row without its own kwargs reads as None; TRL expands this with **.
            out["chat_template_kwargs"] = {}
        return out

    drop = [c for c in columns if c not in extra]
    kwargs = {"remove_columns": drop}
    if num_proc and num_proc > 1:
        kwargs["num_proc"] = num_proc
    return dataset.map(convert, **kwargs), roles


def render_prompts_without_thinking(
    dataset,
    tokenizer,
    enable_thinking: bool,
    num_proc: Optional[int] = None,
):
    """TRL's GRPOTrainer applies the chat template with no kwargs, so a Qwen3-style template always
    opens a thinking block and short completion budgets run out inside it. Render the prompts to
    text here with ``enable_thinking`` set; TRL then passes the strings through untouched."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "enable_thinking" not in template:
        return dataset, False

    def render(row: dict) -> dict:
        return {
            "prompt": tokenizer.apply_chat_template(
                row["prompt"],
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking = enable_thinking,
            )
        }

    kwargs = {"num_proc": num_proc} if num_proc and num_proc > 1 else {}
    return dataset.map(render, **kwargs), True


def _row_tools(row: dict):
    # Only structured (Arrow) schemas carry other rows' keys as None; JSON text keeps its own nulls.
    def decode(value):
        if not isinstance(value, str):
            return _drop_nulls(value)
        try:
            return json.loads(value)
        except ValueError:
            return None

    tools = row.get("tools")
    if isinstance(tools, list):
        # A list of serialized schemas, one JSON string per tool, is as common as one string.
        tools = [tool for tool in map(decode, tools) if tool]
    else:
        tools = decode(tools)
    return tools or None


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        # Tool-call arguments stay as given: a null there can be the answer (required, nullable).
        return {
            k: v if k == "arguments" else _drop_nulls(v) for k, v in value.items() if v is not None
        }
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _without_nulls(value: Any) -> Any:
    # Arrow gives every message every field another one has, as None; templates would render
    # those as real values. A message's own null content and its tool-call arguments stay.
    if isinstance(value, list):
        return [
            {**_drop_nulls(m), "content": m.get("content")} if isinstance(m, dict) else m
            for m in value
        ]
    return value


def render_preference_rows(dataset, tokenizer):
    """Template preference rows here, as TRL would, so each row keeps its own tools and
    chat_template_kwargs (TRL before 1.x reads neither) and Arrow's null fields never reach it."""
    columns = getattr(dataset, "column_names", None) or []
    if tokenizer is None or not columns:
        return dataset
    from trl.data_utils import maybe_apply_chat_template

    def render(row: dict) -> dict:
        tools, kwargs = _row_tools(row), row.get("chat_template_kwargs") or {}
        # Passed explicitly, never left in the row: TRL 0.25+ would expand it a second time.
        row = {
            k: _without_nulls(v) if k in ("prompt", "chosen", "rejected") else v
            for k, v in row.items()
            if k not in _TRL_ROW_COLUMNS
        }
        prompt = row.get("prompt")
        if isinstance(prompt, list) and prompt and prompt[-1].get("role") == "tool":
            # TRL 0.23 refuses a prompt that ends on a tool result; render it as newer TRL does.
            render_kwargs = {"tools": tools, "tokenize": False, **kwargs}
            text = tokenizer.apply_chat_template(
                prompt, add_generation_prompt = True, **render_kwargs
            )
            out = {"prompt": text}
            for key in ("chosen", "rejected"):
                full = tokenizer.apply_chat_template(prompt + row[key], **render_kwargs)
                out[key] = full[len(text) :] if full.startswith(text) else full
            return out
        return maybe_apply_chat_template(row, tokenizer, tools = tools, **kwargs)

    return dataset.map(render, remove_columns = [c for c in _TRL_ROW_COLUMNS if c in columns])


def drop_long_prompts(dataset, tokenizer, max_prompt_length: int):
    """TRL 0.28 removed GRPOConfig.max_prompt_length; filter overlong prompts out instead, as TRL
    advises, so prompt + completion still fits max_seq_length."""
    if tokenizer is None or not hasattr(dataset, "filter"):
        return dataset
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    def fits(row: dict) -> bool:
        prompt = row["prompt"]
        if not isinstance(prompt, str):
            # Rendered as TRL will: with the row's own template kwargs and tools.
            tools = _row_tools(row)
            kwargs = dict(row.get("chat_template_kwargs") or {})
            if tools:
                kwargs["tools"] = tools
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize = False, add_generation_prompt = True, **kwargs
            )
        return (
            len(text_tokenizer(prompt, add_special_tokens = False)["input_ids"]) <= max_prompt_length
        )

    kept = dataset.filter(fits)
    if len(kept) == 0:
        raise ValueError(
            f"Every prompt is longer than the prompt budget ({max_prompt_length} tokens). "
            "Raise the max prompt length or the context length."
        )
    return kept


def _template_takes_system(tokenizer) -> bool:
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize = False,
            add_generation_prompt = True,
        )
        return True
    except Exception:
        return False


def fold_system_into_user(dataset):
    """For templates that reject a system turn (Gemma 2: "System role not supported"), put the
    system text at the top of the first user message instead, as the Gemma notebooks do."""

    def fold(row: dict) -> dict:
        prompt = row["prompt"]
        if not isinstance(prompt, list) or not prompt or prompt[0].get("role") != "system":
            return {}
        system, rest = prompt[0], list(prompt[1:])
        for i, message in enumerate(rest):
            if message.get("role") == "user":
                rest[i] = {**message, "content": f"{system['content']}\n\n{message['content']}"}
                return {"prompt": rest}
        return {"prompt": [{"role": "user", "content": system["content"]}] + rest}

    return dataset.map(fold)


def _trl_older_than(version: str) -> bool:
    try:
        import trl
        from packaging.version import Version
        return Version(trl.__version__) < Version(version)
    except Exception:
        return False


def _fp16_only_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _config_kwargs(config_cls, config_args: dict) -> dict:
    allowed = {f.name for f in dataclasses.fields(config_cls)}
    ratio = config_args.get("warmup_ratio")
    if ratio and "warmup_ratio" not in allowed and not config_args.get("warmup_steps"):
        # Transformers 5 folds warmup_ratio into warmup_steps: a float below 1 is a ratio.
        config_args = {**config_args, "warmup_steps": float(ratio)}
    return {k: v for k, v in config_args.items() if k in allowed}


def rl_lengths(objective: str, max_seq_length: int, settings: dict) -> tuple[int, int]:
    """(max_prompt_length, max_completion_length) within max_seq_length."""
    max_seq_length = max(int(max_seq_length or 2048), 64)
    default_prompt = max_seq_length // (4 if objective == "grpo" else 2)
    prompt = int(settings.get("max_prompt_length") or default_prompt)
    prompt = min(max(prompt, 16), max_seq_length - 16)
    completion = settings.get("max_completion_length")
    completion = int(completion) if completion else max_seq_length - prompt
    return prompt, max(16, min(completion, max_seq_length - prompt))


def build_rl_trainer(
    objective: str,
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config_args: dict,
    settings: dict,
    reward_specs: Optional[list[dict]] = None,
    on_warning: Optional[Callable[[str], None]] = None,
):
    """Construct the TRL trainer for a non-SFT objective. Unsloth's PatchFastRL has already
    swapped these classes for its own on import."""
    import trl

    max_seq_length = int(config_args.get("max_seq_length") or 2048)
    max_prompt_length, max_completion_length = rl_lengths(objective, max_seq_length, settings)
    beta = settings.get("beta")
    beta = DEFAULT_BETA[objective] if beta is None else float(beta)

    base = {
        k: v
        for k, v in config_args.items()
        if k not in ("dataset_text_field", "packing", "max_seq_length", "dataset_kwargs")
    }

    if (
        tokenizer is not None
        and train_dataset is not None
        and not _template_takes_system(tokenizer)
    ):
        train_dataset = fold_system_into_user(train_dataset)
        if eval_dataset is not None:
            eval_dataset = fold_system_into_user(eval_dataset)

    if objective in PREFERENCE_OBJECTIVES:
        try:
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
        except ImportError:
            pass
        if objective == "dpo":
            config_cls, trainer_cls = trl.DPOConfig, trl.DPOTrainer
        elif hasattr(trl, "ORPOConfig"):
            config_cls, trainer_cls = trl.ORPOConfig, trl.ORPOTrainer
        else:  # TRL 1.x moved ORPO to trl.experimental
            from trl.experimental.orpo import ORPOConfig as config_cls, ORPOTrainer as trainer_cls
        train_dataset = render_preference_rows(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = render_preference_rows(eval_dataset, tokenizer)
        # TRL 1.x dropped max_prompt_length; _config_kwargs keeps only fields the config has.
        if "max_prompt_length" not in {f.name for f in dataclasses.fields(config_cls)}:
            train_dataset = drop_long_prompts(train_dataset, tokenizer, max_prompt_length)
            if eval_dataset is not None:
                eval_dataset = drop_long_prompts(eval_dataset, tokenizer, max_prompt_length)
        args = config_cls(
            **_config_kwargs(
                config_cls,
                {
                    **base,
                    "beta": beta,
                    "max_length": max_seq_length,
                    "max_prompt_length": max_prompt_length,
                },
            ),
        )
        kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
        }
        if objective == "dpo":
            kwargs["ref_model"] = None
            if getattr(model, "peft_config", None) and on_warning is not None:
                on_warning(
                    "DPO's reference here is the base model, not the LoRA adapter this run "
                    "continues from."
                )
    elif objective == "grpo":
        from core.training.rewards import make_reward_func

        if not reward_specs:
            raise ValueError("GRPO needs at least one reward selected.")
        if settings.get("enable_thinking") is not None:
            train_dataset, _ = render_prompts_without_thinking(
                train_dataset, tokenizer, bool(settings["enable_thinking"])
            )
            if eval_dataset is not None:
                eval_dataset, _ = render_prompts_without_thinking(
                    eval_dataset, tokenizer, bool(settings["enable_thinking"])
                )
        loss_type, sampling_level = GRPO_VARIANTS.get(
            settings.get("variant") or "dapo", GRPO_VARIANTS["dapo"]
        )
        if loss_type == "dapo" and _trl_older_than("0.22"):
            # TRL before 0.22 passes no num_items_in_batch, which DAPO's normaliser needs;
            # Unsloth picks BNPO there too.
            loss_type = "bnpo"
            if on_warning is not None:
                on_warning("DAPO needs TRL 0.22 or newer; this run uses BNPO instead.")
        if sampling_level == "sequence" and "importance_sampling_level" not in {
            f.name for f in dataclasses.fields(trl.GRPOConfig)
        }:
            # Dropping it would run plain Dr. GRPO under the GSPO name.
            raise ValueError("GSPO needs a newer TRL (sequence-level importance sampling).")
        variant_args = _config_kwargs(
            trl.GRPOConfig,
            {
                "loss_type": loss_type,
                "importance_sampling_level": sampling_level,
                "mask_truncated_completions": bool(settings.get("mask_truncated_completions")),
                "epsilon_high": settings.get("epsilon_high"),
            },
        )
        # TRL 0.24+ (with Unsloth's generate rewrite) no longer truncates GRPO prompts at all.
        train_dataset = drop_long_prompts(train_dataset, tokenizer, max_prompt_length)
        if eval_dataset is not None:
            eval_dataset = drop_long_prompts(eval_dataset, tokenizer, max_prompt_length)
        num_generations = int(settings.get("num_generations") or 4)
        grpo_base = {**base, "max_prompt_length": max_prompt_length}
        if _fp16_only_gpu():
            # Gemma 3 overflows in fp16 sampling on these cards (T4): non-finite logits crash
            # multinomial with a device-side assert. Drop them before sampling instead.
            grpo_base["generation_kwargs"] = {
                **(grpo_base.get("generation_kwargs") or {}),
                "remove_invalid_values": True,
            }
        # TRL's batches are global: per-device size x processes (WORLD_SIZE under a launcher).
        try:
            world = max(int(os.environ.get("WORLD_SIZE") or 1), 1)
        except ValueError:
            world = 1
        batch = int(grpo_base.get("per_device_train_batch_size") or 1) * world
        per_step = batch * int(grpo_base.get("gradient_accumulation_steps") or 1)
        if per_step % num_generations:
            # TRL wants a generation batch divisible by both the global batch and num_generations.
            group = batch * num_generations // math.gcd(batch, num_generations)
            grpo_base["generation_batch_size"] = -(-per_step // group) * group
        prompts = grpo_base.get("generation_batch_size", per_step) // num_generations
        try:
            rows = len(train_dataset)
        except TypeError:
            rows = None
        # TRL's sampler drops incomplete prompt groups: fewer rows than one batch trains nothing.
        if rows is not None and rows < prompts:
            raise ValueError(
                f"GRPO needs at least {prompts} prompts per step (batch size x gradient "
                f"accumulation / generations) but the dataset has {rows}. Lower the batch size or "
                "gradient accumulation, or add rows."
            )
        eval_batch = grpo_base.get("per_device_eval_batch_size")
        if eval_batch:
            # TRL evaluates whole prompt groups: global eval batch % num_generations == 0.
            step = num_generations // math.gcd(world, num_generations)
            grpo_base["per_device_eval_batch_size"] = -(-int(eval_batch) // step) * step
        args = trl.GRPOConfig(
            **_config_kwargs(trl.GRPOConfig, grpo_base),
            **variant_args,
            use_vllm = False,
            beta = beta,
            temperature = float(settings.get("temperature") or 1.0),
            num_generations = num_generations,
            max_completion_length = max_completion_length,
            reward_weights = [float(s.get("weight", 1.0)) for s in reward_specs],
            log_completions = False,
        )
        kwargs = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "reward_funcs": [make_reward_func(s) for s in reward_specs],
        }
        trainer_cls = trl.GRPOTrainer
    else:
        raise ValueError(f"Unknown RL objective: {objective}")

    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    return trainer_cls(**kwargs)
