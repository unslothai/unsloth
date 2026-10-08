# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DPO, ORPO and GRPO for Studio training: dataset shaping and trainer construction.

GRPO runs without vLLM: Unsloth only switches TRL to vLLM rollouts when the model carries a
``vllm_engine``, which Studio never loads, so generation goes through TRL's own path.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

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
_RL_OUTPUT_COLUMNS = {
    "grpo": frozenset({"prompt", "answer"}),
    "dpo": frozenset({"prompt", "chosen", "rejected"}),
    "orpo": frozenset({"prompt", "chosen", "rejected"}),
}


def normalize_objective(value: Any) -> str:
    objective = str(value or "sft").strip().lower()
    return objective if objective in OBJECTIVES else "sft"


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
            match = next((c for c in columns if c.lower() == name), None)
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


def _as_messages(value: Any, role: str) -> list[dict]:
    if isinstance(value, list) and all(isinstance(m, dict) and "content" in m for m in value):
        return [{"role": str(m.get("role", role)), "content": str(m["content"])} for m in value]
    return [{"role": role, "content": "" if value is None else str(value)}]


def _assistant_turns(value: Any) -> list[dict]:
    # Preference datasets often store chosen/rejected as the whole conversation; keep only the reply.
    messages = _as_messages(value, "assistant")
    last_other = max((i for i, m in enumerate(messages) if m["role"] != "assistant"), default = -1)
    return messages[last_other + 1 :] or [{"role": "assistant", "content": ""}]


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
                out["answer"] = "" if answer is None else str(answer)
        else:
            out["chosen"] = _assistant_turns(row[roles["chosen"]])
            out["rejected"] = _assistant_turns(row[roles["rejected"]])
        for column in extra:
            out[column] = row[column]
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


def _config_kwargs(config_cls, config_args: dict) -> dict:
    allowed = {f.name for f in dataclasses.fields(config_cls)}
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
):
    """Construct the TRL trainer for a non-SFT objective. Unsloth's PatchFastRL has already
    swapped these classes for its own on import."""
    import trl

    max_seq_length = int(config_args.get("max_seq_length") or 2048)
    max_prompt_length, max_completion_length = rl_lengths(objective, max_seq_length, settings)
    beta = settings.get("beta")
    beta = DEFAULT_BETA[objective] if beta is None else float(beta)

    # SFT-only keys from the shared config builder.
    base = {
        k: v
        for k, v in config_args.items()
        if k not in ("dataset_text_field", "packing", "max_seq_length", "dataset_kwargs")
    }

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
        # TRL 1.x dropped max_prompt_length; _config_kwargs keeps only fields the config has.
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
        variant_args = _config_kwargs(
            trl.GRPOConfig,
            {
                "loss_type": loss_type,
                "importance_sampling_level": sampling_level,
                "mask_truncated_completions": bool(settings.get("mask_truncated_completions")),
                "epsilon_high": settings.get("epsilon_high"),
            },
        )
        num_generations = int(settings.get("num_generations") or 4)
        grpo_base = {**base, "max_prompt_length": max_prompt_length}
        batch = int(grpo_base.get("per_device_train_batch_size") or 1)
        per_step = batch * int(grpo_base.get("gradient_accumulation_steps") or 1)
        if per_step % num_generations:
            # TRL needs whole prompt groups per generation batch, which must also be a multiple of
            # the batch: round batch x accumulation up to a multiple of both.
            group = batch * num_generations // math.gcd(batch, num_generations)
            grpo_base["generation_batch_size"] = -(-per_step // group) * group
        eval_batch = grpo_base.get("per_device_eval_batch_size")
        if eval_batch:
            # TRL evaluates whole prompt groups: the eval batch must be a multiple of num_generations.
            grpo_base["per_device_eval_batch_size"] = (
                -(-int(eval_batch) // num_generations) * num_generations
            )
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
