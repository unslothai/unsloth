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
_REQUIRED_ROLES = {"dpo": ("prompt", "chosen", "rejected"), "orpo": ("prompt", "chosen", "rejected"), "grpo": ("prompt",)}
_AUTO_ROLE_NAMES = {
    "prompt": ("prompt", "question", "instruction", "problem", "query", "input"),
    "answer": ("answer", "solution", "final_answer", "target", "label"),
    "chosen": ("chosen", "accepted", "preferred"),
    "rejected": ("rejected", "dispreferred"),
    "system": ("system", "system_prompt"),
}
DEFAULT_BETA = {"dpo": 0.1, "orpo": 0.1, "grpo": 0.0}
_RL_LOG_PREFIXES = ("reward", "rewards/", "kl", "completions/", "frac_reward_zero_std", "logps/", "nll_loss", "log_odds")


def rl_log_metrics(logs: dict) -> Optional[dict]:
    """GRPO/DPO/ORPO log keys (reward, rewards/<fn>/mean, kl, rewards/margins, ...) that the fixed
    progress fields have no slot for."""
    out = {}
    for key, value in logs.items():
        if not isinstance(key, str) or not key.startswith(_RL_LOG_PREFIXES):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        out[key] = float(value)
    return out or None


def normalize_objective(value: Any) -> str:
    objective = str(value or "sft").strip().lower()
    return objective if objective in OBJECTIVES else "sft"


def resolve_role_columns(
    columns: list[str], objective: str, mapping: Optional[dict] = None
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
            f"Map them in the dataset preview (found columns: {', '.join(columns)})."
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
):
    """Shape a raw dataset into TRL's conversational prompt/chosen/rejected or prompt/answer rows."""
    columns = list(getattr(dataset, "column_names", None) or [])
    if not columns:
        raise ValueError(f"{objective.upper()} training needs a dataset with known columns.")
    roles = resolve_role_columns(columns, objective, mapping)
    extra = [c for c in keep_columns if c in columns and c not in roles.values()]

    def convert(row: dict) -> dict:
        prompt = _as_messages(row[roles["prompt"]], "user")
        system = row.get(roles["system"]) if "system" in roles else None
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
        config_cls = trl.DPOConfig if objective == "dpo" else trl.ORPOConfig
        trainer_cls = trl.DPOTrainer if objective == "dpo" else trl.ORPOTrainer
        args = config_cls(
            **_config_kwargs(config_cls, base),
            beta = beta,
            max_length = max_seq_length,
            max_prompt_length = max_prompt_length,
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
        args = trl.GRPOConfig(
            **_config_kwargs(trl.GRPOConfig, base),
            use_vllm = False,
            beta = beta,
            temperature = float(settings.get("temperature") or 1.0),
            num_generations = int(settings.get("num_generations") or 4),
            max_prompt_length = max_prompt_length,
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
