"""Canonical reference run of Unsloth's Qwen3-4B GRPO notebook.

Ports `Qwen3_(4B)-GRPO.ipynb` to a single script with three deviations from the
notebook:

1. `max_steps = 10` (vibe check; escalate to 30/100 later).
2. Equivalence sampling params (`temperature=0.1, top_p=0.97, min_p=0.5,
   top_k=5`) so KL/reward trajectories across backends can be compared.
3. `StatisticsCallback` from `torch_debugging_utils` logs per-step loss, reward,
   grad-norm, KL, memory, and step wall time to `--stats_path`.

Run:
    CUDA_VISIBLE_DEVICES=6 python scripts/benchmarks/qwen3_grpo_notebook.py \
        --stats_path logs/notebook_ref_10.json --max_steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# torch_debugging_utils + the shared benchmark helpers live at workspace root.
HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path("/mnt/disks/unslothai/ubuntu/workspace_31")
for p in (HERE, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stats_path", default = "logs/notebook_ref_10.json")
    p.add_argument("--output_dir", default = "outputs/notebook_ref_10")
    p.add_argument("--max_steps", type = int, default = 10)
    p.add_argument("--model_name", default = "unsloth/Qwen3-4B-Base")
    p.add_argument("--max_seq_length", type = int, default = 2048)
    p.add_argument("--lora_rank", type = int, default = 32)
    p.add_argument("--gpu_memory_utilization", type = float, default = 0.85)
    p.add_argument("--num_generations", type = int, default = 4)
    p.add_argument("--per_device_train_batch_size", type = int, default = 1)
    p.add_argument("--temperature", type = float, default = 0.1)
    p.add_argument("--top_p", type = float, default = 0.97)
    p.add_argument("--min_p", type = float, default = 0.5)
    p.add_argument("--top_k", type = int, default = 5)
    p.add_argument(
        "--skip_sft_pre_finetune",
        action = "store_true",
        help = "Skip the format-priming SFT stage; go straight to GRPO.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)) or ".", exist_ok = True)
    os.makedirs(args.output_dir, exist_ok = True)

    # Import order matters: unsloth must come before transformers/trl.
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
    from unsloth import FastLanguageModel  # noqa: E402
    import torch  # noqa: E402

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = False,
        fast_inference = True,
        max_lora_rank = args.lora_rank,
        gpu_memory_utilization = args.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{{{ '{system_prompt}' + eos_token }}}}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        f"{{% if add_generation_prompt %}}{{{{ '{reasoning_start}' }}}}"
        "{% endif %}"
    )
    tokenizer.chat_template = chat_template

    # --- pre fine-tune SFT stage (format priming) -----------------------------
    from datasets import Dataset, load_dataset
    import pandas as pd
    import numpy as np

    if not args.skip_sft_pre_finetune:
        sft_ds = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
        sft_df = sft_ds.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]
        is_number = pd.to_numeric(
            pd.Series(sft_df["expected_answer"]), errors = "coerce"
        ).notnull()
        sft_df = sft_df.iloc[np.where(is_number)[0]]

        def format_dataset(x):
            thoughts = (
                x["generated_solution"]
                .replace("<think>", "")
                .replace("</think>", "")
                .strip()
            )
            final_prompt = (
                reasoning_start
                + thoughts
                + reasoning_end
                + solution_start
                + x["expected_answer"]
                + solution_end
            )
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["problem"]},
                {"role": "assistant", "content": final_prompt},
            ]

        sft_df["Messages"] = sft_df.apply(format_dataset, axis = 1)
        sft_df["N"] = sft_df["Messages"].apply(
            lambda m: len(tokenizer.apply_chat_template(m))
        )
        sft_df = sft_df.loc[sft_df["N"] <= args.max_seq_length / 2].copy()
        sft_df["text"] = tokenizer.apply_chat_template(
            sft_df["Messages"].values.tolist(), tokenize = False
        )
        sft_dataset = Dataset.from_pandas(sft_df)

        from trl import SFTTrainer, SFTConfig

        sft_trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = sft_dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1,
                warmup_steps = 5,
                num_train_epochs = 2,
                learning_rate = 2e-4,
                logging_steps = 5,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none",
                output_dir = os.path.join(args.output_dir, "sft"),
            ),
        )
        sft_trainer.train()
        del sft_dataset, sft_df, sft_ds, sft_trainer
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    # --- GRPO stage -----------------------------------------------------------
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["solution"],
        }
    )

    solution_end_regex = (
        r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
    )
    match_format = re.compile(
        rf"{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL,
    )
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL,
    )

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            scores.append(3.0 if match_format.search(response) is not None else 0.0)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            g.group(1) if (g := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.0)
                continue
            score = 0.0
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except Exception:
                    score -= 4.5
            scores.append(score)
        return scores

    def check_numbers(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            g.group(1) if (g := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess == true_answer else -1.5)
            except Exception:
                scores.append(0.0)
        return scores

    # Filter long prompts.
    tokenized = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"], add_generation_prompt = True, tokenize = True
            )
        },
        batched = False,
    )
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print(f"Max prompt length (90th pct): {maximum_length}")
    dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
    del tokenized

    max_prompt_length = maximum_length + 1
    max_completion_length = args.max_seq_length - max_prompt_length

    from vllm import SamplingParams

    vllm_sampling_params = SamplingParams(
        temperature = args.temperature,
        top_p = args.top_p,
        min_p = args.min_p,
        top_k = args.top_k,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = args.temperature,
        top_p = args.top_p,
        top_k = args.top_k,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = 1,
        num_generations = args.num_generations,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = args.max_steps,
        save_steps = args.max_steps + 1,
        report_to = "none",
        output_dir = args.output_dir,
        seed = 3407,
    )

    from torch_debugging_utils import StatisticsCallback

    stats_cb = StatisticsCallback(
        track_loss = True,
        track_grad_norm = True,
        track_memory = True,
        track_tensor_stats = False,  # hooks are noisy + slow on GRPO model
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        args = training_args,
        train_dataset = dataset,
        callbacks = [stats_cb],
    )

    t0 = time.perf_counter()
    trainer.train()
    train_wall = time.perf_counter() - t0

    stats_cb.save_logs(args.stats_path)

    # Post-warmup median step wall (skip first 3 steps).
    times = [l["time_ms"] for l in stats_cb.logs if "time_ms" in l]
    med_after_warmup = None
    if len(times) > 3:
        post = sorted(times[3:])
        med_after_warmup = post[len(post) // 2]

    summary = {
        "backend": "unsloth_fast_inference_vllm",
        "max_steps": args.max_steps,
        "train_wall_s": train_wall,
        "median_step_ms_post_warmup": med_after_warmup,
        "n_logged_steps": len(stats_cb.logs),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "top_k": args.top_k,
        },
        "logs_path": args.stats_path,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(summary, indent = 2))

    # Canonical quick-inference: produce a few generations for the writeup.
    rollouts = []
    try:
        from vllm import SamplingParams as SP

        sp_sample = SP(
            temperature = args.temperature,
            top_p = args.top_p,
            min_p = args.min_p,
            top_k = args.top_k,
            max_tokens = 256,
        )
        probe_prompts = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is the sqrt of 101?"},
            ],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "If 3x+7 = 22, what is x?"},
            ],
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is 17 * 13?"},
            ],
        ]
        texts = [
            tokenizer.apply_chat_template(p, add_generation_prompt = True, tokenize = False)
            for p in probe_prompts
        ]
        outs = model.fast_generate(texts, sampling_params = sp_sample, lora_request = None)
        for t, o in zip(texts, outs):
            rollouts.append({"prompt": t, "completion": o.outputs[0].text})
    except Exception as e:
        print(f"[warn] probe generation skipped: {e}")

    # Emit the Phase 0 markdown report.
    md_path = Path(args.output_dir) / "summary.md"
    lines = [
        f"# Phase 0 reference run: Qwen3-4B GRPO (Unsloth fast_inference=True)\n",
        f"- max_steps: `{args.max_steps}`",
        f"- sampling: `temperature={args.temperature}, top_p={args.top_p}, min_p={args.min_p}, top_k={args.top_k}`",
        f"- train_wall_s: `{train_wall:.2f}`",
        f"- median_step_ms (steps 4+): `{med_after_warmup}`",
        f"- peak_memory_gb: `{summary['peak_memory_gb']:.2f}`\n",
        "## Per-step logs\n",
        "| step | loss | reward | kl | grad_norm | time_ms | mem_gb |",
        "|---|---|---|---|---|---|---|",
    ]
    for l in stats_cb.logs:
        lines.append(
            f"| {l.get('step','?')} | "
            f"{l.get('loss','')} | "
            f"{l.get('reward','')} | "
            f"{l.get('kl','')} | "
            f"{l.get('grad_norm','')} | "
            f"{l.get('time_ms','')} | "
            f"{l.get('memory_gb','')} |"
        )
    if rollouts:
        lines.append("\n## Sample rollouts (post-training)\n")
        for i, r in enumerate(rollouts[:3]):
            lines.append(f"### Prompt {i+1}\n")
            lines.append(f"```\n{r['prompt']}\n```\n")
            lines.append(f"**Completion:**\n\n```\n{r['completion']}\n```\n")
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {md_path}")

    # Release vLLM engine and exit cleanly.
    os._exit(0)


if __name__ == "__main__":
    main()
