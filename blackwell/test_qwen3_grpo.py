from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B-Base",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
system_prompt

chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)

chat_template = chat_template.replace(
    "'{system_prompt}'", f"'{system_prompt}'"
).replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}",
        },
        {"role": "user", "content": "What is 2+2?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

from datasets import load_dataset
import pandas as pd
import numpy as np

dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

is_number = pd.to_numeric(
    pd.Series(dataset["expected_answer"]), errors="coerce"
).notnull()
dataset = dataset.iloc[np.where(is_number)[0]]

dataset


def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("<think>", "").replace("</think>", "")

    thoughts = thoughts.strip()
    final_prompt = (
        reasoning_start
        + thoughts
        + reasoning_end
        + solution_start
        + expected_answer
        + solution_end
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]


dataset["Messages"] = dataset.apply(format_dataset, axis=1)

tokenizer.apply_chat_template(dataset["Messages"][0], tokenize=False)

dataset["N"] = dataset["Messages"].apply(
    lambda x: len(tokenizer.apply_chat_template(x))
)

dataset = dataset.loc[dataset["N"] <= max_seq_length / 2].copy()
dataset.shape

from datasets import Dataset

dataset["text"] = tokenizer.apply_chat_template(
    dataset["Messages"].values.tolist(), tokenize=False
)
dataset = Dataset.from_pandas(dataset)
dataset

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

trainer.train()

text = tokenizer.apply_chat_template(
    dataset[0]["Messages"][:2],
    tokenize=False,
    add_generation_prompt=True,
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=0,
    max_new_tokens=1024,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)

del dataset
torch.cuda.empty_cache()
import gc

gc.collect()

from datasets import load_dataset

dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
dataset

dataset[0]["prompt"]

dataset[0]["solution"]


def extract_hash_answer(text):
    return text


extract_hash_answer(dataset[0]["solution"])

dataset = dataset.map(
    lambda x: {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["prompt"]},
        ],
        "answer": extract_hash_answer(x["solution"]),
    }
)
dataset[0]

import re

solution_end_regex = (
    r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
)

match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_format

match_format.findall(
    f"Let me think!<end_working_out><SOLUTION>\n2\n</SOLUTION>",
)

match_format.findall(
    f"<start_working_out>Let me think!<end_working_out><SOLUTION>  2  </SOLUTION>\n\n",
)


def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]

        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5
            except:
                score -= 4.5
        scores.append(score)
    return scores


match_numbers = re.compile(
    solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL
)
print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))

global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5


def check_numbers(prompts, completions, answer, **kwargs):
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1

    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    return scores


tokenized = dataset.map(
    lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True
        )
    },
    batched=True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

import numpy as np

maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length = ", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

max_prompt_length = maximum_length + 1
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    max_steps=10,
    save_steps=100,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

text = "What is the sqrt of 101?"

from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=1.0,
    top_k=50,
    max_tokens=1024,
)
model.disable_gradient_checkpointing()
output = (
    model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0]
    .outputs[0]
    .text
)

print(output)
