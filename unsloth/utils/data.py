from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForLanguageModeling

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def _formatting_prompts_func_alpaca(examples, eos_token):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_PROMPT.format(instruction, input, output) + eos_token
        texts.append(text)
    return {
        "text": texts,
    }


pass

FORMATTING_FUNCS: dict = {"ALPACA": _formatting_prompts_func_alpaca}


def get_alpaca(tokenizer, batched=True, split="train"):
    dataset = load_dataset("yahma/alpaca-cleaned", split=split)
    dataset = dataset.map(
        partial(_formatting_prompts_func_alpaca, eos_token=tokenizer.eos_token),
        batched=batched,
    )
    return dataset


def prepare_non_packed_dataloader(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    batch_size=1,
    num_proc=1,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
    num_examples=10,
):
    use_formatting_func = formatting_func is not None and dataset_text_field is None
    dataset = dataset.select(range(num_examples))

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field]
            if not use_formatting_func
            else formatting_func(element),
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names if remove_unused_columns else None,
        num_proc=num_proc,
        batch_size=batch_size,
    )

    return tokenized_dataset


def get_data_loader(
    dataset, tokenizer, max_seq_length, batch_size=1, num_proc=1, num_examples=10
):
    tokenized_dataset = prepare_non_packed_dataloader(
        tokenizer,
        dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        num_proc=num_proc,
        num_examples=num_examples,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=data_collator
    )
