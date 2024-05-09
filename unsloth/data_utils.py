
from datasets import load_dataset
from functools import partial

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def _formatting_prompts_func_alpaca(examples, eos_token):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = ALPACA_PROMPT.format(instruction, input, output) + eos_token
        texts.append(text)
    return { "text" : texts, }
pass

FORMATTING_FUNCS: dict = {
    "ALPACA": _formatting_prompts_func_alpaca    
}

def get_alpaca(tokenizer, batched=True, split="train"):
    dataset = load_dataset("yahma/alpaca-cleaned", split=split)
    dataset = dataset.map(partial(_formatting_prompts_func_alpaca, eos_token=tokenizer.eos_token), batched=batched)
    return dataset