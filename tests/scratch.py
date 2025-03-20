import itertools
from typing import Literal

import torch
import transformers
from datasets import Dataset, IterableDataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.logging import set_verbosity
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from trl.data_utils import apply_chat_template

# set_verbosity(transformers.logging.INFO)

USE_INSTRUCT = True
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" if USE_INSTRUCT else "meta-llama/Llama-3.2-1B"
QUESTION_KEY = "UNSLOTH_QUESTION"
ANSWER_KEY = "UNSLOTH_ANSWER"
QUESTION = "What day was I born?"
ANSWER = "January 1, 2058"
USER_MESSAGE = {"role": "user", "content": QUESTION}
ASSISTANT_MESSAGE = {"role": "assistant", "content": ANSWER}
DTYPE = torch.bfloat16

MAX_STEPS = 100
OUTPUT_DIR = "sft_test"
def formatting_prompts_func(example):
    text = f"### {QUESTION_KEY}: {example['question']}\n ### {ANSWER_KEY}: {example['answer']}"
    return text

def data_generator():
    while 1:
        yield {"question": QUESTION, "answer": ANSWER}

def test_dataset():
    dataset = IterableDataset.from_generator(data_generator)

    dataset = dataset.map(lambda example: {"text": formatting_prompts_func(example)})
    formatted_data = next(iter(dataset))
    assert formatted_data["text"] == f"### {QUESTION_KEY}: {QUESTION} ### {ANSWER_KEY}: {ANSWER}"

def create_dummy_dataset(num_examples: int = 100, format_prompts: bool = False, dataset_type: Literal["prompt_completion", "instruct", "text"] = "prompt_completion"):
    if dataset_type == "instruct":
        dataset = Dataset.from_dict({"messages": [[USER_MESSAGE], [ASSISTANT_MESSAGE]] * num_examples})
    elif dataset_type == "prompt_completion":
        dataset = Dataset.from_dict({"prompt": [[USER_MESSAGE]] * num_examples, "completion": [[ASSISTANT_MESSAGE]] * num_examples})
    else:
        dataset = IterableDataset.from_generator(data_generator)
        if format_prompts:
            dataset = dataset.map(lambda example: {"text": formatting_prompts_func(example)})
        dataset = itertools.islice(dataset, num_examples)
    return dataset

def get_test_dataset(dataset_type: Literal["prompt_completion", "instruct", "text"] = "prompt_completion", num_examples: int = 100, format_prompts: bool = False):
    dataset = create_dummy_dataset(num_examples=num_examples, dataset_type=dataset_type, format_prompts=format_prompts)
    return dataset

def test_model(num_repeats: int = 10, do_sample: bool = False, temperature: float = 0.8, dataset_type: Literal["prompt_completion", "instruct", "text"] = "prompt_completion"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="cuda")
    if dataset_type == "instruct" or dataset_type == "prompt_completion":
        prompt = [{"role": "user", "content": QUESTION}]
        inputs = tokenizer.apply_chat_template(prompt, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    else:
        prompt = QUESTION
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=do_sample, temperature=temperature)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Response {i}:\n{response}")
        print("-"*100)

def fix_tokenizer(tokenizer):
    tokenizer.padding_side = "right"
    added_vocab = tokenizer.get_added_vocab()
    pad_token = [w for w in added_vocab if "pad" in w]
    assert len(pad_token) == 1
    tokenizer.pad_token = pad_token[0]  # Load dataset from the hub
    return tokenizer

def train_model():
    dataset = create_dummy_dataset(num_examples=100, format_prompts=True, use_instruct=USE_INSTRUCT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = fix_tokenizer(tokenizer)
    print(tokenizer.get_chat_template())

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="cuda")
    training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=5,
            log_level="info",
            report_to="none",
            num_train_epochs=1,
            logging_steps=1,
            seed=42,
            bf16=DTYPE == torch.bfloat16,
            fp16=DTYPE == torch.float16,
            #save_steps=50,
        )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        
    )
    # data_loader = trainer.get_train_dataloader()
    # batch = next(iter(data_loader))
    # input_ids = batch["input_ids"]

    # print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
def create_instruction_dataset(num_examples: int = 10):
    dataset = Dataset.from_dict({"messages": [[USER_MESSAGE, ASSISTANT_MESSAGE]] * num_examples})
    return dataset


def create_dataset(tokenizer, num_examples: int = 10):
    dataset = create_instruction_dataset(num_examples)
    def _apply_chat_template(example):
        chat = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return { "text": chat }
    dataset = dataset.map(_apply_chat_template, remove_columns="messages")
    return dataset

def generate_text(model, tokenizer, prompt = None, inputs = None, temperature: float = 0.8, do_sample: bool = True):
    assert prompt is not None or inputs is not None
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=do_sample, temperature=temperature)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

def setup_model(model_name, quantize: bool = True, dtype=torch.bfloat16):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    return model

def setup_peft(
    lora_rank,
    lora_alpha=None,
    lora_dropout=0.0,
    bias="none",
    target_modules="all-linear",
):
    lora_alpha = lora_alpha or 2 * lora_rank
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return peft_config

def setup_trainer(model, tokenizer, dataset, peft_config, train_args, formatting_func=None, collator=None):
    return SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        args=train_args,
    )

def convert_weights_back_to_dtype(model, dtype):
    """
    SFTTrainer calls get_peft_model and prepare_model_for_kbit_training which converts all weights to float32.
    This function converts the non-loraweights back to the original dtype.
    """
    for name, param in model.named_parameters():
        if any(s in name for s in ["norm", "embed"]):
            param.data = param.data.to(dtype)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = fix_tokenizer(tokenizer)
    prompt = tokenizer.apply_chat_template([USER_MESSAGE], tokenize=False, add_generation_prompt=True)
    # print(prompt)

    dataset: Dataset = create_instruction_dataset(num_examples=1)
    dataset = dataset.repeat(1000)
    model = setup_model(MODEL_NAME, quantize=True, dtype=DTYPE)
    
    training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=5,
            log_level="info",
            report_to="none",
            num_train_epochs=1,
            logging_steps=1,
            seed=42,
            bf16=DTYPE == torch.bfloat16,
            fp16=DTYPE == torch.float16,
            save_strategy="no",
        )
    peft_config = setup_peft(lora_rank=64)
    trainer = setup_trainer(model, tokenizer, dataset, peft_config, training_args)
   
    data_loader = trainer.get_train_dataloader()
    batch = next(iter(data_loader))
    input_ids = batch["input_ids"]
    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
   
    # breakpoint()
    # output = trainer.train()
    # print(output)
    # print(prompt)
    # print(generate_text(model, tokenizer, prompt=prompt))
