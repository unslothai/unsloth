import unsloth
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only,standardize_sharegpt

import os
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from trl import SFTTrainer
from accelerate import Accelerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #Select Which devices to use. Or, comment if you want to use all GPUs.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
accelerator = Accelerator()


# to be used in the terminal
#   accelerate launch --config_file acc_config.yaml unsloth_Accelerate.py


device = accelerator.device


def load_model(model_path):
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map=device_map,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    return model, tokenizer

def model_LoRA(base_model):
    model = FastLanguageModel.get_peft_model(
        base_model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128, USE 8
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                 ],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16, # USE 32
        lora_dropout = 0, # Supports any, but = 0 is optimized USE 0.3
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        # use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


    return model

### FUNCTION CALLING
# model_path = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit"
model_path = "unsloth/Qwen2.5-Coder-7B-Instruct"
model, tokenizer = load_model(model_path=model_path)


##APPLY LORA
model = model_LoRA(base_model=model)

def load_data(data_path):
    dataset_train = load_dataset(data_path, split = "train")

    return dataset_train

data_path = "mlabonne/FineTome-100k"
dataset_train = load_data(data_path=data_path)



def split_train_val(dataset):
    # Split training dataset into train and validation sets (80-20 split)
    train_test_split = dataset.train_test_split(test_size=0.1238, seed=42)
    dataset_train = train_test_split["train"]
    dataset_val = train_test_split["test"]  # This becomes the validation set

    return dataset_train, dataset_val

dataset_train, dataset_val = split_train_val(dataset_train)

# Initialize the tokenizer with Qwen2.5 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


dataset_train = standardize_sharegpt(dataset_train)
dataset_val = standardize_sharegpt(dataset_val)



# Apply formatting
dataset_train = dataset_train.map(formatting_prompts_func, batched=True,)
dataset_val = dataset_val.map(formatting_prompts_func, batched=True,)


def def_trainer(model, tokenizer, dataset_train):

    # Create the TrainingArguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        max_steps=30,
        # num_train_epochs=20,  # Set to 20 epochs
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Disable WandB or other reporting
        greater_is_better=False,
        # load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
    )

    # Define the Trainer with the given parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    tokenizer.decode(trainer.train_dataset[5]["input_ids"])

    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

     
    # If the model is wrapped in DDP, access the underlying module:
    if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "_set_static_graph"):
        trainer.model.module._set_static_graph()
    elif hasattr(trainer.model, "_set_static_graph"):
        trainer.model._set_static_graph()
    return trainer

trainer = def_trainer(model=model,tokenizer=tokenizer,dataset_train=dataset_train)
trainer_stats = trainer.train()