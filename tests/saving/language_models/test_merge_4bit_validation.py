from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from datasets import load_dataset
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return {"text": texts}


print(f"\n{'='*80}")
print("🔍 PHASE 1: Loading Base Model and Initial Training")
print(f"{'='*80}")

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    dtype = compute_dtype,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    attn_implementation = attn_implementation,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# Load small dataset for quick training
dataset_train = load_dataset(
    "allenai/openassistant-guanaco-reformatted", split = "train[:100]"
)
dataset_train = dataset_train.map(formatting_prompts_func, batched = True)

print("✅ Base model loaded successfully!")

print(f"\n{'='*80}")
print("🔍 PHASE 2: First Fine-tuning")
print(f"{'='*80}")

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        max_steps = 10,  # Very short training for test
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()
print("✅ First fine-tuning completed!")

print(f"\n{'='*80}")
print("🔍 PHASE 3: Save with Forced 4bit Merge")
print(f"{'='*80}")

model.save_pretrained_merged(
    save_directory = "./test_4bit_model",
    tokenizer = tokenizer,
    save_method = "forced_merged_4bit",
)

print("✅ Model saved with forced 4bit merge!")

print(f"\n{'='*80}")
print("🔍 PHASE 4: Loading 4bit Model and Second Fine-tuning")
print(f"{'='*80}")

# Clean up first model
del model
del tokenizer
torch.cuda.empty_cache()

# Load the 4bit merged model
model_4bit, tokenizer_4bit = FastLanguageModel.from_pretrained(
    model_name = "./test_4bit_model",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
)

tokenizer_4bit = get_chat_template(
    tokenizer_4bit,
    chat_template = "llama-3.1",
)

print("✅ 4bit model loaded successfully!")

# Add LoRA adapters to the 4bit model
model_4bit = FastLanguageModel.get_peft_model(
    model_4bit,
    r = 16,
    target_modules = [
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Second fine-tuning
trainer_4bit = SFTTrainer(
    model = model_4bit,
    tokenizer = tokenizer_4bit,
    train_dataset = dataset_train,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer_4bit),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        max_steps = 10,  # Very short training for test
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_4bit",
        report_to = "none",
    ),
)

trainer_4bit.train()
print("✅ Second fine-tuning on 4bit model completed!")

print(f"\n{'='*80}")
print("🔍 PHASE 5: Testing TypeError on Regular Merge (Should Fail)")
print(f"{'='*80}")

try:
    model_4bit.save_pretrained_merged(
        save_directory = "./test_should_fail",
        tokenizer = tokenizer_4bit,
        # No save_method specified, should default to regular merge
    )
    assert False, "Expected TypeError but merge succeeded!"
except TypeError as e:
    expected_error = "Base model should be a 16bits or mxfp4 base model for a 16bit model merge. Use `save_method=forced_merged_4bit` instead"
    assert expected_error in str(e), f"Unexpected error message: {str(e)}"
    print("✅ Correct TypeError raised for 4bit base model regular merge attempt!")
    print(f"Error message: {str(e)}")

print(f"\n{'='*80}")
print("🔍 PHASE 6: Successful Save with Forced 4bit Method")
print(f"{'='*80}")

try:
    model_4bit.save_pretrained_merged(
        save_directory = "./test_4bit_second",
        tokenizer = tokenizer_4bit,
        save_method = "forced_merged_4bit",
    )
    print("✅ Successfully saved 4bit model with forced 4bit method!")
except Exception as e:
    assert False, f"Phase 6 failed unexpectedly: {e}"

print(f"\n{'='*80}")
print("🔍 CLEANUP")
print(f"{'='*80}")

# Cleanup
safe_remove_directory("./outputs")
safe_remove_directory("./outputs_4bit")
safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./test_4bit_model")
safe_remove_directory("./test_4bit_second")
safe_remove_directory("./test_should_fail")

print("✅ All tests passed successfully!")
