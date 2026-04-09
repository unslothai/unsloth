## Import required libraries

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

import torch
import os
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfFileSystem
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory


## Dataset Preparation"""

print("\n📊 Loading and preparing dataset...")
dataset = load_dataset("lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean", "en", split = "train")
# To select the first 2000 examples
train_dataset = dataset.select(range(2000))

# To select the next 200 examples for evaluation
eval_dataset = dataset.select(range(2000, 2200))

print(f"✅ Dataset loaded successfully!")
print(f"   📈 Training samples: {len(train_dataset)}")
print(f"   📊 Evaluation samples: {len(eval_dataset)}")


# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ],
    }


print("\n🔄 Formatting dataset for vision training...")
system_message = "You are an expert french ocr system."
# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
print("✅ Dataset formatting completed!")

"""## Finetuning Setup and Run"""


print("\n" + "=" * 80)
print("=== MODEL LOADING AND SETUP ===".center(80))
print("=" * 80 + "\n")
# Load Base Model
print("🤖 Loading base vision model...")
try:
    model, tokenizer = FastVisionModel.from_pretrained(
        # model_name = "unsloth/Qwen2-VL-7B-Instruct",
        model_name = "unsloth/Qwen2-VL-7B-Instruct",
        max_seq_length = 2048,  # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False,  # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False,  # [NEW!] We have full finetuning now!
    )
except Exception as e:
    print(f"❌ Failed to load base model: {e}")
    raise

print("\n🔧 Setting up LoRA configuration...")
## Lora Finetuning
try:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = True,  # Turn off for just text!
        finetune_language_layers = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules = True,  # SHould leave on always!
        r = 16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        lora_alpha = 32,
        lora_dropout = 0,  # Supports any, but = 0 is optimized
        bias = "none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )
    print("✅ LoRA configuration applied successfully!")
    print(f"   🎯 LoRA rank (r): 16")
    print(f"   📊 LoRA alpha: 32")
    print(f"   🔍 Vision layers: Enabled")
    print(f"   💬 Language layers: Enabled")
except Exception as e:
    print(f"❌ Failed to apply LoRA configuration: {e}")
    raise

print("\n" + "=" * 80)
print("=== TRAINING SETUP ===".center(80))
print("=" * 80 + "\n")


print("🏋️ Preparing trainer...")
FastVisionModel.for_training(model)  # Enable for training!

try:
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer),
        train_dataset = train_dataset,
        args = SFTConfig(
            # per_device_train_batch_size = 4,
            # gradient_accumulation_steps = 8,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            gradient_checkpointing = True,
            gradient_checkpointing_kwargs = {
                "use_reentrant": False
            },  # use reentrant checkpointing
            max_grad_norm = 0.3,  # max gradient norm based on QLoRA paper
            warmup_ratio = 0.03,
            # num_train_epochs = 2, # Set this instead of max_steps for full training runs
            max_steps = 10,
            learning_rate = 2e-4,
            fp16 = not is_bf16_supported(),
            bf16 = is_bf16_supported(),
            logging_steps = 5,
            save_strategy = "epoch",
            optim = "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "checkpoints",
            report_to = "none",  # For Weights and Biases
            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            dataset_num_proc = 4,
            max_seq_length = 2048,
        ),
    )
    print("✅ Trainer setup completed!")
    print(f"   📦 Batch size: 2")
    print(f"   🔄 Gradient accumulation steps: 4")
    print(f"   📈 Max training steps: 10")
    print(f"   🎯 Learning rate: 2e-4")
    print(f"   💾 Precision: {'BF16' if is_bf16_supported() else 'FP16'}")
except Exception as e:
    print(f"❌ Failed to setup trainer: {e}")
    raise

print("\n" + "=" * 80)
print("=== STARTING TRAINING ===".center(80))
print("=" * 80 + "\n")
# run training
try:
    print("🚀 Starting training process...")
    trainer_stats = trainer.train()
except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

print("\n" + "=" * 80)
print("=== SAVING MODEL ===".center(80))
print("=" * 80 + "\n")

print("💾 Saving adapter model and tokenizer locally...")
try:
    model.save_pretrained("unsloth-qwen2-7vl-french-ocr-adapter", tokenizer)
    tokenizer.save_pretrained("unsloth-qwen2-7vl-french-ocr-adapter")
    print("✅ Model saved locally!")
except Exception as e:
    print(f"❌ Failed to save model locally: {e}")
    raise


hf_username = os.environ.get("HF_USER", "")
if not hf_username:
    hf_username = input("Please enter your Hugging Face username: ").strip()
    os.environ["HF_USER"] = hf_username

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    hf_token = input("Please enter your Hugging Face token: ").strip()
    os.environ["HF_TOKEN"] = hf_token

repo_name = f"{hf_username}/qwen2-7b-ocr-merged"
success = {
    "upload": False,
    "safetensors_check": False,
    "download": False,
}
# Stage 1: Upload model to Hub
try:
    print("\n" + "=" * 80)
    print("=== UPLOADING MODEL TO HUB ===".center(80))
    print("=" * 80 + "\n")
    print(f"🚀 Uploading to repository: {repo_name}")
    model.push_to_hub_merged(repo_name, tokenizer = tokenizer, token = hf_token)
    success["upload"] = True
    print("✅ Model uploaded successfully!")
except Exception as e:
    print(f"❌ Failed to upload model: {e}")
    raise Exception("Model upload failed.")

# Stage 2: Verify safetensors.index.json exists
try:
    print("\n" + "=" * 80)
    print("=== VERIFYING REPO CONTENTS ===".center(80))
    print("=" * 80 + "\n")
    fs = HfFileSystem(token = hf_token)
    file_list = fs.ls(repo_name, detail = True)
    safetensors_found = any(
        file["name"].endswith("model.safetensors.index.json") for file in file_list
    )
    if safetensors_found:
        success["safetensors_check"] = True
        print("✅ model.safetensors.index.json found in repo!")
    else:
        raise Exception("model.safetensors.index.json not found in repo.")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    raise Exception("Repo verification failed.")

# test downloading model even if cached
safe_remove_directory(f"./{hf_username}")

try:
    print("\n" + "=" * 80)
    print("=== TESTING MODEL DOWNLOAD ===".center(80))
    print("=" * 80 + "\n")
    print("📥 Testing model download...")
    # Force download even if cached
    test_model, test_tokenizer = FastVisionModel.from_pretrained(repo_name)
    success["download"] = True
    print("✅ Model downloaded successfully!")

    # Clean up test model
    del test_model, test_tokenizer
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ Download failed: {e}")
    raise Exception("Model download failed.")

# Final report
print("\n" + "=" * 80)
print("=== VALIDATION REPORT ===".center(80))
print("=" * 80 + "\n")
for stage, passed in success.items():
    status = "✅" if passed else "❌"
    print(f"{status} {stage.replace('_', ' ').title()}")
print("\n" + "=" * 80)

if all(success.values()):
    print("\n🎉 All stages completed successfully!")
    print(f"🌐 Your model is available at: https://huggingface.co/{repo_name}")
else:
    raise Exception("Validation failed for one or more stages.")


# Final cleanup
print("\n🧹 Cleaning up temporary files...")
safe_remove_directory("./checkpoints")
safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./unsloth-qwen2-7vl-french-ocr-adapter")

print("\n🎯 Pipeline completed successfully!")
print("=" * 80)
