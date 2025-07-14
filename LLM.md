# Unsloth Library API Reference

**Unsloth** is a library for 2x faster LLM fine-tuning with 80% less VRAM usage. It supports LoRA, full fine-tuning, quantization, and optimized training for various model architectures.

## Installation

```bash
pip install unsloth
```

## Core API Overview

### Main Import
```python
import unsloth
from unsloth import FastLanguageModel, FastVisionModel
```

## 1. Model Loading

### FastLanguageModel
**Primary class for loading and optimizing language models**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",  # Model identifier
    max_seq_length=2048,                          # Maximum sequence length
    dtype=None,                                   # Auto-detect bfloat16/float16
    load_in_4bit=True,                           # Enable 4-bit quantization
    load_in_8bit=False,                          # Enable 8-bit quantization
    full_finetuning=False,                       # Enable full parameter training
    token=None,                                  # HuggingFace token
    device_map="sequential",                     # Device mapping strategy
    rope_scaling=None,                           # RoPE scaling config
    fix_tokenizer=True,                          # Apply tokenizer fixes
    trust_remote_code=False,                     # Allow remote code
    use_gradient_checkpointing="unsloth",        # Gradient checkpointing mode
    resize_model_vocab=None,                     # Resize vocabulary
    revision=None,                               # Model revision/branch
    fast_inference=False,                        # Use vLLM for inference
    gpu_memory_utilization=0.5,                 # GPU memory for vLLM
    random_state=3407,                           # Random seed
    max_lora_rank=64,                           # Maximum LoRA rank
)
```

### FastVisionModel
**For vision-language models**

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    return_logits=False,                         # Return logits in output
    fullgraph=True,                             # No graph breaks
    whisper_language=None,                      # For Whisper models
    whisper_task=None,                          # For Whisper models
)
```

### Model-Specific Classes
```python
# Available model-specific optimizations
from unsloth.models import (
    FastLlamaModel,
    FastMistralModel, 
    FastQwen2Model,
    FastQwen3Model,
    FastGraniteModel,
    FastGemmaModel,
    FastGemma2Model,
    FastCohereModel
)
```

## 2. LoRA Configuration

### Get PEFT Model
```python
model = FastLlamaModel.get_peft_model(
    model,
    r=16,                                       # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],  # Target modules
    lora_alpha=16,                              # LoRA alpha parameter
    lora_dropout=0,                             # LoRA dropout rate
    bias="none",                                # Bias handling
    use_gradient_checkpointing=True,            # Gradient checkpointing
    use_rslora=False,                          # Rank-stabilized LoRA
    modules_to_save=None,                       # Additional modules to save
    init_lora_weights=True,                     # Initialize LoRA weights
)
```

## 3. Training

### Training Arguments
```python
from unsloth import UnslothTrainingArguments

training_args = UnslothTrainingArguments(
    embedding_learning_rate=5e-5,               # Separate LR for embeddings
    learning_rate=2e-4,                         # Learning rate for other params
    # ... standard TrainingArguments parameters
)
```

### Trainer
```python
from unsloth import UnslothTrainer, unsloth_train

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    # ... standard SFTTrainer parameters
)

# Train with gradient accumulation fixes
unsloth_train(trainer)
```

## 4. Chat Templates

### Apply Chat Templates
```python
from unsloth import get_chat_template, apply_chat_template

# Get chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",                     # Template type
    mapping={"role": "role", "content": "content"},
    map_eos_token=True,
    system_message=None,
)

# Apply to dataset
dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template="chatml",
)
```

### Data Standardization
```python
from unsloth import standardize_sharegpt, train_on_responses_only

# Standardize ShareGPT format
dataset = standardize_sharegpt(dataset)

# Train only on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user",
    response_part="<|im_start|>assistant",
)
```

## 5. Model Saving

### Save Model
```python
from unsloth import unsloth_save_model

unsloth_save_model(
    model, tokenizer,
    save_directory="./my_model",
    save_method="lora",                         # "lora", "merged_16bit", "merged_4bit"
    push_to_hub=False,                          # Upload to HF Hub
    token=None,                                 # HF token
    max_shard_size="5GB",                       # Sharding size
    safe_serialization=True,                    # Use safetensors
    commit_message="Trained with Unsloth",     # Hub commit message
    temporary_location="_unsloth_temporary_saved_buffers",
    maximum_memory_usage=0.9,                  # Memory usage limit
)
```

### GGUF Conversion
```python
from unsloth import save_to_gguf, print_quantization_methods

# Show available quantization methods
print_quantization_methods()

# Convert to GGUF
save_to_gguf(
    model_type="llama",                         # Model architecture
    model_dtype="float16",                      # Data type
    is_sentencepiece=False,                     # Tokenizer type
    model_directory="unsloth_finetuned_model",  # Output directory
    quantization_method="q4_k_m",               # Single method or list
    first_conversion=None,                      # Initial conversion format
)
```

## 6. Tokenizer Utilities

### Load and Fix Tokenizer
```python
from unsloth import load_correct_tokenizer, check_tokenizer, add_new_tokens

# Load tokenizer with fixes
tokenizer = load_correct_tokenizer(
    "unsloth/llama-2-7b-bnb-4bit",
    model_max_length=4096,
    padding_side="right",
    token=None,
    trust_remote_code=False,
    fix_tokenizer=True,
)

# Validate tokenizer
check_tokenizer(model, tokenizer, model_max_length=4096)

# Add new tokens
add_new_tokens(tokenizer, ["<special_token>"])
```

## 7. Reinforcement Learning

### DPO/KTO Training
```python
from unsloth import PatchDPOTrainer, PatchKTOTrainer, PatchFastRL

# Patch trainers for Unsloth compatibility
PatchDPOTrainer()
PatchKTOTrainer()

# Patch for vLLM and RL
PatchFastRL(algorithm="ppo", FastLanguageModel=FastLanguageModel)
```

## 8. Utility Functions

### Hardware Checks
```python
from unsloth import is_bfloat16_supported, is_vLLM_available

# Check hardware capabilities
if is_bfloat16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

# Check vLLM availability
if is_vLLM_available():
    fast_inference = True
```

### Model Preparation
```python
from unsloth import prepare_model_for_kbit_training

# Prepare quantized model for training
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    use_reentrant=True,
)
```

## 9. Complete Example

```python
import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from unsloth import get_chat_template, unsloth_save_model, save_to_gguf

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# 3. Setup chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

# 4. Training
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    embedding_learning_rate=5e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# 5. Save model
unsloth_save_model(model, tokenizer, "my_model", save_method="lora")

# 6. Convert to GGUF
save_to_gguf(
    model_type="llama",
    model_dtype="float16",
    quantization_method="q4_k_m"
)
```

## Supported Models

- **Llama**: 3.2, 3.1, 3, 2
- **Mistral**: v0.3, v0.2, v0.1
- **Qwen**: 3, 2.5, 2
- **Gemma**: 3, 2, 1
- **Phi**: 4, 3.5, 3
- **Granite**: Code models
- **Cohere**: Command models
- **Vision**: Llama 3.2 Vision, Qwen 2.5 VL, Pixtral
- **TTS**: Orpheus, CSM, Whisper

## Key Features

- **2x faster training** with 80% less VRAM
- **4-bit and 8-bit quantization** support
- **LoRA and full fine-tuning** capabilities
- **Vision and TTS model** support
- **GGUF export** for llama.cpp
- **Ollama integration** for deployment
- **vLLM support** for fast inference
- **Chat template** utilities
- **Gradient accumulation fixes**
- **Hardware optimization** detection

