import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import time

def test_mps_finetuning():
    print("\n--- Testing MPS Finetuning (16-bit LoRA) ---")
    
    model_name = "unsloth/Llama-3.2-1B"
    max_seq_length = 512
    
    # Load model and tokenizer
    # We use load_in_4bit=False because bnb is not supported on MPS
    # Unsloth will fallback to FP16/BF16 automatically
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = False, 
        dtype = None, # Auto detect (BF16 on M-series)
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Create dummy dataset
    data = {
        "text": [
            "What is the capital of France? ### Paris",
            "What is 2+2? ### 4",
            "Who wrote Romeo and Juliet? ### William Shakespeare",
        ]
    }
    dataset = Dataset.from_dict(data)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1,
        warmup_steps = 1,
        max_steps = 3, # Just 3 steps to verify it works
        learning_rate = 2e-4,
        fp16 = not torch.backends.mps.is_available(), # torch stuff
        bf16 = torch.backends.mps.is_available(),
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    )

    # Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = training_args,
    )

    print("Starting training...")
    start_time = time.time()
    trainer.train()
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Verification: Check if some weights changed or at least it didn't crash
    print("✅ Success: Finetuning run completed on MPS.")

if __name__ == "__main__":
    test_mps_finetuning()
