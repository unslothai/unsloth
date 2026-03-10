# inference_on_merged.py
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import gc
import os
import shutil


def safe_remove_directory(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            return True
        else:
            print(f"Path {path} is not a valid directory")
            return False
    except Exception as e:
        print(f"Failed to remove directory {path}: {e}")
        return False


print("🔥 Loading the 16-bit merged model from disk...")
merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./gpt-oss-finetuned-merged",
    max_seq_length = 1024,
    load_in_4bit = True,
    load_in_8bit = False,
)
print("✅ Merged model loaded successfully.")

# --- Run Inference ---
print("\n🚀 Running inference...")
messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = merged_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "low",  # **NEW!** Set reasoning effort to low, medium or high
).to(merged_model.device)

_ = merged_model.generate(
    **inputs, max_new_tokens = 512, streamer = TextStreamer(merged_tokenizer)
)
print("\n✅ Inference complete.")

# --- Final Cleanup ---
print("\n🧹 Cleaning up merged model directory and cache...")
del merged_model, merged_tokenizer
torch.cuda.empty_cache()
gc.collect()

safe_remove_directory("./gpt-oss-finetuned-merged")
safe_remove_directory(
    "./unsloth_compiled_cache"
)  # Clean up cache created by this process
print("✅ Final cleanup complete. Exiting inference script.")
