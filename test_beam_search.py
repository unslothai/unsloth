# Test beam search with a minimal example
from unsloth import FastLanguageModel
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a small model for testing
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
    device_map=device,
)

# Get PEFT model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)

# Test beam search
inputs = tokenizer("Hello, how are", return_tensors="pt").to(device)

# Check model before generate
print(f"\n--- Pre-generate check ---")
print(f"model.__class__.__module__: {model.__class__.__module__}")
print(f"model.__class__.__name__: {model.__class__.__name__}")
print(f"hasattr(model, '_reorder_cache'): {hasattr(model, '_reorder_cache')}")
print(f"hasattr(model.__class__, '_reorder_cache'): {hasattr(model.__class__, '_reorder_cache')}")

# Check if the method is callable
if hasattr(model, '_reorder_cache'):
    print(f"model._reorder_cache is callable: {callable(model._reorder_cache)}")
    print(f"model._reorder_cache: {model._reorder_cache}")

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=2,
            num_return_sequences=2,
        )
        
    print("Beam search successful!")
    for i, output in enumerate(outputs):
        print(f"Sequence {i}: {tokenizer.decode(output, skip_special_tokens=True)}")
except Exception as e:
    print(f"Error during beam search: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
# Debug info
print("\n--- Debug Info ---")
print(f"Model class: {model.__class__}")
print(f"Model module: {model.__class__.__module__}")
print(f"Has _reorder_cache: {hasattr(model, '_reorder_cache')}")
if hasattr(model, 'base_model'):
    print(f"Base model class: {model.base_model.__class__}")
    print(f"Base model has _reorder_cache: {hasattr(model.base_model, '_reorder_cache')}")
    
# Check the transformers module namespace
import transformers.models.llama.modeling_llama as llama_module
if hasattr(llama_module, 'LlamaForCausalLM'):
    print(f"LlamaForCausalLM in transformers has _reorder_cache: {hasattr(llama_module.LlamaForCausalLM, '_reorder_cache')}")