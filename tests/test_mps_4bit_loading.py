from unsloth import FastLanguageModel
import torch
import os

def test_loading():
    print("Testing 4-bit Loading on MPS...")
    
    # Force MPS for testing (though FastLanguageModel will detect it)
    model_name = "unsloth/Llama-3.2-1B-Instruct" # Small model
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        
        print("✅ Model loaded successfully.")
        
        # Check if weights are quantized
        q_count = 0
        total_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "weight_quant"):
                q_count += 1
            if isinstance(module, torch.nn.Linear):
                total_count += 1
        
        print(f"Quantized {q_count} out of {total_count} linear layers.")
        
        if q_count > 0:
            print("✅ MLX Quantization applied correctly.")
        else:
            print("❌ No modules were quantized. Check loader logic.")
            
        # Test a forward pass
        inputs = tokenizer("Hello, world!", return_tensors = "pt")
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("✅ Forward pass successful.")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")

if __name__ == "__main__":
    test_loading()
