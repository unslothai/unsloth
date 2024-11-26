from unsloth.models import FastCausalModel
from transformers import AutoTokenizer
import torch

def test_causal_model():
    print("Starting FastCausalModel test...")
    
    try:
        # First create tokenizer
        print("\nCreating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        # Create model with custom config
        print("\nCreating model...")
        model, tokenizer, config = FastCausalModel.create_model(
            tokenizer=tokenizer,  # Pass the tokenizer
            hidden_size=768,  # Smaller for testing
            num_hidden_layers=12,
            num_attention_heads=12,
            max_seq_length=2048,
            intermediate_size=3072,
            dtype=torch.float32,
        )
        
        # Print model info
        print("\nModel Configuration:")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Number of layers: {config.num_hidden_layers}")
        print(f"Number of attention heads: {config.num_attention_heads}")
        print(f"Model size: {sum(p.numel() for p in model.parameters())/1000**2:.1f}M parameters")
        
        # Test tokenizer
        print("\nTesting tokenizer...")
        test_text = "Testing the causal model:"
        tokens = tokenizer(test_text, return_tensors="pt")
        print("encoded_tokens",tokens)
        print(f"Input text: {test_text}")
        print(f"Token shape: {tokens['input_ids'].shape}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise e

if __name__ == "__main__":
    test_causal_model()