import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsloth.kernels.utils import profile_generate_method

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)


generate_args = {
    **inputs,  # Assuming model_inputs is a dictionary with appropriate keys
    "max_new_tokens": 100,
    "do_sample": True
}

# Ensure your model and tokenizer are properly loaded and set up as before.

# Now, call the profile_generate_method function
prof = profile_generate_method(model, generate_args)

