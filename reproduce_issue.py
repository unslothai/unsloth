
from unsloth import FastLanguageModel
import torch

# Mocking the model loading to avoid downloading large models if possible, 
# but for reproduction we might need a real model. 
# The issue is about shape mismatch, which depends on model config.
# I'll try to use a very small model or just rely on the fact that I can't run it 
# and trust the traceback and code analysis.
# However, the user provided a script. I will save it as reproduce_issue.py.

max_seq_length = 4096
dtype = None
load_in_4bit = False

# Using a small model if available, otherwise the one from the issue
model_name = "unsloth/Phi-3-mini-4k-instruct" 

print(f"Loading model {model_name}...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        attn_implementation="flash_attention_2",
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    # Fallback to a smaller model or cpu if cuda not available (though issue is cuda specific likely)
    exit(1)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

print("First generation...")
model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=50, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
print("Second generation with past_key_values...")
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

# This is where it fails
try:
    generated_output = model.generate(
        **model_inputs, 
        max_new_tokens=50, 
        return_dict_in_generate=True, 
        past_key_values=generated_output.past_key_values
    )
    text_output = tokenizer.batch_decode(generated_output.sequences)[0]
    print(text_output)
    print("Success!")
except Exception as e:
    print(f"Caught expected exception: {e}")
    import traceback
    traceback.print_exc()
