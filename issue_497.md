# Issue 497: Allow passing in custom `past_key_values`

**Author:** Nisimachluf

## Description

I'm trying to use KV caching with phi3-unsloth model from the HF hub (unsloth/Phi-3-mini-4k-instruct)
How ever it seems that the FastLanguageModel class doesn't suuprt KV caching.
Here is a toy exmaple of asking it a question, and folow it's reply with another question.

```
from unsloth import FastLanguageModel

max_seq_length = 4096  # Can be set arbitrarily, automatically supports RoPE scaling!
dtype = None  # Automatically detect if None. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Reduce memory usage using 4-bit quantization. Can be set to False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/local/models/phi3_unsloth",  # Use "unsloth/mistral-7b" for 16-bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)
```

The second call to model.generate() fails with
```
Traceback (most recent call last):
  File "phi3_unsloth_toy.py", line 31, in <module>
    generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 1736, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2375, in _sample
    outputs = self(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/mistral.py", line 205, in MistralForCausalLM_fast_forward
    outputs = LlamaModel_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 748, in LlamaModel_fast_forward_inference
    hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 154, in LlamaAttention_fast_forward_inference
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
RuntimeError: shape '[1, 1, 32, 96]' is invalid for input of size 61440
```

Works well if not using past_key_values.


## Comments

### danielhanchen
I will check this! Sorry on the issue!


# Issue 497: Allow passing in custom `past_key_values`

**Author:** Nisimachluf

## Description

I'm trying to use KV caching with phi3-unsloth model from the HF hub (unsloth/Phi-3-mini-4k-instruct)
How ever it seems that the FastLanguageModel class doesn't suuprt KV caching.
Here is a toy exmaple of asking it a question, and folow it's reply with another question.

```
from unsloth import FastLanguageModel

max_seq_length = 4096  # Can be set arbitrarily, automatically supports RoPE scaling!
dtype = None  # Automatically detect if None. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Reduce memory usage using 4-bit quantization. Can be set to False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/local/models/phi3_unsloth",  # Use "unsloth/mistral-7b" for 16-bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)
```

The second call to model.generate() fails with
```
Traceback (most recent call last):
  File "phi3_unsloth_toy.py", line 31, in <module>
    generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 1736, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2375, in _sample
    outputs = self(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/mistral.py", line 205, in MistralForCausalLM_fast_forward
    outputs = LlamaModel_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 748, in LlamaModel_fast_forward_inference
    hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 154, in LlamaAttention_fast_forward_inference
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
RuntimeError: shape '[1, 1, 32, 96]' is invalid for input of size 61440
```

Works well if not using past_key_values.


## Comments

### spisupat
Hi @danielhanchen I'm also running into this issue with the `unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit` model - the past_key_values work if I load up the exact same (unsloth tuned) model as an AutoModelForCausalLM instead. 

Here's an example code snippet that works with AutoModelForCausalLM but not with a FastLanguageModel:

```
# Step 1: Encode the instruction
instruction = "Please give me the next number in the sequence"
instruction_input_ids = tokenizer.encode(instruction, return_tensors='pt')
with torch.no_grad():
    instruction_outputs = model(input_ids=instruction_input_ids)
    past_key_values = instruction_outputs.past_key_values

# Step 2: Encode the user message
user = "1, "
user_input_ids = tokenizer.encode(user, return_tensors='pt')
with torch.no_grad():
    user_message_outputs = model(input_ids=user_input_ids, past_key_values=past_key_values)
    logits = user_message_outputs.logits
    
print(tokenizer.decode(torch.argmax(logits[:, -1, :], dim=-1)))
```


# Issue 497: Allow passing in custom `past_key_values`

**Author:** Nisimachluf

## Description

I'm trying to use KV caching with phi3-unsloth model from the HF hub (unsloth/Phi-3-mini-4k-instruct)
How ever it seems that the FastLanguageModel class doesn't suuprt KV caching.
Here is a toy exmaple of asking it a question, and folow it's reply with another question.

```
from unsloth import FastLanguageModel

max_seq_length = 4096  # Can be set arbitrarily, automatically supports RoPE scaling!
dtype = None  # Automatically detect if None. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Reduce memory usage using 4-bit quantization. Can be set to False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/local/models/phi3_unsloth",  # Use "unsloth/mistral-7b" for 16-bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)
```

The second call to model.generate() fails with
```
Traceback (most recent call last):
  File "phi3_unsloth_toy.py", line 31, in <module>
    generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 1736, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2375, in _sample
    outputs = self(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/mistral.py", line 205, in MistralForCausalLM_fast_forward
    outputs = LlamaModel_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 748, in LlamaModel_fast_forward_inference
    hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 154, in LlamaAttention_fast_forward_inference
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
RuntimeError: shape '[1, 1, 32, 96]' is invalid for input of size 61440
```

Works well if not using past_key_values.


## Comments

### danielhanchen
Oh why is `past_key_values = instruction_outputs.past_key_values` there? The KV Cache should be a list of 2 matrices (K and V)


# Issue 497: Allow passing in custom `past_key_values`

**Author:** Nisimachluf

## Description

I'm trying to use KV caching with phi3-unsloth model from the HF hub (unsloth/Phi-3-mini-4k-instruct)
How ever it seems that the FastLanguageModel class doesn't suuprt KV caching.
Here is a toy exmaple of asking it a question, and folow it's reply with another question.

```
from unsloth import FastLanguageModel

max_seq_length = 4096  # Can be set arbitrarily, automatically supports RoPE scaling!
dtype = None  # Automatically detect if None. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Reduce memory usage using 4-bit quantization. Can be set to False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/local/models/phi3_unsloth",  # Use "unsloth/mistral-7b" for 16-bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)
```

The second call to model.generate() fails with
```
Traceback (most recent call last):
  File "phi3_unsloth_toy.py", line 31, in <module>
    generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 1736, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2375, in _sample
    outputs = self(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/mistral.py", line 205, in MistralForCausalLM_fast_forward
    outputs = LlamaModel_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 748, in LlamaModel_fast_forward_inference
    hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 154, in LlamaAttention_fast_forward_inference
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
RuntimeError: shape '[1, 1, 32, 96]' is invalid for input of size 61440
```

Works well if not using past_key_values.


## Comments

### rscmendes
I have the same problem with unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit when using `past_key_values`. 

As @spisupat mentioned, using AutoModelForCasualLM works, but FastLanguageModel doesn't .





# Issue 497: Allow passing in custom `past_key_values`

**Author:** Nisimachluf

## Description

I'm trying to use KV caching with phi3-unsloth model from the HF hub (unsloth/Phi-3-mini-4k-instruct)
How ever it seems that the FastLanguageModel class doesn't suuprt KV caching.
Here is a toy exmaple of asking it a question, and folow it's reply with another question.

```
from unsloth import FastLanguageModel

max_seq_length = 4096  # Can be set arbitrarily, automatically supports RoPE scaling!
dtype = None  # Automatically detect if None. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Reduce memory usage using 4-bit quantization. Can be set to False.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/local/models/phi3_unsloth",  # Use "unsloth/mistral-7b" for 16-bit loading
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support

)

prompt = """<|user|>
My name name is Jon. What is my name?<|end|>
<|assistant|>"""

model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, temperature=0)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)

second_prompt = """
<|user|>
I'm 30 years old. How old am i?<|end|>
<|assistant|>"""

full_prompt = text_output + second_prompt
model_inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
text_output = tokenizer.batch_decode(generated_output.sequences)[0]
print(text_output)
```

The second call to model.generate() fails with
```
Traceback (most recent call last):
  File "phi3_unsloth_toy.py", line 31, in <module>
    generated_output = model.generate(**model_inputs, max_new_tokens=500, return_dict_in_generate=True, past_key_values=generated_output.past_key_values)
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 1736, in generate
    result = self._sample(
  File "/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py", line 2375, in _sample
    outputs = self(
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/mistral.py", line 205, in MistralForCausalLM_fast_forward
    outputs = LlamaModel_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 748, in LlamaModel_fast_forward_inference
    hidden_states, present_key_value = LlamaAttention_fast_forward_inference(
  File "/usr/local/lib/python3.10/dist-packages/unsloth/models/llama.py", line 154, in LlamaAttention_fast_forward_inference
    Qn = Qn.view(bsz, 1, n_heads,    head_dim).transpose(1, 2)
RuntimeError: shape '[1, 1, 32, 96]' is invalid for input of size 61440
```

Works well if not using past_key_values.


## Comments

### danielhanchen
Ok it seems like all past_key_values won't function - I think this will have to be a feature request


