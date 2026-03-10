from unsloth import FastLanguageModel, FastModel
from transformers import CsmForConditionalGeneration
import torch

# ruff: noqa
import sys
from pathlib import Path
from peft import PeftModel
import warnings
import requests

REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.os_utils import require_package, require_python_package

require_package("ffmpeg", "ffmpeg")
require_python_package("soundfile")
require_python_package("snac")

import soundfile as sf
from snac import SNAC

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")
print(f"\n{'='*80}")
print("🔍 SECTION 1: Loading Model and LoRA Adapters")
print(f"{'='*80}")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,  # Choose any for long context!
    dtype = None,  # Select None for auto detection
    load_in_4bit = False,  # Select True for 4bit which reduces memory usage
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

base_model_class = model.__class__.__name__


model = FastLanguageModel.get_peft_model(
    model,
    r = 64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 64,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)
print("✅ Model and LoRA adapters loaded successfully!")


print(f"\n{'='*80}")
print("🔍 SECTION 2: Checking Model Class Type")
print(f"{'='*80}")

assert isinstance(model, PeftModel), "Model should be an instance of PeftModel"
print("✅ Model is an instance of PeftModel!")


print(f"\n{'='*80}")
print("🔍 SECTION 3: Checking Config Model Class Type")
print(f"{'='*80}")


def find_lora_base_model(model_to_inspect):
    current = model_to_inspect
    if hasattr(current, "base_model"):
        current = current.base_model
    if hasattr(current, "model"):
        current = current.model
    return current


config_model = find_lora_base_model(model) if isinstance(model, PeftModel) else model

assert (
    config_model.__class__.__name__ == base_model_class
), f"Expected config_model class to be {base_model_class}"
print("✅ config_model returns correct Base Model class:", str(base_model_class))


print(f"\n{'='*80}")
print("🔍 SECTION 4: Saving and Merging Model")
print(f"{'='*80}")

with warnings.catch_warnings():
    warnings.simplefilter("error")  # Treat warnings as errors
    try:
        model.save_pretrained_merged("orpheus", tokenizer)
        print("✅ Model saved and merged successfully without warnings!")
    except Exception as e:
        assert False, f"Model saving/merging failed with exception: {e}"

print(f"\n{'='*80}")
print("🔍 SECTION 5: Loading Model for Inference")
print(f"{'='*80}")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,  # Choose any for long context!
    dtype = None,  # Select None for auto detection
    load_in_4bit = False,  # Select True for 4bit which reduces memory usage
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# from transformers import AutoProcessor
# processor = AutoProcessor.from_pretrained("unsloth/csm-1b")

print("✅ Model loaded for inference successfully!")


print(f"\n{'='*80}")
print("🔍 SECTION 6: Running Inference")
print(f"{'='*80}")


# @title Run Inference


FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Moving snac_model cuda to cpu
snac_model.to("cpu")
prompts = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
]

chosen_voice = None  # None for single-speaker

prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

all_input_ids = []

for prompt in prompts_:
    input_ids = tokenizer(prompt, return_tensors = "pt").input_ids
    all_input_ids.append(input_ids)

start_token = torch.tensor([[128259]], dtype = torch.int64)  # Start of human
end_tokens = torch.tensor(
    [[128009, 128260]], dtype = torch.int64
)  # End of text, End of human

all_modified_input_ids = []
for input_ids in all_input_ids:
    modified_input_ids = torch.cat(
        [start_token, input_ids, end_tokens], dim = 1
    )  # SOH SOT Text EOT EOH
    all_modified_input_ids.append(modified_input_ids)

all_padded_tensors = []
all_attention_masks = []
max_length = max(
    [modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids]
)
for modified_input_ids in all_modified_input_ids:
    padding = max_length - modified_input_ids.shape[1]
    padded_tensor = torch.cat(
        [torch.full((1, padding), 128263, dtype = torch.int64), modified_input_ids], dim = 1
    )
    attention_mask = torch.cat(
        [
            torch.zeros((1, padding), dtype = torch.int64),
            torch.ones((1, modified_input_ids.shape[1]), dtype = torch.int64),
        ],
        dim = 1,
    )
    all_padded_tensors.append(padded_tensor)
    all_attention_masks.append(attention_mask)

all_padded_tensors = torch.cat(all_padded_tensors, dim = 0)
all_attention_masks = torch.cat(all_attention_masks, dim = 0)

input_ids = all_padded_tensors.to("cuda")
attention_mask = all_attention_masks.to("cuda")
generated_ids = model.generate(
    input_ids = input_ids,
    attention_mask = attention_mask,
    max_new_tokens = 1200,
    do_sample = True,
    temperature = 0.6,
    top_p = 0.95,
    repetition_penalty = 1.1,
    num_return_sequences = 1,
    eos_token_id = 128258,
    use_cache = True,
)
token_to_find = 128257
token_to_remove = 128258

token_indices = (generated_ids == token_to_find).nonzero(as_tuple = True)

if len(token_indices[1]) > 0:
    last_occurrence_idx = token_indices[1][-1].item()
    cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
else:
    cropped_tensor = generated_ids

mask = cropped_tensor != token_to_remove

processed_rows = []

for row in cropped_tensor:
    masked_row = row[row != token_to_remove]
    processed_rows.append(masked_row)

code_lists = []

for row in processed_rows:
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = row[:new_length]
    trimmed_row = [t - 128266 for t in trimmed_row]
    code_lists.append(trimmed_row)


def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]

    # codes = [c.to("cuda") for c in codes]
    audio_hat = snac_model.decode(codes)
    return audio_hat


my_samples = []
for code_list in code_lists:
    samples = redistribute_codes(code_list)
    my_samples.append(samples)
output_path = "orpheus_audio.wav"
try:
    for i, samples in enumerate(my_samples):
        audio_data = samples.detach().squeeze().cpu().numpy()
        import soundfile as sf

        sf.write(output_path, audio_data, 24000)  # Explicitly pass sample rate
        print(f"✅ Audio saved to {output_path}!")
except Exception as e:
    assert False, f"Inference failed with exception: {e}"

# Verify the file exists
import os

assert os.path.exists(output_path), f"Audio file not found at {output_path}"
print("✅ Audio file exists on disk!")
del my_samples, samples
## assert that transcribed_text contains The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It's easy to tell the depth of a well. Four hours of steady work faced us.

print("✅ All sections passed successfully!")


safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./orpheus")
