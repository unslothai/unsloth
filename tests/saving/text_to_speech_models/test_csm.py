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

import soundfile as sf

print(f"\n{'='*80}")
print("🔍 SECTION 1: Loading Model and LoRA Adapters")
print(f"{'='*80}")


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/csm-1b",
    max_seq_length = 2048,  # Choose any for long context!
    dtype = None,  # Leave as None for auto-detection
    auto_model = CsmForConditionalGeneration,
    load_in_4bit = False,  # Select True for 4bit - reduces memory usage
)


base_model_class = model.__class__.__name__


model = FastModel.get_peft_model(
    model,
    r = 32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 32,
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
        model.save_pretrained_merged("csm", tokenizer)
        print("✅ Model saved and merged successfully without warnings!")
    except Exception as e:
        assert False, f"Model saving/merging failed with exception: {e}"

print(f"\n{'='*80}")
print("🔍 SECTION 5: Loading Model for Inference")
print(f"{'='*80}")


model, processor = FastModel.from_pretrained(
    model_name = "./csm",
    max_seq_length = 2048,  # Choose any for long context!
    dtype = None,  # Leave as None for auto-detection
    auto_model = CsmForConditionalGeneration,
    load_in_4bit = False,  # Select True for 4bit - reduces memory usage
)

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("unsloth/csm-1b")

print("✅ Model loaded for inference successfully!")


print(f"\n{'='*80}")
print("🔍 SECTION 6: Running Inference")
print(f"{'='*80}")


from transformers import pipeline
import torch

output_audio_path = "csm_audio.wav"
try:
    text = (
        "We just finished fine tuning a text to speech model... and it's pretty good!"
    )
    speaker_id = 0
    inputs = processor(f"[{speaker_id}]{text}", add_special_tokens = True).to("cuda")
    audio_values = model.generate(
        **inputs,
        max_new_tokens = 125,  # 125 tokens is 10 seconds of audio, for longer speech increase this
        # play with these parameters to get the best results
        depth_decoder_temperature = 0.6,
        depth_decoder_top_k = 0,
        depth_decoder_top_p = 0.9,
        temperature = 0.8,
        top_k = 50,
        top_p = 1.0,
        #########################################################
        output_audio = True,
    )
    audio = audio_values[0].to(torch.float32).cpu().numpy()
    sf.write("example_without_context.wav", audio, 24000)
    print(f"✅ Audio generated and saved to {output_audio_path}!")
except Exception as e:
    assert False, f"Inference failed with exception: {e}"


## assert that transcribed_text contains The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It's easy to tell the depth of a well. Four hours of steady work faced us.

print("✅ All sections passed successfully!")


safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./csm")
