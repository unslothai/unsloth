from unsloth import FastLanguageModel, FastModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
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
    model_name = "unsloth/whisper-large-v3",
    dtype = None,  # Leave as None for auto detection
    load_in_4bit = False,  # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


base_model_class = model.__class__.__name__
# https://github.com/huggingface/transformers/issues/37172
model.generation_config.input_ids = model.generation_config.forced_decoder_ids
model.generation_config.forced_decoder_ids = None


model = FastModel.get_peft_model(
    model,
    r = 64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
    task_type = None,  # ** MUST set this for Whisper **
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
        model.save_pretrained_merged("whisper", tokenizer)
        print("✅ Model saved and merged successfully without warnings!")
    except Exception as e:
        assert False, f"Model saving/merging failed with exception: {e}"

print(f"\n{'='*80}")
print("🔍 SECTION 5: Loading Model for Inference")
print(f"{'='*80}")


model, tokenizer = FastModel.from_pretrained(
    model_name = "./whisper",
    dtype = None,  # Leave as None for auto detection
    load_in_4bit = False,  # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# model = WhisperForConditionalGeneration.from_pretrained("./whisper")
# processor = WhisperProcessor.from_pretrained("./whisper")

print("✅ Model loaded for inference successfully!")

print(f"\n{'='*80}")
print("🔍 SECTION 6: Downloading Sample Audio File")
print(f"{'='*80}")

audio_url = "https://upload.wikimedia.org/wikipedia/commons/5/5b/Speech_12dB_s16.flac"
audio_file = "Speech_12dB_s16.flac"

try:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(audio_url, headers = headers)
    response.raise_for_status()
    with open(audio_file, "wb") as f:
        f.write(response.content)
    print("✅ Audio file downloaded successfully!")
except Exception as e:
    assert False, f"Failed to download audio file: {e}"

print(f"\n{'='*80}")
print("🔍 SECTION 7: Running Inference")
print(f"{'='*80}")


from transformers import pipeline
import torch

FastModel.for_inference(model)
model.eval()
# Create pipeline without specifying the device
whisper = pipeline(
    "automatic-speech-recognition",
    model = model,
    tokenizer = tokenizer.tokenizer,
    feature_extractor = tokenizer.feature_extractor,
    processor = tokenizer,
    return_language = True,
    torch_dtype = torch.float16,  # Remove the device parameter
)
# Example usage
audio_file = "Speech_12dB_s16.flac"
transcribed_text = whisper(audio_file)
# audio, sr = sf.read(audio_file)
# input_features = processor(audio, return_tensors="pt").input_features
# transcribed_text = model.generate(input_features=input_features)
print(f"📝 Transcribed Text: {transcribed_text['text']}")

## assert that transcribed_text contains The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It's easy to tell the depth of a well. Four hours of steady work faced us.

expected_phrases = [
    "birch canoe slid on the smooth planks",
    "sheet to the dark blue background",
    "easy to tell the depth of a well",
    "Four hours of steady work faced us",
]

transcribed_lower = transcribed_text["text"].lower()
all_phrases_found = all(
    phrase.lower() in transcribed_lower for phrase in expected_phrases
)

assert (
    all_phrases_found
), f"Expected phrases not found in transcription: {transcribed_text['text']}"
print("✅ Transcription contains all expected phrases!")


safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./whisper")
