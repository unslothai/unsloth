# ruff: noqa
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from tests.utils.os_utils import require_opt_in as _require_opt_in

_require_opt_in(
    "UNSLOTH_RUN_SAVING_SCRIPTS",
    "GPU + Hub saving script; its body runs at import.",
)

import pytest

try:
    from unsloth import FastLanguageModel, FastModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch
    from peft import PeftModel
    import requests
except ImportError as exc:
    # Imported at collection time, so an absent runtime dep (triton on the Windows CI runner) is a collection error that
    # reports no results at all.
    pytest.skip(
        f"requires the full unsloth runtime: {exc}",
        allow_module_level = True,
    )

import sys
from pathlib import Path
import warnings


REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))


from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.os_utils import require_package, require_python_package

require_package("ffmpeg", "ffmpeg")
require_python_package("soundfile")

import soundfile as sf

print(f"\n{'=' * 80}")
print("🔍 SECTION 1: Loading Model and LoRA Adapters")
print(f"{'=' * 80}")


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/whisper-large-v3",
    dtype = None,
    load_in_4bit = False,
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
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
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
    task_type = None,  # ** MUST set this for Whisper **
)

print("✅ Model and LoRA adapters loaded successfully!")


print(f"\n{'=' * 80}")
print("🔍 SECTION 2: Checking Model Class Type")
print(f"{'=' * 80}")

assert isinstance(model, PeftModel), "Model should be an instance of PeftModel"
print("✅ Model is an instance of PeftModel!")


print(f"\n{'=' * 80}")
print("🔍 SECTION 3: Checking Config Model Class Type")
print(f"{'=' * 80}")


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


print(f"\n{'=' * 80}")
print("🔍 SECTION 4: Saving and Merging Model")
print(f"{'=' * 80}")

with warnings.catch_warnings():
    warnings.simplefilter("error")  # Treat warnings as errors
    try:
        model.save_pretrained_merged("whisper", tokenizer)
        print("✅ Model saved and merged successfully without warnings!")
    except Exception as e:
        assert False, f"Model saving/merging failed with exception: {e}"

print(f"\n{'=' * 80}")
print("🔍 SECTION 5: Loading Model for Inference")
print(f"{'=' * 80}")


model, tokenizer = FastModel.from_pretrained(
    model_name = "./whisper",
    dtype = None,
    load_in_4bit = False,
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "English",
    whisper_task = "transcribe",
)


print("✅ Model loaded for inference successfully!")

print(f"\n{'=' * 80}")
print("🔍 SECTION 6: Downloading Sample Audio File")
print(f"{'=' * 80}")

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
    # Runs at import, so a failure here is a collection error and the whole file reports no results.
    # Wikimedia rate-limits this URL (429 in a batch run) and a fixture we could not fetch says nothing about unsloth,
    # so skip.
    pytest.skip(
        f"could not download the test audio fixture from {audio_url}: {e}",
        allow_module_level = True,
    )

print(f"\n{'=' * 80}")
print("🔍 SECTION 7: Running Inference")
print(f"{'=' * 80}")


from transformers import pipeline
import torch

FastModel.for_inference(model)
model.eval()
whisper = pipeline(
    "automatic-speech-recognition",
    model = model,
    tokenizer = tokenizer.tokenizer,
    feature_extractor = tokenizer.feature_extractor,
    processor = tokenizer,
    return_language = True,
    torch_dtype = torch.float16,
)
audio_file = "Speech_12dB_s16.flac"
transcribed_text = whisper(audio_file)
print(f"📝 Transcribed Text: {transcribed_text['text']}")

expected_phrases = [
    "birch canoe slid on the smooth planks",
    "sheet to the dark blue background",
    "easy to tell the depth of a well",
    "Four hours of steady work faced us",
]

transcribed_lower = transcribed_text["text"].lower()
all_phrases_found = all(phrase.lower() in transcribed_lower for phrase in expected_phrases)

assert all_phrases_found, f"Expected phrases not found in transcription: {transcribed_text['text']}"
print("✅ Transcription contains all expected phrases!")


safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./whisper")
