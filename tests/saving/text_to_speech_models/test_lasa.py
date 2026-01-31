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
require_python_package("xcodec2")

import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model

XCODEC2_MODEL_NAME = "HKUST-Audio/xcodec2"
SAMPLE_RATE = 16000
DEVICE = "cuda"

try:
    codec_model = XCodec2Model.from_pretrained(XCODEC2_MODEL_NAME)

except Exception as e:
    raise f"ERROR loading XCodec2 model: {e}."

codec_model.to("cpu")

print(f"\n{'='*80}")
print("🔍 SECTION 1: Loading Model and LoRA Adapters")
print(f"{'='*80}")

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llasa-1B",
    max_seq_length = max_seq_length,
    dtype = None,  # Select None for auto detection
    load_in_4bit = False,  # Choose True for 4bit which reduces memory
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

base_model_class = model.__class__.__name__


model = FastLanguageModel.get_peft_model(
    model,
    r = 128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 128,
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
        model.save_pretrained_merged("lasa", tokenizer)
        print("✅ Model saved and merged successfully without warnings!")
    except Exception as e:
        assert False, f"Model saving/merging failed with exception: {e}"

print(f"\n{'='*80}")
print("🔍 SECTION 5: Loading Model for Inference")
print(f"{'='*80}")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./lasa",
    max_seq_length = max_seq_length,
    dtype = None,  # Select None for auto detection
    load_in_4bit = False,  # Choose True for 4bit which reduces memory
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# from transformers import AutoProcessor
# processor = AutoProcessor.from_pretrained("unsloth/csm-1b")

print("✅ Model loaded for inference successfully!")


print(f"\n{'='*80}")
print("🔍 SECTION 6: Running Inference")
print(f"{'='*80}")


from transformers import pipeline
import torch

output_audio_path = "lasa_audio.wav"
input_text = "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person."

FastLanguageModel.for_inference(model)


def ids_to_speech_tokens(speech_ids):
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith("<|s_") and token_str.endswith("|>"):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


# TTS start!
with torch.inference_mode():
    with torch.amp.autocast("cuda", dtype = model.dtype):
        formatted_text = (
            f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"
        )

        # Tokenize the text
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
        ]

        input_ids = tokenizer.apply_chat_template(
            chat, tokenize = True, return_tensors = "pt", continue_final_message = True
        )
        input_ids = input_ids.to("cuda")

        speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length = 2048,  # We trained our model with a max length of 2048
            eos_token_id = speech_end_id,
            do_sample = True,
            top_p = 1.2,  #  Adjusts the diversity of generated content
            temperature = 1.2,  #  Controls randomness in output
        )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1] : -1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens = True)

    # Convert  token <|s_23456|> to int 23456
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).cpu().unsqueeze(0).unsqueeze(0)

    # Decode the speech tokens to speech waveform
    gen_wav = codec_model.decode_code(speech_tokens)
try:
    sf.write(output_audio_path, gen_wav[0, 0, :].cpu().numpy(), 16000)
except Exception as e:
    assert False, f"Inference failed with exception: {e}"


## assert that transcribed_text contains The birch canoe slid on the smooth planks. Glued the sheet to the dark blue background. It's easy to tell the depth of a well. Four hours of steady work faced us.

print("✅ All sections passed successfully!")


safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./lasa")
