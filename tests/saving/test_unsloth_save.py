import json
import os
import shutil
import tempfile
import pytest
import importlib

from unsloth import FastLanguageModel, FastModel

# Every param here downloads a checkpoint and merges it on the accelerator;
# the file was 877s, 29% of the repo's total test time, before its matrix was cut to three models.
# CI runs it under `-m gpu`.
pytestmark = pytest.mark.gpu

# One model per save path, and the smallest model that still exercises the path.
# This file runs in the default `pytest tests/` walk (unlike the rest of tests/saving/, which is gated behind
# UNSLOTH_RUN_SAVING_SCRIPTS), so every entry here is paid by anyone who runs the suite.
# The ten-model matrix it grew to cost 877s, 29% of the whole repo's test time; these three cover the same paths.
# Distinct paths, not distinct checkpoints, are what earn a slot: text 16-bit on disk (Qwen2.5-0.5B-Instruct),
# text already 4-bit on disk (tinyllama-bnb-4bit), vision (Qwen2.5-VL-3B-Instruct).
# Dropped: tinyllama and Qwen2.5-0.5B (same text path as Qwen2.5-0.5B-Instruct, and the first is 2x its size);
# Qwen2.5-0.5B-Instruct-bnb-4bit and Phi-4-mini-instruct-bnb-4bit (loading a pre-quantized checkpoint is one path,
# so it gets one model, not four); Phi-4-mini-instruct (3.8B, adds no path a 0.5B does not); gemma-3-4b-it (vision,
# but 4B against Qwen2.5-VL's 3B);
# Llama-3.2-11B-Vision-Instruct-bnb-4bit (11B, and its merged 16-bit write was 21.3GB to $TMPDIR per run).
model_to_test = [
    # Text, 16-bit on disk.
    "unsloth/Qwen2.5-0.5B-Instruct",
    # Text, already 4-bit on disk: from_pretrained has to load a pre-quantized checkpoint and the merge has to
    # dequantize back out of it.
    "unsloth/tinyllama-bnb-4bit",
    # Vision, and the only entry whose merged 16-bit output clears the 5GB safetensors shard limit, so it is what
    # keeps sharded output and its index file covered here. The dedicated sharded-index tests
    # (vision_models/test_index_file_sharded_model.py,
    # language_models/test_push_to_hub_merged_sharded_index_file.py) do NOT cover it in a default run: both are behind
    # UNSLOTH_RUN_SAVING_SCRIPTS. Keep an entry above 5GB here, or that path stops being exercised.
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
]

torchao_models = [
    # One model: both entries drove the same save_pretrained_torchao path, and tinyllama is 1.1B against this
    # one's 0.5B.
    "unsloth/Qwen2.5-0.5B-Instruct",
]


save_file_sizes = {}
save_file_sizes["merged_16bit"] = {}
save_file_sizes["merged_4bit"] = {}
save_file_sizes["torchao"] = {}

tokenizer_files = [
    "tokenizer_config.json",
    "special_tokens_map.json",
]


@pytest.fixture(scope = "session", params = model_to_test)
def loaded_model_tokenizer(request):
    model_name = request.param
    print("Loading model and tokenizer...")

    model, tokenizer = FastModel.from_pretrained(
        model_name,
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = True,
    )

    model = FastModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        use_gradient_checkpointing = "unsloth",
    )

    return model, tokenizer


@pytest.fixture(scope = "session", params = torchao_models)
def fp16_model_tokenizer(request):
    """Load model in FP16 for TorchAO quantization."""
    model_name = request.param
    print(f"Loading model in FP16 for TorchAO: {model_name}")

    model, tokenizer = FastModel.from_pretrained(
        model_name,
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = False,
    )

    model = FastModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        use_gradient_checkpointing = "unsloth",
    )

    return model, tokenizer


@pytest.fixture(scope = "session")
def model(loaded_model_tokenizer):
    return loaded_model_tokenizer[0]


@pytest.fixture(scope = "session")
def tokenizer(loaded_model_tokenizer):
    return loaded_model_tokenizer[1]


@pytest.fixture
def temp_save_dir():
    dir = tempfile.mkdtemp()
    print(f"Temporary directory created at: {dir}")
    yield dir
    print(f"Temporary directory deleted: {dir}")
    shutil.rmtree(dir)


def delete_quantization_config(model):
    old_config = model.config
    new_config = model.config.to_dict()
    if "quantization_config" in new_config:
        del new_config["quantization_config"]
    original_model = model
    new_config = type(model.config).from_dict(new_config)
    while hasattr(original_model, "model"):
        original_model = original_model.model
        original_model.config = new_config
    model.config = new_config


def test_save_merged_16bit(model, tokenizer, temp_save_dir: str):
    save_path = os.path.join(
        temp_save_dir,
        "unsloth_merged_16bit",
        model.config._name_or_path.replace("/", "_"),
    )

    model.save_pretrained_merged(save_path, tokenizer = tokenizer, save_method = "merged_16bit")

    assert os.path.isdir(save_path), f"Directory {save_path} does not exist."
    assert os.path.isfile(os.path.join(save_path, "config.json")), "config.json not found."

    weight_files = [
        f for f in os.listdir(save_path) if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(save_path, file)
        ), f"{file} not found in the save directory."

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert "quantization_config" not in config, "Quantization config not found in the model config."

    total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in weight_files)
    save_file_sizes["merged_16bit"][model.config._name_or_path] = total_size
    print(f"Total size of merged_16bit files: {total_size} bytes")

    loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
        save_path,
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = True,
    )


def test_save_merged_4bit(model, tokenizer, temp_save_dir: str):
    save_path = os.path.join(
        temp_save_dir,
        "unsloth_merged_4bit",
        model.config._name_or_path.replace("/", "_"),
    )

    model.save_pretrained_merged(save_path, tokenizer = tokenizer, save_method = "merged_4bit_forced")

    assert os.path.isdir(save_path), f"Directory {save_path} does not exist."
    assert os.path.isfile(os.path.join(save_path, "config.json")), "config.json not found."

    weight_files = [
        f for f in os.listdir(save_path) if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(save_path, file)
        ), f"{file} not found in the save directory."

    total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in weight_files)
    save_file_sizes["merged_4bit"][model.config._name_or_path] = total_size

    print(f"Total size of merged_4bit files: {total_size} bytes")

    assert (
        total_size < save_file_sizes["merged_16bit"][model.config._name_or_path]
    ), "Merged 4bit files are larger than merged 16bit files."

    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert "quantization_config" in config, "Quantization config not found in the model config."

    loaded_model, loaded_tokenizer = FastModel.from_pretrained(
        save_path,
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = True,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None,
    reason = "require torchao to be installed",
)
def test_save_torchao(fp16_model_tokenizer, temp_save_dir: str):
    model, tokenizer = fp16_model_tokenizer
    save_path = os.path.join(
        temp_save_dir, "unsloth_torchao", model.config._name_or_path.replace("/", "_")
    )

    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    torchao_config = Int8DynamicActivationInt8WeightConfig()
    model.save_pretrained_torchao(
        save_path,
        tokenizer = tokenizer,
        torchao_config = torchao_config,
        push_to_hub = False,
    )

    weight_files_16bit = [
        f for f in os.listdir(save_path) if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    total_16bit_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in weight_files_16bit)
    save_file_sizes["merged_16bit"][model.config._name_or_path] = total_16bit_size

    torchao_save_path = save_path + "-torchao"

    assert os.path.isdir(torchao_save_path), f"Directory {torchao_save_path} does not exist."
    assert os.path.isfile(os.path.join(torchao_save_path, "config.json")), "config.json not found."

    weight_files = [
        f for f in os.listdir(torchao_save_path) if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(torchao_save_path, file)
        ), f"{file} not found in the save directory."

    total_size = sum(os.path.getsize(os.path.join(torchao_save_path, f)) for f in weight_files)
    save_file_sizes["torchao"][model.config._name_or_path] = total_size

    assert (
        total_size < save_file_sizes["merged_16bit"][model.config._name_or_path]
    ), "torchao files are larger than merged 16bit files."

    config_path = os.path.join(torchao_save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert "quantization_config" in config, "Quantization config not found in the model config."

    import torch.serialization

    with torch.serialization.safe_globals([getattr]):
        loaded_model, loaded_tokenizer = FastModel.from_pretrained(
            torchao_save_path,
            max_seq_length = 128,
            dtype = None,
            load_in_4bit = False,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None,
    reason = "require torchao to be installed",
)
def test_save_and_inference_torchao(fp16_model_tokenizer, temp_save_dir: str):
    model, tokenizer = fp16_model_tokenizer
    model_name = model.config._name_or_path

    print(f"Testing TorchAO save and inference for: {model_name}")

    save_path = os.path.join(temp_save_dir, "torchao_models", model_name.replace("/", "_"))

    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    torchao_config = Int8DynamicActivationInt8WeightConfig()

    model.save_pretrained_torchao(
        save_path,
        tokenizer = tokenizer,
        torchao_config = torchao_config,
        push_to_hub = False,
    )

    torchao_save_path = save_path + "-torchao"

    assert os.path.isdir(
        torchao_save_path
    ), f"TorchAO directory {torchao_save_path} does not exist."

    # load_in_4bit must stay False: a torchao-quantized model can't be re-quantized with bitsandbytes.
    import torch.serialization

    with torch.serialization.safe_globals([getattr]):
        loaded_model, loaded_tokenizer = FastModel.from_pretrained(
            torchao_save_path,
            max_seq_length = 128,
            dtype = None,
            load_in_4bit = False,
        )

    FastModel.for_inference(loaded_model)

    messages = [
        {
            "role": "user",
            "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,",
        },
    ]
    inputs = loaded_tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,  # required for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = loaded_model.generate(
        input_ids = inputs,
        max_new_tokens = 64,
        use_cache = False,  # avoid cache issues
        temperature = 1.5,
        min_p = 0.1,
        do_sample = True,
        pad_token_id = loaded_tokenizer.pad_token_id or loaded_tokenizer.eos_token_id,
    )

    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens = True)
    input_text = loaded_tokenizer.decode(inputs[0], skip_special_tokens = True)
    response_part = generated_text[len(input_text) :].strip()

    print(f"Input: {input_text}")
    print(f"Full output: {generated_text}")
    print(f"Response only: {response_part}")
