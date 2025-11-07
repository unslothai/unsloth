import json
import os
import shutil
import tempfile
import pytest
import importlib

from unsloth import FastLanguageModel, FastModel

model_to_test = [
    # Text Models
    "unsloth/tinyllama",
    "unsloth/tinyllama-bnb-4bit",
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "unsloth/Phi-4-mini-instruct",
    "unsloth/Phi-4-mini-instruct-bnb-4bit",
    "unsloth/Qwen2.5-0.5B",
    # Vision Models
    "unsloth/gemma-3-4b-it",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
]

torchao_models = [
    "unsloth/tinyllama",
    "unsloth/Qwen2.5-0.5B-Instruct",
    # "unsloth/Phi-4-mini-instruct",
    # "unsloth/Qwen2.5-0.5B",
    # Skip the -bnb-4bit variants since they're already quantized
]


# Variables
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
        model_name,  # use small model
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = True,
    )

    # Apply LoRA
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
    """Load model in FP16 for TorchAO quantization"""
    model_name = request.param
    print(f"Loading model in FP16 for TorchAO: {model_name}")

    model, tokenizer = FastModel.from_pretrained(
        model_name,
        max_seq_length = 128,
        dtype = None,
        load_in_4bit = False,  # No BnB quantization
    )

    # Apply LoRA
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
    # Since merged, edit quantization_config
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

    model.save_pretrained_merged(
        save_path, tokenizer = tokenizer, save_method = "merged_16bit"
    )

    # Check model files
    assert os.path.isdir(save_path), f"Directory {save_path} does not exist."
    assert os.path.isfile(
        os.path.join(save_path, "config.json")
    ), "config.json not found."

    weight_files = [
        f
        for f in os.listdir(save_path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    # Check tokenizer files
    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(save_path, file)
        ), f"{file} not found in the save directory."

    # Check config to see if it is 16bit by checking for quantization config
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert (
        "quantization_config" not in config
    ), "Quantization config not found in the model config."

    # Store the size of the model files
    total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in weight_files)
    save_file_sizes["merged_16bit"][model.config._name_or_path] = total_size
    print(f"Total size of merged_16bit files: {total_size} bytes")

    # Test loading the model from the saved path
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

    model.save_pretrained_merged(
        save_path, tokenizer = tokenizer, save_method = "merged_4bit_forced"
    )

    # Check model files
    assert os.path.isdir(save_path), f"Directory {save_path} does not exist."
    assert os.path.isfile(
        os.path.join(save_path, "config.json")
    ), "config.json not found."

    weight_files = [
        f
        for f in os.listdir(save_path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    # Check tokenizer files
    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(save_path, file)
        ), f"{file} not found in the save directory."

    # Store the size of the model files
    total_size = sum(os.path.getsize(os.path.join(save_path, f)) for f in weight_files)
    save_file_sizes["merged_4bit"][model.config._name_or_path] = total_size

    print(f"Total size of merged_4bit files: {total_size} bytes")

    assert (
        total_size < save_file_sizes["merged_16bit"][model.config._name_or_path]
    ), "Merged 4bit files are larger than merged 16bit files."

    # Check config to see if it is 4bit
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert (
        "quantization_config" in config
    ), "Quantization config not found in the model config."

    # Test loading the model from the saved path
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
        f
        for f in os.listdir(save_path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    total_16bit_size = sum(
        os.path.getsize(os.path.join(save_path, f)) for f in weight_files_16bit
    )
    save_file_sizes["merged_16bit"][model.config._name_or_path] = total_16bit_size

    torchao_save_path = save_path + "-torchao"

    # Check model files
    assert os.path.isdir(
        torchao_save_path
    ), f"Directory {torchao_save_path} does not exist."
    assert os.path.isfile(
        os.path.join(torchao_save_path, "config.json")
    ), "config.json not found."

    weight_files = [
        f
        for f in os.listdir(torchao_save_path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ]
    assert len(weight_files) > 0, "No weight files found in the save directory."

    # Check tokenizer files
    for file in tokenizer_files:
        assert os.path.isfile(
            os.path.join(torchao_save_path, file)
        ), f"{file} not found in the save directory."

    # Store the size of the model files
    total_size = sum(
        os.path.getsize(os.path.join(torchao_save_path, f)) for f in weight_files
    )
    save_file_sizes["torchao"][model.config._name_or_path] = total_size

    assert (
        total_size < save_file_sizes["merged_16bit"][model.config._name_or_path]
    ), "torchao files are larger than merged 16bit files."

    # Check config to see if it is quantized with torchao
    config_path = os.path.join(torchao_save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    assert (
        "quantization_config" in config
    ), "Quantization config not found in the model config."

    # Test loading the model from the saved path
    # can't set `load_in_4bit` to True because the model is torchao quantized
    # can't quantize again with bitsandbytes
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

    save_path = os.path.join(
        temp_save_dir, "torchao_models", model_name.replace("/", "_")
    )

    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    torchao_config = Int8DynamicActivationInt8WeightConfig()

    # Save with TorchAO
    model.save_pretrained_torchao(
        save_path,
        tokenizer = tokenizer,
        torchao_config = torchao_config,
        push_to_hub = False,
    )

    torchao_save_path = save_path + "-torchao"

    # Verify files exist
    assert os.path.isdir(
        torchao_save_path
    ), f"TorchAO directory {torchao_save_path} does not exist."

    # Load with safe globals
    import torch.serialization

    with torch.serialization.safe_globals([getattr]):
        loaded_model, loaded_tokenizer = FastModel.from_pretrained(
            torchao_save_path,
            max_seq_length = 128,
            dtype = None,
            load_in_4bit = False,
        )

    FastModel.for_inference(loaded_model)  # Enable native 2x faster inference

    messages = [
        {
            "role": "user",
            "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,",
        },
    ]
    inputs = loaded_tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,  # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = loaded_model.generate(  # ← Use loaded_model, not model
        input_ids = inputs,
        max_new_tokens = 64,
        use_cache = False,  # Avoid cache issues
        temperature = 1.5,
        min_p = 0.1,
        do_sample = True,
        pad_token_id = loaded_tokenizer.pad_token_id or loaded_tokenizer.eos_token_id,
    )

    # Decode with the LOADED tokenizer
    generated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens = True)
    input_text = loaded_tokenizer.decode(inputs[0], skip_special_tokens = True)
    response_part = generated_text[len(input_text) :].strip()

    print(f"Input: {input_text}")
    print(f"Full output: {generated_text}")
    print(f"Response only: {response_part}")
