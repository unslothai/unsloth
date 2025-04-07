# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
from .mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit
from packaging.version import Version
from transformers import __version__ as transformers_version, AutoModelForSequenceClassification

transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")

def patch_for_sequence_classification(model, num_labels=2, device_map=None):
    """
    Patch a model to support sequence classification.

    Args:
        model: The base model (e.g., BERT, RoBERTa).
        num_labels: The number of labels for classification.
        device_map: Specifies device placement for efficient training.

    Returns:
        The patched model for sequence classification.
    """
    if not hasattr(model, "config") or not hasattr(model.config, "_name_or_path"):
        raise ValueError("Invalid model object. Ensure it's a valid Hugging Face model.")

    return AutoModelForSequenceClassification.from_pretrained(
        model.config._name_or_path,
        num_labels=num_labels,
        device_map=device_map,
    )


def __get_model_name(model_name, load_in_4bit, INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit):
    """
    Helper function to get the correct model name based on 4-bit compatibility.

    Args:
        model_name (str): The model name (e.g., "mistral-7b").
        load_in_4bit (bool): Whether to load in 4-bit.
        INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit (dict): Mappers for compatibility.

    Returns:
        str: Mapped model name or None if unchanged.
    """
    model_name = model_name.lower()

    if not SUPPORTS_FOURBIT and model_name in INT_TO_FLOAT_MAPPER:
        print(
            f"Unsloth: Your transformers version {transformers_version} does not support native 4-bit loading.\n"
            f"The minimum required version is 4.37. Try `pip install --upgrade \"transformers>=4.37\"`\n"
            f"For now, we shall load `{INT_TO_FLOAT_MAPPER[model_name]}` instead (still 4-bit, just slower)."
        )
        return INT_TO_FLOAT_MAPPER[model_name]

    if not load_in_4bit:
        return INT_TO_FLOAT_MAPPER.get(model_name) or MAP_TO_UNSLOTH_16bit.get(model_name)

    if SUPPORTS_FOURBIT and load_in_4bit:
        return FLOAT_TO_INT_MAPPER.get(model_name)

    return None


def _get_new_mapper():
    """
    Fetches updated mappings from the Unsloth repository.

    Returns:
        tuple: Updated INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit
    """
    new_mapper_url = "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
    
    try:
        response = requests.get(new_mapper_url, timeout=3)
        response.raise_for_status()
        
        # Extract dictionary mappings from the fetched script
        exec_globals = {}
        exec(response.text, exec_globals)

        return (
            exec_globals.get("INT_TO_FLOAT_MAPPER", INT_TO_FLOAT_MAPPER),
            exec_globals.get("FLOAT_TO_INT_MAPPER", FLOAT_TO_INT_MAPPER),
            exec_globals.get("MAP_TO_UNSLOTH_16bit", MAP_TO_UNSLOTH_16bit)
        )
    except Exception as e:
        print(f"Warning: Failed to fetch updated Unsloth mappers. Using existing mappings. Error: {e}")
        return INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit


def get_model_name(model_name, load_in_4bit=True):
    """
    Retrieves the correct model name, ensuring compatibility with Unsloth and 4-bit loading.

    Args:
        model_name (str): The base model name (e.g., "mistral-7b").
        load_in_4bit (bool): Whether to load in 4-bit precision.

    Returns:
        str: The mapped model name.
    """
    new_model_name = __get_model_name(
        model_name,
        load_in_4bit,
        INT_TO_FLOAT_MAPPER,
        FLOAT_TO_INT_MAPPER,
        MAP_TO_UNSLOTH_16bit,
    )

    if new_model_name:
        return new_model_name

    if "/" in model_name and model_name[0].isalnum():
        # Fetch latest Unsloth mappings if model is not recognized
        updated_mappers = _get_new_mapper()
        new_model_name = __get_model_name(model_name, load_in_4bit, *updated_mappers)

        if new_model_name:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version!\n"
                "Please update Unsloth via:\n\n"
                'pip uninstall unsloth unsloth_zoo -y\n'
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
                'pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"\n'
            )

    return model_name
