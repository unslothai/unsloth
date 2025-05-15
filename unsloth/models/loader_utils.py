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

from .mapper import INT_TO_FLOAT_MAPPER, FLOAT_TO_INT_MAPPER, MAP_TO_UNSLOTH_16bit
# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from packaging.version import Version
from transformers import __version__ as transformers_version
transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")

BAD_MAPPINGS = \
{
    "unsloth/Qwen3-32B-unsloth-bnb-4bit".lower()          : "unsloth/Qwen3-32B-bnb-4bit".lower(), # 32B dynamic quant is way too big
    "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit".lower()      : "unsloth/Qwen3-30B-A3B".lower(),      # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-bnb-4bit".lower()              : "unsloth/Qwen3-30B-A3B".lower(),      # We rather do it on the fly
    "unsloth/Qwen3-30B-A3B-Base-unsloth-bnb-4bit".lower() : "unsloth/Qwen3-30B-A3B-Base".lower(), # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-Base-bnb-4bit".lower()         : "unsloth/Qwen3-30B-A3B-Base".lower(), # We rather do it on the fly
}

def __get_model_name(
    model_name,
    load_in_4bit = True,
    INT_TO_FLOAT_MAPPER  = None,
    FLOAT_TO_INT_MAPPER  = None,
    MAP_TO_UNSLOTH_16bit = None,
):
    model_name = str(model_name)
    lower_model_name = model_name.lower()

    if not SUPPORTS_FOURBIT and lower_model_name in INT_TO_FLOAT_MAPPER:

        model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        print(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "\
            f"4bit loading.\nThe minimum required version is 4.37.\n"\
            f'Try `pip install --upgrade "transformers>=4.37"`\n'\
            f"to obtain the latest transformers build, then restart this session.\n"\
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
        return model_name
    
    elif not load_in_4bit and lower_model_name in INT_TO_FLOAT_MAPPER:

        new_model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
        #     f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        # )
        return new_model_name

    elif not load_in_4bit and lower_model_name in MAP_TO_UNSLOTH_16bit:

        new_model_name = MAP_TO_UNSLOTH_16bit[lower_model_name]
        return new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and lower_model_name in FLOAT_TO_INT_MAPPER:

        # Support returning original full -bnb-4bit name if specified specifically
        # since we'll map it to the dynamic version instead
        if lower_model_name.endswith("-bnb-4bit"):
            return lower_model_name
        
        new_model_name = FLOAT_TO_INT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
        #     f"We shall load `{new_model_name}` for 4x faster loading."
        # )
        return new_model_name
    pass

    return None
pass


def _get_new_mapper():
    try:
        import requests
        new_mapper = "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
        with requests.get(new_mapper, timeout = 3) as new_mapper: new_mapper = new_mapper.text
        new_mapper = new_mapper[new_mapper.find("__INT_TO_FLOAT_MAPPER"):]
        new_mapper = new_mapper\
            .replace("INT_TO_FLOAT_MAPPER",  "NEW_INT_TO_FLOAT_MAPPER")\
            .replace("FLOAT_TO_INT_MAPPER",  "NEW_FLOAT_TO_INT_MAPPER")\
            .replace("MAP_TO_UNSLOTH_16bit", "NEW_MAP_TO_UNSLOTH_16bit")

        exec(new_mapper, globals())
        return NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit
    except:
        return {}, {}, {}
    pass
pass


def get_model_name(model_name, load_in_4bit = True):
    new_model_name = __get_model_name(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        INT_TO_FLOAT_MAPPER  = INT_TO_FLOAT_MAPPER,
        FLOAT_TO_INT_MAPPER  = FLOAT_TO_INT_MAPPER,
        MAP_TO_UNSLOTH_16bit = MAP_TO_UNSLOTH_16bit,
    )
    # In the rare case, we convert bad model names to other names
    # For eg too large dynamic quants or MoEs
    if new_model_name is not None and type(new_model_name) is str and \
        new_model_name.lower() in BAD_MAPPINGS:
        new_model_name = BAD_MAPPINGS[new_model_name.lower()]

    if new_model_name is None and model_name.count("/") == 1 and model_name[0].isalnum():
        # Try checking if a new Unsloth version allows it!
        NEW_INT_TO_FLOAT_MAPPER, NEW_FLOAT_TO_INT_MAPPER, NEW_MAP_TO_UNSLOTH_16bit = _get_new_mapper()
        upgraded_model_name = __get_model_name(
            model_name = model_name,
            load_in_4bit = load_in_4bit,
            INT_TO_FLOAT_MAPPER  = NEW_INT_TO_FLOAT_MAPPER,
            FLOAT_TO_INT_MAPPER  = NEW_FLOAT_TO_INT_MAPPER,
            MAP_TO_UNSLOTH_16bit = NEW_MAP_TO_UNSLOTH_16bit,
        )
        if upgraded_model_name is not None:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version! Please update Unsloth via:\n\n"\
                'pip uninstall unsloth unsloth_zoo -y\n'\
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'\
                'pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"\n'\
            )
        pass
    pass
    return new_model_name if new_model_name is not None else model_name
pass
