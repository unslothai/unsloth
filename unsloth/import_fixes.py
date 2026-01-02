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

import os
import importlib.util
from pathlib import Path
from importlib.metadata import version as importlib_version
from packaging.version import Version as TrueVersion
import re
import logging
import textwrap
import warnings

# We cannot do from unsloth_zoo.log import logger since FBGEMM might cause seg faults.
UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") in (
    "1",
    "True",
    "true",
)
logger = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logging.basicConfig(
        level = logging.INFO, format = "[%(name)s|%(levelname)s]%(message)s"
    )
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(
        level = logging.WARNING, format = "[%(name)s|%(levelname)s]%(message)s"
    )
    logger.setLevel(logging.WARNING)


def Version(version):
    try:
        new_version = str(version)
        new_version = re.match(r"[0-9\.]{1,}", new_version)
        if new_version is None:
            raise Exception(str(e))
        new_version = new_version.group(0).rstrip(".")
        if new_version != version:
            new_version += ".1"  # Add .1 for dev / alpha / beta / rc
        return TrueVersion(new_version)
    except:
        from inspect import getframeinfo, stack

        caller = getframeinfo(stack()[1][0])
        raise RuntimeError(
            f"Unsloth: Could not get version for `{version}`\n"
            f"File name = [{caller.filename}] Line number = [{caller.lineno}]"
        )


# Ignore logging messages
class HideLoggingMessage(logging.Filter):
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text

    def filter(self, x):
        return not (self.text in x.getMessage())


class HidePrintMessage:
    def __init__(self, original_stream):
        self._original_stream = original_stream
        self._hidden_texts = []

    def add_filter(self, text):
        self._hidden_texts.append(text)

    def write(self, message):
        if not any(text in message for text in self._hidden_texts):
            self._original_stream.write(message)

    def flush(self):
        self._original_stream.flush()

    def __getattr__(self, name):
        return getattr(self._original_stream, name)


if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") != "1":
    import sys

    # Apply to stderr for FBGEMM
    sys.stderr = HidePrintMessage(sys.stderr)
    # https://github.com/pytorch/FBGEMM/blob/d99cd96490ec4aabac2ee95b1e76ea4dcfcfa628/fbgemm_gpu/experimental/gemm/triton_gemm/utils.py#L43-L52
    sys.stderr.add_filter("TMA benchmarks will be running")
    # Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu128 for torchao version 0.15.0
    logging.getLogger("torchao").setLevel(logging.ERROR)
    # SyntaxWarning: invalid escape sequence '\.'
    warnings.filterwarnings(
        "ignore", message = "invalid escape sequence", category = SyntaxWarning
    )


# Fix up AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
# MUST do this at the start primarily due to tensorflow causing issues
def fix_message_factory_issue():
    try:
        import google.protobuf.message_factory

        class MessageFactory:
            def CreatePrototype(self, *args, **kwargs):
                return

            def GetMessages(self, *args, **kwargs):
                return

            def GetPrototype(self, *args, **kwargs):
                return

        if not hasattr(google.protobuf.message_factory, "MessageFactory"):
            logger.info("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
            google.protobuf.message_factory.MessageFactory = MessageFactory
        elif (
            hasattr(google.protobuf.message_factory, "MessageFactory")
            and not hasattr(
                google.protobuf.message_factory.MessageFactory, "GetPrototype"
            )
            and not hasattr(google.protobuf.message_factory, "GetMessageClass")
        ):
            google.protobuf.message_factory.MessageFactory = MessageFactory
            logger.info("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
        elif (
            hasattr(google.protobuf.message_factory, "MessageFactory")
            and not hasattr(
                google.protobuf.message_factory.MessageFactory, "GetPrototype"
            )
            and hasattr(google.protobuf.message_factory, "GetMessageClass")
        ):
            GetMessageClass = google.protobuf.message_factory.GetMessageClass

            def GetPrototype(self, descriptor):
                return GetMessageClass(descriptor)

            google.protobuf.message_factory.MessageFactory.GetPrototype = GetPrototype
            logger.info("Unsloth: Patching protobuf.MessageFactory.GetPrototype")
        pass
    except:
        pass


# Fix Xformers performance issues since 0.0.25
def fix_xformers_performance_issue():
    spec = importlib.util.find_spec("xformers")
    if spec is None:
        return
    xformers_version = importlib_version("xformers")
    if Version(xformers_version) < Version("0.0.29"):
        xformers_location = spec.origin
        if xformers_location is None:
            xformers_location = spec.submodule_search_locations[0]
        else:
            xformers_location = os.path.split(xformers_location)[0]
        cutlass = Path(xformers_location) / "ops" / "fmha" / "cutlass.py"
        try:
            if cutlass.exists():
                with open(cutlass, "r+", encoding = "utf-8") as f:
                    text = f.read()
                    # See https://github.com/facebookresearch/xformers/issues/1176#issuecomment-2545829591
                    if "num_splits_key=-1," in text:
                        text = text.replace(
                            "num_splits_key=-1,",
                            "num_splits_key=None,",
                        )
                        f.seek(0)
                        f.write(text)
                        f.truncate()
                        logger.info(
                            "Unsloth: Patching Xformers to fix some performance issues."
                        )
        except Exception as e:
            logger.info(f"Unsloth: Failed patching Xformers with error = {str(e)}")


# ValueError: 'aimv2' is already used by a Transformers config, pick another name.
def fix_vllm_aimv2_issue():
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        return
    vllm_version = importlib_version("vllm")
    if Version(vllm_version) < Version("0.10.1"):
        vllm_location = spec.origin
        if vllm_location is None:
            vllm_location = spec.submodule_search_locations[0]
        else:
            vllm_location = os.path.split(vllm_location)[0]
        ovis_config = Path(vllm_location) / "transformers_utils" / "configs" / "ovis.py"
        try:
            if ovis_config.exists():
                with open(ovis_config, "r+", encoding = "utf-8") as f:
                    text = f.read()
                    # See https://github.com/vllm-project/vllm-ascend/issues/2046
                    if 'AutoConfig.register("aimv2", AIMv2Config)' in text:
                        text = text.replace(
                            'AutoConfig.register("aimv2", AIMv2Config)',
                            "",
                        )
                        text = text.replace(
                            """backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type,
                                                       **backbone_config)""",
                            """if model_type != "aimv2":
                    backbone_config.pop('model_type')
                    backbone_config = AutoConfig.for_model(model_type, **backbone_config)
                else:
                    backbone_config = AIMv2Config(**backbone_config)""",
                        )
                        f.seek(0)
                        f.write(text)
                        f.truncate()
                        logger.info(
                            "Unsloth: Patching vLLM to fix `'aimv2' is already used by a Transformers config, pick another name.`"
                        )
        except Exception as e:
            logger.info(f"Unsloth: Failed patching vLLM with error = {str(e)}")


def fix_vllm_guided_decoding_params():
    if importlib.util.find_spec("vllm") is None:
        return
    # GuidedDecodingParmas is renamed to StructuredOutputsParams in vLLM
    # https://github.com/vllm-project/vllm/pull/22772/files
    # trl still wants to use GuidedDecodingParams. This is a temporary patch till trl updates
    import vllm

    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        vllm.sampling_params.GuidedDecodingParams = (
            vllm.sampling_params.StructuredOutputsParams
        )


def ignore_logger_messages():
    # Ignore Environment variable `HF_TOKEN` is set
    try:
        from huggingface_hub._login import logger as huggingface_hub_logger

        huggingface_hub_logger.addFilter(HideLoggingMessage("`HF_TOKEN`"))
        del huggingface_hub_logger
    except:
        pass


def patch_ipykernel_hf_xet():
    # HF-XET == 1.1.10 and ipykernel == 7.0.0 / 7.0.1 causes issues
    # See https://github.com/huggingface/xet-core/issues/526
    # 2025-10-13T20:37:33.028737Z ERROR  Python exception updating progress:, error: PyErr { type: <class 'LookupError'>, value: LookupError(<ContextVar name='shell_parent' at 0x7535b4cebd80>), traceback: Some(<traceback object at 0x753408489f40>) }, caller: "src/progress_update.rs:313"
    # at /home/runner/work/xet-core/xet-core/error_printer/src/lib.rs:28
    if importlib.util.find_spec("hf_xet") is None:
        return
    if importlib.util.find_spec("ipykernel") is None:
        return
    if importlib.util.find_spec("huggingface_hub") is None:
        return

    ipykernel_version = Version(importlib_version("ipykernel"))
    if (
        (Version(importlib_version("hf_xet")) == Version("1.1.10"))
        and (
            (ipykernel_version == Version("7.0.0"))
            or (
                ipykernel_version == Version("7.0.1")
            )  # 7.0.1 seems to also break with LookupError: <ContextVar name='shell_parent' at 0x7a9775143ec0>
        )
    ):
        print(
            "#### Unsloth: `hf_xet==1.1.10` and `ipykernel==7.0.0` or `ipykernel==7.0.1` breaks progress bars. Using ASCII progress bars.\n"
            "#### Unsloth: To re-enable progress bars, please upgrade to `ipykernel>=7.1.0` or wait for a fix to\n"
            "https://github.com/huggingface/xet-core/issues/526"
        )
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()


def patch_trackio():
    # Set some environment variables to customize the Trackio dashboard for experiment tracking
    # See https://github.com/unslothai/notebooks/pull/110
    os.environ["TRACKIO_LOGO_LIGHT_URL"] = (
        "https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png"
    )
    os.environ["TRACKIO_LOGO_DARK_URL"] = (
        "https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png"
    )
    os.environ["TRACKIO_PLOT_ORDER"] = "train/reward"


def patch_datasets():
    # Datasets 4.4.0 and 4.4.1 weirdly have some weird `_thread.RLock_recursion_count` issues
    if importlib.util.find_spec("datasets") is None:
        return

    datasets_version = Version(importlib_version("datasets"))
    if (datasets_version <= Version("4.5.0")) and (
        datasets_version >= Version("4.4.0")
    ):
        raise NotImplementedError(
            f"#### Unsloth: Using `datasets = {str(datasets_version)}` will cause recursion errors.\n"
            "Please downgrade datasets to `datasets==4.3.0"
        )


def check_fbgemm_gpu_version():
    if importlib.util.find_spec("fbgemm_gpu") is None:
        return
    try:
        fbgemm_gpu_version = importlib_version("fbgemm_gpu_genai")
    except:
        return
    # We noticed some SegFault or bad alloc errors on lower versions of fbgemm_gpu.
    if Version(fbgemm_gpu_version) < Version("1.4.0"):
        raise ImportError(
            f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu_version} detected. It might cause unexpected issues like segmentation faults. Please uninstall the current one by doing `pip uninstall fbgemm-gpu` && `pip install fbgemm-gpu` to install fbgemm-gpu 1.4.0 or newer!"
        )

    logger.info(f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu_version} detected.")


def patch_enable_input_require_grads():
    """
    Patch transformers PreTrainedModel.enable_input_require_grads to handle vision models
    that raise NotImplementedError from get_input_embeddings().

    """
    import inspect
    from transformers import PreTrainedModel

    # Check if the original function iterates over self.modules() instead of just returning the enable_input_require_grads
    # Ref: https://github.com/huggingface/transformers/pull/41993/files#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63eaL1979-R1996
    try:
        original_source = inspect.getsource(PreTrainedModel.enable_input_require_grads)
    except:
        return

    # Only patch if the new pattern exists (iterating over self.modules())
    if "for module in self.modules()" not in original_source:
        return

    def _patched_enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()

        for module in self.modules():
            if not (
                isinstance(module, PreTrainedModel)
                and hasattr(module, "get_input_embeddings")
            ):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                # Vision models may not implement get_input_embeddings - skip them
                # For GLM V4.6 for example, this skips only `self.visual`
                continue

            if input_embeddings is None:
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(
                input_embeddings.register_forward_hook(make_inputs_require_grads)
            )

        self._require_grads_hooks = hooks
        if hooks:
            self._require_grads_hook = hooks[0]

    PreTrainedModel.enable_input_require_grads = _patched_enable_input_require_grads

    logger.info(
        "Unsloth: Patched enable_input_require_grads for vision model compatibility"
    )


def torchvision_compatibility_check():
    if importlib.util.find_spec("torch") is None:
        raise ImportError("Unsloth: torch not found. Please install torch first.")
    if importlib.util.find_spec("torchvision") is None:
        return
    torch_version = importlib_version("torch")
    torchvision_version = importlib_version("torchvision")

    # Torch version -> minimum required torchvision version
    # See https://pytorch.org/get-started/previous-versions/
    TORCH_TORCHVISION_COMPAT = [
        ("2.9.0", "0.24.0"),
        ("2.8.0", "0.23.0"),
        ("2.7.0", "0.22.0"),
        ("2.6.0", "0.21.0"),
        ("2.5.0", "0.20.0"),
        ("2.4.0", "0.19.0"),
    ]

    required_torchvision = None
    for min_torch, min_torchvision in TORCH_TORCHVISION_COMPAT:
        if Version(torch_version) >= Version(min_torch):
            required_torchvision = min_torchvision
            break

    if required_torchvision is None:
        # Torch version not in compatibility table, skip check
        return

    if Version(torchvision_version) < Version(required_torchvision):
        raise ImportError(
            f"Unsloth: torch=={torch_version} requires torchvision>={required_torchvision}, "
            f"but found torchvision=={torchvision_version}. "
            f"Please refer to https://pytorch.org/get-started/previous-versions/ for more information."
        )

    logger.info(
        f"Unsloth: torch=={torch_version} and torchvision=={torchvision_version} are compatible."
    )


# Fix TRL OpenEnv 0.26 NameError: name 'SamplingParams' is not defined
def fix_openenv_no_vllm():
    spec = importlib.util.find_spec("trl")
    if spec is None:
        return
    trl_location = spec.origin
    if trl_location is None:
        trl_location = spec.submodule_search_locations[0]
    else:
        trl_location = os.path.split(trl_location)[0]
    openenv = Path(trl_location) / "experimental" / "openenv" / "utils.py"
    if not openenv.exists():
        return

    try:
        with open(openenv, "r+", encoding = "utf-8") as f:
            text = f.read()
            bad = (
                "if is_vllm_available():\n"
                "    from vllm import SamplingParams\n"
                "    from vllm.sampling_params import GuidedDecodingParams\n"
            )
            replace_with = bad + (
                "else:\n"
                "    from typing import Any\n"
                "    SamplingParams = Any\n"
                "    GuidedDecodingParams = Any\n"
                "\n"
            )
            if bad + "\n" + "\n" in text and replace_with not in text:
                text = text.replace(bad + "\n" + "\n", replace_with)
                f.seek(0)
                f.write(text)
                f.truncate()
                logger.info(
                    "Unsloth: Patching TRL OpenEnv to fix SamplingParams not defined"
                )
    except Exception as e:
        logger.info(f"Unsloth: Failed patching TRL OpenEnv with error = {str(e)}")


# Fix Exeuctorch needing get_mapped_key
def fix_executorch():
    spec = importlib.util.find_spec("executorch")
    if spec is None:
        return
    executorch_location = spec.origin
    if executorch_location is None:
        executorch_location = spec.submodule_search_locations[0]
    else:
        executorch_location = os.path.split(executorch_location)[0]
    executorch = Path(executorch_location) / "examples" / "models" / "__init__.py"
    if not executorch.exists():
        return

    try:
        what = r"""
        import sys
        import types
        import re
        from typing import Any, Optional
        def get_mapped_key(key: str, mapping_dict: dict[str, str]) -> str:
            try:
                # Checks if there is a layer # in the key
                if any(k.isdigit() for k in key.split(".")):
                    # Replace layer number with "{}" to create key for lookup
                    abstract_key = re.sub(r"(\.\d+)", ".{}", key)
                    layer_num = re.search(r"\d+", key).group(0)
                    new_key = mapping_dict[abstract_key]
                    new_key = new_key.format(layer_num)
                else:
                    new_key = mapping_dict[key]
            except KeyError as e:
                raise Exception(
                    f'Error converting the state dict. Found unexpected key: "{key}". '
                    "Please make sure you're loading a checkpoint with the right format. "
                ) from e

            return new_key

        torchtune = types.ModuleType("torchtune")
        torchtune.__path__ = []
        models = types.ModuleType("torchtune.models")
        models.__path__ = []
        convert_weights = types.ModuleType("torchtune.models.convert_weights")
        convert_weights.get_mapped_key = get_mapped_key
        torchtune.models = models
        models.convert_weights = convert_weights
        sys.modules["torchtune"] = torchtune
        sys.modules["torchtune.models"] = models
        sys.modules["torchtune.models.convert_weights"] = convert_weights
        """
        what = textwrap.dedent(what)

        with open(executorch, "r+", encoding = "utf-8") as f:
            text = f.read()
            bad = "from enum import Enum\n"
            if bad in text and what not in text:
                text = text.replace(bad + "\n", bad + "\n" + what)
                f.seek(0)
                f.write(text)
                f.truncate()
                logger.info("Unsloth: Patching Executorch to fix get_mapped_key")
    except Exception as e:
        logger.info(f"Unsloth: Failed Executorch with error = {str(e)}")


def fix_diffusers_warnings():
    # Silence Flax classes are deprecated and will be removed in Diffusers v1.0.0.
    os.environ["DIFFUSERS_VERBOSITY"] = "error"


def fix_huggingface_hub():
    # huggingface_hub.is_offline_mode got removed, so add it back
    import huggingface_hub

    if not hasattr(huggingface_hub, "is_offline_mode"):
        huggingface_hub.is_offline_mode = (
            lambda: huggingface_hub.constants.HF_HUB_OFFLINE
        )
