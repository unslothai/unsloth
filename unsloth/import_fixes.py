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
from packaging.version import Version
UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"

# Fix up AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
# MUST do this at the start primarily due to tensorflow causing issues
def fix_message_factory_issue():
    try:
        import google.protobuf.message_factory
        class MessageFactory:
            def CreatePrototype(self, *args, **kwargs): return
            def GetMessages(self, *args, **kwargs): return
            def GetPrototype(self, *args, **kwargs): return
        if not hasattr(google.protobuf.message_factory, "MessageFactory"):
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
            google.protobuf.message_factory.MessageFactory = MessageFactory
        elif hasattr(google.protobuf.message_factory, "MessageFactory") and \
            not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype") and \
            not hasattr(google.protobuf.message_factory, "GetMessageClass"):
            google.protobuf.message_factory.MessageFactory = MessageFactory
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patching protobuf.MessageFactory as it doesn't exist")
        elif hasattr(google.protobuf.message_factory, "MessageFactory") and \
            not hasattr(google.protobuf.message_factory.MessageFactory, "GetPrototype") and \
            hasattr(google.protobuf.message_factory, "GetMessageClass"):
            GetMessageClass = google.protobuf.message_factory.GetMessageClass
            def GetPrototype(self, descriptor):
                return GetMessageClass(descriptor)
            google.protobuf.message_factory.MessageFactory.GetPrototype = GetPrototype
            if UNSLOTH_ENABLE_LOGGING:
                print("Unsloth: Patching protobuf.MessageFactory.GetPrototype")
        pass
    except:
        pass
pass

# Fix Xformers performance issues since 0.0.25
def fix_xformers_performance_issue():
    if importlib.util.find_spec("xformers") is None: return
    xformers_version = importlib_version("xformers")
    if Version(xformers_version) < Version("0.0.29"):
        xformers_location = importlib.util.find_spec("xformers").origin
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
                        if UNSLOTH_ENABLE_LOGGING:
                            print("Unsloth: Patching Xformers to fix some performance issues.")
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"Unsloth: Failed patching Xformers with error = {str(e)}")
pass

# ValueError: 'aimv2' is already used by a Transformers config, pick another name.
def fix_vllm_aimv2_issue():
    if importlib.util.find_spec("vllm") is None: return
    vllm_version = importlib_version("vllm")
    if Version(vllm_version) < Version("0.10.1"):
        vllm_version = importlib.util.find_spec("vllm").origin
        vllm_version = os.path.split(vllm_version)[0]
        ovis_config = Path(vllm_version) / "transformers_utils" / "configs" / "ovis.py"
        try:
            if ovis_config.exists():
                with open(ovis_config, "r+", encoding = "utf-8") as f:
                    text = f.read()
                    # See https://github.com/vllm-project/vllm-ascend/issues/2046
                    if 'AutoConfig.register("aimv2", AIMv2Config)' in text:
                        text = text.replace(
                            'AutoConfig.register("aimv2", AIMv2Config)',
                            '',
                        )
                        text = text.replace(
                            '''backbone_config.pop('model_type')
                backbone_config = AutoConfig.for_model(model_type,
                                                       **backbone_config)''',
                            '''if model_type != "aimv2":
                    backbone_config.pop('model_type')
                    backbone_config = AutoConfig.for_model(model_type, **backbone_config)
                else:
                    backbone_config = AIMv2Config(**backbone_config)'''
                        )
                        f.seek(0)
                        f.write(text)
                        f.truncate()
                        if UNSLOTH_ENABLE_LOGGING:
                            print("Unsloth: Patching vLLM to fix `'aimv2' is already used by a Transformers config, pick another name.`")
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                print(f"Unsloth: Failed patching vLLM with error = {str(e)}")
pass
