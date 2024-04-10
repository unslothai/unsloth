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

import torch
import importlib
import importlib.metadata


def _is_package_available(pkg_name, metadata_name=None):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(pkg_name if metadata_name is None else metadata_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
        
def is_bnb_available():
    return _is_package_available("bitsandbytes")

def is_peft_available():
    return _is_package_available("peft")

def is_transformers_available():
    return _is_package_available("transformers")

def is_flash_attn_available():
    if not torch.version.cuda:
        return False
    return _is_package_available("flashattn")