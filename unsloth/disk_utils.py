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
"""Access to unsloth_zoo.disk_utils that survives an older installed zoo.

The disk sizing and Kaggle detection logic is owned by
`unsloth_zoo.disk_utils` so the two packages cannot disagree about whether we
are on Kaggle. unsloth pins a new enough unsloth_zoo, but a stale environment
that satisfies `import unsloth_zoo` and not `unsloth_zoo.disk_utils` must
still be able to save a model, so the fallbacks below keep the environment
question answered correctly and turn the new disk guard into a no-op rather
than an ImportError.
"""

import os

__all__ = [
    "KAGGLE_TMP",
    "KAGGLE_WORKING",
    "is_kaggle_environment",
    "is_colab_environment",
    "free_bytes",
    "logical_numel",
    "model_16bit_bytes",
    "model_logical_numel",
    "estimate_gguf_export_bytes",
    "kaggle_tmp_redirect",
    "HAS_ZOO_DISK_UTILS",
]

try:
    from unsloth_zoo.disk_utils import (
        KAGGLE_TMP,
        KAGGLE_WORKING,
        is_kaggle_environment,
        is_colab_environment,
        free_bytes,
        logical_numel,
        model_16bit_bytes,
        model_logical_numel,
        estimate_gguf_export_bytes,
        kaggle_tmp_redirect,
    )
    HAS_ZOO_DISK_UTILS = True

except ImportError:
    HAS_ZOO_DISK_UTILS = False

    KAGGLE_TMP = "/tmp"
    KAGGLE_WORKING = "/kaggle/working"
    _TRUE = ("1", "true", "yes", "on")

    def is_kaggle_environment():
        # Kept in step with unsloth_zoo.disk_utils: only a real kernel sets KAGGLE_KERNEL_RUN_TYPE and
        # only the Kaggle image has /kaggle/working, whereas KAGGLE_USERNAME / KAGGLE_KEY /
        # KAGGLE_CONFIG_DIR are what people export for the Kaggle CLI on their own machines.
        override = os.environ.get("UNSLOTH_IS_KAGGLE", None)
        if override is not None:
            return str(override).strip().lower() in _TRUE
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").strip():
            return False
        try:
            return os.path.isdir(KAGGLE_WORKING)
        except Exception:
            return False

    def is_colab_environment():
        for key in os.environ:
            if key.startswith("COLAB_"):
                return True
        return False

    def free_bytes(path):
        import shutil

        try:
            probe = os.path.abspath(os.path.expanduser(str(path)))
        except Exception:
            return None
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        try:
            return shutil.disk_usage(probe).free
        except Exception:
            return None

    def logical_numel(param, name = ""):
        # Packed storage cannot be unpacked without the zoo's knowledge of the packing schemes, so report
        # what numel() says, exactly as the code did before it asked. Barely reachable: model_16bit_bytes
        # is 0 here and every caller returns before sizing anything.
        try:
            return int(param.numel())
        except Exception:
            return 0

    def model_logical_numel(model):
        return 0

    def model_16bit_bytes(model):
        return 0

    def estimate_gguf_export_bytes(*args, **kwargs):
        # 0 means "unmeasurable", and every caller treats that as "do not block", so an old unsloth_zoo
        # behaves exactly as it did.
        return 0

    def kaggle_tmp_redirect(save_directory, *args, **kwargs):
        return save_directory, None
