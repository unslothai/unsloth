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

"""Compatibility helpers for text emitted by platform probing commands."""

import contextlib
import subprocess


@contextlib.contextmanager
def replace_subprocess_decode_errors():
    """Replace malformed bytes from text subprocesses created in this scope.

    GPU libraries probe commands such as ``nvidia-smi`` and ``ldconfig`` while
    they are imported. Some WSL installations return bytes that are invalid in
    the process locale even though the probing library requests UTF-8 text.
    Such diagnostic output must not make ``import unsloth`` fail.

    Binary subprocesses and callers that choose an explicit ``errors`` policy
    retain their existing behavior. The Popen replacement is also removed when
    the scoped import raises.
    """
    original_popen = subprocess.Popen

    class EncodingSafePopen(original_popen):
        def __init__(self, *args, **kwargs):
            text_mode = (
                kwargs.get("text", False)
                or kwargs.get("universal_newlines", False)
                or kwargs.get("encoding") is not None
            )
            if text_mode:
                kwargs.setdefault("errors", "replace")
            super().__init__(*args, **kwargs)

    subprocess.Popen = EncodingSafePopen
    try:
        yield
    finally:
        # Avoid overwriting a replacement installed independently while the
        # context was active.
        if subprocess.Popen is EncodingSafePopen:
            subprocess.Popen = original_popen
