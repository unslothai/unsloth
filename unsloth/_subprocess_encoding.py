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

import contextlib
import subprocess


@contextlib.contextmanager
def replace_invalid_subprocess_text():
    """Replace malformed bytes in text subprocesses created within the context.

    GPU discovery code and imported dependencies may execute system utilities
    such as driver probes. Their diagnostic output is not guaranteed to match
    the process locale, particularly when Windows tools are exposed through
    WSL. Python otherwise decodes text-mode output strictly and can turn one
    malformed diagnostic byte into an import-time UnicodeDecodeError.

    Callers that explicitly select an error policy remain authoritative, and
    binary subprocesses are unchanged.
    """
    original_popen = subprocess.Popen

    def safe_popen(*args, **kwargs):
        text_mode = (
            kwargs.get("text", False)
            or kwargs.get("universal_newlines", False)
            or kwargs.get("encoding") is not None
        )
        if text_mode and kwargs.get("errors") is None:
            kwargs["errors"] = "replace"
        return original_popen(*args, **kwargs)

    subprocess.Popen = safe_popen
    try:
        yield
    finally:
        # Avoid overwriting an unrelated patch installed while the context was
        # active. Nested uses restore correctly because each owns its wrapper.
        if subprocess.Popen is safe_popen:
            subprocess.Popen = original_popen
