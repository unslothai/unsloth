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

# The single source of truth for the version of THIS package.
#
# It lives alone in a module with no imports for two reasons. First, pyproject.toml
# reads it through `version = {attr = "unsloth._version.__version__"}`, and setuptools
# resolves that with a static AST parse as long as the value stays a plain string
# literal -- so building a wheel never has to import torch. Keep it a literal.
# Second, the MLX path in `unsloth/__init__.py` is deliberately torch-free and cannot
# reach `unsloth.models._utils`, where this used to live; importing that module pulls
# in torch, transformers, trl, peft and the Triton kernels. A leaf module both paths
# can import is what stops the MLX branch from having to borrow unsloth_zoo's version
# instead, which reported a different package's number entirely.
__version__ = "2026.8.12"
