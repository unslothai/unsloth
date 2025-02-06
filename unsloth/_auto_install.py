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

try: import torch
except: raise ImportError('Install torch via `pip install torch`')
from packaging.version import Version as V
v = V(torch.__version__)
cuda = str(torch.version.cuda)
is_ampere = torch.cuda.get_device_capability()[0] >= 8
if cuda != "12.1" and cuda != "11.8" and cuda != "12.4" and cuda != "12.6": raise RuntimeError(f"CUDA = {cuda} not supported!")
if   v <= V('2.1.0'): raise RuntimeError(f"Torch = {v} too old!")
elif v <= V('2.1.1'): x = 'cu{}{}-torch211'
elif v <= V('2.1.2'): x = 'cu{}{}-torch212'
elif v  < V('2.3.0'): x = 'cu{}{}-torch220'
elif v  < V('2.4.0'): x = 'cu{}{}-torch230'
elif v  < V('2.5.0'): x = 'cu{}{}-torch240'
elif v  < V('2.5.1'): x = 'cu{}{}-torch250'
elif v <= V('2.5.1'): x = 'cu{}{}-torch251'
elif v  < V('2.7.0'): x = 'cu{}{}-torch260'
else: raise RuntimeError(f"Torch = {v} too new!")
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
print(f'pip install --upgrade pip && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git"')