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


import gc
from contextlib import contextmanager

import os
import pytest
import numpy as np
import torch
import torch._dynamo as dynamo


@contextmanager
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    yield

@pytest.fixture(autouse=True)
def reset_dyno_state():
    cache_limit = dynamo.config.cache_size_limit
    try:
        dynamo.config.cache_size_limit = 8192
        dynamo.reset()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        yield {}
    except RuntimeError as err:
        raise err
    finally:
        dynamo.config.cache_size_limit = cache_limit
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def assert_all_close(a: torch.Tensor, b: torch.Tensor, rtol=0, atol=1e-1) -> None:
    """
    Check that all elements of tensors a and b are within provided thresholds.
    """
    assert a.shape == b.shape, f"Shapes don't match: {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"Dtypes don't match: {a.dtype} != {b.dtype}"
    assert a.device == b.device, f"Devices don't match: {a.device} != {b.device}"
    max_abs_diff = torch.max(torch.abs(a - b))
    rel_diff = torch.abs(a / b)
    max_rel_diff = torch.max(rel_diff)
    mismatch_elements = torch.sum(torch.abs(a - b) > atol + rtol * torch.abs(b))
    nb_elements = torch.numel(a)
    msg = (
        f"Differences: "
        f"{max_abs_diff:.3f} (max abs), "
        f"{max_rel_diff:.3f} (max rel), "
        f"{mismatch_elements}/{nb_elements} (mismatch elements)"
    )
    assert torch.allclose(a, b, rtol=rtol, atol=atol), msg