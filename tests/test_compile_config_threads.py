# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""torch >= 2.12 keeps config assignments per thread. A reentrant checkpoint (Unsloth's gradient
checkpointing) recomputes on autograd's device thread, and the grad graph it compiles there, plus
that graph's lazily compiled backward, saw torch's defaults: no kernel-file cache_key_tag, so a warm
FX cache could serve the previous Unsloth's triton_op kernels, and no readback-barrier cache key.
CPU only: CheckpointFunction.backward is run on a fresh thread, the context the device thread has."""

import importlib.util
import os
import threading
import uuid
from contextvars import ContextVar
from pathlib import Path

import pytest
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.utils.checkpoint as checkpoint_module

pytest.importorskip("triton")
torch_config = pytest.importorskip("unsloth_zoo.torch_config")

ROOT = Path(__file__).resolve().parents[1]
_COMPILER_CONFIG = getattr(torch.compiler, "config", None)
if not hasattr(_COMPILER_CONFIG, "cache_key_tag"):
    pytest.skip("torch has no compile cache key tag", allow_module_level = True)
try:
    from torch._inductor.custom_graph_pass import CustomGraphPass
except ImportError:
    pytest.skip("torch has no CustomGraphPass", allow_module_level = True)

_KEYS = [(_COMPILER_CONFIG, "cache_key_tag"), (inductor_config, "post_grad_custom_post_pass")]


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        Path(module.__file__).resolve().is_relative_to(ROOT)
    )  # this tree, not an installed unsloth
    return module


def _invalidate(module):
    if hasattr(module, "_is_dirty"):
        module._is_dirty = True
    for attr, default in (
        ("_hash_dirty_var", True),
        ("_hash_cache_var", None),
        ("_get_dict_dirty_keys_var", None),
        ("_get_dict_cache_var", None),
    ):
        if isinstance(getattr(module, attr, None), ContextVar):
            setattr(module, attr, ContextVar(attr, default = default))


@pytest.fixture(autouse = True)
def _restore_torch_config():
    saved = []
    for module, name in _KEYS:
        entry = module._config[name]
        override = entry.user_override
        current = override.get() if isinstance(override, ContextVar) else override
        saved.append((module, entry, current, entry.env_value_default, entry.env_value_force))
    torch._dynamo.reset()
    yield
    for module, entry, current, env_default, env_force in saved:
        if isinstance(entry.user_override, ContextVar):
            entry.user_override.set(current)
        else:
            entry.user_override = current
        entry.env_value_default = env_default
        entry.env_value_force = env_force
        _invalidate(module)
    torch._dynamo.reset()


def _in_thread(fn, name = "device"):
    box = {}

    def run():
        try:
            box["value"] = fn()
        except BaseException as error:  # noqa: BLE001
            box["error"] = error

    thread = threading.Thread(target = run, name = name)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _kernel_tag(tmp_path):
    triton_launch = _load("unsloth/kernels/triton_launch.py", "unsloth_triton_launch_under_test")
    kernel = tmp_path / "fake_kernel.py"
    kernel.write_text(f"# {uuid.uuid4()}\n")
    triton_launch.tag_compile_cache(str(kernel))
    tags = _COMPILER_CONFIG.cache_key_tag.split(",")
    tag = next(t for t in tags if t.startswith(f"unsloth/{os.path.basename(kernel)}:"))
    return tag


def test_kernel_cache_tag_reaches_other_threads(tmp_path):
    tag = _kernel_tag(tmp_path)
    assert tag in _in_thread(lambda: _COMPILER_CONFIG.cache_key_tag).split(",")


def test_checkpoint_recompute_compiles_with_the_kernel_cache_tag(tmp_path, monkeypatch):
    original = checkpoint_module.CheckpointFunction.backward

    def device_thread_backward(ctx, *grads):
        return _in_thread(lambda: original(ctx, *grads))

    monkeypatch.setattr(
        checkpoint_module.CheckpointFunction, "backward", staticmethod(device_thread_backward)
    )
    tag = _kernel_tag(tmp_path)
    seen = []
    key = f"unsloth-test-recorder-{uuid.uuid4()}"  # a warm FX cache would skip the pass

    class Recorder(CustomGraphPass):
        def __call__(self, graph):
            seen.append((threading.current_thread().name, _COMPILER_CONFIG.cache_key_tag))

        def uuid(self):
            return key

    # Test plumbing: the recorder itself must reach the device thread.
    torch_config.set_torch_config(inductor_config, "post_grad_custom_post_pass", Recorder())
    compiled = torch.compile(lambda x, w: torch.nn.functional.gelu(x @ w).sum(-1))
    x = torch.randn(4, 8, requires_grad = True)
    w = torch.randn(8, 8, requires_grad = True)
    loss = checkpoint_module.checkpoint(compiled, x, w, use_reentrant = True).sum()
    loss.backward()  # grad-mode guard miss: grad graph + its lazy backward compile on "device"
    on_device = [tags for name, tags in seen if name == "device"]
    assert len(on_device) == 2, seen
    assert all(tag in tags.split(",") for tags in on_device), (tag, on_device)


def test_readback_barrier_cache_key_reaches_other_threads():
    import_fixes = _load("unsloth/import_fixes.py", "unsloth_import_fixes_under_test")
    torch_config.set_torch_config(inductor_config, "post_grad_custom_post_pass", None)
    import_fixes._key_inductor_caches_on_readback_barrier(inductor_config, CustomGraphPass)
    assert getattr(inductor_config.post_grad_custom_post_pass, "_unsloth_readback_barrier", False)
    on_device = _in_thread(lambda: inductor_config.post_grad_custom_post_pass)
    assert getattr(on_device, "_unsloth_readback_barrier", False), on_device
