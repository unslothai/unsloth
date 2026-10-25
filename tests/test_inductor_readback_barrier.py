# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

"""fix_inductor_reduction_readback_race: backport of pytorch/pytorch#190519 for torch < 2.14."""

import importlib.util
import os
import subprocess
import sys
import textwrap
import types

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXES = os.path.join(_ROOT, "unsloth", "import_fixes.py")


def _load_fixes(name):
    spec = importlib.util.spec_from_file_location(name, _FIXES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Buffer:
    def __init__(self):
        self.lines = []

    def writeline(self, line):
        self.lines.append(getattr(line, "line", line))


class _CSE:
    def __init__(self, invalidated):
        self.invalidated_stores = set(invalidated)


def _fake_kernel_class():
    class FakeTritonKernel:
        def __init__(self, invalidated):
            self.cse = _CSE(invalidated)
            self.loads, self.body = _Buffer(), _Buffer()

        def get_load_buffer(self, indexing):
            return self.body if indexing == "hoisted" else self.loads

        def load(self, name, index):
            buffer = self.get_load_buffer(index)
            buffer.writeline(f"tl.load({name})")
            return name

    return FakeTritonKernel


class _Graph:
    def __init__(self, device):
        self.device = torch.device(device)

    def get_current_device_or_throw(self):
        return self.device


@pytest.fixture
def patched(monkeypatch):
    from torch._inductor import config as inductor_config
    from torch._inductor.codegen import triton as triton_codegen

    fake = _fake_kernel_class()
    monkeypatch.setattr(triton_codegen, "TritonKernel", fake)
    monkeypatch.setattr(inductor_config, "post_grad_custom_post_pass", None)
    fixes = _load_fixes("unsloth_import_fixes_readback")
    fixes.fix_inductor_reduction_readback_race()
    return fake, fixes, inductor_config


def _lines(
    kernel,
    device,
    name,
    index = None,
):
    from torch._inductor.virtualized import V
    with V.set_graph_handler(_Graph(device)):
        kernel.load(name, index)


def test_barrier_precedes_reload_of_an_invalidated_store(patched):
    fake, _, _ = patched
    kernel = fake({"buf0"})
    _lines(kernel, "cuda", "buf0")
    assert kernel.loads.lines == ["tl.debug_barrier()", "tl.load(buf0)"]


def test_barrier_goes_to_the_buffer_the_load_goes_to(patched):
    fake, _, _ = patched
    kernel = fake({"buf0"})
    _lines(kernel, "cuda", "buf0", "hoisted")
    assert kernel.body.lines == ["tl.debug_barrier()", "tl.load(buf0)"]
    assert kernel.loads.lines == []


def test_no_barrier_for_other_buffers_or_cpu(patched):
    fake, _, _ = patched
    kernel = fake({"buf0"})
    _lines(kernel, "cuda", "arg0_1")
    _lines(kernel, "cpu", "buf0")
    assert kernel.loads.lines == ["tl.load(arg0_1)", "tl.load(buf0)"]


def test_idempotent_and_keys_the_compile_cache(patched):
    pytest.importorskip("torch._inductor.custom_graph_pass")
    fake, fixes, inductor_config = patched
    first = fake.load
    fixes.fix_inductor_reduction_readback_race()
    assert fake.load is first
    key = inductor_config.post_grad_custom_post_pass
    from torch._inductor.custom_graph_pass import CustomGraphPass

    assert isinstance(key, CustomGraphPass)
    assert key.uuid() == "unsloth-inductor-readback-barrier-v1"
    kernel = fake({"buf0"})
    _lines(kernel, "cuda", "buf0")
    assert kernel.loads.lines.count("tl.debug_barrier()") == 1


def test_keeps_a_user_post_grad_pass(monkeypatch):
    from torch._inductor import config as inductor_config
    from torch._inductor.codegen import triton as triton_codegen

    monkeypatch.setattr(triton_codegen, "TritonKernel", _fake_kernel_class())
    user_pass = object()
    monkeypatch.setattr(inductor_config, "post_grad_custom_post_pass", user_pass)
    _load_fixes("unsloth_import_fixes_readback_user").fix_inductor_reduction_readback_race()
    assert inductor_config.post_grad_custom_post_pass is user_pass


def test_skips_a_torch_that_already_has_the_barrier(monkeypatch):
    from torch._inductor.codegen import triton as triton_codegen

    fake = _fake_kernel_class()

    def load(self, name, index):  # mirrors torch >= 2.14
        if name in self.cse.invalidated_stores:
            pass
        return name

    fake.load = load
    monkeypatch.setattr(triton_codegen, "TritonKernel", fake)
    _load_fixes("unsloth_import_fixes_readback_fixed").fix_inductor_reduction_readback_race()
    assert fake.load is load


def test_torch_without_custom_graph_pass_skips_the_fx_graph_cache():
    # torch 2.4 / 2.5 config is a plain module; newer torch lets TORCHINDUCTOR_FX_GRAPH_CACHE override writes.
    config = types.SimpleNamespace(fx_graph_cache = True, post_grad_custom_post_pass = None)
    fixes = _load_fixes("unsloth_import_fixes_readback_old")
    fixes._key_inductor_caches_on_readback_barrier(config, None)
    assert config.fx_graph_cache is False
    assert config.post_grad_custom_post_pass is None


_GPU_RUNNER = textwrap.dedent(
    """
    import importlib.util, sys, torch
    from torch._inductor.utils import run_and_get_triton_code
    if sys.argv[1] == "patched":
        spec = importlib.util.spec_from_file_location("fixes", sys.argv[2])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes.fix_inductor_reduction_readback_race()

    def fn(a, b, w, bias):
        y = a + b
        mean = y.mean(-1, keepdim = True)
        var = y.var(-1, keepdim = True, unbiased = False)
        return y, (y - mean) * torch.rsqrt(var + 1e-6) * w + bias

    n = 4096
    inputs = [torch.randn(64, n, device = "cuda"), torch.randn(64, n, device = "cuda"),
              torch.randn(n, device = "cuda"), torch.randn(n, device = "cuda")]
    compiled = torch.compile(fn, backend = "inductor")
    code = run_and_get_triton_code(compiled, *inputs)
    for got, want in zip(compiled(*inputs), fn(*inputs)):
        torch.testing.assert_close(got, want)
    print("BARRIER" if "tl.debug_barrier()" in code else "NO_BARRIER")
    """
)


@pytest.mark.skipif(not has_real_cuda(), reason = "Inductor Triton codegen needs a GPU")
def test_generated_reduction_syncs_before_the_reload(tmp_path):
    env = dict(
        os.environ,
        TORCHINDUCTOR_CACHE_DIR = str(tmp_path / "inductor"),
        TORCHINDUCTOR_FX_GRAPH_CACHE = "0",
    )
    run = lambda arm: subprocess.run(
        [sys.executable, "-c", _GPU_RUNNER, arm, _FIXES],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    patched = run("patched")
    assert patched.returncode == 0, patched.stderr[-3000:]
    assert patched.stdout.strip().splitlines()[-1] == "BARRIER"


def _user_pass_class(base_name = "CustomGraphPass"):
    custom_graph_pass = pytest.importorskip("torch._inductor.custom_graph_pass")
    base = getattr(custom_graph_pass, base_name, None)
    if base is None:
        pytest.skip(f"torch has no {base_name}")

    class UserPass(base):
        def __init__(self):
            self.calls = []

        def __call__(self, graph, *args, **kwargs):
            self.calls.append((graph, args, kwargs))

        def uuid(self):
            return "user-pass-v1"

    return UserPass


def _install_with(monkeypatch, existing):
    from torch._inductor import config as inductor_config
    from torch._inductor.codegen import triton as triton_codegen

    monkeypatch.setattr(triton_codegen, "TritonKernel", _fake_kernel_class())
    monkeypatch.setattr(inductor_config, "post_grad_custom_post_pass", existing)
    fixes = _load_fixes("unsloth_import_fixes_readback_keyed")
    fixes.fix_inductor_reduction_readback_race()
    return fixes, inductor_config


def test_a_user_custom_graph_pass_keys_the_cache_on_the_barrier_too(monkeypatch):
    """A pre-set CustomGraphPass must carry the marker in its uuid and still run."""
    from torch._inductor.custom_graph_pass import CustomGraphPass

    user = _user_pass_class()()
    fixes, inductor_config = _install_with(monkeypatch, user)
    keyed = inductor_config.post_grad_custom_post_pass
    # torch keeps the cache only for CustomGraphPass instances with a truthy uuid.
    assert isinstance(keyed, CustomGraphPass) and keyed.uuid()
    assert keyed.uuid() != user.uuid()
    assert keyed.uuid() == ("unsloth-inductor-readback-barrier-v1", "user-pass-v1")
    graph = object()
    keyed(graph)
    assert user.calls == [(graph, (), {})]
    # Idempotent: no re-wrap.
    fixes._key_inductor_caches_on_readback_barrier(inductor_config, CustomGraphPass)
    assert inductor_config.post_grad_custom_post_pass is keyed


def test_an_inference_aware_user_pass_still_gets_is_inference(monkeypatch):
    user = _user_pass_class("CustomInferenceAwareGraphPass")()  # skips on torch without it
    from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass

    _, inductor_config = _install_with(monkeypatch, user)
    keyed = inductor_config.post_grad_custom_post_pass
    # post_grad.py passes is_inference only to CustomInferenceAwareGraphPass instances.
    assert isinstance(keyed, CustomInferenceAwareGraphPass)
    keyed("g", is_inference = True)
    assert user.calls == [("g", (), {"is_inference": True})]


def test_a_list_of_user_passes_gets_the_marker_appended(monkeypatch):
    user = _user_pass_class()()
    _, inductor_config = _install_with(monkeypatch, [user])
    passes = inductor_config.post_grad_custom_post_pass
    assert isinstance(passes, list) and passes[0] is user
    assert [p.uuid() for p in passes] == ["user-pass-v1", "unsloth-inductor-readback-barrier-v1"]


def test_a_raw_callable_user_pass_is_left_alone(monkeypatch):
    # torch bypasses the FX cache for raw callables; wrapping one would make it cacheable unkeyed.
    def raw(graph):
        return None

    _, inductor_config = _install_with(monkeypatch, raw)
    assert inductor_config.post_grad_custom_post_pass is raw


_CACHE_RUNNER = textwrap.dedent(
    """
    import importlib.util, json, sys, torch
    from torch._inductor import config
    from torch._inductor.custom_graph_pass import CustomGraphPass
    from torch._dynamo.utils import counters

    class UserPass(CustomGraphPass):
        def __call__(self, graph):
            return None

        def uuid(self):
            return "user-pass-v1"

    config.post_grad_custom_post_pass = UserPass()
    if sys.argv[1] == "patched":
        spec = importlib.util.spec_from_file_location("fixes", sys.argv[2])
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        fixes._key_inductor_caches_on_readback_barrier(config, CustomGraphPass)
    torch.compile(lambda x: (x.sin() + 1).sum(-1), backend = "inductor")(torch.randn(4, 8))
    print(json.dumps(dict(counters["inductor"])))
    """
)


def test_a_warm_cache_from_before_the_fix_is_not_served_with_a_user_pass(tmp_path):
    """End to end on CPU (the key, not the kernel): a cache warmed without the fix must miss."""
    pytest.importorskip("torch._inductor.custom_graph_pass")
    import json

    runner = tmp_path / "runner.py"
    runner.write_text(_CACHE_RUNNER)
    env = {
        **os.environ,
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }

    def run(arm):
        out = subprocess.run(
            [sys.executable, str(runner), arm, _FIXES],
            env = env,
            capture_output = True,
            text = True,
            timeout = 600,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        return json.loads(out.stdout.strip().splitlines()[-1])

    warm = run("base")
    assert warm.get("fxgraph_cache_miss", 0) >= 1
    assert run("base").get("fxgraph_cache_hit", 0) >= 1  # the key is otherwise stable
    patched = run("patched")
    assert patched.get("fxgraph_cache_hit", 0) == 0
    assert patched.get("fxgraph_cache_miss", 0) >= 1
