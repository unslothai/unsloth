# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Compiled-inference path of _install_offload_embedding_hooks: exact op under compile, module elsewhere."""

import ast, functools, os
import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")

try:
    from unsloth_zoo.offloaded_embedding import offloaded_embedding as _zoo_op
except Exception:
    _zoo_op = None

pytestmark = pytest.mark.skipif(
    _zoo_op is None, reason = "unsloth_zoo without the shared offloaded_embedding op"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


def _load_installer():
    src = open(VISION, encoding = "utf-8").read()
    ns = {"torch": torch}
    wanted = {"_is_scaled_word_embedding_forward", "_install_offload_embedding_hooks"}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(src, node), ns)
    assert wanted <= ns.keys(), "offload embedding installer not found in vision.py"
    return ns["_install_offload_embedding_hooks"]


install = _load_installer()


class ScaledWordEmbedding(nn.Embedding):
    # Same forward as transformers' Gemma*TextScaledWordEmbedding.
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx = None,
        embed_scale = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent = False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Float32ScaledWordEmbedding(ScaledWordEmbedding):
    # Like Unsloth's float32 Gemma patch: a different product.
    def forward(self, input_ids: torch.Tensor):
        return (
            nn.functional.embedding(input_ids, self.weight, self.padding_idx).float()
            * self.embed_scale
        )


V, H = 5003, 64


def _make(kind = "plain", **kw):
    torch.manual_seed(0)
    if kind == "plain":
        emb = nn.Embedding(V, H, **kw)
    elif kind == "scaled":
        emb = ScaledWordEmbedding(V, H, padding_idx = 0, embed_scale = H**0.5)
    else:
        emb = Float32ScaledWordEmbedding(V, H, padding_idx = 0, embed_scale = H**0.5)
    return emb.to(torch.bfloat16).requires_grad_(False)


def _head(device):
    return nn.Linear(H, 8, bias = False, dtype = torch.bfloat16).to(device)


@pytest.mark.parametrize(
    "kind, kw, expected",
    [
        ("plain", {}, True),
        ("scaled", {}, True),
        ("plain", {"max_norm": 1.0}, False),
        ("plain", {"sparse": True}, False),
        ("float32", {}, False),
    ],
)
def test_eligibility(kind, kw, expected):
    emb = _make(kind, **kw)
    assert install(emb, _head("cpu"), torch.device("cpu")) is True
    assert bool(getattr(emb, "_unsloth_offload_op_forward", False)) is expected
    assert ("forward" in emb.__dict__) is expected


class PreprocessingScaledEmbedding(ScaledWordEmbedding):
    # Extra work before the same return: the op would skip the clamp.
    def forward(self, input_ids: torch.Tensor):
        input_ids = input_ids.clamp(max = 10)
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class ClampingEmbedding(nn.Embedding):
    def forward(self, input):
        return super().forward(input.clamp(max = 10))


class OverScaledEmbedding(ClampingEmbedding):
    # The scaled forward verbatim, but super() is not nn.Embedding.forward.
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("embed_scale", torch.tensor(2.0), persistent = False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


@pytest.mark.parametrize(
    "emb",
    [
        lambda: PreprocessingScaledEmbedding(V, H, padding_idx = 0, embed_scale = 2.0),
        lambda: OverScaledEmbedding(V, H),
    ],
)
def test_scaled_forward_must_match_exactly(emb):
    emb = emb().to(torch.bfloat16).requires_grad_(False)
    install(emb, _head("cpu"), torch.device("cpu"))
    assert not getattr(emb, "_unsloth_offload_op_forward", False)


def _doubling(f):
    @functools.wraps(f)
    def wrapper(self, input_ids):
        return f(self, input_ids) * 2

    return wrapper


class DecoratedScaledEmbedding(nn.Embedding):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("embed_scale", torch.tensor(2.0), persistent = False)

    @_doubling
    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


def test_decorated_scaled_forward_is_declined():
    emb = DecoratedScaledEmbedding(V, H, padding_idx = 0)
    emb = emb.to(torch.bfloat16).requires_grad_(False)
    install(emb, _head("cpu"), torch.device("cpu"))
    assert not getattr(emb, "_unsloth_offload_op_forward", False)


@needs_cuda
def test_max_norm_set_after_install_keeps_module():
    ref = _make()
    emb = _make()
    head = _head("cuda")
    install(emb, head, torch.device("cuda"))
    assert getattr(emb, "_unsloth_offload_op_forward", False)
    emb.max_norm = ref.max_norm = 0.5
    ids = torch.randint(0, V, (4, 9), device = "cuda")
    torch._dynamo.reset()
    with torch.no_grad():
        out = torch.compile(lambda i: emb(i), backend = "aot_eager")(ids)
        expected = ref(ids.cpu())
    assert torch.equal(out.cpu(), expected)


def test_globally_patched_embedding_forward_is_declined(monkeypatch):
    original = nn.Embedding.forward
    monkeypatch.setattr(nn.Embedding, "forward", lambda self, input: original(self, input) * 3)
    emb = _make()
    install(emb, _head("cpu"), torch.device("cpu"))
    assert not getattr(emb, "_unsloth_offload_op_forward", False)


@needs_cuda
def test_other_pre_hook_keeps_module_path():
    emb = _make()
    head = _head("cuda")
    install(emb, head, torch.device("cuda"))
    seen = []
    emb.register_forward_pre_hook(lambda m, args: seen.append(args[0].device))
    ids = torch.randint(0, V, (4, 9), device = "cuda")
    torch._dynamo.reset()
    with torch.no_grad():
        out = torch.compile(lambda i: emb(i), backend = "aot_eager")(ids)
    assert seen and all(d.type == "cpu" for d in seen)
    assert torch.equal(out.cpu(), _make()(ids.cpu()))


def test_input_keyword_still_works():
    ref = _make()
    emb = _make()
    install(emb, _head("cpu"), torch.device("cpu"))
    assert getattr(emb, "_unsloth_offload_op_forward", False)
    ids = torch.randint(0, V, (2, 5))
    assert torch.equal(emb(input = ids), ref(ids))


def test_class_forward_patched_after_install_is_used():
    class Patchable(nn.Embedding):
        pass

    torch.manual_seed(0)
    emb = Patchable(V, H).to(torch.bfloat16).requires_grad_(False)
    install(emb, _head("cpu"), torch.device("cpu"))
    Patchable.forward = lambda self, ids: nn.Embedding.forward(self, ids) * 3
    ids = torch.randint(0, V, (2, 5))
    assert torch.equal(emb(ids), nn.Embedding.forward(emb, ids) * 3)


def test_existing_instance_forward_is_left_alone():
    emb = _make()
    marker = lambda ids: nn.Embedding.forward(emb, ids)
    emb.forward = marker
    install(emb, _head("cpu"), torch.device("cpu"))
    assert emb.forward is marker and not getattr(emb, "_unsloth_offload_op_forward", False)


@pytest.mark.parametrize("kind", ["plain", "scaled"])
def test_eager_unchanged_on_cpu(kind):
    ref = _make(kind)
    emb = _make(kind)
    install(emb, _head("cpu"), torch.device("cpu"))
    ids = torch.randint(0, V, (3, 7))
    assert torch.equal(emb(ids), ref(ids))


@needs_cuda
@pytest.mark.parametrize("kind", ["plain", "scaled", "float32"])
def test_compiled_no_grad_matches_module_and_breaks_only_when_declined(kind):
    from torch._dynamo.utils import counters

    ref = _make(kind)
    emb = _make(kind)
    head = _head("cuda")
    install(emb, head, torch.device("cuda"))

    def step(ids):
        return head(emb(ids).to(torch.bfloat16))

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(step, backend = "aot_eager")
    ids = torch.randint(0, V, (4, 9), device = "cuda")
    with torch.no_grad():
        out = compiled(ids)
        expected = head(ref(ids.cpu()).cuda().to(torch.bfloat16))
    breaks = sum(counters["graph_break"].values())
    assert torch.equal(out, expected)
    if kind == "float32":
        assert breaks > 0  # declined: the disabled hooks still run
    else:
        assert breaks == 0, dict(counters["graph_break"])


@needs_cuda
@pytest.mark.parametrize("kind", ["plain", "scaled"])
def test_cuda_graph_decode_loop(kind):
    ref = _make(kind)
    emb = _make(kind)
    head = nn.Linear(H, V, bias = False, dtype = torch.bfloat16).cuda()
    install(emb, head, torch.device("cuda"))

    def step(ids):
        return head(emb(ids)).argmax(-1)

    compiled = torch.compile(step, fullgraph = True, mode = "reduce-overhead")
    ids = torch.randint(0, V, (4, 1), device = "cuda")
    ref_ids = ids.clone()
    with torch.no_grad():
        for _ in range(8):
            torch.compiler.cudagraph_mark_step_begin()
            ids = compiled(ids).clone()
            ref_ids = head(ref(ref_ids.cpu()).cuda()).argmax(-1)
            assert torch.equal(ids, ref_ids)


@needs_cuda
def test_grad_enabled_keeps_module_and_hooks():
    # Frozen table under grad: enable_input_require_grads' hook must still fire.
    emb = _make()
    head = _head("cuda").requires_grad_(True)
    install(emb, head, torch.device("cuda"))
    fired = []
    emb.register_forward_hook(lambda m, i, o: (fired.append(1), o.requires_grad_(True))[1])

    def step(ids):
        return head(emb(ids)).float().sum()

    torch._dynamo.reset()
    loss = torch.compile(step, backend = "aot_eager")(torch.randint(0, V, (2, 5), device = "cuda"))
    loss.backward()
    assert fired and head.weight.grad is not None and head.weight.grad.abs().sum() > 0
