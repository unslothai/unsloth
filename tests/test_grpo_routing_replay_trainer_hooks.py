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

"""GRPO trainer installation of MoE routing replay (UNSLOTH_GRPO_ROUTING_REPLAY=1), CPU only.

Runs the exact post-init code rl.py writes into the compiled GRPO trainer: a cold router must
compile with fullgraph=True, and PEFT reference scoring (disable_adapter) must route naturally."""

import ast
import os
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
try:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter  # noqa: F401
except Exception:
    pytest.skip("needs transformers >= 5 router modules", allow_module_level = True)
rr = pytest.importorskip("unsloth_zoo.routing_replay")

RL_PY = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
LAYERS, EXPERTS, TOP_K, PAD = 2, 8, 2, 0


def _post_init_code():
    # The string constant rl.py appends to RLTrainer_post for grpo_trainer.
    for node in ast.walk(ast.parse(RL_PY.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "UNSLOTH_GRPO_ROUTING_REPLAY', '0') == '1' and hasattr(model, 'vllm_engine')" in node.value:
            return node.value
    raise AssertionError("routing replay post-init code not found in rl.py")


def _run_post_init(model, trainer_model = None):
    trainer = SimpleNamespace(model = trainer_model if trainer_model is not None else model,
                              processing_class = SimpleNamespace(pad_token_id = PAD))
    exec(_post_init_code(), {"os": os, "self": trainer, "model": model})


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    torch._dynamo.reset()
    yield
    rr.reset_routing_replay()
    torch._dynamo.reset()


def _moe(seed = 0):
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = LAYERS, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
        num_experts = EXPERTS, num_experts_per_tok = TOP_K, norm_topk_prob = True, pad_token_id = PAD,
    )
    model = Qwen3MoeForCausalLM(config).float().eval()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    model.vllm_engine = None
    return model


def _natural(model, ids):
    captured = {}
    handles = [
        layer.mlp.gate.register_forward_hook(lambda m, a, out, i = i: captured.__setitem__(i, out[2].clone()))
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        logits = model(input_ids = torch.tensor([ids])).logits
    for h in handles:
        h.remove()
    return logits, torch.stack([captured[i] for i in range(LAYERS)], dim = 1)


def test_cold_router_compiles_fullgraph(replay):
    model = _moe()
    _run_post_init(model)
    router = model.model.layers[0].mlp.gate
    assert router._forward_hooks, "routing replay hooks were not installed"
    x = torch.randn(5, 32, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        natural = router.forward(x)[2]
    forced = (natural + 3) % EXPERTS
    rr._STATE.in_forward, rr._STATE.record = True, None
    rr._STATE.current = forced[:, None, :].expand(-1, LAYERS, -1).contiguous()
    try:
        compiled = torch.compile(router, backend = "eager", fullgraph = True)
        logits, weights, ids = compiled(x.requires_grad_(True))  # first call: no eager warm-up
        weights.square().sum().backward()
    finally:
        rr._STATE.in_forward, rr._STATE.current = False, None
    assert torch.equal(ids.long(), forced.long())  # the replay ran inside the compiled graph
    expected = torch.softmax(logits.detach().float().gather(-1, forced.long()), -1)
    torch.testing.assert_close(weights.detach().float(), expected, rtol = 1e-5, atol = 1e-6)
    assert x.grad is not None and bool(x.grad.abs().sum() > 0)


def test_peft_reference_scoring_routes_naturally(replay):
    peft = pytest.importorskip("peft")
    base = _moe()
    ids = [5, 6, 7, 30, 31, 32]
    natural_logits, natural_ids = _natural(base, ids)
    model = peft.get_peft_model(base, peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))
    model.vllm_engine = None
    _run_post_init(model)
    routers = [layer.mlp.gate for layer in model.base_model.model.model.layers]
    hooks_before = [(dict(r._forward_hooks), dict(r._forward_pre_hooks)) for r in routers]
    assert all(h for h, _ in hooks_before)

    forced = ((natural_ids + 3) % EXPERTS).numpy().astype(np.int16)
    rr.record_routed_experts([SimpleNamespace(
        prompt_token_ids = ids[:3], outputs = [SimpleNamespace(token_ids = ids[3:], routed_experts = forced[:-1])],
    )], pad_token_id = PAD)

    with torch.no_grad():
        policy = model(input_ids = torch.tensor([ids])).logits
        # LoRA B starts at zero: any difference from the base comes from the replayed routes.
        assert not torch.allclose(policy, natural_logits, atol = 1e-4)
        with model.disable_adapter():
            reference = model(input_ids = torch.tensor([ids])).logits
            assert all(not r._forward_hooks for r in routers)
        torch.testing.assert_close(reference, natural_logits, rtol = 1e-5, atol = 1e-6)
        again = model(input_ids = torch.tensor([ids])).logits
    torch.testing.assert_close(again, policy, rtol = 1e-5, atol = 1e-6)
    assert [(dict(r._forward_hooks), dict(r._forward_pre_hooks)) for r in routers] == hooks_before


def test_no_replay_installs_nothing(monkeypatch):
    monkeypatch.delenv(rr.ROUTING_REPLAY_ENV, raising = False)
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(_moe(), peft.LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))
    model.vllm_engine = None
    disable_adapter = model.disable_adapter
    _run_post_init(model)
    assert model.disable_adapter == disable_adapter
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
