import types
from unittest.mock import patch

import pytest
import torch
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

import unsloth.utils.testing as test_utils
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel

torch.manual_seed(0)


@pytest.mark.parametrize("bs, seqlen, dtype, n_loop_iters", [(1, 128, "float32", 1)])
def test_cel(bs, seqlen, dtype, n_loop_iters):
    dtype = getattr(torch, dtype)

    model_config = LlamaConfig.from_pretrained("./llama-10m.json")
    model = LlamaForCausalLM(model_config).to("cuda")
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size

    hidden_states = torch.randn(
        bs, seqlen, hidden_dim, dtype=dtype, device="cuda", requires_grad=True
    )
    model.model.forward = types.MethodType(
        lambda *args, **kwargs: (hidden_states,), model.model
    )

    input_ids = torch.randint(0, vocab_size, size=(bs, seqlen), device="cuda")
    labels = input_ids.detach().clone()

    attention_mask = torch.ones((bs, seqlen), device="cuda")
    loss, *_ = model(
        input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
    )
    dX, dW = torch.autograd.grad(loss, [hidden_states, model.lm_head.weight])
    fused_model = patch_model_fused_cel(
        model, use_fused_cel=True, fused_cel_n_loop_iters=n_loop_iters
    )
    fused_loss, *_ = fused_model(
        input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
    )
    dX_fused, dW_fused = torch.autograd.grad(
        fused_loss, [hidden_states, fused_model.lm_head.weight]
    )
    print((fused_loss - loss).abs().max())
    print((dX - dX_fused).abs().max())
    print((dW - dW_fused).abs().max())


#     with patch(model, "model", return_value=hidden_states):
#         out = model()
#         print((out - hidden_states).abs().max())
#     # ref_out = model(input_ids, labels=labels, attention_mask=attention_mask)


# # ref_head = model.lm_head
