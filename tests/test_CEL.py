import types
from pathlib import Path

import pytest
import torch
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

import unsloth.utils.testing as test_utils
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel

torch.manual_seed(0)

@pytest.fixture
def model_path():
    PARENT_DIR = Path(__file__).parent.absolute()
    MODEL_CONFIG_PATH = PARENT_DIR / "llama-small.json"

    return MODEL_CONFIG_PATH


@pytest.mark.parametrize("bs", [1])  # , 2, 4])
@pytest.mark.parametrize("seqlen", [256])  # , 512, 1024])
@pytest.mark.parametrize("hidden_size", [128])  # , 4096])
@pytest.mark.parametrize(
    "vocab_size",
    [32000],  # , 128256, 256000]
)  # llama-2, llama-3, gemma
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("n_loop_iters", [1])  # , 2, 4])
def test_cel(bs, seqlen, hidden_size, vocab_size, dtype, n_loop_iters, model_path):
    dtype = getattr(torch, dtype)

    model_config = LlamaConfig.from_pretrained(model_path)
    model_config.update({"vocab_size": vocab_size})
    model_config.update({"hidden_size": hidden_size})

    model = LlamaForCausalLM(model_config).to(dtype).to("cuda")

    # Mock LlamaModel.forward so that we can directly test the CEL loss and derivatives wrt the hidden states (input to the LM head)
    hidden_states = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )
    model.model.forward = types.MethodType(
        lambda *args, **kwargs: (hidden_states,), model.model
    )

    # Input ids aren't actually used, but we need to pass them to the model
    input_ids = torch.randint(0, vocab_size, size=(bs, seqlen), device="cuda")
    labels = input_ids.detach().clone()
    attention_mask = torch.ones((bs, seqlen), device="cuda")

    # Reference loss, dX, dW where dX is the gradients wrt to the hidden states and dW is the gradients wrt to the LM head weight
    loss, *_ = model(
        input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
    )
    dX, dW = torch.autograd.grad(loss, [hidden_states, model.lm_head.weight])

    # Patch the model to use fused CEL
    fused_model = patch_model_fused_cel(
        model, use_fused_cel=True, fused_cel_n_loop_iters=n_loop_iters
    )
    fused_loss, *_ = fused_model(
        input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
    )
    dX_fused, dW_fused = torch.autograd.grad(
        fused_loss, [hidden_states, fused_model.lm_head.weight]
    )

    if dtype == torch.bfloat16:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 1e-5
    else:
        atol = 1e-6

    test_utils.check_all(
        [loss, dX, dW],
        [fused_loss, dX_fused, dW_fused],
        ["loss", "dX", "dW"],
        atol=atol,
        rtol=1e-5,
    )
