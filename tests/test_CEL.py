import types
from pathlib import Path

import pytest
import torch
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

import unsloth.utils.testing as test_utils
from unsloth.kernels.fused_cel import fused_cel_layer
from unsloth.utils.memory import empty_cache

torch.manual_seed(0)


@pytest.fixture
def model_path():
    PARENT_DIR = Path(__file__).parent.absolute()
    MODEL_CONFIG_PATH = PARENT_DIR / "llama-small.json"

    return MODEL_CONFIG_PATH


def ref_cel(hidden_states, lm_head_weight, labels):
    vocab_size = lm_head_weight.shape[0]
    logits = hidden_states @ lm_head_weight.T
    logits = logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    return loss


def run_cel(fn, hidden_states, lm_head_weight, labels, **kwargs):
    loss = fn(hidden_states, lm_head_weight, labels, **kwargs)
    dX, dW = torch.autograd.grad(loss, [hidden_states, lm_head_weight])
    return loss, dX, dW


# Test will fail for dX when hidden_size > 4096 and n_loop_iters > 1 and vocab_size == 32000
# Comment out the pytest.skip if you want to run
@pytest.mark.parametrize("bs", [1])  # , 2, 4])
@pytest.mark.parametrize("seqlen", [256])  # , 512, 1024])
@pytest.mark.parametrize("hidden_size", [128, 4096])
@pytest.mark.parametrize(
    "vocab_size",
    [32000, 128256],  # , 256000]
)  # llama-2, llama-3, gemma
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("n_loop_iters", [1, 2])  # , 2])  # , 2, 4])
def test_cel(bs, seqlen, hidden_size, vocab_size, dtype, n_loop_iters):
    dtype = getattr(torch, dtype)
    # if hidden_size >= 4096 and n_loop_iters > 1:
    #     pytest.skip("Failure cases for dX")
    # if not (hidden_size >= 4096 and n_loop_iters > 1 and vocab_size == 32000):
    #     pytest.skip("Skipping, running only failure cases for dX")

    hidden_states = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )

    lm_head_weight = torch.randn(
        (vocab_size, hidden_size), dtype=dtype, device="cuda", requires_grad=True
    )

    # Input ids aren't actually used, but we need to pass them to the model
    labels = torch.randint(0, vocab_size, size=(bs, seqlen), device="cuda")

    # Reference loss, dX, dW where dX is the gradients wrt to the hidden states and dW is the gradients wrt to the LM head weight
    loss, dX, dW = run_cel(ref_cel, hidden_states, lm_head_weight, labels)
    fused_loss, dX_fused, dW_fused = run_cel(
        fused_cel_layer,
        hidden_states,
        lm_head_weight,
        labels,
        n_loop_iters=n_loop_iters,
        ignore_index=-100,
        reduction="mean",
    )
    if dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-3  # Fails if < 1e-3
    elif dtype == torch.float16:
        atol, rtol = 1e-4, 1e-4  # Fails if < 1e-4
    else:
        atol, rtol = 1e-6, 1e-6

    test_utils.check_all(
        [loss, dX, dW],
        [fused_loss, dX_fused, dW_fused],
        ["loss", "dX", "dW"],
        atol=atol,
        rtol=rtol,
    )
    empty_cache()


# @pytest.mark.parametrize("bs", [1])  # , 2, 4])
# @pytest.mark.parametrize("seqlen", [256])  # , 512, 1024])
# @pytest.mark.parametrize("hidden_size", [4096])
# @pytest.mark.parametrize(
#     "vocab_size",
#     [32000, 128256],  # , 256000]
# )  # llama-2, llama-3, gemma
# @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
# @pytest.mark.parametrize("n_loop_iters", [1, 2])  # , 2, 4])
# def test_cel(bs, seqlen, hidden_size, vocab_size, dtype, n_loop_iters, model_path):
#     dtype = getattr(torch, dtype)

#     model_config = LlamaConfig.from_pretrained(model_path)
#     model_config.update({"vocab_size": vocab_size})
#     model_config.update({"hidden_size": hidden_size})

#     model = LlamaForCausalLM(model_config).to(dtype).to("cuda")

#     # Mock LlamaModel.forward so that we can directly test the CEL loss and derivatives wrt the hidden states (input to the LM head)
#     hidden_states = torch.randn(
#         bs, seqlen, hidden_size, dtype=dtype, device="cuda", requires_grad=True
#     )
#     model.model.forward = types.MethodType(
#         lambda *args, **kwargs: (hidden_states,), model.model
#     )

#     # Input ids aren't actually used, but we need to pass them to the model
#     input_ids = torch.randint(0, vocab_size, size=(bs, seqlen), device="cuda")
#     labels = input_ids.detach().clone()
#     attention_mask = torch.ones((bs, seqlen), device="cuda")

#     # Reference loss, dX, dW where dX is the gradients wrt to the hidden states and dW is the gradients wrt to the LM head weight
#     loss, *_ = model(
#         input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
#     )
#     dX, dW = torch.autograd.grad(loss, [hidden_states, model.lm_head.weight])

#     # Patch the model to use fused CEL
#     fused_model = patch_model_fused_cel(
#         model, use_fused_cel=True, fused_cel_n_loop_iters=n_loop_iters
#     )
#     fused_loss, *_ = fused_model(
#         input_ids, labels=labels, attention_mask=attention_mask, return_dict=False
#     )
#     dX_fused, dW_fused = torch.autograd.grad(
#         fused_loss, [hidden_states, fused_model.lm_head.weight]
#     )

#     if dtype == torch.bfloat16:
#         atol, rtol = 1e-3, 1e-3  # Fails if < 1e-3
#     elif dtype == torch.float16:
#         atol, rtol = 1e-4, 1e-4  # Fails if < 1e-4
#     else:
#         atol, rtol = 1e-6, 1e-6

#     test_utils.check_all(
#         [loss, dX, dW],
#         [fused_loss, dX_fused, dW_fused],
#         ["loss", "dX", "dW"],
#         atol=atol,
#         rtol=rtol,
#     )
#     del fused_model
#     empty_cache()
