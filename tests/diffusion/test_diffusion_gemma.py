# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import os
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "DiffusionGemmaForBlockDiffusion"):
    pytest.skip("transformers without DiffusionGemma", allow_module_level = True)

from unsloth.models.diffusion_gemma_objective import DIFFUSION_GEMMA_PROFILE
from unsloth.models.diffusion_profiles import resolve_diffusion_profile

TINY = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"
TRL_EXAMPLE = os.environ.get("UNSLOTH_TRL_DIFFUSION_GEMMA_EXAMPLE")


@pytest.fixture(scope = "module")
def model():
    m = transformers.DiffusionGemmaForBlockDiffusion.from_pretrained(TINY, dtype = torch.float32)
    m._unsloth_diffusion_eos_token_id = 1
    return m.train()


def _batch(
    vocab,
    canvas,
    max_length = 80,
    rows = None,
):
    g = torch.Generator().manual_seed(0)
    if rows is None:
        rows = [(5, 10), (7, canvas * 2 + 3), (4, max_length - 4)]
    input_ids = torch.zeros(len(rows), max_length, dtype = torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, (p, r) in enumerate(rows):
        n = p + r
        input_ids[i, :n] = torch.randint(10, vocab, (n,), generator = g)
        attention_mask[i, :n] = 1
        labels[i, p:n] = input_ids[i, p:n]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _args(**kw):
    base = dict(
        max_length = 80,
        diffusion_eps = None,
        diffusion_self_conditioning_p = None,
        diffusion_encoder_ar_weight = None,
        diffusion_prediction_type = None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_profile_resolves(model):
    assert resolve_diffusion_profile(model.config) is DIFFUSION_GEMMA_PROFILE


def test_canvas_reaches_the_decoder(model):
    # The notebook passed canvas_ids, which forward() silently swallows; decoder_input_ids must change the logits.
    cfg = model.config
    ids = torch.randint(10, 1000, (1, 8))
    a = torch.randint(10, 1000, (1, cfg.canvas_length))
    b = torch.randint(10, 1000, (1, cfg.canvas_length))
    with torch.no_grad():
        la = model(input_ids = ids, decoder_input_ids = a).logits
        lb = model(input_ids = ids, decoder_input_ids = b).logits
    assert not torch.allclose(la, lb)


def test_loss_finite_and_metrics(model):
    batch = _batch(model.config.text_config.vocab_size, model.config.canvas_length)
    torch.manual_seed(0)
    loss, _, metrics = DIFFUSION_GEMMA_PROFILE.compute_loss(model, batch, _args())
    assert torch.isfinite(loss)
    assert set(metrics) == {"diffusion_loss", "encoder_ar_loss", "mean_t"}


@pytest.mark.skipif(
    TRL_EXAMPLE is None, reason = "set UNSLOTH_TRL_DIFFUSION_GEMMA_EXAMPLE to TRL's example file"
)
@pytest.mark.parametrize("prediction_type", ["mean", "mean_loo"])
def test_parity_with_trl_reference(model, prediction_type):
    source = open(TRL_EXAMPLE, encoding = "utf-8").read()
    source = source.replace(
        "from trl.trainer.utils import maybe_gather_lm_head_ctx",
        "import contextlib\nmaybe_gather_lm_head_ctx = lambda *a: contextlib.nullcontext()",
    )
    namespace = {"__name__": "trl_example"}
    exec(compile(source, TRL_EXAMPLE, "exec"), namespace)
    reference = namespace["DiffusionGemmaSFTTrainer"]
    cfg = model.config
    fake = types.SimpleNamespace(
        args = types.SimpleNamespace(max_length = 80),
        canvas_length = cfg.canvas_length,
        vocab_size = cfg.text_config.vocab_size,
        final_logit_softcapping = cfg.text_config.final_logit_softcapping,
        eos_token_id = 1,
        model_prediction_type = prediction_type,
        model = model,
        eps = reference.eps,
        self_conditioning_p = reference.self_conditioning_p,
    )
    batch = _batch(cfg.text_config.vocab_size, cfg.canvas_length)
    params = [p for p in model.parameters() if p.requires_grad]

    def run(fn):
        model.zero_grad(set_to_none = True)
        torch.manual_seed(1234)
        loss = fn()
        loss.backward()
        return loss.detach(), [None if p.grad is None else p.grad.clone() for p in params]

    ref_loss, ref_grads = run(lambda: reference.compute_loss(fake, model, batch))
    our_loss, our_grads = run(
        lambda: DIFFUSION_GEMMA_PROFILE.compute_loss(
            model, batch, _args(diffusion_prediction_type = prediction_type)
        )[0]
    )
    torch.testing.assert_close(our_loss, ref_loss, rtol = 0, atol = 1e-6)
    for a, b in zip(our_grads, ref_grads):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-7)


def test_tied_encoder_decoder_lora_merge_is_refused():
    from peft import LoraConfig, get_peft_model

    from unsloth.models.diffusion import _refuse_tied_lora_merge

    m = transformers.DiffusionGemmaForBlockDiffusion.from_pretrained(TINY, dtype = torch.float32)
    enc = m.model.encoder.language_model.layers[0].self_attn.q_proj
    dec = m.model.decoder.layers[0].self_attn.q_proj
    assert enc is not dec and enc.weight.data_ptr() == dec.weight.data_ptr()
    m = get_peft_model(
        m, LoraConfig(r = 4, target_modules = DIFFUSION_GEMMA_PROFILE.lora_target_modules)
    )
    m = _refuse_tied_lora_merge(m)
    with pytest.raises(RuntimeError, match = "share base weights"):
        m.merge_and_unload()


def test_untied_lora_merge_still_allowed():
    from peft import LoraConfig, get_peft_model

    from unsloth.models.diffusion import _refuse_tied_lora_merge

    m = transformers.AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM"
    )
    m = _refuse_tied_lora_merge(
        get_peft_model(m, LoraConfig(r = 4, target_modules = ["q_proj", "v_proj"]))
    )
    assert "merge_and_unload" not in vars(m)
    m.merge_and_unload()


class _DistributedLike(torch.nn.Module):
    """DDP / FSDP expose neither config nor heads; only forward reaches the module."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, **kwargs):
        return self.module(**kwargs)


def test_loss_through_distributed_wrapper(model):
    batch = _batch(model.config.text_config.vocab_size, model.config.canvas_length)
    torch.manual_seed(0)
    plain = DIFFUSION_GEMMA_PROFILE.compute_loss(model, batch, _args())[0]
    torch.manual_seed(0)
    wrapped = DIFFUSION_GEMMA_PROFILE.compute_loss(_DistributedLike(model), batch, _args())[0]
    torch.testing.assert_close(wrapped, plain)


def test_row_without_supervised_tokens_adds_no_canvas_loss(model):
    batch = _batch(model.config.text_config.vocab_size, model.config.canvas_length, rows = [(6, 0)])
    torch.manual_seed(0)
    _, _, metrics = DIFFUSION_GEMMA_PROFILE.compute_loss(model, batch, _args())
    assert metrics["diffusion_loss"] == 0.0


def test_reloaded_adapter_still_refuses_merge(tmp_path):
    from peft import LoraConfig, get_peft_model

    from unsloth import FastModel

    m = transformers.DiffusionGemmaForBlockDiffusion.from_pretrained(TINY, dtype = torch.float32)
    m = get_peft_model(
        m, LoraConfig(r = 4, target_modules = DIFFUSION_GEMMA_PROFILE.lora_target_modules)
    )
    m.save_pretrained(tmp_path)
    reloaded, _ = FastModel.from_pretrained(str(tmp_path), dtype = torch.float32, device_map = "cpu")
    with pytest.raises(RuntimeError, match = "share base weights"):
        reloaded.merge_and_unload()


def test_reload_uses_the_adapters_saved_tokenizer(tmp_path):
    from peft import LoraConfig, get_peft_model

    from unsloth import FastModel

    m = transformers.DiffusionGemmaForBlockDiffusion.from_pretrained(TINY, dtype = torch.float32)
    m = get_peft_model(
        m, LoraConfig(r = 4, target_modules = DIFFUSION_GEMMA_PROFILE.lora_target_modules)
    )
    m.save_pretrained(tmp_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(TINY)
    tokenizer.chat_template = "{{ 'adapter-template-marker' }}"
    tokenizer.save_pretrained(tmp_path)
    _, reloaded = FastModel.from_pretrained(str(tmp_path), dtype = torch.float32, device_map = "cpu")
    reloaded = getattr(reloaded, "tokenizer", reloaded)
    assert "adapter-template-marker" in (reloaded.chat_template or "")
