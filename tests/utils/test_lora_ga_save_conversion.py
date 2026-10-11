# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import json
import os

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.models._utils import snapshot_residual_lora_init

peft = pytest.importorskip("peft")
if not hasattr(peft, "preprocess_loraga"):
    pytest.skip("PEFT without LoRA-GA", allow_module_level = True)

from peft import LoraConfig, LoraGAConfig, PeftModel, get_peft_model, preprocess_loraga
from transformers import AutoModelForCausalLM

MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
R = 4


def _base():
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype = torch.float32)


def _lora_ga_model(
    seed,
    extra_targets = (),
    **config_kwargs,
):
    torch.manual_seed(seed)
    model = _base()
    config = LoraConfig(
        r = R,
        lora_alpha = 8,
        lora_dropout = 0.0,
        target_modules = ["q_proj", "v_proj", "down_proj", *extra_targets],
        init_lora_weights = "lora_ga",
        lora_ga_config = LoraGAConfig(),
        **config_kwargs,
    )
    ids = torch.randint(
        0, model.config.vocab_size, (2, 12), generator = torch.Generator().manual_seed(0)
    )

    def train_step():
        model(input_ids = ids, labels = ids).loss.backward()

    preprocess_loraga(model, config, train_step)
    peft_model = get_peft_model(model, config)
    snapshot_residual_lora_init(peft_model, "lora_ga")
    return peft_model


def _train(model, seed = 1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.add_(0.05 * torch.randn(p.shape, generator = g))


def _logits(model):
    ids = torch.arange(1, 17).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model(input_ids = ids).logits


def _lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


@pytest.fixture(scope = "module")
def trained():
    model = _lora_ga_model(seed = 3407)
    _train(model)
    return model, _logits(model)


def test_saved_adapter_reloads_on_original_base(trained, tmp_path):
    model, reference = trained
    model.save_pretrained(tmp_path)
    config = json.load(open(tmp_path / "adapter_config.json"))
    assert (
        config["r"] == 2 * R and config["lora_alpha"] == 16 and config["init_lora_weights"] is True
    )
    reloaded = PeftModel.from_pretrained(_base(), tmp_path)
    torch.testing.assert_close(_logits(reloaded), reference, atol = 1e-4, rtol = 1e-4)


def test_embedding_targets_keep_their_rank(tmp_path):
    # PEFT gives the embedding a default init; doubling only the global r would break its reload.
    model = _lora_ga_model(seed = 3407, extra_targets = ["embed_tokens"])
    _train(model)
    reference = _logits(model)
    model.save_pretrained(tmp_path)
    config = json.load(open(tmp_path / "adapter_config.json"))
    assert config["rank_pattern"] == {r"model\.embed_tokens": R}
    reloaded = PeftModel.from_pretrained(_base(), tmp_path)
    torch.testing.assert_close(_logits(reloaded), reference, atol = 1e-4, rtol = 1e-4)


def test_user_pattern_does_not_shadow_the_embedding_pin(tmp_path):
    # "embed_tokens" sorts before the pin and would resolve the embedding to the doubled rank.
    model = _lora_ga_model(
        seed = 3407, extra_targets = ["embed_tokens"], rank_pattern = {"embed_tokens": R}
    )
    _train(model)
    reference = _logits(model)
    model.save_pretrained(tmp_path)
    reloaded = PeftModel.from_pretrained(_base(), tmp_path)
    torch.testing.assert_close(_logits(reloaded), reference, atol = 1e-4, rtol = 1e-4)


def test_empty_adapter_selection_saves_nothing(trained, tmp_path):
    model, _ = trained
    model.save_pretrained(tmp_path, selected_adapters = [])
    assert not (tmp_path / "adapter_config.json").exists()


def test_native_save_is_the_bug(trained, tmp_path):
    model, reference = trained
    model.save_pretrained.__wrapped__(tmp_path)
    reloaded = PeftModel.from_pretrained(_base(), tmp_path)
    assert (
        _logits(reloaded) - reference
    ).abs().max() > 1e-3  # 30x the tolerance the converted save meets


def test_trainer_state_dict_save_is_converted(trained, tmp_path):
    model, reference = trained
    model.save_pretrained(tmp_path, state_dict = model.state_dict(), safe_serialization = True)
    reloaded = PeftModel.from_pretrained(_base(), tmp_path)
    torch.testing.assert_close(_logits(reloaded), reference, atol = 1e-4, rtol = 1e-4)


@pytest.mark.parametrize("as_path", [False, True])
def test_resume_into_live_model_restores_rank_r(trained, tmp_path, as_path):
    model, reference = trained
    model.save_pretrained(tmp_path)
    saved = _lora_state(model)
    _train(model, seed = 2)
    model.load_adapter(tmp_path if as_path else str(tmp_path), "default", is_trainable = True)
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert p.shape == saved[n].shape and p.requires_grad
            torch.testing.assert_close(p.detach(), saved[n], atol = 0, rtol = 0)
    torch.testing.assert_close(_logits(model), reference, atol = 1e-5, rtol = 1e-5)


def test_resume_in_fresh_run_rebases_onto_saved_init(trained, tmp_path):
    model, reference = trained
    model.save_pretrained(tmp_path)
    fresh = _lora_ga_model(
        seed = 7
    )  # svd_lowrank draws differ: another B0 @ A0 subtracted from the base
    model_init = next(m for m in model.modules() if hasattr(m, "_unsloth_initial_lora"))
    fresh_init = next(m for m in fresh.modules() if hasattr(m, "_unsloth_initial_lora"))
    assert not torch.equal(
        model_init._unsloth_initial_lora["default"][0],
        fresh_init._unsloth_initial_lora["default"][0],
    )
    fresh.load_adapter(str(tmp_path), "default", is_trainable = True)
    torch.testing.assert_close(_logits(fresh), reference, atol = 1e-4, rtol = 1e-4)


def test_native_checkpoint_still_loads(trained, tmp_path):
    model, reference = trained
    model.save_pretrained.__wrapped__(tmp_path)  # what earlier Unsloth versions wrote
    _train(model, seed = 3)
    model.load_adapter(str(tmp_path), "default", is_trainable = True)
    torch.testing.assert_close(_logits(model), reference, atol = 1e-5, rtol = 1e-5)


def test_wrapping_is_idempotent(tmp_path):
    import types

    model = _lora_ga_model(seed = 3407)
    inner = model.save_pretrained
    # patch_saving_functions rebinds save_pretrained around the wrapper with types.MethodType.
    model.save_pretrained = types.MethodType(lambda self, *a, **k: inner(*a, **k), model)
    outer = model.save_pretrained
    snapshot_residual_lora_init(model, "lora_ga")
    assert model.save_pretrained is outer
    model.save_pretrained(tmp_path)
    assert json.load(open(tmp_path / "adapter_config.json"))["r"] == 2 * R


def test_trainer_checkpoints_resume_and_reload(tmp_path):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    g = torch.Generator().manual_seed(1)
    rows = [torch.randint(0, 1000, (16,), generator = g).tolist() for _ in range(16)]
    data = Dataset.from_list([{"input_ids": r, "labels": r} for r in rows])

    def train(model, resume = None):
        args = TrainingArguments(
            str(tmp_path),
            max_steps = 4,
            save_steps = 2,
            per_device_train_batch_size = 2,
            learning_rate = 1e-2,
            use_cpu = True,
            report_to = [],
            seed = 3407,
        )
        Trainer(model = model, args = args, train_dataset = data).train(resume_from_checkpoint = resume)
        return _logits(model)

    reference = train(_lora_ga_model(seed = 3407))
    plain = PeftModel.from_pretrained(_base(), tmp_path / "checkpoint-4")
    torch.testing.assert_close(_logits(plain), reference, atol = 1e-4, rtol = 1e-4)
    # A fresh run draws another LoRA-GA init, then resumes steps 3-4 from the converted checkpoint.
    resumed = train(_lora_ga_model(seed = 7), resume = str(tmp_path / "checkpoint-2"))
    torch.testing.assert_close(resumed, reference, atol = 1e-3, rtol = 1e-3)
