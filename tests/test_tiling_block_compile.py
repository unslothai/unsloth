# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Tiled MLP under per-decoder-block compile (old TiledMLP hit a no_grad view error on torch >= 2.11)."""

import os

import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("spelling", ["env", "kwarg"])
def test_tiled_mlp_trains_under_the_default_block_compile(spelling, monkeypatch, tmp_path, capfd):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "cache"))
    monkeypatch.delenv("UNSLOTH_BLOCK_COMPILE", raising = False)
    monkeypatch.delenv("UNSLOTH_BLOCK_COMPILE_LEGACY", raising = False)
    kwargs = {}
    if spelling == "env":
        monkeypatch.setenv("UNSLOTH_TILED_MLP", "arctic")
    else:
        monkeypatch.delenv("UNSLOTH_TILED_MLP", raising = False)
        kwargs["unsloth_tiled_mlp"] = "rows:16"
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        max_seq_length = 512,
        load_in_4bit = False,
        dtype = torch.bfloat16,
        **kwargs,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        random_state = 0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
    )
    try:
        from unsloth_zoo import tiling_registry
    except ImportError:  # a zoo with the previous TiledMLP: the training loop below is the check
        tiling_registry = None
    if tiling_registry is not None:
        assert tiling_registry.tiled_modules(model)
    block = __import__("unsloth.models._block_compile", fromlist = ["_IN_COMPILED_REGION"])
    assert hasattr(block, "_IN_COMPILED_REGION")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-4)
    ids = torch.randint(10, 500, (1, 256), generator = torch.Generator().manual_seed(1)).cuda()
    for _ in range(2):
        hidden = model.base_model.model.model(input_ids = ids).last_hidden_state
        loss = hidden.float().pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
        opt.step()
        opt.zero_grad()
    out = capfd.readouterr()
    text = out.out + out.err
    assert "compiling each transformer block as one region" in text
    assert "block compile failed" not in text
