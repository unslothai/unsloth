# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp32 __init__ tables survive diffusers' from_single_file, as they do from_pretrained."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from core.inference import diffusion_init_buffers as ib  # noqa: E402

QWEN21 = {
    "attention_head_dim": 16,
    "axes_dims_rope": [4, 6, 6],
    "causal_condition": True,
    "context_in_dim": 32,
    "in_channels": 16,
    "mlp_ratio": 2,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 16,
    "patch_size": 1,
}


@pytest.fixture
def qwen21_files(tmp_path):
    cls = getattr(diffusers, "QwenImage21Transformer2DModel", None)
    if cls is None:
        pytest.skip("this diffusers has no QwenImage21Transformer2DModel")
    torch.manual_seed(0)
    model = cls(**QWEN21)
    repo = tmp_path / "repo"
    model.save_pretrained(repo / "transformer")
    single = tmp_path / "qwen21.safetensors"
    safetensors_torch.save_file(
        {
            f"model.diffusion_model.{k}": v.to(torch.bfloat16).contiguous()
            for k, v in model.state_dict().items()
        },
        str(single),
    )
    return cls, repo, single, model.time_text_embed.time_proj.freqs.clone()


def _load(cls, repo, single):
    return cls.from_single_file(
        str(single), config = str(repo), subfolder = "transformer", torch_dtype = torch.bfloat16
    )


def test_single_file_rounds_the_timestep_table_without_the_wrapper(qwen21_files, monkeypatch):
    from diffusers.loaders.single_file_model import FromOriginalModelMixin

    cls, repo, single, _ = qwen21_files
    stock = FromOriginalModelMixin.__dict__["from_single_file"].__func__
    monkeypatch.setattr(
        FromOriginalModelMixin,
        "from_single_file",
        classmethod(getattr(stock, "__wrapped__", stock)),
    )
    assert _load(cls, repo, single).time_text_embed.time_proj.freqs.dtype == torch.bfloat16


def test_single_file_keeps_the_timestep_table_fp32(qwen21_files, monkeypatch):
    from diffusers.loaders.single_file_model import FromOriginalModelMixin

    cls, repo, single, freqs = qwen21_files
    monkeypatch.setattr(
        FromOriginalModelMixin,
        "from_single_file",
        FromOriginalModelMixin.__dict__["from_single_file"],
    )
    assert ib.install_single_file_fp32_buffers() and ib.install_single_file_fp32_buffers()
    model = _load(cls, repo, single)
    table = model.time_text_embed.time_proj.freqs
    assert table.dtype == torch.float32 and torch.equal(table, freqs)
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    ref = cls.from_pretrained(str(repo), subfolder = "transformer", torch_dtype = torch.bfloat16)
    ours = {n: b.dtype for n, b in model.named_buffers()}
    assert ours == {n: b.dtype for n, b in ref.named_buffers()}


def test_only_rounded_fp32_init_tables_are_rebuilt():
    class _Tables(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "table", torch.arange(4, dtype = torch.float32) / 3, persistent = False
            )
            self.register_buffer("wide", torch.arange(4, dtype = torch.float64) / 3, persistent = False)
            self.register_buffer("kept", torch.arange(4, dtype = torch.float32) / 3)
            self.config = {}

        @classmethod
        def from_config(cls, config):
            return cls()

    model = _Tables().to(torch.bfloat16)
    assert ib.restore_fp32_init_buffers(model) == ["table"]
    assert model.table.dtype == torch.float32 and torch.equal(model.table, _Tables().table)
    assert model.wide.dtype == torch.bfloat16 and model.kept.dtype == torch.bfloat16
    assert ib.restore_fp32_init_buffers(model) == []  # nothing rounded left: no rebuild
    assert ib.restore_fp32_init_buffers(torch.nn.Linear(2, 2)) == []  # no from_config: untouched


def test_both_backends_install_the_wrapper_before_loading():
    import ast
    import inspect

    from core.inference import diffusion, video

    for module, cls in ((diffusion, diffusion.DiffusionBackend), (video, video.VideoBackend)):
        body = ast.parse(inspect.getsource(cls)).body[0]
        load = next(n for n in body.body if getattr(n, "name", None) == "load_pipeline")
        assert "install_single_file_fp32_buffers" in ast.unparse(load), module.__name__
