# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local DiT inside a ComfyUI tree takes its text encoder from the tree's ``text_encoders/`` folder instead of
downloading the base repo's (unslothai/unsloth#8950). CPU only, tiny Qwen3 encoders in the ComfyUI layout."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")

import core.inference.diffusion_comfy_components as C  # noqa: E402
from hub.utils.comfy_models import comfy_text_encoder_files  # noqa: E402

INDEX = {"text_encoder": ["transformers", "Qwen3Model"], "vae": ["diffusers", "AutoencoderKL"]}


def _cfg(hidden = 64):
    return transformers.Qwen3Config(
        vocab_size = 64,
        hidden_size = hidden,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 32,
        tie_word_embeddings = False,
    )


def _encoder_file(
    path: Path,
    hidden = 64,
    int8 = False,
    convrot = False,
) -> str:
    torch.manual_seed(0)
    state = transformers.Qwen3Model(_cfg(hidden)).state_dict()
    out = {}
    for key, value in state.items():
        name = "model." + key
        if int8 and ".mlp." in key and key.endswith(".weight"):
            stem = name[: -len(".weight")]
            out[name] = torch.zeros(value.shape, dtype = torch.int8)
            out[stem + ".weight_scale"] = torch.ones(value.shape[0])
            conf = {"format": "int8_tensorwise"}
            if convrot:
                conf.update(convrot = True, convrot_groupsize = 16)
            blob = json.dumps(conf).encode()
            out[stem + ".comfy_quant"] = torch.tensor(list(blob), dtype = torch.uint8)
        else:
            out[name] = value.to(torch.bfloat16)
    path.parent.mkdir(parents = True, exist_ok = True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in out.items()}, str(path))
    return str(path)


def _vae_file(path: Path) -> str:
    path.parent.mkdir(parents = True, exist_ok = True)
    safetensors_torch.save_file({"decoder.up.0.block.0.conv1.weight": torch.zeros(4, 4)}, str(path))
    return str(path)


def _base(tmp_path: Path, hidden = 64) -> str:
    base = tmp_path / "base"
    _cfg(hidden).save_pretrained(base / "text_encoder")
    (base / "model_index.json").write_text(json.dumps(INDEX), encoding = "utf-8")
    return str(base)


def _fits(base: str, quant = None):
    return lambda o: C.verify_override_fit(
        o, base = base, local_files_only = True, text_encoder_quant = quant
    )


def _comfy_tree(tmp_path: Path) -> Path:
    models = tmp_path / "ComfyUI" / "models"
    for name in ("diffusion_models", "text_encoders", "vae", "loras", "checkpoints"):
        (models / name).mkdir(parents = True, exist_ok = True)
    (models / "diffusion_models" / "dit.safetensors").write_bytes(b"x")
    return models


def test_text_encoder_folders_of_the_dits_own_layout(tmp_path):
    models = _comfy_tree(tmp_path)
    te = _encoder_file(models / "text_encoders" / "qwen_3_4b.safetensors")
    legacy = _encoder_file(models / "clip" / "sub" / "t5.safetensors")
    (models / "text_encoders" / "notes.txt").write_text("x")
    found = comfy_text_encoder_files(models / "diffusion_models" / "dit.safetensors")
    assert [str(p) for p in found] == [te, legacy]
    # A file beside the text encoders, or in a plain folder, is no ComfyUI DiT pick.
    assert comfy_text_encoder_files(models / "text_encoders" / "qwen_3_4b.safetensors") == ()
    (tmp_path / "plain").mkdir()
    assert comfy_text_encoder_files(tmp_path / "plain" / "dit.safetensors") == ()


def test_the_one_fitting_encoder_is_picked(tmp_path):
    base = _base(tmp_path)
    good = _encoder_file(tmp_path / "te" / "qwen_3_4b.safetensors")
    wrong_size = _encoder_file(tmp_path / "te" / "qwen_3_8b.safetensors", hidden = 128)
    vae = _vae_file(tmp_path / "te" / "ae.safetensors")
    junk = tmp_path / "te" / "broken.safetensors"
    junk.write_bytes(b"not a safetensors file")
    picked = C.pick_comfy_text_encoders(
        [wrong_size, vae, str(junk), good], model_index = INDEX, fits = _fits(base)
    )
    assert picked == [good]


def test_dense_beats_int8_and_a_tie_falls_back_to_the_base_repo(tmp_path):
    base = _base(tmp_path)
    dense = _encoder_file(tmp_path / "te" / "qwen_3_4b.safetensors")
    int8 = _encoder_file(tmp_path / "te" / "qwen_3_4b_int8.safetensors", int8 = True)
    assert C.pick_comfy_text_encoders([int8, dense], model_index = INDEX, fits = _fits(base)) == [dense]
    twin = _encoder_file(tmp_path / "te" / "qwen_3_4b_finetune.safetensors")
    assert C.pick_comfy_text_encoders([dense, twin], model_index = INDEX, fits = _fits(base)) == []


def test_auto_scheme_family_never_takes_a_dense_file(tmp_path):
    base = _base(tmp_path)
    dense = _encoder_file(tmp_path / "te" / "qwen3vl_8b_bf16.safetensors")
    int8 = _encoder_file(tmp_path / "te" / "qwen3vl_8b_int8.safetensors", int8 = True)
    kwargs = dict(model_index = INDEX, fits = _fits(base), auto_scheme = "int8")
    assert C.pick_comfy_text_encoders([dense], **kwargs) == []
    assert C.pick_comfy_text_encoders([dense, int8], **kwargs) == [int8]


def test_an_explicit_precision_the_file_cannot_take_skips_it(tmp_path):
    base = _base(tmp_path)
    int8 = _encoder_file(tmp_path / "te" / "qwen3_int8.safetensors", int8 = True, convrot = True)
    assert (
        C.pick_comfy_text_encoders([int8], model_index = INDEX, fits = _fits(base, quant = "fp8")) == []
    )


def _backend_pick(
    tmp_path,
    monkeypatch,
    *,
    env = None,
    kind = "single_file",
):
    from core.inference import diffusion

    models = _comfy_tree(tmp_path)
    te = _encoder_file(models / "text_encoders" / "qwen_3_4b.safetensors")
    base = _base(tmp_path)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_COMFY_TEXT_ENCODERS", env)
    fam = types.SimpleNamespace(
        name = "z-image", single_file_is_pipeline = False, pipeline_only = False, te_quant_auto = None
    )
    picked = diffusion.DiffusionBackend._comfy_text_encoders(
        diffusion.DiffusionBackend.__new__(diffusion.DiffusionBackend),
        fam,
        kind,
        str(models / "diffusion_models"),
        "dit.safetensors",
        base,
        None,
        local_files_only = True,
    )
    return picked, te


def test_backend_takes_the_comfy_encoder_for_a_local_dit(tmp_path, monkeypatch):
    picked, te = _backend_pick(tmp_path, monkeypatch)
    assert picked == [str(Path(te))]


@pytest.mark.parametrize("env, kind", [("0", "single_file"), (None, "pipeline")])
def test_backend_opt_out_and_pipeline_loads_keep_the_base_repo(tmp_path, monkeypatch, env, kind):
    picked, _ = _backend_pick(tmp_path, monkeypatch, env = env, kind = kind)
    assert picked == []
