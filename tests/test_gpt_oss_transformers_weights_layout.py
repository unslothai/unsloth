# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`config.transformers_weights` wins over standard filenames, as in transformers' `_get_resolved_checkpoint_files`."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import unsloth  # noqa: F401  (patches before transformers)
from unsloth.models import loader_utils

BNB_QC = {"quant_method": "bitsandbytes", "load_in_4bit": True}
P = "model.layers.0.mlp"
SPLIT = [f"{P}.experts.gate_up_projs.0.weight", f"{P}.experts.down_projs.0.weight"]
FUSED = [f"{P}.experts.gate_up_proj", f"{P}.experts.down_proj", f"{P}.router.weight"]


def _save(path, names):
    save_file({n: torch.zeros(2, 2, dtype = torch.bfloat16) for n in names}, str(path))


def _cfg(transformers_weights = None):
    cfg = SimpleNamespace(quantization_config = BNB_QC)
    if transformers_weights is not None:
        cfg.transformers_weights = transformers_weights
    return cfg


def _repo(
    tmp_path,
    standard,
    chosen,
    chosen_name = "chosen.safetensors",
):
    d = tmp_path / "ckpt"
    d.mkdir()
    _save(d / "model.safetensors", standard)
    _save(d / chosen_name, chosen)
    return str(d)


@pytest.mark.parametrize(
    "standard, chosen, expected",
    [(SPLIT, FUSED, True), (FUSED, SPLIT, False)],
    ids = ["split_standard_fused_chosen", "fused_standard_split_chosen"],
)
def test_explicit_file_decides_the_layout(tmp_path, standard, chosen, expected):
    path = _repo(tmp_path, standard, chosen)
    probe = loader_utils.gpt_oss_checkpoint_has_fused_experts
    assert probe(_cfg("chosen.safetensors"), path) is expected
    assert probe(_cfg(), path) is (not expected)


def test_explicit_index_decides_the_layout(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    _save(d / "model.safetensors", SPLIT)
    _save(d / "chosen-00001.safetensors", FUSED)
    (d / "chosen.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: "chosen-00001.safetensors" for n in FUSED}})
    )
    cfg = _cfg("chosen.safetensors.index.json")
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, str(d)) is True


def test_explicit_file_ignores_variant_and_use_safetensors(tmp_path):
    path = _repo(tmp_path, SPLIT, FUSED)
    _save(tmp_path / "ckpt" / "model.fp16.safetensors", SPLIT)
    dtypes = loader_utils._checkpoint_tensor_dtypes(
        path, variant = "fp16", use_safetensors = False, transformers_weights = "chosen.safetensors"
    )
    assert loader_utils._gpt_oss_bnb_checkpoint_expert_layout(dtypes) == "fused"


def test_invalid_or_escaping_explicit_names_never_read_other_files(tmp_path):
    path = _repo(tmp_path, SPLIT, FUSED)
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(_cfg("chosen.bin"), path) is False
    _save(tmp_path / "outside.safetensors", FUSED)
    dtypes = loader_utils._checkpoint_tensor_dtypes(
        path, transformers_weights = "../outside.safetensors"
    )
    assert dtypes is None


def test_hub_probe_reads_the_explicit_file(monkeypatch, tmp_path):
    import huggingface_hub

    _save(tmp_path / "model.safetensors", SPLIT)
    _save(tmp_path / "chosen.safetensors", FUSED)
    asked = []

    def download(repo, filename, **kwargs):
        asked.append(filename)
        return str(tmp_path / filename)

    def no_metadata(*args, **kwargs):
        raise AssertionError("standard-name metadata must not decide an explicit file")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", no_metadata)
    monkeypatch.setattr(loader_utils, "_hub_live_commit", lambda *a, **k: None)
    cfg = _cfg("chosen.safetensors")
    assert loader_utils.gpt_oss_checkpoint_has_fused_experts(cfg, "org/repo") is True
    assert asked == ["chosen.safetensors"]
