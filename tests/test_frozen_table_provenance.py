# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An adapter of a model with a frozen lookup table (Qwen3.8-Flash-Next's PLE n-gram table read from the checkpoint
by unsloth_zoo, ram / disk mode) never carries the table, so the save writes frozen_tables.json next to it and
loading the adapter checks the model it lands on has the same base table: silent on the same checkpoint, a warning
(an error with UNSLOTH_FROZEN_TABLE_PROVENANCE=strict) on a checkpoint whose table differs. CPU, tiny model.
(The distributed saves are checked in tests/_frozen_table_parallel.py.)"""

import json
import os
import shutil
import sys
import warnings

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _frozen_table_parallel as ftp  # noqa: E402

pytestmark = pytest.mark.skipif(ftp.available() is not None, reason = str(ftp.available()))


def _base(path, mode = "ram"):
    os.environ["UNSLOTH_FROZEN_TABLE"] = mode
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from transformers import AutoConfig, Qwen4ExpForCausalLM
    from unsloth_zoo.temporary_patches.qwen4_exp import patch_qwen4_exp

    patch_qwen4_exp()
    cfg = AutoConfig.from_pretrained(path)
    cfg._attn_implementation = "eager"
    return Qwen4ExpForCausalLM.from_pretrained(path, config = cfg, dtype = torch.float32)


@pytest.fixture(scope = "module")
def checkpoints(tmp_path_factory):
    same = ftp.make_checkpoint(str(tmp_path_factory.mktemp("ckpt")))
    other = str(tmp_path_factory.mktemp("other"))
    for f in os.listdir(same):
        shutil.copy(os.path.join(same, f), os.path.join(other, f))
    # one byte of the table changed: same files, sizes and headers, different rows
    key, fname = sorted(ftp.table_files(other).items())[0]
    from unsloth_zoo.frozen_table_store import resolve_tensor_locations

    loc = resolve_tensor_locations(os.path.join(other, fname), names = [key])[key]
    with open(loc.path, "r+b") as f:
        f.seek(loc.offset + 5)
        b = f.read(1)
        f.seek(loc.offset + 5)
        f.write(bytes([b[0] ^ 0x5A]))
    return same, other


@pytest.fixture
def adapter(checkpoints, tmp_path):
    from peft import LoraConfig, get_peft_model

    provenance = ftp.load_module("frozen_table_provenance")
    assert provenance.install()
    model = get_peft_model(_base(checkpoints[0]), LoraConfig(r = 2, target_modules = ["o_proj"]))
    model.save_pretrained(str(tmp_path / "adapter"))
    return str(tmp_path / "adapter")


def test_adapter_save_writes_the_table_provenance(checkpoints, adapter):
    with open(os.path.join(adapter, "frozen_tables.json")) as f:
        record = json.load(f)
    (table,) = record["tables"].values()
    assert {t["name"] for t in table["tensors"]} == set(ftp.table_files(checkpoints[0]))
    assert all(len(t["sample_sha256"]) == 64 for t in table["tensors"])
    assert set(table["files"]) == {
        os.path.basename(f) for f in ftp.table_files(checkpoints[0]).values()
    }


def test_reload_on_the_same_tables_is_silent(checkpoints, adapter):
    from peft import PeftModel
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PeftModel.from_pretrained(_base(checkpoints[0], mode = "disk"), adapter)


def test_reload_on_other_tables_warns_or_refuses(checkpoints, adapter, monkeypatch):
    from peft import PeftModel

    with pytest.warns(UserWarning, match = "other frozen lookup tables"):
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "strict")
    with pytest.raises(ValueError, match = "contents differ"):
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter)
    monkeypatch.setenv("UNSLOTH_FROZEN_TABLE_PROVENANCE", "off")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PeftModel.from_pretrained(_base(checkpoints[1]), adapter)


def test_pipeline_stages_merge_their_tables():
    provenance = ftp.load_module("frozen_table_provenance")
    a = {"format": "unsloth_frozen_tables", "version": 1, "tables": {"model.layers.1.t": {"n": 1}}}
    b = {"format": "unsloth_frozen_tables", "version": 1, "tables": {"model.layers.4.t": {"n": 2}}}
    merged = provenance.merge_provenance([a, None, b])
    assert merged["tables"] == {"model.layers.1.t": {"n": 1}, "model.layers.4.t": {"n": 2}}
    assert provenance.merge_provenance([None, None]) is None
