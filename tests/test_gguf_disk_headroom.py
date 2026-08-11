# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A GGUF export that holds three copies of the model at once.

`Nemotron-3-Nano-30B-A3B` trained, ran inference, merged to 16-bit, and then
died in llama-quantize at tensor 88 of 401:

    llama_model_quantize: failed to quantize: basic_ios::clear: iostream error

That message is not about the model. It is llama.cpp's own

    fout.exceptions(std::ofstream::failbit); // fail fast on write errors

firing on a write that had nowhere to go. The arithmetic, measured on a 132GB
Colab G4 disk: a 63GB intermediate 16-bit merge, a 60GB BF16 GGUF, and a
Q4_K_M needing about 18GB. 141GB into 132GB.

The first suspicion was the filename -- unsloth passes `initial_files[0]`, and
for this model that was `...BF16-00001-of-00002.gguf`, the first shard of a
split. That is a red herring, and worth writing down so it is not re-derived:
llama.cpp's `llama_model_quantize_impl` walks `ml.weights_map` across every
input shard and, with `keep_split` false, collapses them into a single output.
Handing it shard one is correct usage.

The intermediate merge is what is actually wasted. It is written, converted to
GGUF, and then never read again -- llama-quantize reads the GGUF. So when the
quants will not fit, those bytes are the ones to reclaim.

Two properties matter and are tested here: it frees when the room is not there,
and it does NOT free when the room is there. The second is the one that keeps
this from becoming a surprise for anyone who wanted the merge kept.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

GB = 1024**3


@pytest.fixture
def save_mod():
    import unsloth.save as save
    return save


def _layout(tmp_path, merge_gb, base_gb):
    merge = tmp_path / "model"
    gguf = tmp_path / "model_gguf"
    merge.mkdir()
    gguf.mkdir()
    # Sparse files: the sizes are what the code reads, and writing 60GB of
    # zeroes to prove a point about disk space would be its own joke.
    for i, gb in enumerate(_split(merge_gb)):
        with open(merge / f"model-0000{i + 1}.safetensors", "wb") as fh:
            fh.truncate(int(gb * GB))
    (merge / "config.json").write_text("{}")
    (merge / "tokenizer.json").write_text("{}")
    bases = []
    for i, gb in enumerate(_split(base_gb)):
        path = gguf / f"model.BF16-0000{i + 1}-of-00002.gguf"
        with open(path, "wb") as fh:
            fh.truncate(int(gb * GB))
        bases.append(str(path))
    return str(merge), str(gguf), bases


def _split(total_gb, per = 30):
    while total_gb > per:
        yield per
        total_gb -= per
    yield total_gb


def _with_free(monkeypatch, save_mod, free_gb):
    import shutil

    usage = types.SimpleNamespace(total = 0, used = 0, free = int(free_gb * GB))
    monkeypatch.setattr(save_mod.shutil, "disk_usage", lambda *_a, **_k: usage)
    return usage


def test_a_tight_disk_frees_the_intermediate_merge(tmp_path, monkeypatch, save_mod):
    """The Nemotron case: 60GB of base GGUF, one quant to write, 20GB free."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    freed = save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 1)
    assert freed > 60 * GB
    assert not [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_a_roomy_disk_keeps_everything(tmp_path, monkeypatch, save_mod):
    """The property that stops this from being a surprise. A user who asked for
    a 16-bit merge and a GGUF on a machine with room for both gets both."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 500)
    assert save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 1) == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_config_and_tokenizer_survive(tmp_path, monkeypatch, save_mod):
    """Later steps still read them, so deleting the directory outright would
    trade one failure for another."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 1)
    left = set(os.listdir(merge))
    assert {"config.json", "tokenizer.json"} <= left, left


def test_more_quants_need_more_room(tmp_path, monkeypatch, save_mod):
    """Passes can run concurrently, so three outputs is three times the room.
    A disk that fits one and not three must still be treated as tight."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 100)
    assert save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 1) == 0
    assert save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 3) > 0


def test_nothing_to_quantize_frees_nothing(tmp_path, monkeypatch, save_mod):
    """`first_conversion` alone means no llama-quantize pass runs at all, so
    there is nothing the merge could be in the way of."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 1)
    assert save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 0) == 0


def test_a_missing_directory_is_not_an_error(tmp_path, monkeypatch, save_mod):
    """It runs to make an export succeed and must never be what fails it."""
    _with_free(monkeypatch, save_mod, 1)
    assert (
        save_mod._free_merge_if_disk_is_tight(str(tmp_path / "gone"), str(tmp_path), [], n_quants = 1)
        == 0
    )


def test_an_unreadable_disk_declines_rather_than_deleting(tmp_path, monkeypatch, save_mod):
    """If we cannot tell whether the room is there, the safe answer is to leave
    the user's files alone."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)

    def boom(*_a, **_k):
        raise OSError("no such device")

    monkeypatch.setattr(save_mod.shutil, "disk_usage", boom)
    assert save_mod._free_merge_if_disk_is_tight(merge, gguf, bases, n_quants = 1) == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def _save_to_gguf_source(save_mod):
    import inspect
    return inspect.getsource(save_mod.save_to_gguf)


def test_the_disk_message_is_not_gated_on_kaggle(save_mod):
    """The Nemotron run was on Colab and was told to rebuild llama.cpp, which
    fixes nothing and costs a long compile. Asserted against the source because
    the alternative is driving a 30B export to reproduce one string."""
    source = _save_to_gguf_source(save_mod)
    disk_branch = source.index("elif _gguf_failure_looks_like_disk")
    build_advice = source.index("make clean && make all -j")
    assert (
        disk_branch < build_advice
    ), "the disk explanation must be reached before the rebuild advice"
    assert "not a problem " in source and "with llama.cpp" in source


def test_the_helper_is_called_before_quantizing(save_mod):
    """A helper nothing calls is the same as no fix. Ordering matters too: it
    has to run after the base GGUF exists and before llama-quantize starts."""
    source = _save_to_gguf_source(save_mod)
    assert source.index("_free_merge_if_disk_is_tight(") < source.index("def _quantize_one")
