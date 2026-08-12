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


def _reclaim(save_mod, merge, gguf, bases, quant_methods = ("q4_k_m",), **kwargs):
    """Call the helper the way `save_to_gguf` does for a merge it wrote itself.

    `merge_is_disposable` defaults to off in the helper so that a caller who
    points it at a real checkpoint keeps it; every test below that is about the
    reclamation itself has to opt in, exactly like the real call site.
    """
    kwargs.setdefault("merge_is_disposable", True)
    return save_mod._free_merge_if_disk_is_tight(
        merge, gguf, bases, quant_methods = list(quant_methods), **kwargs
    )


def test_a_tight_disk_frees_the_intermediate_merge(tmp_path, monkeypatch, save_mod):
    """The Nemotron case: 60GB of base GGUF, one quant to write, 20GB free."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    freed = _reclaim(save_mod, merge, gguf, bases)
    assert freed > 60 * GB
    assert not [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_a_roomy_disk_keeps_everything(tmp_path, monkeypatch, save_mod):
    """The property that stops this from being a surprise. A user who asked for
    a 16-bit merge and a GGUF on a machine with room for both gets both."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 500)
    assert _reclaim(save_mod, merge, gguf, bases) == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_config_and_tokenizer_survive(tmp_path, monkeypatch, save_mod):
    """Later steps still read them, so deleting the directory outright would
    trade one failure for another."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    _reclaim(save_mod, merge, gguf, bases)
    left = set(os.listdir(merge))
    assert {"config.json", "tokenizer.json"} <= left, left


def test_more_quants_need_more_room(tmp_path, monkeypatch, save_mod):
    """Every output stays on disk, so three of them is three times the room.
    A disk that fits one and not three must still be treated as tight."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 100)
    assert _reclaim(save_mod, merge, gguf, bases) == 0
    assert _reclaim(save_mod, merge, gguf, bases, ["q4_k_m", "q5_k_m", "q8_0"]) > 0


def test_nothing_to_quantize_frees_nothing(tmp_path, monkeypatch, save_mod):
    """`first_conversion` alone means no llama-quantize pass runs at all, so
    there is nothing the merge could be in the way of."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases, []) == 0


def test_a_missing_directory_is_not_an_error(tmp_path, monkeypatch, save_mod):
    """It runs to make an export succeed and must never be what fails it."""
    _with_free(monkeypatch, save_mod, 1)
    assert (
        _reclaim(save_mod, str(tmp_path / "gone"), str(tmp_path), [])
        == 0
    )


def test_an_unreadable_disk_declines_rather_than_deleting(tmp_path, monkeypatch, save_mod):
    """If we cannot tell whether the room is there, the safe answer is to leave
    the user's files alone."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)

    def boom(*_a, **_k):
        raise OSError("no such device")

    monkeypatch.setattr(save_mod.shutil, "disk_usage", boom)
    assert _reclaim(save_mod, merge, gguf, bases) == 0
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


# ---- what reclamation must never do ---------------------------------------


def test_a_reused_checkpoint_is_never_reclaimed(tmp_path, monkeypatch, save_mod):
    """The one that would have been a data-loss bug.

    A non-PEFT `save_pretrained_gguf` does not write an intermediate at all --
    `unsloth_save_pretrained_gguf` points the converter straight at the local
    directory the model was loaded from. On a tight disk, reclaiming there
    deletes the user's own model, and an ordinary export would have eaten its
    own input. Only a merge this export wrote is disposable, so the flag is off
    unless the caller says otherwise.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    freed = save_mod._free_merge_if_disk_is_tight(
        merge, gguf, bases, quant_methods = ["q4_k_m"],
    )
    assert freed == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


class _FakeTokenizer:
    chat_template = None

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok = True)


class _FakeModel:
    """Stand-in for a loaded model: `unsloth_save_pretrained_gguf` reads only
    the config before it decides what `model_directory` is."""

    def __init__(self, name_or_path):
        self.config = types.SimpleNamespace(
            _name_or_path = name_or_path,
            architectures = ["LlamaForCausalLM"],
            model_type = "llama",
        )
        self.saved_to = []

    def save_pretrained(self, path, *args, **kwargs):
        self.saved_to.append(path)


def _run_export(monkeypatch, save_mod, tmp_path, model, **kwargs):
    """Drive `unsloth_save_pretrained_gguf` with the converter stubbed out, and
    report what `save_to_gguf` was handed."""
    seen = {}
    monkeypatch.setattr(save_mod, "_is_vlm", lambda _m: False)
    monkeypatch.setattr(save_mod, "_is_gpt_oss", lambda _m: False)
    monkeypatch.setattr(save_mod, "fix_tokenizer_bos_token", lambda _t: (False, None))
    monkeypatch.setattr(save_mod, "_resolve_imatrix_file", lambda *_a, **_k: None)
    monkeypatch.setattr(save_mod, "dtype_from_config", lambda _c: save_mod.torch.float16)
    monkeypatch.setattr(save_mod, "create_ollama_modelfile", lambda *_a, **_k: None)

    def _save_to_gguf(**kw):
        seen.update(kw)
        out = tmp_path / "export" / "model_gguf" / "model.Q8_0.gguf"
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_bytes(b"GGUF")
        return [str(out)], True, False

    monkeypatch.setattr(save_mod, "save_to_gguf", _save_to_gguf)
    save_mod.unsloth_save_pretrained_gguf(
        model,
        str(tmp_path / "export" / "model"),
        tokenizer = _FakeTokenizer(),
        quantization_method = "q8_0",
        **kwargs,
    )
    return seen


def test_the_non_peft_branch_marks_the_checkpoint_as_not_disposable(
    tmp_path, monkeypatch, save_mod
):
    """The flag is only as good as the branch that clears it: a non-PEFT model
    loaded from a local directory has `save_directory` pointed at that
    directory, so the export must declare it off limits."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = _FakeModel(str(checkpoint))
    seen = _run_export(monkeypatch, save_mod, tmp_path, model)
    assert seen["model_directory"] == str(checkpoint)
    assert seen["merge_is_disposable"] is False


def test_a_merge_this_export_wrote_is_disposable(tmp_path, monkeypatch, save_mod):
    """The other side of the same branch. Nothing local to reuse, so the export
    writes the weights itself and owns them."""
    model = _FakeModel("some-org/some-model")
    seen = _run_export(monkeypatch, save_mod, tmp_path, model)
    assert seen["model_directory"] == str(tmp_path / "export" / "model")
    assert model.saved_to, "the fallback branch should have written the weights"
    assert seen["merge_is_disposable"] is True


def test_a_caller_can_keep_a_directory_it_owns(tmp_path, monkeypatch, save_mod):
    """`save_directory` is not always scratch. The SentenceTransformer export
    hands over its own module directory, so it opts out, and the opt-out must
    not leak into the merge call that does not accept it."""
    generic = {}

    def _generic_save(**kw):
        generic.update(kw)

    monkeypatch.setattr(save_mod, "unsloth_generic_save", _generic_save)
    # isinstance() against a stand-in class is what puts the export on its PEFT
    # branch, which is the one that calls unsloth_generic_save.
    monkeypatch.setattr(save_mod, "PeftModelForCausalLM", _FakeModel)
    model = _FakeModel("some-org/some-model")
    seen = _run_export(
        monkeypatch, save_mod, tmp_path, model, merge_is_disposable = False,
    )
    assert seen["merge_is_disposable"] is False
    assert generic, "the PEFT branch should have merged through unsloth_generic_save"
    assert "merge_is_disposable" not in generic, (
        "the 16-bit merge does not take this argument; forwarding it is a TypeError"
    )


def test_the_sentence_transformer_export_keeps_its_module_directory(monkeypatch, tmp_path):
    """It writes the module in step 1 and uploads the folder in step 7, so
    reclaiming `0_Transformer` would return a folder that no longer loads."""
    st = pytest.importorskip("unsloth.models.sentence_transformer")
    seen = {}

    class _FakeST:
        tokenizer = _FakeTokenizer()

        def save_pretrained(self, path):
            os.makedirs(os.path.join(path, "0_Transformer"), exist_ok = True)

        def __getitem__(self, _index):
            return types.SimpleNamespace(auto_model = object())

    monkeypatch.setattr(
        st, "unsloth_save_pretrained_gguf",
        lambda *a, **kw: (seen.update(kw), {"gguf_files": []})[1],
    )
    st._save_pretrained_gguf(_FakeST(), str(tmp_path / "st"))
    assert seen.get("merge_is_disposable") is False


def test_a_separate_output_filesystem_is_left_alone(tmp_path, monkeypatch, save_mod):
    """`gguf_directory` can put the outputs on another filesystem. Deleting the
    merge then frees bytes the quantize pass cannot use: the export runs out of
    space anyway and the merge is gone too."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    real_stat = os.stat
    merge_real = os.path.realpath(merge)

    def stat_on_another_device(path, *args, **kwargs):
        """Real `stat_result`, one field changed.

        `os.path.isdir` goes through the same `os.stat`, so a stand-in object
        has to keep `st_mode` and the rest intact or the helper never reaches
        the device comparison this test is about.
        """
        st = real_stat(path, *args, **kwargs)
        if os.path.realpath(path) != merge_real:
            return st
        fields = list(st[:10])
        fields[2] = st.st_dev + 1
        return os.stat_result(tuple(fields))

    monkeypatch.setattr(os, "stat", stat_on_another_device)
    assert _reclaim(save_mod, merge, gguf, bases) == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_an_unreadable_merge_directory_declines_rather_than_raising(
    tmp_path, monkeypatch, save_mod
):
    """The helper promises never to raise. A directory listing that fails must
    end the reclamation, not the export -- llama-quantize does not need that
    directory any more and would have succeeded."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)

    def boom(*_a, **_k):
        raise PermissionError("permission denied")

    monkeypatch.setattr(save_mod.os, "listdir", boom)
    assert _reclaim(save_mod, merge, gguf, bases) == 0


def test_f32_output_is_counted_as_twice_the_base(tmp_path, monkeypatch, save_mod):
    """An f32 output off a bf16 base is four bytes a weight against two, so it
    is twice the file it reads. Bounding every pass by the base size would call
    a disk roomy that is about to fill."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 100)
    assert _reclaim(save_mod, merge, gguf, bases, ["q8_0"], first_conversion = "bf16") == 0
    assert _reclaim(save_mod, merge, gguf, bases, ["f32"], first_conversion = "bf16") > 0


def test_f32_from_an_f32_base_is_not_doubled(save_mod):
    """The ratio is target over base, not a special case for the word f32."""
    assert save_mod._gguf_output_size_ratio("f32", "bf16") == 2.0
    assert save_mod._gguf_output_size_ratio("f32", "f32") == 1.0
    assert save_mod._gguf_output_size_ratio("q4_k_m", "bf16") == 1.0


def test_a_roomy_output_disk_is_not_called_full_by_a_tight_cwd(monkeypatch, save_mod):
    """`_gguf_failure_looks_like_disk` now gates the non-Kaggle message too, so
    a wrong answer here hides the llama.cpp advice on every platform. The
    directory that has to hold the file is the one that decides."""
    tight = types.SimpleNamespace(total = 0, used = 0, free = 1 * GB)
    roomy = types.SimpleNamespace(total = 0, used = 0, free = 400 * GB)
    monkeypatch.setattr(
        save_mod.shutil, "disk_usage",
        lambda path: roomy if str(path) == "/output" else tight,
    )
    exc = RuntimeError("llama-quantize: unknown model architecture")
    assert save_mod._gguf_failure_looks_like_disk(exc, "/output") is False
    # No output directory to consult, so the working directory still answers.
    assert save_mod._gguf_failure_looks_like_disk(exc, None) is True
