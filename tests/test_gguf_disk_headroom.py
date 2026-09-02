# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A GGUF export that holds three copies of the model at once.

`Nemotron-3-Nano-30B-A3B` merged to 16-bit and then died in llama-quantize at
tensor 88 of 401 with `failed to quantize: basic_ios::clear: iostream error`.
That message is not about the model: it is llama.cpp's own
`fout.exceptions(std::ofstream::failbit)` firing on a write with nowhere to go.
The arithmetic, measured on a 132GB Colab G4 disk: a 63GB intermediate 16-bit
merge, a 60GB BF16 GGUF, and a Q4_K_M needing about 18GB. 141GB into 132GB.

The filename was a red herring, worth writing down so it is not re-derived:
unsloth passes `initial_files[0]`, here `...BF16-00001-of-00002.gguf`, but
llama.cpp's `llama_model_quantize_impl` walks `ml.weights_map` across every
input shard and, with `keep_split` false, collapses them into a single output.
Handing it shard one is correct usage.

The intermediate merge is what is actually wasted: written, converted, then
never read again, since llama-quantize reads the GGUF. Two properties are
tested here -- it frees when the room is not there, and it does NOT free when
the room is there. The second is what keeps this from surprising anyone who
wanted the merge kept.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

GB = 1024**3

# A weight shard of the merge, as opposed to anything else that ends in `.safetensors` (an adapter) or `.bin`
# (`training_args.bin`).
_MERGE_SHARD = re.compile(r"^(model|pytorch_model|consolidated)\.|-\d+-of-\d+\.")


def _is_merge_shard(name):
    return name.endswith((".safetensors", ".bin")) and bool(_MERGE_SHARD.search(name))


@pytest.fixture
def save_mod():
    import unsloth.save as save
    return save


def _layout(tmp_path, merge_gb, base_gb):
    merge = tmp_path / "model"
    gguf = tmp_path / "model_gguf"
    merge.mkdir()
    gguf.mkdir()
    # Sparse files: only the sizes are read, and writing 60GB of zeroes to prove a point about disk space would be its
    # own joke.
    # The names are the `-NNNNN-of-NNNNN` set `save_pretrained` really writes, which is what the reclamation matches on;
    # `model-00001.safetensors` is not a shape it produces.
    merge_shards = list(_split(merge_gb))
    for i, gb in enumerate(merge_shards):
        name = f"model-{i + 1:05d}-of-{len(merge_shards):05d}.safetensors"
        with open(merge / name, "wb") as fh:
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


def _reclaim(
    save_mod,
    merge,
    gguf,
    bases,
    quant_methods = ("q4_k_m",),
    **kwargs,
):
    """Call the helper the way `save_to_gguf` does for a merge it wrote itself.

    The helper defaults `merge_is_disposable` off so a caller pointing it at a
    real checkpoint keeps it, so every test about the reclamation itself has to
    opt in, exactly like the real call site. `preexisting_weights` defaults to
    empty for the same reason the layouts do: the ordinary merge writes into a
    directory of its own, so everything in it is the export's. The tests about a
    reused directory pass their own.
    """
    kwargs.setdefault("merge_is_disposable", True)
    kwargs.setdefault("preexisting_weights", frozenset())
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
    """Every output stays on disk, so a disk that fits one quant and not three is
    still tight. Off a 60GB BF16 base one Q4_K_M is about 21GB and
    Q4_K_M + Q5_K_M + Q8_0 about 81GB, so 50GB free fits the first, not the three."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 50)
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
    assert _reclaim(save_mod, str(tmp_path / "gone"), str(tmp_path), []) == 0


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


def test_a_complete_stale_shard_set_is_not_inherited(tmp_path, monkeypatch, save_mod):
    """A finished earlier save in the same directory, index and every shard.

    This is the case self-consistency cannot decide. An index left by a previous
    sharded save lists `-00001-of-00002` and `-00002-of-00002` under one stem and
    both are on disk, so it looks exactly like an index the current merge wrote.
    transformers removes neither: its stale sweep does not match
    `model.safetensors.index`, and it only prunes shards under the stem it is
    writing. What separates them is not their shape but who wrote them.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    stale = ["archive-00001-of-00002.safetensors", "archive-00002-of-00002.safetensors"]
    for name in stale:
        with open(os.path.join(merge, name), "wb") as fh:
            fh.truncate(GB)
    index = {"weight_map": {f"layer.{i}": name for i, name in enumerate(stale)}}
    with open(os.path.join(merge, "model.safetensors.index.json"), "w", encoding = "utf-8") as fh:
        json.dump(index, fh)
    preexisting = frozenset(stale + ["model.safetensors.index.json"])
    _with_free(monkeypatch, save_mod, 20)
    _reclaim(save_mod, merge, gguf, bases, preexisting_weights = preexisting)
    for name in stale:
        assert os.path.isfile(
            os.path.join(merge, name)
        ), f"{name} belonged to an earlier save and was deleted"


def test_a_consolidated_file_the_merge_did_not_write_is_kept(tmp_path, monkeypatch, save_mod):
    """`consolidated.safetensors` is a name the merge uses and may not have written.

    The shard selection drops it whenever ordinary shards coexist, so a caller
    reusing an output directory can hold one this run never touched. The name
    matcher cannot tell the difference, which is the whole reason ownership is
    recorded rather than inferred.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    with open(os.path.join(merge, "consolidated.safetensors"), "wb") as fh:
        fh.truncate(GB)
    _with_free(monkeypatch, save_mod, 20)
    _reclaim(
        save_mod,
        merge,
        gguf,
        bases,
        preexisting_weights = frozenset(["consolidated.safetensors"]),
    )
    assert os.path.isfile(
        os.path.join(merge, "consolidated.safetensors")
    ), "a consolidated checkpoint this export never wrote was deleted"


def test_unknown_provenance_reclaims_nothing(tmp_path, monkeypatch, save_mod):
    """No answer is not the same as an empty answer.

    A caller that could not read the directory before the merge cannot say what
    it owns, and the deletion is permanent, so the reclamation declines.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    freed = _reclaim(save_mod, merge, gguf, bases, preexisting_weights = None)
    assert freed == 0
    assert [f for f in os.listdir(merge) if f.endswith(".safetensors")]


def test_a_small_output_with_modest_free_space_is_not_called_a_full_disk(save_mod):
    """The rebuild advice is only wrong when the disk is the problem.

    A fixed 2GB floor is a claim about the machine, not about the write. An
    incompatible quantizer failing on a sub-gigabyte model with 1.5GB free has
    all the room it needs, and calling that a full disk suppresses the version
    advice that would actually have fixed it.
    """
    import types as _types

    free = _types.SimpleNamespace(total = 0, used = 0, free = int(1.5 * GB))
    original = save_mod.shutil.disk_usage
    save_mod.shutil.disk_usage = lambda *_a, **_k: free
    try:
        failure = RuntimeError("unknown model architecture")
        # 400MB of output: 1.5GB free is ample, so this is not a disk problem.
        assert not save_mod._gguf_failure_looks_like_disk(failure, ".", needed_bytes = int(0.4 * GB))
        # 4GB of output into 1.5GB free is.
        assert save_mod._gguf_failure_looks_like_disk(failure, ".", needed_bytes = 4 * GB)
        # A caller that cannot say what it needed still gets the fixed floor.
        assert save_mod._gguf_failure_looks_like_disk(failure, ".")
    finally:
        save_mod.shutil.disk_usage = original


def test_an_explicit_enospc_is_a_full_disk_whatever_the_size(save_mod):
    """The message and errno signals are independent of the arithmetic, so a
    real ENOSPC is still a full disk even when the output would have fit."""
    failure = OSError(28, "No space left on device")
    assert save_mod._gguf_failure_looks_like_disk(failure, ".", needed_bytes = 1)


def test_the_bytes_already_written_are_not_charged_twice(tmp_path, save_mod):
    """A failed pass leaves its partial output behind, and those bytes are gone
    from the free space this measures while `needed_bytes` still describes the
    whole file.

    llama-quantize streams into the output (`llama-quant.cpp` opens the
    `ofstream` before the tensor loop and writes each tensor as it finishes
    one), so an export that starts with 12GB free, writes 5GB of a 10GB output
    and then dies on an unsupported tensor is measured at 7GB against 10GB and
    called a full disk. It never was: the room was there, and the rebuild advice
    that would have addressed the real failure is the thing suppressed. Credit
    the partial file back before comparing.
    """
    import types as _types

    output = tmp_path / "model.Q4_K_M.gguf"
    with open(output, "wb") as f:
        f.truncate(5 * GB)

    original = save_mod.shutil.disk_usage
    try:
        failure = RuntimeError("unknown model architecture")

        save_mod.shutil.disk_usage = lambda *_a, **_k: _types.SimpleNamespace(
            total = 0, used = 0, free = 7 * GB
        )
        assert not save_mod._gguf_failure_looks_like_disk(
            failure,
            str(tmp_path),
            needed_bytes = 10 * GB,
            partial_output = str(output),
        )

        # A disk that really is short is still short: 1GB left plus the 5GB written is 6GB, and the output wanted 10GB.
        save_mod.shutil.disk_usage = lambda *_a, **_k: _types.SimpleNamespace(
            total = 0, used = 0, free = 1 * GB
        )
        assert save_mod._gguf_failure_looks_like_disk(
            failure,
            str(tmp_path),
            needed_bytes = 10 * GB,
            partial_output = str(output),
        )

        # A pass that wrote nothing at all is unchanged, and so is a caller that names an output which is not there.
        assert save_mod._gguf_failure_looks_like_disk(
            failure,
            str(tmp_path),
            needed_bytes = 10 * GB,
            partial_output = str(tmp_path / "never-written.gguf"),
        )
    finally:
        save_mod.shutil.disk_usage = original


def test_a_reused_checkpoint_is_never_reclaimed(tmp_path, monkeypatch, save_mod):
    """The one that would have been a data-loss bug.

    A non-PEFT `save_pretrained_gguf` writes no intermediate at all: it points
    the converter straight at the local directory the model was loaded from, so
    reclaiming there on a tight disk would eat the user's own model. Only a merge
    this export wrote is disposable, so the flag is off unless the caller says so.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 20)
    freed = save_mod._free_merge_if_disk_is_tight(
        merge,
        gguf,
        bases,
        quant_methods = ["q4_k_m"],
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
    # isinstance() against a stand-in class is what puts the export on its PEFT branch, which is the one that calls
    # unsloth_generic_save.
    monkeypatch.setattr(save_mod, "PeftModelForCausalLM", _FakeModel)
    model = _FakeModel("some-org/some-model")
    seen = _run_export(
        monkeypatch,
        save_mod,
        tmp_path,
        model,
        merge_is_disposable = False,
    )
    assert seen["merge_is_disposable"] is False
    assert generic, "the PEFT branch should have merged through unsloth_generic_save"
    assert (
        "merge_is_disposable" not in generic
    ), "the 16-bit merge does not take this argument; forwarding it is a TypeError"


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
        st,
        "unsloth_save_pretrained_gguf",
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
        """Real `stat_result`, one field changed. `os.path.isdir` goes through the
        same `os.stat`, so `st_mode` and the rest have to stay intact or the helper
        never reaches the device comparison this test is about."""
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


def test_a_roomy_output_disk_is_not_called_full_by_a_tight_cwd(monkeypatch, save_mod):
    """`_gguf_failure_looks_like_disk` now gates the non-Kaggle message too, so
    a wrong answer here hides the llama.cpp advice on every platform. The
    directory that has to hold the file is the one that decides."""
    tight = types.SimpleNamespace(total = 0, used = 0, free = 1 * GB)
    roomy = types.SimpleNamespace(total = 0, used = 0, free = 400 * GB)
    monkeypatch.setattr(
        save_mod.shutil,
        "disk_usage",
        lambda path: roomy if str(path) == "/output" else tight,
    )
    exc = RuntimeError("llama-quantize: unknown model architecture")
    assert save_mod._gguf_failure_looks_like_disk(exc, "/output") is False
    # No output directory to consult, so the working directory still answers.
    assert save_mod._gguf_failure_looks_like_disk(exc, None) is True


def test_unrelated_training_artifacts_are_never_deleted(tmp_path, monkeypatch, save_mod):
    """A merge routinely lands in a training `output_dir`, so deleting by
    extension takes `training_args.bin` and the optimizer state with it. Those
    are not the merge and this export cannot put them back."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    keep = {
        "training_args.bin": b"args",
        "optimizer.pt": b"opt",
        "rng_state.pth": b"rng",
        "scheduler.pt": b"sched",
        # An adapter beside the merge is the user's, and it is tiny anyway.
        "adapter_model.safetensors": b"adapter",
    }
    for name, blob in keep.items():
        (Path(merge) / name).write_bytes(blob)
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    left = set(os.listdir(merge))
    for name in keep:
        assert name in left, f"{name} was deleted but the merge did not write it"
    assert not [f for f in left if _is_merge_shard(f)]


def test_the_shards_the_index_names_are_the_ones_reclaimed(tmp_path, monkeypatch, save_mod):
    """`save_pretrained` writes an index naming its shards, so when there is one
    it decides rather than the naming convention.

    The fixture is a whole shard set under a stem the convention misses, because
    that is the only shape an index is ever written in: a save shards or it does
    not, and an unsharded one writes no index at all.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 30, base_gb = 60)
    odd = [Path(merge) / f"weights-part-0000{i}-of-00002.safetensors" for i in (1, 2)]
    for shard in odd:
        with open(shard, "wb") as fh:
            fh.truncate(int(10 * GB))
    (Path(merge) / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": s.name for i, s in enumerate(odd)}})
    )
    _with_free(monkeypatch, save_mod, 1)
    freed = _reclaim(save_mod, merge, gguf, bases)
    assert freed > 20 * GB
    for shard in odd:
        assert not shard.exists(), "the index named this shard and it was left behind"


def test_a_stale_safetensors_index_does_not_widen_the_deletion(tmp_path, monkeypatch, save_mod):
    """The hazard the index read creates, and the reason it is validated.

    transformers writes an index only when a save shards, and its stale-shard
    sweep never removes one. So an unsharded merge lands in a directory that
    still holds an earlier save's index, and reading that index hands whatever it
    names -- including files under a stem this helper otherwise protects -- to a
    permanent delete.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    theirs = Path(merge) / "users_own-00001-of-00002.safetensors"
    theirs.write_bytes(b"an earlier save under a stem the convention protects")
    (Path(merge) / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                    "c": theirs.name,
                }
            }
        )
    )
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    assert theirs.exists(), "a stale index widened the deletion onto a foreign stem"
    assert not [f for f in os.listdir(merge) if f.startswith("model-")]


def test_an_index_missing_shards_it_names_is_not_trusted(tmp_path, monkeypatch, save_mod):
    """`of-00003` promises three shards. Two means this is not that set, so the
    naming convention decides and the odd stem is left alone."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    theirs = Path(merge) / "leftover-00001-of-00003.safetensors"
    theirs.write_bytes(b"one shard of a set that is no longer whole")
    (Path(merge) / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": theirs.name,
                    "b": "leftover-00002-of-00003.safetensors",
                }
            }
        )
    )
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    assert theirs.exists(), "a partial shard set was read as the merge"


def test_a_shard_set_under_another_stem_is_not_the_merge(tmp_path, monkeypatch, save_mod):
    """`-NNNNN-of-NNNNN` under an unrelated stem is not something `save_pretrained`
    writes, and transformers does not clear it either: its stale-shard sweep wants
    the `model` / `pytorch_model` stem as well as the shard shape. So the user put
    it here, and this helper deletes permanently."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    theirs = Path(merge) / "backup-00001-of-00002.safetensors"
    theirs.write_bytes(b"not ours")
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    assert theirs.exists(), "a foreign shard set was deleted with the merge"
    assert not [f for f in os.listdir(merge) if f.startswith("model-")]


def test_a_checkpoint_in_the_other_serialization_is_not_the_merge(tmp_path, monkeypatch, save_mod):
    """A disposable merge is always safetensors: the PEFT branch goes through
    unsloth_zoo's safetensors rewrite and the non-PEFT fallback calls
    `save_pretrained` with no arguments. So a `pytorch_model.bin` beside it came
    from an earlier save, which transformers' stale sweep leaves alone too."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    theirs = Path(merge) / "pytorch_model.bin"
    theirs.write_bytes(b"an earlier save the export cannot put back")
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    assert theirs.exists(), "a checkpoint in the other serialization was deleted with the merge"
    assert not [f for f in os.listdir(merge) if f.startswith("model-")]


def test_a_stale_sharded_bin_checkpoint_is_left_alone(tmp_path, monkeypatch, save_mod):
    """The sharded form of the same thing. Its index is not this merge's index,
    and reading it would hand the shards it names to the deletion."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    theirs = [Path(merge) / f"pytorch_model-0000{i}-of-00002.bin" for i in (1, 2)]
    for shard in theirs:
        shard.write_bytes(b"an earlier sharded save")
    (Path(merge) / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": shard.name for i, shard in enumerate(theirs)}})
    )
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB
    for shard in theirs:
        assert shard.exists(), f"{shard.name} was deleted but the merge did not write it"
    assert not [f for f in os.listdir(merge) if f.startswith("model-")]


def test_a_malformed_index_falls_back_to_the_naming_convention(tmp_path, monkeypatch, save_mod):
    """A half-written index must not stop the reclamation, nor make it raise."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    (Path(merge) / "model.safetensors.index.json").write_text("{not json")
    _with_free(monkeypatch, save_mod, 1)
    assert _reclaim(save_mod, merge, gguf, bases) > 60 * GB


def test_a_quantized_output_is_priced_below_the_base_it_reads(tmp_path, monkeypatch, save_mod):
    """The merge is kept whenever the export already fits, which is the whole
    promise. A Q4_K_M off a 60GB BF16 base needs about 21GB, so 30GB free has
    the room and charging it a full 60GB copy would delete for nothing."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    _with_free(monkeypatch, save_mod, 30)
    assert _reclaim(save_mod, merge, gguf, bases) == 0
    assert [f for f in os.listdir(merge) if _is_merge_shard(f)]


def test_each_quant_is_priced_by_its_own_width(save_mod):
    """Wider types cost more room, and none of them is charged the base."""
    ratio = lambda m: save_mod._gguf_output_size_ratio(m, "bf16")
    assert ratio("q2_k") < ratio("q4_k_m") < ratio("q6_k") < ratio("q8_0") < 1.0
    # The published llama.cpp 7B sizes: Q4_K_M near 4.5 bits a weight, Q8_0 8.5.
    assert 4.5 / 16 < ratio("q4_k_m") < 8.0 / 16
    assert 8.5 / 16 < ratio("q8_0") < 1.0
    # i-quants carry their width the same way.
    assert ratio("iq2_xxs") < ratio("iq4_xs") < ratio("q8_0")
    # An unrecognised method is still charged a whole copy of the base.
    assert ratio("something_new") == 1.0


def test_a_q8_0_base_is_not_priced_as_sixteen_bit(save_mod):
    """`first_conversion` is a public argument and `q8_0` is one of the types
    convert_hf_to_gguf can emit directly, so the base GGUF is not always 16-bit.
    Pricing an 8-bit base as 16-bit halves every estimate taken off it, and
    under-counting is the direction that costs the export."""
    off_16 = save_mod._gguf_output_size_ratio("q4_k_m", "bf16")
    off_8 = save_mod._gguf_output_size_ratio("q4_k_m", "q8_0")
    assert off_8 > off_16, "a quant off an 8-bit base is a bigger share of it"
    # Still an upper bound: Q4_K_M is about 4.5 bits against Q8_0's real 8.5.
    assert off_8 > 4.5 / 8.5
    # And an f32 output off that base is four bytes a weight against one.
    assert save_mod._gguf_output_size_ratio("f32", "q8_0") >= 32.0 / 8.5


def test_the_vlm_projector_is_not_charged_to_every_quant(tmp_path, monkeypatch, save_mod):
    """llama-quantize copies the `-mmproj` projector rather than quantizing it,
    so it is not part of what a pass writes. Charging it per quant calls a disk
    tight that has the room."""
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 30)
    mmproj = Path(gguf) / "model.BF16-mmproj.gguf"
    with open(mmproj, "wb") as fh:
        fh.truncate(int(200 * GB))
    _with_free(monkeypatch, save_mod, 20)
    assert _reclaim(save_mod, merge, gguf, bases + [str(mmproj)]) == 0
    assert [f for f in os.listdir(merge) if _is_merge_shard(f)]


def test_the_diagnosis_is_not_priced_off_the_reclamation_bound(save_mod):
    """The two callers of the ratio are hurt by opposite errors.

    Reclamation rounds up on purpose -- an estimate that comes out low keeps a
    merge the quants then have no room for. Reusing that same number to decide
    whether a *failure* was about disk inverts the cost: Q4_K_M is charged 5.5
    bits a weight against a real 4.5, so a 60GB BF16 base is called 20.6GB when
    the output is about 17GB, and an unrelated quantizer failure with 19GB free
    is reported as a full disk while the rebuild advice that would have fixed it
    is swallowed. Diagnosis therefore prices each type at its nominal width,
    which no k-quant is ever under.
    """
    upper = save_mod._gguf_output_size_ratio("q4_k_m", "bf16")
    lower = save_mod._gguf_output_size_ratio("q4_k_m", "bf16", upper_bound = False)
    # llama.cpp's own 7B table puts Q4_K_M near 4.8 bits a weight; the two bounds have to sit either side of it.
    assert lower < 4.5 / 16 < upper

    # Codex's case, run through the helper the way the call site does.
    base_bytes = 60 * GB
    failure = RuntimeError("unknown model architecture")
    free = types.SimpleNamespace(total = 0, used = 0, free = int(19 * GB))
    original = save_mod.shutil.disk_usage
    save_mod.shutil.disk_usage = lambda *_a, **_k: free
    try:
        assert not save_mod._gguf_failure_looks_like_disk(
            failure, ".", needed_bytes = int(base_bytes * lower)
        ), "19GB free for a ~17GB output is not a full disk"
        # A disk that genuinely cannot hold even the nominal output still is one.
        short = types.SimpleNamespace(total = 0, used = 0, free = int(8 * GB))
        save_mod.shutil.disk_usage = lambda *_a, **_k: short
        assert save_mod._gguf_failure_looks_like_disk(
            failure, ".", needed_bytes = int(base_bytes * lower)
        )
    finally:
        save_mod.shutil.disk_usage = original

    # Full-precision outtypes carry no block overhead, so both bounds agree.
    for dtype in ("f16", "bf16", "f32"):
        assert save_mod._gguf_output_size_ratio(dtype, "bf16") == (
            save_mod._gguf_output_size_ratio(dtype, "bf16", upper_bound = False)
        )
    # A width the diagnosis cannot measure is not guessed at: None puts the caller back on the fixed floor rather than
    # charging a whole base copy.
    assert save_mod._gguf_output_size_ratio("something_new", "bf16") == 1.0
    assert save_mod._gguf_output_size_ratio("something_new", "bf16", upper_bound = False) is None

    # And the call site actually asks for the lower bound.
    assert "upper_bound = False" in _save_to_gguf_source(save_mod)


def test_the_index_is_reclaimed_with_the_shards_it_named(tmp_path, monkeypatch, save_mod):
    """An index this export wrote goes with the shards it named.

    Reading the index is the one way a shard set under a stem
    `_MERGE_WEIGHT_NAME` does not know still gets reclaimed. Deleting those
    shards and leaving the index behind breaks that on the second export into
    the same directory: the provenance snapshot now sees the leftover index and
    classifies it as the caller's, so it is filtered out before the reading, the
    non-canonical shards are invisible to the name matcher, and a tight-disk
    rerun reclaims nothing. It also leaves an index pointing at files that no
    longer exist.
    """
    merge, gguf, bases = _layout(tmp_path, merge_gb = 63, base_gb = 60)
    for name in os.listdir(merge):
        if _is_merge_shard(name):
            os.remove(os.path.join(merge, name))

    shards = ["archive-00001-of-00002.safetensors", "archive-00002-of-00002.safetensors"]

    def _write_merge():
        for name in shards:
            with open(os.path.join(merge, name), "wb") as fh:
                fh.truncate(int(30 * GB))
        index = {"weight_map": {f"layer.{i}": name for i, name in enumerate(shards)}}
        with open(os.path.join(merge, "model.safetensors.index.json"), "w", encoding = "utf-8") as fh:
            json.dump(index, fh)

    _write_merge()
    _with_free(monkeypatch, save_mod, 20)
    assert _reclaim(save_mod, merge, gguf, bases) > 0
    for name in shards:
        assert not os.path.exists(os.path.join(merge, name))
    assert not os.path.exists(
        os.path.join(merge, "model.safetensors.index.json")
    ), "the index named the shards that were just deleted and cannot outlive them"

    # Second export into the same directory, provenance snapshotted the way `unsloth_save_pretrained_gguf` does, before
    # the merge writes.
    preexisting = frozenset(os.listdir(merge))
    # First export: the directory is this export's own, so everything goes.
    _write_merge()
    freed = _reclaim(save_mod, merge, gguf, bases, preexisting_weights = preexisting)
    assert freed > 0, "the second export reclaimed nothing"
    for name in shards:
        assert not os.path.exists(os.path.join(merge, name))
