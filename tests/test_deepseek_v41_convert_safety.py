# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Safety of the offline DeepSeek-V4.1 released-checkpoint conversion cache."""

import json
import multiprocessing
import os
import time

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch")

from unsloth.models.deepseek_v41 import convert  # noqa: E402


def _tensors(fill = 1.0):
    return {
        "embed.weight": torch.full((8, 4), fill, dtype = torch.bfloat16),
        "layers.0.attn_norm.weight": torch.full((4,), fill, dtype = torch.bfloat16),
        "head.weight": torch.full((8, 4), fill, dtype = torch.bfloat16),
    }


def _make_release(
    root,
    fill = 1.0,
    sharded = False,
):
    os.makedirs(root, exist_ok = True)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"model_type": "deepseek_v41", "hidden_size": 4}, f)
    tensors = _tensors(fill)
    if sharded:
        a = {k: v for k, v in tensors.items() if k.startswith("layers.")}
        b = {k: v for k, v in tensors.items() if not k.startswith("layers.")}
        safetensors_torch.save_file(a, os.path.join(root, "model-00001.safetensors"))
        safetensors_torch.save_file(b, os.path.join(root, "model-00002.safetensors"))
        weight_map = {k: "model-00001.safetensors" for k in a}
        weight_map.update({k: "model-00002.safetensors" for k in b})
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
    else:
        safetensors_torch.save_file(tensors, os.path.join(root, "model.safetensors"))
    return str(root)


def _read_converted(dst):
    assert os.path.isfile(os.path.join(dst, ".unsloth_converted"))
    with open(os.path.join(dst, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for key, shard in weight_map.items():
        path = os.path.join(dst, shard)
        assert os.path.isfile(path), f"published cache references missing shard {shard}"
        with safetensors_torch.safe_open(path, "pt") as f:
            out[key] = f.get_tensor(key)
    return out


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "converted"
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(root))
    monkeypatch.delenv("UNSLOTH_DSV41_FP4_EXPERTS", raising = False)
    return root


def _concurrent_worker(src, barrier, queue):
    real_save = safetensors_torch.save_file

    def slow_save(*args, **kwargs):
        time.sleep(0.4)
        return real_save(*args, **kwargs)

    safetensors_torch.save_file = slow_save
    try:
        barrier.wait(timeout = 60)
        dst = convert.convert_released_checkpoint(src)
        tensors = _read_converted(dst)
        queue.put(("ok", dst, sorted(tensors)))
    except BaseException as e:  # report every failure to the parent
        queue.put(("err", repr(e), None))


def test_concurrent_first_loads_both_return_a_complete_cache(tmp_path, cache_root):
    src = _make_release(tmp_path / "release")
    ctx = multiprocessing.get_context("fork")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target = _concurrent_worker, args = (src, barrier, queue)) for _ in range(2)]
    for p in procs:
        p.start()
    results = [queue.get(timeout = 120) for _ in procs]
    for p in procs:
        p.join(timeout = 60)
    assert all(r[0] == "ok" for r in results), results
    assert results[0][1] == results[1][1]
    assert (
        results[0][2]
        == results[1][2]
        == sorted(
            ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight", "lm_head.weight"]
        )
    )
    leftovers = [n for n in os.listdir(cache_root) if ".partial" in n]
    assert leftovers == []


def test_preplanted_predictable_staging_symlink_is_not_followed(tmp_path, cache_root):
    src = _make_release(tmp_path / "release")
    dst = convert._target_dir(src)
    os.makedirs(os.path.dirname(dst), exist_ok = True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious")
    os.symlink(str(victim), dst + ".partial")

    out = convert.convert_released_checkpoint(src)

    assert sorted(os.listdir(victim)) == ["keep.txt"]
    assert (victim / "keep.txt").read_text() == "precious"
    assert not os.path.islink(out)
    tensors = _read_converted(out)
    assert torch.equal(tensors["lm_head.weight"], _tensors()["head.weight"])


def test_failed_conversion_removes_only_its_own_staging(tmp_path, cache_root, monkeypatch):
    src = _make_release(tmp_path / "release")
    dst = convert._target_dir(src)
    os.makedirs(os.path.dirname(dst), exist_ok = True)
    foreign = dst + ".partial-someoneelse"
    os.makedirs(foreign)
    open(os.path.join(foreign, "x"), "w").close()

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(safetensors_torch, "save_file", boom)
    with pytest.raises(RuntimeError, match = "disk full"):
        convert.convert_released_checkpoint(src)

    names = sorted(n for n in os.listdir(cache_root) if not n.endswith(".lock"))
    assert names == [os.path.basename(foreign)]
    assert os.listdir(foreign) == ["x"]
    assert not os.path.exists(dst)


@pytest.mark.parametrize(
    "bad",
    [
        "../outside.safetensors",
        "ABSOLUTE",
        "sub/../../outside.safetensors",
        "..\\outside.safetensors",
        "C:outside.safetensors",
        "",
    ],
)
def test_index_shard_names_escaping_the_checkpoint_are_rejected(
    tmp_path, cache_root, monkeypatch, bad
):
    src = _make_release(tmp_path / "release", sharded = True)
    outside = tmp_path / "outside.safetensors"
    safetensors_torch.save_file(_tensors(7.0), str(outside))
    if bad == "ABSOLUTE":
        bad = str(outside)
    index_path = os.path.join(src, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"]["embed.weight"] = bad
    with open(index_path, "w") as f:
        json.dump(index, f)

    import safetensors

    opened = []
    real_open = safetensors.safe_open

    def spy(path, *args, **kwargs):
        opened.append(os.path.realpath(str(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", spy)
    with pytest.raises(ValueError, match = "unsafe shard"):
        convert.convert_released_checkpoint(src)
    assert os.path.realpath(str(outside)) not in opened


def test_safe_open_call_validates_the_shard_name_too(tmp_path, cache_root, monkeypatch):
    src = _make_release(tmp_path / "release", sharded = True)
    outside = tmp_path / "outside.safetensors"
    safetensors_torch.save_file(_tensors(7.0), str(outside))
    real_read = convert._read_index

    def unvalidated(path):
        weight_map = dict(real_read(path))
        weight_map["embed.weight"] = str(outside)
        return weight_map

    import safetensors

    opened = []
    real_open = safetensors.safe_open

    def spy(path, *args, **kwargs):
        opened.append(os.path.realpath(str(path)))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(convert, "_read_index", unvalidated)
    monkeypatch.setattr(safetensors, "safe_open", spy)
    with pytest.raises(ValueError, match = "unsafe shard"):
        convert.convert_released_checkpoint(src)
    assert os.path.realpath(str(outside)) not in opened


def test_symlinked_shards_inside_the_checkpoint_still_load(tmp_path, cache_root):
    # Hub snapshots expose shards as symlinks into ../../blobs: the check is lexical, so they keep working.
    blobs = tmp_path / "blobs"
    real = _make_release(blobs, sharded = True)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    for name in os.listdir(real):
        os.symlink(os.path.join(real, name), snap / name)
    tensors = _read_converted(convert.convert_released_checkpoint(str(snap)))
    assert torch.equal(tensors["model.embed_tokens.weight"], _tensors()["embed.weight"])


def test_same_size_replacement_within_the_same_second_gets_a_new_cache(tmp_path, cache_root):
    src = _make_release(tmp_path / "release")
    weights = os.path.join(src, "model.safetensors")
    second = int(time.time()) - 100
    os.utime(weights, ns = (second * 10**9 + 100, second * 10**9 + 100))
    first = convert.convert_released_checkpoint(src)
    assert torch.equal(_read_converted(first)["lm_head.weight"], _tensors(1.0)["head.weight"])

    staged = os.path.join(str(tmp_path), "new.safetensors")
    safetensors_torch.save_file(_tensors(9.0), staged)
    assert os.path.getsize(staged) == os.path.getsize(weights)
    os.replace(staged, weights)
    os.utime(weights, ns = (second * 10**9 + 200, second * 10**9 + 200))

    second_dst = convert.convert_released_checkpoint(src)
    assert second_dst != first
    assert torch.equal(_read_converted(second_dst)["lm_head.weight"], _tensors(9.0)["head.weight"])


def test_cache_identity_keeps_the_expert_mode_suffix(tmp_path, cache_root, monkeypatch):
    src = _make_release(tmp_path / "release")
    packed = convert._target_dir(src)
    monkeypatch.setenv("UNSLOTH_DSV41_FP4_EXPERTS", "bf16")
    assert convert._target_dir(src) != packed


def test_subfolder_source_converts_from_the_nested_dir(tmp_path, cache_root):
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_release(repo / "ckpt", fill = 3.0)
    dst = convert.convert_released_checkpoint(str(repo), subfolder = "ckpt")
    tensors = _read_converted(dst)
    assert torch.equal(tensors["model.embed_tokens.weight"], _tensors(3.0)["embed.weight"])


def test_subfolder_hub_source_prefixes_allow_patterns(tmp_path, cache_root, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _make_release(snapshot / "ckpt", fill = 5.0)
    calls = []

    def fake_snapshot_download(
        repo_id,
        allow_patterns = None,
        **kwargs,
    ):
        calls.append((repo_id, list(allow_patterns), kwargs))
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    dst = convert.convert_released_checkpoint(
        "org/does-not-exist", subfolder = "ckpt", revision = "main", local_files_only = True
    )
    assert len(calls) == 1
    repo_id, patterns, kwargs = calls[0]
    assert repo_id == "org/does-not-exist"
    assert patterns and all(p.startswith("ckpt/") for p in patterns)
    assert kwargs == {"revision": "main", "local_files_only": True}
    tensors = _read_converted(dst)
    assert torch.equal(tensors["lm_head.weight"], _tensors(5.0)["head.weight"])
