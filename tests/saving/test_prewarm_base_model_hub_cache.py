# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pre-warm the hub cache: hf_hub_download(local_dir) never fills it, so later exports re-download."""

from __future__ import annotations

import ast
import json
import os
import types
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent.parent / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")


def _extract_function(name: str) -> str:
    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SOURCE, node)
    raise AssertionError(f"{name} not found in unsloth/save.py")


class _FakePeftModel:
    def __init__(self, name_or_path = "unsloth/gemma-4-31b-it-bnb-4bit"):
        self.config = types.SimpleNamespace(_name_or_path = name_or_path)


class _Recorder:
    """Callable that records calls and returns/raises per configuration."""

    def __init__(
        self,
        result = None,
        exc = None,
        results_fn = None,
    ):
        self.calls = []
        self.result = result
        self.exc = exc
        self.results_fn = results_fn

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        if self.results_fn is not None:
            return self.results_fn(*args, **kwargs)
        return self.result


def _build_env(
    monkeypatch,
    tmp_path,
    shards = None,
    cached = False,
    free_bytes = 10**15,
    base_source = None,
    kaggle = False,
    colab = False,
    hub_cache = None,
    live_hub_cache = "__same__",
    fp8_sibling = None,
    sibling_source = None,
    index_weight_map = None,
):
    """Exec the extracted helper with stubbed collaborators; returns (fn, stubs)."""
    shards = (
        shards
        if shards is not None
        else [
            ("model-00001-of-00002.safetensors", 30 * 1024**3),
            ("model-00002-of-00002.safetensors", 29 * 1024**3),
        ]
    )
    if base_source is None:
        base_source = ("unsloth/gemma-4-31b-it", False, None, False, None)

    class _FS:
        def __init__(self, token = None):
            pass

        def ls(
            self,
            repo,
            detail = True,
        ):
            return [{"name": f"{repo}/{n}", "size": s} for n, s in shards]

    class _LocalMiss(Exception):
        pass

    hf_hub_download = _Recorder(result = str(tmp_path / "cached"))
    if not cached:
        hf_hub_download.exc = _LocalMiss("not cached")
    # Serve the index (the merge's shard filter) while shard probes miss, so no network is needed.
    if index_weight_map is not None:
        _idx_path = tmp_path / "model.safetensors.index.json"
        _idx_path.write_text(json.dumps({"weight_map": index_weight_map}))

        def _hub_dl(
            repo_id = None,
            filename = None,
            **kw,
        ):
            if filename == "model.safetensors.index.json":
                return str(_idx_path)
            raise _LocalMiss("not cached")

        hf_hub_download.exc = None
        hf_hub_download.results_fn = _hub_dl
    snapshot_download = _Recorder()
    determine_base_model_source = _Recorder(result = base_source)
    # FP8 -> 16bit sibling swap: re-resolving the sibling returns its 16bit source.
    if fp8_sibling is not None:
        _sib_src = sibling_source or (fp8_sibling, False, None, False, None)
        determine_base_model_source.results_fn = (
            lambda name, token = None: _sib_src if name == fp8_sibling else base_source
        )
    resolve_fp8_16bit_sibling = _Recorder(result = fp8_sibling)

    cache_dir = tmp_path / "hub_cache"
    cache_dir.mkdir(exist_ok = True)

    hf_module = types.SimpleNamespace(
        HfFileSystem = _FS,
        hf_hub_download = hf_hub_download,
        snapshot_download = snapshot_download,
        constants = types.SimpleNamespace(
            HF_HUB_CACHE = hub_cache if hub_cache is not None else str(cache_dir)
        ),
    )
    zoo_module = types.SimpleNamespace(
        determine_base_model_source = determine_base_model_source,
        _resolve_fp8_16bit_sibling = resolve_fp8_16bit_sibling,
    )
    _live = (
        (hub_cache if hub_cache is not None else str(cache_dir))
        if live_hub_cache == "__same__"
        else live_hub_cache
    )
    hf_cache_module = types.SimpleNamespace(_active_caches = lambda: (None, _live, None))
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hf_module)
    monkeypatch.setitem(__import__("sys").modules, "unsloth_zoo.saving_utils", zoo_module)
    monkeypatch.setitem(__import__("sys").modules, "unsloth_zoo.hf_cache", hf_cache_module)

    fake_shutil = types.SimpleNamespace(
        disk_usage = lambda path: types.SimpleNamespace(free = free_bytes)
    )

    prints = []
    namespace = {
        "os": os,
        "shutil": fake_shutil,
        "PeftModel": _FakePeftModel,
        "get_model_name": lambda name, load_in_4bit: name.removesuffix("-bnb-4bit"),
        "IS_KAGGLE_ENVIRONMENT": kaggle,
        "IS_COLAB_ENVIRONMENT": colab,
        "print": lambda *a, **k: prints.append(" ".join(str(x) for x in a)),
    }
    exec(
        compile(_extract_function("_prewarm_base_model_hub_cache"), str(_SAVE_PY), "exec"),
        namespace,
    )
    stubs = types.SimpleNamespace(
        snapshot_download = snapshot_download,
        hf_hub_download = hf_hub_download,
        determine_base_model_source = determine_base_model_source,
        resolve_fp8_16bit_sibling = resolve_fp8_16bit_sibling,
        prints = prints,
    )
    return namespace["_prewarm_base_model_hub_cache"], stubs


def test_downloads_base_into_hub_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_PREWARM_HUB_CACHE", raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(), save_method = "merged_16bit", token = "tok")
    assert len(stubs.snapshot_download.calls) == 1
    _, kwargs = stubs.snapshot_download.calls[0]
    assert kwargs["repo_id"] == "unsloth/gemma-4-31b-it"
    # No local_dir: the point is populating the persistent cache.
    assert "local_dir" not in kwargs
    assert "model-00001-of-00002.safetensors" in kwargs["allow_patterns"]
    assert "model.safetensors.index.json" in kwargs["allow_patterns"]


def test_skips_download_when_already_cached(monkeypatch, tmp_path):
    fn, stubs = _build_env(monkeypatch, tmp_path, cached = True)
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []
    assert all(kwargs.get("local_files_only") for _, kwargs in stubs.hf_hub_download.calls)


def test_skips_when_disk_too_small_for_cache_copy(monkeypatch, tmp_path):
    fn, stubs = _build_env(monkeypatch, tmp_path, free_bytes = 60 * 1024**3)
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


@pytest.mark.parametrize("env_value", ["0", "false", "NO", "off"])
def test_env_opt_out(monkeypatch, tmp_path, env_value):
    monkeypatch.setenv("UNSLOTH_PREWARM_HUB_CACHE", env_value)
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


@pytest.mark.parametrize("var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_offline_skips(monkeypatch, tmp_path, var):
    monkeypatch.setenv(var, "1")
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


def test_kaggle_and_colab_skip(monkeypatch, tmp_path):
    for flag in ("kaggle", "colab"):
        fn, stubs = _build_env(monkeypatch, tmp_path, **{flag: True})
        fn(_FakePeftModel(), save_method = "merged_16bit")
        assert stubs.snapshot_download.calls == [], f"{flag} must skip pre-warm"


def test_modelscope_skips(monkeypatch, tmp_path):
    # The merge fetches the base from ModelScope (#3726), so a Hub copy would go unused.
    monkeypatch.setenv("UNSLOTH_USE_MODELSCOPE", "1")
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []
    assert stubs.determine_base_model_source.calls == []


@pytest.mark.parametrize("save_method", ["merged_4bit", "forced_merged_4bit", "lora"])
def test_non_downloading_save_methods_skip(monkeypatch, tmp_path, save_method):
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(), save_method = save_method)
    assert stubs.snapshot_download.calls == []


def test_local_base_model_skips(monkeypatch, tmp_path):
    local_dir = tmp_path / "local_base"
    local_dir.mkdir()
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(_FakePeftModel(name_or_path = str(local_dir)), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


def test_quantized_base_skips(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        base_source = ("unsloth/gemma-4-31b-it-bnb-4bit", False, None, True, "nf4"),
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


def test_non_peft_model_skips(monkeypatch, tmp_path):
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(object(), save_method = "merged_16bit")
    assert stubs.determine_base_model_source.calls == []
    assert stubs.snapshot_download.calls == []


def test_consolidated_shard_excluded_when_proper_shards_exist(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        shards = [
            ("consolidated.safetensors", 14 * 1024**3),
            ("model-00001-of-00001.safetensors", 14 * 1024**3),
        ],
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    _, kwargs = stubs.snapshot_download.calls[0]
    assert "consolidated.safetensors" not in kwargs["allow_patterns"]
    assert "model-00001-of-00001.safetensors" in kwargs["allow_patterns"]


def test_consolidated_only_repo_is_kept(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        shards = [("consolidated.safetensors", 14 * 1024**3)],
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    _, kwargs = stubs.snapshot_download.calls[0]
    assert "consolidated.safetensors" in kwargs["allow_patterns"]


def test_listing_failure_is_swallowed(monkeypatch, tmp_path):
    fn, stubs = _build_env(monkeypatch, tmp_path)
    stubs.determine_base_model_source.exc = RuntimeError("HF is down")
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls == []


def test_gpt_oss_bf16_mxfp4_swap_skips(monkeypatch, tmp_path):
    fn, stubs = _build_env(monkeypatch, tmp_path)
    fn(
        _FakePeftModel(name_or_path = "unsloth/gpt-oss-20b-BF16"),
        save_method = "mxfp4",
    )
    assert stubs.snapshot_download.calls == []


def test_missing_config_skips_cleanly(monkeypatch, tmp_path):
    # A None config must skip silently, not print a misleading "Could not pre-cache" warning.
    fn, stubs = _build_env(monkeypatch, tmp_path)
    model = _FakePeftModel()
    model.config = None
    fn(model, save_method = "merged_16bit")
    assert stubs.determine_base_model_source.calls == []
    assert stubs.snapshot_download.calls == []
    assert not any(
        "Could not pre-cache" in p for p in stubs.prints
    ), "missing config took the error path instead of a clean skip"


def test_relative_hub_cache_does_not_falsely_skip(monkeypatch, tmp_path):
    # Without abspath a relative, missing HF_HUB_CACHE walks up to "" and pre-warm is skipped.
    monkeypatch.chdir(tmp_path)
    fn, stubs = _build_env(monkeypatch, tmp_path, hub_cache = "relcache/hub")
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert len(stubs.snapshot_download.calls) == 1, "relative cache path falsely skipped pre-warm"


def test_generic_save_calls_prewarm_before_merge():
    """unsloth_generic_save must pre-warm the cache before merge_and_overwrite_lora."""
    tree = ast.parse(_SOURCE)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "unsloth_generic_save"
    )
    body_src = ast.get_source_segment(_SOURCE, fn)
    prewarm_pos = body_src.find("_prewarm_base_model_hub_cache(")
    merge_pos = body_src.find("merge_and_overwrite_lora(")
    assert prewarm_pos != -1, "unsloth_generic_save no longer pre-warms the hub cache"
    assert merge_pos != -1
    assert prewarm_pos < merge_pos, "pre-warm must run before the merge downloads shards"


def test_prewarm_downloads_into_live_env_cache(monkeypatch, tmp_path):
    # Download must target the live-env cache (what the merge reads).
    fn, stubs = _build_env(monkeypatch, tmp_path, live_hub_cache = "/mnt/persistent/hf/hub")
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls[0][1]["cache_dir"] == "/mnt/persistent/hf/hub"


def test_prewarm_survives_runtime_cache_redirect(monkeypatch, tmp_path):
    # The pre-warm must follow the merge's runtime redirect, not the frozen constants.
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        hub_cache = "/read-only/original/hub",
        live_hub_cache = "/writable/redirect/hub",
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.snapshot_download.calls[0][1]["cache_dir"] == "/writable/redirect/hub"


def test_cached_probe_uses_live_env_cache(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch, tmp_path, cached = True, live_hub_cache = "/mnt/persistent/hf/hub"
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    assert stubs.hf_hub_download.calls, "cached probe did not run"
    assert all(
        kw.get("cache_dir") == "/mnt/persistent/hf/hub" for _, kw in stubs.hf_hub_download.calls
    )


def test_fp8_base_prewarms_16bit_sibling_not_fp8_repo(monkeypatch, tmp_path):
    # An FP8 base with a 16bit sibling merges onto the sibling, so cache the sibling, not the FP8 repo.
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        base_source = ("unsloth/Model-FP8", False, None, True, "fp8"),
        fp8_sibling = "unsloth/Model",
    )
    fn(_FakePeftModel(name_or_path = "unsloth/Model-FP8"), save_method = "merged_16bit")
    assert stubs.resolve_fp8_16bit_sibling.calls, "sibling resolver was not consulted"
    assert len(stubs.snapshot_download.calls) == 1
    assert stubs.snapshot_download.calls[0][1]["repo_id"] == "unsloth/Model"


def test_fp8_base_without_sibling_still_prewarms_fp8_repo(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        base_source = ("unsloth/Model-FP8", False, None, True, "fp8"),
        fp8_sibling = None,
    )
    fn(_FakePeftModel(name_or_path = "unsloth/Model-FP8"), save_method = "merged_16bit")
    assert len(stubs.snapshot_download.calls) == 1
    assert stubs.snapshot_download.calls[0][1]["repo_id"] == "unsloth/Model-FP8"


def test_prewarm_filters_shards_through_index(monkeypatch, tmp_path):
    # The merge keeps only indexed shards, so the pre-warm must skip an unreferenced leftover too.
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        shards = [
            ("model-00001-of-00002.safetensors", 10 * 1024**3),
            ("model-00002-of-00002.safetensors", 10 * 1024**3),
            ("leftover-00001-of-00001.safetensors", 10 * 1024**3),
        ],
        index_weight_map = {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        },
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    allow = stubs.snapshot_download.calls[0][1]["allow_patterns"]
    assert "leftover-00001-of-00001.safetensors" not in allow
    assert "model-00001-of-00002.safetensors" in allow
    assert "model-00002-of-00002.safetensors" in allow


def test_prewarm_keeps_all_shards_when_index_matches(monkeypatch, tmp_path):
    fn, stubs = _build_env(
        monkeypatch,
        tmp_path,
        shards = [
            ("model-00001-of-00002.safetensors", 10 * 1024**3),
            ("model-00002-of-00002.safetensors", 10 * 1024**3),
        ],
        index_weight_map = {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        },
    )
    fn(_FakePeftModel(), save_method = "merged_16bit")
    allow = stubs.snapshot_download.calls[0][1]["allow_patterns"]
    assert "model-00001-of-00002.safetensors" in allow
    assert "model-00002-of-00002.safetensors" in allow
