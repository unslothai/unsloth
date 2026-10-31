# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
import threading
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from core.training import model_weight_prefetch as prefetch
from utils.account_context import AccountContext, current_account_id, run_as


def test_checkpoint_selection_chooses_single_safetensors_without_other_formats():
    files = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "q4.gguf",
    ]
    reads = []
    assert prefetch.checkpoint_files(files, lambda name: reads.append(name) or {}) == [
        "model.safetensors"
    ]
    assert reads == ["config.json"]


def test_checkpoint_selection_uses_only_exact_index_shards():
    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    metadata = {
        "config.json": {},
        "model.safetensors.index.json": {"weight_map": {"a": second, "b": first, "c": first}},
    }
    assert prefetch.checkpoint_files(
        [*metadata, first, second, "other.safetensors"], metadata.__getitem__
    ) == [first, second]


@pytest.mark.parametrize(
    "shard",
    [
        "../secret.safetensors",
        "/model.safetensors",
        "folder/model-1-of-1.safetensors",
        "model-1-of-1.bin",
        123,
        ["bad"],
    ],
)
def test_unfamiliar_index_paths_are_not_downloaded(shard):
    metadata = {"config.json": {}, "model.safetensors.index.json": {"weight_map": {"a": shard}}}
    assert prefetch.checkpoint_files(metadata, metadata.__getitem__) == []


@pytest.mark.parametrize(
    "config",
    [{"auto_map": {"AutoModel": "custom.Model"}}, {"base_model_name_or_path": "other/base"}],
)
def test_custom_and_adapter_configs_are_left_to_loader(config):
    assert (
        prefetch.checkpoint_files(["config.json", "model.safetensors"], lambda _name: config) == []
    )
    assert (
        prefetch.checkpoint_files(
            ["adapter_config.json", "config.json", "model.safetensors"],
            lambda _name: pytest.fail("adapter metadata must not be fetched"),
        )
        == []
    )


@pytest.fixture
def mirror(monkeypatch):
    import utils.models.unsloth_mirror as mapping

    seen = []
    monkeypatch.setattr(mapping, "mirror_lookup_available", lambda: True)
    monkeypatch.setattr(
        mapping,
        "unsloth_public_mirror",
        lambda name, quant: seen.append((name, quant))
        or ("unsloth/model-bnb-4bit" if quant else "unsloth/model"),
    )
    return seen


def test_prefetch_resolves_the_effective_quantization(mirror):
    trainer = SimpleNamespace(should_stop = False, _audio_type_known = True)
    capabilities = SimpleNamespace(ALLOW_BITSANDBYTES = True, ALLOW_PREQUANTIZED_MODELS = True)
    config = {"model_name": "upstream/model", "dataset_streaming": True}
    assert (
        prefetch.prefetch_repo(config, trainer, load_in_4bit = True, capabilities = capabilities)
        == "unsloth/model-bnb-4bit"
    )
    assert (
        prefetch.prefetch_repo(config, trainer, load_in_4bit = False, capabilities = capabilities)
        == "unsloth/model"
    )
    capabilities.ALLOW_BITSANDBYTES = False
    assert (
        prefetch.prefetch_repo(config, trainer, load_in_4bit = True, capabilities = capabilities)
        == "unsloth/model"
    )
    assert mirror == [
        ("upstream/model", True),
        ("upstream/model", False),
        ("upstream/model", False),
    ]


@pytest.mark.parametrize(
    "patch",
    [
        {"model_revision": "sha"},
        {"model_snapshot_path": "snapshot"},
        {"resume_from_checkpoint": "checkpoint"},
        {"require_exact_model_resource": True},
        {"require_exact_resume_resources": True},
        {"trust_remote_code": True},
        {"training_type": "Full Finetuning"},
        {"is_dataset_audio": True},
        {"is_decision": True},
        {"is_embedding": True},
        {"model_name": "/local/model"},
    ],
)
def test_pinned_and_uncertain_jobs_do_not_prefetch(patch, mirror):
    trainer = SimpleNamespace(should_stop = False, _audio_type_known = True)
    capabilities = SimpleNamespace(ALLOW_BITSANDBYTES = True, ALLOW_PREQUANTIZED_MODELS = True)
    assert (
        prefetch.prefetch_repo(
            {"model_name": "upstream/model", **patch},
            trainer,
            load_in_4bit = True,
            capabilities = capabilities,
        )
        is None
    )
    assert mirror == []


def test_unproven_runtime_and_offline_jobs_do_not_prefetch(mirror):
    trainer = SimpleNamespace(should_stop = False, _audio_type_known = True)
    config = {"model_name": "upstream/model"}
    for capabilities in (
        None,
        SimpleNamespace(),
        SimpleNamespace(ALLOW_BITSANDBYTES = True, ALLOW_PREQUANTIZED_MODELS = False),
    ):
        assert (
            prefetch.prefetch_repo(config, trainer, load_in_4bit = True, capabilities = capabilities)
            is None
        )
    assert (
        prefetch.prefetch_repo(
            config, trainer, load_in_4bit = True, capabilities = object(), offline = True
        )
        is None
    )
    assert mirror == []


@pytest.mark.parametrize(
    "num_proc, streaming, expected",
    [(None, False, True), (1, False, True), (4, False, False), (4, True, True)],
)
def test_worker_uses_prefetch_only_without_forking_preparation(
    monkeypatch, tmp_path, mirror, num_proc, streaming, expected
):
    from core.training import worker
    import utils.hardware
    import utils.hf_cache_settings

    capabilities = SimpleNamespace(ALLOW_BITSANDBYTES = True, ALLOW_PREQUANTIZED_MODELS = True)
    monkeypatch.setitem(sys.modules, "unsloth.device_type", capabilities)
    monkeypatch.setattr(worker, "_model_offline_mode_enabled", lambda: False)
    resolved = []
    monkeypatch.setattr(
        worker, "_effective_training_load_in_4bit", lambda *_a: resolved.append(True) or False
    )
    monkeypatch.setattr(utils.hardware, "dataset_map_num_proc", lambda: num_proc)
    monkeypatch.setattr(
        utils.hf_cache_settings, "get_hf_cache_paths", lambda: SimpleNamespace(hub_cache = tmp_path)
    )
    job = worker._training_weight_prefetch(
        {"model_name": "upstream/model", "dataset_streaming": streaming},
        SimpleNamespace(should_stop = False, _audio_type_known = True),
        False,
        SimpleNamespace(put = lambda _ev: None),
    )
    assert (job is not None) is expected
    assert bool(resolved) is expected
    if job:
        assert job.repo == "unsloth/model"
        assert job.token is False
        assert job.cache_dir == str(tmp_path)


def fake_hub(monkeypatch, tmp_path, *, download):
    commit = "a" * 40
    info = SimpleNamespace(
        sha = commit,
        siblings = [
            SimpleNamespace(rfilename = name)
            for name in ("config.json", "model.safetensors", "pytorch_model.bin")
        ],
    )
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        if kwargs["filename"] == "config.json":
            path.write_text(json.dumps({}))
        else:
            download()
            path.write_bytes(b"weights")
        return str(path)

    watchdog = threading.Event()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi = lambda: SimpleNamespace(model_info = lambda *a, **kw: info), hf_hub_download = fetch
        ),
    )
    monkeypatch.setitem(
        sys.modules, "utils.hf_xet_fallback", SimpleNamespace(start_watchdog = lambda **_kw: watchdog)
    )
    return calls, watchdog


def test_model_io_overlaps_dataset_preparation_and_keeps_context(monkeypatch, tmp_path):
    entered, finish = threading.Event(), threading.Event()
    custom = ContextVar("prefetch_test_context", default = "unset")
    observed = []

    def download():
        observed.append((current_account_id(), custom.get(), threading.current_thread().ident))
        entered.set()
        assert finish.wait(3)

    calls, watchdog = fake_hub(monkeypatch, tmp_path, download = download)
    events = []
    account = AccountContext("test-account", "test")
    context_token = custom.set("captured")
    main_thread = threading.current_thread().ident

    class LazyDataset:
        def __iter__(self):
            pytest.fail("startup must not consume streaming rows")

    dataset = LazyDataset()

    def load_dataset():
        assert entered.wait(3), "weight I/O must already be in progress during dataset preparation"
        assert not watchdog.is_set()
        finish.set()
        return dataset, None

    def run():
        job = prefetch.ModelWeightPrefetch(
            "unsloth/model",
            token = False,
            cache_dir = tmp_path / "account-cache",
            emit = events.append,
            should_stop = lambda: False,
        )
        assert prefetch.prepare_dataset_with_prefetch(load_dataset, job) == (dataset, None)
        assert (
            not job._thread.is_alive()
        ), "actual model placement starts after background I/O finishes"

    try:
        run_as(account, run)
    finally:
        custom.reset(context_token)
        finish.set()
    assert observed == [("test-account", "captured", observed[0][2])]
    assert observed[0][2] != main_thread
    assert [call["filename"] for call in calls] == ["config.json", "model.safetensors"]
    assert all(
        call["revision"] == "a" * 40
        and call["token"] is False
        and call["cache_dir"] == str(tmp_path / "account-cache")
        for call in calls
    )
    assert [event["type"] for event in events] == [
        "model_load_started",
        "model_load_resolved",
        "model_load_completed",
    ]
    assert watchdog.is_set()


@pytest.mark.parametrize("outcome", ["error", "empty", "stop"])
def test_failed_or_stopped_dataset_does_not_wait_for_download(monkeypatch, tmp_path, outcome):
    entered, release = threading.Event(), threading.Event()
    calls, watchdog = fake_hub(
        monkeypatch, tmp_path, download = lambda: (entered.set(), release.wait(3))
    )
    events = []
    stopped = []
    job = prefetch.ModelWeightPrefetch(
        "unsloth/model",
        token = "token",
        cache_dir = tmp_path,
        emit = events.append,
        should_stop = lambda: bool(stopped),
    )

    def load_dataset():
        assert entered.wait(3)
        if outcome == "error":
            raise ValueError("invalid dataset")
        if outcome == "stop":
            stopped.append(True)
        return (None, None) if outcome == "empty" else (object(), None)

    try:
        if outcome == "error":
            with pytest.raises(ValueError, match = "invalid dataset"):
                prefetch.prepare_dataset_with_prefetch(load_dataset, job)
        else:
            prefetch.prepare_dataset_with_prefetch(load_dataset, job)
        assert job._thread.is_alive(), "an in-flight Hub call must not block cancellation"
        assert watchdog.is_set()
        assert events[-1]["type"] == "model_load_completed"
        before = list(events)
        job._send("stall", message = "late")
        assert events == before
    finally:
        release.set()
        job._thread.join(timeout = 3)
    assert not job._thread.is_alive()


def test_prefetch_failure_leaves_dataset_result_authoritative(monkeypatch, tmp_path):
    fake_hub(
        monkeypatch,
        tmp_path,
        download = lambda: (_ for _ in ()).throw(OSError("temporary Hub error")),
    )
    events = []
    dataset = object()
    job = prefetch.ModelWeightPrefetch(
        "unsloth/model",
        token = None,
        cache_dir = tmp_path,
        emit = events.append,
        should_stop = lambda: False,
    )
    assert prefetch.prepare_dataset_with_prefetch(lambda: (dataset, None), job) == (dataset, None)
    assert events[-1]["type"] == "model_load_completed"
