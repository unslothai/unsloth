# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util

import pytest

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason = "fastapi is not installed",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("huggingface_hub") is None,
        reason = "huggingface_hub is not installed",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("structlog") is None,
        reason = "structlog is not installed",
    ),
]


def test_tokened_dataset_size_failure_does_not_poison_anonymous_neg_cache(
    monkeypatch,
):
    import routes.datasets as dataset_routes

    repo_id = "org/private-dataset"

    class FailingHfApi:
        def __init__(self, token = None):
            self.token = token

        def dataset_info(self, *_args, **_kwargs):
            raise RuntimeError("temporary HF failure")

    with dataset_routes._dataset_size_cache_lock:
        dataset_routes._dataset_size_cache.clear()
        dataset_routes._dataset_size_neg_cache.clear()

    monkeypatch.setattr("huggingface_hub.HfApi", FailingHfApi)

    try:
        assert dataset_routes._get_dataset_size_cached(repo_id, "hf_token") == 0
        assert repo_id not in dataset_routes._dataset_size_neg_cache

        assert dataset_routes._get_dataset_size_cached(repo_id, None) == 0
        assert repo_id in dataset_routes._dataset_size_neg_cache
    finally:
        with dataset_routes._dataset_size_cache_lock:
            dataset_routes._dataset_size_cache.clear()
            dataset_routes._dataset_size_neg_cache.clear()
