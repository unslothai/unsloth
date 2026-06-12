import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

_BACKEND = Path(__file__).resolve().parent / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from models.datasets import DatasetSplitsRequest  # noqa: E402
import routes.datasets as rd  # noqa: E402


def test_omitted_hf_token_forwards_false_to_library(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds

    captured = {}

    def _configs(name, token = None):
        captured["configs_token"] = token
        return ["default"]

    def _splits(name, config_name = None, token = None):
        captured.setdefault("splits_tokens", []).append(token)
        return ["train"]

    monkeypatch.setattr(ds, "get_dataset_config_names", _configs)
    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)

    req = DatasetSplitsRequest(dataset_name = "owner/public")
    rd.get_dataset_splits(req, current_subject = "t")

    assert captured["configs_token"] is False
    assert all(t is False for t in captured["splits_tokens"])


def test_supplied_hf_token_is_forwarded_verbatim(
    monkeypatch, fake_datasets, fake_structlog
):
    import datasets as ds

    captured = {}

    def _configs(name, token = None):
        captured["configs_token"] = token
        return ["default"]

    def _splits(name, config_name = None, token = None):
        captured.setdefault("splits_tokens", []).append(token)
        return ["train"]

    monkeypatch.setattr(ds, "get_dataset_config_names", _configs)
    monkeypatch.setattr(ds, "get_dataset_split_names", _splits)

    supplied = "hf_USER_TOKEN_abcdef12345"
    req = DatasetSplitsRequest(dataset_name = "owner/private", hf_token = supplied)
    rd.get_dataset_splits(req, current_subject = "t")

    assert captured["configs_token"] == supplied
    assert all(t == supplied for t in captured["splits_tokens"])
