"""Regression coverage for complete encoder caches without a pipeline manifest."""

import json
import pytest
from hub.tests.test_model_services import _diffusion_scan, _file
from hub.services.models import cache_inventory, local_inventory
from hub.utils.inventory_scan import snapshot_is_encoder_companion


def test_encoder_only_cache_is_a_companion_in_both_inventories(monkeypatch, tmp_path):
    real = cache_inventory.hf_cache_scan.is_snapshot_partial
    repo = "unsloth/Qwen-Image-FP8"
    _diffusion_scan(
        monkeypatch,
        tmp_path,
        repo,
        [_file("text_encoder/model.safetensors", 4096)],
        task = "text-to-image",
    )
    monkeypatch.setattr(cache_inventory.hf_cache_scan, "is_snapshot_partial", real)
    (row,) = cache_inventory._scan_cached_models(active_hub_cache = tmp_path / "hub")
    assert row["partial"] is True  # Not a standalone loadable model.
    assert row["companion_prefetch"] is True
    assert row["partial_transport"] is None
    rows = local_inventory._scan_hf_cache(tmp_path / "hub")
    assert rows
    assert all(row.partial and row.companion_prefetch for row in rows)


@pytest.mark.parametrize(
    "case", ["complete", "unknown-repo", "missing-shard", "empty", "denoiser", "bad-index"]
)
def test_encoder_snapshot_requires_positive_component_evidence(tmp_path, case):
    encoder = tmp_path / "text_encoder"
    encoder.mkdir()
    weight = encoder / "model.safetensors"
    weight.write_bytes(b"weights" if case != "empty" else b"")
    repo = "unsloth/Qwen-Image-FP8"
    if case == "unknown-repo":
        repo = "someone/unrecognized"
    elif case == "missing-shard":
        weight.rename(encoder / "model-00001-of-00002.safetensors")
    elif case == "denoiser":
        (tmp_path / "checkpoint.safetensors").write_bytes(b"weights")
    elif case == "bad-index":
        (encoder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "missing.safetensors"}})
        )
    assert snapshot_is_encoder_companion(repo, tmp_path) is (case == "complete")


def test_interrupted_encoder_cache_keeps_partial_status(monkeypatch, tmp_path):
    row = _diffusion_scan(
        monkeypatch,
        tmp_path,
        "unsloth/Qwen-Image-FP8",
        [_file("text_encoder/model.safetensors", 4096)],
        task = "text-to-image",
        download_partial = True,
    )
    assert row["partial"] is True
    assert row["companion_prefetch"] is False
