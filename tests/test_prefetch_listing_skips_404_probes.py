# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Listing the repo first skips only the root index probe it proves absent; patterns and model_info calls match
the unconditional probes, and a failed or empty listing probes as before."""

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import unsloth.models._utils as U
except ImportError as exc:
    pytest.skip(f"unsloth is not importable here: {exc}", allow_module_level = True)

ST_MODULES = [
    {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
]
COSMOS_INDEX = {
    "weight_map": {"a": "transformer/m-00001-of-00002.safetensors", "b": "vision/model.safetensors"}
}

PLAIN = [
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "extra/model.safetensors",
]
SHARDED = [
    "config.json",
    "model.safetensors.index.json",
    "transformer/m-00001-of-00002.safetensors",
    "vision/model.safetensors",
    "pytorch_model.bin",
]
ST_REPO = [
    "modules.json",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "1_Pooling/config.json",
    "2_Dense/pytorch_model.bin",
]
CONTENT = {"modules.json": ST_MODULES, "model.safetensors.index.json": COSMOS_INDEX}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub serving *listing*: hf_hub_download 404s on unlisted files, model_info lists them."""
    import huggingface_hub

    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(flag, raising = False)
    state = {}

    def fake_hf_hub_download(repo_id, filename, *a, **k):
        state["downloads"].append((filename, k.get("revision"), k.get("token")))
        if state["hub_down"] and not state["cached"]:
            raise OSError("connection refused")
        if filename in state["listing"] or (state["hub_down"] and filename in state["cached"]):
            p = tmp_path / filename
            p.write_text(json.dumps(CONTENT[filename]), encoding = "utf-8")
            return str(p)
        raise OSError(f"404 {filename}")

    class _Api:
        def model_info(self, repo_id, *a, **k):
            state["infos"].append((k.get("revision"), k.get("token")))
            if state["hub_down"]:
                raise OSError("connection refused")
            listing = state["listing_override"]
            if listing is None:
                listing = state["listing"]
            if listing == "none":
                return types.SimpleNamespace(siblings = None)
            return types.SimpleNamespace(
                siblings = [types.SimpleNamespace(rfilename = f) for f in listing]
            )

    def fake_snapshot(repo_id, **kwargs):
        state["snapshot"] = kwargs
        return str(tmp_path / "snapshot")

    zoo = types.ModuleType("unsloth_zoo.hf_xet_fallback")
    zoo.snapshot_download_with_xet_fallback = fake_snapshot
    zoo.DownloadStallError = type("DownloadStallError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "unsloth_zoo.hf_xet_fallback", zoo)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)

    def run(
        listing,
        *,
        hub_down = False,
        cached = (),
        listing_override = None,
        **kwargs,
    ):
        state.update(
            listing = listing,
            hub_down = hub_down,
            cached = tuple(cached),
            listing_override = listing_override,
            downloads = [],
            infos = [],
            snapshot = None,
        )
        U.maybe_prefetch_hf_snapshot("org/repo", **{"weights_at_root": True, **kwargs})
        return dict(state)

    return run


def _unconditional(monkeypatch):
    """The pre-change probe behaviour: the listing never proves a file absent."""
    monkeypatch.setattr(U, "_listing_lacks", lambda siblings, filename: False)


KWARGS = [
    {},
    {"revision": "abc123", "token": "hf_x"},
    {"variant": "fp16"},
    {"use_safetensors": True},
    {"use_safetensors": False},
    {"from_tf": True},
    {"from_flax": True},
    {"weights_at_root": False},
    {"subfolder": "transformer"},
]


@pytest.mark.parametrize("listing", [PLAIN, SHARDED, ST_REPO], ids = ["plain", "sharded", "st"])
@pytest.mark.parametrize(
    "kwargs", KWARGS, ids = lambda k: ",".join(f"{a}={b}" for a, b in k.items()) or "auto"
)
def test_patterns_match_the_unconditional_probes(hub, monkeypatch, listing, kwargs):
    head = hub(listing, **kwargs)
    with monkeypatch.context() as m:
        _unconditional(m)
        base = hub(listing, **kwargs)
    assert head["snapshot"] == base["snapshot"]
    assert head["infos"] == base["infos"]
    dropped = [d for d in base["downloads"] if d not in head["downloads"]]
    assert all(name not in listing for name, _, _ in dropped), dropped
    assert [d for d in base["downloads"] if d not in dropped] == head["downloads"]


def test_a_plain_repo_skips_only_the_index_probe(hub):
    got = hub(PLAIN)
    assert got["downloads"] == [("modules.json", None, None)]
    assert got["infos"] == [(None, None)]


def test_modules_json_is_probed_even_when_the_listing_lacks_it(hub):
    """Its 404 writes the .no_exist marker FastSentenceTransformer._modules_json_for_gating reads offline."""
    for listing in (PLAIN, SHARDED):
        got = hub(listing)
        assert [d[0] for d in got["downloads"]][:1] == ["modules.json"], listing


def test_listed_files_are_still_probed_with_the_load_ref(hub):
    got = hub(SHARDED + ["modules.json"], revision = "abc123", token = "hf_x")
    assert got["downloads"] == [
        ("modules.json", "abc123", "hf_x"),
        ("model.safetensors.index.json", "abc123", "hf_x"),
    ]
    assert got["infos"] == [("abc123", "hf_x")]


def test_an_explicit_format_adds_no_model_info_call(hub):
    for kwargs in ({"use_safetensors": True}, {"use_safetensors": False}, {"from_tf": True}):
        got = hub(PLAIN, **kwargs)
        assert got["infos"] == [], kwargs
        assert [d[0] for d in got["downloads"]] == ["modules.json", "model.safetensors.index.json"]


def test_a_hub_outage_still_probes_the_cache(hub, monkeypatch):
    """model_info fails offline-by-accident while hf_hub_download can still answer from the local cache."""
    head = hub(ST_REPO, hub_down = True, cached = ("modules.json",))
    with monkeypatch.context() as m:
        _unconditional(m)
        base = hub(ST_REPO, hub_down = True, cached = ("modules.json",))
    assert [d[0] for d in head["downloads"]] == ["modules.json", "model.safetensors.index.json"]
    assert head["downloads"] == base["downloads"]
    assert head["infos"] == base["infos"] == [(None, None)]
    assert head["snapshot"] == base["snapshot"]
    assert "*/*.safetensors" not in head["snapshot"]["ignore_patterns"]


@pytest.mark.parametrize("override", [[], "none"], ids = ["empty", "none"])
def test_an_unusable_listing_probes_as_before(hub, override):
    got = hub(ST_REPO, listing_override = override)
    assert [d[0] for d in got["downloads"]] == ["modules.json", "model.safetensors.index.json"]
    assert got["infos"] == [(None, None)]


def test_listing_lacks_is_conservative():
    sib = [
        types.SimpleNamespace(rfilename = "a\\modules.json"),
        types.SimpleNamespace(rfilename = "config.json"),
    ]
    assert U._listing_lacks(sib, "modules.json") is True
    assert (
        U._listing_lacks([types.SimpleNamespace(rfilename = "modules.json")], "modules.json") is False
    )
    assert U._listing_lacks(None, "modules.json") is False
    assert U._listing_lacks([], "modules.json") is False
    assert U._listing_lacks(U._SIBLINGS_NOT_FETCHED, "modules.json") is False
    assert U._listing_lacks([object()], "modules.json") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
