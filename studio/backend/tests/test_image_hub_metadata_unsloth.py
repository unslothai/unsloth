# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Hub METADATA call an image plan makes (listings, sizes, manifests, access probes) asks an ``unsloth/*`` repo.

Walks the diffusers plan for every quantised catalog row and every mirrored base picked as a bf16 pipeline, and the
sd.cpp plan for every GGUF row, with the Hub client replaced by a recorder that answers from a synthetic listing and
refuses any HTTP that slips past it. A fresh install: nothing cached, so every byte and every listing comes from a
mirror. (A base whose upstream copy is already cached keeps reading that copy's metadata, as released builds do: the
cache probe pins the vendor commit, and the load itself revalidates against the vendor id.)
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.inference import diffusion_families as df
from core.inference import sd_cpp_backend as bk
from core.inference.diffusion import DiffusionBackend

from .test_image_fetch_ids_unsloth import _CARD_BASES

# Third-party DENOISER rows the catalog still offers (no unsloth GGUF exists): the pick's own repo, never a companion.
_PICK_ONLY_EXCEPTIONS = frozenset({"quantstack/flux.1-krea-dev-gguf"})

_PIPELINE_FILES = {
    "model_index.json": 400,
    "scheduler/scheduler_config.json": 300,
    "text_encoder/config.json": 300,
    "text_encoder/model.safetensors": 9_000,
    "tokenizer/tokenizer.json": 500,
    "transformer/config.json": 300,
    "transformer/diffusion_pytorch_model.safetensors": 20_000,
    "vae/config.json": 300,
    "vae/diffusion_pytorch_model.safetensors": 3_000,
}
_MODEL_INDEX = {
    "_class_name": "DiffusionPipeline",
    **{c: ["diffusers", "Component"] for c in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")},
}
# What an artifact mirror ships beside the upstream folders: root checkpoints and sample renders.
_ARTIFACT_EXTRAS = {"Model-FP8.safetensors": 7_000, "samples/fp8_1.png": 1_000}
_GGUFS = {r: f"{r.split('/')[1].lower()}-Q4_K_M.gguf" for r in _CARD_BASES if r.endswith("-GGUF")}


def _record_hub(monkeypatch, tmp_path) -> list[tuple[str, str]]:
    import httpx
    import huggingface_hub
    from huggingface_hub import HfApi

    asked: list[tuple[str, str]] = []

    def _siblings(repo):
        files = dict(_PIPELINE_FILES)
        if repo in _GGUFS:
            files[_GGUFS[repo]] = 5_000
        if df.mirror_hosts_artifacts(repo):
            files.update(_ARTIFACT_EXTRAS)
        return [
            SimpleNamespace(rfilename = n, size = s, lfs = None, blob_id = f"b-{n}")
            for n, s in files.items()
        ]

    def _info(repo):
        base = _CARD_BASES.get(repo)
        card = {"base_model": base} if base else {}
        return SimpleNamespace(
            id = repo,
            sha = "0" * 40,
            gated = False,
            private = False,
            siblings = _siblings(repo),
            cardData = card,
            card_data = card,
            tags = [],
            library_name = "diffusers",
            pipeline_tag = "text-to-image",
        )

    def _api(kind, answer):
        def call(self, repo_id, *args, **kwargs):
            asked.append((kind, repo_id))
            return answer(repo_id, *args, **kwargs)

        return call

    for name, answer in {
        "model_info": lambda r, *a, **k: _info(r),
        "repo_info": lambda r, *a, **k: _info(r),
        "list_repo_files": lambda r, *a, **k: [s.rfilename for s in _siblings(r)],
        "list_repo_tree": lambda r, *a, **k: [
            SimpleNamespace(path = s.rfilename, size = s.size) for s in _siblings(r)
        ],
        "get_paths_info": lambda r, paths = (), *a, **k: [
            SimpleNamespace(path = p, size = 100, lfs = None) for p in (k.get("paths") or paths)
        ],
        "file_exists": lambda r, *a, **k: True,
    }.items():
        monkeypatch.setattr(HfApi, name, _api(name, answer))

    def hf_hub_download(repo_id, filename, *args, **kwargs):
        asked.append(("hf_hub_download", repo_id))
        if kwargs.get("local_files_only"):
            from huggingface_hub.errors import LocalEntryNotFoundError

            raise LocalEntryNotFoundError(f"{repo_id}/{filename}")
        out = tmp_path / "dl" / repo_id.replace("/", "--") / filename
        out.parent.mkdir(parents = True, exist_ok = True)
        out.write_text(json.dumps(_MODEL_INDEX) if filename == "model_index.json" else "{}")
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub.file_download, "hf_hub_download", hf_hub_download)

    def _refuse(request):
        asked.append(("http", f"{request.url.host}{request.url.path}"))
        return httpx.Response(503)

    monkeypatch.setattr(
        huggingface_hub.utils._http,
        "get_session",
        lambda: httpx.Client(transport = httpx.MockTransport(_refuse)),
    )
    return asked


def _vendor_calls(asked: list[tuple[str, str]], pick: str) -> list[tuple[str, str]]:
    def allowed(target: str) -> bool:
        t = target.lower()
        if t.startswith("huggingface.co/"):
            t = t.split("/", 1)[1]
        if t.startswith("api/models/"):
            t = t.split("/", 2)[2]
        if t.startswith("unsloth/"):
            return True
        return pick.lower() in _PICK_ONLY_EXCEPTIONS and t.startswith(pick.lower())

    return sorted({(kind, target) for kind, target in asked if not allowed(target)})


@pytest.fixture
def hub(monkeypatch, tmp_path):
    monkeypatch.setattr(df, "_upstream_is_cached", lambda *a, **k: False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)
    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_loadable", staticmethod(lambda *a, **k: False))
    return _record_hub(monkeypatch, tmp_path)


def _picks():
    return [(repo, gguf, "gguf") for repo, gguf in _GGUFS.items()] + [
        (upstream, None, "pipeline") for upstream, _mirror in df._MIRROR_PAIRS
    ]


def test_the_picks_cover_every_mirrored_base():
    picked = {r.lower() for r, _g, _k in _picks()} | {b.lower() for b in _CARD_BASES.values()}
    assert {u.lower() for u, _m in df._MIRROR_PAIRS} <= picked


def test_every_diffusers_plan_asks_only_unsloth_repos(hub):
    backend = DiffusionBackend()
    leaks = {}
    for repo, gguf, kind in _picks():
        hub.clear()
        backend.download_plan(
            repo,
            gguf_filename = gguf,
            model_kind = kind,
            allow_device_probe = False,
            memory_verdict = False,
            discover_text_encoders = False,
        )
        assert hub, repo  # the plan really asked the Hub something
        if calls := _vendor_calls(hub, repo):
            leaks[repo] = calls
    assert leaks == {}


def test_every_native_plan_asks_only_unsloth_repos(hub, monkeypatch):
    backend = bk.SdCppDiffusionBackend()
    monkeypatch.setattr(backend, "_flux2_inner_dim", lambda *a, **k: None)
    leaks = {}
    for repo, gguf in _GGUFS.items():
        hub.clear()
        backend.download_plan(repo, gguf_filename = gguf, model_kind = "gguf")
        if calls := _vendor_calls(hub, repo):
            leaks[repo] = calls
    assert leaks == {}


@pytest.mark.parametrize("kind", ["pipeline", "gguf"])
def test_an_artifact_mirror_listing_stages_only_the_upstream_folders(hub, kind):
    """Listing unsloth/Qwen-Image-2.1-Turbo-FP8 for the Turbo base must not stage its root checkpoints or sample
    renders: neither is part of the pipeline, and the vendor listing it replaces never had them."""
    base = "Qwen/Qwen-Image-2.1-Turbo"
    repo, gguf = (base, None) if kind == "pipeline" else ("unsloth/Qwen-Image-2.1-Turbo-GGUF", None)
    _total, files = DiffusionBackend._estimate_download_bytes(
        repo, "x-Q4_K_M.gguf" if kind == "gguf" else gguf, base, None, kind = kind
    )
    assert files and "model_index.json" in files
    assert not [f for f in files if f in _ARTIFACT_EXTRAS]
