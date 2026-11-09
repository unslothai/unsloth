# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image / video downloads and diffusers loads put .safetensors first: a pickle beside its safetensors twin is never
fetched, a pickle that is the only copy still is, and use_safetensors=True is passed where the folder ships
safetensors."""

from __future__ import annotations

import ast
import json
import struct
import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusers_safetensors as ds
from hub.utils.snapshot_filters import (
    resolve_snapshot_ignore_patterns_for_files,
    sibling_matches_ignore,
)

_BACKEND = Path(__file__).resolve().parents[1]


def _write_safetensors(path: Path) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def _pipeline(tmp_path: Path, components: dict) -> Path:
    root = tmp_path / "pipe"
    root.mkdir()
    index = {
        "_class_name": "FluxPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    for name, files in components.items():
        index[name] = ["diffusers", "Y"]
        (root / name).mkdir()
        (root / name / "config.json").write_text("{}")
        for f in files:
            if f.endswith(".safetensors"):
                _write_safetensors(root / name / f)
            else:
                (root / name / f).write_bytes(b"x")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text("{}")
    (root / "model_index.json").write_text(json.dumps(index))
    return root


# --- download filters -------------------------------------------------------------------------------------------


_DIFFUSERS_REPO = [
    "model_index.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.bin",
    "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.bin",
    "safety_checker/pytorch_model.bin",  # its only copy: kept
    "vae/diffusion_pytorch_model.fp16.safetensors",  # a variant does not make the .bin redundant
    "vae/diffusion_pytorch_model.bin",
    "sd_xl_base_1.0.ckpt",  # root pickle with no safetensors twin: the only copy, kept
    "image_encoder/open_clip.pth",
    "Wan2.2-T2V-A14B-FP8.pt",  # pre-quant hosted only as .pt: kept
    "Wan2.2-T2V-A14B-INT8.pt",
    "Wan2.2-T2V-A14B-INT8.safetensors",  # its twin makes the .pt redundant
]


def test_a_pipeline_repo_never_downloads_a_pickle_beside_safetensors():
    skipped = set(ds.pipeline_pickle_files(_DIFFUSERS_REPO))
    assert skipped == {
        "unet/diffusion_pytorch_model.bin",
        "text_encoder/pytorch_model.bin",
        "image_encoder/open_clip.pth",
        "Wan2.2-T2V-A14B-INT8.pt",
    }


def test_a_bin_only_component_still_downloads():
    skipped = set(ds.pipeline_pickle_files(_DIFFUSERS_REPO))
    assert "safety_checker/pytorch_model.bin" not in skipped
    assert "vae/diffusion_pytorch_model.bin" not in skipped
    assert "sd_xl_base_1.0.ckpt" not in skipped
    assert "Wan2.2-T2V-A14B-FP8.pt" not in skipped


def test_a_modular_pipeline_repo_is_filtered_too():
    names = [
        "modular_model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.bin",
        "x.pt",
        "x.safetensors",
    ]
    assert set(ds.pipeline_pickle_files(names)) == {
        "transformer/diffusion_pytorch_model.bin",
        "x.pt",
    }


def test_an_unrelated_safetensors_never_drops_the_only_bin(tmp_path):
    # An adapter / aux .safetensors beside a .bin-only component is not that component's weights.
    names = [
        "model_index.json",
        "transformer/diffusion_pytorch_model.bin",
        "transformer/adapter_model.safetensors",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/pytorch_model.bin",
    ]
    assert set(ds.pipeline_pickle_files(names)) == {"text_encoder/pytorch_model.bin"}
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "diffusion_pytorch_model.bin").write_bytes(b"x")
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    assert ds.component_weight_state(folder) is False
    (folder / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    assert ds.component_weight_state(folder) is True


def test_a_non_pipeline_repo_keeps_its_files():
    # LLM / audio repos keep their own rules: no manifest, nothing extra is skipped here.
    names = [
        "config.json",
        "model.safetensors",
        "consolidated.pth",
        "tokenizer.model",
        "optimizer.pt",
    ]
    assert ds.pipeline_pickle_files(names) == []


def test_the_hub_snapshot_filter_skips_diffusers_pickles():
    ignore = resolve_snapshot_ignore_patterns_for_files(_DIFFUSERS_REPO)
    for name in ("unet/diffusion_pytorch_model.bin", "Wan2.2-T2V-A14B-INT8.pt"):
        assert sibling_matches_ignore(name, ignore), name
    for name in (
        "unet/diffusion_pytorch_model.safetensors",
        "safety_checker/pytorch_model.bin",
        "Wan2.2-T2V-A14B-FP8.pt",
        "sd_xl_base_1.0.ckpt",
    ):
        assert not sibling_matches_ignore(name, ignore), name


def test_the_hub_snapshot_filter_is_unchanged_for_an_llm_repo():
    names = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "training_args.bin",
        "extra.pt",
    ]
    ignore = resolve_snapshot_ignore_patterns_for_files(names)
    assert not sibling_matches_ignore("extra.pt", ignore)
    assert not sibling_matches_ignore("training_args.bin", ignore)


def test_the_frontend_size_estimate_mirrors_the_pipeline_pickle_filter():
    src = (
        _BACKEND.parent / "frontend" / "src" / "features" / "hub" / "lib" / "dataset-size.ts"
    ).read_text(encoding = "utf-8")
    assert "function diffusersPickleFiles(" in src
    assert '"model_index.json", "modular_model_index.json"' in src
    assert r"/\.(pt|pth|ckpt|pkl|pickle)$/i" in src
    # The same component-weight matcher as the backend: an adapter / variant safetensors never drops the .bin.
    backend = ds._MODEL_SAFETENSORS.pattern
    assert (
        backend == r"^(diffusion_pytorch_model|model)(-\d+-of-\d+)?\.safetensors(\.index\.json)?$"
    )
    assert "/" + backend.replace("\\d", "\\d") + "/" in src
    assert "MODEL_SAFETENSORS_RE.test(basename(n))" in src
    assert "pipelinePickles.has(filename)" in src
    assert "safetensorsStems.has(" in src


def test_pipeline_and_base_file_filters_drop_never_read_pickles():
    from core.inference.diffusion import _base_file_downloaded, _pipeline_file_downloaded

    for name in ("unet/x.pt", "vae/y.pth", "text_encoder/z.ckpt", "transformer/w.pkl"):
        assert not _pipeline_file_downloaded(name), name
        assert not _base_file_downloaded(name, include_transformer = True), name
    assert _pipeline_file_downloaded("unet/diffusion_pytorch_model.safetensors")
    assert _base_file_downloaded("vae/diffusion_pytorch_model.safetensors")


def test_the_dense_encoder_fetch_ignores_pickles(monkeypatch):
    from core.inference import diffusion_te_prequant as tp

    listing = [
        "text_encoder/model.safetensors",
        "text_encoder/pytorch_model.bin",
        "text_encoder_2/pytorch_model.bin",
    ]

    class _Api:
        def list_repo_files(self, *_a, **_k):
            return listing

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi = _Api))
    patterns = tp._dense_component_pickles("o/r", ["text_encoder", "text_encoder_2"], None, None)
    assert "text_encoder/pytorch_model.bin" in patterns
    assert "text_encoder_2/pytorch_model.bin" not in patterns  # its only copy
    assert "text_encoder/*.pt" in patterns and "text_encoder_2/*.ckpt" in patterns


def test_the_dense_encoder_fetch_without_a_listing_drops_only_never_read_pickles(monkeypatch):
    from core.inference import diffusion_te_prequant as tp

    class _Api:
        def list_repo_files(self, *_a, **_k):
            raise OSError("offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi = _Api))
    patterns = tp._dense_component_pickles("o/r", ["text_encoder"], None, None)
    assert not any(p.endswith(".bin") for p in patterns)
    assert "text_encoder/*.pt" in patterns


# --- use_safetensors=True ---------------------------------------------------------------------------------------


def test_a_folder_shipping_safetensors_everywhere_loads_with_use_safetensors(tmp_path):
    root = _pipeline(
        tmp_path,
        {
            "transformer": ["diffusion_pytorch_model.safetensors"],
            "text_encoder": ["model.safetensors", "pytorch_model.bin"],
            "vae": ["diffusion_pytorch_model.safetensors"],
        },
    )
    assert ds.pipeline_safetensors_kwargs(str(root)) == {"use_safetensors": True}


def test_a_bin_only_component_keeps_the_library_default(tmp_path):
    root = _pipeline(
        tmp_path,
        {
            "transformer": ["diffusion_pytorch_model.safetensors"],
            "text_encoder": ["pytorch_model.bin"],
        },
    )
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}
    assert ds.pipeline_safetensors_kwargs(str(root), passed = {"text_encoder": object()}) == {
        "use_safetensors": True
    }


def test_a_partly_cached_snapshot_is_unknown_not_complete(tmp_path):
    """A weight-bearing component that is missing, or present with its config only (a partly cached snapshot),
    could still be .bin-only on the Hub: never force safetensors on it."""
    root = _pipeline(tmp_path, {"transformer": ["diffusion_pytorch_model.safetensors"]})
    index = json.loads((root / "model_index.json").read_text())
    index["text_encoder"] = ["transformers", "CLIPTextModel"]
    index["tokenizer"] = ["transformers", "CLIPTokenizer"]
    (root / "model_index.json").write_text(json.dumps(index))
    # tokenizer and scheduler hold no weights, so their absence or config-only folder is no evidence either way
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text("{}")
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}
    _write_safetensors(root / "text_encoder" / "model.safetensors")
    assert ds.pipeline_safetensors_kwargs(str(root)) == {"use_safetensors": True}


def test_a_variant_only_safetensors_does_not_count(tmp_path):
    root = _pipeline(
        tmp_path,
        {"vae": ["diffusion_pytorch_model.fp16.safetensors", "diffusion_pytorch_model.bin"]},
    )
    assert ds.pipeline_safetensors_kwargs(str(root)) == {}


def test_an_unknown_source_keeps_the_library_default(tmp_path):
    assert ds.pipeline_safetensors_kwargs("nobody/not-cached-anywhere-xyz") == {}
    assert ds.pipeline_safetensors_kwargs(str(tmp_path)) == {}  # no manifest
    assert ds.pipeline_safetensors_kwargs("") == {}


def test_a_non_diffusers_pipeline_class_never_gets_the_kwarg(tmp_path, monkeypatch):
    root = _pipeline(tmp_path, {"transformer": ["diffusion_pytorch_model.safetensors"]})

    class DiffusionPipeline:
        pass

    class FluxPipeline(DiffusionPipeline):
        pass

    class Custom:
        pass

    monkeypatch.setitem(
        sys.modules, "diffusers", types.SimpleNamespace(DiffusionPipeline = DiffusionPipeline)
    )
    assert ds.pipeline_safetensors_kwargs(str(root), pipeline_cls = Custom) == {}
    assert ds.pipeline_safetensors_kwargs(str(root), pipeline_cls = FluxPipeline) == {
        "use_safetensors": True
    }


def test_a_component_load_takes_use_safetensors_only_when_its_folder_ships_them(tmp_path):
    root = _pipeline(
        tmp_path,
        {
            "transformer": ["diffusion_pytorch_model.safetensors"],
            "text_encoder": ["pytorch_model.bin"],
        },
    )
    assert ds.component_safetensors_kwargs(str(root), "transformer") == {"use_safetensors": True}
    assert ds.component_safetensors_kwargs(str(root), "text_encoder") == {}
    assert ds.component_safetensors_kwargs(str(root), "missing") == {}


def test_a_hub_id_uses_its_cached_snapshot(tmp_path, monkeypatch):
    root = _pipeline(tmp_path, {"transformer": ["diffusion_pytorch_model.safetensors"]})

    def _cached(
        repo_id,
        filename,
        cache_dir = None,
    ):
        path = root / filename
        return str(path) if path.is_file() else None

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(try_to_load_from_cache = _cached)
    )
    assert ds.pipeline_safetensors_kwargs("org/model") == {"use_safetensors": True}
    assert ds.component_safetensors_kwargs("org/model", "transformer") == {"use_safetensors": True}


def _calls_with_helper(path: Path, helper: str) -> int:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_pretrained":
            for kw in node.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Call):
                    name = getattr(kw.value.func, "id", None) or getattr(
                        kw.value.func, "attr", None
                    )
                    if name == helper:
                        count += 1
    return count


def test_every_dense_diffusers_load_asks_for_safetensors():
    image = _BACKEND / "core" / "inference" / "diffusion.py"
    video = _BACKEND / "core" / "inference" / "video.py"
    assert _calls_with_helper(image, "pipeline_safetensors_kwargs") == 3
    assert _calls_with_helper(image, "component_safetensors_kwargs") == 1
    assert _calls_with_helper(video, "_safetensors_kwargs") == 3
    assert _calls_with_helper(video, "component_safetensors_kwargs") == 1


def test_a_pinned_revision_is_judged_on_its_own_snapshot(tmp_path):
    # Cached main ships safetensors, the pinned commit only a .bin: the pinned load must not be forced to safetensors.
    cache = tmp_path / "hub"
    repo = cache / "models--o--pipe"
    main_rev, pinned = "a" * 40, "b" * 40
    for rev, weight in (
        (main_rev, "diffusion_pytorch_model.safetensors"),
        (pinned, "diffusion_pytorch_model.bin"),
    ):
        snap = repo / "snapshots" / rev
        (snap / "transformer").mkdir(parents = True)
        (snap / "transformer" / "config.json").write_text("{}")
        if weight.endswith(".safetensors"):
            _write_safetensors(snap / "transformer" / weight)
        else:
            (snap / "transformer" / weight).write_bytes(b"x")
        (snap / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Y"]}))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(main_rev)
    assert ds.pipeline_safetensors_kwargs("o/pipe", cache_dir = str(cache)) == {
        "use_safetensors": True
    }
    assert ds.pipeline_safetensors_kwargs(
        "o/pipe", passed = {"revision": pinned}, cache_dir = str(cache)
    ) != {"use_safetensors": True}
    assert ds.component_safetensors_kwargs(
        "o/pipe", "transformer", cache_dir = str(cache), revision = pinned
    ) != {"use_safetensors": True}
    assert ds.component_safetensors_kwargs("o/pipe", "transformer", cache_dir = str(cache)) == {
        "use_safetensors": True
    }


def test_an_orphan_shard_without_its_index_is_not_safetensors_weights(tmp_path):
    folder = tmp_path / "transformer"
    folder.mkdir()
    (folder / "diffusion_pytorch_model.bin").write_bytes(b"x")
    (folder / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"x")
    assert ds.component_weight_state(folder) is False
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
    assert ds.component_weight_state(folder) is True
