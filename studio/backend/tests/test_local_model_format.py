# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local folder's rows carry the formats its weights show; an mmproj or calibration file alone is no GGUF."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from hub.services.models import local_inventory


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0")
    return path


_TRANSFORMER = {"transformer": ["diffusers", "Transformer2DModel"]}


def _write(path: Path, text: str = "{}") -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text)


def _index(
    root: Path,
    name: str = "model_index.json",
    **payload,
) -> Path:
    payload.setdefault("_class_name", "DiffusionPipeline")
    _write(root / name, json.dumps(payload))
    return root / name


def _pipeline_manifest(
    root: Path,
    name: str = "model_index.json",
    **payload,
) -> Path:
    if name.startswith("modular_"):
        payload.setdefault("_blocks_class_name", "TestPipelineBlocks")
    path = _index(root, name, **{**_TRANSFORMER, **payload})
    _write(root / "transformer" / "config.json")
    _touch(root / "transformer" / "diffusion_pytorch_model.safetensors")
    return path


def _weight_map(path: Path, shard: str) -> None:
    _write(path, json.dumps({"weight_map": {"layer": shard}}))


def _complete(root: Path, *name: str, **kwargs) -> bool:
    from core.inference.diffusion_families import local_pipeline_components_are_complete
    return local_pipeline_components_are_complete(root, *(name or ("model_index.json",)), **kwargs)


def _empty_sources(tmp_path: Path):
    empty = tmp_path / "empty"
    return local_inventory._LocalInventorySources(
        hf_cache_dir = empty / "active",
        legacy_hf = empty / "legacy",
        hf_default = empty / "default",
        lm_dirs = (),
        ollama_dirs = (),
        hermes_dirs = (),
        known_hf_caches = (),
    )


def test_compat_inventory_groups_custom_gguf_variants(tmp_path):
    root = tmp_path / "custom"
    model = root / "publisher"
    _touch(model / "model-Q4_K_M.gguf")
    _touch(model / "model-Q8_0.gguf")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}],
        sources = _empty_sources(tmp_path),
    )

    assert [(row.source, Path(row.path)) for row in rows] == [("custom", model)]


def test_compat_inventory_preserves_distinct_custom_ggufs(tmp_path):
    root = tmp_path / "custom"
    holder = root / "publisher"
    first = _touch(holder / "model-a-Q4_K_M.gguf")
    second = _touch(holder / "model-b-Q4_K_M.gguf")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}],
        sources = _empty_sources(tmp_path),
    )

    assert {Path(row.path) for row in rows} == {first, second}


def test_compat_inventory_dedupes_overlapping_custom_roots(tmp_path):
    root = tmp_path / "custom"
    model = root / "publisher"
    _touch(model / "model-Q4_K_M.gguf")
    _touch(model / "model-Q8_0.gguf")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}, {"path": str(model)}],
        sources = _empty_sources(tmp_path),
    )

    assert [(row.source, Path(row.path)) for row in rows] == [("custom", model)]


def test_compat_inventory_dedupes_nested_quant_root(tmp_path):
    root = tmp_path / "custom"
    model = root / "publisher"
    _touch(model / "model-a-Q4_K_M.gguf")
    quant_dir = model / "Q8_0"
    _touch(quant_dir / "model-a-Q8_0.gguf")
    (quant_dir / "config.json").write_text("{}", encoding = "utf-8")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}, {"path": str(quant_dir)}],
        sources = _empty_sources(tmp_path),
    )

    assert [(row.source, Path(row.path)) for row in rows] == [("custom", model)]


def test_compat_inventory_preserves_nested_independent_model_root(tmp_path):
    root = tmp_path / "custom"
    parent = root / "parent"
    _touch(parent / "parent-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "child"
    _touch(child / "child-Q8_0.gguf")
    (child / "config.json").write_text("{}", encoding = "utf-8")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}, {"path": str(child)}],
        sources = _empty_sources(tmp_path),
    )

    assert {(row.source, Path(row.path)) for row in rows} == {
        ("custom", parent),
        ("custom", child),
    }


def test_compat_inventory_keeps_custom_section_copy(tmp_path):
    root = tmp_path / "models"
    model = root / "publisher"
    q4 = _touch(model / "model-Q4_K_M.gguf")
    q8 = _touch(model / "model-Q8_0.gguf")

    rows = models_route.collect_local_models(
        root,
        custom_folders = [{"path": str(model)}],
        sources = _empty_sources(tmp_path),
    )

    assert {(row.source, Path(row.path)) for row in rows} == {
        ("models_dir", model),
        ("custom", q4),
        ("custom", q8),
    }


def test_compat_inventory_does_not_cross_dedupe_default_sources(tmp_path):
    root = tmp_path / "models"
    holder = root / "publisher"
    first = _touch(holder / "model-a-Q4_K_M.gguf")
    second = _touch(holder / "model-b-Q4_K_M.gguf")
    empty = _empty_sources(tmp_path)
    sources = empty._replace(lm_dirs = (root,))

    rows = models_route.collect_local_models(
        root,
        custom_folders = [],
        sources = sources,
    )

    assert {(row.source, Path(row.path)) for row in rows} == {
        ("models_dir", holder),
        ("lmstudio", first),
        ("lmstudio", second),
    }


def test_compat_inventory_lists_hermes_downloads(tmp_path):
    models_root = tmp_path / "models"
    models_root.mkdir()
    hermes = tmp_path / ".hermes" / "models"
    weight = _touch(hermes / "Qwen3.8-27B-UD-Q4_K_M.gguf")
    _touch(hermes / "assets" / "mmproj-Qwen3.8-27B-BF16.gguf")
    empty = _empty_sources(tmp_path)
    sources = empty._replace(hermes_dirs = (hermes,))

    rows = models_route.collect_local_models(
        models_root,
        custom_folders = [],
        sources = sources,
    )

    assert {(row.source, Path(row.path)) for row in rows} == {("hermes", weight)}
    assert rows[0].display_name == "Qwen3.8-27B-UD-Q4_K_M"


def test_local_route_returns_hermes_downloads(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    hermes = tmp_path / ".hermes" / "models"
    weight = _touch(hermes / "Qwen3.8-27B-UD-Q4_K_M.gguf")
    sources = _empty_sources(tmp_path)._replace(hf_cache_dir = models_dir, hermes_dirs = (hermes,))
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr(local_inventory, "list_scan_folders", lambda: [])

    response = asyncio.run(
        models_route.list_local_models(models_dir = str(models_dir), current_subject = "test")
    )

    assert response.hermes_dirs == [str(hermes)]
    assert [(m.source, Path(m.path)) for m in response.models] == [("hermes", weight)]


def test_local_route_reads_its_sources_once_off_the_event_loop(monkeypatch, tmp_path):
    """The probes stat every source root, so a slow drive on the event loop stalls every request."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    sources = _empty_sources(tmp_path)._replace(hf_cache_dir = models_dir)
    on_loop: list[bool] = []

    def probe():
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return sources

    monkeypatch.setattr(local_inventory, "_local_inventory_sources", probe)
    monkeypatch.setattr(local_inventory, "list_scan_folders", lambda: [])

    asyncio.run(models_route.list_local_models(models_dir = str(models_dir), current_subject = "test"))

    assert on_loop == [False]


def test_compat_inventory_preserves_ollama_tags_sharing_a_blob(tmp_path):
    root = tmp_path / "ollama"
    digest = "sha256-model"
    _touch(root / "blobs" / digest)
    manifest = json.dumps(
        {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": digest.replace("-", ":", 1),
                }
            ]
        }
    )
    tags = root / "manifests" / "registry.ollama.ai" / "library" / "model"
    tags.mkdir(parents = True)
    for tag in ("latest", "v1"):
        (tags / tag).write_text(manifest, encoding = "utf-8")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}],
        sources = _empty_sources(tmp_path),
    )

    assert {row.model_id for row in rows} == {"ollama/model:latest", "ollama/model:v1"}


def test_compat_inventory_dedupes_lmstudio_publisher_named_ollama(tmp_path):
    root = tmp_path / "lmstudio"
    model = root / "ollama" / "foo"
    _touch(model / "foo-Q4_K_M.gguf")
    (model / "config.json").write_text("{}", encoding = "utf-8")

    rows = models_route.collect_local_models(
        tmp_path / "models",
        custom_folders = [{"path": str(root)}, {"path": str(model)}],
        sources = _empty_sources(tmp_path),
    )

    assert [Path(row.path) for row in rows] == [model]


def test_local_adapter_chat_capability_uses_its_local_base(tmp_path):
    from hub.services.models.common import _classify_local_path

    base = tmp_path / "whisper-base"
    _touch(base / "model.safetensors")
    (base / "config.json").write_text(
        '{"model_type":"whisper","architectures":["WhisperForConditionalGeneration"]}'
    )
    adapter = tmp_path / "whisper-adapter"
    _touch(adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))

    rows = _classify_local_path(adapter, "custom")

    assert len(rows) == 1
    assert rows[0].base_model == str(base)
    assert models_route._local_model_can_chat(rows[0]) is False


@pytest.mark.parametrize(
    "files, source, expected",
    [
        (("model-Q4_K_M.gguf",), "models_dir", ["gguf"]),
        (("mmproj-F16.gguf",), "models_dir", ["unknown"]),
        (("mmproj-F16.gguf", "model-Q4_K_M.gguf"), "models_dir", ["gguf"]),
        (("config.json", "model-Q4_K_M.gguf"), "models_dir", ["gguf"]),
        (("tokenizer.bin", "model-Q4_K_M.gguf"), "models_dir", ["gguf"]),
        (("pytorch_model.bin", "model-Q4_K_M.gguf"), "models_dir", ["gguf"]),
        (("config.json", "model.safetensors"), "models_dir", ["safetensors"]),
        (
            ("config.json", "model.safetensors", "model-Q4_K_M.gguf"),
            "models_dir",
            ["gguf", "safetensors"],
        ),
        (("UD-Q4_K_XL/model-00001-of-00002.gguf",), "hf_cache", ["gguf"]),
        (("mmproj/mmproj-F16.gguf",), "hf_cache", ["unknown"]),
        (("MTP/mtp-model-Q8_0.gguf",), "hf_cache", ["unknown"]),
    ],
)
def test_folder_rows_follow_their_weights(tmp_path, files, source, expected):
    folder = tmp_path / "model"
    for name in files:
        _touch(folder / name)
    from hub.services.models.common import _classify_local_path

    assert [row.model_format for row in _classify_local_path(folder, source)] == expected


def test_scan_models_dir_skips_an_mmproj_only_folder(tmp_path):
    _touch(tmp_path / "vision" / "mmproj-F16.gguf")
    _touch(tmp_path / "real" / "model-Q4_K_M.gguf")
    formats = {m.display_name: m.model_format for m in local_inventory._scan_models_dir(tmp_path)}
    assert formats == {"real": "gguf"}


def test_scan_models_dir_skips_standalone_mmproj_file(tmp_path):
    _touch(tmp_path / "mmproj-F16.gguf")
    _touch(tmp_path / "model-Q4_K_M.gguf")
    names = {m.display_name for m in local_inventory._scan_models_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_scan_lmstudio_dir_skips_standalone_mmproj_file(tmp_path):
    _touch(tmp_path / "mmproj-F16.gguf")
    _touch(tmp_path / "model-Q4_K_M.gguf")
    names = {m.display_name for m in local_inventory._scan_lmstudio_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_scan_lmstudio_dir_skips_mmproj_under_publisher(tmp_path):
    _touch(tmp_path / "Publisher" / "mmproj-F16.gguf")
    _touch(tmp_path / "Publisher" / "model-Q4_K_M.gguf")
    names = {m.display_name for m in local_inventory._scan_lmstudio_dir(tmp_path)}
    assert names == {"model-Q4_K_M"}


def test_scan_models_dir_classifies_gguf_with_config(tmp_path):
    root = tmp_path / "models"
    _touch(root / "gguf_repo" / "config.json")
    _touch(root / "gguf_repo" / "model-Q4_K_M.gguf")
    _touch(root / "st_repo" / "config.json")
    _touch(root / "st_repo" / "model.safetensors")
    _touch(root / "loose.gguf")

    fmt = {Path(m.path).name: m.model_format for m in local_inventory._scan_models_dir(root)}

    assert fmt == {"gguf_repo": "gguf", "st_repo": "safetensors", "loose.gguf": "gguf"}


def test_scan_models_dir_classifies_root_gguf_with_config(tmp_path):
    root = tmp_path / "SuffixlessRepo"
    _touch(root / "config.json")
    _touch(root / "model-Q4_K_M.gguf")

    [row] = local_inventory._scan_models_dir(root)

    assert row.path == str(root)
    assert row.model_format == "gguf"


def test_scan_models_dir_surfaces_diffusers_pipeline_folder(tmp_path):
    root = tmp_path / "models"
    pipe = root / "my-pipeline"
    _pipeline_manifest(pipe)
    _touch(pipe / "vae" / "diffusion_pytorch_model.safetensors")

    rows = {Path(m.path).name: m for m in local_inventory._scan_models_dir(root)}

    assert rows["my-pipeline"].model_format == "unknown"
    assert rows["my-pipeline"].artifact_kind == "diffusers_pipeline"


def test_scan_models_dir_surfaces_root_diffusers_pipeline(tmp_path):
    root = tmp_path / "my-local-pipeline"
    _pipeline_manifest(root)
    _touch(root / "vae" / "diffusion_pytorch_model.safetensors")

    rows = local_inventory._scan_models_dir(root)

    assert [r.path for r in rows] == [str(root)]
    assert rows[0].artifact_kind == "diffusers_pipeline"


def _bom(pipeline):
    manifest = pipeline / "model_index.json"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())


def _write_manifest(text):
    return lambda p: (p / "model_index.json").write_text(text)


@pytest.mark.parametrize(
    "manifests, mutate, expected",
    [
        (("model_index.json",), None, "diffusers_pipeline"),
        (("model_index.json",), _bom, "diffusers_pipeline"),
        (("modular_model_index.json",), None, "diffusers_modular_pipeline"),
        (("model_index.json", "modular_model_index.json"), None, "diffusers_dual_pipeline"),
        (
            ("model_index.json",),
            lambda p: (p / "transformer" / "diffusion_pytorch_model.safetensors").unlink(),
            "unknown",
        ),
    ]
    + [
        (("model_index.json",), _write_manifest(text), "unknown")
        for text in ["", "{", "[]", "{}", '{"_class_name":" "}', '{"_class_name":"P"}']
    ],
)
def test_hub_inventory_types_pipeline_roots_structurally(tmp_path, manifests, mutate, expected):
    from hub.services.models.common import _classify_local_path

    pipeline = tmp_path / "opaque-model"
    for name in manifests:
        _pipeline_manifest(pipeline, name)
    if mutate:
        mutate(pipeline)
    [row] = _classify_local_path(pipeline, "custom")
    assert (row.model_format, row.artifact_kind) == ("unknown", expected)


def test_hub_inventory_does_not_confuse_transformers_or_adapter_with_pipeline(tmp_path):
    from hub.services.models.common import _classify_local_path
    from hub.services.models.local_inventory import _scan_models_dir as scan_hub_models_dir

    transformer, adapter, loose = tmp_path / "transformer", tmp_path / "adapter", tmp_path / "loose"
    _pipeline_manifest(transformer)
    _touch(transformer / "model.safetensors")
    (transformer / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}')
    _pipeline_manifest(adapter)
    _touch(adapter / "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text("{}")
    _touch(loose / "qwen_3_4b.safetensors")
    _touch(loose / "diffusion_pytorch_model-00001-of-00002.bf16.safetensors")

    assert _classify_local_path(transformer, "custom")[0].artifact_kind == "transformers_model"
    assert _classify_local_path(adapter, "custom")[0].artifact_kind == "adapter"
    assert scan_hub_models_dir(loose) == []


def test_local_pipeline_completeness_checks_configs_and_every_indexed_shard(tmp_path):
    _pipeline_manifest(tmp_path)
    component = tmp_path / "transformer"
    (component / "diffusion_pytorch_model.safetensors").unlink()
    (component / "config.json").write_text("{")
    index = component / "diffusion_pytorch_model.safetensors.index.json"
    _weight_map(index, "weights-00001-of-00001.safetensors")
    assert _complete(tmp_path) is False
    (component / "config.json").write_text("{}")
    assert _complete(tmp_path) is False
    _touch(component / "weights-00001-of-00001.safetensors")
    assert _complete(tmp_path) is True
    _weight_map(index, "..\\outside.safetensors")
    assert _complete(tmp_path) is False
    index.unlink()
    (component / "weights-00001-of-00001.safetensors").unlink()
    _touch(component / "diffusion_pytorch_model.safetensors")
    _weight_map(component / "diffusion_pytorch_model.bin.index.json", "model-00001-of-00001.bin")
    assert _complete(tmp_path) is True
    (component / "diffusion_pytorch_model.safetensors").unlink()
    assert _complete(tmp_path) is False


def test_local_pipeline_completeness_rejects_variant_only_weights(tmp_path):
    _pipeline_manifest(tmp_path)
    component = tmp_path / "transformer"
    (component / "diffusion_pytorch_model.safetensors").unlink()
    shard = "diffusion_pytorch_model.fp16-00001-of-00001.safetensors"
    _touch(component / shard)
    _weight_map(component / "diffusion_pytorch_model.safetensors.index.fp16.json", shard)
    assert _complete(tmp_path) is False
    _touch(component / "diffusion_pytorch_model.safetensors")
    assert _complete(tmp_path) is True


def test_local_pipeline_completeness_accepts_a_tokenizer_config_over_the_manifest_cap(tmp_path):
    _pipeline_manifest(tmp_path, tokenizer = ["transformers", "GemmaTokenizerFast"])
    _write(
        tmp_path / "tokenizer" / "tokenizer_config.json",
        json.dumps({"added_tokens_decoder": "x" * (2 << 20)}),
    )
    _touch(tmp_path / "tokenizer" / "tokenizer.json")
    assert _complete(tmp_path) is True


def test_local_pipeline_completeness_ignores_a_diffusers_index_on_a_transformers_encoder(tmp_path):
    _pipeline_manifest(tmp_path, text_encoder = ["transformers", "Gemma3ForConditionalGeneration"])
    encoder = tmp_path / "text_encoder"
    _write(encoder / "config.json")
    _touch(encoder / "model-00001-of-00001.safetensors")
    for prefix in ("model", "diffusion_pytorch_model"):
        _weight_map(
            encoder / f"{prefix}.safetensors.index.json", f"{prefix}-00001-of-00001.safetensors"
        )
    assert _complete(tmp_path) is True
    (encoder / "model-00001-of-00001.safetensors").unlink()
    assert _complete(tmp_path) is False
    _touch(encoder / "model.safetensors")
    assert _complete(tmp_path) is True


@pytest.mark.parametrize(
    "extra, expected",
    [
        ({}, True),
        ({"text_encoder": [None, None]}, True),
        ({"text_encoder_select_layers": [2, 5, 8, 11]}, True),
        ({"text_encoder": ["transformers", "CLIPTextModel"]}, False),
    ],
)
def test_local_pipeline_optional_component_presence(tmp_path, extra, expected):
    _pipeline_manifest(tmp_path, **extra)
    assert _complete(tmp_path) is expected


def test_local_pipeline_completeness_allows_hidream_caller_supplied_encoder(tmp_path):
    encoders = {
        "text_encoder_4": ["transformers", "LlamaForCausalLM"],
        "tokenizer_4": ["transformers", "PreTrainedTokenizerFast"],
    }
    _pipeline_manifest(tmp_path, _class_name = "HiDreamImagePipeline", **encoders)
    assert _complete(tmp_path) is True
    (tmp_path / "text_encoder_4").mkdir()
    assert _complete(tmp_path) is False
    (tmp_path / "text_encoder_4").rmdir()
    _pipeline_manifest(tmp_path, **encoders)
    assert _complete(tmp_path) is False


def test_local_pipeline_completeness_honors_exclusions_and_config_only_models(tmp_path):
    _index(tmp_path, scheduler = ["diffusers", "FlowMatchEulerDiscreteScheduler"], **_TRANSFORMER)
    _write(tmp_path / "scheduler" / "scheduler_config.json")
    assert _complete(tmp_path) is False
    assert _complete(tmp_path, excluded_components = ("transformer",)) is True
    _index(tmp_path, **_TRANSFORMER)
    assert _complete(tmp_path, excluded_components = ("transformer",)) is False

    _index(tmp_path, unet = ["diffusers", "UNet2DConditionModel"])
    _write(tmp_path / "unet" / "config.json")
    assert _complete(tmp_path) is False
    assert _complete(tmp_path, config_only_model_components = True) is True


@pytest.mark.parametrize(
    "name, class_name, files, expected",
    [
        ("scheduler", "FlowMatchEulerDiscreteScheduler", "README.md", False),
        ("scheduler", "FlowMatchEulerDiscreteScheduler", "scheduler_config.json", True),
        ("tokenizer", "Qwen2Tokenizer", "tokenizer_config.json", False),
        ("tokenizer", "Qwen2Tokenizer", "tokenizer_config.json tokenizer.json", True),
        ("tokenizer_2", "ByT5Tokenizer", "tokenizer_config.json", True),
        ("guider", "ClassifierFreeGuidance", "guider_config.json", True),
        ("processor", "Qwen3VLProcessor", "preprocessor_config.json", False),
        ("processor", "Qwen3VLProcessor", "preprocessor_config.json tokenizer.json", True),
    ],
)
def test_local_pipeline_completeness_validates_metadata_component_contracts(
    tmp_path, name, class_name, files, expected
):
    library = "diffusers" if name in ("scheduler", "guider") else "transformers"
    _index(tmp_path, **{name: [library, class_name]})
    for file in files.split():
        _write(tmp_path / name / file)
    assert _complete(tmp_path) is expected


@pytest.mark.parametrize("filename", ["model_index.json", "modular_model_index.json"])
@pytest.mark.parametrize(
    "class_name, asset",
    [("CustomModel", "custom_weights.safetensors"), ("CustomScheduler", "custom_schedule.json")],
)
def test_local_pipeline_preserves_custom_component_contracts(tmp_path, filename, class_name, asset):
    from hub.services.models.common import _diffusers_pipeline_artifact_kind

    spec = {"component": ["local_extensions", class_name], "optional_component": [None, None]}
    _index(tmp_path, filename, **spec)
    component = tmp_path / "component"
    assert not _complete(tmp_path, filename)
    component.mkdir()
    assert not _complete(tmp_path, filename)
    (component / asset).write_bytes(b"")
    assert not _complete(tmp_path, filename)
    (component / asset).write_text("{}")
    assert _complete(tmp_path, filename)
    assert _diffusers_pipeline_artifact_kind(tmp_path) == (
        "diffusers_pipeline" if filename == "model_index.json" else "diffusers_modular_pipeline"
    )


def test_local_pipeline_completeness_honors_modular_external_component_sources(tmp_path):
    pipeline, modular = tmp_path / "external-modular", "modular_model_index.json"

    def manifest(source, component = ("transformers", "Qwen2Tokenizer", "tokenizer")):
        spec = {"pretrained_model_name_or_path": source, "subfolder": component[2]}
        _index(pipeline, modular, tokenizer = [*component[:2], spec])
        return _complete(pipeline, modular)

    assert manifest("Org/components") is True
    assert manifest("./missing") is False
    source = tmp_path / "component-source"
    _write(source / "tokenizer" / "tokenizer_config.json")
    _write(source / "tokenizer" / "tokenizer.json")
    assert manifest(str(source)) is True
    assert manifest(str(source), ("transformers", "Qwen2Tokenizer", "../outside")) is False
    (source / "custom_weights.safetensors").write_bytes(b"weights")
    assert manifest(str(source), ("local_extensions", "CustomModel", None)) is True


def test_scan_models_dir_surfaces_root_single_file_checkpoint(tmp_path):
    root = tmp_path / "qwen-image-2509"
    _touch(root / "qwen-image-2509.safetensors")

    rows = local_inventory._scan_models_dir(root)

    assert [r.path for r in rows] == [str(root / "qwen-image-2509.safetensors")]
    assert rows[0].artifact_kind == "single_file_checkpoint"


def test_scan_models_dir_root_weights_do_not_hide_child_models(tmp_path):
    root = tmp_path / "models"
    _touch(root / "stray.safetensors")
    _touch(root / "llama" / "config.json")
    _touch(root / "llama" / "model.safetensors")

    assert [Path(r.path).name for r in local_inventory._scan_models_dir(root)] == ["llama"]


from models.models import LocalModelInfo  # noqa: E402
from hub.services.models import catalog_classification as classification


def _local(
    path,
    *,
    model_format = None,
    model_id = None,
    display_name = "m",
    id = "m",
):
    return LocalModelInfo(
        id = id,
        display_name = display_name,
        path = str(path),
        source = "models_dir",
        model_id = model_id,
        model_format = model_format,
    )


def test_windows_cloud_recall_attributes_are_not_local():
    from utils.paths.path_utils import file_contents_available_locally

    # Synology online-only files: 0x400020 via os.stat, 0x401620 via enumeration.
    for attributes in (
        0x00400020,
        0x00401620,
        0x00001000,
        0x00040000,
        0x00400000,
    ):
        assert not file_contents_available_locally(
            "unused", types.SimpleNamespace(st_file_attributes = attributes)
        )

    # Hydrated Synology files stay reparse points (0x420); UNPINNED is not absence.
    for attributes in (0x00000420, 0x00100000):
        assert file_contents_available_locally(
            "unused", types.SimpleNamespace(st_file_attributes = attributes)
        )


def test_local_gguf_task_reads_present_header(tmp_path, monkeypatch):
    """Fully present files retain architecture-based task detection."""
    gguf = _touch(tmp_path / "generic-Q4_K_M.gguf")
    reads = []
    monkeypatch.setattr(classification, "file_contents_available_locally", lambda _path: True)
    monkeypatch.setattr(
        classification,
        "_gguf_architecture",
        lambda path: reads.append(path) or "llama",
    )

    model = _local(
        gguf,
        model_format = "gguf",
        display_name = "generic",
        id = "generic-file-id",
    )

    assert models_route._local_model_task(model) == "text-generation"
    assert reads == [str(gguf)]


def test_local_gguf_task_skips_online_only_contents(tmp_path, monkeypatch):
    """Cloud placeholders stay discoverable by name without opening their data."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("local GGUF listing touched placeholder contents")

    monkeypatch.setattr(classification, "file_contents_available_locally", lambda _path: False)
    monkeypatch.setattr(classification, "_gguf_architecture", forbidden)

    gguf = _touch(tmp_path / "generic-Q4_K_M.gguf")
    model = _local(
        gguf,
        model_format = "gguf",
        display_name = "generic",
        id = "generic-file-id",
    )

    assert models_route._local_model_task(model) is None


def test_local_classification_never_opens_an_online_only_gguf(tmp_path, monkeypatch):
    """Online-only GGUF placeholders must never be opened, including by the audio-type fallback probe."""
    from utils.models import gguf_metadata

    single = _touch(tmp_path / "single" / "generic-Q4_K_M.gguf")
    folder = tmp_path / "generic-GGUF"
    for quant in ("Q4_K_M", "Q8_0"):
        _touch(folder / f"generic-{quant}.gguf")

    opened: list[str] = []
    monkeypatch.setattr(
        classification, "file_contents_available_locally", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        gguf_metadata,
        "read_gguf_architecture",
        lambda path: opened.append(path) or "llama",
    )

    for path in (single, folder):
        model = _local(path, model_format = "gguf", display_name = "generic", id = str(path))
        assert classification._local_model_classification(model) == (None, None)
    assert opened == []


def test_local_classification_probes_the_hf_cache_snapshot(tmp_path):
    repo = tmp_path / "models--unsloth--csm-1b"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "0": {"content": "<|AUDIO|>"},
                    "1": {"content": "<|audio_eos|>"},
                }
            }
        ),
        encoding = "utf-8",
    )
    model = LocalModelInfo(
        id = "unsloth/csm-1b",
        display_name = "csm-1b",
        path = str(repo),
        source = "hf_cache",
        model_id = "unsloth/csm-1b",
    )

    assert classification._local_model_classification(model) == ("text-to-speech", "csm")


def test_local_task_probes_the_hf_cache_pipeline_snapshot(tmp_path):
    repo = tmp_path / "models--hf-internal-testing--tiny-sdxl-pipe"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}),
        encoding = "utf-8",
    )
    model = LocalModelInfo(
        id = "hf-internal-testing/tiny-sdxl-pipe",
        display_name = "tiny-sdxl-pipe",
        path = str(repo),
        source = "hf_cache",
        model_id = "hf-internal-testing/tiny-sdxl-pipe",
    )

    assert models_route._local_model_task(model) == "text-to-image"


def test_an_unhydrated_denoiser_keeps_the_picker_that_would_hydrate_it(tmp_path, monkeypatch):
    """Images and Video filter On Device rows by exact task, so an unclassified denoiser is not
    reachable from the one page whose pick would pull it down; it lists in Chat instead."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("placeholder contents were read to classify it")

    monkeypatch.setattr(
        classification, "file_contents_available_locally", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(classification, "_gguf_architecture", forbidden)

    for name, expected in (
        ("flux1-dev-Q4_K_M.gguf", "text-to-image"),
        ("z-image-turbo-Q4_K_M.gguf", "text-to-image"),
        ("ltx-video-2b-Q4_K_M.gguf", "text-to-video"),
        ("qwen3-4b-instruct-Q4_K_M.gguf", None),
    ):
        gguf = _touch(tmp_path / name)
        model = _local(gguf, model_format = "gguf", display_name = name, id = name)
        assert models_route._local_model_task(model) == expected, name


def test_an_ancestor_directory_does_not_name_an_unhydrated_gguf(tmp_path, monkeypatch):
    """An ancestor folder named after a model family must not classify an unhydrated GGUF placeholder."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("placeholder contents were read to classify it")

    monkeypatch.setattr(
        classification, "file_contents_available_locally", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(classification, "_gguf_architecture", forbidden)

    for relative in (
        "FLUX.1-dev-GGUF/extra/qwen3-4b/qwen3-Q4_K_M.gguf",
        "ltx-2/qwen3-4b/qwen3-Q4_K_M.gguf",
    ):
        gguf = _touch(tmp_path / relative)
        model = _local(gguf, model_format = "gguf", display_name = gguf.name, id = str(gguf))
        assert models_route._local_model_task(model) is None, relative

    folder = tmp_path / "FLUX.1-dev-GGUF"
    _touch(folder / "diffusion_model-Q4_K_M.gguf")
    row = _local(folder, model_format = "gguf", display_name = folder.name, id = str(folder))
    assert models_route._local_model_task(row) == "text-to-image"


def test_local_task_tags_family_named_pipeline_dir(tmp_path):
    d = tmp_path / "flux-pipeline"
    _pipeline_manifest(d)
    _touch(d / "unet" / "diffusion_pytorch_model.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "black-forest-labs/FLUX.1-dev"))
        == "text-to-image"
    )


def test_local_task_none_for_familyless_pipeline_dir(tmp_path):
    d = tmp_path / "my-local-pipeline"
    _pipeline_manifest(d)
    _touch(d / "unet" / "diffusion_pytorch_model.safetensors")
    assert models_route._local_is_diffusers(_local(d)) is True
    assert models_route._local_model_task(_local(d)) is None


def test_local_task_tags_diffusers_by_family_id(tmp_path):
    d = tmp_path / "flux-checkpoint"
    _touch(d / "flux1-dev.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "black-forest-labs/FLUX.1-dev"))
        == "text-to-image"
    )


def test_local_task_none_for_plain_llm(tmp_path):
    d = tmp_path / "llama"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert models_route._local_model_task(_local(d, model_id = "meta-llama/Llama-3.1-8B")) is None


def test_local_task_tags_minimax_music3_modular_pipeline(tmp_path):
    d = tmp_path / "music3"
    d.mkdir()
    _pipeline_manifest(
        d,
        "modular_model_index.json",
        _class_name = "MiniMaxMusic3ModularPipeline",
        _blocks_class_name = "MiniMaxMusic3Blocks",
    )

    assert models_route._local_model_task(_local(d, model_format = "safetensors")) == (
        "text-to-speech"
    )


def test_compat_local_inventory_preserves_minimax_music3_audio_type(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    d = models_dir / "music3"
    d.mkdir(parents = True)
    _pipeline_manifest(
        d,
        "modular_model_index.json",
        _class_name = "MiniMaxMusic3ModularPipeline",
        _blocks_class_name = "MiniMaxMusic3Blocks",
    )
    sources = _empty_sources(tmp_path)._replace(hf_cache_dir = models_dir)
    row = local_inventory._classify_local_path(d, "models_dir")

    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr(
        local_inventory, "_collect_models_from_default_sources", lambda *_a, **_k: row
    )
    monkeypatch.setattr(local_inventory, "list_scan_folders", lambda: [])

    response = asyncio.run(
        models_route.list_local_models(models_dir = str(models_dir), current_subject = "test")
    )

    assert response.models[0].task == "text-to-speech"
    assert response.models[0].audio_type == "minimax_music3"


def test_local_task_tags_video_pipeline_dir(tmp_path):
    d = tmp_path / "wan-local"
    _touch(d / "model_index.json")
    _touch(d / "transformer" / "diffusion_pytorch_model.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
        == models_route._VIDEO_GEN_TASK
    )


def test_local_task_tags_video_single_file_checkpoint(tmp_path):
    d = tmp_path / "ltx-loose"
    _touch(d / "ltx-2.safetensors")
    assert (
        models_route._local_model_task(_local(d, model_id = "Lightricks/LTX-2"))
        == models_route._VIDEO_GEN_TASK
    )


def test_local_task_tags_single_file_by_checkpoint_filename(tmp_path):
    d = tmp_path / "downloads"
    _touch(d / "qwen-image-2509.safetensors")
    m = _local(d, id = str(d), display_name = "downloads")
    assert models_route._local_is_diffusers(m) is True
    assert models_route._local_model_task(m) == "text-to-image"


def test_local_task_tags_video_single_file_by_checkpoint_filename(tmp_path):
    d = tmp_path / "clips"
    _touch(d / "ltx-2.3-distilled.safetensors")
    m = _local(d, id = str(d), display_name = "clips")
    assert models_route._local_model_task(m) == models_route._VIDEO_GEN_TASK


def test_local_task_ignores_family_token_in_parent_path(tmp_path):
    # Family detection is scoped to the leaf name, not parent dirs in model.id.
    d = tmp_path / "misc"
    _touch(d / "unrelated.safetensors")
    m = _local(d, id = "/models/qwen-image/misc", display_name = "misc")
    assert models_route._local_is_diffusers(m) is False
    assert models_route._local_model_task(m) is None
    d2 = tmp_path / "z-image-turbo"
    _touch(d2 / "model.safetensors")
    m2 = _local(d2, id = str(d2), display_name = "z-image-turbo")
    assert models_route._local_is_diffusers(m2) is True


def test_a_modular_pipeline_root_counts_as_a_pipeline_index(tmp_path):
    """A modular_model_index.json alone marks a pipeline root, else its components are offered as models."""
    from hub.services.models.common import _is_diffusers_pipeline_dir

    modular = tmp_path / "modular"
    (modular / "transformer").mkdir(parents = True)
    _pipeline_manifest(modular, "modular_model_index.json")
    assert _is_diffusers_pipeline_dir(modular) is True
    assert (
        models_route._local_is_diffusers(_local(modular, display_name = "opaque", id = str(modular)))
        is True
    )

    conventional = tmp_path / "conventional"
    conventional.mkdir()
    _pipeline_manifest(conventional)
    assert _is_diffusers_pipeline_dir(conventional) is True

    neither = tmp_path / "neither"
    neither.mkdir()
    assert _is_diffusers_pipeline_dir(neither) is False


def test_a_single_file_video_repo_is_flagged_diffusers(monkeypatch):
    """Single-file video repos need detect_video_family too, or their weights reach the text chat picker."""
    from types import SimpleNamespace

    from core.inference.video_families import detect_video_family

    repo_id = "Lightricks/LTX-Video"
    assert detect_video_family(repo_id) is not None, "fixture assumes a known video family"

    info = SimpleNamespace(repo_id = repo_id, repo_path = "/nonexistent")
    assert classification._repo_is_diffusers(info) is True
    chat = SimpleNamespace(repo_id = "unsloth/Qwen3-0.6B", repo_path = "/nonexistent")
    assert classification._repo_is_diffusers(chat) is False


def test_adapter_base_is_found_in_the_cache_root_holding_the_adapter(tmp_path):
    """Adapter bases are probed in the adapter's own cache root, else the adapter stays chat-capable."""
    import json

    from hub.services.models.common import _base_transformers_can_chat, _hub_cache_root_of

    root = tmp_path / "legacy_hub"
    base_snapshot = root / "models--Org--WhisperBase" / "snapshots" / ("b" * 40)
    base_snapshot.mkdir(parents = True)
    (base_snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
            }
        )
    )
    (root / "models--Org--WhisperBase" / "refs").mkdir(parents = True)
    (root / "models--Org--WhisperBase" / "refs" / "main").write_text("b" * 40)

    adapter_snapshot = root / "models--Org--SpeechLora" / "snapshots" / ("a" * 40)
    adapter_snapshot.mkdir(parents = True)

    assert _hub_cache_root_of(adapter_snapshot) == root
    assert _base_transformers_can_chat("Org/WhisperBase", None, adapter_snapshot) is False


def _scan_hub_only(
    monkeypatch,
    tmp_path,
    folders = (),
):
    sources = _empty_sources(tmp_path)._replace(hf_cache_dir = tmp_path / "hub")
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr(
        "storage.studio_db.list_scan_folders", lambda: [{"path": str(f)} for f in folders]
    )
    monkeypatch.chdir(tmp_path)
    return models_route.collect_local_models(Path("models").resolve())


def _hf_snapshot(tmp_path: Path, repo_id: str) -> Path:
    repo = tmp_path / "hub" / f"models--{repo_id.replace('/', '--')}"
    _write(repo / "refs" / "main", "b" * 40)
    return repo / "snapshots" / ("b" * 40)


def _flux_checkpoint(path: Path) -> Path:
    import struct

    header = json.dumps(
        {
            "double_blocks.0.img_attn.qkv.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            }
        }
    ).encode()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 16)
    return path


def test_a_copied_single_file_hf_checkpoint_is_whole_and_media_resolvable(monkeypatch, tmp_path):
    """Windows without symlinks copies files into snapshots/; a lone flux1-dev.safetensors loads as single_file,
    though the Model Hub lists the unrecognized layout as partial."""
    from core.inference import media_model_index

    checkpoint = _flux_checkpoint(
        _hf_snapshot(tmp_path, "Comfy-Org/flux1-dev") / "flux1-dev.safetensors"
    )

    (row,) = _scan_hub_only(monkeypatch, tmp_path)
    assert (row.model_id, row.partial) == ("Comfy-Org/flux1-dev", True)
    pick = media_model_index._build_index("text-to-image")["comfy-org/flux1-dev"]
    assert Path(pick.model_path) == checkpoint.parent


@pytest.mark.parametrize(
    "text_encoder, weight_suffix",
    [("pytorch_model.bin", ".safetensors"), ("model.fp16.safetensors", ".fp16.safetensors")],
)
def test_a_whole_cached_pipeline_without_safetensors_defaults_is_media_resolvable(
    monkeypatch, tmp_path, text_encoder, weight_suffix
):
    """.bin text encoders, or the .fp16 twins an fp16 load stages instead of the defaults, are whole weights."""
    from core.inference import media_model_index

    snapshot = _hf_snapshot(tmp_path, "stabilityai/stable-diffusion-xl-base-1.0")
    components = {
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
    }
    _index(snapshot, _class_name = "StableDiffusionXLPipeline", **components)
    for name in ("unet", "vae"):
        _write(snapshot / name / "config.json")
        _touch(snapshot / name / f"diffusion_pytorch_model{weight_suffix}")
    _write(snapshot / "text_encoder" / "config.json")
    _touch(snapshot / "text_encoder" / text_encoder)
    for name in ("tokenizer_config.json", "vocab.json", "merges.txt"):
        _write(snapshot / "tokenizer" / name)
    _write(snapshot / "scheduler" / "scheduler_config.json")

    _scan_hub_only(monkeypatch, tmp_path)
    pick = media_model_index._build_index("text-to-image")[
        "stabilityai/stable-diffusion-xl-base-1.0"
    ]
    assert Path(pick.model_path) == snapshot


def test_a_custom_folder_keeps_publisher_qualified_ids(monkeypatch, tmp_path):
    """Two publishers' same-named models stay apart in /v1/models and the resolver, as on main."""
    from core.inference import local_model_resolver as resolver

    folder = tmp_path / "custom"
    for publisher in ("alice", "bob"):
        _touch(folder / publisher / "model" / "model-Q4_K_M.gguf")

    rows = _scan_hub_only(monkeypatch, tmp_path, folders = [folder])
    assert sorted(resolver._advertised_loader_id(row) for row in rows) == [
        "alice/model",
        "bob/model",
    ]
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    index = resolver._build_index()
    assert {"alice/model", "bob/model"} <= set(index)
    assert index["alice/model"] is not index["bob/model"]


def test_a_scan_folder_subfolder_holding_one_checkpoint_is_listed(monkeypatch, tmp_path):
    from core.inference import media_model_index

    library = tmp_path / "library"
    checkpoint = _flux_checkpoint(library / "Flux" / "flux1-dev.safetensors")

    (row,) = _scan_hub_only(monkeypatch, tmp_path, folders = [library])
    assert Path(row.path) == checkpoint.parent
    assert (
        Path(media_model_index._build_index("text-to-image")["flux"].model_path)
        == checkpoint.parent
    )


def test_an_empty_single_file_hf_checkpoint_is_not_offered(monkeypatch, tmp_path):
    from core.inference import media_model_index

    checkpoint = _hf_snapshot(tmp_path, "Comfy-Org/flux1-dev") / "flux1-dev.safetensors"
    checkpoint.parent.mkdir(parents = True)
    checkpoint.write_bytes(b"")

    (row,) = _scan_hub_only(monkeypatch, tmp_path)
    assert row.partial is True
    assert "comfy-org/flux1-dev" not in media_model_index._build_index("text-to-image")


def test_a_whole_artifact_mirror_pipeline_loads_but_is_no_hub_pick(monkeypatch, tmp_path):
    """A Qwen/Qwen-Image-2.1-Turbo pick downloads the pipeline into its unsloth/...-FP8 artifact mirror."""
    from core.inference import media_model_index

    snapshot = _hf_snapshot(tmp_path, "unsloth/Qwen-Image-2.1-Turbo-FP8")
    _index(
        snapshot,
        _class_name = "QwenImage21Pipeline",
        transformer = ["diffusers", "QwenImage21Transformer2DModel"],
        vae = ["diffusers", "AutoencoderKLQwenImage21"],
        scheduler = ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    )
    for name in ("transformer", "vae"):
        _write(snapshot / name / "config.json")
        _touch(snapshot / name / "diffusion_pytorch_model.safetensors")
    _write(snapshot / "scheduler" / "scheduler_config.json")

    _scan_hub_only(monkeypatch, tmp_path)
    pick = media_model_index._build_index("text-to-image")["unsloth/qwen-image-2.1-turbo-fp8"]
    assert Path(pick.model_path) == snapshot
    (hub_row,) = local_inventory._scan_hf_cache(tmp_path / "hub")
    assert (hub_row.partial, hub_row.companion_prefetch) == (True, True)


@pytest.mark.parametrize("name", ["flux1-dev.safetensors", "model-00001-of-00003.safetensors"])
def test_a_scan_folder_subfolder_with_an_empty_or_numbered_checkpoint_is_not_listed(
    monkeypatch, tmp_path, name
):
    library = tmp_path / "library"
    checkpoint = library / "Flux" / name
    if name.startswith("flux1"):
        checkpoint.parent.mkdir(parents = True)
        checkpoint.write_bytes(b"")
    else:
        _flux_checkpoint(checkpoint)

    assert _scan_hub_only(monkeypatch, tmp_path, folders = [library]) == []


def _two_revision_gguf_repo(
    hub: Path,
    newer_files = (),
    quant = "Model-Q4_K_M.gguf",
) -> None:
    import os
    import time

    repo = hub / "models--org--Model-GGUF"
    _write(repo / "refs" / "main", "b" * 40)
    older = repo / "snapshots" / ("a" * 40)
    _touch(older / quant)
    newer = repo / "snapshots" / ("b" * 40)
    newer.mkdir(parents = True)
    for name in newer_files:
        _touch(newer / name)
    now = time.time()
    os.utime(older, (now - 3600, now - 3600))
    os.utime(newer, (now, now))


@pytest.mark.parametrize("quant", ["Model-Q4_K_M.gguf", "UD-Q4_K_XL/Model-UD-Q4_K_XL.gguf"])
def test_a_custom_cache_mtp_only_revision_keeps_the_older_quant_resolvable(
    monkeypatch, tmp_path, quant
):
    """A newer revision holding only an MTP drafter must not hide the older revision's quant from the resolver."""
    from core.inference import local_model_resolver as resolver

    folder = tmp_path / "custom"
    _two_revision_gguf_repo(folder / "hub", ("MTP/mtp-model-Q8_0.gguf",), quant)

    rows = _scan_hub_only(monkeypatch, tmp_path, folders = [folder])
    assert len(rows) == 1
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    assert "org/model-gguf" in resolver._build_index()


@pytest.mark.parametrize(
    "real, alias, link",
    [
        ("real", "alias-a", ("alias-a", "real")),
        ("pub/real", "pub-alias/real", ("pub-alias", "pub")),
    ],
)
def test_a_folded_symlink_alias_still_resolves_by_its_own_name(
    monkeypatch, tmp_path, real, alias, link
):
    """The listing keeps the real folder; requests and per-model settings keyed on the alias still resolve."""
    from core.inference import local_model_resolver as resolver

    folder = tmp_path / "custom"
    _touch(folder / real / "model-Q4_K_M.gguf")
    (folder / link[0]).symlink_to(folder / link[1], target_is_directory = True)

    rows = _scan_hub_only(monkeypatch, tmp_path, folders = [folder])
    assert [Path(row.path) for row in rows] == [folder / real]
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    index = resolver._build_index()
    assert {str(folder / alias).lower(), Path(alias).name} <= set(index)
    assert resolver.local_gguf_pinned_variant(f"{folder / alias}:Q4_K_M") == "Q4_K_M"


def test_a_folded_copy_in_a_second_lm_studio_root_resolves_by_its_path(monkeypatch, tmp_path):
    """Two LM Studio roots holding pub/model list once; the second copy's path and quant pin still resolve."""
    from core.inference import local_model_resolver as resolver

    first, second = tmp_path / "lms-a", tmp_path / "lms-b"
    _touch(first / "pub" / "model" / "model-Q4_K_M.gguf")
    _touch(second / "pub" / "model" / "model-Q8_0.gguf")
    sources = _empty_sources(tmp_path)._replace(lm_dirs = (first, second))
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))

    assert len(models_route.collect_local_models(Path("models").resolve())) == 1
    assert str(second / "pub" / "model").lower() in resolver._build_index()
    assert resolver.local_gguf_pinned_variant(f"{second / 'pub' / 'model'}:Q8_0") == "Q8_0"


def test_a_failing_optional_app_root_does_not_empty_the_scan(monkeypatch, tmp_path):
    """An untraversable OMLX_BASE_PATH raises PermissionError from Path.is_file (Python < 3.14)."""
    from core.inference import local_model_resolver as resolver

    def _denied():
        raise PermissionError(13, "Permission denied", "/omlx/settings.json")

    folder = tmp_path / "custom"
    _touch(folder / "publisher" / "model-Q4_K_M.gguf")
    monkeypatch.setattr(local_inventory, "omlx_model_dirs", _denied)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(folder)}])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(local_inventory, "_resolve_hf_cache_dir", lambda: tmp_path / "hub")
    for name in ("legacy_hf_cache_dir", "hf_default_cache_dir"):
        monkeypatch.setattr(local_inventory, name, lambda: tmp_path / "empty")
    for name in ("lmstudio_model_dirs", "ollama_model_dirs", "hermes_model_dirs"):
        monkeypatch.setattr(local_inventory, name, lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [])

    assert local_inventory._local_inventory_sources().omlx_dirs == ()
    assert str(folder / "publisher").lower() in resolver._build_index()


def test_a_native_row_folded_into_its_scan_folder_keeps_the_publisher_id(monkeypatch, tmp_path):
    """An LM Studio model whose folder is also a scan folder lists as the custom row, which has no publisher id."""
    from core.inference import local_model_resolver as resolver

    lms = tmp_path / "lms"
    model = lms / "pub" / "model"
    _write(
        model / "config.json",
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
    )
    _touch(model / "model.safetensors")
    sources = _empty_sources(tmp_path)._replace(lm_dirs = (lms,))
    monkeypatch.setattr(local_inventory, "_local_inventory_sources", lambda: sources)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(model)}])
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))

    assert "pub/model" in resolver._build_index()
