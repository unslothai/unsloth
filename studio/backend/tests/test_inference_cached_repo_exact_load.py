# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test which repo id a chat load is handed, and when Unsloth's mapper picks it."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_STUBBED: list[str] = []


def _stub_if_missing(
    name,
    attrs = (),
    named_spec = False,
):
    """Stub dependencies unavailable in backend CI so inference can be imported."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
        pass
    _STUBBED.append(name)
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None) if named_spec else None
    module.__version__ = "0.0.0"
    module.__getattr__ = lambda _attr: MagicMock()
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


for _torchao in (
    "torchao",
    "torchao.prototype",
    "torchao.prototype.safetensors",
    "torchao.prototype.safetensors.safetensors_support",
    "torchao.prototype.safetensors.safetensors_utils",
    "torchao.quantization",
    "torchao.dtypes",
    "torchao.float8",
    "torchao.utils",
):
    _stub_if_missing(_torchao, named_spec = True)

_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("unsloth_zoo")
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.inference.inference import _exact_model_name_for_load  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


UPSTREAM = "Qwen/Qwen3-VL-4B-Instruct"
PREQUANT = "unsloth/qwen3-vl-4b-instruct-unsloth-bnb-4bit"
UNSLOTH_16BIT = "unsloth/Qwen3-VL-4B-Instruct"


@pytest.fixture
def mapper(monkeypatch):
    """Stub the mapper with one model."""
    loader_utils = types.ModuleType("unsloth.models.loader_utils")
    loader_utils.FLOAT_TO_INT_MAPPER = {UPSTREAM.lower(): PREQUANT, UNSLOTH_16BIT.lower(): PREQUANT}
    loader_utils.MAP_TO_UNSLOTH_16bit = {UPSTREAM.lower(): UNSLOTH_16BIT}
    loader_utils.BAD_MAPPINGS = {}
    loader_utils.calls = []

    def get_model_name(model_name, load_in_4bit = True):
        loader_utils.calls.append(model_name)
        table = (
            loader_utils.FLOAT_TO_INT_MAPPER if load_in_4bit else loader_utils.MAP_TO_UNSLOTH_16bit
        )
        return table.get(model_name.lower(), model_name)

    loader_utils.get_model_name = get_model_name
    loader = types.ModuleType("unsloth.models.loader")
    loader.ALLOW_BITSANDBYTES = True
    loader.ALLOW_PREQUANTIZED_MODELS = True
    loader.USE_MODELSCOPE = False
    loader._strip_unsloth_bnb_4bit_suffix = lambda name: name.removesuffix("-unsloth-bnb-4bit")
    loader_utils.loader = loader
    models = types.ModuleType("unsloth.models")
    models.loader_utils = loader_utils
    models.loader = loader
    unsloth = types.ModuleType("unsloth")
    unsloth.models = models
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.models", models)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", loader_utils)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader", loader)
    return loader_utils


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    import utils.hf_cache_settings as hf_cache_settings
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(tmp_path))
    return tmp_path


def _cache_repo(
    hub_cache: Path,
    repo_id: str,
    *,
    missing_shard: bool = False,
    config: dict | None = None,
) -> None:
    repo_dir = hub_cache / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / "0123abcd"
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("0123abcd")
    (snapshot / "config.json").write_text(json.dumps(config or {}))
    if not missing_shard:
        (snapshot / "model.safetensors").write_bytes(b"\0" * 64)
        return
    shards = [f"model-0000{i}-of-00002.safetensors" for i in (1, 2)]
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}})
    )
    (snapshot / shards[0]).write_bytes(b"\0" * 64)


def _config(path = UPSTREAM, **overrides):
    return types.SimpleNamespace(**{"is_local": False, "is_lora": False, "path": path, **overrides})


@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_cached_repo_loads_as_named_when_the_swap_target_is_missing(
    mapper, hub_cache, load_in_4bit
):
    _cache_repo(hub_cache, UPSTREAM)

    assert _exact_model_name_for_load(_config(), load_in_4bit) == UPSTREAM


@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_cached_swap_target_keeps_the_swap(mapper, hub_cache, load_in_4bit):
    _cache_repo(hub_cache, UPSTREAM)
    _cache_repo(hub_cache, PREQUANT if load_in_4bit else UNSLOTH_16BIT)

    assert _exact_model_name_for_load(_config(), load_in_4bit) is None


@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_a_differently_cased_swap_target_is_loaded_under_its_cached_spelling(
    mapper, hub_cache, load_in_4bit
):
    """The mapper lowercases its targets; the Hub, and Studio's own downloads, do not.

    Answering "cached, let the mapper have it" is not enough: the mapper hands the
    loader its own lowercase spelling, huggingface_hub keys the cache directory on the
    id verbatim (huggingface/huggingface_hub#3838), and the copy on disk is fetched a
    second time. Measured at 1356 MiB on Qwen3-1.7B with Xet dedup off.
    """
    target = (PREQUANT if load_in_4bit else UNSLOTH_16BIT).upper()
    _cache_repo(hub_cache, UPSTREAM)
    _cache_repo(hub_cache, target)

    assert _exact_model_name_for_load(_config(), load_in_4bit) == target


def test_a_differently_cased_target_short_a_shard_still_loads_as_named(mapper, hub_cache):
    # Matching the case is not enough on its own: the copy still has to be loadable.
    _cache_repo(hub_cache, UPSTREAM)
    _cache_repo(hub_cache, PREQUANT.upper(), missing_shard = True)

    assert _exact_model_name_for_load(_config(), True) == UPSTREAM


def test_without_bitsandbytes_the_16bit_target_decides(mapper, hub_cache):
    # The loader drops 4-bit here, so an unusable cached prequant does not count.
    mapper.loader.ALLOW_BITSANDBYTES = False
    _cache_repo(hub_cache, UPSTREAM)
    _cache_repo(hub_cache, PREQUANT)
    assert _exact_model_name_for_load(_config(), True) == UPSTREAM

    _cache_repo(hub_cache, UNSLOTH_16BIT)
    assert _exact_model_name_for_load(_config(), True) is None


def test_without_prequantized_models_the_stripped_target_decides(mapper, hub_cache):
    mapper.loader.ALLOW_PREQUANTIZED_MODELS = False
    _cache_repo(hub_cache, UPSTREAM)
    _cache_repo(hub_cache, PREQUANT)
    assert _exact_model_name_for_load(_config(), True) == UPSTREAM

    _cache_repo(hub_cache, PREQUANT.removesuffix("-unsloth-bnb-4bit"))
    assert _exact_model_name_for_load(_config(), True) is None


def test_modelscope_keeps_the_swap(mapper, hub_cache):
    mapper.loader.USE_MODELSCOPE = True
    _cache_repo(hub_cache, UPSTREAM)

    assert _exact_model_name_for_load(_config(), True) is None


def test_named_repo_short_a_shard_keeps_the_swap(mapper, hub_cache):
    _cache_repo(hub_cache, UPSTREAM, missing_shard = True)

    assert _exact_model_name_for_load(_config(), True) is None


def test_uncached_named_repo_keeps_the_swap(mapper, hub_cache):
    assert _exact_model_name_for_load(_config(), True) is None


def test_checkpoint_in_a_previous_cache_root_keeps_the_swap(
    mapper, hub_cache, tmp_path_factory, monkeypatch
):
    # The loader cannot reuse weights from a previous cache root.
    from utils.utils import hf_cache_snapshot_is_loadable

    previous = tmp_path_factory.mktemp("previous-hub-cache")
    _cache_repo(previous, UPSTREAM)
    monkeypatch.setenv("HF_HUB_CACHE", str(previous))

    assert hf_cache_snapshot_is_loadable(UPSTREAM) is True
    assert _exact_model_name_for_load(_config(), True) is None


def test_known_bad_repo_keeps_the_swap(mapper, hub_cache):
    _cache_repo(hub_cache, UPSTREAM)
    mapper.BAD_MAPPINGS[UPSTREAM.lower()] = UNSLOTH_16BIT

    assert _exact_model_name_for_load(_config(), True) is None


@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_quantized_checkpoint_keeps_the_swap(mapper, hub_cache, load_in_4bit):
    # MXFP4 weights cannot satisfy the requested 4-bit or 16-bit load as-is.
    _cache_repo(hub_cache, UPSTREAM, config = {"quantization_config": {"quant_method": "mxfp4"}})

    assert _exact_model_name_for_load(_config(), load_in_4bit) is None


def test_name_the_installed_tables_do_not_know_skips_the_mapper(mapper, hub_cache):
    # Leave remote mapper lookups to the loader.
    _cache_repo(hub_cache, "someone/custom-model")

    assert _exact_model_name_for_load(_config("someone/custom-model"), True) is None
    assert mapper.calls == []


@pytest.mark.parametrize("overrides", [{"is_local": True}, {"is_lora": True}])
def test_local_and_adapter_loads_are_untouched(mapper, hub_cache, overrides):
    _cache_repo(hub_cache, UPSTREAM)

    assert _exact_model_name_for_load(_config(**overrides), True) is None
    assert mapper.calls == []


def test_load_model_hands_the_verdict_to_both_loaders():
    """Dropping either keyword argument restores the download and leaves every other test
    here passing, so the wiring needs its own assertion. ``model_name`` matters as much as
    the flag: the verdict can name the swap target's cached spelling, and passing
    ``config.path`` there would load the wrong repo. A real ``load_model`` call cannot run
    here (no weights, no network, no unsloth), so read the call sites.
    """
    import ast

    source = (_BACKEND / "core/inference/inference.py").read_text(encoding = "utf-8")
    load_model = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )

    def assigned_from(predicate):
        return {
            target.id
            for node in ast.walk(load_model)
            if isinstance(node, ast.Assign) and predicate(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

    verdicts = assigned_from(
        lambda v: isinstance(v, ast.Call)
        and isinstance(v.func, ast.Name)
        and v.func.id == "_exact_model_name_for_load"
    )
    assert verdicts, "load_model never calls _exact_model_name_for_load"
    # Whatever the flag and the path are called, they have to be computed FROM the
    # verdict, so a rename stays green and a hardcoded True or config.path does not.
    derived = verdicts | assigned_from(
        lambda v: any(isinstance(n, ast.Name) and n.id in verdicts for n in ast.walk(v))
    )

    def receives(kwarg):
        return {
            node.func.value.id
            for node in ast.walk(load_model)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("FastLanguageModel", "FastVisionModel")
            and any(
                kw.arg == kwarg and isinstance(kw.value, ast.Name) and kw.value.id in derived
                for kw in node.keywords
            )
        }

    both = {"FastLanguageModel", "FastVisionModel"}
    for kwarg in ("use_exact_model_name", "model_name"):
        assert receives(kwarg) == both, f"only {sorted(receives(kwarg))} receive {kwarg}"
