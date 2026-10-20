# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-install the SSM/Mamba kernels a hybrid model lazy-imports during ``from_pretrained``;
the inference-path counterpart of the training worker's install.
"""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional

from loggers import get_logger
from utils.kernel_install import (
    CAUSAL_CONV1D,
    MAMBA_SSM,
    PinnedKernel,
    hipcc_gcc_install_dir,
    install_prebuilt,
    source_build_command,
    source_build_run_kwargs,
    uninstall_command,
)
from utils.wheel_utils import (
    direct_wheel_url,
    install_wheel,
    is_windows_wheel_env,
    pinned_source_build_supported,
    probe_torch_wheel_env,
    url_exists,
)

logger = get_logger(__name__)

StatusCb = Optional[Callable[[str], None]]

# Lowercased-id substring matches, shared with the training worker. mamba-ssm models are a subset of the causal-conv1d set.
# The Mamba-2 hybrids whose modeling (or remote) code cannot load without mamba_ssm: a failed install is fatal for them.
REQUIRED_SSM_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
)
SSM_MODEL_SUBSTRINGS = REQUIRED_SSM_MODEL_SUBSTRINGS + (
    "mamba",
    "jamba",
    "zamba",
    "bamba",
)
# config.json model types whose modeling files run mamba_ssm (and so causal_conv1d). Matched on the
# resolved config, so a renamed checkpoint or an id without the substrings above
# (NVIDIA-Nemotron-Nano-9B-v2, granite-4.0-tiny) still counts. The Mamba-1 family and the other
# hybrids have a working torch path in transformers, so mamba_ssm is only an acceleration there.
REQUIRED_SSM_MODEL_TYPES = frozenset({"nemotron_h", "falcon_h1", "granitemoehybrid"})
SSM_MODEL_TYPES = REQUIRED_SSM_MODEL_TYPES | frozenset(
    {"bamba", "jamba", "zamba", "zamba2", "mamba", "mamba2", "falcon_mamba"}
)
CAUSAL_CONV1D_MODEL_SUBSTRINGS = (
    "nemotron_h",
    "nemotron-h",
    "nemotron-3-nano",
    "falcon_h1",
    "falcon-h1",
    "granite-4.0-h",
    "granitemoehybrid",
    "lfm2",
    "mamba",
    "jamba",
    "zamba",
    "bamba",
)
# GatedDeltaNet models unsloth-zoo runs without causal_conv1d: its Triton causal conv for prefill /
# training and Triton causal_conv1d_update for cached decode (temporary_patches/gdn_causal_conv1d.py,
# `_GDN_MODELING`), plus the vendored FLA kernels. Qwen3.6 reuses the qwen3_5 model types. A usable
# causal_conv1d still wins inside zoo; this only stops Studio from downloading it. Zoo's kill switch,
# a zoo without the decode kernel (decode was slower on a T4 without it), ROCm, macOS or no Triton
# bring the request back.
ZOO_GDN_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_next"})
ZOO_GDN_MODEL_SUBSTRINGS = (
    "qwen3.5",
    "qwen3_5",
    "qwen3.6",
    "qwen3_6",
    "qwen3-next",
    "qwen3_next",
)
ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH = "UNSLOTH_DISABLE_TRITON_CAUSAL_CONV1D"
# Present in the zoo source from the release that added the decode kernel.
_ZOO_GDN_DECODE_KERNEL_MARKER = "def triton_causal_conv1d_update("
_TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE: dict[tuple[str, str], bool | None] = {}


def zoo_triton_causal_conv1d_enabled() -> bool:
    return os.environ.get(ZOO_TRITON_CAUSAL_CONV1D_KILL_SWITCH, "0") != "1"


@functools.lru_cache(maxsize = 1)
def _zoo_triton_platform_usable() -> bool:
    """Whether zoo's Triton conv can engage here: a CUDA torch (not ROCm) with Triton importable.
    macOS, ROCm and a Windows install without triton-windows keep the prebuilt causal-conv1d.
    Read from torch's distribution version, so no model load pays for importing torch: ROCm and
    CPU builds carry `+rocm` / `+cpu`, a Windows CUDA build `+cu`, and a Linux PyPI torch with no
    local version is the CUDA build."""
    if sys.platform == "darwin" or importlib.util.find_spec("triton") is None:
        return False
    torch = sys.modules.get("torch")
    if torch is not None:
        version = getattr(torch, "version", None)
        return bool(getattr(version, "cuda", None)) and not getattr(version, "hip", None)
    try:
        local = importlib.metadata.version("torch").partition("+")[2].lower()
    except importlib.metadata.PackageNotFoundError:
        return False
    if local.startswith(("rocm", "cpu")):
        return False
    return local.startswith("cu") or (not local and sys.platform.startswith("linux"))


@functools.lru_cache(maxsize = 1)
def _zoo_gdn_decode_kernel_present() -> bool:
    """Whether the installed unsloth_zoo ships the Triton decode update. Read from its source on
    sys.path, so nothing is imported and an import-guard stub in sys.modules cannot answer."""
    try:
        spec = importlib.machinery.PathFinder.find_spec("unsloth_zoo")
    except (ImportError, ValueError):
        return False
    for location in (spec.submodule_search_locations or ()) if spec is not None else ():
        source = Path(location) / "temporary_patches" / "gdn_causal_conv1d.py"
        try:
            if _ZOO_GDN_DECODE_KERNEL_MARKER in source.read_text(encoding = "utf-8", errors = "ignore"):
                return True
        except OSError:
            continue
    return False


def zoo_triton_causal_conv1d_usable() -> bool:
    """zoo runs Qwen3.5 / 3.6 / Qwen3-Next prefill, training and decode without causal_conv1d here."""
    return (
        zoo_triton_causal_conv1d_enabled()
        and _zoo_triton_platform_usable()
        and _zoo_gdn_decode_kernel_present()
    )


def _is_zoo_gdn_model_type(model_type: str) -> bool:
    """The model type or one of its `_text` / `_vision` sub-configs belongs to a zoo-covered model."""
    return _base_model_type(model_type) in ZOO_GDN_MODEL_TYPES


class SsmKernels(NamedTuple):
    """What one model needs: ``causal_conv1d`` and ``mamba_ssm`` (best effort), and whether a
    missing ``mamba_ssm`` must stop the load. ``probe`` is the identifier the decision was made for
    (an adapter's base when there is one); ``model_types`` the config's model types, if read."""

    causal_conv1d: bool
    mamba_ssm: bool
    mamba_ssm_required: bool
    probe: str
    model_types: frozenset[str] = frozenset()
    # (dynamic-module namespace, implementation module) pairs from the config's auto_map, as
    # transformers names them under transformers_modules (see remote_code_modules).
    remote_code: frozenset = frozenset()
    # Whether the load will import code that cannot run without mamba_ssm, so a failed install
    # must refuse it: remote modeling code importing it unconditionally, or a model type this
    # transformers has no native implementation of. mamba_ssm_required alone (Nemotron-H,
    # Falcon-H1, Granite-4.0-H) only means "worth a source build": their native transformers
    # models run without it.
    mamba_ssm_fatal: bool = False


def _base_model_type(model_type: str) -> str:
    """A config's model type with its `_text` / `_vision` sub-config suffix removed."""
    candidate = model_type.strip().lower().replace("-", "_")
    for suffix in ("_text", "_vision"):
        candidate = candidate.removesuffix(suffix)
    return candidate


def model_is_ssm(model_name: str) -> bool:
    """Whether *model_name* is a Mamba/SSM model whose fast path is ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in SSM_MODEL_SUBSTRINGS)


def model_requires_mamba_ssm(model_name: str) -> bool:
    """Whether *model_name* cannot load at all without ``mamba_ssm``."""
    name = (model_name or "").lower()
    return any(sub in name for sub in REQUIRED_SSM_MODEL_SUBSTRINGS)


def model_wants_causal_conv1d(model_name: str) -> bool:
    """Whether *model_name* needs ``causal_conv1d`` (the SSM set plus linear-attention hybrids like LFM2 whose modeling files lazy-import it; Qwen3.5 / 3.6 / Qwen3-Next only where zoo's Triton conv cannot run)."""
    name = (model_name or "").lower()
    if any(sub in name for sub in CAUSAL_CONV1D_MODEL_SUBSTRINGS):
        return True
    return (
        any(sub in name for sub in ZOO_GDN_MODEL_SUBSTRINGS)
        and not zoo_triton_causal_conv1d_usable()
    )


def _normalized_model_identifier(value: str) -> str:
    return "".join(
        character for character in value.lower() if character.isascii() and character.isalnum()
    )


def _transformers_models_dir() -> Path | None:
    """The active transformers' ``models`` folder, found without importing transformers: the
    inference worker asks before its transformers import, which snapshots the kernel gates."""
    try:
        spec = importlib.util.find_spec("transformers")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    origin = getattr(spec, "origin", None)
    if not locations and origin:
        locations = [str(Path(origin).parent)]
    for location in locations:
        models_dir = Path(location) / "models"
        if models_dir.is_dir():
            return models_dir
    return None


def _transformers_model_type_uses_causal_conv1d(model_type: str) -> bool | None:
    candidate = model_type.strip().lower().replace("-", "_")
    if not candidate or any(
        not (character.isascii() and (character.isalnum() or character == "_"))
        for character in candidate
    ):
        return None
    models_dir = _transformers_models_dir()
    cache_key = (str(models_dir), candidate)
    if cache_key in _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE:
        return _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[cache_key]

    result: bool | None = None
    try:
        model_dir = models_dir / candidate if models_dir is not None else None
        if model_dir is not None and model_dir.is_dir():
            for modeling_file in model_dir.glob("modeling_*.py"):
                try:
                    source = modeling_file.read_text(encoding = "utf-8", errors = "ignore")
                except OSError:
                    continue
                result = False
                if "causal_conv1d" in source:
                    result = True
                    break
    except Exception as exc:
        logger.debug("causal-conv1d model-type inspection skipped: %s", exc)

    _TRANSFORMERS_CAUSAL_CONV1D_MODEL_TYPE_CACHE[cache_key] = result
    return result


def _config_identifiers(model_config: dict) -> tuple[set[str], set[str]]:
    """Every model_type and architecture in *model_config*, nested sub-configs included."""
    model_types: set[str] = set()
    architectures: set[str] = set()
    pending: list[Any] = [model_config]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            model_type = value.get("model_type")
            if isinstance(model_type, str):
                model_types.add(model_type)
            model_architectures = value.get("architectures")
            if isinstance(model_architectures, (list, tuple)):
                architectures.update(
                    architecture
                    for architecture in model_architectures
                    if isinstance(architecture, str)
                )
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return model_types, architectures


def model_config_is_ssm(model_config: dict) -> bool | None:
    """True for a Mamba-family model type, False for another known type, None without one."""
    model_types, _ = _config_identifiers(model_config)
    if not model_types:
        return None
    return any(_base_model_type(model_type) in SSM_MODEL_TYPES for model_type in model_types)


def model_config_requires_mamba_ssm(model_config: dict) -> bool | None:
    """True when the config's model cannot load without ``mamba_ssm``; None without a model type."""
    model_types, _ = _config_identifiers(model_config)
    if not model_types:
        return None
    return any(
        _base_model_type(model_type) in REQUIRED_SSM_MODEL_TYPES for model_type in model_types
    )


# The required families by name, for a decision with no config to read.
_REQUIRED_MODEL_TYPE_BY_SUBSTRING = (
    ("nemotron_h", "nemotron_h"),
    ("nemotron-h", "nemotron_h"),
    ("nemotron-3-nano", "nemotron_h"),
    ("falcon_h1", "falcon_h1"),
    ("falcon-h1", "falcon_h1"),
    ("granite-4.0-h", "granitemoehybrid"),
    ("granitemoehybrid", "granitemoehybrid"),
)
_NATIVE_MODELING_CACHE: dict[tuple[str, str], bool] = {}


def native_modeling_available(model_type: str) -> bool:
    """Whether the active transformers ships a native implementation of *model_type*
    (``models/<type>/modeling_*.py`` on disk), found without importing transformers."""
    models_dir = _transformers_models_dir()
    if models_dir is None:
        return False
    key = (str(models_dir), model_type)
    if key not in _NATIVE_MODELING_CACHE:
        try:
            _NATIVE_MODELING_CACHE[key] = any((models_dir / model_type).glob("modeling_*.py"))
        except OSError:
            _NATIVE_MODELING_CACHE[key] = False
    return _NATIVE_MODELING_CACHE[key]


def _imports_mamba_ssm_unconditionally(source: str) -> bool:
    """Whether module *source* imports ``mamba_ssm`` at import time with no way around it: a
    top-level import, or one in a ``try`` whose handler raises. One under an ``if`` (an
    ``is_mamba_2_ssm_available()`` gate) or with a fallback handler does not count. Unparseable
    source counts: it cannot be shown safe."""
    import ast

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return True

    def imports_mamba(statements) -> bool:
        for node in statements:
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "mamba_ssm" for alias in node.names
            ):
                return True
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").split(".")[0] == "mamba_ssm"
            ):
                return True
        return False

    for node in tree.body:
        if imports_mamba([node]):
            return True
        if isinstance(node, ast.Try) and imports_mamba(node.body):
            if any(
                any(isinstance(inner, ast.Raise) for inner in ast.walk(handler))
                for handler in node.handlers
            ):
                return True
    return False


_REMOTE_SOURCES: dict[tuple[str, str, str | None], str | None] = {}


def _remote_source_text(repo: str, filename: str, hf_token: str | None) -> str | None:
    """The text of *filename* in a local checkpoint or a Hub repo (the local HF cache first, then
    the Hub unless offline), or None when it cannot be read."""
    key = (repo, filename, hf_token)
    if key in _REMOTE_SOURCES:
        return _REMOTE_SOURCES[key]
    text = None
    try:
        if os.path.isdir(repo):
            path = Path(repo) / filename
            text = path.read_text(encoding = "utf-8") if path.is_file() else None
        elif repo.count("/") == 1:
            text = _hub_cached_text(repo, filename)
            from utils.utils import hf_env_offline
            if text is None and not hf_env_offline():
                import urllib.request

                from utils.transformers_version import _hf_raw_url, _hf_urlopen

                headers = {"User-Agent": "unsloth-studio"}
                if isinstance(hf_token, str) and hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                request = urllib.request.Request(_hf_raw_url(repo, filename), headers = headers)
                with _hf_urlopen(request, timeout = 10) as response:
                    text = response.read().decode("utf-8", errors = "replace")
    except Exception as exc:
        logger.debug("Could not read %s from %s: %s", filename, repo, exc)
        text = None
    _REMOTE_SOURCES[key] = text
    return text


def _hub_cached_text(repo: str, filename: str) -> str | None:
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        repo_dir = Path(str(get_hf_cache_paths().hub_cache)) / (
            "models--" + repo.replace("/", "--")
        )
        ref = repo_dir / "refs" / "main"
        candidates = []
        if ref.is_file():
            candidates.append(
                repo_dir / "snapshots" / ref.read_text(encoding = "utf-8").strip() / filename
            )
        candidates += sorted(
            repo_dir.glob(f"snapshots/*/{filename}"), key = lambda p: p.stat().st_mtime, reverse = True
        )
        for path in candidates:
            if path.is_file():
                return path.read_text(encoding = "utf-8")
    except Exception as exc:
        logger.debug("HF cache lookup of %s in %s failed: %s", filename, repo, exc)
    return None


def _remote_model_sources(checkpoint: str, model_config: dict) -> list[tuple[str, str]]:
    """(repo, file) of the model classes the config's ``auto_map`` names."""
    auto_map = model_config.get("auto_map") if isinstance(model_config, dict) else None
    if not isinstance(auto_map, dict):
        return []
    sources = []
    for key, value in auto_map.items():
        if not str(key).startswith("AutoModel"):
            continue
        for reference in value if isinstance(value, (list, tuple)) else (value,):
            if not isinstance(reference, str) or "." not in reference:
                continue
            source, _, reference = reference.rpartition("--")
            module = reference.rsplit(".", 1)[0]
            sources.append((source or checkpoint, module.replace(".", "/") + ".py"))
    return list(dict.fromkeys(sources))


def mamba_ssm_is_fatal(
    checkpoint: str,
    model_config: dict | None,
    model_types,
    *,
    trust_remote_code: bool | None,
    hf_token: str | None = None,
) -> bool:
    """Whether loading this required-family model (Nemotron-H, Falcon-H1, Granite-4.0-H) imports
    code that cannot run without mamba_ssm. Remote code (auto_map, loaded under trust_remote_code
    or because transformers has no native class) counts when a model module imports it
    unconditionally; otherwise it is fatal only when transformers lacks the native model."""
    required_types = {t for t in model_types if t in REQUIRED_SSM_MODEL_TYPES}
    native = bool(required_types) and all(native_modeling_available(t) for t in required_types)
    sources = _remote_model_sources(checkpoint, model_config or {})
    if sources and (trust_remote_code or not native):
        for repo, filename in sources:
            text = _remote_source_text(repo, filename, hf_token)
            if text is None or _imports_mamba_ssm_unconditionally(text):
                return True
        return False
    return not native


def _name_mamba_ssm_is_fatal(model_name: str) -> bool:
    """mamba_ssm_is_fatal from the name alone: the family's native model is missing."""
    name = (model_name or "").lower()
    types = {model_type for sub, model_type in _REQUIRED_MODEL_TYPE_BY_SUBSTRING if sub in name}
    return bool(types) and not all(native_modeling_available(t) for t in types)


def model_config_wants_causal_conv1d(model_config: dict) -> bool | None:
    model_types, architectures = _config_identifiers(model_config)
    if model_config_is_ssm(model_config):
        return True
    # Before the source scan: their modeling files name causal_conv1d, which zoo covers.
    if model_types and all(_is_zoo_gdn_model_type(model_type) for model_type in model_types):
        return not zoo_triton_causal_conv1d_usable()
    source_requirements = {
        _transformers_model_type_uses_causal_conv1d(model_type) for model_type in model_types
    }
    if True in source_requirements:
        return True
    normalized = {_normalized_model_identifier(i) for i in model_types | architectures}

    def _matches(substrings: tuple[str, ...]) -> bool:
        needles = {_normalized_model_identifier(value) for value in substrings}
        return any(needle in identifier for identifier in normalized for needle in needles)

    if _matches(CAUSAL_CONV1D_MODEL_SUBSTRINGS):
        return True
    if _matches(ZOO_GDN_MODEL_SUBSTRINGS) and not zoo_triton_causal_conv1d_usable():
        return True
    if False in source_requirements:
        return False
    return None


def _local_adapter_base(model_load_target: str) -> str | None:
    """A local PEFT adapter folder's base model (``adapter_config.json`` without a ``config.json``)."""
    try:
        folder = Path(model_load_target)
        if not folder.is_dir() or (folder / "config.json").is_file():
            return None
        adapter_config = folder / "adapter_config.json"
        if not adapter_config.is_file():
            return None
        import json

        base = json.loads(adapter_config.read_text(encoding = "utf-8-sig")).get(
            "base_model_name_or_path"
        )
    except Exception as exc:
        logger.debug("Could not read the adapter base of %s: %s", model_load_target, exc)
        return None
    return base if isinstance(base, str) and base.strip() else None


def _dependency_target(model_load_target: str, adapter_base: str | None) -> str:
    """What decides the kernels: an adapter's (pinned, else recorded) base, else the target itself.
    The adapter stays the load target; only the dependency decision moves to its base."""
    if adapter_base:
        return adapter_base
    return _local_adapter_base(model_load_target) or model_load_target


# Configs read for the kernel decision, per process: a load's preflight, its _handle_load and each
# kernel share one read (a cold Hub read can take seconds).
_MODEL_CONFIGS: dict[tuple[str, str | None], dict] = {}


def _resolved_model_config(model_load_target: str, hf_token: str | None) -> dict | None:
    key = (str(model_load_target), hf_token)
    if key in _MODEL_CONFIGS:
        return _MODEL_CONFIGS[key]
    try:
        from utils.transformers_version import _load_config_json
        model_config = _load_config_json(model_load_target, hf_token)
    except Exception as exc:
        logger.debug("Could not inspect model config for SSM kernels: %s", exc)
        model_config = None
    if not isinstance(model_config, dict):
        return None  # a miss may be transient: read again next time
    _MODEL_CONFIGS[key] = model_config
    return model_config


def resolve_ssm_kernels(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
    model_config: dict | None = None,
    trust_remote_code: bool | None = None,
) -> SsmKernels:
    """The kernels the model at *model_load_target* needs, from its config (an adapter's base
    config when it is an adapter); the name lists only when there is no config to read. A caller
    that already read the deciding config passes it as *model_config*."""
    target = _dependency_target(model_load_target, adapter_base)
    probe = target if target != model_load_target else model_name
    if model_config is None:
        model_config = _resolved_model_config(target, hf_token)
    model_types: frozenset[str] = frozenset()
    is_ssm = required = wants = None
    remote_code: frozenset = frozenset()
    if model_config is not None:
        remote_code = remote_code_modules(target, model_config)
        model_types = frozenset(
            _base_model_type(model_type) for model_type in _config_identifiers(model_config)[0]
        )
        is_ssm = model_config_is_ssm(model_config)
        required = model_config_requires_mamba_ssm(model_config)
        wants = model_config_wants_causal_conv1d(model_config)
        logger.info(
            "SSM kernels resolved from model architecture: causal-conv1d=%s mamba-ssm=%s",
            wants,
            is_ssm,
        )
    if is_ssm is None:
        is_ssm = model_is_ssm(probe)
    if required is None:
        required = model_requires_mamba_ssm(probe)
    if wants is None:
        wants = model_wants_causal_conv1d(probe)
    fatal = False
    if is_ssm and required:
        if model_config is not None and model_types:
            fatal = mamba_ssm_is_fatal(
                target,
                model_config,
                model_types,
                trust_remote_code = trust_remote_code,
                hf_token = hf_token,
            )
        else:
            fatal = _name_mamba_ssm_is_fatal(probe)
    # Every mamba-ssm model also runs causal-conv1d.
    return SsmKernels(
        causal_conv1d = bool(wants or is_ssm),
        mamba_ssm = bool(is_ssm),
        mamba_ssm_required = bool(is_ssm and required),
        probe = probe,
        model_types = model_types,
        remote_code = remote_code,
        mamba_ssm_fatal = bool(fatal),
    )


def resolved_model_wants_causal_conv1d(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
) -> bool:
    return resolve_ssm_kernels(
        model_name, model_load_target, hf_token, adapter_base = adapter_base
    ).causal_conv1d


def resolved_model_is_ssm(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
) -> bool:
    """Whether the model at *model_load_target* runs ``mamba_ssm``, from its (adapter's base) config;
    the name list only when there is no config to read."""
    return resolve_ssm_kernels(
        model_name, model_load_target, hf_token, adapter_base = adapter_base
    ).mamba_ssm


def refresh_transformers_kernel_gates() -> None:
    """After an in-process install, let an already-imported transformers see the new kernels: its
    availability gates are lru_cached and the hub-kernel map can hold a cached None. Never raises.
    This cannot rebind functions a modeling module already captured; see
    ``modeling_modules_needing_restart``."""
    if "transformers" not in sys.modules:
        return
    try:
        import_utils = sys.modules.get("transformers.utils.import_utils")
        for gate_name in (
            "is_causal_conv1d_available",
            "is_mamba_ssm_available",
            "is_mamba_2_ssm_available",
        ):
            cache_clear = getattr(getattr(import_utils, gate_name, None), "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    except Exception as exc:
        logger.debug("Could not clear transformers' kernel gates: %s", exc)
    try:
        hub_kernels = sys.modules.get("transformers.integrations.hub_kernels")
        mapping = getattr(hub_kernels, "_KERNEL_MODULE_MAPPING", None)
        if isinstance(mapping, dict):
            for key in [key for key, value in mapping.items() if value is None]:
                if any(
                    kernel in str(key) for kernel in ("causal-conv1d", "causal_conv1d", "mamba")
                ):
                    mapping.pop(key, None)
    except Exception as exc:
        logger.debug("Could not refresh transformers' hub kernel map: %s", exc)


def _sanitized_module_name(name: str) -> str:
    # transformers.dynamic_module_utils._sanitize_module_name (4.57+, 5.x).
    name = name.replace(".", "_dot_").replace("-", "_hyphen_")
    return f"_{name}" if name and name[0].isdigit() else name


def _dynamic_module_namespaces(checkpoint: str) -> set[tuple[str, ...]]:
    """The transformers_modules package(s) transformers puts *checkpoint*'s remote code under: a
    Hub repo by its id (org, name), a local folder by its name. Sanitized as 4.57+ and 5.x spell
    them, and raw as older 4.x did."""
    checkpoint = str(checkpoint).strip()
    if not checkpoint:
        return set()
    if os.path.isdir(checkpoint):
        parts = [os.path.basename(os.path.normpath(checkpoint))]
    else:
        parts = checkpoint.split("/")
    if not all(parts):
        return set()
    return {tuple(parts), tuple(_sanitized_module_name(part) for part in parts)}


def remote_code_modules(checkpoint: str, model_config: dict) -> frozenset:
    """The (namespace, module) pairs of the model classes a config's ``auto_map`` points at: an
    entry ``"model.CustomForCausalLM"`` is module ``model`` of *checkpoint*, and
    ``"org/repo--model.Custom"`` module ``model`` of ``org/repo``. Any module name counts, not just
    ``modeling_*``; AutoConfig / AutoTokenizer / processor entries capture no kernels."""
    auto_map = model_config.get("auto_map") if isinstance(model_config, dict) else None
    if not isinstance(auto_map, dict):
        return frozenset()
    pairs = set()
    for key, value in auto_map.items():
        if not str(key).startswith("AutoModel"):
            continue
        for reference in value if isinstance(value, (list, tuple)) else (value,):
            if not isinstance(reference, str) or "." not in reference:
                continue
            source, _, reference = reference.rpartition("--")
            module = tuple(reference.rsplit(".", 1)[0].split("."))
            for namespace in _dynamic_module_namespaces(source or checkpoint):
                pairs.add((namespace, module))
    return frozenset(pairs)


def _is_remote_code_module(name: str, remote_code) -> bool:
    # transformers_modules.<namespace>[.<commit or source hash>].<module>, matched exactly.
    parts = tuple(name.split("."))
    if parts[0] != "transformers_modules":
        return False
    for namespace, module in remote_code:
        if parts[1 : 1 + len(namespace)] != namespace:
            continue
        rest = parts[1 + len(namespace) :]
        if rest == module or (len(rest) == len(module) + 1 and rest[1:] == module):
            return True
    return False


def modeling_modules_needing_restart(model_types, remote_code = ()) -> list[str]:
    """The already-imported implementation modules of *model_types* (``transformers.models.<type>.
    modeling_*``) and of the selected checkpoint's remote code (*remote_code*, from
    :func:`remote_code_modules`). Each captured its kernel functions (module globals, hub-kernel
    decorators) when it was imported, so a kernel installed after that is only picked up by a fresh
    process. A configuration or tokenizer module captures none, so importing one needs no restart."""
    model_types = {model_type for model_type in model_types if model_type}
    stale = []
    for name in list(sys.modules):
        parts = name.split(".")
        if parts[0] == "transformers":
            if (
                len(parts) == 4
                and parts[1] == "models"
                and parts[2] in model_types
                and parts[3].startswith("modeling_")
            ):
                stale.append(name)
        elif remote_code and _is_remote_code_module(name, remote_code):
            stale.append(name)
    return sorted(stale)


def ssm_probe_identifier(model_name: str, base: str | None = None) -> str:
    """The identifier whose architecture decides the SSM kernels. The substring match needs a real model id: a LoRA adapter id or a local checkpoint's parent folders are unrelated to its architecture (a Llama LoRA at ``user/falcon-h1-lora`` is not SSM). Prefer *base*; for a bare local checkpoint use its basename."""
    probe = base or model_name
    if probe == model_name:
        try:
            from utils.paths import is_local_path
            if is_local_path(model_name):
                probe = os.path.basename((model_name or "").rstrip("/\\")) or model_name
        except Exception:
            pass
    return probe


def _is_importable(import_name: str) -> bool:
    # Invalidate finder caches so a kernel installed earlier in this process is seen.
    importlib.invalidate_caches()
    try:
        __import__(import_name)
        return True
    except Exception as exc:
        # An ABI-incompatible kernel (undefined symbol after a torch/CUDA upgrade) raises OSError/RuntimeError, not ImportError; treat any failure as "not importable" so the caller reinstalls or source-builds instead of hard-failing on a merely broken kernel.
        logger.debug("%s is not importable (%s: %s)", import_name, type(exc).__name__, exc)
        return False


def _emit(status_cb: StatusCb, message: str) -> None:
    logger.info(message)
    if status_cb is None:
        return
    try:
        status_cb(message)
    except Exception:  # status is best-effort; never fail a load over a UI message
        logger.debug("ssm_runtime status callback raised", exc_info = True)


_hipcc_gcc_install_dir = hipcc_gcc_install_dir


# Keep quiet downloads and builds inside the orchestrator's inactivity deadline.
_HEARTBEAT_SECONDS = 60.0


@contextmanager
def _heartbeat(status_cb: StatusCb, message: str) -> Iterator[None]:
    """Emit *message* on a timer while the wrapped work runs. The inference orchestrator treats silence as a dead load, and status messages reset its inactivity deadline; prebuilt wheel installs and source builds can both stay quiet for minutes on aarch64 or slow links, so both paths use this."""
    done = threading.Event()

    def _beat() -> None:
        while not done.wait(_HEARTBEAT_SECONDS):
            _emit(status_cb, message)

    thread = threading.Thread(target = _beat, daemon = True, name = "ssm-install-heartbeat")
    thread.start()
    try:
        yield
    finally:
        done.set()
        # Wait out a tick that already left done.wait().
        thread.join(timeout = 1)


def _run_with_heartbeat(run, cmd, status_cb, display_name, **kwargs):
    """Run *cmd* via *run*, emitting a status every 60s so the parent's inactivity timeout is not tripped by a long (e.g. ROCm) source build."""
    with _heartbeat(
        status_cb,
        f"Still building {display_name} (this can take several minutes)...",
    ):
        return run(cmd, **kwargs)


# The torch build this process installs kernels for, probed once (one torch import in a child).
_TORCH_ENV: dict = {}
# Kernels whose install already failed in this process, keyed by kernel, torch and toolchain, so a
# later load stage (or the next load in the same worker) does not download or compile them again.
_FAILED_INSTALLS: set = set()


def _torch_env() -> dict[str, str] | None:
    key = probe_torch_wheel_env
    if key not in _TORCH_ENV:
        _TORCH_ENV[key] = probe_torch_wheel_env(timeout = 30, include_windows = True)
    return _TORCH_ENV[key]


def _kernel_backend(env: dict[str, str] | None) -> str | None:
    """ "cuda" or "hip" for a torch these kernels build / ship for; None for CPU-only, XPU, MPS,
    or a torch the probe could not read, which take the model's torch path instead."""
    if not env:
        return None
    if env.get("hip_version"):
        return "hip"
    if env.get("cuda_version") or env.get("cuda_major"):
        return "cuda"
    return None


def _failure_key(import_name: str) -> tuple:
    try:
        import importlib.metadata as metadata
        torch_version = metadata.version("torch")
    except Exception:
        torch_version = None
    return (
        import_name,
        sys.executable,
        torch_version,
        shutil.which("nvcc"),
        shutil.which("hipcc"),
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"),
        os.environ.get("ROCM_PATH"),
    )


def _installed_distribution(pypi_name: str) -> bool:
    try:
        import importlib.metadata as metadata
        metadata.version(pypi_name)
        return True
    except Exception:
        return False


def _remove_broken(pypi_name: str, display_name: str, run: Callable[..., Any]) -> None:
    """Uninstall a distribution whose extension does not import, as the training worker does: a
    later availability check would otherwise find it and fail instead of taking the torch path."""
    if not _installed_distribution(pypi_name):
        return
    try:
        removed = (
            run(
                uninstall_command(pypi_name, use_uv = bool(shutil.which("uv"))),
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except Exception:
        removed = False
    logger.warning(
        "%s does not import; %s",
        display_name,
        "removed it" if removed else f"uninstall {pypi_name} manually",
    )


def _install_kernel(
    *,
    import_name: str,
    display_name: str,
    pypi_name: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    status_cb: StatusCb,
    run: Callable[..., Any],
    newly_installed: list[str] | None = None,
    allow_source_build: bool = True,
    required: bool = False,
) -> bool:
    """Install one kernel wheel-first, then (when *allow_source_build*) a HIP-aware PyPI source build
    (never on Windows). Returns True iff importable afterwards; idempotent (no-op when already
    installed). An install this call made is appended to *newly_installed*. One install per kernel
    runs at a time across processes; a failed one is not retried in this process."""
    if _is_importable(import_name):
        logger.info("%s already installed", display_name)
        return True

    from utils.utils import package_installs_offline

    if package_installs_offline():
        # No URL probe, no source build, and no uv-to-pip fallback that would reach the network.
        logger.info("Skipping %s installation while offline", display_name)
        return False

    failure_key = _failure_key(import_name)
    if failure_key in _FAILED_INSTALLS:
        logger.info("%s already failed to install in this process; not retrying", display_name)
        return False

    from utils.kernel_install import (
        KERNEL_LOCK_WAIT,
        OPTIONAL_KERNEL_LOCK_WAIT,
        kernel_install_lock,
    )

    with kernel_install_lock(
        import_name,
        on_wait = lambda: _emit(
            status_cb, f"Waiting for another process installing {display_name}..."
        ),
        # An optional acceleration does not hold a load for a peer's half-hour build.
        wait_seconds = KERNEL_LOCK_WAIT if required else OPTIONAL_KERNEL_LOCK_WAIT,
    ) as held:
        if not held:
            _emit(
                status_cb,
                f"Could not take the {display_name} install lock (another process is installing "
                "it, or the lock is unavailable); continuing without it.",
            )
            return False
        # Another process may have installed it while this one waited: new to this worker all the
        # same, so its already-imported modeling modules are refreshed or restarted.
        if _is_importable(import_name):
            logger.info("%s installed by another process", display_name)
            if newly_installed is not None:
                newly_installed.append(import_name)
            return True
        installed: bool | None = False
        try:
            installed = _install_kernel_locked(
                import_name = import_name,
                display_name = display_name,
                pypi_name = pypi_name,
                package_version = package_version,
                release_tag = release_tag,
                release_base_url = release_base_url,
                status_cb = status_cb,
                run = run,
                allow_source_build = allow_source_build,
            )
        except Exception as exc:
            # An installer that cannot even start (OSError) fails like any other attempt.
            logger.warning("Installing %s failed: %s", display_name, exc, exc_info = True)
            installed = False
        finally:
            if installed is False:
                _remove_broken(pypi_name, display_name, run)
                _FAILED_INSTALLS.add(failure_key)
        if installed is None:
            return False  # nothing attempted (unsupported torch backend)
        if installed and newly_installed is not None:
            newly_installed.append(import_name)
        return bool(installed)


def _install_kernel_locked(
    *,
    import_name: str,
    display_name: str,
    pypi_name: str,
    package_version: str,
    release_tag: str,
    release_base_url: str,
    status_cb: StatusCb,
    run: Callable[..., Any],
    allow_source_build: bool,
) -> bool | None:
    """The install itself, under the kernel's lock. None when nothing was attempted."""
    env = _torch_env()
    backend = _kernel_backend(env)
    if backend is None:
        _emit(
            status_cb,
            f"{display_name} needs a CUDA or ROCm build of torch; using the model's torch path.",
        )
        return None
    wheel_only = sys.platform == "win32" or is_windows_wheel_env(env)
    wheel_url = direct_wheel_url(
        filename_prefix = import_name,
        package_version = package_version,
        release_tag = release_tag,
        release_base_url = release_base_url,
        env = env,
    )
    wheel_available = url_exists(wheel_url) if wheel_url else False
    # Only a 404 proves nothing is published: an unanswered HEAD (a proxy refusing HEAD but serving
    # the download) still tries the wheel, as setup always did.
    if wheel_url and wheel_available is not False:
        if wheel_available is None:
            _emit(
                status_cb,
                f"Could not check the {display_name} prebuilt wheel; trying to download it.",
            )
        _emit(status_cb, f"Installing {display_name} (prebuilt kernel) for this model...")
        # Keep quiet downloads and unpacks within the inactivity deadline (#9398).
        with _heartbeat(
            status_cb,
            f"Still installing {display_name} (prebuilt kernel)...",
        ):
            # A cold first import can also stay quiet for tens of seconds.
            outcome = install_prebuilt(
                wheel_url,
                install = install_wheel,
                verify = lambda: _is_importable(import_name),
                on_failed = lambda installer, result: logger.warning(
                    "%s could not install %s wheel:\n%s",
                    installer,
                    display_name,
                    getattr(result, "stdout", ""),
                ),
                use_uv = bool(shutil.which("uv")),
                run = run,
            )
        if outcome == "installed":
            logger.info("Installed prebuilt %s wheel", display_name)
            return True
        if outcome == "rejected":
            logger.warning("%s wheel installed but not importable", display_name)
            # Remove it before any rebuild: pip / uv would call the broken copy "satisfied".
            _remove_broken(pypi_name, display_name, run)
    elif wheel_only:
        logger.info(
            "No prebuilt %s wheel for this Windows environment (%s); using the torch fallback",
            display_name,
            wheel_url,
        )
    else:
        logger.info("No prebuilt %s wheel for this environment (%s)", display_name, wheel_url)

    if wheel_only:
        return False
    if not allow_source_build:
        _emit(
            status_cb,
            f"No usable prebuilt {display_name} for this torch; using the model's torch path.",
        )
        return False
    if not pinned_source_build_supported(env):
        # The pinned sources build at C++17 and torch 2.13+ needs C++20: a certain failure after
        # minutes of compiling, so only the prebuilt wheel serves this torch.
        _emit(
            status_cb,
            f"No prebuilt {display_name} for torch {env.get('torch_mm')} on this machine, and it "
            "cannot be built from source for it; using the model's torch path.",
        )
        return False

    # Source build (slow). ROCm has no prebuilt wheel and needs hipcc + a gcc-install-dir shim.
    spec = f"{pypi_name}=={package_version}"
    is_hip = backend == "hip"
    if is_hip and not shutil.which("hipcc"):
        _emit(status_cb, f"{display_name}: hipcc not found; install the ROCm HIP SDK to build it.")
        return False
    _emit(
        status_cb,
        f"Building {display_name} from source for this model (this can take several minutes)...",
    )
    # Reinstall so the source build replaces a broken wheel instead of no-opping as "already satisfied".
    cmd = source_build_command(spec, use_uv = bool(shutil.which("uv")), is_hip = is_hip, reinstall = True)
    run_kwargs, _ = source_build_run_kwargs(is_hip = is_hip, gcc_install_dir = _hipcc_gcc_install_dir)
    from utils.kernel_install import run_process_tree

    try:
        result = _run_with_heartbeat(
            lambda command, **kwargs: run_process_tree(command, run = run, **kwargs),
            cmd,
            status_cb,
            display_name,
            **run_kwargs,
        )
    except subprocess.TimeoutExpired:
        logger.error("%s source build timed out", display_name)
        _emit(status_cb, f"{display_name} source build timed out.")
        return False
    if getattr(result, "returncode", 1) != 0:
        logger.warning("%s source install failed:\n%s", display_name, getattr(result, "stdout", ""))
    return _is_importable(import_name)


def _pinned_kwargs(kernel: PinnedKernel) -> dict[str, str]:
    """_install_kernel arguments for a pinned kernel release."""
    return {
        "import_name": kernel.import_name,
        "display_name": kernel.display_name,
        "pypi_name": kernel.pypi_name,
        "package_version": kernel.package_version,
        "release_tag": kernel.release_tag,
        "release_base_url": kernel.release_base_url,
    }


def ensure_ssm_runtime(
    model_name: str,
    *,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
    wants_causal_conv1d: bool | None = None,
    is_ssm: bool | None = None,
    mamba_ssm_required: bool | None = None,
    mamba_ssm_fatal: bool | None = None,
) -> list[str]:
    """Install the SSM kernels *model_name* needs before load, wheel-first; a no-op for non-SSM models and idempotent. Returns the kernels newly installed by this call. Nemotron-H, Falcon-H1 and Granite-4.0-H (*mamba_ssm_required*) are worth a source build; a load fails on a missing ``mamba_ssm`` (``RuntimeError`` instead of a cryptic mid-load failure) only when *mamba_ssm_fatal*: it imports code that cannot run without it. Everything else, and ``causal_conv1d``, leaves transformers' torch path."""
    # The caller's config-resolved answers win; the name match is the default.
    if is_ssm is None:
        is_ssm = model_is_ssm(model_name)
    if mamba_ssm_required is None:
        mamba_ssm_required = model_requires_mamba_ssm(model_name)
    mamba_ssm_required = bool(is_ssm and mamba_ssm_required)
    if mamba_ssm_fatal is None:
        mamba_ssm_fatal = mamba_ssm_required and _name_mamba_ssm_is_fatal(model_name)
    mamba_ssm_fatal = bool(mamba_ssm_required and mamba_ssm_fatal)
    if wants_causal_conv1d is None:
        wants_causal_conv1d = model_wants_causal_conv1d(model_name)
    # Every mamba-ssm model also runs causal-conv1d.
    wants_causal_conv1d = wants_causal_conv1d or is_ssm
    installed: list[str] = []
    if not (wants_causal_conv1d or is_ssm):
        return installed

    def _install(kernel: PinnedKernel, *, allow_source_build: bool = True) -> bool:
        return _install_kernel(
            **_pinned_kwargs(kernel),
            status_cb = status_cb,
            run = run,
            newly_installed = installed,
            allow_source_build = allow_source_build,
            # Nemotron-H, Falcon-H1, Granite-4.0-H wait out a peer's install; the rest move on.
            required = mamba_ssm_required,
        )

    # causal-conv1d first: SSM modeling files lazy-import it, and mamba-ssm's fast path uses it.
    if wants_causal_conv1d and not _install(CAUSAL_CONV1D):
        logger.warning("causal-conv1d unavailable; continuing on the model's torch fallback")

    if not is_ssm:
        return installed
    offline_hint = (
        "If this machine is offline, run `unsloth install-kernels` once while online, "
        "then reload the model."
    )
    if sys.platform == "win32" and importlib.util.find_spec("triton") is None:
        # mamba_ssm imports triton at load and the wheel goes in --no-deps.
        if mamba_ssm_fatal:
            raise RuntimeError(
                "mamba-ssm, required by this Mamba model, needs Triton on Windows "
                "(the triton-windows package); install it and reload the model."
            )
        _emit(
            status_cb, "mamba-ssm needs Triton on Windows (triton-windows); using the torch path."
        )
        return installed
    if not mamba_ssm_required:
        # The families with a torch path get mamba-ssm only as a prebuilt wheel, and only where
        # unsloth_zoo keeps its fast path on (setup's sm80 / sm75 + Triton 3.4 floor): never a
        # source build, as before setup stopped installing it for every model.
        reason = None if _is_importable(MAMBA_SSM.import_name) else _mamba_ssm_floor_reason()
        if reason is not None:
            _emit(status_cb, f"Skipping mamba-ssm, which {reason}; using the model's torch path.")
            return installed
    if not _install(MAMBA_SSM, allow_source_build = mamba_ssm_required):
        if mamba_ssm_fatal:
            # The model cannot load without mamba-ssm (its remote code imports it), so no torch
            # path exists to fall back on. On a CPU / XPU torch no install could have helped, so
            # say what is missing instead of pointing at the network.
            if _kernel_backend(_torch_env()) is None:
                raise RuntimeError(
                    "This Mamba model requires mamba-ssm, which needs a CUDA or ROCm torch build."
                )
            raise RuntimeError(
                f"Could not install mamba-ssm, required by this Mamba model. {offline_hint}"
            )
        if mamba_ssm_required:
            _emit(
                status_cb,
                "Warning: could not install mamba-ssm; loading this model on transformers' torch "
                "path, which is slower and uses more memory.",
            )
        else:
            _emit(status_cb, "mamba-ssm is not available; continuing on the model's torch path.")
    return installed


def _mamba_ssm_floor_reason() -> str | None:
    """Why an optional mamba-ssm stays off this machine's GPU, or None when unsloth_zoo would use
    it: setup's rule (kernel_install), from the torch already imported here when there is one."""
    from utils import kernel_install

    capability = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if getattr(torch.version, "hip", None):
                return None
            if torch.cuda.is_available() and torch.cuda.device_count():
                capability = max(
                    tuple(torch.cuda.get_device_capability(i))
                    for i in range(torch.cuda.device_count())
                )
        except Exception:
            capability = None
    if capability is None:
        capability = kernel_install._gpu_capability(subprocess.run)
    if capability is None:
        return "needs a visible GPU"
    if capability >= (8, 0):
        return None
    return kernel_install._pre_ampere_skip_reason("mamba_ssm", capability, subprocess.run)


def provision_ssm_kernels(
    model_name: str,
    model_load_target: str,
    hf_token: str | None,
    *,
    adapter_base: str | None = None,
    model_config: dict | None = None,
    trust_remote_code: bool | None = None,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
) -> tuple[SsmKernels, list[str]]:
    """Resolve and install the kernels a load needs (training, inference, export): returns what was
    resolved and what this call newly installed. Raises ``RuntimeError`` only for a model that
    cannot load without ``mamba_ssm``."""
    kernels = resolve_ssm_kernels(
        model_name,
        model_load_target,
        hf_token,
        adapter_base = adapter_base,
        model_config = model_config,
        trust_remote_code = trust_remote_code,
    )
    installed = ensure_ssm_runtime(
        kernels.probe,
        status_cb = status_cb,
        run = run,
        wants_causal_conv1d = kernels.causal_conv1d,
        is_ssm = kernels.mamba_ssm,
        mamba_ssm_required = kernels.mamba_ssm_required,
        mamba_ssm_fatal = kernels.mamba_ssm_fatal,
    )
    if installed:
        refresh_transformers_kernel_gates()
    return kernels, installed
