# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across vLLM PyPI minors >= 0.9.0 (GitHub raw-fetch, no pip/GPU).

Catches API drift like vLLM PR #30253 (vllm.lora.models split), 0.14
supports_tower_connector_lora(), 0.15 create_lora_manager rename, the
lora_path -> lora_dir rename, and the 0.11 v0 graph-capture removal.
Asserts every symbol unsloth-zoo's vllm_utils + vllm_lora_* expects is present.
"""

from __future__ import annotations

import functools
import json
import os
import re
import urllib.error
import urllib.request

import pytest


# Every stable vLLM release from here on is covered. A hand-kept list silently
# stops testing the moment a new minor ships, which is how 0.28 moving
# bitsandbytes out of tree went unnoticed, so derive it from PyPI instead.
_VLLM_MIN_VERSION = (0, 9, 0)

# Used when PyPI is unreachable (offline CI, network blip). Stale by design: it
# only needs to keep the suite meaningful, not current.
_VLLM_TAGS_FALLBACK = [
    "v0.9.0",
    "v0.9.1",
    "v0.9.2",
    "v0.10.0",
    "v0.10.1",
    "v0.10.2",
    "v0.11.0",
    "v0.12.0",
    "v0.13.0",
    "v0.14.0",
    "v0.15.0",
    "v0.15.1",
    "v0.16.0",
    "v0.17.0",
    "v0.17.1",
    "v0.18.0",
    "v0.18.1",
    "v0.19.0",
    "v0.19.1",
    "v0.20.0",
    "v0.20.1",
    "v0.20.2",
]


def _stable_release_tags() -> list[str]:
    """Stable vLLM releases >= _VLLM_MIN_VERSION, as git tags, oldest first.

    Only `X.Y.Z` is accepted: release candidates, dev builds and post releases
    are not what users pip install, and a fully yanked release is not one we
    owe compatibility to. Any PyPI failure falls back rather than failing the
    suite, since an unreachable index says nothing about our compatibility.
    """
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/vllm/json", timeout = 20) as r:
            releases = json.loads(r.read().decode("utf-8"))["releases"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return list(_VLLM_TAGS_FALLBACK)

    versions = []
    for version, files in releases.items():
        if not files or all(f.get("yanked") for f in files):
            continue
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
        if m is None:
            continue
        parts = tuple(int(g) for g in m.groups())
        if parts >= _VLLM_MIN_VERSION:
            versions.append(parts)
    if not versions:
        return list(_VLLM_TAGS_FALLBACK)
    return [f"v{major}.{minor}.{patch}" for major, minor, patch in sorted(versions)]


# `main` catches drift before it ships to PyPI.
VLLM_TAGS = _stable_release_tags() + ["main"]


@functools.lru_cache(maxsize = None)
def _tag_exists(tag: str) -> bool:
    return _fetch_text("vllm-project/vllm", tag, "README.md") is not None


# vLLM 0.28 (PR #43529) moved bitsandbytes out of tree to vllm-bnb-plugin. The
# plugin re-exports the same names, so unsloth_zoo resolves whichever is
# installed; the symbols must keep existing in one home or the other.
VLLM_BNB_IN_TREE = "vllm/model_executor/layers/quantization/bitsandbytes.py"
VLLM_BNB_PLUGIN_REPO = "vllm-project/vllm-bnb-plugin"
VLLM_BNB_PLUGIN_PATH = "vllm_bnb_plugin/bitsandbytes.py"
# Only these two are REQUIRED. unsloth_zoo subclasses BitsAndBytesConfig and
# replaces BitsAndBytesLinearMethod._apply_4bit_weight, so both must exist.
# `apply_bnb_4bit` is hasattr-checked (the in-tree module has never defined it
# directly, and unsloth_zoo carries a branch for each case), and
# `is_layer_skipped_bnb` is assigned onto the module rather than read from it.
VLLM_BNB_SYMBOLS = (
    "BitsAndBytesConfig",
    "BitsAndBytesLinearMethod",
)


@functools.lru_cache(maxsize = None)
def _fetch_text(repo: str, ref: str, path: str) -> str | None:
    """Fetch a file's text from GitHub; None on 404 (renamed/removed, informational).

    Cached: the tag list is now every release, and the same few paths are read
    once per tag per test, so without this the suite makes thousands of requests
    and gets rate limited.
    """
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    req = urllib.request.Request(url)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = 15) as r:
            return r.read().decode("utf-8", errors = "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        pytest.skip(f"GitHub fetch failed ({e.code}) for {url}")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"GitHub fetch failed ({e}) for {url}")


@pytest.fixture(autouse = True)
def _skip_when_the_tag_is_absent(request):
    """A PyPI release with no git tag is not a compatibility failure.

    Without this, such a version 404s on every path and reports as broken
    compatibility, which says nothing true about our code.
    """
    if "tag" not in request.fixturenames:
        return
    tag = request.getfixturevalue("tag")
    if not _tag_exists(tag):
        pytest.skip(f"vLLM repo carries no tag {tag}")


def _has_def(
    src: str,
    name: str,
    kind: str = "any",
) -> bool:
    """Grep for `class Name`/`def name`/`Name = ...`; avoids ast.parse so one bad line doesn't false-fail."""
    if kind in ("any", "class") and re.search(rf"^class\s+{re.escape(name)}\b", src, re.MULTILINE):
        return True
    if kind in ("any", "func") and re.search(
        rf"^(?:async\s+)?def\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
        return True
    if kind == "any" and re.search(rf"^{re.escape(name)}\s*[:=]", src, re.MULTILINE):
        return True
    return False


# HARD-import symbols: must be present in every tested version.
@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_lora_request_hard_imports(tag: str):
    """LoRARequest, get_adapter_absolute_path, PEFTHelper -- hard-imported by unsloth-zoo's vllm_lora_worker_manager."""
    src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/request.py")
    assert src is not None, f"vllm/lora/request.py missing in {tag}"
    assert _has_def(
        src, "LoRARequest", "class"
    ), f"vllm/lora/request.py:LoRARequest missing in {tag} (unsloth-zoo HARD-imports it)"

    src_utils = _fetch_text("vllm-project/vllm", tag, "vllm/lora/utils.py")
    assert src_utils is not None, f"vllm/lora/utils.py missing in {tag}"
    assert _has_def(
        src_utils, "get_adapter_absolute_path", "func"
    ), f"vllm/lora/utils.py:get_adapter_absolute_path missing in {tag}"

    src_peft = _fetch_text("vllm-project/vllm", tag, "vllm/lora/peft_helper.py")
    assert src_peft is not None, f"vllm/lora/peft_helper.py missing in {tag}"
    assert _has_def(
        src_peft, "PEFTHelper", "class"
    ), f"vllm/lora/peft_helper.py:PEFTHelper missing in {tag}"


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_config_lora_config(tag: str):
    """vllm.config.LoRAConfig -- hard-imported at module top of unsloth_zoo.vllm_lora_worker_manager."""
    candidates = [
        "vllm/config/__init__.py",
        "vllm/config.py",
        "vllm/config/lora.py",
    ]
    found = False
    for path in candidates:
        src = _fetch_text("vllm-project/vllm", tag, path)
        if src is None:
            continue
        if _has_def(src, "LoRAConfig", "class") or "LoRAConfig" in src:
            found = True
            break
    assert found, f"vllm.config.LoRAConfig missing in {tag} (checked {candidates})"


# SOFT-import symbols: either old path or new post-#30253 path is fine.


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_lora_models_either_path(tag: str):
    """The LoRA model/manager symbols must resolve via EITHER vllm.lora.models OR the post-#30253 split path."""
    needed = {
        "LoRAModel": ("class", None),
        "LoRAModelManager": ("class", None),
        "LRUCacheLoRAModelManager": ("class", None),
        "create_lora_manager": ("func", None),
    }
    # Old path: single vllm/lora/models.py (or models/__init__.py).
    old_candidates = ["vllm/lora/models.py", "vllm/lora/models/__init__.py"]
    old_src = next(
        (s for s in (_fetch_text("vllm-project/vllm", tag, p) for p in old_candidates) if s),
        None,
    )
    if old_src is not None:
        if all(_has_def(old_src, n, k) for n, (k, _) in needed.items()):
            return

    # New path (post vLLM PR #30253):
    lora_model_src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/lora_model.py")
    model_mgr_src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/model_manager.py")

    if lora_model_src is None and model_mgr_src is None:
        pytest.fail(
            f"{tag}: neither legacy vllm/lora/models.py nor split "
            f"vllm/lora/{{lora_model,model_manager}}.py found; "
            f"unsloth-zoo's try/except will fail-closed at import"
        )

    combined = (lora_model_src or "") + "\n" + (model_mgr_src or "")
    missing = [n for n, (k, _) in needed.items() if not _has_def(combined, n, k)]
    if missing:
        pytest.fail(
            f"{tag}: post-#30253 path missing symbols {missing}. "
            f"unsloth-zoo's try/except for vllm.lora.models will fall "
            f"through to the new path and crash."
        )


# Optional / version-gated symbols: assert presence only on minors claiming support.
@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_worker_lora_manager_class(tag: str):
    """vllm.lora.worker_manager.WorkerLoRAManager -- unsloth-zoo subclasses it; signature drives old_init vs new_init."""
    src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/worker_manager.py")
    if src is None:
        # Some vLLM versions split this; check fallback locations.
        alt = _fetch_text("vllm-project/vllm", tag, "vllm/v1/worker/lora_model_runner_mixin.py")
        if alt and ("WorkerLoRAManager" in alt or "LoRAModelRunnerMixin" in alt):
            return
        pytest.fail(
            f"{tag}: vllm/lora/worker_manager.py and "
            f"vllm/v1/worker/lora_model_runner_mixin.py both missing"
        )
    assert (
        _has_def(src, "WorkerLoRAManager", "class") or "WorkerLoRAManager" in src
    ), f"{tag}: vllm.lora.worker_manager.WorkerLoRAManager not in source"


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_lora_request_no_removed_kwargs(tag: str):
    """vLLM renamed lora_local_path -> lora_path -> lora_dir; assert LoRARequest still accepts lora_dir or lora_path."""
    src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/request.py")
    assert src is not None
    has_dir = bool(re.search(r"\blora_dir\b", src))
    has_path = bool(re.search(r"\blora_path\b", src))
    assert has_dir or has_path, f"{tag}: vllm.lora.request has neither lora_dir nor lora_path"


# UNSLOTH_VLLM_STANDBY hard-error windows: unsloth-zoo refuses standby on 0.10.0 <= vllm < 0.11.0 (std::bad_alloc) and
# 0.14.0 <= vllm < 0.15.0 (cudaErrorIllegalAddress).
def _vllm_zoo_local_path() -> str | None:
    """Return the on-runner path to unsloth_zoo.vllm_utils source, or None."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("unsloth_zoo.vllm_utils")
        if spec and spec.origin:
            return spec.origin
    except Exception:
        pass
    return None


def test_unsloth_zoo_standby_guards_present():
    """Sanity: the two hard-error windows exist in unsloth_zoo.vllm_utils; catches a revert that drops them."""
    path = _vllm_zoo_local_path()
    if path is None:
        pytest.skip("unsloth_zoo not installed on runner")
    src = open(path, encoding = "utf-8").read()
    has_10x_guard = re.search(r"0\.10\.0", src) and re.search(r"standby", src, re.IGNORECASE)
    has_14x_guard = re.search(r"0\.14\.0", src) and re.search(r"standby", src, re.IGNORECASE)
    assert has_10x_guard or has_14x_guard, (
        "unsloth_zoo.vllm_utils dropped the UNSLOTH_VLLM_STANDBY "
        "version-gate against vLLM 0.10.x / 0.14.x; that re-introduces the "
        "std::bad_alloc and cudaErrorIllegalAddress crashes the team fixed "
        "in unsloth-zoo commits 664e52ea / fa82dcc2."
    )


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_bitsandbytes_symbols_have_a_home(tag: str):
    """The bnb symbols unsloth_zoo patches must exist in tree OR in the plugin.

    This is the check that was missing when vLLM 0.28 moved bitsandbytes out of
    tree: `import unsloth_zoo.vllm_utils` raised ModuleNotFoundError at module
    scope, taking out every fast_inference GRPO run on 0.28+ rather than only
    the 4-bit ones, and the tag list here stopped at v0.20.1 so nothing noticed.
    """
    in_tree = _fetch_text("vllm-project/vllm", tag, VLLM_BNB_IN_TREE)
    if in_tree is not None:
        missing = [s for s in VLLM_BNB_SYMBOLS if not _has_def(in_tree, s)]
        assert not missing, f"{tag}: in-tree bitsandbytes is missing {missing}"
        return

    # Out of tree from 0.28. The plugin is versioned separately, so check its
    # main rather than trying to map a vLLM tag onto a plugin release.
    plugin = _fetch_text(VLLM_BNB_PLUGIN_REPO, "main", VLLM_BNB_PLUGIN_PATH)
    assert plugin is not None, (
        f"{tag}: bitsandbytes is absent in tree AND {VLLM_BNB_PLUGIN_PATH} could "
        f"not be fetched from {VLLM_BNB_PLUGIN_REPO}; unsloth_zoo has nowhere to "
        f"resolve the bnb linear method from, so load_in_4bit + fast_inference "
        f"has no path on this version"
    )
    missing = [s for s in VLLM_BNB_SYMBOLS if s not in plugin]
    assert not missing, (
        f"{tag}: bitsandbytes moved out of tree and the plugin's compat module "
        f"no longer re-exports {missing}"
    )


# WeightsMapper helper that strips the stacked (fused) weight maps. unsloth_zoo
# must call it before loading LoRA tensors, or q/k/v and gate/up collapse onto
# the fused names and set_lora dies with IndexError.
VLLM_WEIGHTS_MAPPER_PATH = "vllm/model_executor/models/utils.py"
VLLM_UNSTACK_HELPERS = ("get_rename_mapper", "get_unstacked_mapper")


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_weights_mapper_unstack_helper_is_named_as_expected(tag: str):
    """One of the helper spellings unsloth_zoo probes for must still exist.

    vLLM 0.25.0 added `get_unstacked_mapper`; 0.29.0 renamed it to
    `get_rename_mapper`. unsloth_zoo probed only the old name, so on 0.29 the
    full mapper reached the LoRA loader, `.q_proj`/`.k_proj`/`.v_proj` and
    `.gate_proj`/`.up_proj` all rewrote onto `.qkv_proj`/`.gate_up_proj`, and
    GRPO with fast_inference=True died in vLLM's set_lora with
    `IndexError: tuple index out of range` for 4-bit and 16-bit alike.

    Versions with no stacked maps need no helper: skip those rather than fail.
    """
    src = _fetch_text("vllm-project/vllm", tag, VLLM_WEIGHTS_MAPPER_PATH)
    if src is None:
        pytest.skip(f"{tag}: {VLLM_WEIGHTS_MAPPER_PATH} not present")
    if "orig_to_new_stacked" not in src:
        pytest.skip(f"{tag}: WeightsMapper has no stacked maps, nothing to strip")
    # A method, so indented: _has_def anchors at column 0 and would miss it.
    assert any(
        re.search(rf"^\s*def\s+{name}\b", src, re.MULTILINE) for name in VLLM_UNSTACK_HELPERS
    ), (
        f"{tag}: WeightsMapper folds fused weights via orig_to_new_stacked but "
        f"exposes none of {VLLM_UNSTACK_HELPERS}; unsloth_zoo's "
        f"_drop_stacked_weight_maps falls back to clearing the field, so add "
        f"the new spelling there"
    )
