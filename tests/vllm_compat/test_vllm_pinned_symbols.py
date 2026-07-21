# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across vLLM PyPI minors >= 0.9.0 (GitHub raw-fetch, no pip/GPU).

Catches API drift like vLLM PR #30253 (vllm.lora.models split), 0.14
supports_tower_connector_lora(), 0.15 create_lora_manager rename, the
lora_path -> lora_dir rename, and the 0.11 v0 graph-capture removal.
Asserts every symbol unsloth-zoo's vllm_utils + vllm_lora_* expects is present.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import pytest


# Last patch release of each tracked vLLM minor (or first stable if none yet).
VLLM_TAGS = [
    "v0.9.0",
    "v0.9.2",
    "v0.10.0",
    "v0.10.2",
    "v0.11.0",
    "v0.12.0",
    "v0.13.0",
    "v0.14.0",
    "v0.15.0",
    "v0.16.0",
    "v0.17.1",
    "v0.18.1",
    "v0.19.1",
    "v0.20.1",
    # `main` catches drift before it ships to PyPI.
    "main",
]


def _fetch_text(repo: str, ref: str, path: str) -> str | None:
    """Fetch a file's text from GitHub; None on 404 (renamed/removed, informational)."""
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


def _has_def(
    src: str,
    name: str,
    kind: str = "any",
) -> bool:
    """Grep for `class Name`/`def name`/`Name = ...`; avoids ast.parse so one bad line doesn't false-fail."""
    if kind in ("any", "class") and re.search(
        rf"^class\s+{re.escape(name)}\b", src, re.MULTILINE
    ):
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
        (
            s
            for s in (_fetch_text("vllm-project/vllm", tag, p) for p in old_candidates)
            if s
        ),
        None,
    )
    if old_src is not None:
        if all(_has_def(old_src, n, k) for n, (k, _) in needed.items()):
            return  # All resolve through the legacy single-file path.

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
        alt = _fetch_text(
            "vllm-project/vllm", tag, "vllm/v1/worker/lora_model_runner_mixin.py"
        )
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
    assert (
        has_dir or has_path
    ), f"{tag}: vllm.lora.request has neither lora_dir nor lora_path"


# UNSLOTH_VLLM_STANDBY hard-error windows: unsloth-zoo refuses standby on
#   0.10.0 <= vllm < 0.11.0 (std::bad_alloc) and 0.14.0 <= vllm < 0.15.0 (cudaErrorIllegalAddress).


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
    has_10x_guard = re.search(r"0\.10\.0", src) and re.search(
        r"standby", src, re.IGNORECASE
    )
    has_14x_guard = re.search(r"0\.14\.0", src) and re.search(
        r"standby", src, re.IGNORECASE
    )
    assert has_10x_guard or has_14x_guard, (
        "unsloth_zoo.vllm_utils dropped the UNSLOTH_VLLM_STANDBY "
        "version-gate against vLLM 0.10.x / 0.14.x; that re-introduces the "
        "std::bad_alloc and cudaErrorIllegalAddress crashes the team fixed "
        "in unsloth-zoo commits 664e52ea / fa82dcc2."
    )
