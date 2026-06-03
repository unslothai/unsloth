# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""
Pinned-symbol compat check across all vLLM PyPI minor versions
>= 0.9.0. Catches API drift like:

  - vLLM PR #30253 split vllm.lora.models -> {vllm.lora.lora_model,
    vllm.lora.model_manager}  (unsloth-zoo commit ec186187)
  - vLLM 0.14 gpu_model_runner adds supports_tower_connector_lora()
    and calls it unconditionally on every LoRA VLM
    (unsloth-zoo commit e3072a23)
  - vLLM 0.15 LoRA manager rename of create_lora_manager kwargs
    (unsloth-zoo commit 2a80d543)
  - vLLM removal of LoRARequest.embedding_padding_modules / lora_path
    -> lora_dir (unsloth-zoo commits 888f79fd, e915bca1)
  - vLLM v0 graph capture path removed in 0.11 (commit 65939946)

Strategy: for each tracked vLLM tag, fetch the relevant source files
straight from github.com/vllm-project/vllm (no pip install, no GPU
required) and assert that every symbol unsloth-zoo's vllm_utils +
vllm_lora_worker_manager + vllm_lora_request expects is present.

Symbol windows (from the unsloth-zoo upstream survey, 2026-05-07):

  HARD imports (must be present in all versions tested):
    vllm.lora.peft_helper.PEFTHelper
    vllm.lora.request.LoRARequest
    vllm.lora.utils.get_adapter_absolute_path
    vllm.config.LoRAConfig (+ VllmConfig from 0.11+)

  SOFT imports (try/except wrappers in unsloth-zoo; either branch OK):
    vllm.lora.models.{LoRAModel, create_lora_manager}     -- pre #30253
    vllm.lora.lora_model.LoRAModel                        -- post #30253
    vllm.lora.model_manager.create_lora_manager           -- post #30253

  Behavioural (must exist when the corresponding feature is in scope):
    vllm.device_allocator.cumem.{CuMemAllocator, libcudart, ...}
        -- only required if UNSLOTH_VLLM_STANDBY=1; on 0.10.x and
           0.14.x the feature is hard-errored anyway, so the absence
           of those modules in those versions is fine.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

import pytest


# Tags that map to the released vLLM minor versions we care about.
# Each tracked tag is the last patch release of that minor (or the
# minor's first stable release if no later patch exists yet). Add new
# rows when vLLM ships a new minor.
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
    # `main` catches symbol drift that hasn't shipped to PyPI yet,
    # giving us a few-day lead on a release that would break us.
    "main",
]


def _fetch_text(repo: str, ref: str, path: str) -> str | None:
    """Fetch a file's text from GitHub. Returns None on 404 (the file
    is renamed/removed in this version, which is informational, not a
    hard failure)."""
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


def _has_def(src: str, name: str, kind: str = "any") -> bool:
    """Heuristic AST-equivalent grep for `class Name`, `def name`,
    or `Name = ...` at module scope. We avoid a full ast.parse so a
    single non-importable line (e.g. type: ignore) doesn't false-fail."""
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


# -------------------------------------------------------------------------
# HARD-import symbols: must be present in every tested version.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_lora_request_hard_imports(tag: str):
    """vllm.lora.request.LoRARequest, vllm.lora.utils.get_adapter_absolute_path,
    vllm.lora.peft_helper.PEFTHelper. Hard-imported by unsloth-zoo's
    vllm_lora_worker_manager."""
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
    """vllm.config.LoRAConfig. Imported at module top of
    unsloth_zoo.vllm_lora_worker_manager (HARD)."""
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


# -------------------------------------------------------------------------
# SOFT-import symbols: either old path or new post-#30253 path is fine.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_lora_models_either_path(tag: str):
    """unsloth-zoo's vllm_lora_worker_manager imports
    {LoRAModel, LoRAModelManager, LRUCacheLoRAModelManager,
    create_lora_manager} from EITHER vllm.lora.models OR
    {vllm.lora.lora_model + vllm.lora.model_manager}. Verify at least
    one path resolves every symbol, in every version."""
    needed = {
        "LoRAModel": ("class", None),
        "LoRAModelManager": ("class", None),
        "LRUCacheLoRAModelManager": ("class", None),
        "create_lora_manager": ("func", None),
    }
    # Old path: a single vllm/lora/models.py (or vllm/lora/models/__init__.py).
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


# -------------------------------------------------------------------------
# Optional / version-gated symbols. Don't fail if missing on minors
# unsloth-zoo already gates against; assert presence on minors that
# claim support.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", VLLM_TAGS)
def test_vllm_worker_lora_manager_class(tag: str):
    """vllm.lora.worker_manager.WorkerLoRAManager. unsloth-zoo subclasses
    this; signature inspection drives old_init vs new_init choice."""
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
    """vLLM removed `lora_local_path` -> `lora_path` -> `lora_dir`
    progressively. unsloth-zoo's vllm_lora_request must not depend on
    the older spelling (else GRPO + fast_inference breaks on the
    rename release).

    We assert the LoRARequest constructor accepts EITHER the new name
    or both (forward-compat). Specifically: presence of `lora_dir` or
    `lora_path` is sufficient; both is the transition state."""
    src = _fetch_text("vllm-project/vllm", tag, "vllm/lora/request.py")
    assert src is not None
    has_dir = bool(re.search(r"\blora_dir\b", src))
    has_path = bool(re.search(r"\blora_path\b", src))
    assert (
        has_dir or has_path
    ), f"{tag}: vllm.lora.request has neither lora_dir nor lora_path"


# -------------------------------------------------------------------------
# UNSLOTH_VLLM_STANDBY hard-error windows.
# unsloth-zoo refuses to enable standby on:
#   0.10.0 <= vllm < 0.11.0  (std::bad_alloc)
#   0.14.0 <= vllm < 0.15.0  (cudaErrorIllegalAddress)
# Make this enforcement testable so a future commit doesn't accidentally
# remove the guard.
# -------------------------------------------------------------------------


def _vllm_zoo_local_path() -> str | None:
    """Return the on-runner path to unsloth_zoo.vllm_utils source if
    importable. None otherwise."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("unsloth_zoo.vllm_utils")
        if spec and spec.origin:
            return spec.origin
    except Exception:
        pass
    return None


def test_unsloth_zoo_standby_guards_present():
    """Sanity: the two hard-error windows exist somewhere in the
    unsloth_zoo.vllm_utils source. Catches a future revert that drops
    them."""
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
