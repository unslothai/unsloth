# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for transformers-sidecar selection in the Unsloth Docker image.

The image runs unslothai/notebooks unchanged by refusing a notebook's
`transformers==X` install and activating a baked "sidecar" (transformers X plus
its matched huggingface_hub/tokenizers/safetensors) on sys.path instead. The
selection was a pure CEILING -- smallest baked version >= the request -- which
ignored that vLLM is version-locked to transformers. Two of the four baked
sidecars could not be imported by the baked vLLM 0.26.0 at all, and they were
exactly the two the common pins selected:

    sidecar 4.57.6   ImportError: Support for Transformers v4 is deprecated and
                     was removed in vLLM v0.24.0
                     <- pins 4.48 / 4.52.3 / 4.55.4 / 4.56.1 / 4.56.2 / 4.57.x
                        = 241 of the 433 shipped notebooks
    sidecar 5.3.0    ImportError: cannot import name 'ALLOWED_LAYER_TYPES' from
                     transformers.configuration_utils
                     <- pins 5.2.0 / 5.3.0 = 13 more notebooks

254 of 433 notebooks therefore died at `from unsloth import FastModel`, before
the first model cell. Pointing UNSLOTH_TF_SIDECAR_ROOT at an empty directory,
changing nothing else, turned two of them into clean 22/22 and 25/25 passes.

The fix is a FLOOR in front of the ceiling. Which versions are above the floor is
not hardcoded: the Dockerfile imports vllm.transformers_utils.config under every
candidate sidecar (the vLLM module that reads the transformers API -- it
reproduces both failures and needs no GPU, which matters because the build host
has none), deletes the ones that raise, and records the lowest survivor. A
request below the floor is clamped UP to the lowest eligible sidecar, which is
the closest thing to the notebook's pin the image can actually run.

Static: parses the Dockerfile and drives unsloth_nb_compat against a synthetic
sidecar root. No docker, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
COMPAT_PATH = REPO_ROOT / "docker" / "unsloth_nb_compat.py"

# Every distinct transformers pin across the 433 shipped notebooks, and the
# sidecar each must resolve to once 4.57.6 and 5.3.0 are gone.
SHIPPED_PINS = [
    "4.48",
    "4.52.3",
    "4.55.4",
    "4.56.1",
    "4.56.2",
    "4.57.0",
    "4.57.1",
    "4.57.3",
    "5.2.0",
    "5.3.0",
    "5.5.0",
    "5.10.1",
    "5.11.0",
]


@pytest.fixture(scope = "module")
def dockerfile() -> str:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"
    return DOCKERFILE.read_text()


@pytest.fixture(scope = "module")
def sidecar_block(dockerfile: str) -> str:
    start = dockerfile.index("tf-sidecars/t_$(echo")
    block = dockerfile[dockerfile.rindex("RUN set -eux", 0, start) :]
    return block[: block.index("\n\n")]


def _load_compat(root, floor = None):
    """Import a fresh unsloth_nb_compat bound to a synthetic sidecar root."""
    import os

    prev_root = os.environ.get("UNSLOTH_TF_SIDECAR_ROOT")
    prev_min = os.environ.get("UNSLOTH_TF_SIDECAR_MIN")
    os.environ["UNSLOTH_TF_SIDECAR_ROOT"] = str(root)
    os.environ.pop("UNSLOTH_TF_SIDECAR_MIN", None)
    try:
        spec = importlib.util.spec_from_file_location("unsloth_nb_compat_under_test", COMPAT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if prev_root is None:
            os.environ.pop("UNSLOTH_TF_SIDECAR_ROOT", None)
        else:
            os.environ["UNSLOTH_TF_SIDECAR_ROOT"] = prev_root
        if prev_min is not None:
            os.environ["UNSLOTH_TF_SIDECAR_MIN"] = prev_min
    return mod


@pytest.fixture()
def fixed_root(tmp_path):
    """The sidecar root the fixed Dockerfile produces: only verified sidecars,
    plus the recorded floor."""
    for name in ("t_5_5_0", "t_5_10_2"):
        (tmp_path / name).mkdir()
    (tmp_path / ".vllm_min_transformers").write_text("5.5.0\n")
    return tmp_path


@pytest.fixture()
def stale_root(tmp_path):
    """A root that still carries the incompatible sidecars (a bind-mounted or
    pre-fix directory). The recorded floor must keep them unselectable."""
    for name in ("t_4_57_6", "t_5_3_0", "t_5_5_0", "t_5_10_2"):
        (tmp_path / name).mkdir()
    (tmp_path / ".vllm_min_transformers").write_text("5.5.0\n")
    return tmp_path


# --------------------------------------------------------------------------
# The build must decide eligibility by measurement, not by a literal.
# --------------------------------------------------------------------------
def test_build_verifies_every_sidecar_against_the_baked_vllm(sidecar_block: str):
    assert "import vllm.transformers_utils.config" in sidecar_block, (
        "each baked sidecar must be proven importable by the baked vLLM; this is "
        "the module that reads the transformers API and it reproduces both the "
        "v4 refusal and the ALLOWED_LAYER_TYPES break"
    )


def test_build_verification_needs_no_gpu(sidecar_block: str):
    # `import unsloth` raises NotImplementedError("cannot find any torch
    # accelerator") on the build host, so it can never be the gate.
    assert (
        "import unsloth" not in sidecar_block
    ), "the sidecar gate must not import unsloth: the build host has no GPU"


def test_an_unverifiable_sidecar_is_deleted_not_shipped(sidecar_block: str):
    assert re.search(r"DROPPED", sidecar_block), "a failed candidate must be reported"
    assert re.search(r'rm -rf "\$DEST"', sidecar_block), (
        "a sidecar the baked vLLM cannot import must be removed, not shipped: it "
        "can never be selected safely and it costs image size"
    )


def test_build_records_the_selection_floor(sidecar_block: str):
    assert (
        ".vllm_min_transformers" in sidecar_block
    ), "the lowest verified version must be recorded for unsloth_nb_compat"
    assert "sort -V | head -1" in sidecar_block, "the floor is the LOWEST survivor"


def test_build_fails_when_no_sidecar_survives(sidecar_block: str):
    assert "exit 1" in sidecar_block, (
        "an empty sidecar set means the whole per-notebook mechanism is dead; "
        "that must fail the build rather than ship silently"
    )


def test_build_skips_the_gate_when_vllm_is_absent(sidecar_block: str):
    # The vLLM install is fail-soft per arch; with no vLLM there is no constraint
    # and every sidecar must survive rather than the build exploding.
    assert "HAVE_VLLM" in sidecar_block


def test_compat_reads_the_floor_the_build_writes():
    assert ".vllm_min_transformers" in COMPAT_PATH.read_text(), (
        "unsloth_nb_compat must read the floor the Dockerfile records, not a "
        "literal that rots on the next vLLM bump"
    )


# --------------------------------------------------------------------------
# Selection: floor, then ceiling.
# --------------------------------------------------------------------------
def test_floor_is_read_back(fixed_root):
    assert _load_compat(fixed_root).min_version() == "5.5.0"


@pytest.mark.parametrize(
    "pin, expected",
    [
        # every pin below the floor clamps UP to the lowest eligible sidecar
        ("4.48", "t_5_5_0"),
        ("4.52.3", "t_5_5_0"),
        ("4.55.4", "t_5_5_0"),
        ("4.56.1", "t_5_5_0"),
        ("4.56.2", "t_5_5_0"),
        ("4.57.0", "t_5_5_0"),
        ("4.57.1", "t_5_5_0"),
        ("4.57.3", "t_5_5_0"),
        ("5.2.0", "t_5_5_0"),
        ("5.3.0", "t_5_5_0"),
        # at and above the floor, the ceiling still decides
        ("5.5.0", "t_5_5_0"),
        ("5.10.1", "t_5_10_2"),
        # newer than every sidecar -> the baked transformers
        ("5.11.0", None),
    ],
)
def test_every_shipped_pin_resolves_to_a_vllm_compatible_sidecar(fixed_root, pin, expected):
    got = _load_compat(fixed_root).sidecar_for(pin)
    assert (Path(got).name if got else None) == expected


def test_no_shipped_pin_can_reach_an_incompatible_sidecar(stale_root):
    compat = _load_compat(stale_root)
    for pin in SHIPPED_PINS:
        got = compat.sidecar_for(pin)
        name = Path(got).name if got else None
        assert name not in (
            "t_4_57_6",
            "t_5_3_0",
        ), f"pin {pin} selected {name}, which the baked vLLM cannot import"


def test_model_tier_fallback_is_clamped_too(stale_root):
    # tier_for_model maps qwen3-next and friends to 5.3.0; that tier must not
    # reach the 5.3.0 sidecar either.
    compat = _load_compat(stale_root)
    tier = compat.tier_for_model("unsloth/Qwen3-Next-80B-A3B")
    assert tier == "5.3.0"
    assert Path(compat.sidecar_for(tier)).name == "t_5_5_0"


def test_an_unrecorded_floor_keeps_the_old_ceiling_behaviour(tmp_path):
    # No .vllm_min_transformers (an environment that never ran the build-time
    # verification): selection must not silently start dropping sidecars.
    for name in ("t_4_57_6", "t_5_5_0"):
        (tmp_path / name).mkdir()
    compat = _load_compat(tmp_path)
    assert compat.min_version() is None
    assert Path(compat.sidecar_for("4.56.2")).name == "t_4_57_6"
