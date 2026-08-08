# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for what the Docker pip shim protects.

The shim fronts pip/uv inside the notebook kernel so a `!pip install` cell cannot
replace the baked, ABI-matched stack. It protected torch/vLLM/unsloth and stopped
there, which left the training stack wide open. Measured over the 433 shipped
notebooks (probe_notebook_pins.py against the baked image):

    trl          382 notebooks pin an older release -- 378 of them end their
                 install cell with `!pip install --no-deps trl==0.22.2`, against
                 a baked and tested trl 0.24.0
    torchao      273 reinstall it, 2 pin 0.15.0, replacing 0.17.0+cu128 with a
                 generic PyPI build
    torchcodec    92 reinstall it, 26 pin 0.5 / 0.7.0, replacing the 0.11.0+cu128
                 wheel the Dockerfile deliberately paired with torch 2.11
    datasets     254 reinstall it; a trl 0.22.2 resolve was observed pulling it
                 back from 4.3.0 to 3.0.0
    peft         225 reinstall it; observed dropping 0.19.1 -> 0.14.0
    accelerate   225 reinstall it
    hf_hub       240 reinstall it, tokenizers 64 -- both version-locked to
                 transformers, and the sidecars ship their own matched copies

So EVERY notebook run silently mutated the stack the image was validated with,
and printed "Successfully installed trl-0.22.2 peft-0.14.0 datasets-3.0.0" while
the shim reported it was keeping the baked versions.

The criterion for _KEEP is "replacing this invalidates the tested stack or breaks
unsloth", not "any package a notebook mentions": a package the notebook genuinely
needs and the image does not bake still has to install normally.

Static: drives the shim's main() with os.execv captured. No docker, no GPU, no
network.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_PATH = REPO_ROOT / "docker" / "unsloth_pip_shim.py"

# The install cell 378 of the 433 shipped notebooks actually end on.
SHIPPED_TRL_CELL = ["--no-deps", "trl==0.22.2"]
# A package the image does NOT bake: must keep installing normally.
UNBAKED = "snac"


class _Exec(Exception):
    def __init__(self, path, argv):
        self.path = path
        self.argv = list(argv)


@pytest.fixture()
def shim(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NB_TF_MARKER", str(tmp_path / "requested_transformers"))
    monkeypatch.setenv("UNSLOTH_NB_SHIM", "1")
    assert SHIM_PATH.is_file(), f"missing shim: {SHIM_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_pip_shim_stack_test", SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _fake_execv(path, argv):
        raise _Exec(path, argv)

    monkeypatch.setattr(mod.os, "execv", _fake_execv)
    return mod


def _run(
    shim,
    args,
    tool = "pip",
):
    """Return the args that reached the real tool after `install`, or None when
    the shim no-op'd. The always-injected protected-constraints pair is dropped."""
    argv = ["uv", "pip", "install", *args] if tool == "uv" else ["pip", "install", *args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", argv)
        try:
            shim.main()
            return None
        except _Exec as exc:
            i = exc.argv.index("install")
            execd = exc.argv[i + 1 :]
            if (
                len(execd) >= 2
                and execd[-2] == "--constraint"
                and os.path.basename(execd[-1]).startswith("unsloth-nb-protected-")
            ):
                execd = execd[:-2]
            return execd


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pkg",
    [
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "torchao",
        "torchcodec",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
    ],
)
def test_training_stack_is_protected(shim, pkg):
    assert (
        pkg in shim._KEEP
    ), f"{pkg} is baked and tested; a notebook pin replacing it invalidates the image"


def test_the_original_gpu_stack_is_still_protected(shim):
    for pkg in [
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
        "xformers",
        "vllm",
        "bitsandbytes",
        "unsloth",
        "unsloth-zoo",
    ]:
        assert pkg in shim._KEEP


def test_unrelated_packages_are_not_swept_in(shim):
    # The criterion is "invalidates the tested stack", not "a notebook mentions
    # it". These are all installed by shipped notebooks and must stay installable.
    for pkg in [
        "snac",
        "causal-conv1d",
        "mamba-ssm",
        "omegaconf",
        "timm",
        "librosa",
        "trackio",
        "open-spiel",
        "protobuf",
        "sentencepiece",
    ]:
        assert pkg not in shim._KEEP, f"{pkg} must still install for the notebooks that need it"


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------
def test_the_shipped_trl_cell_installs_nothing(shim):
    # `!pip install --no-deps trl==0.22.2` is the last line of 378 notebooks.
    assert _run(shim, SHIPPED_TRL_CELL) is None


def test_a_mixed_cell_keeps_only_the_unbaked_package(shim):
    execd = _run(
        shim,
        [
            "--no-deps",
            "trl==0.22.2",
            "peft==0.14.0",
            "datasets==3.0.0",
            "accelerate==1.0.0",
            UNBAKED,
        ],
    )
    assert execd == ["--no-deps", UNBAKED], execd


def test_cuda_matched_wheels_are_not_replaced_by_pypi_builds(shim):
    # torchao 0.17.0+cu128 and torchcodec 0.11.0+cu128 are resolved from the
    # cu128 index; a PyPI pin swaps in a generic (or cu13) build.
    assert _run(shim, ["torchao==0.15.0", "torchcodec==0.5"]) is None


def test_transformers_companions_cannot_desynchronise_the_sidecars(shim):
    # Each sidecar ships its own matched huggingface_hub/tokenizers/safetensors;
    # replacing the base-venv copies desynchronises every sidecar at once.
    assert (
        _run(shim, ["huggingface_hub==0.30.0", "tokenizers==0.20.0", "safetensors==0.4.0"]) is None
    )


def test_an_unbaked_package_still_installs(shim):
    assert _run(shim, [UNBAKED]) == [UNBAKED]
    assert _run(shim, [UNBAKED], tool = "uv") == [UNBAKED]


def test_protection_survives_a_requirements_file(shim, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text(f"trl==0.22.2\npeft==0.14.0\ndatasets==3.0.0\n{UNBAKED}\n")
    execd = _run(shim, ["-r", str(req)])
    assert execd is not None and execd[0] == "-r"
    filtered = Path(execd[1]).read_text()
    assert UNBAKED in filtered
    for dropped in ("trl", "peft", "datasets"):
        assert dropped not in filtered, f"{dropped} slipped through the requirements file"


def test_protection_survives_a_direct_wheel_url(shim):
    url = "https://files.pythonhosted.org/x/trl-0.22.2-py3-none-any.whl"
    assert _run(shim, [url, UNBAKED]) == [UNBAKED]


def test_protection_survives_an_editable_vcs_install(shim):
    assert _run(shim, ["-e", "git+https://github.com/huggingface/trl.git", UNBAKED]) == [UNBAKED]


def test_forwarded_installs_pin_the_protected_set_for_the_resolver(shim):
    # Argument filtering alone does not stop a dependency of the kept target from
    # dragging peft/datasets back down -- which is how peft 0.19.1 became 0.14.0
    # with no notebook ever naming peft. Every forwarded install carries pins.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", ["pip", "install", UNBAKED])
        with pytest.raises(_Exec) as exc:
            shim.main()
    argv = exc.value.argv
    assert "--constraint" in argv
    pins = Path(argv[argv.index("--constraint") + 1]).read_text()
    names = {line.split("==")[0].lower().replace("_", "-") for line in pins.splitlines() if line}
    # only the installed subset is pinned, but nothing outside the protected set
    assert names, "the constraints file must not be empty"
    assert all(
        n in shim._KEEP or n == "transformers" or n.startswith("nvidia-") for n in names
    ), sorted(names)
