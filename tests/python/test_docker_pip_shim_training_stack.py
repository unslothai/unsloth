# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for what the Docker pip shim protects.

The shim protected torch/vLLM/unsloth and stopped there, so every notebook run
silently mutated the training stack the image was validated with -- while printing
that it was keeping the baked versions.

The criterion for _KEEP is "replacing this invalidates the tested stack or breaks
unsloth", not "any package a notebook mentions".
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_PATH = REPO_ROOT / "docker" / "unsloth_pip_shim.py"

SHIPPED_TRL_CELL = ["--no-deps", "trl==0.22.2"]
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
    """Args after `install`, or None when the shim no-op'd; constraints pair dropped."""
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


def test_the_shipped_trl_cell_installs_nothing(shim):
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
    # these come from the cu128 index; a PyPI pin swaps in a generic (or cu13) build
    assert _run(shim, ["torchao==0.15.0", "torchcodec==0.5"]) is None


def test_transformers_companions_cannot_desynchronise_the_sidecars(shim):
    # each sidecar ships its own matched copies, so a base-venv swap breaks them all
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
    # argument filtering does not stop a DEPENDENCY of the kept target from dragging
    # peft/datasets down, which happened with no notebook ever naming peft
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shim.sys, "argv", ["pip", "install", UNBAKED])
        with pytest.raises(_Exec) as exc:
            shim.main()
    argv = exc.value.argv
    assert "--constraint" in argv
    pins = Path(argv[argv.index("--constraint") + 1]).read_text()
    names = {line.split("==")[0].lower().replace("_", "-") for line in pins.splitlines() if line}
    assert names, "the constraints file must not be empty"
    assert all(
        n in shim._KEEP or n == "transformers" or n.startswith("nvidia-") for n in names
    ), sorted(names)
