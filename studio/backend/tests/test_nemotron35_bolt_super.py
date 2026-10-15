# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA Nemotron 3.5 Bolt and Super GGUF repos load the right files and sampling.

Both repos ship files the quant parser did not know about:

- Super has an ``NVFP4/`` quant folder. ``NVFP4`` was not a quant token (only MXFP was),
  so the ``NVFP4`` variant resolved to ``LiLiCorr/lilicorr-...-NVFP4.gguf``, a 1.3 GB
  drafter, and loaded it as the main model.
- Super's ``LiLiCorr/`` drafter was listed as a selectable quant. Super uses its embedded
  MTP head; the LiLiCorr drafter is never launched, so it must not appear as a variant.
- Bolt's ``dspark-...-NVFP4.gguf`` sidecar kept ``-nvfp4`` in its pairing stem, so a local
  folder load attached no DSpark drafter.

Sampling follows NVIDIA's recommendation for both: temperature 1.0, top_p 0.95.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


SUPER_REPO = "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"
SUPER = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
_SUPER_SPLITS = {
    "NVFP4": 3, "Q8_0": 4, "UD-IQ1_M": 2, "UD-IQ2_M": 3, "UD-IQ2_XXS": 3, "UD-IQ3_XXS": 3,
    "UD-Q2_K_XL": 3, "UD-Q3_K_XL": 3, "UD-Q4_K_XL": 3, "UD-Q5_K_XL": 4, "UD-Q6_K_XL": 4,
}
SUPER_FILES = [".gitattributes", "README.md", "mmproj-BF16.gguf",
               f"LiLiCorr/lilicorr-{SUPER}-NVFP4.gguf"] + [
    f"{q}/{SUPER}-{q}-{i:05d}-of-{n:05d}.gguf"
    for q, n in _SUPER_SPLITS.items() for i in range(1, n + 1)
]

BOLT = "NVIDIA-Nemotron-3.5-Bolt-74B-A9B"
BOLT_FILES = [".gitattributes", "mmproj-BF16.gguf", f"dspark-{BOLT}-NVFP4.gguf"] + [
    f"{BOLT}-{q}.gguf" for q in ("UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_XXS", "official-IQ2_XXS")
]


def _model_info(files):
    return SimpleNamespace(
        siblings = [SimpleNamespace(rfilename = f, size = 1 << 30) for f in files],
        gguf = None,
        sha = "0" * 40,
    )


def test_super_nvfp4_variant_resolves_to_the_weights_not_the_drafter(monkeypatch):
    from core.inference import llama_cpp

    monkeypatch.setattr(
        llama_cpp, "call_hub_with_anonymous_retry", lambda fn, token, repo: list(SUPER_FILES)
    )
    first, extra = llama_cpp._resolve_variant_gguf_files(SUPER_REPO, "NVFP4")
    assert first == f"NVFP4/{SUPER}-NVFP4-00001-of-00003.gguf"
    assert extra == [f"NVFP4/{SUPER}-NVFP4-0000{i}-of-00003.gguf" for i in (2, 3)]


@pytest.mark.parametrize("quant", sorted(_SUPER_SPLITS))
def test_every_super_quant_resolves_to_its_own_folder(monkeypatch, quant):
    from core.inference import llama_cpp

    monkeypatch.setattr(
        llama_cpp, "call_hub_with_anonymous_retry", lambda fn, token, repo: list(SUPER_FILES)
    )
    first, extra = llama_cpp._resolve_variant_gguf_files(SUPER_REPO, quant)
    assert first == f"{quant}/{SUPER}-{quant}-00001-of-{_SUPER_SPLITS[quant]:05d}.gguf"
    assert len(extra) == _SUPER_SPLITS[quant] - 1


def test_lilicorr_is_a_drafter_not_a_variant():
    """All three mirrors of the drafter predicate must agree, or one path lists it."""
    from core.inference import llama_cpp
    from hub.utils import gguf as hub_gguf
    from utils.models import model_config

    path = f"LiLiCorr/lilicorr-{SUPER}-NVFP4.gguf"
    assert llama_cpp._drafter_path_kind(path) == "lilicorr"
    assert hub_gguf.is_mtp_drafter_path(path)
    assert model_config._is_mtp_drafter(path)


def test_super_variant_menu_lists_exactly_the_real_quants(monkeypatch):
    from utils.models import model_config

    monkeypatch.setattr(
        model_config, "_hub_model_info_slow_link",
        lambda repo, token, files_metadata = True: _model_info(SUPER_FILES),
    )
    variants, has_vision = model_config.list_gguf_variants(SUPER_REPO)
    assert has_vision
    assert sorted(v.quant for v in variants) == sorted(_SUPER_SPLITS)


def test_bolt_local_folder_finds_its_dspark_sidecar(tmp_path):
    from utils.models.model_config import detect_dspark_file

    for name in BOLT_FILES:
        (tmp_path / name).write_bytes(b"GGUF")
    found = detect_dspark_file(str(tmp_path / f"{BOLT}-UD-IQ2_XXS.gguf"))
    assert found is not None and Path(found).name == f"dspark-{BOLT}-NVFP4.gguf"


EXPECTED_SAMPLING = {"temperature": 1.0, "top_p": 0.95}

NEMOTRON35_IDS = [
    SUPER_REPO,
    f"{SUPER_REPO}:UD-Q4_K_XL",
    "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF",
    "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF:UD-IQ2_XXS",
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
    "/home/u/.cache/huggingface/hub/models--unsloth--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF/snapshots/deadbeef",
    f"/data/models/{BOLT}-GGUF/{BOLT}-UD-IQ2_XXS.gguf",
]


@pytest.mark.parametrize("model_id", NEMOTRON35_IDS)
def test_nemotron35_sampling_defaults(model_id):
    from utils.inference.inference_config import load_inference_config

    config = load_inference_config(model_id)
    for key, want in EXPECTED_SAMPLING.items():
        assert config[key] == want, f"{model_id}: {key} was {config[key]}, expected {want}"


def test_older_nemotrons_keep_their_sampling():
    from utils.inference.inference_config import load_inference_config

    assert load_inference_config("unsloth/Nemotron-3-Nano-30B-A3B-GGUF")["top_p"] == 1.0
