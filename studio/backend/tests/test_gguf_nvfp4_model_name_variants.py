# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``NVFP4`` inside a community model name is not the quant of every file.

Teaching the quant parser ``NVFP4`` (for the Nemotron 3.5 Super ``NVFP4/`` folder) made it the
first quant token of every file in repos whose model name contains ``-NVFP4-``, so all builds got
the one variant key ``NVFP4`` and the picker kept a single row. NVFP now names the quant only last
in the name, where llama.cpp naming puts the type; earlier it is part of the model name, and those
files keep the keys they had before NVFP was a token. File names below are the real Hub listings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_backend_root = Path(__file__).resolve().parent.parent
_tests_dir = Path(__file__).resolve().parent
for _p in (_backend_root, _tests_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hub.utils.gguf import (  # noqa: E402
    extract_quant_token,
    gguf_variant_key,
    group_gguf_variant_files,
    is_mtp_drafter_path,
)


# repo -> {file: key the picker had before NVFP was a quant token}
COMMUNITY_REPOS = {
    "esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF": {
        f"Qwen3.8-27B-NVFP4-MTP-{build}.gguf": f"Qwen3.8-27B-NVFP4-MTP-{build}"
        for build in (
            "COMPACT-LOW",
            "HIGH",
            "HIGHEST",
            "LOW",
            "MEDIUM",
            "MID-HIGH",
            "ORIG",
            "VERY-HIGH",
            "VERY-LOW",
        )
    },
    "michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF": {
        "Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf": "Qwen3.6-35B-A3B-NVFP4-MTP-HQ",
        "Qwen3.6-35B-A3B-NVFP4-MTP-TURBO.gguf": "Qwen3.6-35B-A3B-NVFP4-MTP-TURBO",
    },
    "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only-GGUF": {
        "Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only-Q8_0.gguf": "Q8_0",
        "Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only.gguf": (
            "Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only"
        ),
    },
    "utautako/Qwen3.8-27B-NVFP4-MTP-Q8attn-GGUF": {
        "Qwen3.8-27B-NVFP4-MTP-Q6attn.gguf": "Qwen3.8-27B-NVFP4-MTP-Q6attn",
        "Qwen3.8-27B-NVFP4-MTP-Q8attn.gguf": "Qwen3.8-27B-NVFP4-MTP-Q8attn",
    },
    "CompiledThoughts/Qwen3.8-Flash-Next-NVFP4-Q8_0": {
        "Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf": "Q8_0",
    },
}

SUPER = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"

# Names whose quant really is the NVFP / MXFP token: unchanged by this fix.
QUANT_SUFFIX_KEYS = {
    f"NVFP4/{SUPER}-NVFP4-00001-of-00003.gguf": "NVFP4",
    f"NVFP4/{SUPER}-NVFP4-00003-of-00003.gguf": "NVFP4",
    "Qwen3.6-27B-NVFP4.gguf": "NVFP4",
    "Qwen3.6-27B-nvfp4.gguf": "nvfp4",
    "Model-UD-NVFP4.gguf": "UD-NVFP4",
    "Model-NVFP4_MOE.gguf": "NVFP4_MOE",
    "Model-NVFP4-4.25bpw.gguf": "NVFP4-4.25bpw",
    "NVFP4": "NVFP4",
    "gpt-oss-20b-MXFP4.gguf": "MXFP4",
    "Qwen3.6-35B-A3B-MXFP4_MOE.gguf": "MXFP4_MOE",
    "Qwen3.6-35B-A3B-UD-MXFP4_MOE.gguf": "UD-MXFP4_MOE",
    "MXFP4_MOE/Qwen3.6-35B-A3B-MXFP4_MOE-00002-of-00004.gguf": "MXFP4_MOE",
    # MXFP was a token before NVFP and still matches anywhere, so stored MXFP keys stay put.
    "DeepSeek-V4-Flash-Dspark-Abliterated-MXFP4-BF16.gguf": "MXFP4",
    "Some-Model-MXFP4-MTP-HIGH.gguf": "MXFP4",
}


@pytest.mark.parametrize("repo", sorted(COMMUNITY_REPOS))
def test_every_build_in_an_nvfp4_named_repo_keeps_its_own_row(repo):
    expected = COMMUNITY_REPOS[repo]
    rows = group_gguf_variant_files([(name, 1) for name in expected])
    assert len(rows) == len(expected)
    assert {key: filename for key, (filename, _size) in rows.items()} == {
        key: name for name, key in expected.items()
    }


@pytest.mark.parametrize(
    "name,key",
    [(name, key) for files in COMMUNITY_REPOS.values() for name, key in files.items()],
)
def test_nvfp4_inside_the_model_name_keeps_the_pre_nvfp_key(name, key):
    assert gguf_variant_key(name) == key


@pytest.mark.parametrize("name,key", sorted(QUANT_SUFFIX_KEYS.items()))
def test_trailing_nvfp_and_mxfp_tokens_are_still_the_quant(name, key):
    assert gguf_variant_key(name) == key
    assert extract_quant_token(name) == key.removesuffix("-4.25bpw")


def test_all_quant_label_mirrors_agree():
    """The loader label, the loader key and the variant-service scan must read the same token."""
    from core.inference import llama_cpp
    from hub.services.models import gguf_variants
    from utils.models import model_config

    name = "Qwen3.6-35B-A3B-uncensored-heretic-NVFP4-Experts-Only-Q8_0.gguf"
    assert model_config._extract_quant_label(name) == "Q8_0"
    assert model_config._gguf_variant_key(name) == "Q8_0"
    assert [m.group(2) for m in gguf_variants._KNOWN_QUANT_RE.finditer(name[:-5])] == ["Q8_0"]
    assert [m.group(2) for m in llama_cpp._GGUF_KNOWN_QUANT_RE.finditer(name[:-5])] == ["Q8_0"]

    bare = "Qwen3.8-27B-NVFP4-MTP-HIGH"
    assert model_config._gguf_variant_key(f"{bare}.gguf") == bare
    assert gguf_variants._KNOWN_QUANT_RE.search(bare) is None
    assert llama_cpp._GGUF_KNOWN_QUANT_RE.search(bare) is None

    super_file = f"NVFP4/{SUPER}-NVFP4-00001-of-00003.gguf"
    assert model_config._extract_quant_label(super_file) == "NVFP4"
    assert model_config._gguf_variant_key(super_file) == "NVFP4"


@pytest.mark.parametrize("repo", sorted(COMMUNITY_REPOS))
def test_each_key_loads_exactly_its_own_file(monkeypatch, repo):
    from core.inference import llama_cpp

    files = [".gitattributes", "README.md", *COMMUNITY_REPOS[repo]]
    monkeypatch.setattr(
        llama_cpp, "call_hub_with_anonymous_retry", lambda fn, token, repo_id: list(files)
    )
    for name, key in COMMUNITY_REPOS[repo].items():
        assert llama_cpp._resolve_variant_gguf_files(repo, key) == (name, [])


@pytest.mark.parametrize("repo", sorted(COMMUNITY_REPOS))
def test_each_download_plan_holds_exactly_its_own_file(repo):
    from hub.utils.gguf_plan import build_gguf_variant_plans

    siblings = [
        SimpleNamespace(rfilename = name, size = 1 << 20, lfs = None) for name in COMMUNITY_REPOS[repo]
    ]
    plans = build_gguf_variant_plans(siblings)
    assert {key: set(plan.main_filenames) for key, plan in plans.items()} == {
        key.lower(): {name} for name, key in COMMUNITY_REPOS[repo].items()
    }


def test_super_nvfp4_shards_group_into_one_row():
    shards = [f"NVFP4/{SUPER}-NVFP4-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    rows = group_gguf_variant_files([(name, 10) for name in shards])
    assert rows == {"NVFP4": (shards[0], 30)}


def test_nvfp4_suffixed_drafters_stay_drafters():
    assert is_mtp_drafter_path(f"LiLiCorr/lilicorr-{SUPER}-NVFP4.gguf")
    assert is_mtp_drafter_path("Bonsai-8B-dspark-NVFP4.gguf")
    assert not is_mtp_drafter_path("Qwen3.8-27B-dspark-NVFP4-MTP-HIGH.gguf")


def test_a_custom_folder_of_nvfp4_named_builds_lists_each_by_its_own_name(tmp_path):
    from hub.utils.gguf import list_local_gguf_variants
    from routes.models import _resolve_quant_gguf
    from utils.models.model_config import _find_local_gguf_by_variant

    folder = tmp_path / "esatapedico" / "Qwen3.8-27B-NVFP4-MTP-GGUF"
    folder.mkdir(parents = True)
    names = [f"Qwen3.8-27B-NVFP4-MTP-{build}.gguf" for build in ("HIGHEST", "LOW", "MID-HIGH")]
    for name in names:
        (folder / name).write_bytes(b"x" * 64)
    variants, _ = list_local_gguf_variants(str(folder))
    listed = {v.quant: v.filename for v in variants}
    assert listed == {name[: -len(".gguf")]: name for name in names}
    for quant, filename in listed.items():
        assert Path(_find_local_gguf_by_variant(str(folder), quant)).name == filename
        assert Path(_resolve_quant_gguf(str(folder), quant, True)[0]).name == filename
