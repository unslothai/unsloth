# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from utils.notebooks import catalog

_COLAB = "https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/"

README = f"""
| **Gemma 4 (E2B)** | Vision | [![Open in Colab](x.svg)]({_COLAB}Gemma4_%28E2B%29-Vision.ipynb) |
| **Llama3.1** **(8B)** | Alpaca | <a href="{_COLAB}Llama3.1_(8B)-Alpaca.ipynb" target="_blank"><img src="y"></a> |
| **Again** | Alpaca | <a href="{_COLAB}Llama3.1_(8B)-Alpaca.ipynb"></a> |
| Kaggle | <a href="{_COLAB}Kaggle-Llama3.1_(8B)-Alpaca.ipynb"></a> |
| AMD | <a href="{_COLAB}AMD-Llama3.1_(8B)-Alpaca.ipynb"></a> |
| Other repo | [x](https://colab.research.google.com/github/someone/else/blob/main/nb/Foo.ipynb) |
| Drive | [x](https://colab.research.google.com/drive/1abc) |
"""


def test_readme_parser_keeps_public_unsloth_notebooks_only():
    assert catalog.parse_readme_notebooks(README) == [
        "Gemma4_(E2B)-Vision.ipynb",
        "Llama3.1_(8B)-Alpaca.ipynb",
    ]


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(catalog, "_cached_files", None)
    monkeypatch.setattr(catalog, "_cache_until", 0.0)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising = False)


def test_catalog_uses_readme_index(monkeypatch, fresh_cache):
    monkeypatch.setattr(
        catalog, "_fetch_repo_notebook_files", lambda: catalog.parse_readme_notebooks(README)
    )
    entries = {entry["notebook_file"]: entry for entry in catalog.build_notebook_catalog()}
    assert set(entries) == {"Gemma4_(E2B)-Vision.ipynb", "Llama3.1_(8B)-Alpaca.ipynb"}
    gemma = entries["Gemma4_(E2B)-Vision.ipynb"]
    assert gemma["featured"] is True
    assert gemma["category"] == "vision"
    assert gemma["colab_url"] == _COLAB + "Gemma4_(E2B)-Vision.ipynb"


def test_catalog_falls_back_to_local_metadata_when_fetch_fails(monkeypatch, fresh_cache):
    monkeypatch.setattr(catalog, "_fetch_repo_notebook_files", lambda: None)
    files = {entry["notebook_file"] for entry in catalog.build_notebook_catalog()}
    assert set(catalog._load_overrides()) <= files
    assert files


def test_offline_mode_never_fetches(monkeypatch, fresh_cache):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def fail():
        raise AssertionError("fetched while offline")

    monkeypatch.setattr(catalog, "_fetch_repo_notebook_files", fail)
    assert catalog.build_notebook_catalog()


@pytest.mark.parametrize(
    "name",
    [
        "Qwen3_(4B)-GRPO.ipynb",
        "gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb",
        "NeMo-Gym-Sudoku.ipynb",
        "Zephyr_(7B)-DPO.ipynb",
        "Llama3_(8B)-ORPO.ipynb",
        "Llama3.2_(1B)-PPO.ipynb",
    ],
)
def test_rl_and_preference_recipes_are_not_sft(name):
    # The frontend keys "Use in Studio" off this category: Studio trains SFT / CPT only.
    assert catalog._category(name) == "grpo"


@pytest.mark.parametrize("name", ["Llama3.1_(8B)-Alpaca.ipynb", "Mistral_Nemo_(12B)-Alpaca.ipynb"])
def test_sft_recipes_stay_sft(name):
    assert catalog._category(name) == "sft"


@pytest.mark.parametrize("name", ["Gemma3N_(4B)-Audio.ipynb", "Gemma4_(E2B)-Audio.ipynb"])
def test_audio_recipes_are_audio(name):
    assert catalog._category(name) == "audio"
