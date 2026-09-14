# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every model a defaults file says it applies to has to actually load it.

A model id reaches its YAML either through MODEL_NAME_MAPPING or through the
`org/model` -> `org_model.yaml` filename convention, and it has to get there
whether it arrives bare or as the tail of a local model directory. When it does
not, the model silently falls back to default.yaml with generic hyperparameters
instead of its tuned ones.
"""

import re
from pathlib import Path

import pytest
import yaml

from utils.models.model_config import load_model_defaults

_DEFAULTS_DIR = Path(__file__).parent.parent / "assets" / "configs" / "model_defaults"
_ALSO_APPLIES_RE = re.compile(r"^#\s*Also applies to:\s*(.+)$", re.MULTILINE)

_DEFAULT_CONFIG = yaml.safe_load((_DEFAULTS_DIR / "default.yaml").read_text(encoding = "utf-8"))


def _configs():
    """Every tuned defaults file, by filename."""
    return sorted(p.name for p in _DEFAULTS_DIR.rglob("*.yaml") if p.name != "default.yaml")


def _claimed_aliases():
    """(config filename, alias) for every name a defaults file claims to cover."""
    for path in sorted(_DEFAULTS_DIR.rglob("*.yaml")):
        if path.name == "default.yaml":
            continue
        header = path.read_text(encoding = "utf-8")[:1000]
        match = _ALSO_APPLIES_RE.search(header)
        if match is None:
            continue
        for alias in match.group(1).split(","):
            alias = alias.strip().strip('"').strip()
            # Skip prose such as "and its GGUF variants".
            if alias and " " not in alias:
                yield path.name, alias


_CONFIGS = _configs()
_CLAIMED = list(_claimed_aliases())


def _primary_name(config_name):
    return config_name[: -len(".yaml")].replace("_", "/", 1)


def _on_disk(model_id):
    """The id as an LM Studio or custom scan folder hands it over: <root>/<publisher>/<model>.

    Those rows carry the filesystem path, not a repo id, so this is the form the defaults
    lookup actually receives for a locally stored model.
    """
    return f"/home/u/.lmstudio/models/{model_id}"


def _load_tuned(model_id, config_name):
    """Load `model_id`'s defaults, failing if it fell through to default.yaml."""
    config = load_model_defaults(model_id)
    assert config and config != _DEFAULT_CONFIG, (
        f"{model_id} got default.yaml, not {config_name}: it reaches its config through "
        f"neither MODEL_NAME_MAPPING nor the org/model -> org_model.yaml convention"
    )
    return config


def test_the_fixtures_are_not_empty():
    """A rename or a header reformat should fail loudly, not quietly pass."""
    assert len(_CONFIGS) > 20, f"only found {len(_CONFIGS)} defaults files"
    assert len(_CLAIMED) > 20, f"only found {len(_CLAIMED)} claimed aliases"


@pytest.mark.parametrize("config_name", _CONFIGS, ids = lambda v: v)
def test_config_loads_under_its_own_name(config_name):
    """Bare id and local directory both have to reach the file named after them."""
    primary = _primary_name(config_name)
    own = _load_tuned(primary, config_name)
    assert _load_tuned(_on_disk(primary), config_name) == own


@pytest.mark.parametrize("config_name, alias", _CLAIMED, ids = lambda v: v)
def test_claimed_alias_loads_its_own_defaults(config_name, alias):
    """Same for every name the header claims, in both forms."""
    own = _load_tuned(_primary_name(config_name), config_name)
    assert _load_tuned(alias, config_name) == own
    assert _load_tuned(_on_disk(alias), config_name) == own


@pytest.mark.parametrize(
    "model_id",
    [
        "LiquidAI/LFM2-1.2B",
        "unsloth/LFM2-1.2B-unsloth-bnb-4bit",
    ],
)
def test_lfm2_supported_ids_use_all_linear_defaults(model_id):
    config = _load_tuned(model_id, "unsloth_LFM2-1.2B.yaml")
    assert config["lora"]["target_modules"] == ["all-linear"]
