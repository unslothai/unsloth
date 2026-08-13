# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every model a defaults file says it applies to has to actually load it.

A model id reaches its YAML either through MODEL_NAME_MAPPING or through the
`org/model` -> `org_model.yaml` filename convention. The primary name is always
covered by the convention, so only the aliases in the header comment can drift,
and when one does the model silently falls back to default.yaml with generic
hyperparameters instead of its tuned ones.
"""

import re
from pathlib import Path

import pytest

from utils.models.model_config import load_model_defaults

_DEFAULTS_DIR = (
    Path(__file__).parent.parent / "assets" / "configs" / "model_defaults"
)
_ALSO_APPLIES_RE = re.compile(r"^#\s*Also applies to:\s*(.+)$", re.MULTILINE)


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


_CLAIMED = list(_claimed_aliases())


def test_the_alias_fixture_is_not_empty():
    """A rename or a header reformat should fail loudly, not quietly pass."""
    assert len(_CLAIMED) > 20, f"only found {len(_CLAIMED)} claimed aliases"


@pytest.mark.parametrize("config_name, alias", _CLAIMED, ids = lambda v: v)
def test_claimed_alias_loads_its_own_defaults(config_name, alias):
    primary = load_model_defaults(config_name[: -len(".yaml")].replace("_", "/", 1))
    assert primary, f"{config_name} did not load through its own name"

    resolved = load_model_defaults(alias)
    assert resolved == primary, (
        f"{alias} is listed in {config_name} but loaded different defaults; "
        f"it needs an entry in MODEL_NAME_MAPPING"
    )
