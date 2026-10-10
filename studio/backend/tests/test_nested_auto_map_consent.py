# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An auto_map on a sub-config is remote code, and the gate has to see it.

transformers reads auto_map off whichever config builds the thing being built, so a
composite model declares it on `text_config` rather than at the top level.
`unsloth/models/_utils.py` resolves `text_config.auto_map["AutoModelForCausalLM"]`
through `get_class_from_dynamic_module`, which downloads and executes that module, and
`unsloth/models/loader.py` walks every sub-config level for the compiler. A gate that
read only `cfg["auto_map"]` therefore reported "ships no remote code" for a repo whose
code the load runs, and allowed it with no scan, no findings and no fingerprint.

The scanner runs for real here; only the local model directory is a fixture.
"""

import json

import pytest

from utils.security import evaluate_remote_code_consent
from utils.security.remote_code_scan import HIGH

# Fix helpers imported inside tests so pre-fix code fails on assertions, not import.


def _model(tmp_path, config):
    """A local model directory whose auto_map target is suspicious enough to block."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (model / "modeling_evil.py").write_text(
        "import subprocess\nsubprocess.Popen(['id'])\n",
        encoding = "utf-8",
    )
    return model


NESTED_AUTO_MAP = {
    "model_type": "llama",
    "text_config": {
        "model_type": "llama",
        "auto_map": {"AutoModelForCausalLM": "modeling_evil.Model"},
    },
}


def test_a_nested_auto_map_is_scanned_rather_than_called_a_noop(tmp_path):
    """The defect, end to end through the public gate.

    Before the fix this returned has_remote_code False with "trust_remote_code is a
    no-op" in the reason, for a repo whose modelling file the load executes.
    """
    model = _model(tmp_path, NESTED_AUTO_MAP)

    decision = evaluate_remote_code_consent(str(model), trust_remote_code = True)

    assert decision.has_remote_code is True
    assert decision.blocked is True
    assert decision.max_severity == HIGH
    assert decision.fingerprint
    assert "no-op" not in decision.reason


def test_the_nested_module_is_named_in_the_refs_the_scan_fetches(tmp_path):
    """The trigger firing is not enough: the ref has to reach the fetch list too."""
    from utils.security.remote_code_scan import _auto_map_refs
    assert _auto_map_refs(NESTED_AUTO_MAP) == {(None, "modeling_evil.py")}


def test_a_top_level_auto_map_is_unchanged(tmp_path):
    """Negative control. The common case must behave exactly as it did."""
    model = _model(tmp_path, {"auto_map": {"AutoModel": "modeling_evil.Model"}})

    decision = evaluate_remote_code_consent(str(model), trust_remote_code = True)

    assert decision.has_remote_code is True
    assert decision.blocked is True
    assert decision.max_severity == HIGH


def test_a_model_with_no_auto_map_anywhere_is_still_a_noop(tmp_path):
    """The fast path for every ordinary model: no scan, no dialog, no slowdown.

    This is the regression that would hurt, because it is every user. A walk that
    reported auto_map where there is none would put the consent dialog in front of
    models that ship no Python at all.
    """
    model = tmp_path / "plain"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "text_config": {"model_type": "llama", "hidden_size": 8},
                "vision_config": {"model_type": "siglip", "hidden_size": 8},
            }
        ),
        encoding = "utf-8",
    )

    decision = evaluate_remote_code_consent(str(model), trust_remote_code = True)

    assert decision.has_remote_code is False
    assert decision.blocked is False
    assert "no-op" in decision.reason


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, False),
        ({"auto_map": {}}, False),
        ({"auto_map": {"AutoModel": "modeling_evil.Model"}}, True),
        ({"text_config": {"auto_map": {"AutoModel": "m.C"}}}, True),
        ({"thinker_config": {"text_config": {"auto_map": {"AutoModel": "m.C"}}}}, True),
        ({"sub_configs": [{"auto_map": {"AutoModel": "m.C"}}]}, True),
        ({"auto_map": "modeling_evil.Model"}, False),
        ({"text_config": None}, False),
        ({"text_config": "llama"}, False),
    ],
)
def test_the_walk_agrees_with_what_transformers_would_read(config, expected):
    from utils.security.remote_code_scan import config_declares_auto_map
    assert config_declares_auto_map(config) is expected


def test_the_walk_is_depth_bounded(tmp_path):
    """A config is attacker-supplied JSON, so the walk must not be unbounded."""
    from utils.security.remote_code_scan import config_declares_auto_map, iter_auto_maps

    config: dict = {"auto_map": {"AutoModel": "modeling_evil.Model"}}
    for _ in range(64):
        config = {"text_config": config}

    assert config_declares_auto_map(config) is False
    assert list(iter_auto_maps(config)) == []
