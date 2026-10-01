# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""scripts/lint_untrusted_sinks.py: every sink class fires, and a constant stays quiet.

A checker nobody has watched fail is a checker that does not work, so each case here is
written as the real shape it exists for: a value read out of a downloaded config
reaching a dynamic import, a `sys.path` entry derived from a download directory, a
hardcoded `trust_remote_code`, and a code fetch with no revision.

The quiet cases matter as much. A gate that reports every `import_module` in the tree
would be turned off within a week, so the literal import, the validated value and the
weights download all have to come back clean.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "lint_untrusted_sinks.py"

_spec = importlib.util.spec_from_file_location("lint_untrusted_sinks", SCRIPT)
L = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = L
_spec.loader.exec_module(L)


def _scan(
    tmp_path,
    source: str,
    name: str = "sample.py",
):
    path = tmp_path / name
    path.write_text(source, encoding = "utf-8")
    return L.scan([path], roots = [tmp_path])


def _sinks(findings, tier = "A"):
    return {f["sink"] for f in findings if f["tier"] == tier}


def test_the_self_test_passes():
    assert L._self_test() == 0


def test_the_checker_runs_clean_against_its_baseline():
    """The committed baseline has to match the tree, or the gate is noise on every PR."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd = str(REPO),
        capture_output = True,
        text = True,
        timeout = 1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_baseline_is_sorted_and_shaped():
    """Sorted so two people regenerating it produce the same file."""
    with (REPO / "scripts" / "untrusted_sinks_baseline.json").open(encoding = "utf-8") as handle:
        payload = json.load(handle)
    entries = payload["entries"]
    assert list(entries) == sorted(entries)
    assert all(isinstance(count, int) and count > 0 for count in entries.values())


def test_a_config_value_reaching_a_dynamic_import_is_reported(tmp_path):
    """The shape unsloth-zoo#1083 removed, with import_module in place of exec."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = json.load(handle)['model_type']\n"
        "    return importlib.import_module('transformers.models.' + model_type)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_taint_crosses_a_function_boundary(tmp_path):
    """The read and the sink are rarely in the same function in real code."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def read(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n"
        "def load(path):\n"
        "    return importlib.import_module('a.' + read(path))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_download_derived_sys_path_entry_is_reported(tmp_path):
    """The basename collision shape: a fixed name under a directory the attacker names."""
    findings = _scan(
        tmp_path,
        "import os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    local = snapshot_download(repo)\n"
        "    sys.path.insert(0, os.path.join(os.path.dirname(local), 'Spark-TTS'))\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_module_getattr_from_a_config_is_reported(tmp_path):
    findings = _scan(
        tmp_path,
        "import json, transformers\n"
        "def build(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        name = json.load(handle)['architectures'][0]\n"
        "    return getattr(transformers, name)\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "from transformers import AutoConfig\n"
            "def probe(name):\n"
            "    return AutoConfig.from_pretrained(name, trust_remote_code = True)\n",
            "trust_remote_code = True",
        ),
        (
            "def probe(name, trust_remote_code = True):\n    return name\n",
            "trust_remote_code = True (default)",
        ),
        (
            "def probe(name):\n    kwargs = {'trust_remote_code': True}\n    return kwargs\n",
            "trust_remote_code = True (dict)",
        ),
    ],
)
def test_remote_code_turned_on_without_the_user_is_reported(tmp_path, source, expected):
    """The dict form carries no keyword at the call, so it reads as the caller's choice."""
    assert expected in _sinks(_scan(tmp_path, source))


def test_an_unpinned_code_fetch_is_reported(tmp_path):
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def install():\n"
        "    local = snapshot_download('owner/name')\n"
        "    sys.path.insert(0, local)\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


@pytest.mark.parametrize(
    "source",
    [
        # A literal import is the overwhelming majority of import_module calls.
        "import importlib\ndef load():\n    return importlib.import_module('transformers')\n",
        # A weights download with no revision, which is normal and not a code fetch.
        "from huggingface_hub import snapshot_download\n"
        "def weights(repo):\n    return snapshot_download(repo)\n",
        # The user's own flag, forwarded. The common and correct case.
        "from transformers import AutoConfig\n"
        "def load(name, trust_remote_code = False):\n"
        "    return AutoConfig.from_pretrained(name, trust_remote_code = trust_remote_code)\n",
        # A plain dict read, not a namespace lookup.
        "import json\n"
        "def read(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        config = json.load(handle)\n"
        "    return getattr(config, 'hidden_size', None)\n",
    ],
)
def test_the_quiet_cases_stay_quiet(tmp_path, source):
    assert _sinks(_scan(tmp_path, source)) == set()


@pytest.mark.parametrize(
    "guard",
    [
        # Validated and refused, which is what unsloth-zoo#1083 added at the producer.
        "    if not re.fullmatch(r'[a-z0-9_]+', model_type):\n        raise ValueError('bad')\n",
        # Validated and the result thrown away, which is not a check at all.
        "    re.fullmatch(r'[a-z0-9_]+', model_type)\n",
    ],
)
def test_a_validated_value_is_still_reported(tmp_path, guard):
    """The checker's main limitation, asserted rather than left for someone to discover.

    It does not model sanitisers. It cannot tell the regex that refuses a bad value from
    the one whose result is discarded, and treating the first as clean would mean
    trusting a pattern this script never read. So a validated value is still reported,
    and the committed baseline is where a human records that they read the validator.
    That is why the baseline is keyed on a hash of the call: changing the call re-opens
    the question instead of inheriting the last answer.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json, re\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = json.load(handle)['model_type']\n"
        + guard
        + "    return importlib.import_module('a.' + model_type)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_parameter_name_alone_is_the_weaker_tier(tmp_path):
    """Tier B is reported and never gated: the name of a parameter is a guess."""
    findings = _scan(
        tmp_path,
        "import importlib\ndef load(model_type):\n    return importlib.import_module('a.' + model_type)\n",
    )
    assert _sinks(findings, tier = "A") == set()
    assert "importlib.import_module" in _sinks(findings, tier = "B")


def test_tests_are_not_scanned(tmp_path):
    """A security test keeps the removed sink around as the thing it asserts about."""
    directory = tmp_path / "tests"
    directory.mkdir()
    source = (
        "import importlib, json\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = json.load(handle)['model_type']\n"
        "    return importlib.import_module('a.' + model_type)\n"
    )
    (directory / "test_sample.py").write_text(source, encoding = "utf-8")

    assert L.scan([directory], roots = [tmp_path]) == []


def test_the_scan_is_deterministic(tmp_path):
    """Same tree, same findings, in the same order: the baseline depends on it."""
    source = (
        "import importlib, json, sys\n"
        "def read(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n"
        "def load(path):\n"
        "    sys.path.insert(0, read(path))\n"
        "    return importlib.import_module('a.' + read(path))\n"
    )
    first = _scan(tmp_path, source)
    second = _scan(tmp_path, source)
    assert first == second
    assert len(first) >= 2
