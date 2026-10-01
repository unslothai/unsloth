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


@pytest.mark.parametrize(
    "source",
    [
        # import json as js: the alias has no suffix in the source table.
        "import importlib\nimport json as js\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = js.load(handle)['model_type']\n"
        "    return importlib.import_module('a.' + model_type)\n",
        # from json import loads: the call is a bare name.
        "import importlib\nfrom json import loads\n"
        "def load(path):\n"
        "    model_type = loads(open(path + '/config.json').read())['model_type']\n"
        "    return importlib.import_module('a.' + model_type)\n",
        # from yaml import safe_load, same shape with a different deserialiser.
        "import importlib\nfrom yaml import safe_load\n"
        "def load(path):\n"
        "    model_type = safe_load(open(path + '/config.json').read())['model_type']\n"
        "    return importlib.import_module('a.' + model_type)\n",
    ],
)
def test_an_aliased_deserialiser_still_taints(tmp_path, source):
    """Suffix matching alone cannot see through an import alias.

    These produced no taint at all, so a sink immediately downstream was accepted. The
    callee is now rewritten through the file's own imports before the tables see it.
    """
    assert "importlib.import_module" in _sinks(_scan(tmp_path, source))


def test_a_sink_before_a_later_tainted_assignment_is_reported(tmp_path):
    """Flow-insensitive has to mean flow-insensitive, including backedges.

    A loop that consumes a name and then rebinds it from a deserialiser for the next
    iteration is a real executable flow. One ordered traversal built the local taint as
    it went, so the sink was visited before the assignment and never reconsidered.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(paths):\n"
        "    name = 'transformers'\n"
        "    for path in paths:\n"
        "        importlib.import_module(name)\n"
        "        name = json.loads(open(path + '/config.json').read())['model_type']\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_no_shipped_package_is_left_out_of_the_default_targets():
    """A package left out of the defaults is a surface CI never looks at.

    The CI invocation passes no paths, so this tuple decides the scope of the gate, and
    `unsloth_cli` was missing from it. Derived from the tree rather than listed, so a
    new top-level package fails here instead of being silently unscanned. The tuple is
    shared with the unsloth-zoo copy, so it also names packages that live only there.
    """
    shipped = {
        entry.name
        for entry in REPO.iterdir()
        if entry.is_dir()
        and (entry / "__init__.py").is_file()
        and entry.name not in {"tests"}
        and not entry.name.startswith(".")
    }
    missing = sorted(shipped - set(L.DEFAULT_TARGETS))
    assert not missing, f"shipped packages outside the gate's default scope: {missing}"


def test_taint_flows_through_an_instance_method(tmp_path):
    """`self.parse(...)` is a call like any other, and it was resolving to nothing.

    Most code in this tree is methods, so an analysis that cannot follow one is not
    following much: a class that read an untrusted config in one method and imported
    the result in another was accepted in full.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Loader:\n"
        "    def parse(self, path):\n"
        "        with open(path + '/config.json') as handle:\n"
        "            return json.load(handle)['model_type']\n"
        "    def load(self, path):\n"
        "        return importlib.import_module('a.' + self.parse(path))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_taint_reaches_a_method_through_an_instance_attribute(tmp_path):
    """The other half of the same shape: one method stores, another consumes."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Loader:\n"
        "    def read(self, path):\n"
        "        with open(path + '/config.json') as handle:\n"
        "            self.model_type = json.load(handle)['model_type']\n"
        "    def load(self):\n"
        "        return importlib.import_module('a.' + self.model_type)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_global_initialised_by_a_helper_is_tainted(tmp_path):
    """Module-level taint has to see the helper's return summary.

    Running the module pass once before the interprocedural fixpoint left the global
    trusted, because the summary for `parse_config` did not exist yet.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def parse_config():\n"
        "    with open('config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n"
        "MODEL_TYPE = parse_config()\n"
        "def load():\n"
        "    return importlib.import_module('a.' + MODEL_TYPE)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_weights_download_beside_an_unrelated_import_is_not_a_code_fetch(tmp_path):
    """Co-location is not correlation.

    Downloading weights and separately importing a fixed optional backend satisfied
    "both appear in this function", and none of the fetched bytes are executed.
    """
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    weights = snapshot_download(repo)\n"
        "    backend = importlib.import_module('my_backend')\n"
        "    return weights, backend\n",
    )
    assert "unpinned code fetch" not in _sinks(findings)


def test_a_stale_baseline_allowance_fails_the_gate(tmp_path, monkeypatch):
    """An allowance for a sink that is gone is an allowance a later change inherits.

    The key is the path, the qualname and a hash of the call, so restoring the identical
    call in the same place would consume it silently.
    """
    baseline = tmp_path / "untrusted_sinks_baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "comment": "test",
                "entries": {"nowhere.py::gone::importlib.import_module::deadbeefdeadbeef": 1},
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(L, "BASELINE_PATH", baseline)
    clean = tmp_path / "clean.py"
    clean.write_text("VALUE = 1\n", encoding = "utf-8")

    assert L.main(["--paths", str(clean)]) == 1
