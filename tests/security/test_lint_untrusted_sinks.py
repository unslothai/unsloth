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

    The allowance is keyed to the file being scanned. It used to name an unrelated file,
    which passed for the wrong reason: the check compared against every entry in the
    baseline regardless of what the run had looked at, so any allowance anywhere failed
    any scan. That is what made --paths unusable, and the scope filter means a test of
    staleness has to put the stale entry inside the scope to be testing staleness at all.
    """
    clean = tmp_path / "clean.py"
    clean.write_text("VALUE = 1\n", encoding = "utf-8")

    baseline = tmp_path / "untrusted_sinks_baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "comment": "test",
                "entries": {
                    f"{L._relative(clean.resolve())}::gone::importlib.import_module"
                    f"::deadbeefdeadbeef::cafecafecafecafe": 1
                },
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(L, "BASELINE_PATH", baseline)

    assert L.main(["--paths", str(clean)]) == 1


def test_an_allowance_does_not_survive_a_change_around_the_call(tmp_path, monkeypatch):
    """A baselined sink is re-opened when its enclosing function changes.

    The case that matters: the sink was accepted because something nearby validated its
    input. This analysis does not model validators, so weakening one leaves the path,
    the qualname, the sink and the call's own text identical. Without the function in
    the identity, the stale allowance would match and the gate would pass on exactly
    the regression it exists to catch.
    """
    guarded = (
        "import importlib, json, re\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = json.load(handle)['model_type']\n"
        "    if not re.fullmatch(r'[a-z0-9_]+', model_type):\n"
        "        raise ValueError('bad')\n"
        "    return importlib.import_module('a.' + model_type)\n"
    )
    weakened = guarded.replace(
        "    if not re.fullmatch(r'[a-z0-9_]+', model_type):\n        raise ValueError('bad')\n",
        "",
    )

    sample = tmp_path / "sample.py"
    baseline = tmp_path / "untrusted_sinks_baseline.json"
    monkeypatch.setattr(L, "BASELINE_PATH", baseline)
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)

    # Review and record the guarded version.
    sample.write_text(guarded, encoding = "utf-8")
    assert L.main(["--paths", str(sample), "--update"]) == 0
    assert L.main(["--paths", str(sample)]) == 0

    # Remove the guard. The call itself is untouched, so only the function differs.
    sample.write_text(weakened, encoding = "utf-8")
    assert L.main(["--paths", str(sample)]) == 1


def test_a_relative_import_resolves_to_the_containing_package(tmp_path):
    """`from .parser import parse` in `pkg.use` is `pkg.parser`, not `pkg.use.parser`.

    Getting this wrong meant the module index could not find the sibling helper, so
    taint returned by it never reached a sink in the caller: a whole class of
    first-party call edges was missing.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding = "utf-8")
    (package / "parser.py").write_text(
        "import json\n"
        "def parse(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    (package / "use.py").write_text(
        "import importlib\n"
        "from .parser import parse\n"
        "def load(path):\n"
        "    return importlib.import_module('a.' + parse(path))\n",
        encoding = "utf-8",
    )

    findings = L.scan([package], roots = [tmp_path])

    assert "importlib.import_module" in {f["sink"] for f in findings if f["tier"] == "A"}


def test_trust_remote_code_set_through_a_kwargs_subscript_is_reported(tmp_path):
    """`kwargs["trust_remote_code"] = True` then `from_pretrained(**kwargs)`.

    The keyword never appears at the call and the call checker cannot look inside an
    expanded dict, so this spelling was reported nowhere at all.
    """
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    kwargs = {}\n"
        "    kwargs['trust_remote_code'] = True\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n",
    )
    assert "trust_remote_code = True (dict item)" in _sinks(findings)


def test_a_long_assignment_chain_still_converges(tmp_path):
    """The old bound of eight silently stopped while taint was still propagating."""
    chain = "".join(f"    x{i} = x{i + 1}\n" for i in range(12))
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(path):\n"
        "    importlib.import_module('a.' + x0)\n"
        + chain
        + "    x12 = json.loads(open(path + '/config.json').read())['model_type']\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_unconverged_body_fails_the_gate(tmp_path, monkeypatch):
    """A partial answer must not be reported as a clean one.

    The bound is a safety valve. If it expires while taint is still growing, the result
    for that body is incomplete, and saying nothing would be the one failure mode this
    script exists to avoid.
    """
    monkeypatch.setattr(L, "_LOCAL_BOUND", 2)
    chain = "".join(f"    x{i} = x{i + 1}\n" for i in range(8))
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(path):\n"
        "    importlib.import_module('a.' + x0)\n"
        + chain
        + "    x8 = json.loads(open(path + '/config.json').read())['model_type']\n",
    )
    assert "analysis did not converge" in _sinks(findings)


def test_settled_local_taint_reaches_a_callee(tmp_path):
    """Taint discovered while settling a body has to reach the functions it calls.

    Settling only at reporting time meant the pending tainted parameter for `execute`
    was written after the interprocedural fixpoint had finished, so nothing ever looked
    inside `execute` again and the sink there was missed entirely.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    subprocess.run(command)\n"
        "def load(paths):\n"
        "    command = ['echo']\n"
        "    for path in paths:\n"
        "        execute(command)\n"
        "        command = json.loads(open(path + '/config.json').read())['cmd']\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_nested_import_root_is_indexed_under_both_names(tmp_path):
    """`studio/backend` modules import each other by top-level name.

    The index used to stop at the first matching root, and the repo root is listed
    first, so those files existed only as `studio.backend.utils.x` and the form they
    actually use, `from utils.x import f`, resolved to nothing. Taint therefore never
    crossed a single backend file boundary, which is the one thing the index is for.
    """
    backend = tmp_path / "studio" / "backend"
    (backend / "utils").mkdir(parents = True)
    (backend / "utils" / "__init__.py").write_text("", encoding = "utf-8")
    (backend / "utils" / "reader.py").write_text(
        "import json\n"
        "def read(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    (backend / "worker.py").write_text(
        "import importlib\n"
        "from utils.reader import read\n"
        "def load(path):\n"
        "    return importlib.import_module('a.' + read(path))\n",
        encoding = "utf-8",
    )

    findings = L.scan([backend], roots = [tmp_path, backend])

    assert "importlib.import_module" in {f["sink"] for f in findings if f["tier"] == "A"}


def test_a_handle_opened_on_a_downloaded_path_is_tainted(tmp_path):
    """The ordinary pickle shape: open the download, then load it."""
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    downloaded = hf_hub_download(repo, 'weights.pkl')\n"
        "    with open(downloaded, 'rb') as handle:\n"
        "        return pickle.load(handle)\n",
    )
    assert "pickle.load" in _sinks(findings)


@pytest.mark.parametrize(
    "expression",
    [
        "[name for name in json.load(handle)['types']]",
        "{name for name in json.load(handle)['types']}",
        "list(name for name in json.load(handle)['types'])",
        "{k: v for k, v in json.load(handle)['types'].items()}",
    ],
)
def test_taint_survives_a_comprehension(tmp_path, expression):
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(path):\n"
        "    handle = open(path + '/config.json')\n"
        f"    names = {expression}\n"
        "    return importlib.import_module('a.' + str(names))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_tainted_keyword_binds_to_a_star_kwargs_parameter(tmp_path):
    """A helper that forwards its options as **kwargs still receives the value.

    Matching only on the keyword's own name dropped the taint for every such helper.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def run(**kwargs):\n"
        "    return importlib.import_module(kwargs['name'])\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        model_type = json.load(handle)['model_type']\n"
        "    return run(name = model_type)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_getattr_on_an_aliased_module_is_reported(tmp_path):
    """`import torch as t` then `getattr(t, name)` is the same sink."""
    findings = _scan(
        tmp_path,
        "import json\nimport torch as t\n"
        "def build(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        name = json.load(handle)['dtype']\n"
        "    return getattr(t, name)\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_an_unsettled_interprocedural_fixpoint_fails_the_gate(tmp_path, monkeypatch):
    """The cross-function bound needs the same honesty as the local one."""
    monkeypatch.setattr(L, "_GLOBAL_BOUND", 1)
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def a(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n"
        "def b(path):\n    return a(path)\n"
        "def c(path):\n    return b(path)\n"
        "def d(path):\n    return importlib.import_module('x.' + c(path))\n",
    )
    assert "analysis did not converge" in _sinks(findings)


def test_a_stronger_reason_is_not_overwritten_by_a_later_weaker_call(tmp_path):
    """Two call sites, one proving taint and one only suggesting it.

    Binding a callee's parameters assigned unconditionally, so whichever call site the
    walk reached last decided the tier. `run(json.load(...))` followed by
    `run(model_type)` left the parameter marked only "untrusted parameter name", and the
    sink inside `run` came out tier B: reported for a human to read, but not gating.
    The order in the sample is the one that used to lose, weaker call last.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def run(name):\n"
        "    return importlib.import_module('x.' + name)\n"
        "def first(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return run(json.load(handle)['model_type'])\n"
        "def second(model_type):\n"
        "    return run(model_type)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_tainted_global_imported_from_another_module_is_tainted(tmp_path):
    """`from producer import MODEL_TYPE`, where the producer read it off a config.

    The global lookup only ever checked the consuming file's own module-level names, so
    each half looked clean on its own: the read is in one file and the sink in the other,
    and nothing was reported anywhere. This is the cross-file guarantee the whole script
    claims, in its shortest possible form.
    """
    (tmp_path / "producer.py").write_text(
        "import json\n"
        "with open('config.json') as handle:\n"
        "    MODEL_TYPE = json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nfrom producer import MODEL_TYPE\n"
        "def load():\n"
        "    return importlib.import_module('transformers.models.' + MODEL_TYPE)\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_a_remote_code_assignment_is_keyed_to_its_enclosing_function(tmp_path):
    """An allowance for `trust_remote_code = True` has to die with the consent around it.

    The three assignment spellings were recorded under the synthetic qualnames `<assign>`,
    `<dict>` and `<item>`, which no function is called, so the context digest in the
    baseline key was always the empty string. A reviewed assignment therefore kept its
    allowance word for word after the consent check beside it was deleted, which is the
    one regression the context digest exists to catch.
    """
    findings = _scan(
        tmp_path,
        "def load(name, approved):\n"
        "    if not approved:\n"
        "        raise ValueError('refused')\n"
        "    trust_remote_code = True\n"
        "    return trust_remote_code\n",
    )
    assignments = [f for f in findings if f["sink"].startswith("trust_remote_code = True (assign")]
    assert len(assignments) == 1
    assert assignments[0]["qualname"] == "load"
    assert assignments[0]["context"]

    weakened = _scan(
        tmp_path,
        "def load(name, approved):\n"
        "    trust_remote_code = True\n"
        "    return trust_remote_code\n",
        name = "weakened.py",
    )
    other = [f for f in weakened if f["sink"].startswith("trust_remote_code = True (assign")]
    assert len(other) == 1
    # Same path, same qualname, same call text, and the key still has to differ.
    assert other[0]["hash"] == assignments[0]["hash"]
    assert L._baseline_key(other[0]).replace("weakened.py", "sample.py") != L._baseline_key(
        assignments[0]
    )


def test_one_alias_bound_to_two_modules_fails_closed(tmp_path):
    """`import json as codec` here, `import pickle as codec` there.

    Imports were a file-wide name -> target map, so the second binding overwrote the
    first and `codec.load` inside `load_json` canonicalised to `pickle.load`. That is not
    merely imprecise: the JSON read stopped counting as an untrusted source, so the value
    was clean and the dynamic import below it was accepted. Both bindings are kept now
    and the strongest answer wins.
    """
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "def load_json(path):\n"
        "    import json as codec\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return importlib.import_module('x.' + codec.load(handle)['model_type'])\n"
        "def load_cache(path):\n"
        "    import pickle as codec\n"
        "    with open(path, 'rb') as handle:\n"
        "        return codec.load(handle)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_unaliased_dotted_import_resolves_its_helper(tmp_path):
    """`import pkg.parser` binds `pkg`, so the tail must not be appended twice.

    Recording `pkg -> pkg.parser` made `pkg.parser.parse` resolve as
    `pkg.parser.parser.parse`, which is nothing, so the helper was outside the analysis:
    taint it returned never reached the caller's sink.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding = "utf-8")
    (package / "parser.py").write_text(
        "import json\n"
        "def parse(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    user = tmp_path / "use.py"
    user.write_text(
        "import importlib\nimport pkg.parser\n"
        "def load(path):\n"
        "    return importlib.import_module('x.' + pkg.parser.parse(path))\n",
        encoding = "utf-8",
    )
    findings = L.scan([package / "parser.py", user], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_taint_propagates_through_an_assignment_expression(tmp_path):
    """The walrus binds a name, and only the statement forms were handled.

    `if (cfg := json.loads(blob)):` left `cfg` out of the local taint entirely, so a
    dynamic import reading it a line later was reported as clean.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    if (cfg := json.loads(blob)):\n"
        "        return importlib.import_module('x.' + cfg['model_type'])\n"
        "    return None\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_only_the_fetch_that_reaches_the_import_is_a_code_fetch(tmp_path):
    """Two downloads in one function, and only one of them is executed.

    `reached` was a per-function boolean, so a function that imports a pinned checkout
    and separately downloads weights had the weights fetch reported as executable code.
    The pinned one carries a revision and is never a candidate, which is what made the
    boolean report the wrong call.
    """
    findings = _scan(
        tmp_path,
        "import importlib, os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, weights_repo):\n"
        "    code = snapshot_download(repo, revision = 'a' * 40)\n"
        "    weights = snapshot_download(weights_repo)\n"
        "    sys.path.insert(0, os.path.join(code, 'src'))\n"
        "    importlib.import_module('vendored')\n"
        "    return weights\n",
    )
    fetches = [f for f in findings if f["sink"] == "unpinned code fetch"]
    assert fetches == [], [f["argument"] for f in fetches]


def test_an_incomplete_analysis_cannot_be_baselined(tmp_path, monkeypatch):
    """`--update` has to refuse, and the gate has to ignore a hand-written allowance.

    An exhausted bound emits a synthetic "analysis did not converge" finding. The writer
    stored it like a reviewed sink and reported success, so every later run matched the
    allowance and printed OK while the scanner was still saying its own answer was
    partial. That is the one outcome this script exists to prevent.
    """
    monkeypatch.setattr(L, "_LOCAL_BOUND", 1)
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        a = json.load(handle)['model_type']\n"
        "    b = a\n    c = b\n    d = c\n    e = d\n"
        "    return importlib.import_module('x.' + e)\n",
    )
    # Spelled out rather than read off the module, so that on a tree without the fix
    # this test fails on the behaviour below and not on a missing constant.
    incomplete = [f for f in findings if f["sink"] == "analysis did not converge"]
    assert incomplete

    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(L, "BASELINE_PATH", baseline)
    with pytest.raises(SystemExit) as refused:
        L._write_baseline(findings)
    assert refused.value.code == 2
    assert not baseline.exists()

    # And an allowance someone writes by hand must not silence it either.
    baseline.write_text(
        json.dumps({"entries": {L._baseline_key(incomplete[0]): 1}}),
        encoding = "utf-8",
    )
    assert "analysis did not converge" in {f["sink"] for f in L._unbaselined(findings)}


def test_a_bare_call_resolves_a_nested_helper(tmp_path):
    """A helper defined inside a function is indexed under its qualified name.

    `def execute` inside `def run` is `run.execute`, while the call to it is the bare
    `execute`, so looking the bare name up on its own found nothing: taint neither entered
    the helper nor came back out, and a sink inside it was invisible.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def run(blob):\n"
        "    def execute(command):\n"
        "        return subprocess.run(command)\n"
        "    return execute(json.loads(blob)['cmd'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_inner_helper_is_preferred_to_a_module_level_namesake(tmp_path):
    """Innermost first, which is how Python resolves the name.

    Resolving outwards-in would attribute the call to the module-level function and look
    for the sink in the wrong body.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    return command\n"
        "def run(blob):\n"
        "    def execute(command):\n"
        "        return subprocess.run(command)\n"
        "    return execute(json.loads(blob)['cmd'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_taint_survives_an_await(tmp_path):
    """`body = await request.json()` is Await(Call(...)), not a bare Call.

    Only the top-level Call was recognised, so every async read of a request body or a
    file came out clean and a dynamic import below it was accepted.
    """
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "async def handle(request):\n"
        "    body = await request.json()\n"
        "    return importlib.import_module('x.' + body['model_type'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_stronger_attribute_reason_survives_a_later_method(tmp_path):
    """Two methods assigning the same attribute, and method order decided the tier.

    The pending write was unconditional, so a tier-B assignment in a method visited later
    overwrote a tier-A one from another method, purely alphabetically, and the sink
    reading that attribute stopped gating. a_config and z_parameter are named so the
    weaker one is visited second, which is the order that used to lose.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def a_config(self, blob):\n"
        "        self.command = json.loads(blob)['cmd']\n"
        "    def z_parameter(self, model_path):\n"
        "        self.command = model_path\n"
        "    def go(self):\n"
        "        return subprocess.run(self.command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_expanded_dictionary_binds_the_named_parameters(tmp_path):
    """`execute(**json.loads(blob))` can supply any named parameter.

    The keyword's arg is None, and the only branch that handled that needed the callee to
    declare **kwargs. A callee declaring `command` directly bound nothing, so it ran a
    value out of the parsed dictionary with nothing reported anywhere.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command, timeout = 5):\n"
        "    return subprocess.run(command, timeout = timeout)\n"
        "def load(blob):\n"
        "    return execute(**json.loads(blob))\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_partial_scan_does_not_call_every_other_allowance_stale(tmp_path, monkeypatch):
    """--paths on one clean file must not exit 1 over files it never looked at.

    The stale check ran against the whole baseline, so scanning a single file reported
    every allowance belonging to every other file as unused. That makes the option
    useless for the one thing it is for, checking a file you just edited.
    """
    sample = tmp_path / "sample.py"
    sample.write_text("x = 1\n", encoding = "utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"entries": {"somewhere/else.py::f::subprocess.run::abc::def": 1}}),
        encoding = "utf-8",
    )
    monkeypatch.setattr(L, "BASELINE_PATH", baseline)

    findings = L.scan([sample], roots = [tmp_path])
    scoped = L._stale_allowances(findings, scope = {L._relative(sample.resolve())})
    assert scoped == []
    # Unscoped is the old behaviour, and it is what made the option unusable.
    assert L._stale_allowances(findings) != []


def test_both_bindings_of_a_reused_alias_are_analysed(tmp_path):
    """One alias, two first-party parsers, and only one of them is dirty.

    Taking the first resolvable target meant the sorted-clean one was chosen for calls in
    both functions, so a value from the dirty parser reached a sink with no finding. Every
    binding is analysed now, which is the same fail-closed choice the source and sink
    tables make. a_clean sorts first, which is the order that used to lose.
    """
    (tmp_path / "a_clean.py").write_text("def parse(path):\n    return 'llama'\n", encoding = "utf-8")
    (tmp_path / "z_dirty.py").write_text(
        "import json\n"
        "def parse(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    user = tmp_path / "use.py"
    user.write_text(
        "import importlib\n"
        "def clean(path):\n"
        "    from a_clean import parse as codec\n"
        "    return importlib.import_module('x.' + codec(path))\n"
        "def dirty(path):\n"
        "    from z_dirty import parse as codec\n"
        "    return importlib.import_module('y.' + codec(path))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "a_clean.py", tmp_path / "z_dirty.py", user], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_a_file_under_two_roots_resolves_by_its_longer_name(tmp_path):
    """The same file is indexed under the repository root and under studio/backend.

    `files[file]` keeps only the shorter name, so stripping it off the longer spelling
    left an empty qualname: the callee resolved to no function at all and taint returned
    by a backend helper never reached a caller that imported it by the full path. The
    qualname now comes from the spelling that actually matched.
    """
    backend = tmp_path / "studio" / "backend" / "utils"
    backend.mkdir(parents = True)
    for marker in (tmp_path / "studio", tmp_path / "studio" / "backend", backend):
        (marker / "__init__.py").write_text("", encoding = "utf-8")
    (backend / "parser.py").write_text(
        "import json\n"
        "def parse(path):\n"
        "    with open(path + '/config.json') as handle:\n"
        "        return json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    caller = tmp_path / "cli.py"
    caller.write_text(
        "import importlib\n"
        "from studio.backend.utils.parser import parse\n"
        "def load(path):\n"
        "    return importlib.import_module('x.' + parse(path))\n",
        encoding = "utf-8",
    )
    findings = L.scan(
        [backend / "parser.py", caller],
        roots = [tmp_path, tmp_path / "studio" / "backend"],
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_starred_argument_binds_the_parameters_it_spreads_over(tmp_path):
    """`execute(*json.loads(blob))` populates more than the first parameter.

    Treated as one positional it tainted only `prefix`, so the parsed second element
    reaching subprocess.run was reported nowhere.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(prefix, command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    return execute(*json.loads(blob))\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_call_through_a_local_class_name_resolves(tmp_path):
    """`Parser.parse(blob)` is already indexed as `Parser.parse`.

    The head is neither an import nor self, so the callee was rejected and a local static
    or class method sat outside the analysis: it could return a parsed config straight
    into an import_module with nothing reported.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Parser:\n"
        "    @staticmethod\n"
        "    def parse(blob):\n"
        "        return json.loads(blob)['model_type']\n"
        "def load(blob):\n"
        "    return importlib.import_module('x.' + Parser.parse(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_taint_survives_a_subscript_assignment(tmp_path):
    """`settings["module"] = json.loads(blob)` is how configuration gets assembled.

    The assignment dispatcher ignored a Subscript target, so a dictionary built up key by
    key arrived clean at the sink.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    settings = {}\n"
        "    settings['module'] = json.loads(blob)['model_type']\n"
        "    return importlib.import_module('x.' + settings['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_declared_global_assigned_in_a_function_is_module_taint(tmp_path):
    """`global MODEL_TYPE` makes the write module state, not a local.

    Module taint was produced only by module-level statements, so a function that parsed a
    config into a declared global and another that read it were each clean on their own.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "MODEL_TYPE = ''\n"
        "def parse(blob):\n"
        "    global MODEL_TYPE\n"
        "    MODEL_TYPE = json.loads(blob)['model_type']\n"
        "def load():\n"
        "    return importlib.import_module('x.' + MODEL_TYPE)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_method_on_a_locally_constructed_object_resolves(tmp_path):
    """`parser = Parser()` then `parser.parse(blob)`.

    The head is a local variable, so nothing resolved the callee and the method sat
    outside the analysis: it could return a parsed config into an import_module unseen.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['model_type']\n"
        "def load(blob):\n"
        "    parser = Parser()\n"
        "    return importlib.import_module('x.' + parser.parse(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_method_called_before_the_constructor_line_still_resolves(tmp_path):
    """The instance map is carried across passes, like the reasons are.

    A single ordered traversal would not have the binding yet when the call appears first,
    which is a real shape in a loop.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['model_type']\n"
        "def load(blobs):\n"
        "    parser = None\n"
        "    out = []\n"
        "    for blob in blobs:\n"
        "        if parser is not None:\n"
        "            out.append(importlib.import_module('x.' + parser.parse(blob)))\n"
        "        parser = Parser()\n"
        "    return out\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_module_qualified_global_is_tainted(tmp_path):
    """`import producer` then `producer.MODEL_TYPE`.

    The imported-global fix handled only `from producer import MODEL_TYPE`, so the
    module-qualified spelling of the same tainted global fell through to the clean base
    name `producer`.
    """
    (tmp_path / "producer.py").write_text(
        "import json\n"
        "with open('config.json') as handle:\n"
        "    MODEL_TYPE = json.load(handle)['model_type']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nimport producer\n"
        "def load():\n"
        "    return importlib.import_module('x.' + producer.MODEL_TYPE)\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_taint_survives_a_decode(tmp_path):
    """`urlopen(url).read().decode().strip()` is the ordinary network read.

    decode was missing from the taint-preserving operations, so the bytes read clean the
    moment they became a str and everything chained after it inherited that.
    """
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "from urllib.request import urlopen\n"
        "def load(url):\n"
        "    name = urlopen(url).read().decode().strip()\n"
        "    return importlib.import_module('x.' + name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_locally_aliased_sink_is_still_a_sink(tmp_path):
    """`loader = importlib.import_module` then `loader(name)`.

    Matched only under the textual name `loader`, so an ordinary local alias walked past
    the gate. A reference to a sink is recorded, as distinct from a call to one.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    loader = importlib.import_module\n"
        "    return loader('x.' + json.loads(blob)['model_type'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_aliased_getattr_is_still_the_getattr_sink(tmp_path):
    """`from builtins import getattr as resolve`.

    The holder alias was already handled; the alias of this sink itself was checked
    against the raw spelling and skipped.
    """
    findings = _scan(
        tmp_path,
        "import json, transformers\n"
        "from builtins import getattr as resolve\n"
        "def build(blob):\n"
        "    return resolve(transformers, json.loads(blob)['cls'])\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_a_locally_aliased_getattr_is_still_the_getattr_sink(tmp_path):
    """The same alias made locally rather than at the import."""
    findings = _scan(
        tmp_path,
        "import json, transformers\n"
        "def build(blob):\n"
        "    resolve = getattr\n"
        "    return resolve(transformers, json.loads(blob)['cls'])\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_taint_enters_an_inherited_method_through_super(tmp_path):
    """`super().execute(tainted)` reduces to the bare method name.

    None of the handled shapes matched, so taint never entered an inherited method and an
    overridden helper that passes its argument to a sink was invisible from every
    subclass call site.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Base:\n"
        "    def execute(self, name):\n"
        "        return importlib.import_module('x.' + name)\n"
        "class Child(Base):\n"
        "    def execute(self, blob):\n"
        "        return super().execute(json.loads(blob)['model_type'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_taint_survives_a_pathlib_read_bytes(tmp_path):
    """`pickle.loads(Path(download).read_bytes())` is the short form of the handle shape.

    read_text was in the method list and read_bytes was not, so the bytes lost the taint
    the path already carried.
    """
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    blob = Path(hf_hub_download(repo, 'state.pkl')).read_bytes()\n"
        "    return pickle.loads(blob)\n",
    )
    assert "pickle.loads" in _sinks(findings) or "pickle.load" in _sinks(findings)


def test_a_method_on_an_imported_class_instance_resolves(tmp_path):
    """`from producer import Parser` then `parser = Parser()`.

    Accepting only classes declared in the same file left an imported first-party class
    unresolvable, so its methods were outside the analysis exactly as local ones had been.
    """
    (tmp_path / "producer.py").write_text(
        "import json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['model_type']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nfrom producer import Parser\n"
        "def load(blob):\n"
        "    parser = Parser()\n"
        "    return importlib.import_module('x.' + parser.parse(blob))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_an_attribute_written_through_a_constructed_local_is_tainted(tmp_path):
    """`runner.module = parsed` then `runner.run()` reading self.module.

    Only the literal `self` receiver was recognised, so the write was discarded and the
    method that reads it saw a clean value.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Runner:\n"
        "    def run(self):\n"
        "        return importlib.import_module('x.' + self.module)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    runner.module = json.loads(blob)['module']\n"
        "    return runner.run()\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_expanded_dictionary_at_a_direct_sink_is_reported(tmp_path):
    """`importlib.import_module(**json.loads(blob))`.

    The expansion can supply the sink's own argument, and the earlier handling covered
    only propagation into a first-party callee, so a direct sink skipped it.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    return importlib.import_module(**json.loads(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_overflow_positional_arguments_bind_to_the_vararg(tmp_path):
    """`def execute(*commands)` takes every positional, not just the first.

    The flattened parameter list names the vararg once, so anything past that position
    was dropped and a value executed out of commands[1] was reported nowhere.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(*commands):\n"
        "    return subprocess.run(commands[1])\n"
        "def load(blob):\n"
        "    return execute('safe', json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_annotated_constructor_is_tracked(tmp_path):
    """`parser: Parser = Parser()` is the same construction as the plain form."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['model_type']\n"
        "def load(blob):\n"
        "    parser: Parser = Parser()\n"
        "    return importlib.import_module('x.' + parser.parse(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_tainted_subprocess_executable_is_reported(tmp_path):
    """`executable=` names the program that actually runs.

    A fixed argv with a tainted executable runs the tainted one, and only `args` was
    inspected.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    return subprocess.run(['safe-command'], executable = json.loads(blob)['binary'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_module_level_sink_alias_reaches_a_function(tmp_path):
    """`loader = importlib.import_module` at module scope.

    The alias map lived on the visitor that scanned the module body and was discarded
    before any function was scanned, so a function calling it read as clean while the
    identical local alias was caught.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return loader('x.' + json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_module_level_instance_reaches_a_function(tmp_path):
    """The same gap for a module-scope construction, which shares the mechanism."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['model_type']\n"
        "parser = Parser()\n"
        "def load(blob):\n"
        "    return importlib.import_module('x.' + parser.parse(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_container_mutated_by_update_is_tainted(tmp_path):
    """`settings.update(json.loads(blob))` is an assignment into the container.

    Only assignments were handled, so a configuration filled in by a mutating call
    arrived clean at the sink.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    settings = {}\n"
        "    settings.update(json.loads(blob))\n"
        "    return importlib.import_module(settings['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_list_mutated_by_append_is_tainted(tmp_path):
    """The same shape for a command list built up by appending."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    cmd = ['tool']\n"
        "    cmd.append(json.loads(blob)['flag'])\n"
        "    return subprocess.run(cmd)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_trust_remote_code_through_the_dict_constructor_is_reported(tmp_path):
    """`kwargs = dict(trust_remote_code = True)` splats into a loader like the literal.

    It is a Call rather than a Dict, so neither the dict scan nor the call-site keyword
    check saw it.
    """
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    kwargs = dict(trust_remote_code = True)\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_taint_flows_out_of_a_generator(tmp_path):
    """A generator's output is what it yields, and only Return was handled.

    So a helper that iterates a parsed config and yields each name had no tainted summary
    and every caller looping over it read clean.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def parse(blob):\n"
        "    for name in json.loads(blob)['modules']:\n"
        "        yield name\n"
        "def load(blob):\n"
        "    for name in parse(blob):\n"
        "        importlib.import_module('x.' + name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_taint_flows_out_of_a_yield_from(tmp_path):
    """The delegating form has the same output."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def inner(blob):\n"
        "    yield json.loads(blob)['model_type']\n"
        "def parse(blob):\n"
        "    yield from inner(blob)\n"
        "def load(blob):\n"
        "    for name in parse(blob):\n"
        "        importlib.import_module('x.' + name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_container_copy_is_not_a_sanitiser(tmp_path):
    """`cfg.copy()` read clean, so an ordinary defensive copy laundered a parsed config."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    cfg = json.loads(blob)\n"
        "    copied = cfg.copy()\n"
        "    return importlib.import_module(copied['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_deepcopy_is_not_a_sanitiser(tmp_path):
    """The module-level function form of the same operation."""
    findings = _scan(
        tmp_path,
        "import copy, importlib, json\n"
        "def load(blob):\n"
        "    cfg = copy.deepcopy(json.loads(blob))\n"
        "    return importlib.import_module(cfg['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_path_passed_to_open_by_keyword_still_taints_the_handle(tmp_path):
    """open takes its path as `file=`, and only positions were checked.

    So `open(file = downloaded, mode = "rb")` handed back a clean handle and the declared
    pickle sink below it never fired.
    """
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    path = snapshot_download(repo)\n"
        "    with open(file = path, mode = 'rb') as handle:\n"
        "        return pickle.load(handle)\n",
    )
    assert "pickle.load" in _sinks(findings)


def test_a_module_scope_instance_of_an_imported_class_resolves(tmp_path):
    """`from producer import Parser` then a module-level `parser = Parser()`.

    The module-scope collector accepted only classes declared in the same file, so this
    instance carried no type, `parser.parse(blob)` resolved to nothing and the sink fed
    by its return value was reported nowhere. The function-local form was already handled,
    which is exactly the drift the shared resolver removes.
    """
    (tmp_path / "producer.py").write_text(
        "import json\n"
        "class Parser:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['module']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nfrom producer import Parser\n"
        "parser = Parser()\n"
        "def load(blob):\n"
        "    return importlib.import_module(parser.parse(blob))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_an_unbound_method_call_does_not_shift_its_arguments(tmp_path):
    """`Runner.execute(runner, "safe", command)` passes the receiver by hand.

    Attribute syntax alone shifted every argument one place right, so the tainted third
    argument landed past the end of the parameter list and was dropped, and the
    subprocess call reading it was reported nowhere.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, prefix, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    return Runner.execute(runner, 'safe', json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_bound_method_call_still_shifts_its_arguments(tmp_path):
    """The ordinary form has an implicit receiver, so the offset has to stay.

    Dropping it everywhere would have bound the first real argument to `self` and shifted
    the rest left, which loses the finding just as surely in the shape nearly all of this
    tree is written in.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, prefix, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    return runner.execute('safe', json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_alias_of_a_first_party_callable_is_followed(tmp_path):
    """`runner = execute` then `runner(...)`.

    The alias is neither an indexed local function named `runner` nor an import target,
    so the callee did not resolve, taint never entered the helper and the subprocess call
    inside it walked past the gate.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = execute\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_alias_of_an_imported_callable_is_followed(tmp_path):
    """The same alias, on a helper imported from another first-party module."""
    (tmp_path / "producer.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import execute\n"
        "runner = execute\n"
        "def load(blob):\n"
        "    return runner(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_package_re_export_is_followed_to_its_defining_module(tmp_path):
    """`from pkg import parse`, where `pkg/__init__.py` re-exports `.parser.parse`.

    Resolution stopped at the package `__init__.py` and handed back a qualname that file
    does not define, so the real helper was never analysed and taint stopped at the
    package API, which is how most of this tree imports its own helpers.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("from .parser import parse\n", encoding = "utf-8")
    (package / "parser.py").write_text(
        "import json\ndef parse(blob):\n    return json.loads(blob)['module']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nfrom pkg import parse\n"
        "def load(blob):\n"
        "    return importlib.import_module(parse(blob))\n",
        encoding = "utf-8",
    )
    findings = L.scan([package / "__init__.py", package / "parser.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_an_inherited_self_call_resolves_through_the_base(tmp_path):
    """`self.parse(...)` on a method the child inherits rather than overrides.

    Only the explicit `super().parse(...)` form was followed, so a base method returning
    a parsed value fed the child's sink with nothing reported.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Base:\n"
        "    def parse(self, blob):\n"
        "        return json.loads(blob)['module']\n"
        "class Child(Base):\n"
        "    def load(self, blob):\n"
        "        return importlib.import_module(self.parse(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_attribute_written_through_cls_is_tracked(tmp_path):
    """Classmethods use `cls`, and the attribute key accepted only `self`.

    So one classmethod storing a parsed value and another executing it both got an empty
    key, and the executable value was reported as clean.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    command = None\n"
        "    @classmethod\n"
        "    def store(cls, blob):\n"
        "        cls.command = json.loads(blob)['command']\n"
        "    @classmethod\n"
        "    def execute(cls):\n"
        "        return subprocess.run(cls.command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_parameter_inherits_taint_from_its_default(tmp_path):
    """`def execute(argv = CONFIG["argv"])` runs the default when the caller omits it.

    A parameter was tainted only by an explicit caller or by its name, so the omitted
    argument executed a parsed global while the parameter itself stayed clean.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "from pathlib import Path\n"
        "CONFIG = json.loads(Path('downloaded.json').read_text())\n"
        "def execute(argv = CONFIG['argv']):\n"
        "    return subprocess.run(argv)\n"
        "def load():\n"
        "    return execute()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_async_for_target_is_tainted(tmp_path):
    """The async loop fell through to generic traversal, leaving its target clean."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "async def parse(blob):\n"
        "    for name in json.loads(blob)['modules']:\n"
        "        yield name\n"
        "async def load(blob):\n"
        "    async for name in parse(blob):\n"
        "        importlib.import_module('x.' + name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_deepcopy_by_keyword_is_not_a_sanitiser(tmp_path):
    """`copy.deepcopy` takes its input as `x=`, and only positions were checked."""
    findings = _scan(
        tmp_path,
        "import copy, importlib, json\n"
        "def load(blob):\n"
        "    cfg = copy.deepcopy(x = json.loads(blob))\n"
        "    return importlib.import_module(cfg['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_next_preserves_a_generators_taint(tmp_path):
    """`next(parse(blob))` is how a generator's first value is taken.

    The return summary the yield handling produces was dropped again at the consumer,
    because `next` was not among the operations that pass their input through.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def parse(blob):\n"
        "    yield json.loads(blob)['module']\n"
        "def load(blob):\n"
        "    return importlib.import_module(next(parse(blob)))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_nested_functions_locals_stay_out_of_the_enclosing_scope(tmp_path):
    """A nested helper's assignments used to taint the enclosing function's names.

    The nested body is indexed and analysed under its own qualname, so keeping its writes
    produced a finding on an outer name that only ever held a fixed value. A gate that
    reports those is a gate that gets switched off.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    def helper():\n"
        "        command = json.loads(blob)['command']\n"
        "        return len(command)\n"
        "    helper()\n"
        "    command = ['ls', '-l']\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_a_closure_reading_a_tainted_enclosing_local_still_reports(tmp_path):
    """Scoping the nested writes must not stop the nested body being scanned at all.

    The read of an enclosing tainted local inside a closure is a real flow, so this is
    the case that pins the nested body is still walked rather than skipped.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = json.loads(blob)['command']\n"
        "    def helper():\n"
        "        return subprocess.run(command)\n"
        "    return helper()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_classmethod_keeps_its_implicit_receiver(tmp_path):
    """`Runner.execute(command)` on a classmethod: Python still supplies `cls`.

    The unbound-call fix read that spelling as an explicit receiver, so the tainted
    argument bound to `cls` and the subprocess call inside the classmethod was missed.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    @classmethod\n"
        "    def execute(cls, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    return Runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_local_reconstructed_from_a_second_class_resolves_both(tmp_path):
    """`runner = Safe()` then `runner = Dirty()` is valid sequential code.

    Keeping only the first constructor resolved calls to the wrong class, so the sink in
    the one that actually runs received tainted data with nothing reported.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Safe:\n"
        "    def execute(self, command):\n"
        "        return len(command)\n"
        "class Dirty:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Safe()\n"
        "    runner = Dirty()\n"
        "    return runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_sink_inside_a_comprehension_is_reported(tmp_path):
    """`[import_module(n) for n in json.loads(blob)["modules"]]`.

    One of the most common shapes in this tree, and the `for` target was never bound, so
    the sink in the element read it as clean. Taking the comprehension's whole value was
    already handled; it was the sink INSIDE one that was invisible.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    return [importlib.import_module(n) for n in json.loads(blob)['modules']]\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_generator_expression_target_is_bound(tmp_path):
    """The lazy form is the one used inside `join`, which is how argv strings get built."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    joined = ' '.join(part for part in json.loads(blob)['argv'])\n"
        "    return subprocess.run(joined, shell = True)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_match_capture_carries_the_subjects_taint(tmp_path):
    """`case {"module": name}` is how a parsed config gets destructured."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    match json.loads(blob):\n"
        "        case {'module': name}:\n"
        "            return importlib.import_module(name)\n"
        "    return None\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_sink_in_a_class_body_is_scanned(tmp_path):
    """A class body runs at import time, and it was skipped outright.

    Module-level statements were scanned and function bodies were scanned, so a sink
    sitting between the two was the one place nothing looked at.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "from pathlib import Path\n"
        "class Loader:\n"
        "    name = json.loads(Path('downloaded.json').read_text())['module']\n"
        "    mod = importlib.import_module(name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_keyword_format_argument_keeps_its_taint(tmp_path):
    """`"x.{n}".format(n = parsed)` is the named spelling of the same interpolation."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    return importlib.import_module('x.{n}'.format(n = json.loads(blob)['m']))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_constructor_argument_reaches_init(tmp_path):
    """`Runner(parsed)` is where a value gets parked on the instance.

    A class name used as a callee resolved to nothing, so a constructor that stores an
    argument on `self` and a method that later executes it were both outside the
    analysis. The receiver offset has to apply here too, with no receiver written out.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def __init__(self, command):\n"
        "        self.command = command\n"
        "    def go(self):\n"
        "        return subprocess.run(self.command)\n"
        "def load(blob):\n"
        "    runner = Runner(json.loads(blob)['command'])\n"
        "    return runner.go()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_bare_call_to_an_imported_helper_still_resolves(tmp_path):
    """The guard for the constructor branch, which claimed calls it had no business in.

    `resolve_construction` resolves an imported NAME, not necessarily a class, so taking
    whatever the index returned sent every bare call to an imported helper to an
    `__init__` that does not exist and the import resolution below never ran. Two dozen
    real findings went quiet, which is how this was caught.
    """
    (tmp_path / "producer.py").write_text(
        "import json\ndef parse(blob):\n    return json.loads(blob)['module']\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import importlib\nfrom producer import parse\n"
        "def load(blob):\n"
        "    return importlib.import_module(parse(blob))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_getattr_on_an_imported_module_is_a_namespace(tmp_path):
    """An alias bound by `import x` is a module by construction, not by its name.

    The name heuristic caught the modules it happens to list and missed every other one
    this tree imports, so reflection with a parsed name on them was reported nowhere.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "import xml.etree.ElementTree as namespace\n"
        "def load(blob):\n"
        "    return getattr(namespace, json.loads(blob)['fn'])\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_getattr_on_a_plain_local_object_stays_quiet(tmp_path):
    """The quiet half: `getattr(config, field)` is a dict-ish read and is everywhere.

    Widening to imported modules must not widen to every receiver, or the gate becomes
    noise and gets switched off.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "def load(blob):\n"
        "    config = json.loads(blob)\n"
        "    settings = object()\n"
        "    return getattr(settings, config['field'])\n",
    )
    assert "getattr(module, ...)" not in _sinks(findings)


def test_a_lookup_in_a_literal_table_is_validation(tmp_path):
    """`getattr(mx, _DTYPES.get(meta["dtype"], "float32"))` can only yield a literal.

    Whatever key the attacker supplies, the value comes from the table written in the
    source, which is what validating at the producer looks like. Reporting these is how
    a gate earns its way into being switched off.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "import mlx.core as mx\n"
        "_DTYPES = {'F32': 'float32', 'F16': 'float16'}\n"
        "def load(blob):\n"
        "    meta = json.loads(blob)\n"
        "    return getattr(mx, _DTYPES.get(meta['dtype'], 'float32'))\n",
    )
    assert "getattr(module, ...)" not in _sinks(findings)


def test_a_subscript_of_a_literal_table_is_validation(tmp_path):
    """The raising form of the same lookup produces the same constants."""
    findings = _scan(
        tmp_path,
        "import json\n"
        "import mlx.core as mx\n"
        "_DTYPES = {'F32': 'float32', 'F16': 'float16'}\n"
        "def load(blob):\n"
        "    return getattr(mx, _DTYPES[json.loads(blob)['dtype']])\n",
    )
    assert "getattr(module, ...)" not in _sinks(findings)


def test_a_table_built_from_parsed_values_is_not_validation(tmp_path):
    """The quiet rule must not cover a table whose VALUES came from the artefact.

    Only a dict of literals constrains the result. A mapping assembled out of parsed
    data constrains nothing, and treating the two alike would have been a hole rather
    than a precision fix.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "import mlx.core as mx\n"
        "from pathlib import Path\n"
        "_TABLE = json.loads(Path('downloaded.json').read_text())\n"
        "def load(blob):\n"
        "    return getattr(mx, _TABLE.get(json.loads(blob)['dtype'], 'float32'))\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_os_execv_is_a_sink(tmp_path):
    """The exec family replaces this process with the named program.

    No shell sits in between, so a parsed path here IS execution, and none of these
    APIs were in the sink table at all.
    """
    findings = _scan(
        tmp_path,
        "import json, os\n"
        "def load(blob):\n"
        "    return os.execv(json.loads(blob)['binary'], ['binary'])\n",
    )
    assert "os.execv" in _sinks(findings)


def test_os_spawnv_takes_its_path_after_the_mode(tmp_path):
    """The spawn family puts a mode first, so the path sits at position 1.

    Registering it at 0 like the exec family would have watched the mode and ignored the
    program, which is a sink that cannot fail.
    """
    findings = _scan(
        tmp_path,
        "import json, os\n"
        "def load(blob):\n"
        "    return os.spawnv(os.P_WAIT, json.loads(blob)['binary'], ['binary'])\n",
    )
    assert "os.spawnv" in _sinks(findings)


def test_a_nonlocal_write_from_a_nested_function_survives(tmp_path):
    """`nonlocal command` writes the ENCLOSING scope, which is the point of it.

    Scoping nested writes dropped these too, so a helper setting an outer name from a
    parsed config left the sink below it clean. Only ordinary nested locals are dropped.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = ['ls']\n"
        "    def parse():\n"
        "        nonlocal command\n"
        "        command = json.loads(blob)['command']\n"
        "    parse()\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_method_whose_receiver_is_not_named_self_is_still_a_method(tmp_path):
    """Python does not require the receiver to be called `self`.

    Reading method status off the first parameter's name bound the tainted argument to
    `this` and left `command` clean, so the sink consuming it was missed.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(this, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    return runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_staticmethod_takes_no_receiver(tmp_path):
    """The other half of deriving method status from class containment.

    A `@staticmethod` is inside a class but takes no receiver, so counting it as a method
    would shift every argument one place right and lose the finding.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    @staticmethod\n"
        "    def execute(command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    return Runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_alias_rebound_to_a_second_callable_resolves_both(tmp_path):
    """`runner = safe` then `runner = dirty` runs the second one.

    Keeping only the first helper resolved to the clean one, so the sink inside the one
    that actually runs received the parsed value with nothing reported. Same fail-closed
    choice the reconstructed instance types already make.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def safe(command):\n"
        "    return len(command)\n"
        "def dirty(command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = safe\n"
        "    runner = dirty\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_inherited_method_resolves_through_a_constructed_instance(tmp_path):
    """`child.execute(...)` where `Child` inherits `execute` from `Base`.

    The `self.execute(...)` spelling already walked the bases, so a call through the
    instance gave up exactly where a call from inside the class did not.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Child(Base):\n"
        "    pass\n"
        "def load(blob):\n"
        "    child = Child()\n"
        "    return child.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_async_context_manager_binds_its_target(tmp_path):
    """`async with downloaded(repo) as path` left `path` clean.

    Only the synchronous `with` bound one, so an async consumer of a downloaded path
    reached the import-path sink unreported.
    """
    findings = _scan(
        tmp_path,
        "import contextlib, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "@contextlib.asynccontextmanager\n"
        "async def downloaded(repo):\n"
        "    yield snapshot_download(repo)\n"
        "async def load(repo):\n"
        "    async with downloaded(repo) as path:\n"
        "        sys.path.append(path)\n",
    )
    assert "sys.path.append" in _sinks(findings)


def test_a_local_list_named_path_is_not_an_import_hook(tmp_path):
    """`path = []` then `path.append(parsed)` only mutates a list.

    `from sys import path` canonicalises to `sys.path.append` through the import table,
    so the bare spelling in the sink table only ever matched an unrelated local and
    blocked ordinary code.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "def load(blob):\n"
        "    path = []\n"
        "    path.append(json.loads(blob))\n"
        "    return path\n",
    )
    assert "path.append" not in _sinks(findings)
    assert "sys.path.append" not in _sinks(findings)


def test_sys_path_imported_by_name_is_still_a_sink(tmp_path):
    """The half that has to keep working: `from sys import path` IS the import hook."""
    findings = _scan(
        tmp_path,
        "from sys import path\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    path.append(snapshot_download(repo))\n",
    )
    assert "sys.path.append" in _sinks(findings)


def test_a_nested_global_declaration_stays_with_its_own_function(tmp_path):
    """`ast.walk` descended into nested functions when collecting `global`.

    So an inner helper's declaration was recorded for the function around it, the outer
    function's own local assignment poisoned the module global, and an unrelated function
    executing the safe global failed the gate.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "command = ['ls', '-l']\n"
        "def outer(blob):\n"
        "    def inner():\n"
        "        global command\n"
        "        command = ['safe']\n"
        "    command = json.loads(blob)['command']\n"
        "    return command\n"
        "def other():\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_a_comprehension_target_does_not_leak_into_the_enclosing_scope(tmp_path):
    """Python 3 gives a comprehension its own scope.

    Keeping the binding afterwards turned a safe outer `command` into a tainted one and
    blocked a sink that only ever sees the fixed value.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = ['echo', 'ok']\n"
        "    [None for command in json.loads(blob)]\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_a_nested_return_is_not_the_outer_functions_summary(tmp_path):
    """A nested helper's `return` is its own output, not the enclosing function's.

    Leaving it set made an outer function that returns a fixed literal carry the nested
    summary, and every caller then failed on a value it never produces.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def outer(blob):\n"
        "    def helper():\n"
        "        return json.loads(blob)['command']\n"
        "    return 'safe'\n"
        "def caller(blob):\n"
        "    return subprocess.run(outer(blob))\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_a_nonlocal_in_a_deeper_closure_does_not_reach_two_levels_up(tmp_path):
    """A `nonlocal` inside a grandchild targets ITS enclosing function, not this one."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def outer(blob):\n"
        "    command = ['ls']\n"
        "    def middle():\n"
        "        command = ['safe']\n"
        "        def inner():\n"
        "            nonlocal command\n"
        "            command = json.loads(blob)['command']\n"
        "        inner()\n"
        "    middle()\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_each_class_body_keeps_its_own_namespace(tmp_path):
    """Flattening every class body into the module merged their attributes.

    A parsed `command` in one class then failed a safe `command` in a different class and
    a safe module global of the same name.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "from pathlib import Path\n"
        "class A:\n"
        "    command = json.loads(Path('downloaded.json').read_text())['command']\n"
        "class B:\n"
        "    command = ['ls', '-l']\n"
        "    result = subprocess.run(command)\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_an_alias_of_a_sink_alias_is_still_the_sink(tmp_path):
    """`loader = importlib.import_module` then `invoke = loader`."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    loader = importlib.import_module\n"
        "    invoke = loader\n"
        "    return invoke(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_alias_of_a_callable_alias_is_followed(tmp_path):
    """`runner = execute` then `invoke = runner`: taint stopped at the first hop."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = execute\n"
        "    invoke = runner\n"
        "    return invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_pydoc_locate_by_keyword_is_a_sink(tmp_path):
    """An empty keyword set meant the named spelling was inspected neither way."""
    findings = _scan(
        tmp_path,
        "import json, pydoc\n"
        "def load(blob):\n"
        "    return pydoc.locate(path = json.loads(blob)['class'])\n",
    )
    assert "pydoc.locate" in _sinks(findings)


def test_map_does_not_validate_its_iterable(tmp_path):
    """`map(str, parsed["modules"])` passes attacker names straight through."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    for name in map(str, json.loads(blob)['modules']):\n"
        "        importlib.import_module(name)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_inherited_method_from_an_imported_base_resolves(tmp_path):
    """`class Child(Base)` on a first-party base imported from another module.

    The lookup searched the current file only, and most base classes in this tree are
    imported, so the inherited method sat outside the analysis.
    """
    (tmp_path / "base.py").write_text(
        "import subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom base import Base\n"
        "class Child(Base):\n"
        "    pass\n"
        "def load(blob):\n"
        "    child = Child()\n"
        "    return child.execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "base.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_method_on_a_fresh_instance_resolves(tmp_path):
    """`Child().execute(...)` with no intermediate variable.

    `_call_name` cannot reduce a Call to a name, so the callee did not resolve at all and
    every method reached this way, inherited ones included, was invisible.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Child(Base):\n"
        "    pass\n"
        "def load(blob):\n"
        "    return Child().execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_method_of_a_nested_class_resolves_against_its_full_name(tmp_path):
    """For `Outer.Runner.run`, taking `Outer` as the class looked up `Outer.identity`."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Outer:\n"
        "    class Runner:\n"
        "        def identity(self, value):\n"
        "            return value\n"
        "        def run(self, blob):\n"
        "            return subprocess.run(self.identity(json.loads(blob)['command']))\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_aliased_source_callable_still_taints(tmp_path):
    """`decode = json.loads` then `decode(blob)`.

    Sink aliases were followed and source aliases were not, so a deserialiser behind an
    ordinary local name read clean and everything downstream of it did too.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    decode = json.loads\n"
        "    return importlib.import_module(decode(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_property_getter_carries_its_taint(tmp_path):
    """`cfg.module` where the getter returns a parsed value.

    An ordinary method call on the same instance was followed, so the one spelling that
    looks like a plain attribute read was the gap.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "class Config:\n"
        "    def __init__(self, blob):\n"
        "        self.blob = blob\n"
        "    @property\n"
        "    def module(self):\n"
        "        return json.loads(self.blob)['module']\n"
        "def load(blob):\n"
        "    cfg = Config(blob)\n"
        "    return importlib.import_module(cfg.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_constant_bound_to_true_still_enables_remote_code(tmp_path):
    """`enabled = True` forwarded as `trust_remote_code = enabled` runs repository code.

    Only the literal written at the call was recognised.
    """
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    enabled = True\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = enabled)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_function_local_class_resolves_under_its_qualified_name(tmp_path):
    """A class declared inside a function is indexed as `load.Runner.execute`.

    Construction returned the bare `Runner`, so the method lookup searched for
    `Runner.execute` and taint never reached the sink inside it.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    class Runner:\n"
        "        def execute(self, command):\n"
        "            return subprocess.run(command)\n"
        "    runner = Runner()\n"
        "    return runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_lambda_is_a_call_target(tmp_path):
    """`runner = lambda command: subprocess.run(command)` is a common wrapper form.

    `_call_name` cannot reduce a Lambda to a name, so the assignment was discarded and
    the sink inside saw an unbound, clean parameter.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    runner = lambda command: subprocess.run(command)\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_lambda_body_is_its_return_value(tmp_path):
    """A lambda has no `return` statement, so it needed a summary of its own."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    pick = lambda cfg: cfg['module']\n"
        "    return importlib.import_module(pick(json.loads(blob)))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_weight_file_load_is_an_untrusted_source(tmp_path):
    """The tensor NAMES in a downloaded checkpoint are attacker-chosen.

    Only the config beside a checkpoint counted as a source, so the only chain the
    scanner could see into an adapter loader was through `adapter_config.json`, and a
    finding driven by tensor names was attributed to the config instead.
    """
    findings = _scan(
        tmp_path,
        "import mlx.core as mx\n"
        "def load(path):\n"
        "    tensors = dict(mx.load(path))\n"
        "    for name in tensors:\n"
        "        module = mx\n"
        "        getattr(module, name)\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_a_container_written_through_setdefault_is_tainted(tmp_path):
    """`full_state.setdefault(path, {})[name] = tensor` writes THROUGH a call.

    The assignment's base was a Call, so it landed nowhere: the dict being built stayed
    clean and the function returning it was summarised clean too.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def group(blob):\n"
        "    grouped = {}\n"
        "    for name, value in json.loads(blob).items():\n"
        "        grouped.setdefault('all', {})[name] = value\n"
        "    return grouped\n"
        "def load(blob):\n"
        "    return importlib.import_module(group(blob)['all']['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)
