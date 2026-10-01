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
