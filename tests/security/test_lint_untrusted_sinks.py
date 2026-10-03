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
import os
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


# This one scans the whole repository in a subprocess, which takes minutes. The
# authoritative run of the same command is the `lint-ci.yml` step, so under GitHub
# Actions it is skipped: running it again inside every other pytest job duplicated a
# five minute scan and pushed the workflow guard job past its own timeout. It stays
# for local pytest, so someone running the suite still sees a baseline that drifted.
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason = "lint-ci.yml runs the same whole-repository scan as a dedicated step",
)
@pytest.mark.timeout(1800)
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
        "    code = snapshot_download(repo, revision = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')\n"
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


def test_an_unpinned_fetch_is_correlated_across_a_helper(tmp_path):
    """`fetch()` returns the download and `load()` puts it on `sys.path`.

    The correlation was keyed by the sink's qualname, so a download returned by one
    helper and imported by another was never correlated at all: removing a `revision`
    changed only the helper, the sink's own allowance stayed valid, and nothing was
    reported.
    """
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def fetch(repo):\n"
        "    return snapshot_download(repo)\n"
        "def load(repo):\n"
        "    sys.path.insert(0, fetch(repo))\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


def test_a_weights_download_through_a_helper_is_still_not_a_code_fetch(tmp_path):
    """The quiet half: following a branch for weights is correct behaviour.

    Making the correlation global must not turn it back into proximity, or every lazy
    import beside a weights download is reported again.
    """
    findings = _scan(
        tmp_path,
        "import torch\n"
        "from huggingface_hub import snapshot_download\n"
        "def fetch(repo):\n"
        "    return snapshot_download(repo)\n"
        "def load(repo):\n"
        "    return torch.load(fetch(repo) + '/model.bin')\n",
    )
    assert "unpinned code fetch" not in _sinks(findings)


def test_a_qualified_imported_base_resolves(tmp_path):
    """`import producer` then `class Child(producer.Base)`.

    Stripping every base to its last segment left `Base`, which the import table has no
    binding for, so this spelling of imported inheritance resolved to nothing while the
    `from producer import Base` spelling worked.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nimport producer\n"
        "class Child(producer.Base):\n"
        "    pass\n"
        "def load(blob):\n"
        "    return Child().execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_super_reaches_a_method_on_an_imported_base(tmp_path):
    """`super().execute(...)` where the base lives in another first-party module.

    Resolving from the class itself would find the overriding method that contains the
    `super()` call, so the inherited lookup has to skip the class's own declaration.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import Base\n"
        "class Child(Base):\n"
        "    def execute(self, command):\n"
        "        return super().execute(command)\n"
        "def load(blob):\n"
        "    return Child().execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_method_inherited_through_two_levels_resolves(tmp_path):
    """`Root.execute`, `Mid(Root)`, `Child(Mid)`: only the direct bases were checked."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Root:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Mid(Root):\n"
        "    pass\n"
        "class Child(Mid):\n"
        "    pass\n"
        "def load(blob):\n"
        "    return Child().execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_chained_alias_collected_at_module_scope_resolves(tmp_path):
    """`loader = import_module` then `invoke = loader`, both at module scope.

    The collector matched the referenced name against imports only and never consulted
    the aliases it had already collected, so function visitors were seeded with the
    first name and never the second.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "loader = importlib.import_module\n"
        "invoke = loader\n"
        "def load(blob):\n"
        "    return invoke(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_alias_rebound_to_a_second_sink_watches_both(tmp_path):
    """`action = subprocess.run` then `action = sys.path.insert`.

    The two sinks watch different argument positions, so keeping only the first examined
    argument 0 and missed the tainted path in argument 1.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess, sys\n"
        "def load(blob):\n"
        "    action = subprocess.run\n"
        "    action = sys.path.insert\n"
        "    return action(0, json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_source_alias_bound_on_a_backedge_is_carried(tmp_path):
    """`decode` used before it is rebound to `json.loads` for the next iteration.

    The fixpoint carried sink and callable aliases between passes and dropped the source
    table, so the sink was never revisited with the name recognised as a deserialiser.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def safe(value):\n"
        "    return {'module': 'fixed'}\n"
        "def load(blob, rounds):\n"
        "    decode = safe\n"
        "    for _ in range(rounds):\n"
        "        importlib.import_module(decode(blob)['module'])\n"
        "        decode = json.loads\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_nested_source_alias_does_not_leak_outward(tmp_path):
    """A nested helper binding `decode = json.loads` must not rename the outer `decode`.

    The saved-state tuple restored the other alias maps and not this one, so an outer
    function whose own `decode` is a safe callable was reported as though it had called
    `json.loads`, even when the nested helper is never invoked.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def safe(value):\n"
        "    return {'module': 'fixed'}\n"
        "def load(blob):\n"
        "    def helper():\n"
        "        decode = json.loads\n"
        "        return decode(blob)\n"
        "    decode = safe\n"
        "    return importlib.import_module(decode(blob)['module'])\n",
    )
    assert "importlib.import_module" not in _sinks(findings)


def test_an_inherited_self_call_crosses_a_file(tmp_path):
    """`self.execute(parsed)` in a child of a base declared in another module.

    The branch checked the current class and its direct bases in this file only, so this
    spelling reached no target at all while the constructed-instance spelling resolved.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "class Base:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import Base\n"
        "class Child(Base):\n"
        "    def load(self, blob):\n"
        "        return self.execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_an_attribute_written_on_a_subclass_reaches_an_inherited_reader(tmp_path):
    """`child.command = parsed` then an inherited `Base.run` executing `self.command`.

    The write recorded `Child.command` and the inherited method read `Base.command`, so
    the two halves never met. Writes bind every ancestor's key and reads consult them.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "class Base:\n"
        "    def run(self):\n"
        "        return subprocess.run(self.command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import Base\n"
        "class Child(Base):\n"
        "    pass\n"
        "def load(blob):\n"
        "    child = Child()\n"
        "    child.command = json.loads(blob)['command']\n"
        "    return child.run()\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_an_annotated_true_flag_still_enables_remote_code(tmp_path):
    """`enabled: bool = True` is the same flag as the plain spelling."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    enabled: bool = True\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = enabled)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_true_flag_set_on_a_backedge_is_carried(tmp_path):
    """Every pass visited the loader call before rediscovering the assignment."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name, rounds):\n"
        "    enabled = False\n"
        "    for _ in range(rounds):\n"
        "        AutoModel.from_pretrained(name, trust_remote_code = enabled)\n"
        "        enabled = True\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_module_level_true_flag_reaches_a_loader(tmp_path):
    """`ENABLED = True` at module scope, forwarded into a loader."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = True\n"
        "def load(name):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = ENABLED)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_an_explicit_null_revision_is_unpinned(tmp_path):
    """`revision = None` reaches the library as the same default as an omitted keyword.

    A presence-only test let the explicit spelling through, which is worse than the
    omitted one because changing a helper to `revision = None` does not alter the sink
    function's context digest and so keeps its baseline allowance valid.
    """
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = None))\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


def test_a_pinned_revision_is_still_quiet(tmp_path):
    """The half that must keep working: a commit revision pins the code that runs."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = '" + "ab12" * 10 + "'))\n",
    )
    assert "unpinned code fetch" not in _sinks(findings)


def test_format_map_preserves_taint(tmp_path):
    """The mapping spelling of the same interpolation."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def load(blob):\n"
        "    return importlib.import_module('x.{n}'.format_map(json.loads(blob)))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_asyncio_subprocess_creation_is_a_sink(tmp_path):
    """The async spellings run a program exactly as the blocking ones do."""
    findings = _scan(
        tmp_path,
        "import asyncio, json\n"
        "async def load(blob):\n"
        "    return await asyncio.create_subprocess_exec(json.loads(blob)['binary'])\n",
    )
    assert "asyncio.create_subprocess_exec" in _sinks(findings)


def test_a_sink_wrapped_in_functools_partial_is_still_a_sink(tmp_path):
    """`functools.partial(import_module)` is a reference to the sink, wrapped.

    The alias handling skipped anything that was a call, so the partial walked past the
    gate while the bare alias of the same sink was caught.
    """
    findings = _scan(
        tmp_path,
        "import functools, importlib, json\n"
        "def load(blob):\n"
        "    loader = functools.partial(importlib.import_module)\n"
        "    return loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_argument_bound_into_a_partial_is_checked(tmp_path):
    """`functools.partial(import_module, parsed)` then `loader()`.

    Recording only the wrapped sink's identity discarded the bound values, and the later
    zero-argument call leaves the sink check nothing to look at. The bound arguments are
    checked at the partial itself, with the wrapped callee dropped off the front so the
    positions line up with the sink's.
    """
    findings = _scan(
        tmp_path,
        "import functools, importlib, json\n"
        "def load(blob):\n"
        "    loader = functools.partial(importlib.import_module, json.loads(blob)['module'])\n"
        "    return loader()\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_keyword_bound_into_a_partial_is_checked(tmp_path):
    """The named spelling, onto a sink that watches a keyword rather than a position."""
    findings = _scan(
        tmp_path,
        "import functools, json, subprocess\n"
        "def load(blob):\n"
        "    runner = functools.partial(subprocess.run, executable = json.loads(blob)['exe'])\n"
        "    return runner()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_sink_in_a_nested_class_body_is_scanned(tmp_path):
    """Python runs a nested class body while defining its parent.

    The nested node was filtered out of the only worklist and never queued on its own,
    so an import-time sink inside `class Outer: class Inner:` was not visited at all.
    """
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "from pathlib import Path\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        module = importlib.import_module(\n"
        "            json.loads(Path('downloaded.json').read_text())['module']\n"
        "        )\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_nested_true_flag_does_not_leak_outward(tmp_path):
    """A nested `enabled = True` must not make the outer `enabled` read as true.

    The saved-state tuple omitted this set, so an outer loader call that always receives
    False reported a gated finding.
    """
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    enabled = False\n"
        "    def helper():\n"
        "        enabled = True\n"
        "        return enabled\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = enabled)\n",
    )
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_an_instance_held_on_an_attribute_resolves(tmp_path):
    """`self.runner = Runner()` in the constructor, called from another method.

    Only plain locals were recorded, and a per-function map could not carry the binding
    from `__init__` to the method that calls through it, so the instance type is kept in
    shared state under the same key an attribute read uses.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.runner = Runner()\n"
        "    def go(self, blob):\n"
        "        return self.runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_plain_self_call_still_resolves(tmp_path):
    """The guard for the branch order.

    The attribute-held receiver has to be tried before the `self.`/`cls.` branch, which
    returns early on anything with that head. Putting it after swallowed the new shape;
    putting it before must not swallow the ordinary one.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Holder:\n"
        "    def helper(self, command):\n"
        "        return subprocess.run(command)\n"
        "    def go(self, blob):\n"
        "        return self.helper(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_true_flag_reached_through_an_alias_is_found(tmp_path):
    """`enabled = True` then `remote = enabled`: one assignment walked past the gate."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    enabled = True\n"
        "    remote = enabled\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = remote)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_pickle_handle_passed_by_keyword_is_checked(tmp_path):
    """`pickle.load(file = handle)` is the same call as `pickle.load(handle)`.

    The declared keyword names for the two `load` entries were never consulted, so
    spelling the handle out by name walked past the sink that exists for it.
    """
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    downloaded = hf_hub_download(repo, 'weights.pkl')\n"
        "    with open(downloaded, 'rb') as handle:\n"
        "        return pickle.load(file = handle)\n",
    )
    assert "pickle.load" in _sinks(findings)


def test_a_dill_handle_passed_by_keyword_is_checked(tmp_path):
    """The same spelling for `dill`, which is the one Unsloth actually imports."""
    findings = _scan(
        tmp_path,
        "import dill\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    downloaded = hf_hub_download(repo, 'weights.pkl')\n"
        "    with open(downloaded, 'rb') as handle:\n"
        "        return dill.load(file = handle)\n",
    )
    assert "dill.load" in _sinks(findings)


def test_a_walrus_used_in_place_carries_its_taint(tmp_path):
    """`subprocess.run(command := parsed['command'])`, binding and using in one go.

    The assignment expression was not a case in the taint test, so an argument written
    this way read clean even though the identical value assigned on its own line fired.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    return subprocess.run(command := json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_dict_built_with_an_untrusted_key_is_tainted(tmp_path):
    """`{parsed['command']: 1}` taints the dict, because iterating it yields the key.

    Only the values of a dict literal were read, so a table keyed by the untrusted half
    of a parsed document looked as clean as an empty one.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    table = {json.loads(blob)['command']: 1}\n"
        "    for key in table:\n"
        "        return subprocess.run(key)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_literal_dict_is_still_quiet(tmp_path):
    """The guard for reading dict keys: a table written out in the file is not a source."""
    findings = _scan(
        tmp_path,
        "import subprocess\n"
        "def load():\n"
        "    table = {'ls': '-l'}\n"
        "    for key in table:\n"
        "        return subprocess.run(key)\n",
    )
    assert _sinks(findings) == set()


def test_a_starred_argument_reaches_a_watched_position(tmp_path):
    """`os.spawnv(*rest)`, whose watched position is 1 rather than 0.

    Positions were counted off the argument list as written, so a star sitting before a
    watched position left that position empty and nothing was compared at all. One
    unpacked sequence can fill any position, so a starred argument is now checked
    against every watched one.
    """
    findings = _scan(
        tmp_path,
        "import json, os\n"
        "def load(blob):\n"
        "    rest = json.loads(blob)['rest']\n"
        "    return os.spawnv(*rest)\n",
    )
    assert "os.spawnv" in _sinks(findings)


def test_a_starred_literal_is_still_quiet(tmp_path):
    """The guard for starred arguments: a list written out here fills no position badly."""
    findings = _scan(
        tmp_path,
        "import os\n"
        "def load():\n"
        "    rest = ['/bin/ls', '-l']\n"
        "    return os.spawnv(*rest)\n",
    )
    assert _sinks(findings) == set()


def test_an_instance_rebound_to_a_second_name_keeps_its_type(tmp_path):
    """`invoke = runner` then `invoke.execute(parsed)`.

    The constructed type was recorded against the name the constructor was assigned to
    and nowhere else, so one plain rebinding lost the type and the method call resolved
    to nothing.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    invoke = runner\n"
        "    return invoke.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_bound_method_pulled_off_an_instance_is_followed(tmp_path):
    """`execute = runner.execute` then `execute(parsed)`.

    A callable alias resolved a plain function but not a method taken off an instance,
    and the argument offset has to drop the receiver: without that the tainted value
    bound to `self` and the real parameter read clean.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    execute = runner.execute\n"
        "    return execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_attribute_write_binds_every_recorded_type(tmp_path):
    """Two constructors on one name, and the sink is in the second class.

    The attribute key used the first recorded type only, so `runner = Safe()` followed
    by `runner = Dirty()` wrote `Safe.command` while `Dirty.run` read `Dirty.command`.
    Method resolution already considered both types; the attribute key now does too.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Safe:\n"
        "    def run(self):\n"
        "        return len(self.command)\n"
        "class Dirty:\n"
        "    def run(self):\n"
        "        return subprocess.run(self.command)\n"
        "def load(blob):\n"
        "    runner = Safe()\n"
        "    runner = Dirty()\n"
        "    runner.command = json.loads(blob)['command']\n"
        "    return runner.run()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_attribute_write_reaches_an_ancestor_two_files_away(tmp_path):
    """Local `Child`, imported `Mid`, and `Root` declared above `Mid` in a third file.

    The ancestor walk stopped at the directly imported class, so the write recorded a
    key for `Mid` but not for `Root`, whose inherited `run` is where the sink is. Each
    base is resolved through the import table of the file that declares it, so the key
    lands on the file that actually holds the reader.
    """
    (tmp_path / "root.py").write_text(
        "import subprocess\n"
        "class Root:\n"
        "    def run(self):\n"
        "        return subprocess.run(self.command)\n",
        encoding = "utf-8",
    )
    (tmp_path / "middle.py").write_text(
        "from root import Root\nclass Mid(Root):\n    pass\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom middle import Mid\n"
        "class Child(Mid):\n"
        "    pass\n"
        "def load(blob):\n"
        "    child = Child()\n"
        "    child.command = json.loads(blob)['command']\n"
        "    return child.run()\n",
        encoding = "utf-8",
    )
    findings = L.scan(
        [tmp_path / "root.py", tmp_path / "middle.py", consumer],
        roots = [tmp_path],
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_class_body_binding_reaches_a_class_read(tmp_path):
    """`class Config: module = parsed["module"]` then `Config.module` at a sink.

    A class body was settled only when reporting, so its bindings were discovered after
    everything that could read them had already run, and the binding was then thrown
    away with the body's locals. A name bound in a class body is a class attribute, so
    it is published under the key attribute reads already use and the fixpoint settles
    the rest.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Config:\n"
        "    module = json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "def go():\n"
        "    return importlib.import_module(Config.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_class_body_binding_reaches_a_method_read(tmp_path):
    """The same binding read as `self.module` from a method of that class."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Config:\n"
        "    module = json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "    def go(self):\n"
        "        return importlib.import_module(self.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_nested_class_body_binding_is_published(tmp_path):
    """`class Outer: class Inner: module = parsed[...]`, read as `Outer.Inner.module`."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Outer:\n"
        "    class Inner:\n"
        "        module = json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "def go():\n"
        "    return importlib.import_module(Outer.Inner.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_another_class_with_the_same_attribute_stays_quiet(tmp_path):
    """The guard for publishing class bodies: the key is per class, not per name.

    Two classes with a `module` attribute, one parsed and one fixed. Merging them is how
    a gate starts failing on the safe one, which is the fastest way to get itself turned
    off.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Dirty:\n"
        "    module = json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "class Clean:\n"
        "    module = 'torch'\n"
        "def go():\n"
        "    return importlib.import_module(Clean.module)\n",
    )
    assert _sinks(findings) == set()


def test_a_partial_shifts_the_watched_sink_position(tmp_path):
    """`add_path = functools.partial(sys.path.insert, 0)` then `add_path(parsed)`.

    The alias recorded the sink's identity and not the partial's layout, so the later
    call was compared at position 1 while the partial had already moved the path to
    position 0. The creation-time check saw only the clean index.
    """
    findings = _scan(
        tmp_path,
        "import json, functools, sys\n"
        "def load(blob):\n"
        "    add_path = functools.partial(sys.path.insert, 0)\n"
        "    return add_path(json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_rebound_partial_keeps_its_layout(tmp_path):
    """`invoke = add_path`: the second name is the same wrapper with the same shift."""
    findings = _scan(
        tmp_path,
        "import json, functools, sys\n"
        "def load(blob):\n"
        "    add_path = functools.partial(sys.path.insert, 0)\n"
        "    invoke = add_path\n"
        "    return invoke(json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_an_unwrapped_sink_is_still_checked_at_its_own_positions(tmp_path):
    """The guard for the shift: an ordinary call and an ordinary alias are not shifted."""
    direct = _scan(
        tmp_path,
        "import json, sys\n"
        "def load(blob):\n"
        "    return sys.path.insert(0, json.loads(blob)['path'])\n",
        name = "direct.py",
    )
    aliased = _scan(
        tmp_path,
        "import json, sys\n"
        "def load(blob):\n"
        "    add = sys.path.insert\n"
        "    return add(0, json.loads(blob)['path'])\n",
        name = "aliased.py",
    )
    assert "sys.path.insert" in _sinks(direct)
    assert "sys.path.insert" in _sinks(aliased)


def test_a_partial_around_a_literal_is_quiet(tmp_path):
    """And the wrapper itself is not a finding when what flows through it is fixed."""
    findings = _scan(
        tmp_path,
        "import functools, sys\n"
        "def load():\n"
        "    add_path = functools.partial(sys.path.insert, 0)\n"
        "    return add_path('/opt/fixed')\n",
    )
    assert _sinks(findings) == set()


def test_a_partial_around_a_first_party_helper_is_followed(tmp_path):
    """`runner = functools.partial(execute)` then `runner(parsed)`.

    The branch recognised a wrapped callable only when it matched a sink or an existing
    sink alias, so wrapping a first-party helper recorded nothing at all: no callable
    alias, no parameter propagation, and the `subprocess.run` inside the helper was
    missed.
    """
    findings = _scan(
        tmp_path,
        "import json, functools, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = functools.partial(execute)\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_partial_around_a_helper_shifts_its_parameters(tmp_path):
    """The same wrapper with one argument pre-bound, so the shift applies to parameters."""
    findings = _scan(
        tmp_path,
        "import json, functools, subprocess\n"
        "def execute(tag, command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = functools.partial(execute, 'build')\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_argument_bound_into_a_partial_reaches_the_helper(tmp_path):
    """And an argument bound at creation time, which the wrapper then never takes."""
    findings = _scan(
        tmp_path,
        "import json, functools, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = functools.partial(execute, json.loads(blob)['command'])\n"
        "    return runner()\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_property_reached_through_a_held_instance_resolves(tmp_path):
    """`self.cfg = Config()` then `import_module(self.cfg.module)`.

    The getter lookup rejected any receiver that was not a plain name, so the held
    instance the constructor records could not be resolved and the property's tainted
    return summary was never consulted.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Config:\n"
        "    @property\n"
        "    def module(self):\n"
        "        return json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.cfg = Config()\n"
        "    def go(self):\n"
        "        return importlib.import_module(self.cfg.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_property_returning_a_literal_stays_quiet(tmp_path):
    """The guard for that: the receiver resolving does not make the getter tainted."""
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "class Config:\n"
        "    @property\n"
        "    def module(self):\n"
        "        return 'torch'\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.cfg = Config()\n"
        "    def go(self):\n"
        "        return importlib.import_module(self.cfg.module)\n",
    )
    assert _sinks(findings) == set()


def test_prefix_and_suffix_removal_carry_taint(tmp_path):
    """`parsed["module"].removeprefix("plugins.")` selects a module, it validates nothing.

    The pass-through list had `strip` and `replace` but not these, so trimming a known
    prefix off an untrusted name laundered it.
    """
    for transform in ("removeprefix('plugins.')", "removesuffix('.main')", "lstrip('.')"):
        findings = _scan(
            tmp_path,
            "import json, importlib\n"
            "def load(blob):\n"
            "    return importlib.import_module(json.loads(blob)['module'].%s)\n" % transform,
            name = "sample_%s.py" % transform[:6],
        )
        assert "importlib.import_module" in _sinks(findings), transform


def test_a_module_level_true_flag_reached_through_a_name(tmp_path):
    """`ENABLED = True` then `REMOTE = ENABLED` forwarded as `trust_remote_code`.

    Only the literal module-scope assignment was collected, so one hop was enough to
    hide remote-code enablement even though the local alias chain was already handled.
    """
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = True\n"
        "REMOTE = ENABLED\n"
        "def load(name):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = REMOTE)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_module_level_false_flag_stays_quiet(tmp_path):
    """The guard: the chain is followed, the value is still what decides."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = False\n"
        "REMOTE = ENABLED\n"
        "def load(name):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = REMOTE)\n",
    )
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_torch_load_with_weights_only_off_is_a_sink(tmp_path):
    """`torch.load(downloaded, weights_only = False)` runs the pickle-based loader.

    That constructs arbitrary objects out of the checkpoint, which is the same execution
    the `pickle` entries exist for, and the table covered only the `pickle` and `dill`
    spellings.
    """
    findings = _scan(
        tmp_path,
        "import torch\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    path = hf_hub_download(repo, 'weights.bin')\n"
        "    return torch.load(path, weights_only = False)\n",
    )
    assert "torch.load(weights_only = False)" in _sinks(findings)


def test_torch_load_through_a_bare_import_is_a_sink(tmp_path):
    """`from torch import load`, resolved through the import table like every sink."""
    findings = _scan(
        tmp_path,
        "from torch import load\n"
        "from huggingface_hub import hf_hub_download\n"
        "def go(repo):\n"
        "    return load(hf_hub_download(repo, 'w.bin'), weights_only = False)\n",
    )
    assert "torch.load(weights_only = False)" in _sinks(findings)


def test_torch_load_with_an_unproven_flag_is_a_sink(tmp_path):
    """A `weights_only` this file cannot read as True is not proof that it is on."""
    findings = _scan(
        tmp_path,
        "import torch\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo, flag):\n"
        "    path = hf_hub_download(repo, 'weights.bin')\n"
        "    return torch.load(path, weights_only = flag)\n",
    )
    assert "torch.load(weights_only = False)" in _sinks(findings)


def test_torch_load_stays_quiet_when_weights_only_holds(tmp_path):
    """The guards: an explicit True, or a local name bound to True, is accepted."""
    head = (
        "import torch\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    path = hf_hub_download(repo, 'weights.bin')\n"
    )
    for tail, label in (
        ("    return torch.load(path, weights_only = True)\n", "explicit True"),
        ("    safe = True\n    return torch.load(path, weights_only = safe)\n", "local True"),
    ):
        findings = _scan(tmp_path, head + tail, name = "sample_%s.py" % label.replace(" ", "_"))
        assert not {
            "torch.load(weights_only = False)",
            "torch.load(weights_only unset)",
        } & _sinks(findings), label


def test_torch_load_with_weights_only_omitted_is_a_sink(tmp_path):
    """`torch.load(downloaded)` with no `weights_only` at all.

    It only defaults to True from torch 2.6, and unsloth-zoo still allows
    `torch>=2.4.0`, so leaving it out unpickles arbitrary objects on a supported
    install. Reported under its own label so a reviewed explicit-False entry and a
    reviewed omission stay distinguishable in the baseline.
    """
    findings = _scan(
        tmp_path,
        "import torch\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    return torch.load(hf_hub_download(repo, 'weights.bin'))\n",
    )
    assert "torch.load(weights_only unset)" in _sinks(findings)


def test_json_load_is_not_mistaken_for_torch_load(tmp_path):
    """The guard for the omitted rule: only the canonical `torch.load` is the sink.

    A bare `load` entry in the name table suffix-matched every `*.load`, `json.load`
    included, which the explicit-False requirement used to hide.
    """
    findings = _scan(
        tmp_path,
        "import json\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    with open(hf_hub_download(repo, 'config.json')) as handle:\n"
        "        return json.load(handle)\n",
    )
    assert _sinks(findings) == set()


def test_a_path_open_receiver_carries_its_taint(tmp_path):
    """`Path(downloaded).open("rb")` carries the path on the receiver, not in the args.

    The arguments are the mode, so the loop over them found nothing and the handle onto
    attacker bytes read clean for the whole instance-method form.
    """
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    handle = Path(hf_hub_download(repo, 'w.pkl'))\n"
        "    with handle.open('rb') as stream:\n"
        "        return pickle.load(stream)\n",
    )
    assert "pickle.load" in _sinks(findings)


def test_a_fixed_path_open_is_still_quiet(tmp_path):
    """The guard: reading the receiver does not make every `.open` a source."""
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from pathlib import Path\n"
        "def load():\n"
        "    handle = Path('/opt/fixed.pkl')\n"
        "    with handle.open('rb') as stream:\n"
        "        return pickle.load(stream)\n",
    )
    assert _sinks(findings) == set()


def test_an_annotated_instance_alias_keeps_its_type(tmp_path):
    """`invoke: Runner = runner`, the annotated spelling of a rebinding.

    This mirrored path called the other alias helpers and omitted the one that records
    the type, so the method behind the second name did not resolve while the identical
    unannotated assignment did.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    invoke: Runner = runner\n"
        "    return invoke.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_sink_stored_on_an_attribute_is_followed(tmp_path):
    """`runner.loader = importlib.import_module` then `runner.loader(parsed)`.

    Only name targets were recorded, so the binding was discarded and the call matched
    neither the canonical table nor the alias table.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Runner:\n"
        "    pass\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    runner.loader = importlib.import_module\n"
        "    return runner.loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_sink_stored_on_self_is_read_by_another_method(tmp_path):
    """The realistic shape: the constructor stores it, a method calls through it.

    Separate passes, so the binding goes into the shared map under the same keys the
    attribute reads use, exactly as the held instances already do.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Runner:\n"
        "    def __init__(self):\n"
        "        self.loader = importlib.import_module\n"
        "    def go(self, blob):\n"
        "        return self.loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_getattr_stored_on_an_attribute_is_still_reflection(tmp_path):
    """And `self.resolve = getattr`, which has its own check rather than a table entry."""
    findings = _scan(
        tmp_path,
        "import json, transformers\n"
        "class Runner:\n"
        "    def __init__(self):\n"
        "        self.resolve = getattr\n"
        "    def go(self, blob):\n"
        "        return self.resolve(transformers, json.loads(blob)['cls'])\n",
    )
    assert "getattr(module, ...)" in _sinks(findings)


def test_an_unrelated_attribute_call_stays_quiet(tmp_path):
    """The guard: a callable on an attribute that is not a sink is not one."""
    findings = _scan(
        tmp_path,
        "import json\n"
        "class Runner:\n"
        "    def __init__(self):\n"
        "        self.loader = len\n"
        "    def go(self, blob):\n"
        "        return self.loader(json.loads(blob)['module'])\n",
    )
    assert _sinks(findings) == set()


def test_a_wildcard_import_resolves_its_callable(tmp_path):
    """`from producer import *` then `execute(parsed)`.

    Only the literal `*` was bound, so the call resolved to nothing and taint never
    reached the helper. The exporting file is not parsed yet when the import is read, so
    the base is kept and expanded against that file's exports on demand.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import *\n"
        "def load(blob):\n"
        "    return execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_wildcard_import_respects_dunder_all(tmp_path):
    """The guard: a name `__all__` leaves out is not bound by `import *`.

    Resolved against the exporting file rather than guessed, so the expansion cannot
    invent a target that the star does not actually bind.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "__all__ = ['other']\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "def other():\n"
        "    pass\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import *\n"
        "def load(blob):\n"
        "    return execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert _sinks(findings) == set()


def test_a_reexport_through_two_packages_is_followed(tmp_path):
    """`pkg` -> `pkg.api` -> `pkg.api.impl`, where only the last one defines `execute`.

    One hop landed on a file that does not define the symbol either, so the hop was
    declined and the implementation was never analysed. Followed through the layers now,
    bounded so a circular re-export terminates.
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "api").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("from pkg.api import execute\n", encoding = "utf-8")
    (tmp_path / "pkg" / "api" / "__init__.py").write_text(
        "from pkg.api.impl import execute\n", encoding = "utf-8"
    )
    (tmp_path / "pkg" / "api" / "impl.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom pkg import execute\n"
        "def load(blob):\n"
        "    return execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan(
        [
            tmp_path / "pkg" / "__init__.py",
            tmp_path / "pkg" / "api" / "__init__.py",
            tmp_path / "pkg" / "api" / "impl.py",
            consumer,
        ],
        roots = [tmp_path],
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_mapped_callback_receives_the_elements(tmp_path):
    """`map(execute, parsed["commands"])` calls `execute` with every element.

    The result carried the taint onward, which is right, but the callback was never
    analysed with a tainted parameter, so a `subprocess.run` inside it ran every
    attacker-chosen command with nothing reported.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import execute\n"
        "def load(blob):\n"
        "    return list(map(execute, json.loads(blob)['commands']))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_mapped_callback_over_a_literal_is_quiet(tmp_path):
    """The guard: the callback is analysed with what it actually receives."""
    (tmp_path / "producer.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from producer import execute\n"
        "def load():\n"
        "    return list(map(execute, ['ls', 'pwd']))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert _sinks(findings) == set()


def test_a_values_view_of_a_key_tainted_dict_is_quiet(tmp_path):
    """`{parsed["label"]: ["echo", "ok"]}.values()` hands back only the fixed half.

    Reading dict keys is right, because iterating a dict yields them, but marking the
    whole literal tainted made `.values()` inherit it and blocked code whose every
    iterated command is written out in the file.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    table = {json.loads(blob)['label']: ['echo', 'ok']}\n"
        "    for command in table.values():\n"
        "        subprocess.run(command)\n",
    )
    assert _sinks(findings) == set()


def test_the_key_half_of_that_dict_is_still_reported(tmp_path):
    """The guard, four ways: everything that can expose the key still reports it."""
    head = "import json, subprocess\ndef load(blob):\n    table = {json.loads(blob)['label']: ['echo']}\n"
    shapes = {
        "keys": "    for command in table.keys():\n        subprocess.run(command)\n",
        "iteration": "    for command in table:\n        subprocess.run(command)\n",
        "items": "    for key, command in table.items():\n        subprocess.run(key)\n",
    }
    for label, tail in shapes.items():
        findings = _scan(tmp_path, head + tail, name = "sample_%s.py" % label)
        assert "subprocess.run" in _sinks(findings), label
    # And a dict whose VALUES are parsed is unaffected by the narrowing.
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    table = {'label': json.loads(blob)['command']}\n"
        "    for command in table.values():\n"
        "        subprocess.run(command)\n",
        name = "sample_values.py",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_rebinding_clears_the_key_only_narrowing(tmp_path):
    """The narrowing is per binding: the second assignment reopens the name.

    This is the one piece of state here that suppresses a finding, so it is also the one
    that is not accumulated across passes of the fixpoint and not carried out of a
    nested scope.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    table = {json.loads(blob)['label']: ['echo', 'ok']}\n"
        "    table = {'label': json.loads(blob)['command']}\n"
        "    for command in table.values():\n"
        "        subprocess.run(command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_nested_key_only_dict_does_not_clear_the_outer_one(tmp_path):
    """And a nested body's narrowing stays inside it."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    table = {'label': json.loads(blob)['command']}\n"
        "    def inner():\n"
        "        table = {json.loads(blob)['label']: ['echo']}\n"
        "        return table\n"
        "    inner()\n"
        "    for command in table.values():\n"
        "        subprocess.run(command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_module_level_partial_is_collected(tmp_path):
    """Module-scope `add_path = functools.partial(sys.path.insert, 0)`.

    The module-scope collector asked only whether the call constructs a class and then
    moved on, so neither the wrapped sink nor its layout was ever seeded into a function
    visitor and a call from any function in the file reported nothing.
    """
    findings = _scan(
        tmp_path,
        "import json, functools, sys\n"
        "add_path = functools.partial(sys.path.insert, 0)\n"
        "def load(blob):\n"
        "    return add_path(json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_module_level_partial_around_a_helper_is_collected(tmp_path):
    """The same at module scope for a first-party helper rather than a sink."""
    (tmp_path / "producer.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json, functools\nfrom producer import execute\n"
        "runner = functools.partial(execute)\n"
        "def load(blob):\n"
        "    return runner(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_module_level_partial_around_a_literal_is_quiet(tmp_path):
    """The guard, and that an ordinary module-scope sink alias is unaffected."""
    wrapped = _scan(
        tmp_path,
        "import functools, sys\n"
        "add_path = functools.partial(sys.path.insert, 0)\n"
        "def load():\n"
        "    return add_path('/opt/fixed')\n",
        name = "wrapped.py",
    )
    plain = _scan(
        tmp_path,
        "import json, importlib\n"
        "loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "plain.py",
    )
    assert _sinks(wrapped) == set()
    assert "importlib.import_module" in _sinks(plain)


def test_nested_partials_accumulate_their_offsets(tmp_path):
    """`invoke = partial(add_path)` around a partial that already bound a position.

    The second wrapper counted only its own arguments, so it recorded no offset and the
    call was checked back at the original sink's position 1, which the first wrapper had
    already filled.
    """
    local = _scan(
        tmp_path,
        "import json, functools, sys\n"
        "def load(blob):\n"
        "    add_path = functools.partial(sys.path.insert, 0)\n"
        "    invoke = functools.partial(add_path)\n"
        "    return invoke(json.loads(blob)['path'])\n",
        name = "local.py",
    )
    at_module = _scan(
        tmp_path,
        "import json, functools, sys\n"
        "add_path = functools.partial(sys.path.insert, 0)\n"
        "invoke = functools.partial(add_path)\n"
        "def load(blob):\n"
        "    return invoke(json.loads(blob)['path'])\n",
        name = "at_module.py",
    )
    assert "sys.path.insert" in _sinks(local)
    assert "sys.path.insert" in _sinks(at_module)


def test_a_partial_checks_every_sink_its_target_can_be(tmp_path):
    """`action = subprocess.run` then `action = sys.path.insert`, wrapped in a partial.

    Keeping the first candidate meant the wrapper was checked against a sink whose
    watched position the offset had already removed, so the sink it really calls was
    never checked at all and nothing was reported.
    """
    findings = _scan(
        tmp_path,
        "import json, functools, subprocess, sys\n"
        "def load(blob):\n"
        "    action = subprocess.run\n"
        "    action = sys.path.insert\n"
        "    wrapper = functools.partial(action, 0)\n"
        "    return wrapper(json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_property_on_a_fresh_instance_resolves(tmp_path):
    """`importlib.import_module(Config().module)`, with no name for the receiver.

    The receiver is the construction itself, which has no spelling, so the getter lookup
    gave up before it started even though the construction is right there to resolve.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Config:\n"
        "    @property\n"
        "    def module(self):\n"
        "        return json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "def go():\n"
        "    return importlib.import_module(Config().module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_fresh_instance_property_returning_a_literal_is_quiet(tmp_path):
    """The guard: resolving the construction is not by itself a finding."""
    findings = _scan(
        tmp_path,
        "import importlib\n"
        "class Config:\n"
        "    @property\n"
        "    def module(self):\n"
        "        return 'torch'\n"
        "def go():\n"
        "    return importlib.import_module(Config().module)\n",
    )
    assert _sinks(findings) == set()


def test_a_nonlocal_sink_rebinding_reaches_the_enclosing_scope(tmp_path):
    """`nonlocal action; action = subprocess.run` in a helper, called in the outer scope.

    Only the taint reason was carried out of a nested body for `nonlocal` names, and the
    alias tables were restored unconditionally, so the outer `action(parsed)` matched
    neither the canonical table nor any alias.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    action = len\n"
        "    def setup():\n"
        "        nonlocal action\n"
        "        action = subprocess.run\n"
        "    setup()\n"
        "    return action(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_nonlocal_taint_rebinding_still_reaches_the_enclosing_scope(tmp_path):
    """The guard for that restructuring: the taint reason itself still comes out."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = 'ls'\n"
        "    def setup():\n"
        "        nonlocal command\n"
        "        command = json.loads(blob)['command']\n"
        "    setup()\n"
        "    return subprocess.run(command)\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_instance_built_in_a_class_body_resolves_for_methods(tmp_path):
    """`class App: runner = Runner()` then `self.runner.execute(parsed)`.

    Class-body settlement published the tainted bindings and discarded the types, so the
    shared collaborator a class declares this way had no type and the method behind it
    resolved to nothing.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class App:\n"
        "    runner = Runner()\n"
        "    def go(self, blob):\n"
        "        return self.runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_nested_parameter_masks_the_enclosing_binding(tmp_path):
    """An outer tainted `command` and `def helper(command)` called with a literal.

    The nested body was walked with every enclosing binding still live, so the parameter
    never shadowed the outer name and the helper reported a finding on a value it cannot
    receive. Masking costs nothing: the nested body is also analysed under its own
    qualname with its parameters bound by its real callers.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = json.loads(blob)['command']\n"
        "    def helper(command):\n"
        "        return subprocess.run(command)\n"
        "    return helper('ls')\n",
    )
    assert _sinks(findings) == set()


def test_a_real_caller_of_that_helper_is_still_reported(tmp_path):
    """The guard, two ways: a tainted argument, and a closure over the outer name.

    Masking the parameter must not lose the case the mask exists to make precise, nor
    the case where the nested body reads the enclosing name rather than shadowing it.
    """
    passed = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    def helper(command):\n"
        "        return subprocess.run(command)\n"
        "    return helper(json.loads(blob)['command'])\n",
        name = "passed.py",
    )
    closed_over = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = json.loads(blob)['command']\n"
        "    def helper():\n"
        "        return subprocess.run(command)\n"
        "    return helper()\n",
        name = "closed_over.py",
    )
    assert "subprocess.run" in _sinks(passed)
    assert "subprocess.run" in _sinks(closed_over)


def test_a_true_name_inside_splatted_kwargs_enables_remote_code(tmp_path):
    """`enabled = True` then `{"trust_remote_code": enabled}` splatted into a loader.

    The dict, `dict(...)` and subscript spellings accepted only a literal True, so the
    same name that is already caught when passed as a keyword went through unreported
    once it was put in a mapping first.
    """
    shapes = {
        "dict": "    kwargs = {'trust_remote_code': enabled}\n",
        "dict_call": "    kwargs = dict(trust_remote_code = enabled)\n",
        "subscript": "    kwargs = {}\n    kwargs['trust_remote_code'] = enabled\n",
    }
    for label, middle in shapes.items():
        findings = _scan(
            tmp_path,
            "from transformers import AutoModel\n"
            "def load(name):\n"
            "    enabled = True\n"
            + middle
            + "    return AutoModel.from_pretrained(name, **kwargs)\n",
            name = "sample_%s.py" % label,
        )
        assert any(f["sink"].startswith("trust_remote_code = True") for f in findings), label


def test_a_false_or_forwarded_name_inside_kwargs_stays_quiet(tmp_path):
    """The guards: the value still decides, and a forwarded parameter is the caller's."""
    for label, head in (
        ("false", "def load(name):\n    enabled = False\n"),
        ("forwarded", "def load(name, enabled):\n"),
    ):
        findings = _scan(
            tmp_path,
            "from transformers import AutoModel\n"
            + head
            + "    kwargs = {'trust_remote_code': enabled}\n"
            "    return AutoModel.from_pretrained(name, **kwargs)\n",
            name = "sample_%s.py" % label,
        )
        assert not any(f["sink"].startswith("trust_remote_code = True") for f in findings), label


def test_a_sink_declared_in_a_class_body_is_followed(tmp_path):
    """`class Hooks: loader = importlib.import_module` then `Hooks.loader(parsed)`.

    Only taint reasons and types left a class body, so the sink stored on the class
    attribute could not be recovered at the call.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Hooks:\n"
        "    loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return Hooks.loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_module_and_class_findings_carry_a_context_digest(tmp_path):
    """A module-scope finding has a non-empty digest that moves with its guard.

    Only indexed functions had one, so a finding at module or class scope was baselined
    with an empty digest, and weakening the validation around a reviewed sink kept the
    allowance while the call text was unchanged.
    """
    head = (
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "MOD = json.load(open(hf_hub_download('r', 'c.json')))['m']\n"
    )
    bare = _scan(tmp_path, head + "importlib.import_module(MOD)\n", name = "bare.py")
    guarded = _scan(
        tmp_path,
        head + "assert MOD in {'torch'}\nimportlib.import_module(MOD)\n",
        name = "guarded.py",
    )
    first = [f["context"] for f in bare if f["tier"] == "A"]
    second = [f["context"] for f in guarded if f["tier"] == "A"]
    assert first and second and all(first) and all(second)
    assert first != second


def test_a_mapped_callback_gets_one_slot_per_iterable(tmp_path):
    """`map(execute, fixed, parsed["commands"])` with `execute(label, command)`.

    The tainted iterable was always put in slot 0, which tainted `label` and left
    `command`, the parameter that reaches the sink, clean.
    """
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "def execute(label, command):\n"
        "    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import execute\n"
        "def load(blob):\n"
        "    return list(map(execute, ['a'], json.loads(blob)['commands']))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_a_mapped_callback_with_the_taint_in_the_other_slot_is_quiet(tmp_path):
    """The guard: the slot that never reaches the sink stays clean."""
    (tmp_path / "producer.py").write_text(
        "import subprocess\n"
        "def execute(label, command):\n"
        "    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import execute\n"
        "def load(blob):\n"
        "    return list(map(execute, json.loads(blob)['labels'], ['ls']))\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", consumer], roots = [tmp_path])
    assert _sinks(findings) == set()


def test_a_wildcard_import_binds_names_the_exporter_imported(tmp_path):
    """`producer.py` doing `from impl import execute`, then `from producer import *`.

    `import *` binds public imported names exactly like local definitions when there is
    no `__all__`, and leaving them out meant the re-exported helper never resolved.
    """
    (tmp_path / "producer.py").write_text("from impl import execute\n", encoding = "utf-8")
    (tmp_path / "impl.py").write_text(
        "import subprocess\ndef execute(command):\n    return subprocess.run(command)\n",
        encoding = "utf-8",
    )
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "import json\nfrom producer import *\n"
        "def load(blob):\n"
        "    return execute(json.loads(blob)['command'])\n",
        encoding = "utf-8",
    )
    findings = L.scan([tmp_path / "producer.py", tmp_path / "impl.py", consumer], roots = [tmp_path])
    assert "subprocess.run" in _sinks(findings)


def test_an_alias_of_torch_load_is_still_checked(tmp_path):
    """`loader = torch.load` then `loader(downloaded, weights_only = False)`.

    The conditional check recognised only the canonical spelling of the callee, so the
    same unsafe unpickling through a local or module-scope name reported nothing.
    """
    tail = "    return loader(hf_hub_download(repo, 'w.bin'), weights_only = False)\n"
    local = _scan(
        tmp_path,
        "import torch\nfrom huggingface_hub import hf_hub_download\n"
        "def load(repo):\n    loader = torch.load\n" + tail,
        name = "local.py",
    )
    at_module = _scan(
        tmp_path,
        "import torch\nfrom huggingface_hub import hf_hub_download\n"
        "loader = torch.load\ndef load(repo):\n" + tail,
        name = "at_module.py",
    )
    assert "torch.load(weights_only = False)" in _sinks(local)
    assert "torch.load(weights_only = False)" in _sinks(at_module)


def test_an_alias_of_torch_load_follows_the_direct_rule(tmp_path):
    """The alias is judged exactly like the direct call: True is quiet, omitted is not."""
    head = (
        "import torch\nfrom huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    loader = torch.load\n"
    )
    held = _scan(
        tmp_path,
        head + "    return loader(hf_hub_download(repo, 'w.bin'), weights_only = True)\n",
        name = "held.py",
    )
    omitted = _scan(
        tmp_path,
        head + "    return loader(hf_hub_download(repo, 'w.bin'))\n",
        name = "omitted.py",
    )
    assert _sinks(held) == set()
    assert "torch.load(weights_only unset)" in _sinks(omitted)


def test_a_partial_of_a_reflection_alias_does_not_crash(tmp_path):
    """`g = getattr; partial(g, transformers)` used to index the sink table with a marker.

    `getattr` is checked separately rather than through the table, so a partial wrapping
    an alias of it must not be treated as a table sink.
    """
    _scan(
        tmp_path,
        "import json, functools, transformers\n"
        "def load(blob):\n"
        "    g = getattr\n"
        "    resolve = functools.partial(g, transformers)\n"
        "    return resolve(json.loads(blob)['cls'])\n",
    )


def test_a_module_alias_bound_inside_a_compound_statement_counts(tmp_path):
    """`if enabled: loader = importlib.import_module`, and the `try` import shim.

    Only direct children of the module were collected, so a global alias bound under a
    module-level `if`, `try`, `with` or loop was skipped and every function calling it
    matched nothing.
    """
    branch = _scan(
        tmp_path,
        "import json, importlib, os\n"
        "if os.environ.get('X'):\n"
        "    loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "branch.py",
    )
    shim = _scan(
        tmp_path,
        "import json\n"
        "try:\n"
        "    from importlib import import_module as loader\n"
        "except ImportError:\n"
        "    loader = None\n"
        "def load(blob):\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "shim.py",
    )
    assert "importlib.import_module" in _sinks(branch)
    assert "importlib.import_module" in _sinks(shim)


def test_a_source_stored_on_an_attribute_is_followed(tmp_path):
    """`self.decode = json.loads` then `import_module(self.decode(blob)[...])`.

    Only name targets were recorded, so the decoder behind an attribute read clean, in
    the same method and in any other one.
    """
    other_method = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Box:\n"
        "    def __init__(self):\n"
        "        self.decode = json.loads\n"
        "    def go(self, blob):\n"
        "        return importlib.import_module(self.decode(blob)['module'])\n",
        name = "other_method.py",
    )
    same_method = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Box:\n"
        "    def go(self, blob):\n"
        "        self.decode = json.loads\n"
        "        return importlib.import_module(self.decode(blob)['module'])\n",
        name = "same_method.py",
    )
    assert "importlib.import_module" in _sinks(other_method)
    assert "importlib.import_module" in _sinks(same_method)


def test_a_local_name_shadows_a_tainted_global(tmp_path):
    """A parsed module-level `command`, and functions whose own `command` is local.

    A parameter or any name a function binds is local for the whole body, so the
    global is unreachable from there, yet falling through to it reported a helper that
    is only ever called with a literal.
    """
    head = (
        "import json, subprocess\n"
        "from huggingface_hub import hf_hub_download\n"
        "command = %s\n" % "json.load(open(hf_hub_download('r', 'c.json')))['command']"
    )
    shapes = {
        "parameter": "def execute(command):\n    return subprocess.run(command)\n"
        "def go():\n    return execute('ls')\n",
        "assignment": "def go():\n    command = 'ls'\n    return subprocess.run(command)\n",
        "nested": "def go():\n    def helper(command):\n        return subprocess.run(command)\n"
        "    return helper('ls')\n",
    }
    for label, tail in shapes.items():
        findings = _scan(tmp_path, head + tail, name = "sample_%s.py" % label)
        assert _sinks(findings) == set(), label


def test_a_tainted_global_is_still_reported_where_it_is_read(tmp_path):
    """The guards: a real global read, a `global` declaration, and a tainted caller."""
    head = (
        "import json, subprocess\n"
        "from huggingface_hub import hf_hub_download\n"
        "command = %s\n" % "json.load(open(hf_hub_download('r', 'c.json')))['command']"
    )
    shapes = {
        "read": "def go():\n    return subprocess.run(command)\n",
        "declared": "def go():\n    global command\n    return subprocess.run(command)\n",
        "caller": "def execute(command):\n    return subprocess.run(command)\n"
        "def go():\n    return execute(command)\n",
    }
    for label, tail in shapes.items():
        findings = _scan(tmp_path, head + tail, name = "sample_%s.py" % label)
        assert "subprocess.run" in _sinks(findings), label


def test_a_class_attribute_does_not_inherit_a_module_global_of_its_name(tmp_path):
    """Module-level `runner = Dirty()` and `class App: runner = Safe()`.

    Every pass was seeded with the module's instances, so the class attribute became
    both types and a call through it was propagated into a class it never holds.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Dirty:\n"
        "    def execute(self, c):\n"
        "        return subprocess.run(c)\n"
        "class Safe:\n"
        "    def execute(self, c):\n"
        "        return len(c)\n"
        "runner = Dirty()\n"
        "class App:\n"
        "    runner = Safe()\n"
        "    def go(self, blob):\n"
        "        return self.runner.execute(json.loads(blob)['c'])\n",
    )
    assert _sinks(findings) == set()


def test_a_class_attribute_of_the_unsafe_type_is_still_reported(tmp_path):
    """The guard, with the two classes swapped."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Dirty:\n"
        "    def execute(self, c):\n"
        "        return subprocess.run(c)\n"
        "class Safe:\n"
        "    def execute(self, c):\n"
        "        return len(c)\n"
        "runner = Safe()\n"
        "class App:\n"
        "    runner = Dirty()\n"
        "    def go(self, blob):\n"
        "        return self.runner.execute(json.loads(blob)['c'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_nonlocal_rebinding_merges_with_the_outer_alias(tmp_path):
    """`action = subprocess.run`, then a helper rebinding it to `sys.path.insert`.

    The nested candidates were carried only when the outer name had no entry, so the
    old `subprocess.run` survived alone and the sink the helper rebound it to, with its
    own watched position, was never checked.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess, sys\n"
        "def load(blob):\n"
        "    action = subprocess.run\n"
        "    def rebind():\n"
        "        nonlocal action\n"
        "        action = sys.path.insert\n"
        "    rebind()\n"
        "    return action(0, json.loads(blob)['path'])\n",
    )
    assert "sys.path.insert" in _sinks(findings)


def test_a_name_assigned_in_a_nested_function_masks_the_outer_one(tmp_path):
    """`def helper(): command = "fixed"; subprocess.run(command)` under a tainted outer.

    Only the nested function's parameters masked the enclosing binding, but any name it
    assigns is local throughout it.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def load(blob):\n"
        "    command = json.loads(blob)['command']\n"
        "    def helper():\n"
        "        command = 'fixed'\n"
        "        return subprocess.run(command)\n"
        "    return helper()\n",
    )
    assert _sinks(findings) == set()


def test_a_mutated_constant_map_is_no_longer_a_sanitiser(tmp_path):
    """`MAP = {"safe": "torch"}` then `MAP.update(parsed)` or `MAP["safe"] = parsed`.

    The exemption recorded only the literal, so a download that replaces the values
    still had `MAP["safe"]` treated as validated.
    """
    shapes = {
        "update": "    MAP.update(json.loads(blob))\n",
        "item": "    MAP['safe'] = json.loads(blob)['m']\n",
    }
    for label, middle in shapes.items():
        findings = _scan(
            tmp_path,
            "import json, importlib\n"
            "MAP = {'safe': 'torch'}\n"
            "def load(blob):\n" + middle + "    return importlib.import_module(MAP['safe'])\n",
            name = "sample_%s.py" % label,
        )
        assert "importlib.import_module" in _sinks(findings), label


def test_an_untouched_constant_map_is_still_a_sanitiser(tmp_path):
    """The guard: a lookup in a table nothing writes to stays the validation it is."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "MAP = {'safe': 'torch'}\n"
        "def load(blob):\n"
        "    return importlib.import_module(MAP.get(json.loads(blob)['k'], 'torch'))\n",
    )
    assert _sinks(findings) == set()


def test_a_partial_of_torch_load_keeps_its_unsafe_option(tmp_path):
    """`partial(torch.load, weights_only = False)` called with a downloaded path.

    The partial branch dropped the conditional sink's marker, so creation recognised
    nothing and the call no longer carried the unsafe keyword. A path bound at creation
    is checked there too.
    """
    head = "import torch, functools\nfrom huggingface_hub import hf_hub_download\ndef load(repo):\n"
    later = _scan(
        tmp_path,
        head + "    unsafe = functools.partial(torch.load, weights_only = False)\n"
        "    return unsafe(hf_hub_download(repo, 'w.bin'))\n",
        name = "later.py",
    )
    bound = _scan(
        tmp_path,
        head + "    unsafe = functools.partial(torch.load, hf_hub_download(repo, 'w.bin'), "
        "weights_only = False)\n"
        "    return unsafe()\n",
        name = "bound.py",
    )
    omitted = _scan(
        tmp_path,
        head + "    loader = functools.partial(torch.load, map_location = 'cpu')\n"
        "    return loader(hf_hub_download(repo, 'w.bin'))\n",
        name = "omitted.py",
    )
    assert "torch.load(weights_only = False)" in _sinks(later)
    assert "torch.load(weights_only = False)" in _sinks(bound)
    assert "torch.load(weights_only unset)" in _sinks(omitted)


def test_a_partial_of_torch_load_pinned_safe_is_quiet(tmp_path):
    """The guards: a wrapper that pins True, and an unsafe wrapper over a fixed path."""
    pinned = _scan(
        tmp_path,
        "import torch, functools\nfrom huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    safe = functools.partial(torch.load, weights_only = True)\n"
        "    return safe(hf_hub_download(repo, 'w.bin'))\n",
        name = "pinned.py",
    )
    literal = _scan(
        tmp_path,
        "import torch, functools\n"
        "def load():\n"
        "    unsafe = functools.partial(torch.load, weights_only = False)\n"
        "    return unsafe('/opt/x.bin')\n",
        name = "literal.py",
    )
    assert _sinks(pinned) == set()
    assert _sinks(literal) == set()


def test_a_partial_of_a_bound_method_is_followed(tmp_path):
    """`invoke = functools.partial(runner.execute)` then `invoke(parsed)`.

    Neither the local alias table nor `callable_alias` resolved `runner` through the
    tracked instances, so the method behind the wrapper never saw the tainted argument.
    """
    findings = _scan(
        tmp_path,
        "import json, functools, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def load(blob):\n"
        "    runner = Runner()\n"
        "    invoke = functools.partial(runner.execute)\n"
        "    return invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_an_aliased_loader_still_gates_remote_code(tmp_path):
    """`loader = AutoModel.from_pretrained` then `loader(..., trust_remote_code = True)`.

    Loaders are handled by their own check rather than the sink table, so an alias was
    never recorded and the explicit opt-in read as a call to nothing in particular.
    """
    local = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = AutoModel.from_pretrained\n"
        "    return loader(name, trust_remote_code = True)\n",
        name = "local.py",
    )
    at_module = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "loader = AutoModel.from_pretrained\n"
        "def load(name):\n"
        "    return loader(name, trust_remote_code = True)\n",
        name = "at_module.py",
    )
    quiet = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = AutoModel.from_pretrained\n"
        "    return loader(name, trust_remote_code = False)\n",
        name = "quiet.py",
    )
    for findings in (local, at_module):
        assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in quiet)


def test_a_flag_rebound_away_from_true_does_not_prove_weights_only(tmp_path):
    """`safe = True; safe = False` then `weights_only = safe`, and other non-proofs.

    "Ever bound True" is the fail-closed reading for enabling remote code and the wrong
    one for proving a load safe, so a name proves `weights_only` only when every binding
    of it in scope is True. A parameter or a loop target is not proof either.
    """
    head = "import torch\nfrom huggingface_hub import hf_hub_download\n"
    tail = "    return torch.load(hf_hub_download(repo, 'w.bin'), weights_only = safe)\n"
    shapes = {
        "rebound": "def load(repo):\n    safe = True\n    safe = False\n",
        "loop": "def load(repo):\n    safe = True\n    for safe in (False,):\n        pass\n",
        "parameter": "def load(repo, safe = True):\n",
    }
    for label, middle in shapes.items():
        findings = _scan(tmp_path, head + middle + tail, name = "sample_%s.py" % label)
        assert "torch.load(weights_only = False)" in _sinks(findings), label


def test_a_flag_that_is_always_true_still_proves_weights_only(tmp_path):
    """The guards: a single True binding, a chain of them, and a module-level one."""
    head = "import torch\nfrom huggingface_hub import hf_hub_download\n"
    tail = "    return torch.load(hf_hub_download(repo, 'w.bin'), weights_only = safe)\n"
    shapes = {
        "single": "def load(repo):\n    safe = True\n",
        "chained": "def load(repo):\n    flag = True\n    safe = flag\n",
        "module": "safe = True\ndef load(repo):\n",
    }
    for label, middle in shapes.items():
        findings = _scan(tmp_path, head + middle + tail, name = "sample_%s.py" % label)
        assert _sinks(findings) == set(), label


def test_ever_true_still_enables_remote_code(tmp_path):
    """And the remote-code side keeps its fail-closed reading: a branch that sets True."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name, cond):\n"
        "    enabled = False\n"
        "    if cond:\n"
        "        enabled = True\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = enabled)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_mutation_through_an_alias_reaches_the_original_table(tmp_path):
    """`alias = MAP` then `alias.update(parsed)`, and the import reads `MAP`.

    Both names hold one object, but only the receiver was tainted, so the original
    table read clean, and it also kept its literal-table exemption.
    """
    for label, head in (
        ("module", "MAP = {'safe': 'torch'}\ndef load(blob):\n"),
        ("local", "def load(blob):\n    MAP = {'safe': 'torch'}\n"),
    ):
        findings = _scan(
            tmp_path,
            "import json, importlib\n" + head + "    alias = MAP\n"
            "    alias.update(json.loads(blob))\n"
            "    return importlib.import_module(MAP['safe'])\n",
            name = "sample_%s.py" % label,
        )
        assert "importlib.import_module" in _sinks(findings), label


def test_a_mutated_copy_leaves_the_original_table_clean(tmp_path):
    """The guard: `dict(table)` is a new object, so its mutation is not the original's."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "def load(blob):\n"
        "    table = {'safe': 'torch'}\n"
        "    other = dict(table)\n"
        "    other.update(json.loads(blob))\n"
        "    return importlib.import_module(table['safe'])\n",
    )
    assert _sinks(findings) == set()


def test_a_module_alias_bound_in_a_match_case_counts(tmp_path):
    """Module-level `match`: each case keeps its statements under its own body."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "match 1:\n"
        "    case 1:\n"
        "        loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_walrus_binding_records_its_alias(tmp_path):
    """`if (loader := importlib.import_module): loader(parsed)`.

    The walrus carried only taint, so the sink it bound was never recorded and the call
    matched neither table.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "def load(blob):\n"
        "    if (loader := importlib.import_module):\n"
        "        return loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_an_annotated_true_flag_inside_kwargs_enables_remote_code(tmp_path):
    """`enabled: bool = True` then `{"trust_remote_code": enabled}` splatted."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    enabled: bool = True\n"
        "    kwargs = {'trust_remote_code': enabled}\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_module_level_partial_of_torch_load_is_collected(tmp_path):
    """`load_cpu = functools.partial(torch.load, map_location = "cpu")` at module scope.

    Setting loader defaults once at module scope is common, and the module collector
    matched wrapped callables only against the sink table, where the conditional
    `torch.load` sink deliberately is not, so a call from any function read clean.
    """
    head = "import torch, functools\nfrom huggingface_hub import hf_hub_download\n"
    tail = "def load(repo):\n    return wrapper(hf_hub_download(repo, 'w.bin'))\n"
    unsafe = _scan(
        tmp_path,
        head + "wrapper = functools.partial(torch.load, weights_only = False)\n" + tail,
        name = "unsafe.py",
    )
    omitted = _scan(
        tmp_path,
        head + "wrapper = functools.partial(torch.load, map_location = 'cpu')\n" + tail,
        name = "omitted.py",
    )
    pinned = _scan(
        tmp_path,
        head + "wrapper = functools.partial(torch.load, weights_only = True)\n" + tail,
        name = "pinned.py",
    )
    assert "torch.load(weights_only = False)" in _sinks(unsafe)
    assert "torch.load(weights_only unset)" in _sinks(omitted)
    assert _sinks(pinned) == set()


def test_a_chained_loader_alias_still_gates_remote_code(tmp_path):
    """`loader = AutoModel.from_pretrained; invoke = loader`, then the opt-in."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = AutoModel.from_pretrained\n"
        "    invoke = loader\n"
        "    return invoke(name, trust_remote_code = True)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_bound_method_stored_on_self_is_followed(tmp_path):
    """`self.invoke = runner.execute` in the constructor, `self.invoke(parsed)` later.

    Callbacks are stored this way, and only name targets were recorded, so the method
    behind the attribute never saw the tainted argument.
    """
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        runner = Runner()\n"
        "        self.invoke = runner.execute\n"
        "    def go(self, blob):\n"
        "        return self.invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_bound_method_on_self_called_with_a_literal_is_quiet(tmp_path):
    """The guard: following the stored method does not taint what it receives."""
    findings = _scan(
        tmp_path,
        "import subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        runner = Runner()\n"
        "        self.invoke = runner.execute\n"
        "    def go(self):\n"
        "        return self.invoke('ls')\n",
    )
    assert _sinks(findings) == set()


def test_a_class_body_binding_inside_a_loop_is_published(tmp_path):
    """A class attribute bound inside a `while` was skipped by the binding allowlist."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "class Config:\n"
        "    while True:\n"
        "        module = json.load(open(hf_hub_download('r', 'c.json')))['module']\n"
        "        break\n"
        "def go():\n"
        "    return importlib.import_module(Config.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_pathlib_transforms_keep_a_download_tainted(tmp_path):
    """`Path(snapshot_download(repo)).resolve()` then `sys.path.insert`.

    `resolve`, `absolute`, `expanduser`, `as_posix` and `joinpath` return the same
    attacker-chosen location in another form, and each one laundered the download
    before it reached the import path.
    """
    head = (
        "import sys\nfrom pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
    )
    for method in ("resolve()", "absolute()", "expanduser()", "as_posix()", "joinpath('src')"):
        findings = _scan(
            tmp_path,
            head
            + "    p = Path(snapshot_download(repo)).%s\n" % method
            + "    sys.path.insert(0, str(p))\n",
            name = "sample_%s.py" % method.split("(")[0],
        )
        assert "sys.path.insert" in _sinks(findings), method


def test_a_resolved_fixed_path_is_quiet(tmp_path):
    """The guard: the transform carries taint, it does not create it."""
    findings = _scan(
        tmp_path,
        "import sys\nfrom pathlib import Path\n"
        "def load():\n"
        "    sys.path.insert(0, str(Path('/opt/fixed').resolve()))\n",
    )
    assert _sinks(findings) == set()


def test_a_callable_default_is_an_alias(tmp_path):
    """`def load(blob, loader = importlib.import_module)` runs the default when omitted."""
    sink = _scan(
        tmp_path,
        "import json, importlib\n"
        "def load(blob, loader = importlib.import_module):\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "sink.py",
    )
    helper = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(c):\n"
        "    return subprocess.run(c)\n"
        "def load(blob, runner = execute):\n"
        "    return runner(json.loads(blob)['c'])\n",
        name = "helper.py",
    )
    literal = _scan(
        tmp_path,
        "import importlib\n"
        "def load(loader = importlib.import_module):\n"
        "    return loader('torch')\n",
        name = "literal.py",
    )
    assert "importlib.import_module" in _sinks(sink)
    assert "subprocess.run" in _sinks(helper)
    assert _sinks(literal) == set()


def test_remote_code_written_by_a_mapping_method_is_reported(tmp_path):
    """`kwargs.update(trust_remote_code = True)` and `kwargs.setdefault(...)`.

    Both write the key into a mapping that is splatted into a loader later, which
    neither the literal nor the `dict(...)` spelling covered.
    """
    shapes = {
        "update": "    kwargs.update(trust_remote_code = True)\n",
        "setdefault": "    kwargs.setdefault('trust_remote_code', True)\n",
    }
    for label, middle in shapes.items():
        findings = _scan(
            tmp_path,
            "from transformers import AutoModel\n"
            "def load(name):\n"
            "    kwargs = {}\n" + middle + "    return AutoModel.from_pretrained(name, **kwargs)\n",
            name = "sample_%s.py" % label,
        )
        assert any(f["sink"].startswith("trust_remote_code = True") for f in findings), label
    quiet = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    kwargs = {}\n"
        "    kwargs.update(trust_remote_code = False)\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n",
        name = "sample_false.py",
    )
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in quiet)


def test_a_local_binding_hides_the_module_alias_of_its_name(tmp_path):
    """Module `loader = importlib.import_module`, function `loader = len`.

    Every reference to `loader` in that function is local, so seeding the module alias
    reported a call to a safe helper as a dynamic import.
    """
    shadowed = _scan(
        tmp_path,
        "import json, importlib\n"
        "loader = importlib.import_module\n"
        "def load(blob):\n"
        "    loader = len\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "shadowed.py",
    )
    plain = _scan(
        tmp_path,
        "import json, importlib\n"
        "loader = importlib.import_module\n"
        "def load(blob):\n"
        "    return loader(json.loads(blob)['module'])\n",
        name = "plain.py",
    )
    assert _sinks(shadowed) == set()
    assert "importlib.import_module" in _sinks(plain)


def test_a_torch_load_partial_pinned_by_a_proven_name_is_quiet(tmp_path):
    """`safe = True; partial(torch.load, weights_only = safe)` is a safe wrapper.

    The creation check accepted the proven name but the wrapper was still marked
    unsafe, so every call through it was reported as explicitly unsafe.
    """
    findings = _scan(
        tmp_path,
        "import torch, functools\nfrom huggingface_hub import hf_hub_download\n"
        "def load(repo):\n"
        "    safe = True\n"
        "    loader = functools.partial(torch.load, weights_only = safe)\n"
        "    return loader(hf_hub_download(repo, 'w.bin'))\n",
    )
    assert _sinks(findings) == set()


def test_a_decoder_stored_after_its_reader_still_converges(tmp_path):
    """`a_read` uses `self.decode`, which a later-sorted `z_install` assigns.

    The convergence snapshot compared only held instances among the shared attribute
    maps, so a decoder discovered after its reader changed nothing it looked at, the
    fixpoint stopped a pass early, and the reader's return summary stayed clean.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Box:\n"
        "    def a_read(self, blob):\n"
        "        return self.decode(blob)['module']\n"
        "    def z_install(self):\n"
        "        self.decode = json.loads\n"
        "def main(blob):\n"
        "    return importlib.import_module(Box().a_read(blob))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_module_container_mutated_in_a_function_is_shared_state(tmp_path):
    """`SETTINGS["module"] = parsed` in one function, imported in another.

    A mutation needs no `global` declaration, and only declared globals were published,
    so the shared dict read clean everywhere else.
    """
    for label, write in (
        ("item", "    SETTINGS['module'] = json.loads(blob)['module']\n"),
        ("update", "    SETTINGS.update(json.loads(blob))\n"),
    ):
        findings = _scan(
            tmp_path,
            "import json, importlib\n"
            "SETTINGS = {}\n"
            "def parse(blob):\n" + write + "def run():\n"
            "    return importlib.import_module(SETTINGS['module'])\n",
            name = "sample_%s.py" % label,
        )
        assert "importlib.import_module" in _sinks(findings), label


def test_a_local_container_of_the_same_name_does_not_touch_the_global(tmp_path):
    """The guards: a function's own `SETTINGS = {}`, and a closure's list."""
    local = _scan(
        tmp_path,
        "import json, importlib\n"
        "SETTINGS = {'module': 'torch'}\n"
        "def parse(blob):\n"
        "    SETTINGS = {}\n"
        "    SETTINGS['module'] = json.loads(blob)['module']\n"
        "def run():\n"
        "    return importlib.import_module(SETTINGS['module'])\n",
        name = "local.py",
    )
    assert _sinks(local) == set()


def test_a_plain_function_stored_as_a_callback_is_followed(tmp_path):
    """`self.invoke = execute` in the constructor, `self.invoke(parsed)` later."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "class Holder:\n"
        "    def __init__(self):\n"
        "        self.invoke = execute\n"
        "    def go(self, blob):\n"
        "        return self.invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_source_declared_in_a_class_body_is_followed(tmp_path):
    """`class Box: decode = json.loads` then `self.decode(blob)` in a method."""
    findings = _scan(
        tmp_path,
        "import json, importlib\n"
        "class Box:\n"
        "    decode = json.loads\n"
        "    def go(self, blob):\n"
        "        return importlib.import_module(self.decode(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_local_flag_hides_a_module_true_flag_of_its_name(tmp_path):
    """Module `ENABLED = True`, function `ENABLED = False` passed as the opt-in."""
    shadowed = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = True\n"
        "def load(name):\n"
        "    ENABLED = False\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = ENABLED)\n",
        name = "shadowed.py",
    )
    plain = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = True\n"
        "def load(name):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = ENABLED)\n",
        name = "plain.py",
    )
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in shadowed)
    assert any(f["sink"].startswith("trust_remote_code = True") for f in plain)


def test_a_module_level_bound_method_alias_is_followed(tmp_path):
    """`runner = Runner()` and `invoke = runner.execute` at module scope."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "runner = Runner()\n"
        "invoke = runner.execute\n"
        "def load(blob):\n"
        "    return invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_function_under_a_module_branch_is_reported_once(tmp_path):
    """A function defined under a module-level `if` was walked by the module pass too.

    It sits inside the module body's statements, so each sink in it was reported once
    under `<module>` and once under its own name, which doubled its baseline entries.
    """
    findings = _scan(
        tmp_path,
        "import json, importlib, os\n"
        "if os.environ.get('X'):\n"
        "    def load(blob):\n"
        "        return importlib.import_module(json.loads(blob)['m'])\n",
    )
    reported = [(f["qualname"], f["sink"]) for f in findings if f["tier"] == "A"]
    assert reported == [("load", "importlib.import_module")]


def test_sys_path_extend_is_an_import_path_sink(tmp_path):
    """`sys.path.extend([downloaded])` adds the directory just as `append` does."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def add(repo):\n"
        "    sys.path.extend([snapshot_download(repo)])\n",
    )
    assert "sys.path.extend" in _sinks(findings)


def test_a_loader_partial_binding_remote_code_is_reported(tmp_path):
    """`partial(AutoModel.from_pretrained, trust_remote_code = True)` and its use."""
    created = _scan(
        tmp_path,
        "import functools\n"
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = functools.partial(AutoModel.from_pretrained, trust_remote_code = True)\n"
        "    return loader(name)\n",
        name = "created.py",
    )
    called = _scan(
        tmp_path,
        "import functools\n"
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = functools.partial(AutoModel.from_pretrained, revision = 'main')\n"
        "    return loader(name, trust_remote_code = True)\n",
        name = "called.py",
    )
    safe = _scan(
        tmp_path,
        "import functools\n"
        "from transformers import AutoModel\n"
        "def load(name):\n"
        "    loader = functools.partial(AutoModel.from_pretrained, trust_remote_code = False)\n"
        "    return loader(name)\n",
        name = "safe.py",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in created)
    assert any(f["sink"].startswith("trust_remote_code = True") for f in called)
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in safe)


def test_a_partial_of_a_source_stays_a_source(tmp_path):
    """`decode = partial(json.loads, ...)`, locally and at module scope."""
    local = _scan(
        tmp_path,
        "import functools, importlib, json\n"
        "def go(blob):\n"
        "    decode = functools.partial(json.loads, object_hook = dict)\n"
        "    return importlib.import_module(decode(blob)['module'])\n",
        name = "local.py",
    )
    module = _scan(
        tmp_path,
        "import functools, importlib, json\n"
        "decode = functools.partial(json.loads, object_hook = dict)\n"
        "def go(blob):\n"
        "    return importlib.import_module(decode(blob)['module'])\n",
        name = "module.py",
    )
    assert "importlib.import_module" in _sinks(local)
    assert "importlib.import_module" in _sinks(module)


def test_match_and_except_captures_are_locals_of_a_nested_helper(tmp_path):
    """A nested helper's own `case {"command": command}` hides the outer name."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def outer(blob):\n"
        "    command = json.loads(blob)['command']\n"
        "    def helper(spec):\n"
        "        match spec:\n"
        "            case {'command': command}:\n"
        "                return subprocess.run(command)\n"
        "    return helper({'command': ['ls']})\n",
    )
    assert "subprocess.run" not in _sinks(findings)


def test_a_sink_chosen_by_a_conditional_expression_is_an_alias(tmp_path):
    """`loader = importlib.import_module if enabled else safe_loader`."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def safe_loader(name):\n"
        "    return name\n"
        "def go(blob, enabled):\n"
        "    loader = importlib.import_module if enabled else safe_loader\n"
        "    return loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_sink_passed_to_a_helper_is_followed_into_it(tmp_path):
    """`invoke(subprocess.run, parsed)` with `def invoke(callback, value)`."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def invoke(callback, value):\n"
        "    return callback(value)\n"
        "def go(blob):\n"
        "    return invoke(subprocess.run, json.loads(blob)['command'])\n",
    )
    quiet = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def invoke(callback, value):\n"
        "    return callback(value)\n"
        "def go(blob):\n"
        "    return invoke(subprocess.run, ['ls'])\n",
        name = "quiet.py",
    )
    assert "subprocess.run" in _sinks(findings)
    assert "subprocess.run" not in _sinks(quiet)


def test_a_callback_declared_in_a_class_body_is_followed(tmp_path):
    """`class Hooks: invoke = execute` then `Hooks.invoke(parsed)`."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def execute(command):\n"
        "    return subprocess.run(command)\n"
        "class Hooks:\n"
        "    invoke = execute\n"
        "def go(blob):\n"
        "    return Hooks.invoke(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_closure_returning_a_captured_value_is_tainted(tmp_path):
    """`def get_module(): return module` with `module` parsed in the enclosing scope."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def go(blob):\n"
        "    module = json.loads(blob)['module']\n"
        "    def get_module():\n"
        "        return module\n"
        "    return importlib.import_module(get_module())\n",
    )
    quiet = _scan(
        tmp_path,
        "import importlib, json\n"
        "def go(blob):\n"
        "    module = json.loads(blob)['module']\n"
        "    def get_module():\n"
        "        return 'json'\n"
        "    return importlib.import_module(get_module())\n",
        name = "quiet.py",
    )
    assert "importlib.import_module" in _sinks(findings)
    assert "importlib.import_module" not in _sinks(quiet)


def test_subprocess_shell_helpers_are_command_sinks(tmp_path):
    """`subprocess.getoutput` and `getstatusoutput` run their argument in a shell."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def go(blob):\n"
        "    subprocess.getoutput(json.loads(blob)['command'])\n"
        "    subprocess.getstatusoutput(json.loads(blob)['command'])\n",
    )
    assert {"subprocess.getoutput", "subprocess.getstatusoutput"} <= _sinks(findings)


def test_a_source_passed_to_a_helper_is_followed_into_it(tmp_path):
    """`decode(json.loads, blob)` with `def decode(parser, value): return parser(value)`."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def decode(parser, value):\n"
        "    return parser(value)\n"
        "def go(blob):\n"
        "    return importlib.import_module(decode(json.loads, blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_constant_map_lookup_with_an_untrusted_fallback_is_tainted(tmp_path):
    """`MODULES.get(kind, cfg["fallback"])` returns the fallback when the key is absent."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "MODULES = {'a': 'json', 'b': 'csv'}\n"
        "def go(blob):\n"
        "    cfg = json.loads(blob)\n"
        "    return importlib.import_module(MODULES.get(cfg['kind'], cfg['fallback']))\n",
    )
    quiet = _scan(
        tmp_path,
        "import importlib, json\n"
        "MODULES = {'a': 'json', 'b': 'csv'}\n"
        "def go(blob):\n"
        "    cfg = json.loads(blob)\n"
        "    return importlib.import_module(MODULES.get(cfg['kind'], 'json'))\n",
        name = "quiet.py",
    )
    assert "importlib.import_module" in _sinks(findings)
    assert "importlib.import_module" not in _sinks(quiet)


def test_a_remote_code_loader_passed_to_a_helper_is_followed(tmp_path):
    """`load_with(AutoModel.from_pretrained, name)` opting in inside the helper."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def load_with(loader, name):\n"
        "    return loader(name, trust_remote_code = True)\n"
        "def go(name):\n"
        "    return load_with(AutoModel.from_pretrained, name)\n",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)


def test_a_construction_chosen_by_a_conditional_expression_resolves(tmp_path):
    """`runner = Runner() if enabled else None` then `runner.execute(parsed)`."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def go(blob, enabled):\n"
        "    runner = Runner() if enabled else None\n"
        "    return runner.execute(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_a_path_split_into_stem_and_suffix_stays_tainted(tmp_path):
    """`os.path.splitext(parsed)[0]` is still the untrusted name."""
    findings = _scan(
        tmp_path,
        "import importlib, json, os\n"
        "def go(blob):\n"
        "    return importlib.import_module(os.path.splitext(json.loads(blob)['module'])[0])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_dispatch_table_of_sinks_is_followed(tmp_path):
    """`LOADERS[kind](parsed)` with a sink in the table, at module and local scope."""
    module = _scan(
        tmp_path,
        "import importlib, json\n"
        "LOADERS = {'dynamic': importlib.import_module}\n"
        "def go(blob, kind):\n"
        "    return LOADERS[kind](json.loads(blob)['module'])\n",
        name = "module.py",
    )
    local = _scan(
        tmp_path,
        "import json, subprocess\n"
        "def go(blob, kind):\n"
        "    runners = [subprocess.run, print]\n"
        "    return runners[kind](json.loads(blob)['command'])\n",
        name = "local.py",
    )
    assert "importlib.import_module" in _sinks(module)
    assert "subprocess.run" in _sinks(local)


def test_a_stored_source_passed_to_a_helper_is_followed(tmp_path):
    """`self.decode = json.loads`, then `parse(self.decode, blob)` in another method."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def parse(parser, value):\n"
        "    return parser(value)\n"
        "class Loader:\n"
        "    def __init__(self):\n"
        "        self.decode = json.loads\n"
        "    def go(self, blob):\n"
        "        data = parse(self.decode, blob)\n"
        "        return importlib.import_module(data['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_remote_code_default_bound_to_a_true_name_is_reported(tmp_path):
    """`ENABLED = True` then `def load(name, trust_remote_code = ENABLED)`."""
    findings = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = True\n"
        "def load(name, trust_remote_code = ENABLED):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = trust_remote_code)\n",
    )
    quiet = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "ENABLED = False\n"
        "def load(name, trust_remote_code = ENABLED):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = trust_remote_code)\n",
        name = "quiet.py",
    )
    assert any(f["sink"].startswith("trust_remote_code = True") for f in findings)
    assert not any(f["sink"].startswith("trust_remote_code = True") for f in quiet)


def test_a_revision_forwarded_from_a_none_default_is_unpinned(tmp_path):
    """`def load(repo, revision = None)` passing `revision = revision` fetches the tip."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, revision = None):\n"
        "    path = snapshot_download(repo, revision = revision)\n"
        "    sys.path.insert(0, path)\n",
    )
    pinned = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, revision):\n"
        "    path = snapshot_download(repo, revision = revision)\n"
        "    sys.path.insert(0, path)\n",
        name = "pinned.py",
    )
    assert "unpinned code fetch" in _sinks(findings)
    assert "unpinned code fetch" not in _sinks(pinned)


def test_calling_an_instance_enters_its_call_method(tmp_path):
    """`runner = Runner()` then `runner(parsed)` runs `Runner.__call__`."""
    findings = _scan(
        tmp_path,
        "import json, subprocess\n"
        "class Runner:\n"
        "    def __call__(self, command):\n"
        "        return subprocess.run(command)\n"
        "def go(blob):\n"
        "    runner = Runner()\n"
        "    return runner(json.loads(blob)['command'])\n",
    )
    assert "subprocess.run" in _sinks(findings)


def test_pure_path_constructors_keep_taint(tmp_path):
    """`PurePath(snapshot_download(repo)).parent` still names the download."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from pathlib import PurePath\n"
        "from huggingface_hub import snapshot_download\n"
        "def add(repo):\n"
        "    sys.path.append(str(PurePath(snapshot_download(repo, revision = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')).parent))\n",
    )
    assert "sys.path.append" in _sinks(findings)


def test_aliases_unpacked_from_a_tuple_are_followed(tmp_path):
    """`loader, fallback = (importlib.import_module, print)`."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def go(blob):\n"
        "    loader, fallback = (importlib.import_module, print)\n"
        "    return loader(json.loads(blob)['module'])\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_branch_name_revision_is_not_a_pin(tmp_path):
    """`revision = "main"` moves like the default; a 40 hex commit pins."""
    branch = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = 'main'))\n",
        name = "branch.py",
    )
    commit = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = '0123456789abcdef0123456789abcdef01234567'))\n",
        name = "commit.py",
    )
    assert "unpinned code fetch" in _sinks(branch)
    assert "unpinned code fetch" not in _sinks(commit)


def test_assigning_to_sys_path_is_an_import_path_sink(tmp_path):
    """Slice, index, plain and augmented writes to `sys.path`."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def a(repo):\n"
        "    sys.path[:] = [snapshot_download(repo)] + sys.path\n"
        "def b(repo):\n"
        "    sys.path += [snapshot_download(repo)]\n",
    )
    quiet = _scan(
        tmp_path,
        "import sys\ndef a(kept):\n    old = list(sys.path)\n    sys.path[:] = old\n",
        name = "quiet.py",
    )
    assert sum(f["sink"] == "sys.path (assignment)" for f in findings if f["tier"] == "A") == 2
    assert "sys.path (assignment)" not in _sinks(quiet)


def test_a_helper_filling_a_container_argument_taints_the_caller(tmp_path):
    """`fill(settings, blob)` writing `settings["module"] = parsed` inside."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "def fill(settings, blob):\n"
        "    settings['module'] = json.loads(blob)['module']\n"
        "def go(blob):\n"
        "    settings = {}\n"
        "    fill(settings, blob)\n"
        "    return importlib.import_module(settings['module'])\n",
    )
    quiet = _scan(
        tmp_path,
        "import importlib, json\n"
        "def fill(settings, blob):\n"
        "    settings = {'module': json.loads(blob)['module']}\n"
        "    return settings\n"
        "def go(blob):\n"
        "    settings = {'module': 'json'}\n"
        "    fill(settings, blob)\n"
        "    return importlib.import_module(settings['module'])\n",
        name = "quiet.py",
    )
    assert "importlib.import_module" in _sinks(findings)
    assert "importlib.import_module" not in _sinks(quiet)


def test_a_config_parser_read_fills_the_parser(tmp_path):
    """`parser.read(downloaded)` then a value pulled back out of the parser."""
    findings = _scan(
        tmp_path,
        "import configparser, importlib\n"
        "from huggingface_hub import hf_hub_download\n"
        "def go(repo):\n"
        "    parser = configparser.ConfigParser()\n"
        "    parser.read(hf_hub_download(repo, 'plugin.cfg'))\n"
        "    return importlib.import_module(parser.get('plugin', 'module'))\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_a_branch_default_for_a_forwarded_revision_is_unpinned(tmp_path):
    """`def load(repo, revision = "main")` forwarding it fetches a moving branch."""
    findings = _scan(
        tmp_path,
        "import sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, revision = 'main'):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = revision))\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


def test_a_regex_substitution_keeps_taint(tmp_path):
    """`re.sub("-", "_", parsed)` still names whatever the config named."""
    findings = _scan(
        tmp_path,
        "import importlib, json, re\n"
        "def go(blob):\n"
        "    module = re.sub('-', '_', json.loads(blob)['module'])\n"
        "    return importlib.import_module(module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_unsafe_yaml_loading_of_a_download_is_a_sink(tmp_path):
    """`yaml.unsafe_load` and `yaml.load(..., Loader = yaml.UnsafeLoader)` run tags."""
    findings = _scan(
        tmp_path,
        "import yaml\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(repo):\n"
        "    yaml.unsafe_load(open(hf_hub_download(repo, 'c.yaml')))\n"
        "def b(repo):\n"
        "    yaml.load(open(hf_hub_download(repo, 'c.yaml')), Loader = yaml.UnsafeLoader)\n",
    )
    quiet = _scan(
        tmp_path,
        "import yaml\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(repo):\n"
        "    yaml.load(open(hf_hub_download(repo, 'c.yaml')), Loader = yaml.SafeLoader)\n",
        name = "quiet.py",
    )
    assert {"yaml.unsafe_load(unsafe loader)", "yaml.load(unsafe loader)"} <= _sinks(findings)
    assert not any("unsafe loader" in sink for sink in _sinks(quiet))


def test_a_dataclass_built_from_parsed_data_carries_it(tmp_path):
    """`cfg = Config(**json.loads(blob))` then `cfg.module` with a generated `__init__`."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Config:\n"
        "    module: str\n"
        "def go(blob):\n"
        "    cfg = Config(**json.loads(blob))\n"
        "    return importlib.import_module(cfg.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_keyword_streams_and_config_inputs_are_read(tmp_path):
    """`yaml.unsafe_load(stream = ...)` and `parser.read_dict(dictionary = ...)`."""
    findings = _scan(
        tmp_path,
        "import configparser, importlib, json, yaml\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(repo):\n"
        "    yaml.unsafe_load(stream = open(hf_hub_download(repo, 'c.yaml')))\n"
        "def b(blob):\n"
        "    parser = configparser.ConfigParser()\n"
        "    parser.read_dict(dictionary = json.loads(blob))\n"
        "    return importlib.import_module(parser.get('plugin', 'module'))\n",
    )
    assert {"yaml.unsafe_load(unsafe loader)", "importlib.import_module"} <= _sinks(findings)


def test_the_unpickler_object_form_is_a_sink(tmp_path):
    """`pickle.Unpickler(open(downloaded, "rb")).load()` runs the same reducers."""
    findings = _scan(
        tmp_path,
        "import pickle\n"
        "from huggingface_hub import hf_hub_download\n"
        "def go(repo):\n"
        "    return pickle.Unpickler(open(hf_hub_download(repo, 'x.pkl'), 'rb')).load()\n",
    )
    assert "pickle.Unpickler" in _sinks(findings)


def test_a_dataclass_over_a_base_with_an_init_still_carries_taint(tmp_path):
    """`@dataclass` generates its own `__init__` even when a base defines one."""
    findings = _scan(
        tmp_path,
        "import importlib, json\n"
        "from dataclasses import dataclass\n"
        "class Base:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "@dataclass\n"
        "class Config(Base):\n"
        "    module: str\n"
        "def go(blob):\n"
        "    cfg = Config(**json.loads(blob))\n"
        "    return importlib.import_module(cfg.module)\n",
    )
    assert "importlib.import_module" in _sinks(findings)


def test_byte_streams_casefold_and_aliased_loaders_keep_taint(tmp_path):
    """`io.BytesIO(...)`, `.casefold()` and `UnsafeLoader as Danger` do not launder."""
    findings = _scan(
        tmp_path,
        "import importlib, io, json, pickle, requests, yaml\n"
        "from yaml import UnsafeLoader as Danger\n"
        "def a(url):\n"
        "    return pickle.load(io.BytesIO(requests.get(url).content))\n"
        "def b(blob):\n"
        "    return importlib.import_module(json.loads(blob)['module'].casefold())\n"
        "def c(url):\n"
        "    return yaml.load(stream = requests.get(url).text, Loader = Danger)\n",
    )
    assert {"pickle.load", "importlib.import_module", "yaml.load(unsafe loader)"} <= _sinks(
        findings
    )


def test_the_default_scope_includes_the_docker_helpers():
    """The images copy `docker/*.py` in, so the CI scan has to cover them."""
    assert "docker" in L.DEFAULT_TARGETS


def test_objects_and_callbacks_handed_to_a_helper_are_followed(tmp_path):
    """`invoke(runner, parsed)` and `invoke(execute, parsed)`."""
    findings = _scan(
        tmp_path,
        "import importlib, json, subprocess\n"
        "class Runner:\n"
        "    def execute(self, command):\n"
        "        return subprocess.run(command)\n"
        "def call_method(runner, command):\n"
        "    return runner.execute(command)\n"
        "def load(name):\n"
        "    return importlib.import_module(name)\n"
        "def call_back(callback, value):\n"
        "    return callback(value)\n"
        "def go(blob):\n"
        "    runner = Runner()\n"
        "    call_method(runner, json.loads(blob)['command'])\n"
        "    call_back(load, json.loads(blob)['module'])\n",
    )
    assert {"subprocess.run", "importlib.import_module"} <= _sinks(findings)


def test_a_parsed_mapping_expanded_into_a_loader_is_reported(tmp_path):
    """`AutoModel.from_pretrained(name, **json.loads(blob))` lets the data opt in."""
    findings = _scan(
        tmp_path,
        "import json\n"
        "from transformers import AutoModel\n"
        "def load(name, blob):\n"
        "    return AutoModel.from_pretrained(name, **json.loads(blob))\n",
    )
    quiet = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo):\n"
        "    kwargs = {'pretrained_model_name_or_path': snapshot_download(repo)}\n"
        "    return AutoModel.from_pretrained(**kwargs)\n",
        name = "quiet.py",
    )
    assert "trust_remote_code (untrusted **kwargs)" in _sinks(findings)
    assert "trust_remote_code (untrusted **kwargs)" not in _sinks(quiet)


def test_normcase_shell_executable_and_local_revision_aliases(tmp_path):
    """`os.path.normcase`, `create_subprocess_shell(executable = ...)`, `revision = "main"`."""
    findings = _scan(
        tmp_path,
        "import asyncio, json, os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def a(repo):\n"
        "    sys.path.append(os.path.normcase(snapshot_download(repo, revision = 'abababababababababababababababababababab')))\n"
        "async def b(blob):\n"
        "    await asyncio.create_subprocess_shell('echo ok', executable = json.loads(blob)['shell'])\n"
        "def c(repo):\n"
        "    revision = 'main'\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = revision))\n",
    )
    sinks = _sinks(findings)
    assert {"sys.path.append", "asyncio.create_subprocess_shell", "unpinned code fetch"} <= sinks


def test_a_full_run_reports_allowances_for_deleted_files(monkeypatch):
    """A whole-repository run checks the complete baseline, deleted files included."""
    monkeypatch.setattr(L, "scan", lambda targets, **kwargs: [])
    monkeypatch.setattr(
        L,
        "_load_baseline",
        lambda: {"gone_for_good.py::f::importlib.import_module::0::0": 1},
    )
    assert L.main([]) != 0


def test_string_and_iterator_deserialisers_and_the_yaml_object_form(tmp_path):
    """`toml.loads`, `yaml.safe_load_all`, and `yaml.Loader(stream)` on a download."""
    findings = _scan(
        tmp_path,
        "import importlib, toml, yaml\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(blob):\n"
        "    return importlib.import_module(toml.loads(blob)['module'])\n"
        "def b(blob):\n"
        "    return importlib.import_module(next(yaml.safe_load_all(blob))['module'])\n"
        "def c(repo):\n"
        "    return yaml.Loader(open(hf_hub_download(repo, 'c.yaml'))).get_data()\n",
    )
    sinks = _sinks(findings)
    assert "yaml.Loader" in sinks
    assert sum(f["sink"] == "importlib.import_module" for f in findings if f["tier"] == "A") == 2


def test_a_revision_read_from_data_is_not_a_pin(tmp_path):
    """`revision = json.loads(blob)["revision"]` lets the data choose a moving branch."""
    findings = _scan(
        tmp_path,
        "import json, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, blob):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = json.loads(blob)['revision']))\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


def test_annotated_and_copied_parsed_mappings_and_annotated_parsers(tmp_path):
    """`kwargs: dict = json.loads(...)`, `dict(json.loads(...))`, `parser: ConfigParser = ...`."""
    findings = _scan(
        tmp_path,
        "import configparser, importlib, json\n"
        "from transformers import AutoModel\n"
        "def a(name, blob):\n"
        "    kwargs: dict = json.loads(blob)\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n"
        "def b(name, blob):\n"
        "    kwargs = dict(json.loads(blob))\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n"
        "def c(blob):\n"
        "    parser: configparser.ConfigParser = configparser.ConfigParser()\n"
        "    parser.read_dict(json.loads(blob))\n"
        "    return importlib.import_module(parser.get('plugin', 'module'))\n",
    )
    assert (
        sum(
            f["sink"] == "trust_remote_code (untrusted **kwargs)"
            for f in findings
            if f["tier"] == "A"
        )
        == 2
    )
    assert "importlib.import_module" in _sinks(findings)


def test_session_and_generic_request_bodies_are_sources(tmp_path):
    """`requests.request(...)` and `session.get(...)` on a `requests.Session()`."""
    findings = _scan(
        tmp_path,
        "import importlib, requests\n"
        "def a(url):\n"
        "    return importlib.import_module(requests.request('GET', url).text)\n"
        "def b(url):\n"
        "    with requests.Session() as session:\n"
        "        return importlib.import_module(session.get(url).text)\n",
    )
    assert sum(f["sink"] == "importlib.import_module" for f in findings if f["tier"] == "A") == 2


def test_runpy_joblib_and_native_loaders_are_sinks(tmp_path):
    """`runpy.run_path`, `joblib.load` and `ctypes.CDLL` on a download all execute it."""
    findings = _scan(
        tmp_path,
        "import ctypes, joblib, runpy\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(repo):\n"
        "    runpy.run_path(hf_hub_download(repo, 'plugin.py', revision = 'abababababababababababababababababababab'))\n"
        "def b(repo):\n"
        "    return joblib.load(hf_hub_download(repo, 'model.joblib', revision = 'abababababababababababababababababababab'))\n"
        "def c(repo):\n"
        "    return ctypes.CDLL(hf_hub_download(repo, 'plugin.so', revision = 'abababababababababababababababababababab'))\n",
    )
    assert {"runpy.run_path", "joblib.load", "ctypes.CDLL"} <= _sinks(findings)


def test_more_http_verbs_and_httpx_helpers_are_sources(tmp_path):
    """`requests.put` and `httpx.get` return the same untrusted body."""
    findings = _scan(
        tmp_path,
        "import httpx, importlib, requests\n"
        "def a(url):\n"
        "    return importlib.import_module(requests.put(url).text)\n"
        "def b(url):\n"
        "    return importlib.import_module(httpx.get(url).text)\n",
    )
    assert sum(f["sink"] == "importlib.import_module" for f in findings if f["tier"] == "A") == 2


def test_a_remote_code_flag_read_from_parsed_data_is_reported(tmp_path):
    """`trust_remote_code = json.loads(blob)["trust_remote_code"]`, direct or via a local."""
    findings = _scan(
        tmp_path,
        "import json\n"
        "from transformers import AutoModel\n"
        "def a(name, blob):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = json.loads(blob)['trust_remote_code'])\n"
        "def b(name, blob):\n"
        "    flag = json.loads(blob).get('trust_remote_code')\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = flag)\n",
    )
    quiet = _scan(
        tmp_path,
        "from transformers import AutoModel\n"
        "def a(name, trust_remote_code = False):\n"
        "    return AutoModel.from_pretrained(name, trust_remote_code = trust_remote_code)\n",
        name = "quiet.py",
    )
    assert (
        sum(
            f["sink"] == "trust_remote_code (untrusted value)" for f in findings if f["tier"] == "A"
        )
        == 2
    )
    assert "trust_remote_code (untrusted value)" not in _sinks(quiet)


def test_a_revision_local_read_from_data_is_not_a_pin(tmp_path):
    """`revision = json.loads(blob)["revision"]` then `revision = revision`."""
    findings = _scan(
        tmp_path,
        "import json, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "def load(repo, blob):\n"
        "    revision = json.loads(blob)['revision']\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = revision))\n",
    )
    assert "unpinned code fetch" in _sinks(findings)


def test_an_unparsable_file_fails_closed_and_a_declared_encoding_is_honoured(tmp_path):
    """A parse failure is an incomplete result, and a PEP 263 Latin-1 file still parses."""
    broken = _scan(tmp_path, "def broken(:\n    pass\n", name = "broken.py")
    assert any(f["sink"] == L.INCOMPLETE_SINK for f in broken)
    latin = tmp_path / "latin.py"
    latin.write_bytes(
        "# -*- coding: latin-1 -*-\n"
        "import importlib, json\n"
        "def café(blob):\n"
        "    return importlib.import_module(json.loads(blob)['module'])\n".encode("latin-1")
    )
    findings = L.scan([latin], roots = [tmp_path])
    assert "importlib.import_module" in _sinks(findings)


def test_file_loaders_read_pickle_and_base64_payloads(tmp_path):
    """`SourceFileLoader`, `pandas.read_pickle` and a base64-wrapped network pickle."""
    findings = _scan(
        tmp_path,
        "import base64, pandas, pickle, requests\n"
        "from importlib.machinery import SourceFileLoader\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(repo):\n"
        "    return SourceFileLoader('p', hf_hub_download(repo, 'p.py', revision = 'abababababababababababababababababababab')).load_module()\n"
        "def b(repo):\n"
        "    return pandas.read_pickle(hf_hub_download(repo, 'x.pkl', revision = 'abababababababababababababababababababab'))\n"
        "def c(url):\n"
        "    return pickle.loads(base64.b64decode(requests.get(url).content))\n",
    )
    assert {"importlib.machinery.SourceFileLoader", "pandas.read_pickle", "pickle.loads"} <= _sinks(
        findings
    )


def test_namespace_calls_computed_revisions_and_written_files(tmp_path):
    """`globals()[name]()`, `revision = "ma" + "in"`, and `write_bytes` then `run_path`."""
    findings = _scan(
        tmp_path,
        "import json, requests, runpy, sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        "def a(blob):\n"
        "    return globals()[json.loads(blob)['function']]()\n"
        "def b(repo):\n"
        "    sys.path.insert(0, snapshot_download(repo, revision = 'ma' + 'in'))\n"
        "def c(url):\n"
        "    plugin = Path('plugin.py')\n"
        "    plugin.write_bytes(requests.get(url).content)\n"
        "    runpy.run_path(plugin)\n",
    )
    assert {"getattr(module, ...)", "unpinned code fetch", "runpy.run_path"} <= _sinks(findings)


def test_streamed_bodies_handles_archives_and_numpy_pickle(tmp_path):
    """Streaming iterators, `raw_decode`, handle writes, extraction and `allow_pickle`."""
    findings = _scan(
        tmp_path,
        "import json, pickle, requests, runpy, sys, zipfile, numpy as np\n"
        "from importlib import import_module\n"
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(url):\n"
        "    return pickle.loads(b''.join(requests.get(url).iter_content()))\n"
        "def b(url):\n"
        "    cfg, _ = json.JSONDecoder().raw_decode(requests.get(url).text)\n"
        "    return import_module(cfg['module'])\n"
        "def c(url):\n"
        "    plugin = Path('plugin.py')\n"
        "    with plugin.open('wb') as out:\n"
        "        out.write(requests.get(url).content)\n"
        "    runpy.run_path(plugin)\n"
        "def d(repo):\n"
        "    target = 'plugins'\n"
        "    zipfile.ZipFile(hf_hub_download(repo, 'p.zip')).extractall(target)\n"
        "    sys.path.insert(0, target)\n"
        "def e(repo):\n"
        "    return np.load(hf_hub_download(repo, 'x.npy'), allow_pickle = True)\n"
        "def f(repo):\n"
        "    return np.load(hf_hub_download(repo, 'x.npy'))\n",
    )
    sinks = {(f["qualname"], f["sink"]) for f in findings}
    assert {
        ("a", "pickle.loads"),
        ("b", "importlib.import_module"),
        ("c", "runpy.run_path"),
        ("d", "sys.path.insert"),
        ("e", "numpy.load(allow_pickle = True)"),
    } <= sinks
    assert not [s for q, s in sinks if q == "f" and s.startswith("numpy")]


def test_keyword_exec_paths_scheduled_callbacks_and_true_attributes(tmp_path):
    """`os.execvp(file = ...)`, `Thread(target = ..., args = ...)`, `self.flag = True`."""
    findings = _scan(
        tmp_path,
        "import asyncio, json, os, subprocess, threading\n"
        "from transformers import AutoModel\n"
        "def run(command):\n"
        "    subprocess.run(command, shell = True)\n"
        "def a(blob):\n"
        "    os.execvp(file = json.loads(blob)['program'], args = ['x'])\n"
        "def b(blob):\n"
        "    threading.Thread(target = run, args = (json.loads(blob)['command'],)).start()\n"
        "async def c(blob):\n"
        "    await asyncio.to_thread(run, json.loads(blob)['command'])\n"
        "class Loader:\n"
        "    def __init__(self):\n"
        "        self.allow_remote = True\n"
        "    def load(self, name):\n"
        "        return AutoModel.from_pretrained(name, trust_remote_code = self.allow_remote)\n",
    )
    sinks = {(f["qualname"], f["sink"]) for f in findings}
    assert ("a", "os.execvp") in sinks
    assert ("run", "subprocess.run") in sinks
    assert any(q == "Loader.load" and s.startswith("trust_remote_code") for q, s in sinks)


def test_copied_kwargs_terminate_and_nested_bindings_stay_local(tmp_path):
    """`kwargs = dict(kwargs)` must not recurse, and an inner function's locals are its own."""
    findings = _scan(
        tmp_path,
        "import json\n"
        "from transformers import AutoModel\n"
        "def a(name, kwargs):\n"
        "    kwargs = dict(kwargs)\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n"
        "def b(name, kwargs, blob):\n"
        "    def inner():\n"
        "        kwargs = json.loads(blob)\n"
        "        return kwargs\n"
        "    return AutoModel.from_pretrained(name, **kwargs)\n",
    )
    assert not [f for f in findings if f["sink"] == "trust_remote_code (untrusted **kwargs)"]


def test_streams_buffers_decompression_and_file_transfers(tmp_path):
    """`httpx.stream`, `bytes(...)`, `gzip`, `copyfileobj` and `os.replace` keep the taint."""
    findings = _scan(
        tmp_path,
        "import gzip, httpx, os, pickle, requests, runpy, shutil\n"
        "from urllib.request import urlopen\n"
        "from huggingface_hub import hf_hub_download\n"
        "def a(url):\n"
        "    with httpx.stream('GET', url) as response:\n"
        "        return pickle.loads(b''.join(response.iter_bytes()))\n"
        "def b(url):\n"
        "    return pickle.loads(bytes(requests.get(url).content))\n"
        "def c(url):\n"
        "    return pickle.load(gzip.GzipFile(fileobj = urlopen(url)))\n"
        "def d(url):\n"
        "    plugin = 'plugin.py'\n"
        "    with open(plugin, 'wb') as out:\n"
        "        shutil.copyfileobj(urlopen(url), out)\n"
        "    runpy.run_path(plugin)\n"
        "def e(repo):\n"
        "    final = 'plugin.py'\n"
        "    os.replace(hf_hub_download(repo, 'plugin.py'), final)\n"
        "    runpy.run_path(final)\n",
    )
    sinks = {(f["qualname"], f["sink"]) for f in findings}
    assert {
        ("a", "pickle.loads"),
        ("b", "pickle.loads"),
        ("c", "pickle.load"),
        ("d", "runpy.run_path"),
        ("e", "runpy.run_path"),
    } <= sinks


def test_import_packages_class_namespaces_and_sink_callbacks(tmp_path):
    """`import_module(package = ...)`, `getattr(LocalClass, ...)`, `Thread(target = sink)`."""
    findings = _scan(
        tmp_path,
        "import httpx, json, subprocess, threading\n"
        "from importlib import import_module\n"
        "class Commands:\n"
        "    def run(self):\n"
        "        return 1\n"
        "def a(blob):\n"
        "    return import_module('.plugin', package = json.loads(blob)['package'])\n"
        "def b(blob):\n"
        "    return getattr(Commands, json.loads(blob)['action'])()\n"
        "def c(blob):\n"
        "    threading.Thread(target = subprocess.run, args = (json.loads(blob)['command'],)).start()\n"
        "def d(client):\n"
        "    def inner():\n"
        "        client = httpx.Client()\n"
        "        return client\n"
        "    return import_module(client.get('module'))\n",
    )
    sinks = {(f["qualname"], f["sink"]) for f in findings}
    assert {
        ("a", "importlib.import_module"),
        ("b", "getattr(module, ...)"),
        ("c", "subprocess.run"),
    } <= sinks
    assert not [f for f in findings if f["qualname"] == "d" and f["tier"] == "A"]
