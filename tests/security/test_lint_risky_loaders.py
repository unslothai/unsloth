# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`scripts/lint_risky_loaders.py` fails CI on a new risky loader call site.

These pin which shapes each rule reports, which it leaves alone, and that the baseline
cannot be used to smuggle a new call site past it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lint_risky_loaders.py"
BASELINE = SCRIPT.parent / "risky_loaders_baseline.json"


def _module():
    spec = importlib.util.spec_from_file_location("lint_risky_loaders_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rules(tmp_path, source):
    sample = tmp_path / "sample.py"
    sample.write_text(source, encoding = "utf-8")
    return [(f["rule"], f["sink"]) for f in _module().scan_file(sample, "sample.py")]


REPORTED = {
    "config model_type into import_module": (
        "import importlib\ndef f(c):\n    importlib.import_module(f'mlx_lm.models.{c.model_type}')\n",
        ("dynamic-import", "import_module"),
    ),
    "aliased import_module": (
        "from importlib import import_module as im\ndef f(n):\n    im(n)\n",
        ("dynamic-import", "import_module"),
    ),
    "modules.json type into import_from_string": (
        "from sentence_transformers.util import import_from_string\n"
        "def f(cfg):\n    import_from_string(cfg['type'])\n",
        ("dynamic-import", "import_from_string"),
    ),
    "auto_map into get_class_from_dynamic_module": (
        "from transformers import dynamic_module_utils as d\n"
        "def f(c, n):\n    d.get_class_from_dynamic_module(c.auto_map['AutoModel'], n)\n",
        ("dynamic-import", "get_class_from_dynamic_module"),
    ),
    "a computed path into spec_from_file_location": (
        "import importlib.util\ndef f(p):\n    importlib.util.spec_from_file_location('x', p)\n",
        ("dynamic-import", "spec_from_file_location"),
    ),
    "a computed path into SourceFileLoader": (
        "import importlib.machinery\ndef f(p):\n    importlib.machinery.SourceFileLoader('x', p)\n",
        ("dynamic-import", "SourceFileLoader"),
    ),
    "trust_remote_code default": (
        "def load(name, trust_remote_code = True):\n    pass\n",
        ("trust-remote-code", "default"),
    ),
    "trust_remote_code keyword": (
        "def f(n):\n    AutoModel.from_pretrained(n, trust_remote_code = True)\n",
        ("trust-remote-code", "keyword"),
    ),
    "trust_remote_code dict item": (
        "def f(kw):\n    kw['trust_remote_code'] = True\n",
        ("trust-remote-code", "item"),
    ),
    "Python from a branch head": (
        "URL = 'https://raw.githubusercontent.com/org/repo/main/tool.py'\n",
        ("unpinned-code-fetch", "branch-head-url"),
    ),
    "revision-less download put on sys.path": (
        "import sys\nfrom huggingface_hub import snapshot_download\n"
        "def f():\n    sys.path.insert(0, snapshot_download('org/codec'))\n",
        ("unpinned-code-fetch", "hub-download-loaded-as-code"),
    ),
    "git clone with no checkout": (
        "import subprocess\ndef f():\n    subprocess.run(['git', 'clone', 'https://github.com/o/r'])\n",
        ("unpinned-code-fetch", "git-clone-unpinned"),
    ),
    "a shell-string git clone": (
        "def f(folder):\n    run(f'git clone https://github.com/o/r {folder}')\n",
        ("unpinned-code-fetch", "git-clone-unpinned"),
    ),
    "a clone next to a message that only mentions checkout": (
        "import subprocess\ndef f():\n    subprocess.run(['git', 'clone', 'u'])\n"
        "    print('source checkout detected')\n",
        ("unpinned-code-fetch", "git-clone-unpinned"),
    ),
    "torch.load with weights_only omitted": (
        "import torch\ndef f(p):\n    torch.load(p)\n",
        ("unsafe-deserialize", "torch.load"),
    ),
    "torch.load weights_only=False": (
        "import torch\ndef f(p):\n    torch.load(p, weights_only = False)\n",
        ("unsafe-deserialize", "torch.load"),
    ),
    "pickle.loads": (
        "import pickle\ndef f(b):\n    pickle.loads(b)\n",
        ("unsafe-deserialize", "pickle.loads"),
    ),
    "yaml.load without a loader": (
        "import yaml\ndef f(s):\n    yaml.load(s)\n",
        ("unsafe-deserialize", "yaml.load"),
    ),
    "np.load allow_pickle": (
        "import numpy as np\ndef f(p):\n    np.load(p, allow_pickle = True)\n",
        ("unsafe-deserialize", "numpy.load"),
    ),
}

QUIET = {
    "a written-out import": "import importlib\nimportlib.import_module('torch.nn')\n",
    "find_spec, which does not execute": "import importlib.util\ndef f(n):\n    importlib.util.find_spec(n)\n",
    "a local function that shares the name": "def import_module(n):\n    return n\ndef f(n):\n    import_module(n)\n",
    "trust_remote_code forwarded from the caller": (
        "def f(n, trust_remote_code = False):\n"
        "    AutoModel.from_pretrained(n, trust_remote_code = trust_remote_code)\n"
    ),
    "a pinned URL": "URL = 'https://raw.githubusercontent.com/org/repo/v1.2/tool.py'\n",
    "a pinned download put on sys.path": (
        "import sys\nfrom huggingface_hub import snapshot_download\n"
        "def f():\n    sys.path.insert(0, snapshot_download('org/codec', revision = 'abc123'))\n"
    ),
    "a download that is not loaded as code": (
        "from huggingface_hub import snapshot_download\ndef f(n):\n    return snapshot_download(n)\n"
    ),
    "git clone followed by a checkout": (
        "import subprocess\ndef f():\n    subprocess.run(['git', 'clone', 'u'])\n"
        "    subprocess.run(['git', 'checkout', 'abc123'])\n"
    ),
    "torch.load with weights_only=True": "import torch\ndef f(p):\n    torch.load(p, weights_only = True)\n",
    "yaml with a safe loader": "import yaml\ndef f(s):\n    yaml.load(s, Loader = yaml.SafeLoader)\n    yaml.safe_load(s)\n",
}


@pytest.mark.parametrize("label", sorted(REPORTED))
def test_reported(tmp_path, label):
    source, expected = REPORTED[label]
    assert _rules(tmp_path, source) == [expected]


@pytest.mark.parametrize("label", sorted(QUIET))
def test_quiet(tmp_path, label):
    assert _rules(tmp_path, QUIET[label]) == []


def test_self_test_passes():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output = True, text = True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_second_identical_call_is_not_covered_by_one_baseline_entry(tmp_path, monkeypatch):
    module = _module()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import pickle\ndef f(b):\n    pickle.loads(b)\n    pickle.loads(b)\n", encoding = "utf-8"
    )
    found = module.scan_file(sample, "sample.py")
    entry = dict(found[0], count = 1, reason = "reviewed")
    entry.pop("line")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"targets": [str(sample)], "entries": [entry]}), encoding = "utf-8"
    )
    monkeypatch.setattr(module, "BASELINE_PATH", baseline)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["lint_risky_loaders.py"])
    assert module.main() == 1


def test_a_removed_duplicate_leaves_no_spare_allowance(tmp_path, monkeypatch):
    module = _module()
    sample = tmp_path / "sample.py"
    sample.write_text("import pickle\ndef f(b):\n    pickle.loads(b)\n", encoding = "utf-8")
    entry = dict(module.scan_file(sample, "sample.py")[0], count = 2, reason = "reviewed")
    entry.pop("line")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"targets": [str(sample)], "entries": [entry]}), encoding = "utf-8"
    )
    monkeypatch.setattr(module, "BASELINE_PATH", baseline)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["lint_risky_loaders.py"])
    assert module.main() == 1


def test_every_baseline_entry_is_reviewed():
    entries = json.loads(BASELINE.read_text(encoding = "utf-8"))["entries"]
    assert entries
    assert all(e.get("reason") and e["reason"] != "REVIEW ME" for e in entries)


def test_the_tree_matches_the_baseline():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output = True, text = True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lint_ci_runs_the_gate():
    workflow = (SCRIPT.parents[1] / ".github" / "workflows" / "lint-ci.yml").read_text(
        encoding = "utf-8"
    )
    assert "lint_risky_loaders.py --self-test" in workflow
    assert "python scripts/lint_risky_loaders.py\n" in workflow
