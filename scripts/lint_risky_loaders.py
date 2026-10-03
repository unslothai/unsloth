#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail CI on a new call site that loads code or objects in a way a model repo can steer.

`lint_exec_literals.py` covers `exec`, `eval` and `compile`. These are the other
shapes behind holes fixed here since July 2026, each as one rule:

  dynamic-import       import_module / __import__ / import_from_string /
                       get_class_from_dynamic_module / spec_from_file_location /
                       runpy / pydoc.locate handed a name that is not written out
                       (sentence-transformers modules.json, mlx `model_type`).
  trust-remote-code    `trust_remote_code` switched on in code, rather than by the
                       caller: a default, an assignment, a dict entry or a literal
                       keyword (the Studio defaults that ran Hub code unasked).
  unpinned-code-fetch  Python fetched from a branch head (`.../main/x.py`), a
                       `git clone` with no checkout of a fixed revision, or a Hub
                       download with no `revision` in a function that then puts
                       the result on `sys.path` or loads a file from it (Spark-TTS).
  unsafe-deserialize   pickle / dill / joblib / marshal / shelve, `torch.load` with
                       `weights_only` not left at the safe default, `np.load` with
                       `allow_pickle=True`, `yaml.load` without a safe loader.

Every rule is one AST pass and a few set lookups, so the whole tree takes seconds.
Existing call sites are recorded in a baseline beside this script with a reason
each, keyed on the call's own text rather than its line, with a count, exactly as
in `lint_exec_literals.py`: only a new site fails.

    python scripts/lint_risky_loaders.py             # check, exit 1 on a new site
    python scripts/lint_risky_loaders.py --update    # rewrite the baseline
    python scripts/lint_risky_loaders.py --self-test # prove the rules still fire
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "risky_loaders_baseline.json"

EXCLUDED_PARTS = frozenset(
    {
        "tests",
        "node_modules",
        "build",
        "dist",
        ".venv",
        "venv",
        "site-packages",
        ".git",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        ".ipynb_checkpoints",
        ".eggs",
        "unsloth_compiled_cache",
    }
)

DYNAMIC_IMPORTS = {
    "importlib.import_module": "import_module",
    "importlib.__import__": "__import__",
    "importlib.util.spec_from_file_location": "spec_from_file_location",
    "importlib.machinery.SourceFileLoader": "SourceFileLoader",
    "runpy.run_path": "runpy.run_path",
    "runpy.run_module": "runpy.run_module",
    "pydoc.locate": "pydoc.locate",
    "pkgutil.resolve_name": "pkgutil.resolve_name",
}
# Matched on the last name alone: distinctive enough, and reached through several
# import paths (`transformers.dynamic_module_utils`, `sentence_transformers.util`).
DYNAMIC_IMPORT_TAILS = {
    "import_from_string": "import_from_string",
    "get_class_from_dynamic_module": "get_class_from_dynamic_module",
    "get_class_in_module": "get_class_in_module",
}

PICKLE_LIKE = {
    f"{module}.{function}"
    for module in ("pickle", "_pickle", "cPickle", "dill", "cloudpickle")
    for function in ("load", "loads", "Unpickler")
} | {
    "joblib.load",
    "marshal.load",
    "marshal.loads",
    "shelve.open",
    "pandas.read_pickle",
    "yaml.unsafe_load",
    "yaml.unsafe_load_all",
}
SAFE_YAML_LOADERS = {"SafeLoader", "CSafeLoader", "BaseLoader", "CBaseLoader"}

HUB_DOWNLOADS = {"snapshot_download", "hf_hub_download"}
CODE_LOADERS = {
    "sys.path.insert",
    "sys.path.append",
    "importlib.util.spec_from_file_location",
    "importlib.machinery.SourceFileLoader",
    "runpy.run_path",
}
# Python source on a moving ref: a branch head rather than a tag or commit.
BRANCH_HEAD_PY = re.compile(
    r"(?:raw\.githubusercontent\.com/[^/\s]+/[^/\s]+/(?:refs/heads/)?(?:main|master)/"
    r"|github\.com/[^/\s]+/[^/\s]+/raw/(?:refs/heads/)?(?:main|master)/)\S*\.py\b"
)
PIN_WORDS = {"checkout", "--revision", "reset"}


def _imports(tree: ast.AST) -> dict:
    """Local name -> dotted origin, so aliases (`import importlib as il`) resolve."""
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                table[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    table[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    table[head] = head
    return table


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _qualified(func: ast.AST, table: dict) -> str:
    name = _dotted(func)
    if not name:
        return ""
    head, _, rest = name.partition(".")
    if head == "__import__":
        return "importlib.__import__"
    if head in table:
        return table[head] + ("." + rest if rest else "")
    return name


def _is_written_out(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, bytes))
    if isinstance(node, ast.JoinedStr):
        return not any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _is_written_out(node.left) and _is_written_out(node.right)
    return False


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _keyword(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _first_argument(call: ast.Call, keyword: str):
    if call.args:
        return call.args[0]
    return _keyword(call, keyword)


# The keyword that names the module or file, where it is not `name`.
_TARGET_KEYWORDS = {
    "runpy.run_path": "path_name",
    "runpy.run_module": "mod_name",
    "pydoc.locate": "path",
}


def _dynamic_import(call: ast.Call, qualified: str):
    sink = DYNAMIC_IMPORTS.get(qualified) or DYNAMIC_IMPORT_TAILS.get(qualified.split(".")[-1])
    if sink is None:
        return None
    if sink in ("spec_from_file_location", "SourceFileLoader"):
        # The module name is a label; the path is what gets executed.
        keyword = "location" if sink == "spec_from_file_location" else "path"
        target = call.args[1] if len(call.args) > 1 else _keyword(call, keyword)
    else:
        target = _first_argument(call, _TARGET_KEYWORDS.get(sink, "name"))
    if target is None or _is_written_out(target):
        return None
    return sink


def _unsafe_deserialize(call: ast.Call, qualified: str):
    if qualified in PICKLE_LIKE:
        return qualified
    if qualified == "torch.load":
        # Omitted is unsafe too: torch < 2.6, still supported, defaults to weights_only=False.
        if not _is_true(_keyword(call, "weights_only")):
            return "torch.load"
    elif qualified == "numpy.load":
        if _is_true(_keyword(call, "allow_pickle")):
            return "numpy.load"
    elif qualified in ("yaml.load", "yaml.load_all"):
        loader = _keyword(call, "Loader")
        if loader is None and len(call.args) > 1:
            loader = call.args[1]
        if loader is None or _dotted(loader).split(".")[-1] not in SAFE_YAML_LOADERS:
            return qualified
    return None


def _trust_remote_code(node: ast.AST):
    """`trust_remote_code` turned on by the code itself."""
    if isinstance(node, ast.Call):
        if _is_true(_keyword(node, "trust_remote_code")):
            return "keyword"
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "setdefault"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "trust_remote_code"
            and _is_true(node.args[1])
        ):
            return "setdefault"
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = node.args
        positional = arguments.posonlyargs + arguments.args
        pairs = list(
            zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults)
        )
        pairs += [
            (a, d) for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults) if d is not None
        ]
        if any(a.arg == "trust_remote_code" and _is_true(d) for a, d in pairs):
            return "default"
    elif (
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and _is_true(node.value)
    ):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, (ast.Name, ast.Attribute)) and (
                getattr(target, "id", None) == "trust_remote_code"
                or getattr(target, "attr", None) == "trust_remote_code"
            ):
                return "assignment"
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "trust_remote_code"
            ):
                return "item"
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "trust_remote_code"
                and _is_true(value)
            ):
                return "dict"
    return None


def _git_words(command) -> list:
    """The written-out words of a git command, from an argv list or a shell string; [] otherwise."""
    if isinstance(command, (ast.List, ast.Tuple)):
        first = command.elts[0] if command.elts else None
        if not (isinstance(first, ast.Constant) and first.value == "git"):
            return []
        words = [
            e.value
            for e in command.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    else:
        if isinstance(command, ast.JoinedStr) and command.values:
            command = command.values[0]
        if (
            not (isinstance(command, ast.Constant) and isinstance(command.value, str))
            or "git " not in command.value
        ):
            return []
        # The first git command in a chain, so `cd llama.cpp && git reset --hard X` counts.
        segments = (part.split() for part in re.split(r"&&|;|\|\|", command.value))
        words = next((w for w in segments if w[:1] == ["git"]), [])
    return words if words[:1] == ["git"] else []


def _unpinned_fetches(calls: list, git_commands: list) -> list:
    """Per function: a clone with no pin, or a revision-less Hub download feeding a code loader."""
    found = []
    loads_code = any(qualified in CODE_LOADERS for _, qualified in calls)
    for call, qualified in calls:
        if (
            qualified.split(".")[-1] in HUB_DOWNLOADS
            and loads_code
            and _keyword(call, "revision") is None
        ):
            found.append((call, "hub-download-loaded-as-code"))
    # Written-out git commands, passed directly or held in a variable first. A pin is a git
    # command that moves to a revision, not any string that mentions one.
    if not any(set(words) & PIN_WORDS for _, words in git_commands):
        found += [
            (node, "git-clone-unpinned") for node, words in git_commands if words[1:2] == ["clone"]
        ]
    return found


def _key(relative: str, rule: str, sink: str, node: ast.AST) -> dict:
    return {
        "file": relative,
        "rule": rule,
        "sink": sink,
        "digest": hashlib.sha256(ast.unparse(node).encode()).hexdigest()[:16],
        "line": getattr(node, "lineno", 0),
    }


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_TRUST_SHAPES = (ast.Call, ast.Lambda, ast.Assign, ast.AnnAssign, ast.Dict) + _FUNCTIONS


def scan_file(path: Path, relative: str) -> list:
    try:
        tree = ast.parse(path.read_bytes(), filename = str(path))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        # Unparsed is unchecked, and reporting it clean is the bypass this gate exists to avoid.
        raise SystemExit(f"{relative}: could not be parsed ({error.__class__.__name__})")
    found = []
    imports = []
    calls_by_owner: dict = {}
    git_by_owner: dict = {}
    covered = set()
    stack = [(tree, tree)]
    while stack:
        node, owner = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _TRUST_SHAPES):
                shape = _trust_remote_code(child)
                if shape:
                    # A function default is keyed on the signature, not the whole body.
                    subject = child.args if isinstance(child, _FUNCTIONS) else child
                    entry = _key(relative, "trust-remote-code", shape, subject)
                    entry["line"] = child.lineno
                    found.append(entry)
                if isinstance(child, ast.Call):
                    calls_by_owner.setdefault(owner, []).append(child)
                elif isinstance(child, _FUNCTIONS):
                    stack.append((child, child))
                    continue
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                imports.append(child)
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                if "/ma" in child.value and BRANCH_HEAD_PY.search(child.value):
                    found.append(_key(relative, "unpinned-code-fetch", "branch-head-url", child))
            if (
                isinstance(child, (ast.List, ast.Tuple, ast.JoinedStr, ast.Constant))
                and id(child) not in covered
            ):
                words = _git_words(child)
                if words:
                    git_by_owner.setdefault(owner, []).append((child, words))
                    if isinstance(child, ast.JoinedStr):
                        covered.add(id(child.values[0]))
            stack.append((child, owner))

    table = _imports(ast.Module(body = imports, type_ignores = []))
    for owner in calls_by_owner.keys() | git_by_owner.keys():
        calls = [(call, _qualified(call.func, table)) for call in calls_by_owner.get(owner, ())]
        for call, qualified in calls:
            sink = _dynamic_import(call, qualified)
            if sink:
                found.append(_key(relative, "dynamic-import", sink, call))
            sink = _unsafe_deserialize(call, qualified)
            if sink:
                found.append(_key(relative, "unsafe-deserialize", sink, call))
        for call, sink in _unpinned_fetches(calls, git_by_owner.get(owner, [])):
            found.append(_key(relative, "unpinned-code-fetch", sink, call))
    return found


def collect(targets: list) -> list:
    found = []
    for target in targets:
        root = REPO_ROOT / target
        if root.is_file() and root.suffix == ".py":
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.rglob("*.py"))
        else:
            raise SystemExit(
                f"{target}: scan target does not exist, so nothing under it was checked"
            )
        for path in paths:
            # This file's self-test sources spell out every bad shape on purpose.
            if not path.is_file() or path.resolve() == Path(__file__).resolve():
                continue
            relative = _relative(path)
            if EXCLUDED_PARTS & set(Path(relative).parts):
                continue
            found.extend(scan_file(path, relative))
    return found


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _identity(entry: dict) -> tuple:
    return (entry["file"], entry["rule"], entry["sink"], entry["digest"])


def _counted(entries: list) -> dict:
    counts: dict = {}
    for entry in entries:
        counts[_identity(entry)] = counts.get(_identity(entry), 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--update", action = "store_true", help = "rewrite the baseline")
    parser.add_argument("--self-test", action = "store_true", help = "check the rules still fire")
    parser.add_argument("--paths", nargs = "*", help = "scan these instead of the defaults")
    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()
    if arguments.update and arguments.paths:
        # --update rewrites the whole baseline, so a partial scan would drop every other entry.
        parser.error("--update rewrites the whole baseline; run it without --paths")

    document = json.loads(BASELINE_PATH.read_text(encoding = "utf-8"))
    found = collect(arguments.paths or document["targets"])
    baseline_rel = BASELINE_PATH.relative_to(REPO_ROOT).as_posix()

    if arguments.update:
        # Carry reasons over, so a reviewed entry never silently becomes an unreviewed one.
        reasons = {_identity(e): e.get("reason", "") for e in document["entries"]}
        document["entries"] = sorted(
            (
                {
                    "file": f,
                    "rule": r,
                    "sink": s,
                    "digest": d,
                    "count": n,
                    "reason": reasons.get((f, r, s, d), "REVIEW ME"),
                }
                for (f, r, s, d), n in _counted(found).items()
            ),
            key = _identity,
        )
        BASELINE_PATH.write_text(json.dumps(document, indent = 2) + "\n", encoding = "utf-8")
        print(f"baseline: {len(document['entries'])} entries, {len(found)} call sites")
        return 0

    entries = document["entries"]
    if arguments.paths:
        # A scoped run judges only the entries under the paths it scanned.
        scopes = [_relative(REPO_ROOT / path) for path in arguments.paths]
        if "." not in scopes:
            entries = [
                e
                for e in entries
                if any(e["file"] == p or e["file"].startswith(p.rstrip("/") + "/") for p in scopes)
            ]
    allowed = {_identity(e): e["count"] for e in entries}
    observed = _counted(found)
    lines = {_identity(e): e["line"] for e in found}

    new = sorted(k for k, n in observed.items() if n > allowed.get(k, 0))
    if new:
        print(f"{len(new)} risky loader call site(s) not in the baseline:\n")
        for key in new:
            f, r, s, _ = key
            print(f"  {f}:{lines[key]}  [{r}] {s}")
        print(
            "\nEither load from a written-out name, a pinned revision or a safe format, or - if "
            "the input really is trusted - record it:\n"
            "  1. python scripts/lint_risky_loaders.py --update\n"
            f'  2. replace the new entr(y/ies)\' "REVIEW ME" reason in {baseline_rel} with why it is safe.'
        )
        return 1

    unreviewed = sorted(
        (e["file"], e["rule"], e["sink"])
        for e in entries
        if e.get("reason", "REVIEW ME") in ("", "REVIEW ME")
    )
    if unreviewed:
        print(f"{len(unreviewed)} baseline entr(y/ies) carry no justification:\n")
        for f, r, s in unreviewed:
            print(f"  {f}  [{r}] {s}")
        print(f"\nSay why the input is trusted in the entry's `reason` field in {baseline_rel}.")
        return 1

    # Fewer calls than allowed counts too, so a removed duplicate cannot make room for a new one.
    stale = sorted(k for k in allowed if observed.get(k, 0) < allowed[k])
    if stale:
        # An entry outliving its call site would re-permit whatever lands on that digest next.
        print(f"{len(stale)} baseline entr(y/ies) match fewer calls than recorded:\n")
        for f, r, s, d in stale:
            print(f"  {f}  [{r}] {s}  {d}")
        print(
            "\nA risky call was removed or rewritten; run `python scripts/lint_risky_loaders.py --update`."
        )
        return 1

    print(f"ok: {len(found)} risky loader call site(s), all recorded")
    return 0


_BAD = {
    "dynamic-import": """
import importlib
from importlib import import_module as im
from sentence_transformers.util import import_from_string
from transformers.dynamic_module_utils import get_class_from_dynamic_module
def f(config, name, path):
    importlib.import_module(f"mlx_lm.models.{config.model_type}")
    im(name)
    __import__(name)
    import_from_string(config["type"])
    get_class_from_dynamic_module(config.auto_map["AutoModel"], name)
    importlib.util.spec_from_file_location("x", path)
    importlib.machinery.SourceFileLoader("x", path)
""",
    "trust-remote-code": """
def load(name, trust_remote_code = True):
    AutoConfig.from_pretrained(name, trust_remote_code = True)
    trust_remote_code = True
    kwargs["trust_remote_code"] = True
    kwargs.setdefault("trust_remote_code", True)
    return {"trust_remote_code": True}
""",
    "unpinned-code-fetch": """
import sys, subprocess
from huggingface_hub import snapshot_download
URL = "https://github.com/ggml-org/llama.cpp/raw/refs/heads/master/convert_hf_to_gguf.py"
def codec():
    path = snapshot_download("org/codec")
    sys.path.insert(0, path)
def clone():
    subprocess.run(["git", "clone", "https://github.com/org/repo"])
def shell_clone(folder):
    run(f"git clone https://github.com/org/repo {folder}")
def held_clone():
    commands = ["git clone --recursive https://github.com/org/repo", "pip install x"]
    try_execute(commands)
""",
    "unsafe-deserialize": """
import pickle, torch, yaml, joblib
import numpy as np
def f(path, flag):
    pickle.loads(open(path, "rb").read())
    torch.load(path, weights_only = False)
    torch.load(path, weights_only = flag)
    torch.load(path)
    np.load(path, allow_pickle = True)
    yaml.load(open(path))
    joblib.load(path)
""",
}
_BAD_COUNTS = {
    "dynamic-import": 7,
    "trust-remote-code": 6,
    "unpinned-code-fetch": 5,
    "unsafe-deserialize": 7,
}

_GOOD = """
import importlib, subprocess, sys, torch, yaml
import numpy as np
from huggingface_hub import snapshot_download
URL = "https://github.com/ggml-org/llama.cpp/raw/b1234/convert_hf_to_gguf.py"
def f(name, path, trust_remote_code = False):
    importlib.import_module("torch.nn")
    importlib.util.find_spec(name)
    __import__("os")
    AutoConfig.from_pretrained(name, trust_remote_code = trust_remote_code)
    torch.load(path, weights_only = True)
    np.load(path)
    yaml.load(open(path), Loader = yaml.SafeLoader)
    yaml.safe_load(open(path))
    snapshot_download(name)
def chained_pin(version):
    try_execute(["git clone https://github.com/org/repo", f"cd repo && git reset --hard {version}"])
def pinned():
    subprocess.run(["git", "clone", "https://github.com/org/repo"])
    subprocess.run(["git", "checkout", "0123abc"])
def codec():
    path = snapshot_download("org/codec", revision = "0123abc")
    sys.path.insert(0, path)
"""


def self_test() -> int:
    """Each rule fires on its bad shapes and stays quiet on the good ones, on a runner with nothing but Python."""
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory() as directory:
        cases = [(rule, source, _BAD_COUNTS[rule], rule) for rule, source in _BAD.items()]
        cases.append(("good", _GOOD, 0, None))
        for label, source, expected, rule in cases:
            path = Path(directory) / f"{label.replace('-', '_')}.py"
            path.write_text(source, encoding = "utf-8")
            hits = scan_file(path, path.name)
            count = len([h for h in hits if rule is None or h["rule"] == rule])
            if count != expected or (rule is not None and len(hits) != expected):
                failures.append(
                    f"{label}: expected {expected} finding(s), got {[(h['rule'], h['sink']) for h in hits]}"
                )
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
