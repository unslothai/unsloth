#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail CI when a value read out of a downloaded artefact reaches an execution sink.

`lint_exec_literals.py` already covers `exec`/`eval`/`compile`, and it covers them
bluntly: the first argument must be a written-out string. That rule is the right one
for those three builtins and it is not the rule needed here, because the sinks this
script exists for are *supposed* to take a computed argument. `import_module` is
called with a variable on purpose. `sys.path.insert` is called with a computed path on
purpose. The question for them is not "is this a literal" but "whose value is it".

So this is a taint checker. It marks the return value of a read from an untrusted
artefact (a `config.json`, an `adapter_config.json`, a `modules.json`, an HTTP body, a
dataset) as tainted, propagates taint through assignments, containers, attributes and
across first-party function calls to a fixpoint, and reports when a tainted value lands
in the argument of a sink that executes it.

The sink list is the shape of three real holes:

    importlib.import_module(name)     a config field naming a module to import
    sys.path.insert(0, directory)     a directory derived from a download path, after
                                      which a plain `import` runs whatever is there
    from_pretrained(trust_remote_code = True)
                                      consent the user never gave

Taint flows *between* files: the read can be in `utils/models/model_config.py` and the
sink in `core/inference/worker.py`. Call targets are resolved against a module index
built from the repo, so a first-party callee has its parameters tainted by its callers
and the analysis continues inside it. Third-party packages are not expanded: they are
the trust boundary, and a call into one is where this script stops and reports.

The analysis is flow-insensitive and deliberately so. If a name is ever assigned a
tainted value anywhere in a function, it is tainted everywhere in that function. That
over-reports on code that overwrites a variable, it never under-reports on ordering,
and it is what makes the result independent of how the file is laid out: the same tree
gives the same findings on every machine and in every order.

Findings come in two tiers. Tier A is a chain that starts at a real untrusted read and
is what the gate fails on. Tier B is a sink whose argument traces back only to a
parameter whose *name* says it carries untrusted data (`model_type`, `model_path`,
`class_ref` and friends) with no first-party caller in the tree, which is the honest
state for a public entry point someone else calls. Tier B is reported and not gated,
because the name of a parameter is a guess and a gate should not rest on one.

Existing call sites live in a baseline beside this script so the gate starts green and
only new ones fail. An entry is keyed on the path, the enclosing qualname and a hash of
the call's own normalised source, so moving a function does not churn the baseline but
editing the call re-opens it. Each entry carries a count, so a new sink cannot hide
behind a removed one.

Stdlib only, so it runs in a lint job that installs nothing.

    python scripts/lint_untrusted_sinks.py              # check, exit 1 on a new finding
    python scripts/lint_untrusted_sinks.py --update     # rewrite the baseline
    python scripts/lint_untrusted_sinks.py --self-test  # prove the rules still fire
    python scripts/lint_untrusted_sinks.py --json out.json --tier b
    python scripts/lint_untrusted_sinks.py --paths a.py b.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "untrusted_sinks_baseline.json"

# Scanned by default. `studio/backend` is also an import root in its own right: its
# modules import each other as `utils.models.model_config`, not as a subpackage of
# anything, so it is added to the module index separately below.
# `unsloth_cli` and the two top-level entry points are here because they ship: the CI
# invocation uses these defaults, so a package left out of this tuple is a production
# execution surface the gate never looks at.
DEFAULT_TARGETS = (
    "unsloth",
    "unsloth_zoo",
    "unsloth_cli",
    "studio",
    "scripts",
    "cli.py",
    "unsloth-cli.py",
)

EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "unsloth_compiled_cache",
        "venv",
    }
)

# Tests are excluded: a security test's whole job is to keep the removed sink around and
# assert that it is refused, so scanning them reports the assertions as findings.
EXCLUDED_DIRS = frozenset({"tests", "test"})


# --------------------------------------------------------------------------------------
# What counts as untrusted
# --------------------------------------------------------------------------------------

# Calls whose return value is attacker-controlled. Matched on the trailing one or two
# dotted segments of the callee, so `json.load`, `js.load` after `import json as js` and
# a bare `load` imported from json all match. Deserialisers are here because every
# artefact a model repo ships arrives through one of them.
UNTRUSTED_CALLS = frozenset(
    {
        "json.load",
        "json.loads",
        "yaml.load",
        "yaml.safe_load",
        "yaml.full_load",
        "tomllib.load",
        "tomllib.loads",
        "toml.load",
        "configparser.read",
        # Downloads. The *return* of these is a path to attacker-written bytes, and the
        # basename of that path is attacker-chosen, which is the Spark-TTS shape.
        "hf_hub_download",
        "snapshot_download",
        "cached_file",
        "try_to_load_from_cache",
        "get_hf_file_metadata",
        # Network bodies.
        "requests.get",
        "requests.post",
        "urlopen",
        "urlretrieve",
        # Hub metadata and datasets.
        "load_dataset",
        "list_repo_files",
        "model_info",
        "repo_info",
        "dataset_info",
        "HfApi.model_info",
    }
)

# Attribute reads that turn a trusted handle into untrusted bytes.
UNTRUSTED_METHODS = frozenset({"read", "read_text", "readlines", "readline", "json"})

# Artefact names. A string literal containing one of these anywhere in a function is
# what makes a `json.load` in that function a *model repository* read rather than a read
# of first-party state, and it is recorded on the finding so a reviewer can see which
# file the value came out of.
UNTRUSTED_ARTEFACTS = (
    "adapter_config.json",
    "chat_template",
    "config.json",
    "generation_config.json",
    "modules.json",
    "preprocessor_config.json",
    "processor_config.json",
    "router_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "wordembedding_config.json",
)

# Tier B only. A parameter named one of these carries a value its caller read out of an
# artefact, which is true of every one of these names in this tree today, but the name
# is a convention and not a proof, so findings that rest on it are never gated.
NAMED_PARAM_REASON = "untrusted parameter name"

UNTRUSTED_PARAM_NAMES = frozenset(
    {
        "architecture",
        "architectures",
        "base_model",
        "base_model_name_or_path",
        "class_ref",
        "config",
        "dataset_name",
        "hf_repo",
        "model_class",
        "model_id",
        "model_location",
        "model_name",
        "model_name_or_path",
        "model_path",
        "model_repo_path",
        "model_type",
        "module_class",
        "pretrained_model_name_or_path",
        "repo_id",
        "repo_path",
        "tokenizer_name",
    }
)


# --------------------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------------------

# callee -> indices of positional arguments that execute, plus keyword names that do.
# `None` for the keyword set means "no keyword form".
SINKS: dict[str, tuple[tuple[int, ...], frozenset[str]]] = {
    # Dynamic import. The whole class of CWE-470 unsafe reflection.
    "importlib.import_module": ((0,), frozenset({"name"})),
    "import_module": ((0,), frozenset({"name"})),
    "__import__": ((0,), frozenset({"name"})),
    "importlib.util.spec_from_file_location": ((1,), frozenset({"location"})),
    "spec_from_file_location": ((1,), frozenset({"location"})),
    "load_source": ((1,), frozenset()),
    "pydoc.locate": ((0,), frozenset()),
    # Resolving a dotted path out of a string, in both libraries that offer it.
    "import_from_string": ((0,), frozenset()),
    "get_class_from_dynamic_module": (
        (0, 1),
        frozenset({"class_reference", "pretrained_model_name_or_path"}),
    ),
    "import_module_class": ((0,), frozenset()),
    # Import path injection: after this, a plain `import` statement is the sink.
    "sys.path.insert": ((1,), frozenset()),
    "sys.path.append": ((0,), frozenset()),
    "path.insert": ((1,), frozenset()),
    "path.append": ((0,), frozenset()),
    "site.addsitedir": ((0,), frozenset()),
    # The builtins the other linter covers. Kept so one report shows every sink, and
    # excluded from this gate's failure set to avoid two scripts failing on one line.
    "exec": ((0,), frozenset()),
    "eval": ((0,), frozenset()),
    "compile": ((0,), frozenset({"source"})),
    "builtins.exec": ((0,), frozenset()),
    "builtins.eval": ((0,), frozenset()),
    # Shell. A tainted argv[0] or a tainted command string is execution.
    "os.system": ((0,), frozenset()),
    "os.popen": ((0,), frozenset()),
    "subprocess.run": ((0,), frozenset({"args"})),
    "subprocess.call": ((0,), frozenset({"args"})),
    "subprocess.check_call": ((0,), frozenset({"args"})),
    "subprocess.check_output": ((0,), frozenset({"args"})),
    "subprocess.Popen": ((0,), frozenset({"args"})),
    # Deserialisers that construct arbitrary objects.
    "pickle.load": ((0,), frozenset()),
    "pickle.loads": ((0,), frozenset()),
    "dill.load": ((0,), frozenset()),
    "dill.loads": ((0,), frozenset()),
}

# Sinks already gated by lint_exec_literals.py / lint_dynamic_exec.py. Reported here for
# a single view of the surface, never the reason this script exits non-zero.
SINKS_GATED_ELSEWHERE = frozenset({"exec", "eval", "compile", "builtins.exec", "builtins.eval"})

# `getattr(x, tainted)` is only a finding when `x` is plausibly a module or a class
# namespace: `getattr(config, field)` is a dict-ish read and is everywhere.
MODULE_ISH_NAMES = frozenset(
    {
        "builtins",
        "diffusers",
        "importlib",
        "module",
        "mod",
        "models",
        "nn",
        "peft",
        "sentence_transformers",
        "st_models",
        "torch",
        "transformers",
        "trl",
    }
)

# Loaders that take a `trust_remote_code`. A True literal here, or a default of True on a
# first-party function that forwards into one, is consent the user did not give.
REMOTE_CODE_LOADERS = (
    "from_pretrained",
    "from_config",
    "pipeline",
    "load_dataset",
    "AutoConfig",
    "AutoModel",
    "AutoTokenizer",
    "AutoProcessor",
    "SentenceTransformer",
    "get_class_from_dynamic_module",
)


def _relative(path: Path) -> str:
    """Repo-relative where possible, forward slashes always.

    The baseline is committed. A Windows checkout that regenerated it with backslashes
    would rewrite every entry.
    """
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _call_name(node: ast.AST) -> str:
    """The dotted text of a call target, or "" when it is not a plain dotted name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        # `HfApi().model_info(...)`: keep the method so the trailing-segment match works.
        return ".".join(reversed(parts))
    return ""


def _matches_any(names, table) -> str | None:
    """First table hit across every spelling an alias can have. Ambiguity fails closed."""
    for name in names:
        hit = _matches(name, table)
        if hit is not None:
            return hit
    return None


def _matches(name: str, table) -> str | None:
    """Match a dotted callee against a table on its trailing segments.

    `json.load`, `js.load` and a bare `load` all have to hit the same entry, because
    which spelling appears is an import style and not a security property.
    """
    if not name:
        return None
    segments = name.split(".")
    for width in (3, 2, 1):
        if len(segments) >= width:
            candidate = ".".join(segments[-width:])
            if candidate in table:
                return candidate
    return None


def _norm_hash(node: ast.AST) -> str:
    """A hash of the call's own normalised source, for baseline identity."""
    try:
        text = ast.unparse(node)
    except Exception:
        text = ast.dump(node)
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def _short(node: ast.AST, limit: int = 160) -> str:
    try:
        text = " ".join(ast.unparse(node).split())
    except Exception:
        text = type(node).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _ModuleIndex:
    """Dotted module name -> file, over every import root in the repo.

    This is the "expand all imported modules" part. `studio/backend` is its own root
    because its modules import each other by top-level name, so without it every
    `from utils.models.model_config import ...` resolves to nothing and taint stops at
    the file boundary.
    """

    def __init__(self, roots: list[Path], files: list[Path]):
        self.modules: dict[str, Path] = {}
        self.files: dict[Path, str] = {}
        # Deepest root first, and no `break`: a file under `studio/backend` has to be
        # registered BOTH as `studio.backend.utils.x` and as `utils.x`, because the
        # backend's own modules import each other by the second form. Stopping at the
        # first matching root registered only the first, so `from utils.models... import`
        # resolved to nothing and taint never crossed a single backend file boundary,
        # which is the one thing this index exists to do.
        ordered = sorted(roots, key = lambda root: len(root.parts), reverse = True)
        for path in sorted(files):
            for root in ordered:
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                parts = list(relative.parts)
                if parts[-1] == "__init__.py":
                    parts = parts[:-1]
                else:
                    parts[-1] = parts[-1][: -len(".py")]
                if not parts:
                    continue
                dotted = ".".join(parts)
                self.modules.setdefault(dotted, path)
                # The deepest root comes first, so this records the name the file's own
                # package uses, which is what relative imports resolve against.
                self.files.setdefault(path, dotted)

    def resolve(self, dotted: str) -> Path | None:
        """The file defining `dotted`, walking up so `a.b.func` finds module `a.b`."""
        segments = dotted.split(".")
        while segments:
            hit = self.modules.get(".".join(segments))
            if hit is not None:
                return hit
            segments.pop()
        return None


class _FileFacts:
    """Everything one file contributes: its functions, its imports, its sinks."""

    def __init__(self, path: Path, tree: ast.AST, index: _ModuleIndex):
        self.path = path
        self.relative = _relative(path)
        self.tree = tree
        self.index = index
        self.module = index.files.get(path, "")
        self.is_package = path.name == "__init__.py"
        # local alias -> dotted targets, for both `import x.y as z` and `from x import y`.
        #
        # A set, not one target, because one file can bind the same alias to two
        # different modules in two different functions. `import json as codec` in one and
        # `import pickle as codec` in another used to leave whichever came last, and
        # canonicalising `codec.load` to the wrong one of those loses the deserialiser:
        # the JSON read stopped being an untrusted source and a dynamic import below it
        # was accepted. Resolution tries every binding and the strongest answer wins, so
        # an ambiguous alias fails closed instead of silently picking one.
        self.imports: dict[str, set[str]] = {}
        # qualname -> FunctionDef
        self.functions: dict[str, ast.AST] = {}
        # qualname -> parameter names in order
        self.params: dict[str, list[str]] = {}
        # qualname -> digest of the whole function, so a baselined sink is re-opened when
        # anything around it changes, including a validator that guarded its input
        self.contexts: dict[str, str] = {}
        # qualname -> name of its **kwargs parameter, so a forwarded keyword lands
        self.star_kwargs: dict[str, str] = {}
        self._collect()

    def _collect(self) -> None:
        scope: list[str] = []

        def walk(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        if alias.asname:
                            self._bind_import(alias.asname, alias.name)
                        else:
                            # `import pkg.parser` binds `pkg`, not `pkg.parser`. Recording
                            # the full dotted name against the head made target_of append
                            # the original tail to it a second time, so `pkg.parser.parse`
                            # resolved as `pkg.parser.parser.parse` and did not resolve at
                            # all: taint returned by that helper never reached its caller.
                            head = alias.name.split(".")[0]
                            self._bind_import(head, head)
                elif isinstance(child, ast.ImportFrom):
                    base = self._absolute(child)
                    for alias in child.names:
                        target = f"{base}.{alias.name}" if base else alias.name
                        self._bind_import(alias.asname or alias.name, target)
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    scope.append(child.name)
                    qualname = ".".join(scope)
                    self.functions[qualname] = child
                    self.params[qualname] = _param_names(child)
                    self.contexts[qualname] = _norm_hash(child)
                    if child.args.kwarg is not None:
                        self.star_kwargs[qualname] = child.args.kwarg.arg
                    walk(child)
                    scope.pop()
                    continue
                elif isinstance(child, ast.ClassDef):
                    scope.append(child.name)
                    walk(child)
                    scope.pop()
                    continue
                walk(child)

        walk(self.tree)

    def _bind_import(self, alias: str, target: str) -> None:
        self.imports.setdefault(alias, set()).add(target)

    def _targets(self, alias: str) -> list[str]:
        """Every module one alias can mean in this file, in a deterministic order."""
        return sorted(self.imports.get(alias, ()))

    def _absolute(self, node: ast.ImportFrom) -> str:
        """Resolve a relative import against this file's containing package.

        One dot means the package the module lives in, so for `pkg.use` it is `pkg` and
        `from .parser import parse` is `pkg.parser.parse`. Keeping the current module in
        the base made it `pkg.use.parser.parse`, which the module index cannot resolve,
        so taint returned by a sibling helper never reached a sink in the caller. For an
        `__init__.py` the module name already IS the package, so nothing is dropped.
        """
        if not node.level:
            return node.module or ""
        segments = self.module.split(".") if self.module else []
        package = segments if self.is_package else segments[:-1]
        extra = node.level - 1
        if extra > len(package):
            return ""
        base = package[: len(package) - extra]
        if node.module:
            base = base + node.module.split(".")
        return ".".join(base)

    def canonicals(self, name: str) -> list[str]:
        """Every spelling a callee can have once this file's imports are applied.

        Suffix matching alone cannot do this, and claiming it could was wrong: after
        `import json as js`, `js.load` has no suffix in the table, and after
        `from yaml import safe_load` the call is the bare name `safe_load`. Both are
        deserialisers of attacker bytes that produced no taint at all, so a dynamic
        import or a subprocess immediately downstream was silently accepted.

        A list rather than one name because an alias can be bound twice in one file. The
        callers try all of them and take the first match, which is what makes an
        ambiguous alias fail closed: `codec.load` is treated as a deserialiser if any
        binding of `codec` makes it one.
        """
        if not name:
            return [name]
        head, separator, tail = name.partition(".")
        targets = self._targets(head)
        if not targets:
            return [name]
        return [f"{dotted}.{tail}" if separator else dotted for dotted in targets]

    def canonical(self, name: str) -> str:
        """One spelling, for a message. Matching goes through `canonicals`."""
        return self.canonicals(name)[0]

    def _local_function(self, name: str, scope: str) -> str | None:
        """A bare name, resolved against the scopes enclosing `scope`.

        A nested helper is indexed under its qualified name, so `def execute` inside
        `def run` is `run.execute` while the call to it is the bare `execute`. Looking the
        bare name up on its own therefore missed every nested helper: taint neither
        entered one nor came back out, and a sink inside it was invisible. Innermost
        first, then outwards, then module level, which is how Python resolves it.
        """
        parts = scope.split(".") if scope else []
        while parts:
            candidate = ".".join(parts + [name])
            if candidate in self.functions:
                return candidate
            parts.pop()
        return name if name in self.functions else None

    def target_of(
        self,
        callee: ast.AST,
        class_name: str = "",
        scope: str = "",
    ) -> tuple[Path, str] | None:
        """Resolve a call target to a first-party (file, qualname), or None.

        This is what makes the analysis inter-procedural. Three forms matter:
        a bare local function, a name imported with `from ... import f`, and a
        `module.f` where `module` was imported.
        """
        name = _call_name(callee)
        if not name:
            return None
        head, _, tail = name.partition(".")
        # Bare call to a function defined in this file, nested helpers included.
        if not tail:
            local = self._local_function(name, scope)
            if local is not None:
                return (self.path, local)
        # `self.method(...)` and `cls.method(...)`. Without this an instance method is
        # outside the analysis entirely: taint neither enters it nor returns from it, so
        # a class that parses an untrusted config in one method and dynamically imports
        # the result in another was accepted. Most code in this tree is methods.
        if head in ("self", "cls") and tail and class_name:
            candidate = f"{class_name}.{tail}"
            if candidate in self.functions:
                return (self.path, candidate)
            return None
        # `from pkg.mod import f` then `f(...)`.
        targets = self._targets(head)
        if not targets:
            if not tail:
                local = self._local_function(name, scope)
                if local is not None:
                    return (self.path, local)
            return None
        for dotted in targets:
            full = f"{dotted}.{tail}" if tail else dotted
            file = self.index.resolve(full)
            if file is None:
                continue
            module = self.index.files.get(file, "")
            qualname = full[len(module) + 1 :] if module and full.startswith(module + ".") else ""
            return (file, qualname)
        return None


def _param_names(node: ast.AST) -> list[str]:
    arguments = node.args
    names = [a.arg for a in list(arguments.posonlyargs) + list(arguments.args)]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    names += [a.arg for a in arguments.kwonlyargs]
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


class _TaintPass(ast.NodeVisitor):
    """One function body, one pass, under the taint state the fixpoint has so far.

    Tainted things are named by string so the state is a plain set and the fixpoint is a
    set comparison: a local `name`, an instance attribute `Class.attr`, or a module
    global `path::NAME`.
    """

    def __init__(self, facts: _FileFacts, qualname: str, state: "_State"):
        self.facts = facts
        self.qualname = qualname
        self.state = state
        # name -> why it is tainted. The reason is the *origin* of the chain, carried
        # through every assignment, because the tier is decided by where the value came
        # from and a reason of "local" would lose exactly that.
        self.local_reasons: dict[str, str] = {}
        key = (facts.path, qualname)
        for name, reason in sorted(state.tainted_params.get(key, {}).items()):
            self.local_reasons[name] = reason
        for name in sorted(state.named_params.get(key, set())):
            self.local_reasons.setdefault(name, NAMED_PARAM_REASON)
        self.artefacts: set[str] = set()
        self.returns_tainted: str = ""
        self.findings: list[dict] = []
        self.class_name = qualname.split(".")[0] if "." in qualname else ""

    # -- taint queries -----------------------------------------------------------------

    def tainted(self, node: ast.AST) -> str | None:
        """Why `node` is tainted, or None. The reason is carried into the finding."""
        if isinstance(node, ast.Name):
            reason = self.local_reasons.get(node.id)
            if reason:
                return reason
            own = self.state.tainted_globals.get(f"{self.facts.relative}::{node.id}")
            if own:
                return own
            return self._imported_global(node.id)
        if isinstance(node, ast.Attribute):
            attribute_reason = self.state.tainted_attrs.get(self._attr_key(node))
            if attribute_reason:
                return attribute_reason
            method = _matches(_call_name(node), UNTRUSTED_METHODS)
            if method:
                return f"read via .{method}"
            return self.tainted(node.value)
        if isinstance(node, ast.Subscript):
            return self.tainted(node.value)
        if isinstance(node, ast.Starred):
            return self.tainted(node.value)
        if isinstance(node, ast.Await):
            # `body = await request.json()` is Await(Call(...)), and only a bare Call was
            # recognised, so every async read of a request body or a file came out clean.
            return self.tainted(node.value)
        if isinstance(node, ast.Call):
            return self._tainted_call(node)
        if isinstance(node, (ast.BinOp,)):
            return self.tainted(node.left) or self.tainted(node.right)
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                reason = self.tainted(value)
                if reason:
                    return reason
            return None
        if isinstance(node, ast.IfExp):
            return self.tainted(node.body) or self.tainted(node.orelse)
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    reason = self.tainted(value.value)
                    if reason:
                        return reason
            return None
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            for generator in node.generators:
                reason = self.tainted(generator.iter)
                if reason:
                    return reason
            return self.tainted(node.elt)
        if isinstance(node, ast.DictComp):
            for generator in node.generators:
                reason = self.tainted(generator.iter)
                if reason:
                    return reason
            return self.tainted(node.key) or self.tainted(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                reason = self.tainted(element)
                if reason:
                    return reason
            return None
        if isinstance(node, ast.Dict):
            for value in node.values:
                reason = self.tainted(value)
                if reason:
                    return reason
            return None
        return None

    def _tainted_call(self, node: ast.Call) -> str | None:
        names = self.facts.canonicals(_call_name(node.func))
        source = _matches_any(names, UNTRUSTED_CALLS)
        if source:
            # The file and line are part of the reason so that two downloads stay
            # distinguishable: the unpinned-fetch rule has to know WHICH download reached
            # an import, not merely that one in the same function did. The file belongs in
            # it as well as the line, because a reason travels across files and a bare
            # line number would collide with an unrelated fetch sitting on that line
            # somewhere else.
            return f"{source}()@{self.facts.relative}:{node.lineno}"
        method = _matches_any(names, UNTRUSTED_METHODS)
        if method:
            return f"read via .{method}"
        # Container and string operations preserve taint.
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "get",
            "pop",
            "format",
            "join",
            "split",
            "strip",
            "lower",
            "upper",
            "replace",
            "rsplit",
            "partition",
            "setdefault",
            # Container views. Without these a dict comprehension over `cfg.items()`
            # lost the taint at the `.items()` call.
            "items",
            "keys",
            "values",
        ):
            reason = self.tainted(node.func.value)
            if reason:
                return reason
            for argument in node.args:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # Container constructors pass their contents straight through, so wrapping a
        # tainted iterable in one of these is not a sanitiser.
        if _matches_any(
            names,
            {"list", "tuple", "set", "dict", "sorted", "reversed", "iter", "enumerate"},
        ):
            for argument in node.args:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # `open(downloaded)` hands back a handle onto attacker bytes, and the usual shape
        # is `with open(downloaded, "rb") as handle: pickle.load(handle)`. Without this
        # the handle read clean and the declared pickle sink never fired.
        if _matches_any(names, {"open", "io.open", "Path.open"}):
            for argument in node.args:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # `os.path.join(tainted, "Spark-TTS")` is still attacker-influenced, and that is
        # the whole basename-collision shape: a fixed name under a controlled parent.
        if _matches_any(
            names,
            {
                "os.path.join",
                "path.join",
                "os.path.dirname",
                "path.dirname",
                "os.path.abspath",
                "os.path.realpath",
                "os.fspath",
                "str",
                "Path",
                "os.path.basename",
                "os.path.normpath",
            },
        ):
            for argument in node.args:
                reason = self.tainted(argument)
                if reason:
                    return reason
            return None
        # A first-party callee that returns tainted data.
        target = self.facts.target_of(node.func, self.class_name, scope = self.qualname)
        if target is not None:
            returned = self.state.returns_tainted.get(target)
            if returned:
                return returned
        return None

    def _imported_global(self, name: str) -> str | None:
        """`from producer import MODEL_TYPE`, where the producer tainted that global.

        Checking only this file's own globals broke the cross-file guarantee for the one
        shape that needs it least ceremony: `MODEL_TYPE = json.load(...)["model_type"]`
        at the top of one module, imported into another and handed straight to
        `import_module`. Neither file reported anything, because the read and the sink
        each looked local and clean.
        """
        for dotted in self.facts._targets(name):
            owner, _, attribute = dotted.rpartition(".")
            if not owner or not attribute:
                continue
            file = self.facts.index.resolve(owner)
            if file is None:
                continue
            reason = self.state.tainted_globals.get(f"{_relative(file)}::{attribute}")
            if reason:
                return reason
        return None

    def _attr_key(self, node: ast.Attribute) -> str:
        if isinstance(node.value, ast.Name) and node.value.id == "self" and self.class_name:
            return f"{self.facts.relative}::{self.class_name}.{node.attr}"
        return ""

    # -- taint writes ------------------------------------------------------------------

    def _assign(self, target: ast.AST, reason: str) -> None:
        if isinstance(target, ast.Name):
            # Flow-insensitive, and the strongest reason wins: a name tainted by a real
            # read stays tier A even if it is also assigned from a named parameter.
            if self.local_reasons.get(target.id, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
                self.local_reasons[target.id] = reason
        elif isinstance(target, ast.Attribute):
            key = self._attr_key(target)
            if key:
                # Same tier-A-wins rule as locals and parameters. Writing unconditionally
                # let one method's tier-B assignment to self.command overwrite another
                # method's tier-A one, purely on the order the methods are visited, and
                # the sink reading that attribute then did not gate.
                self._bind(self.state.pending_attrs, key, reason)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._assign(element, reason)

    def visit_Assign(self, node: ast.Assign) -> None:
        reason = self.tainted(node.value)
        if reason:
            for target in node.targets:
                self._assign(target, reason)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            reason = self.tainted(node.value)
            if reason:
                self._assign(node.target, reason)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # `if (cfg := json.loads(text)):` binds cfg, and only the statement forms were
        # handled, so the walrus carried a parsed config past the analysis untainted.
        reason = self.tainted(node.value)
        if reason:
            self._assign(node.target, reason)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        reason = self.tainted(node.value)
        if reason:
            self._assign(node.target, reason)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        reason = self.tainted(node.iter)
        if reason:
            self._assign(node.target, reason)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                reason = self.tainted(item.context_expr)
                if reason:
                    self._assign(item.optional_vars, reason)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            reason = self.tainted(node.value)
            if reason and (not self.returns_tainted or self.returns_tainted == NAMED_PARAM_REASON):
                self.returns_tainted = reason
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for artefact in UNTRUSTED_ARTEFACTS:
                if artefact in lowered:
                    self.artefacts.add(artefact)
        self.generic_visit(node)

    # -- sinks -------------------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        self._propagate_into_callee(node)
        self._check_sink(node)
        self._check_remote_code(node)
        self.generic_visit(node)

    @staticmethod
    def _bind(bound: dict[str, str], name: str, reason: str) -> None:
        """Strongest reason wins, same rule the settled parameters already used.

        Assigning unconditionally let a weaker later call overwrite a stronger earlier
        one: `execute(json.loads(blob))` followed anywhere by `execute(model_type)` left
        the parameter marked only `untrusted parameter name`, so a sink inside `execute`
        came out tier B and did not fail the gate even though the first call site hands
        it proven untrusted data.
        """
        if bound.get(name, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
            bound[name] = reason

    def _propagate_into_callee(self, node: ast.Call) -> None:
        """Taint the callee's parameters, which is how a chain crosses a file."""
        target = self.facts.target_of(node.func, self.class_name, scope = self.qualname)
        if target is None:
            return
        params = self.state.params.get(target)
        if not params:
            return
        bound = self.state.pending_params.setdefault(target, {})
        offset = (
            1 if self.state.is_method.get(target) and isinstance(node.func, ast.Attribute) else 0
        )
        for position, argument in enumerate(node.args):
            reason = self.tainted(argument)
            if not reason:
                continue
            index = position + offset
            if index < len(params):
                self._bind(bound, params[index], reason)
        star_kwargs = self.state.star_kwargs.get(target)
        for keyword in node.keywords:
            reason = self.tainted(keyword.value)
            if not reason:
                continue
            if keyword.arg is None:
                # `execute(**json.loads(blob))`. Which named parameter the dict supplies is
                # not knowable here, so all of them are bound. Deliberately conservative:
                # the alternative was binding none at all, and a callee that declares
                # `command` rather than `**kwargs` then ran a value out of that dict with
                # nothing reported anywhere.
                if star_kwargs:
                    self._bind(bound, star_kwargs, reason)
                for parameter in params:
                    if parameter not in ("self", "cls"):
                        self._bind(bound, parameter, reason)
                continue
            if keyword.arg in params:
                self._bind(bound, keyword.arg, reason)
            elif star_kwargs:
                # A keyword the callee does not name by hand still arrives, in **kwargs.
                # Matching only on the keyword's own name dropped the taint entirely for
                # every helper that forwards its options that way.
                self._bind(bound, star_kwargs, reason)
            elif keyword.arg:
                self._bind(bound, keyword.arg, reason)

    def _check_sink(self, node: ast.Call) -> None:
        names = self.facts.canonicals(_call_name(node.func))
        sink = _matches_any(names, SINKS)
        if sink is None:
            self._check_getattr(node)
            return
        positions, keywords = SINKS[sink]
        for index in positions:
            if index < len(node.args):
                reason = self.tainted(node.args[index])
                if reason:
                    self._record(node, sink, reason, _short(node.args[index]))
                    return
        for keyword in node.keywords:
            if keyword.arg in keywords:
                reason = self.tainted(keyword.value)
                if reason:
                    self._record(node, sink, reason, _short(keyword.value))
                    return

    def _check_getattr(self, node: ast.Call) -> None:
        """`getattr(transformers, tainted)` resolves an arbitrary name in a namespace."""
        if _call_name(node.func).rpartition(".")[2] != "getattr" or len(node.args) < 2:
            return
        holder = node.args[0]
        holder_names = (
            [candidate.split(".")[0] for candidate in self.facts.canonicals(_call_name(holder))]
            if not isinstance(holder, ast.Call)
            else []
        )
        is_module_ish = any(held in MODULE_ISH_NAMES for held in holder_names) or (
            isinstance(holder, ast.Call)
            and _matches_any(
                self.facts.canonicals(_call_name(holder.func)),
                {"importlib.import_module", "import_module"},
            )
        )
        if not is_module_ish:
            return
        reason = self.tainted(node.args[1])
        if reason:
            self._record(node, "getattr(module, ...)", reason, _short(node.args[1]))

    def _check_remote_code(self, node: ast.Call) -> None:
        """`trust_remote_code = True` written at a loader, or forwarded as a constant."""
        names = self.facts.canonicals(_call_name(node.func))
        name = next(
            (
                candidate
                for candidate in names
                if any(marker in candidate for marker in REMOTE_CODE_LOADERS)
            ),
            "",
        )
        if not name:
            return
        for keyword in node.keywords:
            if keyword.arg != "trust_remote_code":
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self._record(
                    node,
                    "trust_remote_code = True",
                    "written True at the call",
                    f"{name}(trust_remote_code = True)",
                    tier = "A",
                )

    def _record(
        self,
        node: ast.Call,
        sink: str,
        reason: str,
        argument: str,
        tier: str = "",
    ) -> None:
        if not tier:
            tier = "B" if reason == NAMED_PARAM_REASON else "A"
        self.findings.append(
            {
                "path": self.facts.relative,
                "line": node.lineno,
                "qualname": self.qualname,
                "sink": sink,
                "argument": argument,
                "why": reason,
                "artefacts": sorted(self.artefacts),
                "tier": tier,
                "hash": _norm_hash(node),
                "context": self.facts.contexts.get(self.qualname, ""),
                "gated": sink not in SINKS_GATED_ELSEWHERE,
            }
        )


class _State:
    """The fixpoint's mutable state, shared across files."""

    def __init__(self) -> None:
        # Every map is name -> reason, so a finding can say where the value came from.
        self.tainted_params: dict[tuple[Path, str], dict[str, str]] = {}
        self.pending_params: dict[tuple[Path, str], dict[str, str]] = {}
        self.named_params: dict[tuple[Path, str], set[str]] = {}
        self.tainted_attrs: dict[str, str] = {}
        self.pending_attrs: dict[str, str] = {}
        self.tainted_globals: dict[str, str] = {}
        self.returns_tainted: dict[tuple[Path, str], str] = {}
        self.params: dict[tuple[Path, str], list[str]] = {}
        self.is_method: dict[tuple[Path, str], bool] = {}
        # (file, qualname) -> the name of the callee's **kwargs parameter, if it has one
        self.star_kwargs: dict[tuple[Path, str], str] = {}

    def snapshot(self) -> str:
        return json.dumps(
            {
                "params": sorted(
                    f"{path}::{qualname}::{name}={reason}"
                    for (path, qualname), names in self.tainted_params.items()
                    for name, reason in names.items()
                ),
                "attrs": sorted(f"{key}={reason}" for key, reason in self.tainted_attrs.items()),
                "globals": sorted(
                    f"{key}={reason}" for key, reason in self.tainted_globals.items()
                ),
                "returns": sorted(
                    f"{path}::{qualname}={reason}"
                    for (path, qualname), reason in self.returns_tainted.items()
                ),
            },
            sort_keys = True,
        )


def _remote_code_defaults(facts: _FileFacts) -> list[dict]:
    """`trust_remote_code` turned on without a call keyword to show it.

    Three spellings, all of which reached a loader in this tree at some point and none
    of which the keyword check above sees:

        def load(..., trust_remote_code = True)      a default nobody passes
        kwargs = {"trust_remote_code": True}         a dict splatted into the loader
        trust_remote_code = True                     an assignment before the call

    The dict form is the one that matters most, because the keyword never appears at the
    call site at all and so reads as if the caller decided.
    """
    findings: list[dict] = []

    # node -> the function it sits in. The dict and assignment spellings used to be
    # recorded under the synthetic qualnames `<dict>`, `<assign>` and `<item>`, which no
    # function is called, so the context digest in the baseline key was always empty: a
    # reviewed `trust_remote_code = True` kept its allowance word for word after the
    # consent check around it was weakened, which is the one thing the context digest
    # exists to stop. Deeper functions are walked later so the innermost one wins.
    owners: dict[int, str] = {}
    for owner_qualname, owner_node in sorted(
        facts.functions.items(), key = lambda item: (item[0].count("."), item[0])
    ):
        for child in ast.walk(owner_node):
            owners[id(child)] = owner_qualname

    def record(
        node: ast.AST,
        shape: str,
        text: str,
        qualname: str = "",
    ) -> None:
        if not qualname:
            qualname = owners.get(id(node), "<module>")
        findings.append(
            {
                "path": facts.relative,
                "line": node.lineno,
                "qualname": qualname,
                "sink": f"trust_remote_code = True ({shape})",
                "argument": text,
                "why": "on by default, not by the user",
                "artefacts": [],
                "tier": "A",
                "hash": _norm_hash(node),
                "context": facts.contexts.get(qualname, ""),
                "gated": True,
            }
        )

    for qualname, node in sorted(facts.functions.items()):
        arguments = node.args
        positional = list(arguments.posonlyargs) + list(arguments.args)
        pairs = list(
            zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults)
        )
        pairs += [
            (argument, default)
            for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
            if default is not None
        ]
        for argument, default in pairs:
            if argument.arg != "trust_remote_code":
                continue
            if isinstance(default, ast.Constant) and default.value is True:
                record(
                    node,
                    "default",
                    f"def {qualname}(..., trust_remote_code = True)",
                    qualname = qualname,
                )

    for node in ast.walk(facts.tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "trust_remote_code"
                    and isinstance(value, ast.Constant)
                    and value.value is True
                ):
                    record(node, "dict", _short(node))
        elif isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Constant) and node.value.value is True):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "trust_remote_code":
                    record(node, "assignment", _short(node))
                # `kwargs["trust_remote_code"] = True` then `from_pretrained(**kwargs)`.
                # The keyword never appears at the call, and the call checker cannot see
                # inside an expanded dict, so this spelling was reported nowhere at all.
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "trust_remote_code"
                ):
                    record(node, "dict item", _short(node))
    return findings


def _unpinned_code_fetches(facts: _FileFacts, reached: dict[str, frozenset]) -> list[dict]:
    """A download with no `revision` whose bytes the same function then imports.

    `snapshot_download(repo)` without a revision resolves to whatever the branch points
    at when it runs, so the code that executes is not the code that was reviewed. That
    is only a note on a weights download, and it is the whole story when the fetched
    tree is then put on `sys.path` and imported: the import executes the current tip of
    a remote branch.

    Narrow on purpose, twice over. The test is not "does this function contain an
    import" - lazy imports are everywhere in this tree and that version of the rule
    reported sixty weights downloads, where following the branch is correct behaviour
    and not a flaw. Nor is it "does this function contain both" - a function that
    downloads weights and separately imports a fixed optional backend satisfies
    co-location while none of the fetched bytes are executed.

    The test is whether the fetched value REACHES the import, which the taint analysis
    has already decided: `reached` carries, per function, the identity of each download
    whose value arrived at an import sink. Correlation, not proximity, and per download
    rather than per function: a function that pins the code it imports and separately
    downloads weights used to have the weights fetch reported, because a function-wide
    boolean cannot say which of the two the sink actually consumed.
    """
    findings: list[dict] = []
    for qualname, node in sorted(facts.functions.items()):
        arrived = reached.get(qualname) or frozenset()
        if not arrived:
            continue
        fetches: list[ast.Call] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            source = _matches_any(
                facts.canonicals(_call_name(child.func)),
                {"snapshot_download", "hf_hub_download"},
            )
            if not source:
                continue
            if any(keyword.arg == "revision" for keyword in child.keywords):
                continue
            # Same identity the taint reason carries, so this is the download the sink
            # read and not merely one of the downloads in the same body.
            if f"{source}()@{facts.relative}:{child.lineno}" not in arrived:
                continue
            fetches.append(child)
        for call in fetches:
            findings.append(
                {
                    "path": facts.relative,
                    "line": call.lineno,
                    "qualname": qualname,
                    "sink": "unpinned code fetch",
                    "argument": _short(call),
                    "why": "no revision, and the fetched path reaches an import",
                    "artefacts": [],
                    "tier": "A",
                    "hash": _norm_hash(call),
                    "context": facts.contexts.get(qualname, ""),
                    "gated": True,
                }
            )
    return findings


_LOCAL_BOUND = 64
# The interprocedural fixpoint's own valve. Deeper than any call chain in these trees,
# and exceeding it fails the run rather than truncating the analysis quietly.
_GLOBAL_BOUND = 64


def _settle(facts: _FileFacts, qualname: str, nodes: list, state: "_State"):
    """Walk one body until its local taint stops growing. Returns (visitor, converged).

    One ordered traversal is not enough even for a flow-insensitive result, because
    `local_reasons` is built as the walk proceeds: a sink visited before a later tainted
    assignment to the same name would never be reconsidered. A loop that consumes `name`
    and then rebinds it from `json.loads` for the next iteration is a real executable
    flow.

    The bound is a safety valve, not a cutoff to rely on: a chain of assignments longer
    than the bound would still be growing when it expires. Whether it converged is
    returned rather than swallowed, so the caller can fail the run instead of reporting
    a partial answer as a clean one.
    """
    reasons: dict[str, str] = {}
    visitor = None
    for _ in range(_LOCAL_BOUND):
        visitor = _TaintPass(facts, qualname, state)
        visitor.local_reasons.update(reasons)
        for child in nodes:
            visitor.visit(child)
        if visitor.local_reasons == reasons:
            return visitor, True
        reasons = dict(visitor.local_reasons)
    return visitor, False


def _python_files(targets: list[Path]) -> list[Path]:
    found: list[Path] = []
    for target in targets:
        if target.is_file():
            if target.suffix == ".py":
                found.append(target.resolve())
            continue
        for path in target.rglob("*.py"):
            parts = set(path.parts)
            if parts & EXCLUDED_PARTS:
                continue
            if parts & EXCLUDED_DIRS:
                continue
            found.append(path.resolve())
    return sorted(set(found))


def _module_level_taint(facts: _FileFacts, state: _State) -> None:
    """Module-level assignments, so a global read at import time is tainted too."""
    visitor = _TaintPass(facts, "<module>", state)
    for node in ast.iter_child_nodes(facts.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        visitor.visit(node)
    for name, reason in sorted(visitor.local_reasons.items()):
        state.tainted_globals[f"{facts.relative}::{name}"] = reason


def scan(targets: list[Path], roots: list[Path] | None = None) -> list[dict]:
    """Every finding, sorted, for a deterministic report."""
    files = _python_files(targets)
    if roots is None:
        roots = [REPO_ROOT]
        backend = REPO_ROOT / "studio" / "backend"
        if backend.is_dir():
            roots.append(backend)
    index = _ModuleIndex(roots, files)

    facts_by_path: dict[Path, _FileFacts] = {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8", errors = "replace"), filename = str(path))
        except SyntaxError:
            continue
        facts_by_path[path] = _FileFacts(path, tree, index)

    state = _State()
    for path, facts in sorted(facts_by_path.items()):
        for qualname, node in sorted(facts.functions.items()):
            key = (path, qualname)
            state.params[key] = facts.params[qualname]
            state.is_method[key] = bool(facts.params[qualname]) and facts.params[qualname][0] in (
                "self",
                "cls",
            )
            star = facts.star_kwargs.get(qualname)
            if star:
                state.star_kwargs[key] = star
            # Tier B seeding: a parameter whose name says it carries untrusted data.
            seeded = {name for name in facts.params[qualname] if name in UNTRUSTED_PARAM_NAMES}
            if seeded:
                state.named_params[key] = seeded

    # Bodies whose local taint was still growing when the safety bound expired. Reported
    # rather than swallowed: a partial answer presented as a clean one is the failure
    # mode this whole script exists to avoid.
    unconverged: set[str] = set()

    # Interprocedural fixpoint. Bounded, because taint only grows and a pathological
    # tree must not run the lint job forever, but the bound is a safety valve and not a
    # cutoff to rely on: a chain of caller-before-callee helpers deeper than the bound
    # would still be settling when it expires, and sink collection would then run with
    # incomplete return summaries. Whether it settled is recorded, and an unsettled run
    # fails rather than reporting a partial answer as a clean one.
    settled = False
    for _ in range(_GLOBAL_BOUND):
        before = state.snapshot()
        state.pending_params = {}
        state.pending_attrs = {}
        for path, facts in sorted(facts_by_path.items()):
            # Inside the loop, not once before it: a global initialised through a
            # first-party helper (`MODEL_TYPE = parse_config()`) is only tainted once
            # that helper's return summary exists, which the fixpoint produces.
            _module_level_taint(facts, state)
            for qualname, node in sorted(facts.functions.items()):
                # Settle the locals HERE, not only when reporting. The pending tainted
                # parameters a settled body writes for its callees have to be merged by
                # this loop and the callee re-analysed, otherwise a loop that calls
                # execute(command) and then rebinds command from json.loads taints the
                # parameter too late for anything to look inside execute again.
                visitor, converged = _settle(
                    facts, qualname, list(ast.iter_child_nodes(node)), state
                )
                if not converged:
                    unconverged.add(f"{facts.relative}::{qualname}")
                if visitor is not None and visitor.returns_tainted:
                    key = (path, qualname)
                    known = state.returns_tainted.get(key)
                    if not known or known == NAMED_PARAM_REASON:
                        state.returns_tainted[key] = visitor.returns_tainted
        for key, names in state.pending_params.items():
            bound = state.tainted_params.setdefault(key, {})
            for name, reason in names.items():
                if bound.get(name, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
                    bound[name] = reason
        for attribute, reason in state.pending_attrs.items():
            if state.tainted_attrs.get(attribute, NAMED_PARAM_REASON) == NAMED_PARAM_REASON:
                state.tainted_attrs[attribute] = reason
        if state.snapshot() == before:
            settled = True
            break

    # Sinks whose tainted argument came from a download, per function. This is what
    # lets the unpinned-fetch rule ask whether the fetched path reaches an import
    # rather than whether the two merely appear in the same function.
    _DOWNLOADS = ("snapshot_download()@", "hf_hub_download()@")
    _IMPORT_SINKS = frozenset(
        {
            "sys.path.insert",
            "sys.path.append",
            "path.insert",
            "path.append",
            "importlib.import_module",
            "import_module",
            "__import__",
            "spec_from_file_location",
            "importlib.util.spec_from_file_location",
        }
    )

    findings: list[dict] = []
    for path, facts in sorted(facts_by_path.items()):
        reached: dict[str, frozenset] = {}
        for qualname, node in sorted(facts.functions.items()):
            visitor, converged = _settle(facts, qualname, list(ast.iter_child_nodes(node)), state)
            if not converged:
                unconverged.add(f"{facts.relative}::{qualname}")
            found = visitor.findings if visitor is not None else []
            reached[qualname] = frozenset(
                f["why"]
                for f in found
                if f["sink"] in _IMPORT_SINKS
                and any(f["why"].startswith(prefix) for prefix in _DOWNLOADS)
            )
            findings.extend(found)
        body = [
            child
            for child in ast.iter_child_nodes(facts.tree)
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        module_visitor, module_converged = _settle(facts, "<module>", body, state)
        if not module_converged:
            unconverged.add(f"{facts.relative}::<module>")
        if module_visitor is not None:
            findings.extend(module_visitor.findings)
        findings.extend(_remote_code_defaults(facts))
        findings.extend(_unpinned_code_fetches(facts, reached))

    if not settled:
        unconverged.add(f"<whole scan>::interprocedural fixpoint")

    if unconverged:
        for where in sorted(unconverged):
            findings.append(
                {
                    "path": where.split("::")[0],
                    "line": 0,
                    "qualname": where.split("::")[-1],
                    "sink": INCOMPLETE_SINK,
                    "argument": f"local taint still growing after {_LOCAL_BOUND} passes",
                    "why": "the result for this body is incomplete",
                    "artefacts": [],
                    "tier": "A",
                    "hash": "unconverged",
                    "context": "",
                    "gated": True,
                }
            )

    deduplicated = {
        (f["path"], f["qualname"], f["sink"], f["hash"], f["line"]): f for f in findings
    }
    return [deduplicated[key] for key in sorted(deduplicated)]


def _baseline_key(finding: dict) -> str:
    """Identity of a reviewed sink.

    The enclosing function's digest is part of it, not just the call's own text. A
    baselined sink may have been accepted because something nearby validated its input,
    and this analysis does not model validators, so weakening one leaves the path, the
    qualname, the sink and the call text identical: the allowance would still match and
    the gate would pass on exactly the regression it exists to catch. Including the
    function means any change around the call re-opens the question. That costs churn,
    and the churn is the point: if you edit a function containing a reviewed sink, you
    re-justify it.
    """
    return (
        f"{finding['path']}::{finding['qualname']}::{finding['sink']}"
        f"::{finding['hash']}::{finding.get('context', '')}"
    )


def _counted(findings: list[dict]) -> dict[str, int]:
    """How many times each reviewed-sink identity appears in this scan."""
    counted: dict[str, int] = {}
    for finding in findings:
        if not finding["gated"] or finding["tier"] != "A":
            continue
        if finding["sink"] == INCOMPLETE_SINK:
            continue
        key = _baseline_key(finding)
        counted[key] = counted.get(key, 0) + 1
    return counted


def _unbaselined(findings: list[dict], baseline: dict | None = None) -> list[dict]:
    """Gating findings with no allowance behind them. Extracted so it can be tested."""
    if baseline is None:
        baseline = _load_baseline()
    seen: dict[str, int] = {}
    new: list[dict] = []
    for finding in findings:
        if not finding["gated"] or finding["tier"] != "A":
            continue
        if finding["sink"] == INCOMPLETE_SINK:
            # Never allowable, whatever a hand-edited baseline says. --update refuses to
            # write one of these, and consulting the baseline here would be the other
            # half of the same hole.
            new.append(finding)
            continue
        key = _baseline_key(finding)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > baseline.get(key, 0):
            new.append(finding)
    return new


def _stale_allowances(
    findings: list[dict],
    baseline: dict | None = None,
    scope: set | None = None,
) -> list[str]:
    """Allowances for a sink that is no longer there.

    A loaded gun: the key is the path, the qualname, a hash of the call and the enclosing
    function, so a later change that restores the identical call in the same place
    inherits the allowance and is never reported. Removing a sink therefore has to be
    accompanied by --update.

    `scope` is the set of relative paths this run actually looked at. Without it a
    `--paths` run on one file called every allowance for every other file stale and exited
    1 on a clean file, which makes the option useless for the thing it is for: checking
    one file you just edited. An allowance for a file nobody scanned is not evidence of
    anything.
    """
    if baseline is None:
        baseline = _load_baseline()
    counted = _counted(findings)
    return sorted(
        f"{key} ({count - counted.get(key, 0)} unused)"
        for key, count in baseline.items()
        if count > counted.get(key, 0) and (scope is None or key.split("::", 1)[0] in scope)
    )


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    with BASELINE_PATH.open(encoding = "utf-8") as handle:
        return json.load(handle).get("entries", {})


INCOMPLETE_SINK = "analysis did not converge"


def _write_baseline(findings: list[dict]) -> None:
    """Record the reviewed sinks, and refuse outright if the analysis is incomplete.

    An exhausted bound emits a synthetic `analysis did not converge` finding, and the
    writer used to store it like any other reviewed sink and report success. Every later
    run then matched that allowance and printed OK while the scanner was still saying its
    own result was partial, which is the exact failure this script exists to prevent: a
    clean answer nobody can stand behind. There is nothing to review in an incomplete
    analysis, so it cannot be baselined and `--update` fails instead.
    """
    incomplete = [f for f in findings if f["sink"] == INCOMPLETE_SINK]
    if incomplete:
        print("refusing to write a baseline: the analysis did not converge", file = sys.stderr)
        for finding in incomplete[:20]:
            print(
                f"  {finding['path']}::{finding['qualname']}: {finding['argument']}",
                file = sys.stderr,
            )
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more", file = sys.stderr)
        print(
            "Raise the bound or simplify the body. A partial result is not a reviewed sink.",
            file = sys.stderr,
        )
        raise SystemExit(2)
    entries: dict[str, int] = {}
    for finding in findings:
        if not finding["gated"] or finding["tier"] != "A":
            continue
        entries[_baseline_key(finding)] = entries.get(_baseline_key(finding), 0) + 1
    payload = {
        "comment": (
            "Reviewed sinks that a value from a downloaded artefact can reach. "
            "Regenerate with: python scripts/lint_untrusted_sinks.py --update"
        ),
        "entries": dict(sorted(entries.items())),
    }
    with BASELINE_PATH.open("w", encoding = "utf-8") as handle:
        json.dump(payload, handle, indent = 2, sort_keys = True)
        handle.write("\n")


# The two real holes, written out. The taint starts at a download rather than at a bare
# parameter so that both findings have to come out tier A: a parameter name alone is
# tier B by construction and would let a broken fixpoint pass this test.
SELF_TEST_BAD = """
import importlib, json, os, sys
from huggingface_hub import snapshot_download

def read_type(directory):
    with open(os.path.join(directory, "config.json")) as handle:
        return json.load(handle)["model_type"]

def load(repo):
    local = snapshot_download(repo, local_dir = repo.split("/")[-1])
    model_type = read_type(local)
    module = importlib.import_module("transformers.models." + model_type)
    sys.path.insert(0, os.path.join(os.path.dirname(local), "Spark-TTS"))
    return module
"""

SELF_TEST_GOOD = """
import importlib, json, os, re

def read_type(path):
    with open(os.path.join(path, "config.json")) as handle:
        model_type = json.load(handle)["model_type"]
    if not re.fullmatch(r"[a-z0-9_]+", model_type):
        raise ValueError("bad model_type")
    return model_type

def load():
    return importlib.import_module("transformers.models.llama")
"""


def _self_test() -> int:
    """The checker has to fire on the shape it exists for and stay quiet on a constant.

    A linter nobody has seen fail is a linter that does not work. The bad sample is the
    two real holes written out: a config field reaching `import_module` across a
    function boundary, and a sys.path entry derived from a download path.
    """
    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bad = root / "bad_sample.py"
        bad.write_text(SELF_TEST_BAD, encoding = "utf-8")
        good = root / "good_sample.py"
        good.write_text(SELF_TEST_GOOD, encoding = "utf-8")

        bad_findings = scan([bad], roots = [root])
        sinks = {f["sink"] for f in bad_findings if f["tier"] == "A"}
        if "importlib.import_module" not in sinks:
            failures.append("did not flag a config field reaching import_module")
        if "sys.path.insert" not in sinks:
            failures.append("did not flag a download-derived sys.path entry")

        good_findings = [f for f in scan([good], roots = [root]) if f["tier"] == "A"]
        if good_findings:
            failures.append(
                "flagged a literal import: " + ", ".join(f["sink"] for f in good_findings)
            )

    for failure in failures:
        print(f"self-test FAIL: {failure}")
    if failures:
        return 1
    print("self-test OK: fires on the untrusted chain, quiet on the literal")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--paths", nargs = "*", help = "files or directories to scan")
    parser.add_argument(
        "--repo-root",
        help = "scan a checkout other than this script's own, for cross-repo runs",
    )
    parser.add_argument("--update", action = "store_true", help = "rewrite the baseline")
    parser.add_argument("--self-test", action = "store_true", help = "check the checker")
    parser.add_argument("--json", help = "write the full report here")
    parser.add_argument(
        "--tier",
        choices = ("a", "b", "all"),
        default = "a",
        help = "which tier to print; the gate always uses tier A only",
    )
    arguments = parser.parse_args(argv)

    if arguments.self_test:
        return _self_test()

    global REPO_ROOT
    if arguments.repo_root:
        REPO_ROOT = Path(arguments.repo_root).resolve()

    if arguments.paths:
        targets = [Path(p).resolve() for p in arguments.paths]
    else:
        targets = [REPO_ROOT / name for name in DEFAULT_TARGETS]
        targets = [t for t in targets if t.exists()]

    findings = scan(targets)

    if arguments.json:
        with open(arguments.json, "w", encoding = "utf-8") as handle:
            json.dump({"findings": findings}, handle, indent = 2, sort_keys = True)
            handle.write("\n")

    if arguments.update:
        _write_baseline(findings)
        gated = sum(1 for f in findings if f["gated"] and f["tier"] == "A")
        print(f"wrote {BASELINE_PATH.name}: {gated} reviewed tier A sinks")
        return 0

    baseline = _load_baseline()
    new = _unbaselined(findings, baseline = baseline)
    # The stale check only means something for files this run looked at.
    scanned = {_relative(path) for path in _python_files(targets)}
    stale = _stale_allowances(findings, baseline = baseline, scope = scanned)

    shown = [
        f
        for f in findings
        if arguments.tier == "all"
        or (arguments.tier == "a" and f["tier"] == "A")
        or (arguments.tier == "b" and f["tier"] == "B")
    ]
    tier_a = sum(1 for f in findings if f["tier"] == "A")
    tier_b = sum(1 for f in findings if f["tier"] == "B")
    print(f"scanned: {tier_a} tier A findings, {tier_b} tier B, {len(baseline)} baselined")

    if new:
        print(f"\n{len(new)} sink(s) reachable from untrusted input and not in the baseline:\n")
        for finding in new:
            print(f"  {finding['path']}:{finding['line']}  in {finding['qualname']}")
            print(f"    sink     {finding['sink']}")
            print(f"    argument {finding['argument']}")
            print(f"    tainted  {finding['why']}")
            if finding["artefacts"]:
                print(f"    reads    {', '.join(finding['artefacts'])}")
            print()
        print("Validate the value at its producer, or justify it and run --update.")
        return 1

    if stale:
        print(f"\n{len(stale)} baseline allowance(s) no longer match a sink in the tree:\n")
        for entry in stale:
            print(f"  {entry}")
        print("\nRun --update so a later change cannot inherit the allowance.")
        return 1

    if arguments.tier != "a" or shown:
        for finding in shown:
            print(
                f"  [{finding['tier']}] {finding['path']}:{finding['line']} "
                f"{finding['sink']} <- {finding['why']}"
            )
    print("OK: no new untrusted value reaches an execution sink")
    return 0


if __name__ == "__main__":
    sys.exit(main())
